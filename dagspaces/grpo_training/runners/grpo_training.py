"""GRPO Training stage runner.

Supports two vLLM modes (configured via training.grpo.vllm_mode):

  "colocate" (default): Single-process, TRL manages vLLM internally.
    Just calls run_grpo_training_stage directly.

  "server": Runner launches `trl vllm-serve` as a subprocess on dedicated
    GPUs, then calls run_grpo_training_stage for training on remaining GPUs.
    The server subprocess is cleaned up after training completes.

Online R_ground auxiliary servers (embedding + judge) support two modes:

  "managed" (default): Runner launches servers as subprocesses, allocating
    GPUs from the SLURM allocation (1 embed + N judge + remaining train).

  "external": Servers are launched independently and their URLs passed via
    config (embedding_server_url / judge_server_url). All GPUs go to
    training. Enables cross-machine sharing and multi-pipeline reuse.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from typing import Any, Dict, Optional

from omegaconf import OmegaConf

from dagspaces.common.orchestrator import (
    StageExecutionContext,
    StageResult,
)
from .base import StageRunner


def _launch_vllm_server(
    model_path: str,
    grpo_cfg: dict,
) -> subprocess.Popen:
    """Launch `trl vllm-serve` as a background subprocess.

    Returns the Popen handle for cleanup.
    """
    host = grpo_cfg.get("vllm_server_host", "0.0.0.0")
    port = grpo_cfg.get("vllm_server_port", 8000)
    tp = grpo_cfg.get("vllm_server_tensor_parallel_size", 1)
    gpu_mem = grpo_cfg.get("vllm_server_gpu_memory_utilization", 0.9)

    cmd = [
        "trl", "vllm-serve",
        "--model", model_path,
        "--tensor-parallel-size", str(tp),
        "--host", str(host),
        "--port", str(port),
        "--gpu-memory-utilization", str(gpu_mem),
    ]

    # Pin vLLM server to the last GPU(s), leaving earlier GPUs for training.
    # E.g. with CUDA_VISIBLE_DEVICES=0,1 and TP=1, vLLM gets GPU 1.
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    gpu_list = [g.strip() for g in visible.split(",") if g.strip()] if visible else []
    server_env = os.environ.copy()
    if len(gpu_list) > int(tp):
        server_gpus = gpu_list[-int(tp):]  # last TP GPUs for vLLM
        train_gpus = gpu_list[:-int(tp)]   # remaining for training
        server_env["CUDA_VISIBLE_DEVICES"] = ",".join(server_gpus)
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(train_gpus)
        print(f"[grpo_training] GPU split: vLLM={server_gpus}, training={train_gpus}")

    print(f"[grpo_training] Launching vLLM server: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,  # new process group for clean shutdown
        env=server_env,
    )

    # Wait for server to become ready
    timeout = grpo_cfg.get("vllm_server_startup_timeout", 300)
    start = time.time()
    import urllib.request
    health_url = f"http://localhost:{port}/health"
    while time.time() - start < timeout:
        if proc.poll() is not None:
            stdout = proc.stdout.read().decode() if proc.stdout else ""
            raise RuntimeError(
                f"vLLM server exited with code {proc.returncode} during startup.\n{stdout}"
            )
        try:
            urllib.request.urlopen(health_url, timeout=2)
            print(f"[grpo_training] vLLM server ready on {host}:{port} "
                  f"(took {time.time() - start:.1f}s)")
            return proc
        except Exception:
            time.sleep(5)

    # Timeout — kill and raise
    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    raise RuntimeError(
        f"vLLM server did not become ready within {timeout}s"
    )


def _shutdown_server(proc: Optional[subprocess.Popen]) -> None:
    """Gracefully shut down a server subprocess."""
    if proc is None or proc.poll() is not None:
        return
    print("[grpo_training] Shutting down server subprocess...")
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=30)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass


def _check_external_server(
    url: str,
    label: str,
    timeout: int = 60,
    poll_interval: int = 5,
) -> None:
    """Verify an external vLLM server is reachable before starting training.

    Polls the /health endpoint with retries. Raises RuntimeError if the
    server is not reachable within the timeout.
    """
    import urllib.request

    health_url = f"{url.rstrip('/')}/health"
    start = time.time()
    last_error = None

    while time.time() - start < timeout:
        try:
            urllib.request.urlopen(health_url, timeout=5)
            print(f"[grpo_training] External {label} server OK: {url} "
                  f"(checked in {time.time() - start:.1f}s)")
            return
        except Exception as e:
            last_error = e
            time.sleep(poll_interval)

    raise RuntimeError(
        f"External {label} server at {url} not reachable within {timeout}s. "
        f"Last error: {last_error}"
    )


def _launch_auxiliary_server(
    model_path: str,
    port: int,
    gpu_ids: list,
    tp: int = 1,
    startup_timeout: int = 300,
    label: str = "auxiliary",
    max_model_len: Optional[int] = None,
) -> subprocess.Popen:
    """Launch a vLLM server for auxiliary models (embedding or judge).

    Uses ``vllm serve`` (not ``trl vllm-serve``) since these are
    standalone inference servers, not TRL-managed policy models.
    vLLM 0.17+ auto-detects model type (embedding vs generative).

    Args:
        model_path: Path to the model.
        port: Port to serve on.
        gpu_ids: List of GPU indices to pin via CUDA_VISIBLE_DEVICES.
        tp: Tensor parallel size.
        startup_timeout: Max seconds to wait for health check.
        label: Human-readable label for log messages.

    Returns:
        Popen handle for cleanup.
    """
    cmd = [
        "vllm", "serve", model_path,
        "--port", str(port),
        "--tensor-parallel-size", str(tp),
        "--host", "0.0.0.0",
    ]
    # PCIe GPUs: disable custom all-reduce (requires NVLink)
    if tp > 1:
        cmd.extend(["--disable-custom-all-reduce"])
    # AWQ models: use awq_marlin for optimized Marlin kernels (5-8x
    # faster decode than generic awq). vLLM auto-selects this, but
    # only if we don't pass plain "awq" which forces the slow path.
    if "awq" in model_path.lower():
        cmd.extend(["--quantization", "awq_marlin"])
    # Cap KV cache to leave GPU memory for concurrent request batching
    if max_model_len:
        cmd.extend(["--max-model-len", str(max_model_len)])

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
    # Suppress tokenizer parallelism warnings in subprocess
    env["TOKENIZERS_PARALLELISM"] = "false"
    # Ensure NCCL vars for PCIe-only machines are set
    from dagspaces.common.vllm_inference import get_pcie_nccl_env_vars
    for k, v in get_pcie_nccl_env_vars().items():
        env.setdefault(k, v)

    print(f"[grpo_training] Launching {label} server: {' '.join(cmd)}")
    print(f"[grpo_training]   GPUs: {gpu_ids}, port: {port}")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
        env=env,
    )

    # Wait for health check
    import urllib.request
    health_url = f"http://localhost:{port}/health"
    start = time.time()
    while time.time() - start < startup_timeout:
        if proc.poll() is not None:
            stdout = proc.stdout.read().decode() if proc.stdout else ""
            raise RuntimeError(
                f"{label} server exited with code {proc.returncode} "
                f"during startup.\n{stdout}"
            )
        try:
            urllib.request.urlopen(health_url, timeout=2)
            print(f"[grpo_training] {label} server ready on port {port} "
                  f"(took {time.time() - start:.1f}s)")
            return proc
        except Exception:
            time.sleep(5)

    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    raise RuntimeError(
        f"{label} server did not become ready within {startup_timeout}s"
    )


class GRPOTrainingRunner(StageRunner):
    """Runner for the grpo_training stage.

    Produces a checkpoint directory instead of a DataFrame.
    For server mode, manages the vLLM server lifecycle.
    """

    stage_name = "grpo_training"

    def run(self, context: StageExecutionContext) -> StageResult:
        from ..stages.grpo_training import run_grpo_training_stage

        cfg = context.cfg
        sft_checkpoint = context.inputs.get("sft_checkpoint")
        chunks_path = context.inputs.get("chunks")
        reward_cache_path = context.inputs.get("reward_cache", "")
        norm_universes_path = context.inputs.get("norm_universes", "")
        embeddings_dir = context.inputs.get("embeddings", "")

        missing = []
        if not sft_checkpoint:
            missing.append("sft_checkpoint")
        if not chunks_path:
            missing.append("chunks")

        # online_rground needs norm universes for the judge
        grpo_cfg_raw = OmegaConf.select(cfg, "training.grpo")
        if grpo_cfg_raw and OmegaConf.to_container(grpo_cfg_raw, resolve=True).get("online_rground", False):
            if not norm_universes_path:
                missing.append("norm_universes (required by online_rground)")

        if missing:
            raise ValueError(
                f"Node '{context.node.key}' requires inputs: {missing}"
            )
        checkpoint_dir = context.output_paths.get("checkpoint")
        if not checkpoint_dir:
            checkpoint_dir = os.path.join(context.output_dir, "checkpoint")

        os.makedirs(checkpoint_dir, exist_ok=True)

        # Check vLLM mode — server mode requires launching `trl vllm-serve`
        grpo_cfg = OmegaConf.to_container(
            OmegaConf.select(cfg, "training.grpo"), resolve=True
        )
        vllm_mode = grpo_cfg.get("vllm_mode", "colocate")
        use_online_rground = grpo_cfg.get("online_rground", False)

        if use_online_rground and vllm_mode == "server":
            raise ValueError(
                "online_rground and vllm_mode='server' are mutually exclusive. "
                "online_rground manages its own GPU allocation and auxiliary servers; "
                "set vllm_mode='colocate' when using online_rground."
            )

        vllm_server_proc = None
        embedding_server_proc = None
        judge_server_proc = None

        # Determine external vs managed server mode for online R_ground
        embedding_server_url = ""
        judge_server_url = ""
        use_external_servers = False

        if use_online_rground:
            embedding_server_url = str(
                grpo_cfg.get("embedding_server_url") or ""
            )
            judge_server_url = str(
                grpo_cfg.get("judge_server_url") or ""
            )
            use_external_servers = bool(
                embedding_server_url and judge_server_url
            )

            if bool(embedding_server_url) != bool(judge_server_url):
                raise ValueError(
                    "online_rground: either both embedding_server_url and "
                    "judge_server_url must be set, or neither. Got: "
                    f"embedding={embedding_server_url!r}, "
                    f"judge={judge_server_url!r}"
                )

        try:
            # ---- Online R_ground server setup ----
            if use_online_rground and use_external_servers:
                # External mode: connect to pre-existing servers.
                # All GPUs remain available for training.

                # Validate model names are set (needed for vLLM API calls)
                emb_model = str(
                    OmegaConf.select(cfg, "embedding_model.model_source", default=None)
                    or OmegaConf.select(cfg, "model.embedding_model_source", default=None)
                    or ""
                )
                judge_model = str(
                    OmegaConf.select(cfg, "judge_model.model_source") or ""
                )
                if not emb_model:
                    raise ValueError(
                        "online_rground requires embedding_model.model_source "
                        "(must match the model loaded by the external "
                        "embedding server)"
                    )
                if not judge_model:
                    raise ValueError(
                        "online_rground requires judge_model.model_source "
                        "(must match the model loaded by the external "
                        "judge server)"
                    )

                check_timeout = grpo_cfg.get(
                    "external_server_check_timeout", 300
                )
                print(f"[grpo_training] External server mode: "
                      f"embed={embedding_server_url}, "
                      f"judge={judge_server_url}")
                _check_external_server(
                    embedding_server_url, "embedding",
                    timeout=check_timeout,
                )
                _check_external_server(
                    judge_server_url, "judge",
                    timeout=check_timeout,
                )

            elif use_online_rground:
                # Managed mode: launch servers as subprocesses, allocate GPUs.
                visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
                gpu_list = (
                    [g.strip() for g in visible.split(",") if g.strip()]
                    if visible else []
                )
                if not gpu_list:
                    import torch
                    gpu_list = [str(i) for i in range(torch.cuda.device_count())]

                judge_tp = grpo_cfg.get("judge_server_tensor_parallel_size", 2)
                min_gpus = 1 + judge_tp + 1  # embed + judge + train
                if len(gpu_list) < min_gpus:
                    raise ValueError(
                        f"online_rground requires >= {min_gpus} GPUs "
                        f"(1 embed + {judge_tp} judge + 1 train), "
                        f"got {len(gpu_list)}: {gpu_list}"
                    )

                emb_gpus = [gpu_list[0]]            # 1 GPU for embedding
                judge_gpus = gpu_list[1:1 + judge_tp]  # 2 GPUs for judge
                train_gpus = gpu_list[1 + judge_tp:]   # remaining for training

                emb_port = grpo_cfg.get("embedding_server_port", 8001)
                judge_port = grpo_cfg.get("judge_server_port", 8002)

                emb_model = str(
                    OmegaConf.select(cfg, "embedding_model.model_source", default=None)
                    or OmegaConf.select(cfg, "model.embedding_model_source", default=None)
                    or ""
                )
                judge_model = str(
                    OmegaConf.select(cfg, "judge_model.model_source") or ""
                )

                if not emb_model:
                    raise ValueError(
                        "online_rground requires embedding_model.model_source"
                    )
                if not judge_model:
                    raise ValueError(
                        "online_rground requires judge_model.model_source"
                    )

                if not embeddings_dir:
                    print("[grpo_training] No embeddings dir provided; "
                          "NormRetriever will re-embed norms via the "
                          "embedding server at init.")

                print(f"[grpo_training] Online R_ground GPU allocation: "
                      f"embed={emb_gpus}, judge={judge_gpus}, "
                      f"train={train_gpus}")

                # Launch embedding server (vLLM auto-detects embedding model)
                embedding_server_proc = _launch_auxiliary_server(
                    model_path=emb_model,
                    port=emb_port,
                    gpu_ids=emb_gpus,
                    tp=1,
                    label="embedding",
                )

                # Launch judge server — cap context length to leave GPU
                # memory for concurrent request batching (72B AWQ on 2x A6000).
                judge_max_model_len = grpo_cfg.get(
                    "judge_server_max_model_len", 8192
                )
                judge_server_proc = _launch_auxiliary_server(
                    model_path=judge_model,
                    port=judge_port,
                    gpu_ids=judge_gpus,
                    tp=judge_tp,
                    label="judge",
                    max_model_len=judge_max_model_len,
                )

                # Restrict training to remaining GPUs
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(train_gpus)
                print(f"[grpo_training] Training restricted to GPUs: "
                      f"{train_gpus}")

                # Set URLs so the stage can resolve them uniformly
                embedding_server_url = f"http://localhost:{emb_port}"
                judge_server_url = f"http://localhost:{judge_port}"

            elif vllm_mode == "server" and grpo_cfg.get("use_vllm", True):
                # Legacy server mode: serves the base model for policy
                base_model_path = str(
                    OmegaConf.select(cfg, "model.model_source")
                )
                vllm_server_proc = _launch_vllm_server(
                    base_model_path, grpo_cfg
                )

            # Pass resolved server URLs to the stage via env vars.
            # The stage uses these as primary URL source, falling back
            # to localhost:port if unset (backward compat).
            if embedding_server_url:
                os.environ["GRPO_EMBEDDING_SERVER_URL"] = embedding_server_url
            if judge_server_url:
                os.environ["GRPO_JUDGE_SERVER_URL"] = judge_server_url

            run_grpo_training_stage(
                sft_checkpoint=sft_checkpoint,
                chunks_path=chunks_path,
                norm_universes_path=norm_universes_path or "",
                output_dir=checkpoint_dir,
                cfg=cfg,
                embeddings_dir=embeddings_dir or "",
                reward_cache_path=reward_cache_path or "",
            )
        finally:
            _shutdown_server(vllm_server_proc)
            _shutdown_server(embedding_server_proc)
            _shutdown_server(judge_server_proc)

        metadata: Dict[str, Any] = {
            "sft_checkpoint": sft_checkpoint,
            "checkpoint_dir": checkpoint_dir,
            "vllm_mode": vllm_mode,
            "online_rground": use_online_rground,
            "external_servers": use_external_servers,
            "embedding_server_url": embedding_server_url,
            "judge_server_url": judge_server_url,
            "rows": 0,
        }

        return StageResult(
            outputs={"checkpoint": checkpoint_dir},
            metadata=metadata,
        )
