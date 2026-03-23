"""GRPO Training stage runner.

Supports two vLLM modes (configured via training.grpo.vllm_mode):

  "colocate" (default): Single-process, TRL manages vLLM internally.
    Just calls run_grpo_training_stage directly.

  "server": Runner launches `trl vllm-serve` as a subprocess on dedicated
    GPUs, then calls run_grpo_training_stage for training on remaining GPUs.
    The server subprocess is cleaned up after training completes.
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


def _shutdown_vllm_server(proc: Optional[subprocess.Popen]) -> None:
    """Gracefully shut down the vLLM server subprocess."""
    if proc is None or proc.poll() is not None:
        return
    print("[grpo_training] Shutting down vLLM server...")
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=30)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass


def _launch_auxiliary_server(
    model_path: str,
    port: int,
    gpu_ids: list,
    tp: int = 1,
    task: str = "generate",
    startup_timeout: int = 300,
    label: str = "auxiliary",
) -> subprocess.Popen:
    """Launch a vLLM server for auxiliary models (embedding or judge).

    Uses ``vllm serve`` (not ``trl vllm-serve``) since these are
    standalone inference servers, not TRL-managed policy models.

    Args:
        model_path: Path to the model.
        port: Port to serve on.
        gpu_ids: List of GPU indices to pin via CUDA_VISIBLE_DEVICES.
        tp: Tensor parallel size.
        task: vLLM task type ("generate" or "embed").
        startup_timeout: Max seconds to wait for health check.
        label: Human-readable label for log messages.

    Returns:
        Popen handle for cleanup.
    """
    cmd = [
        "vllm", "serve", model_path,
        "--port", str(port),
        "--tensor-parallel-size", str(tp),
        "--task", task,
        "--host", "0.0.0.0",
    ]
    # Judge model (AWQ) needs quantization flag
    if "awq" in model_path.lower() or "AWQ" in model_path:
        cmd.extend(["--quantization", "awq"])

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
    # Suppress tokenizer parallelism warnings in subprocess
    env["TOKENIZERS_PARALLELISM"] = "false"

    print(f"[grpo_training] Launching {label} server: {' '.join(cmd)}")
    print(f"[grpo_training]   GPUs: {gpu_ids}, port: {port}, task: {task}")

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

        sft_checkpoint = context.inputs.get("sft_checkpoint")
        dataset_path = context.inputs.get("dataset")
        reward_cache_path = context.inputs.get("reward_cache", "")
        norm_universes_path = context.inputs.get("norm_universes", "")
        embeddings_dir = context.inputs.get("embeddings", "")

        missing = []
        if not sft_checkpoint:
            missing.append("sft_checkpoint")
        if not dataset_path:
            missing.append("dataset")
        if missing:
            raise ValueError(
                f"Node '{context.node.key}' requires inputs: {missing}"
            )

        cfg = context.cfg
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

        vllm_server_proc = None
        embedding_server_proc = None
        judge_server_proc = None

        try:
            # ---- GPU allocation for online R_ground ----
            if use_online_rground:
                visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
                gpu_list = (
                    [g.strip() for g in visible.split(",") if g.strip()]
                    if visible else []
                )
                if not gpu_list:
                    import torch
                    gpu_list = [str(i) for i in range(torch.cuda.device_count())]

                if len(gpu_list) < 5:
                    raise ValueError(
                        f"online_rground requires >= 5 GPUs, "
                        f"got {len(gpu_list)}: {gpu_list}"
                    )

                emb_gpus = [gpu_list[0]]            # 1 GPU for embedding
                judge_tp = grpo_cfg.get("judge_server_tensor_parallel_size", 2)
                judge_gpus = gpu_list[1:1 + judge_tp]  # 2 GPUs for judge
                train_gpus = gpu_list[1 + judge_tp:]   # remaining for training

                emb_port = grpo_cfg.get("embedding_server_port", 8001)
                judge_port = grpo_cfg.get("judge_server_port", 8002)

                emb_model = str(
                    OmegaConf.select(cfg, "model.embedding_model_source") or ""
                )
                judge_model = str(
                    OmegaConf.select(cfg, "judge_model.model_source") or ""
                )

                if not emb_model:
                    raise ValueError(
                        "online_rground requires model.embedding_model_source"
                    )
                if not judge_model:
                    raise ValueError(
                        "online_rground requires judge_model.model_source"
                    )

                if not embeddings_dir:
                    raise ValueError(
                        "online_rground requires 'embeddings' input "
                        "(from norm_universe stage)"
                    )

                print(f"[grpo_training] Online R_ground GPU allocation: "
                      f"embed={emb_gpus}, judge={judge_gpus}, "
                      f"train={train_gpus}")

                # Launch embedding server (task=embed for vLLM embedding mode)
                embedding_server_proc = _launch_auxiliary_server(
                    model_path=emb_model,
                    port=emb_port,
                    gpu_ids=emb_gpus,
                    tp=1,
                    task="embed",
                    label="embedding",
                )

                # Launch judge server
                judge_server_proc = _launch_auxiliary_server(
                    model_path=judge_model,
                    port=judge_port,
                    gpu_ids=judge_gpus,
                    tp=judge_tp,
                    task="generate",
                    label="judge",
                )

                # Restrict training to remaining GPUs
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(train_gpus)
                print(f"[grpo_training] Training restricted to GPUs: "
                      f"{train_gpus}")

            elif vllm_mode == "server" and grpo_cfg.get("use_vllm", True):
                # Legacy server mode: serves the base model for policy
                base_model_path = str(
                    OmegaConf.select(cfg, "model.model_source")
                )
                vllm_server_proc = _launch_vllm_server(
                    base_model_path, grpo_cfg
                )

            run_grpo_training_stage(
                sft_checkpoint=sft_checkpoint,
                dataset_path=dataset_path,
                reward_cache_path=reward_cache_path or "",
                norm_universes_path=norm_universes_path or "",
                output_dir=checkpoint_dir,
                cfg=cfg,
                embeddings_dir=embeddings_dir or "",
            )
        finally:
            _shutdown_vllm_server(vllm_server_proc)
            _shutdown_vllm_server(embedding_server_proc)
            _shutdown_vllm_server(judge_server_proc)

        metadata: Dict[str, Any] = {
            "sft_checkpoint": sft_checkpoint,
            "checkpoint_dir": checkpoint_dir,
            "vllm_mode": vllm_mode,
            "online_rground": use_online_rground,
            "rows": 0,
        }

        return StageResult(
            outputs={"checkpoint": checkpoint_dir},
            metadata=metadata,
        )
