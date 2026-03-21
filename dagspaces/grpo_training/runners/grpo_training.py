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
        reward_cache_path = context.inputs.get("reward_cache")
        norm_universes_path = context.inputs.get("norm_universes")

        missing = []
        if not sft_checkpoint:
            missing.append("sft_checkpoint")
        if not dataset_path:
            missing.append("dataset")
        if missing:
            raise ValueError(
                f"Node '{context.node.key}' requires inputs: {missing}"
            )
        # reward_cache and norm_universes are optional
        if not reward_cache_path:
            reward_cache_path = ""
        if not norm_universes_path:
            norm_universes_path = ""

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
        vllm_server_proc = None

        try:
            if vllm_mode == "server" and grpo_cfg.get("use_vllm", True):
                # Server mode serves the base model; LoRA is applied by GRPOTrainer
                base_model_path = str(OmegaConf.select(cfg, "model.model_source"))
                vllm_server_proc = _launch_vllm_server(base_model_path, grpo_cfg)

            run_grpo_training_stage(
                sft_checkpoint=sft_checkpoint,
                dataset_path=dataset_path,
                reward_cache_path=reward_cache_path,
                norm_universes_path=norm_universes_path,
                output_dir=checkpoint_dir,
                cfg=cfg,
            )
        finally:
            _shutdown_vllm_server(vllm_server_proc)

        metadata: Dict[str, Any] = {
            "sft_checkpoint": sft_checkpoint,
            "checkpoint_dir": checkpoint_dir,
            "vllm_mode": vllm_mode,
            "rows": 0,
        }

        return StageResult(
            outputs={"checkpoint": checkpoint_dir},
            metadata=metadata,
        )
