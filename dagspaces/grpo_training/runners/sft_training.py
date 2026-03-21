"""SFT Training stage runner.

Launches training via `accelerate launch` for proper multi-GPU DDP support.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from typing import Any, Dict

from omegaconf import OmegaConf

from dagspaces.common.orchestrator import (
    StageExecutionContext,
    StageResult,
)
from .base import StageRunner


class SFTTrainingRunner(StageRunner):
    """Runner for the sft_training stage.

    Spawns `accelerate launch` with the number of GPUs visible to the job.
    Falls back to single-process if only 1 GPU is available.
    """

    stage_name = "sft_training"

    def run(self, context: StageExecutionContext) -> StageResult:
        dataset_path = context.inputs.get("dataset")
        if not dataset_path:
            raise ValueError(f"Node '{context.node.key}' requires 'dataset' input")

        cfg = context.cfg
        base_model = str(OmegaConf.select(cfg, "model.model_source"))
        checkpoint_dir = context.output_paths.get("checkpoint")
        if not checkpoint_dir:
            checkpoint_dir = os.path.join(context.output_dir, "checkpoint")

        os.makedirs(checkpoint_dir, exist_ok=True)

        import torch
        n_gpus = torch.cuda.device_count()

        if n_gpus > 1:
            self._run_with_accelerate(
                dataset_path, base_model, checkpoint_dir, cfg, n_gpus
            )
        else:
            from ..stages.sft_training import run_sft_training_stage
            run_sft_training_stage(dataset_path, base_model, checkpoint_dir, cfg)

        metadata: Dict[str, Any] = {
            "base_model": base_model,
            "checkpoint_dir": checkpoint_dir,
            "num_gpus": n_gpus,
            "rows": 0,
        }

        return StageResult(
            outputs={"checkpoint": checkpoint_dir},
            metadata=metadata,
        )

    def _run_with_accelerate(
        self,
        dataset_path: str,
        base_model: str,
        checkpoint_dir: str,
        cfg: Any,
        n_gpus: int,
    ) -> None:
        """Launch training via accelerate for multi-GPU DDP."""
        # Serialize config to a temp file so the subprocess can load it
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, prefix="sft_cfg_"
        ) as f:
            json.dump(cfg_dict, f)
            cfg_path = f.name

        # Write a small launcher script that the subprocess will execute
        launcher_script = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "stages", "_sft_accelerate_entry.py",
        )
        launcher_script = os.path.normpath(launcher_script)

        cmd = [
            sys.executable, "-m", "accelerate.commands.launch",
            "--num_processes", str(n_gpus),
            "--mixed_precision", "bf16",
            launcher_script,
            "--dataset_path", dataset_path,
            "--base_model", base_model,
            "--output_dir", checkpoint_dir,
            "--cfg_path", cfg_path,
        ]

        print(f"[sft_training] Launching DDP with {n_gpus} GPUs via accelerate")
        print(f"[sft_training] Command: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            env=os.environ.copy(),
            stdout=sys.stdout,
            stderr=sys.stderr,
        )

        # Clean up temp config
        try:
            os.unlink(cfg_path)
        except OSError:
            pass

        if result.returncode != 0:
            raise RuntimeError(
                f"accelerate launch failed with return code {result.returncode}"
            )
