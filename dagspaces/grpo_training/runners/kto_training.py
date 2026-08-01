"""KTO training stage runner (k-series, K2).

Runs in-process on one GPU: k-series arms are 1-GPU by design (plan §7 —
four arms in parallel, one GPU each; a 9B LoRA run with the implicit
adapter-disabled reference fits one A6000). No accelerate spawn needed.
"""

from __future__ import annotations

import os

from omegaconf import OmegaConf

from dagspaces.common.orchestrator import (
    StageExecutionContext,
    StageResult,
)

from .base import StageRunner


class KTOTrainingRunner(StageRunner):
    """Runner for the kto_training stage."""

    stage_name = "kto_training"

    def run(self, context: StageExecutionContext) -> StageResult:
        dataset_path = context.inputs.get("dataset")
        if not dataset_path:
            raise ValueError(
                f"Node '{context.node.key}' requires 'dataset' input")

        cfg = context.cfg
        base_model = OmegaConf.select(cfg, "training.kto.base_model")
        if not base_model:
            base_model = str(OmegaConf.select(cfg, "model.model_source"))
        checkpoint_dir = context.output_paths.get("checkpoint")
        if not checkpoint_dir:
            checkpoint_dir = os.path.join(context.output_dir, "checkpoint")
        os.makedirs(checkpoint_dir, exist_ok=True)

        import torch
        n_gpus = torch.cuda.device_count()
        if n_gpus > 1:
            print(f"[kto_training] {n_gpus} GPUs visible but k-series arms "
                  "are single-GPU by design; training uses GPU 0 only")

        from ..stages.kto_training import run_kto_training_stage
        run_meta = run_kto_training_stage(
            dataset_path=dataset_path,
            base_model=str(base_model),
            output_dir=checkpoint_dir,
            cfg=cfg,
            metadata_path=context.inputs.get("metadata"),
        )

        return StageResult(
            outputs={"checkpoint": checkpoint_dir},
            metadata=run_meta,
        )
