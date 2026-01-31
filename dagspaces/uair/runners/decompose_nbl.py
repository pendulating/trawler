"""Decompose NBL stage runner."""

from __future__ import annotations

from typing import Any, Dict

from omegaconf import DictConfig, OmegaConf

from ..orchestrator import (
    StageExecutionContext,
    StageResult,
    _collect_outputs,
    _convert_to_pandas_if_needed,
    _save_stage_outputs,
    _safe_log_table,
    prepare_stage_input,
)
from ..stages.decompose_nbl import run_decomposition_stage_nbl
from .base import StageRunner


class DecomposeNBLRunner(StageRunner):
    """Runner for the decompose_nbl stage."""
    
    stage_name = "decompose_nbl"

    def run(self, context: StageExecutionContext) -> StageResult:
        """Execute the decompose_nbl stage."""
        dataset_path = context.inputs.get("dataset")
        if not dataset_path:
            raise ValueError(f"Node '{context.node.key}' requires 'dataset' input")
        cfg = context.cfg
        OmegaConf.update(cfg, "data.parquet_path", dataset_path, merge=True)
        df, ds, use_streaming = prepare_stage_input(cfg, dataset_path, self.stage_name)
        in_obj = ds if use_streaming and ds is not None else df
        out = run_decomposition_stage_nbl(in_obj, cfg)
        
        out = _convert_to_pandas_if_needed(out)
        _save_stage_outputs(out, context.output_paths)
        
        # Log results table to wandb (in inspect_results panel group)
        import pandas as pd
        if isinstance(out, pd.DataFrame) and context.logger:
            prefer_cols = ["article_id", "chunk_id", "chunk_text", "article_path"]
            _safe_log_table(context.logger, out, "decompose_nbl/results", prefer_cols=prefer_cols, panel_group="inspect_results")
        
        metadata: Dict[str, Any] = {
            "rows": len(out) if isinstance(out, pd.DataFrame) else None,
            "streaming": bool(use_streaming),
        }
        outputs = _collect_outputs(
            context,
            {name: spec.optional for name, spec in context.node.outputs.items()},
        )
        return StageResult(outputs=outputs, metadata=metadata)

