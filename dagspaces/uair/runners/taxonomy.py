"""Taxonomy stage runner."""

from __future__ import annotations

from typing import Any, Dict

from omegaconf import DictConfig, OmegaConf

from ..orchestrator import (
    StageExecutionContext,
    StageResult,
    _collect_outputs,
    _save_stage_outputs,
    _safe_log_table,
    prepare_stage_input,
)
from ..stages.taxonomy import run_taxonomy_stage
from .base import StageRunner


class TaxonomyRunner(StageRunner):
    """Runner for the taxonomy stage."""
    
    stage_name = "taxonomy"

    def run(self, context: StageExecutionContext) -> StageResult:
        """Execute the taxonomy stage."""
        dataset_path = context.inputs.get("dataset")
        if not dataset_path:
            raise ValueError(f"Node '{context.node.key}' requires 'dataset' input")
        cfg = context.cfg
        OmegaConf.update(cfg, "data.parquet_path", dataset_path, merge=True)
        df, _, _ = prepare_stage_input(cfg, dataset_path, self.stage_name)
        out = run_taxonomy_stage(df, cfg)
        _save_stage_outputs(out, context.output_paths)
        
        # Log results table to wandb (in inspect_results panel group)
        import pandas as pd
        if isinstance(out, pd.DataFrame) and context.logger:
            prefer_cols = ["article_id", "chunk_id", "taxonomy_json", "chunk_text", "article_path"]
            _safe_log_table(context.logger, out, "taxonomy/results", prefer_cols=prefer_cols, panel_group="inspect_results")
        
        metadata: Dict[str, Any] = {
            "rows": len(out) if isinstance(out, pd.DataFrame) else None,
            "streaming": False,
        }
        outputs = _collect_outputs(
            context,
            {name: spec.optional for name, spec in context.node.outputs.items()},
        )
        return StageResult(outputs=outputs, metadata=metadata)

