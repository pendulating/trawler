"""Topic stage runner."""

from __future__ import annotations

import os
from typing import Any, Dict

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from ..orchestrator import (
    StageExecutionContext,
    StageResult,
    _collect_outputs,
    _save_stage_outputs,
    _safe_log_table,
    prepare_stage_input,
)
from ..stages.topic import run_topic_stage
from .base import StageRunner


class TopicRunner(StageRunner):
    """Runner for the topic stage."""
    
    stage_name = "topic"

    @staticmethod
    def _resolve_topic_path(path: str) -> str:
        """Resolve topic path, checking for common filenames if path is a directory."""
        if os.path.isdir(path):
            candidates = [
                os.path.join(path, "classify_relevant.parquet"),
                os.path.join(path, "classify_all.parquet"),
                os.path.join(path, "results.parquet"),
            ]
            for cand in candidates:
                if os.path.exists(cand):
                    return cand
        return path

    def run(self, context: StageExecutionContext) -> StageResult:
        """Execute the topic stage."""
        dataset_path = context.inputs.get("dataset")
        if not dataset_path:
            raise ValueError(f"Node '{context.node.key}' requires 'dataset' input")
        resolved_path = self._resolve_topic_path(dataset_path)
        
        cfg = context.cfg
        OmegaConf.update(cfg, "data.parquet_path", resolved_path, merge=True)
        df, _, _ = prepare_stage_input(cfg, resolved_path, self.stage_name)

        # Gate by core_tuple_verified == True if column exists
        total_rows = None
        gated_rows = None
        in_obj = df
        if isinstance(df, pd.DataFrame):
            try:
                total_rows = len(df)
                if "core_tuple_verified" in df.columns:
                    mask = df["core_tuple_verified"].fillna(False).astype(bool)
                    gated_df = df[mask].copy()
                    gated_rows = len(gated_df)
                    print(f"[topic] Gated by core_tuple_verified: {total_rows} → {gated_rows} rows", flush=True)
                    in_obj = gated_df
                else:
                    print(f"[topic] No core_tuple_verified column found, keeping all {total_rows} rows", flush=True)
                    gated_rows = total_rows
            except Exception as e:
                print(f"[topic] Error gating by core_tuple_verified: {e}, proceeding with unfiltered data", flush=True)
                gated_rows = total_rows

        out = run_topic_stage(in_obj, cfg, logger=context.logger)
        _save_stage_outputs(out, context.output_paths)
        
        # Log results table and plots to wandb
        if isinstance(out, pd.DataFrame) and context.logger:
            prefer_cols = ["unit_id", "topic_id", "topic_prob", "topic_top_terms", "article_keywords", "plot_x", "plot_y"]
            _safe_log_table(context.logger, out, "topic/results", prefer_cols=prefer_cols, panel_group="inspect_results")
            
            try:
                # Log plotly visualization
                from ..stages.topic_plot import log_cluster_scatter_plotly_to_wandb
                log_cluster_scatter_plotly_to_wandb(out, context.logger, title="topic_cluster_map")
            except Exception as e:
                print(f"Warning: Failed to log topic plot to wandb: {e}", flush=True)
        
        # Log gating metrics
        if context.logger:
            try:
                metrics = {
                    "topic/output_rows": int(len(out)) if isinstance(out, pd.DataFrame) else 0,
                }
                if total_rows is not None:
                    metrics["topic/input_rows"] = int(total_rows)
                if gated_rows is not None:
                    metrics["topic/gated_rows"] = int(gated_rows)
                context.logger.log_metrics(metrics)
            except Exception:
                pass
        
        metadata: Dict[str, Any] = {
            "rows": len(out) if isinstance(out, pd.DataFrame) else None,
            "streaming": False,
        }
        outputs = _collect_outputs(
            context,
            {name: spec.optional for name, spec in context.node.outputs.items()},
        )
        return StageResult(outputs=outputs, metadata=metadata)

