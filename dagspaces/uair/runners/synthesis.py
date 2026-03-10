"""Synthesis stage runner."""

from __future__ import annotations

import os
from typing import Any, Dict

import pandas as pd
from omegaconf import DictConfig

from ..orchestrator import (
    StageExecutionContext,
    StageResult,
    _collect_outputs,
    _save_stage_outputs,
    _safe_log_table,
)
from ..stages.synthesis import run_synthesis_stage
from .base import StageRunner


class SynthesisRunner(StageRunner):
    """Runner for the synthesis stage."""
    
    stage_name = "synthesis"

    def run(self, context: StageExecutionContext) -> StageResult:
        """Execute the synthesis stage."""
        # Synthesis requires clusters; articles are optional (enables text-aware join when provided)
        clusters_path = context.inputs.get("clusters")
        articles_path = context.inputs.get("articles")
        
        if not clusters_path:
            raise ValueError(f"Node '{context.node.key}' requires 'clusters' input (topic assignments)")
        
        cfg = context.cfg
        
        # Load both datasets
        try:
            df_clusters = pd.read_parquet(clusters_path)
            print(f"Loaded {len(df_clusters)} rows from clusters: {clusters_path}", flush=True)
        except Exception as e:
            raise ValueError(f"Failed to load clusters from '{clusters_path}': {e}")
        
        df_articles = None
        if articles_path:
            try:
                if os.path.exists(articles_path):
                    df_articles = pd.read_parquet(articles_path)
                    print(f"Loaded {len(df_articles)} rows from articles: {articles_path}", flush=True)
                else:
                    print(f"Articles path not found; proceeding without articles: {articles_path}", flush=True)
            except Exception as e:
                print(f"Warning: Failed to load articles from '{articles_path}': {e}; proceeding without articles.", flush=True)
        
        # Pass both to synthesis stage
        out = run_synthesis_stage(df_clusters, cfg, logger=context.logger, articles_df=df_articles)
        
        _save_stage_outputs(out, context.output_paths)
        
        # Log results table to wandb (in inspect_results panel group)
        prefer_cols = ["cluster_id", "num_articles", "primary_risk_type", "risk_confidence", "synthesis_summary"]
        _safe_log_table(context.logger, out, "synthesis/results", prefer_cols=prefer_cols, panel_group="inspect_results")
        
        metadata: Dict[str, Any] = {
            "rows": len(out) if isinstance(out, pd.DataFrame) else None,
            "streaming": False,  # Synthesis uses dual-input join, not streaming
        }
        outputs = _collect_outputs(
            context,
            {name: spec.optional for name, spec in context.node.outputs.items()},
        )
        return StageResult(outputs=outputs, metadata=metadata)

