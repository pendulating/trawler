"""EU AI Act classification stage runner."""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from ..orchestrator import (
    StageExecutionContext,
    StageResult,
    _collect_outputs,
    _convert_to_pandas_if_needed,
    _safe_log_table,
    prepare_stage_input,
)
from ..stages.classify_shared import get_required_input_columns
from ..stages.classify_eu_act import run_classification_eu_act
from .base import StageRunner


class ClassificationEUActRunner(StageRunner):
    """Runner for the classify_eu_act stage."""
    
    stage_name = "classify_eu_act"

    def run(self, context: StageExecutionContext) -> StageResult:
        """Execute the classify_eu_act stage."""
        dataset_path = context.inputs.get("dataset")
        if not dataset_path:
            raise ValueError(f"Node '{context.node.key}' requires 'dataset' input")
        cfg = context.cfg
        OmegaConf.update(cfg, "data.parquet_path", dataset_path, merge=True)
        # Note: Prompt injection is handled internally by run_classification_eu_act
        # Always load as pandas for gating; streaming not supported for this custom stage
        df, ds, use_streaming = prepare_stage_input(cfg, dataset_path, self.stage_name)
        in_df = df if df is not None else (ds.to_pandas() if hasattr(ds, "to_pandas") else None)
        if in_df is None:
            raise RuntimeError("Failed to load input dataset for classify_eu_act")
        # Gate by core_tuple_verified == True
        try:
            total_rows = len(in_df)
        except Exception:
            total_rows = None
        try:
            if "core_tuple_verified" in in_df.columns:
                mask = in_df["core_tuple_verified"].fillna(False).astype(bool)
            else:
                mask = pd.Series([False] * len(in_df))
            gated_df = in_df[mask].copy()
        except Exception:
            gated_df = in_df.iloc[0:0].copy()
        # Prefer one row per article to avoid duplicate classifications
        try:
            if "article_id" in gated_df.columns and len(gated_df):
                # Keep the first occurrence per article_id
                gated_df = gated_df.sort_values(by=["article_id"]).drop_duplicates(subset=["article_id"], keep="first")
        except Exception:
            pass

        # Drop verification-specific columns before handing off to Ray to avoid schema drift
        try:
            required_cols = get_required_input_columns(is_eu_profile=True, is_risks_benefits_profile=False)
            existing = [c for c in gated_df.columns if c in required_cols]
            if existing:
                gated_df = gated_df[existing].copy()
            else:
                gated_df = gated_df.copy()
        except Exception:
            gated_df = gated_df.copy()

        # Run classification with EU AI Act profile (assumed set via overrides)
        # EU-only classification (prompt injected inside wrapper)
        out = run_classification_eu_act(gated_df, cfg)

        out = _convert_to_pandas_if_needed(out)

        # Save outputs to disk
        if isinstance(out, pd.DataFrame) and "results" in context.output_paths:
            out.to_parquet(context.output_paths["results"], index=False)

        # Log results
        prefer_cols = [
            c for c in [
                "article_id",
                "too_vague_to_process",
                "eu_valid_input_count",
                "eu_ai_label",
                "eu_ai_desc",
                "eu_ai_relevant_text",
                "eu_ai_reason",
                "eu_ai_raw_json",
                "classification_mode",
                "latency_s",
                "token_usage_prompt",
                "token_usage_output",
                "token_usage_total",
                "article_path",
                "country",
                "year",
            ] if c in out.columns
        ]
        try:
            print(
                f"[classify_eu_act] Logging columns to wandb ({len(out.columns)} total): {list(out.columns)}",
                flush=True,
            )
        except Exception:
            pass
        _safe_log_table(context.logger, out, "classify_eu_act/results", prefer_cols=prefer_cols, panel_group="inspect_results")
        
        # Log simple counts
        if context.logger:
            try:
                context.logger.log_metrics({
                    "classify_eu_act/input_rows": int(total_rows or 0),
                    "classify_eu_act/gated_rows": int(len(gated_df)),
                    "classify_eu_act/output_rows": int(len(out)),
                })
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

