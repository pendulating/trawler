"""Risks and Benefits classification stage runner."""

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
from ..stages.classify_risks_benefits import run_classification_risks_benefits
from .base import StageRunner


class ClassificationRisksBenefitsRunner(StageRunner):
    """Runner for the classify_risk_and_benefits stage."""
    
    stage_name = "classify_risk_and_benefits"

    def run(self, context: StageExecutionContext) -> StageResult:
        """Execute the classify_risk_and_benefits stage."""
        dataset_path = context.inputs.get("dataset")
        if not dataset_path:
            raise ValueError(f"Node '{context.node.key}' requires 'dataset' input")
        cfg = context.cfg
        OmegaConf.update(cfg, "data.parquet_path", dataset_path, merge=True)
        # Ensure proper profile
        try:
            OmegaConf.update(cfg, "runtime.classification_profile", "risks_and_benefits", merge=True)
            OmegaConf.update(cfg, "runtime.use_llm_classify", True, merge=True)
        except Exception:
            pass
        # Note: Prompt injection is handled internally by run_classification_risks_benefits
        df, ds, use_streaming = prepare_stage_input(cfg, dataset_path, self.stage_name)
        in_df = df if df is not None else (ds.to_pandas() if hasattr(ds, "to_pandas") else None)
        if in_df is None:
            raise RuntimeError("Failed to load input dataset for classify_risk_and_benefits")
        # Gate by core_tuple_verified == True to mirror EU AI Act classification requirements
        try:
            if "core_tuple_verified" in in_df.columns:
                mask = in_df["core_tuple_verified"].fillna(False).astype(bool)
            else:
                mask = pd.Series([False] * len(in_df))
            gated_df = in_df[mask].copy()
        except Exception:
            gated_df = in_df.iloc[0:0].copy()
        in_df = gated_df
        # Prefer one row per article to avoid duplicate classifications
        try:
            if "article_id" in in_df.columns and len(in_df):
                in_df = in_df.sort_values(by=["article_id"]).drop_duplicates(subset=["article_id"], keep="first")
        except Exception:
            pass

        try:
            required_cols = get_required_input_columns(is_eu_profile=False, is_risks_benefits_profile=True)
            existing = [c for c in in_df.columns if c in required_cols]
            if existing:
                in_df = in_df[existing].copy()
        except Exception:
            pass
        # Risks & Benefits classification (prompt injected inside wrapper)
        out = run_classification_risks_benefits(in_df, cfg)

        out = _convert_to_pandas_if_needed(out)

        # Save outputs to disk
        if isinstance(out, pd.DataFrame) and "results" in context.output_paths:
            out.to_parquet(context.output_paths["results"], index=False)

        # Log results
        prefer_cols = [
            c for c in [
                "article_id",
                "too_vague_to_process",
                "rb_desc",
                "rb_human_rights",
                "rb_sdgs",
                "rb_additional",
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
                f"[classify_risk_and_benefits] Logging columns to wandb ({len(out.columns)} total): {list(out.columns)}",
                flush=True,
            )
        except Exception:
            pass
        _safe_log_table(context.logger, out, "classify_risk_and_benefits/results", prefer_cols=prefer_cols, panel_group="inspect_results")

        metadata: Dict[str, Any] = {
            "rows": len(out) if isinstance(out, pd.DataFrame) else None,
            "streaming": False,
        }
        outputs = _collect_outputs(
            context,
            {name: spec.optional for name, spec in context.node.outputs.items()},
        )
        return StageResult(outputs=outputs, metadata=metadata)

