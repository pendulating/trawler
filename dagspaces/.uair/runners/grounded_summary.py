"""Grounded summary stage runner."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from ..orchestrator import (
    StageExecutionContext,
    StageResult,
    _collect_outputs,
    _save_stage_outputs,
    _safe_log_table,
)
from ..stages.grounded_summary import run_grounded_summary_stage
from .base import StageRunner


class GroundedSummaryRunner(StageRunner):
    """Runner for the grounded summary stage."""

    stage_name = "grounded_summary"

    def run(self, context: StageExecutionContext) -> StageResult:
        classification_path = context.inputs.get("classification")
        if not classification_path:
            raise ValueError(f"Node '{context.node.key}' requires 'classification' input (classify_eu_act results)")

        cfg = context.cfg
        self._ensure_prompt_defaults(cfg)

        df_classification = self._load_parquet(classification_path, required=True, label="classification")
        df_decompose = self._load_parquet(context.inputs.get("decompose"), required=False, label="decompose")
        df_verify = self._load_parquet(context.inputs.get("verification"), required=False, label="verification")

        out = run_grounded_summary_stage(
            df_classification,
            cfg,
            logger=context.logger,
            decompose_df=df_decompose,
            verify_df=df_verify,
        )

        _save_stage_outputs(out, context.output_paths)

        prefer_cols = [
            col
            for col in [
                "article_id",
                "use_case_summary",
                "risk_classification",
                "risk_rationale",
                "company_or_startup",
                "actors_involved",
                "government_or_regulators",
                "deployment_stage",
                "deployment_location",
            ]
            if isinstance(out, pd.DataFrame) and col in out.columns
        ]
        _safe_log_table(
            context.logger,
            out,
            "grounded_summary/results",
            prefer_cols=prefer_cols or None,
            panel_group="inspect_results",
        )

        metadata: Dict[str, Any] = {
            "rows": len(out) if isinstance(out, pd.DataFrame) else None,
            "streaming": False,
        }
        outputs = _collect_outputs(
            context,
            {name: spec.optional for name, spec in context.node.outputs.items()},
        )
        return StageResult(outputs=outputs, metadata=metadata)

    @staticmethod
    def _load_parquet(path: Optional[str], required: bool, label: str) -> Optional[pd.DataFrame]:
        if not path:
            if required:
                raise ValueError(f"Required input '{label}' path is missing")
            return None
        if not os.path.exists(path):
            if required:
                raise FileNotFoundError(f"Input '{label}' not found at '{path}'")
            print(f"[grounded_summary] Optional input '{label}' not found at '{path}', continuing without it.", flush=True)
            return None
        try:
            df = pd.read_parquet(path)
            print(f"[grounded_summary] Loaded {len(df)} rows from {label}: {path}", flush=True)
            return df
        except Exception as exc:
            if required:
                raise RuntimeError(f"Failed to load required input '{label}' from '{path}': {exc}") from exc
            print(f"[grounded_summary] Warning: failed to load optional input '{label}' from '{path}': {exc}", flush=True)
            return None

    @staticmethod
    def _ensure_prompt_defaults(cfg: DictConfig) -> None:
        if OmegaConf.select(cfg, "prompt_grounded_summary") is not None:
            return
        try:
            base_dir = os.path.dirname(os.path.dirname(__file__))
            prompt_path = os.path.join(base_dir, "conf", "prompt", "grounded_summary.yaml")
            if os.path.exists(prompt_path):
                prompt_cfg = OmegaConf.load(prompt_path)
                section = prompt_cfg.get("prompt_grounded_summary")
                if section is not None:
                    OmegaConf.update(cfg, "prompt_grounded_summary", section, merge=True)
                    print("[grounded_summary] Injected default grounded_summary prompt configuration.", flush=True)
        except Exception as exc:
            print(f"[grounded_summary] Warning: failed to inject default prompt configuration: {exc}", flush=True)


