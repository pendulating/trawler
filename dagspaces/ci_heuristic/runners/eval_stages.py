"""Runner classes for ci_heuristic stages.

The read/transform/write bodies live in
``dagspaces/common/runners/eval_base.py``. Only the ci_heuristic-specific
calls are here.

``score_traversal`` keeps the plain ``StageRunner`` form. It reads TWO node
inputs, and one call returns both the metric dict and the output DataFrame,
so :class:`EvalMetricsRunner` — which derives its DataFrame from the metrics —
does not fit it.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd
from omegaconf import OmegaConf

from dagspaces.common.orchestrator import StageResult
from dagspaces.common.runners.base import StageRunner
from dagspaces.common.runners.eval_base import (
    EvalLoadRunner,
    EvalStageRunner,
    runtime_sample_n,
    write_dataset,
)


class LoadCasesRunner(EvalLoadRunner):
    stage_name = "load_cases"

    def _tiers(self, context: Any) -> list[str]:
        cases_cfg = getattr(context.cfg, "cases", None)
        return list(
            OmegaConf.to_container(
                getattr(cases_cfg, "tiers", None) or ["a", "c"], resolve=True
            )
        )

    def load(self, context: Any) -> pd.DataFrame:
        from ..stages.load_cases import load_cases

        cases_cfg = getattr(context.cfg, "cases", None)
        return load_cases(
            tiers=self._tiers(context),
            include_contaminated=bool(
                getattr(cases_cfg, "include_contaminated", True)
            ),
            corpus_root=str(getattr(cases_cfg, "corpus_root", "") or "") or None,
            sample_n=runtime_sample_n(context.cfg),
        )

    def stage_metadata(self, context: Any, df: pd.DataFrame) -> dict[str, Any]:
        return {
            "tiers": self._tiers(context),
            "by_tier": df["tier"].value_counts().to_dict(),
        }


class TraverseRunner(EvalStageRunner):
    stage_name = "traverse"

    def transform(self, df: pd.DataFrame, context: Any) -> pd.DataFrame:
        from ..stages.traverse import run_traversal

        return run_traversal(df, context.cfg)

    def stage_metadata(self, context: Any, df: pd.DataFrame) -> dict[str, Any]:
        # Parse-health summary for the orchestrator's metric logging. This
        # stage reports per-step rather than per-run, so it does not use
        # EvalParseRunner.
        by_step: dict[str, Any] = {}
        for step, sub in df.groupby("step"):
            by_step[str(step)] = {
                "parseable_rate": round(
                    float((sub["parse_status"] != "unparseable").mean()), 6
                ),
                "n": int(len(sub)),
            }
        return {
            "metrics": {
                "ladder_level": str(OmegaConf.select(context.cfg, "ladder.level")),
                "n_cases": int(df["case_id"].nunique()),
                "per_step_parse": by_step,
            }
        }


class TPProbeRunner(EvalStageRunner):
    stage_name = "tp_probe"

    def transform(self, df: pd.DataFrame, context: Any) -> pd.DataFrame:
        from ..stages.tp_probe import run_tp_probe

        return run_tp_probe(df, context.cfg)

    def stage_metadata(self, context: Any, df: pd.DataFrame) -> dict[str, Any]:
        return {
            "metrics": {
                "n_cases": int(len(df)),
                "parseable_rate": round(
                    float((df["parse_status"] != "unparseable").mean()), 6
                ),
                "mean_conditions": round(float(df["n_conditions"].mean()), 6),
            }
        }


class ScoreTraversalRunner(StageRunner):
    stage_name = "score_traversal"

    def run(self, context: Any) -> StageResult:
        from ..stages.score_traversal import score_traversals

        traverse_df = pd.read_parquet(context.inputs["dataset"])
        cases_df = pd.read_parquet(context.inputs["cases"])

        metrics, per_case_df = score_traversals(traverse_df, cases_df)

        metrics_json_path = os.path.join(context.output_dir, "metrics.json")
        with open(metrics_json_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)

        out_path = write_dataset(per_case_df, context.output_paths["dataset"])

        return StageResult(
            outputs={"dataset": out_path, "metrics_json": metrics_json_path},
            metadata={"rows": len(per_case_df), "metrics": metrics},
        )
