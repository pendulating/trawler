"""Runner classes for ci_heuristic stages."""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
from omegaconf import OmegaConf

from dagspaces.common.orchestrator import StageResult
from dagspaces.common.runners.base import StageRunner


class LoadCasesRunner(StageRunner):
    stage_name = "load_cases"

    def run(self, context: Any) -> StageResult:
        from ..stages.load_cases import load_cases

        cfg = context.cfg
        cases_cfg = getattr(cfg, "cases", None)
        tiers = list(OmegaConf.to_container(getattr(cases_cfg, "tiers", None) or ["a", "c"], resolve=True))
        include_contaminated = bool(getattr(cases_cfg, "include_contaminated", True))
        corpus_root = str(getattr(cases_cfg, "corpus_root", "") or "") or None

        sample_n = None
        runtime = getattr(cfg, "runtime", None)
        if runtime is not None and getattr(runtime, "sample_n", None) is not None:
            sample_n = int(runtime.sample_n)

        df = load_cases(
            tiers=tiers,
            include_contaminated=include_contaminated,
            corpus_root=corpus_root,
            sample_n=sample_n,
        )

        out_path = context.output_paths["dataset"]
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_parquet(out_path, index=False)

        return StageResult(
            outputs={"dataset": out_path},
            metadata={"rows": len(df), "tiers": tiers,
                      "by_tier": df["tier"].value_counts().to_dict()},
        )


class TraverseRunner(StageRunner):
    stage_name = "traverse"

    def run(self, context: Any) -> StageResult:
        from ..stages.traverse import run_traversal

        input_path = context.inputs["dataset"]
        df = pd.read_parquet(input_path)

        result_df = run_traversal(df, context.cfg)

        out_path = context.output_paths["dataset"]
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        result_df.to_parquet(out_path, index=False)

        # Parse-health summary for the orchestrator's metric logging
        by_step: dict[str, Any] = {}
        for step, sub in result_df.groupby("step"):
            by_step[str(step)] = {
                "parseable_rate": round(float((sub["parse_status"] != "unparseable").mean()), 6),
                "n": int(len(sub)),
            }
        metrics = {
            "ladder_level": str(OmegaConf.select(context.cfg, "ladder.level")),
            "n_cases": int(result_df["case_id"].nunique()),
            "per_step_parse": by_step,
        }

        return StageResult(
            outputs={"dataset": out_path},
            metadata={"rows": len(result_df), "metrics": metrics},
        )


class TPProbeRunner(StageRunner):
    stage_name = "tp_probe"

    def run(self, context: Any) -> StageResult:
        from ..stages.tp_probe import run_tp_probe

        input_path = context.inputs["dataset"]
        df = pd.read_parquet(input_path)

        result_df = run_tp_probe(df, context.cfg)

        out_path = context.output_paths["dataset"]
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        result_df.to_parquet(out_path, index=False)

        metrics = {
            "n_cases": int(len(result_df)),
            "parseable_rate": round(float((result_df["parse_status"] != "unparseable").mean()), 6),
            "mean_conditions": round(float(result_df["n_conditions"].mean()), 6),
        }
        return StageResult(
            outputs={"dataset": out_path},
            metadata={"rows": len(result_df), "metrics": metrics},
        )


class ScoreTraversalRunner(StageRunner):
    stage_name = "score_traversal"

    def run(self, context: Any) -> StageResult:
        import json

        from ..stages.score_traversal import score_traversals

        traverse_df = pd.read_parquet(context.inputs["dataset"])
        cases_df = pd.read_parquet(context.inputs["cases"])

        metrics, per_case_df = score_traversals(traverse_df, cases_df)

        metrics_json_path = os.path.join(context.output_dir, "metrics.json")
        with open(metrics_json_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)

        out_path = context.output_paths["dataset"]
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        per_case_df.to_parquet(out_path, index=False)

        return StageResult(
            outputs={"dataset": out_path, "metrics_json": metrics_json_path},
            metadata={"rows": len(per_case_df), "metrics": metrics},
        )
