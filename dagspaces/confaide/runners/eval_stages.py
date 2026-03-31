"""Runner classes for CONFAIDE evaluation stages."""

from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd

from dagspaces.common.runners.base import StageRunner
from dagspaces.common.orchestrator import StageResult


class LoadDatasetRunner(StageRunner):
    stage_name = "load_dataset"

    def run(self, context: Any) -> StageResult:
        from ..stages.load_dataset import load_dataset

        cfg = context.cfg
        tier = str(cfg.prompt.tier)

        sample_n = None
        runtime = getattr(cfg, "runtime", None)
        if runtime:
            sample_n = getattr(runtime, "sample_n", None)
            if sample_n is not None:
                sample_n = int(sample_n)

        cache_dir = str(getattr(cfg.data, "cache_dir", "")) or None

        df = load_dataset(tier=tier, cache_dir=cache_dir, sample_n=sample_n)

        out_path = context.output_paths["dataset"]
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_parquet(out_path, index=False)

        return StageResult(
            outputs={"dataset": out_path},
            metadata={"rows": len(df)},
        )


class LLMInferenceRunner(StageRunner):
    stage_name = "llm_inference"

    def run(self, context: Any) -> StageResult:
        from ..stages.llm_inference import run_llm_inference

        input_path = context.inputs["dataset"]
        df = pd.read_parquet(input_path)

        result_df = run_llm_inference(df, context.cfg)

        out_path = context.output_paths["dataset"]
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        result_df.to_parquet(out_path, index=False)

        return StageResult(
            outputs={"dataset": out_path},
            metadata={"rows": len(result_df)},
        )


class ParseResponsesRunner(StageRunner):
    stage_name = "parse_responses"

    def run(self, context: Any) -> StageResult:
        from ..stages.parse_responses import parse_responses

        input_path = context.inputs["dataset"]
        df = pd.read_parquet(input_path)

        tier = str(context.cfg.prompt.tier)
        result_df = parse_responses(df, tier=tier)

        out_path = context.output_paths["dataset"]
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        result_df.to_parquet(out_path, index=False)

        return StageResult(
            outputs={"dataset": out_path},
            metadata={"rows": len(result_df)},
        )


class ComputeMetricsRunner(StageRunner):
    stage_name = "compute_metrics"

    def run(self, context: Any) -> StageResult:
        from ..stages.compute_metrics import compute_metrics, metrics_to_dataframe

        input_path = context.inputs["dataset"]
        df = pd.read_parquet(input_path)

        tier = str(context.cfg.prompt.tier)
        metrics = compute_metrics(df, tier=tier)

        metrics_json_path = os.path.join(context.output_dir, "metrics.json")
        with open(metrics_json_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)

        metrics_df = metrics_to_dataframe(metrics)
        out_path = context.output_paths["dataset"]
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        metrics_df.to_parquet(out_path, index=False)

        return StageResult(
            outputs={"dataset": out_path, "metrics_json": metrics_json_path},
            metadata={"rows": len(metrics_df), "metrics": metrics},
        )
