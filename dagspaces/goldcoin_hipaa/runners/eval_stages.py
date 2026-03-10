"""Runner classes for GoldCoin HIPAA evaluation stages."""

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

        sample_n = None
        runtime = getattr(cfg, "runtime", None)
        if runtime:
            sample_n = getattr(runtime, "sample_n", None)
            if sample_n is not None:
                sample_n = int(sample_n)

        df = load_dataset(
            csv_path=str(cfg.data.csv_path),
            task=str(cfg.prompt.task),
            sample_n=sample_n,
        )

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

        task = str(context.cfg.prompt.task)
        result_df = parse_responses(df, task=task)

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

        task = str(context.cfg.prompt.task)
        metrics = compute_metrics(df, task=task)

        # Save metrics as JSON
        metrics_json_path = os.path.join(context.output_dir, "metrics.json")
        with open(metrics_json_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)

        # Save as parquet for pipeline compatibility
        metrics_df = metrics_to_dataframe(metrics)
        out_path = context.output_paths["dataset"]
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        metrics_df.to_parquet(out_path, index=False)

        # Metrics logging (W&B + structured log output) is handled by the
        # orchestrator via _log_eval_metrics when it sees metrics in metadata.

        return StageResult(
            outputs={"dataset": out_path, "metrics_json": metrics_json_path},
            metadata={"rows": len(metrics_df), "metrics": metrics},
        )
