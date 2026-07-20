"""Stage-runner classes for the mmlu dagspace."""

from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd

from dagspaces.common.eval_sanity import compute_parse_health
from dagspaces.common.orchestrator import StageResult
from dagspaces.common.runners.base import StageRunner
from dagspaces.common.runners.sanity import (
    log_sanity_to_context,
    sanity_overrides,
    task_model_name,
)


class LoadDatasetRunner(StageRunner):
    stage_name = "load_dataset"

    def run(self, context: Any) -> StageResult:
        from ..stages.load_dataset import load_dataset

        cfg = context.cfg
        data_cfg = cfg.data

        sample_n = None
        runtime = getattr(cfg, "runtime", None)
        if runtime:
            sample_n = getattr(runtime, "sample_n", None)
            if sample_n is not None:
                sample_n = int(sample_n)

        prompt_cfg = getattr(cfg, "prompt", None)
        k_shot = int(getattr(prompt_cfg, "k_shot", 0) or 0) if prompt_cfg is not None else 0

        df = load_dataset(
            hf_dataset=str(getattr(data_cfg, "hf_dataset", "cais/mmlu")),
            hf_config=str(getattr(data_cfg, "hf_config", "all")),
            split=str(getattr(data_cfg, "split", "test")),
            hf_token=getattr(data_cfg, "hf_token", None),
            sample_n=sample_n,
            k_shot=k_shot,
        )

        out_path = context.output_paths["dataset"]
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_parquet(out_path, index=False)

        return StageResult(
            outputs={"dataset": out_path},
            metadata={"rows": len(df), "k_shot": k_shot},
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
        input_n = len(df)

        result_df = parse_responses(df)

        out_path = context.output_paths["dataset"]
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        result_df.to_parquet(out_path, index=False)

        thresholds, patterns = sanity_overrides(context.cfg)
        report = compute_parse_health(
            result_df,
            dagspace="mmlu",
            stage=self.stage_name,
            model=task_model_name(context.cfg),
            status_col="parse_status",
            completion_col="generated_text",
            label_col="prediction_letter",
            finish_reason_col="finish_reason",
            expected_input_n=input_n,
            refusal_patterns=patterns,
            thresholds=thresholds,
        )
        metadata: dict[str, Any] = {"rows": len(result_df)}
        log_sanity_to_context(context, report, metadata=metadata)
        return StageResult(outputs={"dataset": out_path}, metadata=metadata)


class ComputeMetricsRunner(StageRunner):
    stage_name = "compute_metrics"

    def run(self, context: Any) -> StageResult:
        from ..stages.compute_metrics import compute_metrics, metrics_to_dataframe

        input_path = context.inputs["dataset"]
        df = pd.read_parquet(input_path)
        metrics = compute_metrics(df)

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
