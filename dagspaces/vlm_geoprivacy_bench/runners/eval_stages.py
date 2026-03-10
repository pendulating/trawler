"""Runner classes for VLM-GeoPrivacyBench evaluation stages."""

from __future__ import annotations

import json
import os
from typing import Any, Dict

import pandas as pd
from omegaconf import OmegaConf

from dagspaces.common.runners.base import StageRunner
from dagspaces.common.orchestrator import StageResult


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

        exclude_sources = list(getattr(data_cfg, "exclude_sources", []) or [])

        df = load_dataset(
            annotations_path=str(data_cfg.annotations_path),
            metadata_path=str(data_cfg.metadata_path),
            image_dir=str(data_cfg.image_dir),
            exclude_sources=exclude_sources,
            sample_n=sample_n,
        )

        out_path = context.output_paths["dataset"]
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_parquet(out_path, index=False)

        return StageResult(
            outputs={"dataset": out_path},
            metadata={"rows": len(df)},
        )


class VLMMCQInferenceRunner(StageRunner):
    stage_name = "vlm_mcq_inference"

    def run(self, context: Any) -> StageResult:
        from ..stages.vlm_mcq_inference import run_mcq_inference

        input_path = context.inputs["dataset"]
        df = pd.read_parquet(input_path)

        result_df = run_mcq_inference(df, context.cfg)

        out_path = context.output_paths["dataset"]
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        result_df.to_parquet(out_path, index=False)

        return StageResult(
            outputs={"dataset": out_path},
            metadata={"rows": len(result_df)},
        )


class VLMFreeformInferenceRunner(StageRunner):
    stage_name = "vlm_freeform_inference"

    def run(self, context: Any) -> StageResult:
        from ..stages.vlm_freeform_inference import run_freeform_inference

        input_path = context.inputs["dataset"]
        df = pd.read_parquet(input_path)

        result_df = run_freeform_inference(df, context.cfg)

        out_path = context.output_paths["dataset"]
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        result_df.to_parquet(out_path, index=False)

        return StageResult(
            outputs={"dataset": out_path},
            metadata={"rows": len(result_df)},
        )


class ParseMCQRunner(StageRunner):
    stage_name = "parse_mcq"

    def run(self, context: Any) -> StageResult:
        from ..stages.parse_responses import parse_mcq_responses

        input_path = context.inputs["dataset"]
        df = pd.read_parquet(input_path)

        result_df = parse_mcq_responses(df)

        out_path = context.output_paths["dataset"]
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        result_df.to_parquet(out_path, index=False)

        return StageResult(
            outputs={"dataset": out_path},
            metadata={"rows": len(result_df)},
        )


class ParseFreeformRunner(StageRunner):
    stage_name = "parse_freeform"

    def run(self, context: Any) -> StageResult:
        from ..stages.parse_responses import parse_freeform_responses

        input_path = context.inputs["dataset"]
        df = pd.read_parquet(input_path)

        result_df = parse_freeform_responses(df)

        out_path = context.output_paths["dataset"]
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        result_df.to_parquet(out_path, index=False)

        return StageResult(
            outputs={"dataset": out_path},
            metadata={"rows": len(result_df)},
        )


class GranularityJudgeRunner(StageRunner):
    stage_name = "granularity_judge"

    def run(self, context: Any) -> StageResult:
        from ..stages.granularity_judge import run_granularity_judge

        input_path = context.inputs["dataset"]
        df = pd.read_parquet(input_path)

        result_df = run_granularity_judge(df, context.cfg)

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

        # Determine if free-form based on columns present
        free_form = "Q7_gen" in df.columns and "Q1_pred" not in df.columns

        metrics = compute_metrics(df, free_form=free_form)

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
