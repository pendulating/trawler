"""Runner classes for CIRL-Vignettes evaluation stages."""

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

        data_cfg = cfg.data
        json_path = str(getattr(data_cfg, "json_path", "")) or None

        # Allow config to select probing levels (default: both seed + vignette)
        probing_levels = None
        raw = getattr(data_cfg, "probing_levels", None)
        if raw is not None:
            from omegaconf import OmegaConf
            probing_levels = list(OmegaConf.to_container(raw, resolve=True)) if not isinstance(raw, list) else list(raw)

        df = load_dataset(
            json_path=json_path,
            probing_levels=probing_levels,
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

        result_df = parse_responses(df)

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

        metrics = compute_metrics(df)

        # Save metrics as JSON
        metrics_json_path = os.path.join(context.output_dir, "metrics.json")
        with open(metrics_json_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)

        # Save as parquet for pipeline compatibility
        metrics_df = metrics_to_dataframe(metrics)
        out_path = context.output_paths["dataset"]
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        metrics_df.to_parquet(out_path, index=False)

        return StageResult(
            outputs={"dataset": out_path, "metrics_json": metrics_json_path},
            metadata={"rows": len(metrics_df), "metrics": metrics},
        )


# ── Trajectory evaluation runners ──────────────────────────────────────


class TrajectoryInferenceRunner(StageRunner):
    stage_name = "trajectory_inference"

    def run(self, context: Any) -> StageResult:
        from ..stages.trajectory_inference import run_trajectory_inference

        input_path = context.inputs["dataset"]
        df = pd.read_parquet(input_path)

        result_df = run_trajectory_inference(df, context.cfg)

        out_path = context.output_paths["dataset"]
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        result_df.to_parquet(out_path, index=False)

        return StageResult(
            outputs={"dataset": out_path},
            metadata={"rows": len(result_df)},
        )


class JudgeLeakageRunner(StageRunner):
    stage_name = "judge_leakage"

    def run(self, context: Any) -> StageResult:
        from ..stages.judge_leakage import judge_leakage

        input_path = context.inputs["dataset"]
        df = pd.read_parquet(input_path)

        result_df = judge_leakage(df, context.cfg)

        out_path = context.output_paths["dataset"]
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        result_df.to_parquet(out_path, index=False)

        return StageResult(
            outputs={"dataset": out_path},
            metadata={"rows": len(result_df)},
        )


class JudgeHelpfulnessRunner(StageRunner):
    stage_name = "judge_helpfulness"

    def run(self, context: Any) -> StageResult:
        from ..stages.judge_helpfulness import judge_helpfulness

        input_path = context.inputs["dataset"]
        df = pd.read_parquet(input_path)

        result_df = judge_helpfulness(df, context.cfg)

        out_path = context.output_paths["dataset"]
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        result_df.to_parquet(out_path, index=False)

        return StageResult(
            outputs={"dataset": out_path},
            metadata={"rows": len(result_df)},
        )


class ComputeTrajectoryMetricsRunner(StageRunner):
    stage_name = "compute_trajectory_metrics"

    def run(self, context: Any) -> StageResult:
        from ..stages.compute_trajectory_metrics import (
            compute_trajectory_metrics,
            metrics_to_dataframe,
        )

        input_path = context.inputs["dataset"]
        df = pd.read_parquet(input_path)

        metrics = compute_trajectory_metrics(df)

        # Save metrics as JSON
        metrics_json_path = os.path.join(context.output_dir, "metrics.json")
        with open(metrics_json_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)

        # Save as parquet for pipeline compatibility
        metrics_df = metrics_to_dataframe(metrics)
        out_path = context.output_paths["dataset"]
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        metrics_df.to_parquet(out_path, index=False)

        return StageResult(
            outputs={"dataset": out_path, "metrics_json": metrics_json_path},
            metadata={"rows": len(metrics_df), "metrics": metrics},
        )
