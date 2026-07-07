"""Runner classes for VLM-GeoPrivacyBench evaluation stages."""

from __future__ import annotations

import json
import os
from typing import Any, Dict

import pandas as pd
from omegaconf import OmegaConf

from dagspaces.common.runners.base import StageRunner
from dagspaces.common.orchestrator import StageResult
from dagspaces.common.eval_sanity import compute_parse_health
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
        input_n = len(df)

        result_df = parse_mcq_responses(df)

        out_path = context.output_paths["dataset"]
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        result_df.to_parquet(out_path, index=False)

        thresholds, patterns = sanity_overrides(context.cfg)
        report = compute_parse_health(
            result_df,
            dagspace="vlm_geoprivacy_aug",
            stage=self.stage_name,
            model=task_model_name(context.cfg),
            status_col="parse_status",
            completion_col="generated_text",
            finish_reason_col="finish_reason",
            expected_input_n=input_n,
            refusal_patterns=patterns,
            thresholds=thresholds,
        )
        metadata: Dict[str, Any] = {"rows": len(result_df)}
        log_sanity_to_context(context, report, metadata=metadata)
        return StageResult(outputs={"dataset": out_path}, metadata=metadata)


class ParseFreeformRunner(StageRunner):
    stage_name = "parse_freeform"

    def run(self, context: Any) -> StageResult:
        from ..stages.parse_responses import parse_freeform_responses

        input_path = context.inputs["dataset"]
        df = pd.read_parquet(input_path)
        input_n = len(df)

        result_df = parse_freeform_responses(df)

        out_path = context.output_paths["dataset"]
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        result_df.to_parquet(out_path, index=False)

        thresholds, patterns = sanity_overrides(context.cfg)
        report = compute_parse_health(
            result_df,
            dagspace="vlm_geoprivacy_aug",
            stage=self.stage_name,
            model=task_model_name(context.cfg),
            status_col="parse_status",
            completion_col="generated_text",
            finish_reason_col="finish_reason",
            expected_input_n=input_n,
            refusal_patterns=patterns,
            thresholds=thresholds,
        )
        metadata: Dict[str, Any] = {"rows": len(result_df)}
        log_sanity_to_context(context, report, metadata=metadata)
        return StageResult(outputs={"dataset": out_path}, metadata=metadata)


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


class InpaintHypotheticalsRunner(StageRunner):
    stage_name = "inpaint_hypotheticals"

    def run(self, context: Any) -> StageResult:
        from ..hypotheticals import load_variants
        from ..stages.inpaint_hypotheticals import expand_with_hypotheticals

        input_path = context.inputs["dataset"]
        df = pd.read_parquet(input_path)

        hyp_cfg = getattr(context.cfg, "hypotheticals", None)
        raw_variants = OmegaConf.to_container(
            getattr(hyp_cfg, "variants", None) or [], resolve=True
        )
        include_bridges = bool(getattr(hyp_cfg, "include_bridges", True))
        variants = load_variants(raw_variants, include_bridges=include_bridges)

        result_df = expand_with_hypotheticals(df, variants)

        out_path = context.output_paths["dataset"]
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        result_df.to_parquet(out_path, index=False)

        return StageResult(
            outputs={"dataset": out_path},
            metadata={
                "rows": len(result_df),
                "n_variants": len(variants),
                "variant_ids": [v.id for v in variants],
                "include_bridges": include_bridges,
            },
        )


class ComputeHypotheticalMetricsRunner(StageRunner):
    stage_name = "compute_hypothetical_metrics"

    def run(self, context: Any) -> StageResult:
        from ..stages.hypothetical_metrics import (
            compute_hypothetical_metrics,
            hypothetical_metrics_to_dataframe,
        )

        input_path = context.inputs["dataset"]
        df = pd.read_parquet(input_path)

        metrics = compute_hypothetical_metrics(df)

        metrics_json_path = os.path.join(context.output_dir, "metrics.json")
        with open(metrics_json_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)

        metrics_df = hypothetical_metrics_to_dataframe(metrics)
        out_path = context.output_paths["dataset"]
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        metrics_df.to_parquet(out_path, index=False)

        # Metrics logging (W&B + structured log output) is handled by the
        # orchestrator via _log_eval_metrics when it sees metrics in metadata.

        return StageResult(
            outputs={"dataset": out_path, "metrics_json": metrics_json_path},
            metadata={"rows": len(metrics_df), "metrics": metrics},
        )
