"""Runner classes for VLM-GeoPrivacyBench evaluation stages.

The read/transform/write bodies live in
``dagspaces/common/runners/eval_base.py``. Only the benchmark-specific calls
are here.

Note that the two parse runners leave ``label_col`` unset. This benchmark
holds its predictions in seven per-question columns (``Q1_pred`` …
``Q7_pred``), so no single column is THE label, and the parse-health report
runs without one.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from dagspaces.common.runners.eval_base import (
    EvalLoadRunner,
    EvalMetricsRunner,
    EvalParseRunner,
    EvalStageRunner,
    runtime_sample_n,
)


class LoadDatasetRunner(EvalLoadRunner):
    stage_name = "load_dataset"

    def load(self, context: Any) -> pd.DataFrame:
        from ..stages.load_dataset import load_dataset

        cfg = context.cfg
        data_cfg = cfg.data
        return load_dataset(
            annotations_path=str(data_cfg.annotations_path),
            metadata_path=str(data_cfg.metadata_path),
            image_dir=str(data_cfg.image_dir),
            exclude_sources=list(getattr(data_cfg, "exclude_sources", []) or []),
            sample_n=runtime_sample_n(cfg),
        )


class VLMMCQInferenceRunner(EvalStageRunner):
    stage_name = "vlm_mcq_inference"

    def transform(self, df: pd.DataFrame, context: Any) -> pd.DataFrame:
        from ..stages.vlm_mcq_inference import run_mcq_inference

        return run_mcq_inference(df, context.cfg)


class VLMFreeformInferenceRunner(EvalStageRunner):
    stage_name = "vlm_freeform_inference"

    def transform(self, df: pd.DataFrame, context: Any) -> pd.DataFrame:
        from ..stages.vlm_freeform_inference import run_freeform_inference

        return run_freeform_inference(df, context.cfg)


class ParseMCQRunner(EvalParseRunner):
    stage_name = "parse_mcq"
    health_dagspace = "vlm_geoprivacy"

    def transform(self, df: pd.DataFrame, context: Any) -> pd.DataFrame:
        from ..stages.parse_responses import parse_mcq_responses

        return parse_mcq_responses(df)


class ParseFreeformRunner(EvalParseRunner):
    stage_name = "parse_freeform"
    health_dagspace = "vlm_geoprivacy"

    def transform(self, df: pd.DataFrame, context: Any) -> pd.DataFrame:
        from ..stages.parse_responses import parse_freeform_responses

        return parse_freeform_responses(df)


class GranularityJudgeRunner(EvalStageRunner):
    stage_name = "granularity_judge"

    def transform(self, df: pd.DataFrame, context: Any) -> pd.DataFrame:
        from ..stages.granularity_judge import run_granularity_judge

        return run_granularity_judge(df, context.cfg)


class ComputeMetricsRunner(EvalMetricsRunner):
    stage_name = "compute_metrics"

    def compute(self, df: pd.DataFrame, context: Any) -> dict[str, Any]:
        from ..stages.compute_metrics import compute_metrics

        # Free-form runs carry the judged Q7 answer and no MCQ predictions.
        free_form = "Q7_gen" in df.columns and "Q1_pred" not in df.columns
        return compute_metrics(df, free_form=free_form)

    def to_dataframe(self, metrics: dict[str, Any]) -> pd.DataFrame:
        from ..stages.compute_metrics import metrics_to_dataframe

        return metrics_to_dataframe(metrics)
