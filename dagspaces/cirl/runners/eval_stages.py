"""Runner classes for the CIRL-729 action benchmark stages.

The read/transform/write bodies live in
``dagspaces/common/runners/eval_base.py``. Only the CIRL-specific calls are
here.
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
            parquet_path=str(getattr(data_cfg, "parquet_path", "")) or None,
            sample_n=runtime_sample_n(cfg),
            shuffle_seed=int(getattr(data_cfg, "shuffle_seed", 42)),
        )


class LLMInferenceRunner(EvalStageRunner):
    stage_name = "llm_inference"

    def transform(self, df: pd.DataFrame, context: Any) -> pd.DataFrame:
        from ..stages.llm_inference import run_llm_inference

        return run_llm_inference(df, context.cfg)


class ParseResponsesRunner(EvalParseRunner):
    stage_name = "parse_responses"
    health_dagspace = "cirl"
    label_col = "prediction"

    def transform(self, df: pd.DataFrame, context: Any) -> pd.DataFrame:
        from ..stages.parse_responses import parse_responses

        return parse_responses(df)


class ComputeMetricsRunner(EvalMetricsRunner):
    stage_name = "compute_metrics"

    def compute(self, df: pd.DataFrame, context: Any) -> dict[str, Any]:
        from ..stages.compute_metrics import compute_metrics

        return compute_metrics(df)

    def to_dataframe(self, metrics: dict[str, Any]) -> pd.DataFrame:
        from ..stages.compute_metrics import metrics_to_dataframe

        return metrics_to_dataframe(metrics)
