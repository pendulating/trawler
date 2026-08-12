"""Runner classes for GoldCoin HIPAA evaluation stages.

The read/transform/write bodies live in
``dagspaces/common/runners/eval_base.py``. Only the GoldCoin-specific calls
are here. Every stage is keyed by ``prompt.task``.
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


def _task(cfg: Any) -> str:
    return str(cfg.prompt.task)


class LoadDatasetRunner(EvalLoadRunner):
    stage_name = "load_dataset"

    def load(self, context: Any) -> pd.DataFrame:
        from ..stages.load_dataset import load_dataset

        cfg = context.cfg
        return load_dataset(
            csv_path=str(cfg.data.csv_path),
            task=_task(cfg),
            sample_n=runtime_sample_n(cfg),
        )


class LLMInferenceRunner(EvalStageRunner):
    stage_name = "llm_inference"

    def transform(self, df: pd.DataFrame, context: Any) -> pd.DataFrame:
        from ..stages.llm_inference import run_llm_inference

        return run_llm_inference(df, context.cfg)


class ParseResponsesRunner(EvalParseRunner):
    stage_name = "parse_responses"
    health_dagspace = "goldcoin"
    label_col = "prediction"

    def transform(self, df: pd.DataFrame, context: Any) -> pd.DataFrame:
        from ..stages.parse_responses import parse_responses

        return parse_responses(df, task=_task(context.cfg))

    def health_stage(self, context: Any) -> str:
        return f"{self.stage_name}_{_task(context.cfg)}"

    def stage_metadata(self, context: Any, df: pd.DataFrame) -> dict[str, Any]:
        return {"task": _task(context.cfg)}


class ComputeMetricsRunner(EvalMetricsRunner):
    stage_name = "compute_metrics"

    def compute(self, df: pd.DataFrame, context: Any) -> dict[str, Any]:
        from ..stages.compute_metrics import compute_metrics

        return compute_metrics(df, task=_task(context.cfg))

    def to_dataframe(self, metrics: dict[str, Any]) -> pd.DataFrame:
        from ..stages.compute_metrics import metrics_to_dataframe

        return metrics_to_dataframe(metrics)
