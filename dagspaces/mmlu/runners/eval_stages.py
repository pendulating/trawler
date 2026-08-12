"""Stage-runner classes for the mmlu dagspace.

The read/transform/write bodies live in
``dagspaces/common/runners/eval_base.py``. Only the mmlu-specific calls are
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


def _k_shot(cfg: Any) -> int:
    prompt_cfg = getattr(cfg, "prompt", None)
    if prompt_cfg is None:
        return 0
    return int(getattr(prompt_cfg, "k_shot", 0) or 0)


class LoadDatasetRunner(EvalLoadRunner):
    stage_name = "load_dataset"

    def load(self, context: Any) -> pd.DataFrame:
        from ..stages.load_dataset import load_dataset

        data_cfg = context.cfg.data
        return load_dataset(
            hf_dataset=str(getattr(data_cfg, "hf_dataset", "cais/mmlu")),
            hf_config=str(getattr(data_cfg, "hf_config", "all")),
            split=str(getattr(data_cfg, "split", "test")),
            hf_token=getattr(data_cfg, "hf_token", None),
            sample_n=runtime_sample_n(context.cfg),
            k_shot=_k_shot(context.cfg),
        )

    def stage_metadata(self, context: Any, df: pd.DataFrame) -> dict[str, Any]:
        return {"k_shot": _k_shot(context.cfg)}


class LLMInferenceRunner(EvalStageRunner):
    stage_name = "llm_inference"

    def transform(self, df: pd.DataFrame, context: Any) -> pd.DataFrame:
        from ..stages.llm_inference import run_llm_inference

        return run_llm_inference(df, context.cfg)


class ParseResponsesRunner(EvalParseRunner):
    stage_name = "parse_responses"
    health_dagspace = "mmlu"
    label_col = "prediction_letter"

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
