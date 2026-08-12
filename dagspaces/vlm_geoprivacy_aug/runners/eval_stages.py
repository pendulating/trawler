"""Runner classes for the augmented VLM-GeoPrivacy evaluation stages.

Four runners come straight from ``vlm_geoprivacy_bench``, because this
dagspace runs the benchmark stages unchanged. Two subclass the benchmark
runner and change only the name that the parse-health report carries. The
rest are specific to this dagspace:

* the two inference runners call the hypothetical-aware stages here, which
  render a per-row prompt when the frame carries a ``hyp_id``;
* ``inpaint_hypotheticals`` expands one row into one row per variant;
* ``compute_hypothetical_metrics`` compares each variant with its baseline.

See ``dagspaces/common/runners/eval_base.py`` for the shared stage bodies.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from omegaconf import OmegaConf

from dagspaces.common.orchestrator import StageResult
from dagspaces.common.runners.base import StageRunner
from dagspaces.common.runners.eval_base import (
    EvalMetricsRunner,
    EvalStageRunner,
    write_dataset,
)
from dagspaces.vlm_geoprivacy_bench.runners.eval_stages import (
    ComputeMetricsRunner,
    GranularityJudgeRunner,
    LoadDatasetRunner,
)
from dagspaces.vlm_geoprivacy_bench.runners.eval_stages import (
    ParseFreeformRunner as _BenchParseFreeformRunner,
)
from dagspaces.vlm_geoprivacy_bench.runners.eval_stages import (
    ParseMCQRunner as _BenchParseMCQRunner,
)

__all__ = [
    "ComputeHypotheticalMetricsRunner",
    "ComputeMetricsRunner",
    "GranularityJudgeRunner",
    "InpaintHypotheticalsRunner",
    "LoadDatasetRunner",
    "ParseFreeformRunner",
    "ParseMCQRunner",
    "VLMFreeformInferenceRunner",
    "VLMMCQInferenceRunner",
]


class ParseMCQRunner(_BenchParseMCQRunner):
    """Same parse as the benchmark; the health report names this dagspace."""

    health_dagspace = "vlm_geoprivacy_aug"


class ParseFreeformRunner(_BenchParseFreeformRunner):
    """Same parse as the benchmark; the health report names this dagspace."""

    health_dagspace = "vlm_geoprivacy_aug"


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


class InpaintHypotheticalsRunner(StageRunner):
    """Expand each image row into one row per hypothetical variant."""

    stage_name = "inpaint_hypotheticals"

    def run(self, context: Any) -> StageResult:
        from ..hypotheticals import load_variants
        from ..stages.inpaint_hypotheticals import expand_with_hypotheticals

        df = pd.read_parquet(context.inputs["dataset"])

        hyp_cfg = getattr(context.cfg, "hypotheticals", None)
        raw_variants = OmegaConf.to_container(
            getattr(hyp_cfg, "variants", None) or [], resolve=True
        )
        include_bridges = bool(getattr(hyp_cfg, "include_bridges", True))
        variants = load_variants(raw_variants, include_bridges=include_bridges)

        result_df = expand_with_hypotheticals(df, variants)
        out_path = write_dataset(result_df, context.output_paths["dataset"])

        return StageResult(
            outputs={"dataset": out_path},
            metadata={
                "rows": len(result_df),
                "n_variants": len(variants),
                "variant_ids": [v.id for v in variants],
                "include_bridges": include_bridges,
            },
        )


class ComputeHypotheticalMetricsRunner(EvalMetricsRunner):
    stage_name = "compute_hypothetical_metrics"

    def compute(self, df: pd.DataFrame, context: Any) -> dict[str, Any]:
        from ..stages.hypothetical_metrics import compute_hypothetical_metrics

        return compute_hypothetical_metrics(df)

    def to_dataframe(self, metrics: dict[str, Any]) -> pd.DataFrame:
        from ..stages.hypothetical_metrics import (
            hypothetical_metrics_to_dataframe,
        )

        return hypothetical_metrics_to_dataframe(metrics)
