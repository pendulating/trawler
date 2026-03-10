"""Stage runner registry for VLM-GeoPrivacyBench dagspace."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dagspaces.common.runners.base import StageRunner

__all__ = ["get_stage_registry"]

_STAGE_REGISTRY: dict[str, "StageRunner"] | None = None


def get_stage_registry() -> dict[str, "StageRunner"]:
    """Get the stage registry mapping stage names to runner instances."""
    global _STAGE_REGISTRY
    if _STAGE_REGISTRY is None:
        from .eval_stages import (
            ComputeMetricsRunner,
            GranularityJudgeRunner,
            LoadDatasetRunner,
            ParseFreeformRunner,
            ParseMCQRunner,
            VLMFreeformInferenceRunner,
            VLMMCQInferenceRunner,
        )

        _STAGE_REGISTRY = {
            "load_dataset": LoadDatasetRunner(),
            "vlm_mcq_inference": VLMMCQInferenceRunner(),
            "vlm_freeform_inference": VLMFreeformInferenceRunner(),
            "parse_mcq": ParseMCQRunner(),
            "parse_freeform": ParseFreeformRunner(),
            "granularity_judge": GranularityJudgeRunner(),
            "compute_metrics": ComputeMetricsRunner(),
        }
    return _STAGE_REGISTRY.copy()
