"""Stage runner registry and exports for contextual integrity evaluation."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import StageRunner

__all__ = [
    "StageRunner",
    "get_stage_registry",
]

_STAGE_REGISTRY: dict[str, "StageRunner"] | None = None


def get_stage_registry() -> dict[str, "StageRunner"]:
    """Get the stage registry mapping stage names to runner instances."""
    global _STAGE_REGISTRY
    if _STAGE_REGISTRY is None:
        # New clean PrivacyLens eval runners
        from .privacylens_runners import (
            LoadDatasetRunner,
            QAProbeInferenceRunner,
            AgentActionInferenceRunner,
            LeakageJudgeInferenceRunner,
            ComputeMetricsRunner,
        )

        _STAGE_REGISTRY = {
            # Clean PrivacyLens evaluation pipeline
            "load_dataset": LoadDatasetRunner(),
            "qa_probe_inference": QAProbeInferenceRunner(),
            "agent_action_inference": AgentActionInferenceRunner(),
            "leakage_judge_inference": LeakageJudgeInferenceRunner(),
            "compute_metrics": ComputeMetricsRunner(),
        }
    return _STAGE_REGISTRY.copy()

