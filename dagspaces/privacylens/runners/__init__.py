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
            AgentActionInferenceRunner,
            ComputeMetricsRunner,
            HelpfulnessJudgeBatchExportRunner,
            HelpfulnessJudgeInferenceRunner,
            LeakageJudgeBatchExportRunner,
            LeakageJudgeInferenceRunner,
            LoadDatasetRunner,
            PerturbCultureRunner,
            PrivacylensFinalizeAsyncRunner,
            QAProbeInferenceRunner,
            RecoveredActionsRunner,
        )

        # PrivacyLens-under-CIRL-protocol stages (ported from the retired
        # cirl_vignettes dagspace — (A)/(B) probing + trajectory leakage on the
        # same PrivacyLens-493 cases, with the CIRL paper prompt framing).
        from ..cirl_protocol.runners import (
            CirlTrajectoryFinalizeAsyncRunner,
            ComputeMetricsRunner as CirlComputeMetricsRunner,
            ComputeTrajectoryMetricsRunner as CirlComputeTrajectoryMetricsRunner,
            JudgeHelpfulnessBatchExportRunner as CirlHelpfulnessJudgeBatchExportRunner,
            JudgeHelpfulnessRunner as CirlJudgeHelpfulnessRunner,
            JudgeLeakageBatchExportRunner as CirlLeakageJudgeBatchExportRunner,
            JudgeLeakageRunner as CirlJudgeLeakageRunner,
            LLMInferenceRunner as CirlLLMInferenceRunner,
            LoadDatasetRunner as CirlLoadDatasetRunner,
            ParseResponsesRunner as CirlParseResponsesRunner,
            TrajectoryInferenceRunner as CirlTrajectoryInferenceRunner,
        )

        _STAGE_REGISTRY = {
            # Clean PrivacyLens evaluation pipeline
            "load_dataset": LoadDatasetRunner(),
            "perturb_culture": PerturbCultureRunner(),
            "qa_probe_inference": QAProbeInferenceRunner(),
            "agent_action_inference": AgentActionInferenceRunner(),
            "leakage_judge_inference": LeakageJudgeInferenceRunner(),
            "helpfulness_judge_inference": HelpfulnessJudgeInferenceRunner(),
            "leakage_judge_batch_export": LeakageJudgeBatchExportRunner(),
            "helpfulness_judge_batch_export": HelpfulnessJudgeBatchExportRunner(),
            "compute_metrics": ComputeMetricsRunner(),
            # Async-judge finalize: drain output.jsonl + parse + metrics.
            "privacylens_finalize_async": PrivacylensFinalizeAsyncRunner(),
            "privacylens_recovered_actions": RecoveredActionsRunner(),
            # PrivacyLens-under-CIRL-protocol
            "cirl_load_dataset": CirlLoadDatasetRunner(),
            "cirl_probe_inference": CirlLLMInferenceRunner(),
            "cirl_parse_responses": CirlParseResponsesRunner(),
            "cirl_compute_metrics": CirlComputeMetricsRunner(),
            "cirl_trajectory_inference": CirlTrajectoryInferenceRunner(),
            "cirl_judge_leakage": CirlJudgeLeakageRunner(),
            "cirl_judge_helpfulness": CirlJudgeHelpfulnessRunner(),
            "cirl_judge_leakage_batch_export": CirlLeakageJudgeBatchExportRunner(),
            "cirl_judge_helpfulness_batch_export": CirlHelpfulnessJudgeBatchExportRunner(),
            "cirl_compute_trajectory_metrics": CirlComputeTrajectoryMetricsRunner(),
            "cirl_finalize_async": CirlTrajectoryFinalizeAsyncRunner(),
        }
    return _STAGE_REGISTRY.copy()

