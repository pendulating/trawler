_STAGE_REGISTRY: dict[str, "StageRunner"] | None = None


def get_stage_registry() -> dict[str, "StageRunner"]:
    """Get the stage registry mapping stage names to runner instances."""
    global _STAGE_REGISTRY
    if _STAGE_REGISTRY is None:
        from .eval_stages import (
            LoadDatasetRunner,
            LLMInferenceRunner,
            ParseResponsesRunner,
            ComputeMetricsRunner,
            TrajectoryInferenceRunner,
            JudgeLeakageRunner,
            JudgeHelpfulnessRunner,
            ComputeTrajectoryMetricsRunner,
        )

        _STAGE_REGISTRY = {
            "load_dataset": LoadDatasetRunner(),
            "llm_inference": LLMInferenceRunner(),
            "parse_responses": ParseResponsesRunner(),
            "compute_metrics": ComputeMetricsRunner(),
            "trajectory_inference": TrajectoryInferenceRunner(),
            "judge_leakage": JudgeLeakageRunner(),
            "judge_helpfulness": JudgeHelpfulnessRunner(),
            "compute_trajectory_metrics": ComputeTrajectoryMetricsRunner(),
        }
    return _STAGE_REGISTRY.copy()
