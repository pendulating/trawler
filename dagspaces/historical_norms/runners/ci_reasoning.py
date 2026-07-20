"""CI Reasoning stage runner."""

from __future__ import annotations

from dagspaces.common.runners.base import DataFrameStageRunner


class CIReasoningRunner(DataFrameStageRunner):
    """Runner for the ci_reasoning stage."""

    stage_name = "ci_reasoning"

    def transform(self, df, cfg):
        from ..stages.ci_reasoning import run_ci_reasoning_stage

        return run_ci_reasoning_stage(df, cfg)
