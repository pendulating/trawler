"""Norm Reasoning stage runner."""

from __future__ import annotations

from dagspaces.common.runners.base import DataFrameStageRunner


class NormReasoningRunner(DataFrameStageRunner):
    """Runner for the norm_reasoning stage."""

    stage_name = "norm_reasoning"

    def transform(self, df, cfg):
        from ..stages.norm_reasoning import run_norm_reasoning_stage

        return run_norm_reasoning_stage(df, cfg)
