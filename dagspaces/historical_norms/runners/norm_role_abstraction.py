"""Norm Role Abstraction stage runner."""

from __future__ import annotations

from dagspaces.common.runners.base import DataFrameStageRunner


class NormRoleAbstractionRunner(DataFrameStageRunner):
    """Runner for the norm_role_abstraction stage."""

    stage_name = "norm_role_abstraction"

    def transform(self, df, cfg):
        from ..stages.norm_role_abstraction import run_norm_role_abstraction_stage

        return run_norm_role_abstraction_stage(df, cfg)
