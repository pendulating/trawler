"""Norm Consolidation stage runner."""

from __future__ import annotations

from dagspaces.common.runners.base import DataFrameStageRunner


class NormConsolidationRunner(DataFrameStageRunner):
    """Runner for the norm_consolidation stage."""

    stage_name = "norm_consolidation"

    def transform(self, df, cfg):
        from ..stages.norm_consolidation import run_norm_consolidation_stage

        return run_norm_consolidation_stage(df, cfg)
