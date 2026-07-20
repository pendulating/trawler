"""Norm Extraction stage runner."""

from __future__ import annotations

from dagspaces.common.runners.base import DataFrameStageRunner


class NormExtractionRunner(DataFrameStageRunner):
    """Runner for the norm_extraction stage."""

    stage_name = "norm_extraction"

    def transform(self, df, cfg):
        from ..stages.norm_extraction import run_norm_extraction_stage

        return run_norm_extraction_stage(df, cfg)
