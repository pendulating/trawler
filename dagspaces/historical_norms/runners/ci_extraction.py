"""CI Extraction stage runner - Contextual Integrity structured tuple extraction."""

from __future__ import annotations

from dagspaces.common.runners.base import DataFrameStageRunner


class CIExtractionRunner(DataFrameStageRunner):
    """Runner for the ci_extraction stage.

    Converts CI reasoning traces into structured 5-component
    information flow tuples (Subject, Sender, Recipient,
    Information Type, Transmission Principle).
    """

    stage_name = "ci_extraction"

    def transform(self, df, cfg):
        from ..stages.ci_extraction import run_ci_extraction_stage

        return run_ci_extraction_stage(df, cfg)
