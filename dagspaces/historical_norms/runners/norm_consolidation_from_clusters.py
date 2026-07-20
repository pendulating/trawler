"""Norm Consolidation from Clusters stage runner."""

from __future__ import annotations

from dagspaces.common.runners.base import DataFrameStageRunner


class NormConsolidationFromClustersRunner(DataFrameStageRunner):
    """Runner for the norm_consolidation_from_clusters stage."""

    stage_name = "norm_consolidation_from_clusters"

    def transform(self, df, cfg):
        from ..stages.norm_consolidation_from_clusters import (
            run_norm_consolidation_from_clusters_stage,
        )

        return run_norm_consolidation_from_clusters_stage(df, cfg)
