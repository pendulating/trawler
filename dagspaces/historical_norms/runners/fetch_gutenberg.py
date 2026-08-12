"""Fetch Gutenberg stage runner."""

from __future__ import annotations

import os
from typing import Any

# Import from common, NOT from ..orchestrator. These helpers moved to
# dagspaces.common.orchestrator and the local re-export was dropped, so
# `from ..orchestrator import _collect_outputs` raised ImportError. Because
# orchestrator.py builds _STAGE_REGISTRY at module scope, that broke
# `import dagspaces.historical_norms.cli` outright — i.e. the whole dagspace,
# not just this stage. Every sibling runner here migrated to
# DataFrameStageRunner; this one was left behind. Matches the grpo_training
# runners' convention.
from dagspaces.common.orchestrator import (
    StageExecutionContext,
    StageResult,
    _collect_outputs,
    _save_stage_outputs,
)
from ..stages.fetch_gutenberg import run_fetch_gutenberg
from .base import StageRunner


class FetchGutenbergRunner(StageRunner):
    """Runner for the fetch_gutenberg stage."""
    
    stage_name = "fetch_gutenberg"

    def run(self, context: StageExecutionContext) -> StageResult:
        """Execute the fetch_gutenberg stage."""
        cfg = context.cfg
        
        # Resolve the config file if provided as input
        config_path = context.inputs.get("config")
        if config_path and os.path.exists(config_path):
            import yaml
            with open(config_path, 'r') as f:
                data_config = yaml.safe_load(f)
                # Merge into cfg for the stage logic
                from omegaconf import OmegaConf
                cfg = OmegaConf.merge(cfg, {"data": data_config})
        
        # This stage uses IDs from merged config
        df = run_fetch_gutenberg(cfg)
        
        _save_stage_outputs(df, context.output_paths)
        
        metadata: dict[str, Any] = {
            "rows": len(df),
            "gutenberg_ids": cfg.data.get("gutenberg_ids", []),
        }
        
        outputs = _collect_outputs(
            context,
            {name: spec.optional for name, spec in context.node.outputs.items()},
        )
        return StageResult(outputs=outputs, metadata=metadata)

