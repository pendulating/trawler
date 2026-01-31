"""Fetch Gutenberg stage runner."""

from __future__ import annotations

from typing import Any, Dict
import os
import pandas as pd

from ..orchestrator import (
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
        
        metadata: Dict[str, Any] = {
            "rows": len(df),
            "gutenberg_ids": cfg.data.get("gutenberg_ids", []),
        }
        
        outputs = _collect_outputs(
            context,
            {name: spec.optional for name, spec in context.node.outputs.items()},
        )
        return StageResult(outputs=outputs, metadata=metadata)

