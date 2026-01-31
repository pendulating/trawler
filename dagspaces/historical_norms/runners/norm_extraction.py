"""Norm Extraction stage runner."""

from __future__ import annotations

from typing import Any, Dict
import pandas as pd
import ray

from ..orchestrator import (
    StageExecutionContext,
    StageResult,
    _collect_outputs,
    _save_stage_outputs,
    prepare_stage_input,
)
from ..stages.norm_extraction import run_norm_extraction_stage
from .base import StageRunner

class NormExtractionRunner(StageRunner):
    """Runner for the norm_extraction stage."""
    
    stage_name = "norm_extraction"

    def run(self, context: StageExecutionContext) -> StageResult:
        """Execute the norm_extraction stage."""
        dataset_path = context.inputs.get("dataset")
        if not dataset_path:
            raise ValueError(f"Node '{context.node.key}' requires 'dataset' input")
            
        cfg = context.cfg
        df, ds, use_streaming = prepare_stage_input(cfg, dataset_path, self.stage_name)
        in_obj = ds if use_streaming and ds is not None else df
        
        out = run_norm_extraction_stage(in_obj, cfg)
        
        if hasattr(out, "to_pandas"):
            # Disable automatic casting to TensorArray to avoid conversion failures
            # for complex nested numpy arrays in columns like ig20_activation_conditions
            ctx = ray.data.DataContext.get_current()
            ctx.enable_tensor_extension_casting = False
            out = out.to_pandas()
            
        _save_stage_outputs(out, context.output_paths)
        
        metadata: Dict[str, Any] = {
            "rows": len(out) if isinstance(out, pd.DataFrame) else None,
            "streaming": bool(use_streaming),
        }
        
        outputs = _collect_outputs(
            context,
            {name: spec.optional for name, spec in context.node.outputs.items()},
        )
        return StageResult(outputs=outputs, metadata=metadata)



