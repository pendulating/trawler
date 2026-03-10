"""Norm Reasoning stage runner."""

from __future__ import annotations

from typing import Any, Dict
import pandas as pd

from ..orchestrator import (
    StageExecutionContext,
    StageResult,
    _collect_outputs,
    _save_stage_outputs,
    prepare_stage_input,
)
from ..stages.norm_reasoning import run_norm_reasoning_stage
from .base import StageRunner

class NormReasoningRunner(StageRunner):
    """Runner for the norm_reasoning stage."""
    
    stage_name = "norm_reasoning"

    def run(self, context: StageExecutionContext) -> StageResult:
        """Execute the norm_reasoning stage."""
        dataset_path = context.inputs.get("dataset")
        if not dataset_path:
            raise ValueError(f"Node '{context.node.key}' requires 'dataset' input")
            
        cfg = context.cfg
        df, _, _ = prepare_stage_input(cfg, dataset_path, self.stage_name)

        out = run_norm_reasoning_stage(df, cfg)

        # Explode the reasoning data into individual norms
        if isinstance(out, pd.DataFrame) and "reasoning_data" in out.columns:
            rows = []
            for _, row in out.iterrows():
                data = row.get("reasoning_data")
                if isinstance(data, dict):
                    # Use .get() to handle missing "norms" key gracefully
                    norms = data.get("norms", [])
                    if norms is None or (hasattr(norms, "__len__") and len(norms) == 0):
                        # Keep the row but mark it as no norms found
                        rows.append(row.to_dict())
                    else:
                        for i, norm in enumerate(norms):
                            new_row = row.to_dict()
                            new_row["norm_index"] = i
                            new_row["reasoning_trace"] = norm.get("reasoning", "")
                            new_row["norm_snippet"] = norm.get("original_text_snippet", "")
                            new_row["preliminary_normative_force"] = norm.get("preliminary_normative_force", "")
                            new_row["governs_information_flow"] = norm.get("governs_information_flow", None)
                            rows.append(new_row)
                else:
                    rows.append(row.to_dict())
            out = pd.DataFrame(rows)

        _save_stage_outputs(out, context.output_paths)

        metadata: Dict[str, Any] = {
            "rows": len(out) if isinstance(out, pd.DataFrame) else None,
            "streaming": False,
        }
        
        outputs = _collect_outputs(
            context,
            {name: spec.optional for name, spec in context.node.outputs.items()},
        )
        return StageResult(outputs=outputs, metadata=metadata)


