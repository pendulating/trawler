"""Norm Role Abstraction stage runner."""

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
from ..stages.norm_role_abstraction import run_norm_role_abstraction_stage
from .base import StageRunner


class NormRoleAbstractionRunner(StageRunner):
    """Runner for the norm_role_abstraction stage."""

    stage_name = "norm_role_abstraction"

    def run(self, context: StageExecutionContext) -> StageResult:
        """Execute the norm_role_abstraction stage."""
        dataset_path = context.inputs.get("dataset")
        if not dataset_path:
            raise ValueError(f"Node '{context.node.key}' requires 'dataset' input")

        cfg = context.cfg
        df, _, _ = prepare_stage_input(cfg, dataset_path, self.stage_name)
        input_rows = len(df) if isinstance(df, pd.DataFrame) else 0
        print(f"[{self.stage_name}] Input: {input_rows} rows")

        out = run_norm_role_abstraction_stage(df, cfg)

        output_rows = len(out) if isinstance(out, pd.DataFrame) else 0
        print(f"[{self.stage_name}] Output: {output_rows} rows")
        _save_stage_outputs(out, context.output_paths)

        metadata: Dict[str, Any] = {
            "rows": output_rows,
            "input_rows": input_rows,
            "streaming": False,
        }

        outputs = _collect_outputs(
            context,
            {name: spec.optional for name, spec in context.node.outputs.items()},
        )
        return StageResult(outputs=outputs, metadata=metadata)
