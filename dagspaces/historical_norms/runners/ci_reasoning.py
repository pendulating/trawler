"""CI Reasoning stage runner - Contextual Integrity information flow identification."""

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
from ..stages.ci_reasoning import run_ci_reasoning_stage
from .base import StageRunner


class CIReasoningRunner(StageRunner):
    """Runner for the ci_reasoning stage.

    Identifies information flows in text chunks through a
    Contextual Integrity lens, producing reasoning traces.
    """

    stage_name = "ci_reasoning"

    def run(self, context: StageExecutionContext) -> StageResult:
        """Execute the ci_reasoning stage."""
        dataset_path = context.inputs.get("dataset")
        if not dataset_path:
            raise ValueError(f"Node '{context.node.key}' requires 'dataset' input")

        cfg = context.cfg
        df, _, _ = prepare_stage_input(cfg, dataset_path, self.stage_name)

        out = run_ci_reasoning_stage(df, cfg)

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
