"""SFT Data Preparation stage runner."""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from dagspaces.common.orchestrator import (
    StageExecutionContext,
    StageResult,
    _collect_outputs,
    _save_stage_outputs,
    prepare_stage_input,
)
from .base import StageRunner


class SFTDataPrepRunner(StageRunner):
    """Runner for the sft_data_prep stage."""

    stage_name = "sft_data_prep"

    def run(self, context: StageExecutionContext) -> StageResult:
        from ..stages.sft_data_prep import run_sft_data_prep_stage

        ci_reasoning_path = context.inputs.get("ci_reasoning")
        ci_extraction_path = context.inputs.get("ci_extraction")
        if not ci_reasoning_path or not ci_extraction_path:
            raise ValueError(
                f"Node '{context.node.key}' requires 'ci_reasoning' and 'ci_extraction' inputs"
            )

        cfg = context.cfg
        ci_reasoning_df = pd.read_parquet(ci_reasoning_path)
        ci_extraction_df = pd.read_parquet(ci_extraction_path)
        print(
            f"[{self.stage_name}] Input: {len(ci_reasoning_df)} reasoning rows, "
            f"{len(ci_extraction_df)} extraction rows"
        )

        out = run_sft_data_prep_stage(ci_reasoning_df, ci_extraction_df, cfg)

        output_rows = len(out) if isinstance(out, pd.DataFrame) else 0
        _save_stage_outputs(out, context.output_paths)

        metadata: Dict[str, Any] = {
            "rows": output_rows,
            "ci_reasoning_rows": len(ci_reasoning_df),
            "ci_extraction_rows": len(ci_extraction_df),
        }

        outputs = _collect_outputs(
            context,
            {name: spec.optional for name, spec in context.node.outputs.items()},
        )
        return StageResult(outputs=outputs, metadata=metadata)
