"""Runner for norm_consolidation_from_clusters stage."""

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
from .base import StageRunner


class NormConsolidationFromClustersRunner(StageRunner):
    """Runner for the norm_consolidation_from_clusters stage."""

    stage_name = "norm_consolidation_from_clusters"

    def run(self, context: StageExecutionContext) -> StageResult:
        from ..stages.norm_consolidation_from_clusters import (
            run_norm_consolidation_from_clusters_stage,
        )

        dataset_path = context.inputs.get("dataset")
        if not dataset_path:
            raise ValueError(f"Node '{context.node.key}' requires 'dataset' input")

        cfg = context.cfg
        df, _, _ = prepare_stage_input(cfg, dataset_path, self.stage_name)
        input_rows = len(df) if isinstance(df, pd.DataFrame) else 0
        print(f"[{self.stage_name}] Input: {input_rows} rows")

        out = run_norm_consolidation_from_clusters_stage(df, cfg)

        output_rows = len(out) if isinstance(out, pd.DataFrame) else 0
        print(f"[{self.stage_name}] Output: {output_rows} consolidated norms "
              f"(ratio: {output_rows / max(input_rows, 1):.2f}x)")
        _save_stage_outputs(out, context.output_paths)

        metadata: Dict[str, Any] = {
            "rows": output_rows,
            "input_rows": input_rows,
        }

        outputs = _collect_outputs(
            context,
            {name: spec.optional for name, spec in context.node.outputs.items()},
        )
        return StageResult(outputs=outputs, metadata=metadata)
