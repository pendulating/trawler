"""Reward Preparation stage runner."""

from __future__ import annotations

import json
from typing import Any, Dict

import pandas as pd

from dagspaces.common.orchestrator import (
    StageExecutionContext,
    StageResult,
    _collect_outputs,
    _save_stage_outputs,
)
from .base import StageRunner


class RewardPrepRunner(StageRunner):
    """Runner for the reward_prep stage."""

    stage_name = "reward_prep"

    def run(self, context: StageExecutionContext) -> StageResult:
        from ..stages.reward_prep import run_reward_prep_stage

        dataset_path = context.inputs.get("dataset")
        norm_universes_path = context.inputs.get("norm_universes")
        if not dataset_path or not norm_universes_path:
            raise ValueError(
                f"Node '{context.node.key}' requires 'dataset' and 'norm_universes' inputs"
            )

        cfg = context.cfg

        # Load SFT pairs
        sft_df = pd.read_parquet(dataset_path)
        print(f"[{self.stage_name}] Input: {len(sft_df)} SFT pairs")

        # Load norm universes (JSON file)
        with open(norm_universes_path, "r", encoding="utf-8") as f:
            norm_universes = json.load(f)
        print(f"[{self.stage_name}] Loaded {len(norm_universes)} normative universes")

        # Pre-computed embeddings directory (from norm_universe stage)
        embeddings_dir = context.inputs.get("embeddings")
        if embeddings_dir:
            print(f"[{self.stage_name}] Using pre-computed embeddings from {embeddings_dir}")

        out = run_reward_prep_stage(
            sft_df, norm_universes, cfg, embeddings_dir=embeddings_dir,
        )

        output_rows = len(out) if isinstance(out, pd.DataFrame) else 0
        _save_stage_outputs(out, context.output_paths)

        metadata: Dict[str, Any] = {
            "rows": output_rows,
            "input_pairs": len(sft_df),
            "num_universes": len(norm_universes),
        }

        outputs = _collect_outputs(
            context,
            {name: spec.optional for name, spec in context.node.outputs.items()},
        )
        return StageResult(outputs=outputs, metadata=metadata)
