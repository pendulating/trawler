"""Norm Universe stage runner."""

from __future__ import annotations

import json
import os
from typing import Any, Dict

import pandas as pd

from dagspaces.common.orchestrator import (
    StageExecutionContext,
    StageResult,
)
from .base import StageRunner


class NormUniverseRunner(StageRunner):
    """Runner for the norm_universe stage.

    Inputs:
        dataset: structured_norms.parquet (raw extracted norms)

    Outputs:
        dataset: norm_universes.json (per-book norm dicts)
        embeddings: directory with per-book .npy embedding matrices
    """

    stage_name = "norm_universe"

    def run(self, context: StageExecutionContext) -> StageResult:
        from ..stages.norm_universe import run_norm_universe_stage

        dataset_path = context.inputs.get("dataset")
        if not dataset_path:
            raise ValueError(f"Node '{context.node.key}' requires 'dataset' input")

        cfg = context.cfg
        df = pd.read_parquet(dataset_path)
        input_rows = len(df)
        print(f"[{self.stage_name}] Input: {input_rows} raw norms")

        # Output dir is the parent of the JSON output path
        json_output_path = context.output_paths.get("dataset")
        if json_output_path:
            output_dir = os.path.dirname(json_output_path)
        else:
            output_dir = os.path.join(
                context.output_paths.get("embeddings", "outputs/norm_universe"),
            )

        norm_universes = run_norm_universe_stage(df, cfg, output_dir)

        # Save JSON universe
        if json_output_path:
            os.makedirs(os.path.dirname(json_output_path), exist_ok=True)
            with open(json_output_path, "w", encoding="utf-8") as f:
                json.dump(norm_universes, f, indent=2, ensure_ascii=False)
            print(f"[{self.stage_name}] Saved norm universes to {json_output_path}")

        metadata: Dict[str, Any] = {
            "rows": sum(len(v) for v in norm_universes.values()),
            "num_sources": len(norm_universes),
            "input_rows": input_rows,
        }

        outputs = {}
        if json_output_path:
            outputs["dataset"] = json_output_path
        emb_dir = context.output_paths.get("embeddings") or os.path.join(output_dir, "embeddings")
        if os.path.isdir(emb_dir):
            outputs["embeddings"] = emb_dir

        return StageResult(outputs=outputs, metadata=metadata)
