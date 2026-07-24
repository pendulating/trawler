"""Norm Universe stage runner."""

from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd

from dagspaces.common.orchestrator import (
    StageExecutionContext,
    StageResult,
)
from dagspaces.common.stage_utils import sanitize_for_json

from .base import StageRunner


class NormUniverseRunner(StageRunner):
    """Runner for the norm_universe stage.

    Inputs:
        dataset: abstracted_norms.parquet (role-abstracted norms)

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
                # Norm dicts carry parquet scalars verbatim; a fully non-null
                # bool column yields numpy.bool_ values json.dump rejects.
                json.dump(
                    sanitize_for_json(norm_universes), f, indent=2, ensure_ascii=False
                )
            print(f"[{self.stage_name}] Saved norm universes to {json_output_path}")

        metadata: dict[str, Any] = {
            "rows": sum(len(v) for v in norm_universes.values()),
            "num_sources": len(norm_universes),
            "input_rows": input_rows,
        }

        # Surface the universe shape on the W&B run, and version the
        # universe JSON as an artifact so downstream GRPO runs can record
        # exactly which universe build they trained against.
        logger = getattr(context, "logger", None)
        if logger is not None:
            try:
                total_norms = metadata["rows"]
                logger.log_metrics({
                    "norm_universe/books": len(norm_universes),
                    "norm_universe/total_norms": total_norms,
                    "norm_universe/input_rows": input_rows,
                    "norm_universe/dropped_invalid_or_duplicate": input_rows - total_norms,
                })
                if json_output_path:
                    logger.log_artifact(
                        json_output_path,
                        name="norm_universes",
                        type="norm_universe",
                        metadata={
                            "n_books": len(norm_universes),
                            "total_norms": total_norms,
                            "norms_per_book": {
                                k: len(v) for k, v in sorted(norm_universes.items())
                            },
                        },
                    )
            except Exception as e:
                print(f"[{self.stage_name}] WARNING: W&B logging failed: {e}")

        outputs = {}
        if json_output_path:
            outputs["dataset"] = json_output_path
        emb_dir = context.output_paths.get("embeddings") or os.path.join(output_dir, "embeddings")
        if os.path.isdir(emb_dir):
            outputs["embeddings"] = emb_dir

        return StageResult(outputs=outputs, metadata=metadata)
