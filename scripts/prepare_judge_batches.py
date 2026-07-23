"""Prepare OpenAI Batch API JSONLs from a task-LLM stage parquet.

Given a single stage-output parquet (e.g. privacylens'
``agent_action_inference/results.parquet``), this script emits one
Batch API input JSONL per downstream LLM-judged evaluation attached to
that stage — without running the full pipeline. Useful when you
already have task-LLM outputs on disk and just want to (re)produce the
judge batches to submit.

Under the hood it delegates to the same ``export_*_judge_batch``
functions the batch-export pipelines use, so the emitted JSONLs are
byte-identical to what you would get from:

    python -m dagspaces.privacylens.cli pipeline=privacylens_clean_batch \\
        judge.mode=batch_export ...

Usage:

    # Default: privacylens.agent_action_inference → leakage + helpfulness
    python scripts/prepare_judge_batches.py \\
        --input outputs/.../agent_action_inference/results.parquet \\
        --stage privacylens.agent_action_inference

    # Only the helpfulness judge
    python scripts/prepare_judge_batches.py \\
        --input outputs/.../agent_action_inference/results.parquet \\
        --stage privacylens.agent_action_inference \\
        --judges helpfulness

    # Override the target model and output directory
    python scripts/prepare_judge_batches.py \\
        --input outputs/.../trajectory_inference/dataset.parquet \\
        --stage cirl_vignettes.trajectory_inference \\
        --target-model gpt-4o \\
        --output-dir /tmp/cirl_batches

Output layout (one directory per selected judge, under ``--output-dir``):

    <output_dir>/
      leakage_judge_batch/
        requests.jsonl        (Batch API input — submit this)
        pending.parquet       (input df + judge_custom_id column)
        manifest.json
        [items.parquet]       (cirl_vignettes only: row × secret mapping)
      helpfulness_judge_batch/
        ...

Submit each ``requests.jsonl`` via::

    python -m dagspaces.common.batch_api submit <path>/requests.jsonl
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List

import pandas as pd
from omegaconf import DictConfig, OmegaConf


# ---------------------------------------------------------------------------
# Stage → judge registry
# ---------------------------------------------------------------------------

@dataclass
class JudgeEntry:
    """One downstream LLM judge attached to a task-LLM stage."""
    name: str                # CLI-facing name (e.g. "leakage")
    out_subdir: str          # directory to write under --output-dir
    export_fn: Callable[[pd.DataFrame, DictConfig, str], pd.DataFrame]
    # Function signature matches dagspace export_*_judge_batch:
    #   (df, cfg, output_dir) -> df_with_judge_custom_id


@dataclass
class StageEntry:
    """A task-LLM stage whose outputs feed one or more LLM judges."""
    label: str               # CLI-facing stage id, e.g. "privacylens.agent_action_inference"
    default_judges: List[str]
    judges: Dict[str, JudgeEntry]


def _build_registry() -> Dict[str, StageEntry]:
    """Lazy imports so a typo in one dagspace doesn't break the whole CLI."""
    from dagspaces.privacylens.stages.llm_inference import (
        export_helpfulness_judge_batch as pl_helpfulness,
        export_leakage_judge_batch as pl_leakage,
    )
    from dagspaces.privacylens.cirl_protocol.stages.judge_helpfulness import (
        export_helpfulness_judge_batch as cv_helpfulness,
    )
    from dagspaces.privacylens.cirl_protocol.stages.judge_leakage import (
        export_leakage_judge_batch as cv_leakage,
    )

    privacylens_aai = StageEntry(
        label="privacylens.agent_action_inference",
        default_judges=["leakage", "helpfulness"],
        judges={
            "leakage": JudgeEntry(
                name="leakage",
                out_subdir="leakage_judge_batch",
                export_fn=pl_leakage,
            ),
            "helpfulness": JudgeEntry(
                name="helpfulness",
                out_subdir="helpfulness_judge_batch",
                export_fn=pl_helpfulness,
            ),
        },
    )

    cirl_trajectory = StageEntry(
        label="cirl_vignettes.trajectory_inference",
        default_judges=["leakage", "helpfulness"],
        judges={
            "leakage": JudgeEntry(
                name="leakage",
                out_subdir="judge_leakage_batch",
                export_fn=cv_leakage,
            ),
            "helpfulness": JudgeEntry(
                name="helpfulness",
                out_subdir="judge_helpfulness_batch",
                export_fn=cv_helpfulness,
            ),
        },
    )

    return {s.label: s for s in [privacylens_aai, cirl_trajectory]}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_cfg(args: argparse.Namespace) -> DictConfig:
    """Minimal Hydra-style config the export functions expect.

    Mirrors the ``judge.batch.*`` schema in each dagspace's config.yaml.
    Only batch-export-relevant fields are set; every live-mode field is
    intentionally omitted to make it obvious this path doesn't touch
    any live judge endpoint.
    """
    return OmegaConf.create({
        "judge": {
            "mode": "batch_export",
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "batch": {
                "target_model": args.target_model,
                "target_endpoint": args.target_endpoint,
            },
        },
    })


def _default_output_dir(input_path: str) -> str:
    """Default: sibling ``judge_batches/`` next to the input parquet's parent."""
    parent = os.path.dirname(os.path.abspath(input_path))
    return os.path.join(os.path.dirname(parent), "judge_batches")


def _run_judge(
    judge: JudgeEntry,
    df: pd.DataFrame,
    cfg: DictConfig,
    output_dir: str,
) -> str:
    """Execute one judge's export function and write pending.parquet."""
    subdir = os.path.join(output_dir, judge.out_subdir)
    os.makedirs(subdir, exist_ok=True)

    result_df = judge.export_fn(df, cfg, subdir)

    pending_path = os.path.join(subdir, "pending.parquet")
    result_df.to_parquet(pending_path, index=False)
    return subdir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    registry = _build_registry()

    parser = argparse.ArgumentParser(
        description="Emit OpenAI Batch API JSONLs for a task-LLM stage's "
                    "downstream LLM judges, from an existing parquet.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to the stage-output parquet (e.g. "
             "agent_action_inference/results.parquet).",
    )
    parser.add_argument(
        "--stage", required=True, choices=sorted(registry.keys()),
        help="Which task-LLM stage this parquet came from. Determines "
             "which judges are available.",
    )
    parser.add_argument(
        "--judges", nargs="+", default=None, metavar="JUDGE",
        help="Judges to emit. Default: every judge registered for the "
             "chosen --stage.",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory to write <judge>_batch/ subfolders under. "
             "Default: judge_batches/ sibling of the input's parent dir.",
    )
    parser.add_argument(
        "--target-model", default="gpt-5.2",
        help="Model name written into each JSONL body.model. "
             "Default: gpt-5.2.",
    )
    parser.add_argument(
        "--target-endpoint", default="/v1/chat/completions",
        help="Endpoint URL written into each JSONL line's url field. "
             "Default: /v1/chat/completions.",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="body.temperature for every request. Default: 0.0.",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=1024,
        help="body.max_tokens for every request. Default: 1024.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"error: input parquet not found: {args.input}", file=sys.stderr)
        return 1

    stage = registry[args.stage]
    judges_to_run = args.judges or stage.default_judges
    unknown = [j for j in judges_to_run if j not in stage.judges]
    if unknown:
        print(
            f"error: unknown judge(s) {unknown} for stage {args.stage!r}. "
            f"Available: {sorted(stage.judges.keys())}",
            file=sys.stderr,
        )
        return 1

    output_dir = os.path.abspath(args.output_dir or _default_output_dir(args.input))
    os.makedirs(output_dir, exist_ok=True)

    print(f"[prepare_judge_batches] reading {args.input}", flush=True)
    df = pd.read_parquet(args.input)
    print(f"[prepare_judge_batches] {len(df)} rows, writing {len(judges_to_run)} "
          f"batch(es) to {output_dir}", flush=True)

    cfg = _build_cfg(args)

    written: List[str] = []
    for name in judges_to_run:
        print(f"[prepare_judge_batches] --- {name} ---", flush=True)
        subdir = _run_judge(stage.judges[name], df, cfg, output_dir)
        written.append(subdir)

    print(f"\n[prepare_judge_batches] Done. Wrote {len(written)} judge batch(es):",
          flush=True)
    for subdir in written:
        jsonl = os.path.join(subdir, "requests.jsonl")
        print(f"  {jsonl}", flush=True)
    print(
        "\nSubmit each with:\n"
        "  python -m dagspaces.common.batch_api submit <path>/requests.jsonl",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
