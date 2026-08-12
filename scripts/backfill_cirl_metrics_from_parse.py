#!/usr/bin/env python3
"""Write CIRL metrics for cells whose compute_metrics stage aborted on the parse gate.

WHY THIS EXISTS. A CIRL cell run WITHOUT `+runtime.allow_unreliable_metrics=true`
raises `SanityFailure` in `parse_responses` when strict `<think>/<answer>` parsing
falls below the 0.7 `parseable_rate` fail threshold, and the DAG stops before
`compute_metrics`. The expensive work survives — `llm_inference` and
`parse_responses` are both on disk, 729 rows — but no `metrics.json` is written,
so the camera-ready table renders "—" ("we chose not to report") for a cell that
actually means "the benchmark ran and this model is structurally unscoreable
under the strict format".

Those are different claims, and the difference is not a property of the model:
`qwen3.5-9b/k3-base` renders VALUES in the quartet batch (which passed the flag)
and "—" in the k3 batch (which did not), from byte-identical weights. This script
removes that asymmetry without a GPU: it runs the production
`dagspaces.cirl.stages.compute_metrics` over the existing parse artifact, exactly
as the stage would have.

It is a RE-SCORE, not a re-run. No inference, no judge, no sampling. Cells that
already have a metrics.json are skipped unless --overwrite is passed, so this can
never quietly replace a real pipeline result.

The recovered cells still fail the strict-format gate — that is the finding, and
the table should mark them as such rather than hide them.

Usage:
    python -m scripts.backfill_cirl_metrics_from_parse --run <multirun_dir> [--dry-run]
    python -m scripts.backfill_cirl_metrics_from_parse \\
        --run multirun/2026-08-03_k3_arms_ci_eval/14-28-30
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = "/share/pierson/matt/UAIR"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from dagspaces.cirl.stages.compute_metrics import (  # noqa: E402
    compute_metrics,
    metrics_to_dataframe,
)


def cell_model(cell: Path) -> str:
    ov = cell / ".hydra" / "overrides.yaml"
    if not ov.is_file():
        return "?"
    for line in ov.read_text(errors="ignore").splitlines():
        line = line.strip().lstrip("- ").strip()
        if line.startswith("model="):
            return line.split("=", 1)[1]
    return "?"


def process(cell: Path, overwrite: bool, dry_run: bool) -> dict | None:
    out_dir = cell / "cirl" / "cirl" / "outputs" / "compute_metrics"
    parse_pq = cell / "cirl" / "cirl" / "outputs" / "parse_responses" / "dataset.parquet"
    model = cell_model(cell)
    if not parse_pq.is_file():
        print(f"  {cell.name} {model}: no parse_responses — skip (needs a re-run)")
        return None
    if (out_dir / "metrics.json").is_file() and not overwrite:
        print(f"  {cell.name} {model}: metrics.json already present — skip")
        return None

    df = pd.read_parquet(parse_pq)
    metrics = compute_metrics(df)
    p, t = metrics.get("parseable"), metrics.get("total")
    print(f"  {cell.name} {model}: strict_parseable={p}/{t} "
          f"scorable_rate={metrics.get('scorable_rate'):.4f} "
          f"net={metrics.get('net_score'):.4f} "
          f"net_scorable={metrics.get('net_score_scorable'):.4f}"
          + ("   [DRY RUN]" if dry_run else ""))
    if dry_run:
        return metrics

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    metrics_to_dataframe(metrics).to_parquet(out_dir / "metrics.parquet", index=False)
    # Provenance sidecar rather than an extra key in metrics.json, so the
    # artifact stays byte-shaped like every other cell's.
    (out_dir / "RECOVERED_FROM_PARSE.txt").write_text(
        "metrics.json was NOT written by the pipeline run in this directory.\n"
        "The compute_metrics stage aborted on the parse-rate sanity gate "
        "(parseable_rate < 0.7) because this sweep omitted "
        "+runtime.allow_unreliable_metrics=true.\n"
        "It was recomputed from outputs/parse_responses/dataset.parquet by\n"
        "scripts/backfill_cirl_metrics_from_parse.py using the production\n"
        "dagspaces.cirl.stages.compute_metrics — no inference, no judge, no "
        "sampling.\n"
        f"model: {model}\n"
        f"recovered_at: {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n"
        f"strict_parseable: {p}/{t}\n"
    )
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", action="append", required=True,
                    help="Multirun dir holding numbered cells. Repeatable.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--overwrite", action="store_true",
                    help="Recompute even where a metrics.json already exists.")
    args = ap.parse_args()

    n = 0
    for run in args.run:
        root = Path(run).resolve()
        print(f"{root}")
        cells = sorted((p for p in root.iterdir() if p.is_dir() and p.name.isdigit()),
                       key=lambda p: int(p.name))
        for cell in cells:
            if process(cell, args.overwrite, args.dry_run) is not None:
                n += 1
    print(f"\n{'would write' if args.dry_run else 'wrote'} {n} cell(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
