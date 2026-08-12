#!/usr/bin/env python3
"""CLI wrapper around the PrivacyLens recovered-actions stage.

The logic lives in ``dagspaces/privacylens/stages/recovered_actions.py`` and is
the SAME code the pipeline runs when ``judge.recover_mislabelled_actions=true``
— this script exists to apply it to cells that were already evaluated, without
re-running the benchmark.

Writes only ``*_recovered`` artifacts (output_recovered.jsonl,
recovered_items.parquet, recovered_actions.parquet, metrics_recovered.json).
``metrics.json``, ``results.parquet`` and ``output.jsonl`` are never touched.

Usage:
    # Dry run — what would be recovered, no judge calls, no writes:
    python -m scripts.patch_privacylens_recovered_actions --cell <cell_dir>

    # Recover, judge, write:
    python -m scripts.patch_privacylens_recovered_actions --cell <cell_dir> --execute

    # Rebuild only recovered_items.parquet for an already-judged cell:
    python -m scripts.patch_privacylens_recovered_actions --cell <cell_dir> --items-only
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = "/share/pierson/matt/UAIR"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from dagspaces.privacylens.stages.recovered_actions import (  # noqa: E402
    build_plan,
    load_judged_frame,
    run_recovered_actions,
)

PL_SUBDIR = Path("privacylens") / "privacylens_eval" / "outputs"


# ── Driver ────────────────────────────────────────────────────────────────

def process_cell(cell: Path, execute: bool, url: str, concurrency: int,
                 items_only: bool = False) -> dict:
    out_dir = cell / PL_SUBDIR
    if not (out_dir / "agent_action_inference" / "results.parquet").is_file():
        print(f"  SKIP {cell}: no agent_action_inference/results.parquet")
        return {}

    print(f"  {cell}")
    if not execute and not items_only:
        df = load_judged_frame(str(out_dir))
        plan = build_plan(df)
        counts = plan["recovery_kind"].value_counts().to_dict() if len(plan) else {}
        print(f"    rows={len(df)} gate-failing={len(plan)} -> {counts or 'none'}")
        return {"cell": str(cell), "counts": counts}

    meta = run_recovered_actions(str(out_dir), judge_url=url,
                                 concurrency=concurrency, items_only=items_only)
    print(f"    rows={meta.get('rows')} gate-failing={meta.get('gate_failing')} "
          f"-> {meta.get('recovery_kinds') or 'none'}")
    if items_only:
        print(f"    wrote recovered_items.parquet ({meta.get('items', 0)} items)")
    elif meta.get("metrics_json"):
        print(f"    wrote {meta['metrics_json']}")
        print(f"    leak parity={meta['leakage_rate_parity_only']} "
              f"union={meta['leakage_rate_union']} "
              f"(coverage {meta['coverage_rate']}, "
              f"judge truncations {meta.get('n_judge_truncated', 0)})")
    return {"cell": str(cell), "meta": meta}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cell", action="append", required=True,
                    help="eval_all cell dir (the one holding privacylens/). Repeatable.")
    ap.add_argument("--execute", action="store_true",
                    help="Actually judge and write. Default is a dry run.")
    ap.add_argument("--judge-url",
                    default=os.environ.get("JUDGE_SERVER_URL",
                                           "http://klara.tech.cornell.edu:8002"))
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--items-only", action="store_true",
                    help="Only (re)write recovered_items.parquet for cells that "
                         "were already judged. No judge calls, no metrics.")
    args = ap.parse_args()

    mode = "ITEMS-ONLY" if args.items_only else ("EXECUTE" if args.execute else "DRY RUN")
    print(f"{mode} — judge: {args.judge_url}\n")
    summary = [process_cell(Path(c).resolve(), args.execute, args.judge_url,
                            args.concurrency, args.items_only) for c in args.cell]
    if not args.execute and not args.items_only:
        print("\nDry run only. Re-run with --execute to judge and write "
              "*_recovered artifacts (metrics.json is never touched).")
    return 0 if summary else 1


if __name__ == "__main__":
    raise SystemExit(main())
