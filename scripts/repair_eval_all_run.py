#!/usr/bin/env python3
"""Derive a repair plan for an eval_all multirun: which cells lost which benchmarks.

WHY THIS IS DERIVED AND NOT HARDCODED. The 2026-08-04 quartet sweep lost two
benchmarks to a ten-minute SLURM controller wobble (`sbatch: Socket timed out
on send/recv` in one cell; submitit misreading a still-running job as COMPLETED
in another). Which cells that hits is a property of when the wobble landed, not
of the sweep design — a hand-written repair list written before the sweep ends
is stale by the time it runs. This reads the run's own artifacts instead.

AUTHORITY. `<cell>/failures.json` is what the eval_all monitor writes on exit
(dagspaces/eval_all/orchestrator.py) and is the primary source. Two things it
does not cover, both handled here:

  * A cell whose monitor was killed never writes one at all. That cell is
    reported as INCOMPLETE-UNKNOWN, never silently skipped — an absent
    failures.json means "we don't know", not "nothing failed".
  * A benchmark can be dispatched "ok" and still leave no metrics behind. Those
    are reported too (`ok-but-no-metrics`), because the table reads metrics, not
    dispatch status.

Metrics detection is deliberately coarse (any `metrics.parquet` / `metrics.json`
under the benchmark dir). It is a cross-check on failures.json, not a
completeness proof: ConfAIde writes per-tier metrics, so a partly-run ConfAIde
can carry some. failures.json catches that case, which is why it leads.

Usage:
    python -m scripts.repair_eval_all_run <run_dir>
    python -m scripts.repair_eval_all_run <run_dir> --json
    python -m scripts.repair_eval_all_run <run_dir> --emit-cmds \\
        --sweep eval_rl_quartet_repair_2026_08_04

`<run_dir>` is the timestamped multirun dir holding the numbered cells, e.g.
multirun/2026-08-04_eval_rl_quartet_recovery/15-51-04.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Benchmarks the all_benchmarks pipeline can dispatch. Only used for the
# fallback scan when a cell has no failures.json; the normal path takes the
# benchmark set from failures.json itself. simpleqa_verified is enabled=false
# in the pipeline and is never expected.
KNOWN_BENCHMARKS = [
    "goldcoin",
    "privacylens",
    "cirl",
    "confaide",
    "vlm_geoprivacy",
    "mmlu",
]


def cell_model(cell: Path) -> str | None:
    """Read the model override Hydra recorded for this cell."""
    overrides = cell / ".hydra" / "overrides.yaml"
    if not overrides.is_file():
        return None
    for line in overrides.read_text().splitlines():
        line = line.strip().lstrip("- ").strip()
        if line.startswith("model="):
            return line.split("=", 1)[1].strip().strip("'\"")
    return None


def has_metrics(cell: Path, bench: str) -> bool:
    """True if the benchmark dir holds any metrics artifact."""
    bdir = cell / bench
    if not bdir.is_dir():
        return False
    for pattern in ("metrics.parquet", "metrics.json"):
        if next(bdir.rglob(pattern), None) is not None:
            return True
    return False


def scan_cell(cell: Path) -> dict:
    """Classify one cell: which benchmarks need re-running, and why."""
    model = cell_model(cell)
    fpath = cell / "failures.json"
    result: dict = {
        "cell": cell.name,
        "path": str(cell),
        "model": model,
        "repair": [],
        "reasons": {},
        "status": "ok",
    }

    if not fpath.is_file():
        # No summary: the cell is STILL RUNNING, or its monitor was killed.
        # These look identical from disk, so the plan is reported as
        # provisional and the caller is told to re-check. Do not launch a
        # repair off this branch — run_eval_quartet_repair.sh refuses while any
        # monitor is alive for exactly this reason. (PrivacyLens metrics land
        # in the finalize phase at the very end of a cell, so a healthy
        # in-flight cell shows several "no metrics" lines here.)
        missing = [b for b in KNOWN_BENCHMARKS if not has_metrics(cell, b)]
        result["status"] = "no-summary"
        result["repair"] = missing
        result["reasons"] = {
            b: "no failures.json — cell still running or monitor killed"
            for b in missing
        }
        return result

    try:
        data = json.loads(fpath.read_text())
    except Exception as e:  # noqa: BLE001 — a corrupt summary is a finding
        result["status"] = "unreadable-failures-json"
        result["reasons"] = {"_": f"{type(e).__name__}: {e}"}
        return result

    result["model"] = data.get("model") or model
    dispatch: dict[str, str] = data.get("dispatch", {})

    for bench, status in dispatch.items():
        if status == "skipped" or str(status).startswith("skipped"):
            continue
        if status != "ok":
            result["repair"].append(bench)
            result["reasons"][bench] = f"dispatch={status}"
        elif not has_metrics(cell, bench):
            result["repair"].append(bench)
            result["reasons"][bench] = "dispatch=ok but no metrics written"

    if result["repair"]:
        result["status"] = "needs-repair"
    return result


def scan_run(run_dir: Path) -> list[dict]:
    """Scan every numbered cell under a multirun dir, in cell order."""
    cells = sorted(
        (p for p in run_dir.iterdir() if p.is_dir() and p.name.isdigit()),
        key=lambda p: int(p.name),
    )
    if not cells:
        raise SystemExit(f"ERROR: no numbered cell dirs under {run_dir}")
    return [scan_cell(c) for c in cells]


def emit_cmds(plan: list[dict], sweep: str, driver: str) -> list[str]:
    """One eval_all invocation per cell needing repair.

    One invocation per cell, NOT one sweep over all of them: the repair lists
    differ per cell, and Hydra's sweeper takes a cartesian product — a single
    sweep would re-run benchmarks that are already fine in cells that don't
    need them.
    """
    cmds = []
    for cell in plan:
        if not cell["repair"]:
            continue
        if not cell["model"]:
            cmds.append(f"# SKIPPED {cell['cell']}: model override unreadable")
            continue
        include = ",".join(sorted(set(cell["repair"])))
        cmds.append(
            f"{driver} -m dagspaces.eval_all.cli --multirun "
            f"+sweep={sweep} model={cell['model']} "
            f"'benchmark_filter.include=[{include}]'"
        )
    return cmds


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir", help="Timestamped multirun dir holding numbered cells")
    ap.add_argument("--json", action="store_true", help="Emit the plan as JSON")
    ap.add_argument(
        "--emit-cmds", action="store_true",
        help="Emit one eval_all repair command per affected cell",
    )
    ap.add_argument(
        "--sweep", default="eval_rl_quartet_repair_2026_08_04",
        help="Sweep config name used by --emit-cmds",
    )
    ap.add_argument(
        "--driver",
        default="/share/pierson/matt/UAIR/.venv-vllm025cu129/bin/python",
        help="Driver python for --emit-cmds (absolute; submitit bakes it in)",
    )
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_dir():
        print(f"ERROR: not a directory: {run_dir}", file=sys.stderr)
        return 2

    plan = scan_run(run_dir)

    if args.json:
        print(json.dumps({"run_dir": str(run_dir), "cells": plan}, indent=2))
        return 0

    if args.emit_cmds:
        for line in emit_cmds(plan, args.sweep, args.driver):
            print(line)
        return 0

    print(f"Run: {run_dir}\n")
    needs = 0
    for cell in plan:
        tag = {
            "ok": "  OK",
            "needs-repair": "REPAIR",
            "no-summary": "RUNNING?",
            "unreadable-failures-json": "CORRUPT",
        }.get(cell["status"], cell["status"])
        print(f"[{tag:>7}] cell {cell['cell']}  {cell['model']}")
        for bench in cell["repair"]:
            print(f"            - {bench}: {cell['reasons'][bench]}")
        if cell["status"] == "unreadable-failures-json":
            for k, v in cell["reasons"].items():
                print(f"            - {v}")
        needs += bool(cell["repair"])

    print()
    provisional = sum(1 for c in plan if c["status"] == "no-summary")
    if provisional:
        print(f"WARNING: {provisional} cell(s) have no failures.json — still "
              f"running, or the monitor was killed. Their lines above are "
              f"provisional; re-run this after the monitors exit.")
    if needs:
        print(f"{needs} cell(s) need repair. Commands:")
        print("  python -m scripts.repair_eval_all_run "
              f"{run_dir} --emit-cmds")
    else:
        print("Nothing to repair.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
