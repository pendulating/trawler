#!/usr/bin/env python3
"""Two-way sync/verify between local metrics.json trees and W&B.

Counterpart of the always-on mirror in ``dagspaces/common/metrics_sync.py``
(see ``wiki/integrations/wandb-parity.md``). The mirror keeps the two sides
in lockstep for new runs; this script covers everything else: verifying
parity, backfilling W&B for runs that crashed / ran offline / predate the
mirror, and restoring metrics.json trees from W&B after local data loss.

Usage:
    # Parity report for a sweep dir (no writes anywhere):
    python -m scripts.wandb_local_sync verify multirun/2026-07-19_eval_sft_per_checkpoint_all/22-48-47

    # Backfill W&B from disk (resumes the linked run when a wandb_run.json
    # sidecar exists, else creates a run tagged `backfill`):
    python -m scripts.wandb_local_sync push multirun/<sweep>/<time> [--dry-run]

    # Restore metrics.json files from W&B into a directory tree:
    python -m scripts.wandb_local_sync pull --group "<sweep>/<time>" --dest restored/

Notes:
    - verify compares every numeric leaf on disk against the run's mirrored
      summary keys (`<subdir>/metrics_json/<dotted>`); a cell can be MISSING
      (key absent in W&B), MISMATCH (differs beyond --tol), or UNLINKED (no
      sidecar and no group match — typically a run from before the mirror).
    - push never overwrites a differing W&B value silently: it logs what it
      writes and tags backfilled runs so they are distinguishable from
      live-logged ones.
    - pull writes only under --dest; it never touches an existing multirun
      tree in place.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from dagspaces.common.metrics_sync import (
    MIRROR_SEGMENT,
    SIDECAR_FILENAME,
    derive_group_from_output_dir,
    flatten_numeric,
)
from dagspaces.common.stage_utils import ensure_dotenv

DEFAULT_ENTITY = os.environ.get("WANDB_ENTITY", "uair")
DEFAULT_PROJECT = "eval-all"


# ---------------------------------------------------------------------------
# Local discovery
# ---------------------------------------------------------------------------

def discover_local(root: Path) -> list[dict[str, Any]]:
    """Every metrics.json under *root*, with its flattened values, mirror
    key prefix (the parent-dir name), sidecar linkage, and enough path
    context to label reports."""
    cells = []
    for mp in sorted(root.rglob("metrics.json")):
        if ".hydra" in mp.parts or "wandb" in mp.parts:
            continue
        try:
            metrics = json.loads(mp.read_text())
        except (ValueError, OSError) as exc:
            print(f"!! unreadable {mp}: {exc}", file=sys.stderr)
            continue
        sidecar = None
        sidecar_path = mp.parent / SIDECAR_FILENAME
        if sidecar_path.is_file():
            try:
                sidecar = json.loads(sidecar_path.read_text())
            except (ValueError, OSError):
                sidecar = None
        subdir = mp.parent.name
        cells.append({
            "path": mp,
            "rel": str(mp.relative_to(root)),
            "subdir": subdir,
            "flat": flatten_numeric(metrics),
            "sidecar": sidecar,
        })
    return cells


def _bench_of(cell: dict[str, Any]) -> str | None:
    """Benchmark (dagspace inner dir) from the path layout
    ``.../<bench_inner>/outputs/<subdir>/metrics.json``."""
    parts = Path(cell["rel"]).parts
    try:
        i = parts.index("outputs")
    except ValueError:
        return None
    return parts[i - 1] if i >= 1 else None


def _model_override_near(mp: Path, stop: Path) -> str | None:
    """Walk up from a metrics.json looking for .hydra/overrides.yaml and the
    ``model=`` override in it (for labelling backfilled runs)."""
    cur = mp.parent
    while True:
        ov = cur / ".hydra" / "overrides.yaml"
        if ov.is_file():
            try:
                for line in ov.read_text(errors="ignore").splitlines():
                    line = line.strip().lstrip("- ").strip()
                    if line.startswith("model="):
                        return line.split("=", 1)[1]
            except OSError:
                pass
        if cur == stop or cur.parent == cur:
            return None
        cur = cur.parent


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------

def cmd_verify(args: argparse.Namespace) -> int:
    import wandb

    root = Path(args.root).resolve()
    cells = discover_local(root)
    print(f"{len(cells)} metrics.json files under {root}")

    api = wandb.Api()
    run_cache: dict[str, Any] = {}

    def _get_run(entity: str, project: str, run_id: str):
        key = f"{entity}/{project}/{run_id}"
        if key not in run_cache:
            try:
                run_cache[key] = api.run(key)
            except Exception:
                run_cache[key] = None
        return run_cache[key]

    n_ok = n_missing = n_mismatch = n_unlinked = 0
    problems: list[str] = []
    for cell in cells:
        sc = cell["sidecar"]
        if not sc or not sc.get("run_id"):
            n_unlinked += 1
            problems.append(f"UNLINKED  {cell['rel']} (no {SIDECAR_FILENAME})")
            continue
        run = _get_run(sc.get("entity") or args.entity,
                       sc.get("project") or args.project,
                       sc["run_id"])
        if run is None:
            n_missing += len(cell["flat"])
            problems.append(f"NO-RUN    {cell['rel']} → run {sc['run_id']} "
                            "not found in W&B")
            continue
        summary = dict(run.summary)
        for dotted, disk_val in cell["flat"].items():
            wb_key = f"{cell['subdir']}/{MIRROR_SEGMENT}/{dotted}"
            wb_val = summary.get(wb_key)
            if wb_val is None:
                n_missing += 1
                problems.append(f"MISSING   {cell['rel']} :: {wb_key}")
            elif abs(float(wb_val) - float(disk_val)) > args.tol:
                n_mismatch += 1
                problems.append(f"MISMATCH  {cell['rel']} :: {wb_key} "
                                f"disk={disk_val} wandb={wb_val}")
            else:
                n_ok += 1

    for line in problems[: args.max_report]:
        print(line)
    if len(problems) > args.max_report:
        print(f"... and {len(problems) - args.max_report} more")
    print(f"\nverify: {n_ok} values in parity, {n_missing} missing in W&B, "
          f"{n_mismatch} mismatched, {n_unlinked} files unlinked")
    print("Fix missing/unlinked with: "
          f"python -m scripts.wandb_local_sync push {args.root}")
    return 1 if (n_missing or n_mismatch or n_unlinked) else 0


# ---------------------------------------------------------------------------
# push (disk → W&B backfill)
# ---------------------------------------------------------------------------

def cmd_push(args: argparse.Namespace) -> int:
    import wandb

    root = Path(args.root).resolve()
    cells = discover_local(root)
    group = args.group or derive_group_from_output_dir(str(root))
    print(f"{len(cells)} metrics.json files under {root} (group={group})")

    # One W&B run per linked run_id (a single_run pipeline has several
    # metrics.json linked to the same run); unlinked cells are grouped per
    # benchmark tree so a backfilled arm×bench also gets one run.
    by_run: dict[str, list[dict[str, Any]]] = {}
    for cell in cells:
        sc = cell["sidecar"] or {}
        if sc.get("run_id"):
            key = f"resume::{sc.get('entity') or args.entity}::" \
                  f"{sc.get('project') or args.project}::{sc['run_id']}"
        else:
            bench = _bench_of(cell) or "unknown"
            arm = Path(cell["rel"]).parts[0]
            key = f"new::{arm}::{bench}"
        by_run.setdefault(key, []).append(cell)

    n_pushed = 0
    for key, group_cells in sorted(by_run.items()):
        kind, *rest = key.split("::")
        if args.dry_run:
            print(f"[dry-run] {key}: would push "
                  f"{sum(len(c['flat']) for c in group_cells)} values from "
                  f"{[c['rel'] for c in group_cells]}")
            continue

        if kind == "resume":
            entity, project, run_id = rest
            init_kwargs: dict[str, Any] = {
                "entity": entity, "project": project,
                "id": run_id, "resume": "allow",
            }
        else:
            arm, bench = rest
            model_ov = _model_override_near(group_cells[0]["path"], root)
            init_kwargs = {
                "entity": args.entity, "project": args.project,
                "group": group,
                "job_type": "backfill",
                "name": f"backfill-{bench}-{arm}-"
                        f"{(model_ov or 'unknown').replace('/', '-')}",
                "tags": ["backfill", f"bench:{bench}"]
                        + ([f"eval_all_run:{group}"] if group else []),
                "config": {
                    "local_output_dir": str(root),
                    "model_override": model_ov,
                    "backfill_source": str(root),
                },
            }

        run = wandb.init(**init_kwargs)
        try:
            for cell in group_cells:
                flat = {
                    f"{cell['subdir']}/{MIRROR_SEGMENT}/{k}": v
                    for k, v in cell["flat"].items()
                }
                run.log(flat)
                wandb.save(str(cell["path"]),
                           base_path=str(cell["path"].parent.parent),
                           policy="now")
                n_pushed += len(flat)
                print(f"pushed {len(flat):4d} values  {cell['rel']} → "
                      f"{run.id}")
                # Leave/refresh the linkage sidecar so verify (and future
                # pushes) resolve this run.
                sidecar = {
                    "entity": run.entity, "project": run.project,
                    "run_id": run.id, "run_name": run.name,
                    "run_url": run.url, "group": group,
                    "tags": list(run.tags or ()),
                    "backfilled": True,
                }
                (cell["path"].parent / SIDECAR_FILENAME).write_text(
                    json.dumps(sidecar, indent=2, default=str)
                )
        finally:
            run.finish()
    print(f"\npush: {n_pushed} values"
          + (" (dry run — nothing written)" if args.dry_run else ""))
    return 0


# ---------------------------------------------------------------------------
# pull (W&B → disk restore)
# ---------------------------------------------------------------------------

def cmd_pull(args: argparse.Namespace) -> int:
    import wandb

    api = wandb.Api()
    dest = Path(args.dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)

    if args.run_id:
        runs = [api.run(f"{args.entity}/{args.project}/{r}")
                for r in args.run_id]
    else:
        if not args.group:
            print("pull needs --group or --run-id", file=sys.stderr)
            return 2
        runs = list(api.runs(f"{args.entity}/{args.project}",
                             filters={"group": args.group}))
    print(f"{len(runs)} runs to restore from")

    n_files = 0
    for run in runs:
        # Reconstruct a stable tree: <dest>/<run_name or id>/<stored path>.
        # The stored path already carries the stage subdir
        # ("compute_metrics_tier2b/metrics.json") thanks to the mirror's
        # base_path choice; local_output_dir in the run config records where
        # the original lived.
        run_dir = dest / (run.name or run.id)
        for f in run.files():
            if not f.name.endswith(("metrics.json", SIDECAR_FILENAME)):
                continue
            f.download(root=str(run_dir), replace=True)
            n_files += 1
            print(f"restored {run_dir / f.name}")
        origin = (run.config or {}).get("local_output_dir")
        if origin:
            (run_dir / "ORIGIN.txt").write_text(
                f"local_output_dir: {origin}\nrun: {run.url}\n"
            )
    print(f"\npull: {n_files} files restored under {dest}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ensure_dotenv()
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_verify = sub.add_parser("verify", help="parity report (read-only)")
    p_verify.add_argument("root", help="multirun sweep dir (or any subtree)")
    p_verify.add_argument("--entity", default=DEFAULT_ENTITY)
    p_verify.add_argument("--project", default=DEFAULT_PROJECT)
    p_verify.add_argument("--tol", type=float, default=1e-9)
    p_verify.add_argument("--max-report", type=int, default=200)
    p_verify.set_defaults(func=cmd_verify)

    p_push = sub.add_parser("push", help="backfill W&B from disk")
    p_push.add_argument("root")
    p_push.add_argument("--entity", default=DEFAULT_ENTITY)
    p_push.add_argument("--project", default=DEFAULT_PROJECT)
    p_push.add_argument("--group", default=None,
                        help="override the derived W&B group")
    p_push.add_argument("--dry-run", action="store_true")
    p_push.set_defaults(func=cmd_push)

    p_pull = sub.add_parser("pull", help="restore metrics.json from W&B")
    p_pull.add_argument("--entity", default=DEFAULT_ENTITY)
    p_pull.add_argument("--project", default=DEFAULT_PROJECT)
    p_pull.add_argument("--group", default=None)
    p_pull.add_argument("--run-id", nargs="*", default=None)
    p_pull.add_argument("--dest", required=True)
    p_pull.set_defaults(func=cmd_pull)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
