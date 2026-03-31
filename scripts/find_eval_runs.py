#!/usr/bin/env python3
"""Find eval_all runs by searching Hydra configs.

Scans multirun directories for eval_all runs, extracts model info from
Hydra overrides/configs, and displays them in a filterable table.
Selected runs can be printed as a ready-to-use completion_inspector command.

Usage:
    # List all eval_all runs:
    python -m scripts.find_eval_runs

    # Filter by model name (regex):
    python -m scripts.find_eval_runs --model "qwen3.5-9b"

    # Filter by date range:
    python -m scripts.find_eval_runs --since 2026-03-28
    python -m scripts.find_eval_runs --since 2026-03-28 --until 2026-03-29

    # Filter by model family:
    python -m scripts.find_eval_runs --family qwen3.5

    # Show only runs with a specific benchmark completed:
    python -m scripts.find_eval_runs --bench privacylens

    # Select runs interactively and emit a completion_inspector command:
    python -m scripts.find_eval_runs --model "qwen3.5-9b" --select

    # Pipe directly (select by index):
    python -m scripts.find_eval_runs --model "qwen3.5-9b" --pick 0,2,5
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# ── Constants ─────────────────────────────────────────────────────────────

DEFAULT_MULTIRUN_ROOT = "/share/pierson/matt/UAIR/multirun"
EVAL_ALL_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}_eval_all$")
TIMESTAMP_PATTERN = re.compile(r"^\d{2}-\d{2}-\d{2}$")


# ── Discovery ─────────────────────────────────────────────────────────────

def find_eval_all_runs(multirun_root: str) -> list[dict[str, Any]]:
    """Scan multirun root for eval_all runs and extract metadata."""
    root = Path(multirun_root)
    if not root.is_dir():
        print(f"ERROR: Multirun root does not exist: {root}", file=sys.stderr)
        sys.exit(1)

    runs = []
    # Find date-level eval_all dirs
    for date_dir in sorted(root.iterdir()):
        if not date_dir.is_dir() or not EVAL_ALL_PATTERN.match(date_dir.name):
            continue
        date_str = date_dir.name[:10]  # YYYY-MM-DD

        # Find timestamp-level run dirs
        for ts_dir in sorted(date_dir.iterdir()):
            if not ts_dir.is_dir() or not TIMESTAMP_PATTERN.match(ts_dir.name):
                continue

            run_info = _extract_run_info(ts_dir, date_str)
            if run_info:
                runs.append(run_info)

    return runs


def _extract_run_info(ts_dir: Path, date_str: str) -> dict[str, Any] | None:
    """Extract metadata from a single eval_all run directory."""
    # Resolve the actual run root (handle /0/ multirun subdir)
    run_root = ts_dir / "0" if (ts_dir / "0").is_dir() else ts_dir

    # Find Hydra config
    hydra_dir = run_root / ".hydra"
    if not hydra_dir.is_dir():
        return None

    overrides_path = hydra_dir / "overrides.yaml"
    config_path = hydra_dir / "config.yaml"

    # Extract model name from overrides (most reliable)
    model_name = None
    if overrides_path.is_file():
        try:
            for line in overrides_path.read_text().splitlines():
                line = line.strip().lstrip("- ")
                if line.startswith("model="):
                    model_name = line.split("=", 1)[1]
                    break
        except Exception:
            pass

    # Extract additional info from config.yaml
    model_source = None
    model_family = None
    lora_path = None
    if config_path.is_file():
        try:
            text = config_path.read_text()
            for line in text.splitlines():
                stripped = line.strip()
                if stripped.startswith("model_source:"):
                    model_source = stripped.split(":", 1)[1].strip()
                elif stripped.startswith("model_family:"):
                    model_family = stripped.split(":", 1)[1].strip()
                elif stripped.startswith("lora_path:"):
                    val = stripped.split(":", 1)[1].strip()
                    if val and val != "null":
                        lora_path = val
        except Exception:
            pass

    if not model_name and not model_source:
        return None

    # Determine which benchmarks completed (have output directories)
    benchmarks = []
    for child in sorted(run_root.iterdir()):
        if child.is_dir() and child.name not in (".hydra", ".submitit"):
            # Check if benchmark has any output parquets
            parquets = list(child.rglob("*.parquet"))
            if parquets:
                benchmarks.append(child.name)

    # Determine run status
    n_parquets = sum(1 for _ in run_root.rglob("*.parquet"))

    # Classify: base vs finetuned
    is_finetuned = lora_path is not None or (
        model_name and any(k in model_name for k in ("sft", "grpo", "ppo"))
    )

    return {
        "path": str(ts_dir),
        "date": date_str,
        "time": ts_dir.name,
        "model": model_name or os.path.basename(model_source or "unknown"),
        "model_family": model_family or "",
        "model_source": model_source or "",
        "lora_path": lora_path or "",
        "is_finetuned": is_finetuned,
        "benchmarks": benchmarks,
        "n_benchmarks": len(benchmarks),
        "n_parquets": n_parquets,
    }


# ── Filtering ─────────────────────────────────────────────────────────────

def filter_runs(
    runs: list[dict],
    model: str | None = None,
    family: str | None = None,
    since: str | None = None,
    until: str | None = None,
    bench: str | None = None,
    finetuned_only: bool = False,
    base_only: bool = False,
) -> list[dict]:
    """Filter runs by criteria."""
    filtered = runs

    if model:
        pat = re.compile(model, re.IGNORECASE)
        filtered = [r for r in filtered if pat.search(r["model"])]

    if family:
        pat = re.compile(family, re.IGNORECASE)
        filtered = [r for r in filtered if pat.search(r["model_family"])]

    if since:
        filtered = [r for r in filtered if r["date"] >= since]

    if until:
        filtered = [r for r in filtered if r["date"] <= until]

    if bench:
        pat = re.compile(bench, re.IGNORECASE)
        filtered = [r for r in filtered if any(pat.search(b) for b in r["benchmarks"])]

    if finetuned_only:
        filtered = [r for r in filtered if r["is_finetuned"]]

    if base_only:
        filtered = [r for r in filtered if not r["is_finetuned"]]

    return filtered


# ── Display ───────────────────────────────────────────────────────────────

def _trunc(s: str, maxlen: int) -> str:
    return s if len(s) <= maxlen else s[: maxlen - 1] + "…"


def display_runs(runs: list[dict], show_paths: bool = False):
    """Print a formatted table of runs."""
    if not runs:
        print("No runs found matching criteria.")
        return

    # Header
    idx_w = max(3, len(str(len(runs) - 1)))
    model_w = min(40, max(5, max(len(r["model"]) for r in runs)))
    fam_w = min(12, max(6, max(len(r["model_family"]) for r in runs)))

    hdr = (
        f"{'#':>{idx_w}}  {'Date':10}  {'Time':8}  "
        f"{'Model':<{model_w}}  {'Family':<{fam_w}}  "
        f"{'Type':8}  {'Bench':5}  {'Files':5}  Benchmarks"
    )
    print(hdr)
    print("─" * len(hdr))

    for i, r in enumerate(runs):
        model_type = "tuned" if r["is_finetuned"] else "base"
        bench_str = ", ".join(r["benchmarks"]) if r["benchmarks"] else "(none)"
        line = (
            f"{i:>{idx_w}}  {r['date']:10}  {r['time']:8}  "
            f"{_trunc(r['model'], model_w):<{model_w}}  "
            f"{_trunc(r['model_family'], fam_w):<{fam_w}}  "
            f"{model_type:8}  {r['n_benchmarks']:5}  "
            f"{r['n_parquets']:5}  {bench_str}"
        )
        print(line)

    if show_paths:
        print()
        for i, r in enumerate(runs):
            print(f"  [{i}] {r['path']}")

    print(f"\n{len(runs)} runs total")


def build_inspector_cmd(
    selected: list[dict],
    output: str = "inspection.html",
    extra_args: str = "",
) -> str:
    """Build a completion_inspector command from selected runs."""
    run_args = []
    for r in selected:
        # Use model name as label, deduplicate by appending date+time if needed
        label = r["model"]
        run_args.append(f'"{label}={r["path"]}"')

    cmd = f"python -m scripts.completion_inspector --runs {' '.join(run_args)} -o {output}"
    if extra_args:
        cmd += f" {extra_args}"
    return cmd


def interactive_select(runs: list[dict]) -> list[dict]:
    """Interactively select runs from the list."""
    display_runs(runs, show_paths=True)
    print()
    try:
        raw = input("Select runs (comma-separated indices, e.g. 0,3,5): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nAborted.")
        sys.exit(0)

    if not raw:
        print("No selection made.")
        sys.exit(0)

    selected = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            idx = int(part)
        except ValueError:
            print(f"ERROR: '{part}' is not a valid index", file=sys.stderr)
            sys.exit(1)
        if idx < 0 or idx >= len(runs):
            print(f"ERROR: Index {idx} out of range (0..{len(runs) - 1})", file=sys.stderr)
            sys.exit(1)
        selected.append(runs[idx])

    return selected


def parse_pick(spec: str, runs: list[dict]) -> list[dict]:
    """Parse --pick spec into selected runs."""
    selected = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            idx = int(part)
        except ValueError:
            print(f"ERROR: '{part}' is not a valid index", file=sys.stderr)
            sys.exit(1)
        if idx < 0 or idx >= len(runs):
            print(f"ERROR: Index {idx} out of range (0..{len(runs) - 1})", file=sys.stderr)
            sys.exit(1)
        selected.append(runs[idx])
    return selected


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Find and select eval_all runs for the completion inspector.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Filters
    parser.add_argument(
        "--model", type=str, default=None,
        help="Filter by model name (regex, case-insensitive)",
    )
    parser.add_argument(
        "--family", type=str, default=None,
        help="Filter by model family (regex, case-insensitive)",
    )
    parser.add_argument(
        "--since", type=str, default=None, metavar="YYYY-MM-DD",
        help="Only runs on or after this date",
    )
    parser.add_argument(
        "--until", type=str, default=None, metavar="YYYY-MM-DD",
        help="Only runs on or before this date",
    )
    parser.add_argument(
        "--bench", type=str, default=None,
        help="Only runs with this benchmark completed (regex)",
    )
    parser.add_argument(
        "--finetuned", action="store_true",
        help="Only show finetuned (SFT/GRPO) models",
    )
    parser.add_argument(
        "--base", action="store_true",
        help="Only show base (zero-shot) models",
    )

    # Selection
    parser.add_argument(
        "--select", action="store_true",
        help="Interactive mode: display runs then prompt for selection",
    )
    parser.add_argument(
        "--pick", type=str, default=None, metavar="INDICES",
        help="Non-interactive selection: comma-separated indices (e.g. 0,3,5)",
    )

    # Output
    parser.add_argument(
        "--cmd", action="store_true",
        help="Emit a completion_inspector command (implied by --select/--pick)",
    )
    parser.add_argument(
        "-o", "--output", type=str, default="inspection.html",
        help="Output filename for the generated inspector command (default: inspection.html)",
    )
    parser.add_argument(
        "--inspector-args", type=str, default="",
        help='Extra args to pass to completion_inspector (e.g. \'--rows "0:50"\')',
    )
    parser.add_argument(
        "--paths", action="store_true",
        help="Show full paths in the listing",
    )

    # Scan location
    parser.add_argument(
        "--root", type=str, default=DEFAULT_MULTIRUN_ROOT,
        help=f"Multirun root directory (default: {DEFAULT_MULTIRUN_ROOT})",
    )

    args = parser.parse_args()

    if args.finetuned and args.base:
        parser.error("--finetuned and --base are mutually exclusive")

    # Validate date formats
    for date_arg, name in [(args.since, "--since"), (args.until, "--until")]:
        if date_arg:
            try:
                datetime.strptime(date_arg, "%Y-%m-%d")
            except ValueError:
                parser.error(f"{name} must be YYYY-MM-DD format, got '{date_arg}'")

    # Discover and filter
    all_runs = find_eval_all_runs(args.root)
    runs = filter_runs(
        all_runs,
        model=args.model,
        family=args.family,
        since=args.since,
        until=args.until,
        bench=args.bench,
        finetuned_only=args.finetuned,
        base_only=args.base,
    )

    # Selection mode
    if args.select:
        selected = interactive_select(runs)
        cmd = build_inspector_cmd(selected, output=args.output, extra_args=args.inspector_args)
        print(f"\n{cmd}")
    elif args.pick is not None:
        display_runs(runs, show_paths=args.paths)
        selected = parse_pick(args.pick, runs)
        cmd = build_inspector_cmd(selected, output=args.output, extra_args=args.inspector_args)
        print(f"\n{cmd}")
    elif args.cmd:
        # --cmd without --pick/--select: use all filtered runs
        display_runs(runs, show_paths=args.paths)
        if runs:
            cmd = build_inspector_cmd(runs, output=args.output, extra_args=args.inspector_args)
            print(f"\n{cmd}")
    else:
        display_runs(runs, show_paths=args.paths)


if __name__ == "__main__":
    main()
