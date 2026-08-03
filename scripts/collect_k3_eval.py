#!/usr/bin/env python3
"""Collect the K4 (revived) CI-benchmark sweep into one comparison table.

Reads every cell of an eval_all multirun, flattens each benchmark's
metrics.json, and prints each arm against the reference cell.

COMPARISON DISCIPLINE (see the sweep yaml): arms are compared to `k3-base` —
the merged SFT they were trained from — IN THIS SWEEP. Never against the v9
numbers in the paper: different SFT lineage and PrivacyLens protocol drift make
those non-comparable.

Every metric is printed next to its parseable_rate. That is not decoration:
the scrutinize arm gate-fails 12-32% on the extract task (wiki 16), so a low
score there may be format damage rather than reasoning. A metric whose
parseable_rate moved materially is not a like-for-like comparison.

Usage:
    python scripts/collect_k3_eval.py \
        --sweep multirun/2026-08-03_k3_arms_ci_eval/11-46-24
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Metric keys worth surfacing, in priority order per benchmark. Anything
# matching is printed; the list is deliberately generous because each
# benchmark's metrics.json has its own shape.
INTERESTING = (
    "f1_macro", "macro_f1", "accuracy", "accuracy_among_parseable",
    "leakage_rate", "helpfulness", "leakage_among_helpful",
    "pearson_r", "correlation", "r", "parseable_rate", "n_valid", "n_total",
)
# Substrings marking a metric as a parse-health signal rather than a score.
HEALTH = ("parseable_rate", "n_valid", "n_total")
# Subtrees that record denominator provenance, not results. Keeping them buries
# the actual scores under hundreds of identical n_total rows.
SKIP_SUBTREES = ("metric_provenance",)


def flatten(obj, prefix=""):
    """Flatten nested metrics.json into dotted keys."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}{k}." if not prefix else f"{prefix}{k}."))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out[prefix.rstrip(".")] = float(obj)
    return out


def cell_model(cell: Path) -> str:
    """Read the model override for a sweep cell."""
    ov = cell / ".hydra" / "overrides.yaml"
    if ov.is_file():
        for line in ov.read_text().splitlines():
            line = line.strip().lstrip("- ")
            if line.startswith("model="):
                return line.split("=", 1)[1].split("/")[-1]
    return cell.name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--ref", default="k3-base",
                    help="reference cell to diff against")
    args = ap.parse_args()

    sweep = Path(args.sweep)
    cells = sorted([p for p in sweep.iterdir() if p.is_dir()
                    and p.name.isdigit()], key=lambda p: int(p.name))
    if not cells:
        raise SystemExit(f"no cells under {sweep}")

    # model -> benchmark -> {metric: value}
    data: dict[str, dict[str, dict[str, float]]] = {}
    for cell in cells:
        model = cell_model(cell)
        data.setdefault(model, {})
        for mfile in sorted(cell.rglob("metrics.json")):
            # .../<benchmark>/<dagspace>/outputs/<stage>/metrics.json
            try:
                bench = mfile.relative_to(cell).parts[0]
            except ValueError:
                continue
            try:
                raw = json.loads(mfile.read_text())
            except Exception as e:
                print(f"[warn] unreadable {mfile}: {e}")
                continue
            flat = flatten(raw)
            keep = {
                k: v for k, v in flat.items()
                if any(k.split(".")[-1] == m for m in INTERESTING)
                and not any(s in k.split(".") for s in SKIP_SUBTREES)
            }
            data[model].setdefault(bench, {}).update(keep)

    models = list(data)
    if args.ref not in models:
        print(f"[warn] reference {args.ref!r} not among cells {models}; "
              f"printing absolute values only")
    benches = sorted({b for m in data.values() for b in m})

    for bench in benches:
        print(f"\n{'=' * 78}\n{bench}\n{'=' * 78}")
        keys = sorted({k for m in models for k in data[m].get(bench, {})})
        if not keys:
            print("  (no metrics yet)")
            continue
        w = max(len(k) for k in keys) + 2
        header = f"{'metric':<{w}}" + "".join(f"{m:>22}" for m in models)
        print(header)
        print("-" * len(header))
        for k in keys:
            is_health = any(h in k for h in HEALTH)
            row = f"{k:<{w}}"
            ref_v = data.get(args.ref, {}).get(bench, {}).get(k)
            for m in models:
                v = data[m].get(bench, {}).get(k)
                if v is None:
                    row += f"{'-':>22}"
                elif m == args.ref or ref_v is None or is_health:
                    row += f"{v:>22.4f}"
                else:
                    row += f"{v:>15.4f} ({v - ref_v:+.3f})"
            print(row + ("   [parse-health]" if is_health else ""))

    print(f"\n{'=' * 78}")
    print("Deltas are vs the reference cell IN THIS SWEEP. Read every score "
          "next to its parseable_rate before\ncalling a difference a reasoning "
          "result — see the sweep yaml's caveats.")


if __name__ == "__main__":
    main()
