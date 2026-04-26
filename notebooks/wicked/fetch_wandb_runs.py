#!/usr/bin/env python3
"""Fetch all W&B runs from the eval-all project and cache enriched JSON + CSV.

Usage:
    python fetch_wandb_runs.py                 # fetch all runs
    python fetch_wandb_runs.py --force          # overwrite existing cache
    python fetch_wandb_runs.py --output-dir .   # custom output directory

The script saves two files into <output_dir>/wandb_cache/:
    runs.json   — list of enriched run dicts (full config, summary, tags, computed metrics)
    runs.csv    — flat CSV with one row per (run, dagspace) containing pre-computed metrics
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import wandb

WANDB_ENTITY = "uair"
WANDB_PROJECT = "eval-all"


# ---------------------------------------------------------------------------
# Metric extraction (mirrors logic from analysis notebooks)
# ---------------------------------------------------------------------------

def _extract_benchmark_metrics(summary: dict, dagspace: str) -> dict:
    """Extract pre-computed benchmark metrics from a run summary."""
    metrics = {}

    if dagspace == "goldcoin_hipaa":
        app_f1 = summary.get("compute_metrics/eval/applicable/f1")
        notapp_f1 = summary.get("compute_metrics/eval/not_applicable/f1")
        permit_f1 = summary.get("compute_metrics/eval/permit/f1")
        forbid_f1 = summary.get("compute_metrics/eval/forbid/f1")
        if app_f1 is not None and notapp_f1 is not None:
            metrics["gc_applicability_f1"] = (app_f1 + notapp_f1) / 2
        if permit_f1 is not None and forbid_f1 is not None:
            metrics["gc_compliance_f1"] = (permit_f1 + forbid_f1) / 2
        # Also store raw components
        metrics["gc_applicable_f1"] = app_f1
        metrics["gc_not_applicable_f1"] = notapp_f1
        metrics["gc_permit_f1"] = permit_f1
        metrics["gc_forbid_f1"] = forbid_f1

    elif dagspace == "privacylens":
        metrics["pl_qa_accuracy"] = summary.get("compute_metrics/eval/qa_accuracy")
        metrics["pl_leakage_rate"] = summary.get("compute_metrics/eval/leakage_rate")
        metrics["pl_adjusted_leakage_rate"] = summary.get("compute_metrics/eval/adjusted_leakage_rate")
        metrics["pl_helpful_rate"] = summary.get("compute_metrics/eval/helpful_rate")

    elif dagspace == "vlm_geoprivacy_bench":
        metrics["vlm_q7_accuracy"] = summary.get("compute_metrics/eval/Q7/accuracy")

    elif dagspace == "confaide":
        metrics["ca_pearson_r"] = summary.get("compute_metrics/eval/pearson_r")
        metrics["ca_accuracy"] = summary.get("compute_metrics/eval/accuracy")
        metrics["ca_leak_rate"] = summary.get("compute_metrics/eval/leak_rate")

    elif dagspace == "cirl_vignettes":
        metrics["cirl_accuracy"] = summary.get("compute_metrics/eval/accuracy")
        # Trajectory metrics (from cirl_trajectory_eval orchestrator runs)
        metrics["cirl_integrity"] = summary.get("compute_trajectory_metrics/eval/integrity")
        metrics["cirl_utility"] = summary.get("compute_trajectory_metrics/eval/utility")
        metrics["cirl_complete"] = summary.get("compute_trajectory_metrics/eval/complete")

    return metrics


def _get_dagspace(tags: list[str]) -> str | None:
    for t in tags:
        if t.startswith("bench:"):
            return t.split(":", 1)[1]
    return None


def enrich_run(run) -> dict:
    """Convert a wandb Run object into a serializable enriched dict."""
    config = run.config
    summary = {k: v for k, v in run.summary.items() if not k.startswith("_")}
    tags = list(run.tags)
    dagspace = _get_dagspace(tags)

    checkpoint = config.get("model", {}).get("checkpoint_name", "")

    # Compute benchmark metrics
    bench_metrics = _extract_benchmark_metrics(summary, dagspace) if dagspace else {}

    return {
        # Identity
        "run_id": run.id,
        "run_name": run.name,
        "run_url": run.url,
        "state": run.state,
        "created_at": run.created_at,
        # Tags
        "tags": tags,
        "dagspace": dagspace,
        # Model info
        "checkpoint_name": checkpoint,
        "model_config": config.get("model", {}),
        # Full config and summary (for arbitrary filtering)
        "config": config,
        "summary": summary,
        # Pre-computed metrics
        **bench_metrics,
    }


def build_flat_csv(runs: list[dict]) -> pd.DataFrame:
    """Build a flat DataFrame suitable for quick filtering."""
    rows = []
    for r in runs:
        row = {
            "run_id": r["run_id"],
            "run_name": r["run_name"],
            "state": r["state"],
            "created_at": r["created_at"],
            "dagspace": r["dagspace"],
            "checkpoint_name": r["checkpoint_name"],
            "tags": "|".join(r["tags"]),
        }
        # Add all pre-computed metric columns
        for k in [
            "gc_applicability_f1", "gc_compliance_f1",
            "gc_applicable_f1", "gc_not_applicable_f1",
            "gc_permit_f1", "gc_forbid_f1",
            "pl_qa_accuracy", "pl_leakage_rate",
            "pl_adjusted_leakage_rate", "pl_helpful_rate",
            "vlm_q7_accuracy",
            "ca_pearson_r", "ca_accuracy", "ca_leak_rate",
            "cirl_accuracy", "cirl_integrity", "cirl_utility", "cirl_complete",
        ]:
            if k in r:
                row[k] = r[k]
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent,
                        help="Parent dir; cache goes into <output-dir>/wandb_cache/")
    parser.add_argument("--force", action="store_true", help="Overwrite existing cache")
    parser.add_argument("--entity", default=WANDB_ENTITY)
    parser.add_argument("--project", default=WANDB_PROJECT)
    args = parser.parse_args()

    cache_dir = args.output_dir / "wandb_cache"
    json_path = cache_dir / "runs.json"
    csv_path = cache_dir / "runs.csv"

    if json_path.exists() and not args.force:
        print(f"Cache already exists at {json_path}. Use --force to overwrite.")
        sys.exit(0)

    cache_dir.mkdir(parents=True, exist_ok=True)

    api = wandb.Api()
    path = f"{args.entity}/{args.project}"
    print(f"Fetching all runs from {path} ...")
    raw_runs = api.runs(path)
    print(f"  Found {len(raw_runs)} runs total")

    enriched = []
    for i, run in enumerate(raw_runs):
        enriched.append(enrich_run(run))
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(raw_runs)} runs")

    print(f"  Processed {len(enriched)}/{len(raw_runs)} runs")

    # Save JSON
    with open(json_path, "w") as f:
        json.dump(enriched, f, indent=2, default=str)
    print(f"Saved {json_path} ({json_path.stat().st_size / 1024:.0f} KB)")

    # Save CSV
    df = build_flat_csv(enriched)
    df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path} ({len(df)} rows)")

    # Metadata
    meta = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "entity": args.entity,
        "project": args.project,
        "total_runs": len(enriched),
    }
    with open(cache_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
