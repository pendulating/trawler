"""Backfill helpfulness and adjusted leakage metrics into existing W&B runs.

Finds all completed PrivacyLens eval runs with metrics.json sidecars,
matches them to W&B runs by name/timestamp, and patches the missing
helpfulness and adjusted leakage metrics into the W&B summary.

Usage:
    python scripts/backfill_privacylens_wandb.py [--dry-run]
"""

import argparse
import json
import os
import re
from glob import glob
from pathlib import Path

import wandb

WANDB_ENTITY = "uair"
WANDB_PROJECT = "eval-all"
MULTIRUN_BASE = "/share/pierson/matt/UAIR/multirun"


def find_metrics_files():
    """Find all completed PrivacyLens metrics.json files."""
    pattern = os.path.join(
        MULTIRUN_BASE, "*_eval_all", "*", "0", "privacylens",
        "*", "outputs", "compute_metrics", "metrics.json",
    )
    return sorted(glob(pattern))


def extract_run_timestamp(metrics_path: str) -> str | None:
    """Extract the multirun timestamp from a metrics path.

    Path format: .../multirun/2026-03-28_eval_all/00-27-36/0/privacylens/...
    The orchestrator W&B run name contains a timestamp derived from the
    Hydra output dir, formatted as YYYYMMDD-HHMMSS.
    """
    parts = Path(metrics_path).parts
    for i, p in enumerate(parts):
        if p.endswith("_eval_all") and i + 1 < len(parts):
            date_part = p.split("_eval_all")[0]  # e.g., "2026-03-28"
            time_part = parts[i + 1]  # e.g., "00-27-36"
            # Convert to the format used in W&B run names
            date_compact = date_part.replace("-", "")  # "20260328"
            # The orchestrator run name uses the UTC time
            # from Hydra, which may differ. We'll match fuzzily.
            return f"{date_compact}_{time_part}"
    return None


def extract_output_root(metrics_path: str) -> str:
    """Extract the privacylens output_root from a metrics path."""
    # .../privacylens/privacylens_eval/outputs/compute_metrics/metrics.json
    # → .../privacylens/privacylens_eval
    return str(Path(metrics_path).parent.parent.parent)


def load_metrics(metrics_path: str) -> dict:
    """Load helpfulness and adjusted leakage from metrics.json."""
    with open(metrics_path) as f:
        m = json.load(f)

    result = {}
    helpfulness = m.get("helpfulness", {})
    if helpfulness:
        result["compute_metrics/eval/helpful_rate"] = helpfulness.get("helpful_rate", 0.0)
        result["compute_metrics/eval/helpfulness_mean_score"] = helpfulness.get("mean_score", 0.0)
        result["compute_metrics/eval/helpful_count"] = helpfulness.get("helpful_count", 0)
        result["compute_metrics/eval/helpfulness_total"] = helpfulness.get("total", 0)

    adj = m.get("adjusted_leakage", {})
    if adj:
        result["compute_metrics/eval/adjusted_leakage_rate"] = adj.get("adjusted_leakage_rate", 0.0)
        result["compute_metrics/eval/adjusted_leakage_total_helpful"] = adj.get("total_helpful", 0)
        result["compute_metrics/eval/adjusted_leakage_leaking_among_helpful"] = adj.get("leaking_among_helpful", 0)

    return result


def match_wandb_run(api, output_root: str, wandb_runs: list) -> wandb.apis.public.Run | None:
    """Match a local output_root to a W&B run.

    Strategy: the orchestrator W&B run logs output_root in its config
    or we match by run name containing the Hydra timestamp.
    """
    # Strategy 1: match by output_root in the W&B run name
    # The orchestrator run name format: privacylens_clean-privacylens_eval-orchestrator-YYYYMMDD-HHMMSS
    # We need to find the run whose name matches our directory's time portion.

    # Extract time from output_root: .../00-27-36/0/privacylens/privacylens_eval
    parts = Path(output_root).parts
    for i, p in enumerate(parts):
        if p == "0" and i >= 1:
            time_dir = parts[i - 1]  # e.g., "00-27-36"
            break
    else:
        return None

    for run in wandb_runs:
        if "orchestrator" not in run.name:
            continue
        # Run name like: privacylens_clean-privacylens_eval-orchestrator-20260328-050048
        # The time_dir 00-27-36 in EDT is 04:27:36 UTC → "042736" in the run name
        # But actually the Hydra dir uses local time and the run name uses local time too.
        # Let's just match by checking if the run's summary already has leakage_rate
        # (confirming it's a compute_metrics orchestrator run) and comparing timestamps.

        # Simpler: check if run logged the output_root path
        # The orchestrator logs it but not always to W&B config.
        # Let's use a different approach: match by the leakage metrics that ARE present
        pass

    return None


def main():
    parser = argparse.ArgumentParser(description="Backfill PrivacyLens helpfulness metrics to W&B")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be updated without writing")
    args = parser.parse_args()

    api = wandb.Api()

    # Fetch all privacylens orchestrator runs
    all_runs = api.runs(
        f"{WANDB_ENTITY}/{WANDB_PROJECT}",
        filters={"tags": {"$in": ["bench:privacylens"]}},
        per_page=500,
    )
    # Keep only orchestrator runs (they have compute_metrics data)
    orchestrator_runs = [r for r in all_runs if "orchestrator" in r.name]
    print(f"Found {len(orchestrator_runs)} PrivacyLens orchestrator W&B runs")

    # Build lookup: (leakage_rate, qa_accuracy) → run
    # This is a fuzzy match since those values are unique per model/condition combo
    run_lookup = {}
    for r in orchestrator_runs:
        lr = r.summary.get("compute_metrics/eval/leakage_rate")
        qa = r.summary.get("compute_metrics/eval/qa_accuracy")
        if lr is not None and qa is not None:
            key = (round(lr, 4), round(qa, 4))
            run_lookup[key] = r

    print(f"Built lookup with {len(run_lookup)} runs that have leakage+qa metrics")

    # Find all local metrics files
    metrics_files = find_metrics_files()
    print(f"Found {len(metrics_files)} local metrics.json files")

    updated = 0
    skipped = 0
    no_match = 0

    for mf in metrics_files:
        metrics = load_metrics(mf)
        if not metrics:
            skipped += 1
            continue

        # Load the full metrics to get leakage_rate and qa_accuracy for matching
        with open(mf) as f:
            full = json.load(f)
        lr = full.get("leakage", {}).get("leakage_rate")
        qa = full.get("qa_probing", {}).get("accuracy")
        if lr is None or qa is None:
            skipped += 1
            continue

        key = (round(lr, 4), round(qa, 4))
        run = run_lookup.get(key)
        if run is None:
            # Try without rounding
            for rkey, rval in run_lookup.items():
                if abs(rkey[0] - lr) < 0.001 and abs(rkey[1] - qa) < 0.001:
                    run = rval
                    break

        if run is None:
            output_root = extract_output_root(mf)
            print(f"  NO MATCH: lr={lr:.4f} qa={qa:.4f} | {output_root}")
            no_match += 1
            continue

        # Check if already backfilled
        existing_hr = run.summary.get("compute_metrics/eval/helpful_rate")
        if existing_hr is not None:
            skipped += 1
            continue

        helpful = full.get("helpfulness", {})
        adj = full.get("adjusted_leakage", {})
        hr = helpful.get("helpful_rate", "?")
        ar = adj.get("adjusted_leakage_rate", "?")

        if args.dry_run:
            print(f"  WOULD UPDATE: {run.name} (id={run.id})")
            print(f"    helpful_rate={hr}, adjusted_leakage_rate={ar}")
        else:
            for k, v in metrics.items():
                run.summary[k] = v
            run.summary.update()
            print(f"  UPDATED: {run.name} (id={run.id})")
            print(f"    helpful_rate={hr}, adjusted_leakage_rate={ar}")

        updated += 1

    print(f"\nDone: {updated} updated, {skipped} skipped, {no_match} no match")


if __name__ == "__main__":
    main()
