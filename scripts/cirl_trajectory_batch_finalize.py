"""Finalize a CIRL-Vignettes trajectory batch-export run.

After you've run ``pipeline=cirl_trajectory_batch`` and downloaded both
judge output JSONLs via ``python -m dagspaces.common.batch_api fetch``,
this script:

    1. Joins each ``output.jsonl`` with the ``items.parquet`` sidecar to
       recover per-request (row_idx, secret) mapping.
    2. Aggregates leakage per row (has_leakage = any secret leaked) using
       ``parse_leakage_judgment`` from the dagspace prompts module.
    3. Aggregates helpfulness per row using ``parse_helpfulness_score``.
    4. Writes the combined judged dataframe to
       ``<run_dir>/outputs/judge_helpfulness_batch/results.parquet`` and
       runs ``compute_trajectory_metrics`` against it, emitting
       ``metrics.json`` + ``metrics.parquet``.

Usage::

    python scripts/cirl_trajectory_batch_finalize.py \\
        --run-dir outputs/2026-04-10_cirl_vignettes/14-22-01/cirl_vignettes
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict

import pandas as pd

from dagspaces.common.batch_api import extract_content, load_jsonl
from dagspaces.cirl_vignettes.prompts import (
    parse_helpfulness_score,
    parse_leakage_judgment,
)
from dagspaces.cirl_vignettes.stages.compute_trajectory_metrics import (
    compute_trajectory_metrics,
    metrics_to_dataframe,
)


def _responses_by_cid(output_jsonl: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for line in load_jsonl(output_jsonl):
        cid = line.get("custom_id")
        if not cid:
            continue
        out[cid] = extract_content(line)
    return out


def _finalize_leakage(run_dir: str, trajectory_df: pd.DataFrame) -> pd.DataFrame:
    base = os.path.join(run_dir, "outputs", "judge_leakage_batch")
    pending_path = os.path.join(base, "pending.parquet")
    items_path = os.path.join(base, "items.parquet")
    output_jsonl = os.path.join(base, "output.jsonl")
    for p in (pending_path, items_path, output_jsonl):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{p} not found. Did you run pipeline=cirl_trajectory_batch "
                f"and download the judge output.jsonl?"
            )

    items_df = pd.read_parquet(items_path)
    responses = _responses_by_cid(output_jsonl)

    # Parse each (row_idx, secret) response into a leaked bool.
    items_df["response"] = items_df["judge_custom_id"].map(responses).fillna("")
    items_df["leaked"] = items_df["response"].apply(parse_leakage_judgment)

    row_judgments: Dict[object, list] = {idx: [] for idx in trajectory_df.index}
    for _, r in items_df.iterrows():
        row_judgments.setdefault(r["row_idx"], []).append(
            (r["secret"], bool(r["leaked"]))
        )

    df = trajectory_df.copy()
    leaked_items_col, has_leakage_col, leakage_count_col, judgments_col = [], [], [], []
    for idx in df.index:
        judgments = row_judgments.get(idx, [])
        leaked = [s for s, leaked in judgments if leaked]
        leaked_items_col.append(json.dumps(leaked))
        has_leakage_col.append(len(leaked) > 0)
        leakage_count_col.append(len(leaked))
        judgments_col.append(json.dumps(
            [{"secret": s, "leaked": l} for s, l in judgments]
        ))
    df["leaked_items"] = leaked_items_col
    df["has_leakage"] = has_leakage_col
    df["leakage_count"] = leakage_count_col
    df["leakage_judgments"] = judgments_col

    results_path = os.path.join(base, "results.parquet")
    df.to_parquet(results_path, index=False)
    print(f"[finalize] wrote {results_path} ({sum(has_leakage_col)}/{len(df)} "
          f"cases leaking)", flush=True)
    return df


def _finalize_helpfulness(run_dir: str, df: pd.DataFrame) -> pd.DataFrame:
    base = os.path.join(run_dir, "outputs", "judge_helpfulness_batch")
    pending_path = os.path.join(base, "pending.parquet")
    items_path = os.path.join(base, "items.parquet")
    output_jsonl = os.path.join(base, "output.jsonl")
    for p in (pending_path, items_path, output_jsonl):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{p} not found. Did you run pipeline=cirl_trajectory_batch "
                f"and download the judge output.jsonl?"
            )

    items_df = pd.read_parquet(items_path)
    responses = _responses_by_cid(output_jsonl)
    items_df["response"] = items_df["judge_custom_id"].map(responses).fillna("")
    items_df["score"] = items_df["response"].apply(parse_helpfulness_score)

    scores = dict(zip(items_df["row_idx"], items_df["score"]))
    raw = dict(zip(items_df["row_idx"], items_df["response"]))

    out = df.copy()
    out["helpfulness_score"] = out.index.map(lambda idx: int(scores.get(idx, 0)))
    out["helpfulness_raw"] = out.index.map(lambda idx: str(raw.get(idx, "")))

    results_path = os.path.join(base, "results.parquet")
    out.to_parquet(results_path, index=False)
    avg = out["helpfulness_score"].mean() if len(out) else 0.0
    print(f"[finalize] wrote {results_path} (avg helpfulness {avg:.2f})",
          flush=True)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", required=True,
        help="pipeline.output_root from a cirl_trajectory_batch run, "
             "e.g. outputs/YYYY-MM-DD_cirl_vignettes/HH-MM-SS/cirl_vignettes",
    )
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    if not os.path.isdir(run_dir):
        print(f"error: run-dir not found: {run_dir}", file=sys.stderr)
        return 1

    trajectory_path = os.path.join(
        run_dir, "outputs", "trajectory_inference", "dataset.parquet"
    )
    if not os.path.exists(trajectory_path):
        print(f"error: trajectory parquet not found: {trajectory_path}",
              file=sys.stderr)
        return 1

    trajectory_df = pd.read_parquet(trajectory_path)
    leakage_df = _finalize_leakage(run_dir, trajectory_df)
    combined = _finalize_helpfulness(run_dir, leakage_df)

    metrics = compute_trajectory_metrics(combined)
    out_dir = os.path.join(run_dir, "outputs", "compute_trajectory_metrics")
    os.makedirs(out_dir, exist_ok=True)
    metrics_json = os.path.join(out_dir, "metrics.json")
    metrics_parquet = os.path.join(out_dir, "metrics.parquet")
    with open(metrics_json, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    metrics_to_dataframe(metrics).to_parquet(metrics_parquet, index=False)

    print(f"[finalize] wrote {metrics_json}", flush=True)
    print(f"[finalize] wrote {metrics_parquet}", flush=True)
    print("=" * 60)
    print("  CIRL-VIGNETTES BATCH-FINALIZE RESULTS")
    print("=" * 60)
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
