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

from dagspaces.common.batch_api import classify_response_line, load_jsonl
from dagspaces.cirl_vignettes.prompts import (
    parse_helpfulness_score,
    parse_leakage_judgment,
)
from dagspaces.cirl_vignettes.stages.compute_trajectory_metrics import (
    compute_trajectory_metrics,
    metrics_to_dataframe,
)


def _classified_by_cid(output_jsonl: str) -> Dict[str, Dict]:
    """Return ``{custom_id: classification}`` distinguishing api errors
    from real responses (see batch_api.classify_response_line). Lets
    finalize stamp ``leakage_judged=False`` on rows whose judge call
    errored, so compute_trajectory_metrics defaults them with
    explicit provenance instead of silently zero-substituting.
    """
    out: Dict[str, Dict] = {}
    for line in load_jsonl(output_jsonl):
        cid = line.get("custom_id")
        if not cid:
            continue
        out[cid] = classify_response_line(line)
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
    classified = _classified_by_cid(output_jsonl)

    items_df["judge_ok"] = items_df["judge_custom_id"].map(
        lambda cid: bool((classified.get(cid) or {}).get("ok", False))
    )
    items_df["response"] = items_df["judge_custom_id"].map(
        lambda cid: str((classified.get(cid) or {}).get("content", "") or "")
    )
    # Errored rows produce no parsed leak signal (False), and the row
    # is demoted below.
    items_df["leaked"] = items_df.apply(
        lambda r: bool(parse_leakage_judgment(r["response"])) if bool(r["judge_ok"]) else False,
        axis=1,
    )

    row_judgments: Dict[object, list] = {idx: [] for idx in trajectory_df.index}
    for _, r in items_df.iterrows():
        row_judgments.setdefault(r["row_idx"], []).append(
            (r["secret"], bool(r["leaked"]))
        )
    errors_per_row = (
        items_df.assign(_err=~items_df["judge_ok"].astype(bool))
        .groupby("row_idx")["_err"].all().to_dict()
    )
    successes_per_row = (
        items_df.groupby("row_idx")["judge_ok"].any().to_dict()
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

    # Stamp judge_api_error on rows whose every fanout item errored.
    if "leakage_judged" not in df.columns:
        df["leakage_judged"] = True
    if "leakage_skip_reason" not in df.columns:
        df["leakage_skip_reason"] = ""
    n_judge_errors = 0
    for idx in df.index:
        if idx not in successes_per_row and idx not in errors_per_row:
            continue
        if successes_per_row.get(idx, False):
            continue
        if errors_per_row.get(idx, False):
            df.at[idx, "leakage_judged"] = False
            df.at[idx, "leakage_skip_reason"] = "judge_api_error"
            n_judge_errors += 1
    if n_judge_errors:
        print(f"[finalize] {n_judge_errors} cases demoted to leakage_judged=False "
              f"(judge_api_error)", flush=True)

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
    classified = _classified_by_cid(output_jsonl)
    items_df["judge_ok"] = items_df["judge_custom_id"].map(
        lambda cid: bool((classified.get(cid) or {}).get("ok", False))
    )
    items_df["response"] = items_df["judge_custom_id"].map(
        lambda cid: str((classified.get(cid) or {}).get("content", "") or "")
    )
    items_df["score"] = items_df.apply(
        lambda r: int(parse_helpfulness_score(r["response"])) if bool(r["judge_ok"]) else 0,
        axis=1,
    )

    scores = dict(zip(items_df["row_idx"], items_df["score"]))
    raw = dict(zip(items_df["row_idx"], items_df["response"]))
    ok = dict(zip(items_df["row_idx"], items_df["judge_ok"]))

    out = df.copy()
    out["helpfulness_score"] = out.index.map(lambda idx: int(scores.get(idx, 0)))
    out["helpfulness_raw"] = out.index.map(lambda idx: str(raw.get(idx, "")))

    if "helpfulness_judged" not in out.columns:
        out["helpfulness_judged"] = True
    if "helpfulness_skip_reason" not in out.columns:
        out["helpfulness_skip_reason"] = ""
    n_judge_errors = 0
    for idx in out.index:
        if idx not in ok:
            continue
        if not ok[idx]:
            out.at[idx, "helpfulness_judged"] = False
            out.at[idx, "helpfulness_skip_reason"] = "judge_api_error"
            n_judge_errors += 1
    if n_judge_errors:
        print(f"[finalize] {n_judge_errors} cases demoted to helpfulness_judged=False "
              f"(judge_api_error)", flush=True)

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
