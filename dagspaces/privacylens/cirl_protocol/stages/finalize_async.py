"""Async-judge finalize for CIRL-Vignettes trajectory evaluation.

Runner-shaped port of ``scripts/cirl_trajectory_batch_finalize.py`` (which
now delegates here). After ``pipeline=cirl_trajectory_async`` has exported
judge requests and the eval_all sidecar has filled each stage's
``output.jsonl`` (+ ``done.flag``), this:

    1. Joins each ``output.jsonl`` with ``items.parquet`` to recover the
       per-request (row_idx, secret) mapping.
    2. OR-aggregates leakage per row; parses helpfulness scores.
    3. Demotes rows whose judge calls all errored to ``*_judged=False``
       with ``skip_reason=judge_api_error`` (explicit provenance instead
       of silently zero-substituting).
    4. Runs ``compute_trajectory_metrics`` and writes ``metrics.json`` +
       ``metrics.parquet``.

Same on-disk schema as the manual OpenAI-Batch flow — async mode only
changes who fills ``output.jsonl`` (the CPU sidecar in real time vs. a
human submitting to the Batch API).
"""

from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd

from dagspaces.common.batch_api import classify_response_line, load_jsonl

from ..prompts import parse_helpfulness_score, parse_leakage_judgment

LEAKAGE_DIR = "outputs/judge_leakage_batch"
HELPFULNESS_DIR = "outputs/judge_helpfulness_batch"
TRAJECTORY_PATH = "outputs/trajectory_inference/dataset.parquet"


def _classified_by_cid(output_jsonl: str) -> dict[str, dict]:
    """Return ``{custom_id: classification}`` distinguishing api errors
    from real responses (see batch_api.classify_response_line)."""
    out: dict[str, dict] = {}
    for line in load_jsonl(output_jsonl):
        cid = line.get("custom_id")
        if not cid:
            continue
        out[cid] = classify_response_line(line)
    return out


def _require(base: str, pipeline_hint: str) -> dict[str, str]:
    paths = {
        "pending": os.path.join(base, "pending.parquet"),
        "items": os.path.join(base, "items.parquet"),
        "output": os.path.join(base, "output.jsonl"),
    }
    for p in paths.values():
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{p} not found. Did {pipeline_hint} run, and has the judge "
                f"sidecar (or a manual batch fetch) filled output.jsonl?"
            )
    return paths


def finalize_leakage(run_dir: str, trajectory_df: pd.DataFrame) -> pd.DataFrame:
    """OR-aggregate per-secret leakage judgments into row-level columns."""
    base = os.path.join(run_dir, *LEAKAGE_DIR.split("/"))
    paths = _require(base, "the leakage export")

    items_df = pd.read_parquet(paths["items"])
    classified = _classified_by_cid(paths["output"])

    items_df["judge_ok"] = items_df["judge_custom_id"].map(
        lambda cid: bool((classified.get(cid) or {}).get("ok", False))
    )
    items_df["response"] = items_df["judge_custom_id"].map(
        lambda cid: str((classified.get(cid) or {}).get("content", "") or "")
    )
    items_df["leaked"] = items_df.apply(
        lambda r: bool(parse_leakage_judgment(r["response"])) if bool(r["judge_ok"]) else False,
        axis=1,
    )

    row_judgments: dict[object, list] = {idx: [] for idx in trajectory_df.index}
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
        print(f"[cirl_finalize] {n_judge_errors} cases demoted to "
              f"leakage_judged=False (judge_api_error)", flush=True)

    results_path = os.path.join(base, "results.parquet")
    df.to_parquet(results_path, index=False)
    print(f"[cirl_finalize] wrote {results_path} "
          f"({sum(has_leakage_col)}/{len(df)} cases leaking)", flush=True)
    return df


def finalize_helpfulness(run_dir: str, df: pd.DataFrame) -> pd.DataFrame:
    """Attach per-row helpfulness scores from the sidecar-filled outputs."""
    base = os.path.join(run_dir, *HELPFULNESS_DIR.split("/"))
    paths = _require(base, "the helpfulness export")

    items_df = pd.read_parquet(paths["items"])
    classified = _classified_by_cid(paths["output"])
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
        print(f"[cirl_finalize] {n_judge_errors} cases demoted to "
              f"helpfulness_judged=False (judge_api_error)", flush=True)

    results_path = os.path.join(base, "results.parquet")
    out.to_parquet(results_path, index=False)
    avg = out["helpfulness_score"].mean() if len(out) else 0.0
    print(f"[cirl_finalize] wrote {results_path} (avg helpfulness {avg:.2f})",
          flush=True)
    return out


def finalize_trajectory_async(
    run_dir: str,
    metrics_dir: str | None = None,
) -> dict[str, Any]:
    """Merge both judges' sidecar outputs and compute trajectory metrics.

    Args:
        run_dir: the pipeline ``output_root`` of the export run (contains
            ``outputs/trajectory_inference`` and both judge export dirs).
        metrics_dir: where to write metrics.{json,parquet}; defaults to
            ``<run_dir>/outputs/compute_trajectory_metrics``.

    Returns dict with ``combined_df``, ``metrics``, ``metrics_json``,
    ``metrics_parquet``.
    """
    from .compute_trajectory_metrics import (
        compute_trajectory_metrics,
        metrics_to_dataframe,
    )

    trajectory_path = os.path.join(run_dir, *TRAJECTORY_PATH.split("/"))
    if not os.path.exists(trajectory_path):
        raise FileNotFoundError(
            f"trajectory parquet not found: {trajectory_path}"
        )
    trajectory_df = pd.read_parquet(trajectory_path)

    leakage_df = finalize_leakage(run_dir, trajectory_df)
    combined = finalize_helpfulness(run_dir, leakage_df)

    metrics = compute_trajectory_metrics(combined)
    out_dir = metrics_dir or os.path.join(
        run_dir, "outputs", "compute_trajectory_metrics"
    )
    os.makedirs(out_dir, exist_ok=True)
    metrics_json = os.path.join(out_dir, "metrics.json")
    metrics_parquet = os.path.join(out_dir, "metrics.parquet")
    with open(metrics_json, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    metrics_to_dataframe(metrics).to_parquet(metrics_parquet, index=False)
    print(f"[cirl_finalize] wrote {metrics_json}", flush=True)
    print(f"[cirl_finalize] wrote {metrics_parquet}", flush=True)

    return {
        "combined_df": combined,
        "metrics": metrics,
        "metrics_json": metrics_json,
        "metrics_parquet": metrics_parquet,
    }
