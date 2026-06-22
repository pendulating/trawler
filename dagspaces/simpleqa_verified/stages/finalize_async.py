"""Async-judge finalize: merge sidecar output.jsonl back into pending parquet.

Reads ``pending.parquet`` + ``items.parquet`` + ``output.jsonl`` from the
judge_grade export dir, joins on ``custom_id``, parses each row's
A/B/C grade, computes SimpleQA metrics, and writes
``metrics.json`` + ``metrics.parquet`` under
``<output_root>/outputs/compute_metrics/``.

Same on-disk schema and parsing semantics as the live mode — async only
changes who fills ``output.jsonl`` (the eval_all sidecar in real time
vs. a manual one-shot smoke test).
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import pandas as pd

from dagspaces.common.batch_api import classify_response_line, load_jsonl

from .judge_grade import parse_grade_letter, letter_to_verdict


JUDGE_DIR = "outputs/judge_grade"


def _load_classified_responses(output_jsonl: str) -> Dict[str, Dict[str, Any]]:
    """``{custom_id: classified-response-dict}`` for every JSONL line."""
    out: Dict[str, Dict[str, Any]] = {}
    for line in load_jsonl(output_jsonl):
        cid = line.get("custom_id")
        if not cid:
            continue
        out[cid] = classify_response_line(line)
    return out


def finalize_async(
    output_root: str,
    *,
    metrics_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Drain + parse + compute_metrics for one simpleqa-verified async run.

    Args:
        output_root: The pipeline ``output_root`` (typically
            ``${hydra:run.dir}/simpleqa_verified``).
        metrics_dir: Where to write metrics outputs. Defaults to
            ``<output_root>/outputs/compute_metrics``.

    Returns a dict with the verdict DataFrame + metrics paths so the
    runner can hand them to the sanity layer.
    """
    from .compute_metrics import compute_metrics, metrics_to_dataframe

    judge_dir = os.path.join(output_root, JUDGE_DIR)
    pending_path = os.path.join(judge_dir, "pending.parquet")
    items_path = os.path.join(judge_dir, "items.parquet")
    output_jsonl = os.path.join(judge_dir, "output.jsonl")

    for p in (pending_path, items_path, output_jsonl):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"[simpleqa_verified.finalize_async] {p} not found — "
                f"async judging hasn't drained yet. Run the sidecar against "
                f"{judge_dir} or hand-submit "
                f"{os.path.join(judge_dir, 'requests.jsonl')} to OpenAI Batch."
            )

    pending_df = pd.read_parquet(pending_path)
    items_df = pd.read_parquet(items_path)
    classified = _load_classified_responses(output_jsonl)

    # Per-item judge content and ok flag.
    items_df["judge_ok"] = items_df["judge_custom_id"].map(
        lambda cid: bool((classified.get(cid) or {}).get("ok", False))
    )
    items_df["judge_content"] = items_df["judge_custom_id"].map(
        lambda cid: str((classified.get(cid) or {}).get("content", "") or "")
    )
    items_df["judge_error_kind"] = items_df["judge_custom_id"].map(
        lambda cid: (classified.get(cid) or {}).get("error_kind")
    )

    # Join back onto pending by row_idx (== question_id).
    pending_df = pending_df.copy()
    pending_df["row_idx"] = pending_df["question_id"].astype(int)
    merged = pending_df.merge(
        items_df[["row_idx", "judge_ok", "judge_content", "judge_error_kind"]],
        on="row_idx", how="left",
    )
    merged["judge_response"] = merged["judge_content"].fillna("").astype(str)
    merged["grade_letter"] = merged.apply(
        lambda r: parse_grade_letter(r["judge_response"]) if bool(r.get("judge_ok", False)) else "unparseable",
        axis=1,
    )
    merged["verdict"] = merged["grade_letter"].apply(letter_to_verdict)
    merged["parse_status"] = merged.apply(
        lambda r: (
            "judge_api_error" if not bool(r.get("judge_ok", False)) and r.get("judge_error_kind")
            else ("unparseable" if r["verdict"] == "unparseable" else "parsed")
        ),
        axis=1,
    )

    n_api_errors = int((~merged["judge_ok"].fillna(False).astype(bool)).sum())
    n_unp = int((merged["verdict"] == "unparseable").sum())
    print(
        f"[finalize_async] {len(merged)} rows | judge_api_errors={n_api_errors} | "
        f"unparseable={n_unp} | distribution: {merged['verdict'].value_counts().to_dict()}",
        flush=True,
    )

    metrics = compute_metrics(merged)

    metrics_dir = metrics_dir or os.path.join(output_root, "outputs", "compute_metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_json = os.path.join(metrics_dir, "metrics.json")
    metrics_parquet = os.path.join(metrics_dir, "metrics.parquet")
    with open(metrics_json, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    metrics_to_dataframe(metrics).to_parquet(metrics_parquet, index=False)

    return {
        "metrics_json": metrics_json,
        "metrics_parquet": metrics_parquet,
        "verdicts_df": merged,
        "metrics": metrics,
    }
