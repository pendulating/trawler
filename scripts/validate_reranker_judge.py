#!/usr/bin/env python
"""Offline validation: does the reranker judge agree with the LLM judge?

The decisive, cheap experiment before committing GRPO compute to the reranker
R_ground backend. Reads a run's ``reward_traces.jsonl`` (which records the
Qwen3.6-27B judge's per-flow grounding scores alongside the retrieved norm
snippets and the completion), re-scores the same (norms, candidate) pairs with
Qwen3-Reranker-8B via vLLM ``/rerank``, and reports:

  * Spearman / Pearson correlation of reranker grounding vs teacher grounding
    (per flow) — does the cheap judge rank groundedness like the expensive one?
  * For ``ranked``-mode traces: per-group rank agreement (teacher rank vs
    reranker-derived rank) and the reranker's within-group **tie rate** — the
    pathology (60% tied groups → zero advantage) that listwise judging fixed.
    A reranker with continuous scores should tie far less.

Caveat: traces store norm *snippets* (truncated to 120 chars), not the full
retrieved norm JSON, so this is a slightly lossy proxy for what the reranker
would see during training. High correlation here is strong positive evidence;
a weak result warrants the full-fidelity re-retrieval path before concluding.

Usage:
  # 1. Launch the reranker server: sbatch scripts/reranker_server.sub
  # 2. export RERANKER_SERVER_URL=http://<host>:8003
  python scripts/validate_reranker_judge.py \
    --traces outputs/<run>/reward_traces.jsonl \
    --reranker-url $RERANKER_SERVER_URL \
    --reranker-model /share/pierson/matt/zoo/models/Qwen3-Reranker-8B \
    --limit 2000
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

# Import the production client so the validation scores flows exactly as
# training would (same query construction, same /rerank call).
from dagspaces.grpo_training.stages.clients import RerankerJudgeClient
from dagspaces.grpo_training.stages.parsing import parse_completion as _parse_completion
from dagspaces.grpo_training.stages.online_rground import _flow_to_query, _flatten_flow


def _spearman(xs: List[float], ys: List[float]) -> Optional[float]:
    """Spearman rank correlation; scipy if available, else a numpy fallback."""
    if len(xs) < 3:
        return None
    try:
        from scipy.stats import spearmanr
        rho, _ = spearmanr(xs, ys)
        return float(rho)
    except Exception:
        import numpy as np

        def _rank(a):
            order = np.argsort(a, kind="mergesort")
            ranks = np.empty(len(a), dtype=float)
            ranks[order] = np.arange(len(a), dtype=float)
            # average ties
            a = np.asarray(a, dtype=float)
            _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
            csum = np.cumsum(counts)
            start = csum - counts
            avg = (start + csum - 1) / 2.0
            return avg[inv]

        rx, ry = _rank(xs), _rank(ys)
        rx -= rx.mean(); ry -= ry.mean()
        denom = (np.sqrt((rx ** 2).sum()) * np.sqrt((ry ** 2).sum()))
        return float((rx * ry).sum() / denom) if denom else None


def _pearson(xs: List[float], ys: List[float]) -> Optional[float]:
    import numpy as np
    if len(xs) < 3:
        return None
    x = np.asarray(xs, float); y = np.asarray(ys, float)
    x -= x.mean(); y -= y.mean()
    denom = np.sqrt((x ** 2).sum()) * np.sqrt((y ** 2).sum())
    return float((x * y).sum() / denom) if denom else None


def _candidate_doc_from_completion(completion: str) -> Optional[str]:
    """Parse a completion into the same flow-text the reranker would score."""
    parsed = _parse_completion(completion)
    if not parsed:
        return None
    extractions = parsed.get("extraction", [])
    if not isinstance(extractions, list):
        return None
    flows = [e for e in extractions if isinstance(e, dict)]
    if flows:
        return json.dumps(flows, ensure_ascii=False)
    reasoning = parsed.get("reasoning", {})
    if isinstance(reasoning, dict) and reasoning.get("has_information_exchange") is False:
        return "This candidate declares the passage contains NO information flows."
    return None


def collect_pairs(
    traces_path: str,
    limit: int,
) -> Tuple[List[Dict[str, Any]], str]:
    """Pull (norms, candidate, teacher_score, group_key, teacher_rank) records.

    Auto-detects ranked vs absolute traces from the diag ``type`` field.
    """
    records: List[Dict[str, Any]] = []
    mode = "unknown"
    with open(traces_path) as f:
        for line in f:
            if len(records) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            flows = entry.get("rground_flows")
            if not flows:
                continue
            group_key = (entry.get("call"), entry.get("prompt_id"))
            completion = entry.get("completion", "")
            for diag in flows:
                norms = " ".join(diag.get("correct_norm_snippets", []) or [])
                if not norms:
                    continue
                if diag.get("type") == "ranked":
                    mode = "ranked"
                    doc = _candidate_doc_from_completion(completion)
                    teacher = diag.get("grounding_score")
                    teacher_rank = diag.get("rank")
                else:
                    mode = "absolute" if mode == "unknown" else mode
                    doc = diag.get("query")
                    nm = diag.get("correct_norm_match")
                    gov = diag.get("correct_governance")
                    teacher = (0.5 * (nm + gov)) if (nm is not None and gov is not None) else None
                    teacher_rank = None
                if doc and teacher is not None:
                    records.append({
                        "norms": norms,
                        "doc": doc,
                        "teacher": float(teacher),
                        "group_key": group_key,
                        "teacher_rank": teacher_rank,
                    })
    return records, mode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traces", required=True, help="path to reward_traces.jsonl")
    ap.add_argument("--reranker-url", default=None,
                    help="vLLM reranker /rerank base url (default: $RERANKER_SERVER_URL)")
    ap.add_argument("--reranker-model",
                    default="/share/pierson/matt/zoo/models/Qwen3-Reranker-8B")
    ap.add_argument("--limit", type=int, default=2000,
                    help="max trace entries to read")
    ap.add_argument("--max-workers", type=int, default=16)
    args = ap.parse_args()

    import os
    url = args.reranker_url or os.environ.get("RERANKER_SERVER_URL")
    if not url:
        print("ERROR: pass --reranker-url or set RERANKER_SERVER_URL", file=sys.stderr)
        return 2

    records, mode = collect_pairs(args.traces, args.limit)
    if not records:
        print(f"No usable rground_flows found in {args.traces}. "
              f"(Was the run traced with R_ground diagnostics?)", file=sys.stderr)
        return 1
    print(f"Loaded {len(records)} (norms, candidate) pairs from {args.traces} "
          f"[trace mode: {mode}]")

    client = RerankerJudgeClient(
        base_url=url, model_name=args.reranker_model, max_workers=args.max_workers,
    )
    # Score each pair: reuse the production per-flow path (norm_match==score).
    items = [
        {"chunk_text": "", "flow_json": r["doc"], "norm_universe_json": r["norms"]}
        for r in records
    ]
    print(f"Scoring {len(items)} pairs via reranker at {url} ...")
    results = client.judge_batch(items)
    client.close()

    teacher = [r["teacher"] for r in records]
    reranker = [res.get("norm_match_score", 0.0) for res in results]

    rho = _spearman(reranker, teacher)
    r = _pearson(reranker, teacher)
    print("\n=== Per-flow agreement (reranker vs teacher grounding) ===")
    print(f"  n            : {len(records)}")
    print(f"  Spearman rho : {rho:.4f}" if rho is not None else "  Spearman rho : n/a")
    print(f"  Pearson  r   : {r:.4f}" if r is not None else "  Pearson  r   : n/a")

    # Group-level analysis: rank agreement + reranker tie rate (the metric that
    # actually decides viability for GRPO advantage).
    groups: Dict[Any, List[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        groups[rec["group_key"]].append(i)
    multi = {k: idxs for k, idxs in groups.items() if len(idxs) >= 2}
    if multi:
        import numpy as np
        tie_groups = 0
        rank_rhos = []
        for k, idxs in multi.items():
            rr = [reranker[i] for i in idxs]
            if len(set(round(x, 6) for x in rr)) < len(rr):
                tie_groups += 1
            t_ranks = [records[i]["teacher_rank"] for i in idxs]
            if all(tr is not None for tr in t_ranks):
                # teacher_rank: lower = better; reranker: higher = better →
                # negate one so positive rho == agreement.
                gr = _spearman(rr, [-tr for tr in t_ranks])
                if gr is not None:
                    rank_rhos.append(gr)
        print("\n=== Group-level (within-prompt) ===")
        print(f"  groups (>=2 cand) : {len(multi)}")
        print(f"  reranker tie rate : {tie_groups / len(multi):.3f}  "
              f"(lower is better; the 60% tie rate is what killed absolute LLM scoring)")
        if rank_rhos:
            print(f"  mean per-group rank agreement (Spearman) : "
                  f"{sum(rank_rhos) / len(rank_rhos):.4f}  over {len(rank_rhos)} groups")

    print("\nVerdict guide: strong per-flow Spearman (>~0.6) AND low reranker "
          "tie rate (<~0.2) ⇒ the reranker backend is viable zero-shot. Weak "
          "correlation ⇒ distill the reranker on these traces before training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
