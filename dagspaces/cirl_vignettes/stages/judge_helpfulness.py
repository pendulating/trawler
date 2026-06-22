"""Judge helpfulness stage for CIRL-Vignettes trajectory evaluation.

Scores each case's generated final action on a 0-3 helpfulness rubric
using a judge LLM via JudgeClient.

Reference: CI-RL evaluate_final_action.py (step=helpfulness)
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

import pandas as pd
from omegaconf import DictConfig

from dagspaces.common.judge_client import JudgeClient
from .judge_leakage import _get_batch_export_client, _get_batch_export_endpoint
from ..prompts import build_helpfulness_judge_prompt, parse_helpfulness_score


def _action_has_format(action: str) -> bool:
    """True iff the trajectory's final action contains an ``Action:`` line.

    Same gate CI-RL applies before grading. Without it, the helpfulness
    judge can't run, and pre-2026-04-27 those rows were silently scored
    0 — which then baked into ``avg_helpfulness_score`` and dragged the
    headline number down without any provenance flag.
    """
    a = str(action or "")
    return bool(a) and "Action:" in a


def _build_helpfulness_items(
    df: pd.DataFrame,
) -> "tuple[List[Dict[str, Any]], set]":
    """Flatten rows into helpfulness-judge inputs, skipping no-action rows.

    Returns ``(items, skipped_indices)``. ``skipped_indices`` is the set
    of df row-indices whose ``final_action_generated`` lacks ``Action:``
    — propagated to the result df as ``helpfulness_judged=False`` so
    metrics can compute conditional vs. defaulted variants.
    """
    items: List[Dict[str, Any]] = []
    skipped: set = set()
    for idx, row in df.iterrows():
        action = str(row.get("final_action_generated", ""))
        if not _action_has_format(action):
            skipped.add(idx)
            continue
        items.append({
            "row_idx": idx,
            "user_name": str(row.get("user_name", "")),
            "user_instruction": str(row.get("user_instruction", "")),
            "trajectory": str(row.get("executable_trajectory", "")),
            "action": action,
        })
    return items, skipped


def _stamp_format_columns(
    df: pd.DataFrame,
    *,
    helpfulness_skipped: "set | None" = None,
    leakage_skipped: "set | None" = None,
) -> pd.DataFrame:
    """Return ``df`` with ``agent_action_format_status`` and the relevant
    ``*_judged`` boolean columns populated.

    Idempotent: if ``agent_action_format_status`` is already present
    (e.g. set by a previous judge stage) we don't overwrite it. The
    ``*_judged`` columns are always (re)written from the supplied skip
    set.
    """
    df = df.copy()
    if "agent_action_format_status" not in df.columns and "final_action_generated" in df.columns:
        df["agent_action_format_status"] = df["final_action_generated"].apply(
            lambda a: "valid" if _action_has_format(a) else "no_action_format"
        )
    if helpfulness_skipped is not None:
        df["helpfulness_judged"] = ~df.index.isin(helpfulness_skipped)
    if leakage_skipped is not None:
        df["leakage_judged"] = ~df.index.isin(leakage_skipped)
    return df


def judge_helpfulness(df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """Judge helpfulness of the generated final action.

    Adds columns: helpfulness_score (0-3), helpfulness_raw (judge text).
    """
    judge_cfg = cfg.judge
    client = JudgeClient(
        base_url=str(judge_cfg.base_url),
        model_name=str(getattr(judge_cfg, "model_name", "default")),
        max_workers=int(getattr(judge_cfg, "max_workers", 8)),
        temperature=float(getattr(judge_cfg, "temperature", 0.0)),
        max_tokens=int(getattr(judge_cfg, "max_tokens", 400)),
        provider=(getattr(judge_cfg, "provider", None) or None),
        api_key=(getattr(judge_cfg, "api_key", None) or None),
        api_key_env=(getattr(judge_cfg, "api_key_env", None) or None),
    )

    if not client.health_check():
        raise ConnectionError(
            f"Judge server not reachable at {judge_cfg.base_url}. "
            "Start it via scripts/judge_server.sub or set judge.base_url."
        )
    print(f"[judge_helpfulness] Connected to judge: {client.model_name}",
          flush=True)

    # Build items for judging. Passes full action text (with Action: prefix) —
    # CI-RL evaluate_final_action.py strips the prefix for leakage but not
    # for helpfulness.
    items, skipped_indices = _build_helpfulness_items(df)

    if skipped_indices:
        print(
            f"[judge_helpfulness] {len(skipped_indices)}/{len(df)} cases "
            f"skipped (no_action_format). compute_format_health will "
            f"surface this; metrics report `helpful_rate_among_judged` "
            f"and `helpful_rate_overall_with_default_zero` separately.",
            flush=True,
        )
    print(f"[judge_helpfulness] Judging {len(items)}/{len(df)} cases",
          flush=True)

    if items:
        def build_messages(item: Dict[str, Any]) -> List[Dict[str, str]]:
            prompt = build_helpfulness_judge_prompt(
                user_name=item["user_name"],
                user_instruction=item["user_instruction"],
                trajectory=item["trajectory"],
                action=item["action"],
            )
            return [{"role": "user", "content": prompt}]

        responses = client.judge_batch(items, build_messages)
    else:
        responses = []

    # Map scores back to dataframe
    scores: Dict[Any, int] = {}
    raw_texts: Dict[Any, str] = {}
    for item, response in zip(items, responses):
        scores[item["row_idx"]] = parse_helpfulness_score(response)
        raw_texts[item["row_idx"]] = response

    df = _stamp_format_columns(df, helpfulness_skipped=skipped_indices)
    df["helpfulness_score"] = df.index.map(lambda idx: scores.get(idx, 0))
    df["helpfulness_raw"] = df.index.map(lambda idx: raw_texts.get(idx, ""))

    judged_mask = df["helpfulness_judged"].astype(bool)
    n_judged = int(judged_mask.sum())
    if n_judged > 0:
        avg_score_judged = float(df.loc[judged_mask, "helpfulness_score"].mean())
        helpful_rate_judged = float((df.loc[judged_mask, "helpfulness_score"] >= 2).mean())
    else:
        avg_score_judged = 0.0
        helpful_rate_judged = 0.0
    print(
        f"[judge_helpfulness] Among judged ({n_judged}/{len(df)}): "
        f"avg score {avg_score_judged:.2f}, helpful rate (>=2) {helpful_rate_judged:.1%}",
        flush=True,
    )

    return df


def export_helpfulness_judge_batch(
    df: pd.DataFrame, cfg: DictConfig, output_dir: str
) -> pd.DataFrame:
    """Export helpfulness-judge requests as an OpenAI Batch API JSONL.

    One request per row (rows with no valid Action: are skipped exactly
    as live-mode does). Writes ``requests.jsonl``, ``items.parquet``
    (row_idx ↔ custom_id mapping so finalize can join responses back),
    ``pending.parquet`` (the trajectory dataframe), and ``manifest.json``.
    """
    client = _get_batch_export_client(cfg)

    items, skipped_indices = _build_helpfulness_items(df)
    if skipped_indices:
        print(
            f"[judge_helpfulness_export] {len(skipped_indices)}/{len(df)} "
            f"cases skipped (no_action_format). compute_format_health will "
            f"surface this; metrics distinguish judged vs. defaulted.",
            flush=True,
        )

    def custom_id_fn(item: Dict[str, Any], idx: int) -> str:
        return f"cirl_vignettes:judge_helpfulness:{item['row_idx']}"

    def build_messages(item: Dict[str, Any]) -> List[Dict[str, str]]:
        prompt = build_helpfulness_judge_prompt(
            user_name=item["user_name"],
            user_instruction=item["user_instruction"],
            trajectory=item["trajectory"],
            action=item["action"],
        )
        return [{"role": "user", "content": prompt}]

    os.makedirs(output_dir, exist_ok=True)
    jsonl_path = os.path.join(output_dir, "requests.jsonl")
    manifest = client.export_batch_jsonl(
        items=items,
        build_messages_fn=build_messages,
        output_path=jsonl_path,
        custom_id_fn=custom_id_fn,
        endpoint_url=_get_batch_export_endpoint(cfg),
    )

    items_rows = [
        {"judge_custom_id": custom_id_fn(item, i), "row_idx": item["row_idx"]}
        for i, item in enumerate(items)
    ]
    items_df = pd.DataFrame(items_rows)
    items_path = os.path.join(output_dir, "items.parquet")
    items_df.to_parquet(items_path, index=False)

    manifest.update({
        "dagspace": "cirl_vignettes",
        "stage": "judge_helpfulness",
        "items_parquet": items_path,
    })
    with open(os.path.join(output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    print(
        f"[judge_helpfulness_export] wrote {manifest['count']} requests to "
        f"{jsonl_path} ({len(items)}/{len(df)} cases; "
        f"model={manifest['model']})",
        flush=True,
    )
    return _stamp_format_columns(df, helpfulness_skipped=skipped_indices)
