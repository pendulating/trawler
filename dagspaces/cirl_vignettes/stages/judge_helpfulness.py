"""Judge helpfulness stage for CIRL-Vignettes trajectory evaluation.

Scores each case's generated final action on a 0-3 helpfulness rubric
using a judge LLM via JudgeClient.

Reference: CI-RL evaluate_final_action.py (step=helpfulness)
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
from omegaconf import DictConfig

from dagspaces.common.judge_client import JudgeClient
from ..prompts import build_helpfulness_judge_prompt, parse_helpfulness_score


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
    )

    if not client.health_check():
        raise ConnectionError(
            f"Judge server not reachable at {judge_cfg.base_url}. "
            "Start it via scripts/judge_server.sub or set judge.base_url."
        )
    print(f"[judge_helpfulness] Connected to judge: {client.model_name}",
          flush=True)

    # Build items for judging
    items: List[Dict[str, Any]] = []
    for i, (idx, row) in enumerate(df.iterrows()):
        action = str(row.get("final_action_generated", ""))
        if not action or "Action:" not in action:
            continue

        # Pass full action text (with Action: prefix) to helpfulness judge —
        # CI-RL evaluate_final_action.py strips the prefix for leakage but
        # NOT for helpfulness.
        items.append({
            "row_idx": idx,
            "user_name": str(row.get("user_name", "")),
            "user_instruction": str(row.get("user_instruction", "")),
            "trajectory": str(row.get("executable_trajectory", "")),
            "action": action,
        })

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

    df = df.copy()
    df["helpfulness_score"] = df.index.map(lambda idx: scores.get(idx, 0))
    df["helpfulness_raw"] = df.index.map(lambda idx: raw_texts.get(idx, ""))

    avg_score = df["helpfulness_score"].mean()
    helpful_rate = (df["helpfulness_score"] >= 2).mean()
    print(f"[judge_helpfulness] Avg score: {avg_score:.2f}, "
          f"helpful rate (>=2): {helpful_rate:.1%}", flush=True)

    return df
