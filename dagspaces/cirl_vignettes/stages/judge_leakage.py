"""Judge leakage stage for CIRL-Vignettes trajectory evaluation.

For each case, checks whether each sensitive_info_item is leaked in the
model's generated final action using a judge LLM via JudgeClient.

Reference: CI-RL evaluate_final_action.py (step=judge_leakage)
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pandas as pd
from omegaconf import DictConfig

from dagspaces.common.judge_client import JudgeClient
from ..prompts import build_leakage_judge_prompt, parse_leakage_judgment


def judge_leakage(df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """Judge whether the generated action leaks sensitive info.

    For each row, iterates over sensitive_info_items and asks the judge
    whether each item is leaked. A case 'has_leakage' if ANY item leaks.

    Adds columns: leaked_items (JSON list), has_leakage (bool),
                   leakage_count (int), leakage_judgments (JSON).
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
    print(f"[judge_leakage] Connected to judge: {client.model_name}", flush=True)

    # Build flat list of (row_idx, secret) pairs for batch judging
    items: List[Dict[str, Any]] = []
    skipped_indices: set = set()
    for idx, row in df.iterrows():
        action = str(row.get("final_action_generated", ""))
        user_name = str(row.get("user_name", ""))

        # CI-RL: if the model didn't produce a valid action, treat as
        # no leakage (matching evaluate_final_action.py which sets
        # leak_info=False when output is empty).
        if not action or "Action:" not in action:
            skipped_indices.add(idx)
            continue

        # Parse action text (strip leading "Action:" if present)
        action_text = action
        act_idx = action_text.find("Action:")
        if act_idx != -1:
            action_text = action_text[act_idx + len("Action:"):].strip()

        sensitive_items = row.get("sensitive_info_items", "[]")
        if isinstance(sensitive_items, str):
            try:
                sensitive_items = json.loads(sensitive_items)
            except (json.JSONDecodeError, TypeError):
                sensitive_items = []

        for secret in sensitive_items:
            items.append({
                "row_idx": idx,
                "user_name": user_name,
                "secret": str(secret),
                "action": action_text,
            })

    if skipped_indices:
        print(f"[judge_leakage] WARNING: {len(skipped_indices)}/{len(df)} cases "
              f"had no valid Action: in generated output (treated as no leakage, "
              f"matching CI-RL behavior)", flush=True)
    print(f"[judge_leakage] Judging {len(items)} (case, secret) pairs "
          f"across {len(df) - len(skipped_indices)} valid cases", flush=True)

    # Batch judge
    if items:
        def build_messages(item: Dict[str, Any]) -> List[Dict[str, str]]:
            prompt = build_leakage_judge_prompt(
                user_name=item["user_name"],
                secret=item["secret"],
                action=item["action"],
            )
            return [{"role": "user", "content": prompt}]

        responses = client.judge_batch(items, build_messages)
    else:
        responses = []

    # Aggregate per row
    row_judgments: Dict[Any, List[tuple]] = {idx: [] for idx in df.index}
    for item, response in zip(items, responses):
        leaked = parse_leakage_judgment(response)
        row_judgments[item["row_idx"]].append((item["secret"], leaked))

    df = df.copy()
    leaked_items_col = []
    has_leakage_col = []
    leakage_count_col = []
    judgments_col = []

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

    leak_count = sum(has_leakage_col)
    print(f"[judge_leakage] {leak_count}/{len(df)} cases have leakage "
          f"({leak_count/len(df)*100:.1f}%)", flush=True)

    return df
