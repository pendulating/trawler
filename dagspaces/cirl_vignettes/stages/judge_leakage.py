"""Judge leakage stage for CIRL-Vignettes trajectory evaluation.

For each case, checks whether each sensitive_info_item is leaked in the
model's generated final action using a judge LLM via JudgeClient.

Reference: CI-RL evaluate_final_action.py (step=judge_leakage)
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

import pandas as pd
from omegaconf import DictConfig

from omegaconf import OmegaConf

from dagspaces.common.judge_client import JudgeClient
from ..prompts import build_leakage_judge_prompt, parse_leakage_judgment


def _get_batch_export_client(cfg: DictConfig) -> JudgeClient:
    """Offline JudgeClient used by every CIRL-Vignettes batch-export stage.

    Ignores every live-mode ``judge.*`` field (``base_url``, ``provider``,
    ``api_key*``) because the emitted JSONL targets OpenAI's Batch API
    regardless of how live judging is wired. Reads only:

      - ``judge.batch.target_model`` (default: ``gpt-5.2``)
      - ``judge.temperature``        (shared with live mode; default 0.0)
      - ``judge.max_tokens``         (shared with live mode; default 400)
    """
    target_model = str(
        OmegaConf.select(cfg, "judge.batch.target_model", default=None)
        or "gpt-5.2"
    )
    temperature = float(
        OmegaConf.select(cfg, "judge.temperature", default=0.0) or 0.0
    )
    max_tokens = int(
        OmegaConf.select(cfg, "judge.max_tokens", default=400) or 400
    )
    client = JudgeClient(
        base_url="https://api.openai.com/v1",
        model_name=target_model,
        temperature=temperature,
        max_tokens=max_tokens,
        provider="openai",
        offline=True,
    )
    print(
        f"[cirl_vignettes] Judge (batch-export, offline): "
        f"target_model={target_model}", flush=True,
    )
    return client


def _get_batch_export_endpoint(cfg: DictConfig) -> str:
    return str(
        OmegaConf.select(cfg, "judge.batch.target_endpoint", default=None)
        or "/v1/chat/completions"
    )


def _build_leakage_items(df: pd.DataFrame) -> Tuple[List[Dict[str, Any]], set]:
    """Flatten (row × sensitive_info_items) into a list of judge inputs.

    Returns ``(items, skipped_indices)``. Rows whose ``final_action_generated``
    is empty or lacks an ``Action:`` marker contribute no items and are
    recorded in ``skipped_indices`` (CI-RL behavior: treat as no leakage).
    """
    items: List[Dict[str, Any]] = []
    skipped: set = set()
    for idx, row in df.iterrows():
        action = str(row.get("final_action_generated", ""))
        user_name = str(row.get("user_name", ""))

        if not action or "Action:" not in action:
            skipped.add(idx)
            continue

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

        for sub_idx, secret in enumerate(sensitive_items):
            items.append({
                "row_idx": idx,
                "sub_idx": sub_idx,
                "user_name": user_name,
                "secret": str(secret),
                "action": action_text,
            })
    return items, skipped


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

    items, skipped_indices = _build_leakage_items(df)

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


def export_leakage_judge_batch(
    df: pd.DataFrame, cfg: DictConfig, output_dir: str
) -> pd.DataFrame:
    """Export leakage-judge requests as an OpenAI Batch API JSONL.

    Because CIRL-Vignettes fans each row out into N (row × sensitive_item)
    judge calls, this writes:

        - requests.jsonl      (one line per (row, secret) pair)
        - items.parquet       (row_idx, sub_idx, secret, judge_custom_id) —
                               used by the finalize script to aggregate
                               per-row leakage flags after merging
        - pending.parquet     (the original trajectory dataframe, returned
                               as the node's ``dataset`` output)
        - manifest.json
    """
    client = _get_batch_export_client(cfg)

    items, skipped_indices = _build_leakage_items(df)
    if skipped_indices:
        print(f"[judge_leakage_export] WARNING: {len(skipped_indices)}/{len(df)} "
              f"cases skipped (no valid Action:), treated as no leakage",
              flush=True)

    def custom_id_fn(item: Dict[str, Any], idx: int) -> str:
        return f"cirl_vignettes:judge_leakage:{item['row_idx']}:{item['sub_idx']}"

    def build_messages(item: Dict[str, Any]) -> List[Dict[str, str]]:
        prompt = build_leakage_judge_prompt(
            user_name=item["user_name"],
            secret=item["secret"],
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

    # Persist the items table so finalize can re-aggregate per-row.
    items_rows: List[Dict[str, Any]] = []
    for i, item in enumerate(items):
        items_rows.append({
            "judge_custom_id": custom_id_fn(item, i),
            "row_idx": item["row_idx"],
            "sub_idx": item["sub_idx"],
            "secret": item["secret"],
        })
    items_df = pd.DataFrame(items_rows)
    items_path = os.path.join(output_dir, "items.parquet")
    items_df.to_parquet(items_path, index=False)

    manifest.update({
        "dagspace": "cirl_vignettes",
        "stage": "judge_leakage",
        "items_parquet": items_path,
        "skipped_row_count": len(skipped_indices),
    })
    with open(os.path.join(output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    print(
        f"[judge_leakage_export] wrote {manifest['count']} requests to "
        f"{jsonl_path} ({len(df) - len(skipped_indices)} cases × secrets; "
        f"model={manifest['model']})",
        flush=True,
    )
    return df.copy()
