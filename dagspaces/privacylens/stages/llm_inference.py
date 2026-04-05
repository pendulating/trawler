"""LLM inference stages for PrivacyLens evaluation.

Three inference passes using dagspaces/common/vllm_inference.py:
1. QA probing (3 axes per row → expanded DataFrame)
2. Agent action generation
3. Leakage judgment
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from dagspaces.common.vllm_inference import run_vllm_inference
from ..prompts import (
    build_qa_prompt, build_action_prompt,
    build_leakage_judge_prompt, build_helpfulness_judge_prompt,
)


# ---------------------------------------------------------------------------
# 1. QA Probing
# ---------------------------------------------------------------------------

def run_qa_probe_inference(df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """Run QA probing inference across S, V, T axes.

    Expands each row into 3 (one per axis), runs inference, then returns
    the expanded DataFrame with generated responses.
    """
    axes = ["S", "V", "T"]

    # Expand: one row per (original_row, axis)
    expanded_rows = []
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        for axis in axes:
            expanded = dict(row_dict)
            expanded["_qa_axis"] = axis
            expanded_rows.append(expanded)

    df_expanded = pd.DataFrame(expanded_rows)
    print(
        f"[qa_probe_inference] Expanded {len(df)} rows × {len(axes)} axes = {len(df_expanded)} prompts",
        flush=True,
    )

    sp_base = dict(OmegaConf.to_container(cfg.sampling_params, resolve=True))

    # Reasoning models (enable_thinking=false + think-block stripping) burn
    # tokens on <think> blocks before producing the answer.  512 is too low —
    # the model hits the limit mid-reasoning and the stripped output is empty.
    _strips_thinking = False
    try:
        ctk = getattr(cfg.model, "chat_template_kwargs", None) or {}
        if hasattr(ctk, "enable_thinking"):
            _strips_thinking = not bool(ctk.enable_thinking)
        elif isinstance(ctk, dict):
            _strips_thinking = not bool(ctk.get("enable_thinking", True))
    except Exception:
        pass
    qa_max_tokens = 4096 if _strips_thinking else sp_base.get("max_tokens", 2048)
    sp_qa = dict(sp_base, max_tokens=qa_max_tokens)

    # Structured decoding for QA probing
    from dagspaces.common.eval_schemas import YesNoResult
    _qa_schema = YesNoResult.model_json_schema()

    def preprocess(row: Dict[str, Any]) -> Dict[str, Any]:
        axis = row["_qa_axis"]
        prompt = build_qa_prompt(row, axis)
        row["messages"] = [
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user"]},
        ]
        row["sampling_params"] = dict(sp_qa, guided_decoding={"json": _qa_schema})
        return row

    def postprocess(row: Dict[str, Any]) -> Dict[str, Any]:
        return row

    result_df = run_vllm_inference(
        df=df_expanded,
        cfg=cfg,
        preprocess=preprocess,
        postprocess=postprocess,
        stage_name="qa_probe_inference",
    )

    return result_df


# ---------------------------------------------------------------------------
# 2. Agent Action Generation
# ---------------------------------------------------------------------------

def run_action_inference(df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """Run agent action generation inference.

    For each row, generates the final action from the trajectory.
    """
    sp_base = dict(OmegaConf.to_container(cfg.sampling_params, resolve=True))
    sp_action = dict(sp_base, max_tokens=4096)

    def preprocess(row: Dict[str, Any]) -> Dict[str, Any]:
        prompt = build_action_prompt(row)
        row["messages"] = [
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user"]},
        ]
        row["sampling_params"] = sp_action
        return row

    def postprocess(row: Dict[str, Any]) -> Dict[str, Any]:
        row["generated_action"] = row.get("generated_text", "")
        return row

    result_df = run_vllm_inference(
        df=df,
        cfg=cfg,
        preprocess=preprocess,
        postprocess=postprocess,
        stage_name="action_inference",
    )

    return result_df


# ---------------------------------------------------------------------------
# 3. Leakage Judgment (via external judge server)
# ---------------------------------------------------------------------------

def _get_judge_client(cfg: DictConfig):
    """Build a JudgeClient from config.

    Reads ``cfg.judge.*`` when present (preferred, supports commercial APIs);
    falls back to the legacy ``cfg.judge_server_url`` / ``JUDGE_SERVER_URL``
    env var for vLLM-only setups.
    """
    from dagspaces.common.judge_client import JudgeClient

    judge_cfg = OmegaConf.select(cfg, "judge", default=None)
    if judge_cfg is not None:
        url = str(getattr(judge_cfg, "base_url", "") or "")
    else:
        url = ""
    if not url:
        url = str(OmegaConf.select(cfg, "judge_server_url", default="") or "")
    if not url:
        import os
        url = os.environ.get("JUDGE_SERVER_URL", "")
    if not url:
        raise RuntimeError(
            "No judge endpoint configured. Set judge.base_url or JUDGE_SERVER_URL. "
            "Launch a vLLM server with: sbatch scripts/judge_server.sub, or point at "
            "a commercial API (e.g. judge.base_url=https://api.openai.com/v1 "
            "judge.model_name=gpt-4o judge.api_key_env=OPENAI_API_KEY)."
        )

    if judge_cfg is not None:
        client = JudgeClient(
            base_url=url,
            model_name=str(getattr(judge_cfg, "model_name", "default") or "default"),
            max_workers=int(getattr(judge_cfg, "max_workers", 8) or 8),
            temperature=float(getattr(judge_cfg, "temperature", 0.0) or 0.0),
            max_tokens=int(getattr(judge_cfg, "max_tokens", 1024) or 1024),
            provider=(getattr(judge_cfg, "provider", None) or None),
            api_key=(getattr(judge_cfg, "api_key", None) or None),
            api_key_env=(getattr(judge_cfg, "api_key_env", None) or None),
        )
    else:
        client = JudgeClient(base_url=url, max_tokens=1024)
    if not client.health_check():
        raise RuntimeError(f"Judge endpoint not reachable at {url}")
    print(f"[privacylens] Judge OK: {url} (provider={client.provider}, model={client.model_name})",
          flush=True)
    return client


def run_leakage_judge_inference(df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """Run leakage judgment via the external judge server.

    For each row (which must have 'generated_action'), judge whether
    the action leaks the sensitive seed information.
    """
    from dagspaces.common.eval_schemas import YesNoResult

    client = _get_judge_client(cfg)
    schema = YesNoResult.model_json_schema()

    items = df.to_dict("records")

    def build_messages(item: Dict[str, Any]):
        action = str(item.get("generated_action", ""))
        prompt = build_leakage_judge_prompt(item, action)
        return [
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user"]},
        ]

    print(f"[leakage_judge] Judging {len(items)} actions via judge server...",
          flush=True)
    responses = client.judge_batch(items, build_messages, json_schema=schema)

    result_df = df.copy()
    result_df["leak_judge_text"] = responses
    return result_df


# ---------------------------------------------------------------------------
# 4. Helpfulness Judgment (via external judge server)
# ---------------------------------------------------------------------------

def run_helpfulness_judge_inference(df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """Run helpfulness judgment via the external judge server.

    For each row (which must have 'generated_action'), evaluate whether
    the action helpfully accomplishes the user's instruction (0-3 scale).
    """
    from dagspaces.common.eval_schemas import HelpfulnessResult

    client = _get_judge_client(cfg)
    schema = HelpfulnessResult.model_json_schema()

    items = df.to_dict("records")

    def build_messages(item: Dict[str, Any]):
        action = str(item.get("generated_action", ""))
        prompt = build_helpfulness_judge_prompt(item, action)
        return [
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user"]},
        ]

    print(f"[helpfulness_judge] Judging {len(items)} actions via judge server...",
          flush=True)
    responses = client.judge_batch(items, build_messages, json_schema=schema)

    result_df = df.copy()
    result_df["helpfulness_judge_text"] = responses
    return result_df
