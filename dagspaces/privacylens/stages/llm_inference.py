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
from ..prompts import build_qa_prompt, build_action_prompt, build_leakage_judge_prompt


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
    sp_qa = dict(sp_base, max_tokens=512)

    def preprocess(row: Dict[str, Any]) -> Dict[str, Any]:
        axis = row["_qa_axis"]
        prompt = build_qa_prompt(row, axis)
        row["messages"] = [
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user"]},
        ]
        row["sampling_params"] = sp_qa
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
# 3. Leakage Judgment
# ---------------------------------------------------------------------------

def run_leakage_judge_inference(df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """Run leakage judgment inference.

    For each row (which must have 'generated_action'), judge whether
    the action leaks the sensitive seed information.
    """
    sp_base = dict(OmegaConf.to_container(cfg.sampling_params, resolve=True))
    sp_judge = dict(sp_base, max_tokens=4096)

    def preprocess(row: Dict[str, Any]) -> Dict[str, Any]:
        action = str(row.get("generated_action", ""))
        prompt = build_leakage_judge_prompt(row, action)
        row["messages"] = [
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user"]},
        ]
        row["sampling_params"] = sp_judge
        return row

    def postprocess(row: Dict[str, Any]) -> Dict[str, Any]:
        row["leak_judge_text"] = row.get("generated_text", "")
        return row

    result_df = run_vllm_inference(
        df=df,
        cfg=cfg,
        preprocess=preprocess,
        postprocess=postprocess,
        stage_name="leakage_judge_inference",
    )

    return result_df
