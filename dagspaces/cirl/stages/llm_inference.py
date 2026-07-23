"""LLM inference stage for the CIRL-729 action benchmark.

Uses ``dagspaces/common/vllm_inference.py``. The model must generate the message
completing ``user_task`` (reasoning in ``<think>``, answer in ``<answer>``), so
this is a free-form generation stage — no guided decoding, generous max_tokens.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from dagspaces.common.vllm_inference import (
    model_needs_reasoning_budget,
    run_vllm_inference,
)

from ..prompts import build_action_prompt


def run_llm_inference(df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """Run CIRL-729 action-generation inference."""
    sp_dict = dict(OmegaConf.to_container(cfg.sampling_params, resolve=True))

    # The action task always emits reasoning + a full message; give every model
    # a generous budget (the CIRL paper uses max_tokens=2048). Reasoning models
    # spend extra tokens on hidden CoT before <answer>, so bump further.
    base = max(int(sp_dict.get("max_tokens", 2048)), 2048)
    if model_needs_reasoning_budget(cfg.model):
        base = max(base, 4096)
    sp_dict["max_tokens"] = base

    def preprocess(row: dict[str, Any]) -> dict[str, Any]:
        prompt_text = build_action_prompt(row)
        row["messages"] = [{"role": "user", "content": prompt_text}]
        row["sampling_params"] = dict(sp_dict)
        return row

    def postprocess(row: dict[str, Any]) -> dict[str, Any]:
        return row

    result_df = run_vllm_inference(
        df=df,
        cfg=cfg,
        preprocess=preprocess,
        postprocess=postprocess,
        stage_name="cirl_llm_inference",
    )

    return result_df
