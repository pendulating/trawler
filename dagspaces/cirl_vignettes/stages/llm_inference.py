"""LLM inference stage for CIRL-Vignettes probing evaluation.

Uses dagspaces/common/vllm_inference.py. No guided decoding — the probing
prompt instructs the model to directly output (A) or (B).
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from dagspaces.common.vllm_inference import (
    model_needs_reasoning_budget,
    run_vllm_inference,
)
from ..prompts import build_prompt_for_row


def run_llm_inference(df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """Run probing inference on the CIRL dataset."""
    prompt_cfg = cfg.prompt
    think = bool(getattr(prompt_cfg, "think", False))

    sp_dict = dict(OmegaConf.to_container(cfg.sampling_params, resolve=True))
    # Remove trajectory-specific keys that would crash vLLM SamplingParams
    sp_dict.pop("trajectory_temperature", None)
    sp_dict.pop("trajectory_max_tokens", None)

    # Reasoning models (enable_thinking=false OR a harmony/qwen3/deepseek
    # reasoning parser, e.g. gpt-oss) spend tokens on hidden CoT before the
    # answer — give them a generous budget so the (A)/(B) choice survives.
    if model_needs_reasoning_budget(cfg.model):
        sp_dict["max_tokens"] = max(sp_dict.get("max_tokens", 16), 4096)
    elif think:
        sp_dict["max_tokens"] = max(sp_dict.get("max_tokens", 16), 512)

    def preprocess(row: Dict[str, Any]) -> Dict[str, Any]:
        prompt_text = build_prompt_for_row(row, think=think)
        row["messages"] = [{"role": "user", "content": prompt_text}]
        row["sampling_params"] = dict(sp_dict)
        return row

    def postprocess(row: Dict[str, Any]) -> Dict[str, Any]:
        return row

    result_df = run_vllm_inference(
        df=df,
        cfg=cfg,
        preprocess=preprocess,
        postprocess=postprocess,
        stage_name="cirl_vignettes_llm_inference",
    )

    return result_df
