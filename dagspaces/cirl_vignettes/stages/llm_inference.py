"""LLM inference stage for CIRL-Vignettes probing evaluation.

Uses dagspaces/common/vllm_inference.py. No guided decoding — the probing
prompt instructs the model to directly output (A) or (B).
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from dagspaces.common.vllm_inference import run_vllm_inference
from ..prompts import build_prompt_for_row


def run_llm_inference(df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """Run probing inference on the CIRL dataset."""
    prompt_cfg = cfg.prompt
    think = bool(getattr(prompt_cfg, "think", False))

    sp_dict = dict(OmegaConf.to_container(cfg.sampling_params, resolve=True))
    # Remove trajectory-specific keys that would crash vLLM SamplingParams
    sp_dict.pop("trajectory_temperature", None)
    sp_dict.pop("trajectory_max_tokens", None)

    # Thinking models need more tokens for reasoning before the answer
    _strips_thinking = False
    try:
        ctk = getattr(cfg.model, "chat_template_kwargs", None) or {}
        if hasattr(ctk, "enable_thinking"):
            _strips_thinking = not bool(ctk.enable_thinking)
        elif isinstance(ctk, dict):
            _strips_thinking = not bool(ctk.get("enable_thinking", True))
    except Exception:
        pass
    if _strips_thinking:
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
