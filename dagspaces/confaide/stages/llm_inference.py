"""LLM inference stage for CONFAIDE evaluation.

No guided decoding — outputs vary by tier: numeric rating (Tier 2),
yes/no (Tier 3 control), free-form response (Tier 3 free), or
character listing (Tier 3 info/sharing).
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from dagspaces.common.vllm_inference import run_vllm_inference
from ..prompts import build_prompt_for_row


def run_llm_inference(df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """Run vLLM inference on CONFAIDE data."""
    sp_dict = dict(OmegaConf.to_container(cfg.sampling_params, resolve=True))

    # Thinking models need more tokens
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
        sp_dict["max_tokens"] = max(sp_dict.get("max_tokens", 32), 4096)

    def preprocess(row: Dict[str, Any]) -> Dict[str, Any]:
        prompt_text = build_prompt_for_row(row)
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
        stage_name="confaide_llm_inference",
    )

    return result_df
