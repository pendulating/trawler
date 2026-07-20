"""LLM inference stage for CONFAIDE evaluation.

No guided decoding — outputs vary by tier: numeric rating (Tier 2),
yes/no (Tier 3 control), free-form response (Tier 3 free), or
character listing (Tier 3 info/sharing).
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from dagspaces.common.vllm_inference import (
    model_needs_reasoning_budget,
    run_vllm_inference,
)

from ..prompts import build_prompt_for_row


def run_llm_inference(df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """Run vLLM inference on CONFAIDE data."""
    sp_dict = dict(OmegaConf.to_container(cfg.sampling_params, resolve=True))

    # Reasoning models (enable_thinking=false OR a harmony/qwen3/deepseek
    # reasoning parser, e.g. gpt-oss) spend tokens on hidden CoT before the
    # rating — give them a generous budget so the rating survives.
    if model_needs_reasoning_budget(cfg.model):
        sp_dict["max_tokens"] = max(sp_dict.get("max_tokens", 32), 4096)

    # Opt-in per-model nudge for verbose instruct models that hedge in prose and
    # truncate before stating a rating (e.g. phi-4: 100% finish_reason=length at
    # 32 tokens in the 2026-05-27 sweep). Force the rating to appear first AND
    # give it room to land. The paper's reference prompt is untouched for models
    # without this flag.
    force_answer_format = bool(
        OmegaConf.select(cfg, "model.force_answer_format") or False
    )
    if force_answer_format:
        sp_dict["max_tokens"] = max(sp_dict.get("max_tokens", 32), 256)

    def preprocess(row: dict[str, Any]) -> dict[str, Any]:
        prompt_text = build_prompt_for_row(
            row, force_answer_format=force_answer_format
        )
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
        stage_name="confaide_llm_inference",
    )

    return result_df
