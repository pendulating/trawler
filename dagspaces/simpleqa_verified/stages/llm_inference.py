"""LLM inference stage for SimpleQA-Verified.

Free-form answer generation via the shared vLLM helper. No guided decoding
— SimpleQA measures factual recall in natural language; constraining the
shape of the answer would change what the grader sees.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from dagspaces.common.vllm_inference import run_vllm_inference

from ..prompts import build_answer_prompt


def run_llm_inference(df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """Run vLLM inference, one user-turn per question.

    Bumps ``max_tokens`` for thinking-strip models so the answer survives
    after ``<think>`` blocks are stripped (mirrors goldcoin_hipaa's
    handling — same root cause, same fix).
    """
    sp_dict = dict(OmegaConf.to_container(cfg.sampling_params, resolve=True))

    # Reasoning models with enable_thinking=false burn tokens on hidden
    # <think> blocks before the user-visible answer. Bump max_tokens so
    # the model can finish reasoning + emit the short answer.
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
        sp_dict["max_tokens"] = max(sp_dict.get("max_tokens", 256), 4096)

    def preprocess(row: dict[str, Any]) -> dict[str, Any]:
        row["messages"] = [{"role": "user", "content": build_answer_prompt(row["question"])}]
        row["sampling_params"] = dict(sp_dict)
        return row

    def postprocess(row: dict[str, Any]) -> dict[str, Any]:
        return row

    return run_vllm_inference(
        df=df,
        cfg=cfg,
        preprocess=preprocess,
        postprocess=postprocess,
        stage_name="simpleqa_llm_inference",
    )
