"""LLM inference stage for SimpleQA-Verified.

Free-form answer generation via the shared vLLM helper. No guided decoding
— SimpleQA measures factual recall in natural language; constraining the
shape of the answer would change what the grader sees.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from dagspaces.common.vllm_inference import (
    model_needs_reasoning_budget,
    run_vllm_inference,
)

from ..prompts import build_answer_prompt


def run_llm_inference(df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """Run vLLM inference, one user-turn per question.

    Bumps ``max_tokens`` for thinking-strip models so the answer survives
    after ``<think>`` blocks are stripped (mirrors goldcoin_hipaa's
    handling — same root cause, same fix).
    """
    sp_dict = dict(OmegaConf.to_container(cfg.sampling_params, resolve=True))

    # Reasoning models spend tokens on hidden chain-of-thought before the
    # answer. model_needs_reasoning_budget covers BOTH triggers: an explicit
    # enable_thinking=false, AND a model that reasons structurally (a vLLM
    # reasoning parser, or harmony). The hand-rolled check this replaced only
    # had the first, so gpt-oss and openthinker3 — whose configs carry a bare
    # `chat_template_kwargs: {}` — silently kept the small budget and
    # truncated. Measured on the 2026-07-17 canonical instruct run, GoldCoin
    # compliance: gpt-oss 24/107 rows hit finish_reason=length and 17 came
    # back EMPTY; openthinker3 12/107.
    if model_needs_reasoning_budget(cfg.model):
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
