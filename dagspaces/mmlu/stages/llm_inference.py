"""LLM inference stage for MMLU.

Uses :func:`dagspaces.common.vllm_inference.run_vllm_inference` with
guided JSON decoding enforcing the ``MMLUAnswer`` schema, so each
response is structurally ``{"answer": "<letter>"}``. parse_responses
then extracts the letter deterministically.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from dagspaces.common.vllm_inference import (
    model_needs_reasoning_budget,
    run_vllm_inference,
)

from ..prompts import MMLU_LETTER_SCHEMA, build_mmlu_prompt


def run_llm_inference(df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """Run guided-MCQ vLLM inference, one row per question."""
    prompt_cfg = getattr(cfg, "prompt", None)
    instruction_json = bool(
        getattr(prompt_cfg, "instruction_response_json", True)
        if prompt_cfg is not None else True
    )

    sp_dict = dict(OmegaConf.to_container(cfg.sampling_params, resolve=True))

    # Reasoning models burn tokens on hidden CoT (stripped <think> blocks, or
    # an always-on harmony/qwen3/deepseek reasoning channel like gpt-oss) before
    # emitting the JSON answer. Bump max_tokens so they can finish reasoning AND
    # emit the answer letter.
    if model_needs_reasoning_budget(cfg.model):
        sp_dict["max_tokens"] = max(sp_dict.get("max_tokens", 64), 4096)

    def preprocess(row: dict[str, Any]) -> dict[str, Any]:
        # Parquet → pandas → preprocess() converts list-typed columns
        # into numpy object arrays. `arr or None` / `arr or []` would
        # raise "truth value of an empty array is ambiguous" on every
        # row; coerce to a plain Python list at this boundary so the
        # downstream prompt builder can use truthy idioms safely.
        fs = row.get("few_shot_examples")
        fs_list = list(fs) if fs is not None else []
        choices_list = list(row["choices"])
        prompt = build_mmlu_prompt(
            question=row["question"],
            choices=choices_list,
            subject=row["subject"],
            few_shot_examples=fs_list,
            instruction_response_json=instruction_json,
        )
        row["messages"] = [{"role": "user", "content": prompt}]
        row["sampling_params"] = dict(sp_dict, guided_decoding={"json": MMLU_LETTER_SCHEMA})
        return row

    def postprocess(row: dict[str, Any]) -> dict[str, Any]:
        return row

    return run_vllm_inference(
        df=df,
        cfg=cfg,
        preprocess=preprocess,
        postprocess=postprocess,
        stage_name="mmlu_llm_inference",
    )
