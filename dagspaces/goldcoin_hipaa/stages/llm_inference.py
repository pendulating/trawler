"""LLM inference stage for GoldCoin HIPAA evaluation.

Uses dagspaces/common/vllm_inference.py (text-only, no multimodal).
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
    """Run text-only vLLM inference on the GoldCoin dataset.

    Reads task/mode/few_shot from cfg.prompt and sampling_params from cfg.
    """
    prompt_cfg = cfg.prompt
    task = str(prompt_cfg.task)
    mode = str(prompt_cfg.mode)
    few_shot = bool(getattr(prompt_cfg, "few_shot", False))

    # Build sampling params dict.
    # Reasoning models spend tokens on hidden chain-of-thought before the
    # answer. model_needs_reasoning_budget covers BOTH triggers: an explicit
    # enable_thinking=false, AND a model that reasons structurally (a vLLM
    # reasoning parser, or harmony). The hand-rolled check this replaced only
    # had the first, so gpt-oss and openthinker3 — whose configs carry a bare
    # `chat_template_kwargs: {}` — silently kept the small budget and
    # truncated. Measured on the 2026-07-17 canonical instruct run, GoldCoin
    # compliance: gpt-oss 24/107 rows hit finish_reason=length and 17 came
    # back EMPTY; openthinker3 12/107.
    sp_dict = dict(OmegaConf.to_container(cfg.sampling_params, resolve=True))
    if model_needs_reasoning_budget(cfg.model):
        sp_dict["max_tokens"] = max(sp_dict.get("max_tokens", 1024), 4096)

    # Structured decoding: enforce JSON schema for deterministic parsing
    from dagspaces.common.eval_schemas import ApplicabilityResult, ComplianceResult
    _schema_cls = ComplianceResult if task == "compliance" else ApplicabilityResult
    _json_schema = _schema_cls.model_json_schema()
    _json_instruction = (
        '\n\nRespond with a JSON object: {"classification": "<your answer>", "reasoning": "<brief explanation>"}.'
    )

    def preprocess(row: dict[str, Any]) -> dict[str, Any]:
        prompt_text = build_prompt_for_row(row, task=task, mode=mode, few_shot=few_shot)
        prompt_text += _json_instruction
        row["messages"] = [{"role": "user", "content": prompt_text}]
        row["sampling_params"] = dict(sp_dict, guided_decoding={"json": _json_schema})
        return row

    def postprocess(row: dict[str, Any]) -> dict[str, Any]:
        # Keep generated_text as-is; parsing happens in parse_responses stage
        return row

    result_df = run_vllm_inference(
        df=df,
        cfg=cfg,
        preprocess=preprocess,
        postprocess=postprocess,
        stage_name="goldcoin_llm_inference",
    )

    return result_df
