"""LLM inference stage for GoldCoin HIPAA evaluation.

Uses dagspaces/common/vllm_inference.py (text-only, no multimodal).
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from dagspaces.common.vllm_inference import run_vllm_inference
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
    # Reasoning models (enable_thinking=false + think-block stripping) burn
    # tokens on <think> blocks before producing the answer.  Bump max_tokens
    # to 2048 so the model can finish reasoning and emit the classification.
    sp_dict = dict(OmegaConf.to_container(cfg.sampling_params, resolve=True))
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
        sp_dict["max_tokens"] = max(sp_dict.get("max_tokens", 1024), 4096)

    # Structured decoding: enforce JSON schema for deterministic parsing
    from dagspaces.common.eval_schemas import ApplicabilityResult, ComplianceResult
    _schema_cls = ComplianceResult if task == "compliance" else ApplicabilityResult
    _json_schema = _schema_cls.model_json_schema()
    _json_instruction = (
        '\n\nRespond with a JSON object: {"classification": "<your answer>", "reasoning": "<brief explanation>"}.'
    )

    def preprocess(row: Dict[str, Any]) -> Dict[str, Any]:
        prompt_text = build_prompt_for_row(row, task=task, mode=mode, few_shot=few_shot)
        prompt_text += _json_instruction
        row["messages"] = [{"role": "user", "content": prompt_text}]
        row["sampling_params"] = dict(sp_dict, guided_decoding={"json": _json_schema})
        return row

    def postprocess(row: Dict[str, Any]) -> Dict[str, Any]:
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
