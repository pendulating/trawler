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

    # Build sampling params dict
    sp_dict = dict(OmegaConf.to_container(cfg.sampling_params, resolve=True))

    def preprocess(row: Dict[str, Any]) -> Dict[str, Any]:
        prompt_text = build_prompt_for_row(row, task=task, mode=mode, few_shot=few_shot)
        row["messages"] = [{"role": "user", "content": prompt_text}]
        row["sampling_params"] = sp_dict
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
