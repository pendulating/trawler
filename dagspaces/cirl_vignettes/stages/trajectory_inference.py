"""Trajectory-level agent inference for CIRL-Vignettes.

Given the past trajectory and user instruction, the model generates the
final action (e.g., sending an email, creating a post). Uses the simplified
agent prompt from prompts.py (without the toolemu dependency).

Reference: CI-RL get_final_action.py
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from dagspaces.common.vllm_inference import run_vllm_inference
from ..prompts import build_agent_prompt, post_process_action


def run_trajectory_inference(df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """Run trajectory-level inference: model generates the final action."""
    prompt_cfg = cfg.prompt
    think = bool(getattr(prompt_cfg, "think", False))
    prompt_type = str(getattr(prompt_cfg, "trajectory_prompt_type", "naive"))

    # Trajectory inference uses higher temp and tokens than probing
    sp_dict = dict(OmegaConf.to_container(cfg.sampling_params, resolve=True))
    sp_dict["temperature"] = sp_dict.get("trajectory_temperature", 0.7)
    sp_dict["max_tokens"] = sp_dict.get("trajectory_max_tokens", 7000)
    # Remove non-vllm keys
    sp_dict.pop("trajectory_temperature", None)
    sp_dict.pop("trajectory_max_tokens", None)

    def preprocess(row: Dict[str, Any]) -> Dict[str, Any]:
        prompt_text = build_agent_prompt(row, prompt_type=prompt_type, think=think)
        row["messages"] = [{"role": "user", "content": prompt_text}]
        row["sampling_params"] = dict(sp_dict)
        return row

    def postprocess(row: Dict[str, Any]) -> Dict[str, Any]:
        raw = str(row.get("generated_text", ""))
        row["final_action_generated"] = post_process_action(raw, think=think)
        return row

    result_df = run_vllm_inference(
        df=df,
        cfg=cfg,
        preprocess=preprocess,
        postprocess=postprocess,
        stage_name="cirl_trajectory_inference",
    )

    return result_df
