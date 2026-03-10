"""MCQ VLM inference stage: run VLM on images with MCQ prompts (Q1-Q7)."""

from __future__ import annotations

import pandas as pd
from omegaconf import DictConfig

from ..model_prompts import get_prompt_builder
from ..prompts import prepare_question_prompt
from ..vlm_inference import run_vlm_inference


def run_mcq_inference(df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """Run MCQ inference on the dataset.

    Builds the prompt from config (mode, heuristics), formats it for the
    model family, then runs VLM inference.
    """
    mode = str(getattr(cfg.prompt, "mode", "zs"))
    include_heuristics = bool(getattr(cfg.prompt, "heuristics", True))

    sys_msg, usr_prompts = prepare_question_prompt(
        mode=mode,
        is_free_form=False,
        include_heuristics=include_heuristics,
        enforce_format=True,
    )

    model_family = str(cfg.model.model_family)
    model_source = str(cfg.model.model_source)
    builder = get_prompt_builder(model_family)
    prompt_text = builder(model_source, sys_msg, usr_prompts)

    print(f"[vlm_mcq_inference] Prompt ({len(prompt_text)} chars):\n{prompt_text[:500]}...")

    result_df = run_vlm_inference(
        df=df,
        cfg=cfg,
        prompt_text=prompt_text,
        image_col="image_path",
        stage_name="vlm_mcq_inference",
    )

    return result_df
