"""Free-form VLM inference stage: ask VLM to describe image location.

When the incoming dataset is hypothetical-expanded (has ``hyp_id``), each
variant's capture-context frame is rendered into its own prompt and
inference runs with per-row prompts. Otherwise this behaves exactly like
the un-augmented benchmark.
"""

from __future__ import annotations

import pandas as pd
from omegaconf import DictConfig

from ..model_prompts import get_prompt_builder
from ..prompts import prepare_question_prompt
from ..vlm_inference import run_vlm_inference
from .inpaint_hypotheticals import PROMPT_COLUMNS, attach_variant_prompts


def run_freeform_inference(df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """Run free-form inference on the dataset.

    Asks the VLM to provide a location at appropriate granularity.
    """
    mode = str(getattr(cfg.prompt, "mode", "zs"))

    # Baseline (un-framed) prompt: used directly on the constant-prompt
    # path, and as the fallback/log sample in hypothetical mode.
    sys_msg, usr_prompts = prepare_question_prompt(
        mode=mode,
        is_free_form=True,
        include_heuristics=False,
        enforce_format=False,
    )

    model_family = str(cfg.model.model_family)
    model_source = str(cfg.model.model_source)
    builder = get_prompt_builder(model_family)
    prompt_text = builder(model_source, sys_msg, usr_prompts)
    usr_text = "".join(usr_prompts)

    prompt_kwargs = {}
    if "hyp_id" in df.columns:
        df = attach_variant_prompts(df, cfg, is_free_form=True)
        prompt_kwargs = {
            "prompt_col": PROMPT_COLUMNS["prompt"],
            "sys_msg_col": PROMPT_COLUMNS["sys"],
            "usr_text_col": PROMPT_COLUMNS["usr"],
        }
        n_variants = df["hyp_id"].nunique()
        print(f"[vlm_freeform_inference] Hypothetical mode: {n_variants} variants, {len(df)} rows")

    print(f"[vlm_freeform_inference] Prompt ({len(prompt_text)} chars):\n{prompt_text[:500]}...")

    result_df = run_vlm_inference(
        df=df,
        cfg=cfg,
        prompt_text=prompt_text,
        image_col="image_path",
        stage_name="vlm_freeform_inference",
        sys_msg=sys_msg,
        usr_text=usr_text,
        **prompt_kwargs,
    )

    return result_df
