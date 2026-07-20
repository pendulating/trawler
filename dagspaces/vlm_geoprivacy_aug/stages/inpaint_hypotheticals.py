"""Expand the dataset with hypothetical capture-context variants.

Cross-products every dataset row with the configured hypothetical variants
(baseline control always included), so downstream inference runs each image
once per capture context and shift metrics can pair variant judgments with
the baseline on the same image.
"""

from __future__ import annotations

import json
import logging

import pandas as pd

from ..hypotheticals import HypotheticalVariant

logger = logging.getLogger(__name__)

# Columns added by expansion; prefixed hyp_ to avoid clashing with dataset columns.
HYP_COLUMNS = ["hyp_id", "hyp_dimension", "hyp_frame", "hyp_position", "hyp_ci_params"]


def expand_with_hypotheticals(
    df: pd.DataFrame,
    variants: list[HypotheticalVariant],
) -> pd.DataFrame:
    """Cross-product dataset rows with hypothetical variants.

    Returns:
        DataFrame with len(df) * len(variants) rows, original columns
        preserved, plus ``hyp_*`` columns (``hyp_ci_params`` is JSON-encoded
        for parquet friendliness). Row order: variants in declared order,
        original row order within each variant.
    """
    if not variants:
        raise ValueError("expand_with_hypotheticals requires at least one variant")

    clashes = [c for c in HYP_COLUMNS if c in df.columns]
    if clashes:
        raise ValueError(f"Dataset already has hypothetical columns: {clashes}")

    blocks: list[pd.DataFrame] = []
    for variant in variants:
        block = df.copy()
        block["hyp_id"] = variant.id
        block["hyp_dimension"] = variant.dimension
        block["hyp_frame"] = variant.frame
        block["hyp_position"] = variant.position
        block["hyp_ci_params"] = json.dumps(variant.ci_params, sort_keys=True)
        blocks.append(block)

    expanded = pd.concat(blocks, ignore_index=True)
    logger.info(
        f"Expanded {len(df)} rows x {len(variants)} hypothetical variants "
        f"({[v.id for v in variants]}) -> {len(expanded)} rows"
    )
    return expanded


# Per-row prompt columns consumed by run_vlm_inference(prompt_col=...).
PROMPT_COLUMNS = {
    "prompt": "hyp_prompt_text",
    "sys": "hyp_sys_msg",
    "usr": "hyp_usr_text",
}


def attach_variant_prompts(
    df: pd.DataFrame,
    cfg,
    *,
    is_free_form: bool,
) -> pd.DataFrame:
    """Render per-variant prompts onto a hypothetical-expanded dataset.

    Builds the prompt once per distinct variant (not per row) and maps it
    onto rows via ``hyp_id``, adding ``hyp_prompt_text`` (model-rendered,
    for vLLM), plus ``hyp_sys_msg``/``hyp_usr_text`` (raw text, for the
    transformers fallback).
    """
    from ..model_prompts import get_prompt_builder
    from ..prompts import prepare_question_prompt

    if "hyp_id" not in df.columns:
        raise ValueError("attach_variant_prompts requires a hypothetical-expanded dataset")

    mode = str(getattr(cfg.prompt, "mode", "zs"))
    include_heuristics = False if is_free_form else bool(getattr(cfg.prompt, "heuristics", True))
    model_family = str(cfg.model.model_family)
    model_source = str(cfg.model.model_source)
    builder = get_prompt_builder(model_family)

    out = df.copy()
    for col in PROMPT_COLUMNS.values():
        out[col] = ""

    variant_rows = (
        df[HYP_COLUMNS]
        .drop_duplicates(subset=["hyp_id"])
        .to_dict("records")
    )
    for entry in variant_rows:
        variant = HypotheticalVariant(
            id=str(entry["hyp_id"]),
            dimension=str(entry["hyp_dimension"]),
            frame=str(entry["hyp_frame"]),
            position=str(entry["hyp_position"]),
            ci_params=json.loads(entry["hyp_ci_params"] or "{}"),
        )
        sys_msg, usr_prompts = prepare_question_prompt(
            mode=mode,
            is_free_form=is_free_form,
            include_heuristics=include_heuristics,
            enforce_format=True,
            hypothetical=variant,
        )
        prompt_text = builder(model_source, sys_msg, usr_prompts)

        mask = out["hyp_id"] == variant.id
        out.loc[mask, PROMPT_COLUMNS["prompt"]] = prompt_text
        out.loc[mask, PROMPT_COLUMNS["sys"]] = sys_msg
        out.loc[mask, PROMPT_COLUMNS["usr"]] = "".join(usr_prompts)

    logger.info(
        f"Attached per-variant prompts for {len(variant_rows)} variants "
        f"across {len(out)} rows"
    )
    return out
