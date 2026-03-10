"""Parse VLM-generated text into per-question labels."""

from __future__ import annotations

import logging

import pandas as pd

from ..prompts import NUM_QUESTIONS, parse_answers

logger = logging.getLogger(__name__)


def parse_mcq_responses(df: pd.DataFrame) -> pd.DataFrame:
    """Parse MCQ responses into Q1_pred..Q7_pred columns."""
    result_df = df.copy()

    for i in range(1, NUM_QUESTIONS + 1):
        result_df[f"Q{i}_pred"] = "N/A"

    parsed_count = 0
    for idx, row in result_df.iterrows():
        generated = str(row.get("generated_text", ""))
        if not generated:
            continue

        answers = parse_answers(generated, free_form=False)
        for i, ans in enumerate(answers):
            result_df.at[idx, f"Q{i + 1}_pred"] = ans.strip()
        parsed_count += 1

    logger.info(f"Parsed MCQ responses for {parsed_count}/{len(df)} rows")
    return result_df


def parse_freeform_responses(df: pd.DataFrame) -> pd.DataFrame:
    """Parse free-form responses: keep raw text as Q7_gen."""
    result_df = df.copy()
    result_df["Q7_gen"] = result_df["generated_text"].fillna("").astype(str).str.strip()

    non_empty = (result_df["Q7_gen"] != "").sum()
    logger.info(f"Parsed free-form responses: {non_empty}/{len(df)} non-empty")
    return result_df
