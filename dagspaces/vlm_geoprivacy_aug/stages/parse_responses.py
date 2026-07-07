"""Parse VLM-generated text into per-question labels."""

from __future__ import annotations

import logging

import pandas as pd

from ..prompts import NUM_QUESTIONS, parse_answers

logger = logging.getLogger(__name__)


def parse_mcq_responses(df: pd.DataFrame) -> pd.DataFrame:
    """Parse MCQ responses into Q1_pred..Q7_pred columns.

    Adds a ``parse_status`` column distinguishing:
    - ``"empty"``: generated_text was empty/whitespace.
    - ``"unparseable"``: parser returned no answers at all.
    - ``"partial"``: parser returned fewer answers than NUM_QUESTIONS;
      the missing positions stay ``"N/A"``.
    - ``"parsed"``: full set of answers extracted.
    """
    result_df = df.copy()

    for i in range(1, NUM_QUESTIONS + 1):
        result_df[f"Q{i}_pred"] = "N/A"
    result_df["parse_status"] = "unparseable"

    parsed_count = 0
    for idx, row in result_df.iterrows():
        generated = str(row.get("generated_text", ""))
        if not generated.strip():
            result_df.at[idx, "parse_status"] = "empty"
            continue

        # parse_answers always pads to NUM_QUESTIONS with "N/A" for missing
        # questions, so count non-"N/A" entries to discriminate parsed /
        # partial / unparseable.
        answers = parse_answers(generated, free_form=False)
        for i, ans in enumerate(answers):
            result_df.at[idx, f"Q{i + 1}_pred"] = ans.strip()
        answered = sum(1 for a in answers if str(a).strip() != "N/A")
        if answered == 0:
            result_df.at[idx, "parse_status"] = "unparseable"
        elif answered < NUM_QUESTIONS:
            result_df.at[idx, "parse_status"] = "partial"
        else:
            result_df.at[idx, "parse_status"] = "parsed"
            parsed_count += 1

    logger.info(f"Parsed MCQ responses for {parsed_count}/{len(df)} rows")
    return result_df


def parse_freeform_responses(df: pd.DataFrame) -> pd.DataFrame:
    """Parse free-form responses: keep raw text as Q7_gen.

    Adds a ``parse_status`` column: ``"parsed"`` when Q7_gen is
    non-empty, ``"empty"`` when the model returned blank text.
    """
    result_df = df.copy()
    result_df["Q7_gen"] = result_df["generated_text"].fillna("").astype(str).str.strip()
    result_df["parse_status"] = (result_df["Q7_gen"] != "").map(
        lambda b: "parsed" if b else "empty"
    )

    non_empty = (result_df["Q7_gen"] != "").sum()
    logger.info(f"Parsed free-form responses: {non_empty}/{len(df)} non-empty")
    return result_df
