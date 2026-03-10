"""Parse LLM responses for GoldCoin HIPAA evaluation.

Ported from GoldCoin/eval/parse_eval_result.py with improvements:
- No random fallback; unparseable responses are marked as "unparseable".
"""

from __future__ import annotations

from typing import Optional

import pandas as pd


def clean_response(response: str) -> str:
    """Strip template phrases that confuse keyword matching."""
    replacements = [
        "determine whether the hipaa privacy rule permits or forbids the case:",
        "permit, forbid, or not applicable",
        "permitted, forbidden, or not applicable",
        "permit, forbid, not applicable",
        "permit/forbid/not applicable",
        "permitted or forbidden",
        "permits or forbids",
        "permit or forbid",
        "determination of permissibility",
        "determine whether the hipaa privacy rule is applicable to the case",
        "whether the hipaa privacy rule is applicable",
        "applicability",
    ]
    response = response.replace('"', "")
    for phrase in replacements:
        response = response.replace(phrase, "")
    return response


def extract_step_result(response: str) -> str:
    """For CoT responses, extract text after the final step marker."""
    if "step 3" in response:
        idx = response.find("step 3")
        response = response[idx:]
    elif "step 2" in response:
        idx = response.find("step 2")
        response = response[idx:]
    return response


def first_compliance_result(response: str) -> Optional[str]:
    """Find the first compliance keyword in the response."""
    permit_keywords = [
        "permit", "permis", "complies with", "not violat",
        "allow", "not explicitly prohibit", "not forbid",
    ]
    forbid_keywords = [
        "forbid", "not permit", "prohibit", "not comply",
        "not fully comply", "violat",
    ]
    all_labels = permit_keywords + forbid_keywords

    first_index = len(response)
    first_label = ""
    for label in all_labels:
        if label in response:
            idx = response.index(label)
            if idx < first_index:
                first_index = idx
                first_label = label

    if first_label in permit_keywords:
        return "Permit"
    elif first_label in forbid_keywords:
        return "Forbid"
    return None


def first_applicability_result(response: str) -> Optional[str]:
    """Find the first applicability keyword in the response."""
    applicable_keywords = ["applicable", "apply to", "applies to"]
    not_keywords = [" not "]
    all_labels = applicable_keywords + not_keywords

    first_index = len(response)
    first_label = ""
    for label in all_labels:
        if label in response:
            idx = response.index(label)
            if idx < first_index:
                first_index = idx
                first_label = label

    if first_label in applicable_keywords:
        return "Applicable"
    elif first_label in not_keywords:
        return "Not Applicable"
    return None


def parse_compliance_response(response: str) -> str:
    """Parse a compliance response into Permit/Forbid/unparseable."""
    response = response.lower()
    response = extract_step_result(response)
    response = clean_response(response)
    result = first_compliance_result(response)
    return result if result is not None else "unparseable"


def parse_applicability_response(response: str) -> str:
    """Parse an applicability response into Applicable/Not Applicable/unparseable."""
    response = response.lower()
    response = extract_step_result(response)
    response = clean_response(response)
    result = first_applicability_result(response)
    return result if result is not None else "unparseable"


def parse_responses(df: pd.DataFrame, task: str) -> pd.DataFrame:
    """Parse generated_text column and add prediction column.

    Args:
        df: DataFrame with ``generated_text`` column.
        task: "compliance" or "applicability".

    Returns:
        DataFrame with ``prediction`` column added.
    """
    df = df.copy()

    if task == "compliance":
        df["prediction"] = df["generated_text"].apply(
            lambda x: parse_compliance_response(str(x))
        )
    elif task == "applicability":
        df["prediction"] = df["generated_text"].apply(
            lambda x: parse_applicability_response(str(x))
        )
    else:
        raise ValueError(f"Unknown task: {task!r}")

    # Report parsing stats
    total = len(df)
    unparseable = (df["prediction"] == "unparseable").sum()
    print(f"[parse_responses] Task={task}: {total} responses, {unparseable} unparseable ({unparseable/total*100:.1f}%)")
    print(f"[parse_responses] Prediction distribution:")
    print(df["prediction"].value_counts().to_string())

    return df
