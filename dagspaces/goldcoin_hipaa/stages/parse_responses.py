"""Parse LLM responses for GoldCoin HIPAA evaluation.

Ported from GoldCoin/eval/parse_eval_result.py with improvements:
- No random fallback; unparseable responses are marked as "unparseable".
- Structured JSON responses (from guided decoding) are parsed first.
"""

from __future__ import annotations

import json
from typing import Optional

import pandas as pd

from dagspaces.common.vllm_inference import _strip_think_blocks


def clean_response(response: str) -> str:
    """Strip reasoning blocks and template phrases that confuse keyword matching."""
    response = _strip_think_blocks(response)
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


def _try_json_classification(response: str) -> Optional[str]:
    """Try to extract classification from a structured JSON response."""
    text = _strip_think_blocks(response).strip()
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            obj = json.loads(text[start:end])
            if isinstance(obj, dict) and "classification" in obj:
                return obj["classification"]
    except (json.JSONDecodeError, TypeError):
        pass
    return None


def parse_compliance_response(response: str) -> str:
    """Parse a compliance response into Permit/Forbid/unparseable."""
    # Structured JSON path (from guided decoding)
    cls = _try_json_classification(response)
    if cls in ("Permit", "Forbid"):
        return cls

    # Legacy keyword-matching path
    response = response.lower()
    response = extract_step_result(response)

    # Pre-clean check: "does not directly permit or forbid" / "does not
    # directly apply" → model is saying HIPAA is silent on this, which the
    # GoldCoin benchmark treats as Permit (no prohibition).
    if any(p in response for p in [
        "does not directly permit or forbid",
        "does not directly apply",
        "does not permit or forbid",
        "neither permits nor forbids",
    ]):
        return "Permit"

    response = clean_response(response)
    result = first_compliance_result(response)
    return result if result is not None else "unparseable"


def parse_applicability_response(response: str) -> str:
    """Parse an applicability response into Applicable/Not Applicable/unparseable."""
    # Structured JSON path (from guided decoding)
    cls = _try_json_classification(response)
    if cls in ("Applicable", "Not Applicable"):
        return cls

    # Legacy keyword-matching path
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
    empty = (df["generated_text"].apply(lambda x: len(str(x).strip()) == 0)).sum()
    unparseable_rate = unparseable / total if total else 0

    print(f"[parse_responses] Task={task}: {total} responses, "
          f"{unparseable} unparseable ({unparseable/total*100:.1f}%), "
          f"{empty} empty", flush=True)
    print(f"[parse_responses] Prediction distribution:", flush=True)
    print(df["prediction"].value_counts().to_string(), flush=True)

    if unparseable_rate > 0.2:
        msg = (
            f"WARNING: {unparseable}/{total} ({unparseable_rate:.0%}) responses are unparseable "
            f"for task={task}. {empty} responses are empty strings. "
            "This usually means the model's output was entirely consumed by <think> "
            "blocks that were stripped (enable_thinking=false + low max_tokens). "
            "Consider increasing max_tokens or setting enable_thinking=true."
        )
        print(f"\n{'!'*60}", flush=True)
        print(f"  {msg}", flush=True)
        print(f"{'!'*60}\n", flush=True)
        import warnings
        warnings.warn(msg, stacklevel=2)

    return df
