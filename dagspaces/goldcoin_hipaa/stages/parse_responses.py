"""Parse LLM responses for GoldCoin HIPAA evaluation.

Ported from GoldCoin/eval/parse_eval_result.py with improvements:
- No random fallback; unparseable responses are marked as "unparseable".
- Structured JSON responses (from guided decoding) are parsed first.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

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


# Word-boundary keyword matching. Substring matching mis-classified negated
# verdicts: "permis" matched inside "impermissible" (→ Permit, should be Forbid)
# and "allow" inside "not allowed" (→ Permit, should be Forbid). Patterns are
# anchored with ``\b`` so a positive keyword cannot match inside a negating word,
# and negated/forbidding phrases are listed first so that on an index tie they
# beat the bare positive keyword they contain. Earliest match position wins;
# "not X" phrases start before the bare "X" they contain, so double-negations
# ("not violate", "not forbid" → Permit) resolve correctly by position.
_COMPLIANCE_PATTERNS: List[Tuple["re.Pattern[str]", str]] = [
    (re.compile(p), label) for p, label in [
        # Forbid (including negated-permit phrases)
        (r"impermissib", "Forbid"),
        (r"\bnot\s+permit", "Forbid"),
        (r"\bnot\s+allow", "Forbid"),
        (r"\bnot\s+(?:fully\s+)?compl", "Forbid"),
        (r"\bprohibit", "Forbid"),
        (r"\bforbid", "Forbid"),
        (r"\bviolat", "Forbid"),
        # Permit (including double-negations that mean "complies")
        (r"\bnot\s+violat", "Permit"),
        (r"\bnot\s+explicitly\s+prohibit", "Permit"),
        (r"\bnot\s+forbid", "Permit"),
        (r"\bcompl(?:y|ies|ies\s+with|iant)", "Permit"),
        (r"\bpermit", "Permit"),
        (r"\bpermis", "Permit"),
        (r"\ballow", "Permit"),
    ]
]

_APPLICABILITY_PATTERNS: List[Tuple["re.Pattern[str]", str]] = [
    (re.compile(p), label) for p, label in [
        # Not Applicable (negations first — a leading "Not applicable" has no
        # space-padded " not " and must still be caught)
        (r"\bnot\s+applicable", "Not Applicable"),
        (r"\binapplicable", "Not Applicable"),
        (r"\bdoes\s+not\s+apply", "Not Applicable"),
        (r"\bdo\s+not\s+apply", "Not Applicable"),
        (r"\bdoesn'?t\s+apply", "Not Applicable"),
        (r"\bnot\s+apply", "Not Applicable"),
        (r"\bn/a\b", "Not Applicable"),
        # Applicable
        (r"\bapplicable", "Applicable"),
        (r"\bapplies\s+to", "Applicable"),
        (r"\bapply\s+to", "Applicable"),
    ]
]


def _earliest_regex_label(
    response: str, patterns: List[Tuple["re.Pattern[str]", str]]
) -> Optional[str]:
    """Return the label whose pattern matches earliest; ties favour list order."""
    best_index = len(response)
    best_label: Optional[str] = None
    for pat, label in patterns:
        m = pat.search(response)
        if m is not None and m.start() < best_index:
            best_index = m.start()
            best_label = label
    return best_label


def first_compliance_result(response: str) -> Optional[str]:
    """Find the first compliance keyword in the response."""
    return _earliest_regex_label(response, _COMPLIANCE_PATTERNS)


def first_applicability_result(response: str) -> Optional[str]:
    """Find the first applicability keyword in the response."""
    return _earliest_regex_label(response, _APPLICABILITY_PATTERNS)


def _try_json_classification(response: str) -> Optional[str]:
    """Try to extract classification from a structured JSON response."""
    from dagspaces.common.json_extraction import extract_json_from_text

    text = _strip_think_blocks(response).strip()
    obj, _ = extract_json_from_text(text)
    if isinstance(obj, dict) and "classification" in obj:
        return obj["classification"]
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
    """Parse generated_text column and add prediction + parse_status columns.

    Args:
        df: DataFrame with ``generated_text`` column.
        task: "compliance" or "applicability".

    Returns:
        DataFrame with ``prediction`` and ``parse_status`` columns added.
        ``parse_status`` ∈ {empty, unparseable, parsed} for eval_sanity.
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

    df["parse_status"] = df.apply(
        lambda r: (
            "empty" if not str(r["generated_text"]).strip()
            else ("unparseable" if r["prediction"] == "unparseable" else "parsed")
        ),
        axis=1,
    )

    # Report parsing stats
    total = len(df)
    unparseable = (df["prediction"] == "unparseable").sum()
    empty = (df["generated_text"].apply(lambda x: len(str(x).strip()) == 0)).sum()
    unparseable_rate = unparseable / total if total else 0

    print(f"[parse_responses] Task={task}: {total} responses, "
          f"{unparseable} unparseable ({unparseable/total*100:.1f}%), "
          f"{empty} empty", flush=True)
    print("[parse_responses] Prediction distribution:", flush=True)
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
