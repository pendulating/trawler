"""Parity locks for the GoldCoin-HIPAA dagspace (2026-07-21 review).

Anchored to HKUST-KnowComp/GoldCoin ``eval/build_instruction_*.py``,
``eval/eval_api.py`` / ``eval_llm.py``, and ``eval/parse_eval_result.py``
(fetched 2026-07-21).
"""

from __future__ import annotations

import pandas as pd
import pytest

from dagspaces.goldcoin_hipaa.prompts import (
    ALPACA_TEMPLATE_WITH_INPUT,
    APPLICABILITY_COT_INSTRUCTION,
    COMPLIANCE_COT_INSTRUCTION,
    build_prompt_for_row,
)
from dagspaces.goldcoin_hipaa.stages.compute_metrics import compute_metrics
from dagspaces.goldcoin_hipaa.stages.parse_responses import (
    parse_applicability_response,
    parse_compliance_response,
)

# ---------------------------------------------------------------------------
# Prompt parity — verbatim reference strings from upstream
# ---------------------------------------------------------------------------

_REF_ALPACA = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes "
    "the request.\n### Instruction: {instruction}\n### Input: {input}\n### Response:"
)

_REF_COMPLIANCE_COT = (
    "Please assess the case for compliance with the HIPAA Privacy Rule through the "
    "following steps: Step 1: Annotate the eleven message characteristics [Sender, "
    "Sender Role, Recipient, Recipient Role, About, About Role, Type, Purpose, In "
    "Reply To, Consented By, Belief] about the flow of private information in the "
    "case as a list. Step 2: Identify and list all applicable HIPAA regulation IDs "
    "(e.g., 164.XXX) and their content. Step 3: Determine whether the HIPAA Privacy "
    "Rule permits or forbids the case."
)

_REF_APPLICABILITY_COT = (
    "Please assess the applicability of the HIPAA Privacy Rule to the case through "
    "the following steps: Step 1: Annotate the message characteristics [Sender, "
    "Sender Role, Recipient, Recipient Role, About, About Role, Type] about the flow "
    "of private information in the case as a list. Step 2: Determine whether the "
    "HIPAA Privacy Rule is applicable to the case."
)


def test_alpaca_template_verbatim():
    assert ALPACA_TEMPLATE_WITH_INPUT == _REF_ALPACA


def test_cot_instructions_verbatim():
    assert COMPLIANCE_COT_INSTRUCTION == _REF_COMPLIANCE_COT
    assert APPLICABILITY_COT_INSTRUCTION == _REF_APPLICABILITY_COT


def test_prompt_assembly_matches_upstream_shape():
    row = {"generate_background": "A hospital shares PHI with an insurer."}
    p = build_prompt_for_row(row, task="compliance", mode="cot")
    assert p == _REF_ALPACA.format(
        instruction=_REF_COMPLIANCE_COT,
        input="Read the case: A hospital shares PHI with an insurer.",
    )


# ---------------------------------------------------------------------------
# Parser — documented improvements over upstream's substring scan
# ---------------------------------------------------------------------------


def test_compliance_negation_fixes():
    # Upstream's substring scan matched "permis" inside "impermissible" →
    # Permit. The 2026-07-14 review fixed this deliberately; lock it.
    assert parse_compliance_response("the disclosure is impermissible") == "Forbid"
    assert parse_compliance_response("this is not allowed under HIPAA") == "Forbid"
    assert parse_compliance_response("HIPAA Privacy Rule permits the case.") == "Permit"
    assert parse_compliance_response("does not violate the Privacy Rule") == "Permit"


def test_applicability_basic():
    assert parse_applicability_response("HIPAA Privacy Rule is applicable to the case.") == "Applicable"
    assert (
        parse_applicability_response("HIPAA Privacy Rule is not applicable to the case.")
        == "Not Applicable"
    )


def test_unparseable_marked_not_guessed():
    # Deliberate deviation from upstream's random-wrong fallback: the PARSER
    # reports "unparseable"; the forced-wrong substitution happens in
    # compute_metrics where it carries provenance.
    assert parse_compliance_response("I need more information.") == "unparseable"


def test_json_guided_path_first():
    assert (
        parse_compliance_response('{"classification": "Forbid", "reasoning": "permits everything"}')
        == "Forbid"
    )
    assert (
        parse_applicability_response('{"classification": "Not Applicable", "reasoning": "applies to X"}')
        == "Not Applicable"
    )


# ---------------------------------------------------------------------------
# Metrics — upstream forced-wrong denominator (Matt-approved 2026-07-21)
# ---------------------------------------------------------------------------


def test_headline_counts_unparseable_as_wrong():
    df = pd.DataFrame(
        {
            "ground_truth": ["Permit", "Permit", "Forbid", "Forbid"],
            "prediction": ["Permit", "unparseable", "Forbid", "unparseable"],
        }
    )
    m = compute_metrics(df, "compliance")
    # Upstream: unparseable → wrong label, stays in denominator → 2/4.
    assert m["accuracy"] == pytest.approx(0.5)
    assert m["accuracy_among_parseable"] == pytest.approx(1.0)
    prov = m["metric_provenance"]["accuracy"]
    assert prov["n_total"] == 4 and prov["n_real"] == 2 and prov["n_defaulted"] == 2
    assert prov["default_reason"] == "unparseable_forced_wrong"
    # The substituted wrong labels flow into the confusion matrix too
    # (upstream parity): Permit-truth row predicted Forbid, and vice versa.
    assert m["confusion_matrix"]["Permit"]["Forbid"] == 1
    assert m["confusion_matrix"]["Forbid"]["Permit"] == 1


def test_all_unparseable_scores_zero():
    df = pd.DataFrame(
        {
            "ground_truth": ["Applicable", "Not Applicable"],
            "prediction": ["unparseable", "unparseable"],
        }
    )
    m = compute_metrics(df, "applicability")
    assert m["accuracy"] == 0.0
    assert m["parseable_rate"] == 0.0
