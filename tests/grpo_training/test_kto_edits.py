"""k-series edit surgery (wiki/2026-07-31_kto_plan.md §4).

The supervision-depth ladder's desirable counterfactuals are built by pure
functions over gate-parsed completions. Load-bearing invariants: inputs are
never mutated; edits round-trip through the production ``valid_gate``; the
citation replaces (not appends to) ``norms_invoked``; the rationale states
the gold conclusion and survives the refraining inversion; validation
rejects rationales that cite nothing or conclude the wrong label.
"""
from __future__ import annotations

import copy
import json

from dagspaces.grpo_training.stages.kto_edits import (
    Correction,
    apply_citation_edit,
    apply_scrutinize_edit,
    apply_verdict_edit,
    rationale_is_valid,
    render_rationale,
    serialize_completion,
)
from dagspaces.grpo_training.stages.modular_reward import valid_gate

_NORM = {
    "articulation": "A servant must never disclose the private affairs of "
                    "the household to outsiders without leave.",
    "normative_force": "prohibited",
    "act_polarity": "performing",
    "norm_subject": "a household servant",
    "norm_act": "disclose the family's private affairs to outsiders",
    "condition_of_application": "when speaking beyond the household",
    "context": "domestic service",
}


def _completion():
    return {
        "reasoning": "The passage shows two exchanges within the estate.",
        "has_information_exchange": True,
        "flows": [
            {"sender": "maid", "recipient": "grocer", "subject": "the family",
             "information_type": "the family's finances",
             "transmission_principle": "gossip", "context": "village",
             "appropriateness": "appropriate",
             "norms_invoked": ["general discretion"], "norm_source": "implicit",
             "is_new_flow": False, "confidence": 7},
            {"sender": "butler", "recipient": "master", "subject": "household",
             "information_type": "daily accounts",
             "transmission_principle": "duty", "context": "household",
             "appropriateness": "appropriate",
             "norms_invoked": [], "norm_source": "implicit",
             "is_new_flow": False, "confidence": 8},
        ],
    }


_CORR = [Correction(flow_index=0, gold="inappropriate", norm=_NORM,
                    match_sim=0.81)]


class TestVerdictEdit:
    def test_flips_only_the_corrected_flow(self):
        out = apply_verdict_edit(_completion(), _CORR)
        assert out["flows"][0]["appropriateness"] == "inappropriate"
        assert out["flows"][1]["appropriateness"] == "appropriate"

    def test_input_not_mutated(self):
        src = _completion()
        snapshot = copy.deepcopy(src)
        apply_verdict_edit(src, _CORR)
        assert src == snapshot

    def test_everything_else_byte_identical(self):
        src = _completion()
        out = apply_verdict_edit(src, _CORR)
        out["flows"][0]["appropriateness"] = src["flows"][0]["appropriateness"]
        assert out == src

    def test_bad_index_raises(self):
        import pytest
        with pytest.raises(ValueError):
            apply_verdict_edit(_completion(), [
                Correction(9, "inappropriate", _NORM, 0.8)])


class TestCitationEdit:
    def test_replaces_norms_invoked_with_gold_articulation(self):
        out = apply_citation_edit(_completion(), _CORR)
        assert out["flows"][0]["norms_invoked"] == [_NORM["articulation"]]
        assert out["flows"][0]["norm_source"] == "explicit"
        # untouched flow keeps its own citations
        assert out["flows"][1]["norms_invoked"] == []

    def test_missing_articulation_degrades_to_verdict(self):
        corr = [Correction(0, "inappropriate",
                           {**_NORM, "articulation": None}, 0.8)]
        out = apply_citation_edit(_completion(), corr)
        assert out["flows"][0]["appropriateness"] == "inappropriate"
        assert out["flows"][0]["norms_invoked"] == ["general discretion"]


class TestScrutinizeEdit:
    def test_appends_rationale_with_gold_conclusion(self):
        out = apply_scrutinize_edit(_completion(), _CORR)
        r = out["reasoning"]
        assert r.startswith("The passage shows")          # narrative kept
        assert "prohibited from" in r                     # force phrase
        assert r.rstrip(".").endswith("inappropriate")    # gold conclusion
        assert _NORM["articulation"].rstrip(".") in r     # citation in prose

    def test_refraining_norm_inverts_the_link_clause(self):
        norm = {**_NORM, "normative_force": "obligatory",
                "act_polarity": "refraining",
                "norm_act": "keep the family's affairs private"}
        corr = [Correction(0, "inappropriate", norm, 0.8)]
        text = render_rationale(0, corr[0])
        assert "refraining" in text
        assert text.rstrip(".").endswith("inappropriate")

    def test_teacher_rationales_override_template(self):
        art = _NORM["articulation"]
        teacher = (f'The norm "{art}" clearly governs: telling the grocer is '
                   "exactly the disclosure it forbids, so the flow is "
                   "inappropriate.")
        out = apply_scrutinize_edit(_completion(), _CORR, rationales=[teacher])
        assert teacher in out["reasoning"]


class TestRationaleValidation:
    def test_template_output_always_valid(self):
        assert rationale_is_valid(render_rationale(0, _CORR[0]), _CORR[0])

    def test_rejects_missing_citation(self):
        assert not rationale_is_valid(
            "This flow is inappropriate because privacy matters.", _CORR[0])

    def test_rejects_wrong_conclusion(self):
        art = _NORM["articulation"]
        assert not rationale_is_valid(
            f'The norm "{art}" applies, so the flow is appropriate.',
            _CORR[0])

    def test_gold_conclusion_after_opposite_mention_ok(self):
        art = _NORM["articulation"]
        assert rationale_is_valid(
            f'One might think this appropriate, but the norm "{art}" forbids '
            "it — the flow is inappropriate.", _CORR[0])


class TestRoundTrip:
    def test_all_three_depths_survive_the_production_gate(self):
        for edit in (apply_verdict_edit, apply_citation_edit,
                     apply_scrutinize_edit):
            text = serialize_completion(edit(_completion(), _CORR))
            g = valid_gate(text)
            assert g.passed, f"{edit.__name__} failed the gate"
            assert len(g.flows) == 2
            assert g.flows[0]["appropriateness"] == "inappropriate"
            # and the serialization is stable under re-parse
            assert json.loads(text)["flows"][1]["appropriateness"] == "appropriate"
