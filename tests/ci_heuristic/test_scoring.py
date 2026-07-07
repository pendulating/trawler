"""Tests for the Phase 4 scoring machinery: matchers, extraction, prima
facie, misapplication probes (a)-(f), consistency, coverage,
contextualization, and the end-to-end score_traversals pass."""

from __future__ import annotations

import json

import pandas as pd

from dagspaces.ci_heuristic.scoring.consistency import check_entailment, flip_rate
from dagspaces.ci_heuristic.scoring.contextualization import (
    aggregate,
    build_judge_prompts,
    entry_score,
    parse_judge_reply,
)
from dagspaces.ci_heuristic.scoring.coverage import score_factor_coverage, viewpoint_diversity
from dagspaces.ci_heuristic.scoring.extraction import score_vs_tier_a, score_vs_tier_b
from dagspaces.ci_heuristic.scoring.matchers import alias_match, set_prf
from dagspaces.ci_heuristic.scoring.prima_facie import score_prima_facie
from dagspaces.ci_heuristic.scoring.probes import (
    probe_a_context_place,
    probe_b_change_as_violation,
    probe_c_purpose_in_tp,
    probe_d_premature_evaluation,
    probe_e_violation_auto_reject,
    probe_f_incompleteness_blindness,
    sentiment_leakage,
)
from dagspaces.ci_heuristic.stages.score_traversal import score_traversals


def _clean_state(**overrides):
    """A well-formed traversal state; override steps per test."""
    state = {
        "s1": {"flows": [{"sender": "the patient", "recipient": "their doctor",
                           "subject": "the patient", "information_type": "symptoms", "medium": "", "novelty": ""}]},
        "s2": {"domain": "health care", "nested_contexts": [], "activities": [],
                "purposes": ["treatment"], "values_ends": ["managing illness"]},
        "s3": {"senders": ["the patient"], "recipients": ["their doctor"],
                "subjects": ["the patient"], "nonhuman_roles": []},
        "s4": {"transmission_principles": [{"principle": "consent", "explicit": False, "evidence": ""}]},
        "s5": {"norms": [{"norm_flow": "patient tells doctor", "entrenchment_evidence": "settled",
                           "departures": ["new recipient"], "completeness": "entrenched"}]},
        "s6": {"violation": "yes", "departed_parameters": ["recipient"], "justification": "j"},
        "s7": {"factors": [{"factor": "coerced disclosure", "kind": "autonomy",
                             "affected_parties": ["patients"], "direction": "harm"}]},
        "s8": {"meanings": [{"factor_ref": 0, "contextual_end": "managing illness",
                              "advances_or_undermines": "undermines", "argument": "a"}]},
        "s9": {"decision": "modify", "conditions": ["voluntary only"], "carrying_findings": ["s7 harm"]},
    }
    state.update(overrides)
    return state


class TestMatchers:
    def test_alias_synonyms_and_containment(self):
        assert alias_match("their doctor", "the patient's physician")
        assert alias_match("consent", "the subject's permission")
        assert not alias_match("their doctor", "an advertising network")

    def test_set_prf_greedy(self):
        r = set_prf(["the patient", "their doctor"], ["a physician", "the patient", "a stranger"])
        assert r["n_matched"] == 2
        assert r["recall"] == 1.0
        assert 0 < r["precision"] < 1


class TestExtraction:
    def test_tier_b_hits(self):
        gold_values = str({"sender": "the patient", "recipient": "the marketing department of an insurance company",
                            "subject": "the patient", "information_type": "symptoms",
                            "transmission_principle": "consent", "_norm": {}})
        state = _clean_state(s3={"senders": ["the patient"],
                                   "recipients": ["an insurance company's marketing arm"],
                                   "subjects": ["the patient"], "nonhuman_roles": []})
        res = score_vs_tier_b(gold_values, state)
        assert res["sender"]["hit"] == 1
        assert res["recipient"]["hit"] == 1

    def test_tier_a_prf_keys(self):
        gold = json.load(open("dagspaces/ci_heuristic/corpus/tier_a/kumar2024_fitbit.json"))
        res = score_vs_tier_a(gold, _clean_state())
        assert set(res) == {"senders", "recipients", "subjects", "information_types", "transmission_principles"}
        assert 0 <= res["transmission_principles"]["recall"] <= 1


class TestProbes:
    def test_a_flags_platform_and_place_not_domain(self):
        assert probe_a_context_place(_clean_state()) is False
        assert probe_a_context_place(_clean_state(s2={"domain": "Facebook"})) is True
        assert probe_a_context_place(_clean_state(s2={"domain": "the hospital"})) is True
        assert probe_a_context_place(_clean_state(s2={"domain": ""})) is None

    def test_b_violation_without_entrenchment(self):
        assert probe_b_change_as_violation(_clean_state()) is False
        assert probe_b_change_as_violation(_clean_state(s5={"norms": []})) is True
        no_dep = {"norms": [{"norm_flow": "n", "departures": [], "completeness": "entrenched"}]}
        assert probe_b_change_as_violation(_clean_state(s5=no_dep)) is True
        incomplete = {"norms": [{"norm_flow": "n", "departures": ["d"], "completeness": "incomplete"}]}
        assert probe_b_change_as_violation(_clean_state(s5=incomplete)) is True

    def test_c_purpose_language_in_tp(self):
        assert probe_c_purpose_in_tp(_clean_state()) is False
        bad = {"transmission_principles": [{"principle": "in order to improve engagement", "explicit": False}]}
        assert probe_c_purpose_in_tp(_clean_state(s4=bad)) is True
        # negation-style TP constraints must NOT trip the detector
        ok = {"transmission_principles": [{"principle": "not to be shared beyond the care team", "explicit": False}]}
        assert probe_c_purpose_in_tp(_clean_state(s4=ok)) is False

    def test_d_sentiment_leak_and_degenerate_tail(self):
        assert probe_d_premature_evaluation(_clean_state()) is False
        leaky = {"flows": [{"sender": "x", "recipient": "y", "subject": "z",
                             "information_type": "a concerning violation of trust", "medium": "", "novelty": ""}]}
        assert probe_d_premature_evaluation(_clean_state(s1=leaky)) is True
        assert probe_d_premature_evaluation(_clean_state(s7={"factors": []})) is True

    def test_d_s5_departure_language_not_flagged(self):
        # s5 legitimately uses change-language; excluded from the leak scan
        state = _clean_state()
        assert sentiment_leakage(state)["leaked"] is False

    def test_e_auto_reject(self):
        assert probe_e_violation_auto_reject(_clean_state()) is False  # modify
        rejecting = _clean_state(
            s9={"decision": "reject", "conditions": [], "carrying_findings": ["x"]},
            s7={"factors": [{"factor": "f", "kind": "trust", "affected_parties": [], "direction": "benefit"}]},
            s8={"meanings": [{"factor_ref": 0, "contextual_end": "e", "advances_or_undermines": "advances", "argument": "a"}]},
        )
        assert probe_e_violation_auto_reject(rejecting) is True
        justified = _clean_state(s9={"decision": "reject", "conditions": [], "carrying_findings": ["x"]})
        assert probe_e_violation_auto_reject(justified) is False

    def test_f_incompleteness_blindness(self):
        forced = _clean_state()  # says yes
        assert probe_f_incompleteness_blindness(forced, "incomplete_norms") is True
        recognizing = _clean_state(s6={"violation": "incomplete_norms", "departed_parameters": [], "justification": "j"})
        assert probe_f_incompleteness_blindness(recognizing, "incomplete_norms") is False
        assert probe_f_incompleteness_blindness(forced, "yes") is None


class TestPrimaFacie:
    def test_scorer_on_mixed_cases(self):
        yes_state = _clean_state()
        no_state = _clean_state(s6={"violation": "no", "departed_parameters": [], "justification": "j"})
        inc_state = _clean_state(s6={"violation": "incomplete_norms", "departed_parameters": [], "justification": "j"})
        cases = [
            {"gold_prima_facie": "yes", "gold_departed_parameter": "recipient", "state": yes_state},
            {"gold_prima_facie": "yes", "gold_departed_parameter": "sender", "state": no_state},  # miss
            {"gold_prima_facie": "no", "gold_departed_parameter": "none", "state": no_state},
            {"gold_prima_facie": "incomplete_norms", "gold_departed_parameter": "none", "state": inc_state},
        ]
        m = score_prima_facie(cases)
        assert m["violation_sensitivity"] == 0.5
        assert m["violation_specificity"] == 1.0
        assert m["incompleteness_recognition_rate"] == 1.0
        assert m["attribution_accuracy"] == 1.0  # the one flagged case named 'recipient'


class TestConsistency:
    def test_entailment_rules(self):
        assert check_entailment(_clean_state())["consistent"] is True
        bad = _clean_state(s9={"decision": "modify", "conditions": [], "carrying_findings": ["x"]})
        assert "modify_without_conditions" in check_entailment(bad)["violations"]
        unassessable = _clean_state(s9={"parse_error": "x"})
        assert check_entailment(unassessable)["assessable"] is False

    def test_flip_rate(self):
        a = [_clean_state(), _clean_state()]
        b = [_clean_state(), _clean_state(s9={"decision": "reject", "conditions": [], "carrying_findings": ["x"]})]
        assert flip_rate(a, b)["flip_rate"] == 0.5


class TestCoverageAndContextualization:
    def test_factor_coverage(self):
        gold = [{"factor": "coerced disclosure via premiums", "kind": "coercion", "affected_parties": ["policyholders"]},
                 {"factor": "power imbalance", "kind": "power", "affected_parties": ["insurers"]}]
        pred = [{"factor": "people are coerced into disclosing", "kind": "coercion", "affected_parties": ["policyholders"]}]
        cov = score_factor_coverage(gold, pred)
        assert cov["factor_recall"] == 0.5
        assert cov["kind_recall"] == 0.5

    def test_viewpoint_diversity_bounds(self):
        clones = viewpoint_diversity(["the same words here", "the same words here"])
        distinct = viewpoint_diversity(["autonomy of patients", "profits of insurers"])
        assert clones["mean_pairwise_distance"] == 0.0
        assert distinct["mean_pairwise_distance"] == 1.0

    def test_judge_prompts_and_scoring(self):
        prompts = build_judge_prompts(_clean_state())
        assert len(prompts) == 1
        assert "managing illness" in prompts[0][1]
        good = {"specific_end": True, "argued_relative": True, "transplant_survives": False}
        generic = {"specific_end": True, "argued_relative": True, "transplant_survives": True}
        assert entry_score(good) == 1 and entry_score(generic) == 0
        agg = aggregate([good, generic])
        assert agg["contextualization_score"] == 0.5
        assert agg["generic_rate"] == 0.5
        assert parse_judge_reply('junk {"specific_end": true, "argued_relative": false, "transplant_survives": false} ok')
        assert parse_judge_reply("no json") is None


class TestEndToEnd:
    def test_score_traversals_over_synthetic_run(self):
        state = _clean_state()
        traverse_rows = [
            {"case_id": "b_case", "tier": "b", "ladder_level": "l3", "step": s,
             "prompt_sys": "", "prompt_usr": "", "generated_text": "",
             "artifact_json": json.dumps(state[s]), "parse_status": "parsed"}
            for s in state
        ]
        traverse_df = pd.DataFrame(traverse_rows)
        cases_df = pd.DataFrame([{
            "case_id": "b_case", "tier": "b", "practice_input": "p", "contaminated": False,
            "gold_path": "", "prima_facie": "yes", "departed_parameter": "recipient",
            "gold_values": str({"sender": "the patient", "recipient": "their doctor",
                                 "subject": "the patient", "information_type": "symptoms",
                                 "transmission_principle": "consent", "_norm": {}}),
        }])
        metrics, per_case = score_traversals(traverse_df, cases_df)

        assert metrics["n_cases"] == 1
        assert metrics["prima_facie"]["violation_sensitivity"] == 1.0
        assert metrics["probes"]["a_context_place"]["rate"] == 0.0
        assert per_case.iloc[0]["entailment_consistent"]
        assert per_case.iloc[0]["extract_hit_sender"] == 1

    def test_l1_monolithic_reconstruction(self):
        state = _clean_state()
        l1_artifact = {
            "s1_flows": state["s1"], "s2_context": state["s2"], "s3_actors": state["s3"],
            "s4_transmission_principles": state["s4"], "s5_norms": state["s5"],
            "s6_prima_facie": state["s6"], "s7_factors": state["s7"],
            "s8_contextual_meaning": state["s8"], "s9_recommendation": state["s9"],
        }
        traverse_df = pd.DataFrame([{
            "case_id": "c1", "tier": "c", "ladder_level": "l1", "step": "monolithic",
            "prompt_sys": "", "prompt_usr": "", "generated_text": "",
            "artifact_json": json.dumps(l1_artifact), "parse_status": "parsed",
        }])
        cases_df = pd.DataFrame([{"case_id": "c1", "tier": "c", "practice_input": "p",
                                    "contaminated": False, "gold_path": ""}])
        metrics, per_case = score_traversals(traverse_df, cases_df)
        assert per_case.iloc[0]["entailment_consistent"]
        assert metrics["probes"]["b_change_as_violation"]["rate"] == 0.0
