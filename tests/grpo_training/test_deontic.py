"""Deterministic deontic-force → appropriateness reasoning.

Pins the mapping the reranker judge uses to recover the appropriateness-
consistency axis it cannot see, and the parsing of governing-norm force and
candidate appropriateness labels. Also guards that the force→gold mapping stays
in sync with the judgment-vignette gold labels (single source of truth).
"""

import json

import pytest

from dagspaces.grpo_training.stages import deontic as d


class TestExpectedAppropriateness:
    @pytest.mark.parametrize("force,expected", [
        ("obligatory", "appropriate"),
        ("recommended", "appropriate"),
        ("prohibited", "inappropriate"),
        ("discouraged", "inappropriate"),
        ("permitted", None),
        ("unknown", None),
        ("", None),
        (None, None),
        ("PROHIBITED", "inappropriate"),  # case-insensitive
    ])
    def test_mapping(self, force, expected):
        assert d.expected_appropriateness(force) == expected


class TestAppropriatenessConsistency:
    def test_match_is_one(self):
        assert d.appropriateness_consistency("inappropriate", "prohibited") == 1.0
        assert d.appropriateness_consistency("appropriate", "obligatory") == 1.0

    def test_contradiction_is_zero(self):
        assert d.appropriateness_consistency("appropriate", "prohibited") == 0.0
        assert d.appropriateness_consistency("inappropriate", "obligatory") == 0.0

    def test_ambiguous_is_neutral(self):
        assert d.appropriateness_consistency("ambiguous", "prohibited") == d.NEUTRAL_CONSISTENCY

    def test_undetermined_force_is_neutral(self):
        assert d.appropriateness_consistency("appropriate", "permitted") == d.NEUTRAL_CONSISTENCY
        assert d.appropriateness_consistency("inappropriate", "") == d.NEUTRAL_CONSISTENCY

    def test_missing_label_is_neutral(self):
        assert d.appropriateness_consistency(None, "prohibited") == d.NEUTRAL_CONSISTENCY


class TestGoverningNormForce:
    def test_top_norm_wins(self):
        norms = json.dumps([
            {"normative_force": "prohibited", "norm_articulation": "x"},
            {"normative_force": "obligatory", "norm_articulation": "y"},
        ])
        assert d.governing_norm_force(norms) == "prohibited"

    def test_raz_prefixed_field(self):
        norms = json.dumps([{"raz_normative_force": "discouraged"}])
        assert d.governing_norm_force(norms) == "discouraged"

    def test_empty_and_malformed(self):
        assert d.governing_norm_force("[]") is None
        assert d.governing_norm_force("not json") is None
        assert d.governing_norm_force(None) is None

    def test_accepts_parsed_list(self):
        assert d.governing_norm_force([{"normative_force": "permitted"}]) == "permitted"


class TestFlowLabels:
    def test_extraction_level(self):
        doc = json.dumps([{"appropriateness": "inappropriate"}])
        assert d.flow_appropriateness_labels(doc) == ["inappropriate"]

    def test_nested_flow(self):
        doc = json.dumps([{"flow": {"appropriateness": "appropriate"}}])
        assert d.flow_appropriateness_labels(doc) == ["appropriate"]

    def test_potential_appropriateness_fallback(self):
        doc = json.dumps([{"potential_appropriateness": "ambiguous"}])
        assert d.flow_appropriateness_labels(doc) == ["ambiguous"]

    def test_no_flow_text_yields_none(self):
        assert d.flow_appropriateness_labels("This declares NO information flows.") == []


class TestCandidateConsistency:
    def test_mean_over_flows(self):
        doc = json.dumps([
            {"appropriateness": "inappropriate"},  # matches prohibited → 1.0
            {"appropriateness": "appropriate"},    # contradicts        → 0.0
        ])
        assert d.candidate_appropriateness_consistency(doc, "prohibited") == 0.5

    def test_no_labels_is_neutral(self):
        assert d.candidate_appropriateness_consistency("no flows", "prohibited") == d.NEUTRAL_CONSISTENCY


class TestSingleSourceOfTruth:
    def test_force_to_gold_matches_appropriateness(self):
        # "appropriate" ⇔ "yes", "inappropriate" ⇔ "no".
        appr_to_gold = {"appropriate": "yes", "inappropriate": "no"}
        for force, appr in d.FORCE_TO_APPROPRIATENESS.items():
            assert d.FORCE_TO_GOLD[force] == appr_to_gold[appr]
        assert set(d.FORCE_TO_GOLD) == set(d.FORCE_TO_APPROPRIATENESS)

    def test_grpo_vignettes_use_shared_mapping(self):
        # The vignette builder imports FORCE_TO_GOLD from deontic; confirm the
        # symbol is the one it relies on so the two cannot drift.
        from dagspaces.grpo_training.stages.deontic import FORCE_TO_GOLD
        assert FORCE_TO_GOLD["prohibited"] == "no"
        assert FORCE_TO_GOLD["obligatory"] == "yes"


class TestDirectionMultiplier:
    """v9: affine map consistency∈[0,1] → reward multiplier∈[floor,1]."""

    def test_endpoints_and_midpoint(self):
        assert d.direction_multiplier(1.0) == pytest.approx(1.0)   # correct verdict
        assert d.direction_multiplier(0.5) == pytest.approx(0.7)   # hedge (floor 0.4)
        assert d.direction_multiplier(0.0) == pytest.approx(0.4)   # wrong verdict

    def test_floor_is_configurable(self):
        assert d.direction_multiplier(0.0, floor=0.0) == pytest.approx(0.0)
        assert d.direction_multiplier(0.0, floor=0.6) == pytest.approx(0.6)
        # The correct verdict always reaches 1.0 regardless of floor.
        assert d.direction_multiplier(1.0, floor=0.6) == pytest.approx(1.0)

    def test_monotone_in_consistency(self):
        assert (d.direction_multiplier(0.0) < d.direction_multiplier(0.5)
                < d.direction_multiplier(1.0))

    def test_consistency_clamped(self):
        assert d.direction_multiplier(1.5) == pytest.approx(1.0)
        assert d.direction_multiplier(-0.5) == pytest.approx(0.4)

    def test_invalid_floor_rejected(self):
        with pytest.raises(ValueError):
            d.direction_multiplier(0.5, floor=1.5)


class TestAppropriatenessMultiplier:
    """v10: cost-sensitive (asymmetric) direction multiplier — false-permits
    (prohibited-governed flow called "appropriate") can floor lower than
    false-forbids, to counter the ~4:1 appropriate:inappropriate norm prior."""

    def test_correct_verdict_full_reward_either_direction(self):
        assert d.appropriateness_multiplier("inappropriate", "prohibited") == pytest.approx(1.0)
        assert d.appropriateness_multiplier("appropriate", "obligatory") == pytest.approx(1.0)

    def test_false_permit_floors_lower_than_false_forbid(self):
        # false-permit: said "appropriate" on a prohibited-governed flow.
        fp = d.appropriateness_multiplier(
            "appropriate", "prohibited", floor=0.4, floor_prohibit=0.1)
        # false-forbid: said "inappropriate" on an obligatory-governed flow.
        ff = d.appropriateness_multiplier(
            "inappropriate", "obligatory", floor=0.4, floor_prohibit=0.1)
        assert fp == pytest.approx(0.1)
        assert ff == pytest.approx(0.4)
        assert fp < ff

    def test_discouraged_is_also_prohibitive(self):
        assert d.appropriateness_multiplier(
            "appropriate", "discouraged", floor=0.4, floor_prohibit=0.1) == pytest.approx(0.1)

    def test_hedge_unaffected_by_prohibit_floor(self):
        assert d.appropriateness_multiplier(
            "ambiguous", "prohibited", floor=0.4, floor_prohibit=0.1) == pytest.approx(0.7)

    def test_none_floor_prohibit_reproduces_symmetric_v9(self):
        # floor_prohibit=None must equal direction_multiplier(consistency, floor).
        for lab, force in [("inappropriate", "prohibited"), ("appropriate", "prohibited"),
                           ("appropriate", "obligatory"), ("inappropriate", "obligatory"),
                           ("ambiguous", "prohibited"), ("appropriate", "permitted")]:
            cons = d.appropriateness_consistency(lab, force)
            assert d.appropriateness_multiplier(lab, force, floor=0.4) == pytest.approx(
                d.direction_multiplier(cons, 0.4)), (lab, force)

    def test_invalid_prohibit_floor_rejected(self):
        with pytest.raises(ValueError):
            d.appropriateness_multiplier("appropriate", "prohibited", floor_prohibit=1.5)


class TestCandidateMultiplier:
    """Mean cost-sensitive multiplier over a candidate's flows."""

    def test_single_false_permit(self):
        doc = json.dumps([{"appropriateness": "appropriate"}])  # prohibited → false-permit
        assert d.candidate_appropriateness_multiplier(
            doc, "prohibited", floor=0.4, floor_prohibit=0.1) == pytest.approx(0.1)

    def test_means_over_flows(self):
        doc = json.dumps([
            {"appropriateness": "inappropriate"},  # correct on prohibited → 1.0
            {"appropriateness": "appropriate"},    # false-permit          → 0.1
        ])
        assert d.candidate_appropriateness_multiplier(
            doc, "prohibited", floor=0.4, floor_prohibit=0.1) == pytest.approx(0.55)

    def test_no_labels_neutral(self):
        assert d.candidate_appropriateness_multiplier(
            "no flows", "prohibited", floor=0.4, floor_prohibit=0.1) == pytest.approx(0.7)

    def test_single_flow_matches_v9_path_when_symmetric(self):
        # A single-flow candidate under floor_prohibit=None equals the v9
        # consistency→direction path exactly.
        doc = json.dumps([{"appropriateness": "appropriate"}])
        cons = d.candidate_appropriateness_consistency(doc, "prohibited")
        assert d.candidate_appropriateness_multiplier(doc, "prohibited", floor=0.4) == pytest.approx(
            d.direction_multiplier(cons, 0.4))
