"""Tests for prompt construction, the descriptive firewall, exemplar
contamination guard, and case loading."""

from __future__ import annotations

import json

import pytest

from dagspaces.ci_heuristic.heuristic_text import FIREWALL_GUARD, HEURISTIC_STEPS
from dagspaces.ci_heuristic.prompts import (
    build_step_prompt,
    build_tp_elicitation_prompt,
    render_exemplar,
)
from dagspaces.ci_heuristic.schemas import DESCRIPTIVE_STEPS, STEP_ORDER
from dagspaces.ci_heuristic.stages.load_cases import load_cases

GOLD_PATH = "dagspaces/ci_heuristic/corpus/tier_a/kumar2024_fitbit.json"


class TestFirewall:
    def test_guard_on_descriptive_steps_only(self):
        for step in STEP_ORDER:
            _, usr = build_step_prompt("P", step, {})
            if step in DESCRIPTIVE_STEPS:
                assert FIREWALL_GUARD in usr, f"{step} must carry the firewall"
            else:
                assert FIREWALL_GUARD not in usr, f"{step} must NOT carry the firewall"

    def test_step_kinds_align_with_firewall(self):
        # The firewall list and the step 'kind' metadata must agree
        for step, meta in HEURISTIC_STEPS.items():
            if meta["kind"] == "descriptive":
                assert step in DESCRIPTIVE_STEPS
            else:
                assert step not in DESCRIPTIVE_STEPS


class TestStatePropagation:
    def test_prior_state_serialized_in_order(self):
        state = {"s2": {"domain": "d"}, "s1": {"flows": []}}
        _, usr = build_step_prompt("P", "s3", state)
        assert usr.index('"s1"') < usr.index('"s2"')
        assert "treat as given" in usr

    def test_no_state_block_on_first_step(self):
        _, usr = build_step_prompt("P", "s1", {})
        assert "prior steps" not in usr


class TestExemplar:
    def test_render_excludes_meta_and_steps_present(self):
        gold = json.load(open(GOLD_PATH))
        ex = render_exemplar(gold)
        assert "steps_present" not in ex
        assert "contaminated" not in ex
        assert "s9_recommendation" in ex

    def test_refuses_non_contaminated(self):
        gold = json.load(open(GOLD_PATH))
        gold["meta"]["contaminated"] = False
        with pytest.raises(ValueError, match="contaminated"):
            render_exemplar(gold)


class TestTPElicitation:
    def test_probe_contains_the_if_frame(self):
        _, usr = build_tp_elicitation_prompt("patient tells doctor their step count")
        assert 'fine IF ___' in usr
        _, usr_p = build_tp_elicitation_prompt("flow", persona="a nurse")
        assert "a nurse" in usr_p


class TestLoadCases:
    def test_tiers_a_and_c_load(self):
        df = load_cases(tiers=["a", "c"], include_contaminated=True)
        assert "kumar2024_fitbit" in set(df["case_id"])
        assert "tier_c_delivery_robot" in set(df["case_id"])
        assert set(df.columns) >= {"case_id", "tier", "practice_input", "contaminated", "gold_path"}

    def test_held_out_excludes_contaminated(self):
        df = load_cases(tiers=["a", "c"], include_contaminated=False)
        assert "kumar2024_fitbit" not in set(df["case_id"])  # flagged contaminated
        assert (df["tier"] == "c").all()

    def test_tier_c_keys_match_part1_variant_ids(self):
        df = load_cases(tiers=["c"])
        expected = {f"tier_c_{v}" for v in
                    ["smartphone", "smart_glasses", "camera_earphones", "dashcam", "delivery_robot"]}
        assert set(df["case_id"]) == expected

    def test_unknown_tier_raises(self):
        with pytest.raises(ValueError, match="Unknown tier"):
            load_cases(tiers=["z"])
