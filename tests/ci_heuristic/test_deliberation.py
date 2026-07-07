"""Tests for the L5 deliberative structures: norm-elicitation aggregation,
stakeholder panel + McDonald-Forte toggle, factor merging, and the full L5
chain with injected inference."""

from __future__ import annotations

import json

import pandas as pd
from omegaconf import OmegaConf

from dagspaces.ci_heuristic.deliberation import (
    NORM_POPULATION,
    STAKEHOLDERS,
    aggregate_expectations,
    build_moderator_prompt,
    build_norm_synthesis_prompt,
    build_s8_analyst_prompt,
    build_stakeholder_prompt,
    merge_factor_artifacts,
    stakeholder_set,
)
from dagspaces.ci_heuristic.stages.traverse import run_traversal
from tests.ci_heuristic.test_traverse import MINIMAL_ARTIFACTS, _cases_df


class TestAggregation:
    def test_entrenched_positive_and_proscriptive(self):
        pos = [{"familiar": True, "appropriate": "yes", "expectation": "e"}] * 9 + \
               [{"familiar": True, "appropriate": "no", "expectation": "e"}]
        neg = [{"familiar": True, "appropriate": "no", "expectation": "e"}] * 9 + \
               [{"familiar": True, "appropriate": "yes", "expectation": "e"}]
        assert aggregate_expectations(pos)["completeness"] == "entrenched"
        assert aggregate_expectations(neg)["completeness"] == "entrenched"  # settled proscription counts

    def test_contested_split(self):
        split = [{"familiar": True, "appropriate": "yes", "expectation": "e"}] * 5 + \
                 [{"familiar": True, "appropriate": "no", "expectation": "e"}] * 5
        assert aggregate_expectations(split)["completeness"] == "contested"

    def test_incomplete_when_unfamiliar(self):
        novel = [{"familiar": False, "appropriate": "unsure", "expectation": ""}] * 6 + \
                 [{"familiar": True, "appropriate": "yes", "expectation": "e"}] * 4
        assert aggregate_expectations(novel)["completeness"] == "incomplete"

    def test_unparseable_members_dropped(self):
        stats = aggregate_expectations([{"parse_error": "x"}, {"familiar": True, "appropriate": "yes", "expectation": "e"}])
        assert stats["n_valid"] == 1


class TestPanel:
    def test_mcdonald_forte_toggle(self):
        full = stakeholder_set(True)
        ablated = stakeholder_set(False)
        assert any(p.marginalized for p in full)
        assert not any(p.marginalized for p in ablated)
        assert len(full) > len(ablated)
        assert len(NORM_POPULATION) >= 10 and any(p.marginalized for p in NORM_POPULATION)

    def test_stakeholder_prompt_carries_prior_with_second_wave(self):
        p = STAKEHOLDERS[0]
        _, usr = build_stakeholder_prompt("P", {"s6": {"violation": "yes"}}, p,
                                            prior_responses=["the operator said X"])
        assert "the operator said X" in usr and "Respect and weigh" in usr
        _, usr_solo = build_stakeholder_prompt("P", {}, p)
        assert "Respect and weigh" not in usr_solo


class TestMerging:
    def test_alias_dedup_with_provenance(self):
        a = {"factors": [{"factor": "coerced disclosure of location", "kind": "coercion",
                           "affected_parties": ["residents"], "direction": "harm"}]}
        b = {"factors": [{"factor": "people are coerced into disclosing location", "kind": "coercion",
                           "affected_parties": ["residents"], "direction": "harm"},
                          {"factor": "chilling of assembly", "kind": "freedom",
                           "affected_parties": ["protesters"], "direction": "harm"}]}
        merged = merge_factor_artifacts([("subject", a), ("civil_liberties", b)])
        assert len(merged["factors"]) == 2
        dup = next(f for f in merged["factors"] if f["kind"] == "coercion")
        assert dup["raised_by"] == ["subject", "civil_liberties"]

    def test_unparseable_member_skipped(self):
        merged = merge_factor_artifacts([("x", {"parse_error": "bad"}), ("y", {"factors": []})])
        assert merged == {"factors": []}


class TestPromptBuilders:
    def test_s8_and_s9_prompts_embed_state(self):
        state = {"s2": {"domain": "health care", "values_ends": ["care"]},
                  "s6": {"violation": "yes"}, "s7": {"factors": []}, "s8": {"meanings": []}}
        _, s8 = build_s8_analyst_prompt("P", state)
        assert "STEP 8" in s8 and "health care" in s8
        _, s9 = build_moderator_prompt("P", state)
        assert "presumption favors entrenched" in " ".join(s9.split())

    def test_norm_synthesis_embeds_population_stats(self):
        stats = {"completeness": "contested", "agreement": 0.6, "unfamiliar_rate": 0.1,
                  "expectations": ["patients tell doctors things in confidence"]}
        _, usr = build_norm_synthesis_prompt("P", {"s1": {"flows": []}}, stats)
        assert "contested" in usr and "in confidence" in usr


class L5FakeInference:
    """Round-aware fake: stage names route to appropriate artifacts."""

    def __init__(self):
        self.stages = []

    def __call__(self, df, cfg, preprocess, postprocess, stage_name):
        self.stages.append((stage_name, len(df)))
        rows = []
        for row in df.to_dict("records"):
            row = preprocess(row)
            if "s5_elicit" in stage_name:
                reply = {"familiar": True, "appropriate": "yes", "expectation": "settled expectation"}
            elif "s7" in stage_name and ("ensemble" in stage_name or "chain" in stage_name or "debate" in stage_name):
                reply = {"factors": [{"factor": f"factor from {row['case_id']}", "kind": "autonomy",
                                        "affected_parties": ["x"], "direction": "harm"}]}
            else:
                step = stage_name.split("_")[-1] if stage_name.split("_")[-1] in MINIMAL_ARTIFACTS else \
                        next((s for s in MINIMAL_ARTIFACTS if f"_{s}_" in stage_name or stage_name.endswith(s)), None)
                reply = MINIMAL_ARTIFACTS.get(step, MINIMAL_ARTIFACTS["s9"])
            row["generated_text"] = json.dumps(reply)
            rows.append(postprocess(row))
        return pd.DataFrame(rows)


class TestL5Chain:
    def _cfg(self, **ladder):
        return OmegaConf.create({
            "ladder": {"level": "l5", "s5_n_personas": 3, "s7_structure": "ensemble",
                        "include_marginalized": True, **ladder},
            "sampling_params": {"temperature": 0.0},
        })

    def test_l5_produces_merged_artifacts_and_member_rows(self):
        fake = L5FakeInference()
        out = run_traversal(_cases_df(2), self._cfg(), run_inference=fake)

        # merged s5/s7 rows exist alongside member rows
        assert (out["step"] == "s5").sum() == 2
        assert (out["step"] == "s7").sum() == 2
        assert out["step"].str.startswith("s5:elicit:").sum() == 2 * 3
        n_panel = len(stakeholder_set(True))
        assert out["step"].str.startswith("s7:member:").sum() == 2 * n_panel

        # merged s5 artifact carries population stats
        s5 = json.loads(out[out["step"] == "s5"].iloc[0]["artifact_json"])
        assert s5["_population_stats"]["completeness"] == "entrenched"

        # s8/s9 used the deliberative builders (moderator language in prompt)
        s9_row = out[out["step"] == "s9"].iloc[0]
        assert "moderating" in s9_row["prompt_usr"]

    def test_mcdonald_forte_arm_reduces_panel(self):
        fake = L5FakeInference()
        out = run_traversal(_cases_df(1), self._cfg(include_marginalized=False), run_inference=fake)
        n_ablated = len(stakeholder_set(False))
        assert out["step"].str.startswith("s7:member:").sum() == n_ablated

    def test_debate_structure_runs_cycles(self):
        fake = L5FakeInference()
        out = run_traversal(_cases_df(1), self._cfg(s7_structure="debate"), run_inference=fake)
        member_rows = out[out["step"].str.startswith("s7:member:")]
        # 2 debaters x 2 cycles
        assert len(member_rows) == 4
        merged = json.loads(out[out["step"] == "s7"].iloc[0]["artifact_json"])
        assert merged["factors"], "final debate positions must merge into s7"
