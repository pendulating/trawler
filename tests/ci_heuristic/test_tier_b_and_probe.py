"""Tests for the Tier B vignette generator, extended case loading, and the
TP-elicitation probe stage."""

from __future__ import annotations

import json

import pandas as pd
import pytest
from omegaconf import OmegaConf

from dagspaces.ci_heuristic.stages.load_cases import load_cases
from dagspaces.ci_heuristic.stages.tp_probe import run_tp_probe
from dagspaces.ci_heuristic.tier_b_generator import (
    CONTEXTS,
    PARAMETERS,
    generate,
    validate_structure,
)


class TestGenerator:
    def test_deterministic_for_seed(self):
        a = generate(seed=7, phrasings_per_cell=2)
        b = generate(seed=7, phrasings_per_cell=2)
        assert a.equals(b)
        c = generate(seed=8, phrasings_per_cell=2)
        assert not a["practice_input"].equals(c["practice_input"])

    def test_structure_validates_and_covers_all_classes(self):
        df = generate(seed=7, phrasings_per_cell=2)
        counts = validate_structure(df)
        assert counts["problems"] == 0
        assert counts["departures"] > 0 and counts["controls"] > 0 and counts["incomplete"] > 0
        assert counts["multi_tp"] > 0 and counts["sender_ne_subject"] > 0
        # every parameter appears as a departure class
        assert set(df[df.prima_facie == "yes"]["departed_parameter"].unique()) == set(PARAMETERS)

    def test_departure_vignette_contains_departing_value_not_norm_value(self):
        df = generate(seed=7, phrasings_per_cell=1)
        row = df[(df.context_key == "health_gp") & (df.departed_parameter == "recipient")].iloc[0]
        flow = row["flow_statement"]
        assert "insurance" in flow or "landlord" in flow
        assert "their doctor" not in flow
        # ...but the norm statement still names the entrenched recipient
        assert "their doctor" in row["norm_statement"]

    def test_incomplete_cases_have_no_norm_statement(self):
        df = generate(seed=7, phrasings_per_cell=1)
        inc = df[df.prima_facie == "incomplete_norms"]
        assert (inc["norm_statement"] == "").all()
        assert (inc["departed_parameter"] == "none").all()

    def test_controls_render_the_norm_flow(self):
        df = generate(seed=7, phrasings_per_cell=1)
        ctrl = df[(df.context_key == "voting") & (df.prima_facie == "no")].iloc[0]
        assert "ballot" in ctrl["flow_statement"]

    def test_context_coverage(self):
        # 5+ social domains as required by the corpus plan
        assert len(CONTEXTS) >= 8


class TestTierBLoading:
    def test_loader_carries_labels_through(self):
        df = load_cases(tiers=["b"])
        assert len(df) >= 500
        for col in ["prima_facie", "departed_parameter", "flow_statement", "norm_statement"]:
            assert col in df.columns
        assert set(df["prima_facie"].unique()) == {"yes", "no", "incomplete_norms"}

    def test_mixed_tiers_do_not_collide(self):
        df = load_cases(tiers=["b", "c"])
        assert df["case_id"].is_unique


class FakeInference:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def __call__(self, df, cfg, preprocess, postprocess, stage_name):
        rows = []
        for row in df.to_dict("records"):
            row = preprocess(row)
            self.calls.append(row)
            row["generated_text"] = self.reply
            rows.append(postprocess(row))
        return pd.DataFrame(rows)


class TestTPProbe:
    def _df(self):
        return pd.DataFrame({
            "case_id": ["b1", "b2"],
            "tier": ["b", "b"],
            "practice_input": ["full vignette 1", "full vignette 2"],
            "flow_statement": ["patient tells doctor symptoms", ""],
        })

    def test_probe_uses_flow_statement_with_practice_fallback(self):
        fake = FakeInference(json.dumps({"conditions": ["the doctor needs it", "the patient consents"]}))
        cfg = OmegaConf.create({"sampling_params": {"temperature": 0.0}})
        out = run_tp_probe(self._df(), cfg, run_inference=fake)

        assert list(out["n_conditions"]) == [2, 2]
        assert (out["parse_status"] == "parsed").all()
        # row 1 probed on its flow_statement; row 2 fell back to practice_input
        assert "patient tells doctor" in fake.calls[0]["messages"][1]["content"]
        assert "full vignette 2" in fake.calls[1]["messages"][1]["content"]

    def test_unparseable_reply_yields_empty_conditions(self):
        fake = FakeInference("not json at all")
        cfg = OmegaConf.create({"sampling_params": {}})
        out = run_tp_probe(self._df(), cfg, run_inference=fake)
        assert (out["parse_status"] == "unparseable").all()
        assert (out["n_conditions"] == 0).all()
