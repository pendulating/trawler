"""Tests for the traversal stage: state threading, ladder levels, parse
resilience — all with an injected fake inference function (no vLLM)."""

from __future__ import annotations

import json

import pandas as pd
import pytest
from omegaconf import OmegaConf

from dagspaces.ci_heuristic.schemas import STEP_ORDER
from dagspaces.ci_heuristic.stages.traverse import (
    _parse_json_artifact,
    run_traversal,
)

MINIMAL_ARTIFACTS = {
    "s1": {"flows": [{"sender": "x", "recipient": "y", "subject": "z", "information_type": "t", "medium": "", "novelty": ""}]},
    "s2": {"domain": "health care", "nested_contexts": [], "activities": [], "purposes": [], "values_ends": ["care"]},
    "s3": {"senders": ["x"], "recipients": ["y"], "subjects": ["z"], "nonhuman_roles": []},
    "s4": {"transmission_principles": [{"principle": "consent", "explicit": False, "evidence": ""}]},
    "s5": {"norms": [{"norm_flow": "n", "entrenchment_evidence": "", "departures": ["d"], "completeness": "entrenched"}]},
    "s6": {"violation": "yes", "departed_parameters": ["recipient"], "justification": "j"},
    "s7": {"factors": [{"factor": "f", "kind": "autonomy", "affected_parties": [], "direction": "harm"}]},
    "s8": {"meanings": [{"factor_ref": 0, "contextual_end": "care", "advances_or_undermines": "undermines", "argument": "a"}]},
    "s9": {"decision": "modify", "conditions": ["c"], "carrying_findings": ["f"]},
}


def _cases_df(n: int = 2) -> pd.DataFrame:
    return pd.DataFrame({
        "case_id": [f"case{i}" for i in range(n)],
        "tier": ["c"] * n,
        "practice_input": [f"Practice {i}" for i in range(n)],
        "contaminated": [False] * n,
        "gold_path": [""] * n,
    })


def _cfg(level: str, **ladder_extra):
    return OmegaConf.create({
        "ladder": {"level": level, **ladder_extra},
        "sampling_params": {"temperature": 0.0, "max_tokens": 64},
    })


class FakeInference:
    """Mirrors run_vllm_inference's contract; records every round."""

    def __init__(self, reply_fn=None):
        self.rounds = []
        self.reply_fn = reply_fn

    def __call__(self, df, cfg, preprocess, postprocess, stage_name):
        rows = []
        for row in df.to_dict("records"):
            row = preprocess(row)
            self.rounds.append({
                "stage_name": stage_name,
                "case_id": row["case_id"],
                "messages": row["messages"],
                "sampling_params": row["sampling_params"],
            })
            step = stage_name.rsplit("_", 1)[-1]
            if self.reply_fn:
                row["generated_text"] = self.reply_fn(stage_name, row)
            elif step in MINIMAL_ARTIFACTS:
                row["generated_text"] = json.dumps(MINIMAL_ARTIFACTS[step])
            else:
                row["generated_text"] = json.dumps({"violates_privacy": "yes", "decision": "reject", "reasoning": "r"})
            rows.append(postprocess(row))
        return pd.DataFrame(rows)


class TestChainTraversal:
    def test_nine_rounds_state_threading_and_schema_per_round(self):
        fake = FakeInference()
        out = run_traversal(_cases_df(2), _cfg("l3"), run_inference=fake)

        # 9 steps x 2 cases, long format
        assert len(out) == 18
        assert list(out["step"].unique()) == STEP_ORDER

        # State threading: the s6 prompt must contain s5's artifact for the SAME case
        s6_calls = [r for r in fake.rounds if r["stage_name"].endswith("_s6")]
        for call in s6_calls:
            usr = call["messages"][1]["content"]
            assert '"s5"' in usr and '"entrenched"' in usr
            assert "STEP 6 of 9" in usr

        # Guided decoding schema differs per round and matches the step
        s1_call = next(r for r in fake.rounds if r["stage_name"].endswith("_s1"))
        s9_call = next(r for r in fake.rounds if r["stage_name"].endswith("_s9"))
        assert "flows" in json.dumps(s1_call["sampling_params"]["guided_decoding"]["json"])
        assert "carrying_findings" in json.dumps(s9_call["sampling_params"]["guided_decoding"]["json"])

    def test_l2_omits_guiding_questions_l3_includes(self):
        fake2, fake3 = FakeInference(), FakeInference()
        run_traversal(_cases_df(1), _cfg("l2"), run_inference=fake2)
        run_traversal(_cases_df(1), _cfg("l3"), run_inference=fake3)
        usr2 = next(r for r in fake2.rounds if r["stage_name"].endswith("_s3"))["messages"][1]["content"]
        usr3 = next(r for r in fake3.rounds if r["stage_name"].endswith("_s3"))["messages"][1]["content"]
        assert "Guiding questions" not in usr2
        assert "Guiding questions" in usr3 and "Who/what is sending information?" in usr3

    def test_parse_failure_degrades_not_aborts(self):
        def reply(stage_name, row):
            if stage_name.endswith("_s4"):
                return "utter garbage, no json"
            step = stage_name.rsplit("_", 1)[-1]
            return json.dumps(MINIMAL_ARTIFACTS[step])

        fake = FakeInference(reply_fn=reply)
        out = run_traversal(_cases_df(1), _cfg("l2"), run_inference=fake)

        s4_row = out[out["step"] == "s4"].iloc[0]
        assert s4_row["parse_status"] == "unparseable"
        assert "parse_error" in json.loads(s4_row["artifact_json"])
        # Chain continued: all 9 steps present, s5 prompt carries the error artifact
        assert len(out) == 9
        s5_usr = next(r for r in fake.rounds if r["stage_name"].endswith("_s5"))["messages"][1]["content"]
        assert "parse_error" in s5_usr

    def test_l4_requires_exemplar_and_injects_it(self):
        with pytest.raises(ValueError, match="exemplar_path"):
            run_traversal(_cases_df(1), _cfg("l4"), run_inference=FakeInference())

        gold_path = "dagspaces/ci_heuristic/corpus/tier_a/kumar2024_fitbit.json"
        fake = FakeInference()
        run_traversal(_cases_df(1), _cfg("l4", exemplar_path=gold_path), run_inference=fake)
        usr = fake.rounds[0]["messages"][1]["content"]
        assert "Worked example" in usr and "Fitbit" in usr


class TestMonolithicLevels:
    def test_l0_single_round_no_heuristic(self):
        fake = FakeInference()
        out = run_traversal(_cases_df(3), _cfg("l0"), run_inference=fake)
        assert len(out) == 3 and set(out["step"]) == {"monolithic"}
        usr = fake.rounds[0]["messages"][1]["content"]
        assert "Step 5" not in usr and "heuristic" not in usr.lower()

    def test_l1_single_round_with_full_heuristic(self):
        fake = FakeInference()
        out = run_traversal(_cases_df(1), _cfg("l1"), run_inference=fake)
        assert len(out) == 1
        usr = fake.rounds[0]["messages"][1]["content"]
        assert "Step 1" in usr and "Step 9" in usr and "prima facie" in usr.lower()

    def test_unknown_level_raises(self):
        with pytest.raises(ValueError, match="ladder.level"):
            run_traversal(_cases_df(1), _cfg("l7"), run_inference=FakeInference())


class TestParseJsonArtifact:
    def test_clean_json(self):
        artifact, status = _parse_json_artifact('{"a": 1}')
        assert artifact == {"a": 1} and status == "parsed"

    def test_recovery_from_wrapped_json(self):
        artifact, status = _parse_json_artifact('Sure! Here: {"a": 1} hope that helps')
        assert artifact == {"a": 1} and status == "recovered"

    def test_garbage_and_empty(self):
        assert _parse_json_artifact("no braces")[1] == "unparseable"
        assert _parse_json_artifact("")[1] == "unparseable"
