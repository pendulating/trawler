"""Judge requests must enforce schemas via ``response_format`` (2026-06-10).

The 2026-06-09 redesign smoke run had a 100% ranking-judge failure rate:
the requests carried the legacy ``guided_json`` extra-body param, which
vLLM >= 0.19 silently ignores (no error, no enforcement). The judge
free-formed ``candidate_id`` instead of the schema's ``candidate_index``,
candidate-coverage validation rejected every response, and R_ground
collapsed to a uniform 0.5 for the whole run. These tests pin the
``response_format`` envelope on all three judge endpoints and the lenient
``candidate_id`` alias in the ranking parser.
"""

import json

from dagspaces.grpo_training.stages.clients import JudgeClient
from dagspaces.grpo_training.schemas import CompletionRankingJudgment

RANK_SCHEMA = CompletionRankingJudgment.model_json_schema()


class _FakeResponse:
    def __init__(self, content):
        self._content = content

    def raise_for_status(self):
        pass

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


def _capture_client(content, captured):
    client = JudgeClient(
        system_prompt="sys",
        prompt_template="{{chunk_text}}",
        json_schema=RANK_SCHEMA,
        max_retries=1,
    )

    def fake_post(url, json=None, timeout=None):
        captured.append(json)
        return _FakeResponse(content)

    client._session.post = fake_post
    return client


def _rankings_content(index_key, n):
    return json.dumps({
        "rankings": [
            {index_key: i, "rank": i + 1, "grounding_score": 0.5}
            for i in range(n)
        ],
        "explanation": "x",
    })


class TestStructuredOutputEnvelope:
    def test_ranking_request_uses_response_format(self):
        captured = []
        client = _capture_client(_rankings_content("candidate_index", 2), captured)
        client._ranking_single(
            {"chunk_text": "t", "candidates_block": "c", "n_candidates": 2},
            "sys", "{{candidates_block}}", RANK_SCHEMA,
        )
        body = captured[0]
        assert "guided_json" not in body
        assert body["response_format"]["type"] == "json_schema"
        assert body["response_format"]["json_schema"]["schema"] == RANK_SCHEMA

    def test_flow_judge_request_uses_response_format(self):
        captured = []
        client = _capture_client(json.dumps({"grounding_score": 0.5}), captured)
        client._judge_single({"chunk_text": "t", "flow_json": "{}"})
        body = captured[0]
        assert "guided_json" not in body
        assert body["response_format"]["type"] == "json_schema"

    def test_coverage_request_uses_response_format(self):
        captured = []
        client = _capture_client(
            json.dumps({"coverage_score": 0.5,
                        "passage_contains_governed_flows": False}),
            captured,
        )
        client._coverage_single({"chunk_text": "t"}, "sys", "{{chunk_text}}",
                                RANK_SCHEMA)
        body = captured[0]
        assert "guided_json" not in body
        assert body["response_format"]["type"] == "json_schema"


class TestRankingParsing:
    def test_candidate_index_parses(self):
        client = _capture_client(_rankings_content("candidate_index", 3), [])
        out = client._ranking_single(
            {"chunk_text": "t", "candidates_block": "c", "n_candidates": 3},
            "sys", "{{candidates_block}}", RANK_SCHEMA,
        )
        assert out is not None
        assert {e["candidate_index"] for e in out} == {0, 1, 2}

    def test_candidate_id_alias_parses(self):
        client = _capture_client(_rankings_content("candidate_id", 3), [])
        out = client._ranking_single(
            {"chunk_text": "t", "candidates_block": "c", "n_candidates": 3},
            "sys", "{{candidates_block}}", RANK_SCHEMA,
        )
        assert out is not None
        assert {e["candidate_index"] for e in out} == {0, 1, 2}

    def test_partial_coverage_returns_none(self):
        client = _capture_client(_rankings_content("candidate_index", 2), [])
        out = client._ranking_single(
            {"chunk_text": "t", "candidates_block": "c", "n_candidates": 4},
            "sys", "{{candidates_block}}", RANK_SCHEMA,
        )
        assert out is None
