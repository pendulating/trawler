"""RerankerJudgeClient must satisfy the JudgeClient contract OnlineRGround relies on.

The reranker backend (Qwen3-Reranker via vLLM /rerank) is a drop-in for the
generative judge in ``rground_scoring=ranked``. These tests pin the bits
OnlineRGround and ``_rankings_to_scores`` depend on:

  * judge_ranking_batch returns one entry per candidate (full coverage), with
    distinct ranks derived from the continuous relevance scores and
    grounding_score == the relevance score;
  * scores are realigned to input order even when /rerank sorts its results;
  * any coverage gap or HTTP failure yields None (treated as judge-failed →
    uniform neutral score, never a spurious gradient);
  * the structured ``candidates`` list is preferred over the joined block;
  * continuous scores break the within-group ties that motivated listwise
    judging (the 60%-tied-groups pathology).
"""

import json

import pytest

from dagspaces.grpo_training.stages.clients import (
    RerankerJudgeClient,
    _split_candidates_block,
)
from dagspaces.grpo_training.stages.online_rground import _rankings_to_scores


class _FakeRerankResponse:
    """Mimics vLLM /rerank: {"results": [{"index", "relevance_score"}, ...]}."""

    def __init__(self, scores, sort_desc=False, drop_last=False):
        results = [
            {"index": i, "relevance_score": s} for i, s in enumerate(scores)
        ]
        if drop_last:
            results = results[:-1]
        if sort_desc:
            results = sorted(results, key=lambda r: r["relevance_score"], reverse=True)
        self._results = results

    def raise_for_status(self):
        pass

    def json(self):
        return {"results": self._results}


def _client_returning(response, captured=None, app_weight=0.0):
    # Rank-mechanics tests use app_weight=0 to isolate the reranker relevance
    # from the deontic appropriateness blend (which has its own tests below).
    client = RerankerJudgeClient(model_name="rr", max_retries=1, app_weight=app_weight)

    def fake_post(url, json=None, timeout=None):
        if captured is not None:
            captured.append({"url": url, "body": json})
        if isinstance(response, Exception):
            raise response
        return response

    client._session.post = fake_post
    return client


class TestRankingContract:
    def test_full_coverage_distinct_ranks(self):
        # scores ascending by index; rank 1 should go to the highest score.
        client = _client_returning(_FakeRerankResponse([0.1, 0.9, 0.5]))
        out = client._ranking_single({"candidates": ["a", "b", "c"]})
        assert out is not None
        assert {e["candidate_index"] for e in out} == {0, 1, 2}
        ranks = sorted(e["rank"] for e in out)
        assert ranks == [1, 2, 3]  # distinct, contiguous
        by_idx = {e["candidate_index"]: e for e in out}
        assert by_idx[1]["rank"] == 1  # highest score → best rank
        assert by_idx[0]["rank"] == 3  # lowest score → worst rank
        assert by_idx[1]["grounding_score"] == pytest.approx(0.9)

    def test_scores_realigned_when_server_sorts(self):
        client = _client_returning(_FakeRerankResponse([0.1, 0.9, 0.5], sort_desc=True))
        out = client._ranking_single({"candidates": ["a", "b", "c"]})
        by_idx = {e["candidate_index"]: e["grounding_score"] for e in out}
        assert by_idx[0] == pytest.approx(0.1)
        assert by_idx[1] == pytest.approx(0.9)
        assert by_idx[2] == pytest.approx(0.5)

    def test_partial_coverage_returns_none(self):
        client = _client_returning(_FakeRerankResponse([0.1, 0.9, 0.5], drop_last=True))
        out = client._ranking_single({"candidates": ["a", "b", "c"]})
        assert out is None

    def test_http_failure_returns_none(self):
        client = _client_returning(RuntimeError("boom"))
        out = client._ranking_single({"candidates": ["a", "b"]})
        assert out is None

    def test_batch_returns_none_per_failed_group(self):
        client = _client_returning(RuntimeError("boom"))
        out = client.judge_ranking_batch([{"candidates": ["a", "b"]}])
        assert out == [None]

    def test_prefers_structured_candidates_over_block(self):
        captured = []
        client = _client_returning(_FakeRerankResponse([0.2, 0.8]), captured)
        client._ranking_single({
            "candidates": ["flow-A", "flow-B"],
            "candidates_block": "### Candidate 0\nIGNORED",
            "n_candidates": 2,
        })
        docs = captured[0]["body"]["documents"]
        assert docs == ["flow-A", "flow-B"]

    def test_falls_back_to_block_when_no_candidates(self):
        captured = []
        client = _client_returning(_FakeRerankResponse([0.2, 0.8]), captured)
        block = "### Candidate 0\nflow-A\n\n### Candidate 1\nflow-B"
        client._ranking_single({"candidates_block": block, "n_candidates": 2})
        docs = captured[0]["body"]["documents"]
        assert docs == ["flow-A", "flow-B"]


class TestPerFlowAndCoverage:
    def test_judge_batch_maps_relevance(self):
        client = _client_returning(_FakeRerankResponse([0.7]))
        out = client.judge_batch([{"flow_json": "{}", "norm_universe_json": "[]"}])
        assert out[0]["norm_match_score"] == pytest.approx(0.7)
        assert out[0]["governance_score"] == pytest.approx(0.7)

    def test_judge_batch_appropriateness_from_deontic_not_score(self):
        # Low relevance must NOT flip appropriateness false (that was the old
        # proxy); appropriateness is now decided by deontic agreement.
        norms = json.dumps([{"normative_force": "prohibited"}])
        client = _client_returning(_FakeRerankResponse([0.3]))
        # flow labeled inappropriate, governed by a prohibiting norm → consistent
        consistent = client.judge_batch([{
            "flow_json": json.dumps({"appropriateness": "inappropriate"}),
            "norm_universe_json": norms,
        }])
        assert consistent[0]["appropriateness_consistent"] is True
        # flow labeled appropriate under the same prohibiting norm → inconsistent
        client2 = _client_returning(_FakeRerankResponse([0.9]))
        inconsistent = client2.judge_batch([{
            "flow_json": json.dumps({"appropriateness": "appropriate"}),
            "norm_universe_json": norms,
        }])
        assert inconsistent[0]["appropriateness_consistent"] is False

    def test_coverage_batch(self):
        client = _client_returning(_FakeRerankResponse([0.6]))
        out = client.judge_coverage_batch([{"chunk_text": "t", "norm_universe_json": "[]"}])
        assert out[0]["coverage_score"] == pytest.approx(0.6)
        assert out[0]["passage_contains_governed_flows"] is True


class TestSpreadPreservation:
    def test_continuous_scores_avoid_group_ties(self):
        # The reranker's continuous scores → distinct ranks → non-zero
        # within-group spread, the property absolute LLM scoring lost.
        client = _client_returning(_FakeRerankResponse([0.41, 0.42, 0.43, 0.44]))
        out = client._ranking_single({"candidates": ["a", "b", "c", "d"]})
        scores = _rankings_to_scores(out, n_candidates=4, rank_weight=0.5)
        assert len(set(round(s, 6) for s in scores)) == 4  # all distinct
        assert max(scores) - min(scores) > 0.0  # non-zero advantage spread


class TestDeonticBlend:
    def test_appropriateness_breaks_equal_relevance(self):
        # Equal reranker relevance; deontic consistency must decide the order.
        norms = json.dumps([{"normative_force": "prohibited"}])
        cand_consistent = json.dumps([{"appropriateness": "inappropriate"}])  # matches
        cand_contradict = json.dumps([{"appropriateness": "appropriate"}])    # contradicts
        client = _client_returning(_FakeRerankResponse([0.5, 0.5]), app_weight=0.2)
        out = client._ranking_single({
            "candidates": [cand_consistent, cand_contradict],
            "norm_universe_json": norms,
        })
        by_idx = {e["candidate_index"]: e for e in out}
        assert by_idx[0]["rank"] == 1  # consistent candidate wins
        assert by_idx[1]["rank"] == 2
        # grounding = 0.8*0.5 + 0.2*consistency
        assert by_idx[0]["grounding_score"] == pytest.approx(0.6)   # 0.4 + 0.2*1.0
        assert by_idx[1]["grounding_score"] == pytest.approx(0.4)   # 0.4 + 0.2*0.0

    def test_app_weight_zero_ignores_deontic(self):
        norms = json.dumps([{"normative_force": "prohibited"}])
        client = _client_returning(_FakeRerankResponse([0.5, 0.5]), app_weight=0.0)
        out = client._ranking_single({
            "candidates": [json.dumps([{"appropriateness": "appropriate"}]),
                           json.dumps([{"appropriateness": "inappropriate"}])],
            "norm_universe_json": norms,
        })
        # Pure relevance: equal scores → grounding equal, deontic ignored.
        assert all(e["grounding_score"] == pytest.approx(0.5) for e in out)

    def test_no_flow_candidate_stays_neutral(self):
        # A no-flow declaration exposes no appropriateness label → neutral 0.5,
        # so it isn't punished on the deontic axis.
        norms = json.dumps([{"normative_force": "prohibited"}])
        client = _client_returning(_FakeRerankResponse([0.5]), app_weight=0.2)
        out = client._ranking_single({
            "candidates": ["This candidate declares NO information flows."],
            "norm_universe_json": norms,
        })
        # grounding = 0.8*0.5 + 0.2*0.5 = 0.5
        assert out[0]["grounding_score"] == pytest.approx(0.5)


class TestSplitCandidatesBlock:
    def test_roundtrip(self):
        block = "### Candidate 0\nfoo\n\n### Candidate 1\nbar baz"
        assert _split_candidates_block(block, 2) == ["foo", "bar baz"]

    def test_empty(self):
        assert _split_candidates_block("", 0) == []
