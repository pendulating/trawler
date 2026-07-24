"""Tests for the production `R-GROUND` / `R-CONTRAST` auxiliary scorers.

All HTTP is mocked (CPU-only): the keeper's ``EmbeddingClient`` /
``JudgeClient`` / ``NormRetriever`` are replaced with fakes that record their
calls, so these tests exercise the *scorer contracts* — the listwise call
shape, rank+grounding parse, seeded wrong-book determinism (and never-own-book),
the ``1 − grounding`` contrast inversion, and the judge-failure → group-neutral
convention — with no GPU and no network.

Covers wiki/grpo_redesign/{reward-ground,reward-contrast}.md and the injected
``ground_scorer`` / ``contrast_scorer`` contract of :mod:`modular_reward`.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from dagspaces.grpo_training.stages.aux_scorers import (
    _ContrastScorer,
    _GroundScorer,
    make_aux_scorers,
    seeded_wrong_book,
)
from dagspaces.grpo_training.stages.modular_reward import ModularReward
from dagspaces.grpo_training.stages.online_rground import _rankings_to_scores


# ---------------------------------------------------------------------------
# Fakes (record calls; no network)
# ---------------------------------------------------------------------------
class FakeEmbedding:
    """Returns a fixed non-degenerate embedding per query."""

    def __init__(self, dim: int = 4):
        self.dim = dim
        self.calls: list[list[str]] = []

    def encode_batch(self, texts):
        self.calls.append(list(texts))
        n = len(texts)
        if n == 0:
            return np.empty((0, 0))
        # Deterministic, non-zero rows so the mean group embedding is non-zero.
        return np.tile(np.arange(1, self.dim + 1, dtype=np.float32), (n, 1))


class FakeRetriever:
    """Records every ``retrieve`` call; returns a fixed norm-universe JSON."""

    def __init__(self, norms_json: str = '[{"norm_articulation": "n"}]'):
        self.norms_json = norms_json
        self.calls: list[dict] = []

    def retrieve(
        self,
        query_embedding,
        source_id,
        contrastive_source=None,
        return_scores=False,
        top_k=None,
    ):
        self.calls.append(
            {
                "source_id": source_id,
                "contrastive_source": contrastive_source,
                "top_k": top_k,
            }
        )
        if return_scores:
            return self.norms_json, [0.9]
        return self.norms_json


class FakeJudge:
    """Returns a scripted ranking (or None) and records the items it saw."""

    def __init__(self, rankings):
        # rankings: a list[dict] to return for every group, or None to fail.
        self._rankings = rankings
        self.items: list[dict] = []

    def judge_ranking_batch(self, items, system_prompt="", prompt_template="", json_schema=None):
        self.items.extend(items)
        return [self._rankings for _ in items]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def extraction(sender="alice") -> str:
    return json.dumps(
        {
            "reasoning": "a flow is present",
            "has_information_exchange": True,
            "flows": [
                {
                    "sender": sender,
                    "recipient": "bob",
                    "subject": "carol",
                    "information_type": "medical history",
                    "transmission_principle": "confidentiality",
                }
            ],
        }
    )


def meta(source_id="1342", chunk_id="c1") -> dict:
    return {"task_type": "extract", "source_id": source_id, "chunk_id": chunk_id}


def rankings(*triples) -> list[dict]:
    """Build a rankings list from (candidate_index, rank, grounding) triples."""
    return [
        {"candidate_index": ci, "rank": r, "grounding_score": g}
        for ci, r, g in triples
    ]


# ---------------------------------------------------------------------------
# R-GROUND: call shape + listwise parse + own-book retrieval
# ---------------------------------------------------------------------------
def test_ground_call_shape_and_listwise_parse():
    emb = FakeEmbedding()
    ret = FakeRetriever()
    ranks = rankings((0, 1, 0.9), (1, 2, 0.5), (2, 3, 0.1))
    judge = FakeJudge(ranks)
    scorer = _GroundScorer(emb, judge, ret, rank_top_k=5, rank_weight=0.5)

    comps = [extraction("alice"), extraction("dave"), extraction("erin")]
    metas = [meta(), meta(), meta()]
    scores = scorer(completions=comps, prompts=["p"] * 3, metadata_list=metas)

    # Exactly one listwise judge call, covering all 3 candidates.
    assert len(judge.items) == 1
    item = judge.items[0]
    assert item["n_candidates"] == 3
    assert len(item["candidates"]) == 3
    assert "### Candidate 0" in item["candidates_block"]
    assert "### Candidate 2" in item["candidates_block"]

    # Own-book retrieval: no contrastive_source, source_id threaded through.
    assert len(ret.calls) == 1
    assert ret.calls[0]["contrastive_source"] is None
    assert ret.calls[0]["source_id"] == "1342"

    # Scores are the keeper's rank/grounding blend, candidate-index aligned.
    assert scores == _rankings_to_scores(ranks, 3, rank_weight=0.5)
    # Sanity: rank-1/high-grounding candidate outscores rank-3/low.
    assert scores[0] > scores[1] > scores[2]


def test_ground_judge_failure_returns_none():
    scorer = _GroundScorer(FakeEmbedding(), FakeJudge(None), FakeRetriever())
    out = scorer(
        completions=[extraction(), extraction()],
        prompts=["p", "p"],
        metadata_list=[meta(), meta()],
    )
    assert out is None  # → modular applies uniform 0.5 group-neutral


def test_ground_no_flows_returns_none():
    # No embeddable queries (empty flows) → neutral fallback, no judge call.
    judge = FakeJudge(rankings((0, 1, 0.5)))
    scorer = _GroundScorer(FakeEmbedding(), judge, FakeRetriever())
    empty = json.dumps({"reasoning": "x", "has_information_exchange": False, "flows": []})
    out = scorer(completions=[empty], prompts=["p"], metadata_list=[meta()])
    assert out is None
    assert judge.items == []


# ---------------------------------------------------------------------------
# Wrong-book seeding: determinism + never own book
# ---------------------------------------------------------------------------
BOOKS = ["1342", "84", "1661", "2701", "1400"]


def test_seeded_wrong_book_deterministic_and_never_own():
    for own in BOOKS:
        picks = {seeded_wrong_book("chunk-42", own, BOOKS) for _ in range(5)}
        assert len(picks) == 1  # same chunk_id ⇒ same wrong book every time
        wrong = picks.pop()
        assert wrong != own  # never the own book
        assert wrong in BOOKS


def test_seeded_wrong_book_varies_by_chunk_and_single_book_none():
    # Different chunk ids generally map to different wrong books (not a hard
    # guarantee, but with 4 candidates the spread should exceed one value).
    picks = {seeded_wrong_book(f"c{i}", "1342", BOOKS) for i in range(20)}
    assert len(picks) > 1
    # A single-book universe has no valid wrong book.
    assert seeded_wrong_book("c1", "1342", ["1342"]) is None
    assert seeded_wrong_book("c1", "1342", []) is None


# ---------------------------------------------------------------------------
# R-CONTRAST: 1 − grounding inversion + seeded wrong-universe retrieval
# ---------------------------------------------------------------------------
def test_contrast_inversion_and_wrong_universe_retrieval():
    emb = FakeEmbedding()
    ret = FakeRetriever()
    # Ranks present but IGNORED by contrast; only grounding_score is read.
    judge = FakeJudge(rankings((0, 2, 0.2), (1, 1, 0.9)))
    scorer = _ContrastScorer(emb, judge, ret, BOOKS, rank_top_k=5)

    metas = [meta(source_id="1342", chunk_id="cX"), meta(source_id="1342", chunk_id="cX")]
    scores = scorer(
        completions=[extraction("a"), extraction("b")],
        prompts=["p", "p"],
        metadata_list=metas,
    )

    # r_contrast = 1 − grounding_wrong, candidate-index aligned.
    assert scores == pytest.approx([1.0 - 0.2, 1.0 - 0.9])

    # Retrieval went against the seeded WRONG book, not the own book.
    assert len(ret.calls) == 1
    wrong = seeded_wrong_book("cX", "1342", BOOKS)
    assert ret.calls[0]["contrastive_source"] == wrong
    assert wrong != "1342"


def test_contrast_seeding_stable_across_calls():
    ret = FakeRetriever()
    judge = FakeJudge(rankings((0, 1, 0.5)))
    scorer = _ContrastScorer(FakeEmbedding(), judge, ret, BOOKS)
    for _ in range(3):
        scorer(
            completions=[extraction()],
            prompts=["p"],
            metadata_list=[meta(source_id="1342", chunk_id="cSTAY")],
        )
    wrongs = {c["contrastive_source"] for c in ret.calls}
    assert len(wrongs) == 1  # same chunk_id ⇒ same wrong universe each call
    assert wrongs.pop() != "1342"


def test_contrast_judge_failure_returns_none():
    scorer = _ContrastScorer(FakeEmbedding(), FakeJudge(None), FakeRetriever(), BOOKS)
    out = scorer(
        completions=[extraction()],
        prompts=["p"],
        metadata_list=[meta(source_id="1342", chunk_id="cZ")],
    )
    assert out is None


def test_contrast_single_book_returns_none_without_judging():
    judge = FakeJudge(rankings((0, 1, 0.5)))
    scorer = _ContrastScorer(FakeEmbedding(), judge, FakeRetriever(), ["1342"])
    out = scorer(
        completions=[extraction()],
        prompts=["p"],
        metadata_list=[meta(source_id="1342", chunk_id="c1")],
    )
    assert out is None
    assert judge.items == []  # never reaches the judge


# ---------------------------------------------------------------------------
# Factory: injection path + active-aux subsetting
# ---------------------------------------------------------------------------
def test_make_aux_scorers_injection_and_subsetting():
    norm_universes = {b: [{"norm_articulation": "n"}] for b in BOOKS}
    emb, ret, judge = FakeEmbedding(), FakeRetriever(), FakeJudge(rankings((0, 1, 0.7)))

    # Only "ground" active → contrast is None.
    ground, contrast = make_aux_scorers(
        {}, {}, norm_universes, ["ground"],
        embedding_client=emb, judge_client=judge, norm_retriever=ret,
    )
    assert ground is not None and contrast is None
    out = ground(completions=[extraction()], prompts=["p"], metadata_list=[meta()])
    assert out == _rankings_to_scores(rankings((0, 1, 0.7)), 1, rank_weight=0.5)

    # Both active → both callables.
    ground2, contrast2 = make_aux_scorers(
        {}, {}, norm_universes, ["ground", "contrast"],
        embedding_client=emb, judge_client=judge, norm_retriever=ret,
    )
    assert ground2 is not None and contrast2 is not None

    # No active aux → both None (no client construction attempted).
    assert make_aux_scorers({}, {}, norm_universes, []) == (None, None)


# ---------------------------------------------------------------------------
# End-to-end: judge failure resolves to group-neutral via modular convention
# ---------------------------------------------------------------------------
def test_modular_group_neutral_on_judge_failure():
    """A judge-failed ground group scores uniform 0.5 and increments
    reward/ground/judge_failed_group_frac — the modular_reward convention the
    scorer's ``None`` return is designed to trigger."""
    ground = _GroundScorer(FakeEmbedding(), FakeJudge(None), FakeRetriever())
    # reward_core=False so the single ground auxiliary carries the whole weight.
    reward = ModularReward(auxiliaries=["ground"], reward_core=False, ground_scorer=ground)

    prompt = "USER PROMPT"
    reward.prompt_metadata = {
        prompt: {"task_type": "extract", "gold_has_exchange": True, "source_id": "1342", "chunk_id": "c1"}
    }
    comps = [extraction("a"), extraction("b")]
    scores = reward(prompts=[prompt, prompt], completions=comps)

    assert scores == [0.5, 0.5]  # group-neutral, zero advantage
    assert reward.last_metrics.get("reward/ground/judge_failed_group_frac") == 1.0
