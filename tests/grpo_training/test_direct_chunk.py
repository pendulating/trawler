"""Chunk-denominator R-DIRECT (R2 fix, 2026-07-28).

The m1 wave's per-completion macro-EM collapsed to plain accuracy on the
50.7% of scored completions that carried a single gold class — the policy
controlled its own denominator twice (labels AND flow selection) and held
reward 0.73 at balanced accuracy 0.56. The fix scores every completion
against the CHUNK's teacher flows: a fixed denominator shared by all G
siblings (68.3% of m1 chunks are mixed-gold at chunk level), with unmatched
teacher flows scoring 0.0 to their class so omitting a governed flow costs
recall.

These tests exercise the pure paths — ``match_flows`` and
``_score_direct_chunk`` with an injected fake index — no embedding server.
"""
from __future__ import annotations

import numpy as np

from dagspaces.grpo_training.stages.modular_reward import (
    ModularReward,
    _MetricAccumulator,
    match_flows,
)


# ---------------------------------------------------------------------------
# match_flows (pure)
# ---------------------------------------------------------------------------
class TestMatchFlows:
    def test_greedy_one_to_one_highest_first(self):
        sims = [[0.9, 0.8],
                [0.85, 0.2]]
        got = match_flows(sims, tau=0.0)
        # (0,0)=0.9 claims both; (1,0) blocked -> (1,1)=0.2
        assert got == [(0, 0, 0.9), (1, 1, 0.2)] or [
            (t, p) for t, p, _ in got] == [(0, 0), (1, 1)]

    def test_threshold_cuts_low_pairs(self):
        sims = [[0.9, 0.1],
                [0.2, 0.3]]
        got = match_flows(sims, tau=0.5)
        assert [(t, p) for t, p, _ in got] == [(0, 0)]

    def test_empty_inputs(self):
        assert match_flows(np.zeros((0, 3)), tau=0.5) == []
        assert match_flows(np.zeros((3, 0)), tau=0.5) == []

    def test_more_policy_than_teacher(self):
        sims = [[0.6, 0.9, 0.7]]
        got = match_flows(sims, tau=0.5)
        assert [(t, p) for t, p, _ in got] == [(0, 1)]


# ---------------------------------------------------------------------------
# _score_direct_chunk with a fake index
# ---------------------------------------------------------------------------
class _FakeChunkGold:
    """Index with one chunk: two teacher flows, gold [appropriate,
    inappropriate], orthogonal unit embeddings. Policy flows embed by a
    lookup on their `information_type` so tests steer matching exactly."""

    _E = {"good_flow": np.array([1.0, 0.0], dtype=np.float32),
          "bad_flow": np.array([0.0, 1.0], dtype=np.float32),
          "unrelated": np.array([-1.0, 0.0], dtype=np.float32)}

    def __init__(self):
        self.index = {("135", "c1"): {
            "golds": ["appropriate", "inappropriate"],
            "emb": np.stack([self._E["good_flow"], self._E["bad_flow"]]),
            "texts": ["good_flow", "bad_flow"],
        }}

    def get(self, source_id, chunk_id):
        return self.index.get((str(source_id), str(chunk_id)))

    def embed_flows(self, flows):
        return np.stack([
            self._E.get(str(f.get("information_type")), self._E["unrelated"])
            for f in flows
        ])


def _reward(**kw):
    kw.setdefault("core_mode", "direct")
    kw.setdefault("direct_chunk_gold", _FakeChunkGold())
    kw.setdefault("direct_match_threshold", 0.5)
    kw.setdefault("direct_gold_fn", lambda flow, sid: (None, 0.0))
    return ModularReward(reward_core=True, **kw)


def _flow(itype, label):
    return {"information_type": itype, "appropriateness": label}


_META = {"source_id": "135", "chunk_id": "c1"}


def _score(r, flows):
    return r._score_direct_chunk(flows, dict(_META), _MetricAccumulator(), 0)


class TestChunkDenominator:
    def test_fully_correct_scores_one(self):
        r = _reward()
        s = _score(r, [_flow("good_flow", "appropriate"),
                       _flow("bad_flow", "inappropriate")])
        assert s == 1.0

    def test_blanket_appropriate_on_mixed_chunk_scores_half(self):
        # The collapse regression: blanket labelling must price at 0.5 on a
        # mixed chunk EVEN IF the policy extracts both flows.
        r = _reward()
        s = _score(r, [_flow("good_flow", "appropriate"),
                       _flow("bad_flow", "appropriate")])
        assert s == 0.5

    def test_omitting_minority_flow_costs_its_class(self):
        # Selection gaming: extract only the safe flow -> inappropriate class
        # scores 0.0 (miss), macro = 0.5. Same as blanket — no escape route.
        r = _reward()
        s = _score(r, [_flow("good_flow", "appropriate")])
        assert s == 0.5

    def test_empty_extraction_scores_zero(self):
        r = _reward()
        assert _score(r, []) == 0.0

    def test_hedge_on_matched_flow_scores_zero_for_that_class(self):
        r = _reward()
        s = _score(r, [_flow("good_flow", "appropriate"),
                       _flow("bad_flow", "ambiguous")])
        assert s == 0.5

    def test_denominator_identical_across_siblings(self):
        # The whole point: two completions with different extractions face
        # the same two-class denominator.
        r = _reward()
        blanket = _score(r, [_flow("good_flow", "appropriate")])
        catcher = _score(r, [_flow("good_flow", "appropriate"),
                             _flow("bad_flow", "inappropriate")])
        assert (blanket, catcher) == (0.5, 1.0)

    def test_spurious_policy_flow_not_scored_but_counted(self):
        r = _reward()
        acc = _MetricAccumulator()
        s = r._score_direct_chunk(
            [_flow("good_flow", "appropriate"),
             _flow("bad_flow", "inappropriate"),
             _flow("unrelated", "inappropriate")],
            dict(_META), acc, 0)
        assert s == 1.0  # the unrelated flow neither helps nor hurts the score
        assert acc.direct_spurious == 1

    def test_missing_chunk_falls_back_to_per_flow_path(self):
        gold_calls = []

        def gold_fn(flow, sid):
            gold_calls.append(flow)
            return "appropriate", 0.9

        r = _reward(direct_gold_fn=gold_fn)
        s = r._score_direct_chunk(
            [_flow("good_flow", "appropriate")],
            {"source_id": "999", "chunk_id": "absent"},
            _MetricAccumulator(), 0)
        assert s == 1.0 and len(gold_calls) == 1

    def test_metrics_split_label_vs_recall_semantics(self):
        # Audit 2026-07-28: by_class/balanced_accuracy are LABEL-only
        # (matched flows), so they keep m1-comparable meaning; recall
        # metrics carry the misses; the REWARD itself still prices recall.
        r = _reward()
        acc = _MetricAccumulator()
        r._score_direct_chunk([_flow("good_flow", "appropriate")],
                              dict(_META), acc, 0)
        out = acc.build(r)
        assert out["reward/direct/miss_frac"] == 0.5
        assert out["diag/match_sim"] == 1.0
        # Label semantics: the one matched flow was labelled correctly.
        assert out["reward/direct/balanced_accuracy"] == 1.0
        # Recall semantics: appropriate 1/1, inappropriate 0/1 (missed).
        assert out["reward/direct/recall_by_class/appropriate"] == 1.0
        assert out["reward/direct/recall_by_class/inappropriate"] == 0.0
        assert out["reward/direct/balanced_recall"] == 0.5
        assert out["reward/direct/spurious_flow_frac"] == 0.0
        assert out["reward/direct/embed_failed_frac"] == 0.0
        # Constant-0 masquerade guard: the chunk path never emits the
        # per-flow path's unscored_flow_frac.
        assert "reward/direct/unscored_flow_frac" not in out

    def test_zero_embedding_rows_are_group_neutral(self):
        # Audit 2026-07-28 (R2-M2): EmbeddingClient degrades to zero vectors
        # on transient faults — that must be group-neutral, never a 0.0.
        class ZeroGold(_FakeChunkGold):
            def embed_flows(self, flows):
                return np.zeros((len(flows), 2), dtype=np.float32)

        r = _reward(direct_chunk_gold=ZeroGold())
        acc = _MetricAccumulator()
        s = r._score_direct_chunk([_flow("good_flow", "appropriate")],
                                  dict(_META), acc, 0)
        assert s is None
        assert acc.direct_embed_failed == 1

    def test_nan_similarity_never_matches(self):
        got = match_flows(np.array([[np.nan, 0.9], [0.6, 0.4]]), tau=0.5)
        assert [(t, p) for t, p, _ in got] == [(0, 1), (1, 0)]

    def test_trace_carries_per_flow_gold_pred(self):
        # R5: traces must make discrimination recomputable from disk.
        r = _reward()
        r._score_direct_chunk(
            [_flow("good_flow", "appropriate")], dict(_META),
            _MetricAccumulator(), 7)
        io = r._direct_io[7]
        assert io["direct_flows"] == [
            {"gold": "appropriate", "pred": "appropriate", "sim": 1.0},
            {"gold": "inappropriate", "pred": None, "sim": None},
        ]

    def test_embedding_failure_returns_none_for_group_neutral(self):
        class BrokenGold(_FakeChunkGold):
            def embed_flows(self, flows):
                raise RuntimeError("server down")

        r = _reward(direct_chunk_gold=BrokenGold())
        s = _score(r, [_flow("good_flow", "appropriate")])
        assert s is None


# ---------------------------------------------------------------------------
# majority_gold (k=3 gold, aux_scorers)
# ---------------------------------------------------------------------------
class TestMajorityGold:
    def _n(self, force, pol="performing"):
        return {"normative_force": force, "act_polarity": pol,
                "governs_info_flow": True}

    def test_majority_overrules_top1(self):
        from dagspaces.grpo_training.stages.aux_scorers import majority_gold
        norms = [self._n("prohibited"), self._n("permitted"),
                 self._n("recommended")]
        assert majority_gold(norms, k=3) == "appropriate"

    def test_k1_preserves_top1_semantics(self):
        from dagspaces.grpo_training.stages.aux_scorers import majority_gold
        norms = [self._n("prohibited"), self._n("permitted"),
                 self._n("recommended")]
        assert majority_gold(norms, k=1) == "inappropriate"

    def test_tie_falls_back_to_top1(self):
        from dagspaces.grpo_training.stages.aux_scorers import majority_gold
        norms = [self._n("prohibited"), self._n("permitted")]
        assert majority_gold(norms, k=2) == "inappropriate"

    def test_refraining_polarity_inverts_votes(self):
        from dagspaces.grpo_training.stages.aux_scorers import majority_gold
        norms = [self._n("obligatory", pol="refraining")]
        assert majority_gold(norms, k=3) == "inappropriate"

    def test_all_unrecognised_forces_unscored(self):
        from dagspaces.grpo_training.stages.aux_scorers import majority_gold
        norms = [self._n("mysterious"), self._n("")]
        assert majority_gold(norms, k=3) is None


# ---------------------------------------------------------------------------
# Factory: retired-core surfaces stay dead in direct mode (2026-07-28)
# ---------------------------------------------------------------------------
class TestFactoryAnswererGating:
    def test_direct_mode_builds_no_answerer(self):
        # The m1 wave wired the frozen answerer client into every direct-mode
        # cell despite the core never calling it. The factory now builds it
        # only for the retired core_mode="outcome".
        from dagspaces.grpo_training.stages.modular_reward import (
            make_modular_reward_from_cfg,
        )

        r = make_modular_reward_from_cfg(
            cfg=None,
            grpo_cfg={"reward_core": True, "core_mode": "direct",
                      "reward_auxiliaries": []},
            norm_universes={},
        )
        assert r.answerer is None
        assert r.core_mode == "direct"
