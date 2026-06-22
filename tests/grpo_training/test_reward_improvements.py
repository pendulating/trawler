"""Tests for the June 2026 GRPO reward/optimizer improvements.

Covers the pure-logic pieces introduced by the Phase 2–5 redesign
(see wiki/changelog/2026-06-09_grpo_phase1_optimizer_revision.md):

* ``_rankings_to_scores`` — listwise rank → per-candidate reward blending
* ``CompositeRewardFunction`` gated vs additive composition
* gold-aware ``r_context`` for no-flow completions
* ``select_prompts_by_reward_std`` — variance pre-screening selection
* ``check_promotion_gates`` — trainer-state / trace gate evaluation

Network-dependent paths (judge/embedding clients, vLLM sampling) are
exercised on the cluster, not here.
"""

from __future__ import annotations

import json
import os

import pytest

from dagspaces.grpo_training.gates import check_promotion_gates
from dagspaces.grpo_training.stages.online_rground import _rankings_to_scores
from dagspaces.grpo_training.stages.prompt_screening import (
    _apply_flow_variance_filter,
    select_prompts_by_reward_std,
)
from dagspaces.grpo_training.stages.rewards import (
    CompositeRewardFunction,
    r_context,
)

WEIGHTS = [0.10, 0.05, 0.05, 0.20, 0.10, 0.50]


# ---------------------------------------------------------------------------
# _rankings_to_scores
# ---------------------------------------------------------------------------

class TestRankingsToScores:
    def test_full_ranking_orders_scores(self):
        rankings = [
            {"candidate_index": 0, "rank": 2, "grounding_score": 0.8},
            {"candidate_index": 1, "rank": 1, "grounding_score": 0.9},
            {"candidate_index": 2, "rank": 3, "grounding_score": 0.1},
        ]
        scores = _rankings_to_scores(rankings, 3, rank_weight=0.5)
        # rank 1 → rank_component 1.0; rank 3 → 0.0
        assert scores[1] == pytest.approx(0.5 * 1.0 + 0.5 * 0.9)
        assert scores[0] == pytest.approx(0.5 * 0.5 + 0.5 * 0.8)
        assert scores[2] == pytest.approx(0.5 * 0.0 + 0.5 * 0.1)
        assert scores[1] > scores[0] > scores[2]

    def test_strict_ranking_always_discriminates(self):
        # Identical grounding scores (the absolute-mode failure) must still
        # produce distinct rewards — this is the point of ranked mode.
        rankings = [
            {"candidate_index": i, "rank": i + 1, "grounding_score": 0.9}
            for i in range(4)
        ]
        scores = _rankings_to_scores(rankings, 4, rank_weight=0.5)
        assert len(set(scores)) == 4

    def test_single_candidate_uses_grounding_only(self):
        scores = _rankings_to_scores(
            [{"candidate_index": 0, "rank": 1, "grounding_score": 0.7}], 1,
        )
        assert scores == [pytest.approx(0.7)]

    def test_out_of_range_indices_ignored(self):
        rankings = [
            {"candidate_index": 5, "rank": 1, "grounding_score": 1.0},
            {"candidate_index": -1, "rank": 2, "grounding_score": 1.0},
            {"candidate_index": 0, "rank": 1, "grounding_score": 0.4},
        ]
        scores = _rankings_to_scores(rankings, 2, rank_weight=0.5)
        assert scores[0] == pytest.approx(0.5 * 1.0 + 0.5 * 0.4)
        assert scores[1] == 0.0

    def test_rank_and_grounding_clamped(self):
        rankings = [
            {"candidate_index": 0, "rank": 99, "grounding_score": 7.0},
            {"candidate_index": 1, "rank": 0, "grounding_score": -3.0},
        ]
        scores = _rankings_to_scores(rankings, 2, rank_weight=0.5)
        assert scores[0] == pytest.approx(0.5 * 0.0 + 0.5 * 1.0)
        assert scores[1] == pytest.approx(0.5 * 1.0 + 0.5 * 0.0)


# ---------------------------------------------------------------------------
# Gated composition
# ---------------------------------------------------------------------------

class TestComposition:
    def test_additive_matches_weighted_sum(self):
        fn = CompositeRewardFunction(weights=WEIGHTS, composition="additive")
        components = [1.0, 1.0, 1.0, 0.5, 0.5, 0.5]
        expected = sum(w * c for w, c in zip(WEIGHTS, components))
        assert fn._combine(components) == pytest.approx(expected)

    def test_gated_is_gate_times_disc(self):
        fn = CompositeRewardFunction(weights=WEIGHTS, composition="gated")
        components = [1.0, 1.0, 1.0, 0.5, 0.5, 0.5]
        # Perfect gating → gate=1.0; disc = weight-normalized mean = 0.5
        assert fn._combine(components) == pytest.approx(0.5)

    def test_gated_zero_gate_zeroes_reward(self):
        fn = CompositeRewardFunction(weights=WEIGHTS, composition="gated")
        assert fn._combine([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]) == 0.0

    def test_gated_saturated_gate_preserves_disc_ordering(self):
        # Two completions with identical (saturated) gates must be ordered
        # purely by their discriminative components — the additive offset
        # dilution this mode removes.
        fn = CompositeRewardFunction(weights=WEIGHTS, composition="gated")
        better = fn._combine([1.0, 1.0, 1.0, 0.2, 0.2, 0.9])
        worse = fn._combine([1.0, 1.0, 1.0, 0.2, 0.2, 0.3])
        # With gate=1.0 the gap is the r_ground difference at its
        # normalized discriminative weight.
        expected_gap = (0.9 - 0.3) * WEIGHTS[5] / sum(WEIGHTS[3:])
        assert better - worse == pytest.approx(expected_gap)

    def test_unknown_composition_rejected(self):
        with pytest.raises(ValueError):
            CompositeRewardFunction(weights=WEIGHTS, composition="multiplicative")


# ---------------------------------------------------------------------------
# Unjustified-abstention penalty (2026-06-12)
# ---------------------------------------------------------------------------

class TestAbstentionPenalty:
    """The penalty fires ONLY on abstain-when-gold-says-flow, and only when
    enabled. It must never touch correct abstention (gold False), unknown gold
    (None), or any extracting completion — otherwise it would punish the model
    for correctly declining to hallucinate a flow.
    """

    def test_disabled_by_default(self):
        fn = CompositeRewardFunction(weights=WEIGHTS, composition="gated")
        assert fn.abstention_penalty == 0.0
        # Even an abstain-on-gold-flow completion is untouched when disabled.
        assert fn._is_unjustified_abstention(True, True) is False

    def test_negative_penalty_rejected(self):
        with pytest.raises(ValueError):
            CompositeRewardFunction(weights=WEIGHTS, abstention_penalty=-0.1)

    def test_fires_only_on_abstain_with_gold_flow(self):
        fn = CompositeRewardFunction(weights=WEIGHTS, abstention_penalty=0.2)
        # The one case that should be penalized:
        assert fn._is_unjustified_abstention(True, True) is True
        # Correct abstention (gold says no flow) — never penalized:
        assert fn._is_unjustified_abstention(True, False) is False
        # Unknown gold — hedged, never penalized:
        assert fn._is_unjustified_abstention(True, None) is False
        # Extracting completions — never penalized regardless of gold:
        assert fn._is_unjustified_abstention(False, True) is False
        assert fn._is_unjustified_abstention(False, False) is False


# ---------------------------------------------------------------------------
# Gold-aware r_context for no-flow completions
# ---------------------------------------------------------------------------

NO_FLOW_COMPLETION = json.dumps({
    "reasoning": "This passage is descriptive scenery with no exchange.",
    "has_information_exchange": False,
    "flows": [],
})


class TestRContextNoFlow:
    def test_correct_no_flow_rewarded(self):
        assert r_context(NO_FLOW_COMPLETION, None, [], None,
                         gold_has_exchange=False) == pytest.approx(0.9)

    def test_false_no_flow_zeroed(self):
        assert r_context(NO_FLOW_COMPLETION, None, [], None,
                         gold_has_exchange=True) == 0.0

    def test_unknown_gold_hedged(self):
        assert r_context(NO_FLOW_COMPLETION, None, [], None,
                         gold_has_exchange=None) == pytest.approx(0.4)

    def test_parse_failure_still_zero(self):
        assert r_context("not json at all", None, [], None,
                         gold_has_exchange=False) == 0.0


# ---------------------------------------------------------------------------
# Variance pre-screening selection
# ---------------------------------------------------------------------------

class TestPrescreenSelection:
    def test_threshold_drops_degenerate_groups(self):
        rewards = {
            "tied": [0.6, 0.6, 0.6, 0.6],
            "informative": [0.2, 0.8, 0.5, 0.6],
            "near_tied": [0.600, 0.601, 0.600, 0.601],
        }
        kept, stds = select_prompts_by_reward_std(rewards, reward_std_min=0.05,
                                                  min_keep=1)
        assert kept == ["informative"]
        assert stds["tied"] == 0.0

    def test_min_keep_fallback_on_aggressive_threshold(self):
        rewards = {f"p{i}": [0.5, 0.5 + i * 0.001] for i in range(10)}
        kept, _ = select_prompts_by_reward_std(rewards, reward_std_min=0.9,
                                               min_keep=4)
        assert len(kept) == 4
        # Fallback keeps the highest-variance prompts
        assert "p9" in kept and "p0" not in kept

    def test_keeps_input_order(self):
        rewards = {
            "b": [0.0, 1.0],
            "a": [0.0, 1.0],
        }
        kept, _ = select_prompts_by_reward_std(rewards, reward_std_min=0.1,
                                               min_keep=1)
        assert kept == ["b", "a"]


class TestFlowVarianceFilter:
    """Option 3: drop prompts whose SFT samples unanimously abstain.

    extract_frac == 0 means no sample extracted a flow → no extract-vs-abstain
    contrast for GRPO and the abstention penalty cancels under centering.
    Pure-extract (1.0) and mixed (0 < f < 1) prompts are kept.
    """

    STDS = {"unanimous_abstain": 0.3, "mixed": 0.3, "pure_extract": 0.3}
    FRACS = {"unanimous_abstain": 0.0, "mixed": 0.5, "pure_extract": 1.0}

    def test_disabled_is_noop(self):
        eligible, dropped = _apply_flow_variance_filter(
            self.STDS, self.FRACS, require_flow_variance=False)
        assert dropped == 0
        assert set(eligible) == set(self.STDS)

    def test_drops_only_unanimous_abstain(self):
        eligible, dropped = _apply_flow_variance_filter(
            self.STDS, self.FRACS, require_flow_variance=True)
        assert dropped == 1
        assert set(eligible) == {"mixed", "pure_extract"}

    def test_missing_fracs_noop(self):
        # No extract fractions available (e.g. legacy cache) → cannot filter,
        # must not empty the set.
        eligible, dropped = _apply_flow_variance_filter(
            self.STDS, None, require_flow_variance=True)
        assert dropped == 0
        assert set(eligible) == set(self.STDS)

    def test_missing_key_fails_safe_to_keep(self):
        fracs = {"unanimous_abstain": 0.0}  # other keys absent
        eligible, dropped = _apply_flow_variance_filter(
            self.STDS, fracs, require_flow_variance=True)
        # Only the known-zero prompt is dropped; absent keys are kept.
        assert dropped == 1
        assert set(eligible) == {"mixed", "pure_extract"}

    def test_vignettes_exempt_from_drop(self):
        # A judgment vignette structurally has extract_frac 0 (it never emits
        # an extraction array). It must NOT be dropped — only CI-extraction
        # prompts in ci_prompt_keys are eligible. Regression for 2026-06-14,
        # when a G=8 screen wiped out all 331 vignettes (Phase 4 disabled).
        stds = {"ci_abstain": 0.3, "vignette": 0.3, "ci_mixed": 0.3}
        fracs = {"ci_abstain": 0.0, "vignette": 0.0, "ci_mixed": 0.5}
        ci_keys = {"ci_abstain", "ci_mixed"}  # vignette excluded
        eligible, dropped = _apply_flow_variance_filter(
            stds, fracs, require_flow_variance=True, ci_prompt_keys=ci_keys)
        # Only the CI unanimous-abstain prompt is dropped; the vignette stays.
        assert dropped == 1
        assert set(eligible) == {"vignette", "ci_mixed"}


# ---------------------------------------------------------------------------
# Promotion gates
# ---------------------------------------------------------------------------

def _write_checkpoint(tmp_path, rewards, frac_zero_std=0.05, kl=0.02,
                      no_flow_frac=0.5, gold_no_flow_rate=0.5,
                      eval_rewards=None):
    """Synthesize the artifacts check_promotion_gates reads."""
    ckpt = tmp_path / "checkpoint"
    step_dir = ckpt / f"checkpoint-{len(rewards) * 10}"
    step_dir.mkdir(parents=True)

    log_history = []
    for i, r in enumerate(rewards):
        entry = {"step": (i + 1) * 10, "reward": r,
                 "frac_reward_zero_std": frac_zero_std}
        if kl is not None:
            entry["kl"] = kl
        log_history.append(entry)
    for i, r in enumerate(eval_rewards or []):
        log_history.append({"step": (i + 1) * 10, "eval_reward": r})
    (step_dir / "trainer_state.json").write_text(
        json.dumps({"log_history": log_history})
    )

    n_no_flow_rows = int(round(no_flow_frac * 8))
    trace_rows = []
    for call in range(30):
        for idx in range(8):
            trace_rows.append({
                "call": call,
                "task_type": "ci_extraction",
                "is_no_flow": idx < n_no_flow_rows,
            })
    (ckpt / "reward_traces.jsonl").write_text(
        "\n".join(json.dumps(r) for r in trace_rows)
    )

    n_total = 100
    n_no_flow = int(round(gold_no_flow_rate * n_total))
    (ckpt / "training_metadata.json").write_text(json.dumps({
        "n_flow_chunks": n_total - n_no_flow,
        "n_no_flow_chunks": n_no_flow,
    }))
    return str(ckpt)


class TestPromotionGates:
    def test_healthy_run_promotes(self, tmp_path):
        ckpt = _write_checkpoint(
            tmp_path, rewards=[0.40, 0.42, 0.45, 0.47, 0.50, 0.55],
        )
        report = check_promotion_gates(ckpt)
        assert report["promote"] is True
        assert all(g["status"] == "pass" for g in report["gates"].values())

    def test_flat_reward_fails_trend_gate(self, tmp_path):
        # The May 2026 production curve: flat-to-declining.
        ckpt = _write_checkpoint(
            tmp_path,
            rewards=[0.458, 0.405, 0.450, 0.399, 0.423, 0.419, 0.403, 0.393, 0.405],
            kl=None,  # beta=0 → no KL logged
        )
        report = check_promotion_gates(ckpt)
        assert report["gates"]["reward_trend"]["status"] == "fail"
        assert report["gates"]["reward_trend"]["source"] == "reward"
        assert report["gates"]["kl_bounded"]["status"] == "skipped"
        assert report["promote"] is False

    def test_trend_gate_prefers_eval_reward(self, tmp_path):
        # 2026-06-09 review, S3: a rising training-batch reward can be
        # reward hacking; the held-out dev curve is authoritative when
        # logged. Train rises but eval is flat → fail.
        ckpt = _write_checkpoint(
            tmp_path,
            rewards=[0.40, 0.42, 0.45, 0.47, 0.50, 0.55],
            eval_rewards=[0.45, 0.44, 0.46, 0.45, 0.44, 0.45],
        )
        report = check_promotion_gates(ckpt)
        assert report["gates"]["reward_trend"]["source"] == "eval_reward"
        assert report["gates"]["reward_trend"]["status"] == "fail"

    def test_trend_gate_eval_reward_rising_promotes(self, tmp_path):
        ckpt = _write_checkpoint(
            tmp_path,
            rewards=[0.40, 0.41, 0.40, 0.41, 0.40, 0.41],
            eval_rewards=[0.40, 0.43, 0.46, 0.49, 0.52, 0.55],
        )
        report = check_promotion_gates(ckpt)
        assert report["gates"]["reward_trend"]["source"] == "eval_reward"
        assert report["gates"]["reward_trend"]["status"] == "pass"

    def test_trend_gate_falls_back_with_sparse_eval_logs(self, tmp_path):
        ckpt = _write_checkpoint(
            tmp_path,
            rewards=[0.40, 0.42, 0.45, 0.47, 0.50, 0.55],
            eval_rewards=[0.45, 0.44],  # < 3 points → train fallback
        )
        report = check_promotion_gates(ckpt)
        assert report["gates"]["reward_trend"]["source"] == "reward"
        assert report["gates"]["reward_trend"]["status"] == "pass"

    def test_tied_groups_fail_zero_std_gate(self, tmp_path):
        ckpt = _write_checkpoint(
            tmp_path, rewards=[0.4, 0.45, 0.5, 0.55], frac_zero_std=0.4,
        )
        report = check_promotion_gates(ckpt)
        assert report["gates"]["zero_std_groups"]["status"] == "fail"
        assert report["promote"] is False

    def test_no_flow_collapse_fails_gate(self, tmp_path):
        # 75% no-flow completions against a 50% gold base rate (May sweep
        # observed 77.6%).
        ckpt = _write_checkpoint(
            tmp_path, rewards=[0.4, 0.45, 0.5, 0.55],
            no_flow_frac=0.75, gold_no_flow_rate=0.5,
        )
        report = check_promotion_gates(ckpt)
        assert report["gates"]["no_flow_rate"]["status"] == "fail"
        assert report["promote"] is False

    def test_kl_blowup_fails_gate(self, tmp_path):
        ckpt = _write_checkpoint(
            tmp_path, rewards=[0.4, 0.45, 0.5, 0.55], kl=2.5,
        )
        report = check_promotion_gates(ckpt)
        assert report["gates"]["kl_bounded"]["status"] == "fail"
        assert report["promote"] is False

    def test_missing_trainer_state_blocks_promotion(self, tmp_path):
        report = check_promotion_gates(str(tmp_path / "nonexistent"))
        assert report["promote"] is False
        assert "error" in report


# ---------------------------------------------------------------------------
# Ranked OnlineRGround end-to-end (faked clients)
# ---------------------------------------------------------------------------

FLOW_COMPLETION = json.dumps({
    "reasoning": "Alice tells Bob a secret about her health.",
    "has_information_exchange": True,
    "flows": [{
        "sender": "Alice", "recipient": "Bob",
        "information_type": "health secret",
        "transmission_principle": "in confidence",
        "subject": "Alice", "context": "friendship",
        "appropriateness": "appropriate",
        "norms_invoked": ["confidences between friends are kept"],
        "confidence": 8,
    }],
})


class _FakeEmbeddingClient:
    def encode_batch(self, texts):
        import numpy as np
        return np.full((len(texts), 4), 0.5)


class _FakeRetriever:
    def __init__(self):
        self.calls = []

    def retrieve(self, emb, source_id, contrastive_source=None,
                 return_scores=False, top_k=None):
        self.calls.append({"source_id": source_id,
                           "contrastive_source": contrastive_source,
                           "top_k": top_k})
        norms = json.dumps([{"norm_articulation": "keep confidences"}])
        return (norms, [0.9]) if return_scores else norms


class _FakeRankingJudge:
    """Returns rank 1 / grounding 0.9 for candidate 0, descending after;
    wrong-universe calls (detected by call order) return low grounding."""

    def __init__(self, fail=False, fail_correct_pass_only=False):
        self.fail = fail
        self.fail_correct_pass_only = fail_correct_pass_only
        self.batches = []

    def judge_ranking_batch(self, items, system_prompt, prompt_template,
                            json_schema=None):
        self.batches.append(items)
        if self.fail or (self.fail_correct_pass_only and len(self.batches) == 1):
            return [None] * len(items)
        is_wrong_pass = len(self.batches) > 1
        results = []
        for item in items:
            n = item["n_candidates"]
            results.append([
                {"candidate_index": i, "rank": i + 1,
                 "grounding_score": (0.1 * (i + 1)) if is_wrong_pass
                 else (0.9 - 0.1 * i)}
                for i in range(n)
            ])
        return results


class _FakeRetrieverWithForce:
    """Like _FakeRetriever but the retrieved norm carries a normative_force,
    so the deontic appropriateness blend has a direction to score against."""

    def __init__(self, force="prohibited"):
        self.force = force

    def retrieve(self, emb, source_id, contrastive_source=None,
                 return_scores=False, top_k=None):
        norms = json.dumps([
            {"norm_articulation": "keep confidences", "normative_force": self.force}
        ])
        return (norms, [0.9]) if return_scores else norms


def _make_ranked_rground(judge, contrastive_lambda=0.5, app_weight=0.0, retriever=None):
    from dagspaces.grpo_training.stages.online_rground import OnlineRGround
    return OnlineRGround(
        embedding_client=_FakeEmbeddingClient(),
        judge_client=judge,
        norm_retriever=retriever or _FakeRetriever(),
        all_source_ids=["book_a", "book_b"],
        contrastive_lambda=contrastive_lambda,
        scoring_mode="ranked",
        ranking_system_prompt="sys",
        ranking_prompt_template="{{chunk_text}} {{norm_universe_json}} "
                                "{{candidates_block}} {{n_candidates}}",
        rank_top_k=5,
        rank_weight=0.5,
        app_weight=app_weight,
    )


class TestRankedOnlineRGround:
    META = {"source_id": "book_a", "prompt_id": "p1",
            "chunk_text": "Alice whispered her secret to Bob."}

    def test_group_scoring_with_contrastive(self):
        judge = _FakeRankingJudge()
        rg = _make_ranked_rground(judge, contrastive_lambda=0.5)
        prompts = ["prompt-1"] * 3
        completions = [FLOW_COMPLETION, NO_FLOW_COMPLETION, "garbage{{{"]
        scores = rg(completions=completions, prompts=prompts,
                    metadata_list=[self.META] * 3)

        # Candidate 0 (flows): rank 1, grounding 0.9, wrong grounding 0.1
        assert scores[0] == pytest.approx(0.5 * 1.0 + 0.5 * 0.9 - 0.5 * 0.1)
        # Candidate 1 (no-flow declaration is a judged candidate): rank 2,
        # grounding 0.8, wrong grounding 0.2
        assert scores[1] == pytest.approx(0.5 * 0.0 + 0.5 * 0.8 - 0.5 * 0.2)
        # Parse failure: never judged, scores 0
        assert scores[2] == 0.0
        # Two listwise passes (correct + wrong), one item each, 2 candidates
        assert len(judge.batches) == 2
        assert judge.batches[0][0]["n_candidates"] == 2
        # Diagnostics attached for judged candidates
        assert rg.last_diagnostics[0][0]["rank"] == 1
        assert rg.last_diagnostics[2] == []

    def test_lambda_zero_skips_wrong_universe_pass(self):
        judge = _FakeRankingJudge()
        rg = _make_ranked_rground(judge, contrastive_lambda=0.0)
        scores = rg(completions=[FLOW_COMPLETION, NO_FLOW_COMPLETION],
                    prompts=["p"] * 2, metadata_list=[self.META] * 2)
        assert len(judge.batches) == 1  # no contrastive pass
        assert scores[0] == pytest.approx(0.5 * 1.0 + 0.5 * 0.9)

    def test_judge_failure_yields_neutral_tie(self):
        judge = _FakeRankingJudge(fail=True)
        rg = _make_ranked_rground(judge, contrastive_lambda=0.5)
        scores = rg(completions=[FLOW_COMPLETION, NO_FLOW_COMPLETION],
                    prompts=["p"] * 2, metadata_list=[self.META] * 2)
        # Neutral identical scores → zero advantage, not a spurious gradient
        assert scores == [0.5, 0.5]

    def test_correct_judge_failure_neutral_even_if_wrong_pass_succeeds(self):
        # Regression (2026-06-09 review, F2): when only the correct-side
        # judge failed, surviving wrong-universe grounding scores used to be
        # subtracted from the neutral 0.5, varying within the group — a
        # spurious gradient driven entirely by the wrong-universe judge.
        judge = _FakeRankingJudge(fail_correct_pass_only=True)
        rg = _make_ranked_rground(judge, contrastive_lambda=0.5)
        scores = rg(completions=[FLOW_COMPLETION, NO_FLOW_COMPLETION],
                    prompts=["p"] * 2, metadata_list=[self.META] * 2)
        assert len(judge.batches) == 2  # wrong-universe pass ran and succeeded
        assert scores == [0.5, 0.5]

    def test_health_metrics_populated(self):
        # 2026-06-09 logging review, item B: every reward call records a
        # bounded rground/* health snapshot (pushed to W&B when a run is
        # active; always kept on last_health).
        judge = _FakeRankingJudge()
        rg = _make_ranked_rground(judge, contrastive_lambda=0.5)
        rg(completions=[FLOW_COMPLETION, NO_FLOW_COMPLETION, "garbage{{{"],
           prompts=["p"] * 3, metadata_list=[self.META] * 3)
        h = rg.last_health
        assert h["rground/n_completions"] == 3
        assert h["rground/parse_fail_frac"] == pytest.approx(1 / 3, abs=1e-4)
        assert h["rground/no_flow_frac"] == pytest.approx(1 / 3, abs=1e-4)
        assert h["rground/judge_failed_group_frac"] == 0.0
        assert h["rground/consecutive_zero_batches"] == 0.0
        assert 0.0 <= h["rground/mean_score"] <= 1.0

    def test_health_reports_judge_failure(self):
        judge = _FakeRankingJudge(fail=True)
        rg = _make_ranked_rground(judge, contrastive_lambda=0.5)
        rg(completions=[FLOW_COMPLETION, NO_FLOW_COMPLETION],
           prompts=["p"] * 2, metadata_list=[self.META] * 2)
        assert rg.last_health["rground/judge_failed_group_frac"] == 1.0

    def test_separate_prompts_form_separate_groups(self):
        judge = _FakeRankingJudge()
        rg = _make_ranked_rground(judge, contrastive_lambda=0.0)
        rg(completions=[FLOW_COMPLETION] * 4,
           prompts=["p1", "p1", "p2", "p2"],
           metadata_list=[self.META] * 4)
        # One correct-universe batch with two single groups of 2 candidates
        assert len(judge.batches) == 1
        assert [it["n_candidates"] for it in judge.batches[0]] == [2, 2]

    def test_zero_embeddings_route_group_to_neutral_without_judging(self):
        # Embedding-server zero fallback → degenerate group embedding →
        # uniform neutral 0.5 (zero advantage), and the judge is never
        # asked to rank against garbage retrieval.
        import numpy as np

        class _ZeroEmbeddingClient:
            def encode_batch(self, texts):
                return np.zeros((len(texts), 4), dtype=np.float32)

        judge = _FakeRankingJudge()
        rg = _make_ranked_rground(judge, contrastive_lambda=0.5)
        rg.embedding_client = _ZeroEmbeddingClient()
        scores = rg(completions=[FLOW_COMPLETION, FLOW_COMPLETION],
                    prompts=["p"] * 2, metadata_list=[self.META] * 2)
        assert scores == [0.5, 0.5]
        assert judge.batches == []
        assert rg.last_health["rground/judge_failed_group_frac"] == 1.0


def _flow_completion(appropriateness):
    return json.dumps({
        "reasoning": "Alice tells Bob a secret about her health.",
        "has_information_exchange": True,
        "flows": [{
            "sender": "Alice", "recipient": "Bob",
            "information_type": "health secret",
            "transmission_principle": "in confidence", "subject": "Alice",
            "context": "friendship", "appropriateness": appropriateness,
            "norms_invoked": ["confidences between friends are kept"], "confidence": 8,
        }],
    })


class TestAppropriatenessBlend:
    """rground_app_weight blends the deterministic deontic appropriateness check
    (norm normative_force → expected appropriateness) into the ranked R_ground."""

    META = {"source_id": "book_a", "prompt_id": "p1",
            "chunk_text": "Alice whispered her secret to Bob."}

    def test_appropriateness_can_flip_grounding_order(self):
        # Norm force = prohibited → expected "inappropriate". The judge ranks
        # candidate 0 (appropriate) higher on grounding, but the appropriateness
        # blend rewards the candidate whose verdict matches the prohibiting norm.
        judge = _FakeRankingJudge()
        rg = _make_ranked_rground(
            judge, contrastive_lambda=0.5, app_weight=0.5,
            retriever=_FakeRetrieverWithForce("prohibited"))
        scores = rg(
            completions=[_flow_completion("appropriate"), _flow_completion("inappropriate")],
            prompts=["p"] * 2, metadata_list=[self.META] * 2)
        # base0=0.90 (rank1/grnd0.9 - 0.5*0.1); base1=0.30 (rank2/grnd0.8 - 0.5*0.2)
        # app: cand0 appropriate vs prohibited -> 0.0; cand1 inappropriate -> 1.0
        assert scores[0] == pytest.approx(0.5 * 0.90 + 0.5 * 0.0)
        assert scores[1] == pytest.approx(0.5 * 0.30 + 0.5 * 1.0)
        assert scores[1] > scores[0]  # context-relative verdict wins
        d0 = rg.last_diagnostics[0][0]
        assert d0["norm_force"] == "prohibited"
        assert d0["app_consistency"] == pytest.approx(0.0)

    def test_app_weight_zero_is_legacy_grounding_only(self):
        # With app_weight=0 the force retriever must not change anything.
        judge = _FakeRankingJudge()
        rg = _make_ranked_rground(
            judge, contrastive_lambda=0.5, app_weight=0.0,
            retriever=_FakeRetrieverWithForce("prohibited"))
        scores = rg(
            completions=[_flow_completion("appropriate"), _flow_completion("inappropriate")],
            prompts=["p"] * 2, metadata_list=[self.META] * 2)
        assert scores[0] == pytest.approx(0.5 * 1.0 + 0.5 * 0.9 - 0.5 * 0.1)
        assert rg.last_diagnostics[0][0]["app_consistency"] is None

    def test_invalid_app_weight_rejected(self):
        from dagspaces.grpo_training.stages.online_rground import OnlineRGround
        with pytest.raises(ValueError):
            OnlineRGround(
                embedding_client=_FakeEmbeddingClient(), judge_client=_FakeRankingJudge(),
                norm_retriever=_FakeRetriever(), scoring_mode="ranked",
                ranking_prompt_template="x", app_weight=1.5)
