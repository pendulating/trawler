"""Prescreen cache invalidation (2026-06-09 review, F4).

The reward signature must capture every knob that changes sampled
completions or their scores; before the fix, rank_top_k / rank_weight were
absent, so a sweep cell changing only retrieval depth silently reused a
stale screening result.
"""

from dagspaces.grpo_training.stages.prompt_screening import (
    _cache_key,
    _reward_signature,
)


class _FakeJudgeClient:
    def __init__(self, model_name="qwen3.6-27b"):
        self.model_name = model_name


class _FakeOnlineRGround:
    def __init__(self, **kw):
        self.contrastive_lambda = kw.get("contrastive_lambda", 1.0)
        self.scoring_mode = kw.get("scoring_mode", "ranked")
        self.rank_top_k = kw.get("rank_top_k", 5)
        self.rank_weight = kw.get("rank_weight", 0.5)
        self.app_floor_prohibit = kw.get("app_floor_prohibit", None)
        self.app_hedge_prohibit = kw.get("app_hedge_prohibit", None)
        self.judge_client = _FakeJudgeClient(kw.get("judge_model", "qwen3.6-27b"))


class _FakeRewardFn:
    def __init__(self, **kw):
        self.weights = [0.10, 0.05, 0.05, 0.20, 0.10, 0.50]
        self.composition = "gated"
        self.no_flow_scoring = kw.get("no_flow_scoring", "independent")
        self.judgment_weights = kw.get("judgment_weights", [0.5, 0.25, 0.25])
        self.confidence_fallthrough = kw.get("confidence_fallthrough", False)
        self.online_rground = _FakeOnlineRGround(**kw)


def _key(**kw):
    sig = _reward_signature(_FakeRewardFn(**kw), temperature=1.0, max_tokens=3072)
    return _cache_key("sft-ckpt", ["p1", "p2"], 8, sig)


class TestRewardSignatureInvalidation:
    def test_identical_setup_hits_cache(self):
        assert _key() == _key()

    def test_rank_top_k_invalidates(self):
        assert _key(rank_top_k=5) != _key(rank_top_k=3)

    def test_rank_weight_invalidates(self):
        assert _key(rank_weight=0.5) != _key(rank_weight=0.7)

    def test_lambda_invalidates(self):
        assert _key(contrastive_lambda=1.0) != _key(contrastive_lambda=0.5)

    def test_scoring_mode_invalidates(self):
        assert _key(scoring_mode="ranked") != _key(scoring_mode="absolute")

    def test_judge_model_invalidates(self):
        assert _key(judge_model="qwen3.6-27b") != _key(judge_model="gpt-4o")

    def test_floor_prohibit_invalidates(self):
        assert _key(app_floor_prohibit=0.1) != _key(app_floor_prohibit=None)

    def test_hedge_prohibit_invalidates(self):
        # v12a: a cell turning on the cost-sensitive hedge tier must never
        # reuse a screen computed under v10 hedge economics.
        assert _key(app_hedge_prohibit=0.5) != _key(app_hedge_prohibit=None)
        assert _key(app_hedge_prohibit=0.5) != _key(app_hedge_prohibit=0.6)

    def test_no_flow_scoring_invalidates(self):
        # "independent" vs "flat" rewrites all five components for no-flow
        # completions; a cell changing it must not reuse the prior screen.
        assert _key(no_flow_scoring="independent") != _key(no_flow_scoring="flat")

    def test_judgment_weights_invalidates(self):
        # Rescales every vignette reward (and thus its group std).
        assert _key(judgment_weights=[0.5, 0.25, 0.25]) != _key(judgment_weights=[0.34, 0.33, 0.33])

    def test_confidence_fallthrough_invalidates(self):
        # r_uncert facet-3 fix: keeper (False) vs corrected (True) change the
        # composite reward value, so the prescreen must be recomputed.
        assert _key(confidence_fallthrough=False) != _key(confidence_fallthrough=True)

    def test_offline_reward_fn_without_rground_still_works(self):
        class _Offline:
            weights = [0.2] * 5
            composition = "additive"

        sig = _reward_signature(_Offline(), temperature=1.0, max_tokens=3072)
        assert _cache_key("ckpt", ["p"], 8, sig)
