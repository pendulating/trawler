"""Keeper-freeze regression guard (wiki/grpo_redesign/migration.md §"Test plan").

The v9-ckpt100 keeper must stay byte-reproducible until the camera-ready ships.
The m-series is *additive* code (migration.md §"The parallel-stack rule"): the
frozen surfaces — `training/grpo/online_rground_external.yaml`, the
`directional` composition path in `rewards.py`, and `online_rground.py` /
`deontic.py`'s clamp/multiplier machinery — are never edited, refactored, or
"cleaned up in passing." Deletion happens once, after the keeper is obsolete,
as its own commit.

This test FAILS LOUDLY if anyone:
  1. changes a training-relevant value in the keeper yaml (the frozen snapshot
     below is captured key-by-key, so the failure names the exact drifted key),
     or
  2. removes/renames the directional composition path or the appropriateness
     multiplier machinery the keeper's reward depends on.

The snapshot was captured 2026-07-24 from `online_rground_external.yaml` at the
v9-ckpt100 keeper commit. A legitimate keeper change (should be never, pre
camera-ready) updates this snapshot in the same commit *with justification*;
silently editing the snapshot to make CI green defeats the guard's purpose.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import dagspaces.grpo_training

CONF_DIR = Path(dagspaces.grpo_training.__file__).parent / "conf"

# ── The frozen keeper snapshot (training-relevant resolved values) ───────────
# Server URLs (env-interpolated) and cache_path are deliberately excluded — they
# are deployment-specific, not part of the reward/optimizer definition. Every
# value here is load-bearing for the v9-ckpt100 result.
FROZEN: dict[str, object] = {
    # ── optimizer preset (optimizer.md §"The preset") ──
    "seed": 42,
    "num_generations": 8,
    "learning_rate": 2e-05,
    "lr_scheduler_type": "cosine_with_min_lr",
    "min_lr_rate": 0.3,
    "warmup_ratio": 0.1,
    "per_device_batch_size": 1,
    "gradient_accumulation_steps": 32,
    "num_epochs": 3,
    "max_completion_length": 3072,
    "gradient_checkpointing": True,
    "bf16": True,
    "logging_steps": 10,
    "save_strategy": "steps",
    "save_steps": 50,
    "beta": 0.02,
    "epsilon_high": 0.28,
    "num_iterations": 1,
    "vllm_importance_sampling_mode": "token_truncate",
    "scale_rewards": "none",
    "mask_truncated_completions": True,
    # ── vLLM (colocate, 1-GPU cell economics) ──
    "use_vllm": True,
    "vllm_mode": "colocate",
    "vllm_tensor_parallel_size": 1,
    "vllm_gpu_memory_utilization": 0.45,
    "vllm_enable_sleep_mode": True,
    "vllm_max_model_length": 16384,
    # ── reward shape — the v9→v12a lineage that DEFINES the keeper ──
    "online_rground": True,
    "enable_thinking_grpo": False,
    "contrastive_ratio": 0.0,
    "contrastive_lambda": 1.0,
    "no_flow_scoring": "independent",
    "confidence_fallthrough": False,
    "rground_scoring": "ranked",
    "rank_top_k": 5,
    "rank_weight": 0.5,
    "judge_max_workers": 16,
    "rground_app_weight": 0.3,
    "rground_app_mode": "multiplicative",
    "rground_app_floor": 0.4,
    "rground_app_floor_prohibit": 0.1,
    "rground_app_hedge_prohibit": 0.5,
    # The migration's headline frozen surface: the directional composition path.
    "reward_composition": "directional",
    "abstention_penalty": 0.0,
    "dev_fraction": 0.0,
    "eval_steps": 50,
    "vignette_ratio": 0.3,
    "context_embedding_model": "all-MiniLM-L6-v2",
}

FROZEN_LISTS: dict[str, list] = {
    "reward_weights": [0.10, 0.05, 0.05, 0.20, 0.10, 0.50],
    "judgment_reward_weights": [0.50, 0.25, 0.25],
}

# Prescreen sub-block (cache_path excluded — env-interpolated deployment knob).
FROZEN_PRESCREEN: dict[str, object] = {
    "enabled": True,
    "num_samples": 8,
    "reward_std_min": 0.05,
    "min_keep": 8,
    "temperature": 1.0,
    "require_flow_variance": True,
}


@pytest.fixture(scope="module")
def keeper_grpo():
    """Compose the keeper config exactly as the CLI would."""
    # The keeper yaml interpolates two server URLs with no default; set dummies
    # so composition never fails. We never assert on them.
    import os

    os.environ.setdefault("EMBEDDING_SERVER_URL", "http://keeper-guard.invalid")
    os.environ.setdefault("JUDGE_SERVER_URL", "http://keeper-guard.invalid")
    os.environ.setdefault("GRPO_PRESCREEN_CACHE", "")
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=["training/grpo=online_rground_external"],
        )
    return cfg.training.grpo


class TestKeeperFrozenValues:
    """Every training-relevant scalar in the keeper yaml is pinned."""

    @pytest.mark.parametrize("key,expected", sorted(FROZEN.items()))
    def test_scalar_frozen(self, keeper_grpo, key, expected):
        actual = OmegaConf.select(keeper_grpo, key)
        assert actual == expected, (
            f"KEEPER DRIFT: online_rground_external.yaml key '{key}' is now "
            f"{actual!r}, frozen snapshot expects {expected!r}. The v9-ckpt100 "
            f"keeper must stay byte-reproducible until camera-ready "
            f"(migration.md §parallel-stack rule). If this change is "
            f"intentional, update FROZEN in this test IN THE SAME COMMIT with "
            f"justification — do not silently edit the snapshot to go green."
        )

    @pytest.mark.parametrize("key,expected", sorted(FROZEN_LISTS.items()))
    def test_list_frozen(self, keeper_grpo, key, expected):
        actual = OmegaConf.select(keeper_grpo, key)
        actual = list(actual) if actual is not None else None
        assert actual == expected, (
            f"KEEPER DRIFT: online_rground_external.yaml list '{key}' is now "
            f"{actual!r}, frozen snapshot expects {expected!r}. "
            f"(migration.md §parallel-stack rule.)"
        )

    @pytest.mark.parametrize("key,expected", sorted(FROZEN_PRESCREEN.items()))
    def test_prescreen_frozen(self, keeper_grpo, key, expected):
        actual = OmegaConf.select(keeper_grpo, f"prescreen.{key}")
        assert actual == expected, (
            f"KEEPER DRIFT: online_rground_external.yaml prescreen.{key} is now "
            f"{actual!r}, frozen snapshot expects {expected!r}. "
            f"(migration.md §parallel-stack rule.)"
        )

    def test_reward_composition_is_directional(self, keeper_grpo):
        """The single most load-bearing keeper value gets its own alarm."""
        assert keeper_grpo.reward_composition == "directional", (
            "KEEPER DRIFT: reward_composition must stay 'directional' — it is "
            "the composition path that produced v9-ckpt100 (migration.md maps "
            "it as a frozen surface). Any other value silently changes the "
            "keeper's reward."
        )

    def test_no_unexpected_keys_appeared(self, keeper_grpo):
        """A new key in the keeper yaml is itself an edit to a frozen file —
        catch additions, not just value drift."""
        present = set(keeper_grpo.keys())
        known = (
            set(FROZEN)
            | set(FROZEN_LISTS)
            | {"prescreen", "embedding_server_url", "judge_server_url"}
        )
        unexpected = present - known
        assert not unexpected, (
            f"KEEPER DRIFT: online_rground_external.yaml grew new key(s) "
            f"{sorted(unexpected)}. The keeper file is frozen (migration.md "
            f"§parallel-stack rule); adding a key is editing it. If genuinely "
            f"intended pre-camera-ready, add the key to this guard with "
            f"justification in the same commit."
        )


class TestKeeperFrozenCodeSurfaces:
    """The directional reward path + appropriateness machinery must exist and
    keep their contract (migration.md: 'the directional composition path in
    rewards.py, online_rground.py's clamp/multiplier machinery ... not removed').
    """

    def test_composite_reward_importable(self):
        from dagspaces.grpo_training.stages.rewards import CompositeRewardFunction

        assert callable(CompositeRewardFunction)

    def test_directional_composition_accepted(self):
        """Constructing with composition='directional' must not raise — the
        keeper's reward_composition value depends on it being a valid path."""
        from dagspaces.grpo_training.stages.rewards import CompositeRewardFunction

        fn = CompositeRewardFunction(
            weights=[0.10, 0.05, 0.05, 0.20, 0.10, 0.50],
            composition="directional",
        )
        assert fn.composition == "directional"

    def test_directional_combine_is_gate_times_content(self):
        """Freeze the directional combine contract: R = gate({0,1,2,4}) *
        content({3,5}). If a refactor changes the index partition, this fails."""
        from dagspaces.grpo_training.stages.rewards import CompositeRewardFunction

        fn = CompositeRewardFunction(
            weights=[0.10, 0.05, 0.05, 0.20, 0.10, 0.50],
            composition="directional",
        )
        # gate components (idx 0,1,2,4) = 1.0, content (idx 3,5) = 0.5 → 1.0*0.5
        comps = [1.0, 1.0, 1.0, 0.5, 1.0, 0.5]
        assert fn._combine(comps) == pytest.approx(0.5)
        # content components = 1.0, gate = 0.5 → 0.5*1.0
        comps2 = [0.5, 0.5, 0.5, 1.0, 0.5, 1.0]
        assert fn._combine(comps2) == pytest.approx(0.5)

    def test_bogus_composition_rejected(self):
        """The validation guard that keeps 'directional' meaningful is intact."""
        from dagspaces.grpo_training.stages.rewards import CompositeRewardFunction

        with pytest.raises(ValueError):
            CompositeRewardFunction(
                weights=[0.10, 0.05, 0.05, 0.20, 0.10, 0.50],
                composition="not_a_real_composition",
            )

    def test_appropriateness_multiplier_machinery_present(self):
        """online_rground.py's multiplier machinery (deontic.py) — the v9→v12a
        direction/floor/hedge tiers the keeper's r_ground depends on."""
        import dagspaces.grpo_training.stages.deontic as deontic

        assert hasattr(deontic, "direction_multiplier")
        assert hasattr(deontic, "appropriateness_multiplier")
        # The v12a hedge-tier contract: ordering correct > hedge-prohibited >
        # false-forbid > false-permit (online_rground_external.yaml comment).
        from dagspaces.grpo_training.stages.online_rground import OnlineRGround

        assert OnlineRGround is not None
