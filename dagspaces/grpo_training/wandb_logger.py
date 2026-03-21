"""Wandb logger for the grpo_training dagspace.

Thin wrapper around dagspaces.common.wandb_logger that supplies
grpo_training-specific defaults:

- project:                "grpo-ci-training"
- env_var_prefix:         "GRPO_TRAINING"
- default_experiment_name: "grpo_training"
- full_column_stages:     sft_data_prep / reward_prep
"""

from __future__ import annotations

from dagspaces.common.wandb_logger import (
    WandbConfig as _WandbConfigBase,
    WandbLogger as _WandbLoggerBase,
    collect_compute_metadata,
    ensure_local_tmpdir,
)
from typing import Any, Dict, Optional

ensure_local_tmpdir("grpo_training")

_GRPO_FULL_COLUMN_STAGES = frozenset(
    {
        "sft_data_prep",
        "reward_prep",
    }
)

_GRPO_FULL_COLUMN_KEY_PREFIXES = frozenset(
    {
        "sft_data_prep/",
        "reward_prep/",
    }
)

_GRPO_EXTRA_INTERNAL_COLUMNS = frozenset(
    {
        "messages",  # chat-format list, not suitable for wandb tables
        "norm_universe_json",  # large nested JSON
    }
)


class WandbConfig(_WandbConfigBase):
    """WandbConfig with grpo_training defaults baked in."""

    @classmethod
    def from_hydra_config(cls, cfg, **kwargs) -> "WandbConfig":  # type: ignore[override]
        kwargs.setdefault("default_project", "grpo-ci-training")
        kwargs.setdefault("default_experiment_name", "grpo_training")
        kwargs.setdefault("env_var_prefix", "GRPO_TRAINING")
        kwargs.setdefault("full_column_stages", _GRPO_FULL_COLUMN_STAGES)
        kwargs.setdefault(
            "full_column_key_prefixes", _GRPO_FULL_COLUMN_KEY_PREFIXES
        )
        kwargs.setdefault(
            "extra_internal_columns", _GRPO_EXTRA_INTERNAL_COLUMNS
        )
        kwargs.setdefault("extra_pattern_prefixes", [])
        kwargs.setdefault("extra_pattern_names", frozenset())
        kwargs.setdefault("extra_runtime_keys", [])
        kwargs.setdefault("classify_variant_field", None)
        kwargs.setdefault("dagspace_name", "grpo_training")
        return super().from_hydra_config(cfg, **kwargs)


class WandbLogger(_WandbLoggerBase):
    """WandbLogger that auto-applies grpo_training WandbConfig defaults."""

    def __init__(
        self,
        cfg,
        stage: str,
        run_id: Optional[str] = None,
        run_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(cfg, stage=stage, run_id=run_id, run_config=run_config)
        self.wb_config = WandbConfig.from_hydra_config(cfg)


__all__ = [
    "WandbConfig",
    "WandbLogger",
    "ensure_local_tmpdir",
    "collect_compute_metadata",
]
