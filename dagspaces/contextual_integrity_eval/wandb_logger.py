"""Wandb logger for the contextual_integrity_eval (PrivacyLens) dagspace.

Thin wrapper around dagspaces.common.wandb_logger that supplies
PrivacyLens-specific defaults:

- project:                "privacylens-eval"
- default_experiment_name: "privacylens"
- full_column_stages:     qa_probe_inference / agent_action_inference /
                          leakage_judge_inference / compute_metrics

All public names are re-exported so that existing
``from .wandb_logger import WandbLogger, WandbConfig`` imports continue to work.

Tmpdir strategy: On import the module ensures TMPDIR points at a writable local
path (/scratch or /tmp) rather than a /share network mount.
"""

from __future__ import annotations

from dagspaces.common.wandb_logger import (
    WandbConfig as _WandbConfigBase,
    WandbLogger as _WandbLoggerBase,
    collect_compute_metadata,
    ensure_local_tmpdir,
)
from typing import Any, Dict, Optional

# Set a suitable local TMPDIR at import time
ensure_local_tmpdir("contextual_integrity_eval")

# ---- PrivacyLens-specific defaults -----------------------------------------

_PL_FULL_COLUMN_STAGES = frozenset(
    {
        "qa_probe_inference",
        "agent_action_inference",
        "leakage_judge_inference",
        "compute_metrics",
    }
)

_PL_FULL_COLUMN_KEY_PREFIXES = frozenset(
    {
        "qa_probe_inference/",
        "agent_action_inference/",
        "leakage_judge_inference/",
        "compute_metrics/",
    }
)


class WandbConfig(_WandbConfigBase):
    """WandbConfig with PrivacyLens defaults baked in."""

    @classmethod
    def from_hydra_config(cls, cfg, **kwargs) -> "WandbConfig":  # type: ignore[override]
        kwargs.setdefault("default_project", "privacylens-eval")
        kwargs.setdefault("default_experiment_name", "privacylens")
        kwargs.setdefault("env_var_prefix", "")
        kwargs.setdefault("full_column_stages", _PL_FULL_COLUMN_STAGES)
        kwargs.setdefault("full_column_key_prefixes", _PL_FULL_COLUMN_KEY_PREFIXES)
        kwargs.setdefault("extra_internal_columns", frozenset())
        kwargs.setdefault("extra_pattern_prefixes", [])
        kwargs.setdefault("extra_pattern_names", frozenset())
        kwargs.setdefault("extra_runtime_keys", [])
        kwargs.setdefault("classify_variant_field", None)
        return super().from_hydra_config(cfg, **kwargs)


class WandbLogger(_WandbLoggerBase):
    """WandbLogger that auto-applies PrivacyLens WandbConfig defaults."""

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
