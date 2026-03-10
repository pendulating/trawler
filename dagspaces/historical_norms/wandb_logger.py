"""Wandb logger for the historical_norms dagspace.

Thin wrapper around dagspaces.common.wandb_logger that supplies
historical_norms-specific defaults:

- project:                "historical-norms-extraction"
- env_var_prefix:         "HISTORICAL_NORMS"  (reads HISTORICAL_NORMS_GPU_SANITIZE_* etc.)
- default_experiment_name: "historical_norms"
- full_column_stages:     norm_reasoning / norm_extraction / ci_reasoning /
                          ci_extraction / fetch_gutenberg
- extra_internal_columns: reasoning_data, ci_flows_raw
- extra_pattern_prefixes: (none — these columns are handled by extra_internal_columns)
- extra_pattern_names:    (none)

All public names are re-exported so that existing
``from .wandb_logger import WandbLogger, WandbConfig`` imports continue to work
without modification.

Tmpdir strategy: On import the module ensures TMPDIR points at a writable local
path (/scratch or /tmp) rather than a /share network mount, using the
`ensure_local_tmpdir` helper from the common module.
"""

from __future__ import annotations

from dagspaces.common.wandb_logger import (
    WandbConfig as _WandbConfigBase,
    WandbLogger as _WandbLoggerBase,
    collect_compute_metadata,
    ensure_local_tmpdir,
)
from typing import Any, Dict, Optional

# Set a suitable local TMPDIR at import time (same side-effect as before)
ensure_local_tmpdir("historical_norms")

# ---- historical_norms-specific defaults ------------------------------------

_HN_FULL_COLUMN_STAGES = frozenset(
    {
        "norm_reasoning",
        "norm_extraction",
        "ci_reasoning",
        "ci_extraction",
        "fetch_gutenberg",
    }
)

_HN_FULL_COLUMN_KEY_PREFIXES = frozenset(
    {
        "norm_reasoning/",
        "norm_extraction/",
        "ci_reasoning/",
        "ci_extraction/",
        "fetch_gutenberg/",
    }
)

# Nested dicts/arrays that cause wandb schema issues
_HN_EXTRA_INTERNAL_COLUMNS = frozenset(
    {
        "reasoning_data",  # contains nested 'norms' array with varying lengths
        "ci_flows_raw",   # contains nested CI flow objects with varying lengths
    }
)


class WandbConfig(_WandbConfigBase):
    """WandbConfig with historical_norms defaults baked in.

    Callers may still instantiate WandbConfig() directly and override any
    field, or use WandbConfig.from_hydra_config() without extra arguments.
    """

    @classmethod
    def from_hydra_config(cls, cfg, **kwargs) -> "WandbConfig":  # type: ignore[override]
        """Build a WandbConfig from a Hydra config with historical_norms defaults."""
        kwargs.setdefault("default_project", "historical-norms-extraction")
        kwargs.setdefault("default_experiment_name", "historical_norms")
        kwargs.setdefault("env_var_prefix", "HISTORICAL_NORMS")
        kwargs.setdefault("full_column_stages", _HN_FULL_COLUMN_STAGES)
        kwargs.setdefault(
            "full_column_key_prefixes", _HN_FULL_COLUMN_KEY_PREFIXES
        )
        kwargs.setdefault(
            "extra_internal_columns", _HN_EXTRA_INTERNAL_COLUMNS
        )
        kwargs.setdefault("extra_pattern_prefixes", [])
        kwargs.setdefault("extra_pattern_names", frozenset())
        kwargs.setdefault("extra_runtime_keys", [])
        kwargs.setdefault("classify_variant_field", None)
        return super().from_hydra_config(cfg, **kwargs)


class WandbLogger(_WandbLoggerBase):
    """WandbLogger that auto-applies historical_norms WandbConfig defaults.

    When callers do ``WandbLogger(cfg, stage=...)``, this subclass ensures the
    historical_norms-flavoured WandbConfig is created automatically.
    """

    def __init__(
        self,
        cfg,
        stage: str,
        run_id: Optional[str] = None,
        run_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(cfg, stage=stage, run_id=run_id, run_config=run_config)
        # Override wb_config with the historical_norms-specific version
        self.wb_config = WandbConfig.from_hydra_config(cfg)


__all__ = [
    "WandbConfig",
    "WandbLogger",
    "ensure_local_tmpdir",
    "collect_compute_metadata",
]
