"""Wandb logger for the contextual_integrity_eval dagspace.

Thin wrapper around dagspaces.common.wandb_logger that supplies
contextual_integrity_eval-specific defaults:

- project:                "nov10-workshop"
- env_var_prefix:         ""  (no GPU sanitizer env vars for this dagspace)
- default_experiment_name: "UAIR"
- full_column_stages:     decompose / decompose_nbl / verify_nbl /
                          classify / classify_eu_act /
                          classify_risk_and_benefits /
                          reasoning / extraction /
                          norm_reasoning / norm_extraction /
                          ci_reasoning / ci_extraction
- extra_internal_columns: reasoning_data, ci_flows_raw
- extra_pattern_prefixes: ["eu_ai_raw_json"]
- extra_pattern_names:    {"llm_json"}

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
ensure_local_tmpdir("contextual_integrity_eval")

# ---- contextual_integrity_eval-specific defaults ---------------------------

_CI_FULL_COLUMN_STAGES = frozenset(
    {
        "decompose",
        "decompose_nbl",
        "verify_nbl",
        "classify",
        "classify_eu_act",
        "classify_risk_and_benefits",
        "reasoning",
        "extraction",
        "norm_reasoning",
        "norm_extraction",
        "ci_reasoning",
        "ci_extraction",
    }
)

_CI_FULL_COLUMN_KEY_PREFIXES = frozenset(
    {
        "decompose/",
        "decompose_nbl/",
        "verify_nbl/",
        "classify/",
        "reasoning/",
        "extraction/",
        "norm_reasoning/",
        "norm_extraction/",
        "ci_reasoning/",
        "ci_extraction/",
    }
)

# Nested dicts/arrays that cause wandb schema issues
_CI_EXTRA_INTERNAL_COLUMNS = frozenset(
    {
        "reasoning_data",  # contains nested 'norms' array with varying lengths
        "ci_flows_raw",   # contains nested CI flow objects with varying lengths
    }
)

_CI_EXTRA_PATTERN_PREFIXES = ["eu_ai_raw_json"]

_CI_EXTRA_PATTERN_NAMES = frozenset({"llm_json"})


class WandbConfig(_WandbConfigBase):
    """WandbConfig with contextual_integrity_eval defaults baked in.

    Callers may still instantiate WandbConfig() directly and override any
    field, or use WandbConfig.from_hydra_config() without extra arguments.
    """

    @classmethod
    def from_hydra_config(cls, cfg, **kwargs) -> "WandbConfig":  # type: ignore[override]
        """Build WandbConfig from Hydra config with contextual_integrity_eval defaults."""
        kwargs.setdefault("default_project", "nov10-workshop")
        kwargs.setdefault("default_experiment_name", "UAIR")
        kwargs.setdefault("env_var_prefix", "")
        kwargs.setdefault("full_column_stages", _CI_FULL_COLUMN_STAGES)
        kwargs.setdefault(
            "full_column_key_prefixes", _CI_FULL_COLUMN_KEY_PREFIXES
        )
        kwargs.setdefault(
            "extra_internal_columns", _CI_EXTRA_INTERNAL_COLUMNS
        )
        kwargs.setdefault(
            "extra_pattern_prefixes", _CI_EXTRA_PATTERN_PREFIXES
        )
        kwargs.setdefault("extra_pattern_names", _CI_EXTRA_PATTERN_NAMES)
        kwargs.setdefault("extra_runtime_keys", [])
        kwargs.setdefault("classify_variant_field", None)
        return super().from_hydra_config(cfg, **kwargs)


class WandbLogger(_WandbLoggerBase):
    """WandbLogger that auto-applies contextual_integrity_eval WandbConfig defaults.

    When callers do ``WandbLogger(cfg, stage=...)``, this subclass ensures the
    contextual_integrity_eval-flavoured WandbConfig is created automatically.
    """

    def __init__(
        self,
        cfg,
        stage: str,
        run_id: Optional[str] = None,
        run_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(cfg, stage=stage, run_id=run_id, run_config=run_config)
        # Override wb_config with the contextual_integrity_eval-specific version
        self.wb_config = WandbConfig.from_hydra_config(cfg)


__all__ = [
    "WandbConfig",
    "WandbLogger",
    "ensure_local_tmpdir",
    "collect_compute_metadata",
]
