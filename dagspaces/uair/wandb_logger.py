"""Wandb logger for the uair dagspace.

Thin wrapper around dagspaces.common.wandb_logger that supplies uair-specific
defaults:

- project:                "nov10-workshop"
- env_var_prefix:         "UAIR"  (reads UAIR_GPU_SANITIZE_* env vars)
- default_experiment_name: "UAIR"
- classify_variant_field: "classification_profile"
- full_column_stages:     decompose / decompose_nbl / verify_nbl / classify /
                          classify_eu_act / classify_risk_and_benefits
- extra_pattern_prefixes: ["eu_ai_raw_json"]
- extra_pattern_names:    {"llm_json"}
- extra_runtime_keys:     use_llm_classify, use_llm_decompose, prefilter_mode,
                          keyword_buffering

Note: the pipeline name env-var fallback was renamed from ``UAIR_PIPELINE_NAME``
to the generic ``WANDB_PIPELINE_NAME`` in the common module.

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
ensure_local_tmpdir("uair")

# ---- uair-specific defaults ------------------------------------------------

_UAIR_FULL_COLUMN_STAGES = frozenset(
    {
        "decompose",
        "decompose_nbl",
        "verify_nbl",
        "classify",
        "classify_eu_act",
        "classify_risk_and_benefits",
    }
)

_UAIR_FULL_COLUMN_KEY_PREFIXES = frozenset(
    {
        "decompose/",
        "decompose_nbl/",
        "verify_nbl/",
        "classify/",
    }
)

_UAIR_EXTRA_INTERNAL_COLUMNS: frozenset = frozenset()

_UAIR_EXTRA_PATTERN_PREFIXES = ["eu_ai_raw_json"]

_UAIR_EXTRA_PATTERN_NAMES = frozenset({"llm_json"})

_UAIR_EXTRA_RUNTIME_KEYS = [
    "use_llm_classify",
    "use_llm_decompose",
    "prefilter_mode",
    "keyword_buffering",
]


class WandbConfig(_WandbConfigBase):
    """WandbConfig with uair defaults baked in.

    Callers may still instantiate WandbConfig() directly and override any field,
    or use WandbConfig.from_hydra_config() without any extra arguments.
    """

    @classmethod
    def from_hydra_config(cls, cfg, **kwargs) -> "WandbConfig":  # type: ignore[override]
        """Build a WandbConfig from a Hydra config with uair defaults."""
        kwargs.setdefault("default_project", "nov10-workshop")
        kwargs.setdefault("default_experiment_name", "UAIR")
        kwargs.setdefault("env_var_prefix", "UAIR")
        kwargs.setdefault("full_column_stages", _UAIR_FULL_COLUMN_STAGES)
        kwargs.setdefault(
            "full_column_key_prefixes", _UAIR_FULL_COLUMN_KEY_PREFIXES
        )
        kwargs.setdefault(
            "extra_internal_columns", _UAIR_EXTRA_INTERNAL_COLUMNS
        )
        kwargs.setdefault(
            "extra_pattern_prefixes", _UAIR_EXTRA_PATTERN_PREFIXES
        )
        kwargs.setdefault("extra_pattern_names", _UAIR_EXTRA_PATTERN_NAMES)
        kwargs.setdefault("extra_runtime_keys", _UAIR_EXTRA_RUNTIME_KEYS)
        kwargs.setdefault(
            "classify_variant_field", "classification_profile"
        )
        return super().from_hydra_config(cfg, **kwargs)


class WandbLogger(_WandbLoggerBase):
    """WandbLogger that auto-applies uair WandbConfig defaults on construction.

    When callers do ``WandbLogger(cfg, stage=...)``, this subclass ensures the
    uair-flavoured WandbConfig (with correct project, full_column_stages, etc.)
    is created automatically via WandbConfig.from_hydra_config, rather than the
    generic common defaults.
    """

    def __init__(
        self,
        cfg,
        stage: str,
        run_id: Optional[str] = None,
        run_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(cfg, stage=stage, run_id=run_id, run_config=run_config)
        # Override wb_config with the uair-specific version
        self.wb_config = WandbConfig.from_hydra_config(cfg)


__all__ = [
    "WandbConfig",
    "WandbLogger",
    "ensure_local_tmpdir",
    "collect_compute_metadata",
]
