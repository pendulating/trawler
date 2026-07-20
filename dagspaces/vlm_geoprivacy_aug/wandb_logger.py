"""Wandb logger for the vlm_geoprivacy_aug dagspace.

Thin wrapper around dagspaces.common.wandb_logger with
VLM-GeoPrivacyBench-specific defaults.
"""

from __future__ import annotations

from typing import Any

from dagspaces.common.wandb_logger import (
    WandbConfig as _WandbConfigBase,
)
from dagspaces.common.wandb_logger import (
    WandbLogger as _WandbLoggerBase,
)
from dagspaces.common.wandb_logger import (
    collect_compute_metadata,
    ensure_local_tmpdir,
)

ensure_local_tmpdir("vlm_geoprivacy_aug")

_VLM_FULL_COLUMN_STAGES = frozenset(
    {"vlm_mcq_inference", "vlm_freeform_inference", "granularity_judge"}
)


class WandbConfig(_WandbConfigBase):
    @classmethod
    def from_hydra_config(cls, cfg, **kwargs) -> WandbConfig:
        kwargs.setdefault("default_project", "vlm-geoprivacy-bench")
        kwargs.setdefault("default_experiment_name", "VLM-GeoPrivacyBench")
        kwargs.setdefault("env_var_prefix", "")
        kwargs.setdefault("full_column_stages", _VLM_FULL_COLUMN_STAGES)
        kwargs.setdefault("extra_runtime_keys", [])
        kwargs.setdefault("dagspace_name", "vlm_geoprivacy_aug")
        return super().from_hydra_config(cfg, **kwargs)


class WandbLogger(_WandbLoggerBase):
    def __init__(
        self,
        cfg,
        stage: str,
        run_id: str | None = None,
        run_config: dict[str, Any] | None = None,
        *,
        wandb_id: str | None = None,
        resume: str | None = None,
    ) -> None:
        super().__init__(
            cfg, stage=stage, run_id=run_id, run_config=run_config,
            wandb_id=wandb_id, resume=resume,
        )
        self.wb_config = WandbConfig.from_hydra_config(cfg)


__all__ = ["WandbConfig", "WandbLogger", "ensure_local_tmpdir", "collect_compute_metadata"]
