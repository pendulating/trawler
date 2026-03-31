"""Wandb logger for the confaide dagspace."""

from __future__ import annotations

from typing import Any, Dict, Optional

from dagspaces.common.wandb_logger import (
    WandbConfig as _WandbConfigBase,
    WandbLogger as _WandbLoggerBase,
    collect_compute_metadata,
    ensure_local_tmpdir,
)

ensure_local_tmpdir("confaide")

_FULL_COLUMN_STAGES = frozenset({"llm_inference"})


class WandbConfig(_WandbConfigBase):
    @classmethod
    def from_hydra_config(cls, cfg, **kwargs) -> "WandbConfig":
        kwargs.setdefault("default_project", "confaide")
        kwargs.setdefault("default_experiment_name", "CONFAIDE")
        kwargs.setdefault("env_var_prefix", "")
        kwargs.setdefault("full_column_stages", _FULL_COLUMN_STAGES)
        kwargs.setdefault("extra_runtime_keys", [])
        kwargs.setdefault("dagspace_name", "confaide")
        return super().from_hydra_config(cfg, **kwargs)


class WandbLogger(_WandbLoggerBase):
    def __init__(
        self,
        cfg,
        stage: str,
        run_id: Optional[str] = None,
        run_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(cfg, stage=stage, run_id=run_id, run_config=run_config)
        self.wb_config = WandbConfig.from_hydra_config(cfg)


__all__ = ["WandbConfig", "WandbLogger", "ensure_local_tmpdir", "collect_compute_metadata"]
