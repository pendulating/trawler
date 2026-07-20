"""Wandb logger for the ci_heuristic dagspace.

Thin wrapper around dagspaces.common.wandb_logger with
ci_heuristic-specific defaults.
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

ensure_local_tmpdir("ci_heuristic")

_FULL_COLUMN_STAGES = frozenset({"traverse", "tp_probe"})


class WandbConfig(_WandbConfigBase):
    @classmethod
    def from_hydra_config(cls, cfg, **kwargs) -> WandbConfig:
        kwargs.setdefault("default_project", "ci-heuristic")
        kwargs.setdefault("default_experiment_name", "ci_heuristic")
        kwargs.setdefault("env_var_prefix", "")
        kwargs.setdefault("full_column_stages", _FULL_COLUMN_STAGES)
        kwargs.setdefault("extra_runtime_keys", [])
        kwargs.setdefault("dagspace_name", "ci_heuristic")
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
