"""W&B logger shim for the mmlu dagspace.

Thin wrapper around dagspaces.common.wandb_logger with mmlu-specific
defaults (project name, dagspace tag, full-column stages).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from dagspaces.common.wandb_logger import (
    WandbConfig as _WandbConfigBase,
    WandbLogger as _WandbLoggerBase,
    collect_compute_metadata,
    ensure_local_tmpdir,
    pipeline_run_id,
)

ensure_local_tmpdir("mmlu")

_FULL_COLUMN_STAGES = frozenset({"llm_inference"})


class WandbConfig(_WandbConfigBase):
    @classmethod
    def from_hydra_config(cls, cfg, **kwargs) -> "WandbConfig":
        kwargs.setdefault("default_project", "mmlu")
        kwargs.setdefault("default_experiment_name", "MMLU")
        kwargs.setdefault("env_var_prefix", "")
        kwargs.setdefault("full_column_stages", _FULL_COLUMN_STAGES)
        kwargs.setdefault("extra_runtime_keys", [])
        kwargs.setdefault("dagspace_name", "mmlu")
        return super().from_hydra_config(cfg, **kwargs)


class WandbLogger(_WandbLoggerBase):
    def __init__(
        self,
        cfg,
        stage: str,
        run_id: Optional[str] = None,
        run_config: Optional[Dict[str, Any]] = None,
        *,
        wandb_id: Optional[str] = None,
        resume: Optional[str] = None,
    ) -> None:
        super().__init__(
            cfg, stage=stage, run_id=run_id, run_config=run_config,
            wandb_id=wandb_id, resume=resume,
        )
        self.wb_config = WandbConfig.from_hydra_config(cfg)


__all__ = ["WandbConfig", "WandbLogger", "ensure_local_tmpdir", "collect_compute_metadata"]
