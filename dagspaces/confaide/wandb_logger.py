"""W&B logger for the confaide dagspace.

Built by ``dagspaces/common/wandb_shim.py``; see that module for the two
details a dagspace must keep here (the import-time ``ensure_local_tmpdir``
call, and the ``pipeline_run_id`` re-export).
"""

from __future__ import annotations

from dagspaces.common.wandb_logger import (
    collect_compute_metadata,
    ensure_local_tmpdir,
    pipeline_run_id,  # re-export: common/orchestrator.make_wandb_logger calls wl.pipeline_run_id on this shim
)
from dagspaces.common.wandb_shim import make_wandb_shim

ensure_local_tmpdir("confaide")

WandbConfig, WandbLogger = make_wandb_shim(
    "confaide",
    default_project='confaide',
    default_experiment_name='CONFAIDE',
    env_var_prefix='',
    full_column_stages=frozenset({'llm_inference'}),
    extra_runtime_keys=[],
)

__all__ = ["WandbConfig", "WandbLogger", "ensure_local_tmpdir", "collect_compute_metadata"]
