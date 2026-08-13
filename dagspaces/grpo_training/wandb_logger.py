"""W&B logger for the grpo_training dagspace.

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

ensure_local_tmpdir("grpo_training")

WandbConfig, WandbLogger = make_wandb_shim(
    "grpo_training",
    default_project='grpo-ci-training',
    default_experiment_name='grpo_training',
    env_var_prefix='GRPO_TRAINING',
    full_column_stages=frozenset(
        {
            'reward_prep',
            'sft_data_prep',
        }
    ),
    full_column_key_prefixes=frozenset(
        {
            'reward_prep/',
            'sft_data_prep/',
        }
    ),
    extra_internal_columns=frozenset(
        {
            'messages',
            'norm_universe_json',
        }
    ),
    extra_pattern_prefixes=[],
    extra_pattern_names=frozenset(),
    extra_runtime_keys=[],
    classify_variant_field=None,
)

__all__ = ["WandbConfig", "WandbLogger", "ensure_local_tmpdir", "collect_compute_metadata"]
