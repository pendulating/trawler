"""W&B logger for the privacylens dagspace.

``extra_internal_columns`` holds nested dicts and numpy arrays of mixed shape
(``seed``, ``vignette``, ``trajectory``, ``S``, ``V``, ``T``). W&B tables break
on those.

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

ensure_local_tmpdir("privacylens")

WandbConfig, WandbLogger = make_wandb_shim(
    "privacylens",
    default_project='privacylens-eval',
    default_experiment_name='privacylens',
    env_var_prefix='',
    full_column_stages=frozenset(
        {
            'agent_action_inference',
            'compute_metrics',
            'leakage_judge_inference',
            'qa_probe_inference',
        }
    ),
    full_column_key_prefixes=frozenset(
        {
            'agent_action_inference/',
            'compute_metrics/',
            'leakage_judge_inference/',
            'qa_probe_inference/',
        }
    ),
    extra_internal_columns=frozenset(
        {
            'S',
            'T',
            'V',
            'seed',
            'trajectory',
            'vignette',
        }
    ),
    extra_pattern_prefixes=[],
    extra_pattern_names=frozenset(),
    extra_runtime_keys=[],
    classify_variant_field=None,
)

__all__ = ["WandbConfig", "WandbLogger", "ensure_local_tmpdir", "collect_compute_metadata"]
