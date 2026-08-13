"""W&B logger for the historical_norms dagspace.

``extra_internal_columns`` holds the two columns whose nested arrays vary in
length per row (``reasoning_data`` carries a ``norms`` array, ``ci_flows_raw``
carries CI flow objects). W&B tables cannot take those.

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

ensure_local_tmpdir("historical_norms")

WandbConfig, WandbLogger = make_wandb_shim(
    "historical_norms",
    default_project='historical-norms-extraction',
    default_experiment_name='historical_norms',
    env_var_prefix='HISTORICAL_NORMS',
    full_column_stages=frozenset(
        {
            'ci_extraction',
            'ci_reasoning',
            'fetch_gutenberg',
            'norm_extraction',
            'norm_reasoning',
            'norm_role_abstraction',
        }
    ),
    full_column_key_prefixes=frozenset(
        {
            'ci_extraction/',
            'ci_reasoning/',
            'fetch_gutenberg/',
            'norm_extraction/',
            'norm_reasoning/',
            'norm_role_abstraction/',
        }
    ),
    extra_internal_columns=frozenset(
        {
            'ci_flows_raw',
            'reasoning_data',
        }
    ),
    extra_pattern_prefixes=[],
    extra_pattern_names=frozenset(),
    extra_runtime_keys=[],
    classify_variant_field=None,
)

__all__ = ["WandbConfig", "WandbLogger", "ensure_local_tmpdir", "collect_compute_metadata"]
