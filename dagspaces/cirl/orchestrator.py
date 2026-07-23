"""Orchestrator for the cirl dagspace (CIRL-729 action benchmark).

Thin wrapper over the shared eval run loop in ``dagspaces/common/orchestrator.py``
(see ``wiki/jul19_orchestrator_unification_plan.md``). The only cirl-specific
code below is the metric formatter; everything else is the generic loop
parameterized by ``ORCHESTRATOR_HOOKS``.
"""

from __future__ import annotations

import os
from typing import Any

from omegaconf import DictConfig

from dagspaces.common.orchestrator import (
    OrchestratorHooks,
)
from dagspaces.common.orchestrator import (
    run_experiment as _run_experiment,
)

# Re-exported for the generic SLURM worker, which recovers the registry via
# ``importlib.import_module("dagspaces.cirl.orchestrator").get_stage_registry()``.
from .runners import get_stage_registry  # noqa: F401

_CONF_DIR = os.path.join(os.path.dirname(__file__), "conf")


def _log_eval_metrics(logger, metrics: dict[str, Any], stage: str) -> None:
    """Log CIRL-729 action leakage / utility / net metrics to W&B + stdout."""
    prefix = f"{stage}/eval"

    leakage = metrics.get("leakage", {}) or {}
    utility = metrics.get("utility", {}) or {}

    wb_metrics: dict[str, Any] = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            wb_metrics[f"{prefix}/{k}"] = v
    for k, v in leakage.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            wb_metrics[f"{prefix}/leakage/{k}"] = v
    for k, v in utility.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            wb_metrics[f"{prefix}/utility/{k}"] = v
    if wb_metrics:
        logger.log_metrics(wb_metrics)

    print(flush=True)
    print("=" * 60, flush=True)
    print("  CIRL-729 ACTION RESULTS", flush=True)
    print("=" * 60, flush=True)
    print(f"  Cases:            {metrics.get('total', '?')} "
          f"({metrics.get('parseable', '?')} strict-parseable, "
          f"{metrics.get('unparseable_count', '?')} unparseable, "
          f"{metrics.get('unparseable_rate', 0) * 100:.1f}%)", flush=True)
    print("-" * 60, flush=True)
    print(f"  Leakage rate  (↓): {leakage.get('leakage_rate', '?')}  "
          f"[micro {leakage.get('leakage_rate_micro', '?')}, "
          f"wb {leakage.get('leakage_rate_word_boundary', '?')}]", flush=True)
    print(f"  Utility rate  (↑): {utility.get('utility_rate', '?')}  "
          f"[micro {utility.get('utility_rate_micro', '?')}]", flush=True)
    print(f"  Net score     (↑): {metrics.get('net_score', '?')}", flush=True)
    print("-" * 60, flush=True)
    print(f"  Lenient — leakage {leakage.get('leakage_rate_lenient', '?')}, "
          f"utility {utility.get('utility_rate_lenient', '?')}, "
          f"net {metrics.get('net_score_lenient', '?')}", flush=True)
    print("=" * 60, flush=True)
    print(flush=True)


ORCHESTRATOR_HOOKS = OrchestratorHooks(
    dagspace_module="dagspaces.cirl.orchestrator",
    dagspace_name="cirl",
    output_subdir="cirl",
    job_prefix="CIRL",
    config_dir=_CONF_DIR,
    log_eval_metrics=_log_eval_metrics,
    wandb_dagspace=lambda cfg: "cirl",
    use_srun=False,
)


def run_experiment(cfg: DictConfig) -> None:
    """Execute the cirl (CIRL-729) evaluation pipeline."""
    _run_experiment(cfg, ORCHESTRATOR_HOOKS)
