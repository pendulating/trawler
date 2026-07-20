"""Orchestrator for the confaide dagspace.

Thin wrapper over the shared eval run loop in ``dagspaces/common/orchestrator.py``
(see ``wiki/jul19_orchestrator_unification_plan.md``).  The only confaide-specific
code below is the metric formatter; everything else is the generic
loop parameterized by ``ORCHESTRATOR_HOOKS``.
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
# ``importlib.import_module("dagspaces.confaide.orchestrator").get_stage_registry()``.
from .runners import get_stage_registry  # noqa: F401

_CONF_DIR = os.path.join(os.path.dirname(__file__), "conf")



def _log_eval_metrics(logger, metrics: dict[str, Any], stage: str) -> None:
    """Log CONFAIDE evaluation metrics to W&B and print a structured summary."""
    prefix = f"{stage}/eval"

    # Log all numeric metrics to W&B (skip 'task' string identifier)
    wb_metrics: dict[str, Any] = {}
    for key, value in metrics.items():
        if key == "task":
            continue
        if isinstance(value, (int, float)):
            wb_metrics[f"{prefix}/{key}"] = value

    if wb_metrics:
        logger.log_metrics(wb_metrics)

    task = metrics.get("task", "unknown")
    print(flush=True)
    print("=" * 60, flush=True)
    print(f"  EVALUATION RESULTS — {task.upper()}", flush=True)
    print("=" * 60, flush=True)
    print(f"  Samples:      {metrics.get('total', '?')} total", flush=True)

    if "pearson_r" in metrics:
        print(f"  Parseable:    {metrics.get('parseable', '?')}  "
              f"Unparseable: {metrics.get('unparseable_count', '?')} "
              f"({metrics.get('unparseable_rate', 0) * 100:.1f}%)", flush=True)
        print(f"  Pearson r:    {metrics['pearson_r']}", flush=True)
        print(f"  Model mean:   {metrics.get('mean_model_rating', '?')}  "
              f"Human mean: {metrics.get('mean_human_rating', '?')}", flush=True)
    elif "reject_count" in metrics:
        print(f"  Parseable:    {metrics.get('parseable', '?')}  "
              f"Unparseable: {metrics.get('unparseable_count', '?')} "
              f"({metrics.get('unparseable_rate', 0) * 100:.1f}%)", flush=True)
        print(f"  Rejection acc: {metrics.get('accuracy', '?')}", flush=True)
        print(f"  Reject (No): {metrics.get('reject_count', '?')}  "
              f"Accept (Yes): {metrics.get('accept_count', '?')}", flush=True)
    elif "leak_rate" in metrics:
        print(f"  Leak rate:    {metrics['leak_rate']}", flush=True)
        print(f"  Leaked: {metrics.get('leak_count', '?')}  "
              f"No leak: {metrics.get('no_leak_count', '?')}", flush=True)
    elif "error_count" in metrics:
        print(f"  Error rate:   {metrics.get('error_rate', '?')}", flush=True)
        print(f"  Errors: {metrics.get('error_count', '?')}  "
              f"Correct: {metrics.get('correct_count', '?')}", flush=True)
    print("=" * 60, flush=True)
    print(flush=True)

ORCHESTRATOR_HOOKS = OrchestratorHooks(
    dagspace_module="dagspaces.confaide.orchestrator",
    dagspace_name="confaide",
    output_subdir="confaide",
    job_prefix="CONFAIDE",
    config_dir=_CONF_DIR,
    log_eval_metrics=_log_eval_metrics,
    wandb_dagspace=lambda cfg: "confaide",
    use_srun=False,
)


def run_experiment(cfg: DictConfig) -> None:
    """Execute the confaide evaluation pipeline."""
    _run_experiment(cfg, ORCHESTRATOR_HOOKS)
