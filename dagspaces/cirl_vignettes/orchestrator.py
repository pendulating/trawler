"""Orchestrator for the cirl_vignettes dagspace.

Thin wrapper over the shared eval run loop in ``dagspaces/common/orchestrator.py``
(see ``wiki/jul19_orchestrator_unification_plan.md``).  The only cirl_vignettes-specific
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
# ``importlib.import_module("dagspaces.cirl_vignettes.orchestrator").get_stage_registry()``.
from .runners import get_stage_registry  # noqa: F401

_CONF_DIR = os.path.join(os.path.dirname(__file__), "conf")



def _log_eval_metrics(logger, metrics: dict[str, Any], stage: str) -> None:
    """Log evaluation metrics to W&B and print a structured summary."""
    task = metrics.get("task", "unknown")
    if task == "cirl_trajectory":
        _log_trajectory_metrics(logger, metrics, stage)
    else:
        _log_probing_metrics(logger, metrics, stage)


# The two formatters below were dropped by the 5aedadb orchestrator
# unification while _log_eval_metrics kept calling them (NameError on the
# first trajectory finalize, sweep 2026-07-19/19-20-27 cells 0/2).
# Restored verbatim from 5aedadb^.

def _log_probing_metrics(logger, metrics: dict[str, Any], stage: str) -> None:
    """Log probing evaluation metrics."""
    prefix = f"{stage}/eval"

    wb_metrics: dict[str, Any] = {}

    for key in ("accuracy", "unparseable_rate", "unparseable_count",
                "total", "parseable", "reject_count", "accept_count"):
        if key in metrics:
            wb_metrics[f"{prefix}/{key}"] = metrics[key]

    for level, level_m in metrics.get("per_level", {}).items():
        for k, v in level_m.items():
            wb_metrics[f"{prefix}/{level}/{k}"] = v

    if wb_metrics:
        logger.log_metrics(wb_metrics)

    task = metrics.get("task", "unknown")
    print(flush=True)
    print("=" * 60, flush=True)
    print(f"  PROBING RESULTS — {task.upper()}", flush=True)
    print("=" * 60, flush=True)
    print(f"  Samples:      {metrics.get('total', '?')} total, "
          f"{metrics.get('parseable', '?')} parseable, "
          f"{metrics.get('unparseable_count', '?')} unparseable "
          f"({metrics.get('unparseable_rate', 0) * 100:.1f}%)", flush=True)
    print(f"  Rejection acc (overall): {metrics.get('accuracy', '?')}", flush=True)
    print(f"  Reject (B): {metrics.get('reject_count', '?')}  "
          f"Accept (A): {metrics.get('accept_count', '?')}", flush=True)
    print("-" * 60, flush=True)
    for level, level_m in metrics.get("per_level", {}).items():
        print(f"  {level:>12s}:  "
              f"{level_m.get('reject_count', '?')}/{level_m.get('total', '?')} reject  "
              f"(acc={level_m.get('accuracy', '?')})", flush=True)
    print("=" * 60, flush=True)
    print(flush=True)


def _log_trajectory_metrics(logger, metrics: dict[str, Any], stage: str) -> None:
    """Log trajectory I/U/C evaluation metrics.

    The compute_trajectory_metrics schema moved to suffixed keys
    (``integrity_among_judged`` / ``*_overall_with_default_*``) after the
    original hardcoded key list was written, which made this formatter
    silently log almost nothing. W&B now gets every top-level numeric
    key; the printed summary reads the ``_among_judged`` variants with a
    fallback to the legacy bare names.
    """
    prefix = f"{stage}/eval"

    wb_metrics: dict[str, Any] = {
        f"{prefix}/{k}": v
        for k, v in metrics.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }
    if wb_metrics:
        logger.log_metrics(wb_metrics)

    def _m(*names: str) -> Any:
        for n in names:
            if n in metrics:
                return metrics[n]
        return "?"

    print(flush=True)
    print("=" * 60, flush=True)
    print("  TRAJECTORY RESULTS — CIRL I/U/C", flush=True)
    print("=" * 60, flush=True)
    print(f"  Cases:          {_m('total')}", flush=True)
    print(f"  Integrity (I):  {_m('integrity_among_judged', 'integrity')}", flush=True)
    print(f"  Utility (U):    {_m('utility_among_judged', 'utility')}", flush=True)
    print(f"  Complete (C):   {_m('complete_among_judged', 'complete')}", flush=True)
    print("-" * 60, flush=True)
    print(f"  Leakage rate:          {_m('leakage_rate_among_judged', 'leakage_rate')}", flush=True)
    print(f"  Adj. leakage rate:     {_m('adjusted_leakage_rate')}", flush=True)
    print(f"  Avg helpfulness:       {_m('avg_helpfulness_score_among_judged', 'avg_helpfulness_score')}", flush=True)
    print(f"  Helpful rate (>=2):    {_m('helpful_rate_among_judged', 'helpful_rate')}", flush=True)
    hdist = metrics.get("helpfulness_distribution", {})
    if hdist:
        print(f"  Helpfulness dist:      {hdist}", flush=True)
    print("=" * 60, flush=True)
    print(flush=True)


ORCHESTRATOR_HOOKS = OrchestratorHooks(
    dagspace_module="dagspaces.cirl_vignettes.orchestrator",
    dagspace_name="cirl_vignettes",
    output_subdir="cirl_vignettes",
    job_prefix="CIRLVignettes",
    config_dir=_CONF_DIR,
    log_eval_metrics=_log_eval_metrics,
    wandb_dagspace=lambda cfg: "cirl_vignettes",
    use_srun=False,
)


def run_experiment(cfg: DictConfig) -> None:
    """Execute the cirl_vignettes evaluation pipeline."""
    _run_experiment(cfg, ORCHESTRATOR_HOOKS)
