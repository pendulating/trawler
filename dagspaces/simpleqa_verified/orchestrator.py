"""Orchestrator for the simpleqa_verified dagspace.

Thin wrapper over the shared eval run loop in ``dagspaces/common/orchestrator.py``
(see ``wiki/jul19_orchestrator_unification_plan.md``).  The only simpleqa_verified-specific
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
# ``importlib.import_module("dagspaces.simpleqa_verified.orchestrator").get_stage_registry()``.
from .runners import get_stage_registry  # noqa: F401

_CONF_DIR = os.path.join(os.path.dirname(__file__), "conf")



def _log_eval_metrics(logger, metrics: dict[str, Any], stage: str) -> None:
    """Flatten SimpleQA metrics to W&B-friendly scalars + print summary.

    Metric shape produced by stages/compute_metrics.py:
        total, judged, unparseable, unparseable_rate,
        correct, incorrect, not_attempted,
        correct_rate, incorrect_rate, not_attempted_rate,
        attempted_rate, accuracy_given_attempted, f1
    """
    prefix = f"{stage}/eval"
    wb_metrics: dict[str, Any] = {}

    scalar_keys = (
        "total", "judged", "unparseable", "unparseable_rate",
        "correct", "incorrect", "not_attempted",
        "correct_rate", "incorrect_rate", "not_attempted_rate",
        "attempted_rate", "accuracy_given_attempted", "f1",
    )
    for key in scalar_keys:
        if key in metrics:
            wb_metrics[f"{prefix}/{key}"] = metrics[key]

    if wb_metrics:
        logger.log_metrics(wb_metrics)

    # Structured stdout block for SLURM .out capture.
    print(flush=True)
    print("=" * 60, flush=True)
    print("  SIMPLEQA-VERIFIED EVALUATION RESULTS", flush=True)
    print("=" * 60, flush=True)
    print(f"  Total:               {metrics.get('total', '?')}", flush=True)
    print(f"  Judged:              {metrics.get('judged', '?')}", flush=True)
    print(f"  Unparseable:         {metrics.get('unparseable', '?')} "
          f"({metrics.get('unparseable_rate', 0) * 100:.1f}%)", flush=True)
    print("-" * 60, flush=True)
    print(f"  Correct:             {metrics.get('correct', '?')} "
          f"({metrics.get('correct_rate', 0) * 100:.1f}%)", flush=True)
    print(f"  Incorrect:           {metrics.get('incorrect', '?')} "
          f"({metrics.get('incorrect_rate', 0) * 100:.1f}%)", flush=True)
    print(f"  Not attempted:       {metrics.get('not_attempted', '?')} "
          f"({metrics.get('not_attempted_rate', 0) * 100:.1f}%)", flush=True)
    print("-" * 60, flush=True)
    print(f"  Attempted rate:      {metrics.get('attempted_rate', 0):.4f}", flush=True)
    print(f"  Acc | attempted:     {metrics.get('accuracy_given_attempted', 0):.4f}", flush=True)
    print(f"  SimpleQA F1:         {metrics.get('f1', 0):.4f}", flush=True)
    print("=" * 60, flush=True)
    print(flush=True)

ORCHESTRATOR_HOOKS = OrchestratorHooks(
    dagspace_module="dagspaces.simpleqa_verified.orchestrator",
    dagspace_name="simpleqa_verified",
    output_subdir="simpleqa_verified",
    job_prefix="SimpleQAVerified",
    config_dir=_CONF_DIR,
    log_eval_metrics=_log_eval_metrics,
    wandb_dagspace=lambda cfg: "simpleqa_verified",
    use_srun=False,
)


def run_experiment(cfg: DictConfig) -> None:
    """Execute the simpleqa_verified evaluation pipeline."""
    _run_experiment(cfg, ORCHESTRATOR_HOOKS)
