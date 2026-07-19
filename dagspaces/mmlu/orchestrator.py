"""Orchestrator for the MMLU dagspace.

Thin wrapper over the shared eval run loop in ``dagspaces/common/orchestrator.py``
(see ``wiki/jul19_orchestrator_unification_plan.md``).  The only MMLU-specific
code is ``_log_eval_metrics``; everything else is the generic loop parameterized
by ``ORCHESTRATOR_HOOKS``.
"""

from __future__ import annotations

import os
from typing import Any, Dict

from omegaconf import DictConfig

from dagspaces.common.orchestrator import (
    OrchestratorHooks,
    run_experiment as _run_experiment,
)

# Re-exported for the generic SLURM worker, which recovers the registry via
# ``importlib.import_module("dagspaces.mmlu.orchestrator").get_stage_registry()``.
from .runners import get_stage_registry  # noqa: F401

_CONF_DIR = os.path.join(os.path.dirname(__file__), "conf")


def _log_eval_metrics(logger, metrics: Dict[str, Any], stage: str) -> None:
    """Flatten MMLU metrics to W&B-friendly scalars + print summary.

    Metric shape produced by stages/compute_metrics.py:
        total, parseable, unparseable_count, unparseable_rate,
        overall_accuracy, by_category: {STEM,humanities,social_sciences,other},
        per_subject: {<subject>: {accuracy, total, correct}}
    """
    prefix = f"{stage}/eval"
    wb_metrics: Dict[str, Any] = {}

    for key in ("total", "parseable", "unparseable_count",
                "unparseable_rate", "overall_accuracy"):
        if key in metrics:
            wb_metrics[f"{prefix}/{key}"] = metrics[key]

    # Category-level accuracies (4 numbers, the standard MMLU headline split).
    for cat, cat_m in metrics.get("by_category", {}).items():
        cat_safe = cat.replace(" ", "_").lower()
        if isinstance(cat_m, dict):
            for k, v in cat_m.items():
                wb_metrics[f"{prefix}/by_category/{cat_safe}/{k}"] = v

    # Per-subject accuracies (57 entries). Keep the flat key prefix so a
    # W&B run history filter on `eval/per_subject/*` shows every subject.
    for subj, subj_m in metrics.get("per_subject", {}).items():
        subj_safe = str(subj).replace(" ", "_").lower()
        if isinstance(subj_m, dict):
            for k, v in subj_m.items():
                wb_metrics[f"{prefix}/per_subject/{subj_safe}/{k}"] = v

    if wb_metrics:
        logger.log_metrics(wb_metrics)

    print(flush=True)
    print("=" * 60, flush=True)
    print("  MMLU EVALUATION RESULTS", flush=True)
    print("=" * 60, flush=True)
    print(f"  Total:               {metrics.get('total', '?')}", flush=True)
    print(f"  Parseable:           {metrics.get('parseable', '?')}", flush=True)
    print(f"  Unparseable:         {metrics.get('unparseable_count', '?')} "
          f"({metrics.get('unparseable_rate', 0) * 100:.1f}%)", flush=True)
    print(f"  Overall accuracy:    {metrics.get('overall_accuracy', 0):.4f}", flush=True)
    print("-" * 60, flush=True)
    print(f"  By category:", flush=True)
    for cat, cat_m in metrics.get("by_category", {}).items():
        if isinstance(cat_m, dict):
            acc = cat_m.get("accuracy", 0)
            n = cat_m.get("total", 0)
            print(f"    {cat:<20s} acc={acc:.4f} (n={n})", flush=True)
    print("=" * 60, flush=True)
    print(flush=True)


ORCHESTRATOR_HOOKS = OrchestratorHooks(
    dagspace_module="dagspaces.mmlu.orchestrator",
    dagspace_name="mmlu",
    output_subdir="mmlu",
    job_prefix="MMLU",
    config_dir=_CONF_DIR,
    log_eval_metrics=_log_eval_metrics,
    wandb_dagspace=lambda cfg: "mmlu",
    use_srun=False,
)


def run_experiment(cfg: DictConfig) -> None:
    """Execute the MMLU evaluation pipeline."""
    _run_experiment(cfg, ORCHESTRATOR_HOOKS)
