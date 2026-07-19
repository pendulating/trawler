"""Orchestrator for the privacylens dagspace.

Thin wrapper over the shared eval run loop in ``dagspaces/common/orchestrator.py``
(see ``wiki/jul19_orchestrator_unification_plan.md``).  The only privacylens-specific
code below is the metric formatter (and _perturb_qualified_dagspace); everything else is the generic
loop parameterized by ``ORCHESTRATOR_HOOKS``.
"""

from __future__ import annotations

import os
from typing import Any, Dict

from omegaconf import DictConfig, OmegaConf

from dagspaces.common.orchestrator import (
    OrchestratorHooks,
    run_experiment as _run_experiment,
)

# Re-exported for the generic SLURM worker, which recovers the registry via
# ``importlib.import_module("dagspaces.privacylens.orchestrator").get_stage_registry()``.
from .runners import get_stage_registry  # noqa: F401

_CONF_DIR = os.path.join(os.path.dirname(__file__), "conf")


def _perturb_qualified_dagspace(cfg: DictConfig) -> str:
    """Dagspace key for W&B run-id derivation, qualified by perturb culture.

    A cultural sweep's per-culture runs share (model, group), so without
    qualification their resumable run ids would collide into one run. Folding
    ``perturb.culture`` into the dagspace gives each (model, culture) its own
    run id, while export and finalize for a culture share it (both receive
    ``+perturb.culture=<c>``). Absent perturb.culture this is exactly
    ``"privacylens"`` — no change for existing runs.
    """
    culture = OmegaConf.select(cfg, "perturb.culture")
    return f"privacylens:{culture}" if culture else "privacylens"


def _log_eval_metrics(logger, metrics: Dict[str, Any], stage: str) -> None:
    """Log PrivacyLens evaluation metrics to W&B."""
    prefix = f"{stage}/eval"
    wb_metrics: Dict[str, Any] = {}

    # QA probing metrics
    qa = metrics.get("qa_probing", {})
    if qa:
        wb_metrics[f"{prefix}/qa_accuracy"] = qa.get("accuracy", 0.0)
        wb_metrics[f"{prefix}/qa_accuracy_parseable"] = qa.get("accuracy_parseable", 0.0)
        wb_metrics[f"{prefix}/qa_total"] = qa.get("total", 0)
        wb_metrics[f"{prefix}/qa_unparseable_count"] = qa.get("unparseable_count", 0)
        for axis, axis_m in qa.get("per_axis", {}).items():
            wb_metrics[f"{prefix}/qa_{axis}_accuracy"] = axis_m.get("accuracy", 0.0)
            wb_metrics[f"{prefix}/qa_{axis}_total"] = axis_m.get("total", 0)

    # Leakage metrics
    leak = metrics.get("leakage", {})
    if leak:
        wb_metrics[f"{prefix}/leakage_rate"] = leak.get("leakage_rate", 0.0)
        wb_metrics[f"{prefix}/leaking_count"] = leak.get("leaking_count", 0)
        wb_metrics[f"{prefix}/leakage_total"] = leak.get("total", 0)
        wb_metrics[f"{prefix}/mean_leak_probability"] = leak.get("mean_leak_probability", 0.0)

    # Helpfulness metrics
    helpfulness = metrics.get("helpfulness", {})
    if helpfulness:
        wb_metrics[f"{prefix}/helpful_rate"] = helpfulness.get("helpful_rate", 0.0)
        wb_metrics[f"{prefix}/helpfulness_mean_score"] = helpfulness.get("mean_score", 0.0)
        wb_metrics[f"{prefix}/helpful_count"] = helpfulness.get("helpful_count", 0)
        wb_metrics[f"{prefix}/helpfulness_total"] = helpfulness.get("total", 0)

    # Adjusted leakage (leakage among helpful responses only)
    adj = metrics.get("adjusted_leakage", {})
    if adj:
        wb_metrics[f"{prefix}/adjusted_leakage_rate"] = adj.get("adjusted_leakage_rate", 0.0)
        wb_metrics[f"{prefix}/adjusted_leakage_total_helpful"] = adj.get("total_helpful", 0)
        wb_metrics[f"{prefix}/adjusted_leakage_leaking_among_helpful"] = adj.get("leaking_among_helpful", 0)

    if wb_metrics:
        logger.log_metrics(wb_metrics)

ORCHESTRATOR_HOOKS = OrchestratorHooks(
    dagspace_module="dagspaces.privacylens.orchestrator",
    dagspace_name="privacylens",
    output_subdir="privacylens_eval",
    job_prefix="PLens",
    config_dir=_CONF_DIR,
    log_eval_metrics=_log_eval_metrics,
    wandb_dagspace=_perturb_qualified_dagspace,
    use_srun=False,
)


def run_experiment(cfg: DictConfig) -> None:
    """Execute the privacylens evaluation pipeline."""
    _run_experiment(cfg, ORCHESTRATOR_HOOKS)
