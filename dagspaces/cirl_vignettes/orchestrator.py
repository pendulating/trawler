"""Orchestrator for the cirl_vignettes dagspace.

Thin wrapper over the shared eval run loop in ``dagspaces/common/orchestrator.py``
(see ``wiki/jul19_orchestrator_unification_plan.md``).  The only cirl_vignettes-specific
code below is the metric formatter; everything else is the generic
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
# ``importlib.import_module("dagspaces.cirl_vignettes.orchestrator").get_stage_registry()``.
from .runners import get_stage_registry  # noqa: F401

_CONF_DIR = os.path.join(os.path.dirname(__file__), "conf")



def _log_eval_metrics(logger, metrics: Dict[str, Any], stage: str) -> None:
    """Log evaluation metrics to W&B and print a structured summary."""
    task = metrics.get("task", "unknown")
    if task == "cirl_trajectory":
        _log_trajectory_metrics(logger, metrics, stage)
    else:
        _log_probing_metrics(logger, metrics, stage)

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
