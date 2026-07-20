"""Orchestrator for the ci_heuristic dagspace.

Thin wrapper over the shared eval run loop in ``dagspaces/common/orchestrator.py``
(see ``wiki/jul19_orchestrator_unification_plan.md``).  The only ci_heuristic-specific
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
# ``importlib.import_module("dagspaces.ci_heuristic.orchestrator").get_stage_registry()``.
from .runners import get_stage_registry  # noqa: F401

_CONF_DIR = os.path.join(os.path.dirname(__file__), "conf")



def _log_eval_metrics(logger, metrics: dict[str, Any], stage: str) -> None:
    """Log ci_heuristic metrics to W&B and print a structured summary.

    Metrics dicts here are heterogeneous (traversal parse health, per-step
    scorers, probe suites), so this is a generic recursive flattener: every
    numeric leaf becomes a W&B scalar under <stage>/eval/<dotted.path>.
    """
    prefix = f"{stage}/eval"
    wb_metrics: dict[str, Any] = {}

    def _walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                if k == "metric_provenance":
                    continue
                _walk(v, f"{path}/{k}" if path else str(k))
        elif isinstance(node, bool):
            wb_metrics[f"{prefix}/{path}"] = int(node)
        elif isinstance(node, (int, float)):
            wb_metrics[f"{prefix}/{path}"] = node

    _walk(metrics, "")
    if wb_metrics:
        logger.log_metrics(wb_metrics)

    print(flush=True)
    print("=" * 64, flush=True)
    print(f"  CI-HEURISTIC RESULTS [{stage}]", flush=True)
    print("=" * 64, flush=True)
    for key in sorted(wb_metrics):
        print(f"  {key[len(prefix) + 1:]}: {wb_metrics[key]}", flush=True)
    print("=" * 64, flush=True)
    print(flush=True)

ORCHESTRATOR_HOOKS = OrchestratorHooks(
    dagspace_module="dagspaces.ci_heuristic.orchestrator",
    dagspace_name="ci_heuristic",
    output_subdir="ci_heuristic",
    job_prefix="CIH",
    config_dir=_CONF_DIR,
    log_eval_metrics=_log_eval_metrics,
    wandb_dagspace=lambda cfg: "ci_heuristic",
    use_srun=False,
)


def run_experiment(cfg: DictConfig) -> None:
    """Execute the ci_heuristic evaluation pipeline."""
    _run_experiment(cfg, ORCHESTRATOR_HOOKS)
