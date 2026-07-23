"""Orchestrator for the privacylens dagspace.

Thin wrapper over the shared eval run loop in ``dagspaces/common/orchestrator.py``
(see ``wiki/jul19_orchestrator_unification_plan.md``).  The only privacylens-specific
code below is the metric formatter (and _perturb_qualified_dagspace); everything else is the generic
loop parameterized by ``ORCHESTRATOR_HOOKS``.
"""

from __future__ import annotations

import os
from typing import Any

from omegaconf import DictConfig, OmegaConf

from dagspaces.common.orchestrator import (
    OrchestratorHooks,
)
from dagspaces.common.orchestrator import (
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


def _log_cirl_probing_metrics(logger, metrics: dict[str, Any], stage: str) -> None:
    """Log PrivacyLens-under-CIRL-protocol probing metrics (rejection accuracy)."""
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

    print(flush=True)
    print("=" * 60, flush=True)
    print("  CIRL-PROTOCOL PROBING RESULTS (PrivacyLens)", flush=True)
    print("=" * 60, flush=True)
    print(f"  Samples:      {metrics.get('total', '?')} total, "
          f"{metrics.get('parseable', '?')} parseable, "
          f"{metrics.get('unparseable_count', '?')} unparseable "
          f"({metrics.get('unparseable_rate', 0) * 100:.1f}%)", flush=True)
    print(f"  Rejection acc (overall): {metrics.get('accuracy', '?')}", flush=True)
    for level, level_m in metrics.get("per_level", {}).items():
        print(f"  {level:>12s}:  "
              f"{level_m.get('reject_count', '?')}/{level_m.get('total', '?')} reject  "
              f"(acc={level_m.get('accuracy', '?')})", flush=True)
    print("=" * 60, flush=True)
    print(flush=True)


def _log_cirl_trajectory_metrics(logger, metrics: dict[str, Any], stage: str) -> None:
    """Log PrivacyLens-under-CIRL-protocol trajectory I/U/C leakage metrics."""
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
    print("  CIRL-PROTOCOL TRAJECTORY RESULTS (PrivacyLens I/U/C)", flush=True)
    print("=" * 60, flush=True)
    print(f"  Cases:          {_m('total')}", flush=True)
    print(f"  Leakage rate:          {_m('leakage_rate_among_judged', 'leakage_rate')}", flush=True)
    print(f"  Adj. leakage rate:     {_m('adjusted_leakage_rate')}", flush=True)
    print(f"  Avg helpfulness:       {_m('avg_helpfulness_score_among_judged', 'avg_helpfulness_score')}", flush=True)
    print(f"  Helpful rate (>=2):    {_m('helpful_rate_among_judged', 'helpful_rate')}", flush=True)
    print("=" * 60, flush=True)
    print(flush=True)


def _log_eval_metrics(logger, metrics: dict[str, Any], stage: str) -> None:
    """Log PrivacyLens evaluation metrics to W&B."""
    # PrivacyLens-under-CIRL-protocol pipelines emit their own metric schemas
    # (ported from the retired cirl_vignettes dagspace); dispatch on task.
    task = metrics.get("task")
    if task == "cirl_trajectory":
        _log_cirl_trajectory_metrics(logger, metrics, stage)
        return
    if task == "cirl_vignettes_probing":
        _log_cirl_probing_metrics(logger, metrics, stage)
        return

    prefix = f"{stage}/eval"
    wb_metrics: dict[str, Any] = {}

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

    # Leakage metrics. Keys must match compute_metrics.py's emitted names —
    # the pre-refactor bare names (leakage_rate, leaking_count, ...) resolved
    # to the 0.0 defaults and logged constant zeros to W&B (2026-07-21
    # parity review). ``*_among_parseable`` is the primary, paper-quoted
    # variant; the ``*_overall_with_default_zero`` audit variant is logged
    # alongside under its full name.
    leak = metrics.get("leakage", {})
    if leak:
        wb_metrics[f"{prefix}/leakage_rate"] = leak.get("leakage_rate_among_parseable", 0.0)
        wb_metrics[f"{prefix}/leakage_rate_overall_with_default_zero"] = leak.get(
            "leakage_rate_overall_with_default_zero", 0.0
        )
        wb_metrics[f"{prefix}/leaking_count"] = leak.get("leaking_count_among_parseable", 0)
        wb_metrics[f"{prefix}/leakage_total"] = leak.get("total", 0)
        wb_metrics[f"{prefix}/agent_action_format_rate"] = leak.get(
            "agent_action_format_rate", 0.0
        )
        wb_metrics[f"{prefix}/mean_leak_probability"] = leak.get(
            "mean_leak_probability_among_parseable", 0.0
        )

    # Helpfulness metrics
    helpfulness = metrics.get("helpfulness", {})
    if helpfulness:
        wb_metrics[f"{prefix}/helpful_rate"] = helpfulness.get(
            "helpful_rate_among_parseable", 0.0
        )
        wb_metrics[f"{prefix}/helpful_rate_overall_with_default_zero"] = helpfulness.get(
            "helpful_rate_overall_with_default_zero", 0.0
        )
        wb_metrics[f"{prefix}/helpfulness_mean_score"] = helpfulness.get(
            "mean_score_among_parseable", 0.0
        )
        wb_metrics[f"{prefix}/helpful_count"] = helpfulness.get(
            "helpful_count_among_parseable", 0
        )
        wb_metrics[f"{prefix}/helpfulness_total"] = helpfulness.get("total", 0)

    # Adjusted leakage (leakage among helpful responses only)
    adj = metrics.get("adjusted_leakage", {})
    if adj:
        wb_metrics[f"{prefix}/adjusted_leakage_rate"] = adj.get("adjusted_leakage_rate", 0.0)
        wb_metrics[f"{prefix}/adjusted_leakage_total_helpful"] = adj.get(
            "total_helpful_and_judged", 0
        )
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
