"""Orchestrator for the vlm_geoprivacy_bench dagspace.

Thin wrapper over the shared eval run loop in ``dagspaces/common/orchestrator.py``
(see ``wiki/jul19_orchestrator_unification_plan.md``).  The only vlm_geoprivacy_bench-specific
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
# ``importlib.import_module("dagspaces.vlm_geoprivacy_bench.orchestrator").get_stage_registry()``.
from .runners import get_stage_registry  # noqa: F401

_CONF_DIR = os.path.join(os.path.dirname(__file__), "conf")



def _log_eval_metrics(logger, metrics: dict[str, Any], stage: str) -> None:
    """Log VLM-GeoPrivacyBench evaluation metrics to W&B and print a structured summary.

    Flattens per-question metrics, Q7 directionality stats, confusion matrix,
    and subgroup analysis into W&B-friendly scalar keys.
    """
    prefix = f"{stage}/eval"
    wb_metrics: dict[str, Any] = {}

    wb_metrics[f"{prefix}/n_samples"] = metrics.get("n_samples", 0)

    # Per-question accuracy and F1
    for q, q_m in metrics.get("per_question", {}).items():
        for k in ("accuracy", "f1_macro"):
            if k in q_m:
                wb_metrics[f"{prefix}/{q}/{k}"] = q_m[k]
        # Q7 directionality stats
        for k in ("over_disclosure_rate", "under_disclosure_rate", "mae",
                   "mae_over", "mae_under", "abstention_violation_rate"):
            if k in q_m:
                wb_metrics[f"{prefix}/{q}/{k}"] = q_m[k]
        # Confusion matrix cells (Q7 only, labels A/B/C)
        cm = q_m.get("confusion_matrix")
        if cm and isinstance(cm, list):
            labels = ["A", "B", "C"]
            for i, true_label in enumerate(labels):
                for j, pred_label in enumerate(labels):
                    if i < len(cm) and j < len(cm[i]):
                        wb_metrics[f"{prefix}/{q}/cm/{true_label}_pred_{pred_label}"] = cm[i][j]
        # Error distribution
        err_dist = q_m.get("error_distribution")
        if err_dist and isinstance(err_dist, dict):
            for bucket, count in err_dist.get("counts", {}).items():
                wb_metrics[f"{prefix}/{q}/error_dist/{bucket}"] = count

    # Subgroup metrics
    for sg_name, sg_m in metrics.get("subgroups", {}).items():
        for k, v in sg_m.items():
            if isinstance(v, (int, float)):
                wb_metrics[f"{prefix}/subgroup/{sg_name}/{k}"] = v

    if wb_metrics:
        logger.log_metrics(wb_metrics)

    # Structured log output
    questions = sorted(metrics.get("per_question", {}).keys(),
                       key=lambda x: int(x[1:]) if x[1:].isdigit() else 99)
    free_form = questions == ["Q7"]

    print(flush=True)
    print("=" * 64, flush=True)
    print(f"  VLM-GEOPRIVACYBENCH RESULTS ({'free-form' if free_form else 'MCQ'})"
          f"  [n={metrics.get('n_samples', '?')}]", flush=True)
    print("=" * 64, flush=True)

    for q in questions:
        q_m = metrics["per_question"][q]
        acc = q_m.get("accuracy", "?")
        f1 = q_m.get("f1_macro", "?")
        line = f"  {q}: accuracy={acc}, F1(macro)={f1}"
        print(line, flush=True)
        if q == "Q7":
            for k in ("over_disclosure_rate", "under_disclosure_rate", "mae",
                       "abstention_violation_rate"):
                if k in q_m:
                    print(f"       {k}: {q_m[k]}", flush=True)
            cm = q_m.get("confusion_matrix")
            if cm and isinstance(cm, list):
                labels = ["A", "B", "C"]
                print("       Confusion Matrix (rows=true, cols=pred):", flush=True)
                header = "".ljust(10) + "".join(f"{l:>8s}" for l in labels)
                print(f"       {header}", flush=True)
                for i, tl in enumerate(labels):
                    if i < len(cm):
                        row_vals = "".join(f"{cm[i][j]:>8d}" for j in range(len(labels)) if j < len(cm[i]))
                        print(f"       {tl:>8s}{row_vals}", flush=True)

    if "subgroups" in metrics:
        print("-" * 64, flush=True)
        print("  Subgroup analysis:", flush=True)
        for sg, sg_m in metrics["subgroups"].items():
            parts = [f"n={sg_m.get('n')}",
                     f"acc={sg_m.get('accuracy')}",
                     f"over={sg_m.get('over_disclosure_rate')}",
                     f"under={sg_m.get('under_disclosure_rate')}"]
            print(f"    {sg}: {', '.join(parts)}", flush=True)

    print("=" * 64, flush=True)
    print(flush=True)

ORCHESTRATOR_HOOKS = OrchestratorHooks(
    dagspace_module="dagspaces.vlm_geoprivacy_bench.orchestrator",
    dagspace_name="vlm_geoprivacy_bench",
    output_subdir="vlm_geoprivacy_bench",
    job_prefix="VLM",
    config_dir=_CONF_DIR,
    log_eval_metrics=_log_eval_metrics,
    wandb_dagspace=lambda cfg: "vlm_geoprivacy",
    use_srun=False,
)


def run_experiment(cfg: DictConfig) -> None:
    """Execute the vlm_geoprivacy_bench evaluation pipeline."""
    _run_experiment(cfg, ORCHESTRATOR_HOOKS)
