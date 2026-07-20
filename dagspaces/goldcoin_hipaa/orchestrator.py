"""Orchestrator for the goldcoin_hipaa dagspace.

Thin wrapper over the shared eval run loop in ``dagspaces/common/orchestrator.py``
(see ``wiki/jul19_orchestrator_unification_plan.md``).  The only goldcoin_hipaa-specific
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
# ``importlib.import_module("dagspaces.goldcoin_hipaa.orchestrator").get_stage_registry()``.
from .runners import get_stage_registry  # noqa: F401

_CONF_DIR = os.path.join(os.path.dirname(__file__), "conf")



def _log_eval_metrics(logger, metrics: dict[str, Any], stage: str) -> None:
    """Log evaluation metrics to W&B and print a structured summary to the log.

    Flattens the metrics dict into W&B-friendly scalar keys and logs them.
    Also prints a clear summary block to stdout for log file capture.
    """
    prefix = f"{stage}/eval"

    # Build flat dict of scalar metrics for W&B
    wb_metrics: dict[str, Any] = {}

    for key in ("accuracy", "macro_f1", "unparseable_rate", "unparseable_count",
                "total", "parseable"):
        if key in metrics:
            wb_metrics[f"{prefix}/{key}"] = metrics[key]

    # Per-class metrics
    for label, class_m in metrics.get("per_class", {}).items():
        label_safe = label.replace(" ", "_").lower()
        for k, v in class_m.items():
            wb_metrics[f"{prefix}/{label_safe}/{k}"] = v

    # Per-class counts
    for label, counts in metrics.get("class_counts", {}).items():
        label_safe = label.replace(" ", "_").lower()
        for k, v in counts.items():
            wb_metrics[f"{prefix}/{label_safe}/{k}"] = v

    # Confusion matrix as individual cells
    for true_label, row in metrics.get("confusion_matrix", {}).items():
        true_safe = true_label.replace(" ", "_").lower()
        for pred_label, count in row.items():
            pred_safe = pred_label.replace(" ", "_").lower()
            wb_metrics[f"{prefix}/cm/{true_safe}_pred_{pred_safe}"] = count

    if wb_metrics:
        logger.log_metrics(wb_metrics)

    # Structured log output (captured in SLURM .out files / stdout)
    task = metrics.get("task", "unknown")
    print(flush=True)
    print("=" * 60, flush=True)
    print(f"  EVALUATION RESULTS — {task.upper()}", flush=True)
    print("=" * 60, flush=True)
    print(f"  Samples:      {metrics.get('total', '?')} total, "
          f"{metrics.get('parseable', '?')} parseable, "
          f"{metrics.get('unparseable_count', '?')} unparseable "
          f"({metrics.get('unparseable_rate', 0) * 100:.1f}%)", flush=True)
    print(f"  Accuracy:     {metrics.get('accuracy', '?')}", flush=True)
    print(f"  Macro F1:     {metrics.get('macro_f1', '?')}", flush=True)
    print("-" * 60, flush=True)
    for label, counts in metrics.get("class_counts", {}).items():
        per_class = metrics.get("per_class", {}).get(label, {})
        print(f"  {label:>20s}:  {counts['correct']}/{counts['total']} correct  "
              f"(P={per_class.get('precision', '?')}, "
              f"R={per_class.get('recall', '?')}, "
              f"F1={per_class.get('f1', '?')})", flush=True)
    cm = metrics.get("confusion_matrix", {})
    if cm:
        labels = list(cm.keys())
        print("-" * 60, flush=True)
        print(f"  Confusion Matrix (rows=true, cols=pred):", flush=True)
        header = "".ljust(22) + "".join(f"{l:>16s}" for l in labels)
        print(f"  {header}", flush=True)
        for true_label in labels:
            row_vals = "".join(f"{cm[true_label].get(pl, 0):>16d}" for pl in labels)
            print(f"  {true_label:>20s}{row_vals}", flush=True)
    if "classification_report" in metrics:
        print("-" * 60, flush=True)
        print(metrics["classification_report"], flush=True)
    print("=" * 60, flush=True)
    print(flush=True)

ORCHESTRATOR_HOOKS = OrchestratorHooks(
    dagspace_module="dagspaces.goldcoin_hipaa.orchestrator",
    dagspace_name="goldcoin_hipaa",
    output_subdir="goldcoin_hipaa",
    job_prefix="GoldCoin",
    config_dir=_CONF_DIR,
    log_eval_metrics=_log_eval_metrics,
    wandb_dagspace=lambda cfg: "goldcoin",
    use_srun=False,
)


def run_experiment(cfg: DictConfig) -> None:
    """Execute the goldcoin_hipaa evaluation pipeline."""
    _run_experiment(cfg, ORCHESTRATOR_HOOKS)
