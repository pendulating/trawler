"""Compute evaluation metrics for GoldCoin HIPAA benchmark.

GoldCoin's parser correctly labels rows that don't yield a Permit/Forbid
or Applicable/Not-Applicable as ``"unparseable"``. ``compute_metrics``
already filters those out before computing accuracy / F1 / confusion
matrix, so unparseable rows do not silently contribute to the headline
metric. The migration here is **provenance-only**: every numeric metric
records ``n_total / n_real / n_defaulted`` so a reader of
``metrics.json`` can tell what fraction of inputs the metric was
actually computed on.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from dagspaces.common.metric_provenance import MetricEmitter


def compute_metrics(df: pd.DataFrame, task: str) -> dict[str, Any]:
    """Accuracy, macro F1, per-class metrics, and confusion matrix.

    Args:
        df: DataFrame with ``ground_truth`` and ``prediction`` columns.
        task: ``"compliance"`` or ``"applicability"``.

    Returns:
        Dict with all metrics + ``metric_provenance`` block.
    """
    em = MetricEmitter()
    em.emit_raw("task", task)

    total = len(df)
    em.emit_raw("total", int(total))

    unparseable_mask = df["prediction"] == "unparseable"
    unparseable_count = int(unparseable_mask.sum())
    df_parseable = df[~unparseable_mask].copy()
    parseable_count = len(df_parseable)

    em.emit_raw("parseable", parseable_count)
    em.emit_raw("unparseable_count", unparseable_count)
    em.emit_simple(
        "unparseable_rate",
        round(unparseable_count / total, 6) if total else 0.0,
        n_total=total,
    )
    em.emit_simple(
        "parseable_rate",
        round(parseable_count / total, 6) if total else 0.0,
        n_total=total,
    )

    if task == "compliance":
        labels = ["Permit", "Forbid"]
    else:
        labels = ["Applicable", "Not Applicable"]

    if parseable_count == 0:
        em.emit_simple("accuracy", 0.0, n_total=0)
        em.emit_simple("macro_f1", 0.0, n_total=0)
        em.emit_raw("per_class", {})
        em.emit_raw("confusion_matrix", {})
        return em.to_dict()

    true_labels = df_parseable["ground_truth"].tolist()
    predictions = df_parseable["prediction"].tolist()

    # Accuracy + macro F1 — provenance reflects that unparseable rows
    # were dropped, not zero-defaulted.
    em.emit(
        "accuracy",
        round(accuracy_score(true_labels, predictions), 6),
        n_total=total,
        n_real=parseable_count,
        n_defaulted=unparseable_count,
        default_reason="unparseable_dropped" if unparseable_count else None,
    )
    em.emit(
        "macro_f1",
        round(
            f1_score(true_labels, predictions, average="macro", labels=labels, zero_division=0),
            6,
        ),
        n_total=total,
        n_real=parseable_count,
        n_defaulted=unparseable_count,
        default_reason="unparseable_dropped" if unparseable_count else None,
    )

    # Per-class metrics via classification_report
    report = classification_report(
        true_labels, predictions, labels=labels, output_dict=True, zero_division=0
    )
    per_class: dict[str, Any] = {}
    for label in labels:
        if label in report:
            per_class[label] = {
                "precision": round(report[label]["precision"], 6),
                "recall": round(report[label]["recall"], 6),
                "f1": round(report[label]["f1-score"], 6),
                "support": int(report[label]["support"]),
            }
    em.emit_raw("per_class", per_class)

    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions, labels=labels)
    cm_dict: dict[str, dict[str, int]] = {}
    for i, true_label in enumerate(labels):
        cm_dict[true_label] = {}
        for j, pred_label in enumerate(labels):
            cm_dict[true_label][pred_label] = int(cm[i, j])
    em.emit_raw("confusion_matrix", cm_dict)

    # Per-class correct/error counts
    class_counts: dict[str, dict[str, int]] = {}
    for label in labels:
        mask = df_parseable["ground_truth"] == label
        class_total = int(mask.sum())
        class_correct = int((df_parseable.loc[mask, "prediction"] == label).sum())
        class_counts[label] = {
            "total": class_total,
            "correct": class_correct,
            "error": class_total - class_correct,
        }
    em.emit_raw("class_counts", class_counts)

    # Full classification report as string for human inspection
    em.emit_raw(
        "classification_report",
        classification_report(
            true_labels, predictions, labels=labels, digits=4, zero_division=0
        ),
    )

    return em.to_dict()


def metrics_to_dataframe(metrics: dict[str, Any]) -> pd.DataFrame:
    """Flatten metrics dict into a single-row DataFrame for parquet storage."""
    flat: dict[str, Any] = {}
    for k, v in metrics.items():
        if isinstance(v, dict):
            flat[k] = json.dumps(v, default=str)
        elif isinstance(v, str) and "\n" in v:
            flat[k] = v
        else:
            flat[k] = v
    return pd.DataFrame([flat])
