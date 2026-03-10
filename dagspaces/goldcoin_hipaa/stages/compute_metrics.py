"""Compute evaluation metrics for GoldCoin HIPAA benchmark."""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def compute_metrics(df: pd.DataFrame, task: str) -> Dict[str, Any]:
    """Compute accuracy, macro F1, per-class metrics, and confusion matrix.

    Args:
        df: DataFrame with ``ground_truth`` and ``prediction`` columns.
        task: "compliance" or "applicability".

    Returns:
        Dict with all metrics.
    """
    # Separate parseable from unparseable
    total = len(df)
    unparseable_mask = df["prediction"] == "unparseable"
    unparseable_count = int(unparseable_mask.sum())
    unparseable_rate = unparseable_count / total if total > 0 else 0.0

    # For metrics, only use parseable predictions
    df_parseable = df[~unparseable_mask].copy()
    parseable_count = len(df_parseable)

    true_labels = df_parseable["ground_truth"].tolist()
    predictions = df_parseable["prediction"].tolist()

    # Determine label set
    if task == "compliance":
        labels = ["Permit", "Forbid"]
    else:
        labels = ["Applicable", "Not Applicable"]

    metrics: Dict[str, Any] = {
        "task": task,
        "total": total,
        "parseable": parseable_count,
        "unparseable_count": unparseable_count,
        "unparseable_rate": round(unparseable_rate, 4),
    }

    if parseable_count == 0:
        metrics["accuracy"] = 0.0
        metrics["macro_f1"] = 0.0
        metrics["per_class"] = {}
        metrics["confusion_matrix"] = {}
        return metrics

    # Overall metrics
    metrics["accuracy"] = round(accuracy_score(true_labels, predictions), 4)
    metrics["macro_f1"] = round(
        f1_score(true_labels, predictions, average="macro", labels=labels, zero_division=0), 4
    )

    # Per-class metrics via classification_report
    report = classification_report(
        true_labels, predictions, labels=labels, output_dict=True, zero_division=0
    )
    per_class = {}
    for label in labels:
        if label in report:
            per_class[label] = {
                "precision": round(report[label]["precision"], 4),
                "recall": round(report[label]["recall"], 4),
                "f1": round(report[label]["f1-score"], 4),
                "support": int(report[label]["support"]),
            }
    metrics["per_class"] = per_class

    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions, labels=labels)
    cm_dict = {}
    for i, true_label in enumerate(labels):
        cm_dict[true_label] = {}
        for j, pred_label in enumerate(labels):
            cm_dict[true_label][pred_label] = int(cm[i, j])
    metrics["confusion_matrix"] = cm_dict

    # Per-class correct/error counts
    class_counts = {}
    for label in labels:
        mask = df_parseable["ground_truth"] == label
        class_total = int(mask.sum())
        class_correct = int((df_parseable.loc[mask, "prediction"] == label).sum())
        class_counts[label] = {
            "total": class_total,
            "correct": class_correct,
            "error": class_total - class_correct,
        }
    metrics["class_counts"] = class_counts

    # Full classification report as string
    metrics["classification_report"] = classification_report(
        true_labels, predictions, labels=labels, digits=4, zero_division=0
    )

    return metrics


def metrics_to_dataframe(metrics: Dict[str, Any]) -> pd.DataFrame:
    """Flatten metrics dict into a single-row DataFrame for parquet storage."""
    flat = {}
    for k, v in metrics.items():
        if isinstance(v, dict):
            flat[k] = json.dumps(v, default=str)
        elif isinstance(v, str) and "\n" in v:
            flat[k] = v
        else:
            flat[k] = v
    return pd.DataFrame([flat])
