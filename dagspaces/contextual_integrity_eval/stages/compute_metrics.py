"""Compute evaluation metrics for PrivacyLens benchmark."""

from __future__ import annotations

import json
from typing import Any, Dict

import pandas as pd


def compute_qa_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute QA probing accuracy metrics per axis and overall.

    Args:
        df: DataFrame with '_qa_axis', 'predicted_label', 'correct' columns.

    Returns:
        Dict with per-axis accuracy, overall accuracy, and unparseable rate.
    """
    total = len(df)
    unknown_count = (df["predicted_label"] == "unknown").sum()

    metrics: Dict[str, Any] = {
        "task": "qa_probing",
        "total": total,
        "unparseable_count": int(unknown_count),
        "unparseable_rate": round(unknown_count / total, 4) if total > 0 else 0.0,
    }

    # Overall accuracy
    metrics["accuracy"] = round(df["correct"].mean(), 4) if total > 0 else 0.0

    # Per-axis accuracy
    per_axis = {}
    for axis in sorted(df["_qa_axis"].unique()):
        mask = df["_qa_axis"] == axis
        axis_df = df[mask]
        axis_total = len(axis_df)
        axis_correct = axis_df["correct"].sum()
        per_axis[axis] = {
            "accuracy": round(axis_correct / axis_total, 4) if axis_total > 0 else 0.0,
            "correct": int(axis_correct),
            "total": axis_total,
            "unknown": int((axis_df["predicted_label"] == "unknown").sum()),
        }
    metrics["per_axis"] = per_axis

    # Per-axis accuracy for the "parseable only" subset
    parseable = df[df["predicted_label"] != "unknown"]
    if len(parseable) > 0:
        metrics["accuracy_parseable"] = round(parseable["correct"].mean(), 4)
    else:
        metrics["accuracy_parseable"] = 0.0

    return metrics


def compute_leakage_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute action-based leakage rate metrics.

    Args:
        df: DataFrame with 'leak_flag', 'leak_probability' columns.

    Returns:
        Dict with leakage rate, count, and probability stats.
    """
    total = len(df)
    leaking = int(df["leak_flag"].sum())
    leakage_rate = round(leaking / total, 4) if total > 0 else 0.0

    metrics: Dict[str, Any] = {
        "task": "leakage",
        "total": total,
        "leaking_count": leaking,
        "non_leaking_count": total - leaking,
        "leakage_rate": leakage_rate,
        "mean_leak_probability": round(float(df["leak_probability"].mean()), 4) if total > 0 else 0.0,
    }

    return metrics


def compute_metrics(
    qa_df: pd.DataFrame,
    leakage_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Compute all PrivacyLens evaluation metrics.

    Args:
        qa_df: QA probing results (with '_qa_axis', 'predicted_label', 'correct').
        leakage_df: Leakage results (with 'leak_flag', 'leak_probability').

    Returns:
        Combined metrics dict.
    """
    qa_metrics = compute_qa_metrics(qa_df)
    leakage_metrics = compute_leakage_metrics(leakage_df)

    return {
        "benchmark": "PrivacyLens",
        "qa_probing": qa_metrics,
        "leakage": leakage_metrics,
    }


def metrics_to_dataframe(metrics: Dict[str, Any]) -> pd.DataFrame:
    """Flatten metrics dict into a single-row DataFrame for parquet storage."""
    flat: Dict[str, Any] = {"benchmark": metrics.get("benchmark", "PrivacyLens")}

    qa = metrics.get("qa_probing", {})
    flat["qa_accuracy"] = qa.get("accuracy", 0.0)
    flat["qa_accuracy_parseable"] = qa.get("accuracy_parseable", 0.0)
    flat["qa_total"] = qa.get("total", 0)
    flat["qa_unparseable_count"] = qa.get("unparseable_count", 0)
    for axis, axis_m in qa.get("per_axis", {}).items():
        flat[f"qa_{axis}_accuracy"] = axis_m.get("accuracy", 0.0)
        flat[f"qa_{axis}_total"] = axis_m.get("total", 0)

    leak = metrics.get("leakage", {})
    flat["leakage_rate"] = leak.get("leakage_rate", 0.0)
    flat["leaking_count"] = leak.get("leaking_count", 0)
    flat["leakage_total"] = leak.get("total", 0)
    flat["mean_leak_probability"] = leak.get("mean_leak_probability", 0.0)

    flat["metrics_json"] = json.dumps(metrics, default=str)

    return pd.DataFrame([flat])
