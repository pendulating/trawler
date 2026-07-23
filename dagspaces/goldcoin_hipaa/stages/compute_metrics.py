"""Compute evaluation metrics for GoldCoin HIPAA benchmark.

**Denominator semantics (2026-07-21 parity review, approved by Matt).**
Upstream ``eval/parse_eval_result.py`` never drops an unparseable
response — its fallback assigns the WRONG label
(``gt.remove(truth); random.choice(gt)`` over a binary label set is
deterministically the opposite class) and keeps the row in accuracy and
macro-F1. The headline ``accuracy`` / ``macro_f1`` (and the confusion
matrix / per-class blocks) mirror that: unparseable predictions are
substituted with the wrong label over ALL rows, with provenance
``unparseable_forced_wrong``. The former drop-unparseable behavior is
preserved as the ``accuracy_among_parseable`` diagnostic (house style:
headline = paper parity, cf. ``wiki/metric-trust.md``). Quantified
impact at flip time: 19/266 July cells had ``parseable_rate`` < 0.99
(all Gemma-4-E2B-it and GPT-OSS-20B; worst 0.715), i.e. the old
headline overstated those cells by up to ~12 points.
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

    if total == 0:
        em.emit_simple("accuracy", 0.0, n_total=0)
        em.emit_simple("macro_f1", 0.0, n_total=0)
        em.emit_simple("accuracy_among_parseable", 0.0, n_total=0)
        em.emit_raw("per_class", {})
        em.emit_raw("confusion_matrix", {})
        return em.to_dict()

    # Upstream forced-wrong substitution: an unparseable prediction becomes
    # the opposite of the ground truth (deterministic for a binary label
    # set — exactly what upstream's ``gt.remove(truth); random.choice(gt)``
    # produces) and stays in every metric below.
    def _wrong(gt: str) -> str:
        return labels[1] if gt == labels[0] else labels[0]

    true_labels = df["ground_truth"].tolist()
    predictions = [
        p if p != "unparseable" else _wrong(g)
        for g, p in zip(true_labels, df["prediction"].tolist())
    ]

    # Headline accuracy + macro F1 — paper parity: all rows, unparseable
    # counted wrong via substitution.
    em.emit(
        "accuracy",
        round(accuracy_score(true_labels, predictions), 6),
        n_total=total,
        n_real=parseable_count,
        n_defaulted=unparseable_count,
        default_reason="unparseable_forced_wrong" if unparseable_count else None,
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
        default_reason="unparseable_forced_wrong" if unparseable_count else None,
    )

    # Diagnostic: accuracy among rows the parser actually labeled.
    em.emit_simple(
        "accuracy_among_parseable",
        round(
            accuracy_score(
                df_parseable["ground_truth"].tolist(),
                df_parseable["prediction"].tolist(),
            ),
            6,
        )
        if parseable_count
        else 0.0,
        n_total=parseable_count,
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

    # Per-class correct/error counts (substituted full set, matching the
    # headline denominator).
    class_counts: dict[str, dict[str, int]] = {}
    for label in labels:
        class_total = sum(1 for t in true_labels if t == label)
        class_correct = sum(
            1 for t, p in zip(true_labels, predictions) if t == label and p == label
        )
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
