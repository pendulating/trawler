"""Compute MMLU metrics from a parsed verdict DataFrame.

Three levels of accuracy:

1. **overall_accuracy** — across all 57 subjects.
2. **by_category** — STEM / humanities / social_sciences / other, the
   four-bucket grouping every published MMLU number uses.
3. **per_subject** — one entry per subject (57 of them on the canonical
   cais/mmlu test split).

Unparseable rows are EXCLUDED from accuracy denominators by default and
surfaced as ``unparseable_rate``. The provenance block records
``n_total / n_real / n_defaulted`` for every scalar so a reader can
audit whether the headline number rested on a parseable majority.
"""

from __future__ import annotations

import json
from typing import Any, Dict

import pandas as pd

from dagspaces.common.metric_provenance import MetricEmitter

from ..subject_categories import CATEGORIES, category_for


def _acc(correct: int, total: int) -> float:
    return round(correct / total, 6) if total else 0.0


def compute_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Return the MMLU metric dict.

    Args:
        df: Must have ``answer`` (ground-truth int 0-3), ``prediction``
            (int 0-3 or -1 for unparseable), ``is_correct`` (bool),
            ``parse_status``, and ``subject``.
    """
    em = MetricEmitter()

    total = len(df)
    em.emit_raw("total", int(total))

    unparseable_mask = df["prediction"] == -1
    unparseable_count = int(unparseable_mask.sum())
    df_parseable = df[~unparseable_mask].copy()
    parseable_count = len(df_parseable)

    em.emit_raw("parseable", parseable_count)
    em.emit_raw("unparseable_count", unparseable_count)
    em.emit_simple(
        "unparseable_rate",
        _acc(unparseable_count, total),
        n_total=total,
    )

    # ── overall ───────────────────────────────────────────────────────
    overall_correct = int(df_parseable["is_correct"].sum())
    em.emit(
        "overall_accuracy",
        _acc(overall_correct, parseable_count),
        n_total=total,
        n_real=parseable_count,
        n_defaulted=unparseable_count,
        default_reason="unparseable_dropped" if unparseable_count else None,
    )

    # ── by category (STEM / humanities / social_sciences / other) ────
    df_parseable["category"] = df_parseable["subject"].apply(category_for)
    cat_breakdown: Dict[str, Dict[str, Any]] = {}
    for cat in CATEGORIES:
        sub = df_parseable[df_parseable["category"] == cat]
        n = len(sub)
        n_correct = int(sub["is_correct"].sum())
        # Total in this category (including unparseable) so attempted-rate
        # is interpretable.
        cat_total_inc_unp = int(
            (df["subject"].apply(category_for) == cat).sum()
        )
        cat_breakdown[cat] = {
            "accuracy": _acc(n_correct, n),
            "correct": n_correct,
            "total": n,
            "total_including_unparseable": cat_total_inc_unp,
        }
    em.emit_raw("by_category", cat_breakdown)

    # ── per subject (57 rows) ────────────────────────────────────────
    per_subject: Dict[str, Dict[str, Any]] = {}
    for subj, sub in df_parseable.groupby("subject"):
        n = len(sub)
        n_correct = int(sub["is_correct"].sum())
        subj_total_inc_unp = int((df["subject"] == subj).sum())
        per_subject[str(subj)] = {
            "accuracy": _acc(n_correct, n),
            "correct": n_correct,
            "total": n,
            "total_including_unparseable": subj_total_inc_unp,
            "category": category_for(subj),
        }
    em.emit_raw("per_subject", per_subject)

    return em.to_dict()


def metrics_to_dataframe(metrics: Dict[str, Any]) -> pd.DataFrame:
    """Flatten metrics dict into a single-row DataFrame for parquet storage."""
    flat: Dict[str, Any] = {}
    for k, v in metrics.items():
        if isinstance(v, dict):
            flat[k] = json.dumps(v, default=str)
        elif isinstance(v, str) and "\n" in v:
            flat[k] = v
        else:
            flat[k] = v
    return pd.DataFrame([flat])
