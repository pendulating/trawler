"""Compute SimpleQA-Verified metrics from a verdict DataFrame.

Verdict labels (from :mod:`stages.judge_grade.letter_to_verdict`):
    correct        — A (gold target fully present, no contradictions)
    incorrect      — B (predicted answer contradicts the gold target)
    not_attempted  — C (predicted answer hedges / dodges)
    unparseable    — judge response neither parsable nor a bare A/B/C

Headline metrics:
    correct, incorrect, not_attempted, unparseable counts
    correct_rate, incorrect_rate, not_attempted_rate, unparseable_rate
        — fractions of *judged* rows (excluding unparseable defaulted rows)
    attempted_rate = (correct + incorrect) / judged
    accuracy_given_attempted = correct / (correct + incorrect)
    f1 = 2 * accuracy_given_attempted * attempted_rate /
         (accuracy_given_attempted + attempted_rate)
        — SimpleQA's published harmonic-mean composite

Provenance via :class:`dagspaces.common.metric_provenance.MetricEmitter`
so a reader of metrics.json can see n_total / n_real / n_defaulted for
every scalar.
"""

from __future__ import annotations

import json
from typing import Any, Dict

import pandas as pd

from dagspaces.common.metric_provenance import MetricEmitter


_VERDICTS = ("correct", "incorrect", "not_attempted")


def compute_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Return the SimpleQA metric dict for a verdict-stamped DataFrame.

    Args:
        df: Must have a ``verdict`` column ∈ {correct, incorrect,
            not_attempted, unparseable}. Other columns are ignored.
    """
    em = MetricEmitter()

    total = len(df)
    em.emit_raw("total", int(total))

    verdict_counts = df["verdict"].value_counts().to_dict() if total else {}
    n_correct = int(verdict_counts.get("correct", 0))
    n_incorrect = int(verdict_counts.get("incorrect", 0))
    n_not_attempted = int(verdict_counts.get("not_attempted", 0))
    n_unparseable = int(verdict_counts.get("unparseable", 0))
    n_judged = n_correct + n_incorrect + n_not_attempted

    em.emit_raw("correct", n_correct)
    em.emit_raw("incorrect", n_incorrect)
    em.emit_raw("not_attempted", n_not_attempted)
    em.emit_raw("unparseable", n_unparseable)
    em.emit_raw("judged", n_judged)

    def _rate(numer: int, denom: int) -> float:
        return round(numer / denom, 6) if denom else 0.0

    em.emit_simple("unparseable_rate", _rate(n_unparseable, total), n_total=total)

    # Rates are computed against *judged* rows so unparseable judge calls
    # don't silently bias the headline F1 downward (they show up as a
    # separate sanity signal).
    em.emit(
        "correct_rate", _rate(n_correct, n_judged),
        n_total=total, n_real=n_judged, n_defaulted=n_unparseable,
        default_reason="unparseable_dropped" if n_unparseable else None,
    )
    em.emit(
        "incorrect_rate", _rate(n_incorrect, n_judged),
        n_total=total, n_real=n_judged, n_defaulted=n_unparseable,
        default_reason="unparseable_dropped" if n_unparseable else None,
    )
    em.emit(
        "not_attempted_rate", _rate(n_not_attempted, n_judged),
        n_total=total, n_real=n_judged, n_defaulted=n_unparseable,
        default_reason="unparseable_dropped" if n_unparseable else None,
    )

    attempted = n_correct + n_incorrect
    attempted_rate = _rate(attempted, n_judged)
    accuracy_given_attempted = _rate(n_correct, attempted)
    em.emit(
        "attempted_rate", attempted_rate,
        n_total=total, n_real=n_judged, n_defaulted=n_unparseable,
        default_reason="unparseable_dropped" if n_unparseable else None,
    )
    em.emit(
        "accuracy_given_attempted", accuracy_given_attempted,
        n_total=total, n_real=attempted,
        n_defaulted=(n_judged - attempted) + n_unparseable,
        default_reason="not_attempted_or_unparseable" if (n_judged - attempted + n_unparseable) else None,
    )

    f1_denom = accuracy_given_attempted + attempted_rate
    f1 = round(
        2 * accuracy_given_attempted * attempted_rate / f1_denom, 6
    ) if f1_denom else 0.0
    em.emit(
        "f1", f1,
        n_total=total, n_real=n_judged, n_defaulted=n_unparseable,
        default_reason="unparseable_dropped" if n_unparseable else None,
    )

    # Topic-level breakdown when present — useful for the paper appendix
    # but never a headline metric.
    if "topic" in df.columns and df["topic"].astype(bool).any():
        topic_breakdown: Dict[str, Dict[str, Any]] = {}
        for topic, sub in df.groupby("topic"):
            if not topic:
                continue
            sub_total = len(sub)
            sub_judged = int((sub["verdict"] != "unparseable").sum())
            sub_correct = int((sub["verdict"] == "correct").sum())
            sub_attempted = int(sub["verdict"].isin(("correct", "incorrect")).sum())
            topic_breakdown[str(topic)] = {
                "total": sub_total,
                "judged": sub_judged,
                "correct": sub_correct,
                "accuracy_given_attempted": (
                    round(sub_correct / sub_attempted, 6) if sub_attempted else 0.0
                ),
            }
        em.emit_raw("per_topic", topic_breakdown)

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
