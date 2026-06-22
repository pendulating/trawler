"""Primary metric registry for the eval_all sweep summary.

Single source of truth for "which metric does each benchmark surface
as its headline number" — used by the eval_all_summary W&B run to
build the per-benchmark row in the sweep table without each benchmark
needing to standardize on a shared metric key.

Each benchmark contributes one or more :class:`PrimaryMetric` entries.
The summary loader walks ``benchmark_root/outputs/<subdir>/metrics.json``
for each entry and pulls out the dotted ``path`` value.

This is intentionally a thin static registry — adding a new judged
benchmark is a one-line append, and the sidecar / orchestrator never
needs to know which metric is primary, only the summary loader does.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class PrimaryMetric:
    """One headline metric to surface for a benchmark.

    Args:
        name: Display name in the summary table column (and W&B key).
        subdir: Subdirectory under ``<benchmark_root>/outputs`` holding
            ``metrics.json``. Most benchmarks use ``compute_metrics``;
            those with sub-task fan-out (goldcoin, confaide) write one
            metrics.json per sub-task into distinct subdirs.
        path: Dotted lookup into the metrics.json (e.g.
            ``"leakage.leakage_rate"`` or ``"per_question.Q7.accuracy"``).
        higher_is_better: For colorization in the summary table; the
            sidecar otherwise ignores it.
        format_spec: f-string format used when rendering for human
            consumption (e.g. ``".4f"`` for accuracy-style metrics).
    """

    name: str
    subdir: str
    path: str
    higher_is_better: bool = True
    format_spec: str = ".4f"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

#: Per-benchmark headline metrics. Adding a new judged benchmark is a
#: one-line append: e.g. for cirl_trajectory's eventual return to the
#: default eval set, append a (judge_leakage_rate, helpfulness) pair
#: under the "cirl_trajectory" key.
PRIMARY_METRICS: Dict[str, List[PrimaryMetric]] = {
    "privacylens": [
        # Lower leakage = better; higher helpfulness = better. Both
        # surface separately; the summary table shows both columns.
        PrimaryMetric(
            name="leakage_rate",
            subdir="compute_metrics",
            path="leakage.leakage_rate",
            higher_is_better=False,
        ),
        PrimaryMetric(
            name="helpfulness",
            subdir="compute_metrics",
            path="helpfulness.mean_score",
            higher_is_better=True,
            format_spec=".3f",
        ),
        PrimaryMetric(
            name="adjusted_leakage_rate",
            subdir="compute_metrics",
            path="adjusted_leakage.adjusted_leakage_rate",
            higher_is_better=False,
        ),
    ],
    "goldcoin": [
        PrimaryMetric(
            name="applicability_acc",
            subdir="compute_metrics_applicability",
            path="accuracy",
        ),
        PrimaryMetric(
            name="compliance_acc",
            subdir="compute_metrics_compliance",
            path="accuracy",
        ),
    ],
    "cirl_vignettes": [
        PrimaryMetric(
            name="accuracy",
            subdir="compute_metrics",
            path="accuracy",
        ),
    ],
    "confaide": [
        # tier 2{a,b}: pearson correlation between model and human ratings.
        PrimaryMetric(
            name="tier2a_pearson",
            subdir="compute_metrics_tier2a",
            path="pearson_r",
        ),
        PrimaryMetric(
            name="tier2b_pearson",
            subdir="compute_metrics_tier2b",
            path="pearson_r",
        ),
        # tier 3: error_rate = how often the model leaked. Lower = better.
        PrimaryMetric(
            name="tier3_info_error",
            subdir="compute_metrics_tier3_info",
            path="error_rate",
            higher_is_better=False,
        ),
        PrimaryMetric(
            name="tier3_free_error",
            subdir="compute_metrics_tier3_free",
            path="error_rate",
            higher_is_better=False,
        ),
        PrimaryMetric(
            name="tier3_control_error",
            subdir="compute_metrics_tier3_control",
            path="error_rate",
            higher_is_better=False,
        ),
        PrimaryMetric(
            name="tier3_sharing_error",
            subdir="compute_metrics_tier3_sharing",
            path="error_rate",
            higher_is_better=False,
        ),
    ],
    "vlm_geoprivacy": [
        # Q7 only — see the COLM26 paper for why Q7 is the headline
        # privacy-relevant question among the 7-question battery.
        PrimaryMetric(
            name="Q7",
            subdir="compute_metrics",
            path="per_question.Q7.accuracy",
        ),
    ],
    "simpleqa_verified": [
        # SimpleQA's published harmonic-mean composite. Higher = better
        # factual recall *and* attempted-rate; replaces accuracy as the
        # headline because the dataset penalizes both wrong answers AND
        # over-hedging that lets the model dodge.
        PrimaryMetric(
            name="f1",
            subdir="compute_metrics",
            path="f1",
            higher_is_better=True,
        ),
        # Lower = better: high not_attempted_rate signals the model
        # routinely dodges questions to avoid penalty. Both columns
        # surface together so reviewers can spot dodge-heavy strategies.
        PrimaryMetric(
            name="not_attempted_rate",
            subdir="compute_metrics",
            path="not_attempted_rate",
            higher_is_better=False,
        ),
    ],
    "mmlu": [
        # Overall 57-subject accuracy is the headline. The 4 category
        # subscores are useful but secondary — surface them via
        # eval/by_category/* in W&B rather than blowing up the summary
        # table with 4 + 57 columns.
        PrimaryMetric(
            name="overall_accuracy",
            subdir="compute_metrics",
            path="overall_accuracy",
            higher_is_better=True,
        ),
    ],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dotted_get(d: Any, path: str) -> Optional[Any]:
    """Walk ``path`` (``"a.b.c"``) through nested dicts. Returns None on miss."""
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        if part not in cur:
            return None
        cur = cur[part]
    return cur


def extract_primary_metrics(
    benchmark_root: str,
    dagspace: str,
) -> Dict[str, Optional[float]]:
    """Pull every registered primary metric for one benchmark run.

    Args:
        benchmark_root: The benchmark's per-run output directory (the
            same path the dagspace's CLI used as ``hydra.run.dir`` plus
            its inner ``output_root`` segment, e.g.
            ``<run>/privacylens/privacylens_eval``).
        dagspace: Lookup key into :data:`PRIMARY_METRICS`. Returns an
            empty dict if the benchmark isn't registered (the summary
            loader then falls back to "no headline metric available").

    Each entry maps ``<metric.name>`` → float (or None when the
    metrics.json is missing or the dotted path doesn't resolve, so the
    summary table can render a dash without raising).
    """
    out: Dict[str, Optional[float]] = {}
    entries = PRIMARY_METRICS.get(dagspace, [])
    for m in entries:
        metrics_path = os.path.join(
            benchmark_root, "outputs", m.subdir, "metrics.json"
        )
        value: Optional[float] = None
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path) as f:
                    data = json.load(f)
                raw = _dotted_get(data, m.path)
                if raw is not None:
                    value = float(raw)
            except (json.JSONDecodeError, OSError, TypeError, ValueError):
                value = None
        out[m.name] = value
    return out


def format_primary_metrics(
    values: Dict[str, Optional[float]],
    dagspace: str,
) -> Dict[str, str]:
    """Render a primary-metric dict for a human-readable summary cell."""
    entries = {m.name: m for m in PRIMARY_METRICS.get(dagspace, [])}
    out: Dict[str, str] = {}
    for name, val in values.items():
        spec = entries[name].format_spec if name in entries else ".4f"
        if val is None:
            out[name] = "—"
        else:
            try:
                out[name] = format(float(val), spec)
            except (TypeError, ValueError):
                out[name] = str(val)
    return out
