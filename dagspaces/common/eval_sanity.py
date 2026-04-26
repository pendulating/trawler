"""Evaluation-stage sanity checks.

Catches silent metric corruption: parse-stage warnings that today get
buried inside ``compute_metrics`` outputs without surfacing to the
sweep level. A model that refuses 40% of the time produces a
"complete" eval whose accuracy is computed on the remaining 60% with
no flag — this module makes that visible.

Two entry points:

- :func:`compute_parse_health` — for task-LLM parse stages
  (``parse_responses``, ``parse_mcq``, etc.). Reports parseable rate,
  truncation rate (when ``finish_reason`` is captured), empty /
  refusal rates, schema-violation rate, label class balance, row-count
  drop vs. expected, and duplicate id counts.

- :func:`compute_judge_health` — for LLM-judge stages. Reports
  judge-label distribution entropy, unparseable rate, and
  per-secret-skip rate (privacylens leakage's per-secret fan-out).

Both produce a :class:`SanityReport` whose ``failure_rows`` DataFrame
follows the canonical :data:`FAILURE_ROW_COLUMNS` schema, shared with
the async-judge sidecar's per-row failure log so a unified
"all failures across this sweep" filter works in W&B.

Warnings are warnings, not errors — pipelines never fail on a sanity
threshold cross. They print a loud banner to stderr and surface as a
``sanity/<stage>/warnings`` W&B table so the sweep summary can flag
them.
"""

from __future__ import annotations

import math
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd


# ---------------------------------------------------------------------------
# Shared per-row failure schema
# ---------------------------------------------------------------------------

#: Canonical column order for the per-row failure log table. Both parse
#: stages (via :func:`compute_parse_health`) and the async-judge sidecar
#: write rows in this schema, so a single W&B filter unifies them.
FAILURE_ROW_COLUMNS: List[str] = [
    "custom_id",
    "dagspace",
    "stage",
    "failure_type",
    "raw_response_preview",
    "parse_error",
    "model",
    "manifest_path",
    "attempt_count",
    "error_preview",
    "last_attempt_at",
]

#: Maximum rows kept in any single ``failure_rows`` DataFrame. Overflow
#: increments a ``failures_dropped`` counter on the report instead.
FAILURE_ROW_CAP: int = 1000

#: Maximum chars kept from a raw completion / judge response in a
#: failure-row preview. Keeps W&B tables small while preserving enough
#: context to triage the failure mode.
RAW_PREVIEW_CHARS: int = 500


# ---------------------------------------------------------------------------
# Default refusal patterns (each benchmark may extend or replace)
# ---------------------------------------------------------------------------

#: Conservative shared defaults — patterns that are very likely a
#: refusal regardless of benchmark. Each benchmark's parse stage should
#: pass its own ``refusal_patterns`` list via cfg, optionally extending
#: this set.
DEFAULT_REFUSAL_PATTERNS: List[str] = [
    r"\bI cannot\b",
    r"\bI can't\b",
    r"\bI won't\b",
    r"\bI will not\b",
    r"\bAs an AI\b",
    r"\bI'm unable\b",
    r"\bagainst my (?:guidelines|policy|policies)\b",
    r"\bI must decline\b",
]


# ---------------------------------------------------------------------------
# Default thresholds
# ---------------------------------------------------------------------------

#: Warning thresholds. ``"<metric>:gt"`` means warn when value > threshold;
#: ``"<metric>:lt"`` means warn when value < threshold. Each benchmark's
#: ``conf/config.yaml`` may override individual entries.
DEFAULT_THRESHOLDS: Dict[str, float] = {
    "parseable_rate:lt": 0.95,
    "truncated_rate:gt": 0.02,
    "empty_response_rate:gt": 0.005,
    "refusal_rate:gt": 0.02,
    "schema_violation_rate:gt": 0.01,
    "class_balance_min:lt": 0.05,
    "row_count_drop:lt": 0.99,
    "duplicate_id_count:gt": 0,
    "judge_unparseable_rate:gt": 0.01,
    "judge_per_secret_skip_rate:gt": 0.02,
    "judge_label_entropy:lt": 0.30,
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SanityWarning:
    """A single threshold violation, ready to log as a W&B table row."""

    metric: str
    value: float
    threshold: float
    comparison: str  # "gt" (warn when value > threshold) or "lt"
    severity: str = "warn"

    def message(self) -> str:
        sym = ">" if self.comparison == "gt" else "<"
        return f"{self.metric}={self.value:.4f} {sym} {self.threshold} (warn)"


@dataclass
class SanityReport:
    """Result of a sanity check on one stage's output."""

    dagspace: str
    stage: str
    metrics: Dict[str, float] = field(default_factory=dict)
    warnings: List[SanityWarning] = field(default_factory=list)
    failure_rows: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=FAILURE_ROW_COLUMNS))
    failures_dropped: int = 0
    n_rows: int = 0

    def has_warnings(self) -> bool:
        return len(self.warnings) > 0

    def worst_warning(self) -> Optional[SanityWarning]:
        if not self.warnings:
            return None
        # Severity ranking: ratio of (value − threshold) magnitude to threshold,
        # i.e. how far past the line we are. Larger ⇒ worse.
        def _badness(w: SanityWarning) -> float:
            if w.threshold == 0:
                return abs(w.value)
            return abs(w.value - w.threshold) / max(abs(w.threshold), 1e-9)
        return max(self.warnings, key=_badness)

    def print_loud(self, *, prefix: str = "") -> None:
        """Print a loud banner to stderr summarizing any warnings."""
        if not self.has_warnings():
            return
        bar = "!" * 70
        head = f"{prefix}SANITY WARNINGS — {self.dagspace}.{self.stage} ({len(self.warnings)} threshold(s))"
        lines = ["", bar, head, bar]
        for w in self.warnings:
            lines.append(f"  - {w.message()}")
        if self.failures_dropped > 0:
            lines.append(
                f"  + {self.failures_dropped} additional failure rows dropped "
                f"(failure_rows capped at {FAILURE_ROW_CAP})"
            )
        lines.append(bar)
        print("\n".join(lines), file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_threshold(
    metric: str,
    comparison: str,
    overrides: Optional[Dict[str, float]],
) -> Optional[float]:
    key = f"{metric}:{comparison}"
    if overrides and key in overrides:
        return float(overrides[key])
    return DEFAULT_THRESHOLDS.get(key)


def _emit_warning(
    report: SanityReport,
    metric: str,
    value: float,
    comparison: str,
    overrides: Optional[Dict[str, float]],
) -> None:
    threshold = _resolve_threshold(metric, comparison, overrides)
    if threshold is None:
        return
    cross = (value > threshold) if comparison == "gt" else (value < threshold)
    if cross:
        report.warnings.append(
            SanityWarning(
                metric=metric,
                value=float(value),
                threshold=float(threshold),
                comparison=comparison,
            )
        )


def _truncate(text: Any, n: int = RAW_PREVIEW_CHARS) -> str:
    s = "" if text is None else str(text)
    if len(s) <= n:
        return s
    return s[:n] + f"…(+{len(s) - n} chars)"


def _compile_refusal_patterns(patterns: Optional[Sequence[str]]) -> List[re.Pattern]:
    pats = list(patterns) if patterns is not None else DEFAULT_REFUSAL_PATTERNS
    out: List[re.Pattern] = []
    for p in pats:
        try:
            out.append(re.compile(p, re.IGNORECASE))
        except re.error:
            # Treat as literal if the regex is malformed.
            out.append(re.compile(re.escape(p), re.IGNORECASE))
    return out


def _matches_any(text: str, patterns: List[re.Pattern]) -> bool:
    if not text:
        return False
    return any(p.search(text) for p in patterns)


def _build_failure_rows(
    flagged: pd.DataFrame,
    *,
    dagspace: str,
    stage: str,
    model: str,
    id_col: Optional[str],
    completion_col: Optional[str],
    failure_type_col: str = "_sanity_failure_type",
    parse_error_col: Optional[str] = None,
) -> tuple[pd.DataFrame, int]:
    """Materialize a capped failure-row table in the shared schema."""
    n = len(flagged)
    dropped = max(0, n - FAILURE_ROW_CAP)
    sub = flagged.head(FAILURE_ROW_CAP)
    rows = []
    now = datetime.now(timezone.utc).isoformat()
    for i, (_, row) in enumerate(sub.iterrows()):
        cid = row.get(id_col) if id_col and id_col in sub.columns else f"row_{i}"
        completion = row.get(completion_col, "") if completion_col else ""
        rows.append({
            "custom_id": str(cid),
            "dagspace": dagspace,
            "stage": stage,
            "failure_type": str(row.get(failure_type_col, "unknown")),
            "raw_response_preview": _truncate(completion),
            "parse_error": str(row.get(parse_error_col, "") or "") if parse_error_col else "",
            "model": str(model or ""),
            "manifest_path": "",
            "attempt_count": 1,
            "error_preview": "",
            "last_attempt_at": now,
        })
    df = pd.DataFrame(rows, columns=FAILURE_ROW_COLUMNS) if rows else pd.DataFrame(columns=FAILURE_ROW_COLUMNS)
    return df, dropped


# ---------------------------------------------------------------------------
# Parse-stage health
# ---------------------------------------------------------------------------

def compute_parse_health(
    df: pd.DataFrame,
    *,
    dagspace: str,
    stage: str,
    model: str = "",
    status_col: str = "parse_status",
    completion_col: str = "generated_text",
    label_col: Optional[str] = None,
    id_col: Optional[str] = None,
    finish_reason_col: Optional[str] = "finish_reason",
    expected_input_n: Optional[int] = None,
    refusal_patterns: Optional[Sequence[str]] = None,
    thresholds: Optional[Dict[str, float]] = None,
    parsed_status_value: str = "parsed",
    schema_violation_status_value: str = "schema_violation",
) -> SanityReport:
    """Compute health metrics for a task-LLM parse stage's output.

    The parse stage is expected to have populated a ``status_col``
    column (default ``parse_status``) with values like ``"parsed"``
    (success), ``"unparseable"``, ``"empty"``, ``"refusal"``,
    ``"schema_violation"``, or ``"fallback_default"``. Any value other
    than ``parsed_status_value`` is counted as a parse failure and
    contributes a row to ``failure_rows``.

    Args:
        df: Parse-stage output DataFrame. Must contain ``completion_col``;
            other column references are conditional on presence.
        dagspace: e.g. ``"privacylens"``.
        stage: e.g. ``"parse_responses"``, ``"parse_qa_responses"``.
        model: Task-LLM identifier (for the failure-row ``model`` column).
        status_col: Column name carrying per-row parse status values.
        completion_col: Column with the raw LLM output (for empty /
            refusal detection and failure previews).
        label_col: Optional parsed-label column; presence enables the
            class-balance check.
        id_col: Optional row-identifier column; presence enables the
            duplicate-id check and populates ``custom_id`` on failures.
        finish_reason_col: Column with vLLM's ``finish_reason`` per row;
            presence enables the truncation check.
        expected_input_n: Expected row count entering the parse stage;
            presence enables the row-count-drop check.
        refusal_patterns: Override the default refusal regex list.
        thresholds: Override default ``DEFAULT_THRESHOLDS`` entries.
        parsed_status_value: Value of ``status_col`` that means "parse
            succeeded." Anything else counts as a failure.
        schema_violation_status_value: Specific status value treated as
            a schema violation for the dedicated metric.
    """
    report = SanityReport(dagspace=dagspace, stage=stage, n_rows=len(df))
    n = len(df)
    if n == 0:
        return report

    refusal_pats = _compile_refusal_patterns(refusal_patterns)

    # ---- parseable_rate ---------------------------------------------------
    if status_col in df.columns:
        parsed_mask = df[status_col].astype(str) == parsed_status_value
        parseable_rate = float(parsed_mask.sum()) / n
    else:
        # No parse_status column → assume everything parsed; nothing useful
        # to flag besides empty / refusal patterns.
        parsed_mask = pd.Series([True] * n, index=df.index)
        parseable_rate = 1.0
    report.metrics["parseable_rate"] = round(parseable_rate, 6)
    _emit_warning(report, "parseable_rate", parseable_rate, "lt", thresholds)

    # ---- schema_violation_rate -------------------------------------------
    if status_col in df.columns:
        schema_v = float((df[status_col].astype(str) == schema_violation_status_value).sum()) / n
        report.metrics["schema_violation_rate"] = round(schema_v, 6)
        _emit_warning(report, "schema_violation_rate", schema_v, "gt", thresholds)

    # ---- empty_response_rate ---------------------------------------------
    if completion_col in df.columns:
        empty_mask = df[completion_col].fillna("").astype(str).str.strip() == ""
        empty_rate = float(empty_mask.sum()) / n
        report.metrics["empty_response_rate"] = round(empty_rate, 6)
        _emit_warning(report, "empty_response_rate", empty_rate, "gt", thresholds)
    else:
        empty_mask = pd.Series([False] * n, index=df.index)

    # ---- refusal_rate ----------------------------------------------------
    if completion_col in df.columns:
        refusal_mask = df[completion_col].fillna("").astype(str).map(
            lambda t: _matches_any(t, refusal_pats)
        )
        refusal_rate = float(refusal_mask.sum()) / n
        report.metrics["refusal_rate"] = round(refusal_rate, 6)
        _emit_warning(report, "refusal_rate", refusal_rate, "gt", thresholds)
    else:
        refusal_mask = pd.Series([False] * n, index=df.index)

    # ---- truncated_rate (conditional on finish_reason being present) -----
    if finish_reason_col and finish_reason_col in df.columns:
        truncated_mask = df[finish_reason_col].astype(str).str.lower() == "length"
        truncated_rate = float(truncated_mask.sum()) / n
        report.metrics["truncated_rate"] = round(truncated_rate, 6)
        _emit_warning(report, "truncated_rate", truncated_rate, "gt", thresholds)

    # ---- class_balance_min (conditional on label column) -----------------
    if label_col and label_col in df.columns:
        labels = df[label_col].dropna()
        if len(labels) > 0:
            counts = labels.value_counts(normalize=True)
            min_share = float(counts.min())
            report.metrics["class_balance_min"] = round(min_share, 6)
            _emit_warning(report, "class_balance_min", min_share, "lt", thresholds)

    # ---- row_count_drop --------------------------------------------------
    if expected_input_n is not None and expected_input_n > 0:
        ratio = n / float(expected_input_n)
        report.metrics["row_count_drop"] = round(ratio, 6)
        _emit_warning(report, "row_count_drop", ratio, "lt", thresholds)

    # ---- duplicate_id_count ---------------------------------------------
    if id_col and id_col in df.columns:
        dup = int(df[id_col].duplicated().sum())
        report.metrics["duplicate_id_count"] = float(dup)
        _emit_warning(report, "duplicate_id_count", dup, "gt", thresholds)

    # ---- failure rows ----------------------------------------------------
    # Anything where parse_status != parsed, or completion is empty, or
    # the row matches a refusal pattern. We tag with the most-specific
    # failure_type we can.
    failed_idx = ~parsed_mask | empty_mask | refusal_mask
    if status_col not in df.columns:
        # Without parse_status, only empty/refusal contribute to failures.
        failed_idx = empty_mask | refusal_mask
    if failed_idx.any():
        flagged = df.loc[failed_idx].copy()
        # Discriminate failure_type
        def _ftype(row) -> str:
            if status_col in df.columns:
                s = str(row.get(status_col, "")).strip()
                if s and s != parsed_status_value:
                    return s
            comp = str(row.get(completion_col, "")) if completion_col in df.columns else ""
            if not comp.strip():
                return "empty"
            if _matches_any(comp, refusal_pats):
                return "refusal"
            return "unknown"
        flagged["_sanity_failure_type"] = flagged.apply(_ftype, axis=1)
        rows_df, dropped = _build_failure_rows(
            flagged,
            dagspace=dagspace,
            stage=stage,
            model=model,
            id_col=id_col,
            completion_col=completion_col,
        )
        report.failure_rows = rows_df
        report.failures_dropped = dropped

    return report


# ---------------------------------------------------------------------------
# Judge-stage health
# ---------------------------------------------------------------------------

def compute_judge_health(
    df: pd.DataFrame,
    *,
    dagspace: str,
    stage: str,
    judge_model: str = "",
    label_col: str,
    valid_labels: Sequence[Any],
    raw_response_col: Optional[str] = None,
    id_col: Optional[str] = None,
    skipped_input_n: Optional[int] = None,
    thresholds: Optional[Dict[str, float]] = None,
) -> SanityReport:
    """Compute health metrics for an LLM-judge stage's output.

    Args:
        df: Judge-stage output DataFrame.
        dagspace: e.g. ``"privacylens"``.
        stage: e.g. ``"leakage_judge_inference"``.
        judge_model: Judge LLM identifier (for failure rows).
        label_col: Column with the parsed canonical label per row
            (e.g. ``"leak_flag"``, ``"helpfulness_score"``).
        valid_labels: Iterable of values considered valid for the task.
            Rows whose ``label_col`` value is not in this set count as
            unparseable.
        raw_response_col: Optional raw judge response column (for
            failure-row previews).
        id_col: Optional row-identifier column; populates ``custom_id``.
        skipped_input_n: When the judge fan-out drops rows on purpose
            (e.g. privacylens leakage's "no Action: substring" rule),
            pass ``len(skipped)`` here to surface the skip rate.
        thresholds: Override default ``DEFAULT_THRESHOLDS`` entries.
    """
    report = SanityReport(dagspace=dagspace, stage=stage, n_rows=len(df))
    n = len(df)
    if n == 0:
        return report

    valid_set = set(valid_labels)

    # ---- judge_unparseable_rate ------------------------------------------
    if label_col in df.columns:
        unparseable_mask = ~df[label_col].isin(valid_set)
        unparseable_rate = float(unparseable_mask.sum()) / n
        report.metrics["judge_unparseable_rate"] = round(unparseable_rate, 6)
        _emit_warning(report, "judge_unparseable_rate", unparseable_rate, "gt", thresholds)
    else:
        unparseable_mask = pd.Series([False] * n, index=df.index)

    # ---- judge_label_entropy --------------------------------------------
    if label_col in df.columns and len(df[label_col].dropna()) > 0:
        counts = df[label_col].dropna().value_counts(normalize=True)
        # Entropy in nats, normalized to [0, log(k)] then to [0, 1] by
        # dividing by log(k). 0 = degenerate, 1 = uniform.
        k = max(len(counts), 1)
        if k > 1:
            ent_nats = -sum(p * math.log(p) for p in counts if p > 0)
            ent_norm = ent_nats / math.log(k)
        else:
            ent_norm = 0.0
        report.metrics["judge_label_entropy"] = round(ent_norm, 6)
        _emit_warning(report, "judge_label_entropy", ent_norm, "lt", thresholds)

    # ---- judge_per_secret_skip_rate -------------------------------------
    if skipped_input_n is not None and (skipped_input_n + n) > 0:
        skip_rate = float(skipped_input_n) / float(skipped_input_n + n)
        report.metrics["judge_per_secret_skip_rate"] = round(skip_rate, 6)
        _emit_warning(report, "judge_per_secret_skip_rate", skip_rate, "gt", thresholds)

    # ---- failure rows ----------------------------------------------------
    if unparseable_mask.any():
        flagged = df.loc[unparseable_mask].copy()
        flagged["_sanity_failure_type"] = "judge_unparseable"
        rows_df, dropped = _build_failure_rows(
            flagged,
            dagspace=dagspace,
            stage=stage,
            model=judge_model,
            id_col=id_col,
            completion_col=raw_response_col,
        )
        report.failure_rows = rows_df
        report.failures_dropped = dropped

    return report
