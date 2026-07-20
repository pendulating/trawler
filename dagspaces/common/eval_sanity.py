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
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Shared per-row failure schema
# ---------------------------------------------------------------------------

#: Canonical column order for the per-row failure log table. Both parse
#: stages (via :func:`compute_parse_health`) and the async-judge sidecar
#: write rows in this schema, so a single W&B filter unifies them.
FAILURE_ROW_COLUMNS: list[str] = [
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
DEFAULT_REFUSAL_PATTERNS: list[str] = [
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
DEFAULT_THRESHOLDS: dict[str, float] = {
    "parseable_rate:lt": 0.95,
    "format_adherence_rate:lt": 0.95,
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
    "judge_api_error_rate:gt": 0.01,
}

#: Fail thresholds. Crossing these signals that the resulting metrics
#: cannot be trusted — by default the pipeline halts (see
#: :class:`SanityFailure` + ``runners/sanity.py``). Set
#: ``runtime.allow_unreliable_metrics=true`` to demote fails to warnings
#: for a single run. Only metrics that directly compromise metric
#: trustworthiness when crossed get default fail entries; other metrics
#: stay warn-only unless a benchmark opts in via cfg.sanity.fail_thresholds.
#:
#: Note: ``format_adherence_rate`` is intentionally **not** in this dict.
#: Upstream SALT-NLP/PrivacyLens has no adherence gate — they report
#: leakage/helpfulness conditioned on parseable rows and let the
#: denominator speak for itself. We mirror that: format adherence is a
#: warn-only metric (``DEFAULT_THRESHOLDS["format_adherence_rate:lt"] =
#: 0.95``) that surfaces as a loud banner but never halts the pipeline.
#: This is essential for base-model sweeps, where weak instruction
#: followers can easily fall below any reasonable gate even though the
#: downstream judge numbers on the parseable subset are still meaningful.
DEFAULT_FAIL_THRESHOLDS: dict[str, float] = {
    "parseable_rate:lt": 0.7,
    "judge_unparseable_rate:gt": 0.2,
    "judge_api_error_rate:gt": 0.05,
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

class SanityFailure(RuntimeError):
    """Raised by the sanity runner when a fail-severity threshold is crossed.

    Stage runners do not catch this — it propagates to the orchestrator,
    which marks the stage as failed and surfaces it in the run summary.
    Override with ``cfg.runtime.allow_unreliable_metrics=true`` (escape
    hatch — only for debugging known-broken runs).
    """

    def __init__(self, dagspace: str, stage: str, failures: list[SanityWarning]):
        self.dagspace = dagspace
        self.stage = stage
        self.failures = list(failures)
        msg_lines = [
            f"sanity FAILURE in {dagspace}.{stage} ({len(self.failures)} fail-tier threshold(s) crossed):"
        ]
        for f in self.failures:
            msg_lines.append(f"  - {f.message()}")
        msg_lines.append(
            "Metrics from this stage cannot be trusted. Set "
            "runtime.allow_unreliable_metrics=true to demote to warnings."
        )
        super().__init__("\n".join(msg_lines))


@dataclass(frozen=True)
class SanityWarning:
    """A single threshold violation, ready to log as a W&B table row."""

    metric: str
    value: float
    threshold: float
    comparison: str  # "gt" (warn when value > threshold) or "lt"
    severity: str = "warn"  # "warn" | "fail"

    def message(self) -> str:
        sym = ">" if self.comparison == "gt" else "<"
        return f"{self.metric}={self.value:.4f} {sym} {self.threshold} ({self.severity})"


@dataclass
class SanityReport:
    """Result of a sanity check on one stage's output."""

    dagspace: str
    stage: str
    metrics: dict[str, float] = field(default_factory=dict)
    warnings: list[SanityWarning] = field(default_factory=list)
    failure_rows: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=FAILURE_ROW_COLUMNS))
    failures_dropped: int = 0
    n_rows: int = 0

    @property
    def warns(self) -> list[SanityWarning]:
        """Threshold violations at warn severity only."""
        return [w for w in self.warnings if w.severity == "warn"]

    @property
    def failures(self) -> list[SanityWarning]:
        """Threshold violations at fail severity (pipeline-halting)."""
        return [w for w in self.warnings if w.severity == "fail"]

    def has_warnings(self) -> bool:
        return len(self.warnings) > 0

    def has_failures(self) -> bool:
        return any(w.severity == "fail" for w in self.warnings)

    def worst_warning(self) -> SanityWarning | None:
        if not self.warnings:
            return None
        # Severity ranking: fail beats warn, then ratio of (value − threshold)
        # magnitude to threshold (how far past the line we are).
        def _badness(w: SanityWarning) -> tuple[int, float]:
            sev_rank = 1 if w.severity == "fail" else 0
            if w.threshold == 0:
                return (sev_rank, abs(w.value))
            return (sev_rank, abs(w.value - w.threshold) / max(abs(w.threshold), 1e-9))
        return max(self.warnings, key=_badness)

    def worst_failure(self) -> SanityWarning | None:
        fails = self.failures
        if not fails:
            return None
        def _badness(w: SanityWarning) -> float:
            if w.threshold == 0:
                return abs(w.value)
            return abs(w.value - w.threshold) / max(abs(w.threshold), 1e-9)
        return max(fails, key=_badness)

    def print_loud(self, *, prefix: str = "") -> None:
        """Print a loud banner to stderr summarizing any warnings/failures.

        Failures get a distinct ``XXX`` bar so they read differently from
        plain warnings in stderr scrollback. Both banners are emitted when
        a stage has both kinds of violations.
        """
        if not self.has_warnings():
            return
        fails = self.failures
        warns = self.warns
        if fails:
            bar = "X" * 70
            head = f"{prefix}SANITY FAILURE — {self.dagspace}.{self.stage} ({len(fails)} fail-tier threshold(s))"
            lines = ["", bar, head, bar]
            for w in fails:
                lines.append(f"  - {w.message()}")
            lines.append(bar)
            print("\n".join(lines), file=sys.stderr, flush=True)
        if warns:
            bar = "!" * 70
            head = f"{prefix}SANITY WARNINGS — {self.dagspace}.{self.stage} ({len(warns)} threshold(s))"
            lines = ["", bar, head, bar]
            for w in warns:
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
    overrides: dict[str, float] | None,
    severity: str = "warn",
) -> float | None:
    """Resolve a threshold for ``(metric, comparison, severity)``.

    Per-run overrides take precedence over the built-in defaults. Override
    keys may be ``"<metric>:<comparison>"`` (legacy, treated as warn) or
    ``"<metric>:<comparison>:<severity>"`` (new explicit form).
    """
    key_sev = f"{metric}:{comparison}:{severity}"
    if overrides and key_sev in overrides:
        return float(overrides[key_sev])
    if severity == "warn":
        # Legacy key form (no severity) defaults to warn.
        legacy_key = f"{metric}:{comparison}"
        if overrides and legacy_key in overrides:
            return float(overrides[legacy_key])
        return DEFAULT_THRESHOLDS.get(legacy_key)
    if severity == "fail":
        legacy_fail_key = f"{metric}:{comparison}"
        return DEFAULT_FAIL_THRESHOLDS.get(legacy_fail_key)
    return None


def _emit_warning(
    report: SanityReport,
    metric: str,
    value: float,
    comparison: str,
    overrides: dict[str, float] | None,
) -> None:
    """Check fail-tier and warn-tier thresholds for ``(metric, comparison)``.

    If the fail threshold is crossed, emit a single fail-severity warning
    (warn is implied). Otherwise, if the warn threshold is crossed, emit a
    warn-severity warning. At most one warning per ``(metric, comparison)``
    pair so the same violation doesn't appear twice in the report.
    """
    def _crossed(thr: float | None) -> bool:
        if thr is None:
            return False
        return (value > thr) if comparison == "gt" else (value < thr)

    fail_thr = _resolve_threshold(metric, comparison, overrides, severity="fail")
    if _crossed(fail_thr):
        report.warnings.append(
            SanityWarning(
                metric=metric,
                value=float(value),
                threshold=float(fail_thr),
                comparison=comparison,
                severity="fail",
            )
        )
        return

    warn_thr = _resolve_threshold(metric, comparison, overrides, severity="warn")
    if _crossed(warn_thr):
        report.warnings.append(
            SanityWarning(
                metric=metric,
                value=float(value),
                threshold=float(warn_thr),
                comparison=comparison,
                severity="warn",
            )
        )


def _truncate(text: Any, n: int = RAW_PREVIEW_CHARS) -> str:
    s = "" if text is None else str(text)
    if len(s) <= n:
        return s
    return s[:n] + f"…(+{len(s) - n} chars)"


def _compile_refusal_patterns(patterns: Sequence[str] | None) -> list[re.Pattern]:
    pats = list(patterns) if patterns is not None else DEFAULT_REFUSAL_PATTERNS
    out: list[re.Pattern] = []
    for p in pats:
        try:
            out.append(re.compile(p, re.IGNORECASE))
        except re.error:
            # Treat as literal if the regex is malformed.
            out.append(re.compile(re.escape(p), re.IGNORECASE))
    return out


def _matches_any(text: str, patterns: list[re.Pattern]) -> bool:
    if not text:
        return False
    return any(p.search(text) for p in patterns)


def _build_failure_rows(
    flagged: pd.DataFrame,
    *,
    dagspace: str,
    stage: str,
    model: str,
    id_col: str | None,
    completion_col: str | None,
    failure_type_col: str = "_sanity_failure_type",
    parse_error_col: str | None = None,
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
    label_col: str | None = None,
    id_col: str | None = None,
    finish_reason_col: str | None = "finish_reason",
    expected_input_n: int | None = None,
    refusal_patterns: Sequence[str] | None = None,
    thresholds: dict[str, float] | None = None,
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
    # Treat any status starting with ``parsed_status_value`` as success so
    # parsers can carry sub-type info (``parsed``, ``parsed_json``,
    # ``parsed_yes_no_normalize``) without all the variants counting as
    # failures.
    if status_col in df.columns:
        parsed_mask = df[status_col].astype(str).str.startswith(parsed_status_value)
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
                if s and not s.startswith(parsed_status_value):
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
# Format-extraction health
# ---------------------------------------------------------------------------

def compute_format_health(
    df: pd.DataFrame,
    *,
    dagspace: str,
    stage: str,
    format_col: str,
    model: str = "",
    valid_value: str = "valid",
    raw_response_col: str | None = None,
    id_col: str | None = None,
    thresholds: dict[str, float] | None = None,
) -> SanityReport:
    """Compute health metrics for a downstream format-extraction step.

    Many benchmarks have a "did the model produce something we can grade"
    gate that runs *after* parse and *before* judging — e.g.
    PrivacyLens's ``Action:`` regex extraction, MCQ answer extraction.
    Today these gates either silently default the row's metric (treat as
    non-leaking, score=0, etc.) or print a one-off WARNING. Either way
    the resulting rate flows into ``metrics.json`` as if every input
    contributed, which inflates or deflates the headline number.

    This helper reframes the gate as a first-class sanity check:

    * The stage that runs the extraction populates a ``format_col``
      column on every row with values ``"valid"`` (gate passed) or any
      other string naming the failure mode (e.g. ``"no_action"``,
      ``"no_sensitive_info"``, ``"empty"``).
    * This function reads ``format_col``, computes
      ``format_adherence_rate = #valid / #total``, and emits the
      appropriate WARN / FAIL warning. Default thresholds: WARN < 0.95,
      FAIL < 0.9 (paper-quality bar — overrideable via cfg).
    * Non-adherent rows are written to ``failure_rows`` in the shared
      schema, so they show up in the same W&B failure-row dashboard
      that parse/judge failures already feed.

    Args:
        df: Stage output. Must contain ``format_col``.
        dagspace: e.g. ``"privacylens"``.
        stage: e.g. ``"leakage_judge_export"``.
        format_col: Per-row format-status column.
        model: Task-LLM identifier (for the failure-row ``model`` field).
        valid_value: Value of ``format_col`` that means "extraction
            succeeded." Anything else counts as a failure.
        raw_response_col: Optional raw model output column for failure
            previews.
        id_col: Optional row identifier; populates ``custom_id``.
        thresholds: Override default thresholds. Use the explicit
            severity-keyed form
            ``"format_adherence_rate:lt:fail": 0.85`` to set per-benchmark
            fail thresholds.
    """
    report = SanityReport(dagspace=dagspace, stage=stage, n_rows=len(df))
    n = len(df)
    if n == 0:
        return report

    if format_col not in df.columns:
        # Defensive: the caller is responsible for populating this. Surface
        # the misconfiguration rather than silently passing.
        report.warnings.append(
            SanityWarning(
                metric="format_col_missing",
                value=0.0,
                threshold=1.0,
                comparison="lt",
                severity="fail",
            )
        )
        return report

    valid_mask = df[format_col].astype(str) == valid_value
    adherence_rate = float(valid_mask.sum()) / n
    report.metrics["format_adherence_rate"] = round(adherence_rate, 6)
    _emit_warning(report, "format_adherence_rate", adherence_rate, "lt", thresholds)

    # Per-failure-mode breakdown so callers can tell *why* extraction
    # failed (no_action vs. no_sensitive_info vs. empty …) without
    # rerunning the parquet.
    if (~valid_mask).any():
        mode_counts = (
            df.loc[~valid_mask, format_col].astype(str).value_counts().to_dict()
        )
        for mode, count in mode_counts.items():
            safe_mode = re.sub(r"\W+", "_", str(mode)).strip("_") or "unknown"
            report.metrics[f"format_failure_rate__{safe_mode}"] = round(count / n, 6)

    if (~valid_mask).any():
        flagged = df.loc[~valid_mask].copy()
        flagged["_sanity_failure_type"] = flagged[format_col].astype(str)
        rows_df, dropped = _build_failure_rows(
            flagged,
            dagspace=dagspace,
            stage=stage,
            model=model,
            id_col=id_col,
            completion_col=raw_response_col,
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
    raw_response_col: str | None = None,
    id_col: str | None = None,
    skipped_input_n: int | None = None,
    thresholds: dict[str, float] | None = None,
    n_api_errors: int | None = None,
    api_error_denominator: int | None = None,
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
        n_api_errors: Number of judge requests that returned an HTTP
            error or sidecar-exhausted-retries error — distinct from
            unparseable (the response *was* a real assistant message,
            we just couldn't parse it). When present, surfaces a
            ``judge_api_error_rate`` metric whose denominator defaults
            to total fanout requests (``len(df) + skipped_input_n +
            n_api_errors`` if no ``api_error_denominator`` is given).
            FAIL >0.05 by default — async runs that the live judge
            server 404'd cannot be quoted.
        api_error_denominator: Override the denominator used when
            computing ``judge_api_error_rate``. Set this to the total
            fanout count when the dagspace knows it explicitly (e.g.
            len(items_df) for per-secret leakage fanout).
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

    # ---- judge_api_error_rate -------------------------------------------
    # The async-judge sidecar (and the OpenAI Batch API) write per-row
    # error lines into output.jsonl on HTTP failure. Those rows make it
    # into ``df`` with ``label_col`` populated by the parser's *default*
    # branch (no Yes / score=0), so judge_unparseable_rate alone won't
    # catch them. The dagspace-level finalize counts them and passes the
    # tally here; if anything > 5% errored the run is unquotable.
    if n_api_errors is not None:
        denom = api_error_denominator
        if denom is None:
            # Default denominator: every judge request that should have
            # produced a response — judged rows + upstream-skipped rows
            # + errored rows. ``n`` is the row count post-finalize, which
            # already includes the errored rows.
            denom = n + (skipped_input_n or 0)
        if denom and denom > 0:
            api_err_rate = float(n_api_errors) / float(denom)
            report.metrics["judge_api_error_rate"] = round(api_err_rate, 6)
            report.metrics["n_judge_api_errors"] = int(n_api_errors)
            _emit_warning(report, "judge_api_error_rate", api_err_rate, "gt", thresholds)

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
