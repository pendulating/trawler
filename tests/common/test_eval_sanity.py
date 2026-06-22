"""Tests for ``dagspaces/common/eval_sanity.py``.

Each test maps to a documented invariant. The two regression tests
(``test_parse_status_variants_*``, ``test_failure_rows_dataframe_truth``)
cover bugs that hit production on 2026-04-26 — keep them passing.
"""

from __future__ import annotations

import pytest
import pandas as pd

from dagspaces.common.eval_sanity import (
    DEFAULT_FAIL_THRESHOLDS,
    DEFAULT_THRESHOLDS,
    FAILURE_ROW_COLUMNS,
    SanityFailure,
    SanityReport,
    SanityWarning,
    _emit_warning,
    _resolve_threshold,
    compute_format_health,
    compute_judge_health,
    compute_parse_health,
)


# ---------------------------------------------------------------------------
# SanityWarning + SanityReport basics
# ---------------------------------------------------------------------------

class TestSanityWarning:
    def test_message_renders_severity(self):
        w_warn = SanityWarning(metric="m", value=0.5, threshold=0.9, comparison="lt", severity="warn")
        w_fail = SanityWarning(metric="m", value=0.5, threshold=0.9, comparison="lt", severity="fail")
        assert "(warn)" in w_warn.message()
        assert "(fail)" in w_fail.message()
        assert "0.5000" in w_warn.message()
        assert "< 0.9" in w_warn.message()

    def test_default_severity_is_warn(self):
        w = SanityWarning(metric="m", value=0.5, threshold=0.9, comparison="lt")
        assert w.severity == "warn"


class TestSanityReport:
    def test_warns_failures_split(self):
        r = SanityReport(dagspace="d", stage="s")
        r.warnings.append(SanityWarning("a", 0.5, 0.9, "lt", "warn"))
        r.warnings.append(SanityWarning("b", 0.5, 0.9, "lt", "fail"))
        r.warnings.append(SanityWarning("c", 0.5, 0.9, "lt", "warn"))
        assert len(r.warns) == 2
        assert len(r.failures) == 1
        assert r.has_warnings()
        assert r.has_failures()

    def test_no_failures_clean(self):
        r = SanityReport(dagspace="d", stage="s")
        r.warnings.append(SanityWarning("a", 0.5, 0.9, "lt", "warn"))
        assert r.has_warnings()
        assert not r.has_failures()
        assert r.failures == []

    def test_worst_warning_prefers_fail(self):
        r = SanityReport(dagspace="d", stage="s")
        # Mild warn vs. mild fail — fail should still beat warn even when
        # the magnitude past the threshold is smaller.
        r.warnings.append(SanityWarning("a", 0.50, 0.95, "lt", "warn"))  # big miss, warn
        r.warnings.append(SanityWarning("b", 0.89, 0.90, "lt", "fail"))  # tiny miss, fail
        worst = r.worst_warning()
        assert worst.severity == "fail"

    def test_worst_failure_only_among_fails(self):
        r = SanityReport(dagspace="d", stage="s")
        r.warnings.append(SanityWarning("a", 0.50, 0.95, "lt", "warn"))
        r.warnings.append(SanityWarning("b", 0.50, 0.90, "lt", "fail"))
        r.warnings.append(SanityWarning("c", 0.10, 0.90, "lt", "fail"))
        worst = r.worst_failure()
        assert worst is not None
        assert worst.metric == "c"  # furthest past threshold

    def test_worst_failure_none_when_no_fails(self):
        r = SanityReport(dagspace="d", stage="s")
        r.warnings.append(SanityWarning("a", 0.50, 0.95, "lt", "warn"))
        assert r.worst_failure() is None


# ---------------------------------------------------------------------------
# _resolve_threshold + _emit_warning
# ---------------------------------------------------------------------------

class TestResolveThreshold:
    def test_warn_default(self):
        # Built-in default: parseable_rate:lt = 0.95
        assert _resolve_threshold("parseable_rate", "lt", None, severity="warn") == 0.95

    def test_fail_default(self):
        # Built-in default: parseable_rate:lt:fail = 0.7
        assert _resolve_threshold("parseable_rate", "lt", None, severity="fail") == 0.7

    def test_legacy_override_treated_as_warn(self):
        overrides = {"parseable_rate:lt": 0.8}
        assert _resolve_threshold("parseable_rate", "lt", overrides, severity="warn") == 0.8
        # Fail tier still uses default (no legacy fail key)
        assert _resolve_threshold("parseable_rate", "lt", overrides, severity="fail") == 0.7

    def test_severity_keyed_override(self):
        overrides = {
            "parseable_rate:lt:warn": 0.99,
            "parseable_rate:lt:fail": 0.5,
        }
        assert _resolve_threshold("parseable_rate", "lt", overrides, severity="warn") == 0.99
        assert _resolve_threshold("parseable_rate", "lt", overrides, severity="fail") == 0.5

    def test_unknown_metric_returns_none(self):
        assert _resolve_threshold("totally_unknown_metric_xyz", "lt", None) is None


class TestEmitWarning:
    def test_fail_beats_warn_when_both_crossed(self):
        # 0.5 < 0.95 (warn) AND < 0.7 (fail) on parseable_rate — emit fail only.
        # parseable_rate has both default tiers; format_adherence_rate is now
        # warn-only (matching upstream SALT-NLP/PrivacyLens, which has no
        # adherence gate — see eval_sanity.py DEFAULT_FAIL_THRESHOLDS doc).
        r = SanityReport(dagspace="d", stage="s")
        _emit_warning(r, "parseable_rate", 0.5, "lt", None)
        assert len(r.warnings) == 1
        assert r.warnings[0].severity == "fail"

    def test_warn_only_when_below_warn_above_fail(self):
        # 0.8: < 0.95 (warn) but > 0.7 (fail) on parseable_rate → warn only.
        r = SanityReport(dagspace="d", stage="s")
        _emit_warning(r, "parseable_rate", 0.8, "lt", None)
        assert len(r.warnings) == 1
        assert r.warnings[0].severity == "warn"

    def test_no_emit_when_clean(self):
        r = SanityReport(dagspace="d", stage="s")
        _emit_warning(r, "parseable_rate", 0.99, "lt", None)
        assert r.warnings == []

    def test_per_run_override_promotes_to_fail(self):
        # Force a metric that doesn't have a default fail tier into one
        r = SanityReport(dagspace="d", stage="s")
        _emit_warning(
            r, "refusal_rate", 0.05, "gt",
            {"refusal_rate:gt:fail": 0.04},
        )
        assert len(r.warnings) == 1
        assert r.warnings[0].severity == "fail"


# ---------------------------------------------------------------------------
# compute_parse_health — regression for the two 2026-04-26 bugs
# ---------------------------------------------------------------------------

class TestComputeParseHealth:
    def test_parsed_variants_all_count_as_success(self):
        """Regression: parse_status values like 'parsed_json' /
        'parsed_yes_no_normalize' must count as parsed, not failures.
        Pre-fix, only 'parsed' literal matched."""
        df = pd.DataFrame({
            "parse_status": ["parsed", "parsed_json", "parsed_yes_no_normalize", "unparseable"],
            "generated_text": ["yes", "{\"answer\":\"yes\"}", "Yes.", ""],
        })
        r = compute_parse_health(df, dagspace="d", stage="s")
        assert r.metrics["parseable_rate"] == 0.75  # 3/4 success

    def test_failure_rows_dataframe_truth_value(self):
        """Regression: SanityReport.failure_rows is a DataFrame; any
        downstream code doing ``rows or default`` blew up with ambiguous
        DataFrame truth value."""
        df = pd.DataFrame({
            "parse_status": ["unparseable"] * 5,
            "generated_text": ["x"] * 5,
        })
        r = compute_parse_health(df, dagspace="d", stage="s")
        # The bug: any code path that did `getattr(report, "failure_rows", []) or []`
        # would raise ValueError. We assert here that the field is well-typed
        # and that ``len()`` works without error.
        assert isinstance(r.failure_rows, pd.DataFrame)
        assert len(r.failure_rows) == 5
        # Bool-converting a DataFrame raises — make sure callers know this
        # by exercising it explicitly (the runner uses `is not None`):
        with pytest.raises(ValueError):
            bool(r.failure_rows)

    def test_empty_df_returns_clean_report(self):
        r = compute_parse_health(pd.DataFrame(), dagspace="d", stage="s")
        assert r.n_rows == 0
        assert r.metrics == {}
        assert not r.has_warnings()

    def test_failure_rows_in_canonical_schema(self):
        df = pd.DataFrame({
            "parse_status": ["unparseable"],
            "generated_text": ["garbage"],
        })
        r = compute_parse_health(df, dagspace="d", stage="s")
        assert list(r.failure_rows.columns) == FAILURE_ROW_COLUMNS

    def test_class_balance_min_emitted(self):
        df = pd.DataFrame({
            "parse_status": ["parsed"] * 100,
            "generated_text": ["x"] * 100,
            "label": ["yes"] * 90 + ["no"] * 10,
        })
        r = compute_parse_health(
            df, dagspace="d", stage="s", label_col="label",
        )
        assert r.metrics["class_balance_min"] == 0.1


# ---------------------------------------------------------------------------
# compute_format_health
# ---------------------------------------------------------------------------

class TestComputeFormatHealth:
    def _make(self, n_valid: int, n_total: int, *, mode: str = "no_action"):
        statuses = ["valid"] * n_valid + [mode] * (n_total - n_valid)
        return pd.DataFrame({
            "format_status": statuses,
            "raw": ["x"] * n_total,
            "id": list(range(n_total)),
        })

    def test_clean_at_100pct(self):
        r = compute_format_health(
            self._make(100, 100), dagspace="d", stage="s",
            format_col="format_status",
        )
        assert r.metrics["format_adherence_rate"] == 1.0
        assert not r.has_warnings()

    def test_warn_at_92pct(self):
        # 0.92 < 0.95 (warn) → warn. format_adherence_rate is warn-only by
        # default (no fail tier) — mirrors upstream SALT-NLP/PrivacyLens.
        r = compute_format_health(
            self._make(92, 100), dagspace="d", stage="s",
            format_col="format_status",
        )
        assert r.metrics["format_adherence_rate"] == 0.92
        assert len(r.warns) == 1
        assert not r.has_failures()

    def test_warn_only_at_85pct_no_default_fail(self):
        # Even at 0.85 — well below any reasonable adherence bar — the
        # default behavior is WARN, not FAIL. Upstream PrivacyLens has no
        # adherence gate; halting weak instruction followers would silently
        # disqualify them from the table even though their denominator-
        # conditioned numbers are still meaningful.
        r = compute_format_health(
            self._make(85, 100), dagspace="d", stage="s",
            format_col="format_status",
        )
        assert r.metrics["format_adherence_rate"] == 0.85
        assert len(r.warns) == 1
        assert not r.has_failures(), (
            "format_adherence_rate must NOT halt the pipeline by default; "
            "operators who want a hard gate set "
            "cfg.sanity.fail_thresholds.format_adherence_rate:lt explicitly."
        )

    def test_fail_at_85pct_with_explicit_override(self):
        # Operators who DO want a hard adherence gate can opt in per-run
        # via a severity-keyed threshold override.
        r = compute_format_health(
            self._make(85, 100), dagspace="d", stage="s",
            format_col="format_status",
            thresholds={"format_adherence_rate:lt:fail": 0.9},
        )
        assert r.metrics["format_adherence_rate"] == 0.85
        assert r.has_failures()
        assert len(r.failures) == 1

    def test_per_mode_breakdown(self):
        df = pd.DataFrame({
            "format_status": ["valid"] * 50 + ["no_action"] * 30 + ["no_secrets"] * 20,
            "raw": ["x"] * 100,
            "id": range(100),
        })
        r = compute_format_health(df, dagspace="d", stage="s", format_col="format_status")
        assert r.metrics["format_failure_rate__no_action"] == 0.30
        assert r.metrics["format_failure_rate__no_secrets"] == 0.20

    def test_missing_column_emits_defensive_fail(self):
        df = pd.DataFrame({"other": [1, 2, 3]})
        r = compute_format_health(df, dagspace="d", stage="s", format_col="format_status")
        assert r.has_failures()
        assert any(w.metric == "format_col_missing" for w in r.failures)

    def test_failure_rows_in_canonical_schema(self):
        r = compute_format_health(
            self._make(50, 100), dagspace="d", stage="s",
            format_col="format_status", raw_response_col="raw", id_col="id",
        )
        assert list(r.failure_rows.columns) == FAILURE_ROW_COLUMNS
        assert len(r.failure_rows) == 50


# ---------------------------------------------------------------------------
# compute_judge_health (smoke)
# ---------------------------------------------------------------------------

class TestComputeJudgeHealth:
    def test_basic_unparseable_rate(self):
        df = pd.DataFrame({
            "label": ["yes"] * 90 + ["no"] * 5 + ["???"] * 5,
            "raw": ["x"] * 100,
            "id": range(100),
        })
        r = compute_judge_health(
            df, dagspace="d", stage="s",
            label_col="label", valid_labels=["yes", "no"],
            raw_response_col="raw", id_col="id",
        )
        assert r.metrics["judge_unparseable_rate"] == 0.05
        assert "judge_label_entropy" in r.metrics


# ---------------------------------------------------------------------------
# SanityFailure exception
# ---------------------------------------------------------------------------

class TestSanityFailure:
    def test_message_lists_all_failures(self):
        fails = [
            SanityWarning("a", 0.5, 0.9, "lt", "fail"),
            SanityWarning("b", 0.05, 0.01, "gt", "fail"),
        ]
        err = SanityFailure("d", "s", fails)
        msg = str(err)
        assert "d.s" in msg
        assert "(2 fail-tier" in msg
        assert "a=0.5000" in msg
        assert "b=0.0500" in msg
        assert "allow_unreliable_metrics" in msg

    def test_attributes_preserved(self):
        fails = [SanityWarning("a", 0.5, 0.9, "lt", "fail")]
        err = SanityFailure("d", "s", fails)
        assert err.dagspace == "d"
        assert err.stage == "s"
        assert err.failures == fails
