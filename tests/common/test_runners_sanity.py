"""Tests for the FAIL-halt behavior in
``dagspaces/common/runners/sanity.py``."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from dagspaces.common.eval_sanity import (
    SanityFailure,
    SanityReport,
    SanityWarning,
)
from dagspaces.common.runners.sanity import log_sanity_to_context


def _ctx(cfg_dict: dict | None = None):
    cfg = OmegaConf.create(cfg_dict or {})
    return SimpleNamespace(cfg=cfg, logger=None)


def _report_with_failure() -> SanityReport:
    r = SanityReport(dagspace="d", stage="s")
    r.warnings.append(
        SanityWarning("format_adherence_rate", 0.5, 0.9, "lt", severity="fail")
    )
    return r


def _report_with_warn_only() -> SanityReport:
    r = SanityReport(dagspace="d", stage="s")
    r.warnings.append(
        SanityWarning("format_adherence_rate", 0.92, 0.95, "lt", severity="warn")
    )
    return r


class TestHaltBehavior:
    def test_raises_sanity_failure_by_default(self):
        ctx = _ctx()
        metadata: dict = {}
        with pytest.raises(SanityFailure):
            log_sanity_to_context(ctx, _report_with_failure(), metadata=metadata)

    def test_metadata_recorded_before_raise(self):
        ctx = _ctx()
        metadata: dict = {}
        with pytest.raises(SanityFailure):
            log_sanity_to_context(ctx, _report_with_failure(), metadata=metadata)
        # Even though we raised, metadata is populated so the manifest
        # captures the failure for postmortem.
        assert "sanity" in metadata
        assert "s" in metadata["sanity"]
        entry = metadata["sanity"]["s"]
        assert entry["n_failures"] == 1
        assert entry["halted"] is True
        assert entry["failures"]  # non-empty list

    def test_allow_unreliable_metrics_demotes(self):
        ctx = _ctx({"runtime": {"allow_unreliable_metrics": True}})
        metadata: dict = {}
        # Should NOT raise
        log_sanity_to_context(ctx, _report_with_failure(), metadata=metadata)
        entry = metadata["sanity"]["s"]
        assert entry["n_failures"] == 1
        assert entry["halted"] is False

    def test_allow_unreliable_metrics_false_still_raises(self):
        ctx = _ctx({"runtime": {"allow_unreliable_metrics": False}})
        with pytest.raises(SanityFailure):
            log_sanity_to_context(ctx, _report_with_failure(), metadata={})

    def test_warn_only_does_not_raise(self):
        ctx = _ctx()
        metadata: dict = {}
        log_sanity_to_context(ctx, _report_with_warn_only(), metadata=metadata)
        entry = metadata["sanity"]["s"]
        assert entry["n_failures"] == 0
        assert entry["n_warnings"] == 1
        assert entry["halted"] is False

    def test_no_warnings_clean(self):
        ctx = _ctx()
        metadata: dict = {}
        clean_report = SanityReport(dagspace="d", stage="s")
        log_sanity_to_context(ctx, clean_report, metadata=metadata)
        entry = metadata["sanity"]["s"]
        assert entry["n_warnings"] == 0
        assert entry["n_failures"] == 0

    def test_multiple_stages_coexist(self):
        """Each report's metadata is keyed by stage so multiple sanity
        reports per pipeline (e.g. parse + judge) don't overwrite."""
        ctx = _ctx({"runtime": {"allow_unreliable_metrics": True}})
        metadata: dict = {}
        r1 = SanityReport(dagspace="d", stage="parse")
        r2 = SanityReport(dagspace="d", stage="judge")
        r2.warnings.append(SanityWarning("a", 0.5, 0.9, "lt", "warn"))
        log_sanity_to_context(ctx, r1, metadata=metadata)
        log_sanity_to_context(ctx, r2, metadata=metadata)
        assert "parse" in metadata["sanity"]
        assert "judge" in metadata["sanity"]
        assert metadata["sanity"]["judge"]["n_warnings"] == 1
