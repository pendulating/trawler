"""Tests for ``dagspaces/common/metric_provenance.py``."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from dagspaces.common.metric_provenance import MetricEmitter, MetricRecord


class TestMetricEmitterBasics:
    def test_simple_emit_no_defaults(self):
        em = MetricEmitter()
        em.emit_simple("accuracy", 0.95, n_total=100)
        out = em.to_dict()
        assert out["accuracy"] == 0.95
        assert out["metric_provenance"]["accuracy"] == {
            "n_total": 100,
            "n_real": 100,
            "n_defaulted": 0,
            "defaulted_rate": 0.0,
            "default_reason": None,
        }

    def test_emit_with_defaults(self):
        em = MetricEmitter()
        em.emit("leakage_rate", 0.0, n_total=493, n_real=44, n_defaulted=449,
                default_reason="no_action_format")
        prov = em.to_dict()["metric_provenance"]["leakage_rate"]
        assert prov["n_total"] == 493
        assert prov["n_real"] == 44
        assert prov["n_defaulted"] == 449
        assert abs(prov["defaulted_rate"] - 449 / 493) < 1e-6
        assert prov["default_reason"] == "no_action_format"

    def test_emit_raw_no_provenance(self):
        em = MetricEmitter()
        em.emit_raw("benchmark", "PrivacyLens")
        out = em.to_dict()
        assert out["benchmark"] == "PrivacyLens"
        assert "metric_provenance" not in out  # no emit yet

    def test_nested_dotted_keys(self):
        em = MetricEmitter()
        em.emit_simple("leakage.rate", 0.0, n_total=100)
        em.emit_simple("leakage.count", 0.0, n_total=100)
        em.emit_raw("leakage.task", "leakage")
        out = em.to_dict()
        assert out["leakage"]["rate"] == 0.0
        assert out["leakage"]["count"] == 0.0
        assert out["leakage"]["task"] == "leakage"


class TestMetricEmitterValidation:
    def test_n_real_plus_defaulted_exceeds_total_raises(self):
        em = MetricEmitter()
        with pytest.raises(ValueError, match="n_real\\+n_defaulted"):
            em.emit("bad", 0.5, n_total=10, n_real=8, n_defaulted=5)

    def test_negative_counts_raise(self):
        em = MetricEmitter()
        with pytest.raises(ValueError):
            em.emit("bad", 0.5, n_total=-1, n_real=0, n_defaulted=0)

    def test_defaulted_without_reason_raises(self):
        em = MetricEmitter()
        with pytest.raises(ValueError, match="default_reason"):
            em.emit("bad", 0.5, n_total=10, n_real=5, n_defaulted=5)

    def test_defaulted_zero_no_reason_ok(self):
        em = MetricEmitter()
        em.emit("ok", 1.0, n_total=10, n_real=10, n_defaulted=0)
        # No raise

    def test_path_conflict_raises(self):
        em = MetricEmitter()
        em.emit_raw("x", 1.0)  # x is a scalar
        with pytest.raises(ValueError, match="path conflict"):
            em.emit_raw("x.y", 2.0)


class TestMetricEmitterSerialization:
    def test_write_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "a/b/c/metrics.json")
            em = MetricEmitter()
            em.emit_simple("v", 1.0, n_total=1)
            em.write(path)
            assert os.path.exists(path)
            with open(path) as f:
                data = json.load(f)
            assert data["v"] == 1.0

    def test_to_dict_idempotent(self):
        em = MetricEmitter()
        em.emit_simple("v", 1.0, n_total=1)
        a = em.to_dict()
        b = em.to_dict()
        assert a == b

    def test_provenance_only_view(self):
        em = MetricEmitter()
        em.emit_simple("v", 1.0, n_total=1)
        em.emit_raw("notes", "x")
        prov = em.provenance()
        assert "v" in prov
        assert "notes" not in prov


class TestMaxDefaultedRate:
    def test_zero_when_empty(self):
        em = MetricEmitter()
        assert em.max_defaulted_rate() == 0.0

    def test_picks_max(self):
        em = MetricEmitter()
        em.emit_simple("a", 1.0, n_total=100)
        em.emit("b", 0.0, n_total=100, n_real=10, n_defaulted=90,
                default_reason="x")
        em.emit("c", 0.0, n_total=100, n_real=80, n_defaulted=20,
                default_reason="y")
        assert em.max_defaulted_rate() == 0.9
