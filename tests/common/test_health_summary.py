"""Tests for ``dagspaces/common/health_summary.py``.

The aggregator must:
1. Roll up sanity reports from per-benchmark manifests under one tree.
2. Distinguish v1 (legacy) schemas where ``n_failures`` was a row count
   from v2 (post-2026-04-27) schemas where ``n_failures`` counts fail-tier
   warnings and ``n_failure_rows`` is the row count.
3. Flag metrics with high ``defaulted_rate`` from ``metric_provenance``.
"""

from __future__ import annotations

import json
import os
import tempfile

from dagspaces.common.health_summary import collect_health, write_health_summary


def _write_manifest(path: str, *, sanity: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    manifest = {
        "output_root": os.path.dirname(path),
        "nodes": {
            "stage_a": {
                "stage": "stage_a",
                "metadata": {"sanity": sanity},
            }
        },
    }
    with open(path, "w") as f:
        json.dump(manifest, f)


def _write_metrics(path: str, *, provenance: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"metric_provenance": provenance}, f)


class TestCollectHealth:
    def test_v2_fail_severity_recognized(self):
        with tempfile.TemporaryDirectory() as run_dir:
            _write_manifest(
                os.path.join(run_dir, "privacylens", "pipeline_manifest.json"),
                sanity={
                    "agent_action_format": {
                        "n_warnings": 0,
                        "n_failures": 1,
                        "n_failure_rows": 449,
                        "halted": True,
                        "warnings": [],
                        "failures": ["format_adherence_rate=0.0892 < 0.9 (fail)"],
                        "metrics": {"format_adherence_rate": 0.089},
                    }
                },
            )
            s = collect_health(run_dir)
            assert s["totals"]["n_fail_stages"] == 1
            assert s["totals"]["n_halted_stages"] == 1
            assert s["sanity"][0]["schema"] == "v2"
            assert s["sanity"][0]["n_failures"] == 1
            assert s["sanity"][0]["n_failure_rows"] == 449

    def test_v1_legacy_schema_treated_as_row_count(self):
        with tempfile.TemporaryDirectory() as run_dir:
            _write_manifest(
                os.path.join(run_dir, "confaide", "pipeline_manifest.json"),
                sanity={
                    "parse_responses_3_free": {
                        "n_warnings": 1,
                        # Legacy: this was the row-count, not fail-tier count.
                        "n_failures": 35,
                        "warnings": ["refusal_rate=0.1296 > 0.02 (warn)"],
                        "metrics": {"parseable_rate": 1.0, "refusal_rate": 0.1296},
                    }
                },
            )
            s = collect_health(run_dir)
            entry = s["sanity"][0]
            assert entry["schema"] == "v1_legacy"
            # Legacy "35" should land in n_failure_rows, NOT n_failures.
            assert entry["n_failures"] == 0
            assert entry["n_failure_rows"] == 35
            assert s["totals"]["n_fail_stages"] == 0

    def test_provenance_aggregation(self):
        with tempfile.TemporaryDirectory() as run_dir:
            _write_metrics(
                os.path.join(run_dir, "privacylens", "outputs", "compute_metrics", "metrics.json"),
                provenance={
                    "leakage.leakage_rate_among_parseable": {
                        "n_total": 44, "n_real": 44, "n_defaulted": 0,
                        "defaulted_rate": 0.0, "default_reason": None,
                    },
                    "leakage.leakage_rate_overall_with_default_zero": {
                        "n_total": 493, "n_real": 44, "n_defaulted": 449,
                        "defaulted_rate": 0.910751,
                        "default_reason": "judge_skipped_default_no_leak",
                    },
                },
            )
            s = collect_health(run_dir)
            assert s["totals"]["n_provenance_records"] == 2
            assert s["totals"]["max_defaulted_rate"] > 0.9
            assert any("0.91" in m for m in s["totals"]["high_default_metrics"]) or \
                   any("91" in m for m in s["totals"]["high_default_metrics"])

    def test_write_creates_output(self):
        with tempfile.TemporaryDirectory() as run_dir:
            _write_manifest(
                os.path.join(run_dir, "x", "pipeline_manifest.json"),
                sanity={"a": {"n_warnings": 0, "n_failures": 0, "n_failure_rows": 0,
                              "halted": False, "warnings": [], "failures": [],
                              "metrics": {}}},
            )
            path = write_health_summary(run_dir)
            assert os.path.exists(path)
            with open(path) as f:
                data = json.load(f)
            assert "totals" in data
            assert "sanity" in data
            assert "metric_provenance" in data
