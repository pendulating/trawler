"""Integration tests for async-judge finalize error handling.

End-to-end coverage of the silent-corruption fix (2026-04-27): when
the live judge server 404s every request (e.g. wrong model name in
body.model), the finalize stage must demote those rows to
``*_judged=False`` with ``*_skip_reason="judge_api_error"`` and surface
``judge_api_error_rate`` as a fail-tier sanity metric, NOT silently
zero-substitute and pass them through compute_metrics with
judged=True.
"""

from __future__ import annotations

import json
import os
import tempfile

import pandas as pd
import pytest

from dagspaces.privacylens.stages.finalize_async import (
    _finalize_helpfulness,
    _finalize_leakage,
)
from dagspaces.common.eval_sanity import (
    DEFAULT_FAIL_THRESHOLDS,
    compute_judge_health,
)


def _write_parquet(path: str, df: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)


def _write_jsonl(path: str, lines: list) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for ln in lines:
            f.write(json.dumps(ln) + "\n")


def _success(cid: str, content: str) -> dict:
    return {
        "custom_id": cid,
        "response": {
            "status_code": 200,
            "body": {"choices": [{"message": {"content": content}}]},
        },
    }


def _http_404(cid: str, model: str = "gpt-5.2") -> dict:
    """Simulate the actual 404 the smoke run surfaced — sidecar
    classifies HTTP non-2xx as a top-level error."""
    return {
        "custom_id": cid,
        "error": f"[other] HTTP 404: The model '{model}' does not exist",
    }


class TestFinalizeLeakageWithErrors:
    def test_all_errors_demote_rows_to_unjudged(self):
        """When every per-secret call errors, the row must be marked
        leakage_judged=False with skip_reason=judge_api_error — not
        silently defaulted to no-leak with judged=True.
        """
        with tempfile.TemporaryDirectory() as base:
            pending = pd.DataFrame({
                "record_id": [101, 102],
                "user_name": ["alice", "bob"],
                "leakage_judged": [True, True],     # set by export stage
                "leakage_skip_reason": ["", ""],
                "agent_action_format_status": ["valid", "valid"],
            })
            items = pd.DataFrame({
                "judge_custom_id": [
                    "privacylens:leakage_judge:0:0",
                    "privacylens:leakage_judge:0:1",
                    "privacylens:leakage_judge:1:0",
                ],
                "row_idx": [0, 0, 1],
                "sub_idx": [0, 1, 0],
                "secret": ["s1", "s2", "s3"],
            })
            _write_parquet(os.path.join(base, "pending.parquet"), pending)
            _write_parquet(os.path.join(base, "items.parquet"), items)
            _write_jsonl(
                os.path.join(base, "output.jsonl"),
                [
                    _http_404("privacylens:leakage_judge:0:0"),
                    _http_404("privacylens:leakage_judge:0:1"),
                    _http_404("privacylens:leakage_judge:1:0"),
                ],
            )

            df, meta = _finalize_leakage(base)
            assert (df["leakage_judged"] == False).all()  # noqa: E712
            assert (df["leakage_skip_reason"] == "judge_api_error").all()
            assert meta["n_response_errors"] == 3
            assert meta["n_rows_judge_api_error"] == 2
            assert meta["items"] == 3

    def test_mixed_success_keeps_judged_true(self):
        """If even one fanout item succeeded, the row is judged."""
        with tempfile.TemporaryDirectory() as base:
            pending = pd.DataFrame({
                "record_id": [101],
                "user_name": ["alice"],
                "leakage_judged": [True],
                "leakage_skip_reason": [""],
                "agent_action_format_status": ["valid"],
            })
            items = pd.DataFrame({
                "judge_custom_id": [
                    "privacylens:leakage_judge:0:0",
                    "privacylens:leakage_judge:0:1",
                ],
                "row_idx": [0, 0],
                "sub_idx": [0, 1],
                "secret": ["s1", "s2"],
            })
            _write_parquet(os.path.join(base, "pending.parquet"), pending)
            _write_parquet(os.path.join(base, "items.parquet"), items)
            _write_jsonl(
                os.path.join(base, "output.jsonl"),
                [
                    _success("privacylens:leakage_judge:0:0", "Answer: Yes."),
                    _http_404("privacylens:leakage_judge:0:1"),
                ],
            )
            df, meta = _finalize_leakage(base)
            assert df.loc[0, "leakage_judged"] == True  # noqa: E712
            assert df.loc[0, "leakage_skip_reason"] == ""
            # OR-aggregation: one Yes is enough.
            assert df.loc[0, "leak_judge_text"] == "Answer: Yes."
            assert meta["n_response_errors"] == 1
            assert meta["n_rows_judge_api_error"] == 0

    def test_clean_run_no_demotions(self):
        with tempfile.TemporaryDirectory() as base:
            pending = pd.DataFrame({
                "record_id": [101, 102],
                "user_name": ["a", "b"],
                "leakage_judged": [True, True],
                "leakage_skip_reason": ["", ""],
                "agent_action_format_status": ["valid", "valid"],
            })
            items = pd.DataFrame({
                "judge_custom_id": [
                    "privacylens:leakage_judge:0:0",
                    "privacylens:leakage_judge:1:0",
                ],
                "row_idx": [0, 1],
                "sub_idx": [0, 0],
                "secret": ["s1", "s2"],
            })
            _write_parquet(os.path.join(base, "pending.parquet"), pending)
            _write_parquet(os.path.join(base, "items.parquet"), items)
            _write_jsonl(os.path.join(base, "output.jsonl"), [
                _success("privacylens:leakage_judge:0:0", "Answer: No."),
                _success("privacylens:leakage_judge:1:0", "Answer: Yes."),
            ])
            df, meta = _finalize_leakage(base)
            assert (df["leakage_judged"] == True).all()  # noqa: E712
            assert meta["n_rows_judge_api_error"] == 0
            assert df.loc[1, "leak_judge_text"] == "Answer: Yes."


class TestFinalizeHelpfulnessWithErrors:
    def test_error_rows_demoted_not_judged_zero(self):
        """The smoke run's exact failure mode: 470/493 error lines must
        produce helpfulness_judged=False, not judged=True with score=0.
        """
        with tempfile.TemporaryDirectory() as base:
            pending = pd.DataFrame({
                "record_id": [101, 102, 103],
                "helpfulness_judged": [True, True, True],
                "agent_action_format_status": ["valid", "valid", "valid"],
            })
            items = pd.DataFrame({
                "judge_custom_id": [
                    "privacylens:helpfulness_judge:0",
                    "privacylens:helpfulness_judge:1",
                    "privacylens:helpfulness_judge:2",
                ],
                "row_idx": [0, 1, 2],
            })
            _write_parquet(os.path.join(base, "pending.parquet"), pending)
            _write_parquet(os.path.join(base, "items.parquet"), items)
            _write_jsonl(os.path.join(base, "output.jsonl"), [
                _http_404("privacylens:helpfulness_judge:0"),
                _http_404("privacylens:helpfulness_judge:1"),
                _success("privacylens:helpfulness_judge:2", "Answer: Excellent (3)."),
            ])
            df, meta = _finalize_helpfulness(base)
            assert df.loc[0, "helpfulness_judged"] == False  # noqa: E712
            assert df.loc[0, "helpfulness_skip_reason"] == "judge_api_error"
            assert df.loc[1, "helpfulness_judged"] == False  # noqa: E712
            assert df.loc[2, "helpfulness_judged"] == True   # noqa: E712
            assert meta["n_response_errors"] == 2
            assert meta["n_rows_judge_api_error"] == 2


class TestJudgeApiErrorRateSanity:
    def test_rate_above_fail_threshold_creates_failure(self):
        """5%+ judge_api_error_rate is fail-tier by default."""
        # 6 of 100 rows errored → 6%, exceeds 5% fail threshold.
        df = pd.DataFrame({
            "leak_flag": [False] * 100,
        })
        report = compute_judge_health(
            df,
            dagspace="privacylens",
            stage="leakage_judge_api",
            label_col="leak_flag",
            valid_labels=[True, False],
            n_api_errors=6,
            api_error_denominator=100,
        )
        assert report.has_failures()
        assert any(w.metric == "judge_api_error_rate" and w.severity == "fail"
                   for w in report.warnings)

    def test_rate_above_warn_below_fail(self):
        """2% triggers warn (default warn 1%), not fail (default fail 5%)."""
        df = pd.DataFrame({"leak_flag": [False] * 100})
        report = compute_judge_health(
            df,
            dagspace="privacylens",
            stage="leakage_judge_api",
            label_col="leak_flag",
            valid_labels=[True, False],
            n_api_errors=2,
            api_error_denominator=100,
        )
        assert not report.has_failures()
        warn_metrics = [w.metric for w in report.warns]
        assert "judge_api_error_rate" in warn_metrics

    def test_zero_errors_clean(self):
        df = pd.DataFrame({"leak_flag": [False] * 100})
        report = compute_judge_health(
            df,
            dagspace="privacylens",
            stage="leakage_judge_api",
            label_col="leak_flag",
            valid_labels=[True, False],
            n_api_errors=0,
            api_error_denominator=100,
        )
        assert not report.has_failures()
        # 0% does not trip warn-tier 1%.
        assert report.metrics.get("judge_api_error_rate") == 0.0


class TestSmokeRunRecreation:
    """End-to-end recreation of the 2026-04-27 smoke run failure mode.

    470 of 493 helpfulness rows received HTTP 404 from the live judge
    server (judge.batch.target_model='gpt-5.2' on a Qwen-serving vLLM).
    With the fix, those rows must end up with helpfulness_judged=False
    and the runner-level sanity layer must FAIL on judge_api_error_rate.
    """

    def test_smoke_pattern_demotes_and_fails(self):
        with tempfile.TemporaryDirectory() as base:
            n_total = 100
            n_errors = 90
            pending = pd.DataFrame({
                "record_id": list(range(n_total)),
                "helpfulness_judged": [True] * n_total,
                "agent_action_format_status": ["valid"] * n_total,
            })
            items = pd.DataFrame({
                "judge_custom_id": [f"privacylens:helpfulness_judge:{i}" for i in range(n_total)],
                "row_idx": list(range(n_total)),
            })
            _write_parquet(os.path.join(base, "pending.parquet"), pending)
            _write_parquet(os.path.join(base, "items.parquet"), items)
            _write_jsonl(
                os.path.join(base, "output.jsonl"),
                [_http_404(f"privacylens:helpfulness_judge:{i}") for i in range(n_errors)] +
                [_success(f"privacylens:helpfulness_judge:{i}", "Answer: Excellent (3).")
                 for i in range(n_errors, n_total)],
            )
            df, meta = _finalize_helpfulness(base)
            assert int((df["helpfulness_judged"] == False).sum()) == n_errors  # noqa: E712
            assert meta["n_response_errors"] == n_errors

            report = compute_judge_health(
                df,
                dagspace="privacylens",
                stage="helpfulness_judge_api",
                label_col="helpfulness_score",
                valid_labels=[0, 1, 2, 3],
                n_api_errors=meta["n_response_errors"],
                api_error_denominator=meta["rows"],
            )
            # 90% > 5% fail threshold → FAIL.
            assert report.has_failures()
            assert DEFAULT_FAIL_THRESHOLDS["judge_api_error_rate:gt"] == pytest.approx(0.05)
