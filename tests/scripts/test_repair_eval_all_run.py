"""Repair-plan derivation for eval_all multiruns.

The plan these tests pin down decides which GPU work a repair pass re-runs, so
the invariants are: never silently treat an unfinished cell as clean, never
trust dispatch=ok over missing metrics, and never fold cells with different
holes into one invocation.
"""

from __future__ import annotations

import json

import pytest

from scripts import repair_eval_all_run as rr


def _cell(run_dir, idx: int, model: str, *, failures: dict | None = None,
          metrics: dict[str, bool] | None = None):
    """Build a fake cell dir: hydra override, optional failures.json, metrics."""
    cell = run_dir / str(idx)
    (cell / ".hydra").mkdir(parents=True)
    (cell / ".hydra" / "overrides.yaml").write_text(f"- model={model}\n")
    if failures is not None:
        (cell / "failures.json").write_text(json.dumps(failures))
    for bench, present in (metrics or {}).items():
        bdir = cell / bench / f"{bench}_dag" / "outputs" / "compute_metrics"
        bdir.mkdir(parents=True)
        if present:
            (bdir / "metrics.parquet").write_text("")
    return cell


def _failures(model: str, dispatch: dict[str, str]) -> dict:
    failed = [b for b, s in dispatch.items()
              if s != "ok" and not s.startswith("skipped")]
    return {"model": model, "dispatch": dispatch, "finalize": {},
            "failed": failed, "success": not failed}


class TestScanCell:
    def test_clean_cell_needs_no_repair(self, tmp_path):
        cell = _cell(
            tmp_path, 0, "qwen3.5-9b/k3-base",
            failures=_failures("qwen3.5-9b/k3-base",
                               {"goldcoin": "ok", "cirl": "ok"}),
            metrics={"goldcoin": True, "cirl": True},
        )
        out = rr.scan_cell(cell)
        assert out["status"] == "ok"
        assert out["repair"] == []

    def test_failed_dispatch_is_repaired(self, tmp_path):
        cell = _cell(
            tmp_path, 1, "qwen3.5-9b/instruct",
            failures=_failures("qwen3.5-9b/instruct", {
                "goldcoin": "ok",
                "confaide": "FAILED (rc=1)",
                "vlm_geoprivacy": "FAILED (rc=1)",
            }),
            metrics={"goldcoin": True},
        )
        out = rr.scan_cell(cell)
        assert out["status"] == "needs-repair"
        assert sorted(out["repair"]) == ["confaide", "vlm_geoprivacy"]
        assert "FAILED" in out["reasons"]["confaide"]

    def test_ok_dispatch_without_metrics_is_repaired(self, tmp_path):
        """dispatch=ok is not proof: the table reads metrics, not status."""
        cell = _cell(
            tmp_path, 2, "qwen3.5-9b/k3-verdict",
            failures=_failures("qwen3.5-9b/k3-verdict",
                               {"cirl": "ok", "mmlu": "ok"}),
            metrics={"cirl": False, "mmlu": True},
        )
        out = rr.scan_cell(cell)
        assert out["repair"] == ["cirl"]
        assert "no metrics" in out["reasons"]["cirl"]

    def test_skipped_benchmarks_are_not_repaired(self, tmp_path):
        cell = _cell(
            tmp_path, 3, "qwen3.5-9b/m2-full-ckpt450",
            failures=_failures("qwen3.5-9b/m2-full-ckpt450", {
                "goldcoin": "ok",
                "simpleqa_verified": "skipped",
                "privacylens": "skipped:export_skipped",
            }),
            metrics={"goldcoin": True},
        )
        out = rr.scan_cell(cell)
        assert out["repair"] == []

    def test_missing_summary_is_flagged_not_assumed_clean(self, tmp_path):
        """A live or killed cell must never read as 'nothing failed'."""
        cell = _cell(tmp_path, 4, "qwen3.5-9b/instruct",
                     metrics={"goldcoin": True})
        out = rr.scan_cell(cell)
        assert out["status"] == "no-summary"
        assert "goldcoin" not in out["repair"]
        assert "cirl" in out["repair"]
        assert "still running or monitor killed" in out["reasons"]["cirl"]

    def test_corrupt_summary_is_reported(self, tmp_path):
        cell = _cell(tmp_path, 5, "qwen3.5-9b/instruct", metrics={})
        (cell / "failures.json").write_text("{not json")
        out = rr.scan_cell(cell)
        assert out["status"] == "unreadable-failures-json"


class TestEmitCmds:
    def test_one_invocation_per_cell_with_its_own_filter(self, tmp_path):
        """Cells with different holes must not be folded into one sweep."""
        _cell(tmp_path, 0, "qwen3.5-9b/k3-base",
              failures=_failures("qwen3.5-9b/k3-base", {"goldcoin": "ok"}),
              metrics={"goldcoin": True})
        _cell(tmp_path, 1, "qwen3.5-9b/instruct",
              failures=_failures("qwen3.5-9b/instruct",
                                 {"confaide": "FAILED (rc=1)"}))
        _cell(tmp_path, 2, "qwen3.5-9b/k3-verdict",
              failures=_failures("qwen3.5-9b/k3-verdict",
                                 {"cirl": "FAILED (rc=1)", "mmlu": "FAILED (rc=1)"}))

        cmds = rr.emit_cmds(rr.scan_run(tmp_path), "sweep_x", "/venv/bin/python")
        assert len(cmds) == 2
        assert "model=qwen3.5-9b/instruct" in cmds[0]
        assert "'benchmark_filter.include=[confaide]'" in cmds[0]
        assert "model=qwen3.5-9b/k3-verdict" in cmds[1]
        assert "'benchmark_filter.include=[cirl,mmlu]'" in cmds[1]
        assert all("+sweep=sweep_x" in c for c in cmds)

    def test_unreadable_model_is_surfaced_not_dropped(self, tmp_path):
        cell = tmp_path / "0"
        (cell / ".hydra").mkdir(parents=True)
        (cell / ".hydra" / "overrides.yaml").write_text("- foo=bar\n")
        (cell / "failures.json").write_text(json.dumps(
            _failures("", {"cirl": "FAILED (rc=1)"})))
        cmds = rr.emit_cmds(rr.scan_run(tmp_path), "sweep_x", "/venv/bin/python")
        assert len(cmds) == 1
        assert cmds[0].startswith("# SKIPPED")


class TestScanRun:
    def test_cells_are_ordered_numerically(self, tmp_path):
        for i in (0, 1, 2, 10):
            _cell(tmp_path, i, f"m{i}",
                  failures=_failures(f"m{i}", {"goldcoin": "ok"}),
                  metrics={"goldcoin": True})
        assert [c["cell"] for c in rr.scan_run(tmp_path)] == ["0", "1", "2", "10"]

    def test_empty_run_dir_is_an_error(self, tmp_path):
        with pytest.raises(SystemExit):
            rr.scan_run(tmp_path)
