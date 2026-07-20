"""Tests for the local ↔ W&B metrics parity layer.

Covers the invariants ``wiki/integrations/wandb-parity.md`` promises:
mirror keys are byte-identical to metrics.json dotted paths, the file
travels with the run, linkage sidecars never orphan a previous run, sweep
groups derive from output paths, and judge tags come from manifests only.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dagspaces.common.metrics_sync import (
    MIRROR_SEGMENT,
    SIDECAR_FILENAME,
    derive_group_from_output_dir,
    flatten_numeric,
    mirror_metrics_to_wandb,
    served_judges_near,
    write_wandb_sidecar,
)


class FakeLogger:
    """Duck-typed WandbLogger capturing what the mirror does."""

    def __init__(self, run_id: str | None = "run123") -> None:
        self.logged: dict = {}
        self.saved: list = []
        self.tags: list = []
        self._run_id = run_id

    def log_metrics(self, metrics, step=None, commit=True):
        self.logged.update(metrics)

    def save_file(self, path, *, base_path=None):
        self.saved.append((path, base_path))

    def add_tags(self, tags):
        self.tags.extend(tags)

    def run_info(self):
        if self._run_id is None:
            return None
        return {
            "entity": "uair",
            "project": "eval-all",
            "run_id": self._run_id,
            "run_name": f"name-{self._run_id}",
            "run_url": f"https://wandb.test/{self._run_id}",
            "group": "2026-07-19_sweep/22-48-47",
            "tags": ["bench:privacylens"],
        }


class TestFlattenNumeric:
    def test_dotted_keys_match_json_paths(self):
        metrics = {
            "qa_probing": {"accuracy": 0.91, "per_axis": {"who": {"accuracy": 0.5}}},
            "leakage": {"leakage_rate": 0.25},
        }
        flat = flatten_numeric(metrics)
        assert flat["qa_probing.accuracy"] == 0.91
        assert flat["qa_probing.per_axis.who.accuracy"] == 0.5
        assert flat["leakage.leakage_rate"] == 0.25

    def test_non_numeric_leaves_skipped(self):
        flat = flatten_numeric({
            "task": "cirl_trajectory",       # str
            "ids": [1, 2, 3],                # list
            "missing": None,
            "total": 493,
            "nested": {"note": "hi", "rate": 0.1},
        })
        assert flat == {"total": 493, "nested.rate": 0.1}

    def test_bools_become_ints(self):
        flat = flatten_numeric({"passed": True, "failed": False})
        assert flat == {"passed": 1, "failed": 0}

    def test_provenance_counts_are_mirrored(self):
        # metric_provenance blocks are numeric-heavy and must survive —
        # they are the trust contract (wiki/metric-trust.md).
        flat = flatten_numeric({
            "metric_provenance": {
                "leakage.leakage_rate": {"n_total": 493, "n_real": 44,
                                         "n_defaulted": 449,
                                         "default_reason": "no_action"},
            }
        })
        assert flat["metric_provenance.leakage.leakage_rate.n_total"] == 493
        assert flat["metric_provenance.leakage.leakage_rate.n_defaulted"] == 449


class TestDeriveGroup:
    @pytest.mark.parametrize("path,expected", [
        ("/x/multirun/2026-07-19_eval_sft_per_checkpoint_all/22-48-47/3",
         "2026-07-19_eval_sft_per_checkpoint_all/22-48-47"),
        # eval_all child, deep below the arm — same group as the parent
        ("/x/multirun/2026-07-19_eval_sft_per_checkpoint_all/22-48-47/3/"
         "privacylens/privacylens_eval",
         "2026-07-19_eval_sft_per_checkpoint_all/22-48-47"),
        ("/x/outputs/2026-07-20_goldcoin_hipaa/10-15-33",
         "2026-07-20_goldcoin_hipaa/10-15-33"),
        ("/x/outputs/2026-07-20_goldcoin_hipaa/10-15-33/goldcoin_hipaa",
         "2026-07-20_goldcoin_hipaa/10-15-33"),
    ])
    def test_derivation(self, path, expected):
        assert derive_group_from_output_dir(path) == expected

    @pytest.mark.parametrize("path", [
        None, "", "/tmp/somewhere/else",
        "/x/multirun/2026-07-19_sweep",          # no time component
        "/x/multirun/22-48-47/2026-07-19_sweep",  # wrong order
    ])
    def test_no_match(self, path):
        assert derive_group_from_output_dir(path) is None


def _make_outputs_tree(tmp_path: Path, *, with_judge: bool = True) -> Path:
    outputs = tmp_path / "privacylens_eval" / "outputs"
    stage_dir = outputs / "compute_metrics"
    stage_dir.mkdir(parents=True)
    metrics = {
        "qa_probing": {"accuracy": 0.9},
        "leakage": {"leakage_rate": 0.2},
        "task": "privacylens",
    }
    (stage_dir / "metrics.json").write_text(json.dumps(metrics))
    if with_judge:
        judge_dir = outputs / "leakage_judge_batch"
        judge_dir.mkdir()
        (judge_dir / "manifest.json").write_text(json.dumps(
            {"model": "/share/zoo/models/Gemma-4-31B-it"}
        ))
    return stage_dir / "metrics.json"


class TestMirror:
    def test_keys_file_judge_and_sidecar(self, tmp_path):
        mp = _make_outputs_tree(tmp_path)
        logger = FakeLogger()
        flat = mirror_metrics_to_wandb(
            logger, metrics_json_path=str(mp), stage="compute_metrics"
        )
        # Scalar mirror: subdir prefix + MIRROR_SEGMENT + dotted disk key.
        assert flat[f"compute_metrics/{MIRROR_SEGMENT}/qa_probing.accuracy"] == 0.9
        assert logger.logged == flat
        # File upload with the outputs/ root as base_path so the stored
        # name keeps the stage subdir.
        (saved_path, base_path), = logger.saved
        assert saved_path == str(mp)
        assert Path(base_path) == mp.parent.parent
        # Judge from the manifest, not config.
        assert logger.tags == ["judge:Gemma-4-31B-it"]
        # Linkage sidecar written next to metrics.json.
        sidecar = json.loads((mp.parent / SIDECAR_FILENAME).read_text())
        assert sidecar["run_id"] == "run123"
        assert sidecar["run_url"].endswith("run123")

    def test_metrics_dict_preferred_over_file_load(self, tmp_path):
        mp = _make_outputs_tree(tmp_path, with_judge=False)
        logger = FakeLogger()
        flat = mirror_metrics_to_wandb(
            logger,
            metrics={"only": {"this": 1.0}},
            metrics_json_path=str(mp),
            stage="compute_metrics",
        )
        assert set(flat) == {f"compute_metrics/{MIRROR_SEGMENT}/only.this"}

    def test_no_file_still_mirrors_scalars(self):
        logger = FakeLogger()
        flat = mirror_metrics_to_wandb(
            logger, metrics={"accuracy": 0.5}, stage="compute_metrics"
        )
        assert flat == {f"compute_metrics/{MIRROR_SEGMENT}/accuracy": 0.5}
        assert logger.saved == []

    def test_disabled_logger_writes_no_sidecar(self, tmp_path):
        mp = _make_outputs_tree(tmp_path, with_judge=False)
        mirror_metrics_to_wandb(
            FakeLogger(run_id=None), metrics_json_path=str(mp),
            stage="compute_metrics",
        )
        assert not (mp.parent / SIDECAR_FILENAME).exists()


class TestServedJudges:
    def test_reads_all_judge_batch_manifests(self, tmp_path):
        mp = _make_outputs_tree(tmp_path)
        outputs = mp.parent.parent
        helpful = outputs / "helpfulness_judge_batch"
        helpful.mkdir()
        (helpful / "manifest.json").write_text(json.dumps(
            {"model": "/share/zoo/models/Gemma-4-31B-it"}
        ))
        assert served_judges_near(str(mp)) == {"Gemma-4-31B-it"}

    def test_no_manifests(self, tmp_path):
        mp = _make_outputs_tree(tmp_path, with_judge=False)
        assert served_judges_near(str(mp)) == set()


class TestSidecar:
    def test_same_run_is_idempotent(self, tmp_path):
        logger = FakeLogger("aaa")
        write_wandb_sidecar(str(tmp_path), logger)
        write_wandb_sidecar(str(tmp_path), logger)
        sc = json.loads((tmp_path / SIDECAR_FILENAME).read_text())
        assert sc["run_id"] == "aaa"
        assert "previous_runs" not in sc

    def test_new_run_preserves_previous(self, tmp_path):
        write_wandb_sidecar(str(tmp_path), FakeLogger("aaa"))
        write_wandb_sidecar(str(tmp_path), FakeLogger("bbb"))
        sc = json.loads((tmp_path / SIDECAR_FILENAME).read_text())
        assert sc["run_id"] == "bbb"
        assert [r["run_id"] for r in sc["previous_runs"]] == ["aaa"]


class TestOrchestratorMirrorHook:
    """The shared-loop glue: metadata['metrics'] / outputs['metrics_json']
    reach the mirror with no per-dagspace code."""

    def test_mirror_stage_metrics(self, tmp_path):
        from dagspaces.common.orchestrator import _mirror_stage_metrics

        class _Result:
            def __init__(self, outputs, metadata):
                self.outputs = outputs
                self.metadata = metadata

        mp = _make_outputs_tree(tmp_path, with_judge=False)
        logger = FakeLogger()
        _mirror_stage_metrics(
            logger,
            _Result({"metrics_json": str(mp)},
                    {"metrics": {"leakage": {"leakage_rate": 0.2}}}),
            "compute_metrics",
        )
        key = f"compute_metrics/{MIRROR_SEGMENT}/leakage.leakage_rate"
        assert logger.logged[key] == 0.2
        assert (mp.parent / SIDECAR_FILENAME).exists()

    def test_missing_metrics_is_a_noop(self):
        from dagspaces.common.orchestrator import _mirror_stage_metrics

        class _Result:
            outputs = {}
            metadata = {}

        logger = FakeLogger()
        _mirror_stage_metrics(logger, _Result(), "compute_metrics")
        assert logger.logged == {}


class TestSyncScriptDiscovery:
    """scripts/wandb_local_sync.py local-side helpers (no wandb import)."""

    def _make_arm(self, root: Path) -> Path:
        arm = root / "3"
        hydra = arm / "privacylens" / ".hydra"
        hydra.mkdir(parents=True)
        (hydra / "overrides.yaml").write_text(
            "- model=qwen3.5-9b/sft-canonical-ckpt342\n"
        )
        stage = arm / "privacylens" / "privacylens_eval" / "outputs" / "compute_metrics"
        stage.mkdir(parents=True)
        (stage / "metrics.json").write_text(json.dumps(
            {"qa_probing": {"accuracy": 0.8}}
        ))
        return stage / "metrics.json"

    def test_discover_bench_and_model(self, tmp_path):
        from scripts.wandb_local_sync import (
            _bench_of,
            _model_override_near,
            discover_local,
        )

        mp = self._make_arm(tmp_path)
        cells = discover_local(tmp_path)
        assert len(cells) == 1
        cell = cells[0]
        assert cell["subdir"] == "compute_metrics"
        assert cell["flat"] == {"qa_probing.accuracy": 0.8}
        assert cell["sidecar"] is None
        assert _bench_of(cell) == "privacylens_eval"
        assert (_model_override_near(mp, tmp_path)
                == "qwen3.5-9b/sft-canonical-ckpt342")

    def test_discover_picks_up_sidecar(self, tmp_path):
        from scripts.wandb_local_sync import discover_local

        mp = self._make_arm(tmp_path)
        (mp.parent / SIDECAR_FILENAME).write_text(json.dumps(
            {"run_id": "zzz", "entity": "uair", "project": "eval-all"}
        ))
        (cell,) = discover_local(tmp_path)
        assert cell["sidecar"]["run_id"] == "zzz"
