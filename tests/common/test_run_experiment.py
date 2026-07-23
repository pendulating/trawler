"""Tests for the generic eval-dagspace run loop in dagspaces/common/orchestrator.py.

Covers Finding 1 / Finding 2 of wiki/jul19_refactoring.md: the SLURM/NFS
result-waiting block (``await_slurm_result``), the wandb logger factory
(``make_wandb_logger``), context serialization, and a local ``run_experiment``
run against a fake dagspace module.

These tests gate the orchestrator-unification refactor: the seven eval
dagspaces are migrated onto this shared loop one at a time, and a parity test
(``test_migrated_dagspaces_expose_hooks``) is extended as each lands.
"""

from __future__ import annotations

import pickle
import sys
import types
from typing import Any, Dict, List

import pytest
from omegaconf import OmegaConf

from dagspaces.common import orchestrator as orch
from dagspaces.common.orchestrator import (
    OrchestratorHooks,
    await_slurm_result,
    make_wandb_logger,
    run_experiment,
    serialize_context_data,
    _rebuild_node,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class _FakePaths:
    def __init__(self, result_pickle: str) -> None:
        self.result_pickle = result_pickle


class _FakeJob:
    """Mimics the slice of a submitit job handle that await_slurm_result uses."""

    def __init__(
        self,
        *,
        job_id: str = "12345",
        result_value: Any = None,
        result_exc: Exception | None = None,
        result_pickle: str = "/nonexistent/result.pkl",
    ) -> None:
        self.job_id = job_id
        self._result_value = result_value
        self._result_exc = result_exc
        self.paths = _FakePaths(result_pickle)

    def result(self):
        if self._result_exc is not None:
            raise self._result_exc
        return self._result_value


class _FakeCompleted:
    def __init__(self, stdout: str) -> None:
        self.stdout = stdout
        self.stderr = ""


def _payload() -> Dict[str, Any]:
    return {"outputs": {"dataset": "/tmp/x.parquet"}, "metadata": {"rows": 3}}


# ---------------------------------------------------------------------------
# await_slurm_result
# ---------------------------------------------------------------------------

class TestAwaitSlurmResult:
    def test_happy_path_dict(self):
        job = _FakeJob(result_value=_payload())
        result = await_slurm_result(job, OmegaConf.create({}), "node_a")
        assert result.outputs == {"dataset": "/tmp/x.parquet"}
        assert result.metadata == {"rows": 3}

    def test_tuple_unpacked_success(self):
        job = _FakeJob(result_value=("done", _payload()))
        result = await_slurm_result(job, OmegaConf.create({}), "node_a")
        assert result.outputs == {"dataset": "/tmp/x.parquet"}

    def test_tuple_error_raises(self):
        job = _FakeJob(result_value=("error", "boom traceback"))
        with pytest.raises(RuntimeError, match="boom traceback"):
            await_slurm_result(job, OmegaConf.create({}), "node_a")

    def test_non_dict_result_raises(self):
        job = _FakeJob(result_value="just a string")
        with pytest.raises(RuntimeError, match="unexpected result type"):
            await_slurm_result(job, OmegaConf.create({}), "node_a")

    def test_result_pickle_recovery(self, tmp_path, monkeypatch):
        """A 'has not produced any output' error recovers from the NFS pickle."""
        monkeypatch.setattr(orch.time, "sleep", lambda *_: None)
        pickle_path = tmp_path / "result.pkl"
        pickle_path.write_bytes(pickle.dumps(("done", _payload())))
        job = _FakeJob(
            result_exc=RuntimeError("Job has not produced any output"),
            result_pickle=str(pickle_path),
        )
        cfg = OmegaConf.create({"runtime": {"submitit_result_wait_s": 5}})
        result = await_slurm_result(job, cfg, "node_a")
        assert result.outputs == {"dataset": "/tmp/x.parquet"}

    def test_squeue_fallback(self, tmp_path, monkeypatch):
        """If the pickle is absent at first, poll squeue then re-read the pickle."""
        monkeypatch.setattr(orch.time, "sleep", lambda *_: None)
        pickle_path = tmp_path / "result.pkl"

        def fake_subprocess_run(cmd, *args, **kwargs):
            # First squeue call: job has left the queue; drop the result pickle
            # so the post-squeue wait finds it (simulates NFS propagation).
            pickle_path.write_bytes(pickle.dumps(("done", _payload())))
            return _FakeCompleted(stdout="")

        monkeypatch.setattr(orch.subprocess, "run", fake_subprocess_run)
        job = _FakeJob(
            result_exc=RuntimeError("result_pickle missing"),
            result_pickle=str(pickle_path),
        )
        # wait_s=0 => the first pickle-wait loop is skipped, forcing the squeue path.
        cfg = OmegaConf.create({"runtime": {"submitit_result_wait_s": 0}})
        result = await_slurm_result(job, cfg, "node_a")
        assert result.outputs == {"dataset": "/tmp/x.parquet"}

    def test_all_recovery_fails_reraises(self, tmp_path, monkeypatch):
        """With no pickle and a failed squeue fallback, the original error propagates."""
        monkeypatch.setattr(orch.time, "sleep", lambda *_: None)

        def fake_subprocess_run(cmd, *args, **kwargs):
            raise OSError("squeue not available")

        monkeypatch.setattr(orch.subprocess, "run", fake_subprocess_run)
        job = _FakeJob(
            result_exc=RuntimeError("Job has not produced any output"),
            result_pickle=str(tmp_path / "never.pkl"),
        )
        cfg = OmegaConf.create({"runtime": {"submitit_result_wait_s": 0}})
        with pytest.raises(RuntimeError, match="has not produced any output"):
            await_slurm_result(job, cfg, "node_a")


# ---------------------------------------------------------------------------
# Fake dagspace module plumbing (for make_wandb_logger / run_experiment)
# ---------------------------------------------------------------------------

def _install_fake_dagspace(monkeypatch, *, metrics=None, enabled=False):
    """Register a fake ``fakeds`` dagspace package in sys.modules.

    Returns (hooks, recorded) where ``recorded`` collects log_eval_metrics calls.
    """
    recorded: List[Dict[str, Any]] = []

    def log_eval_metrics(logger, m, stage):
        recorded.append({"metrics": m, "stage": stage})

    hooks = OrchestratorHooks(
        dagspace_module="fakeds.orchestrator",
        dagspace_name="fakeds",
        output_subdir="fakeds",
        job_prefix="FAKE",
        config_dir="/tmp/fakeds-conf",
        log_eval_metrics=log_eval_metrics,
        wandb_dagspace=lambda cfg: "",  # empty => pipeline_run_id not called
        use_srun=False,
    )

    class _NoopRunner:
        stage_name = "noop"

        def run(self, context):
            meta: Dict[str, Any] = {"rows": 0}
            if metrics is not None:
                meta["metrics"] = metrics
            return orch.StageResult(outputs={}, metadata=meta)

    orch_mod = types.ModuleType("fakeds.orchestrator")
    orch_mod.ORCHESTRATOR_HOOKS = hooks
    orch_mod.get_stage_registry = lambda: {"noop": _NoopRunner()}

    _enabled = enabled

    class _WBConfig:
        enabled = _enabled

        @staticmethod
        def from_hydra_config(cfg):
            return _WBConfig()

    wl_mod = types.ModuleType("fakeds.wandb_logger")
    wl_mod.WandbConfig = _WBConfig
    wl_mod.WandbLogger = object  # not used when disabled
    wl_mod.pipeline_run_id = lambda cfg, dagspace="": "fake-run-id"

    pkg_mod = types.ModuleType("fakeds")
    pkg_mod.__path__ = []  # mark as package

    monkeypatch.setitem(sys.modules, "fakeds", pkg_mod)
    monkeypatch.setitem(sys.modules, "fakeds.orchestrator", orch_mod)
    monkeypatch.setitem(sys.modules, "fakeds.wandb_logger", wl_mod)
    return hooks, recorded


class TestMakeWandbLogger:
    def test_disabled_returns_noop(self, monkeypatch):
        hooks, _ = _install_fake_dagspace(monkeypatch, enabled=False)
        logger = make_wandb_logger(OmegaConf.create({}), hooks, stage="x")
        assert isinstance(logger, orch._NoOpLogger)

    def test_wandb_dagspace_callable_honored(self, monkeypatch):
        hooks, _ = _install_fake_dagspace(monkeypatch, enabled=False)
        # A non-empty wandb_dagspace must be forwarded to pipeline_run_id.
        seen: Dict[str, Any] = {}
        wl = sys.modules["fakeds.wandb_logger"]
        wl.pipeline_run_id = lambda cfg, dagspace="": seen.setdefault("dagspace", dagspace) or "id"
        hooks2 = OrchestratorHooks(
            dagspace_module="fakeds.orchestrator",
            dagspace_name="fakeds",
            output_subdir="fakeds",
            job_prefix="FAKE",
            config_dir="/tmp/fakeds-conf",
            log_eval_metrics=lambda *a: None,
            wandb_dagspace=lambda cfg: "fakeds:culture_x",
        )
        make_wandb_logger(OmegaConf.create({}), hooks2, stage="x")
        assert seen["dagspace"] == "fakeds:culture_x"


# ---------------------------------------------------------------------------
# serialize_context_data / _rebuild_node round-trip
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_round_trip(self):
        from dagspaces.common.config_schema import PipelineNodeSpec, OutputSpec

        node = PipelineNodeSpec(
            key="n1",
            stage="noop",
            depends_on=["n0"],
            inputs={"dataset": "src"},
            outputs={"dataset": OutputSpec(key="dataset", type="parquet", path="out.parquet")},
            overrides={"a": 1},
            launcher="slurm_gpu_1x",
            wandb_suffix="suffix",
        )
        node_cfg = OmegaConf.create({"runtime": {"stage": "noop"}})
        data = serialize_context_data(
            node_cfg, node, {"dataset": "src"}, {"dataset": "/abs/out.parquet"},
            "/abs", "/abs", "fakeds.orchestrator",
        )
        assert data["dagspace_module"] == "fakeds.orchestrator"
        rebuilt = _rebuild_node(data["node"])
        assert rebuilt.key == "n1"
        assert rebuilt.stage == "noop"
        assert rebuilt.depends_on == ["n0"]
        assert rebuilt.launcher == "slurm_gpu_1x"
        assert rebuilt.wandb_suffix == "suffix"
        assert rebuilt.outputs["dataset"].path == "out.parquet"


# ---------------------------------------------------------------------------
# run_experiment — local (no-launcher) path against the fake dagspace
# ---------------------------------------------------------------------------

class TestRunExperimentLocal:
    def _cfg(self, tmp_path, with_metrics_node: bool = False):
        nodes: Dict[str, Any] = {"only": {"stage": "noop"}}
        return OmegaConf.create({
            "pipeline": {
                "sources": {},
                "graph": {"nodes": nodes},
                "output_root": str(tmp_path / "out"),
            },
            "runtime": {},
        })

    def test_local_run_writes_manifest_and_logs_metrics(self, tmp_path, monkeypatch):
        hooks, recorded = _install_fake_dagspace(
            monkeypatch, metrics={"accuracy": 0.5}, enabled=False
        )
        cfg = self._cfg(tmp_path)
        run_experiment(cfg, hooks)

        manifest_path = tmp_path / "out" / "pipeline_manifest.json"
        assert manifest_path.exists()
        import json
        manifest = json.loads(manifest_path.read_text())
        assert "only" in manifest["nodes"]
        assert manifest["nodes"]["only"]["stage"] == "noop"
        # log_eval_metrics fired on the monitor logger because metrics were present.
        assert recorded and recorded[0]["metrics"] == {"accuracy": 0.5}
        assert recorded[0]["stage"] == "noop"

    def test_local_run_no_metrics_no_log(self, tmp_path, monkeypatch):
        hooks, recorded = _install_fake_dagspace(monkeypatch, metrics=None, enabled=False)
        run_experiment(self._cfg(tmp_path), hooks)
        assert recorded == []


# ---------------------------------------------------------------------------
# Migration parity gate — extended as each eval dagspace is migrated.
# ---------------------------------------------------------------------------

_MIGRATED: List[str] = [  # extended as each eval dagspace lands
    "dagspaces.mmlu.orchestrator",
    "dagspaces.simpleqa_verified.orchestrator",
    "dagspaces.goldcoin_hipaa.orchestrator",
    "dagspaces.vlm_geoprivacy_bench.orchestrator",
    "dagspaces.vlm_geoprivacy_aug.orchestrator",
    "dagspaces.confaide.orchestrator",
    "dagspaces.cirl.orchestrator",
    "dagspaces.privacylens.orchestrator",
    "dagspaces.ci_heuristic.orchestrator",
]


@pytest.mark.parametrize("module_path", _MIGRATED or ["__skip__"])
def test_migrated_dagspaces_expose_hooks(module_path):
    """A migrated dagspace exposes ORCHESTRATOR_HOOKS + get_stage_registry,
    and its hooks.dagspace_module points back at itself."""
    if module_path == "__skip__":
        pytest.skip("no dagspaces migrated yet")
    import importlib
    mod = importlib.import_module(module_path)
    hooks = mod.ORCHESTRATOR_HOOKS
    assert isinstance(hooks, OrchestratorHooks)
    assert hooks.dagspace_module == module_path
    assert callable(mod.get_stage_registry)
    assert isinstance(mod.get_stage_registry(), dict)


# ---------------------------------------------------------------------------
# Behavior-preservation guard: each migrated dagspace's hooks must reproduce
# the OLD hardcoded orchestrator params exactly (output paths, W&B run-id key,
# SLURM job names). This is the local half of the byte-parity acceptance check
# (the other half — identical outputs/W&B keys on a real GPU+data run — needs a
# cluster job and is a manual step; see jul19_orchestrator_unification_plan.md §8).
# wandb_dagspace is asserted with an empty cfg (no perturb culture).
# ---------------------------------------------------------------------------

_GOLDEN_PARAMS: Dict[str, Dict[str, Any]] = {
    "dagspaces.mmlu.orchestrator": dict(
        dagspace_name="mmlu", output_subdir="mmlu", job_prefix="MMLU",
        wandb_dagspace="mmlu", use_srun=False),
    "dagspaces.simpleqa_verified.orchestrator": dict(
        dagspace_name="simpleqa_verified", output_subdir="simpleqa_verified",
        job_prefix="SimpleQAVerified", wandb_dagspace="simpleqa_verified", use_srun=False),
    "dagspaces.goldcoin_hipaa.orchestrator": dict(
        dagspace_name="goldcoin_hipaa", output_subdir="goldcoin_hipaa",
        job_prefix="GoldCoin", wandb_dagspace="goldcoin", use_srun=False),
    "dagspaces.vlm_geoprivacy_bench.orchestrator": dict(
        dagspace_name="vlm_geoprivacy_bench", output_subdir="vlm_geoprivacy_bench",
        job_prefix="VLM", wandb_dagspace="vlm_geoprivacy", use_srun=False),
    "dagspaces.vlm_geoprivacy_aug.orchestrator": dict(
        dagspace_name="vlm_geoprivacy_aug", output_subdir="vlm_geoprivacy_aug",
        job_prefix="VLM", wandb_dagspace="vlm_geoprivacy_aug", use_srun=False),
    "dagspaces.confaide.orchestrator": dict(
        dagspace_name="confaide", output_subdir="confaide", job_prefix="CONFAIDE",
        wandb_dagspace="confaide", use_srun=False),
    "dagspaces.cirl.orchestrator": dict(
        dagspace_name="cirl", output_subdir="cirl",
        job_prefix="CIRL", wandb_dagspace="cirl", use_srun=False),
    "dagspaces.privacylens.orchestrator": dict(
        dagspace_name="privacylens", output_subdir="privacylens_eval",
        job_prefix="PLens", wandb_dagspace="privacylens", use_srun=False),
    "dagspaces.ci_heuristic.orchestrator": dict(
        dagspace_name="ci_heuristic", output_subdir="ci_heuristic",
        job_prefix="CIH", wandb_dagspace="ci_heuristic", use_srun=False),
}


@pytest.mark.parametrize("module_path", sorted(_GOLDEN_PARAMS))
def test_migrated_dagspace_golden_params(module_path):
    """Hooks reproduce the pre-refactor hardcoded params exactly."""
    import importlib
    mod = importlib.import_module(module_path)
    hooks = mod.ORCHESTRATOR_HOOKS
    expected = _GOLDEN_PARAMS[module_path]
    assert hooks.dagspace_name == expected["dagspace_name"]
    assert hooks.output_subdir == expected["output_subdir"]
    assert hooks.job_prefix == expected["job_prefix"]
    assert hooks.use_srun is expected["use_srun"]
    assert hooks.wandb_dagspace(OmegaConf.create({})) == expected["wandb_dagspace"]


def test_privacylens_culture_qualifies_wandb_dagspace():
    """privacylens folds perturb.culture into the run-id key (preserves the
    pre-refactor _perturb_qualified_dagspace behavior)."""
    import importlib
    mod = importlib.import_module("dagspaces.privacylens.orchestrator")
    hooks = mod.ORCHESTRATOR_HOOKS
    assert hooks.wandb_dagspace(OmegaConf.create({})) == "privacylens"
    assert hooks.wandb_dagspace(OmegaConf.create({"perturb": {"culture": "us"}})) == "privacylens:us"
