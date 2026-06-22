"""Guards for the PrivacyLens cultural-perturbation eval_all wiring.

Covers the config composition (eval_all culture pipeline + model sweep, and the
per-culture subprocess overrides eval_all issues) plus the W&B run-id
qualification that keeps each (model, culture) a distinct, resumable run.
"""

from __future__ import annotations

import os

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

REPO = "/share/pierson/matt/UAIR"
EVAL_ALL_CONF = os.path.join(REPO, "dagspaces/eval_all/conf")
PRIVACYLENS_CONF = os.path.join(REPO, "dagspaces/privacylens/conf")

CULTURES = ["western", "east_asian", "south_asian", "arabic_me", "african", "african_american"]


def _compose(config_dir, overrides, hydra_cfg=False):
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        return compose(config_name="config", overrides=overrides, return_hydra_config=hydra_cfg)


class TestEvalAllCultureComposition:
    def test_culture_pipeline_has_one_benchmark_per_culture(self):
        cfg = _compose(EVAL_ALL_CONF, ["pipeline=privacylens_culture"])
        benchmarks = OmegaConf.to_container(cfg.benchmarks, resolve=True)
        assert len(benchmarks) == len(CULTURES)
        seen = set()
        for name, bc in benchmarks.items():
            assert bc["pipeline"] == "privacylens_perturb_async"
            assert bc["finalize_pipeline"] == "privacylens_async_finalize"
            assert bc["module"] == "dagspaces.privacylens.cli"
            extra = list(bc.get("extra_args") or [])
            assert len(extra) == 1 and extra[0].startswith("+perturb.culture=")
            seen.add(extra[0].split("=", 1)[1])
        assert seen == set(CULTURES)

    def test_culture_pipeline_uses_async_judge_with_sidecar(self):
        cfg = _compose(EVAL_ALL_CONF, ["pipeline=privacylens_culture"])
        assert cfg.judge.mode == "async"
        assert bool(cfg.judge_sidecar.enabled) is True

    def test_sweep_sets_model_axis_and_multirun(self):
        cfg = _compose(
            EVAL_ALL_CONF,
            ["pipeline=privacylens_culture", "+sweep=privacylens_culture"],
            hydra_cfg=True,
        )
        models = [m.strip() for m in str(cfg.hydra.sweeper.params.model).split(",") if m.strip()]
        # Off-the-shelf instruct baselines across a size range (no fine-tunes).
        assert "qwen3.5-9b/instruct" in models and len(models) >= 2
        assert all(m.endswith("/instruct") for m in models)
        assert str(cfg.hydra.mode) == "RunMode.MULTIRUN"
        # Judge endpoint baked into setup so it survives the sbatch ssh-hop.
        assert any("JUDGE_SERVER_URL=" in line for line in cfg.hydra.launcher.setup)


class TestPerturbSubprocessOverrides:
    """The exact CLIs eval_all builds for export + finalize must compose."""

    @pytest.mark.parametrize("culture", CULTURES)
    def test_export_pipeline_accepts_culture(self, culture):
        cfg = _compose(
            PRIVACYLENS_CONF,
            ["pipeline=privacylens_perturb_async", "model=qwen3.5-9b/base", f"+perturb.culture={culture}"],
        )
        assert OmegaConf.select(cfg, "perturb.culture") == culture
        assert "perturb_culture" in cfg.pipeline.graph.nodes
        # perturb stage feeds the inference stages
        assert cfg.pipeline.graph.nodes.qa_probe_inference.inputs.dataset == "perturb_culture.dataset"

    @pytest.mark.parametrize("culture", CULTURES)
    def test_finalize_pipeline_accepts_culture(self, culture):
        # Shared finalize has no /perturb default; the `+` add must still work.
        cfg = _compose(
            PRIVACYLENS_CONF,
            ["pipeline=privacylens_async_finalize", "model=qwen3.5-9b/base", f"+perturb.culture={culture}"],
        )
        assert OmegaConf.select(cfg, "perturb.culture") == culture


class TestRunIdQualification:
    def test_distinct_per_culture_shared_export_finalize(self):
        from dagspaces.privacylens.orchestrator import _perturb_qualified_dagspace

        ids = set()
        for culture in CULTURES:
            exp = _compose(
                PRIVACYLENS_CONF,
                ["pipeline=privacylens_perturb_async", "model=qwen3.5-9b/base", f"+perturb.culture={culture}"],
            )
            fin = _compose(
                PRIVACYLENS_CONF,
                ["pipeline=privacylens_async_finalize", "model=qwen3.5-9b/base", f"+perturb.culture={culture}"],
            )
            ds_exp = _perturb_qualified_dagspace(exp)
            ds_fin = _perturb_qualified_dagspace(fin)
            assert ds_exp == ds_fin == f"privacylens:{culture}"  # export & finalize share
            ids.add(ds_exp)
        assert len(ids) == len(CULTURES)  # every culture distinct

    def test_plain_run_unchanged(self):
        from dagspaces.privacylens.orchestrator import _perturb_qualified_dagspace

        cfg = _compose(PRIVACYLENS_CONF, ["pipeline=privacylens_async", "model=qwen3.5-9b/base"])
        assert _perturb_qualified_dagspace(cfg) == "privacylens"
