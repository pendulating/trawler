"""Guard: each dagspace's W&B identity and shim contract.

``dagspaces/common/wandb_shim.py`` builds all eleven logger pairs. The values
below are the ones the hand-written shims carried before the 2026-08-12
migration. They are pinned because a wrong project or experiment name does
not fail — it silently files a benchmark's runs under a second name and
splits its dashboard history.
"""

from __future__ import annotations

import importlib
import inspect

import pytest

from dagspaces.common.wandb_logger import WandbConfig as BaseWandbConfig
from dagspaces.common.wandb_logger import WandbLogger as BaseWandbLogger

# dagspace -> (project, experiment name, env var prefix)
IDENTITY = {
    "ci_heuristic":         ("ci-heuristic",                "ci_heuristic",        ""),
    "cirl":                 ("cirl-729",                    "CIRL-729",            ""),
    "confaide":             ("confaide",                    "CONFAIDE",            ""),
    "goldcoin_hipaa":       ("goldcoin-hipaa",              "GoldCoin-HIPAA",      ""),
    "grpo_training":        ("grpo-ci-training",            "grpo_training",       "GRPO_TRAINING"),
    "historical_norms":     ("historical-norms-extraction", "historical_norms",    "HISTORICAL_NORMS"),
    "mmlu":                 ("mmlu",                        "MMLU",                ""),
    "privacylens":          ("privacylens-eval",            "privacylens",         ""),
    "simpleqa_verified":    ("simpleqa-verified",           "SimpleQA-Verified",   ""),
    "vlm_geoprivacy_aug":   ("vlm-geoprivacy-bench",        "VLM-GeoPrivacyBench", ""),
    "vlm_geoprivacy_bench": ("vlm-geoprivacy-bench",        "VLM-GeoPrivacyBench", ""),
}

FULL_COLUMN_STAGES = {
    "ci_heuristic":         {"traverse", "tp_probe"},
    "cirl":                 {"llm_inference"},
    "confaide":             {"llm_inference"},
    "goldcoin_hipaa":       {"llm_inference"},
    "grpo_training":        {"sft_data_prep", "reward_prep"},
    "historical_norms":     {"norm_reasoning", "norm_extraction",
                             "norm_role_abstraction", "ci_reasoning",
                             "ci_extraction", "fetch_gutenberg"},
    "mmlu":                 {"llm_inference"},
    "privacylens":          {"qa_probe_inference", "agent_action_inference",
                             "leakage_judge_inference", "compute_metrics"},
    "simpleqa_verified":    {"llm_inference"},
    "vlm_geoprivacy_aug":   {"vlm_mcq_inference", "vlm_freeform_inference",
                             "granularity_judge"},
    "vlm_geoprivacy_bench": {"vlm_mcq_inference", "vlm_freeform_inference",
                             "granularity_judge"},
}

# The three dagspaces that also pin key prefixes and internal columns.
EXTRA_INTERNAL_COLUMNS = {
    "grpo_training":    {"messages", "norm_universe_json"},
    "historical_norms": {"reasoning_data", "ci_flows_raw"},
    "privacylens":      {"seed", "vignette", "trajectory", "S", "V", "T"},
}

DAGSPACES = sorted(IDENTITY)


def _shim(dagspace):
    return importlib.import_module(f"dagspaces.{dagspace}.wandb_logger")


@pytest.mark.parametrize("dagspace", DAGSPACES)
def test_wandb_identity_is_pinned(dagspace):
    project, experiment, prefix = IDENTITY[dagspace]
    defaults = _shim(dagspace).WandbConfig.trawler_shim_defaults

    assert defaults["default_project"] == project, (
        f"{dagspace} would log to project {defaults['default_project']!r}. "
        f"A renamed project splits this benchmark's dashboard history."
    )
    assert defaults["default_experiment_name"] == experiment
    assert defaults["env_var_prefix"] == prefix
    assert defaults["dagspace_name"] == dagspace


@pytest.mark.parametrize("dagspace", DAGSPACES)
def test_full_column_stages_are_pinned(dagspace):
    """These stages log their FULL table rather than a trimmed one."""
    defaults = _shim(dagspace).WandbConfig.trawler_shim_defaults
    assert set(defaults["full_column_stages"]) == FULL_COLUMN_STAGES[dagspace]


@pytest.mark.parametrize("dagspace", sorted(EXTRA_INTERNAL_COLUMNS))
def test_columns_that_break_wandb_tables_stay_excluded(dagspace):
    """Nested/ragged columns must stay out of W&B tables.

    These are dicts and arrays whose shape varies per row. W&B raises on them,
    so the exclusion is what keeps these dagspaces able to log at all.
    """
    defaults = _shim(dagspace).WandbConfig.trawler_shim_defaults
    assert set(defaults["extra_internal_columns"]) == EXTRA_INTERNAL_COLUMNS[dagspace]


@pytest.mark.parametrize("dagspace", sorted(EXTRA_INTERNAL_COLUMNS))
def test_key_prefixes_match_the_full_column_stages(dagspace):
    defaults = _shim(dagspace).WandbConfig.trawler_shim_defaults
    assert set(defaults["full_column_key_prefixes"]) == {
        f"{s}/" for s in FULL_COLUMN_STAGES[dagspace]
    }


@pytest.mark.parametrize("dagspace", DAGSPACES)
def test_shim_exports_pipeline_run_id(dagspace):
    """common/orchestrator.make_wandb_logger calls wl.pipeline_run_id(...).

    It imports the dagspace's wandb_logger module BY NAME and reads that
    attribute off it. A shim that drops the re-export breaks every W&B run of
    that dagspace, and only at runtime.
    """
    mod = _shim(dagspace)
    assert hasattr(mod, "pipeline_run_id"), (
        f"dagspaces/{dagspace}/wandb_logger.py must re-export pipeline_run_id"
    )
    assert callable(mod.pipeline_run_id)


@pytest.mark.parametrize("dagspace", DAGSPACES)
def test_shim_calls_ensure_local_tmpdir_at_import(dagspace):
    """TMPDIR must move off the /share network mount when the module loads."""
    src = inspect.getsource(_shim(dagspace))
    assert f'ensure_local_tmpdir("{dagspace}")' in src, (
        f"dagspaces/{dagspace}/wandb_logger.py must call "
        f'ensure_local_tmpdir("{dagspace}") at import time'
    )


@pytest.mark.parametrize("dagspace", DAGSPACES)
def test_shim_classes_subclass_the_common_ones(dagspace):
    mod = _shim(dagspace)
    assert issubclass(mod.WandbConfig, BaseWandbConfig)
    assert issubclass(mod.WandbLogger, BaseWandbLogger)


@pytest.mark.parametrize("dagspace", DAGSPACES)
def test_logger_accepts_the_orchestrator_call_signature(dagspace):
    """common/orchestrator.make_wandb_logger passes wandb_id and resume."""
    sig = inspect.signature(_shim(dagspace).WandbLogger.__init__)
    for name in ("cfg", "stage", "run_id", "run_config", "wandb_id", "resume"):
        assert name in sig.parameters, f"{dagspace} logger lacks {name!r}"


@pytest.mark.parametrize("dagspace", DAGSPACES)
def test_explicit_kwargs_beat_the_shim_defaults(dagspace):
    """The defaults apply with setdefault, so a caller can still override."""
    defaults = _shim(dagspace).WandbConfig.trawler_shim_defaults
    assert "default_project" in defaults
    # Rebuild through the factory with an override and confirm it wins.
    from dagspaces.common.wandb_shim import make_wandb_shim

    Config, _ = make_wandb_shim("probe", default_project="from-shim")
    assert Config.trawler_shim_defaults["default_project"] == "from-shim"
    assert Config.trawler_shim_defaults["dagspace_name"] == "probe"


def test_two_vlm_dagspaces_share_a_project_but_not_a_tag():
    """The augmented arm and the base benchmark belong in one dashboard.

    They must stay separable by the dagspace tag.
    """
    aug = _shim("vlm_geoprivacy_aug").WandbConfig.trawler_shim_defaults
    bench = _shim("vlm_geoprivacy_bench").WandbConfig.trawler_shim_defaults
    assert aug["default_project"] == bench["default_project"]
    assert aug["dagspace_name"] != bench["dagspace_name"]


def test_every_dagspace_with_a_shim_is_covered_here():
    """A new dagspace must be added to IDENTITY, not silently skipped."""
    import glob
    import os

    on_disk = {
        p.split(os.sep)[1]
        for p in glob.glob("dagspaces/*/wandb_logger.py")
        if p.split(os.sep)[1] != "common"
    }
    assert on_disk == set(DAGSPACES), (
        f"shims present but untested: {sorted(on_disk - set(DAGSPACES))}; "
        f"tested but absent: {sorted(set(DAGSPACES) - on_disk)}"
    )
