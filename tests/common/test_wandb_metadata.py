"""Tests for ``dagspaces/common/wandb_logger.py`` and the dagspace shims.

Verifies that the cross-dagspace W&B metadata helpers (compute metadata,
run config, tags, WandbConfig) emit the fields the cross-model /
cross-benchmark dashboards depend on.

Migrated from ``scripts/test_wandb_metadata.py`` (custom runner) to
pytest on 2026-05-12. The ``dagspaces.uair`` shim check was dropped at
that time: ``.uair`` is a deprecated dot-prefixed dagspace not used for
COLM (per ``CLAUDE.md``).
"""

from __future__ import annotations

from omegaconf import OmegaConf

from dagspaces.common.config_schema import OutputSpec, PipelineNodeSpec
from dagspaces.common.orchestrator import build_run_config
from dagspaces.common.wandb_logger import (
    WandbConfig,
    _derive_checkpoint_name,
    build_wandb_tags,
    collect_compute_metadata,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_cfg(**overrides):
    """Mock OmegaConf config resembling a real Hydra config."""
    base = {
        "model": {
            "model_source": "/share/pierson/matt/zoo/models/Qwen3-8B",
            "model_family": "qwen3",
            "engine_kwargs": {
                "max_model_len": 8192,
                "tensor_parallel_size": 1,
                "enable_lora": True,
                "max_lora_rank": 64,
            },
            "batch_size": 0,
            "concurrency": 1,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        "prompt": {"task": "compliance"},
        "runtime": {"debug": False, "sample_n": None},
        "wandb": {"enabled": True, "project": "test-project"},
        "experiment": {"name": "test_exp"},
    }
    base.update(overrides)
    return OmegaConf.create(base)


def _make_node(**overrides):
    defaults = dict(
        key="llm_inference_compliance",
        stage="llm_inference",
        depends_on=["load_dataset"],
        inputs={"dataset": "load_dataset.dataset"},
        outputs={
            "dataset": OutputSpec.from_config(
                "dataset", {"path": "out.parquet", "type": "parquet"}
            )
        },
        overrides={"prompt": {"task": "compliance"}},
    )
    defaults.update(overrides)
    return PipelineNodeSpec(**defaults)


# ---------------------------------------------------------------------------
# _derive_checkpoint_name
# ---------------------------------------------------------------------------

class TestDeriveCheckpointName:
    def test_lora_path_with_sft_only_suffix(self):
        name = _derive_checkpoint_name(
            "/share/pierson/matt/UAIR/multirun/2026-03-17/12-37-26/sft_only/outputs/sft/checkpoint",
            "/share/pierson/matt/zoo/models/Qwen3-8B",
        )
        assert name == "Qwen3-8B+sft_only"

    def test_grpo_merged_sft_checkpoint(self):
        name = _derive_checkpoint_name(
            "/share/pierson/matt/UAIR/multirun/2026-03-18/18-10-39/grpo_only/outputs/grpo/checkpoint",
            "/share/pierson/matt/UAIR/multirun/2026-03-18/18-10-39/grpo_only/outputs/grpo/checkpoint/_merged_sft",
        )
        assert name == "_merged_sft+grpo_only"

    def test_specific_checkpoint_step(self):
        name = _derive_checkpoint_name(
            "/path/to/outputs/grpo/checkpoint/checkpoint-227",
            "/path/to/Qwen3-8B",
        )
        assert name == "Qwen3-8B+checkpoint-227"

    def test_no_model_source_falls_back_to_unknown(self):
        name = _derive_checkpoint_name(
            "/path/to/sft_only/outputs/sft/checkpoint", ""
        )
        assert name == "unknown+sft_only"


# ---------------------------------------------------------------------------
# collect_compute_metadata
# ---------------------------------------------------------------------------

class TestCollectComputeMetadata:
    def test_full_config_includes_lora_and_finetuned_flag(self):
        cfg = _make_cfg()
        cfg.model.lora_path = (
            "/share/pierson/matt/UAIR/multirun/2026-03-17/12-37-26/sft_only/"
            "outputs/sft/checkpoint"
        )
        meta = collect_compute_metadata(cfg)
        model = meta.get("model", {})
        assert model.get("model_source") == "/share/pierson/matt/zoo/models/Qwen3-8B"
        assert model.get("model_family") == "qwen3"
        assert model.get("lora_path") is not None
        assert model.get("is_finetuned") is True
        assert model.get("checkpoint_name") == "Qwen3-8B+sft_only"
        assert model.get("chat_template_kwargs", {}).get("enable_thinking") is False

    def test_base_model_has_no_lora_and_not_finetuned(self):
        cfg = _make_cfg()
        meta = collect_compute_metadata(cfg)
        model = meta.get("model", {})
        assert model.get("is_finetuned") is False
        assert model.get("lora_path") is None
        assert model.get("checkpoint_name") == "Qwen3-8B"

    def test_no_model_section_returns_meta_without_model_key(self):
        cfg = OmegaConf.create({"runtime": {"debug": False}})
        meta = collect_compute_metadata(cfg)
        assert "model" not in meta

    def test_explicit_checkpoint_name_wins_over_derived(self):
        cfg = _make_cfg()
        cfg.model.checkpoint_name = "MAR19_K20_GRPO_QWEN3_8B"
        cfg.model.lora_path = "/some/path"
        meta = collect_compute_metadata(cfg)
        assert meta.get("model", {}).get("checkpoint_name") == "MAR19_K20_GRPO_QWEN3_8B"


# ---------------------------------------------------------------------------
# build_run_config
# ---------------------------------------------------------------------------

class TestBuildRunConfig:
    def test_emits_node_stage_dagspace_eval_task_and_checkpoint(self):
        cfg = _make_cfg()
        cfg.model.lora_path = "/path/to/sft_only/outputs/sft/checkpoint"
        node = _make_node()
        inputs = {"dataset": "/path/to/data.parquet"}
        output_paths = {"dataset": "/path/to/out.parquet"}

        rc = build_run_config(
            cfg, node, inputs, output_paths, dagspace_name="goldcoin_hipaa"
        )

        assert rc["node"] == "llm_inference_compliance"
        assert rc["stage"] == "llm_inference"
        assert rc["dagspace"] == "goldcoin_hipaa"
        assert rc["eval_task"] == "compliance"
        assert rc["checkpoint_name"] == "Qwen3-8B+sft_only"
        assert "inputs" in rc
        assert "outputs" in rc


# ---------------------------------------------------------------------------
# build_wandb_tags
# ---------------------------------------------------------------------------

class TestBuildWandbTags:
    def test_finetuned_model_tags(self):
        cfg = _make_cfg()
        cfg.model.lora_path = "/path/to/adapter"
        tags = build_wandb_tags(cfg, dagspace_name="goldcoin_hipaa")
        assert "bench:goldcoin_hipaa" in tags
        assert "family:qwen3" in tags
        assert "finetuned" in tags
        assert "task:compliance" in tags

    def test_base_model_tags_drop_finetuned(self):
        cfg = _make_cfg()
        tags = build_wandb_tags(cfg, dagspace_name="privacylens")
        assert "bench:privacylens" in tags
        assert "base" in tags
        assert "finetuned" not in tags


# ---------------------------------------------------------------------------
# WandbConfig.from_hydra_config — common + dagspace shims
# ---------------------------------------------------------------------------

class TestWandbConfigFromHydra:
    def test_common_auto_tags_finetuned(self):
        cfg = _make_cfg()
        cfg.model.lora_path = "/path/to/adapter"
        wc = WandbConfig.from_hydra_config(cfg, dagspace_name="goldcoin_hipaa")
        assert "bench:goldcoin_hipaa" in wc.tags
        assert "family:qwen3" in wc.tags
        assert "finetuned" in wc.tags
        assert wc.dagspace_name == "goldcoin_hipaa"


class TestDagspaceShims:
    """Each dagspace's ``wandb_logger`` shim must default ``dagspace_name``
    to its own slug so the cross-benchmark dashboards group runs correctly.

    The deprecated ``.uair`` dagspace is intentionally NOT tested here —
    it has been dot-prefixed out of COLM scope (see ``CLAUDE.md``)."""

    def test_goldcoin_hipaa_shim(self):
        from dagspaces.goldcoin_hipaa.wandb_logger import WandbConfig as GC_WC
        gc = GC_WC.from_hydra_config(_make_cfg())
        assert gc.dagspace_name == "goldcoin_hipaa"
        assert "bench:goldcoin_hipaa" in gc.tags

    def test_privacylens_shim(self):
        from dagspaces.privacylens.wandb_logger import WandbConfig as PL_WC
        pl = PL_WC.from_hydra_config(_make_cfg())
        assert pl.dagspace_name == "privacylens"

    def test_vlm_geoprivacy_bench_shim(self):
        from dagspaces.vlm_geoprivacy_bench.wandb_logger import WandbConfig as VLM_WC
        vlm = VLM_WC.from_hydra_config(_make_cfg())
        assert vlm.dagspace_name == "vlm_geoprivacy_bench"

    def test_historical_norms_shim(self):
        from dagspaces.historical_norms.wandb_logger import WandbConfig as HN_WC
        hn = HN_WC.from_hydra_config(_make_cfg())
        assert hn.dagspace_name == "historical_norms"

    def test_grpo_training_shim(self):
        from dagspaces.grpo_training.wandb_logger import WandbConfig as GRPO_WC
        grpo = GRPO_WC.from_hydra_config(_make_cfg())
        assert grpo.dagspace_name == "grpo_training"
