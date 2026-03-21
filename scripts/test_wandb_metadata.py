"""Verification script for enriched W&B metadata logging.

Tests that collect_compute_metadata(), build_run_config(), and build_wandb_tags()
produce the expected fields for cross-model, cross-benchmark comparison queries.

Usage:
    python scripts/test_wandb_metadata.py
"""

import os
import sys

from omegaconf import OmegaConf

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dagspaces.common.wandb_logger import (
    _derive_checkpoint_name,
    build_wandb_tags,
    collect_compute_metadata,
)
from dagspaces.common.orchestrator import build_run_config
from dagspaces.common.config_schema import PipelineNodeSpec, OutputSpec


def _make_cfg(**overrides):
    """Build a mock OmegaConf config resembling a real Hydra config."""
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
        outputs={"dataset": OutputSpec.from_config("dataset", {"path": "out.parquet", "type": "parquet"})},
        overrides={"prompt": {"task": "compliance"}},
    )
    defaults.update(overrides)
    return PipelineNodeSpec(**defaults)


# ── Test 1: _derive_checkpoint_name ──────────────────────────────────────

def test_derive_checkpoint_name():
    print("Test 1: _derive_checkpoint_name")

    # LoRA path with generic suffixes
    name = _derive_checkpoint_name(
        "/share/pierson/matt/UAIR/multirun/2026-03-17/12-37-26/sft_only/outputs/sft/checkpoint",
        "/share/pierson/matt/zoo/models/Qwen3-8B",
    )
    assert name == "Qwen3-8B+sft_only", f"  FAIL: got '{name}'"

    # GRPO checkpoint
    name = _derive_checkpoint_name(
        "/share/pierson/matt/UAIR/multirun/2026-03-18/18-10-39/grpo_only/outputs/grpo/checkpoint",
        "/share/pierson/matt/UAIR/multirun/2026-03-18/18-10-39/grpo_only/outputs/grpo/checkpoint/_merged_sft",
    )
    assert name == "_merged_sft+grpo_only", f"  FAIL: got '{name}'"

    # Specific checkpoint step
    name = _derive_checkpoint_name(
        "/path/to/outputs/grpo/checkpoint/checkpoint-227",
        "/path/to/Qwen3-8B",
    )
    assert name == "Qwen3-8B+checkpoint-227", f"  FAIL: got '{name}'"

    # No model source
    name = _derive_checkpoint_name("/path/to/sft_only/outputs/sft/checkpoint", "")
    assert name == "unknown+sft_only", f"  FAIL: got '{name}'"

    print("  PASS")


# ── Test 2: collect_compute_metadata — full config ───────────────────────

def test_collect_metadata_full():
    print("Test 2: collect_compute_metadata (full config)")

    cfg = _make_cfg()
    cfg.model.lora_path = "/share/pierson/matt/UAIR/multirun/2026-03-17/12-37-26/sft_only/outputs/sft/checkpoint"
    meta = collect_compute_metadata(cfg)

    model = meta.get("model", {})
    assert model.get("model_source") == "/share/pierson/matt/zoo/models/Qwen3-8B", f"  FAIL model_source: {model}"
    assert model.get("model_family") == "qwen3", f"  FAIL model_family: {model}"
    assert model.get("lora_path") is not None, f"  FAIL lora_path missing: {model}"
    assert model.get("is_finetuned") is True, f"  FAIL is_finetuned: {model}"
    assert model.get("checkpoint_name") == "Qwen3-8B+sft_only", f"  FAIL checkpoint_name: {model.get('checkpoint_name')}"
    assert model.get("chat_template_kwargs", {}).get("enable_thinking") is False, f"  FAIL chat_template_kwargs: {model}"

    print("  PASS")


# ── Test 3: collect_compute_metadata — base model (no LoRA) ─────────────

def test_collect_metadata_base():
    print("Test 3: collect_compute_metadata (base model, no LoRA)")

    cfg = _make_cfg()
    meta = collect_compute_metadata(cfg)

    model = meta.get("model", {})
    assert model.get("is_finetuned") is False, f"  FAIL is_finetuned should be False: {model}"
    assert model.get("lora_path") is None, f"  FAIL lora_path should be None: {model}"
    assert model.get("checkpoint_name") == "Qwen3-8B", f"  FAIL checkpoint_name: {model.get('checkpoint_name')}"

    print("  PASS")


# ── Test 4: collect_compute_metadata — no model config ───────────────────

def test_collect_metadata_no_model():
    print("Test 4: collect_compute_metadata (no model config)")

    cfg = OmegaConf.create({"runtime": {"debug": False}})
    meta = collect_compute_metadata(cfg)
    assert "model" not in meta, f"  FAIL: model key should not be present: {meta}"

    print("  PASS")


# ── Test 5: collect_compute_metadata — explicit checkpoint_name ──────────

def test_collect_metadata_explicit_checkpoint():
    print("Test 5: collect_compute_metadata (explicit checkpoint_name)")

    cfg = _make_cfg()
    cfg.model.checkpoint_name = "MAR19_K20_GRPO_QWEN3_8B"
    cfg.model.lora_path = "/some/path"
    meta = collect_compute_metadata(cfg)

    model = meta.get("model", {})
    assert model.get("checkpoint_name") == "MAR19_K20_GRPO_QWEN3_8B", f"  FAIL: {model.get('checkpoint_name')}"

    print("  PASS")


# ── Test 6: build_run_config ─────────────────────────────────────────────

def test_build_run_config():
    print("Test 6: build_run_config")

    cfg = _make_cfg()
    cfg.model.lora_path = "/path/to/sft_only/outputs/sft/checkpoint"
    node = _make_node()
    inputs = {"dataset": "/path/to/data.parquet"}
    output_paths = {"dataset": "/path/to/out.parquet"}

    rc = build_run_config(cfg, node, inputs, output_paths, dagspace_name="goldcoin_hipaa")

    assert rc["node"] == "llm_inference_compliance"
    assert rc["stage"] == "llm_inference"
    assert rc["dagspace"] == "goldcoin_hipaa"
    assert rc["eval_task"] == "compliance", f"  FAIL eval_task: {rc.get('eval_task')}"
    assert rc["checkpoint_name"] == "Qwen3-8B+sft_only", f"  FAIL checkpoint_name: {rc.get('checkpoint_name')}"
    assert "inputs" in rc
    assert "outputs" in rc

    print("  PASS")


# ── Test 7: build_wandb_tags ─────────────────────────────────────────────

def test_build_wandb_tags():
    print("Test 7: build_wandb_tags")

    # Finetuned model
    cfg = _make_cfg()
    cfg.model.lora_path = "/path/to/adapter"
    tags = build_wandb_tags(cfg, dagspace_name="goldcoin_hipaa")
    assert "bench:goldcoin_hipaa" in tags, f"  FAIL bench tag: {tags}"
    assert "family:qwen3" in tags, f"  FAIL family tag: {tags}"
    assert "finetuned" in tags, f"  FAIL finetuned tag: {tags}"
    assert "task:compliance" in tags, f"  FAIL task tag: {tags}"

    # Base model
    cfg2 = _make_cfg()
    tags2 = build_wandb_tags(cfg2, dagspace_name="privacylens")
    assert "bench:privacylens" in tags2, f"  FAIL bench tag: {tags2}"
    assert "base" in tags2, f"  FAIL base tag: {tags2}"
    assert "finetuned" not in tags2, f"  FAIL should not have finetuned: {tags2}"

    print("  PASS")


# ── Test 8: WandbConfig auto-tags ────────────────────────────────────────

def test_wandb_config_auto_tags():
    print("Test 8: WandbConfig auto-tags via from_hydra_config")

    cfg = _make_cfg()
    cfg.model.lora_path = "/path/to/adapter"

    from dagspaces.common.wandb_logger import WandbConfig
    wc = WandbConfig.from_hydra_config(cfg, dagspace_name="goldcoin_hipaa")
    assert "bench:goldcoin_hipaa" in wc.tags, f"  FAIL bench tag: {wc.tags}"
    assert "family:qwen3" in wc.tags, f"  FAIL family tag: {wc.tags}"
    assert "finetuned" in wc.tags, f"  FAIL finetuned tag: {wc.tags}"
    assert wc.dagspace_name == "goldcoin_hipaa", f"  FAIL dagspace_name: {wc.dagspace_name}"

    print("  PASS")


# ── Test 9: Dagspace shims pass dagspace_name ────────────────────────────

def test_dagspace_shims():
    print("Test 9: Dagspace shims pass dagspace_name")

    cfg = _make_cfg()

    from dagspaces.goldcoin_hipaa.wandb_logger import WandbConfig as GC_WC
    gc = GC_WC.from_hydra_config(cfg)
    assert gc.dagspace_name == "goldcoin_hipaa", f"  FAIL goldcoin: {gc.dagspace_name}"
    assert "bench:goldcoin_hipaa" in gc.tags, f"  FAIL goldcoin tags: {gc.tags}"

    from dagspaces.privacylens.wandb_logger import WandbConfig as CI_WC
    ci = CI_WC.from_hydra_config(cfg)
    assert ci.dagspace_name == "privacylens", f"  FAIL privacylens: {ci.dagspace_name}"

    from dagspaces.vlm_geoprivacy_bench.wandb_logger import WandbConfig as VLM_WC
    vlm = VLM_WC.from_hydra_config(cfg)
    assert vlm.dagspace_name == "vlm_geoprivacy_bench", f"  FAIL vlm: {vlm.dagspace_name}"

    from dagspaces.historical_norms.wandb_logger import WandbConfig as HN_WC
    hn = HN_WC.from_hydra_config(cfg)
    assert hn.dagspace_name == "historical_norms", f"  FAIL hn: {hn.dagspace_name}"

    from dagspaces.uair.wandb_logger import WandbConfig as UAIR_WC
    uair = UAIR_WC.from_hydra_config(cfg)
    assert uair.dagspace_name == "uair", f"  FAIL uair: {uair.dagspace_name}"

    from dagspaces.grpo_training.wandb_logger import WandbConfig as GRPO_WC
    grpo = GRPO_WC.from_hydra_config(cfg)
    assert grpo.dagspace_name == "grpo_training", f"  FAIL grpo: {grpo.dagspace_name}"

    print("  PASS")


if __name__ == "__main__":
    tests = [
        test_derive_checkpoint_name,
        test_collect_metadata_full,
        test_collect_metadata_base,
        test_collect_metadata_no_model,
        test_collect_metadata_explicit_checkpoint,
        test_build_run_config,
        test_build_wandb_tags,
        test_wandb_config_auto_tags,
        test_dagspace_shims,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            failed += 1

    print(f"\n{'=' * 40}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
    print("All tests passed!")
