"""Guards for the distilled-grounding flow re-extraction.

Plan: wiki/2026-08-05_distilled_grounding_plan.md

The experiment re-runs the fiction flow pipeline with the fine-tuned policies
substituted for the Gemma-4-31B teacher, then recompares each arm's own
appropriateness label against the norm-grounded one. The measurement only means
anything if the ONLY thing that changed is the weights, so the tests here pin
the two invariants that make that true:

  1. `COLM_flows_fiction_policy` prompts and generation settings are identical
     to `COLM_flows_fiction_prefetched_gemma4`'s. If someone "tidies" a prompt
     selection or drops the 24576 context override, the arms stop being
     comparable to the published 30.9% and nothing downstream would notice.
  2. LoRA survives the pipeline's engine overrides. A silently-empty
     `lora_path` turns three tuned arms into three copies of the base model —
     a failure that produces perfectly plausible numbers.

The contamination-bookkeeping test (the 503-chunk doubly-held-out set) is
artifact-dependent and skips when the training records are not on disk.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from hydra import compose, initialize_config_dir

import dagspaces.historical_norms

CONF_DIR = Path(dagspaces.historical_norms.__file__).parent / "conf"
PIPELINE_DIR = CONF_DIR / "pipeline"

POLICY = "COLM_flows_fiction_policy"
TEACHER = "COLM_flows_fiction_prefetched_gemma4"

ROOT = Path("/share/pierson/matt/UAIR")
M2_TRACES = ROOT / (
    "multirun/2026-07-28_grpo_m2_full/21-31-11/cell=full/"
    "grpo_only_online_external/outputs/grpo/checkpoint/reward_traces.jsonl"
)
K1_META = ROOT / "outputs/2026-07-31_k1_full/kto_metadata.json"

#: Measured 2026-08-05 from the two training records above. These are the
#: numbers the plan's contamination table is built on; if the artifacts change,
#: the plan's holdout claims change with them.
EXPECTED = {
    "kto_heldout": 599,
    "grpo_extract_chunks": 492,
    "grpo_books": {"1023", "11", "1342", "135"},
    "double_heldout": 503,
}


def _body(pipeline: str) -> dict:
    return yaml.safe_load((PIPELINE_DIR / f"{pipeline}.yaml").read_text()) or {}


def _nodes(pipeline: str) -> dict:
    return _body(pipeline)["pipeline"]["graph"]["nodes"]


def test_policy_pipeline_prompts_match_the_teacher_run():
    """Same prompt selections as the gold-label run — the whole comparison
    rests on prompt identity, so this is not a style preference."""
    pol, teach = _body(POLICY), _body(TEACHER)
    for key in ("prompt_ci_reasoning", "prompt_ci_extraction"):
        assert pol[key] == teach[key], (
            f"{POLICY} selects {key}={pol[key]!r} but the teacher run used "
            f"{teach[key]!r}. The arms are no longer comparable to the "
            f"published 30.9% reclassification rate."
        )


def test_policy_pipeline_keeps_the_teacher_generation_budget():
    """max_tokens per stage must match the teacher's.

    A shorter budget truncates mid-JSON, which parses as
    has_information_exchange=False and biases against long chunks — exactly
    the failure the teacher config's 24576 comment documents.
    """
    pol, teach = _nodes(POLICY), _nodes(TEACHER)
    for stage in ("ci_reasoning", "ci_extraction"):
        p = pol[stage]["overrides"]["sampling_params.max_tokens"]
        t = teach[stage]["overrides"]["sampling_params.max_tokens"]
        assert p == t, f"{stage}: policy max_tokens {p} != teacher {t}"


def test_policy_pipeline_keeps_the_24576_context_override():
    """706 of 2,993 fiction10 prompts exceed 16384 with the system prompt +
    book summary. The model yamls say 16384; the pipeline must override."""
    for stage, spec in _nodes(POLICY).items():
        got = spec["overrides"]["model.engine_kwargs.max_model_len"]
        assert got == 24576, f"{stage}: max_model_len {got}, expected 24576"


def test_policy_pipeline_preserves_lora():
    """enable_lora must survive the engine overrides on every stage.

    An adapter that fails to load makes the tuned arms silently identical to
    the base model, and the resulting table would look entirely reasonable.
    """
    for stage, spec in _nodes(POLICY).items():
        ov = spec["overrides"]
        assert ov.get("model.engine_kwargs.enable_lora") is True, (
            f"{stage}: enable_lora not set in overrides"
        )
        assert ov.get("model.engine_kwargs.max_lora_rank") == 64, (
            f"{stage}: max_lora_rank must be 64 (all camera-ready adapters "
            f"are r=64)"
        )


@pytest.mark.parametrize(
    "arm",
    ["qwen3.5-9b/sft-canonical", "qwen3.5-9b/m2-full-ckpt450",
     "qwen3.5-9b/k3-verdict"],
)
def test_tuned_arms_resolve_a_lora_path(arm):
    """Each tuned arm composes with a non-empty lora_path under this pipeline."""
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[f"pipeline={POLICY}", f"model={arm}"],
        )
    lora = str(cfg.model.get("lora_path") or "")
    assert lora, f"{arm} composed with an empty lora_path under {POLICY}"


def test_base_arm_has_no_lora_path():
    """The control arm must be adapter-free, or it is not a control."""
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[f"pipeline={POLICY}", "model=qwen3.5-9b/instruct"],
        )
    assert not str(cfg.model.get("lora_path") or "")


LAUNCHER = ROOT / "scripts/run_policy_flow_extraction.sh"


@pytest.mark.skipif(not LAUNCHER.exists(), reason="launcher not on disk")
def test_launcher_selects_corpus_by_hydra_override_not_env():
    """The corpus must NOT be chosen by exporting FICTION_CHUNKS_PATH.

    Shell exports do not cross the submitit boundary. On the compute node the
    var is unset, ``ensure_dotenv()`` loads the project-root ``.env`` with
    ``override=False``, and ``.env`` points FICTION_CHUNKS_PATH at
    ``chunks_top100_fiction_en.parquet``. On 2026-08-06 that silently ran this
    sweep on top100 (15,875 chunks / 100 books) instead of fiction10 (2,993 /
    10) for ~8 GPU-hours, while the driver's own echo showed the fiction10 path.

    A Hydra override IS serialized into the job, so it crosses.
    """
    src = LAUNCHER.read_text()
    assert "export FICTION_CHUNKS_PATH" not in src, (
        "run_policy_flow_extraction.sh exports FICTION_CHUNKS_PATH again. That "
        "export does not reach the stage job; .env wins there and selects "
        "top100."
    )
    assert "pipeline.sources.prefetched_chunks.path=" in src, (
        "the chunk path must be passed as a Hydra override so it crosses "
        "submitit"
    )


@pytest.mark.skipif(not LAUNCHER.exists(), reason="launcher not on disk")
def test_launcher_preflights_the_corpus():
    """A corpus assertion must run BEFORE submission.

    The smoke test cannot catch a corpus mismatch — ``runtime.sample_n=8``
    truncates any corpus to 8 chunks, which is exactly why the 2026-08-06 smoke
    passed on the wrong data. This preflight is the only guard that would have
    caught it.
    """
    src = LAUNCHER.read_text()
    assert "preflight" in src.lower(), "no corpus preflight in the launcher"
    assert "CORPUS MISMATCH" in src, (
        "the preflight must fail loudly and by name on a mismatch"
    )
    # The preflight has to run before the CLI is invoked, or it guards nothing.
    assert src.index("preflight") < src.index(
        "dagspaces.historical_norms.cli"
    ), "preflight runs after submission"


@pytest.mark.skipif(
    not (M2_TRACES.exists() and K1_META.exists()),
    reason="training records not on disk",
)
def test_contamination_bookkeeping_is_reproducible():
    """The plan's holdout table must be derivable, not remembered.

    chunk_id is per-book, so every key here is (source_id, chunk_id) — keying
    on chunk_id alone silently merges chunks across novels and was the first
    thing that went wrong when this was measured by hand.
    """
    grpo = set()
    with M2_TRACES.open() as fh:
        for line in fh:
            rec = json.loads(line)
            if rec.get("task_type") == "extract":
                grpo.add(f"{rec.get('source_id')}|{rec.get('chunk_id')}")

    held = set(json.loads(K1_META.read_text())["heldout_keys"])

    assert len(held) == EXPECTED["kto_heldout"]
    assert len(grpo) == EXPECTED["grpo_extract_chunks"]
    assert {k.split("|", 1)[0] for k in grpo} == EXPECTED["grpo_books"], (
        "GRPO's extract prompts span a different set of novels than the plan "
        "records; the book-level holdout claim needs re-deriving."
    )
    assert len(held - grpo) == EXPECTED["double_heldout"]
