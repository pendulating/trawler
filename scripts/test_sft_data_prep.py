#!/usr/bin/env python3
"""Test SFT data prep stage end-to-end with fiction10 data.

Covers:
  - baseline (full schema, byte-identical to pre-toggle behavior)
  - minimal_tuple (all four flow_* off; instruction prose drops metadata clause)
  - no_norms_meta (norms_invoked + norm_source + is_new_flow drop together)

Usage:
    source /share/pierson/matt/UAIR/.venv/bin/activate
    python scripts/test_sft_data_prep.py
"""

import json
import os
import sys
import tempfile

import pandas as pd
from omegaconf import OmegaConf

# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = "/share/pierson/matt/n2s4cir/data/fiction10"
CI_REASONING_PATH = os.path.join(DATA_DIR, "ci_reasoning.parquet")
CI_FLOWS_PATH = os.path.join(DATA_DIR, "ci_flows.parquet")

BASE_FIELDS = [
    "sender", "recipient", "subject",
    "information_type", "transmission_principle",
]
ALL_META_FIELDS = [
    "context", "appropriateness",
    "norms_invoked", "norm_source", "is_new_flow",
    "confidence",
]


def _load_inputs():
    ci_reasoning = pd.read_parquet(CI_REASONING_PATH)
    ci_flows = pd.read_parquet(CI_FLOWS_PATH)
    return ci_reasoning, ci_flows


def _make_cfg(**flow_overrides):
    """Build a minimal cfg with the given training.sft.flow_* overrides."""
    return OmegaConf.create({"training": {"sft": dict(flow_overrides)}}) if flow_overrides else OmegaConf.create({})


def _iter_positive_pair(result_df):
    """Yield (i, user_text, completion) for the first positive pair (flows non-empty)."""
    for i, row in result_df.iterrows():
        msgs = json.loads(row["messages"])
        completion = json.loads(msgs[1]["content"])
        if completion.get("flows"):
            return i, msgs[0]["content"], completion
    raise AssertionError("No positive pair (with non-empty flows) found in result")


def test_data_prep_baseline(ci_reasoning, ci_flows):
    """Baseline: all toggles default-on. Asserts the full schema is present."""
    from dagspaces.grpo_training.stages.sft_data_prep import run_sft_data_prep_stage

    cfg = _make_cfg()
    result = run_sft_data_prep_stage(ci_reasoning.copy(), ci_flows.copy(), cfg)

    assert len(result) > 0, "No SFT pairs produced"
    assert set(result.columns) == {"messages", "source_id", "task_type"}
    assert (result["task_type"] == "ci_extraction").all()

    errors = []
    for i, row in result.head(50).iterrows():
        msgs = json.loads(row["messages"])
        if len(msgs) != 2:
            errors.append(f"Row {i}: expected 2 messages, got {len(msgs)}")
            continue
        if msgs[0]["role"] != "user" or msgs[1]["role"] != "assistant":
            errors.append(f"Row {i}: wrong roles: {[m['role'] for m in msgs]}")
            continue

        try:
            completion = json.loads(msgs[1]["content"])
        except json.JSONDecodeError as e:
            errors.append(f"Row {i}: completion not valid JSON: {e}")
            continue

        if "reasoning" not in completion:
            errors.append(f"Row {i}: missing 'reasoning' key")
        if not isinstance(completion.get("reasoning"), str):
            errors.append(f"Row {i}: 'reasoning' should be a string trace")
        if "has_information_exchange" not in completion:
            errors.append(f"Row {i}: missing 'has_information_exchange' key")
        if "flows" not in completion:
            errors.append(f"Row {i}: missing 'flows' key")
            continue

        for j, flow in enumerate(completion["flows"]):
            for field in BASE_FIELDS + ["context", "appropriateness", "confidence"]:
                if field not in flow:
                    errors.append(f"Row {i}, flow {j}: missing '{field}'")

    if errors:
        print(f"\n[baseline] FAILED with {len(errors)} errors:")
        for e in errors[:10]:
            print(f"  {e}")
        return False, result

    # Baseline instruction text must be byte-identical to the historical constant.
    from dagspaces.grpo_training.stages.sft_data_prep import _CI_INSTRUCTION
    _, user_text, _ = _iter_positive_pair(result)
    assert user_text.startswith(_CI_INSTRUCTION + "\n\n"), (
        "Baseline instruction prose has drifted from historical _CI_INSTRUCTION"
    )

    # Parquet round-trip + TRL Dataset compatibility (only run on baseline).
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        tmp_path = f.name
    try:
        result.to_parquet(tmp_path, index=False)
        reloaded = pd.read_parquet(tmp_path)
        assert len(reloaded) == len(result), "Parquet round-trip lost rows"
        sample_msgs = json.loads(reloaded.iloc[0]["messages"])
        assert len(sample_msgs) == 2
        print(f"[baseline] Parquet round-trip OK ({os.path.getsize(tmp_path) / 1024:.0f} KB)")
    finally:
        os.unlink(tmp_path)

    from datasets import Dataset
    dataset = Dataset.from_pandas(result)
    dataset = dataset.map(
        lambda row: {"messages": json.loads(row["messages"])
                     if isinstance(row["messages"], str) else row["messages"]}
    )
    sample = dataset[0]
    assert isinstance(sample["messages"], list)
    assert sample["messages"][0]["role"] == "user"
    print(f"[baseline] TRL Dataset compatibility OK ({len(dataset)} rows)")

    print("[baseline] PASSED")
    return True, result


def test_data_prep_minimal(ci_reasoning, ci_flows):
    """Minimal: all four flow_* toggles off. Only the base 5-tuple per flow,
    and the instruction prose drops the metadata clause entirely."""
    from dagspaces.grpo_training.stages.sft_data_prep import run_sft_data_prep_stage

    cfg = _make_cfg(
        flow_context=False,
        flow_appropriateness=False,
        flow_norms_meta=False,
        flow_confidence=False,
    )
    result = run_sft_data_prep_stage(ci_reasoning.copy(), ci_flows.copy(), cfg)
    assert len(result) > 0, "No SFT pairs produced under minimal_tuple"

    _, user_text, completion = _iter_positive_pair(result)

    # Per-flow schema: only the base 5-tuple.
    for j, flow in enumerate(completion["flows"]):
        for field in BASE_FIELDS:
            assert field in flow, f"[minimal] flow {j}: missing required base field '{field}'"
        for field in ALL_META_FIELDS:
            assert field not in flow, (
                f"[minimal] flow {j}: forbidden meta field '{field}' present under minimal mode"
            )

    # Scan only the instruction prose (everything before the article body, which
    # is appended after a "\n\n" separator). The article body can contain any of
    # the forbidden words verbatim (e.g. Orwell's "along with him").
    instruction_text = user_text.split("\n\n", 1)[0]
    forbidden_phrases = [
        "context",
        "appropriateness",
        "confidence",
        "norms",
        "metadata",
        "along with",
    ]
    for phrase in forbidden_phrases:
        assert phrase not in instruction_text, (
            f"[minimal] instruction prose still mentions '{phrase}': {instruction_text!r}"
        )
    # Sentence ends after `transmission_principle).`
    assert instruction_text.rstrip().endswith("transmission_principle)."), (
        f"[minimal] instruction does not end with 'transmission_principle).': "
        f"{instruction_text!r}"
    )

    print("[minimal] PASSED")
    return True


def test_data_prep_norms_bundle(ci_reasoning, ci_flows):
    """norms_meta off bundles three fields together; the other 3 toggles stay on."""
    from dagspaces.grpo_training.stages.sft_data_prep import run_sft_data_prep_stage

    cfg = _make_cfg(flow_norms_meta=False)
    result = run_sft_data_prep_stage(ci_reasoning.copy(), ci_flows.copy(), cfg)
    assert len(result) > 0, "No SFT pairs produced under no_norms_meta"

    _, user_text, completion = _iter_positive_pair(result)

    for j, flow in enumerate(completion["flows"]):
        # Base + the three still-on fields must be present.
        for field in BASE_FIELDS + ["context", "appropriateness", "confidence"]:
            assert field in flow, f"[norms_bundle] flow {j}: missing '{field}'"
        # The norms-meta trio must all be absent together.
        for field in ["norms_invoked", "norm_source", "is_new_flow"]:
            assert field not in flow, (
                f"[norms_bundle] flow {j}: forbidden field '{field}' "
                f"(should drop with flow_norms_meta=false)"
            )

    # Scan only the instruction prose (article body can contain any word).
    instruction_text = user_text.split("\n\n", 1)[0]
    assert "norms" not in instruction_text, (
        f"[norms_bundle] instruction still mentions 'norms': {instruction_text!r}"
    )
    assert "context" in instruction_text, "[norms_bundle] instruction lost 'context' fragment"
    assert "appropriateness" in instruction_text, "[norms_bundle] instruction lost 'appropriateness' fragment"
    assert "confidence" in instruction_text, "[norms_bundle] instruction lost 'confidence' fragment"

    print("[norms_bundle] PASSED")
    return True


def main():
    ci_reasoning, ci_flows = _load_inputs()
    print(f"Loaded: {len(ci_reasoning)} reasoning rows, {len(ci_flows)} flow rows")

    ok_baseline, baseline_result = test_data_prep_baseline(ci_reasoning, ci_flows)
    if not ok_baseline:
        return False

    ok_minimal = test_data_prep_minimal(ci_reasoning, ci_flows)
    ok_norms = test_data_prep_norms_bundle(ci_reasoning, ci_flows)

    if not (ok_minimal and ok_norms):
        return False

    # Summary stats from the baseline run (for parity with the prior script).
    unique_chunks = len(baseline_result)
    unique_sources = baseline_result["source_id"].nunique()
    flows_per_chunk = []
    for _, row in baseline_result.head(100).iterrows():
        msgs = json.loads(row["messages"])
        completion = json.loads(msgs[1]["content"])
        flows_per_chunk.append(len(completion["flows"]))

    print(f"\n=== Summary (baseline) ===")
    print(f"SFT pairs: {unique_chunks}")
    print(f"Unique sources: {unique_sources}")
    if flows_per_chunk:
        print(f"Flows/chunk (first 100): min={min(flows_per_chunk)}, "
              f"max={max(flows_per_chunk)}, avg={sum(flows_per_chunk)/len(flows_per_chunk):.1f}")
    print(f"Source distribution:")
    for src, count in baseline_result["source_id"].value_counts().items():
        print(f"  {src}: {count}")

    print("\nALL TESTS PASSED")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
