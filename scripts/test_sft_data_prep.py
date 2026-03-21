#!/usr/bin/env python3
"""Test SFT data prep stage end-to-end with fiction10 data.

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


def test_data_prep():
    from dagspaces.grpo_training.stages.sft_data_prep import run_sft_data_prep_stage

    cfg = OmegaConf.create({})

    ci_reasoning = pd.read_parquet(CI_REASONING_PATH)
    ci_flows = pd.read_parquet(CI_FLOWS_PATH)
    print(f"Loaded: {len(ci_reasoning)} reasoning rows, {len(ci_flows)} flow rows")

    result = run_sft_data_prep_stage(ci_reasoning, ci_flows, cfg)

    # Basic shape checks
    assert len(result) > 0, "No SFT pairs produced"
    assert set(result.columns) == {"messages", "source_id", "task_type"}
    assert (result["task_type"] == "ci_extraction").all()

    # Validate message format
    errors = []
    for i, row in result.head(50).iterrows():
        msgs = json.loads(row["messages"])
        if len(msgs) != 2:
            errors.append(f"Row {i}: expected 2 messages, got {len(msgs)}")
            continue
        if msgs[0]["role"] != "user" or msgs[1]["role"] != "assistant":
            errors.append(f"Row {i}: wrong roles: {[m['role'] for m in msgs]}")
            continue

        # Validate completion is parseable JSON with correct structure
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
            for field in ["sender", "recipient", "subject",
                          "information_type", "transmission_principle",
                          "context", "appropriateness", "confidence"]:
                if field not in flow:
                    errors.append(f"Row {i}, flow {j}: missing '{field}'")

    if errors:
        print(f"\nFAILED with {len(errors)} errors:")
        for e in errors[:10]:
            print(f"  {e}")
        return False

    # Save to temp file and verify round-trip
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        tmp_path = f.name
    try:
        result.to_parquet(tmp_path, index=False)
        reloaded = pd.read_parquet(tmp_path)
        assert len(reloaded) == len(result), "Parquet round-trip lost rows"
        # Verify the messages column survives serialization
        sample_msgs = json.loads(reloaded.iloc[0]["messages"])
        assert len(sample_msgs) == 2
        print(f"Parquet round-trip OK ({os.path.getsize(tmp_path) / 1024:.0f} KB)")
    finally:
        os.unlink(tmp_path)

    # Verify TRL dataset compatibility
    from datasets import Dataset

    dataset = Dataset.from_pandas(result)
    dataset = dataset.map(
        lambda row: {"messages": json.loads(row["messages"])
                     if isinstance(row["messages"], str) else row["messages"]}
    )
    sample = dataset[0]
    assert isinstance(sample["messages"], list)
    assert sample["messages"][0]["role"] == "user"
    print(f"TRL Dataset compatibility OK ({len(dataset)} rows)")

    # Summary stats
    unique_chunks = len(result)
    unique_sources = result["source_id"].nunique()
    flows_per_chunk = []
    for _, row in result.head(100).iterrows():
        msgs = json.loads(row["messages"])
        completion = json.loads(msgs[1]["content"])
        flows_per_chunk.append(len(completion["flows"]))

    print(f"\n=== Summary ===")
    print(f"SFT pairs: {unique_chunks}")
    print(f"Unique sources: {unique_sources}")
    print(f"Flows/chunk (first 100): min={min(flows_per_chunk)}, "
          f"max={max(flows_per_chunk)}, avg={sum(flows_per_chunk)/len(flows_per_chunk):.1f}")
    print(f"Source distribution:")
    for src, count in result["source_id"].value_counts().items():
        print(f"  {src}: {count}")

    print("\nALL TESTS PASSED")
    return True


if __name__ == "__main__":
    success = test_data_prep()
    sys.exit(0 if success else 1)
