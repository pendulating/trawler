"""R5 trace self-sufficiency additions (2026-07-28).

The m1 lesson: what isn't in reward_traces.jsonl cannot be recovered — the
W&B crash ate core's discrimination tail, and diagnosing the parse failures
required regenerating completions on a GPU. Wave-2 traces must carry:

  * per-row battery forensics (``model_forces`` + ``vig_result``) so hedge
    drift under the re-anchored scale is recomputable from disk;
  * sampled completion text on R-VALID gate failures (~1-in-8, 600-char cap)
    so the residual failure mode is diagnosable without a GPU probe.

(The third piece, per-flow ``direct_flows`` on scored rows, shipped with the
chunk-denominator core — covered in test_direct_chunk.py.)
"""
from __future__ import annotations

import json

from dagspaces.grpo_training.stages.modular_reward import ModularReward


def _traced_reward(tmp_path, metadata):
    r = ModularReward(
        reward_core=True,
        core_mode="direct",
        direct_gold_fn=lambda flow, sid: ("appropriate", 0.9),
        trace_log_path=str(tmp_path / "reward_traces.jsonl"),
        trace_every_n_calls=1,
    )
    r.prompt_metadata = metadata
    return r


def _read_traces(tmp_path):
    return [json.loads(line)
            for line in open(tmp_path / "reward_traces.jsonl", encoding="utf-8")]


class TestVignetteTraceForensics:
    def test_model_forces_and_result_in_trace(self, tmp_path):
        meta = {"vp": {
            "task_type": "vignette",
            "battery_id": "b1",
            "source_id": "135",
            "gold_items": [
                {"gold_force": "obligatory", "articulation": "share the record"},
                {"gold_force": "prohibited", "articulation": "never disclose"},
            ],
        }}
        completion = json.dumps({"items": [
            {"id": 1, "force": "obligatory", "governing_norm": "share the record"},
            {"id": 2, "force": "permitted", "governing_norm": ""},
        ]})
        r = _traced_reward(tmp_path, meta)
        r(prompts=["vp"], completions=[completion])

        rows = _read_traces(tmp_path)
        assert len(rows) == 1
        row = rows[0]
        assert row["route"] == "vignette"
        assert row["model_forces"] == ["obligatory", "permitted"]
        vig = row["vig_result"]
        # exact (1.0) + hedge (0.15) on the re-anchored scale
        assert vig["battery"] == round((1.0 + 0.15) / 2, 4)
        assert vig["r_vig"] == vig["battery"]  # cite is diagnostic-only
        assert vig["hedge_frac"] == 0.5
        assert vig["parsed_frac"] == 1.0

    def test_unparsed_battery_traces_none_forces(self, tmp_path):
        meta = {"vp": {
            "task_type": "vignette",
            "battery_id": "b1",
            "source_id": "135",
            "gold_items": [{"gold_force": "obligatory", "articulation": "a"}],
        }}
        r = _traced_reward(tmp_path, meta)
        r(prompts=["vp"], completions=["utter prose, no json"])
        row = _read_traces(tmp_path)[0]
        assert row["model_forces"] == [None]
        assert row["vig_result"]["parsed_frac"] == 0.0


class TestGateFailTextSampling:
    def test_first_gate_failure_carries_text_sample(self, tmp_path):
        meta = {"ep": {"task_type": "extract", "gold_has_exchange": True,
                       "chunk_id": "c1", "source_id": "135", "probes": []}}
        bad = "this is not json at all " * 50  # > 600 chars, unparseable
        r = _traced_reward(tmp_path, meta)
        r(prompts=["ep"], completions=[bad])
        row = _read_traces(tmp_path)[0]
        assert row["route"] == "gate_fail"
        assert row["gate_reason"] == "parse"
        sample = row["completion_text_sample"]
        assert sample == bad[:600]
        assert len(sample) <= 600

    def test_sampling_is_one_in_eight(self, tmp_path):
        meta = {f"p{i}": {"task_type": "extract", "gold_has_exchange": True,
                          "chunk_id": f"c{i}", "source_id": "135",
                          "probes": []}
                for i in range(10)}
        r = _traced_reward(tmp_path, meta)
        r(prompts=[f"p{i}" for i in range(10)],
          completions=["not json"] * 10)
        rows = _read_traces(tmp_path)
        sampled = [i for i, row in enumerate(rows)
                   if "completion_text_sample" in row]
        assert sampled == [0, 8]  # failures 1 and 9 of the call
