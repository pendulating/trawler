"""Unit tests for the GRPO field-note metric extractors.

Guards the derivations cited in ``wiki/grpo_training_field_notes/`` against
drift: optimizer-signal summaries from ``trainer_state.json`` and gold-label
behaviour from ``reward_traces.jsonl``. Synthetic fixtures only — no run
artifacts on disk — so the invariants are explicit and fast.
"""

from __future__ import annotations

import json

import pytest

from dagspaces.grpo_training.trace_metrics import (
    find_latest_trainer_state,
    gold_label_metrics,
    is_abstention,
    load_jsonl,
    pearson,
    per_step_table,
    reward_trend,
    summarize_log_history,
    summarize_signal,
    within_group_advantage,
)


# ── Optimizer signals ───────────────────────────────────────────────────────


def _log_history():
    """A log_history where entropy runs away and drags logp_diff up / IS down.

    Mirrors the v6 instability fingerprint (entropy~logp_diff > 0,
    entropy~IS < 0) so the correlation signs are meaningful.
    """
    rows = []
    for i, ent in enumerate([0.6, 1.5, 3.0, 6.0]):
        rows.append({
            "step": (i + 1) * 10,
            "entropy": ent,
            "kl": 0.01 * (i + 1),
            "reward": 0.20 + 0.02 * i,  # mild upward trend
            "frac_reward_zero_std": 0.0,
            "completions/mean_length": 150.0,
            "sampling/sampling_logp_difference/mean": 0.1 * (i + 1),  # grows with entropy
            "sampling/importance_sampling_ratio/mean": 1.0 - 0.2 * i,  # falls with entropy
        })
    return rows


def test_summarize_signal_first_last_min_max():
    lh = _log_history()
    s = summarize_signal(lh, "entropy")
    assert s == {"n": 4, "first": 0.6, "last": 6.0, "min": 0.6, "max": 6.0}


def test_summarize_signal_absent_key_is_none():
    assert summarize_signal(_log_history(), "does/not/exist") is None


def test_reward_trend_matches_gate_definition():
    # last-third minus first-third over the reward series; floor(n/3) window.
    vals = [0.20, 0.22, 0.24, 0.26]  # n=4 -> third=1 -> first=0.20 last=0.26
    t = reward_trend(vals)
    assert t["first_third_mean"] == pytest.approx(0.20)
    assert t["last_third_mean"] == pytest.approx(0.26)
    assert t["gain"] == pytest.approx(0.06)


def test_reward_trend_empty():
    assert reward_trend([])["gain"] == 0.0


def test_correlation_signs_capture_instability_fingerprint():
    out = summarize_log_history(_log_history())
    corr = out["correlations"]
    # entropy up => logp_diff up (positive), IS down (negative)
    assert corr["entropy_vs_logp_diff"] > 0.9
    assert corr["entropy_vs_is_ratio"] < -0.9


def test_pearson_degenerate_returns_none():
    assert pearson([1.0], [2.0]) is None          # too short
    assert pearson([1.0, 1.0], [3.0, 4.0]) is None  # zero variance


def test_per_step_table_selects_keys():
    rows = per_step_table(_log_history(), keys=["reward", "entropy"])
    assert rows[0] == {"step": 10, "reward": 0.20, "entropy": 0.6}
    assert rows[-1]["entropy"] == 6.0


# ── Gold-label behaviour ────────────────────────────────────────────────────


def test_is_abstention_prefers_flag_then_falls_back():
    assert is_abstention({"is_no_flow": True}) is True
    assert is_abstention({"is_no_flow": False}) is False
    # no flag -> structural fallback on completion JSON
    assert is_abstention({"completion": '{"flows": []}'}) is True
    assert is_abstention({"completion": '{"has_information_exchange": false}'}) is True
    assert is_abstention({"completion": '{"flows": [{"x": 1}]}'}) is False


def _ci_traces():
    """Two GRPO groups (call 0/1), gold=YES, each mixed (1 abstain + 1 extract)."""
    def rec(call, no_flow, gold_yes, rground, composite):
        return {
            "task_type": "ci_extraction",
            "call": call,
            "is_no_flow": no_flow,
            "gold_has_exchange": gold_yes,
            "components": {"r_ground": rground},
            "composite": composite,
        }
    return [
        rec(0, True, True, 0.0, -0.10),   # abstain on gold-YES (wrong)
        rec(0, False, True, 0.80, 0.62),  # extract on gold-YES
        rec(1, True, True, 0.0, -0.08),   # abstain on gold-YES (wrong)
        rec(1, False, True, 0.00, 0.40),  # extract, but R_ground==0
    ]


def test_gold_label_metrics_conditional_abstention():
    m = gold_label_metrics(_ci_traces())
    assert m["n_ci"] == 4
    assert m["gold_yes_frac"] == pytest.approx(1.0)
    assert m["abstain_frac"] == pytest.approx(0.5)
    # 2 of 4 gold-YES rows abstained
    assert m["abstain_given_gold_yes"] == pytest.approx(0.5)
    assert m["abstain_given_gold_no"] is None  # no gold-NO rows present


def test_gold_label_rground_and_composite():
    m = gold_label_metrics(_ci_traces())
    # extractors: r_ground 0.80 and 0.00 -> mean 0.40, half are zero
    assert m["rground_mean_on_extractors"] == pytest.approx(0.40)
    assert m["rground_zero_frac_on_extractors"] == pytest.approx(0.5)
    assert m["composite_mean_extract"] == pytest.approx(0.51)   # (0.62+0.40)/2
    assert m["composite_mean_abstain"] == pytest.approx(-0.09)  # (-0.10-0.08)/2


def test_within_group_advantage_gold_yes():
    a = within_group_advantage(_ci_traces(), gold_yes_only=True)
    assert a["n_mixed_groups"] == 2
    # group0: 0.62-(-0.10)=0.72 ; group1: 0.40-(-0.08)=0.48 ; mean 0.60
    assert a["mean_advantage"] == pytest.approx(0.60)
    assert a["frac_groups_extract_wins"] == pytest.approx(1.0)


def test_within_group_skips_homogeneous_groups():
    # A group with only extractors contributes no advantage (no contrast).
    traces = [
        {"task_type": "ci_extraction", "call": 5, "is_no_flow": False,
         "gold_has_exchange": True, "composite": 0.5},
        {"task_type": "ci_extraction", "call": 5, "is_no_flow": False,
         "gold_has_exchange": True, "composite": 0.6},
    ]
    a = within_group_advantage(traces, gold_yes_only=True)
    assert a["n_mixed_groups"] == 0
    assert a["mean_advantage"] is None


# ── IO helpers ──────────────────────────────────────────────────────────────


def test_load_jsonl_skips_bad_lines(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text('{"a": 1}\n\nnot json\n{"a": 2}\n')
    assert load_jsonl(p) == [{"a": 1}, {"a": 2}]


def test_find_latest_trainer_state_picks_highest_step(tmp_path):
    for step in (50, 150, 100):
        d = tmp_path / "outputs" / "grpo" / "checkpoint" / f"checkpoint-{step}"
        d.mkdir(parents=True)
        (d / "trainer_state.json").write_text(json.dumps({"log_history": []}))
    found = find_latest_trainer_state(tmp_path)
    assert found is not None and found.parent.name == "checkpoint-150"
