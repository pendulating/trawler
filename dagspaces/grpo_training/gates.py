"""Promotion gates for GRPO checkpoints.

A trained cell graduates to full benchmark evaluation only if its training
dynamics show it actually learned. The May 2026 sweep spent 15 cells of
eval compute on checkpoints whose reward curves were flat — every one of
them would have failed gate (a) below.

Gates (all computed from artifacts the training stage already writes):
  a. reward_trend       — mean reward over the last third of logged steps
                          must exceed the first third by ``min_reward_gain``.
  b. zero_std_groups    — mean ``frac_reward_zero_std`` must stay below
                          ``max_frac_zero_std`` (tied groups carry no
                          advantage).
  c. kl_bounded         — mean KL to the SFT reference must stay below
                          ``max_kl`` (skipped when beta=0 logged no KL).
  d. no_flow_rate       — fraction of no-flow completions in the most
                          recent reward traces must stay within
                          ``no_flow_tolerance`` of the gold base rate
                          (guards against collapse onto the lazy path).

Use via ``scripts/check_grpo_promotion_gates.py <checkpoint_dir>`` (exits
non-zero on failure, for sweep scripting) or import ``check_promotion_gates``.
"""

from __future__ import annotations

import json
import os
from typing import Any

DEFAULT_THRESHOLDS: dict[str, float] = {
    "min_reward_gain": 0.0,      # last-third mean reward − first-third mean
    "max_frac_zero_std": 0.2,    # mean fraction of zero-advantage groups
    "max_kl": 1.0,               # mean KL to the SFT reference
    "no_flow_tolerance": 0.15,   # |trace no-flow rate − gold base rate|
    "trace_tail_calls": 20,      # reward-trace calls used for gate (d)
}


def _find_trainer_state(checkpoint_dir: str) -> str | None:
    """Locate the highest-step checkpoint-N/trainer_state.json."""
    direct = os.path.join(checkpoint_dir, "trainer_state.json")
    if os.path.exists(direct):
        return direct
    best_step, best_path = -1, None
    try:
        entries = os.listdir(checkpoint_dir)
    except OSError:
        return None
    for name in entries:
        if name.startswith("checkpoint-"):
            try:
                step = int(name.split("-", 1)[1])
            except ValueError:
                continue
            candidate = os.path.join(checkpoint_dir, name, "trainer_state.json")
            if step > best_step and os.path.exists(candidate):
                best_step, best_path = step, candidate
    return best_path


def _reward_entries(trainer_state: dict[str, Any]) -> list[dict[str, Any]]:
    return [e for e in trainer_state.get("log_history", []) if "reward" in e]


def _eval_reward_entries(trainer_state: dict[str, Any]) -> list[dict[str, Any]]:
    return [e for e in trainer_state.get("log_history", []) if "eval_reward" in e]


def _gate_reward_trend(
    entries: list[dict[str, Any]],
    min_gain: float,
    key: str = "reward",
) -> dict[str, Any]:
    """Last-third vs first-third trend over ``key``.

    Prefers the held-out ``eval_reward`` curve when the caller passes it
    (2026-06-09 review, S3): the training-batch reward can trend up via
    reward hacking; the dev-split curve is the generalization signal.
    """
    if len(entries) < 3:
        return {"status": "skipped", "reason": f"only {len(entries)} {key} logs"}
    third = max(1, len(entries) // 3)
    first = sum(e[key] for e in entries[:third]) / third
    last = sum(e[key] for e in entries[-third:]) / third
    gain = last - first
    return {
        "status": "pass" if gain > min_gain else "fail",
        "source": key,
        "first_third_mean": round(first, 4),
        "last_third_mean": round(last, 4),
        "gain": round(gain, 4),
        "threshold": min_gain,
    }


def _gate_zero_std(entries: list[dict[str, Any]], max_frac: float) -> dict[str, Any]:
    vals = [e["frac_reward_zero_std"] for e in entries if "frac_reward_zero_std" in e]
    if not vals:
        return {"status": "skipped", "reason": "frac_reward_zero_std not logged"}
    mean = sum(vals) / len(vals)
    return {
        "status": "pass" if mean < max_frac else "fail",
        "mean_frac_zero_std": round(mean, 4),
        "threshold": max_frac,
    }


def _gate_kl(entries: list[dict[str, Any]], max_kl: float) -> dict[str, Any]:
    vals = [e["kl"] for e in entries if "kl" in e]
    if not vals:
        return {"status": "skipped", "reason": "kl not logged (beta=0?)"}
    mean = sum(vals) / len(vals)
    return {
        "status": "pass" if mean < max_kl else "fail",
        "mean_kl": round(mean, 4),
        "max_logged_kl": round(max(vals), 4),
        "threshold": max_kl,
    }


def _gate_no_flow(
    traces_path: str,
    gold_base_rate: float | None,
    tolerance: float,
    tail_calls: int,
) -> dict[str, Any]:
    if gold_base_rate is None:
        return {"status": "skipped", "reason": "gold base rate unavailable"}
    if not os.path.exists(traces_path):
        return {"status": "skipped", "reason": f"no traces at {traces_path}"}

    rows: list[dict[str, Any]] = []
    with open(traces_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("task_type") == "ci_extraction":
                rows.append(row)
    if not rows:
        return {"status": "skipped", "reason": "no ci_extraction trace rows"}

    max_call = max(r.get("call", 0) for r in rows)
    tail = [r for r in rows if r.get("call", 0) > max_call - tail_calls]
    rate = sum(1 for r in tail if r.get("is_no_flow")) / len(tail)
    deviation = abs(rate - gold_base_rate)
    return {
        "status": "pass" if deviation <= tolerance else "fail",
        "trace_no_flow_rate": round(rate, 4),
        "gold_base_rate": round(gold_base_rate, 4),
        "deviation": round(deviation, 4),
        "tolerance": tolerance,
        "n_tail_completions": len(tail),
    }


def check_promotion_gates(
    checkpoint_dir: str,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Evaluate all promotion gates for a GRPO checkpoint directory.

    Expects the layout the training stage writes: ``checkpoint-N/
    trainer_state.json``, ``reward_traces.jsonl``, ``training_metadata.json``.

    Returns a report dict with per-gate results and an overall ``promote``
    bool (true only if no gate failed; skipped gates don't block).
    """
    th = {**DEFAULT_THRESHOLDS, **(thresholds or {})}

    state_path = _find_trainer_state(checkpoint_dir)
    if state_path is None:
        return {
            "promote": False,
            "error": f"no trainer_state.json under {checkpoint_dir}",
        }
    with open(state_path, "r", encoding="utf-8") as f:
        trainer_state = json.load(f)
    entries = _reward_entries(trainer_state)

    gold_base_rate: float | None = None
    meta_path = os.path.join(checkpoint_dir, "training_metadata.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            n_flow = meta.get("n_flow_chunks")
            n_no_flow = meta.get("n_no_flow_chunks")
            if n_flow is not None and n_no_flow is not None and (n_flow + n_no_flow) > 0:
                gold_base_rate = n_no_flow / (n_flow + n_no_flow)
        except Exception:
            pass

    # Trend gate: use the held-out dev-split curve when it has enough
    # points; fall back to the training-batch reward otherwise (runs with
    # dev_fraction=0 or too-sparse eval_steps).
    eval_entries = _eval_reward_entries(trainer_state)
    if len(eval_entries) >= 3:
        trend = _gate_reward_trend(eval_entries, th["min_reward_gain"],
                                   key="eval_reward")
    else:
        trend = _gate_reward_trend(entries, th["min_reward_gain"])

    gates = {
        "reward_trend": trend,
        "zero_std_groups": _gate_zero_std(entries, th["max_frac_zero_std"]),
        "kl_bounded": _gate_kl(entries, th["max_kl"]),
        "no_flow_rate": _gate_no_flow(
            os.path.join(checkpoint_dir, "reward_traces.jsonl"),
            gold_base_rate,
            th["no_flow_tolerance"],
            int(th["trace_tail_calls"]),
        ),
    }

    return {
        "checkpoint_dir": checkpoint_dir,
        "trainer_state": state_path,
        "n_reward_logs": len(entries),
        "thresholds": th,
        "gates": gates,
        "promote": all(g["status"] != "fail" for g in gates.values()),
    }
