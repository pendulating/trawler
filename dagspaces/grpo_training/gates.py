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
  e. direct_discrimination — pooled Youden's J over the per-flow gold/pred
                          records in the trace tail must reach
                          ``min_youden_j`` (m-series chunk-denominator core
                          only; total reward can rise for reasons that are
                          not discrimination — format, hedge mass — and the
                          m1 wave promoted four cells at the blanket floor).

Use via ``scripts/check_grpo_promotion_gates.py <checkpoint_dir>`` (exits
non-zero on failure, for sweep scripting) or import ``check_promotion_gates``.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

DEFAULT_THRESHOLDS: dict[str, float] = {
    # 0.0 promoted the entire flat m1 grid (core's gain: +0.0027). 0.02 sits
    # above the m1 per-bin wobble (±0.02-0.03 on the per-call mean) — a run
    # must beat launch noise, not just tie its starting point.
    # SEMANTICS (final review M-2): this gates the COMPOSITE reward, whose
    # module weights differ by cell (core outcome=1.0; full outcome=0.5).
    # Gate (a) is therefore an INTRA-cell "did it learn at all" check —
    # never compare gate-(a) margins across cells, and never read a
    # multi-module cell's HOLD as "its core learned less than core's":
    # cross-cell verdicts use the discrimination metrics (gate e /
    # reward/direct/*), which are weight-independent. The v9 keeper's
    # recorded verdict (+0.015 PROMOTE) predates this threshold.
    "min_reward_gain": 0.02,     # last-third mean reward − first-third mean
    "max_frac_zero_std": 0.2,    # mean fraction of zero-advantage groups
    "max_kl": 1.0,               # mean KL to the SFT reference
    "no_flow_tolerance": 0.15,   # |trace no-flow rate − gold base rate|
    "trace_tail_calls": 20,      # reward-trace calls used for gate (d)
    "min_youden_j": 0.05,        # LABEL-only pooled J floor over the trace
                                 # tail (gate e); matched flows only, so 0 =
                                 # blanket floor and ±0.05 = m1 noise band
                                 # (recall is reported as miss_frac, not
                                 # gated — audit 2026-07-28)
    "j_trace_tail_calls": 100,   # reward-trace calls pooled for gate (e)
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
    with open(traces_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Keeper traces: task_type == "ci_extraction" with `is_no_flow`.
            # Modular traces: task_type == "extract" with `no_flow` set only
            # on abstain-routed rows. The gate was DEAD on every modular run
            # until 2026-07-28 (m1's gates all show it skipped) because it
            # only knew the keeper schema.
            if row.get("task_type") in ("ci_extraction", "extract"):
                rows.append(row)
    if not rows:
        return {"status": "skipped", "reason": "no extract-task trace rows"}

    max_call = max(r.get("call", 0) for r in rows)
    tail = [r for r in rows if r.get("call", 0) > max_call - tail_calls]
    rate = sum(
        1 for r in tail if r.get("is_no_flow", r.get("no_flow", False))
    ) / len(tail)
    deviation = abs(rate - gold_base_rate)
    return {
        "status": "pass" if deviation <= tolerance else "fail",
        "trace_no_flow_rate": round(rate, 4),
        "gold_base_rate": round(gold_base_rate, 4),
        "deviation": round(deviation, 4),
        "tolerance": tolerance,
        "n_tail_completions": len(tail),
    }


def _gate_direct_discrimination(
    traces_path: str,
    min_j: float,
    tail_calls: int,
) -> dict[str, Any]:
    """Gate (e): pooled Youden's J from the traces' per-flow gold/pred records.

    Reads the ``direct_flows`` lists the chunk-denominator R-DIRECT scorer
    logs per scored completion (modular_reward, 2026-07-28) — deliberately
    disk-only so a W&B crash cannot lose the verdict (m1's core lost its
    last 150 steps of discrimination metrics exactly that way). A hit is
    ``pred == gold``; a miss, hedge, or unmatched teacher flow all score 0
    for their class — J here prices recall as well as labelling, matching
    the reward. Skipped (never failed) when the traces carry no
    ``direct_flows`` (−outcome cells, per-flow fallback path, pre-R2 runs).
    """
    if not os.path.exists(traces_path):
        return {"status": "skipped", "reason": f"no traces at {traces_path}"}

    rows: list[dict[str, Any]] = []
    global_max_call = -1
    with open(traces_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            global_max_call = max(global_max_call, int(row.get("call", 0) or 0))
            if row.get("direct_flows"):
                rows.append(row)
    if not rows:
        return {"status": "skipped",
                "reason": "no direct_flows trace rows (not a chunk-"
                          "denominator direct-core run?)"}

    max_call = max(r.get("call", 0) for r in rows)
    # Staleness guard (audit 2026-07-28, gates C1): if the direct core
    # silently stopped scoring (embedding outage -> group-neutral), the last
    # direct_flows rows predate the end of training and a verdict on them
    # would describe a policy from N calls earlier — while LOOKING healthy.
    if global_max_call - max_call > 2:
        return {
            "status": "fail",
            "reason": (f"direct core stopped scoring at call {max_call} of "
                       f"{global_max_call} — the tail is stale (embedding "
                       "outage?); no discrimination verdict is possible"),
            "last_direct_call": max_call,
            "last_trace_call": global_max_call,
            "threshold": min_j,
        }
    tail = [r for r in rows if r.get("call", 0) > max_call - tail_calls]
    # LABEL-only J (audit 2026-07-28, R2-M5): unmatched teacher flows
    # (pred None) are EXCLUDED here — including them makes J carry the match
    # rate, whose blanket floor is m-1 (launch J ~= -0.23), so the 0.05
    # threshold would fail every cell for a recall reason. Recall is reported
    # alongside as miss_frac; the REWARD still prices it.
    hits_by_class: dict[str, list[int]] = {}
    n_missed = 0
    n_teacher = 0
    for r in tail:
        for fl in r["direct_flows"]:
            gold = fl.get("gold")
            if gold is None:
                continue
            n_teacher += 1
            if fl.get("pred") is None:
                n_missed += 1
                continue
            hits_by_class.setdefault(str(gold), []).append(
                1 if fl.get("pred") == gold else 0
            )
    if len(hits_by_class) < 2:
        return {"status": "skipped",
                "reason": f"tail carries {len(hits_by_class)} matched gold "
                          "class(es); J needs both"}

    recalls = {c: sum(v) / len(v) for c, v in hits_by_class.items()}
    j = sum(recalls.values()) - 1.0
    return {
        "status": "pass" if j >= min_j else "fail",
        "youden_j": round(j, 4),
        "recalls": {c: round(r, 4) for c, r in recalls.items()},
        "miss_frac": round(n_missed / n_teacher, 4) if n_teacher else None,
        "n_flow_judgments": sum(len(v) for v in hits_by_class.values()),
        "n_tail_completions": len(tail),
        "threshold": min_j,
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
        except Exception as e:
            # Do NOT swallow silently. gold_base_rate stays None, and
            # _gate_no_flow then returns status="skipped". A silently skipped
            # promotion gate is a known failure mode in this file — see the
            # comment in _gate_no_flow about the gate being DEAD on every
            # modular run until 2026-07-28. The verdict must say WHY.
            print(
                f"[gates] could not read n_flow_chunks / n_no_flow_chunks from "
                f"{meta_path}, so the no-flow gate will SKIP: "
                f"{type(e).__name__}: {e}",
                file=sys.stderr,
                flush=True,
            )

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
        "direct_discrimination": _gate_direct_discrimination(
            os.path.join(checkpoint_dir, "reward_traces.jsonl"),
            th["min_youden_j"],
            int(th["j_trace_tail_calls"]),
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
