"""Reproducible metrics for the GRPO training field notes.

The dated notes under ``wiki/grpo_training_field_notes/`` are built from two
artifact families a GRPO run emits:

* ``trainer_state.json`` (``log_history``) -> **optimizer signals**: policy
  entropy, KL, importance-sampling ratios, the vLLM-rollout vs HF-trainer
  logprob mismatch, reward trend. These are what diagnosed the v6 instability
  (entropy runaway -> logprob divergence -> IS collapse -> masked gradient) and
  showed the v7 beta=0.02 anchor fixed it.
* ``reward_traces.jsonl`` -> **gold-label behaviour**: abstention conditioned on
  the gold label, R_ground on extractions, and the within-group extract-vs-
  abstain advantage.

Pass/fail promotion gates live in :mod:`dagspaces.grpo_training.gates`
(``promotion_gates.json``); this module is the *descriptive* companion that
regenerates the field-note tables for any run.

The compute functions are pure (they take already-parsed lists/dicts) so they
are unit-testable without fixtures on disk; the ``load_*`` / ``find_*`` helpers
are thin IO around them. CLI: ``scripts/grpo_field_metrics.py``.
"""

from __future__ import annotations

import json
import math
import statistics as st
from pathlib import Path
from typing import Any

# ── Optimizer signals (trainer_state.json log_history) ──────────────────────

# Short name -> log_history key. These are the signals the field notes tabulate.
SIGNAL_KEYS: dict[str, str] = {
    "reward": "reward",
    "reward_std": "reward_std",
    "entropy": "entropy",
    "kl": "kl",
    "grad_norm": "grad_norm",
    "logp_diff_mean": "sampling/sampling_logp_difference/mean",
    "logp_diff_max": "sampling/sampling_logp_difference/max",
    "is_ratio_mean": "sampling/importance_sampling_ratio/mean",
    "is_ratio_min": "sampling/importance_sampling_ratio/min",
    "is_ratio_max": "sampling/importance_sampling_ratio/max",
    "clipped_ratio": "completions/clipped_ratio",
    "frac_reward_zero_std": "frac_reward_zero_std",
    "mean_length": "completions/mean_length",
}


def series(log_history: list[dict], key: str) -> list[tuple[Any, float]]:
    """Return ``[(step, value), ...]`` for every log entry carrying ``key``."""
    return [(e.get("step"), e[key]) for e in log_history if key in e]


def summarize_signal(log_history: list[dict], key: str) -> dict | None:
    """First/last/min/max summary for one log_history key, or None if absent."""
    vals = [v for _, v in series(log_history, key)]
    if not vals:
        return None
    return {
        "n": len(vals),
        "first": vals[0],
        "last": vals[-1],
        "min": min(vals),
        "max": max(vals),
    }


def reward_trend(values: list[float], frac: float = 1.0 / 3.0) -> dict:
    """First-third vs last-third mean + gain (mirrors the promotion gate).

    Matches :func:`dagspaces.grpo_training.gates` ``reward_trend`` so the
    descriptive table and the pass/fail gate never disagree on the sign.
    """
    n = len(values)
    if n == 0:
        return {"first_third_mean": 0.0, "last_third_mean": 0.0, "gain": 0.0, "n": 0}
    k = max(1, int(n * frac))
    first = st.mean(values[:k])
    last = st.mean(values[-k:])
    return {
        "first_third_mean": first,
        "last_third_mean": last,
        "gain": last - first,
        "n": n,
    }


def pearson(xs: list[float], ys: list[float]) -> float | None:
    """Pearson correlation of two equal-length series, or None if undefined."""
    n = min(len(xs), len(ys))
    if n < 2:
        return None
    xs, ys = xs[:n], ys[:n]
    mx, my = st.mean(xs), st.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / math.sqrt(sxx * syy)


def _aligned(log_history: list[dict], key_a: str, key_b: str) -> tuple[list, list]:
    """Values of two keys over the log entries where BOTH are present."""
    xs, ys = [], []
    for e in log_history:
        if key_a in e and key_b in e:
            xs.append(e[key_a])
            ys.append(e[key_b])
    return xs, ys


def summarize_log_history(
    log_history: list[dict], keys: dict[str, str] | None = None
) -> dict:
    """The optimizer-signal block for one run.

    Returns ``{"signals": {name: summary|None}, "reward_trend": {...},
    "correlations": {...}}``. Correlations are the instability fingerprint:
    entropy~logp_diff (v6 was +0.92) and entropy~IS ratio (v6 was -0.96).
    """
    keys = keys or SIGNAL_KEYS
    signals = {name: summarize_signal(log_history, lk) for name, lk in keys.items()}

    reward_vals = [v for _, v in series(log_history, SIGNAL_KEYS["reward"])]

    ent_lp_x, ent_lp_y = _aligned(
        log_history, SIGNAL_KEYS["entropy"], SIGNAL_KEYS["logp_diff_mean"]
    )
    ent_is_x, ent_is_y = _aligned(
        log_history, SIGNAL_KEYS["entropy"], SIGNAL_KEYS["is_ratio_mean"]
    )
    return {
        "signals": signals,
        "reward_trend": reward_trend(reward_vals),
        "correlations": {
            "entropy_vs_logp_diff": pearson(ent_lp_x, ent_lp_y),
            "entropy_vs_is_ratio": pearson(ent_is_x, ent_is_y),
        },
    }


def per_step_table(log_history: list[dict], keys: list[str] | None = None) -> list[dict]:
    """Compact per-log-step trajectory (the note's step/reward/entropy/kl table)."""
    keys = keys or ["reward", "entropy", "kl", "mean_length"]
    rows = []
    for e in log_history:
        row = {"step": e.get("step")}
        for name in keys:
            lk = SIGNAL_KEYS.get(name, name)
            if lk in e:
                row[name] = e[lk]
        rows.append(row)
    return rows


# ── Gold-label behaviour (reward_traces.jsonl) ──────────────────────────────


def is_abstention(rec: dict) -> bool:
    """True if a CI-extraction completion abstained (predicted no flow).

    Prefers the trace's stored ``is_no_flow`` flag; falls back to a structural
    check on the completion JSON for older traces that predate the flag.
    """
    if "is_no_flow" in rec and rec["is_no_flow"] is not None:
        return bool(rec["is_no_flow"])
    c = rec.get("completion") or ""
    return ('"has_information_exchange": false' in c) or ('"flows": []' in c)


def gold_label_metrics(ci_traces: list[dict]) -> dict:
    """Gold-conditional abstention + reward levels for CI-extraction traces.

    The smoking-gun metric is ``abstain_given_gold_yes`` (abstaining on a chunk
    that truly has a flow is wrong); an ideal policy drives it toward 0.
    """
    n = len(ci_traces)
    gold_yes = [r for r in ci_traces if r.get("gold_has_exchange") is True]
    gold_no = [r for r in ci_traces if r.get("gold_has_exchange") is False]
    abst = [r for r in ci_traces if is_abstention(r)]
    ext = [r for r in ci_traces if not is_abstention(r)]
    abst_y = [r for r in gold_yes if is_abstention(r)]
    abst_n = [r for r in gold_no if is_abstention(r)]
    ext_y = [r for r in gold_yes if not is_abstention(r)]
    ext_n = [r for r in gold_no if not is_abstention(r)]

    rg_ext = [
        r["components"]["r_ground"]
        for r in ext
        if r.get("components") and "r_ground" in r["components"]
    ]
    rg_zero = [x for x in rg_ext if x == 0.0]
    comp_abs = [r["composite"] for r in abst if "composite" in r]
    comp_ext = [r["composite"] for r in ext if "composite" in r]

    def frac(num: list, den: list) -> float | None:
        return len(num) / len(den) if den else None

    ext_rate_y = frac(ext_y, gold_yes)
    ext_rate_n = frac(ext_n, gold_no)
    gap = (
        (ext_rate_y - ext_rate_n)
        if (ext_rate_y is not None and ext_rate_n is not None)
        else None
    )
    return {
        "n_ci": n,
        "gold_yes_frac": frac(gold_yes, ci_traces),
        "abstain_frac": frac(abst, ci_traces),
        "abstain_given_gold_yes": frac(abst_y, gold_yes),
        "abstain_given_gold_no": frac(abst_n, gold_no),
        "extract_gap": gap,
        "rground_mean_on_extractors": st.mean(rg_ext) if rg_ext else None,
        "rground_zero_frac_on_extractors": frac(rg_zero, rg_ext),
        "composite_mean_abstain": st.mean(comp_abs) if comp_abs else None,
        "composite_mean_extract": st.mean(comp_ext) if comp_ext else None,
    }


def within_group_advantage(ci_traces: list[dict], gold_yes_only: bool = True) -> dict:
    """Mean extract-vs-abstain composite advantage over MIXED groups.

    A group (one GRPO ``call``) is MIXED when it contains >=1 abstain AND >=1
    extract completion; its advantage = mean(extract composite) - mean(abstain
    composite). This is the +0.72 figure the v6/v7 notes cite (gold-YES groups).
    """
    groups: dict[Any, list[dict]] = {}
    for r in ci_traces:
        if gold_yes_only and r.get("gold_has_exchange") is not True:
            continue
        groups.setdefault(r.get("call"), []).append(r)

    advantages = []
    n_positive = 0
    for members in groups.values():
        ext = [r["composite"] for r in members if not is_abstention(r) and "composite" in r]
        abs = [r["composite"] for r in members if is_abstention(r) and "composite" in r]
        if not ext or not abs:
            continue
        adv = st.mean(ext) - st.mean(abs)
        advantages.append(adv)
        if adv > 0:
            n_positive += 1
    return {
        "n_mixed_groups": len(advantages),
        "mean_advantage": st.mean(advantages) if advantages else None,
        "frac_groups_extract_wins": (n_positive / len(advantages)) if advantages else None,
    }


# ── IO helpers ──────────────────────────────────────────────────────────────


def load_jsonl(path: str | Path) -> list[dict]:
    """Load a JSONL file, skipping unparseable lines (truncated traces happen)."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def split_by_task(rows: list[dict]) -> dict[str, list[dict]]:
    """Partition trace rows by ``task_type`` (ci_extraction / norm_judgment)."""
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(r.get("task_type", "unknown"), []).append(r)
    return out


def _checkpoint_step(p: Path) -> int:
    name = p.parent.name  # checkpoint-150
    try:
        return int(name.split("-")[-1])
    except ValueError:
        return -1


def find_latest_trainer_state(run_dir: str | Path) -> Path | None:
    """Newest (highest-step) ``checkpoint-*/trainer_state.json`` under a run dir."""
    cands = list(Path(run_dir).rglob("checkpoint-*/trainer_state.json"))
    if not cands:
        return None
    return max(cands, key=_checkpoint_step)


def find_reward_traces(run_dir: str | Path) -> Path | None:
    """Newest ``reward_traces.jsonl`` under a run dir."""
    cands = list(Path(run_dir).rglob("reward_traces.jsonl"))
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)


def load_log_history(trainer_state_path: str | Path) -> list[dict]:
    """Read ``log_history`` out of a ``trainer_state.json``."""
    with open(trainer_state_path) as f:
        return json.load(f).get("log_history", [])
