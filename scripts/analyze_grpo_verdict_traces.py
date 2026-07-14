#!/usr/bin/env python3
"""Verdict-behaviour forensics over GRPO reward traces (v11+ steering tool).

Reads one or more reward_traces.jsonl files (or run directories containing
one) and computes the metrics the v9→v11 reward iterations steer on:

  (1) judgment-vignette gold mix — the REALISED yes:no ratio at the
      completion level (the configured pool ratio drifts through the
      force-blind variance screen; v10: pool 3.07:1 → realised 5.2:1),
  (2) per-gold-class vignette verdict rates + accuracy over training —
      the drift check (v10's 5:1 mix eroded gold-"no" accuracy 0.84→0.77;
      the v11 probe's 2:1 mix held it ~0.91),
  (3) extraction-side appropriateness-direction tier mass on
      prohibited/discouraged-governed flows (correct-commit ×1.0 /
      hedge ×0.7 / v12a hedge-on-prohibited ×0.5 / false-permit ×0.1) —
      frozen at ~72% hedge through v10 AND the v11 probe; the binding
      constraint on GoldCoin Forbid recall. v12a's falsifiable prediction
      reads off this table: correct-commit share off ~0.10,
  (4) exploration guard — fraction of prohibited-governed traced groups
      containing ≥1 correct commit (GRPO can only amplify what the policy
      samples; if this decays, no reward shape can help),
  (5) governing-norm force mix among traced extraction flows.

NB traces log only the first 8 completions per reward call (one G=8 group),
so these are sampled estimates; the ``vignette/*`` W&B series pushed by
``CompositeRewardFunction._push_vignette_health`` covers every completion
live. See wiki/grpo_training_field_notes/2026-07-01_v11_probe_midrun_forensics.md.

Usage:
  python scripts/analyze_grpo_verdict_traces.py RUN_DIR_OR_TRACE [MORE ...] [--bins 3]
  python scripts/analyze_grpo_verdict_traces.py \
      multirun/2026-06-30_grpo_probe_top100_vignettes \
      multirun/2026-06-24_grpo_redesign_full_v10
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

from dagspaces.grpo_training.stages.rewards import _parse_judgment_completion

# Direction-multiplier tier boundaries (deontic.appropriateness_multiplier at
# the production floors app_floor=0.4 / app_floor_prohibit=0.1): correct 1.0,
# hedge 0.7, false-permit 0.1. v12a adds rground_app_hedge_prohibit=0.5 — a
# hedge on a prohibited-governed flow lands at 0.5 exactly, binned as its own
# "hedge_prohibit" tier (tight ±0.01 window; in pre-v12a runs a 0.50 mean is a
# rare 3-flow coincidence, e.g. 2×0.7+0.1). Multi-flow candidates mean per-flow
# multipliers, so anything strictly between tiers is binned "mixed".
_CORRECT_MIN = 0.99
_HEDGE_LO, _HEDGE_HI = 0.65, 0.99
_HEDGE_PROHIBIT_LO, _HEDGE_PROHIBIT_HI = 0.49, 0.51
_FALSE_PERMIT_MAX = 0.11

_NO_GOLD_FORCES = ("prohibited", "discouraged")


def resolve_trace_path(path: str) -> Optional[str]:
    """Accept a reward_traces.jsonl path or a run dir containing one."""
    if os.path.isfile(path):
        return path
    hits = sorted(glob.glob(os.path.join(path, "**", "reward_traces.jsonl"),
                            recursive=True))
    return hits[0] if hits else None


def load_entries(trace_path: str) -> List[Dict[str, Any]]:
    entries = []
    with open(trace_path, encoding="utf-8") as f:
        for line in f:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def model_verdict(completion: str) -> str:
    """The model's yes/no answer, normalised as r_judgment normalises it."""
    parsed = _parse_judgment_completion(completion or "")
    if not isinstance(parsed, dict):
        return ""
    return str(parsed.get("judgment", "")).lower().strip()


def _bin_of(call: int, max_call: int, n_bins: int) -> int:
    if max_call <= 0:
        return 0
    return min(int(n_bins * call / (max_call + 1)), n_bins - 1)


def _tier(direction: float) -> str:
    if direction >= _CORRECT_MIN:
        return "correct"
    if direction <= _FALSE_PERMIT_MAX:
        return "false_permit"
    if _HEDGE_LO <= direction < _HEDGE_HI:
        return "hedge"
    if _HEDGE_PROHIBIT_LO <= direction <= _HEDGE_PROHIBIT_HI:
        return "hedge_prohibit"
    return "mixed"


def analyze(entries: List[Dict[str, Any]], n_bins: int) -> Dict[str, Any]:
    """Compute all report tables from raw trace entries (pure; unit-testable)."""
    max_call = max((e.get("call", 0) for e in entries), default=0)

    # (call, gold, verdict) per vignette completion
    vig: List[Tuple[int, str, str]] = []
    # (call, force, direction) per traced extraction flow with a governing force
    ext: List[Tuple[int, str, Optional[float]]] = []

    for e in entries:
        call = e.get("call", 0)
        if e.get("task_type") == "norm_judgment":
            gold = str(e.get("gold_judgment") or "").lower()
            if gold in ("yes", "no"):
                vig.append((call, gold, model_verdict(e.get("completion", ""))))
        elif e.get("task_type") == "ci_extraction":
            for fl in (e.get("rground_flows") or []):
                if isinstance(fl, dict) and fl.get("norm_force"):
                    ext.append((call, str(fl["norm_force"]).lower(),
                                fl.get("app_direction")))

    gold_mix = Counter(g for _, g, _ in vig)

    # Per-bin, per-gold-class verdict rates
    verdicts: Dict[int, Dict[str, Counter]] = defaultdict(
        lambda: {"yes": Counter(), "no": Counter()})
    for call, gold, verdict in vig:
        verdicts[_bin_of(call, max_call, n_bins)][gold][verdict or "<unparsed>"] += 1

    # Per-bin direction-tier mass on prohibited/discouraged-governed flows
    tiers: Dict[int, Counter] = defaultdict(Counter)
    # Exploration guard: per traced group (= one call), does any prohibited-
    # governed flow commit correctly?
    groups: Dict[int, List[float]] = defaultdict(list)
    for call, force, direction in ext:
        if force in _NO_GOLD_FORCES and direction is not None:
            tiers[_bin_of(call, max_call, n_bins)][_tier(float(direction))] += 1
            groups[call].append(float(direction))
    explore: Dict[int, List[bool]] = defaultdict(list)
    for call, dirs in groups.items():
        explore[_bin_of(call, max_call, n_bins)].append(
            any(d >= _CORRECT_MIN for d in dirs))

    force_mix = Counter(f for _, f, _ in ext)

    return {
        "max_call": max_call,
        "n_vignette_completions": len(vig),
        "n_extraction_flows": len(ext),
        "gold_mix": dict(gold_mix),
        "verdicts": verdicts,
        "tiers": tiers,
        "explore": explore,
        "force_mix": dict(force_mix),
    }


def print_report(name: str, r: Dict[str, Any], n_bins: int) -> None:
    labels = [f"bin{i}" for i in range(n_bins)]
    if n_bins == 3:
        labels = ["early", "mid", "late"]

    print("=" * 88)
    print(f"{name} — calls ≤ {r['max_call']}, "
          f"{r['n_vignette_completions']} vignette completions, "
          f"{r['n_extraction_flows']} governed extraction flows traced")
    print("=" * 88)

    gm = r["gold_mix"]
    yes, no = gm.get("yes", 0), gm.get("no", 0)
    print(f"\n(1) realised vignette gold mix: {yes} yes : {no} no "
          f"= {yes / max(no, 1):.2f}:1")

    print("\n(2) vignette verdicts by gold class over training")
    for gold in ("yes", "no"):
        for b in range(n_bins):
            cnt = r["verdicts"].get(b, {}).get(gold, Counter())
            n = sum(cnt.values())
            if not n:
                continue
            y, nn = cnt.get("yes", 0) / n, cnt.get("no", 0) / n
            print(f"  gold={gold:3s} {labels[b]:5s} n={n:5d}  "
                  f"says yes {y:.2f} / no {nn:.2f} / other {1 - y - nn:.2f}  "
                  f"acc={cnt.get(gold, 0) / n:.2f}")

    print("\n(3) direction-tier mass on prohibited/discouraged-governed flows")
    for b in range(n_bins):
        cnt = r["tiers"].get(b, Counter())
        n = sum(cnt.values())
        if not n:
            continue
        hp = cnt.get("hedge_prohibit", 0)
        print(f"  {labels[b]:5s} n={n:5d}  "
              f"correct-commit {cnt.get('correct', 0) / n:.2f}  "
              f"hedge(0.7) {cnt.get('hedge', 0) / n:.2f}  "
              + (f"hedge-prohibit(0.5) {hp / n:.2f}  " if hp else "")
              + f"false-permit(0.1) {cnt.get('false_permit', 0) / n:.2f}  "
              f"mixed {cnt.get('mixed', 0) / n:.2f}")

    print("\n(4) exploration guard: traced prohibited-governed groups with ≥1 correct commit")
    for b in range(n_bins):
        flags = r["explore"].get(b, [])
        if flags:
            print(f"  {labels[b]:5s} {sum(flags)}/{len(flags)} groups "
                  f"({sum(flags) / len(flags):.2f})")

    print(f"\n(5) governing-norm force mix (traced flows): {r['force_mix']}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("paths", nargs="+",
                    help="reward_traces.jsonl file(s) or run dir(s) containing one")
    ap.add_argument("--bins", type=int, default=3,
                    help="number of training-time bins (default 3: early/mid/late)")
    args = ap.parse_args()

    for path in args.paths:
        trace = resolve_trace_path(path)
        if trace is None:
            print(f"!! no reward_traces.jsonl under {path} — skipping")
            continue
        entries = load_entries(trace)
        if not entries:
            print(f"!! empty/unparseable trace at {trace} — skipping")
            continue
        print_report(trace, analyze(entries, args.bins), args.bins)


if __name__ == "__main__":
    main()
