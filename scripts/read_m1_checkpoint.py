#!/usr/bin/env python3
"""Read the m1 grid's discrimination metrics from W&B.

The four cells launched 2026-07-26 loaded ModularReward BEFORE
balanced_accuracy / youden_j were added, so they emit only the two per-class
recalls. Those recalls are conditioned on disjoint gold subsets and do NOT sum
to 1 — any non-discriminating policy (blanket label, coin flip) sums to exactly
1.0, a perfect one to 2.0. This derives the decision-relevant numbers:

    balanced accuracy = mean(per-class recall)   # 0.5 = blanket floor
    Youden's J        = sum(recalls) - 1         # 0 = no discrimination

Balanced accuracy is also exactly what the macro-EM reward computes, so it is
the quantity the kill criterion should be read against. `agreement_mean` is
micro-averaged over a ~72/28 split and mostly tracks the majority class.
"""
from __future__ import annotations

import argparse
import statistics
import sys

import wandb

APPR = "reward/direct/agreement_by_class/appropriate"
INAP = "reward/direct/agreement_by_class/inappropriate"
EXTRA = ["reward/direct/agreement_mean", "reward/direct/hedge_frac",
         "reward/direct/antithesis_frac", "reward/direct/unscored_flow_frac",
         "reward/valid/gate_frac"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="uair/grpo-ci-training")
    ap.add_argument("--tag", default="grpo_m1_")
    ap.add_argument("--window", type=float, default=0.2,
                    help="fraction of history averaged at each end")
    ap.add_argument("--state", default="running",
                    help="filter runs by state ('running', or 'any' for all). "
                         "Without this, cancelled earlier waves match the tag "
                         "too and swamp the output.")
    args = ap.parse_args()

    api = wandb.Api(timeout=60)
    runs = [r for r in api.runs(args.project, order="-created_at", per_page=60)
            if args.tag in r.name and "grpo_training" in r.name
            and (args.state == "any" or r.state == args.state)]
    if not runs:
        print("no matching runs", file=sys.stderr)
        return 1

    print(f"{'cell':13s} {'n':>5s} {'bal-acc':>16s} {'Youden J':>16s} {'verdict':>10s}")
    for r in sorted(runs, key=lambda x: x.name):
        rows = list(r.scan_history(keys=[APPR, INAP] + EXTRA, page_size=2000))
        pairs = [(x[APPR], x[INAP]) for x in rows
                 if x.get(APPR) is not None and x.get(INAP) is not None]
        cell = r.name.split("grpo_m1_")[1].split("-grpo_training")[0]
        if len(pairs) < 6:
            print(f"{cell:13s} {len(pairs):5d}   (no direct core — expected for -outcome)")
            continue
        w = max(1, int(len(pairs) * args.window))
        ba = lambda ps: statistics.mean((a + i) / 2 for a, i in ps)
        b0, b1 = ba(pairs[:w]), ba(pairs[-w:])
        j0, j1 = 2 * b0 - 1, 2 * b1 - 1
        verdict = "RISING" if j1 > j0 + 0.01 else ("FALLING" if j1 < j0 - 0.01 else "flat")
        print(f"{cell:13s} {len(pairs):5d}   {b0:.3f} -> {b1:.3f}   "
              f"{j0:.3f} -> {j1:.3f}   {verdict:>8s}")
    print("\nbal-acc 0.5 / J 0.0 = blanket floor (no discrimination).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
