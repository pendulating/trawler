#!/usr/bin/env python3
"""Apply the pre-registered §8 promotion gates to k-series probe output.

Pure offline pandas over ``probe_results.parquet`` (written by
``scripts/kto_heldout_probe.py``); gate logic lives in
``dagspaces/grpo_training/stages/kto_probe.py`` (unit-tested). Prints a
per-arm verdict table and the promoted checkpoint list.

  python scripts/check_kto_promotion.py outputs/2026-08-01_k3_probe
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path("/share/pierson/matt/UAIR")
sys.path.insert(0, str(ROOT))

import pandas as pd

from dagspaces.grpo_training.stages.kto_probe import (
    GATE_FAIL_MAX,
    NOISE_FLOOR,
    PROMOTION_BAR,
    evaluate_promotion_gates,
    summarize_checkpoint,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("probe_dir")
    ap.add_argument("--noise-floor", type=float, default=NOISE_FLOOR)
    ap.add_argument("--promotion-bar", type=float, default=PROMOTION_BAR)
    ap.add_argument("--gate-fail-max", type=float, default=GATE_FAIL_MAX)
    args = ap.parse_args()

    d = Path(args.probe_dir)
    df = pd.read_parquet(d / "probe_results.parquet")
    meta = json.load(open(d / "probe_meta.json"))
    print(f"[promotion] tier={meta['tier']} n_chunks={meta['n_chunks']} "
          f"k={meta['n_samples']} | bar>{args.promotion_bar} "
          f"noise={args.noise_floor} gate_fail<={args.gate_fail_max}")

    baseline = summarize_checkpoint(df[df["slice"] == "baseline"])
    print("\nbaseline (epoch-0 SFT policy):")
    for k, v in baseline.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    arms: dict[str, list] = {}
    for s in meta["slices"]:
        if s == "baseline":
            continue
        arm, _, ckpt = s.partition("/")
        arms.setdefault(arm, []).append((ckpt, s))

    promoted_all = []
    for arm, slices in arms.items():
        def _step(t):  # checkpoint-N -> N; "final" sorts last
            return (1 << 30) if t[0] == "final" else int(t[0].rsplit("-", 1)[1])
        curve = []
        for ckpt, s in sorted(slices, key=_step):
            m = summarize_checkpoint(df[df["slice"] == s])
            m["checkpoint"] = s
            curve.append(m)
        verdicts = evaluate_promotion_gates(
            curve, baseline, args.noise_floor, args.promotion_bar,
            args.gate_fail_max)
        print(f"\narm {arm}:")
        vdf = pd.DataFrame(verdicts).set_index("checkpoint")
        cdf = pd.DataFrame(curve).set_index("checkpoint")[
            ["minority_acc", "majority_acc", "gate_fail_rate",
             "abstain_rate_gold_no", "miss_rate"]]
        print(cdf.round(4).join(vdf).to_string())
        promoted_all += [v["checkpoint"] for v in verdicts if v["promoted"]]

    print(f"\nPROMOTED: {promoted_all or 'none'}")
    (d / "promotion_verdict.json").write_text(json.dumps({
        "baseline": baseline, "promoted": promoted_all,
        "knobs": {"noise_floor": args.noise_floor,
                  "promotion_bar": args.promotion_bar,
                  "gate_fail_max": args.gate_fail_max},
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
