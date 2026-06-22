#!/usr/bin/env python
"""Check GRPO promotion gates for one or more checkpoint directories.

A checkpoint should only graduate to full benchmark evaluation if its
training dynamics show actual learning (see dagspaces/grpo_training/gates.py
for the gate definitions and rationale).

Usage:
    python scripts/check_grpo_promotion_gates.py CHECKPOINT_DIR [CHECKPOINT_DIR ...]
        [--min-reward-gain 0.0] [--max-frac-zero-std 0.2]
        [--max-kl 1.0] [--no-flow-tolerance 0.15]

Writes ``promotion_gates.json`` into each checkpoint dir and exits non-zero
if any checkpoint fails any gate — suitable for gating eval sweeps:

    python scripts/check_grpo_promotion_gates.py multirun/.../checkpoint \
        && python -m dagspaces.goldcoin_hipaa.cli ...
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dagspaces.grpo_training.gates import check_promotion_gates  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_dirs", nargs="+")
    parser.add_argument("--min-reward-gain", type=float, default=None)
    parser.add_argument("--max-frac-zero-std", type=float, default=None)
    parser.add_argument("--max-kl", type=float, default=None)
    parser.add_argument("--no-flow-tolerance", type=float, default=None)
    args = parser.parse_args()

    thresholds = {
        k: v for k, v in {
            "min_reward_gain": args.min_reward_gain,
            "max_frac_zero_std": args.max_frac_zero_std,
            "max_kl": args.max_kl,
            "no_flow_tolerance": args.no_flow_tolerance,
        }.items() if v is not None
    }

    any_failed = False
    for ckpt_dir in args.checkpoint_dirs:
        report = check_promotion_gates(ckpt_dir, thresholds)
        out_path = os.path.join(ckpt_dir, "promotion_gates.json")
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
        except OSError as e:
            print(f"[gates] could not write {out_path}: {e}")

        verdict = "PROMOTE" if report.get("promote") else "HOLD"
        print(f"\n=== {ckpt_dir} → {verdict} ===")
        if "error" in report:
            print(f"  error: {report['error']}")
        for name, gate in report.get("gates", {}).items():
            detail = {k: v for k, v in gate.items() if k != "status"}
            print(f"  [{gate['status']:>7s}] {name}: {detail}")
        if not report.get("promote"):
            any_failed = True

    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())
