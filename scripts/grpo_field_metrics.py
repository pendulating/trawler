#!/usr/bin/env python3
"""Regenerate the GRPO training field-note metrics for any run.

The dated notes under ``wiki/grpo_training_field_notes/`` (v2-v5, v6+v7, ...)
cite optimizer-signal and gold-label tables. This CLI reproduces those tables
for an arbitrary run directory, so the notes stay falsifiable as runs change.

All computation lives in :mod:`dagspaces.grpo_training.trace_metrics` (unit
tested in ``tests/grpo_training/test_trace_metrics.py``); this file is a thin
discover-load-print wrapper.

Usage:
    python scripts/grpo_field_metrics.py <RUN_DIR> [<RUN_DIR> ...]
    python scripts/grpo_field_metrics.py <RUN_DIR> --json      # machine-readable

Example (the v7 pilot):
    python scripts/grpo_field_metrics.py \
        multirun/2026-06-21_grpo_v7pilot_beta/20-18-12
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dagspaces.grpo_training.trace_metrics import (  # noqa: E402
    SIGNAL_KEYS,
    find_latest_trainer_state,
    find_reward_traces,
    gold_label_metrics,
    load_jsonl,
    load_log_history,
    per_step_table,
    split_by_task,
    summarize_log_history,
    within_group_advantage,
)


def _fmt(v, spec="+.4f"):
    return "  -  " if v is None else format(v, spec)


def collect(run_dir: str | Path) -> dict:
    """All field-note metrics for one run, plus the artifact paths used."""
    out: dict = {"run_dir": str(run_dir), "artifacts": {}}

    ts = find_latest_trainer_state(run_dir)
    out["artifacts"]["trainer_state"] = str(ts) if ts else None
    if ts:
        lh = load_log_history(ts)
        out["optimizer"] = summarize_log_history(lh)
        out["trajectory"] = per_step_table(lh)

    rt = find_reward_traces(run_dir)
    out["artifacts"]["reward_traces"] = str(rt) if rt else None
    if rt:
        ci = split_by_task(load_jsonl(rt)).get("ci_extraction", [])
        out["gold_label"] = gold_label_metrics(ci)
        out["within_group_advantage_gold_yes"] = within_group_advantage(
            ci, gold_yes_only=True
        )

    # Gate verdict, if the run already wrote one (see gates.py).
    gates_path = None
    if ts:
        cand = ts.parent.parent / "promotion_gates.json"
        if cand.exists():
            gates_path = cand
    out["artifacts"]["promotion_gates"] = str(gates_path) if gates_path else None
    if gates_path:
        with open(gates_path) as f:
            g = json.load(f)
        out["promotion"] = {
            "promote": g.get("promote"),
            "gates": {k: v.get("status") for k, v in g.get("gates", {}).items()},
            "no_flow_rate": g.get("gates", {}).get("no_flow_rate", {}),
        }
    return out


def render(m: dict) -> str:
    lines: list[str] = []
    lines.append(f"### {m['run_dir']}")
    for k, v in m["artifacts"].items():
        lines.append(f"  {k}: {v}")

    if "optimizer" in m:
        lines.append("\nOPTIMIZER SIGNALS (first -> last [min / max])")
        for name in SIGNAL_KEYS:
            s = m["optimizer"]["signals"].get(name)
            if not s:
                continue
            lines.append(
                f"  {name:22s} {s['first']:+.4f} -> {s['last']:+.4f}"
                f"   [{s['min']:+.4f} / {s['max']:+.4f}]  (n={s['n']})"
            )
        rt = m["optimizer"]["reward_trend"]
        lines.append(
            f"  reward_trend           first3={rt['first_third_mean']:+.4f} "
            f"last3={rt['last_third_mean']:+.4f} gain={rt['gain']:+.4f}"
        )
        c = m["optimizer"]["correlations"]
        lines.append(
            f"  corr(entropy,logp_diff)={_fmt(c['entropy_vs_logp_diff'], '+.2f')}  "
            f"corr(entropy,IS)={_fmt(c['entropy_vs_is_ratio'], '+.2f')}"
        )

    if "gold_label" in m:
        g = m["gold_label"]
        lines.append("\nGOLD-LABEL BEHAVIOUR (ci_extraction)")
        lines.append(f"  n_ci={g['n_ci']}  gold_yes_frac={_fmt(g['gold_yes_frac'], '.3f')}")
        lines.append(
            f"  abstain_frac={_fmt(g['abstain_frac'], '.3f')}  "
            f"abst|gold=YES={_fmt(g['abstain_given_gold_yes'], '.3f')} (wrong)  "
            f"abst|gold=NO={_fmt(g['abstain_given_gold_no'], '.3f')} (correct)"
        )
        lines.append(
            f"  R_ground|ext={_fmt(g['rground_mean_on_extractors'], '.3f')}  "
            f"R_ground=0|ext={_fmt(g['rground_zero_frac_on_extractors'], '.3f')}  "
            f"composite abstain={_fmt(g['composite_mean_abstain'])} "
            f"extract={_fmt(g['composite_mean_extract'])}"
        )
        a = m["within_group_advantage_gold_yes"]
        lines.append(
            f"  within-group extract-vs-abstain advantage (gold=YES mixed): "
            f"mean={_fmt(a['mean_advantage'])} over {a['n_mixed_groups']} groups, "
            f"extract wins in {_fmt(a['frac_groups_extract_wins'], '.2%')}"
        )

    if "promotion" in m:
        p = m["promotion"]
        lines.append(f"\nPROMOTION: promote={p['promote']}  gates={p['gates']}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+", help="GRPO run director(ies)")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of text")
    args = ap.parse_args()

    results = [collect(d) for d in args.run_dirs]
    if args.json:
        print(json.dumps(results, indent=2, default=str))
    else:
        print("\n\n".join(render(m) for m in results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
