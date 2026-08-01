#!/usr/bin/env python
"""F1 rescue for the keeper-era v9 PL trio (live-mode judge layout).

``rescue_privacylens_f1_refinalize.py`` covered the canonical-sweep cells
(batch layout, ``output.jsonl``) and SKIPPED live-mode cells. The keeper
comparison (base / sft-contentless-v6 / v9-ckpt100,
``multirun/2026-06-24_privacylens_v9_vs_sft_vs_base``) ran its judges in
LIVE mode, but the raw guided-JSON judge responses survive in
``*_judge_inference/results.parquet`` (``helpfulness_judge_text`` /
``leak_judge_text``), so the same parse-only rescue applies: re-parse with
the fixed JSON-first parsers and recompute metrics through the production
``compute_metrics``. No GPU, no judge calls, no protocol change.

Backups follow the F1 convention (``*.pre_f1_rescue.bak``, never
overwritten). Prints old→new for the paper-facing metrics.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

REPO = Path("/share/pierson/matt/UAIR")
sys.path.insert(0, str(REPO))

import pandas as pd

from dagspaces.privacylens.stages.compute_metrics import (
    compute_metrics,
    metrics_to_dataframe,
)
from dagspaces.privacylens.stages.parse_responses import (
    parse_helpfulness_responses,
    parse_leakage_responses,
)

TRIO = REPO / "multirun/2026-06-24_privacylens_v9_vs_sft_vs_base/20-25-49"
CELLS = {0: "base (instruct)", 1: "sft-contentless-v6", 2: "v9-ckpt100"}
KEYS = [
    ("leakage", "leakage_rate_among_parseable"),
    ("helpfulness", "helpful_rate_among_parseable"),
    ("helpfulness", "mean_score_among_parseable"),
    ("adjusted_leakage", "adjusted_leakage_rate"),
]


def _backup(p: Path) -> None:
    bak = p.with_suffix(p.suffix + ".pre_f1_rescue.bak")
    if p.exists() and not bak.exists():
        shutil.copy2(p, bak)


def main() -> int:
    for i, name in CELLS.items():
        out = TRIO / str(i) / "privacylens_eval/outputs"
        qa = pd.read_parquet(out / "qa_probe_inference/results.parquet")
        leak = pd.read_parquet(out / "leakage_judge_inference/results.parquet")
        hlp = pd.read_parquet(
            out / "helpfulness_judge_inference/results.parquet")

        n_h_flip = n_l_flip = 0
        old_h = hlp["helpfulness_score"].copy()
        hlp = parse_helpfulness_responses(hlp)
        n_h_flip = int((old_h != hlp["helpfulness_score"]).sum())
        old_l = leak["leak_flag"].copy()
        leak = parse_leakage_responses(leak)
        n_l_flip = int((old_l != leak["leak_flag"]).sum())

        mdir = out / "compute_metrics"
        old = json.load(open(mdir / "metrics.json"))
        for f in ("metrics.json", "metrics.parquet"):
            _backup(mdir / f)
        _backup(out / "helpfulness_judge_inference/results.parquet")
        _backup(out / "leakage_judge_inference/results.parquet")

        new = compute_metrics(qa, leak, hlp)
        (mdir / "metrics.json").write_text(json.dumps(new, indent=2))
        metrics_to_dataframe(new).to_parquet(mdir / "metrics.parquet")
        hlp.to_parquet(out / "helpfulness_judge_inference/results.parquet")
        leak.to_parquet(out / "leakage_judge_inference/results.parquet")

        print(f"\n== cell {i}: {name} (reparse flips: "
              f"helpfulness {n_h_flip}/{len(hlp)}, leakage {n_l_flip}/{len(leak)})")
        for sect, key in KEYS:
            o = old.get(sect, {}).get(key)
            n = new.get(sect, {}).get(key)
            print(f"  {sect}.{key}: {o} -> {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
