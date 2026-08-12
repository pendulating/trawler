#!/usr/bin/env python
"""Re-score existing CIRL-729 cells so their metrics.json carries ``*_scorable``.

Context. The camera-ready table blanked CIRL Lk/Util on 15 of 23 cells because
the strict (paper-parity) extractor requires ``</think>`` + ``<answer>...
</answer>`` and fewer than half the 729 actions cleared it. Investigation
(2026-08-03) showed those failures are three different things:

  * missing ``</think>`` only — the model emits a clean, complete
    ``<answer>`` block; our family yaml just runs it with thinking disabled
    (the whole Qwen family: qwen3.5-4b/sft is 728/729 well-formed answers,
    88/729 strict);
  * unclosed ``</answer>`` — a complete message, ``finish_reason=stop``, the
    model simply never emits the closing tag (llama / harc, ~400 rows each);
  * genuinely unscoreable output — an EMPTY final channel (gpt-oss harmony,
    631/729) or a fragment cut off at ``max_tokens`` (phi-4/sft 394,
    openthinker3-7b/sft 407).

Only the third is a real "this model cannot be evaluated" finding. The
``*_scorable`` metrics added to ``dagspaces/cirl/stages/compute_metrics.py``
separate them: rates over rows with a complete message, plus ``scorable_rate``
so a caller can refuse a cell built from a sliver.

This is a **re-score, not a re-run**: the per-row parse artifacts
(``outputs/parse_responses/dataset.parquet``) already hold everything needed,
so no GPU and no inference is involved, and no evaluated condition changes.
Each cell is backed up to ``*.pre_scorable_rescore.bak`` (never overwritten, so
re-running cannot destroy the original artifacts).

Usage:
    /share/pierson/matt/UAIR/.venv-vllm025cu129/bin/python \
        scripts/rescore_cirl_scorable.py [--dry-run]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys

REPO = "/share/pierson/matt/UAIR"
sys.path.insert(0, REPO)

# The CIRL-729 sweeps the camera-ready table reads (kept in sync with
# notebooks/colm-camera-ready/benchmark_results.py SWEEP_GLOBS) plus the
# judge-free variance record, whose CIRL cells feed the noise floor and so
# must be scored on the same quantity as the table.
SWEEP_GLOBS = [
    "*_eval_cirl729_canonical/*",
    "*_eval_cirl729_teacher/*",
    "*_eval_judgefree_variance/*",
    "*_eval_judgefree_variance_gptoss/*",
]

REPORT_KEYS = [
    ("scorable_rate", "scorable rate"),
    ("leakage.leakage_rate_scorable", "leak (scorable)"),
    ("utility.utility_rate_scorable", "util (scorable)"),
    ("leakage.leakage_rate", "leak (strict)"),
]


def _dig(d: dict, dotted: str):
    cur = d
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _backup(path: str) -> None:
    if os.path.exists(path):
        bak = path + ".pre_scorable_rescore.bak"
        if not os.path.exists(bak):
            shutil.copy2(path, bak)


def _model_of(cell_dir: str) -> str:
    ov = os.path.join(cell_dir, ".hydra", "overrides.yaml")
    if not os.path.exists(ov):
        # eval_all children keep the override one level up, next to the
        # benchmark dir.
        ov = os.path.join(os.path.dirname(cell_dir), ".hydra", "overrides.yaml")
    if not os.path.exists(ov):
        return "?"
    for line in open(ov, errors="ignore"):
        line = line.strip().lstrip("- ").strip()
        if line.startswith("model="):
            return line.split("=", 1)[1]
    return "?"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--multirun-root", default=os.path.join(REPO, "multirun"))
    args = ap.parse_args()

    import pandas as pd

    from dagspaces.cirl.stages.compute_metrics import (
        compute_metrics,
        metrics_to_dataframe,
    )

    # Every `<sweep>/<time>/<arm>/.../cirl/outputs/compute_metrics/` in scope.
    cells: list[str] = []
    for g in SWEEP_GLOBS:
        cells += glob.glob(
            os.path.join(args.multirun_root, g, "*", "**", "cirl", "outputs"),
            recursive=True,
        )
    cells = sorted({c for c in cells if os.path.isdir(c)})
    if not cells:
        print("no CIRL cells matched — check SWEEP_GLOBS / --multirun-root")
        return 1

    print(f"{len(cells)} CIRL cells in scope\n")
    n_done = n_skip = n_already = 0
    for out_dir in cells:
        parsed = os.path.join(out_dir, "parse_responses", "dataset.parquet")
        mjson = os.path.join(out_dir, "compute_metrics", "metrics.json")
        cell = os.path.relpath(out_dir, args.multirun_root)
        model = _model_of(os.path.dirname(os.path.dirname(out_dir)))
        if not os.path.exists(parsed):
            print(f"SKIP  {model:<32} no parse_responses parquet  [{cell}]")
            n_skip += 1
            continue

        old = {}
        if os.path.exists(mjson):
            try:
                old = json.load(open(mjson))
            except ValueError:
                old = {}
        if "scorable_rate" in old and not args.dry_run:
            n_already += 1
            continue

        df = pd.read_parquet(parsed)
        new = compute_metrics(df)

        # A re-score must not move the paper-parity headline. If it does, the
        # parse artifacts and the recorded metrics disagree and the cell needs
        # a human, not an overwrite.
        for key in ("net_score", "leakage.leakage_rate", "utility.utility_rate"):
            o, n = _dig(old, key), _dig(new, key)
            if o is not None and n is not None and abs(float(o) - float(n)) > 1e-6:
                print(
                    f"!! {model}: re-score CHANGED the strict {key} "
                    f"({o} -> {n}) — refusing this cell. [{cell}]"
                )
                break
        else:
            sr = new.get("scorable_rate")
            exc = new.get("scorable_exclusions", {})
            print(
                f"{'DRY ' if args.dry_run else 'OK  '}{model:<32} "
                f"scorable {new.get('scorable')}/{new.get('total')} "
                f"({sr:.1%})  leak {_dig(new, 'leakage.leakage_rate_scorable')}  "
                f"util {_dig(new, 'utility.utility_rate_scorable')}  "
                f"[empty {exc.get('empty_answer', 0)}, "
                f"trunc {exc.get('truncated', 0)}]"
            )
            if not args.dry_run:
                _backup(mjson)
                _backup(os.path.join(out_dir, "compute_metrics", "metrics.parquet"))
                os.makedirs(os.path.dirname(mjson), exist_ok=True)
                with open(mjson, "w") as f:
                    json.dump(new, f, indent=2, default=str)
                metrics_to_dataframe(new).to_parquet(
                    os.path.join(out_dir, "compute_metrics", "metrics.parquet"),
                    index=False,
                )
            n_done += 1

    print(
        f"\n{n_done} cells re-scored, {n_already} already had *_scorable, "
        f"{n_skip} skipped."
    )
    if args.dry_run:
        print("(dry run — nothing written)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
