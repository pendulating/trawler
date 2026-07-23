#!/usr/bin/env python
"""F1 rescue: re-finalize canonical PrivacyLens cells with the fixed parser.

Context (wiki/changelog/2026-07-21_privacylens_parity_review.md, F1): the
judge-response parsers applied the upstream free-text substring scan to
guided-JSON responses, mis-scoring ~21.5% of helpfulness judgments (and a
handful of leakage flips) in every run finalized since 2026-04-26 — including
the keeper-era camera-ready columns. The raw judge responses live on disk
(``output.jsonl`` next to each ``*_judge_batch``), so re-running
``finalize_async`` with the fixed JSON-first parser recovers correct numbers
for existing runs WITHOUT any GPU or judge work, and WITHOUT changing the
eval protocol (this is a parse-only rescue: the F3/F4 protocol fixes only
affect new runs).

For every canonical PL cell this script:
  1. backs up ``outputs/compute_metrics/metrics.{json,parquet}`` and both
     judge ``results.parquet`` files to ``*.pre_f1_rescue.bak`` (first run
     only — an existing backup is never overwritten, so re-running the
     script cannot destroy the original corrupted artifacts);
  2. calls ``dagspaces.privacylens.stages.finalize_async.finalize_async``
     in place (regenerates results.parquet + metrics.{json,parquet};
     the metrics.json mtime bump is what clears the ‡ stale flag in
     notebooks/colm-camera-ready/benchmark_results.py);
  3. prints old→new for the table-facing metrics.

Cells whose judges ran in live mode (``*_judge_inference/`` layout, no
``output.jsonl``) are reported and SKIPPED — the teacher reference row is the
known case; its judged columns stay ‡ until a re-run.

Usage:
    /share/pierson/matt/UAIR/.venv-vllm025cu129/bin/python \
        scripts/rescue_privacylens_f1_refinalize.py [--dry-run]
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

# The canonical-sweep globs the camera-ready table draws from (kept in sync
# with notebooks/colm-camera-ready/benchmark_results.py SWEEP_GLOBS).
SWEEP_GLOBS = [
    "*_eval_canonical_instruct/*",
    "*_eval_canonical_sft_gemma4/*",
    "*_eval_canonical_repair/*",
    "*_eval_canonical_gptoss_refix/*",
    "*_eval_gemma4_q7_backfill/*",
    "*_eval_teacher_gemma4_31b/*",
]

PL_INNER = "privacylens/privacylens_eval"
METRIC_KEYS = [
    ("helpfulness.mean_score_among_parseable", "Help (mean)"),
    ("helpfulness.helpful_rate_among_parseable", "Helpful rate"),
    ("adjusted_leakage.adjusted_leakage_rate", "Adj leakage"),
    ("leakage.leakage_rate_among_parseable", "Leakage"),
]


def _dig(d: dict, dotted: str):
    cur = d
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _backup(path: str) -> None:
    """Copy ``path`` to ``path.pre_f1_rescue.bak`` unless a backup exists."""
    if os.path.exists(path):
        bak = path + ".pre_f1_rescue.bak"
        if not os.path.exists(bak):
            shutil.copy2(path, bak)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="List cells and their status; change nothing.")
    args = ap.parse_args()

    from dagspaces.privacylens.stages.finalize_async import finalize_async

    roots = sorted(
        {
            os.path.join(cell, PL_INNER)
            for g in SWEEP_GLOBS
            for cell in glob.glob(os.path.join(REPO, "multirun", g, "*"))
            if os.path.isdir(os.path.join(cell, PL_INNER))
        }
    )
    print(f"[rescue] {len(roots)} privacylens cell(s) found under canonical sweeps\n")

    rescued, skipped, failed = [], [], []
    for root in roots:
        rel = os.path.relpath(root, os.path.join(REPO, "multirun"))
        leak_jsonl = os.path.join(root, "outputs/leakage_judge_batch/output.jsonl")
        help_jsonl = os.path.join(root, "outputs/helpfulness_judge_batch/output.jsonl")
        metrics_json = os.path.join(root, "outputs/compute_metrics/metrics.json")

        if not (os.path.exists(leak_jsonl) and os.path.exists(help_jsonl)):
            mode = "live-mode (no output.jsonl)" if glob.glob(
                os.path.join(root, "outputs/*_judge_inference")
            ) else "missing judge artifacts"
            print(f"SKIP  {rel}  [{mode}]")
            skipped.append((rel, mode))
            continue

        old = {}
        if os.path.exists(metrics_json):
            with open(metrics_json) as f:
                old_metrics = json.load(f)
            old = {k: _dig(old_metrics, k) for k, _ in METRIC_KEYS}

        if args.dry_run:
            print(f"WOULD RESCUE  {rel}")
            rescued.append(rel)
            continue

        try:
            _backup(metrics_json)
            _backup(os.path.join(root, "outputs/compute_metrics/metrics.parquet"))
            _backup(os.path.join(root, "outputs/leakage_judge_batch/results.parquet"))
            _backup(os.path.join(root, "outputs/helpfulness_judge_batch/results.parquet"))

            result = finalize_async(root)
            new_metrics = result["metrics"]

            print(f"OK    {rel}")
            for key, label in METRIC_KEYS:
                o, n = old.get(key), _dig(new_metrics, key)
                if o is None and n is None:
                    continue
                delta = (
                    f"  ({n - o:+.4f})"
                    if isinstance(o, (int, float)) and isinstance(n, (int, float))
                    else ""
                )
                print(f"        {label:>14s}: {o} -> {n}{delta}")
            rescued.append(rel)
        except Exception as exc:  # noqa: BLE001 — report and continue the batch
            print(f"FAIL  {rel}: {exc!r}")
            failed.append((rel, repr(exc)))

    print(
        f"\n[rescue] done: {len(rescued)} rescued, {len(skipped)} skipped, "
        f"{len(failed)} failed"
    )
    for rel, why in skipped:
        print(f"  skipped: {rel} — {why}")
    for rel, why in failed:
        print(f"  FAILED:  {rel} — {why}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
