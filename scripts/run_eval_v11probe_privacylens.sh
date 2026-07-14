#!/bin/bash
# JUDGE-BASED PrivacyLens eval of the top-2 v11-probe GRPO checkpoints vs the
# v9 keeper and the v10 best — the four-way head-to-head on the privacy-utility
# frontier, all judged in ONE batch (Qwen3.6-27B @ klara:8002, job 488113) so the
# LLM-judge state is identical across arms (the v9-vs-SFT leakage margin was ~1pp
# at n=493, so same-batch judging is required for a trustworthy comparison).
#
# Four arms on the SAME Qwen3.5-9B contentless-v6 SFT base, advancing only the LoRA:
#   arm 0  v11probe-ckpt350  v11 probe (top100 vignettes), epoch ~1.99
#                            (GoldCoin comp-F1 0.755, probe best)
#   arm 1  v11probe-ckpt200  v11 probe, epoch ~1.14 (comp-F1 0.733, runner-up)
#   arm 2  v9-ckpt100        the paper keeper — CONTROL
#   arm 3  v10-ckpt250       v10 best (comp-F1 0.755) — CONTROL
# Top-2 probe arms per the 2026-07-02 GoldCoin sweep (350 > 200; 528 regressed).
#
# Headline: does any probe checkpoint beat v9-ckpt100 on the frontier? Privacy =
# leakage.leakage_rate_among_parseable (LOWER better); utility =
# helpfulness.helpful_rate_among_parseable (higher); adjusted_leakage =
# leak-among-helpful (LOWER). Over-permit watch: the probe's late "say yes more"
# vignette drift may RAISE leakage at later checkpoints — a leakage regression at
# ckpt-350 vs 200 is signal, not noise. Full split (max_examples=0).
# See wiki/grpo_training_field_notes/2026-07-01_v11_probe_midrun_forensics.md.

set -euo pipefail

PROJECT_ROOT=/share/pierson/matt/UAIR
source "$PROJECT_ROOT/.venv/bin/activate"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
cd "$PROJECT_ROOT"

# Pin the judge to the live server (also in server.env; set explicitly to be safe).
export JUDGE_SERVER_URL="http://klara:8002"

python -m dagspaces.privacylens.cli -m \
  pipeline=privacylens_clean \
  model=qwen3.5-9b/v11probe-ckpt350,qwen3.5-9b/v11probe-ckpt200,qwen3.5-9b/v9-ckpt100,qwen3.5-9b/v10-ckpt250 \
  experiment.name=privacylens_v11probe_vs_v9_v10
