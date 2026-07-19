#!/bin/bash
# JUDGE-BASED PrivacyLens eval of the top-3 v10 GRPO checkpoints vs the v9 keeper.
# Leakage + helpfulness are scored by the external judge server (Qwen3.6-27B @
# klara:8002, job 725605 — the SAME server instance that judged the v9 sweep on
# 2026-06-24, so the judge environment is identical to the baseline). v9 training
# is long done so klara GPUs are free for per-arm qa_probe + agent_action inference.
#
# Four arms on the SAME Qwen3.5-9B contentless-v6 SFT base, advancing only the LoRA:
#   arm 0  v10-ckpt100   v10 GRPO LoRA, epoch ~0.58  (GoldCoin comp-F1 0.710)
#   arm 1  v10-ckpt200   v10 GRPO LoRA, epoch ~1.16  (GoldCoin comp-F1 0.744)
#   arm 2  v10-ckpt250   v10 GRPO LoRA, epoch ~1.45  (GoldCoin comp-F1 0.755, best)
#   arm 3  v9-ckpt100    v9 GRPO LoRA, epoch ~0.58   (the paper keeper) — CONTROL
# The top-3 v10 checkpoints are the three best on the 2026-06-26 GoldCoin sweep
# (250 > 200 > 100 by compliance macro-F1; 350/500 regressed and are excluded).
# v9-ckpt100 is co-run (not just compared to stored numbers) so the LLM-judge state
# is byte-identical across all four arms — the v9-vs-SFT leakage margin was ~1pp at
# n=493, so same-batch judging is required for a trustworthy head-to-head.
#
# Headline: does any v10 checkpoint BEAT v9-ckpt100 on the privacy-utility frontier?
# Privacy metric = leakage.leakage_rate_among_parseable (LOWER better); utility =
# helpfulness.helpful_rate_among_parseable (higher better); adjusted_leakage =
# leak-among-helpful (LOWER better); qa_probing.accuracy = secondary probe. A win =
# lower leakage at comparable-or-better helpfulness. Full split (max_examples=0).

set -euo pipefail

PROJECT_ROOT=/share/pierson/matt/UAIR
source "${TRAWLER_DRIVER_VENV:-$PROJECT_ROOT/.venv-vllm025cu129}/bin/activate"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
cd "$PROJECT_ROOT"

# Pin the judge to the live server (also in server.env; set explicitly to be safe).
export JUDGE_SERVER_URL="http://klara:8002"

python -m dagspaces.privacylens.cli -m \
  pipeline=privacylens_clean \
  model=qwen3.5-9b/v10-ckpt100,qwen3.5-9b/v10-ckpt200,qwen3.5-9b/v10-ckpt250,qwen3.5-9b/v9-ckpt100 \
  experiment.name=privacylens_v10_ckpt_sweep
