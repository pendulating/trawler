#!/bin/bash
# Judge-free GoldCoin-HIPAA eval placing v9 checkpoint-200 (epoch ~1.16) on the
# compliance trajectory. GoldCoin full_eval is judge-free (sklearn F1 over the
# Permit/Forbid labels), so zero contention with the live v9 run's judge:8002.
#
# GoldCoin generation is temperature 0.2 (samples), so all arms run in ONE sweep
# to share the sampling environment — the ckpt-100 -> ckpt-200 delta is then
# free of cross-run noise:
#   arm 0  sft-contentless-v6   _merged_sft, NO adapter (== GRPO step 0) — Forbid-recall anchor
#   arm 1  v9-ckpt100           _merged_sft + checkpoint-100 LoRA (epoch ~0.58)
#   arm 2  v9-ckpt200           _merged_sft + checkpoint-200 LoRA (epoch ~1.16)
#
# Headline: does the compliance Forbid recall keep climbing toward the SFT 0.65
# baseline at ckpt-200 (was sft 0.65 -> v8 0.35 -> v9-ckpt100 0.55), or does the
# epoch-2 entropy drift (entropy 0.65->0.86, IS_mean 0.984->0.969 by step ~290)
# start to cost held-out behavior? This is the comparison that decides
# run-to-completion vs keep-ckpt-200.

set -euo pipefail

PROJECT_ROOT=/share/pierson/matt/UAIR
source "$PROJECT_ROOT/.venv/bin/activate"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
cd "$PROJECT_ROOT"

python -m dagspaces.goldcoin_hipaa.cli -m \
  pipeline=full_eval \
  model=qwen3.5-9b/sft-contentless-v6,qwen3.5-9b/v9-ckpt100,qwen3.5-9b/v9-ckpt200 \
  experiment.name=goldcoin_v9_ckpt200_traj
