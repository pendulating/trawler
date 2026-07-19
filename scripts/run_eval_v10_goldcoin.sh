#!/bin/bash
# Judge-free GoldCoin-HIPAA eval — the HELD-OUT test of whether v10's
# cost-sensitive floor lifts compliance Forbid recall. 3-arm matched sweep (temp
# 0.2 samples, so all arms run in ONE environment):
#   arm 0  sft-contentless-v6   _merged_sft, NO adapter (== GRPO step 0) — anchor
#   arm 1  v9-ckpt100           v9 symmetric multiplier (epoch ~0.58)
#   arm 2  v10-ckpt100          v10 cost-sensitive floor=0.1 (epoch ~0.58)
#
# Headline: does v10-ckpt100 raise Forbid recall above v9-ckpt100's 0.55 toward
# SFT 0.65, while holding Permit recall / applicability? The v10 trace signal at
# this checkpoint is Forbid commit-accuracy 30%->50% (Permit held); this checks
# whether that transfers to GoldCoin. Judge-free (sklearn F1) so zero contention
# with the live v10 training run's judge calls.

set -euo pipefail

PROJECT_ROOT=/share/pierson/matt/UAIR
source "${TRAWLER_DRIVER_VENV:-$PROJECT_ROOT/.venv-vllm025cu129}/bin/activate"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
cd "$PROJECT_ROOT"

python -m dagspaces.goldcoin_hipaa.cli -m \
  pipeline=full_eval \
  model=qwen3.5-9b/sft-contentless-v6,qwen3.5-9b/v9-ckpt100,qwen3.5-9b/v10-ckpt100 \
  experiment.name=goldcoin_v10_vs_v9_vs_sft
