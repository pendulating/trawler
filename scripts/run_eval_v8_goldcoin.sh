#!/bin/bash
# Ground-truth eval of the in-progress v8 GRPO run on GoldCoin-HIPAA — the
# JUDGE-FREE benchmark (sklearn accuracy/F1 on applicability + compliance; no
# LLM judge, so zero contention with the live v8 run's judge:8002 calls).
#
# Three arms, all sharing the v8 merged-SFT base so the comparison isolates the
# GRPO LoRA effect:
#   arm 0  sft-contentless-v6   the SFT base (== GRPO step 0; no adapter)
#   arm 1  v8-ckpt200           clean snapshot, epoch ~0.55, PRE entropy breakout
#   arm 2  v8-ckpt400           epoch ~1.1, POST breakout onset (entropy ~1.7)
#
# Question: did v8 move the policy on held-out GoldCoin, and is the abstention
# drop seen in the training traces real learning (visible already at ckpt200) or
# entropy noise (only at ckpt400)? See
# wiki/grpo_training_field_notes/2026-06-22_v8_plan.md (ground-truth lever).
#
# The already-saved checkpoint-{200,400} adapters are complete and immutable
# (the live trainer only writes new checkpoint-NNN dirs), so reading them does
# not interfere with job 725705. GoldCoin eval is light (~1.5k applicability +
# ~0.8k compliance cases, single generation each); each arm runs in well under
# an hour on a free klara A6000.

set -euo pipefail

PROJECT_ROOT=/share/pierson/matt/UAIR
source "${TRAWLER_DRIVER_VENV:-$PROJECT_ROOT/.venv-vllm025cu129}/bin/activate"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
cd "$PROJECT_ROOT"

python -m dagspaces.goldcoin_hipaa.cli -m \
  pipeline=full_eval \
  model=qwen3.5-9b/sft-contentless-v6,qwen3.5-9b/v8-ckpt200,qwen3.5-9b/v8-ckpt400 \
  experiment.name=goldcoin_v8_vs_sft
