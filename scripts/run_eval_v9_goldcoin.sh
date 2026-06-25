#!/bin/bash
# Ground-truth eval of the in-progress v9 GRPO run on GoldCoin-HIPAA — the
# JUDGE-FREE benchmark (sklearn accuracy/F1 on applicability + compliance; no
# LLM judge, so zero contention with the live v9 run's judge:8002 calls).
#
# THE DECISIVE v9 TEST. v8's ground-truth eval showed GRPO learned an
# indiscriminate "engage/permit" bias: GoldCoin compliance Forbid recall
# collapsed 0.70 -> 0.35 because the reward was one-directional. v9 makes the
# appropriateness DIRECTION a multiplier-with-floor (correct x1.0, hedge x0.7,
# wrong x0.4). Training traces already show the hedge fraction collapsing
# 73.5% -> ~6%. Question: does that translate to held-out Forbid recall
# recovering toward/above 0.70 while applicability holds ~0.97?
#
# Three arms, matched at near-identical training progress so the comparison is
# fair and isolates v8-vs-v9 reward design:
#   arm 0  sft-contentless-v6   the SFT base (== GRPO step 0; no adapter)
#   arm 1  v8-ckpt200           v8 clean snapshot, epoch ~0.55 (one-directional reward)
#   arm 2  v9-ckpt100           v9 snapshot,       epoch ~0.58 (two-sided directional reward)
#
# All three share the qwen3.5-9b SFT base (v8 and v9 _merged_sft are the same
# SFT checkpoint merged), so the comparison isolates the GRPO LoRA effect.
#
# The already-saved checkpoint-100 adapter is complete and immutable (the live
# trainer only writes new checkpoint-NNN dirs), so reading it does not interfere
# with job 810112. GoldCoin eval is light (~1.5k applicability + ~0.8k
# compliance cases, single generation each); each arm runs well under an hour on
# a free klara A6000 (judge:2 + embed:1 + v9-train:1 = 4 of 8 GPUs busy; 4 free).
#
# SUCCESS METRIC: compliance Forbid recall back toward/above 0.70 at v9-ckpt100
# (vs v8-ckpt200's collapse), with applicability F1 holding ~0.97.

set -euo pipefail

PROJECT_ROOT=/share/pierson/matt/UAIR
source "$PROJECT_ROOT/.venv/bin/activate"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
cd "$PROJECT_ROOT"

python -m dagspaces.goldcoin_hipaa.cli -m \
  pipeline=full_eval \
  model=qwen3.5-9b/sft-contentless-v6,qwen3.5-9b/v8-ckpt200,qwen3.5-9b/v9-ckpt100 \
  experiment.name=goldcoin_v9_vs_v8_vs_sft
