#!/bin/bash
# Judge-free VLM-GeoPrivacy MCQ eval of v9 GRPO checkpoint-100 vs the SFT base.
# This is the remaining judge-free benchmark (privacylens needs an LLM judge;
# goldcoin already done; vlm_geoprivacy freeform_eval needs a judge — mcq_eval
# is sklearn MCQ accuracy, no judge, so zero contention with the live v9 run's
# judge:8002 calls).
#
# Two arms, both on the v9 run's _merged_sft base (vision encoder intact,
# model_type qwen3_5_vision), so the comparison isolates the GRPO LoRA effect:
#   arm 0  sft-contentless-v6   _merged_sft, NO adapter (== GRPO step 0)
#   arm 1  v9-ckpt100           _merged_sft + checkpoint-100 GRPO LoRA (epoch ~0.58)
#
# Question: does v9's text-CI fine-tuning transfer to (or harm) visual-geoprivacy
# CI choices? The checkpoint-100 base+adapter pair already loaded cleanly in the
# GoldCoin eval; the LoRA touches language layers only, vision tower untouched.
# MCQ inference is light (single generation per item); each arm well under an
# hour on a free klara A6000.

set -euo pipefail

PROJECT_ROOT=/share/pierson/matt/UAIR
source "${TRAWLER_DRIVER_VENV:-$PROJECT_ROOT/.venv-vllm025cu129}/bin/activate"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
cd "$PROJECT_ROOT"

python -m dagspaces.vlm_geoprivacy_bench.cli -m \
  pipeline=mcq_eval \
  model=qwen3.5-9b/sft-contentless-v6,qwen3.5-9b/v9-ckpt100 \
  experiment.name=geoprivacy_v9_vs_sft
