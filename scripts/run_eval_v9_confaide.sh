#!/bin/bash
# Judge-free ConfAIDE eval of v9 GRPO checkpoint-100 vs the SFT base.
# Entire confaide_eval pipeline is judge-free:
#   tier 2a / 2b -> Pearson r between model Likert ratings and human ground truth
#   tier 3 control -> rejection accuracy; 3_free/info/sharing -> rule-based string-match
# (grep 'judge' over dagspaces/confaide/ returns nothing). Zero contention with
# the live v9 run's judge:8002 calls.
#
# Two arms, both on the v9 run's _merged_sft text base, isolating the GRPO LoRA:
#   arm 0  sft-contentless-v6   _merged_sft, NO adapter (== GRPO step 0)
#   arm 1  v9-ckpt100           _merged_sft + checkpoint-100 GRPO LoRA (epoch ~0.58)
# (reuses the common-conf qwen3.5-9b text configs from the GoldCoin eval.)
#
# Headline: tier2a_pearson / tier2b_pearson for v9-ckpt100 vs SFT — does CI
# fine-tuning improve agreement with human contextual-privacy expectations?
# ConfAIDE tier 2/3 prompts are short; each branch is light, well under an hour
# on a free klara A6000 (6 branches x 2 arms fan out, SLURM-queued behind the
# live train + geoprivacy jobs).

set -euo pipefail

PROJECT_ROOT=/share/pierson/matt/UAIR
source "${TRAWLER_DRIVER_VENV:-$PROJECT_ROOT/.venv-vllm025cu129}/bin/activate"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
cd "$PROJECT_ROOT"

python -m dagspaces.confaide.cli -m \
  pipeline=confaide_eval \
  model=qwen3.5-9b/sft-contentless-v6,qwen3.5-9b/v9-ckpt100 \
  experiment.name=confaide_v9_vs_sft
