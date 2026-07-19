#!/bin/bash
# CI-RL vignettes eval — the training-shaped benchmark (vignette completeness,
# GOLD-/programmatically-scored, judge-free) and the still-missing COLM main-table
# row for finetuned models. Nine arms in one matched batch (temp-0.2 sampling, so
# cross-arm comparison needs same-batch runs):
#   arm 0  sft-contentless-v6  SFT anchor (bridge to the 2026-06 base/SFT/v9 batch)
#   arm 1  v9-ckpt100          paper keeper (bridge / batch-stability check)
#   arm 2  v10-ckpt250         v10 best
#   arms 3-8  v11probe-ckpt{50,100,150,200,350,528}  full probe checkpoint curve
#
# The full probe curve is cheap here (one light 1-GPU inference per arm) and
# diagnostic: does CIRL completeness peak at ckpt-350 where GoldCoin does, and
# does the late over-permit vignette drift (says-yes-on-gold-no 0.01->0.07) show
# up at ckpt-528? Zero judge dependency -> no contention with the PrivacyLens
# sweep's klara:8002 calls.
# See wiki/grpo_training_field_notes/2026-07-01_v11_probe_midrun_forensics.md.

set -euo pipefail

PROJECT_ROOT=/share/pierson/matt/UAIR
source "${TRAWLER_DRIVER_VENV:-$PROJECT_ROOT/.venv-vllm025cu129}/bin/activate"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
cd "$PROJECT_ROOT"

python -m dagspaces.cirl_vignettes.cli -m \
  pipeline=cirl_vignettes_eval \
  model=qwen3.5-9b/sft-contentless-v6,qwen3.5-9b/v9-ckpt100,qwen3.5-9b/v10-ckpt250,qwen3.5-9b/v11probe-ckpt50,qwen3.5-9b/v11probe-ckpt100,qwen3.5-9b/v11probe-ckpt150,qwen3.5-9b/v11probe-ckpt200,qwen3.5-9b/v11probe-ckpt350,qwen3.5-9b/v11probe-ckpt528 \
  experiment.name=cirl_v11probe_ckpt_sweep
