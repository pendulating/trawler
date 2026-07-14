#!/bin/bash
# Judge-free ConfAIDE eval of the v11 lever-(a) probe checkpoints vs the SFT base.
# ConfAIDE-2b (Pearson r vs human contextual-privacy expectations) is the
# PRE-REGISTERED metric lever (a) should move: the balanced vignette mix halted
# v10's gold-"no" verdict erosion, and 2b is the judgment-shaped held-out metric
# where GRPO regressed below SFT (63.2 vs 68.6). Trimmed arm set (peak candidates
# + late-drift checks); backfill ckpt-50/150 if the GoldCoin sweep peaks there:
#   arm 0  sft-contentless-v6   anchor (v9-era measurement: 2b = 68.6)
#   arm 1  v11probe-ckpt100     epoch ~0.57  (v9/v10 peak region)
#   arm 2  v11probe-ckpt200     epoch ~1.14
#   arm 3  v11probe-ckpt350     epoch ~1.99  (v10 verdict-freeze point)
#   arm 4  v11probe-ckpt528     epoch 3.00   (over-permit drift check:
#                               vignette says-yes-on-gold-no crept 0.01->0.07 late)
# Judge-free (Pearson r / rule-based), zero aux-server dependency.
# See wiki/grpo_training_field_notes/2026-07-01_v11_probe_midrun_forensics.md.

set -euo pipefail

PROJECT_ROOT=/share/pierson/matt/UAIR
source "$PROJECT_ROOT/.venv/bin/activate"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
cd "$PROJECT_ROOT"

python -m dagspaces.confaide.cli -m \
  pipeline=confaide_eval \
  model=qwen3.5-9b/sft-contentless-v6,qwen3.5-9b/v11probe-ckpt100,qwen3.5-9b/v11probe-ckpt200,qwen3.5-9b/v11probe-ckpt350,qwen3.5-9b/v11probe-ckpt528 \
  experiment.name=confaide_v11probe_ckpt_sweep
