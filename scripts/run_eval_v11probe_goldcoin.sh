#!/bin/bash
# Judge-free GoldCoin-HIPAA eval — the HELD-OUT test of the v11 lever-(a) probe
# (top100-universe judgment vignettes, realised 2.15:1 yes:no vs v10's 5.2:1;
# all else v10-identical). 7-arm matched sweep (temp 0.2 samples, one environment):
#   arm 0  sft-contentless-v6   _merged_sft-equivalent, NO GRPO adapter — anchor
#   arm 1  v11probe-ckpt50      epoch ~0.28
#   arm 2  v11probe-ckpt100     epoch ~0.57  (v9/v10 peak region)
#   arm 3  v11probe-ckpt150     epoch ~0.85
#   arm 4  v11probe-ckpt200     epoch ~1.14
#   arm 5  v11probe-ckpt350     epoch ~1.99  (v10 verdict-freeze point)
#   arm 6  v11probe-ckpt528     epoch 3.00   (final)
#
# Pre-registered (2026-07-01 mid-run forensics): Forbid recall expected to stay
# ~=0.55 (n=20 — weak evidence alone); judge the probe on compliance macro-F1
# (n=107) vs v10's best 0.755, plus ConfAIDE-2b (separate script). Over-permit
# watch: later checkpoints may score LOWER on Forbid. Judge-free (sklearn F1),
# no aux-server dependency.
# See wiki/grpo_training_field_notes/2026-07-01_v11_probe_midrun_forensics.md.

set -euo pipefail

PROJECT_ROOT=/share/pierson/matt/UAIR
source "${TRAWLER_DRIVER_VENV:-$PROJECT_ROOT/.venv-vllm025cu129}/bin/activate"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
cd "$PROJECT_ROOT"

python -m dagspaces.goldcoin_hipaa.cli -m \
  pipeline=full_eval \
  model=qwen3.5-9b/sft-contentless-v6,qwen3.5-9b/v11probe-ckpt50,qwen3.5-9b/v11probe-ckpt100,qwen3.5-9b/v11probe-ckpt150,qwen3.5-9b/v11probe-ckpt200,qwen3.5-9b/v11probe-ckpt350,qwen3.5-9b/v11probe-ckpt528 \
  experiment.name=goldcoin_v11probe_ckpt_sweep
