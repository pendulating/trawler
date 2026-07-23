#!/usr/bin/env bash
# Relaunch arms 21-31 of the 2026-07-19 per-checkpoint SFT eval after the
# klara job-launch wedge (see the header of
# dagspaces/eval_all/conf/sweep/eval_sft_per_checkpoint_restart21_2026_07_20.yaml).
#
# Protocol-identical to the original sweep; roster = models 18-19 + 21-31
# (openthinker3-7b ckpt342/513 rerun whole, llama3.1-8b ckpt342/513,
# harc-llama3.1-8b x3, phi-4 x3, gpt-oss-20b x3).
# Combined record = 22-48-47 arms 0-17 + 20 + this run.
#
# Judge server on klara:8002 must be up (check /v1/models) before launch.

set -euo pipefail

PROJECT_ROOT=/share/pierson/matt/UAIR
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
cd "$PROJECT_ROOT"

export JUDGE_SERVER_URL="http://klara.tech.cornell.edu:8002"

# Absolute driver python, NOT shell activation: submitit bakes the monitor
# srun python from the driver's sys.executable (see
# memory feedback_sweep_driver_venv / the 2026-07-17 CPU-fallback trap).
exec "${TRAWLER_DRIVER_VENV:-$PROJECT_ROOT/.venv-vllm025cu129}/bin/python" \
  -m dagspaces.eval_all.cli --multirun \
  +sweep=eval_sft_per_checkpoint_restart21_2026_07_20
