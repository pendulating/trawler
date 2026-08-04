#!/usr/bin/env bash
# Backfill CIRL-729 (paper protocol) + MMLU for the five k-series cells — the
# two columns the main K4 sweep left empty. See
# dagspaces/eval_all/conf/sweep/k3_arms_cirl_mmlu_backfill_2026_08_03.yaml.
#
# Neither benchmark is judged, so the judge server is not required; it is
# preflighted anyway only if already configured, to keep one launch pattern.

set -euo pipefail

PROJECT_ROOT=/share/pierson/matt/UAIR
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
cd "$PROJECT_ROOT"

# Absolute driver python, NOT shell activation: submitit bakes the monitor
# srun python from the driver's sys.executable (memory feedback_sweep_driver_venv).
exec "${TRAWLER_DRIVER_VENV:-$PROJECT_ROOT/.venv-vllm025cu129}/bin/python" \
  -m dagspaces.eval_all.cli --multirun \
  +sweep=k3_arms_cirl_mmlu_backfill_2026_08_03
