#!/usr/bin/env bash
# Recovery launcher for the gpt-oss companion half of the judge-free
# variance sweep. The original chain
# (scripts/run_eval_judgefree_variance_n3.sh, phases 2→3) died when the
# terminal's SLURM session job recycled 2026-07-22 — the main array
# (175833) is submitit-managed and kept running, but the process that
# would have chained the companion did not survive. This script re-arms
# just phase 3.
#
# Usage:
#   WAIT_FOR_JOB=175833 nohup scripts/run_variance_companion_after.sh \
#       > logs/eval_judgefree_variance_gptoss.log 2>&1 &

set -uo pipefail

PROJECT_ROOT=/share/pierson/matt/UAIR
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
cd "$PROJECT_ROOT"

PYTHON="${TRAWLER_DRIVER_VENV:-$PROJECT_ROOT/.venv-vllm025cu129}/bin/python"

# squeue is an ssh shim: require 3 consecutive empty polls so a transient
# failure can't fire the launch early.
if [ -n "${WAIT_FOR_JOB:-}" ]; then
  echo "[companion] waiting for job ${WAIT_FOR_JOB} to drain..."
  empty=0
  while [ "${empty}" -lt 3 ]; do
    if squeue -h -j "${WAIT_FOR_JOB}" 2>/dev/null | grep -q .; then
      empty=0
    else
      empty=$((empty + 1))
      echo "[companion] queue empty for ${WAIT_FOR_JOB} (${empty}/3) at $(date)"
    fi
    sleep 300
  done
  echo "[companion] job ${WAIT_FOR_JOB} drained at $(date)"
fi

"$PYTHON" -m dagspaces.eval_all.cli --multirun \
  +sweep=eval_judgefree_variance_n3_gptoss_2026_07_20
rc=$?
echo "[companion] gpt-oss companion finished rc=${rc} at $(date)"
exit $rc
