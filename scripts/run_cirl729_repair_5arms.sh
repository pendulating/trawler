#!/usr/bin/env bash
# Requeue the 5 camera-ready CIRL-729 arms that hard-failed the strict
# format sanity gate before the allow_unreliable escape hatch was added
# to eval_cirl729_canonical_2026_07_22.yaml (Matt's 2026-07-22 ruling:
# format misses score -1 per paper protocol, cells populated).
set -uo pipefail
PROJECT_ROOT=/share/pierson/matt/UAIR
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
cd "$PROJECT_ROOT"
PYTHON="${TRAWLER_DRIVER_VENV:-$PROJECT_ROOT/.venv-vllm025cu129}/bin/python"
if [ -n "${WAIT_FOR_JOB:-}" ]; then
  echo "[cirl-repair] waiting for job ${WAIT_FOR_JOB} to drain..."
  empty=0
  while [ "${empty}" -lt 3 ]; do
    if squeue -h -j "${WAIT_FOR_JOB}" 2>/dev/null | grep -q .; then empty=0
    else empty=$((empty + 1)); echo "[cirl-repair] queue empty (${empty}/3) at $(date)"; fi
    sleep 300
  done
fi
"$PYTHON" -m dagspaces.eval_all.cli --multirun \
  +sweep=eval_cirl729_canonical_2026_07_22 \
  "model=qwen3.5-2b/instruct,qwen3.5-4b/instruct,qwen3.5-9b/instruct,openthinker3-7b/instruct,llama3.1-8b/instruct"
echo "[cirl-repair] finished rc=$? at $(date)"
