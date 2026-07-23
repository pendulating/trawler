#!/usr/bin/env bash
# Repair the 5 top-up arms whose servers timed out pending behind the
# 13:30 camera-ready sweep (2026-07-22): explicit roster, no re-selection
# (the auto-selector would wastefully re-select already-topped configs).
set -uo pipefail
PROJECT_ROOT=/share/pierson/matt/UAIR
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
cd "$PROJECT_ROOT"
PYTHON="${TRAWLER_DRIVER_VENV:-$PROJECT_ROOT/.venv-vllm025cu129}/bin/python"
if [ -n "${WAIT_FOR_JOB:-}" ]; then
  echo "[repair] waiting for job ${WAIT_FOR_JOB} to drain..."
  empty=0
  while [ "${empty}" -lt 3 ]; do
    if squeue -h -j "${WAIT_FOR_JOB}" 2>/dev/null | grep -q .; then empty=0
    else empty=$((empty + 1)); echo "[repair] queue empty (${empty}/3) at $(date)"; fi
    sleep 300
  done
fi
"$PYTHON" -m dagspaces.eval_all.cli --multirun \
  +sweep=eval_judgefree_variance_topup_goldcoin_2026_07_21 \
  "model=llama3.1-8b/instruct,llama3.1-8b/sft-canonical-ckpt171,openthinker3-7b/instruct,qwen3.5-2b/instruct,qwen3.5-2b/sft-canonical-ckpt171"
echo "[repair] finished rc=$? at $(date)"
