#!/usr/bin/env bash
# Overnight batch 2026-07-22 (Matt-approved, all four items):
#   1. gpt-oss GoldCoin variance top-up (4 in-process arms, seeds 104-108)
#   2. ConfAIde Option-B repair (6 arms, escape hatch, pools into variance)
#   3. gemma-4-12b ckpt513 variance N=3 cell (3 server-mode arms)
#   4. gpt-oss ckpt258 seed102 cirl rep requeue (1 arm; original was
#      externally scancelled mid-engine-load 16:55, environmental)
#
# Gate: waits until the WHOLE pierson queue (this user, minus the
# persistent judge-server) is empty for 3 consecutive polls — stronger
# than a single-job gate because several repair arrays with unknown ids
# (cirl729 repair, GC top-up repair) fire tonight, and phase 3's servers
# must not sit PENDING into the 900s health timeout (the 2026-07-22
# top-up lesson). Phases run sequentially to avoid self-contention.

set -uo pipefail

PROJECT_ROOT=/share/pierson/matt/UAIR
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
cd "$PROJECT_ROOT"
PYTHON="${TRAWLER_DRIVER_VENV:-$PROJECT_ROOT/.venv-vllm025cu129}/bin/python"

_queue_busy() {
  squeue -h -u mwf62 -p pierson -o "%j" 2>/dev/null | grep -v "^judge-server$" | grep -q .
}

echo "[overnight] waiting for the pierson queue to drain (judge-server exempt)..."
empty=0
while [ "${empty}" -lt 3 ]; do
  if _queue_busy; then
    empty=0
  else
    empty=$((empty + 1))
    echo "[overnight] queue quiet (${empty}/3) at $(date)"
  fi
  sleep 300
done
echo "[overnight] queue drained; starting batch at $(date)"

echo "[overnight] phase 1/4: gpt-oss GoldCoin top-up"
"$PYTHON" -m dagspaces.eval_all.cli --multirun \
  +sweep=eval_judgefree_variance_topup_gptoss_gc_2026_07_22
echo "[overnight] phase 1 rc=$? at $(date)"

echo "[overnight] phase 2/4: ConfAIde Option-B repair"
"$PYTHON" -m dagspaces.eval_all.cli --multirun \
  +sweep=eval_judgefree_variance_confaide_repair_2026_07_22
echo "[overnight] phase 2 rc=$? at $(date)"

echo "[overnight] phase 3/4: gemma-4-12b ckpt513 variance N=3"
"$PYTHON" -m dagspaces.eval_all.cli --multirun \
  +sweep=eval_judgefree_variance_n3_12b513_2026_07_22
echo "[overnight] phase 3 rc=$? at $(date)"

echo "[overnight] phase 4/4: gpt-oss ckpt258 cirl rep requeue"
"$PYTHON" -m dagspaces.eval_all.cli --multirun \
  +sweep=eval_judgefree_variance_gptoss_cirl_rep_2026_07_22
echo "[overnight] phase 4 rc=$? at $(date)"

echo "[overnight] batch complete at $(date)"
