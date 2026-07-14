#!/usr/bin/env bash
# Overnight chain, queued 2026-07-12: fiction10 flows -> top100 norms -> top100 flows.
#
# Sequential by necessity, not preference: klara has 8 GPUs, ~3 are held by another
# project, and each pipeline stage takes 4 -- so two of these can never run at once.
# Running them concurrently would just have SLURM queue one behind the other anyway,
# while risking two Hydra drivers racing on the same GPUs.
#
# Each run keeps going if the previous one fails, so one bad pipeline does not eat
# the whole night. Check STATUS at the bottom of the log in the morning.
#
# Launch (detached, survives logout):
#   setsid nohup bash scripts/run_overnight_gemma4_chain.sh > /dev/null 2>&1 &

set -u
cd /share/pierson/matt/UAIR
source .venv/bin/activate 2>/dev/null || true

LOGDIR=/share/pierson/matt/UAIR/.bench/overnight_2026_07_12
mkdir -p "$LOGDIR"
MASTER="$LOGDIR/chain.log"

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$MASTER"; }

# The fiction10 NORMS extraction stage is still finishing and holds the 4 GPUs.
# Wait it out rather than piling a second driver on top of it.
log "waiting for in-flight fiction10 norms extraction to clear the queue..."
while squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -qi "HNORMS-extraction"; do
  sleep 120
done
log "GPUs clear. starting chain."

run() {
  local name=$1; shift
  local t0=$SECONDS
  log "START  $name"
  if "$@" >"$LOGDIR/$name.log" 2>&1; then
    log "DONE   $name  ($(( (SECONDS-t0)/60 )) min)"
    echo "$name OK" >> "$LOGDIR/STATUS"
  else
    log "FAILED $name  ($(( (SECONDS-t0)/60 )) min) -- see $LOGDIR/$name.log"
    echo "$name FAILED" >> "$LOGDIR/STATUS"
  fi
}

# Ordered by value: fiction10 flows completes the corpus that norms was just validated
# on, so it is the one that unblocks paper-facing work first. top100 is bulk.
run fiction10_flows bash scripts/run_extract_fiction10_flows_gemma4.sh
run top100_norms    bash scripts/run_extract_top100_norms_gemma4.sh
run top100_flows    bash scripts/run_extract_top100_flows_gemma4.sh

log "CHAIN COMPLETE"
log "summary:"
cat "$LOGDIR/STATUS" | tee -a "$MASTER"
