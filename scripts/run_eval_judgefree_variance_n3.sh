#!/usr/bin/env bash
# Launch the judge-free variance run: a single-cell server-mode SMOKE TEST,
# then the 117-cell server-mode sweep, then the 12-cell in-process gpt-oss
# companion. See the sweep yaml headers for the design:
#   dagspaces/eval_all/conf/sweep/eval_judgefree_variance_n3_2026_07_20.yaml
#   dagspaces/eval_all/conf/sweep/eval_judgefree_variance_n3_gptoss_2026_07_20.yaml
#
# The smoke test exists because server_mode was repaired 2026-07-19 but has
# never been deployed at sweep scale. It runs ONE cell (a LoRA checkpoint —
# the risky path: adapter key-remap + --enable-lora serving) over the two
# fastest benchmarks with sample_n=25, and the big sweep only launches if
# it exits clean AND produced metrics.
#
# Optionally gate on another SLURM job/array draining first:
#   WAIT_FOR_JOB=144832 nohup scripts/run_eval_judgefree_variance_n3.sh \
#       > logs/eval_judgefree_variance_n3.log 2>&1 &
#
# No judge server needed at any stage.

set -uo pipefail

PROJECT_ROOT=/share/pierson/matt/UAIR
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
cd "$PROJECT_ROOT"

# Absolute driver python, NOT shell activation: submitit bakes the monitor
# srun python from the driver's sys.executable (the 2026-07-17
# CPU-fallback trap).
PYTHON="${TRAWLER_DRIVER_VENV:-$PROJECT_ROOT/.venv-vllm025cu129}/bin/python"

# ── Phase 0: wait for a prior job/array to drain ──────────────────────────
# squeue here is an ssh shim to the login node; a transient ssh failure
# prints nothing, exactly like a drained queue. Require 3 consecutive empty
# polls (15 min stable) so a hiccup can't fire the launch early.
if [ -n "${WAIT_FOR_JOB:-}" ]; then
  echo "[variance] waiting for job ${WAIT_FOR_JOB} to drain..."
  empty=0
  while [ "${empty}" -lt 3 ]; do
    if squeue -h -j "${WAIT_FOR_JOB}" 2>/dev/null | grep -q .; then
      empty=0
    else
      empty=$((empty + 1))
      echo "[variance] queue empty for ${WAIT_FOR_JOB} (${empty}/3 checks) at $(date)"
    fi
    sleep 300
  done
  echo "[variance] job ${WAIT_FOR_JOB} drained; starting smoke test at $(date)"
fi

# ── Phase 1: server-mode smoke test (one cell, LoRA model, sampled) ───────
SMOKE_NAME=judgefree_variance_smoke
# cirl included: the 2026-07-21 benchmark swap rewired it (new dagspace,
# new key, CIRL-729 deterministic scoring) — smoke the new wiring through
# eval_all + server mode before committing 117 cells to it.
"$PYTHON" -m dagspaces.eval_all.cli \
  model=qwen3.5-2b/sft-canonical-ckpt171 \
  server_mode.enabled=true \
  judge_sidecar.enabled=false \
  'benchmark_filter.include=[goldcoin,confaide,cirl]' \
  benchmarks.confaide.pipeline=confaide_tier2_only \
  '+benchmarks.cirl.extra_args=["+runtime.allow_unreliable_metrics=true"]' \
  runtime.sample_n=25 \
  experiment.name="$SMOKE_NAME"
smoke_rc=$?

SMOKE_DIR=$(ls -td outputs/*_"$SMOKE_NAME"/*/ 2>/dev/null | head -1)
n_metrics=$(find "${SMOKE_DIR:-/nonexistent}" -name metrics.json 2>/dev/null | wc -l)
echo "[variance] smoke rc=${smoke_rc}, metrics.json files=${n_metrics} in ${SMOKE_DIR:-<none>}"

# goldcoin emits 2 metrics.json (applicability+compliance), confaide tier2
# emits 2 (tier2a+tier2b), cirl emits 1 — require at least 4 so one
# missing file is tolerated but a dead server (0 metrics) aborts.
if [ "$smoke_rc" -ne 0 ] || [ "$n_metrics" -lt 4 ]; then
  echo "[variance] SMOKE TEST FAILED — NOT launching the 117-cell sweep."
  echo "[variance] Inspect ${SMOKE_DIR:-outputs/*_${SMOKE_NAME}/} and re-run manually."
  exit 1
fi
echo "[variance] smoke test passed; launching server-mode sweep at $(date)"

# ── Phase 2: server-mode sweep (39 models x 3 seeds, 5-wide) ─────────────
"$PYTHON" -m dagspaces.eval_all.cli --multirun \
  +sweep=eval_judgefree_variance_n3_2026_07_20
main_rc=$?
echo "[variance] server-mode sweep finished rc=${main_rc} at $(date)"

# ── Phase 3: gpt-oss companion (in-process, 4 models x 3 seeds) ──────────
# Runs even if some main-sweep arms failed (their cells are independent).
"$PYTHON" -m dagspaces.eval_all.cli --multirun \
  +sweep=eval_judgefree_variance_n3_gptoss_2026_07_20
gptoss_rc=$?
echo "[variance] gpt-oss companion finished rc=${gptoss_rc} at $(date)"

exit $(( main_rc || gptoss_rc ))
