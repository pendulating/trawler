#!/usr/bin/env bash
# Repair the holes left in eval_rl_quartet_recovery by the 2026-08-04 16:39-16:50
# SLURM controller wobble. See
# dagspaces/eval_all/conf/sweep/eval_rl_quartet_repair_2026_08_04.yaml for what
# failed and why CIRL is re-run rather than salvaged.
#
# The repair list is DERIVED from the run's artifacts at launch time
# (scripts/repair_eval_all_run.py reading each cell's failures.json), not
# hardcoded — a list written before the sweep ends is stale by the time this
# runs, and a second wobble would put holes in different cells.
#
# One single-cell invocation per affected cell, run SEQUENTIALLY. This pass
# exists because the controller wedged under three concurrent monitors; it does
# not get to make that mistake again.
#
# Usage:
#   bash scripts/run_eval_quartet_repair.sh                 # plan, confirm, launch
#   bash scripts/run_eval_quartet_repair.sh --dry-run       # print the plan only
#   bash scripts/run_eval_quartet_repair.sh --run-dir <dir> # repair another run
#   bash scripts/run_eval_quartet_repair.sh --force         # skip the live-monitor gate

set -euo pipefail

PROJECT_ROOT=/share/pierson/matt/UAIR
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
cd "$PROJECT_ROOT"

RUN_DIR="${RUN_DIR:-$PROJECT_ROOT/multirun/2026-08-04_eval_rl_quartet_recovery/15-51-04}"
SWEEP="eval_rl_quartet_repair_2026_08_04"
MONITOR_NAME="eval_rl_quartet_recovery-monitor"
DRIVER="${TRAWLER_DRIVER_VENV:-$PROJECT_ROOT/.venv-vllm025cu129}/bin/python"
DRY_RUN=0
FORCE=0

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1 ;;
        --force)   FORCE=1 ;;
        --run-dir) RUN_DIR="$2"; shift ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
    shift
done

if [ ! -d "$RUN_DIR" ]; then
    echo "ERROR: run dir not found: $RUN_DIR" >&2
    exit 1
fi

# ── Gate 1: the run being repaired must be FINISHED ───────────────────────
# A cell with no failures.json is indistinguishable from disk between "still
# running" and "monitor was killed", so planning against a live run produces a
# plan that re-runs benchmarks currently in flight.
live=$(squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -c "$MONITOR_NAME" || true)
if [ "${live:-0}" -gt 0 ] && [ "$FORCE" -ne 1 ]; then
    echo "ERROR: $live '$MONITOR_NAME' job(s) still running." >&2
    echo "       The repair plan is derived from each cell's failures.json," >&2
    echo "       which the monitor writes on exit. Planning now would target" >&2
    echo "       benchmarks that are still in flight." >&2
    echo "       Wait for them to finish, or pass --force if you know better." >&2
    exit 1
fi

# ── Gate 2: judge health, same bar as the sweep being repaired ────────────
# Only matters if the derived plan includes privacylens, but checking is cheap
# and a wrong-judge cell is unusable in the camera-ready table.
export JUDGE_SERVER_URL="${JUDGE_SERVER_URL:-http://klara.tech.cornell.edu:8002}"
EXPECTED_JUDGE="${EXPECTED_JUDGE:-Gemma-4-31B-it}"
models_json=$(curl -sf --max-time 15 "${JUDGE_SERVER_URL}/v1/models" || true)
if [ -z "$models_json" ]; then
    echo "ERROR: judge server not reachable at ${JUDGE_SERVER_URL}" >&2
    echo "       launch it first:  sbatch scripts/judge_server.sub" >&2
    exit 1
fi
if ! printf '%s' "$models_json" | grep -q "$EXPECTED_JUDGE"; then
    echo "ERROR: judge at ${JUDGE_SERVER_URL} does not serve ${EXPECTED_JUDGE}." >&2
    printf '       /v1/models returned: %s\n' "$models_json" >&2
    exit 1
fi
echo "[preflight] judge OK: ${EXPECTED_JUDGE} served at ${JUDGE_SERVER_URL}"

# ── Derive the plan ───────────────────────────────────────────────────────
echo
"$DRIVER" -m scripts.repair_eval_all_run "$RUN_DIR"
echo

mapfile -t CMDS < <("$DRIVER" -m scripts.repair_eval_all_run "$RUN_DIR" \
                      --emit-cmds --sweep "$SWEEP" --driver "$DRIVER")

if [ "${#CMDS[@]}" -eq 0 ]; then
    echo "Nothing to repair. Exiting."
    exit 0
fi

if [ "$DRY_RUN" -eq 1 ]; then
    echo "--dry-run: would run ${#CMDS[@]} invocation(s), sequentially:"
    printf '  %s\n' "${CMDS[@]}"
    exit 0
fi

echo "About to run ${#CMDS[@]} invocation(s), sequentially:"
printf '  %s\n' "${CMDS[@]}"
echo
read -r -p "Proceed? [y/N] " reply
case "$reply" in
    [yY]*) ;;
    *) echo "Aborted."; exit 1 ;;
esac

rc_any=0
for cmd in "${CMDS[@]}"; do
    case "$cmd" in \#*) echo "$cmd"; continue ;; esac
    echo
    echo "============================================================"
    echo "LAUNCH: $cmd"
    echo "============================================================"
    # Deliberately not `exec`: each invocation blocks until its cell finishes,
    # and the next one starts only then.
    if ! eval "$cmd"; then
        echo "WARNING: invocation returned non-zero — continuing with the rest." >&2
        rc_any=1
    fi
done

echo
echo "Repair pass complete. Re-check with:"
echo "  $DRIVER -m scripts.repair_eval_all_run $RUN_DIR"
exit "$rc_any"
