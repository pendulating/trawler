#!/usr/bin/env bash
# PrivacyLens variance instrument: the RL quartet, 3 seeds each. See
# dagspaces/eval_all/conf/sweep/eval_pl_variance_n3_2026_08_07.yaml for the
# design and for why the fixed-seed component is estimated from the two
# existing 777-seeded runs rather than swept here.
#
# Produces the noise floor the camera-ready variance gate is missing for
# PrivacyLens — currently its three columns render as unmeasurable because the
# judge-free instrument deliberately excludes judged benchmarks.

set -euo pipefail

PROJECT_ROOT=/share/pierson/matt/UAIR
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
cd "$PROJECT_ROOT"

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

# A pre-2026-08-07 judge truncates 4.6-9.8% of leakage calls on unterminated
# guided JSON. That would show up as variance this sweep is not measuring, so
# refuse to launch against one rather than silently contaminate the floor.
judge_log=$(ls -t "$PROJECT_ROOT"/.slurm_jobs/judge-server/*.out 2>/dev/null | head -1 || true)
if [ -n "$judge_log" ] && grep -q "disable_any_whitespace" "$judge_log"; then
    echo "[preflight] judge OK: ${EXPECTED_JUDGE}, whitespace guard ON"
else
    echo "ERROR: could not confirm disable_any_whitespace on the running judge." >&2
    echo "       Checked: ${judge_log:-<no judge-server log found>}" >&2
    echo "       Truncation noise would contaminate the variance estimate." >&2
    echo "       Restart: scancel <judge job>; sbatch scripts/judge_server.sub" >&2
    if [ "${FORCE_UNVERIFIED_JUDGE:-0}" != "1" ]; then exit 1; fi
fi

# Absolute driver python, NOT shell activation: submitit bakes the monitor
# srun python from the driver's sys.executable (memory feedback_sweep_driver_venv).
#
# NOTE: run this under nohup/tmux, not a foreground timeout — a 12-cell sweep
# at 3-wide runs ~3-3.5 h and killing the driver mid-flight loses the
# aggregate result (the SLURM monitors survive, but you lose the wait).
# Extra overrides are forwarded, so a second block of seeds runs from the same
# script and the same design:
#   SLURM_BEGIN=2026-08-08T01:30:00 bash scripts/run_eval_pl_variance_n3.sh \
#       variance_seed=104,105,106
# Use DISTINCT seeds for a second block: repeating 101/102/103 yields n=2 per
# seed (an engine-nondeterminism estimate), not n=6 independent draws of the
# total noise. SLURM_BEGIN serializes the blocks — 6 concurrent cells would put
# ~96 requests in flight against a judge whose observed capacity is ~55.
exec "${TRAWLER_DRIVER_VENV:-$PROJECT_ROOT/.venv-vllm025cu129}/bin/python" \
  -m dagspaces.eval_all.cli --multirun \
  +sweep=eval_pl_variance_n3_2026_08_07 \
  +benchmark_filter=privacylens_only \
  "$@"
