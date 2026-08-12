#!/usr/bin/env bash
# PrivacyLens re-run for the RL quartet on the fixed judge stack. See
# dagspaces/eval_all/conf/sweep/eval_pl_quartet_clean_2026_08_07.yaml for what
# was broken, what a re-run fixes, and what it deliberately does not.
#
# privacylens_only: the other benchmarks in the 2026-08-04 batch are unaffected
# by the judge fixes (only PrivacyLens uses the leakage/helpfulness judges) and
# re-running them would burn GPU hours to reproduce identical numbers.

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

# The whole point of this re-run is a judge that does not truncate. A server
# started from the PRE-2026-08-07 .sub has no disable_any_whitespace and will
# silently reintroduce 4.6-9.8% truncated leakage calls, so verify the running
# server actually carries the flag rather than trusting that someone restarted
# it. The launch log is the only place this is observable from outside.
judge_log=$(ls -t "$PROJECT_ROOT"/.slurm_jobs/judge-server/*.out 2>/dev/null | head -1 || true)
if [ -n "$judge_log" ] && grep -q "disable_any_whitespace" "$judge_log"; then
    echo "[preflight] judge OK: ${EXPECTED_JUDGE}, whitespace guard ON"
else
    echo "WARNING: could not confirm disable_any_whitespace on the running judge." >&2
    echo "         Checked: ${judge_log:-<no judge-server log found>}" >&2
    echo "         If this server predates 2026-08-07, restart it:" >&2
    echo "           scancel <judge job>; sbatch scripts/judge_server.sub" >&2
    if [ "${FORCE_UNVERIFIED_JUDGE:-0}" != "1" ]; then
        echo "         Refusing to launch. Set FORCE_UNVERIFIED_JUDGE=1 to override." >&2
        exit 1
    fi
fi

# Absolute driver python, NOT shell activation: submitit bakes the monitor
# srun python from the driver's sys.executable (memory feedback_sweep_driver_venv
# / the 2026-07-17 CPU-fallback trap).
exec "${TRAWLER_DRIVER_VENV:-$PROJECT_ROOT/.venv-vllm025cu129}/bin/python" \
  -m dagspaces.eval_all.cli --multirun \
  +sweep=eval_pl_quartet_clean_2026_08_07 \
  +benchmark_filter=privacylens_only
