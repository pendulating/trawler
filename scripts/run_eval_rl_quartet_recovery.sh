#!/usr/bin/env bash
# Recovery sweep: the four-way RL comparison (Instruct / SFT / K-VERDICT /
# M2-FULL) re-measured in ONE batch on the full benchmark suite. See
# dagspaces/eval_all/conf/sweep/eval_rl_quartet_recovery_2026_08_04.yaml for
# which cells were drifted vs missing and why re-running the two complete ones
# is the point rather than waste.
#
# NO +benchmark_filter here, deliberately: `ci_only` is what cost k3-verdict
# its MMLU column on 2026-08-03.
#
# Judge server on klara:8002 must be up AND serving Gemma-4-31B-it before
# launch — PrivacyLens leakage/helpfulness route through it, and the
# camera-ready notebook rejects any judged cell whose manifest attests a
# different judge.

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

# The served name is what judge_export.py attests into every judge-batch
# manifest. Checking it here turns a silent wrong-judge sweep into a
# pre-launch failure.
if ! printf '%s' "$models_json" | grep -q "$EXPECTED_JUDGE"; then
    echo "ERROR: judge at ${JUDGE_SERVER_URL} does not serve ${EXPECTED_JUDGE}." >&2
    echo "       /v1/models returned:" >&2
    printf '       %s\n' "$models_json" >&2
    echo "       PrivacyLens cells from this sweep would be unusable in the" >&2
    echo "       camera-ready table. Fix the judge server, then relaunch." >&2
    exit 1
fi
echo "[preflight] judge OK: ${EXPECTED_JUDGE} served at ${JUDGE_SERVER_URL}"

# Absolute driver python, NOT shell activation: submitit bakes the monitor
# srun python from the driver's sys.executable (memory feedback_sweep_driver_venv
# / the 2026-07-17 CPU-fallback trap).
exec "${TRAWLER_DRIVER_VENV:-$PROJECT_ROOT/.venv-vllm025cu129}/bin/python" \
  -m dagspaces.eval_all.cli --multirun \
  +sweep=eval_rl_quartet_recovery_2026_08_04
