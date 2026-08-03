#!/usr/bin/env bash
# K4 (revived): external CI-benchmark eval of the four k-series arms + the
# merged SFT base they started from. See
# dagspaces/eval_all/conf/sweep/k3_arms_ci_2026_08_03.yaml for the design
# rationale and the comparison discipline, and wiki §17 of
# 2026-07-31_kto_plan.md for why the original K4 NO-GO does not stand.
#
# Judge server on klara:8002 must be up (check /v1/models) before launch —
# PrivacyLens leakage/helpfulness route through it.

set -euo pipefail

PROJECT_ROOT=/share/pierson/matt/UAIR
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
cd "$PROJECT_ROOT"

export JUDGE_SERVER_URL="${JUDGE_SERVER_URL:-http://klara.tech.cornell.edu:8002}"

if ! curl -sf "${JUDGE_SERVER_URL}/v1/models" >/dev/null 2>&1; then
    echo "ERROR: judge server not reachable at ${JUDGE_SERVER_URL}" >&2
    echo "       launch it first:  sbatch scripts/judge_server.sub" >&2
    exit 1
fi

# Absolute driver python, NOT shell activation: submitit bakes the monitor
# srun python from the driver's sys.executable (memory feedback_sweep_driver_venv
# / the 2026-07-17 CPU-fallback trap).
exec "${TRAWLER_DRIVER_VENV:-$PROJECT_ROOT/.venv-vllm025cu129}/bin/python" \
  -m dagspaces.eval_all.cli --multirun \
  +sweep=k3_arms_ci_2026_08_03 \
  +benchmark_filter=ci_only
