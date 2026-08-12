#!/usr/bin/env bash
# m2 wave-A GRPO arms on external gold — full benchmark suite.
#
# The m2 HOLD was decided by the internal reward instrument, whose
# appropriateness gold agrees with the teacher's `ci_appropriateness` at
# kappa 0.053 (wiki §17 of 2026-07-31_kto_plan.md). External benchmarks do not
# touch that gold. See
# dagspaces/eval_all/conf/sweep/m2_arms_all_2026_08_03.yaml for the cell
# design, the selection hazard on the 9-checkpoint `core` trajectory, and the
# comparison discipline (compare to k3-base IN-SWEEP, never to the paper's v9).
#
# NO benchmark_filter: this runs the full all_benchmarks set (goldcoin,
# privacylens, cirl, confaide, vlm_geoprivacy, mmlu). Judge server on
# klara:8002 must be up — PrivacyLens leakage/helpfulness route through it.

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
  +sweep=m2_arms_all_2026_08_03
