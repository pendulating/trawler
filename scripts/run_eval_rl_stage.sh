#!/usr/bin/env bash
# The GRPO keeper (v9-ckpt100) + its own matched SFT base, evaluated under the
# CURRENT camera-ready protocol so both are readable by
# notebooks/colm-camera-ready/benchmark_results.py.
#
# The keeper's published numbers come from the 2026-06-24 per-benchmark
# multiruns, which predate every 2026-07-21 parity review and are
# Qwen3.6-27B-judged — the camera-ready notebook requires Gemma-4-31B-it and
# post-flip GoldCoin / post-parser-fix PrivacyLens. See
# dagspaces/eval_all/conf/sweep/eval_rl_stage_keeper_2026_08_03.yaml.
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
# NO benchmark_filter: the camera-ready table's canonical rows come from the
# FULL roster (goldcoin, privacylens, cirl, confaide, vlm_geoprivacy, mmlu).
# Filtering to ci_only would leave the RL rows with a blank MMLU column while
# every other row has one.
exec "${TRAWLER_DRIVER_VENV:-$PROJECT_ROOT/.venv-vllm025cu129}/bin/python" \
  -m dagspaces.eval_all.cli --multirun \
  +sweep=eval_rl_stage_keeper_2026_08_03
