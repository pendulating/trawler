#!/bin/bash
# COLM paper sweep launcher — 15 cells over (contrastive_lambda × contrastive_ratio).
#
# Runs three Hydra multirun sweeps sequentially:
#   1. lambda_axis  — 7 cells (λ ∈ {0, 0.25, 0.5, 0.75, 1, 1.5, 2}, ρ=0)
#   2. ratio_axis   — 4 cells (λ=1.0, ρ ∈ {0.05, 0.10, 0.20, 0.50})
#   3. offaxis      — 4 cells (λ ∈ {0.5, 1.5} × ρ ∈ {0.10, 0.50})
# Total: 15 unique (λ, ρ) cells. Each cell is one GRPO training job.
#
# Why three sweeps instead of one: Hydra's default sweeper does Cartesian
# products of params, so non-Cartesian 2D grids need to be split. The three
# pieces above are each Cartesian internally.
#
# Prerequisites: server.env (auto-sourced below) provides
#   EMBEDDING_SERVER_URL   e.g. http://klara:8001
#   JUDGE_SERVER_URL       e.g. http://klara:8002 (Qwen3.6-27B)
#   NORM_UNIVERSES_PATH    norm_universes.json
#   CI_REASONING_PATH      chunks parquet
#   NORM_EMBEDDINGS_PATH   (optional) per-book .npy embeddings dir
# The SFT LoRA dir comes from model.lora_path (qwen3.5-9b/sft-ci.yaml).
#
# Usage:
#   ./scripts/launch_lambda_ratio_sweep.sh                # all 15 cells
#   ./scripts/launch_lambda_ratio_sweep.sh lambda_axis    # only the λ axis (7)
#   ./scripts/launch_lambda_ratio_sweep.sh ratio_axis     # only the ρ axis (4)
#   ./scripts/launch_lambda_ratio_sweep.sh offaxis        # only off-axis (4)
#
# Each sub-sweep blocks until its submitit array completes (Hydra default).
# Wall-time per cell ≈ 6 h on slurm_train_1x; with array_parallelism=3 and
# sequential sub-sweeps, expect ~30 h end-to-end for all 15 cells.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/share/pierson/matt/UAIR}"
cd "$PROJECT_ROOT"

# -------- source server.env so pre-flight sees the same values Hydra will --------
if [ -f "$PROJECT_ROOT/server.env" ]; then
    set -a
    # shellcheck disable=SC1091
    . "$PROJECT_ROOT/server.env"
    set +a
fi

# -------- environment sanity --------
required_vars=(
    EMBEDDING_SERVER_URL
    JUDGE_SERVER_URL
    NORM_UNIVERSES_PATH
    CI_REASONING_PATH
)
missing=0
for v in "${required_vars[@]}"; do
    if [ -z "${!v:-}" ]; then
        echo "ERROR: env var $v is not set (expected in server.env or shell)." >&2
        missing=1
    fi
done
if [ "$missing" -ne 0 ]; then
    echo "Set the missing variables and retry." >&2
    exit 1
fi

# -------- aux server health checks --------
for url_label in "${EMBEDDING_SERVER_URL}:embed" "${JUDGE_SERVER_URL}:judge"; do
    url="${url_label%:*}"
    label="${url_label##*:}"
    if ! curl -sf "${url}/health" > /dev/null 2>&1; then
        echo "ERROR: ${label} server unhealthy at ${url}/health." >&2
        echo "  Launch with: sbatch scripts/judge_server.sub  (or scripts/launch_auxiliary_servers.sh)" >&2
        exit 1
    fi
    echo "  ✓ ${label} server reachable at ${url}"
done

# -------- judge model sanity (Qwen3.6-27B is the canonical paper judge) --------
judge_model=$(curl -sf "${JUDGE_SERVER_URL}/v1/models" 2>/dev/null \
    | python -c "import sys, json; data = json.load(sys.stdin); print(data['data'][0]['id'])" 2>/dev/null \
    || echo "<unknown>")
echo "  ✓ Judge model served: ${judge_model}"
case "$judge_model" in
    *Qwen3.6-27B*) ;;
    *)
        echo "WARNING: judge model is NOT Qwen3.6-27B (the COLM paper judge)." >&2
        echo "         If this is intentional, press enter to continue. Ctrl-C to abort." >&2
        read -r _confirm
        ;;
esac

# -------- dispatch sub-sweeps --------
TARGET="${1:-all}"

run_sweep() {
    local sweep_name="$1"
    echo ""
    echo "=========================================="
    echo "  Launching sub-sweep: +sweep=${sweep_name}"
    echo "  $(date)"
    echo "=========================================="
    python -m dagspaces.grpo_training.cli "+sweep=${sweep_name}"
}

case "$TARGET" in
    all)
        run_sweep lambda_axis
        run_sweep ratio_axis
        run_sweep offaxis
        ;;
    lambda_axis|ratio_axis|offaxis)
        run_sweep "$TARGET"
        ;;
    *)
        echo "ERROR: unknown target '${TARGET}'. Choices: all, lambda_axis, ratio_axis, offaxis." >&2
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "  All requested sub-sweeps dispatched."
echo "  Run scripts/build_sweep_model_yamls.py once trainings complete"
echo "  to generate eval-time model yamls under"
echo "  dagspaces/common/conf/model/qwen3.5-9b/grpo-l<L>-r<R>.yaml"
echo "=========================================="
