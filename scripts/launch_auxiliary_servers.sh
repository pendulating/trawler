#!/bin/bash
# Launch embedding + judge vLLM servers for shared use by GRPO pipelines.
#
# These servers run independently and can be reused across multiple
# training pipelines. Pass the printed export vars to training jobs.
#
# Usage:
#   ./scripts/launch_auxiliary_servers.sh [EMB_GPU] [JUDGE_GPUS] [EMB_PORT] [JUDGE_PORT]
#
# Examples:
#   # Default: embed on GPU 0, judge on GPUs 1,2
#   ./scripts/launch_auxiliary_servers.sh
#
#   # Custom GPU/port assignment
#   ./scripts/launch_auxiliary_servers.sh 4 5,6 9001 9002
#
#   # Via SLURM interactive session
#   srun --gres=gpu:3 --cpus-per-task=8 --mem=64G --pty bash
#   ./scripts/launch_auxiliary_servers.sh 0 1,2

set -euo pipefail

EMB_GPU="${1:-0}"
JUDGE_GPUS="${2:-1,2}"
EMB_PORT="${3:-8001}"
JUDGE_PORT="${4:-8002}"

EMB_MODEL="${EMBEDDING_MODEL:-/share/pierson/matt/zoo/models/Qwen3-Embedding-8B}"
JUDGE_MODEL="${JUDGE_MODEL:-/share/pierson/matt/zoo/models/Qwen2.5-72B-Instruct-AWQ}"
JUDGE_MAX_MODEL_LEN="${JUDGE_MAX_MODEL_LEN:-8192}"

# Count judge GPUs for tensor parallelism
JUDGE_TP=$(echo "$JUDGE_GPUS" | tr ',' '\n' | wc -l)

echo "=== Auxiliary Server Launcher ==="
echo "Embedding: model=${EMB_MODEL}, gpu=${EMB_GPU}, port=${EMB_PORT}"
echo "Judge:     model=${JUDGE_MODEL}, gpus=${JUDGE_GPUS}, tp=${JUDGE_TP}, port=${JUDGE_PORT}"
echo ""

# NCCL env vars for PCIe-only machines (no NVLink)
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=1

# Launch embedding server
echo "[$(date)] Launching embedding server on GPU ${EMB_GPU}..."
CUDA_VISIBLE_DEVICES="$EMB_GPU" vllm serve "$EMB_MODEL" \
    --port "$EMB_PORT" \
    --host 0.0.0.0 \
    --tensor-parallel-size 1 \
    2>&1 | sed 's/^/[embed] /' &
EMB_PID=$!

# Launch judge server
echo "[$(date)] Launching judge server on GPUs ${JUDGE_GPUS}..."
CUDA_VISIBLE_DEVICES="$JUDGE_GPUS" vllm serve "$JUDGE_MODEL" \
    --port "$JUDGE_PORT" \
    --host 0.0.0.0 \
    --tensor-parallel-size "$JUDGE_TP" \
    --quantization awq \
    --max-model-len "$JUDGE_MAX_MODEL_LEN" \
    --disable-custom-all-reduce \
    2>&1 | sed 's/^/[judge] /' &
JUDGE_PID=$!

# Wait for health checks
echo ""
echo "[$(date)] Waiting for servers to become ready..."
for port_label in "${EMB_PORT}:embedding" "${JUDGE_PORT}:judge"; do
    port="${port_label%%:*}"
    label="${port_label##*:}"
    for i in $(seq 1 120); do
        if curl -sf "http://localhost:${port}/health" > /dev/null 2>&1; then
            echo "[$(date)] ${label} server ready on port ${port}"
            break
        fi
        if [ "$i" -eq 120 ]; then
            echo "[$(date)] ERROR: ${label} server not ready after 600s"
            kill $EMB_PID $JUDGE_PID 2>/dev/null || true
            exit 1
        fi
        sleep 5
    done
done

HOSTNAME=$(hostname)
echo ""
echo "============================================"
echo "  Servers ready. Export for training jobs:"
echo ""
echo "  export EMBEDDING_SERVER_URL=http://${HOSTNAME}:${EMB_PORT}"
echo "  export JUDGE_SERVER_URL=http://${HOSTNAME}:${JUDGE_PORT}"
echo ""
echo "  PIDs: embedding=${EMB_PID}, judge=${JUDGE_PID}"
echo "  Kill: kill ${EMB_PID} ${JUDGE_PID}"
echo "============================================"

# Wait for either to exit
wait -n $EMB_PID $JUDGE_PID 2>/dev/null || true
echo "[$(date)] A server exited. Shutting down..."
kill $EMB_PID $JUDGE_PID 2>/dev/null || true
wait
