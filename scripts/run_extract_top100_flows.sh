#!/bin/bash
# Job 1 of the top100-flows GRPO plan: CI-flow REASONING over the top100
# fiction chunk cache (stage 1 only — the only artifact the GRPO run blocks on).
#
# Produces: multirun/<date>_historical_norms/<time>/0/COLM_flows_reasoning_qwen36/
#             outputs/ci_reasoning/reasoning.parquet
#   schema: gutenberg_id / chunk_id / article_text / ... /
#           has_information_exchange / ci_flow_count / ci_reasoning_text
#   (same shape as the fiction10 CI_REASONING_PATH the GRPO loader consumes).
#
# Cost: ~3h on 4 GPUs (DP=2 x TP=2, Qwen3.6-27B) — the norms-track reasoning
# over the same 15,875 chunks took 2.8h. The 5-tuple ci_extraction stage
# (~19h) is deliberately NOT run here; resume it later, wasting nothing, via
# pipeline=ci_extraction_from_reasoning_fiction reasoning_dataset_path=<output>.
#
# Next step after completion: scripts/run_grpo_top100_flows.sh
# Plan: wiki/grpo_training_field_notes/2026-07-08_top100_flows_plan.md

set -euo pipefail

PROJECT_ROOT=/share/pierson/matt/UAIR
source "${TRAWLER_DRIVER_VENV:-$PROJECT_ROOT/.venv-vllm025cu129}/bin/activate"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
cd "$PROJECT_ROOT"

# The top100 fiction chunk cache (100/100 books, 15,875 chunks, 6000/1000
# chunking — the same cache the top100 norms extraction consumed).
FICTION_CHUNKS_PATH="${FICTION_CHUNKS_PATH:-/share/pierson/matt/zoo/datasets/gutenberg_cache/chunks_top100_fiction_en.parquet}"
if [[ ! -f "$FICTION_CHUNKS_PATH" ]]; then
  echo "ERROR: chunk cache not found: $FICTION_CHUNKS_PATH" >&2
  exit 1
fi
export FICTION_CHUNKS_PATH
echo "[top100-flows] chunks: $FICTION_CHUNKS_PATH"

python -m dagspaces.historical_norms.cli \
  pipeline=COLM_flows_reasoning_prefetched_qwen36 \
  model=qwen3.6-27b/instruct \
  experiment.name=top100_flows_reasoning
