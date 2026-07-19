#!/bin/bash
# Gold-label regeneration (top100 norms) with google/gemma-4-31B-it
# and the FICTION prompts — one of the 4 runs of the 2026-07-12 re-extraction
# (fiction10/top100 x norms/flows) after the prompt-wiring fix: every prior
# fiction extraction run had silently used the prescriptive prompts (group
# defaults in config.yaml clobbered pipeline prompt selections; guarded now
# by tests/historical_norms/test_prompt_wiring.py).
#
# Judge: gemma-4-31b/instruct (dense). The 26B-A4B MoE was tried first and ran
# ~25x slower under vLLM 0.19's Triton MoE path on Ampere (2026-07-12) — the
# dense model is both faster on this stack AND slightly stronger. TP=4 single
# replica (at TP=2 the 31B was KV-starved: 25k KV tokens, ~3 concurrent seqs).
# Chunks: the top100 fiction chunk cache (100/100 books).
# Estimate: ~2-3h reasoning + extraction long tail (15,875 chunks; qwen3.6-27B dense took 2.8h + 19.3h — expect similar).
#
# SANITY: after the stage starts, confirm the job log prints
#   PROMPT PROVENANCE: norm_reasoning_fiction
# and that output parquets carry prompt_name == 'norm_reasoning_fiction'.
#
# W&B: project historical-norms-extraction, run top100_norms_gemma4_<stage suffixes>.

set -euo pipefail

PROJECT_ROOT=/share/pierson/matt/UAIR
source "${TRAWLER_DRIVER_VENV:-$PROJECT_ROOT/.venv-vllm025cu129}/bin/activate"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
cd "$PROJECT_ROOT"

FICTION_CHUNKS_PATH="${FICTION_CHUNKS_PATH:-/share/pierson/matt/zoo/datasets/gutenberg_cache/chunks_top100_fiction_en.parquet}"
if [[ ! -f "$FICTION_CHUNKS_PATH" ]]; then
  echo "ERROR: chunk source not found: $FICTION_CHUNKS_PATH" >&2
  exit 1
fi
export FICTION_CHUNKS_PATH
echo "[top100_norms_gemma4] chunks : $FICTION_CHUNKS_PATH"
echo "[top100_norms_gemma4] model  : gemma-4-31b/instruct | pipeline: COLM_norms_fiction_prefetched_gemma4"

python -m dagspaces.historical_norms.cli \
  pipeline=COLM_norms_fiction_prefetched_gemma4 \
  model=gemma-4-31b/instruct \
  experiment.name=top100_norms_gemma4
