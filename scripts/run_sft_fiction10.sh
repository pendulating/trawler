#!/bin/bash
# Run SFT-only pipeline on fiction10 CI extraction data.
#
# Usage:
#   bash scripts/run_sft_fiction10.sh
#
# Data prep runs locally (no GPU needed).
# SFT training is submitted to SLURM with 4x A6000.

set -euo pipefail

source /share/pierson/matt/UAIR/.venv/bin/activate
export PYTHONPATH=/share/pierson/matt/UAIR:$PYTHONPATH

# Source data paths
export CI_REASONING_PATH=/share/pierson/matt/n2s4cir/data/fiction10/ci_reasoning.parquet
export CI_EXTRACTION_PATH=/share/pierson/matt/n2s4cir/data/fiction10/ci_flows.parquet

echo "=== SFT Fiction10 Pipeline ==="
echo "CI Reasoning: $CI_REASONING_PATH"
echo "CI Extraction: $CI_EXTRACTION_PATH"
echo ""

python -m dagspaces.grpo_training.cli \
    pipeline=sft_only \
    model=qwen3.5-9b \
    experiment.name=sft_fiction10 \
    wandb.enabled=false
