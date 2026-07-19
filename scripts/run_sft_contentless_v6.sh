#!/bin/bash
# v6 re-SFT: replicate the production SFT recipe (qwen3.5-9b/instruct, lr 2e-5,
# 3 epochs, LoRA r64/a128, all flow_* toggles on — i.e. the 2026-04-28
# ctx-True_appr-True_norms-True_conf-True cell that v2-v5 GRPO trained on top of)
# but curate the negatives to genuinely-contentless chunks.
#
# WHY: has_information_exchange=False is a *prescriptive-norm* label, not a
# *descriptive-flow* label — ~60% of gold=False chunks describe a real
# disclosure/conversation that simply isn't norm-governed. Training on those as
# "abstain" negatives installs the over-abstention prior that v2-v5 GRPO could
# not escape. negative_selection=contentless drops them (see
# sft_data_prep._is_contentless_chunk; audit in scripts/audit_goldno_labels.py).
#
# Produces the SFT checkpoint that v6 GRPO will train on top of.

set -euo pipefail

source "${TRAWLER_DRIVER_VENV:-/share/pierson/matt/UAIR/.venv-vllm025cu129}/bin/activate"
export PYTHONPATH=/share/pierson/matt/UAIR:${PYTHONPATH:-}

export CI_REASONING_PATH=/share/pierson/matt/n2s4cir/data/fiction10/ci_reasoning.parquet
export CI_EXTRACTION_PATH=/share/pierson/matt/n2s4cir/data/fiction10/ci_flows.parquet

echo "=== v6 re-SFT (contentless-curated negatives) ==="
echo "CI Reasoning:  $CI_REASONING_PATH"
echo "CI Extraction: $CI_EXTRACTION_PATH"
echo ""

python -m dagspaces.grpo_training.cli \
    pipeline=sft_only \
    model=qwen3.5-9b/instruct \
    training.sft.negative_selection=contentless \
    experiment.name=sft_contentless_v6 \
    wandb.enabled=false
