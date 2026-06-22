#!/bin/bash
# v6 full GRPO run — "teach context-relative privacy reasoning".
#
# Trains on top of the contentless-curated SFT (qwen3.5-9b/sft-contentless-v6),
# with the v6 recipe baked into training/grpo=online_rground_external:
#   L2  rground_app_weight=0.3   (deontic appropriateness blended into R_ground)
#   L3  learning_rate=2e-5 + lr_scheduler_type=cosine_with_min_lr / min_lr_rate=0.3
#   kept: lambda=1.0, abstention_penalty=0.4, require_flow_variance=true,
#         beta=0, epsilon_high=0.28, vllm_importance_sampling_mode=token_truncate
# (No false-extraction penalty — gold=False is a prescriptive-norm label, not a
#  flow label; see scripts/audit_goldno_labels.py.)
#
# New SFT base => prescreen CACHE MISS => ~4.5h re-screen before training, then
# ~25-30h (~522 steps / 3 epochs). Servers (embed:8001, judge:8002 = Qwen3.6-27B)
# must be up; server.env is auto-sourced for URLs + GRPO_PRESCREEN_CACHE.
#
# WATCH (the falsifiable test): the prescreen sft_no_flow_rate (vs old SFT 0.54)
# and the FIRST reward_traces window's gold-YES conditional abstention (vs the
# flat ~0.62 of v2-v5). If curated-SFT alone already dropped it, the prior was
# the driver; GRPO should then push it further down while gold-NO stays high.

set -euo pipefail

PROJECT_ROOT=/share/pierson/matt/UAIR
source "$PROJECT_ROOT/.venv/bin/activate"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
cd "$PROJECT_ROOT"

python -m dagspaces.grpo_training.cli -m \
  pipeline=grpo_only_online_external \
  model=qwen3.5-9b/sft-contentless-v6 \
  training/grpo=online_rground_external \
  experiment.name=grpo_redesign_full_v6
