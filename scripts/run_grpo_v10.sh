#!/bin/bash
# v10 GRPO launch — cost-sensitive (asymmetric) appropriateness multiplier.
#
# Single reward variable vs v9: rground_app_floor_prohibit=0.1. A FALSE-PERMIT
# (model calls a prohibited/discouraged-governed flow "appropriate") now floors
# at 0.1 instead of the symmetric 0.4, widening the within-group gradient on
# prohibited flows (0.6 -> 0.9) toward the correct "inappropriate" verdict.
# Motivation: the 2026-06-24 verdict-balance diagnostic found the governing norms
# are ~3.98:1 appropriate:inappropriate, and the v9 policy commits the CORRECT
# verdict on prohibited-governed flows only 30% of the time (53% false-permits)
# on well-retrieved norms (H2 retrieval-noise ruled out). Everything else held
# from v9: multiplicative direction, app_floor 0.4, mu=1, beta=0.02,
# token_truncate, G=8, ranked judge, prescreen, scale=none, contentless-v6 SFT.
# Regime: save_steps 50 (denser, for kill-at-peak + judge-free GoldCoin
# checkpoint selection). num_epochs kept 3 but we kill at the held-out peak
# (~epoch 1, as in v8/v9) on the entropy/IS tripwire.
#
# The reward change bumps rground_formula_version -> "v10_cost_sensitive_floor",
# so the prescreen cache MISSES and re-screens (~4.5h) with the corrected reward.
# A FRESH cache path guarantees no collision with the v9 cache.
#
# Servers must be up (they are: judge 725605 @ klara:8002, embed 725607 @
# klara:8001). NORM_UNIVERSES_PATH / CI_REASONING_PATH / *_SERVER_URL load from
# server.env via ensure_dotenv.
# See wiki/grpo_training_field_notes/2026-06-24_v10_plan.md.

set -euo pipefail

PROJECT_ROOT=/share/pierson/matt/UAIR
source "$PROJECT_ROOT/.venv/bin/activate"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
cd "$PROJECT_ROOT"

python -m dagspaces.grpo_training.cli \
  pipeline=grpo_only_online_external \
  training/grpo=online_rground_external \
  model=qwen3.5-9b/sft-contentless-v6 \
  training.grpo.prescreen.cache_path=/share/pierson/matt/UAIR/cache/grpo_prescreen_v10.json \
  experiment.name=grpo_redesign_full_v10
