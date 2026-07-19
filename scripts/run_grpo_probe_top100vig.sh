#!/bin/bash
# v11 probe — top100-balanced judgment vignettes (single variable vs v10).
#
# Tests the v11 lever-(a) hypothesis WITHOUT the ~32h flows run: the judgment
# vignettes are drawn from the force-balanced top100 norm universe (CI-relevant
# appropriate:inappropriate 1.72:1, 1,975 "no" candidates) instead of fiction10
# (3.07:1, 296 "no"). EVERYTHING ELSE IS v10-IDENTICAL — R_ground grounding and
# the 70% CI-extraction prompts keep the fiction10 universe / ci_reasoning via
# server.env; only VIGNETTE_NORM_UNIVERSES_PATH changes. So this is a clean
# single-variable comparison against the v10 run.
#
# Falsifiable: GoldCoin Forbid recall moves off the v9/v10 plateau (0.55) toward
# SFT's 0.65. If it stays at 0.55 even with balanced judgment data, balancing the
# judgment task alone is insufficient -> revisit before paying the flows run.
#
# PREREQUISITES (both the user's call — this script does NOT start them):
#   1. Aux servers up: embedding @ :8001, judge @ :8002 (scripts/launch_auxiliary_servers.sh).
#      As of 2026-06-30 the klara servers in server.env were unreachable — relaunch
#      and update EMBEDDING_SERVER_URL / JUDGE_SERVER_URL in server.env if the host changed.
#   2. The top100 norm_universe build finished (pipeline=norm_universe_only). This
#      script auto-discovers the newest such build unless VIGNETTE_NORM_UNIVERSES_PATH is set.
#
# See wiki/grpo_training_field_notes/2026-06-27_v11_plan.md.

set -euo pipefail

PROJECT_ROOT=/share/pierson/matt/UAIR
source "${TRAWLER_DRIVER_VENV:-$PROJECT_ROOT/.venv-vllm025cu129}/bin/activate"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
cd "$PROJECT_ROOT"

# Resolve the top100 vignette universe: explicit override wins, else newest build.
if [[ -z "${VIGNETTE_NORM_UNIVERSES_PATH:-}" ]]; then
  VIGNETTE_NORM_UNIVERSES_PATH=$(ls -t \
    "$PROJECT_ROOT"/outputs/*/*/norm_universe_only/outputs/norm_universe/norm_universes.json \
    "$PROJECT_ROOT"/multirun/*/*/norm_universe_only/outputs/norm_universe/norm_universes.json \
    2>/dev/null | head -1 || true)
fi
if [[ -z "${VIGNETTE_NORM_UNIVERSES_PATH:-}" || ! -f "$VIGNETTE_NORM_UNIVERSES_PATH" ]]; then
  echo "ERROR: no top100 norm_universes.json found. Build it first:" >&2
  echo "  ABSTRACTED_NORMS_PATH=.../role_abstraction/abstracted_norms.parquet \\" >&2
  echo "    python -m dagspaces.grpo_training.cli pipeline=norm_universe_only model=qwen3.5-9b/sft-contentless-v6" >&2
  exit 1
fi
export VIGNETTE_NORM_UNIVERSES_PATH
echo "[probe] vignette universe : $VIGNETTE_NORM_UNIVERSES_PATH"
echo "[probe] grounding universe: ${NORM_UNIVERSES_PATH:-<from server.env>} (unchanged)"

python -m dagspaces.grpo_training.cli \
  pipeline=grpo_only_online_external \
  training/grpo=online_rground_external \
  model=qwen3.5-9b/sft-contentless-v6 \
  training.grpo.prescreen.cache_path=/share/pierson/matt/UAIR/cache/grpo_prescreen_probe_top100vig.json \
  experiment.name=grpo_probe_top100_vignettes
