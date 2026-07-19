#!/bin/bash
# v12a — cost-sensitive HEDGE tier (single variable vs the v11 probe).
#
# The v11 probe (top100-balanced vignettes) confirmed its pre-registered
# prediction: the vignette rebalance halted v10's verdict erosion (ConfAIDE-2b
# gap halved, CIRL held at SFT level vs v10's over-accept outlier) but left the
# extraction-side constraint untouched — prohibited-flow hedge mass frozen at
# ~72% (= v10), GoldCoin Forbid recall plateaued at 0.55 for the THIRD
# consecutive iteration (SFT 0.65), best macro-F1 0.755 identical to v10's.
# Exploration guard ~0.41-0.50: a correct committer exists in half the groups
# and still loses — hedge EV, not exploration, binds. v12a attacks that EV
# directly: rground_app_hedge_prohibit=0.5 drops a hedged ("ambiguous")
# verdict on a prohibited/discouraged-governed flow from the neutral 0.7 tier
# to 0.5, widening the commit-vs-hedge direction gap 0.3 -> 0.5 exactly where
# it binds. Hedges elsewhere stay 0.7; no-flow declarations stay neutral;
# v10's false-permit floor (0.1) and the v11 top100 vignettes are both kept —
# so the v11 probe is the control and the hedge tier is the single variable.
#
# Falsifiable (mid-run, scripts/analyze_grpo_verdict_traces.py table 3):
# prohibited-flow correct-commit share rises off ~0.10 (v10/v11 range
# 0.06-0.12) and hedge(0.7)+hedge-prohibit(0.5) mass falls below ~0.70.
# Held-out: GoldCoin Forbid recall finally off 0.55 toward SFT 0.65 without
# tanking Permit recall / applicability (over-correction watch = the v8
# indiscriminate-permit mirror). If traces move but held-out doesn't, escalate
# to v12b (prohibited-rich extraction upweighting) / the top100 flows run.
#
# PREREQUISITES (both the user's call — this script does NOT start them):
#   1. Aux servers up: embedding @ :8001, judge @ :8002
#      (scripts/launch_auxiliary_servers.sh; running on klara as of 2026-07-03).
#   2. The top100 norm_universe build (auto-discovered below, as in the v11 probe).
#
# See wiki/grpo_training_field_notes/2026-07-03_v12a_plan.md.

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
echo "[v12a] vignette universe : $VIGNETTE_NORM_UNIVERSES_PATH"
echo "[v12a] grounding universe: ${NORM_UNIVERSES_PATH:-<from server.env>} (unchanged)"

# hedge tier pinned explicitly (also the yaml default) — cell discipline: the
# knob that defines the cell never rides on a config default.
python -m dagspaces.grpo_training.cli \
  pipeline=grpo_only_online_external \
  training/grpo=online_rground_external \
  model=qwen3.5-9b/sft-contentless-v6 \
  training.grpo.rground_app_hedge_prohibit=0.5 \
  training.grpo.prescreen.cache_path=/share/pierson/matt/UAIR/cache/grpo_prescreen_v12a_hedge_tier.json \
  experiment.name=grpo_v12a_hedge_tier
