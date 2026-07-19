#!/bin/bash
# v13a — μ=2 / Clip-Higher activation (single variable vs the v11 probe).
#
# epsilon_high=0.28 has been configured since v4 but is INERT at
# num_iterations=1: the rollout logp snapshot sees no optimizer step before the
# single loss pass, so the PPO ratio ≡ 1 and neither clip ever binds (v4
# measured clip_ratio/high_mean ~1e-4 all run). μ=2 makes the second inner pass
# depart from ratio 1, finally letting advantage>0 (commit/extraction) tokens
# take asymmetrically larger up-steps — DAPO's anti-mode-collapse lever, echoed
# by URPO (clip ceiling 1.28). The v9 plan deferred μ>1 with "re-evaluate only
# if movement stalls"; it has stalled: prohibited-flow hedge mass ~72% and
# GoldCoin Forbid recall 0.55 held across v10 AND the v11 probe. v8 already
# showed μ=2 produces real held-out movement (applicability 0.921→0.972) on the
# pre-directional reward before an entropy breakout ~step 240; v13a asks
# whether the v9+ CONCENTRATING reward keeps that movement without the breakout.
#
# Sibling to v12a (unrun as of 2026-07-16): v12a attacks hedge EV (reward),
# v13a attacks update strength (optimizer) — disjoint mechanisms, disjoint
# trace fingerprints, SHARED control (the v11 probe). Hence the hedge tier is
# pinned back to null here: the yaml default became 0.5 with v12a, and riding
# it would silently make this a two-variable cell.
#
# Falsifiable (scripts/analyze_grpo_verdict_traces.py table 3 +
# trace_metrics.summarize_log_history):
#   - clip_ratio/high_mean nonzero (direct activation check; ~1e-4 = arm dead)
#   - prohibited-flow correct-commit share off ~0.10; hedge mass < ~0.70
#   - held-out: GoldCoin Forbid recall off 0.55 toward SFT 0.65 WITHOUT the v8
#     indiscriminate-permit mirror (watch Permit recall + applicability)
# KILL: v8 entropy-breakout fingerprint — entropy trend up with
# corr(entropy, logp_diff) → +0.9 or IS collapse → kill, keep best pre-breakout
# ckpt (save_steps=50). If the breakout reproduces despite the concentrating
# reward, μ>1 is ruled out for good.
#
# PREREQUISITES (both the user's call — this script does NOT start them):
#   1. Aux servers up: embedding @ :8001, judge @ :8002
#      (scripts/launch_auxiliary_servers.sh).
#   2. The top100 norm_universe build (auto-discovered below, as in v11/v12a).
#
# See wiki/grpo_training_field_notes/2026-07-16_rl_papers_synthesis.md.

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
echo "[v13a] vignette universe : $VIGNETTE_NORM_UNIVERSES_PATH"
echo "[v13a] grounding universe: ${NORM_UNIVERSES_PATH:-<from server.env>} (unchanged)"

# Cell discipline: every knob that defines (or could silently redefine) the
# cell is pinned on the CLI, never ridden on a config default.
#   num_iterations=2            — THE single variable vs the v11 probe
#   epsilon_high=0.28           — what μ=2 activates (yaml value, pinned)
#   beta=0.02                   — μ>1 stability precondition (v8; yaml value, pinned)
#   rground_app_hedge_prohibit=null — v11-identical control; yaml default is
#                                 now 0.5 (the v12a cell), MUST be pinned back
python -m dagspaces.grpo_training.cli \
  pipeline=grpo_only_online_external \
  training/grpo=online_rground_external \
  model=qwen3.5-9b/sft-contentless-v6 \
  training.grpo.num_iterations=2 \
  training.grpo.epsilon_high=0.28 \
  training.grpo.beta=0.02 \
  training.grpo.rground_app_hedge_prohibit=null \
  training.grpo.prescreen.cache_path=/share/pierson/matt/UAIR/cache/grpo_prescreen_v13a_mu2.json \
  experiment.name=grpo_v13a_mu2_cliphigher
