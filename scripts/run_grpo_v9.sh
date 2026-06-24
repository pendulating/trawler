#!/bin/bash
# v9 full GRPO run — two-sided reward (appropriateness direction) + simplification.
# Plan: wiki/grpo_training_field_notes/2026-06-23_v9_plan.md
# Motivation: wiki/grpo_training_field_notes/2026-06-23_v8_results_groundtruth.md
#
# Trains on the contentless-curated SFT (qwen3.5-9b/sft-contentless-v6), same
# base as v6/v7/v8. v8's ground-truth eval proved the reward is the binding
# constraint: it was ONE-directional (reward extraction / penalize abstention,
# never penalize mis-judging a flow), so its gradient pushed a behavioral
# direction ("extract/permit more") instead of a correct completion → entropy
# diffused and the policy learned an indiscriminate engage/permit bias (GoldCoin
# compliance Forbid recall 0.70→0.35). v8 traces: 97% of extracted flows have a
# DIRECTIONAL governing norm, yet the model HEDGES ("ambiguous") 73% of the time
# — the reward made hedging the safe local optimum.
#
# v9 levers (all baked into training/grpo=online_rground_external; 548 tests green):
#
#   Two-sidedness (the fix) — appropriateness as a gold-INDEPENDENT direction:
#     rground_app_mode: multiplicative   R_ground = grounding · direction, where
#       direction = app_floor + (1-app_floor)·app_consistency (deontic Raz force →
#       expected appropriateness). Correct verdict ×1.0, hedge ×0.7, wrong (e.g. a
#       violation called "appropriate") ×0.4. Promotes appropriateness from a 0.3
#       ADDITIVE blend (~0.19 of composite, mis-judgment nearly free → hedging) to a
#       MULTIPLIER that makes correct violation-detection worth as much as correct
#       endorsement. app_floor=0.4 DISCOUNTS rather than zeroes (deontic-noise guard).
#
#   Simplification (cut bug surface, stay faithful):
#     reward_composition: directional    R = gate · content for extractions:
#       gate    = {r_uncert, r_complete, r_consist, r_cohere}  (well-formedness)
#       content = {r_context (kept, light), r_ground (dominant, =grounding·direction)}
#     no-flow → no_flow_reward(gold) directly (gold=T→0.1, gold=F→0.6, unknown→0.4).
#     Drops the gated gate×disc AND the post-hoc abstention_penalty (=0.0 now).
#
#   Stability:
#     num_iterations: 1   reverted from v8's μ=2 (which reintroduced the entropy
#                         breakout for no held-out gain). With a CONCENTRATING reward
#                         the stable μ=1 regime should now move on its own.
#     beta: 0.02          held (the v7 stabilizer).
#
# R_ground semantics changed → prescreen cache MUST re-screen (prompt_screening
# rground_formula_version → v9_directional). Fresh cache path below.
#
# SUCCESS METRIC (the falsifiable test, via a fresh judge-free GoldCoin eval of
# the v9 final checkpoint vs the SFT base — see scripts/run_eval_v8_goldcoin.sh
# for the pattern): GoldCoin compliance **Forbid recall back toward/above 0.70**
# while applicability holds ~0.97.
#
# WATCH (training, via scripts/grpo_field_metrics.py <run_dir>):
#   - app_consistency mean rises off 0.575; hedge fraction (c=0.5) falls from 73.5%.
#   - entropy stays bounded (no v8-style breakout): μ=1 + β=0.02 + concentrating
#     reward → entropy flat/down, not the +0.0046/step climb. summarize_log_history
#     correlations catch a relapse early.
#   - reward_trend turns positive as wrong/hedged verdicts get discounted.
#
# COST: fresh prescreen (~4.5h, cache miss by design) + ~25-30h training
# (~520 steps / 3 epochs at μ=1). Servers (embed:8001, judge:8002 = Qwen3.6-27B)
# must be up; server.env auto-sources their URLs + the source data paths.
#
# GPUs: pipeline node grpo_training uses slurm_train_1x (1 GPU, tp=1, vLLM
# colocate). GPU0 is fixed, so no spare-request override is needed.

set -euo pipefail

PROJECT_ROOT=/share/pierson/matt/UAIR
source "$PROJECT_ROOT/.venv/bin/activate"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
cd "$PROJECT_ROOT"

# Fresh prescreen cache for v9 (the directional + multiplicative reward changes
# R_ground; the rground_formula_version bump already forces a cache MISS, but a
# new path keeps the v6/v7/v8 caches pristine). Overridden via the hydra key (a
# literal that ships in the pickled config to the compute node), NOT an env var —
# submitit re-sources server.env on the node and would otherwise resolve
# GRPO_PRESCREEN_CACHE back to the old path.
python -m dagspaces.grpo_training.cli -m \
  pipeline=grpo_only_online_external \
  model=qwen3.5-9b/sft-contentless-v6 \
  training/grpo=online_rground_external \
  training.grpo.prescreen.cache_path=/share/pierson/matt/UAIR/cache/grpo_prescreen_v9.json \
  experiment.name=grpo_redesign_full_v9
