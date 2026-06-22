#!/bin/bash
# v8 full GRPO run — update strength + symmetric reward.
# Plan: wiki/grpo_training_field_notes/2026-06-22_v8_plan.md
#
# Trains on the contentless-curated SFT (qwen3.5-9b/sft-contentless-v6), same
# base as v6/v7. Two single-purpose levers vs v6/v7, both already implemented
# (530+ tests green) and baked into the recipe / stage code:
#
#   Lever 1 (config, training/grpo=online_rground_external):
#     beta 0.0 -> 0.02        the v7 stabilizer (entropy bounded, IS~1); REQUIRED
#                             precondition for mu>1. beta*KL~0.001 = floor, not anchor.
#     num_iterations 1 -> 2   mu=2. At mu=1 the PPO ratio is identically 1 so the
#                             configured epsilon_high=0.28 (Clip-Higher) NEVER binds
#                             (v4's clip_ratio/high_mean ~1e-4). mu=2 recomputes
#                             logp at updated weights -> ratio departs from 1 ->
#                             Clip-Higher activates and 2x the optimizer steps per
#                             rollout, JUDGE-COST-NEUTRAL (same judged batch reused).
#
#   Lever 2 (code, stages/online_rground.py _call_ranked):
#     symmetric contrastive clamp — contrast the GROUNDING component only and
#     protect the rank component. Fixes the asymmetry that clamped ~1/3 of
#     well-grounded extractions to R_ground=0 (mean R_ground|ext 0.26 -> ~0.51
#     counterfactual). lambda=1.0 and the contrastive thesis are UNCHANGED (bug
#     fix, not a weakening). Changes R_ground semantics -> prescreen cache MUST
#     re-screen (prompt_screening _reward_signature bumps rground_formula_version).
#
# WHY NOT the simpler levers: steepening abstention_penalty was falsified 3x
# (P=0/0.2/0.4 dead on no-flow rate); concentrating on contested chunks would
# LOWER gold_base_rate and tighten the binding gate. Raise the winning side
# (Lever 2) + convert the present-but-unfollowed +0.72 advantage into movement
# (Lever 1). See the plan's "Why NOT" section.
#
# THE BINDING GATE is (d) no_flow_rate: |tail no-flow - gold_base(~0.31)| <= 0.15.
# Policy sits at ~0.69 (dev ~0.38). v8 must drive no-flow output ~0.69 -> <=0.46;
# then reward_trend turns positive automatically as the -0.4 penalties drop out.
#
# FALSIFIABLE PREDICTIONS (watch via scripts/grpo_field_metrics.py <run_dir>):
#   - abstain_given_gold_yes bends DOWN from flat ~0.62; tail no-flow -> <=0.46.
#   - Lever 1 shows in clip_ratio/high_mean (off ~1e-4) + grad_norm (off ~0.1).
#   - Lever 2 shows in rground_zero_frac_on_extractors (~0.35 -> <0.15) +
#     rground_mean_on_extractors (off 0.265).
#   - INSTABILITY GUARD (mu=2 is the new risk): entropy bounded (<~1.0), IS~1,
#     logp_diff flat, KL<1.0 — the v6 fingerprint surfaces in the first ~30-50
#     steps if mu destabilizes. summarize_log_history() correlations catch it.
#   - If the conditional STAYS flat -> the prior is SFT-baked / a labeling
#     artifact, and the next move is SFT-side, not another GRPO knob.
#
# COST: fresh prescreen (~4.5h, cache miss by design) + ~25-30h training
# (~520 steps / 3 epochs). Servers (embed:8001, judge:8002 = Qwen3.6-27B) must be
# up; server.env auto-sources their URLs + the source data paths.
#
# GPUs: pipeline node grpo_training uses slurm_train_1x (1 GPU, tp=1, vLLM
# colocate). GPU0 is fixed, so no spare-request override is needed.

set -euo pipefail

PROJECT_ROOT=/share/pierson/matt/UAIR
source "$PROJECT_ROOT/.venv/bin/activate"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
cd "$PROJECT_ROOT"

# Fresh prescreen cache for v8 (the symmetric clamp changes R_ground; the
# rground_formula_version bump already forces a cache MISS, but point at a new
# path so the v6/v7 cache stays pristine and versions never mix). Overridden via
# the hydra key (a literal that ships in the pickled config to the compute node),
# NOT an env var — submitit re-sources server.env on the node and would otherwise
# resolve GRPO_PRESCREEN_CACHE back to the old path.
python -m dagspaces.grpo_training.cli -m \
  pipeline=grpo_only_online_external \
  model=qwen3.5-9b/sft-contentless-v6 \
  training/grpo=online_rground_external \
  training.grpo.prescreen.cache_path=/share/pierson/matt/UAIR/cache/grpo_prescreen_v8.json \
  experiment.name=grpo_redesign_full_v8
