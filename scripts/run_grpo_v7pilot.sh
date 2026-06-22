#!/bin/bash
# v7 PILOT — prove GRPO grows reward by fixing the optimizer, not the reward.
#
# DIAGNOSIS (from the v6 full run, job 537141):
#   The reward signal is CLEAN and STRONG — within gold-YES groups, extracting
#   the flow beats abstaining by +0.72 composite, in 100% of mixed groups. But
#   the policy never followed it (gold-YES abstention flat ~0.50 over all 537
#   steps; reward flat ~0.24; promotion gates: promote=false).
#
#   Why: with beta=0 the policy ENTROPY ran away ~10x (0.6 -> 6.0 nats). That
#   pushed the vLLM-rollout vs HF-trainer logprob mismatch up ~80x (0.09 -> 7.5),
#   which collapsed the importance-sampling ratios (min -> 1e-15) so token_truncate
#   MASKED the gradient. The clean +0.72 advantage never reached the weights.
#   Evidence: corr(entropy, logp_diff)=+0.92, corr(entropy, IS_ratio)=-0.96.
#
# HYPOTHESIS:
#   A MODERATE KL anchor (beta>0) bounds the entropy runaway. Because the KL
#   term penalizes the huge-KL entropy explosion far more than the small-KL
#   abstain->extract shift, it should stabilize rollout/trainer agreement
#   (logp_diff stays small -> IS ratios ~1 -> token_truncate stops masking),
#   letting the clean gradient flow. Then reward should GROW.
#
#   NB: beta and token_truncate were never combined before. v4 ran beta=0.01 but
#   with the old sequence_mask (whole-sequence zeroing); v5/v6 fixed that with
#   token_truncate but ran beta=0. This pilot tests the untried stable regime.
#
# SINGLE-VARIABLE CHANGE vs v6: beta 0.0 -> 0.02. Everything else identical
#   (same contentless-curated SFT base, same reward, token_truncate, eps_high
#   0.28, scale_rewards=none, app_weight 0.3, abstention_penalty 0.4).
#
# FALSIFIABLE TEST (watch over 150 steps):
#   SUCCESS  -> entropy stays bounded, logp_diff stays small, reward CLIMBS,
#              gold-YES abstention FALLS.
#   beta too high (over-anchored to abstaining SFT) -> entropy bounded but
#              reward flat + abstention frozen -> lower beta.
#   beta too low -> entropy still runs away -> raise beta.
#
# Same SFT base => prescreen CACHE HIT (beta is not in the cache key) => no
# 4.5h re-screen; straight into ~150 steps (~8h). Servers (embed:8001,
# judge:8002 = Qwen3.6-27B) must be up; server.env auto-sources URLs + cache.

set -euo pipefail

PROJECT_ROOT=/share/pierson/matt/UAIR
source "$PROJECT_ROOT/.venv/bin/activate"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
cd "$PROJECT_ROOT"

# GPU request: the job needs 1 GPU (vllm colocate, tp=1). Request 2 (one spare)
# so that if SLURM pulls in the busted klara GPU0, the orchestrator's GPU
# sanitize probe (a real matmul per device in a subprocess) drops it and trains
# on the healthy GPU. slurm_train_1x had no fallback -> 717862 landed on GPU0
# and died with "CUDA-capable device(s) is/are busy or unavailable".
python -m dagspaces.grpo_training.cli -m \
  pipeline=grpo_only_online_external \
  model=qwen3.5-9b/sft-contentless-v6 \
  training/grpo=online_rground_external \
  training.grpo.beta=0.02 \
  +training.grpo.max_steps=150 \
  pipeline.graph.nodes.grpo_training.launcher=slurm_train_2x \
  experiment.name=grpo_v7pilot_beta
