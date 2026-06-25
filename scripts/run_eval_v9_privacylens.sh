#!/bin/bash
# JUDGE-BASED PrivacyLens eval of v9 GRPO checkpoint-100 vs the base + SFT arms.
# This is the one remaining judge-based benchmark (leakage + helpfulness are
# scored by the external judge server, Qwen3.6-27B @ klara:8002). The judge
# server (job 725605) is already up; v9 training is cancelled so klara GPUs are
# free for the per-arm qa_probe + agent_action inference.
#
# Three arms on the SAME Qwen3.5-9B, isolating each stage of the pipeline:
#   arm 0  instruct             raw Qwen3.5-9B instruct (NO SFT, NO GRPO) — "base"
#   arm 1  sft-contentless-v6   instruct + contentless-v6 SFT LoRA        — "SFT"
#   arm 2  v9-ckpt100           SFT(merged) + v9 GRPO LoRA (epoch ~0.58)  — candidate
# All three share thinking-off / max_model_len 16384, so the comparison is clean.
#
# Headline (paper-ready bar, per the decision on 2026-06-24): does v9-ckpt100
# BEAT base AND SFT on PrivacyLens? The privacy metric is
# leakage.leakage_rate_among_parseable (LOWER is better — less sensitive
# information disclosed in the agent's final action); the utility metric is
# helpfulness.helpful_rate_among_parseable (higher is better). A win = lower
# leakage at comparable-or-better helpfulness. Run on the FULL PrivacyLens
# split (max_examples=0). All arms judged by the same server in one sweep, so
# the judge environment is identical across arms.

set -euo pipefail

PROJECT_ROOT=/share/pierson/matt/UAIR
source "$PROJECT_ROOT/.venv/bin/activate"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
cd "$PROJECT_ROOT"

# Pin the judge to the live server (also in server.env; set explicitly to be safe).
export JUDGE_SERVER_URL="http://klara:8002"

python -m dagspaces.privacylens.cli -m \
  pipeline=privacylens_clean \
  model=qwen3.5-9b/instruct,qwen3.5-9b/sft-contentless-v6,qwen3.5-9b/v9-ckpt100 \
  experiment.name=privacylens_v9_vs_sft_vs_base
