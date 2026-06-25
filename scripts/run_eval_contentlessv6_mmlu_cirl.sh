#!/bin/bash
# Gap-fill eval for the COLM main table after switching the Qwen3.5-9B SFT row
# to sft-contentless-v6 (the checkpoint v9-GRPO was actually initialized from).
#
# The v9 sweeps already give us contentless-v6 MATCHED against v9-ckpt100 on
# GoldCoin / ConfAIde / VLM-GeoPrivacy / PrivacyLens (SFT and GRPO ran in the
# same sweep, so they are mutually matched). Re-running those standalone would
# UN-match them from the GRPO sweep, so we KEEP the sweep numbers for those.
#
# The only table columns NOT yet covered for the contentless-v6 lineage are:
#   - MMLU  (capability/knowledge control; table SFT currently shows sft-ci 78.5)
#   - CIRL  (vignette completeness; table currently "--" for all finetuned rows)
# Both are GOLD-/programmatically-scored -> NO judge stage -> zero contention
# with the live v10 GRPO run on klara (judge:8002 / embed:8001). We disable the
# judge sidecar explicitly and disable every judged/expensive benchmark.
#
# Three arms run together so the new MMLU + CIRL columns are MATCHED across the
# 9B zero-shot -> SFT -> GRPO progression:
#   base                 zero-shot (Qwen3.5-9B-Base)
#   sft-contentless-v6   the SFT row (== v9-GRPO step 0)
#   v9-ckpt100           SFT + GRPO (v9)
#
# Runs on the pierson partition via slurm_monitor (NOT klara). See
# wiki/grpo_training_field_notes/2026-06-24_v10_plan.md for the v9/v10 context
# and CONGRUENCE.md (paper repo) for the contentless-v6-is-primary-SFT intent.

set -euo pipefail

PROJECT_ROOT=/share/pierson/matt/UAIR
source "$PROJECT_ROOT/.venv/bin/activate"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
cd "$PROJECT_ROOT"

python -m dagspaces.eval_all.cli -m \
  pipeline=all_benchmarks \
  model=qwen3.5-9b/base,qwen3.5-9b/sft-contentless-v6,qwen3.5-9b/v9-ckpt100 \
  benchmarks.goldcoin.enabled=false \
  benchmarks.privacylens.enabled=false \
  benchmarks.confaide.enabled=false \
  benchmarks.vlm_geoprivacy.enabled=false \
  benchmarks.cirl_vignettes.enabled=true \
  benchmarks.mmlu.enabled=true \
  judge_sidecar.enabled=false \
  experiment.name=eval_contentlessv6_mmlu_cirl
