#!/bin/bash
# Judge-free VLM-GeoPrivacy MCQ eval — epoch-2 out-of-domain drift check for v9
# checkpoint-200. mcq_eval is sklearn MCQ accuracy (no judge), zero contention
# with the live v9 run's judge:8002.
#
# MCQ generation is temperature 0.2 / top_p 0.95 (samples), so the matched pair
# runs in ONE sweep to share the sampling environment:
#   arm 0  v9-ckpt100   _merged_sft + checkpoint-100 LoRA (epoch ~0.58)
#   arm 1  v9-ckpt200   _merged_sft + checkpoint-200 LoRA (epoch ~1.16)
# (SFT <-> ckpt-100 already established flat in the prior geoprivacy_v9_vs_sft run.)
#
# Question: does one more epoch of text-CI fine-tuning — during which entropy
# started drifting up — begin to harm visual-geoprivacy CI choices, or is
# transfer still flat? LoRA touches language layers only; vision tower untouched.

set -euo pipefail

PROJECT_ROOT=/share/pierson/matt/UAIR
source "$PROJECT_ROOT/.venv/bin/activate"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
cd "$PROJECT_ROOT"

python -m dagspaces.vlm_geoprivacy_bench.cli -m \
  pipeline=mcq_eval \
  model=qwen3.5-9b/v9-ckpt100,qwen3.5-9b/v9-ckpt200 \
  experiment.name=geoprivacy_v9_ckpt200_vs_ckpt100
