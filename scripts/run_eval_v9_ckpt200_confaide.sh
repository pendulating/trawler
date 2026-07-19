#!/bin/bash
# Judge-free ConfAIDE eval of v9 checkpoint-200 (epoch ~1.16). Entire
# confaide_eval pipeline is judge-free (tier 2a/2b Pearson r vs human ground
# truth; tier 3 rule-based string-match), so zero contention with the live v9
# run's judge:8002.
#
# ConfAIDE generation is temperature 0.0 (greedy/deterministic), so a SINGLE
# ckpt-200 arm is exactly comparable to the already-logged sft-contentless-v6 and
# v9-ckpt100 numbers from the identical pipeline — no need to re-run those arms.
#   arm 0  v9-ckpt200   _merged_sft + checkpoint-200 LoRA (epoch ~1.16)
#
# Headline: tier2a_pearson / tier2b_pearson for ckpt-200 vs the logged ckpt-100 —
# does the second epoch improve, hold, or erode agreement with human
# contextual-privacy expectations?

set -euo pipefail

PROJECT_ROOT=/share/pierson/matt/UAIR
source "${TRAWLER_DRIVER_VENV:-$PROJECT_ROOT/.venv-vllm025cu129}/bin/activate"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
cd "$PROJECT_ROOT"

python -m dagspaces.confaide.cli -m \
  pipeline=confaide_eval \
  model=qwen3.5-9b/v9-ckpt200 \
  experiment.name=confaide_v9_ckpt200
