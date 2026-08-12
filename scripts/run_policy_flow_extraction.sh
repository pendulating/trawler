#!/bin/bash
# Fiction-10 flow reasoning + extraction, run with the FINE-TUNED policies as
# the extractor instead of the Gemma-4-31B teacher.
#
# Plan: wiki/2026-08-05_distilled_grounding_plan.md
#
# WHY: §5.2 of the paper measures normative grounding on the teacher's flows —
# grounding reclassifies 30.9% of them. This re-runs the SAME pipeline with the
# SAME prompts on the SAME chunks, varying only the weights, so the per-arm
# reclassification rate is directly comparable to that 30.9%. The hypothesis is
# that the fine-tuned arms reclassify less, i.e. grounding was partly distilled
# into the weights.
#
# ARMS (in sweep order; `.hydra/overrides.yaml` in each subdir records which):
#   0  qwen3.5-9b/instruct         stock backbone, no adapter  <- the control
#   1  qwen3.5-9b/sft-canonical    + SFT LoRA
#   2  qwen3.5-9b/m2-full-ckpt450  + GRPO LoRA  (camera-ready GRPO)
#   3  qwen3.5-9b/k3-verdict       + KTO LoRA   (camera-ready KTO)
#
# Arm 0 is not optional: without it, a lower rate for arms 1-3 is
# indistinguishable from "Qwen judges appropriateness differently than Gemma".
#
# SERIAL BY DESIGN: this uses --multirun so all four arms land in ONE submitit
# array, and both launchers pin array_parallelism=1. Four concurrent arms would
# want 16 GPUs. Do NOT split this into four separate CLI invocations.
#
# Chunks: the exact fiction10 lineage every other fiction10 artifact descends
# from (COLM_fetch_fiction, 2026-03-20) — same file the teacher run consumed.
#
# Estimate: teacher was ~1h reasoning + ~1h extraction for 2,993 chunks at
# TP=2xDP=2 for a 31B dense. A 9B at TP=1xDP=4 should be materially faster;
# budget ~45min/stage/arm => ~6h for the sweep until measured.
#
# SMOKE=1 runs arm 2 only at sample_n=8, to shake out the DP+LoRA+eager combo
# and the 24576 context override before committing a night. Run it first.
#
# SANITY once stages start (plan §5.1):
#   * "PROMPT PROVENANCE: ci_reasoning_fiction" / "ci_extraction_fiction"
#   * "LoRA path resolved: '<...>'" — non-empty for arms 1-3, empty for arm 0.
#     A silently-empty lora_path turns three arms into three copies of the base.
#   * output parquets carry prompt_name == 'ci_{reasoning,extraction}_fiction'
#
# W&B: project historical-norms-extraction, run policy_flows_fiction10_<suffix>.

set -euo pipefail

PROJECT_ROOT=/share/pierson/matt/UAIR
source "${TRAWLER_DRIVER_VENV:-$PROJECT_ROOT/.venv-vllm025cu129}/bin/activate"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
cd "$PROJECT_ROOT"

# Same chunk file the 2026-07-12 gemma4 gold-label run consumed.
CHUNKS="${CHUNKS:-$PROJECT_ROOT/outputs/2026-03-20_historical_norms/23-05-10/COLM_fetch_fiction/outputs/fetch/chunks.parquet}"
if [[ ! -f "$CHUNKS" ]]; then
  echo "ERROR: chunk source not found: $CHUNKS" >&2
  exit 1
fi

# ── DO NOT select the corpus by exporting FICTION_CHUNKS_PATH ────────────────
# The export does NOT cross the submitit boundary. On the compute node the var
# is unset, ensure_dotenv() loads the project-root .env with override=False, and
# `.env:16` points FICTION_CHUNKS_PATH at chunks_top100_fiction_en.parquet. On
# 2026-08-06 that silently ran this sweep on top100 (15,875 chunks, 100 books)
# instead of fiction10 (2,993 / 10) for ~8 GPU-hours, while the driver's own
# echo showed the fiction10 path — the driver echo describes the DRIVER SHELL
# only. Any ${oc.env:VAR} in a pipeline yaml resolves on the compute node.
#
# A Hydra override IS serialized into the job, so it crosses. Hence:
CHUNKS_OVERRIDE="++pipeline.sources.prefetched_chunks.path=$CHUNKS"

# ── Preflight: assert the corpus, at launch rather than four hours in ────────
# The smoke test CANNOT catch a corpus mismatch (runtime.sample_n=8 truncates
# any corpus to 8 chunks), so this is the only guard that would have caught it.
EXPECT_CHUNKS="${EXPECT_CHUNKS:-2993}"
EXPECT_BOOKS="${EXPECT_BOOKS:-10}"
"$PROJECT_ROOT/.venv-vllm025cu129/bin/python" - "$CHUNKS" "$EXPECT_CHUNKS" "$EXPECT_BOOKS" <<'PYEOF' || exit 1
import sys
import pandas as pd
path, want_rows, want_books = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
df = pd.read_parquet(path, columns=["gutenberg_id"])
rows, books = len(df), df["gutenberg_id"].nunique()
print(f"[preflight] {path}")
print(f"[preflight] {rows} chunks / {books} books "
      f"(expect {want_rows} / {want_books})")
if (rows, books) != (want_rows, want_books):
    sys.exit(
        f"[preflight] CORPUS MISMATCH — refusing to launch. Override with "
        f"EXPECT_CHUNKS/EXPECT_BOOKS if this is deliberate."
    )
print("[preflight] corpus OK")
PYEOF

ARMS="qwen3.5-9b/instruct,qwen3.5-9b/sft-canonical,qwen3.5-9b/m2-full-ckpt450,qwen3.5-9b/k3-verdict"
EXTRA=()

if [[ "${SMOKE:-0}" == "1" ]]; then
  ARMS="qwen3.5-9b/m2-full-ckpt450"
  EXTRA+=(runtime.debug=true runtime.sample_n=8)
  echo "[policy-flows] SMOKE MODE: one arm, 8 chunks"
fi

echo "[policy-flows] chunks : $CHUNKS (passed as a Hydra override, not env)"
echo "[policy-flows] arms   : $ARMS"
echo "[policy-flows] pipeline: COLM_flows_fiction_policy (serial, array_parallelism=1)"

# The ABSOLUTE venv python, not the activated one: submitit bakes the monitor
# srun python from the driver sys.executable, and shell activation does not
# survive the hop. (Recorded in the m2 grid header; cost a launch once.)
"$PROJECT_ROOT/.venv-vllm025cu129/bin/python" \
  -m dagspaces.historical_norms.cli --multirun \
  pipeline=COLM_flows_fiction_policy \
  model="$ARMS" \
  "$CHUNKS_OVERRIDE" \
  experiment.name=policy_flows_fiction10 \
  "${EXTRA[@]}"
