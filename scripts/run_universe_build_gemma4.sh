#!/bin/bash
# Gemma-4-teacher norm-universe build: structured_norms -> universe+embeddings.
#
# The gating prerequisite for the modular GRPO redesign (wiki/grpo_redesign/data.md):
# every reward module consumes norm_universes.json or its embeddings, and no
# Gemma-4-teacher universe exists for either corpus — the only built universes
# are qwen-era (teacher mismatch for gemma4-based training).
#
# Single step (per corpus): norm_universe_only on the corpus's
# structured_norms.parquet -> norm_universes.json + Qwen3-Embedding-8B embeddings
# (1 GPU). Role abstraction is deliberately SKIPPED — the gemma-4 extractor
# already emits name-free functional-role subjects, so the abstraction pass is
# redundant and (given its qwen-era name-stripping prompt) would degrade them.
# See the block above the universe call for the full rationale (review 2026-07-17).
#
# Usage (run under nohup; each CLI call blocks until its SLURM jobs finish):
#   nohup scripts/run_universe_build_gemma4.sh fiction10 \
#     > outputs/_launch_logs/universe_build_fiction10_gemma4.log 2>&1 &
#   nohup scripts/run_universe_build_gemma4.sh top100 \
#     > outputs/_launch_logs/universe_build_top100_gemma4.log 2>&1 &

set -euo pipefail

CORPUS="${1:?usage: $0 fiction10|top100}"
PROJECT_ROOT=/share/pierson/matt/UAIR

# Load site env (TRAWLER_DRIVER_VENV, SLURM_PARTITION, ...). ensure_dotenv() loads
# these inside the python CLI, but the driver *shell* needs TRAWLER_DRIVER_VENV
# here to pick the venv it runs from. The stage jobs do NOT rely on this shell's
# exports for the mirror: the orchestrator bakes `export TRAWLER_DRIVER_VENV=
# {sys.prefix}` into each stage's setup block (so it names whatever venv this
# driver runs from), and activate_stage_venv.sh derives the /scratch mirror path
# from that basename.
set -a; [[ -f "$PROJECT_ROOT/server.env" ]] && source "$PROJECT_ROOT/server.env"; set +a

# Run the driver from the vLLM-025/cu129 stack (the gemma-4 teacher venv), NOT the
# legacy .venv. This makes the submitit-baked sys.executable — the fallback the
# stage runs from when a node has no scratch mirror — the *correct* venv, and it
# matches the venv the /scratch mirror is built from (TRAWLER_SCRATCH_VENV), so a
# mirrored node (activate_stage_venv.sh -> TRAWLER_STAGE_PYTHON) runs the same
# interpreter from node-local disk. That mirror is the real fix for the earlier
# GPU-sanitize false positive: cold `import torch` was NFS-latency-bound (>60s
# probe timeout); from /scratch it's seconds, so the preflight probe now passes
# on its own and we no longer bypass it.
DRIVER_VENV="${TRAWLER_DRIVER_VENV:-$PROJECT_ROOT/.venv-vllm025cu129}"
source "$DRIVER_VENV/bin/activate"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
cd "$PROJECT_ROOT"

case "$CORPUS" in
  fiction10)
    EXTRACTION_PARQUET="$PROJECT_ROOT/outputs/2026-07-12_fiction10_norms_gemma4/18-36-28/COLM_norms_fiction_gemma4/outputs/extraction/structured_norms.parquet"
    ;;
  top100)
    EXTRACTION_PARQUET="$PROJECT_ROOT/outputs/2026-07-13_top100_norms_extraction_gemma4/16-23-09/COLM_norms_fiction_gemma4/outputs/extraction/structured_norms.parquet"
    ;;
  *)
    echo "ERROR: corpus must be fiction10 or top100, got '$CORPUS'" >&2; exit 1
    ;;
esac
[[ -f "$EXTRACTION_PARQUET" ]] || { echo "ERROR: missing $EXTRACTION_PARQUET" >&2; exit 1; }
echo "[universe_build:$CORPUS] structured_norms: $EXTRACTION_PARQUET"

# Paths go in as CONFIG OVERRIDES, not env vars: the submitit orchestrator job
# does not inherit this shell's exports, so ${oc.env:...} interpolations fall
# through to their (unset) defaults there — overrides are baked into the
# pickled config and survive. (Bit us on first launch, 2026-07-16.)

# ── Role abstraction is SKIPPED for the gemma-4 teacher (2026-07-17). ──
# The abstraction prompt + all its examples target raw character-name subjects
# (the qwen-era extraction: "Mrs. Fenwick", "Winston", ...). The gemma-4
# extractor already emits name-free functional-role subjects: on top100 only
# 0.52% of 53,494 norms carry ANY name flag, and those flags are dominated by
# NER false positives on titles/roles/places (Queen, Musketeer, Prophet, Sabbath,
# Eton, ...) — abstracting them would DEGRADE legitimate role content. Running the
# stage would also re-abstract 99.5% already-good subjects under its "ALWAYS
# rewrite/enrich" directive, drifting role granularity that the universe/battery
# clustering depends on. So structured_norms feeds the universe build directly;
# norm_universe's own dedup handles near-duplicates. (Review: 2026-07-17.)
#
# The universe stage (grpo_training/stages/norm_universe.py) reads only raz_*
# fields + gutenberg_id — all present in structured_norms.parquet — and filters
# solely on non-empty raz_norm_articulation, so this is a schema drop-in.

# ── Universe + embeddings (path as override, same trap as above). ──
# Single 1-GPU stage (slurm_gpu_1x default); no >1-GPU sanitize probe, but the
# node-local /scratch venv mirror still gives fast torch/vllm imports (both
# pierson nodes mirrored as of 2026-07-17; driver runs from .venv-vllm025cu129 so
# the orchestrator's TRAWLER_DRIVER_VENV export names the mirrored venv).
python -m dagspaces.grpo_training.cli \
  pipeline=norm_universe_only \
  model=qwen3.5-9b/sft-contentless-v6 \
  pipeline.sources.abstracted_norms.path="$EXTRACTION_PARQUET" \
  experiment.name=universe_${CORPUS}_gemma4

# grpo_training's hydra run dir defaults to multirun/ (not outputs/ like
# historical_norms); check both roots in case HYDRA_RUN_DIR overrides it.
UNIVERSE=$(ls -t \
  "$PROJECT_ROOT"/{outputs,multirun}/*_universe_${CORPUS}_gemma4/*/norm_universe_only/outputs/norm_universe/norm_universes.json \
  2>/dev/null | head -1 || true)
echo "[universe_build:$CORPUS] DONE. universe: ${UNIVERSE:-NOT FOUND (check step-2 logs)}"
