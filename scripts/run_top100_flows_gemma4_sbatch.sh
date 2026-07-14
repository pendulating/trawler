#!/bin/bash
#SBATCH --job-name=matt-top100-flows
#SBATCH --partition=pierson
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --output=/share/pierson/matt/UAIR/.bench/top100_resume_2026_07_13/flows_%j.log
#
# top100 FLOWS (ci_reasoning + ci_extraction), Gemma-4 gold labels.
#
# Never started: the 2026-07-12 overnight chain died with SLURM session job
# 845966 before reaching it. This is a full run of both flow stages.
#
# This script IS the driver — itself an sbatch job (CPU-only), so it cannot die
# with an interactive session the way the original chain did. GPU work is
# submitted from here by submitit (launcher: slurm_gpu_4x) and queues on its own.
#
# Submit:
#   sbatch scripts/run_top100_flows_gemma4_sbatch.sh
#
# SANITY: confirm the stage logs print `PROMPT PROVENANCE: ci_reasoning_fiction`
# / `ci_extraction_fiction` and that output parquets carry the matching
# prompt_name. Every fiction flows run before 2026-07-12 silently used the
# PRESCRIPTIVE prompt.

set -euo pipefail

# PATH, and why the order matters. Two separate traps, both hit on 2026-07-13:
#
#  1. ~/.local/bin FIRST — the SLURM clients here are ssh-forwarding shims
#     (compute nodes have the binaries but no slurm.conf/munge, so they forward
#     to unicorn-login-04). An sbatch job does not get ~/.local/bin on PATH, and
#     without it the driver dies with `FileNotFoundError: 'squeue'`.
#
#  2. /usr/local/slurm/current/bin LAST — there is NO `srun` shim, and submitit's
#     backend detection is `-1 if shutil.which("srun") is None else 2`. With no
#     srun on PATH, AutoExecutor SILENTLY falls back to its LOCAL executor and
#     runs the GPU stage as a subprocess on this CPU node. No error — just a 31B
#     model with CUDA_VISIBLE_DEVICES=''. The native dir supplies a real `srun`
#     for the check; it is never executed (slurm_use_srun=False). It must come
#     LAST so the working shims still win for sbatch/squeue/scancel.
export PATH="$HOME/.local/bin:$PATH:/usr/local/slurm/current/bin"

# Preflight: refuse to run rather than degrade to CPU.
if ! command -v srun >/dev/null; then
  echo "FATAL: srun not on PATH — submitit would fall back to its local executor" >&2
  echo "       and run this GPU stage on the CPU. Refusing to start." >&2
  exit 1
fi
command -v squeue >/dev/null || { echo "FATAL: squeue not on PATH" >&2; exit 1; }

PROJECT_ROOT=/share/pierson/matt/UAIR
source "$PROJECT_ROOT/.venv/bin/activate"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
cd "$PROJECT_ROOT"

mkdir -p "$PROJECT_ROOT/.bench/top100_resume_2026_07_13"

FICTION_CHUNKS_PATH="${FICTION_CHUNKS_PATH:-/share/pierson/matt/zoo/datasets/gutenberg_cache/chunks_top100_fiction_en.parquet}"
if [[ ! -f "$FICTION_CHUNKS_PATH" ]]; then
  echo "ERROR: chunk source not found: $FICTION_CHUNKS_PATH" >&2
  exit 1
fi
export FICTION_CHUNKS_PATH

echo "[top100_flows] chunks : $FICTION_CHUNKS_PATH"
echo "[top100_flows] model  : gemma-4-31b/instruct"
echo "[top100_flows] pipeline: COLM_flows_fiction_prefetched_gemma4"
echo "[top100_flows] NOTE: flow_quality_passed was REMOVED 2026-07-13 (it enforced"
echo "[top100_flows]       a role requirement the CI prompt never stated). Do not"
echo "[top100_flows]       expect those columns in the output."

python -m dagspaces.historical_norms.cli \
  pipeline=COLM_flows_fiction_prefetched_gemma4 \
  model=gemma-4-31b/instruct \
  experiment.name=top100_flows_gemma4
