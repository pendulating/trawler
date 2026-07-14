#!/bin/bash
#SBATCH --job-name=matt-top100-norms-ext
#SBATCH --partition=pierson
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --output=/share/pierson/matt/UAIR/.bench/top100_resume_2026_07_13/norms_extraction_%j.log
#
# RESUME top100 norms EXTRACTION from the completed reasoning parquet.
#
# Why: the 2026-07-12 overnight chain ran inside SLURM session job 845966, which
# timed out at 09:52 and killed the driver. The reasoning job had already been
# submitted and survived to completion (15,875/15,875 chunks); extraction was
# never submitted. The orchestrator has no node-level resume, so re-running the
# full pipeline would redo ~12h of reasoning. This runs extraction only.
#
# This script IS the driver — it is itself an sbatch job (CPU-only), so it does
# not die with an interactive session. The GPU work is submitted from here by
# submitit (launcher: slurm_gpu_4x) and will queue for GPUs on its own.
#
# Submit:
#   sbatch scripts/run_resume_top100_norms_extraction_gemma4.sh

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

# The reasoning output the dead chain left behind. Complete and validated:
# 15,875 rows == 15,875 source chunks, prompt_name == norm_reasoning_fiction.
REASONING_PATH="${REASONING_PATH:-$PROJECT_ROOT/outputs/2026-07-13_top100_norms_gemma4/04-30-49/COLM_norms_fiction_gemma4/outputs/reasoning/reasoning.parquet}"
if [[ ! -f "$REASONING_PATH" ]]; then
  echo "ERROR: reasoning parquet not found: $REASONING_PATH" >&2
  exit 1
fi
export REASONING_PATH

echo "[top100_norms_ext] reasoning : $REASONING_PATH"
echo "[top100_norms_ext] model     : gemma-4-31b/instruct"
echo "[top100_norms_ext] pipeline  : COLM_norms_extraction_from_reasoning_gemma4"
echo "[top100_norms_ext] NOTE: this run picks up the FIXED character blocklist"
echo "[top100_norms_ext]       (name_detection.AMBIGUOUS_NAMES, 2026-07-13), so"
echo "[top100_norms_ext]       norm_quality_passed is trustworthy here where"
echo "[top100_norms_ext]       fiction10's was ~85% modal-verb false positives."

python -m dagspaces.historical_norms.cli \
  pipeline=COLM_norms_extraction_from_reasoning_gemma4 \
  model=gemma-4-31b/instruct \
  experiment.name=top100_norms_extraction_gemma4
