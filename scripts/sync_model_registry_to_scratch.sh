#!/usr/bin/env bash
# Mirror canonical zoo models to node-local /scratch (the model "registry").
#
# Same marker convention as sync_venv_to_scratch.sh / activate_stage_venv.sh:
# each mirror carries <mirror>/.sync_complete with a `src=<zoo path>` line,
# and dagspaces.common.model_registry redirects model loads to the mirror
# only when that marker matches. No marker (or a mismatched one) → stages
# fall back to the NFS zoo path, so a partial or interrupted sync is safe.
#
# Usage:
#   scripts/sync_model_registry_to_scratch.sh              # canonical set
#   scripts/sync_model_registry_to_scratch.sh NAME [...]   # specific zoo dirs
#
# Run once per node:
#   lisbeth:  bash scripts/sync_model_registry_to_scratch.sh
#   klara:    sbatch -p pierson -w klara -c 4 --mem=8G \
#               --wrap 'bash /share/pierson/matt/UAIR/scripts/sync_model_registry_to_scratch.sh'
#
# Zoo models are immutable once downloaded; if one IS ever replaced in place,
# re-run this script on every node (mirrors do not check freshness).
set -euo pipefail

ZOO="${TRAWLER_MODEL_ZOO:-/share/pierson/matt/zoo/models}"
REG="${TRAWLER_MODEL_REGISTRY:-/scratch/mwf62/registry/models}"

# Canonical COLM set: 11 SFT/GRPO backbones (instruct-tuned), their base
# (pre-instruct) variants where the zoo has one, and the aux servers.
CANONICAL=(
  Qwen3.5-2B
  Qwen3.5-4B
  Qwen3.5-9B
  Gemma-4-E2B-it
  Gemma-4-E4B-it
  gemma-4-12B-it
  OpenThinker3-7B
  Llama-3.1-8B-Instruct
  HARC-Llama-3.1-8B-Instruct
  Phi-4
  GPT-OSS-20B
  Qwen3.5-2B-Base       # base variants
  Qwen3.5-4B-Base
  Qwen3.5-9B-Base
  Gemma-4-E2B
  Gemma-4-E4B
  gemma-4-12B
  Llama-3.1-8B
  Gemma-4-31B-it        # R_ground judge server
  Qwen3-Embedding-8B    # embedding server
)

if [ "$#" -gt 0 ]; then MODELS=("$@"); else MODELS=("${CANONICAL[@]}"); fi

mkdir -p "$REG"
for name in "${MODELS[@]}"; do
  SRC="$ZOO/$name"
  DST="$REG/$name"
  if [ ! -d "$SRC" ]; then
    echo "sync_model_registry: skip $name — $SRC not found" >&2
    continue
  fi
  mkdir -p "$DST"
  if [ -f "$DST/.sync_complete" ] && ! grep -q "src=$SRC\$" "$DST/.sync_complete"; then
    echo "sync_model_registry: refusing $name — $DST mirrors a different source:" >&2
    sed 's/^/  /' "$DST/.sync_complete" >&2
    continue
  fi
  (
    flock -n 9 || { echo "sync_model_registry: skip $name — another sync holds the lock"; exit 0; }
    rm -f "$DST/.sync_complete"
    echo "sync_model_registry: syncing $name ..."
    rsync -a --delete --exclude=/.sync_complete --exclude=/.sync_lock "$SRC/" "$DST/"
    files=$(find "$DST" -type f ! -name '.sync_lock' | wc -l)
    bytes=$(du -sb "$DST" | cut -f1)
    {
      echo "src=$SRC"
      echo "host=$(hostname -s)"
      echo "date=$(date -Is)"
      echo "files=$files bytes=$bytes"
    } > "$DST/.sync_complete"
    echo "sync_model_registry: done $name ($files files, $bytes bytes)"
  ) 9>"$DST/.sync_lock"
done
echo "sync_model_registry: complete on $(hostname -s) → $REG"
