#!/bin/bash
# Mirror the shared-NFS vLLM venv onto node-local /scratch for fast imports.
#
# Why: the venv holds ~83k mostly-small files. NFS sequential bandwidth is fine
# (~170 MB/s measured on klara) but per-file round trips are what dominate, so
# anything that walks the tree file-by-file (python imports, single-stream
# rsync) is latency-bound. Cold `import torch`+`import vllm`+`import
# flashinfer` from NFS costs ~13 min per process spawn (x3 processes per vLLM
# stage). From local XFS it's seconds.
#
# Fast path: a venv tarball on NFS (built by --make-tarball from an existing
# local mirror) is ONE sequential stream — ~2 min to deploy to a new node.
# Fallback: parallel rsync fan-out (NFS latency-bound work scales ~linearly
# with streams). Either way a final serial rsync --delete pass against the live
# NFS venv enforces exact 1:1 parity, and only then is the completion marker
# written — launchers refuse mirrors without it (scripts/activate_stage_venv.sh).
#
# Usage:
#   sync_venv_to_scratch.sh [SRC_VENV] [DST_VENV]   # deploy/refresh a mirror
#   sync_venv_to_scratch.sh --make-tarball [SRC_VENV] [DST_VENV]
#       # (re)build the NFS bootstrap tarball from a completed local mirror
#
# Safe to re-run any time the shared venv changes; incremental after first run.
set -euo pipefail

MAKE_TARBALL=0
if [ "${1:-}" = "--make-tarball" ]; then MAKE_TARBALL=1; shift; fi

SRC="${1:-/share/pierson/matt/UAIR/.venv-vllm025cu129}"
NAME="$(basename "$SRC" | sed 's/^\.//')"
DST="${2:-/scratch/$USER/venvs/$NAME}"
TARBALL="${SYNC_VENV_TARBALL:-$(dirname "$SRC")/.venv-mirrors/$NAME.tar.zst}"
JOBS="${SYNC_VENV_JOBS:-24}"

[ -d "$SRC" ] || { echo "sync_venv_to_scratch: SRC not found: $SRC" >&2; exit 1; }
case "$DST" in /scratch/*) ;; *) echo "sync_venv_to_scratch: DST must be under /scratch, got: $DST" >&2; exit 1;; esac

if [ "$MAKE_TARBALL" = 1 ]; then
  # Build from the LOCAL mirror (fast reads), never from NFS. Requires a
  # completed mirror so the tarball can't encode a half-synced tree.
  [ -f "$DST/.sync_complete" ] || { echo "sync_venv_to_scratch: no completed mirror at $DST — sync first" >&2; exit 1; }
  grep -q "src=$SRC\$" "$DST/.sync_complete" || { echo "sync_venv_to_scratch: mirror at $DST is not of $SRC" >&2; exit 1; }
  mkdir -p "$(dirname "$TARBALL")"
  echo "[sync_venv] building $TARBALL from $DST"
  start=$SECONDS
  tar -C "$DST" --exclude=./.sync_complete -cf - . | zstd -T0 -3 -q -o "$TARBALL.tmp" -f
  mv -f "$TARBALL.tmp" "$TARBALL"
  echo "[sync_venv] tarball done in $((SECONDS - start))s: $(du -sh "$TARBALL" | cut -f1)"
  exit 0
fi

mkdir -p "$DST"
echo "[sync_venv] $SRC -> $DST (jobs=$JOBS)"
start=$SECONDS

# A mirror without a marker is partial/stale garbage from an interrupted sync;
# a marker for a different src venv means the name collided. Start clean.
if [ -e "$DST/.sync_complete" ] && ! grep -q "src=$SRC\$" "$DST/.sync_complete"; then
  echo "sync_venv_to_scratch: $DST mirrors a different venv ($(cat "$DST/.sync_complete")) — refusing" >&2
  exit 1
fi
rm -f "$DST/.sync_complete"

if [ ! -x "$DST/bin/python" ] && [ -f "$TARBALL" ]; then
  # Fresh node + tarball available: one sequential NFS stream, then untar
  # locally. ~2 min instead of ~30 min of parallel per-file round trips.
  echo "[sync_venv] bootstrap from tarball $TARBALL ($(du -sh "$TARBALL" | cut -f1))"
  zstd -dc -T0 "$TARBALL" | tar -C "$DST" -xf -
  echo "[sync_venv] tarball extracted at ${SECONDS}s; reconciling against live venv"
else
  # No tarball (or refreshing an existing mirror): parallel rsync fan-out at
  # the deepest broad level — site-packages children (one rsync per package
  # dir) plus every non-site-packages venv entry.
  SP_REL="lib/python3.12/site-packages"
  {
    # venv top-level entries except lib (bin, include, pyvenv.cfg, ...)
    find "$SRC" -mindepth 1 -maxdepth 1 ! -name lib
    # lib subtree down to site-packages, exclusive
    find "$SRC/lib" -mindepth 1 -maxdepth 2 ! -path "$SRC/$SP_REL" ! -name python3.12
    # every package inside site-packages
    find "$SRC/$SP_REL" -mindepth 1 -maxdepth 1
  } | sed "s|^$SRC/||" \
    | xargs -P "$JOBS" -I{} rsync -a --relative "$SRC/./{}" "$DST/"
fi

# Serial parity pass against the live NFS venv: propagates deletions, drift
# since the tarball was built, and anything the fan-out missed.
rsync -a --delete --exclude=/.sync_complete "$SRC/" "$DST/"

# Stamp completion: launchers only trust the mirror when this marker exists,
# so a killed/partial sync can never be picked up as a working venv.
date -u +"%Y-%m-%dT%H:%M:%SZ src=$SRC" > "$DST/.sync_complete"

echo "[sync_venv] done in $((SECONDS - start))s, $(du -sh "$DST" | cut -f1) on $(hostname -s)"
