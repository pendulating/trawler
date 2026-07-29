# Sourced from SLURM launcher setup blocks (not executed) — picks the fastest
# venv available on the node a stage job landed on.
#
# If this node holds a *complete* /scratch mirror of the venv the driver runs
# from (see scripts/sync_venv_to_scratch.sh), activate it and export
# TRAWLER_STAGE_PYTHON so the submitit command line (see
# dagspaces/common/orchestrator.py::_create_submitit_executor) launches the
# stage from node-local disk. Cold torch/vllm/flashinfer imports over NFS cost
# ~13 min per process spawn; from local disk, seconds.
#
# Fallbacks are deliberately conservative — on any doubt (no mirror, partial
# sync, mirror of a *different* venv, TRAWLER_DRIVER_VENV unset) we leave
# TRAWLER_STAGE_PYTHON unset, and the stage runs exactly as it always has,
# from the driver's own interpreter over NFS.

_scratch_venv="${TRAWLER_SCRATCH_VENV:-}"
if [ -z "$_scratch_venv" ] && [ -n "${TRAWLER_DRIVER_VENV:-}" ]; then
  # Mirror naming convention of sync_venv_to_scratch.sh: leading dot stripped.
  _scratch_venv="/scratch/$USER/venvs/$(basename "$TRAWLER_DRIVER_VENV" | sed 's/^\.//')"
fi

if [ -n "$_scratch_venv" ] \
   && [ -f "$_scratch_venv/.sync_complete" ] \
   && [ -x "$_scratch_venv/bin/python" ] \
   && [ -n "${TRAWLER_DRIVER_VENV:-}" ] \
   && grep -q "src=$TRAWLER_DRIVER_VENV\$" "$_scratch_venv/.sync_complete"; then
  source "$_scratch_venv/bin/activate"
  export TRAWLER_STAGE_PYTHON="$_scratch_venv/bin/python"
  echo "[stage_venv] node-local venv on $(hostname -s): $_scratch_venv ($(cat "$_scratch_venv/.sync_complete"))"
else
  echo "[stage_venv] no matching scratch mirror on $(hostname -s) (driver venv: ${TRAWLER_DRIVER_VENV:-unset}); stage runs from NFS"
fi

unset _scratch_venv

# ── FFmpeg shared libs for torchcodec (2026-07-24) ─────────────────────
# Prepend a self-contained FFmpeg 7.1 lib dir so torchcodec's dlopen of
# libtorchcodec_coreN.so resolves libav*.so.N here instead of failing (klara
# lacks FFmpeg 4-7; system FFmpeg 8 hits a GLIBCXX_3.4.32 wall from the
# anaconda-base libstdc++). These libav*.so have no libstdc++ dependency, so
# they load cleanly. LD_LIBRARY_PATH is consulted by ld.so at process spawn,
# so this MUST run in the shell setup block BEFORE the stage python launches
# (it does — every launcher sources this file in `setup:`). Prepend ONLY the
# ffmpeg dir; idempotent guard avoids duplicate entries across nested sources.
_ff_libdir="${TRAWLER_FFMPEG_LIBDIR:-/share/pierson/matt/zoo/ffmpeg-libs/n7.1/lib}"
if [ -d "$_ff_libdir" ] && [ -e "$_ff_libdir/libavutil.so.59" ]; then
  case ":${LD_LIBRARY_PATH:-}:" in
    *":$_ff_libdir:"*) : ;;                       # already present — no-op
    *) export LD_LIBRARY_PATH="$_ff_libdir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
  esac
fi
unset _ff_libdir
