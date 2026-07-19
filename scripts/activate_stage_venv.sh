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
