"""Node-local model registry: redirect zoo model loads to a /scratch mirror.

The model zoo lives on NFS (``/share/pierson/matt/zoo/models``). Weight
loading is bandwidth-bound sequential reads paid on every vLLM engine
spin-up and every training start, so the canonical models are mirrored to
node-local /scratch by ``scripts/sync_model_registry_to_scratch.sh`` (one
mirror per node — the same pattern as the venv mirror). This module resolves
a model path to the local mirror when a completed, matching mirror exists on
the current node, and returns the original path otherwise.

Contract:

- Resolution happens at the LOAD BOUNDARY only (``from_pretrained`` /
  vLLM engine kwargs). Hydra configs, wandb records, and run metadata keep
  the canonical /share path.
- The mirror preserves the zoo basename, so every path-substring heuristic
  (harmony detection, reasoning-parser detection, AWQ detection, checkpoint
  naming) behaves identically on the resolved path.
- A mirror is trusted iff ``<mirror>/.sync_complete`` exists and contains a
  line ``src=<original path>`` (the ``activate_stage_venv.sh`` marker
  convention). Freshness is the sync script's responsibility — zoo models
  are effectively immutable once downloaded.
- The registry root comes from ``TRAWLER_MODEL_REGISTRY`` (set in
  ``server.env``). Unset/empty → this module is a no-op, so machines
  without a mirror are unaffected.
- Any probe failure falls back to the original path.
"""

from __future__ import annotations

import os

__all__ = ["resolve_model_source"]


def resolve_model_source(path, *, stage_name: str = "model_registry") -> str:
    """Return the node-local mirror of ``path`` if one is synced, else ``path``.

    Safe to call on anything model-shaped: HF hub ids, empty values, and
    paths outside the zoo simply pass through unchanged.
    """
    src = str(path or "")
    try:
        root = (os.environ.get("TRAWLER_MODEL_REGISTRY") or "").strip()
        if not root or not src.startswith("/"):
            return src
        src_norm = src.rstrip("/")
        if not os.path.isdir(src_norm):
            # Never redirect a source we can't see — a basename collision
            # with a stale mirror would otherwise load the wrong weights.
            return src
        mirror = os.path.join(root, os.path.basename(src_norm))
        marker = os.path.join(mirror, ".sync_complete")
        if not os.path.isfile(marker):
            return src
        with open(marker, "r", encoding="utf-8") as fh:
            lines = [ln.strip() for ln in fh.read().splitlines()]
        if f"src={src_norm}" not in lines:
            return src
        print(f"[{stage_name}] model registry: {src_norm} -> {mirror} "
              f"(node-local mirror)")
        return mirror
    except Exception as exc:  # pragma: no cover - defensive fallback
        print(f"[{stage_name}] model registry probe failed ({exc}); "
              f"using {src}")
        return src
