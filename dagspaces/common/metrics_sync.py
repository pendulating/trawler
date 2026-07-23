"""Local ↔ W&B metrics parity utilities.

The contract this module enforces (see ``wiki/integrations/wandb-parity.md``):

1. **metrics.json is the wire format.** Whatever a benchmark writes to disk
   is mirrored to W&B *mechanically* — every numeric leaf, dotted keys
   byte-identical to the disk paths — under
   ``<subdir>/metrics_json/<dotted.key>`` where ``<subdir>`` is the
   metrics.json's parent directory name (``compute_metrics_tier2b``, …).
   Per-dagspace curated keys (``<stage>/eval/...``) are unchanged and remain
   for dashboards; the mirror is the parity/backup namespace. A notebook that
   reads disk ``(subdir, dotted_key)`` can therefore address the same cell in
   W&B without a per-benchmark key map.
2. **The file itself is uploaded** to the run (``wandb.save``), so W&B holds
   a byte-exact backup restorable by ``scripts/wandb_local_sync.py pull``.
3. **Linkage is bidirectional.** ``wandb_run.json`` next to each metrics.json
   records which run mirrors it; the run's config records
   ``local_output_dir``. Sweep identity (the W&B group) is derived from the
   hydra output dir when ``WANDB_GROUP`` is not set, so runs are *always*
   scopeable back to their ``multirun/<date>_<name>/<HH-MM-SS>`` directory —
   the gap that made W&B unusable as a source for the 2026-07-19
   per-checkpoint sweep analysis.
4. **Judge provenance travels.** The served judge is read from judge-batch
   manifests (never config, which carries a stale default) and attached as a
   ``judge:<model>`` tag.

Import discipline: this module must not import ``wandb_logger`` (which
imports it for the group fallback); it treats the logger as a duck-typed
object exposing ``log_metrics`` / ``save_file`` / ``add_tags`` /
``run_info`` / ``update_config``, all of which no-op when W&B is disabled.
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Any

# Path segment separating curated keys from the mechanical mirror. Also the
# marker `scripts/wandb_local_sync.py` uses to recognize mirrored keys.
MIRROR_SEGMENT = "metrics_json"

# Filename of the local → W&B linkage sidecar, written next to metrics.json.
SIDECAR_FILENAME = "wandb_run.json"

_DATE_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_.+")
_TIME_DIR_RE = re.compile(r"^\d{2}-\d{2}-\d{2}$")


# ---------------------------------------------------------------------------
# Flattening
# ---------------------------------------------------------------------------

def flatten_numeric(obj: Any, prefix: str = "") -> dict[str, float | int]:
    """Flatten every numeric leaf of a nested dict into dotted keys.

    The dotted keys are exactly the paths a reader of metrics.json would
    use (``qa_probing.accuracy``, ``per_question.Q7.accuracy``), so the
    W&B mirror and the disk file address cells identically. Booleans are
    logged as 0/1; strings, lists, and None are skipped (they live only in
    the uploaded file, which is the byte-exact backup).
    """
    flat: dict[str, float | int] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            dotted = f"{prefix}.{key}" if prefix else str(key)
            flat.update(flatten_numeric(value, dotted))
    elif isinstance(obj, bool):
        if prefix:
            flat[prefix] = int(obj)
    elif isinstance(obj, (int, float)):
        if prefix:
            flat[prefix] = obj
    return flat


# ---------------------------------------------------------------------------
# Sweep-group derivation
# ---------------------------------------------------------------------------

def derive_group_from_output_dir(output_dir: str | None) -> str | None:
    """Derive the W&B group ``<date>_<name>/<HH-MM-SS>`` from a hydra
    output path.

    Works for both layouts, at any depth below them::

        multirun/2026-07-19_eval_sft_per_checkpoint_all/22-48-47/<arm>/...
        outputs/2026-07-20_goldcoin_hipaa/10-15-33/...

    An eval_all child running in ``<sweep>/<time>/<arm>/<bench>/`` therefore
    derives the *same* group as its parent, even without the env var. Returns
    None when no ``<date>_<name>/<time>`` pair appears in the path.
    """
    if not output_dir:
        return None
    parts = os.path.normpath(str(output_dir)).split(os.sep)
    for i in range(len(parts) - 1):
        if _DATE_DIR_RE.match(parts[i]) and _TIME_DIR_RE.match(parts[i + 1]):
            return f"{parts[i]}/{parts[i + 1]}"
    return None


def resolve_hydra_runtime_dir() -> str | None:
    """Current hydra runtime output dir, or None outside hydra."""
    try:
        from hydra.core.hydra_config import HydraConfig

        hc = HydraConfig.get()
        if hc and hc.runtime and hc.runtime.output_dir:
            return str(hc.runtime.output_dir)
    except Exception:
        return None
    return None


def derive_default_group() -> str | None:
    """The group to use when neither cfg nor WANDB_GROUP provides one."""
    return derive_group_from_output_dir(resolve_hydra_runtime_dir())


# ---------------------------------------------------------------------------
# Judge provenance (artifact-derived, never config-derived)
# ---------------------------------------------------------------------------

def served_judges_near(metrics_json_path: str) -> set[str]:
    """Judge models recorded by judge-batch manifests in the same
    ``outputs/`` tree as a metrics.json.

    Reads ``<outputs>/*judge*batch*/manifest.json`` → ``model`` — the value
    ``judge_export.py`` resolved from the live server's ``/v1/models`` at
    export time. Deliberately not read from config: the config carries the
    stale ``${oc.env:JUDGE_MODEL,...}`` default and lies.
    """
    judges: set[str] = set()
    outputs_root = os.path.dirname(os.path.dirname(os.path.abspath(metrics_json_path)))
    if not os.path.isdir(outputs_root):
        return judges
    try:
        entries = os.listdir(outputs_root)
    except OSError:
        return judges
    for entry in entries:
        if "judge" not in entry or "batch" not in entry:
            continue
        manifest = os.path.join(outputs_root, entry, "manifest.json")
        if not os.path.isfile(manifest):
            continue
        try:
            with open(manifest) as f:
                model = json.load(f).get("model")
        except (ValueError, OSError):
            continue
        if model:
            judges.add(str(model).rstrip("/").split("/")[-1])
    return judges


# ---------------------------------------------------------------------------
# The mirror
# ---------------------------------------------------------------------------

def mirror_metrics_to_wandb(
    logger: Any,
    *,
    metrics: dict[str, Any] | None = None,
    metrics_json_path: str | None = None,
    stage: str = "",
) -> dict[str, float | int]:
    """Mirror one metrics.json (dict and/or file) into the active W&B run.

    Called by the shared orchestrator loop whenever a stage returns
    ``metadata["metrics"]`` / ``outputs["metrics_json"]`` — no per-dagspace
    code involved, which is the point: the cirl formatter silently logging
    nothing for months is the failure mode this replaces.

    - Scalars land under ``<subdir>/metrics_json/<dotted>`` (``<subdir>`` =
      the metrics.json parent dir name, falling back to ``stage``).
    - The file is uploaded so the run holds a byte-exact copy.
    - Judge tags are attached when judge-batch manifests exist alongside.
    - The linkage sidecar is written next to the file.

    Returns the flat dict that was logged (for tests / callers).
    """
    if metrics is None and metrics_json_path and os.path.isfile(metrics_json_path):
        try:
            with open(metrics_json_path) as f:
                metrics = json.load(f)
        except (ValueError, OSError) as exc:
            print(f"[metrics_sync] unreadable {metrics_json_path}: {exc}",
                  file=sys.stderr)
            metrics = None

    subdir = stage
    if metrics_json_path:
        subdir = os.path.basename(os.path.dirname(os.path.abspath(metrics_json_path))) or stage

    flat: dict[str, float | int] = {}
    if isinstance(metrics, dict):
        flat = {
            f"{subdir}/{MIRROR_SEGMENT}/{key}": value
            for key, value in flatten_numeric(metrics).items()
        }
        if flat:
            try:
                logger.log_metrics(flat)
            except Exception as exc:
                print(f"[metrics_sync] mirror log failed: {exc}", file=sys.stderr)

    if metrics_json_path and os.path.isfile(metrics_json_path):
        try:
            # base_path = the outputs/ root, so the run stores the file as
            # "<subdir>/metrics.json" — several stages of one (single_run)
            # pipeline coexist without clobbering each other.
            logger.save_file(
                metrics_json_path,
                base_path=os.path.dirname(os.path.dirname(os.path.abspath(metrics_json_path))),
            )
        except Exception as exc:
            print(f"[metrics_sync] file upload failed: {exc}", file=sys.stderr)

        judges = served_judges_near(metrics_json_path)
        if judges:
            try:
                logger.add_tags([f"judge:{j}" for j in sorted(judges)])
            except Exception as exc:
                print(f"[metrics_sync] judge tag failed: {exc}", file=sys.stderr)

        write_wandb_sidecar(os.path.dirname(metrics_json_path), logger)

    return flat


# ---------------------------------------------------------------------------
# Linkage sidecar (local → W&B)
# ---------------------------------------------------------------------------

def write_wandb_sidecar(output_dir: str, logger: Any) -> str | None:
    """Write ``wandb_run.json`` into *output_dir* recording the active run.

    Idempotent per (dir, run): re-mirroring the same run overwrites with the
    same content; a *different* run id appends to a ``previous_runs`` list
    rather than losing the old linkage (re-runs into the same dir are rare
    but must not silently orphan the first run).
    """
    info = None
    try:
        info = logger.run_info()
    except Exception:
        info = None
    if not info or not info.get("run_id"):
        return None

    path = os.path.join(output_dir, SIDECAR_FILENAME)
    previous: list[dict[str, Any]] = []
    if os.path.isfile(path):
        try:
            with open(path) as f:
                existing = json.load(f)
            if existing.get("run_id") == info["run_id"]:
                previous = existing.get("previous_runs", [])
            else:
                previous = existing.get("previous_runs", [])
                previous.append({k: v for k, v in existing.items()
                                 if k != "previous_runs"})
        except (ValueError, OSError):
            pass

    payload = dict(info)
    if previous:
        payload["previous_runs"] = previous
    try:
        os.makedirs(output_dir, exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        os.replace(tmp, path)
        return path
    except OSError as exc:
        print(f"[metrics_sync] sidecar write failed: {exc}", file=sys.stderr)
        return None
