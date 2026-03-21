"""Shared stage utilities for all dagspace stage modules.

This module consolidates utility functions that were previously duplicated across
multiple stage files. Import from here rather than defining local copies.

Public API
----------
maybe_silence_vllm_logs()
    Configure vLLM logger levels based on environment variables.

to_json_str(value)
    Serialize an arbitrary Python value to a JSON string.

serialize_arrow_unfriendly_in_row(row, columns)
    In-place convert nested/non-serializable columns to JSON strings in a row dict.

extract_last_json(text)
    Extract the last JSON object or array from a text string.

sanitize_for_json(value)
    Recursively convert a value to JSON-serializable builtins.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Iterable, List, Optional

__all__ = [
    "ensure_dotenv",
    "maybe_silence_vllm_logs",
    "to_json_str",
    "serialize_arrow_unfriendly_in_row",
    "extract_last_json",
    "sanitize_for_json",
]

# Module-level guards so one-time setup work only runs once per process.
_DOTENV_LOADED = False
_VLLM_LOGS_SILENCED = False


def ensure_dotenv() -> None:
    """Load project-root ``.env`` and ``server.env`` into ``os.environ``.

    Walks up from this file's directory to find the project root, then
    loads both ``.env`` (pipeline inputs, checkpoint paths) and
    ``server.env`` (SLURM partition, project paths, NCCL settings).
    Existing env vars are NOT overwritten (``override=False``).
    Idempotent — subsequent calls are no-ops.
    """
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    try:
        from dotenv import load_dotenv as _load

        search_dir = os.path.dirname(os.path.abspath(__file__))
        # Walk up to repo root (max 5 levels: common/ -> dagspaces/ -> repo/)
        for _ in range(5):
            env_file = os.path.join(search_dir, ".env")
            if os.path.isfile(env_file):
                _load(env_file, override=False)
                # Also load server.env for SLURM/cluster settings
                server_env = os.path.join(search_dir, "server.env")
                if os.path.isfile(server_env):
                    _load(server_env, override=False)
                break
            search_dir = os.path.dirname(search_dir)
    except ImportError:
        pass
    _DOTENV_LOADED = True


def maybe_silence_vllm_logs() -> None:
    """Configure vLLM logger levels according to environment variables.

    Reads ``VLLM_LOGGING_LEVEL`` (default ``"WARNING"``) and, when
    ``RULE_TUPLES_SILENT`` is set, forces the level to ``ERROR`` and stops
    log propagation.  Also installs a ``PatternModuloFilter`` on the root
    ``vllm`` logger to throttle the high-frequency "Elapsed time for batch"
    message; the throttle period is controlled by ``UAIR_VLLM_LOG_EVERY``
    (default ``10``).

    The function is idempotent: subsequent calls after the first are no-ops.
    All exceptions are silently swallowed so that a missing ``vllm`` package
    does not break callers.
    """
    global _VLLM_LOGS_SILENCED
    if _VLLM_LOGS_SILENCED:
        return
    try:
        vllm_log_level = os.environ.get("VLLM_LOGGING_LEVEL", "WARNING")
        if os.environ.get("RULE_TUPLES_SILENT"):
            vllm_log_level = "ERROR"
            os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")

        _LEVEL_MAP = {
            "ERROR": logging.ERROR,
            "WARNING": logging.WARNING,
            "INFO": logging.INFO,
            "DEBUG": logging.DEBUG,
        }
        level = _LEVEL_MAP.get(vllm_log_level.upper(), logging.WARNING)

        for name in ("vllm", "vllm.logger", "vllm.engine", "vllm.core", "vllm.worker"):
            lg = logging.getLogger(name)
            lg.setLevel(level)
            if os.environ.get("RULE_TUPLES_SILENT"):
                lg.propagate = False
            else:
                lg.propagate = True

        # Install a modulo filter on the root vllm logger to throttle the
        # high-frequency "Elapsed time for batch" message.
        try:
            from dagspaces.uair.logging_filters import PatternModuloFilter  # type: ignore

            root_vllm = logging.getLogger("vllm")
            try:
                n = int(os.environ.get("UAIR_VLLM_LOG_EVERY", "10") or "10")
            except Exception:
                n = 10
            existing = getattr(root_vllm, "filters", [])
            if not any(
                getattr(f, "__class__", object).__name__ == "PatternModuloFilter"
                for f in existing
            ):
                root_vllm.addFilter(PatternModuloFilter(mod=n, pattern="Elapsed time for batch"))
        except Exception:
            pass

        _VLLM_LOGS_SILENCED = True
    except Exception:
        pass


def to_json_str(value: Any) -> Optional[str]:
    """Serialize *value* to a JSON string.

    Returns ``None`` when *value* is ``None``.  Falls back to ``str(value)``
    if ``json.dumps`` raises.

    Parameters
    ----------
    value:
        Any Python value.

    Returns
    -------
    str | None
        A JSON-encoded string, or ``None`` for ``None`` input.
    """
    try:
        if value is None:
            return None
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        try:
            return str(value)
        except Exception:
            return None


def serialize_arrow_unfriendly_in_row(
    row: Dict[str, Any],
    columns: Iterable[str],
) -> None:
    """In-place convert non-Arrow-serializable column values to JSON strings.

    Handles ``dict``, ``list``, ``tuple``, and vLLM ``GuidedDecodingParams`` /
    ``SamplingParams`` objects.  The vLLM types are imported lazily so that
    the function remains usable in environments where vLLM is not installed.

    Parameters
    ----------
    row:
        A ``dict`` representing one data row, mutated in place.
    columns:
        Names of columns to inspect and potentially serialize.
    """
    # Lazy import so we don't hard-require vllm at module load time.
    _GuidedDecodingParams = None
    _SamplingParams = None
    try:
        from vllm.sampling_params import GuidedDecodingParams, SamplingParams  # type: ignore

        _GuidedDecodingParams = GuidedDecodingParams
        _SamplingParams = SamplingParams
    except Exception:
        pass

    for col in columns:
        if col not in row:
            continue
        val = row[col]
        if isinstance(val, (dict, list, tuple)):
            row[col] = to_json_str(val)
        elif _GuidedDecodingParams is not None and isinstance(val, _GuidedDecodingParams):
            row[col] = str(val)
        elif _SamplingParams is not None and isinstance(val, _SamplingParams):
            row[col] = str(val)


def extract_last_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract the last JSON object from *text*.

    First attempts to parse the entire string.  On failure, uses a regex to
    locate all ``{…}`` blocks and tries each from the last one backwards.

    Parameters
    ----------
    text:
        Raw text that may contain one or more embedded JSON objects.

    Returns
    -------
    dict | None
        The parsed dict, or ``None`` if no valid JSON object was found.
    """
    try:
        if not isinstance(text, str) or not text.strip():
            return None
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        try:
            import re as _re

            snippets = _re.findall(r"\{[\s\S]*\}", text)
            for snip in reversed(snippets or []):
                try:
                    obj = json.loads(snip)
                    if isinstance(obj, dict):
                        return obj
                except Exception:
                    continue
        except Exception:
            pass
        return None
    except Exception:
        return None


def sanitize_for_json(value: Any) -> Any:
    """Recursively convert *value* to JSON-serializable Python builtins.

    Conversion rules:

    * ``None``, ``str``, ``int``, ``float``, ``bool`` — returned as-is.
    * vLLM ``GuidedDecodingParams`` — converted to a plain dict with the
      ``json``, ``disable_fallback``, and ``disable_additional_properties``
      attributes; falls back to ``str(value)`` on error.
    * vLLM ``SamplingParams`` — converted via ``str()``.
    * ``dict`` — keys coerced to ``str``; values recursed.
    * ``list``, ``tuple``, ``set`` — converted to ``list``; elements recursed.
    * Objects with a ``.tolist()`` method (e.g. numpy arrays) — called and recursed.
    * Everything else — ``str(value)``.

    Parameters
    ----------
    value:
        Any Python value.

    Returns
    -------
    Any
        A JSON-serializable representation, or ``None`` on unrecoverable error.
    """
    # Lazy import so we don't hard-require vllm at module load time.
    _GuidedDecodingParams = None
    _SamplingParams = None
    try:
        from vllm.sampling_params import GuidedDecodingParams, SamplingParams  # type: ignore

        _GuidedDecodingParams = GuidedDecodingParams
        _SamplingParams = SamplingParams
    except Exception:
        pass

    try:
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        if _GuidedDecodingParams is not None and isinstance(value, _GuidedDecodingParams):
            try:
                return {
                    "json": getattr(value, "json", None),
                    "disable_fallback": getattr(value, "disable_fallback", True),
                    "disable_additional_properties": getattr(
                        value, "disable_additional_properties", True
                    ),
                }
            except Exception:
                return str(value)
        if _SamplingParams is not None and isinstance(value, _SamplingParams):
            return str(value)
        if isinstance(value, dict):
            out: Dict[str, Any] = {}
            for k, v in value.items():
                try:
                    key = str(k)
                except Exception:
                    key = repr(k)
                out[key] = sanitize_for_json(v)
            return out
        if isinstance(value, (list, tuple, set)):
            return [sanitize_for_json(v) for v in value]
        tolist = getattr(value, "tolist", None)
        if callable(tolist):
            try:
                return sanitize_for_json(tolist())
            except Exception:
                pass
        return str(value)
    except Exception:
        return None
