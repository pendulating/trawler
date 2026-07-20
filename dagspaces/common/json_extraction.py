"""Canonical JSON-from-LLM-text extraction.

One well-tested extractor replaces the ≥9 divergent implementations that
sprawled across the codebase (see wiki/jul19_refactoring.md, Finding 8).

Algorithm
---------
1. **Fast path** — ``json.loads(text)`` on the full string (clean output).
2. **Outermost span** — slice from the first ``{`` to the last ``}`` and
   parse.  This is what every prior implementation actually did: the old
   ``extract_last_json`` regex ``\\{[\\s\\S]*\\}`` is greedy and always
   produces one match identical to the ``find("{")/rfind("}")`` slice.
3. **Optional repair** — when *repair* is ``True`` and ``json_repair`` is
   installed, a failed parse is retried through
   ``repair_json(…, return_objects=True)``.  Previously only available in
   the ``historical_norms`` variant.

Return convention
-----------------
``(dict | None, str | None)`` — the parsed object and an error message
(``None`` on success).  Callers that only need the dict index ``[0]``.
"""

from __future__ import annotations

import json
from typing import Any

try:
    from json_repair import repair_json as _repair_json

    _JSON_REPAIR_OK = True
except ImportError:  # pragma: no cover
    _JSON_REPAIR_OK = False

__all__ = ["extract_json_from_text", "extract_last_json"]


def _try_repair(text: str) -> tuple[dict[str, Any] | None, str | None]:
    """Attempt ``json_repair`` on *text*; return ``(dict, None)`` or ``(None, err)``."""
    if not _JSON_REPAIR_OK:
        return None, "json_repair not installed"
    try:
        repaired = _repair_json(text, return_objects=True)
        if isinstance(repaired, dict):
            return repaired, None
        return None, f"json_repair returned {type(repaired).__name__}, not dict"
    except Exception as exc:
        return None, f"json_repair failed: {exc}"


def extract_json_from_text(
    text: str,
    *,
    repair: bool = False,
) -> tuple[dict[str, Any] | None, str | None]:
    """Extract a JSON object from LLM-generated *text*.

    Parameters
    ----------
    text:
        Raw model output that may contain a JSON object wrapped in prose,
        ``<think>`` blocks, or markdown fences.
    repair:
        If ``True``, fall back to ``json_repair`` when ``json.loads``
        fails on the candidate span.

    Returns
    -------
    (dict | None, str | None)
        ``(parsed_dict, None)`` on success, ``(None, error_message)``
        on failure.
    """
    if not isinstance(text, str) or not text.strip():
        return None, "empty or non-string input"

    # ── Fast path: the whole string is valid JSON ──────────────────────
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj, None
    except (json.JSONDecodeError, ValueError):
        pass

    # ── Outermost {…} span ─────────────────────────────────────────────
    start = text.find("{")
    end = text.rfind("}") + 1
    if start < 0 or end <= start:
        return None, "no JSON object found in text"

    candidate = text[start:end]
    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj, None
        err = f"parsed {type(obj).__name__}, not dict"
    except (json.JSONDecodeError, ValueError) as exc:
        err = str(exc)

    # ── Optional repair ────────────────────────────────────────────────
    if repair:
        repaired, repair_err = _try_repair(candidate)
        if repaired is not None:
            return repaired, None
        err = repair_err or err

    return None, err


def extract_last_json(text: str) -> dict[str, Any] | None:
    """Extract a JSON object from *text*.

    Backward-compatible wrapper around :func:`extract_json_from_text`.
    Returns the dict or ``None``.

    .. note:: Despite the historical name, this extracts the *outermost*
       ``{…}`` span (first ``{`` to last ``}``), which is what the old
       regex-based implementation actually did — the greedy
       ``\\{[\\s\\S]*\\}`` pattern always matched the full span.
    """
    return extract_json_from_text(text)[0]
