"""Shared scaffolding for static HTML inspector tools.

Factors CSS, core JS, HTML skeleton, and Python helpers out of
`completion_inspector.py` / `norms_inspector.py` so the SFT and GRPO
inspectors (and future tools) stay small.

Each inspector provides:
  - stage data dict:  {stage_key: {rows: [...], id_cols: [...], ...}}
  - an overview HTML blob (pre-rendered, tool-specific aggregate view)
  - a `buildRowBody(row, stageData)` JS function body (tool-specific)
  - optional `getRowBadges(row, stageData)` hook
  - optional `getRowIdLine(row, stageData)` hook
  - optional extra CSS/JS

Then calls `render_page(...)` to assemble the full HTML document.

The core JS handles: stage switching, field-filter discovery, search,
row cards, virtualized rendering, keyboard nav, bookmarks panel, export
modal (HTML + PDF).
"""

from __future__ import annotations

import html as html_lib

import json
from pathlib import Path
from typing import Any

import numpy as np


# ── Python helpers ────────────────────────────────────────────────────────


def _esc(s: Any) -> str:
    """HTML-escape a value for inclusion in an inspector page."""
    return html_lib.escape(str(s))


def _serialize(v: Any) -> Any:
    """Make a value JSON-serializable (numpy scalars, arrays, bytes, etc)."""
    if v is None:
        return None
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return None
    if isinstance(v, np.ndarray):
        return [_serialize(x) for x in v]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, dict):
        return {k: _serialize(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_serialize(x) for x in v]
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    return v


def resolve_root(run_path: str | Path) -> Path:
    """Resolve a run root, descending into a `0/` multirun subdir if present."""
    p = Path(run_path)
    if (p / "0").is_dir():
        return p / "0"
    return p


def parse_run_arg(arg: str) -> tuple[str, str]:
    """Parse 'Label=/path' or '/path' into (label, path)."""
    if "=" in arg:
        label, path = arg.split("=", 1)
        return label.strip(), path.strip()
    p = Path(arg)
    return p.name, arg


def parse_row_slice(spec: str, n: int) -> list[int]:
    """Parse Python slice syntax into a deduplicated list of row indices.

    Supports::

        "42"          -> [42]
        "10:20"       -> [10, 11, ..., 19]
        ":50"         -> [0, 1, ..., 49]
        "100:"        -> [100, 101, ..., n-1]
        "::2"         -> [0, 2, 4, ...]
        "-10:"        -> the last 10 rows
        "0:100:5"     -> [0, 5, 10, ..., 95]
        "0,5,10,42"   -> [0, 5, 10, 42]
        "0:10,50:60"  -> [0..9, 50..59]

    Args:
        spec: The row specification.
        n: The number of rows available.

    Returns:
        A sorted list of valid row indices, without duplicates.

    Raises:
        ValueError: if the spec is malformed, or gives no valid index.
    """
    indices: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            pieces = part.split(":")
            if len(pieces) > 3:
                raise ValueError(
                    f"Invalid slice '{part}': too many colons "
                    f"(the format is start:stop:step)"
                )
            try:
                args = [int(p) if p.strip() else None for p in pieces]
            except ValueError:
                raise ValueError(f"Invalid slice '{part}': non-integer component")
            sl = slice(*args)
            resolved = range(*sl.indices(n))
            if len(resolved) == 0:
                raise ValueError(
                    f"Slice '{part}' produces no rows (the dataset has {n})"
                )
            indices.update(resolved)
        else:
            try:
                idx = int(part)
            except ValueError:
                raise ValueError(f"Invalid index '{part}': not an integer")
            resolved_idx = idx if idx >= 0 else n + idx
            if resolved_idx < 0 or resolved_idx >= n:
                raise ValueError(
                    f"Index {idx} is out of range (the dataset has {n} rows; "
                    f"the valid range is {-n}..{n - 1})"
                )
            indices.add(resolved_idx)
    if not indices:
        raise ValueError(f"Row spec '{spec}' produced no indices")
    return sorted(indices)


def extract_user_prompt(messages: Any) -> str | None:
    """Pull the last user-role message content from a messages list."""
    if messages is None:
        return None
    msgs = list(messages) if isinstance(messages, np.ndarray) else messages
    if isinstance(msgs, str):
        try:
            msgs = json.loads(msgs)
        except Exception:
            return msgs
    if not isinstance(msgs, list):
        return None
    user_msgs = [m for m in msgs if isinstance(m, dict) and m.get("role") == "user"]
    if not user_msgs:
        return None
    content = user_msgs[-1].get("content", "")
    if isinstance(content, list):
        parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
        return "\n".join(parts) if parts else str(content)
    return str(content)


def extract_assistant_content(messages: Any) -> str | None:
    """Pull the last assistant-role message content from a messages list."""
    if messages is None:
        return None
    msgs = list(messages) if isinstance(messages, np.ndarray) else messages
    if isinstance(msgs, str):
        try:
            msgs = json.loads(msgs)
        except Exception:
            return None
    if not isinstance(msgs, list):
        return None
    asst = [m for m in msgs if isinstance(m, dict) and m.get("role") == "assistant"]
    if not asst:
        return None
    content = asst[-1].get("content", "")
    if isinstance(content, list):
        parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
        return "\n".join(parts) if parts else str(content)
    return str(content)


# ── Shared CSS ────────────────────────────────────────────────────────────

COMMON_CSS = r"""
:root {
  --bg: #fafafa; --fg: #1a1a1a; --border: #e0e0e0;
  --accent: __ACCENT_COLOR__; --accent-light: __ACCENT_LIGHT__;
  --green: #2e7d32; --green-bg: #e8f5e9;
  --red: #c62828; --red-bg: #ffebee;
  --orange: #ef6c00; --orange-bg: #fff3e0;
  --blue: #1565c0; --blue-bg: #e3f2fd;
  --purple: #6a1b9a; --purple-bg: #f3e5f5;
  --mono: 'SF Mono', 'Cascadia Code', 'Fira Code', Consolas, monospace;
  --sans: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: var(--sans); background: var(--bg); color: var(--fg); font-size: 14px; }

/* ── Top bar ─────────────────────────────────────────────── */
.topbar {
  position: sticky; top: 0; z-index: 100;
  background: #fff; border-bottom: 1px solid var(--border);
  padding: 8px 16px; display: flex; flex-wrap: wrap; gap: 8px; align-items: center;
  box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.topbar select, .topbar input, .topbar button {
  font-size: 13px; padding: 4px 8px; border: 1px solid var(--border);
  border-radius: 4px; background: #fff;
}
.topbar select { max-width: 320px; }
.topbar input[type="text"] { width: 220px; }
.topbar button {
  cursor: pointer; background: var(--accent); color: #fff; border: none;
  padding: 5px 12px; border-radius: 4px; font-weight: 500;
}
.topbar button:hover { opacity: 0.9; }
.topbar button.secondary { background: #757575; }
.topbar .sep { width: 1px; height: 24px; background: var(--border); }
.topbar .info { font-size: 12px; color: #666; }
.topbar .kbd {
  display: inline-block; background: #eee; border: 1px solid #ccc;
  border-radius: 3px; padding: 1px 5px; font-size: 11px; font-family: var(--mono);
  color: #555;
}

/* ── Field filters ───────────────────────────────────────── */
.field-filters {
  display: flex; gap: 6px; flex-wrap: wrap; align-items: center;
  padding: 4px 16px 6px; background: #fafafa;
  border-bottom: 1px solid var(--border);
}
.field-filters:empty { display: none; }
.field-filter {
  display: inline-flex; align-items: center; gap: 3px; font-size: 12px;
}
.field-filter label { font-weight: 600; color: #555; white-space: nowrap; }
.field-filter select {
  font-size: 12px; padding: 2px 6px; border: 1px solid var(--border);
  border-radius: 4px; background: #fff; max-width: 200px;
}
.field-filter select.active-filter {
  border-color: var(--accent); background: var(--accent-light);
}
.field-filters .ff-reset {
  font-size: 11px; color: var(--accent); cursor: pointer;
  text-decoration: underline; margin-left: 4px;
}
.field-filters .ff-reset:hover { opacity: 0.75; }

/* ── Searchable filter ──────────────────────────────────── */
.search-filter {
  position: relative; display: inline-flex; align-items: center; gap: 3px; font-size: 12px;
}
.search-filter label { font-weight: 600; color: #555; white-space: nowrap; }
.search-filter input {
  font-size: 12px; padding: 2px 6px; border: 1px solid var(--border);
  border-radius: 4px; background: #fff; width: 170px;
}
.search-filter input.active-filter {
  border-color: var(--accent); background: var(--accent-light);
}
.search-filter .sf-dropdown {
  display: none; position: absolute; top: 100%; left: 0; z-index: 200;
  background: #fff; border: 1px solid var(--border); border-radius: 6px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.12); max-height: 260px; overflow-y: auto;
  min-width: 220px; margin-top: 2px;
}
.search-filter.open .sf-dropdown { display: block; }
.sf-item {
  padding: 5px 10px; cursor: pointer; font-size: 12px;
  display: flex; justify-content: space-between; gap: 8px;
}
.sf-item:hover, .sf-item.highlighted { background: var(--accent-light); }
.sf-item .sf-count { color: #999; font-size: 11px; min-width: 30px; text-align: right; }
.sf-clear {
  display: none; position: absolute; right: 4px; top: 50%; transform: translateY(-50%);
  cursor: pointer; font-size: 14px; color: #999; line-height: 1;
  background: none; border: none; padding: 0 2px;
}
.sf-clear:hover { color: var(--red); }
.search-filter.has-value .sf-clear { display: block; }
.search-filter.has-value input { padding-right: 18px; }

/* ── Overview panel (pre-rendered, tool-specific) ──────── */
.overview-panel {
  background: #fff; border: 1px solid var(--border); border-radius: 8px;
  margin: 12px 16px; padding: 14px 18px;
  max-width: 1400px;
}
.overview-panel h2 {
  font-size: 14px; font-weight: 700; margin-bottom: 10px;
  color: var(--accent);
}
.overview-grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 10px 16px; margin-bottom: 10px;
}
.overview-stat {
  background: #fafafa; border: 1px solid #f0f0f0; border-radius: 4px;
  padding: 6px 10px;
}
.overview-stat .stat-label {
  font-size: 10px; font-weight: 600; color: #888;
  text-transform: uppercase; letter-spacing: 0.5px;
}
.overview-stat .stat-value {
  font-size: 15px; font-weight: 600; color: var(--fg); font-family: var(--mono);
}

/* ── Main container & row cards ─────────────────────────── */
.container { max-width: 1400px; margin: 0 auto; padding: 12px 16px; }

.row-card {
  background: #fff; border: 1px solid var(--border); border-radius: 8px;
  margin-bottom: 16px; overflow: hidden;
  transition: border-color 0.15s;
}
.row-card.current { border-color: var(--accent); box-shadow: 0 0 0 2px var(--accent-light); }
.row-card.bookmarked { border-left: 4px solid var(--orange); }

.row-header {
  display: flex; align-items: center; gap: 8px;
  padding: 8px 12px; background: #f5f5f5; border-bottom: 1px solid var(--border);
  cursor: pointer; user-select: none;
}
.row-header:hover { background: #eeeeee; }
.row-num { font-weight: 700; font-size: 13px; color: var(--accent); min-width: 50px; }
.row-ids { font-size: 12px; color: #666; flex: 1; }
.row-ids b { color: #333; }
.row-badges { display: flex; gap: 4px; flex-wrap: wrap; }

.badge { font-size: 11px; padding: 2px 8px; border-radius: 10px; font-weight: 500; }
.badge.correct, .badge.ok, .badge.pass { background: var(--green-bg); color: var(--green); }
.badge.wrong, .badge.fail { background: var(--red-bg); color: var(--red); }
.badge.warn { background: var(--orange-bg); color: var(--orange); }
.badge.info { background: var(--blue-bg); color: var(--blue); }
.badge.neutral { background: #eee; color: #555; }
.badge.accent { background: var(--accent-light); color: var(--accent); }
.badge.bookmark-badge { background: var(--orange-bg); color: var(--orange); cursor: pointer; }

.row-body { display: none; padding: 12px; }
.row-card.expanded .row-body { display: block; }

/* ── Text/prompt boxes ──────────────────────────────────── */
.text-box, .prompt-box {
  background: #f5f5f5; border: 1px solid #e8e8e8; border-radius: 6px;
  padding: 10px 14px; margin-bottom: 12px; font-size: 13px;
  max-height: 360px; overflow-y: auto;
}
.text-box summary, .prompt-box summary {
  cursor: pointer; font-weight: 600; font-size: 12px; color: #555;
}
.text-box pre, .prompt-box pre {
  white-space: pre-wrap; word-break: break-word; margin: 6px 0 0;
  font-family: var(--mono); font-size: 12px; line-height: 1.45;
}
.text-box.completion { background: #faf5ff; border-color: #e1bee7; }
.text-box.user { background: #f5f5f5; }
.text-box.assistant { background: #faf5ff; border-color: #e1bee7; }

/* ── Field groups ────────────────────────────────────────── */
.field-group {
  margin-bottom: 12px; border: 1px solid var(--border); border-radius: 6px;
  overflow: hidden;
}
.field-group-header {
  padding: 6px 12px; font-weight: 600; font-size: 12px;
  background: #f5f5f5; border-bottom: 1px solid var(--border);
  cursor: pointer; user-select: none;
}
.field-group-header:hover { background: #eee; }
.field-group-body { display: none; }
.field-group.open .field-group-body { display: block; }
.field-row {
  display: flex; padding: 4px 12px; border-bottom: 1px solid #f0f0f0;
  font-size: 13px;
}
.field-row:last-child { border-bottom: none; }
.field-label {
  width: 200px; min-width: 200px; font-weight: 500; color: #555;
  padding-right: 12px;
}
.field-value {
  flex: 1; font-family: var(--mono); font-size: 12px;
  white-space: pre-wrap; word-break: break-word;
}
.field-value.bool-true { color: var(--green); font-weight: 600; }
.field-value.bool-false { color: var(--red); font-weight: 600; }

/* ── CI tuple card ───────────────────────────────────────── */
.ci-tuple-card {
  display: grid; grid-template-columns: repeat(5, 1fr); gap: 1px;
  background: var(--border); border-radius: 6px; overflow: hidden;
  margin-bottom: 12px;
}
.ci-tuple-cell { background: #fff; padding: 8px 10px; text-align: center; }
.ci-tuple-cell .cell-label {
  font-size: 10px; font-weight: 600; color: #888;
  text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px;
}
.ci-tuple-cell .cell-value { font-size: 13px; font-weight: 500; color: var(--fg); }

/* ── JSON pretty-print ───────────────────────────────────── */
.json-block {
  background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 4px;
  padding: 8px 10px; margin: 4px 0; font-size: 12px;
  font-family: var(--mono); white-space: pre-wrap; word-break: break-word;
}
.json-key { color: #881391; }
.json-str { color: #0b7285; }
.json-num { color: #d9480f; }
.json-bool { color: #5c940d; font-weight: 600; }
.json-null { color: #868e96; font-style: italic; }

/* ── Search highlight ────────────────────────────────────── */
mark { background: #fff176; padding: 1px 2px; border-radius: 2px; }

/* ── Reward bars (GRPO) ─────────────────────────────────── */
.reward-bar-row {
  display: flex; align-items: center; gap: 8px;
  font-family: var(--mono); font-size: 12px;
  padding: 3px 0;
}
.reward-bar-label {
  width: 90px; min-width: 90px; font-weight: 600; color: #555;
}
.reward-bar-track {
  flex: 1; height: 14px; background: #eee; border-radius: 3px;
  position: relative; overflow: hidden;
}
.reward-bar-fill {
  position: absolute; left: 0; top: 0; bottom: 0;
  background: linear-gradient(90deg, var(--accent), var(--accent-light));
}
.reward-bar-fill.muted { background: #bbb; }
.reward-bar-value {
  width: 90px; min-width: 90px; text-align: right;
  color: #333; font-size: 11.5px;
}
.reward-bar-value .weighted { color: #999; font-size: 10.5px; }

/* ── R_ground flow cards ────────────────────────────────── */
.rground-flow-card {
  border: 1px solid var(--border); border-radius: 6px;
  margin-bottom: 10px; overflow: hidden;
}
.rground-flow-header {
  padding: 6px 12px; font-weight: 600; font-size: 12px;
  background: #f5f5f5; border-bottom: 1px solid var(--border);
}
.rground-flow-body { padding: 10px 12px; }
.rground-query {
  background: #f8f9fa; border-left: 3px solid var(--accent);
  padding: 6px 10px; margin-bottom: 10px;
  font-family: var(--mono); font-size: 11px; color: #444;
}
.rground-two-col {
  display: grid; grid-template-columns: 1fr 1fr; gap: 10px;
}
.rground-col {
  border: 1px solid #e8e8e8; border-radius: 4px; padding: 8px 10px;
}
.rground-col.correct { background: #f1f8e9; border-color: #c5e1a5; }
.rground-col.wrong { background: #fff3e0; border-color: #ffcc80; }
.rground-col-header {
  font-weight: 600; font-size: 11px;
  text-transform: uppercase; letter-spacing: 0.4px;
  margin-bottom: 6px;
}
.rground-col.correct .rground-col-header { color: #2e7d32; }
.rground-col.wrong .rground-col-header { color: #ef6c00; }
.rground-norm-list { margin-bottom: 6px; }
.rground-norm-item {
  font-family: var(--mono); font-size: 11px; color: #444;
  padding: 3px 0;
}
.rground-norm-item .sim {
  display: inline-block; min-width: 44px; color: #888;
  font-size: 10.5px;
}
.rground-scores {
  display: flex; gap: 8px; flex-wrap: wrap; font-size: 11px;
}
.rground-scores .sc {
  background: #fff; padding: 2px 8px; border: 1px solid #e0e0e0;
  border-radius: 10px; font-family: var(--mono);
}
.rground-contrastive {
  margin-top: 8px; padding: 6px 10px;
  background: #e3f2fd; border-left: 3px solid var(--blue);
  font-family: var(--mono); font-size: 11px; color: #1565c0;
}
.rground-flow-card.sign-flip { border-color: var(--red); }
.rground-flow-card.sign-flip .rground-flow-header {
  background: var(--red-bg); color: var(--red);
}

/* ── Bookmarks panel ─────────────────────────────────────── */
.bookmarks-panel {
  position: fixed; right: 0; top: 45px; width: 280px; max-height: calc(100vh - 55px);
  background: #fff; border-left: 1px solid var(--border);
  box-shadow: -2px 0 8px rgba(0,0,0,0.05);
  transform: translateX(100%); transition: transform 0.2s;
  z-index: 90; overflow-y: auto; padding: 12px;
}
.bookmarks-panel.open { transform: translateX(0); }
.bookmarks-panel h3 { font-size: 14px; margin-bottom: 8px; }
.bookmark-item {
  padding: 6px 8px; font-size: 12px; border-radius: 4px; cursor: pointer;
  border: 1px solid var(--border); margin-bottom: 4px;
}
.bookmark-item:hover { background: var(--accent-light); }

/* ── Export modal ────────────────────────────────────────── */
.export-overlay { position:fixed;inset:0;z-index:500;background:rgba(0,0,0,0.5);display:flex;align-items:center;justify-content:center; }
.export-dialog { background:#fff;border-radius:10px;width:92vw;max-width:1000px;max-height:92vh;display:flex;flex-direction:column;box-shadow:0 8px 32px rgba(0,0,0,0.2); }
.export-header { padding:12px 16px;border-bottom:1px solid var(--border);display:flex;justify-content:space-between;align-items:center; }
.export-header h3 { margin:0;font-size:15px; }
.export-controls { padding:10px 16px;border-bottom:1px solid var(--border);display:flex;flex-wrap:wrap;gap:10px 18px;align-items:center;font-size:13px;background:#fafafa; }
.export-controls label { display:inline-flex;align-items:center;gap:4px;cursor:pointer;white-space:nowrap; }
.export-preview { flex:1;overflow:auto;padding:16px;background:#f9f9f9; }
.export-preview [data-export].export-hidden { display:none !important; }
.export-preview .export-frame { background:#fff;border:1px solid var(--border);border-radius:8px;padding:14px;margin:0 auto;font-size:13px; }
.export-preview .export-frame details[open] > summary { margin-bottom:4px; }
.export-preview .export-frame pre { max-height:none !important; overflow:visible !important; }
.export-preview .export-frame .text-box,
.export-preview .export-frame .prompt-box { max-height:none !important; overflow:visible !important; }
.export-actions { padding:10px 16px;border-top:1px solid var(--border);display:flex;gap:8px;justify-content:flex-end; }
.export-actions button { font-size:13px;padding:6px 16px;border:none;border-radius:4px;cursor:pointer;font-weight:500; }
.export-actions .btn-pdf { background:#1976d2;color:#fff; }
.export-actions .btn-html { background:#2e7d32;color:#fff; }
.export-actions .btn-cancel { background:#757575;color:#fff; }
.export-actions button:hover { opacity:0.9; }

/* ── Training curves (simple inline SVG) ─────────────── */
.curve-chart { width: 100%; height: 180px; background: #fff; border: 1px solid #eee; border-radius: 4px; }
.curve-legend { display: flex; gap: 12px; font-size: 11px; margin-top: 4px; }
.curve-legend .legend-item { display: inline-flex; align-items: center; gap: 4px; }
.curve-legend .legend-swatch { width: 12px; height: 3px; display: inline-block; }

/* ── Responsive ──────────────────────────────────────────── */
@media (max-width: 900px) {
  .rground-two-col { grid-template-columns: 1fr; }
  .ci-tuple-card { grid-template-columns: repeat(2, 1fr); }
  .field-label { width: 140px; min-width: 140px; }
}
"""


# ── Shared HTML skeleton ──────────────────────────────────────────────────

HTML_SKELETON = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
__COMMON_CSS__
__EXTRA_CSS__
</style>
</head>
<body>

<div class="topbar" id="topbar">
  <select id="stage-select"></select>
  <div class="sep"></div>
  <input type="number" id="jump-input" placeholder="#" min="0" style="width:80px;">
  <button id="jump-btn" class="secondary">Go</button>
  <div class="sep"></div>
  <input type="text" id="search-input" placeholder="Search (regex)...">
  <button id="search-btn">Search</button>
  <button id="clear-btn" class="secondary">Clear</button>
  <div class="sep"></div>
  <button id="bookmarks-btn" class="secondary">Bookmarks</button>
  <button id="exportrow-btn" class="secondary">Export Row</button>
  <button id="export-btn" class="secondary">Export BMs</button>
  <div class="info" id="status-info"></div>
  <div class="info">
    <span class="kbd">j</span>/<span class="kbd">k</span> nav
    <span class="kbd">g</span> jump
    <span class="kbd">b</span> bookmark
    <span class="kbd">/</span> search
  </div>
</div>

<div class="field-filters" id="field-filters"></div>

<div id="overview-slot">
__OVERVIEW_HTML__
</div>

<div class="container" id="container"></div>

<div class="bookmarks-panel" id="bookmarks-panel">
  <h3>Bookmarks</h3>
  <div id="bookmarks-list"></div>
</div>

<div id="export-overlay" class="export-overlay" style="display:none">
  <div class="export-dialog">
    <div class="export-header"><h3 id="export-title">Export Row</h3></div>
    <div class="export-controls" id="export-controls"></div>
    <div class="export-preview"><div class="export-frame" id="export-frame"></div></div>
    <div class="export-actions">
      <button class="btn-html" onclick="doExportHTML()">Save HTML</button>
      <button class="btn-pdf" onclick="doExportPDF()">Save PDF</button>
      <button class="btn-cancel" onclick="closeExportModal()">Close</button>
    </div>
  </div>
</div>

<script>
const DATA = __DATA__;
const TOOL_NAME = "__TOOL_NAME__";

__CORE_JS__

__EXTRA_JS__

// ── Start ────────────────────────────────────────────────────────────
init();
</script>
</body>
</html>
"""


# ── Shared core JS ────────────────────────────────────────────────────────
# Everything except buildRowBody / getRowBadges / getRowIdLine, which are
# provided by each tool via inspectorHooks overrides in extra_js.

CORE_JS = r"""
const STAGE_KEYS = Object.keys(DATA);

// Tool-specific hooks — overridden by extra_js
window.inspectorHooks = window.inspectorHooks || {};
const HOOKS = window.inspectorHooks;
if (!HOOKS.buildRowBody) HOOKS.buildRowBody = function(row, stageData) {
  return '<pre>' + esc(JSON.stringify(row, null, 2)) + '</pre>';
};
if (!HOOKS.getRowBadges) HOOKS.getRowBadges = function(row, stageData) { return []; };
if (!HOOKS.getRowIdLine) HOOKS.getRowIdLine = function(row, stageData) {
  const idCols = stageData.id_cols || [];
  return idCols.filter(c => row[c] !== undefined)
    .map(c => `<b>${esc(c)}</b>=${esc(String(row[c]))}`)
    .join('&ensp;');
};
if (!HOOKS.onStageChange) HOOKS.onStageChange = function(stageKey, stageData) {};

// ── State ────────────────────────────────────────────────────────────
let currentStage = STAGE_KEYS[0] || '';
let currentIdx = 0;
let filteredRows = [];
let allRows = [];
let searchQuery = '';
let bookmarks = new Set();
let bookmarksPanelOpen = false;
let fieldFilters = {};
let _searchFilterWidgets = [];

// ── Init ─────────────────────────────────────────────────────────────
function init() {
  const sel = document.getElementById('stage-select');
  STAGE_KEYS.forEach(k => {
    const opt = document.createElement('option');
    opt.value = k;
    const sd = DATA[k];
    const nRows = sd.rows ? sd.rows.length : 0;
    const nTotal = sd.n_total || nRows;
    opt.textContent = `${k} (${nRows}${nTotal > nRows ? ' / ' + nTotal : ''} rows)`;
    sel.appendChild(opt);
  });
  sel.addEventListener('change', () => { loadStage(sel.value); });

  document.getElementById('jump-btn').addEventListener('click', doJump);
  document.getElementById('jump-input').addEventListener('keydown', e => {
    if (e.key === 'Enter') doJump();
  });
  document.getElementById('search-btn').addEventListener('click', doSearch);
  document.getElementById('clear-btn').addEventListener('click', clearSearch);
  document.getElementById('search-input').addEventListener('keydown', e => {
    if (e.key === 'Enter') doSearch();
  });
  document.getElementById('bookmarks-btn').addEventListener('click', toggleBookmarks);
  document.getElementById('exportrow-btn').addEventListener('click', showExportModal);
  document.getElementById('export-overlay').addEventListener('click', (e) => {
    if (e.target === e.currentTarget) closeExportModal();
  });
  document.getElementById('export-btn').addEventListener('click', exportBookmarks);

  document.addEventListener('keydown', handleKeyboard);

  if (currentStage) loadStage(currentStage);
}

function loadStage(key) {
  currentStage = key;
  const stageData = DATA[key] || {};
  allRows = stageData.rows || [];
  bookmarks.clear();
  searchQuery = '';
  fieldFilters = {};
  document.getElementById('search-input').value = '';
  buildFieldFilters();
  HOOKS.onStageChange(key, stageData);
  applyFilters();
}

// ── Field filters ───────────────────────────────────────────────────
function _discoverFacets() {
  const DROPDOWN_MAX = 30;
  const SEARCHABLE_MAX = 5000;
  const facetCounts = new Map();
  for (const row of allRows) {
    const facets = row._facets;
    if (!facets) continue;
    for (const [key, val] of Object.entries(facets)) {
      if (!facetCounts.has(key)) facetCounts.set(key, new Map());
      const vc = facetCounts.get(key);
      vc.set(val, (vc.get(val) || 0) + 1);
    }
  }
  const dropdowns = [];
  const searchable = [];
  for (const [key, valCounts] of facetCounts) {
    if (valCounts.size < 2 || valCounts.size > SEARCHABLE_MAX) continue;
    const entries = [...valCounts.entries()].sort((a, b) => {
      if (b[1] !== a[1]) return b[1] - a[1];
      return a[0].localeCompare(b[0]);
    });
    if (valCounts.size <= DROPDOWN_MAX) {
      const sorted = entries.map(e => e[0]).sort((a, b) => {
        const na = Number(a), nb = Number(b);
        if (!isNaN(na) && !isNaN(nb)) return na - nb;
        return a.localeCompare(b);
      });
      dropdowns.push({key, values: sorted});
    } else {
      searchable.push({key, entries});
    }
  }
  dropdowns.sort((a, b) => a.key.localeCompare(b.key));
  searchable.sort((a, b) => a.key.localeCompare(b.key));
  return {dropdowns, searchable};
}

function _buildSearchableFilter(container, key, entries) {
  const wrapper = document.createElement('span');
  wrapper.className = 'search-filter';
  const lbl = document.createElement('label');
  lbl.textContent = key + ':';
  wrapper.appendChild(lbl);
  const input = document.createElement('input');
  input.type = 'text';
  input.placeholder = `Search (${entries.length} values)...`;
  wrapper.appendChild(input);
  const clearBtn = document.createElement('button');
  clearBtn.className = 'sf-clear';
  clearBtn.textContent = '\u00d7';
  wrapper.appendChild(clearBtn);
  const dropdown = document.createElement('div');
  dropdown.className = 'sf-dropdown';
  wrapper.appendChild(dropdown);

  let highlighted = -1;
  let visible = [];

  function renderDropdown(query) {
    dropdown.innerHTML = '';
    const q = query.toLowerCase();
    visible = q ? entries.filter(([val]) => val.toLowerCase().includes(q))
                : entries.slice(0, 50);
    highlighted = -1;
    if (visible.length === 0) {
      const empty = document.createElement('div');
      empty.style.cssText = 'padding:8px 10px;color:#999;font-size:12px;';
      empty.textContent = 'No matches';
      dropdown.appendChild(empty);
      return;
    }
    visible.forEach(([val, count]) => {
      const item = document.createElement('div');
      item.className = 'sf-item';
      const name = document.createElement('span');
      if (q) {
        const idx = val.toLowerCase().indexOf(q);
        if (idx >= 0) {
          name.innerHTML = esc(val.slice(0, idx))
            + '<mark>' + esc(val.slice(idx, idx + q.length)) + '</mark>'
            + esc(val.slice(idx + q.length));
        } else { name.textContent = val; }
      } else { name.textContent = val; }
      item.appendChild(name);
      const cnt = document.createElement('span');
      cnt.className = 'sf-count';
      cnt.textContent = count.toLocaleString();
      item.appendChild(cnt);
      item.addEventListener('mousedown', (e) => {
        e.preventDefault();
        selectValue(val);
      });
      dropdown.appendChild(item);
    });
  }

  function selectValue(val) {
    input.value = val;
    fieldFilters[key] = val;
    wrapper.classList.add('has-value');
    wrapper.classList.remove('open');
    input.classList.add('active-filter');
    refilter();
  }
  function clearValue() {
    input.value = '';
    delete fieldFilters[key];
    wrapper.classList.remove('has-value', 'open');
    input.classList.remove('active-filter');
    refilter();
  }
  clearBtn.addEventListener('click', (e) => { e.stopPropagation(); clearValue(); });
  input.addEventListener('focus', () => {
    renderDropdown(fieldFilters[key] ? '' : input.value);
    wrapper.classList.add('open');
  });
  input.addEventListener('input', () => {
    renderDropdown(input.value);
    wrapper.classList.add('open');
  });
  input.addEventListener('blur', () => {
    setTimeout(() => { wrapper.classList.remove('open'); }, 150);
  });
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') { wrapper.classList.remove('open'); input.blur(); return; }
    if (e.key === 'Enter') {
      e.preventDefault();
      if (highlighted >= 0 && highlighted < visible.length) selectValue(visible[highlighted][0]);
      else if (visible.length === 1) selectValue(visible[0][0]);
      return;
    }
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      if (highlighted < visible.length - 1) highlighted++;
      _highlightItem(dropdown, highlighted);
      return;
    }
    if (e.key === 'ArrowUp') {
      e.preventDefault();
      if (highlighted > 0) highlighted--;
      _highlightItem(dropdown, highlighted);
      return;
    }
  });
  container.appendChild(wrapper);
  return {input, wrapper};
}

function _highlightItem(dropdown, idx) {
  dropdown.querySelectorAll('.sf-item').forEach((el, i) => {
    el.classList.toggle('highlighted', i === idx);
    if (i === idx) el.scrollIntoView({block: 'nearest'});
  });
}

function buildFieldFilters() {
  const container = document.getElementById('field-filters');
  container.innerHTML = '';
  fieldFilters = {};
  _searchFilterWidgets = [];

  const {dropdowns, searchable} = _discoverFacets();
  if (dropdowns.length === 0 && searchable.length === 0) return;

  dropdowns.forEach(({key, values}) => {
    const wrapper = document.createElement('span');
    wrapper.className = 'field-filter';
    const lbl = document.createElement('label');
    lbl.textContent = key + ':';
    wrapper.appendChild(lbl);
    const sel = document.createElement('select');
    const allOpt = document.createElement('option');
    allOpt.value = '';
    allOpt.textContent = 'All (' + values.length + ')';
    sel.appendChild(allOpt);
    values.forEach(v => {
      const opt = document.createElement('option');
      opt.value = v; opt.textContent = v;
      sel.appendChild(opt);
    });
    const facetKey = key;
    sel.addEventListener('change', () => {
      if (sel.value) {
        fieldFilters[facetKey] = sel.value;
        sel.classList.add('active-filter');
      } else {
        delete fieldFilters[facetKey];
        sel.classList.remove('active-filter');
      }
      refilter();
    });
    wrapper.appendChild(sel);
    container.appendChild(wrapper);
  });

  searchable.forEach(({key, entries}) => {
    _searchFilterWidgets.push(_buildSearchableFilter(container, key, entries));
  });

  const reset = document.createElement('span');
  reset.className = 'ff-reset';
  reset.textContent = 'Reset all';
  reset.addEventListener('click', () => {
    _resetAllFilterWidgets();
    refilter();
  });
  container.appendChild(reset);
}

function _resetAllFilterWidgets() {
  fieldFilters = {};
  searchQuery = '';
  document.getElementById('search-input').value = '';
  document.getElementById('field-filters').querySelectorAll('select').forEach(s => {
    s.value = '';
    s.classList.remove('active-filter');
  });
  _searchFilterWidgets.forEach(({input, wrapper}) => {
    input.value = '';
    input.classList.remove('active-filter');
    wrapper.classList.remove('has-value', 'open');
  });
}

function matchesFieldFilters(row) {
  const facets = row._facets;
  for (const [key, expected] of Object.entries(fieldFilters)) {
    if (!facets || facets[key] !== expected) return false;
  }
  return true;
}

function _buildSearchText(row) {
  if (row._searchText !== undefined) return row._searchText;
  const parts = [];
  // Generic: serialize every string/number top-level value
  for (const [k, v] of Object.entries(row)) {
    if (k.startsWith('_')) continue;
    if (v == null) continue;
    if (typeof v === 'string' || typeof v === 'number') parts.push(String(v));
    else if (typeof v === 'object') {
      try { parts.push(JSON.stringify(v)); } catch (e) {}
    }
  }
  row._searchText = parts.join('\n');
  return row._searchText;
}

function matchesSearch(row) {
  if (!searchQuery) return true;
  try {
    return new RegExp(searchQuery, 'i').test(_buildSearchText(row));
  } catch (e) {
    return true;
  }
}

function refilter() {
  filteredRows = allRows.filter(r => matchesSearch(r) && matchesFieldFilters(r));
  currentIdx = 0;
  render('header');
  updateStatus();
}
function applyFilters() { refilter(); }

function updateStatus() {
  const nActive = Object.keys(fieldFilters).length;
  const parts = [`${filteredRows.length} / ${allRows.length} rows`];
  if (searchQuery) parts.push(`search: "${searchQuery}"`);
  if (nActive) parts.push(`${nActive} filter${nActive > 1 ? 's' : ''}`);
  document.getElementById('status-info').textContent = parts.join(' | ');
}

// ── Jump / search ─────────────────────────────────────────────────
function doJump() {
  const input = document.getElementById('jump-input');
  const target = parseInt(input.value, 10);
  if (isNaN(target)) return;
  const fi = filteredRows.findIndex(r => r.idx === target);
  if (fi >= 0) { currentIdx = fi; render('center'); }
  else {
    const exists = allRows.some(r => r.idx === target);
    if (exists) {
      _resetAllFilterWidgets();
      filteredRows = allRows.slice();
      const fi2 = filteredRows.findIndex(r => r.idx === target);
      if (fi2 >= 0) { currentIdx = fi2; render('center'); }
      updateStatus();
    }
  }
  input.value = '';
}
function doSearch() {
  searchQuery = document.getElementById('search-input').value.trim();
  applyFilters();
}
function clearSearch() {
  searchQuery = '';
  document.getElementById('search-input').value = '';
  applyFilters();
}

// ── Render ───────────────────────────────────────────────────────────
function render(scrollMode) {
  if (!scrollMode) scrollMode = 'header';
  const container = document.getElementById('container');
  container.innerHTML = '';

  if (filteredRows.length === 0) {
    container.innerHTML = '<p style="padding:40px;text-align:center;color:#999;">No rows match the current filters.</p>';
    return;
  }

  const start = Math.max(0, currentIdx - 25);
  const end = Math.min(filteredRows.length, currentIdx + 50);
  for (let fi = start; fi < end; fi++) {
    container.appendChild(buildRowCard(filteredRows[fi], fi));
  }

  if (scrollMode === 'none') return;
  requestAnimationFrame(() => {
    const el = document.querySelector('.row-card.current');
    if (!el) return;
    if (scrollMode === 'center') {
      el.scrollIntoView({block: 'center', behavior: 'smooth'});
    } else {
      const topbar = document.getElementById('topbar');
      const offset = topbar ? topbar.getBoundingClientRect().height + 8 : 50;
      const rect = el.getBoundingClientRect();
      const scrollY = window.scrollY + rect.top - offset;
      window.scrollTo({top: Math.max(0, scrollY), behavior: 'smooth'});
    }
  });
}

function buildRowCard(row, filterIdx) {
  const stageData = DATA[currentStage];
  const card = document.createElement('div');
  card.className = 'row-card' +
    (filterIdx === currentIdx ? ' expanded current' : '') +
    (bookmarks.has(row.idx) ? ' bookmarked' : '');
  card.dataset.filterIdx = filterIdx;

  // Header
  const header = document.createElement('div');
  header.className = 'row-header';
  header.addEventListener('click', () => {
    if (currentIdx === filterIdx) return;
    currentIdx = filterIdx;
    render('header');
  });

  const num = document.createElement('span');
  num.className = 'row-num';
  num.textContent = `#${row.idx}`;
  header.appendChild(num);

  const ids = document.createElement('span');
  ids.className = 'row-ids';
  ids.innerHTML = HOOKS.getRowIdLine(row, stageData);
  header.appendChild(ids);

  const badges = document.createElement('span');
  badges.className = 'row-badges';
  const badgeList = HOOKS.getRowBadges(row, stageData) || [];
  badgeList.forEach(b => {
    const span = document.createElement('span');
    span.className = 'badge ' + (b.cls || 'neutral');
    span.textContent = b.text;
    if (b.title) span.title = b.title;
    badges.appendChild(span);
  });
  const bmBadge = document.createElement('span');
  bmBadge.className = 'badge bookmark-badge';
  bmBadge.textContent = bookmarks.has(row.idx) ? '★' : '☆';
  bmBadge.title = 'Toggle bookmark (b)';
  bmBadge.addEventListener('click', (e) => {
    e.stopPropagation();
    toggleBookmark(row.idx);
  });
  badges.appendChild(bmBadge);
  header.appendChild(badges);
  card.appendChild(header);

  if (filterIdx === currentIdx) {
    const body = document.createElement('div');
    body.className = 'row-body';
    body.style.display = 'block';
    body.innerHTML = HOOKS.buildRowBody(row, stageData);
    card.appendChild(body);
  }

  return card;
}

// ── Helpers ──────────────────────────────────────────────────────────
function esc(s) {
  if (s == null) return '';
  const d = document.createElement('div');
  d.textContent = String(s);
  return d.innerHTML;
}

function stringify(val) {
  if (val == null) return '';
  if (typeof val === 'string') return val;
  if (typeof val === 'object') return JSON.stringify(val, null, 2);
  return String(val);
}

function highlightSearch(html) {
  if (!searchQuery) return html;
  try {
    const re = new RegExp(`(${searchQuery})`, 'gi');
    return html.replace(re, '<mark>$1</mark>');
  } catch (e) { return html; }
}

function syntaxHighlightJson(jsonStr) {
  return jsonStr.replace(
    /("(?:\\.|[^"\\])*")\s*:/g, '<span class="json-key">$1</span>:'
  ).replace(
    /:\s*("(?:\\.|[^"\\])*")/g, ': <span class="json-str">$1</span>'
  ).replace(
    /:\s*(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\b/g, ': <span class="json-num">$1</span>'
  ).replace(
    /:\s*(true|false)\b/g, ': <span class="json-bool">$1</span>'
  ).replace(
    /:\s*(null)\b/g, ': <span class="json-null">$1</span>'
  );
}

function _tryParseJsonAt(text, i) {
  const ch = text[i];
  if (ch !== '{' && ch !== '[') return null;
  const close = ch === '{' ? '}' : ']';
  let depth = 0, inStr = false, esc2 = false;
  for (let j = i; j < text.length; j++) {
    const c = text[j];
    if (esc2) { esc2 = false; continue; }
    if (c === '\\' && inStr) { esc2 = true; continue; }
    if (c === '"' && !inStr) { inStr = true; continue; }
    if (c === '"' && inStr) { inStr = false; continue; }
    if (inStr) continue;
    if (c === ch) depth++;
    else if (c === close) depth--;
    if (depth === 0) {
      const candidate = text.slice(i, j + 1);
      try { return {end: j + 1, parsed: JSON.parse(candidate)}; }
      catch (e) { return null; }
    }
  }
  return null;
}

function formatTextWithJson(rawText) {
  const result = [];
  let i = 0;
  const len = rawText.length;
  let plainStart = 0;
  while (i < len) {
    const ch = rawText[i];
    if (ch === '{' || ch === '[') {
      const match = _tryParseJsonAt(rawText, i);
      if (match) {
        if (i > plainStart) result.push(esc(rawText.slice(plainStart, i)));
        const pretty = JSON.stringify(match.parsed, null, 2);
        result.push('<div class="json-block">' + syntaxHighlightJson(esc(pretty)) + '</div>');
        i = match.end;
        plainStart = i;
        continue;
      }
    }
    i++;
  }
  if (plainStart < len) result.push(esc(rawText.slice(plainStart)));
  return result.join('');
}

// ── Bookmarks ────────────────────────────────────────────────────────
function toggleBookmark(idx) {
  if (bookmarks.has(idx)) bookmarks.delete(idx);
  else bookmarks.add(idx);
  const isBookmarked = bookmarks.has(idx);
  document.querySelectorAll('.row-card').forEach(card => {
    const fi = parseInt(card.dataset.filterIdx, 10);
    if (filteredRows[fi] && filteredRows[fi].idx === idx) {
      card.classList.toggle('bookmarked', isBookmarked);
      const badge = card.querySelector('.bookmark-badge');
      if (badge) badge.textContent = isBookmarked ? '★' : '☆';
    }
  });
  renderBookmarksPanel();
}

function toggleBookmarks() {
  bookmarksPanelOpen = !bookmarksPanelOpen;
  document.getElementById('bookmarks-panel').classList.toggle('open', bookmarksPanelOpen);
  renderBookmarksPanel();
}

function renderBookmarksPanel() {
  const list = document.getElementById('bookmarks-list');
  if (bookmarks.size === 0) {
    list.innerHTML = '<p style="color:#999;font-size:12px;">No bookmarks yet. Press <b>b</b> to bookmark the current row.</p>';
    return;
  }
  list.innerHTML = '';
  const sorted = [...bookmarks].sort((a, b) => a - b);
  const stageData = DATA[currentStage];
  sorted.forEach(idx => {
    const row = allRows.find(r => r.idx === idx);
    if (!row) return;
    const item = document.createElement('div');
    item.className = 'bookmark-item';
    const idLine = HOOKS.getRowIdLine(row, stageData).replace(/<[^>]+>/g, '');
    item.textContent = `#${idx}${idLine ? ' — ' + idLine : ''}`;
    item.addEventListener('click', () => {
      const fi = filteredRows.findIndex(r => r.idx === idx);
      if (fi >= 0) { currentIdx = fi; render('center'); }
      else {
        _resetAllFilterWidgets();
        filteredRows = allRows.slice();
        const fi2 = filteredRows.findIndex(r => r.idx === idx);
        if (fi2 >= 0) { currentIdx = fi2; render('center'); }
        updateStatus();
      }
    });
    list.appendChild(item);
  });
}

function exportBookmarks() {
  if (bookmarks.size === 0) { alert('No bookmarks to export.'); return; }
  const exported = [];
  const sorted = [...bookmarks].sort((a, b) => a - b);
  sorted.forEach(idx => {
    const row = allRows.find(r => r.idx === idx);
    if (row) exported.push({stage: currentStage, ...row});
  });
  const blob = new Blob([JSON.stringify(exported, null, 2)], {type: 'application/json'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `bookmarks_${TOOL_NAME}_${currentStage.replace(/[^a-zA-Z0-9]/g, '_')}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

// ── Keyboard ─────────────────────────────────────────────────────────
function handleKeyboard(e) {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
    if (e.key === 'Escape') e.target.blur();
    return;
  }
  switch (e.key) {
    case 'j': case 'ArrowDown':
      e.preventDefault();
      if (currentIdx < filteredRows.length - 1) { currentIdx++; render('header'); }
      break;
    case 'k': case 'ArrowUp':
      e.preventDefault();
      if (currentIdx > 0) { currentIdx--; render('header'); }
      break;
    case 'g':
      e.preventDefault();
      document.getElementById('jump-input').focus();
      break;
    case 'b':
      if (filteredRows[currentIdx]) toggleBookmark(filteredRows[currentIdx].idx);
      break;
    case '/':
      e.preventDefault();
      document.getElementById('search-input').focus();
      break;
    case 'Escape':
      if (document.getElementById('export-overlay').style.display !== 'none') { closeExportModal(); break; }
      if (bookmarksPanelOpen) toggleBookmarks();
      break;
  }
}

// ── Export modal ──────────────────────────────────────────────────
let _exportRow = null;

function showExportModal() {
  const row = filteredRows[currentIdx];
  if (!row) return;
  _exportRow = row;
  document.getElementById('export-title').textContent = `Export Row #${row.idx} — ${currentStage}`;
  const ctrl = document.getElementById('export-controls');
  ctrl.innerHTML = '';

  const wlbl = document.createElement('span');
  wlbl.style.fontWeight = '600';
  wlbl.textContent = 'Width:';
  ctrl.appendChild(wlbl);
  const wInput = document.createElement('input');
  wInput.type = 'number';
  wInput.id = 'export-width';
  wInput.value = 1100;
  wInput.min = 400; wInput.max = 2400; wInput.step = 50;
  wInput.style.cssText = 'width:70px;font-size:12px;padding:2px 4px;border:1px solid #ccc;border-radius:3px;';
  wInput.addEventListener('input', updateExportPreview);
  ctrl.appendChild(wInput);
  const wpx = document.createElement('span');
  wpx.style.cssText = 'font-size:11px;color:#888;';
  wpx.textContent = 'px';
  ctrl.appendChild(wpx);

  updateExportPreview();
  document.getElementById('export-overlay').style.display = 'flex';
}

function _getExportWidth() {
  const el = document.getElementById('export-width');
  return el ? parseInt(el.value, 10) || 1100 : 1100;
}

function updateExportPreview() {
  const frame = document.getElementById('export-frame');
  frame.style.maxWidth = _getExportWidth() + 'px';
  const currentBody = document.querySelector('.row-card.current .row-body');
  if (!currentBody) return;
  const clone = currentBody.cloneNode(true);
  clone.style.display = 'block';
  clone.querySelectorAll('details').forEach(d => d.setAttribute('open', ''));
  clone.querySelectorAll('mark').forEach(m => {
    m.replaceWith(document.createTextNode(m.textContent));
  });
  clone.querySelectorAll('pre').forEach(p => { p.style.maxHeight = 'none'; });
  clone.querySelectorAll('.text-box, .prompt-box').forEach(p => { p.style.maxHeight = 'none'; });
  frame.innerHTML = '';
  frame.appendChild(clone);
}

function closeExportModal() {
  document.getElementById('export-overlay').style.display = 'none';
  _exportRow = null;
}

function _getExportCSS() {
  return `
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; font-size: 13px; color: #1a1a1a; }
    .row-body { display: block; padding: 0; }
    details { margin-bottom: 8px; }
    details > summary { cursor: default; font-weight: 600; font-size: 12px; color: #555; margin-bottom: 4px; }
    pre { white-space: pre-wrap; word-break: break-word; font-family: 'SF Mono','Cascadia Code','Fira Code',Consolas,monospace; font-size: 11.5px; line-height: 1.5; margin: 0; }
    .text-box, .prompt-box { background: #f5f5f5; border: 1px solid #e8e8e8; border-radius: 6px; padding: 10px 14px; margin-bottom: 10px; }
    .text-box.completion, .text-box.assistant { background: #faf5ff; border-color: #e1bee7; }
    .json-block { background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 4px; padding: 6px 8px; margin: 4px 0; font-size: 11px; font-family: 'SF Mono',Consolas,monospace; white-space: pre-wrap; }
    .json-key { color: #881391; } .json-str { color: #0b7285; } .json-num { color: #d9480f; }
    .json-bool { color: #5c940d; font-weight: 600; } .json-null { color: #868e96; font-style: italic; }
    .reward-bar-row { display: flex; align-items: center; gap: 8px; font-family: 'SF Mono',Consolas,monospace; font-size: 11px; padding: 3px 0; }
    .reward-bar-label { width: 90px; min-width: 90px; font-weight: 600; color: #555; }
    .reward-bar-track { flex: 1; height: 12px; background: #eee; border-radius: 3px; position: relative; overflow: hidden; }
    .reward-bar-fill { position: absolute; left: 0; top: 0; bottom: 0; background: #1976d2; }
    .reward-bar-fill.muted { background: #bbb; }
    .reward-bar-value { width: 90px; min-width: 90px; text-align: right; font-size: 10.5px; }
    .rground-flow-card { border: 1px solid #e0e0e0; border-radius: 6px; margin-bottom: 10px; overflow: hidden; }
    .rground-flow-header { padding: 5px 10px; font-weight: 600; font-size: 11px; background: #f5f5f5; border-bottom: 1px solid #e0e0e0; }
    .rground-flow-body { padding: 8px 10px; }
    .rground-query { background: #f8f9fa; border-left: 3px solid #1976d2; padding: 5px 10px; margin-bottom: 8px; font-family: 'SF Mono',Consolas,monospace; font-size: 10px; color: #444; }
    .rground-two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .rground-col { border: 1px solid #e8e8e8; border-radius: 4px; padding: 6px 10px; }
    .rground-col.correct { background: #f1f8e9; border-color: #c5e1a5; }
    .rground-col.wrong { background: #fff3e0; border-color: #ffcc80; }
    .rground-col-header { font-weight: 600; font-size: 10px; text-transform: uppercase; margin-bottom: 5px; }
    .rground-col.correct .rground-col-header { color: #2e7d32; }
    .rground-col.wrong .rground-col-header { color: #ef6c00; }
    .rground-norm-item { font-family: 'SF Mono',Consolas,monospace; font-size: 10px; color: #444; padding: 2px 0; }
    .rground-norm-item .sim { display: inline-block; min-width: 40px; color: #888; font-size: 9.5px; }
    .rground-scores { display: flex; gap: 6px; flex-wrap: wrap; font-size: 10px; }
    .rground-scores .sc { background: #fff; padding: 1px 6px; border: 1px solid #e0e0e0; border-radius: 8px; font-family: 'SF Mono',Consolas,monospace; }
    .rground-contrastive { margin-top: 6px; padding: 5px 8px; background: #e3f2fd; border-left: 3px solid #1565c0; font-family: 'SF Mono',Consolas,monospace; font-size: 10px; color: #1565c0; }
    .field-group { margin-bottom: 10px; border: 1px solid #e0e0e0; border-radius: 6px; overflow: hidden; }
    .field-group-header { padding: 5px 12px; font-weight: 600; font-size: 11px; background: #f5f5f5; border-bottom: 1px solid #e0e0e0; }
    .field-group-body { display: block !important; }
    .field-row { display: flex; padding: 3px 12px; border-bottom: 1px solid #f0f0f0; font-size: 12px; }
    .field-label { width: 180px; min-width: 180px; font-weight: 500; color: #555; padding-right: 10px; }
    .field-value { flex: 1; font-family: 'SF Mono',Consolas,monospace; font-size: 11px; white-space: pre-wrap; word-break: break-word; }
    .ci-tuple-card { display: grid; grid-template-columns: repeat(5, 1fr); gap: 1px; background: #e0e0e0; border-radius: 6px; overflow: hidden; margin-bottom: 10px; }
    .ci-tuple-cell { background: #fff; padding: 6px 8px; text-align: center; }
    .ci-tuple-cell .cell-label { font-size: 9px; font-weight: 600; color: #888; text-transform: uppercase; margin-bottom: 3px; }
    .ci-tuple-cell .cell-value { font-size: 11px; font-weight: 500; }
  `;
}

function _getExportHTML() {
  const frame = document.getElementById('export-frame');
  const clone = frame.cloneNode(true);
  clone.querySelectorAll('.export-hidden').forEach(el => el.remove());
  clone.querySelectorAll('[data-export]').forEach(el => el.removeAttribute('data-export'));
  return clone.innerHTML;
}

function _exportFilename(ext) {
  const row = _exportRow;
  if (!row) return `export.${ext}`;
  const stage = currentStage.replace(/[^a-zA-Z0-9]/g, '_');
  return `${TOOL_NAME}_row${row.idx}_${stage}.${ext}`;
}

function doExportHTML() {
  const content = _getExportHTML();
  const css = _getExportCSS();
  const w = _getExportWidth();
  const htmlDoc = `<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>${esc(_exportFilename('html'))}</title>
<style>${css}</style>
</head><body style="max-width:${w}px;margin:0 auto;padding:16px;">
${content}
</body></html>`;
  const blob = new Blob([htmlDoc], {type: 'text/html'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = _exportFilename('html');
  a.click();
  URL.revokeObjectURL(url);
}

function doExportPDF() {
  const content = _getExportHTML();
  const css = _getExportCSS();
  const w = _getExportWidth();
  const win = window.open('', '_blank');
  win.document.write(`<!DOCTYPE html><html><head><meta charset="utf-8"><title>${_exportFilename('pdf')}</title>
<style>
${css}
@media print { @page { margin: 0.5in; size: auto; } body { -webkit-print-color-adjust: exact; print-color-adjust: exact; } }
</style></head><body style="max-width:${w}px;padding:12px;">
${content}
<script>window.onload=function(){window.print();}<\/script>
</body></html>`);
  win.document.close();
}
"""


# ── Page assembly ─────────────────────────────────────────────────────────


# Default palette: (accent, accent_light)
ACCENTS = {
    "blue":   ("#1976d2", "#e3f2fd"),
    "purple": ("#6a1b9a", "#f3e5f5"),
    "teal":   ("#00796b", "#e0f2f1"),
    "orange": ("#e65100", "#fff3e0"),
    "green":  ("#2e7d32", "#e8f5e9"),
}


def render_page(
    *,
    tool_name: str,
    title: str,
    accent: str,
    stages_data: dict[str, dict],
    overview_html: str = "",
    body_renderer_js: str = "",
    extra_css: str = "",
    extra_js: str = "",
) -> str:
    """Assemble the full inspector HTML string.

    Args:
        tool_name: Short identifier used for export filenames, e.g. "sft" or "grpo".
        title: Browser tab title.
        accent: Name of accent color (key in ACCENTS) or a hex string.
        stages_data: {stage_key: {rows, id_cols, ...}}.  Each stage's rows are
            injected into DATA[stage_key] on the JS side.  Keep them already
            serialized (use _serialize() on values).
        overview_html: Pre-rendered tool-specific overview HTML, injected
            above the row container.
        body_renderer_js: JS code that defines HOOKS.buildRowBody (and
            optionally HOOKS.getRowBadges, HOOKS.getRowIdLine, HOOKS.onStageChange).
            Will be placed inside the existing <script>.
        extra_css: Extra CSS appended after COMMON_CSS.
        extra_js: Extra JS appended after body_renderer_js.
    """
    if accent in ACCENTS:
        accent_hex, accent_light = ACCENTS[accent]
    elif accent.startswith("#"):
        accent_hex = accent
        accent_light = "#f0f0f0"
    else:
        accent_hex, accent_light = ACCENTS["blue"]

    css = COMMON_CSS.replace("__ACCENT_COLOR__", accent_hex)
    css = css.replace("__ACCENT_LIGHT__", accent_light)

    data_json = json.dumps(stages_data, ensure_ascii=False, separators=(",", ":"))

    # Combine body renderer + extra js
    combined_extra_js = (body_renderer_js or "") + "\n\n" + (extra_js or "")

    html = HTML_SKELETON
    html = html.replace("__TITLE__", title)
    html = html.replace("__TOOL_NAME__", tool_name)
    html = html.replace("__COMMON_CSS__", css)
    html = html.replace("__EXTRA_CSS__", extra_css)
    html = html.replace("__OVERVIEW_HTML__", overview_html)
    html = html.replace("__DATA__", data_json)
    html = html.replace("__CORE_JS__", CORE_JS)
    html = html.replace("__EXTRA_JS__", combined_extra_js)
    return html
