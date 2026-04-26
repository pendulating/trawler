#!/usr/bin/env python3
"""Norms Inspector — static HTML browser for extracted norms and CI flows.

Reads the parquet outputs from the historical_norms dagspace (abstracted_norms,
ci_flows) and generates a self-contained HTML file for interactive browsing.

Usage:
    python -m scripts.norms_inspector \
        --data /share/pierson/matt/n2s4cir/data/fiction10 \
        -o norms_inspector.html

    # Only CI flows:
    python -m scripts.norms_inspector \
        --data /share/pierson/matt/n2s4cir/data/fiction10 \
        --stages ci_flows -o flows_only.html

    # Limit rows per stage:
    python -m scripts.norms_inspector \
        --data /share/pierson/matt/n2s4cir/data/fiction10 \
        --max-rows 500 -o norms_inspector.html

    # Specific books only:
    python -m scripts.norms_inspector \
        --data /share/pierson/matt/n2s4cir/data/fiction10 \
        --books "Pride and Prejudice" "1984" -o subset.html
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ── Data loading ────────────────────────────────────────────────────────

KNOWN_STAGES = {
    "abstracted_norms": "abstracted_norms.parquet",
    "ci_flows": "ci_flows.parquet",
}


def discover_stages(data_dir: Path) -> dict[str, Path]:
    """Find available parquet files in the data directory."""
    found = {}
    for key, filename in KNOWN_STAGES.items():
        path = data_dir / filename
        if path.exists():
            found[key] = path
    return found


def _serialize(v: Any) -> Any:
    """Make a value JSON-serializable."""
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


# ── Stage-specific schemas ──────────────────────────────────────────────

# Which columns to include, in what order, and how to group them for display.
# Each group: (group_label, [(display_name, column_name), ...])

NORMS_SCHEMA: list[tuple[str, list[tuple[str, str]]]] = [
    ("Source", [
        ("Book", "book_title"),
        ("Author", "book_author"),
        ("Gutenberg ID", "gutenberg_id"),
        ("Chunk ID", "chunk_id"),
    ]),
    ("Extraction — Preliminary", [
        ("Norm Snippet", "norm_snippet"),
        ("Reasoning Trace", "reasoning_trace"),
        ("Normative Force (prelim)", "preliminary_normative_force"),
        ("Governs Info Flow?", "governs_information_flow"),
        ("Norm Index", "norm_index"),
    ]),
    ("Raz Norm — Structured", [
        ("Subject", "raz_norm_subject"),
        ("Act", "raz_norm_act"),
        ("Condition of Application", "raz_condition_of_application"),
        ("Normative Force", "raz_normative_force"),
        ("Articulation", "raz_norm_articulation"),
        ("Prescriptive Element", "raz_prescriptive_element"),
        ("Source", "raz_norm_source"),
        ("Governs Info Flow?", "raz_governs_info_flow"),
        ("Info Flow Note", "raz_info_flow_note"),
        ("Context", "raz_context"),
        ("Confidence (qual)", "raz_confidence_qual"),
        ("Confidence (quant)", "raz_confidence_quant"),
    ]),
    ("Role Abstraction", [
        ("Original Subject", "orig_raz_norm_subject"),
        ("Original Act", "orig_raz_norm_act"),
        ("Original Condition", "orig_raz_condition_of_application"),
        ("Original Articulation", "orig_raz_norm_articulation"),
        ("Abstracted Subject", "raz_norm_subject"),
        ("Abstracted Act", "raz_norm_act"),
        ("Abstracted Condition", "raz_condition_of_application"),
        ("Abstracted Articulation", "raz_norm_articulation"),
        ("Role Rationale", "role_rationale"),
        ("Abstraction Failed?", "role_abstraction_failed"),
    ]),
    ("Quality", [
        ("Quality Passed?", "norm_quality_passed"),
        ("Quality Flags", "norm_quality_flags"),
        ("Extraction Failed?", "extraction_failed"),
        ("Reasoning Error", "reasoning_error"),
    ]),
]

FLOWS_SCHEMA: list[tuple[str, list[tuple[str, str]]]] = [
    ("Source", [
        ("Book", "book_title"),
        ("Author", "book_author"),
        ("Gutenberg ID", "gutenberg_id"),
        ("Chunk ID", "chunk_id"),
    ]),
    ("CI Tuple", [
        ("Subject", "ci_subject"),
        ("Sender", "ci_sender"),
        ("Recipient", "ci_recipient"),
        ("Information Type", "ci_information_type"),
        ("Transmission Principle", "ci_transmission_principle"),
    ]),
    ("Flow Context", [
        ("Snippet", "ci_flow_snippet"),
        ("Context", "ci_context"),
        ("Direction", "ci_flow_direction"),
        ("Appropriateness", "ci_appropriateness"),
        ("Norms Invoked", "ci_norms_invoked"),
        ("Norm Source", "ci_norm_source"),
        ("Is New Flow?", "ci_is_new_flow"),
        ("New Flow Reasoning", "ci_is_new_flow_reasoning"),
    ]),
    ("Extraction Meta", [
        ("Reasoning Trace", "ci_reasoning_trace"),
        ("Reasoning Text", "ci_reasoning_text"),
        ("Confidence (qual)", "ci_confidence_qual"),
        ("Confidence (quant)", "ci_confidence_quant"),
        ("Flow Index", "ci_flow_index"),
        ("Flow Count", "ci_flow_count"),
        ("Has Info Exchange?", "has_information_exchange"),
    ]),
]

STAGE_CONFIG = {
    "abstracted_norms": {
        "schema": NORMS_SCHEMA,
        "text_col": "article_text",
        "completion_col": "generated_text",
        "id_cols": ["gutenberg_id", "chunk_id", "norm_index"],
        "facet_cols": [
            "book_title", "raz_normative_force", "raz_context",
            "raz_governs_info_flow", "raz_confidence_qual",
            "norm_quality_passed", "extraction_failed",
            "preliminary_normative_force", "governs_information_flow",
        ],
    },
    "ci_flows": {
        "schema": FLOWS_SCHEMA,
        "text_col": "article_text",
        "completion_col": "generated_text",
        "id_cols": ["gutenberg_id", "chunk_id", "ci_flow_index"],
        "facet_cols": [
            "book_title", "ci_context", "ci_appropriateness",
            "ci_norm_source", "ci_is_new_flow", "ci_confidence_qual",
            "has_information_exchange",
        ],
    },
}


def build_stage_data(
    df: pd.DataFrame,
    stage_key: str,
    max_rows: int | None = None,
    include_completions: bool = False,
) -> tuple[list[dict], dict]:
    """Build JSON-serializable row data for one stage.

    Returns (rows, stage_meta) where stage_meta has col_info and schema.
    """
    cfg = STAGE_CONFIG[stage_key]
    schema = cfg["schema"]
    text_col = cfg["text_col"]
    completion_col = cfg["completion_col"]
    id_cols = cfg["id_cols"]
    facet_cols = cfg["facet_cols"]

    n = len(df)
    if max_rows and max_rows < n:
        n = max_rows

    # Collect all column names referenced by the schema
    schema_cols = set()
    for _, fields in schema:
        for _, col in fields:
            schema_cols.add(col)

    rows = []
    for i in range(n):
        row = df.iloc[i]
        record: dict[str, Any] = {"idx": i}

        # IDs
        for c in id_cols:
            if c in row.index:
                record[c] = _serialize(row[c])

        # Source text
        if text_col in row.index and row[text_col] is not None:
            record["source_text"] = _serialize(row[text_col])

        # LLM completion
        if include_completions and completion_col in row.index and row[completion_col] is not None:
            record["completion"] = _serialize(row[completion_col])

        # All schema fields
        fields: dict[str, Any] = {}
        for _, group_fields in schema:
            for display_name, col in group_fields:
                if col in row.index:
                    val = _serialize(row[col])
                    if val is not None:
                        fields[col] = val
        record["fields"] = fields

        # Facets for filtering
        facets: dict[str, str] = {}
        for c in facet_cols:
            if c in row.index:
                val = _serialize(row[c])
                if val is not None:
                    facets[c] = str(val)
        if facets:
            record["_facets"] = facets

        # Note: _searchText omitted to save space; JS builds it on demand

        rows.append(record)

    # Schema as JSON-friendly structure
    schema_json = [
        {"group": group, "fields": [{"label": label, "col": col} for label, col in group_fields]}
        for group, group_fields in schema
    ]

    stage_meta = {
        "rows": rows,
        "schema": schema_json,
        "id_cols": id_cols,
        "text_col": text_col,
        "completion_col": completion_col,
        "n_total": len(df),
    }
    return rows, stage_meta


# ── HTML template ─────────────────────────────────────────────────────────

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Norms Inspector</title>
<style>
:root {
  --bg: #fafafa; --fg: #1a1a1a; --border: #e0e0e0;
  --accent: #6a1b9a; --accent-light: #f3e5f5;
  --green: #2e7d32; --green-bg: #e8f5e9;
  --red: #c62828; --red-bg: #ffebee;
  --orange: #ef6c00; --orange-bg: #fff3e0;
  --blue: #1565c0; --blue-bg: #e3f2fd;
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
.topbar select { max-width: 280px; }
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
.field-filter label {
  font-weight: 600; color: #555; white-space: nowrap;
}
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
.field-filters .ff-reset:hover { color: #4a148c; }

/* ── Searchable filter ──────────────────────────────────── */
.search-filter {
  position: relative; display: inline-flex; align-items: center; gap: 3px; font-size: 12px;
}
.search-filter label {
  font-weight: 600; color: #555; white-space: nowrap;
}
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
.sf-item .sf-count {
  color: #999; font-size: 11px; min-width: 30px; text-align: right;
}
.sf-clear {
  display: none; position: absolute; right: 4px; top: 50%; transform: translateY(-50%);
  cursor: pointer; font-size: 14px; color: #999; line-height: 1;
  background: none; border: none; padding: 0 2px;
}
.sf-clear:hover { color: var(--red); }
.search-filter.has-value .sf-clear { display: block; }
.search-filter.has-value input { padding-right: 18px; }

/* ── Main content ────────────────────────────────────────── */
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
.row-badges { display: flex; gap: 4px; }
.badge {
  font-size: 11px; padding: 2px 8px; border-radius: 10px; font-weight: 500;
}
.badge.info-flow { background: var(--blue-bg); color: var(--blue); }
.badge.quality-pass { background: var(--green-bg); color: var(--green); }
.badge.quality-fail { background: var(--red-bg); color: var(--red); }
.badge.bookmark-badge { background: var(--orange-bg); color: var(--orange); cursor: pointer; }

.row-body { display: none; padding: 12px; }
.row-card.expanded .row-body { display: block; }

/* ── Source text / completion ────────────────────────────── */
.text-box {
  background: #f5f5f5; border: 1px solid #e8e8e8; border-radius: 6px;
  padding: 10px 14px; margin-bottom: 12px; font-size: 13px;
  max-height: 300px; overflow-y: auto;
}
.text-box summary { cursor: pointer; font-weight: 600; font-size: 12px; color: #555; }
.text-box pre {
  white-space: pre-wrap; word-break: break-word; margin: 6px 0 0;
  font-family: var(--mono); font-size: 12px;
}
.text-box.completion { background: #faf5ff; border-color: #e1bee7; }

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
.field-value.changed {
  background: #fff9c4; padding: 2px 4px; border-radius: 3px;
}

/* ── CI tuple card ───────────────────────────────────────── */
.ci-tuple-card {
  display: grid; grid-template-columns: repeat(5, 1fr); gap: 1px;
  background: var(--border); border-radius: 6px; overflow: hidden;
  margin-bottom: 12px;
}
.ci-tuple-cell {
  background: #fff; padding: 8px 10px; text-align: center;
}
.ci-tuple-cell .cell-label {
  font-size: 10px; font-weight: 600; color: #888;
  text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px;
}
.ci-tuple-cell .cell-value {
  font-size: 13px; font-weight: 500; color: var(--fg);
}

/* ── Norm articulation highlight ─────────────────────────── */
.norm-articulation {
  background: var(--accent-light); border-left: 3px solid var(--accent);
  padding: 8px 14px; margin-bottom: 12px; border-radius: 0 6px 6px 0;
  font-size: 14px; font-style: italic; line-height: 1.5;
}

/* ── JSON pretty-print ───────────────────────────────────── */
.json-block {
  background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 4px;
  padding: 8px 10px; margin: 4px 0; font-size: 12px;
}
.json-key { color: #881391; }
.json-str { color: #0b7285; }
.json-num { color: #d9480f; }
.json-bool { color: #5c940d; font-weight: 600; }
.json-null { color: #868e96; font-style: italic; }

/* ── Search highlight ────────────────────────────────────── */
mark { background: #fff176; padding: 1px 2px; border-radius: 2px; }

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

/* ── Responsive ──────────────────────────────────────────── */
@media (max-width: 700px) {
  .ci-tuple-card { grid-template-columns: repeat(2, 1fr); }
  .field-label { width: 140px; min-width: 140px; }
}

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
.export-preview .export-frame .text-box { max-height:none !important; overflow:visible !important; }
.export-preview .export-frame .field-group { }
.export-preview .export-frame .field-group .field-group-body { display:block; }
.export-actions { padding:10px 16px;border-top:1px solid var(--border);display:flex;gap:8px;justify-content:flex-end; }
.export-actions button { font-size:13px;padding:6px 16px;border:none;border-radius:4px;cursor:pointer;font-weight:500; }
.export-actions .btn-pdf { background:#1976d2;color:#fff; }
.export-actions .btn-html { background:#2e7d32;color:#fff; }
.export-actions .btn-cancel { background:#757575;color:#fff; }
.export-actions button:hover { opacity:0.9; }
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
// ── Data (injected by Python) ────────────────────────────────────────
const DATA = __DATA_PLACEHOLDER__;
const STAGE_KEYS = Object.keys(DATA);

// ── State ────────────────────────────────────────────────────────────
let currentStage = STAGE_KEYS[0] || '';
let currentIdx = 0;
let filteredRows = [];
let allRows = [];
let searchQuery = '';
let bookmarks = new Set();
let bookmarksPanelOpen = false;
let fieldFilters = {};

// ── Init ─────────────────────────────────────────────────────────────
function init() {
  const sel = document.getElementById('stage-select');
  STAGE_KEYS.forEach(k => {
    const opt = document.createElement('option');
    opt.value = k;
    const sd = DATA[k];
    opt.textContent = `${k} (${sd.rows.length}${sd.n_total > sd.rows.length ? ' / ' + sd.n_total : ''} rows)`;
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

  loadStage(currentStage);
}

function loadStage(key) {
  currentStage = key;
  allRows = DATA[key].rows;
  bookmarks.clear();
  searchQuery = '';
  fieldFilters = {};
  document.getElementById('search-input').value = '';
  buildFieldFilters();
  applyFilters();
}

// ── Field filters (facet-based) ─────────────────────────────────────
function _discoverFacets() {
  const DROPDOWN_MAX = 30;  // use <select> up to this many unique values
  const SEARCHABLE_MAX = 5000;  // beyond this, skip entirely
  // Count occurrences (not just unique values)
  const facetCounts = new Map();  // key -> Map<value, count>
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
    // Sort by count descending, then alphabetically
    const entries = [...valCounts.entries()].sort((a, b) => {
      if (b[1] !== a[1]) return b[1] - a[1];
      return a[0].localeCompare(b[0]);
    });
    if (valCounts.size <= DROPDOWN_MAX) {
      // For dropdowns, sort alphabetically/numerically
      const sorted = entries.map(e => e[0]).sort((a, b) => {
        const na = Number(a), nb = Number(b);
        if (!isNaN(na) && !isNaN(nb)) return na - nb;
        return a.localeCompare(b);
      });
      dropdowns.push({key, values: sorted, searchable: false});
    } else {
      // For searchable, keep sorted by count (most common first)
      searchable.push({key, entries, searchable: true});
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
  clearBtn.title = 'Clear filter';
  wrapper.appendChild(clearBtn);

  const dropdown = document.createElement('div');
  dropdown.className = 'sf-dropdown';
  wrapper.appendChild(dropdown);

  let highlighted = -1;
  let visible = [];

  function renderDropdown(query) {
    dropdown.innerHTML = '';
    const q = query.toLowerCase();
    visible = q
      ? entries.filter(([val]) => val.toLowerCase().includes(q))
      : entries.slice(0, 50);  // show top 50 by count when no query
    highlighted = -1;

    if (visible.length === 0) {
      const empty = document.createElement('div');
      empty.style.cssText = 'padding:8px 10px;color:#999;font-size:12px;';
      empty.textContent = 'No matches';
      dropdown.appendChild(empty);
      return;
    }

    visible.forEach(([val, count], i) => {
      const item = document.createElement('div');
      item.className = 'sf-item';

      const name = document.createElement('span');
      // Highlight the matching substring
      if (q) {
        const idx = val.toLowerCase().indexOf(q);
        if (idx >= 0) {
          name.innerHTML = esc(val.slice(0, idx))
            + '<mark>' + esc(val.slice(idx, idx + q.length)) + '</mark>'
            + esc(val.slice(idx + q.length));
        } else {
          name.textContent = val;
        }
      } else {
        name.textContent = val;
      }
      item.appendChild(name);

      const cnt = document.createElement('span');
      cnt.className = 'sf-count';
      cnt.textContent = count.toLocaleString();
      item.appendChild(cnt);

      item.addEventListener('mousedown', (e) => {
        e.preventDefault();  // prevent blur
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

  clearBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    clearValue();
  });

  input.addEventListener('focus', () => {
    // If a value is selected, show it but also open dropdown for changing
    renderDropdown(fieldFilters[key] ? '' : input.value);
    wrapper.classList.add('open');
  });

  input.addEventListener('input', () => {
    // If user starts typing, they're searching — clear active filter temporarily
    renderDropdown(input.value);
    wrapper.classList.add('open');
  });

  input.addEventListener('blur', () => {
    // Small delay so mousedown on item fires first
    setTimeout(() => { wrapper.classList.remove('open'); }, 150);
  });

  input.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      wrapper.classList.remove('open');
      input.blur();
      return;
    }
    if (e.key === 'Enter') {
      e.preventDefault();
      if (highlighted >= 0 && highlighted < visible.length) {
        selectValue(visible[highlighted][0]);
      } else if (visible.length === 1) {
        selectValue(visible[0][0]);
      }
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

let _searchFilterWidgets = [];  // track for reset

function buildFieldFilters() {
  const container = document.getElementById('field-filters');
  container.innerHTML = '';
  fieldFilters = {};
  _searchFilterWidgets = [];

  const {dropdowns, searchable} = _discoverFacets();
  if (dropdowns.length === 0 && searchable.length === 0) return;

  // Regular dropdown filters
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
      opt.value = v;
      opt.textContent = v;
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

  // Searchable filters
  searchable.forEach(({key, entries}) => {
    const w = _buildSearchableFilter(container, key, entries);
    _searchFilterWidgets.push(w);
  });

  if (dropdowns.length > 0 || searchable.length > 0) {
    const reset = document.createElement('span');
    reset.className = 'ff-reset';
    reset.textContent = 'Reset all';
    reset.addEventListener('click', () => {
      _resetAllFilterWidgets();
      refilter();
    });
    container.appendChild(reset);
  }
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
  if (row.source_text) parts.push(row.source_text);
  if (row.completion) parts.push(row.completion);
  const f = row.fields || {};
  for (const v of Object.values(f)) {
    if (v != null) parts.push(String(v));
  }
  const idCols = DATA[currentStage].id_cols || [];
  for (const c of idCols) { if (row[c] != null) parts.push(String(row[c])); }
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

// ── Jump to row ─────────────────────────────────────────────────────
function doJump() {
  const input = document.getElementById('jump-input');
  const target = parseInt(input.value, 10);
  if (isNaN(target)) return;
  // Find this original row index in the filtered list
  const fi = filteredRows.findIndex(r => r.idx === target);
  if (fi >= 0) {
    currentIdx = fi;
    render('center');
  } else {
    // Row exists but is filtered out — clear filters and try again
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

// ── Search ───────────────────────────────────────────────────────────
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
// scrollMode: 'header' = pin to the clicked card's header top position (default)
//             'center' = scroll card to center (for jump/search)
//             'none'   = don't scroll at all
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
    const row = filteredRows[fi];
    container.appendChild(buildRowCard(row, fi));
  }

  if (scrollMode === 'none') return;

  requestAnimationFrame(() => {
    const el = document.querySelector('.row-card.current');
    if (!el) return;
    if (scrollMode === 'center') {
      el.scrollIntoView({block: 'center', behavior: 'smooth'});
    } else {
      // 'header': scroll so the card header is at the top of the viewport,
      // just below the sticky topbar. This prevents the confusing upward jump
      // when a tall card body expands below.
      const topbar = document.getElementById('topbar');
      const offset = topbar ? topbar.getBoundingClientRect().height + 8 : 50;
      const rect = el.getBoundingClientRect();
      const scrollY = window.scrollY + rect.top - offset;
      window.scrollTo({top: Math.max(0, scrollY), behavior: 'smooth'});
    }
  });
}

function buildRowCard(row, filterIdx) {
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

  // ID columns
  const ids = document.createElement('span');
  ids.className = 'row-ids';
  const stageData = DATA[currentStage];
  const idCols = stageData.id_cols || [];
  ids.innerHTML = idCols
    .filter(c => row[c] !== undefined)
    .map(c => `<b>${esc(c)}</b>=${esc(String(row[c]))}`)
    .join('&ensp;');
  header.appendChild(ids);

  // Badges
  const badges = document.createElement('span');
  badges.className = 'row-badges';

  const f = row.fields || {};
  if (currentStage === 'abstracted_norms') {
    // Book badge
    if (f.book_title) {
      const bb = document.createElement('span');
      bb.className = 'badge'; bb.style.background = '#e8eaf6'; bb.style.color = '#283593';
      bb.textContent = f.book_title;
      badges.appendChild(bb);
    }
    // Info flow badge
    if (f.raz_governs_info_flow === true || f.raz_governs_info_flow === 'True') {
      const ib = document.createElement('span');
      ib.className = 'badge info-flow';
      ib.textContent = 'Info Flow';
      badges.appendChild(ib);
    }
    // Quality badge
    if (f.norm_quality_passed !== undefined) {
      const qb = document.createElement('span');
      qb.className = 'badge ' + (f.norm_quality_passed ? 'quality-pass' : 'quality-fail');
      qb.textContent = f.norm_quality_passed ? 'QA pass' : 'QA fail';
      badges.appendChild(qb);
    }
  } else if (currentStage === 'ci_flows') {
    if (f.book_title) {
      const bb = document.createElement('span');
      bb.className = 'badge'; bb.style.background = '#e8eaf6'; bb.style.color = '#283593';
      bb.textContent = f.book_title;
      badges.appendChild(bb);
    }
    if (f.ci_appropriateness) {
      const ab = document.createElement('span');
      const appr = String(f.ci_appropriateness).toLowerCase();
      ab.className = 'badge';
      if (appr === 'appropriate') { ab.style.background = '#e8f5e9'; ab.style.color = '#2e7d32'; }
      else if (appr === 'inappropriate') { ab.style.background = '#ffebee'; ab.style.color = '#c62828'; }
      else { ab.style.background = '#fff3e0'; ab.style.color = '#ef6c00'; }
      ab.textContent = f.ci_appropriateness;
      badges.appendChild(ab);
    }
  }

  // Bookmark
  const bmBadge = document.createElement('span');
  bmBadge.className = 'badge bookmark-badge';
  bmBadge.textContent = bookmarks.has(row.idx) ? '\u2605' : '\u2606';
  bmBadge.title = 'Toggle bookmark (b)';
  bmBadge.addEventListener('click', (e) => {
    e.stopPropagation();
    toggleBookmark(row.idx);
  });
  badges.appendChild(bmBadge);

  header.appendChild(badges);
  card.appendChild(header);

  // Body
  if (filterIdx === currentIdx) {
    const body = document.createElement('div');
    body.className = 'row-body';
    body.style.display = 'block';
    body.innerHTML = buildRowBody(row);
    card.appendChild(body);
  }

  return card;
}

function buildRowBody(row) {
  const stageData = DATA[currentStage];
  const schema = stageData.schema;
  const f = row.fields || {};
  let html = '';

  // For ci_flows: show the CI tuple card prominently
  if (currentStage === 'ci_flows') {
    const tupleFields = [
      {label: 'Subject', col: 'ci_subject'},
      {label: 'Sender', col: 'ci_sender'},
      {label: 'Recipient', col: 'ci_recipient'},
      {label: 'Info Type', col: 'ci_information_type'},
      {label: 'Transmission', col: 'ci_transmission_principle'},
    ];
    html += '<div class="ci-tuple-card" data-export="ci-tuple">';
    tupleFields.forEach(({label, col}) => {
      html += `<div class="ci-tuple-cell">
        <div class="cell-label">${esc(label)}</div>
        <div class="cell-value">${esc(f[col] || '—')}</div>
      </div>`;
    });
    html += '</div>';
  }

  // For abstracted_norms: show the articulation prominently
  if (currentStage === 'abstracted_norms' && f.raz_norm_articulation) {
    html += `<div class="norm-articulation" data-export="norm-articulation">${highlightSearch(esc(f.raz_norm_articulation))}</div>`;
  }

  // Source text
  if (row.source_text) {
    const isLong = row.source_text.length > 500;
    html += `<div class="text-box" data-export="source-text">
      <details${isLong ? '' : ' open'}>
        <summary>Source Text (${row.source_text.length.toLocaleString()} chars)</summary>
        <pre>${highlightSearch(esc(row.source_text))}</pre>
      </details>
    </div>`;
  }

  // LLM completion
  if (row.completion) {
    const isLong = row.completion.length > 300;
    html += `<div class="text-box completion" data-export="completion">
      <details${isLong ? '' : ' open'}>
        <summary>LLM Completion (${row.completion.length.toLocaleString()} chars)</summary>
        <pre>${highlightSearch(formatTextWithJson(row.completion))}</pre>
      </details>
    </div>`;
  }

  // Structured field groups
  schema.forEach(({group, fields}, gi) => {
    // Skip CI Tuple group for ci_flows (already shown as card above)
    if (currentStage === 'ci_flows' && group === 'CI Tuple') return;

    const hasData = fields.some(({col}) => f[col] !== undefined && f[col] !== null);
    if (!hasData) return;

    // Role Abstraction: open by default to highlight orig vs abstracted
    const defaultOpen = (group === 'Role Abstraction' || group === 'Raz Norm \u2014 Structured'
                         || group === 'CI Tuple' || group === 'Flow Context');

    html += `<div class="field-group${defaultOpen ? ' open' : ''}" data-group="${gi}" data-export="group-${gi}">`;
    html += `<div class="field-group-header" onclick="this.parentElement.classList.toggle('open')">${esc(group)}</div>`;
    html += '<div class="field-group-body">';

    fields.forEach(({label, col}) => {
      const val = f[col];
      if (val === undefined || val === null) return;

      let valClass = 'field-value';
      let displayVal;
      if (typeof val === 'boolean') {
        valClass += val ? ' bool-true' : ' bool-false';
        displayVal = String(val);
      } else if (typeof val === 'object') {
        displayVal = formatTextWithJson(JSON.stringify(val, null, 2));
      } else {
        displayVal = highlightSearch(esc(String(val)));
      }

      // Highlight role abstraction changes
      if (group === 'Role Abstraction') {
        const origPrefix = 'orig_raz_';
        const abstracted = col;
        const isOrigCol = col.startsWith('orig_');
        if (!isOrigCol && f['orig_' + col] !== undefined && String(f['orig_' + col]) !== String(val)) {
          valClass += ' changed';
        }
      }

      html += `<div class="field-row">
        <span class="field-label">${esc(label)}</span>
        <span class="${valClass}">${displayVal}</span>
      </div>`;
    });

    html += '</div></div>';
  });

  return html;
}

// ── Helpers ──────────────────────────────────────────────────────────
function esc(s) {
  if (s == null) return '';
  const d = document.createElement('div');
  d.textContent = String(s);
  return d.innerHTML;
}

function highlightSearch(html) {
  if (!searchQuery) return html;
  try {
    const re = new RegExp(`(${searchQuery})`, 'gi');
    return html.replace(re, '<mark>$1</mark>');
  } catch (e) {
    return html;
  }
}

function syntaxHighlightJson(jsonStr) {
  return jsonStr.replace(
    /("(?:\\.|[^"\\])*")\s*:/g,
    '<span class="json-key">$1</span>:'
  ).replace(
    /:\s*("(?:\\.|[^"\\])*")/g,
    ': <span class="json-str">$1</span>'
  ).replace(
    /:\s*(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\b/g,
    ': <span class="json-num">$1</span>'
  ).replace(
    /:\s*(true|false)\b/g,
    ': <span class="json-bool">$1</span>'
  ).replace(
    /:\s*(null)\b/g,
    ': <span class="json-null">$1</span>'
  );
}

function _tryParseJsonAt(text, i) {
  const ch = text[i];
  if (ch !== '{' && ch !== '[') return null;
  const close = ch === '{' ? '}' : ']';
  let depth = 0, inStr = false, esc = false;
  for (let j = i; j < text.length; j++) {
    const c = text[j];
    if (esc) { esc = false; continue; }
    if (c === '\\' && inStr) { esc = true; continue; }
    if (c === '"' && !inStr) { inStr = true; continue; }
    if (c === '"' && inStr) { inStr = false; continue; }
    if (inStr) continue;
    if (c === ch) depth++;
    else if (c === close) depth--;
    if (depth === 0) {
      const candidate = text.slice(i, j + 1);
      try {
        const parsed = JSON.parse(candidate);
        return {end: j + 1, parsed};
      } catch (e) { return null; }
    }
  }
  return null;
}

function formatTextWithJson(rawText) {
  const result = [];
  let i = 0, plainStart = 0;
  while (i < rawText.length) {
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
  if (plainStart < rawText.length) result.push(esc(rawText.slice(plainStart)));
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
      if (badge) badge.textContent = isBookmarked ? '\u2605' : '\u2606';
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
  sorted.forEach(idx => {
    const row = allRows.find(r => r.idx === idx);
    if (!row) return;
    const item = document.createElement('div');
    item.className = 'bookmark-item';
    const idCols = DATA[currentStage].id_cols || [];
    const idStr = idCols.filter(c => row[c] !== undefined).map(c => `${c}=${row[c]}`).join(', ');
    item.textContent = `#${idx}${idStr ? ' \u2014 ' + idStr : ''}`;
    item.addEventListener('click', () => {
      const fi = filteredRows.findIndex(r => r.idx === idx);
      if (fi >= 0) { currentIdx = fi; render('center'); }
      else {
        _resetAllFilterWidgets();
        applyFilters();
        const fi2 = filteredRows.findIndex(r => r.idx === idx);
        if (fi2 >= 0) { currentIdx = fi2; render('center'); }
      }
    });
    list.appendChild(item);
  });
}

function exportBookmarks() {
  if (bookmarks.size === 0) { alert('No bookmarks to export.'); return; }
  const exported = [];
  [...bookmarks].sort((a, b) => a - b).forEach(idx => {
    const row = allRows.find(r => r.idx === idx);
    if (row) exported.push({stage: currentStage, ...row});
  });
  const blob = new Blob([JSON.stringify(exported, null, 2)], {type: 'application/json'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = `bookmarks_${currentStage}.json`;
  a.click(); URL.revokeObjectURL(url);
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

// ── Export Row ──────────────────────────────────────────────────────
let _exportRow = null;

function showExportModal() {
  const row = filteredRows[currentIdx];
  if (!row) return;
  _exportRow = row;

  const stageData = DATA[currentStage];
  const schema = stageData.schema;
  const f = row.fields || {};

  document.getElementById('export-title').textContent = `Export Row #${row.idx} — ${currentStage}`;

  // Build toggle controls dynamically from the row content
  const ctrl = document.getElementById('export-controls');
  ctrl.innerHTML = '';

  const sections = [];

  // Stage-specific top-level sections
  if (currentStage === 'ci_flows') {
    sections.push({id: 'ci-tuple', label: 'CI Tuple', on: true});
  }
  if (currentStage === 'abstracted_norms' && f.raz_norm_articulation) {
    sections.push({id: 'norm-articulation', label: 'Norm Articulation', on: true});
  }
  if (row.source_text) sections.push({id: 'source-text', label: 'Source Text', on: false});
  if (row.completion) sections.push({id: 'completion', label: 'LLM Completion', on: false});

  // Field groups from schema
  schema.forEach(({group, fields}, gi) => {
    if (currentStage === 'ci_flows' && group === 'CI Tuple') return;
    const hasData = fields.some(({col}) => f[col] !== undefined && f[col] !== null);
    if (!hasData) return;
    sections.push({id: `group-${gi}`, label: group, on: true});
  });

  sections.forEach(s => {
    const lbl = document.createElement('label');
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = s.on;
    cb.dataset.section = s.id;
    cb.addEventListener('change', updateExportPreview);
    lbl.appendChild(cb);
    lbl.appendChild(document.createTextNode(' ' + s.label));
    ctrl.appendChild(lbl);
  });

  // Width control
  const wsep = document.createElement('span');
  wsep.style.cssText = 'width:1px;height:18px;background:#ccc;margin:0 4px;';
  ctrl.appendChild(wsep);

  const wlbl = document.createElement('span');
  wlbl.style.fontWeight = '600';
  wlbl.textContent = 'Width:';
  ctrl.appendChild(wlbl);

  const wInput = document.createElement('input');
  wInput.type = 'number';
  wInput.id = 'export-width';
  wInput.value = 1100;
  wInput.min = 400;
  wInput.max = 2400;
  wInput.step = 50;
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

function getExportOptions() {
  const opts = {};
  document.querySelectorAll('#export-controls input[type=checkbox]').forEach(cb => {
    opts[cb.dataset.section] = cb.checked;
  });
  return opts;
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

  // Force all <details> open and field-groups open
  clone.querySelectorAll('details').forEach(d => d.setAttribute('open', ''));
  clone.querySelectorAll('.field-group').forEach(g => g.classList.add('open'));

  // Remove search highlights
  clone.querySelectorAll('mark').forEach(m => {
    m.replaceWith(document.createTextNode(m.textContent));
  });

  // Remove max-height constraints
  clone.querySelectorAll('pre').forEach(p => { p.style.maxHeight = 'none'; });
  clone.querySelectorAll('.text-box').forEach(p => { p.style.maxHeight = 'none'; });

  // Apply section visibility from toggles
  const opts = getExportOptions();
  clone.querySelectorAll('[data-export]').forEach(el => {
    const key = el.getAttribute('data-export');
    if (key in opts) {
      el.classList.toggle('export-hidden', !opts[key]);
    }
  });

  frame.innerHTML = '';
  frame.appendChild(clone);
}

function closeExportModal() {
  document.getElementById('export-overlay').style.display = 'none';
  _exportRow = null;
}

function _getExportCSS() {
  // Self-contained export CSS — no extracted page rules, no CSS variables
  return `
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; font-size: 13px; color: #1a1a1a; }
    .row-body { display: block; padding: 0; }
    details { margin-bottom: 8px; }
    details > summary { cursor: default; font-weight: 600; font-size: 12px; color: #555; margin-bottom: 4px; text-transform: capitalize; }
    pre { white-space: pre-wrap; word-break: break-word; font-family: 'SF Mono','Cascadia Code',Consolas,monospace; font-size: 11.5px; line-height: 1.5; margin: 0; }
    .text-box { background: #f5f5f5; border: 1px solid #e8e8e8; border-radius: 6px; padding: 10px 14px; margin-bottom: 10px; }
    .text-box.completion { background: #faf5ff; border-color: #e1bee7; }
    .norm-articulation { background: #f3e5f5; border-left: 3px solid #6a1b9a; padding: 8px 14px; margin-bottom: 10px; border-radius: 0 6px 6px 0; font-size: 14px; font-style: italic; line-height: 1.5; }
    .ci-tuple-card { display: grid; grid-template-columns: repeat(5, 1fr); gap: 1px; background: #e0e0e0; border-radius: 6px; overflow: hidden; margin-bottom: 10px; }
    .ci-tuple-cell { background: #fff; padding: 8px 10px; text-align: center; }
    .ci-tuple-cell .cell-label { font-size: 10px; font-weight: 600; color: #888; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px; }
    .ci-tuple-cell .cell-value { font-size: 13px; font-weight: 500; }
    .field-group { margin-bottom: 10px; border: 1px solid #e0e0e0; border-radius: 6px; overflow: hidden; }
    .field-group-header { padding: 6px 12px; font-weight: 600; font-size: 12px; background: #f5f5f5; border-bottom: 1px solid #e0e0e0; cursor: default; }
    .field-group-body { display: block !important; }
    .field-row { display: flex; padding: 4px 12px; border-bottom: 1px solid #f0f0f0; font-size: 13px; }
    .field-row:last-child { border-bottom: none; }
    .field-label { width: 200px; min-width: 200px; font-weight: 500; color: #555; padding-right: 12px; }
    .field-value { flex: 1; font-family: 'SF Mono',Consolas,monospace; font-size: 12px; white-space: pre-wrap; word-break: break-word; }
    .field-value.bool-true { color: #2e7d32; font-weight: 600; }
    .field-value.bool-false { color: #c62828; font-weight: 600; }
    .field-value.changed { background: #fff9c4; padding: 2px 4px; border-radius: 3px; }
    .json-block { background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 4px; padding: 6px 8px; margin: 4px 0; font-size: 11px; }
    .json-key { color: #881391; } .json-str { color: #0b7285; } .json-num { color: #d9480f; }
    .json-bool { color: #5c940d; font-weight: 600; } .json-null { color: #868e96; font-style: italic; }
  `;
}

function _getExportHTML() {
  const frame = document.getElementById('export-frame');
  const clone = frame.cloneNode(true);
  clone.querySelectorAll('.export-hidden').forEach(el => el.remove());
  clone.querySelectorAll('[data-export]').forEach(el => el.removeAttribute('data-export'));
  // Remove onclick handlers from field-group-headers
  clone.querySelectorAll('.field-group-header').forEach(el => el.removeAttribute('onclick'));
  return clone.innerHTML;
}

function _exportFilename(ext) {
  const row = _exportRow;
  if (!row) return `export.${ext}`;
  const stage = currentStage.replace(/[^a-zA-Z0-9]/g, '_');
  return `row${row.idx}_${stage}.${ext}`;
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

// ── Start ────────────────────────────────────────────────────────────
init();
</script>
</body>
</html>"""


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate a static HTML inspector for extracted norms and CI flows.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data", required=True,
        help="Path to data directory containing parquet files",
    )
    parser.add_argument(
        "-o", "--output", default="norms_inspector.html",
        help="Output HTML file path (default: norms_inspector.html)",
    )
    parser.add_argument(
        "--stages", nargs="*", default=None,
        help=f"Stages to include (default: all). Choices: {list(KNOWN_STAGES.keys())}",
    )
    parser.add_argument(
        "--max-rows", type=int, default=None,
        help="Maximum rows per stage (default: all)",
    )
    parser.add_argument(
        "--books", nargs="*", default=None,
        help="Filter to specific book titles (substring match)",
    )
    parser.add_argument(
        "--truncate-text", type=int, default=2000, metavar="N",
        help="Truncate source_text to N chars (default: 2000, 0 = no truncation)",
    )
    parser.add_argument(
        "--include-completions", action="store_true",
        help="Include raw LLM completion text (excluded by default since fields are extracted)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data)
    if not data_dir.is_dir():
        print(f"ERROR: Data directory does not exist: {data_dir}", file=sys.stderr)
        sys.exit(1)

    stages = discover_stages(data_dir)
    if args.stages:
        stages = {k: v for k, v in stages.items() if k in args.stages}

    if not stages:
        print(f"ERROR: No parquet files found in {data_dir}", file=sys.stderr)
        print(f"  Expected: {list(KNOWN_STAGES.values())}", file=sys.stderr)
        sys.exit(1)

    print(f"Data directory: {data_dir}")
    print(f"Stages found: {list(stages.keys())}")

    data = {}
    for stage_key, pq_path in sorted(stages.items()):
        print(f"\nProcessing: {stage_key} ...", end=" ", flush=True)
        df = pd.read_parquet(pq_path)

        # Drop book_summary (same per book, very large)
        if "book_summary" in df.columns:
            df = df.drop(columns=["book_summary"])

        # Truncate source text to keep HTML manageable
        text_col = STAGE_CONFIG[stage_key]["text_col"]
        if args.truncate_text and text_col in df.columns:
            df[text_col] = df[text_col].apply(
                lambda x: (x[:args.truncate_text] + "\n... [truncated]")
                if isinstance(x, str) and len(x) > args.truncate_text else x
            )

        # Filter by book title if requested
        if args.books and "book_title" in df.columns:
            mask = pd.Series(False, index=df.index)
            for book in args.books:
                mask |= df["book_title"].str.contains(book, case=False, na=False)
            df = df[mask].reset_index(drop=True)
            print(f"(filtered to {len(df)} rows by book)", end=" ", flush=True)

        rows, stage_meta = build_stage_data(
            df, stage_key, max_rows=args.max_rows,
            include_completions=args.include_completions,
        )
        data[stage_key] = stage_meta
        print(f"{len(rows)} rows")

    # Generate HTML
    print(f"\nGenerating HTML ...", end=" ", flush=True)
    data_json = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    html = HTML_TEMPLATE.replace("__DATA_PLACEHOLDER__", data_json)

    out_path = Path(args.output)
    out_path.write_text(html, encoding="utf-8")
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"done ({size_mb:.1f} MB)")
    print(f"\nOutput: {out_path.resolve()}")
    print(f"Open in browser: file://{out_path.resolve()}")


if __name__ == "__main__":
    main()
