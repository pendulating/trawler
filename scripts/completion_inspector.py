#!/usr/bin/env python3
"""Completion Inspector — static HTML generator for side-by-side model comparison.

Reads eval_all run outputs (parquet files) for N models and generates a single
self-contained HTML file with an interactive comparison UI.

Usage:
    python -m scripts.completion_inspector \\
        --runs "Base=/path/to/run_a" "SFT=/path/to/run_b" "GRPO=/path/to/run_c" \\
        -o inspection.html

    # Compare two runs with auto-labels:
    python -m scripts.completion_inspector \\
        --runs /path/to/run_a /path/to/run_b \\
        -o inspection.html

    # Limit rows per stage (for large datasets):
    python -m scripts.completion_inspector \\
        --runs "A=/path/a" "B=/path/b" --max-rows 500 -o inspection.html

    # Row subset with Python slice syntax:
    python -m scripts.completion_inspector \\
        --runs "A=/path/a" "B=/path/b" --rows "0:100" -o first100.html

    # Last 50 rows (use = for negative indices):
    python -m scripts.completion_inspector \\
        --runs "A=/path/a" "B=/path/b" --rows="-50:" -o last50.html

    # Every 10th row:
    python -m scripts.completion_inspector \\
        --runs "A=/path/a" "B=/path/b" --rows "::10" -o sampled.html

    # Specific ranges combined:
    python -m scripts.completion_inspector \\
        --runs "A=/path/a" "B=/path/b" --rows "0:10,50:60,100" -o selection.html
"""

from __future__ import annotations

import argparse
import html as html_lib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ── Auto-discovery ────────────────────────────────────────────────────────

def resolve_root(run_path: str) -> Path:
    """Resolve eval_all run root, handling /0/ multirun subdirectory."""
    p = Path(run_path)
    if (p / "0").is_dir():
        return p / "0"
    return p


def _parse_stage_key(pq: Path, root: Path) -> tuple[str, str, str] | None:
    """Parse a parquet path into (benchmark, stage_name, stage_key)."""
    rel = pq.relative_to(root)
    parts = rel.parts
    if "outputs" in parts:
        oi = parts.index("outputs")
        benchmark = parts[0] if len(parts) > 0 else "unknown"
        stage = parts[oi + 1] if oi + 1 < len(parts) else pq.stem
        return benchmark, stage, f"{benchmark} / {stage}"
    return None


def discover_stages(root: Path) -> dict[str, Path]:
    """Walk a run root and find all parquet files with generated_text.

    Returns {stage_key: parquet_path} where stage_key is like
    "goldcoin / llm_inference_applicability".
    """
    found = {}
    for pq in sorted(root.rglob("*.parquet")):
        # Skip metrics files (typically 1 row of aggregated stats)
        if "metrics" in pq.stem:
            continue
        # Quick check: does this file have generated_text? (schema-only read)
        try:
            import pyarrow.parquet as _pq
            schema = _pq.read_schema(pq)
            cols = schema.names
        except Exception:
            continue
        if "generated_text" not in cols:
            continue

        parsed = _parse_stage_key(pq, root)
        if parsed:
            _, _, key = parsed
        else:
            key = str(pq.relative_to(root))
        found[key] = pq
    return found


# Judge stages and which primary stage they attach to.
# key = judge stage name, value includes the parent stage it should attach to.
_JUDGE_COLUMNS = {
    "leakage_judge_inference": {
        "text_col": "leak_judge_text",
        "verdict_cols": ["leak_probability", "leak_flag"],
        "display_name": "Leakage Judge",
        "parent_stage": "agent_action_inference",
    },
    "helpfulness_judge_inference": {
        "text_col": "helpfulness_judge_text",
        "verdict_cols": ["helpfulness_score", "helpfulness_binary"],
        "display_name": "Helpfulness Judge",
        "parent_stage": "agent_action_inference",
    },
}


def discover_judge_stages(root: Path, primary_stages: dict[str, Path]) -> dict[str, list[dict]]:
    """Find judge stages related to primary stages.

    Returns {primary_stage_key: [{path, judge_name, text_col, verdict_cols, display_name}, ...]}.
    Judges are only attached to their designated parent stage.
    """
    # Build index of all parquets (including non-generated_text ones)
    all_parquets: dict[tuple[str, str], Path] = {}
    for pq in sorted(root.rglob("*.parquet")):
        if "metrics" in pq.stem:
            continue
        parsed = _parse_stage_key(pq, root)
        if parsed:
            benchmark, stage, _ = parsed
            all_parquets[(benchmark, stage)] = pq

    result: dict[str, list[dict]] = {}
    for primary_key, primary_path in primary_stages.items():
        parsed = _parse_stage_key(primary_path, root)
        if not parsed:
            continue
        benchmark, primary_stage, _ = parsed

        judges = []
        for judge_stage, judge_info in _JUDGE_COLUMNS.items():
            # Only attach judge to its designated parent stage
            if judge_info["parent_stage"] != primary_stage:
                continue
            if (benchmark, judge_stage) in all_parquets:
                judges.append({
                    "path": all_parquets[(benchmark, judge_stage)],
                    "judge_name": judge_stage,
                    **{k: v for k, v in judge_info.items() if k != "parent_stage"},
                })
        if judges:
            result[primary_key] = judges

    return result


# ── Column classification ────────────────────────────────────────────────

def _classify_columns(df: pd.DataFrame) -> dict:
    """Classify DataFrame columns into semantic categories."""
    cols = list(df.columns)
    info: dict[str, Any] = {
        "id_cols": [],
        "context_cols": [],
        "ground_truth_cols": [],
        "prediction_cols": [],
        "completion_col": "generated_text" if "generated_text" in cols else None,
        "has_messages": "messages" in cols,
    }

    for c in cols:
        cl = c.lower()
        # IDs
        if cl.endswith("_id") or cl in ("case_id", "record_id", "image_id", "row_id", "name"):
            info["id_cols"].append(c)
        # Ground truth
        elif cl == "ground_truth" or cl.endswith("_true"):
            info["ground_truth_cols"].append(c)
        # Predictions
        elif cl in ("prediction", "predicted_label") or cl.endswith("_pred"):
            info["prediction_cols"].append(c)
        # Context (verbose text fields that provide background)
        elif cl in ("vignette", "generate_background", "story", "text"):
            info["context_cols"].append(c)

    return info


# ── Data serialization ───────────────────────────────────────────────────

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


def _extract_user_prompt(messages: Any) -> str | None:
    """Pull last user-role message content from messages array."""
    if messages is None:
        return None
    msgs = list(messages) if isinstance(messages, np.ndarray) else messages
    if not isinstance(msgs, list):
        return None
    user_msgs = [m for m in msgs if isinstance(m, dict) and m.get("role") == "user"]
    if not user_msgs:
        return None
    content = user_msgs[-1].get("content", "")
    # Content may be a list of dicts (multimodal) or a string
    if isinstance(content, list):
        text_parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
        return "\n".join(text_parts) if text_parts else str(content)
    return str(content)


def build_stage_data(
    label_to_df: dict[str, pd.DataFrame],
    col_info: dict,
    max_rows: int | None = None,
    judge_dfs: dict[str, dict[str, pd.DataFrame]] | None = None,
) -> list[dict]:
    """Build the row-level comparison data for one stage across all models.

    Args:
        label_to_df: {model_label: DataFrame} for the primary stage.
        col_info: Column classification from _classify_columns.
        max_rows: Limit rows.
        judge_dfs: {judge_display_name: {model_label: DataFrame}} for related
            judge stages. Each judge DataFrame must have the same row count/order
            as the primary stage.
    """
    labels = list(label_to_df.keys())
    dfs = list(label_to_df.values())
    n = min(len(df) for df in dfs)
    if max_rows and max_rows < n:
        n = max_rows

    rows = []
    for i in range(n):
        record: dict[str, Any] = {"idx": i}

        # Use first model's data for shared fields
        ref = dfs[0].iloc[i]

        # IDs
        for c in col_info["id_cols"]:
            if c in ref.index:
                record[c] = _serialize(ref[c])

        # Prompt (from messages)
        if col_info["has_messages"] and "messages" in ref.index:
            record["prompt"] = _extract_user_prompt(ref["messages"])

        # Context
        for c in col_info["context_cols"]:
            if c in ref.index and ref[c] is not None:
                record[c] = _serialize(ref[c])

        # Ground truth
        for c in col_info["ground_truth_cols"]:
            if c in ref.index:
                record[c] = _serialize(ref[c])

        # Per-model: completions, predictions, usage
        completions = {}
        predictions = {}
        for label, df in zip(labels, dfs):
            row = df.iloc[i]
            if col_info["completion_col"] and col_info["completion_col"] in row.index:
                completions[label] = _serialize(row[col_info["completion_col"]])
            for c in col_info["prediction_cols"]:
                if c in row.index:
                    predictions.setdefault(c, {})[label] = _serialize(row[c])

        record["completions"] = completions
        if predictions:
            record["predictions"] = predictions

        # Agreement/correctness flags
        if col_info["ground_truth_cols"] and col_info["prediction_cols"]:
            gt_col = col_info["ground_truth_cols"][0]
            pred_col = col_info["prediction_cols"][0]
            gt_val = _serialize(ref.get(gt_col))
            correctness = {}
            for label, df in zip(labels, dfs):
                row = df.iloc[i]
                if pred_col in row.index:
                    pv = _serialize(row[pred_col])
                    if gt_val is not None and pv is not None:
                        correctness[label] = str(pv).strip().lower() == str(gt_val).strip().lower()
            if correctness:
                record["correctness"] = correctness

        # Judge results (per model, per judge type)
        if judge_dfs:
            judges: dict[str, dict] = {}
            for judge_name, label_to_jdf in judge_dfs.items():
                per_model: dict[str, dict] = {}
                for label in labels:
                    if label not in label_to_jdf:
                        continue
                    jdf = label_to_jdf[label]
                    if i >= len(jdf):
                        continue
                    jrow = jdf.iloc[i]
                    entry: dict[str, Any] = {}
                    # Include all judge-specific columns
                    for c in jdf.columns:
                        if c in ("name", "record_id", "messages", "sampling_params",
                                 "usage", "generated_text", "generated_action",
                                 "seed", "vignette", "trajectory", "S", "V", "T",
                                 "benchmark_name", "split"):
                            continue  # skip shared/bulky columns
                        entry[c] = _serialize(jrow[c])
                    if entry:
                        per_model[label] = entry
                if per_model:
                    judges[judge_name] = per_model
            if judges:
                record["judges"] = judges

        # Pre-compute flat facets for filtering (no JS path traversal needed)
        facets: dict[str, str] = {}
        for c in col_info["ground_truth_cols"]:
            val = record.get(c)
            if val is not None:
                facets[c] = str(val)
        for pred_col, label_map in predictions.items():
            for label, val in label_map.items():
                if val is not None:
                    key = pred_col if len(labels) == 1 else f"{pred_col} ({label})"
                    facets[key] = str(val)
        if judge_dfs and "judges" in record:
            for judge_name, per_model in record["judges"].items():
                short_judge = judge_name.replace(" Judge", "")
                for label, entry in per_model.items():
                    for field, val in entry.items():
                        if field.endswith("_text"):
                            continue
                        if val is not None:
                            key = (f"{short_judge}: {field}"
                                   if len(labels) == 1
                                   else f"{short_judge}: {field} ({label})")
                            facets[key] = str(val)
        if facets:
            record["_facets"] = facets

        # Build searchable text blob for fast full-text search
        search_parts: list[str] = []
        if record.get("prompt"):
            search_parts.append(record["prompt"])
        for c in col_info["context_cols"]:
            val = record.get(c)
            if val is not None:
                search_parts.append(str(val) if isinstance(val, str) else json.dumps(val))
        for text in completions.values():
            if text is not None:
                search_parts.append(str(text))
        for c in col_info["id_cols"]:
            val = record.get(c)
            if val is not None:
                search_parts.append(str(val))
        for c in col_info["ground_truth_cols"]:
            val = record.get(c)
            if val is not None:
                search_parts.append(str(val))
        if search_parts:
            record["_searchText"] = "\n".join(search_parts)

        rows.append(record)

    return rows


# ── HTML template ─────────────────────────────────────────────────────────

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Completion Inspector</title>
<style>
:root {
  --bg: #fafafa; --fg: #1a1a1a; --border: #e0e0e0;
  --accent: #1976d2; --accent-light: #e3f2fd;
  --green: #2e7d32; --green-bg: #e8f5e9;
  --red: #c62828; --red-bg: #ffebee;
  --orange: #ef6c00; --orange-bg: #fff3e0;
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

/* ── Filter chips ────────────────────────────────────────── */
.filters { display: flex; gap: 4px; flex-wrap: wrap; }
.chip {
  font-size: 12px; padding: 3px 10px; border-radius: 12px; cursor: pointer;
  border: 1px solid var(--border); background: #fff; transition: all 0.15s;
}
.chip:hover { border-color: var(--accent); }
.chip.active { background: var(--accent); color: #fff; border-color: var(--accent); }

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
  border-radius: 4px; background: #fff; max-width: 180px;
}
.field-filter select.active-filter {
  border-color: var(--accent); background: var(--accent-light);
}
.field-filters .ff-reset {
  font-size: 11px; color: var(--accent); cursor: pointer;
  text-decoration: underline; margin-left: 4px;
}
.field-filters .ff-reset:hover { color: #1565c0; }

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
.badge.correct { background: var(--green-bg); color: var(--green); }
.badge.wrong { background: var(--red-bg); color: var(--red); }
.badge.bookmark-badge { background: var(--orange-bg); color: var(--orange); cursor: pointer; }

.row-body { display: none; padding: 12px; }
.row-card.expanded .row-body { display: block; }

/* ── Prompt / context ────────────────────────────────────── */
.prompt-box {
  background: #f5f5f5; border: 1px solid #e8e8e8; border-radius: 6px;
  padding: 10px 14px; margin-bottom: 12px; font-size: 13px;
  max-height: 300px; overflow-y: auto;
}
.prompt-box summary { cursor: pointer; font-weight: 600; font-size: 12px; color: #555; }
.prompt-box pre { white-space: pre-wrap; word-break: break-word; margin: 6px 0 0; font-family: var(--mono); font-size: 12px; }

/* ── Completion grid ─────────────────────────────────────── */
.completions-grid {
  display: grid; gap: 8px; margin-bottom: 12px;
}
.completion-col {
  border: 1px solid var(--border); border-radius: 6px; overflow: hidden;
}
.completion-col .col-header {
  padding: 6px 10px; font-weight: 600; font-size: 12px;
  border-bottom: 1px solid var(--border);
}
.completion-col pre {
  padding: 10px; white-space: pre-wrap; word-break: break-word;
  font-family: var(--mono); font-size: 12px; line-height: 1.5;
  max-height: 500px; overflow-y: auto; margin: 0;
}
.diff-add { background: #d4edda; }
.diff-del { background: #f8d7da; }

/* ── Ground truth / predictions ──────────────────────────── */
.meta-row {
  display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 8px;
  font-size: 13px;
}
.meta-row .label { font-weight: 600; color: #555; }
.meta-row .val { font-family: var(--mono); }
.meta-row .val.correct { color: var(--green); font-weight: 600; }
.meta-row .val.wrong { color: var(--red); font-weight: 600; }

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

/* ── Judge panels ────────────────────────────────────────── */
.judge-section { margin-bottom: 12px; }
.judge-toggle {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 4px 10px; font-size: 12px; font-weight: 600;
  background: #f5f5f5; border: 1px solid var(--border); border-radius: 4px;
  cursor: pointer; user-select: none; margin-bottom: 6px;
}
.judge-toggle:hover { background: #eee; }
.judge-toggle .arrow { transition: transform 0.15s; display: inline-block; }
.judge-toggle.open .arrow { transform: rotate(90deg); }
.judge-grid {
  display: none; gap: 8px; margin-top: 6px;
}
.judge-grid.open { display: grid; }
.judge-card {
  border: 1px solid #e0d6f5; border-radius: 6px; overflow: hidden;
}
.judge-card .judge-header {
  padding: 5px 10px; font-size: 11px; font-weight: 600;
  background: #f3e5f5; border-bottom: 1px solid #e0d6f5;
  display: flex; gap: 8px; flex-wrap: wrap;
}
.judge-card .judge-header .verdict {
  padding: 1px 6px; border-radius: 8px; font-weight: 700; font-size: 11px;
}
.judge-card .judge-header .verdict.leak { background: var(--red-bg); color: var(--red); }
.judge-card .judge-header .verdict.no-leak { background: var(--green-bg); color: var(--green); }
.judge-card .judge-header .verdict.helpful { background: var(--green-bg); color: var(--green); }
.judge-card .judge-header .verdict.not-helpful { background: var(--orange-bg); color: var(--orange); }
.judge-card pre {
  padding: 8px 10px; white-space: pre-wrap; word-break: break-word;
  font-family: var(--mono); font-size: 11px; line-height: 1.4;
  max-height: 300px; overflow-y: auto; margin: 0; background: #faf5ff;
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

/* ── Responsive ──────────────────────────────────────────── */
@media (max-width: 900px) {
  .completions-grid { grid-template-columns: 1fr !important; }
}
</style>
</head>
<body>

<div class="topbar" id="topbar">
  <select id="stage-select"></select>
  <div class="sep"></div>
  <input type="text" id="search-input" placeholder="Search (regex)...">
  <button id="search-btn">Search</button>
  <button id="clear-btn" class="secondary">Clear</button>
  <div class="sep"></div>
  <div class="filters" id="filter-chips"></div>
  <div class="sep"></div>
  <button id="bookmarks-btn" class="secondary">Bookmarks</button>
  <button id="export-btn" class="secondary">Export</button>
  <div class="info" id="status-info"></div>
  <div class="info">
    <span class="kbd">j</span>/<span class="kbd">k</span> nav
    <span class="kbd">Enter</span> expand
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

<script>
// ── Data (injected by Python) ────────────────────────────────────────
const DATA = __DATA_PLACEHOLDER__;
const ALL_LABELS = __LABELS_PLACEHOLDER__;
const STAGE_KEYS = Object.keys(DATA);

function getStageLabels() {
  const sd = DATA[currentStage];
  return sd && sd.labels ? sd.labels : ALL_LABELS;
}

// ── Color palette for model columns ──────────────────────────────────
const PALETTE = [
  {bg: '#fff3e0', border: '#ffe0b2', header: '#fff3e0'},
  {bg: '#e8f5e9', border: '#c8e6c9', header: '#e8f5e9'},
  {bg: '#e3f2fd', border: '#bbdefb', header: '#e3f2fd'},
  {bg: '#f3e5f5', border: '#ce93d8', header: '#f3e5f5'},
  {bg: '#fce4ec', border: '#f48fb1', header: '#fce4ec'},
  {bg: '#e0f7fa', border: '#80deea', header: '#e0f7fa'},
];

// ── State ────────────────────────────────────────────────────────────
let currentStage = STAGE_KEYS[0] || '';
let currentIdx = 0;  // index into filteredRows
let filteredRows = [];
let allRows = [];
let searchQuery = '';
let activeFilter = 'all';
let bookmarks = new Set();  // row indices (original, not filtered)
let bookmarksPanelOpen = false;
let fieldFilters = {};  // {fieldPath: selectedValue} — empty string means "all"

// ── Init ─────────────────────────────────────────────────────────────
function init() {
  const sel = document.getElementById('stage-select');
  STAGE_KEYS.forEach(k => {
    const opt = document.createElement('option');
    opt.value = k;
    opt.textContent = `${k} (${DATA[k].rows.length} rows)`;
    sel.appendChild(opt);
  });
  sel.addEventListener('change', () => { loadStage(sel.value); });

  document.getElementById('search-btn').addEventListener('click', doSearch);
  document.getElementById('clear-btn').addEventListener('click', clearSearch);
  document.getElementById('search-input').addEventListener('keydown', e => {
    if (e.key === 'Enter') doSearch();
  });
  document.getElementById('bookmarks-btn').addEventListener('click', toggleBookmarks);
  document.getElementById('export-btn').addEventListener('click', exportBookmarks);

  document.addEventListener('keydown', handleKeyboard);

  loadStage(currentStage);
}

function loadStage(key) {
  currentStage = key;
  const stageData = DATA[key];
  allRows = stageData.rows;
  bookmarks.clear();
  activeFilter = 'all';
  searchQuery = '';
  fieldFilters = {};
  document.getElementById('search-input').value = '';
  buildFilterChips(stageData);
  buildFieldFilters();
  applyFilters();
}

// ── Filters ──────────────────────────────────────────────────────────
function buildFilterChips(stageData) {
  const container = document.getElementById('filter-chips');
  container.innerHTML = '';

  const hasCorrectness = allRows.some(r => r.correctness);
  const filters = [['all', 'All']];

  if (hasCorrectness) {
    // Build dynamic filters based on which labels have correctness data
    const labels = getStageLabels();
    if (labels.length === 2) {
      filters.push(['disagree', 'Disagree']);
      filters.push(['a_right_b_wrong', `${labels[0]} right, ${labels[1]} wrong`]);
      filters.push(['a_wrong_b_right', `${labels[0]} wrong, ${labels[1]} right`]);
      filters.push(['both_correct', 'Both correct']);
      filters.push(['both_wrong', 'Both wrong']);
    } else {
      filters.push(['any_wrong', 'Any wrong']);
      filters.push(['all_correct', 'All correct']);
      filters.push(['disagree', 'Disagree']);
    }
  }
  filters.push(['bookmarked', 'Bookmarked']);

  filters.forEach(([id, label]) => {
    const chip = document.createElement('span');
    chip.className = 'chip' + (id === activeFilter ? ' active' : '');
    chip.textContent = label;
    chip.dataset.filter = id;
    chip.addEventListener('click', () => {
      activeFilter = id;
      applyFilters();
    });
    container.appendChild(chip);
  });
}

function matchesChipFilter(row) {
  if (activeFilter === 'all') return true;
  if (activeFilter === 'bookmarked') return bookmarks.has(row.idx);

  const c = row.correctness;
  if (!c) return false;

  const labels = getStageLabels();
  const vals = labels.map(l => c[l]);

  switch (activeFilter) {
    case 'disagree': {
      const defined = vals.filter(v => v !== undefined);
      return defined.length > 1 && new Set(defined).size > 1;
    }
    case 'a_right_b_wrong': return vals[0] === true && vals[1] === false;
    case 'a_wrong_b_right': return vals[0] === false && vals[1] === true;
    case 'both_correct': return vals.every(v => v === true);
    case 'both_wrong': return vals.every(v => v === false);
    case 'any_wrong': return vals.some(v => v === false);
    case 'all_correct': return vals.every(v => v === true);
    default: return true;
  }
}

function matchesSearch(row) {
  if (!searchQuery) return true;
  // Use pre-computed _searchText blob for fast matching across all fields
  const haystack = row._searchText || '';
  try {
    return new RegExp(searchQuery, 'i').test(haystack);
  } catch (e) {
    return true;  // invalid regex, show all
  }
}

// ── Field filters (facet-based, pre-computed in Python) ──────────────
// row._facets is a flat {label: stringValue} dict, no path traversal needed.

function _discoverFacets() {
  // Scan all rows' _facets to find filterable fields and their unique values.
  const MAX_UNIQUE = 25;
  const facetValues = new Map();  // facetKey -> Set<string>

  for (const row of allRows) {
    const facets = row._facets;
    if (!facets) continue;
    for (const [key, val] of Object.entries(facets)) {
      if (!facetValues.has(key)) facetValues.set(key, new Set());
      facetValues.get(key).add(val);
    }
  }

  const result = [];
  for (const [key, values] of facetValues) {
    if (values.size >= 2 && values.size <= MAX_UNIQUE) {
      const sorted = [...values].sort((a, b) => {
        const na = Number(a), nb = Number(b);
        if (!isNaN(na) && !isNaN(nb)) return na - nb;
        return a.localeCompare(b);
      });
      result.push({key, values: sorted});
    }
  }
  return result.sort((a, b) => a.key.localeCompare(b.key));
}

function buildFieldFilters() {
  const container = document.getElementById('field-filters');
  container.innerHTML = '';
  fieldFilters = {};

  const facets = _discoverFacets();
  if (facets.length === 0) return;

  facets.forEach(({key, values}) => {
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

    // Use a closure variable — no dataset/path tricks needed
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

  if (facets.length > 0) {
    const reset = document.createElement('span');
    reset.className = 'ff-reset';
    reset.textContent = 'Reset all';
    reset.addEventListener('click', () => {
      fieldFilters = {};
      container.querySelectorAll('select').forEach(s => {
        s.value = '';
        s.classList.remove('active-filter');
      });
      refilter();
    });
    container.appendChild(reset);
  }
}

function matchesFieldFilters(row) {
  const facets = row._facets;
  for (const [key, expected] of Object.entries(fieldFilters)) {
    if (!facets || facets[key] !== expected) return false;
  }
  return true;
}

function refilter() {
  filteredRows = allRows.filter(r =>
    matchesChipFilter(r) && matchesSearch(r) && matchesFieldFilters(r)
  );
  currentIdx = 0;
  render();
  updateStatus();
}

function applyFilters() { refilter(); }

function updateStatus() {
  document.querySelectorAll('.chip').forEach(chip => {
    chip.classList.toggle('active', chip.dataset.filter === activeFilter);
  });
  const nActive = Object.keys(fieldFilters).length;
  const parts = [`${filteredRows.length} / ${allRows.length} rows`];
  if (searchQuery) parts.push(`search: "${searchQuery}"`);
  if (nActive) parts.push(`${nActive} filter${nActive > 1 ? 's' : ''}`);
  document.getElementById('status-info').textContent = parts.join(' | ');
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
function render() {
  const container = document.getElementById('container');
  container.innerHTML = '';

  if (filteredRows.length === 0) {
    container.innerHTML = '<p style="padding:40px;text-align:center;color:#999;">No rows match the current filters.</p>';
    return;
  }

  // Render visible rows (virtualized: show ±25 around current)
  const start = Math.max(0, currentIdx - 25);
  const end = Math.min(filteredRows.length, currentIdx + 50);

  for (let fi = start; fi < end; fi++) {
    const row = filteredRows[fi];
    container.appendChild(buildRowCard(row, fi));
  }

  // Scroll current into view
  requestAnimationFrame(() => {
    const el = document.querySelector('.row-card.current');
    if (el) el.scrollIntoView({block: 'nearest', behavior: 'smooth'});
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
    if (currentIdx === filterIdx) return;  // already expanded
    currentIdx = filterIdx;
    render();
  });

  const num = document.createElement('span');
  num.className = 'row-num';
  num.textContent = `#${row.idx}`;
  header.appendChild(num);

  // ID columns
  const ids = document.createElement('span');
  ids.className = 'row-ids';
  const stageData = DATA[currentStage];
  const idCols = stageData.col_info.id_cols || [];
  ids.innerHTML = idCols
    .filter(c => row[c] !== undefined)
    .map(c => `<b>${esc(c)}</b>=${esc(String(row[c]))}`)
    .join('&ensp;');
  header.appendChild(ids);

  // Badges
  const badges = document.createElement('span');
  badges.className = 'row-badges';
  if (row.correctness) {
    getStageLabels().forEach(label => {
      if (row.correctness[label] !== undefined) {
        const b = document.createElement('span');
        b.className = 'badge ' + (row.correctness[label] ? 'correct' : 'wrong');
        b.textContent = `${label}: ${row.correctness[label] ? 'correct' : 'wrong'}`;
        badges.appendChild(b);
      }
    });
  }
  // Bookmark badge
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

  // Body (only rendered if expanded)
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
  const colInfo = stageData.col_info;
  let html = '';

  // Prompt
  if (row.prompt) {
    const promptFormatted = formatTextWithJson(row.prompt);
    const isLong = row.prompt.length > 500;
    html += `<div class="prompt-box">
      <details${isLong ? '' : ' open'}>
        <summary>Prompt (${row.prompt.length.toLocaleString()} chars)</summary>
        <pre>${highlightSearch(promptFormatted)}</pre>
      </details>
    </div>`;
  }

  // Context columns
  (colInfo.context_cols || []).forEach(c => {
    if (row[c]) {
      const raw = stringify(row[c]);
      const formatted = formatTextWithJson(raw);
      html += `<div class="prompt-box">
        <details><summary>${esc(c)} (${raw.length.toLocaleString()} chars)</summary>
        <pre>${formatted}</pre></details>
      </div>`;
    }
  });

  // Completions grid
  const completions = row.completions || {};
  const stageLabels = getStageLabels();
  const nCols = stageLabels.length;
  html += `<div class="completions-grid" style="grid-template-columns: repeat(${nCols}, 1fr);">`;
  stageLabels.forEach((label, i) => {
    const p = PALETTE[i % PALETTE.length];
    const text = completions[label];
    const formatted = text != null ? formatTextWithJson(String(text)) : '<em>(no completion)</em>';
    const correctnessClass = row.correctness && row.correctness[label] !== undefined
      ? (row.correctness[label] ? ' correct' : ' wrong') : '';
    html += `<div class="completion-col">
      <div class="col-header${correctnessClass}" style="background:${p.header};border-color:${p.border};">
        ${esc(label)}
      </div>
      <pre style="background:${p.bg};">${highlightSearch(formatted)}</pre>
    </div>`;
  });
  html += '</div>';

  // Ground truth
  const gtCols = colInfo.ground_truth_cols || [];
  if (gtCols.length > 0) {
    html += '<div class="meta-row">';
    gtCols.forEach(c => {
      if (row[c] !== undefined && row[c] !== null) {
        html += `<span><span class="label">${esc(c)}:</span> <span class="val">${esc(String(row[c]))}</span></span>`;
      }
    });
    html += '</div>';
  }

  // Predictions
  if (row.predictions) {
    Object.entries(row.predictions).forEach(([predCol, labelVals]) => {
      html += '<div class="meta-row">';
      html += `<span class="label">${esc(predCol)}:</span> `;
      getStageLabels().forEach((label, i) => {
        const val = labelVals[label];
        if (val !== undefined) {
          // Check correctness for this prediction
          let cls = '';
          if (row.correctness && row.correctness[label] !== undefined) {
            cls = row.correctness[label] ? ' correct' : ' wrong';
          }
          const p = PALETTE[i % PALETTE.length];
          html += `<span class="val${cls}" style="background:${p.bg};padding:2px 6px;border-radius:3px;margin-right:4px;">${esc(label)}: ${esc(String(val))}</span> `;
        }
      });
      html += '</div>';
    });
  }

  // Judge panels
  if (row.judges) {
    const stageData = DATA[currentStage];
    const judgeNames = stageData.judge_names || Object.keys(row.judges);
    judgeNames.forEach(judgeName => {
      const judgeData = row.judges[judgeName];
      if (!judgeData) return;
      const toggleId = `judge-${row.idx}-${judgeName.replace(/\s+/g, '_')}`;
      html += `<div class="judge-section">`;
      html += `<div class="judge-toggle" onclick="
        this.classList.toggle('open');
        document.getElementById('${toggleId}').classList.toggle('open');
      "><span class="arrow">&#9654;</span> ${esc(judgeName)}</div>`;
      const judgeLabels = getStageLabels().filter(l => judgeData[l]);
      const nJCols = judgeLabels.length;
      html += `<div class="judge-grid" id="${toggleId}" style="grid-template-columns: repeat(${nJCols}, 1fr);">`;
      judgeLabels.forEach((label, i) => {
        const jEntry = judgeData[label];
        if (!jEntry) return;
        html += `<div class="judge-card">`;
        // Header with verdict badges
        html += `<div class="judge-header"><span>${esc(label)}</span>`;
        // Render verdict columns as badges
        for (const [k, v] of Object.entries(jEntry)) {
          if (k.endsWith('_text')) continue;  // skip the full text, shown in pre
          if (k === 'leak_flag') {
            html += `<span class="verdict ${v ? 'leak' : 'no-leak'}">${v ? 'LEAK' : 'No leak'}</span>`;
          } else if (k === 'leak_probability') {
            html += `<span class="verdict" style="background:#eee;">P=${typeof v === 'number' ? v.toFixed(2) : v}</span>`;
          } else if (k === 'helpfulness_binary') {
            html += `<span class="verdict ${v ? 'helpful' : 'not-helpful'}">${v ? 'Helpful' : 'Not helpful'}</span>`;
          } else if (k === 'helpfulness_score') {
            html += `<span class="verdict" style="background:#eee;">Score=${v}</span>`;
          } else {
            html += `<span class="verdict" style="background:#eee;">${esc(k)}=${esc(String(v))}</span>`;
          }
        }
        html += '</div>';
        // Judge reasoning text
        const textKeys = Object.keys(jEntry).filter(k => k.endsWith('_text'));
        textKeys.forEach(tk => {
          if (jEntry[tk]) {
            html += `<pre>${highlightSearch(formatTextWithJson(String(jEntry[tk])))}</pre>`;
          }
        });
        html += '</div>';
      });
      html += '</div></div>';
    });
  }

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

function stringify(val) {
  if (val == null) return '';
  if (typeof val === 'string') return val;
  if (typeof val === 'object') return JSON.stringify(val, null, 2);
  return String(val);
}

function syntaxHighlightJson(jsonStr) {
  // Highlight a raw JSON string (already indented) with span classes.
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
  // Try to parse a JSON value starting at position i.
  // Returns {end, parsed} on success, null on failure.
  const ch = text[i];
  if (ch !== '{' && ch !== '[') return null;
  const close = ch === '{' ? '}' : ']';

  // Find matching close bracket, respecting nesting and strings.
  let depth = 0;
  let inStr = false;
  let esc = false;
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
      } catch (e) {
        return null;
      }
    }
  }
  return null;
}

function formatTextWithJson(rawText) {
  // Detect JSON objects/arrays in text, pretty-print them with syntax
  // highlighting, and leave surrounding plain text as escaped HTML.
  // When a brace doesn't start valid JSON, it's emitted as plain text
  // and scanning continues — so inner JSON fragments are still found.
  const result = [];
  let i = 0;
  const len = rawText.length;
  let plainStart = 0;  // start of current plain-text run

  while (i < len) {
    const ch = rawText[i];
    if (ch === '{' || ch === '[') {
      const match = _tryParseJsonAt(rawText, i);
      if (match) {
        // Flush accumulated plain text before this JSON block
        if (i > plainStart) {
          result.push(esc(rawText.slice(plainStart, i)));
        }
        const pretty = JSON.stringify(match.parsed, null, 2);
        const highlighted = syntaxHighlightJson(esc(pretty));
        result.push('<div class="json-block">' + highlighted + '</div>');
        i = match.end;
        plainStart = i;
        continue;
      }
    }
    // Not JSON (or not a brace) — advance one character
    i++;
  }
  // Flush remaining plain text
  if (plainStart < len) {
    result.push(esc(rawText.slice(plainStart)));
  }
  return result.join('');
}

// ── Bookmarks ────────────────────────────────────────────────────────
function toggleBookmark(idx) {
  if (bookmarks.has(idx)) bookmarks.delete(idx);
  else bookmarks.add(idx);

  // If the bookmarked filter is active, the row set changes — must re-render.
  if (activeFilter === 'bookmarked') {
    render();
  } else {
    // Patch the affected card in-place instead of re-rendering everything.
    const isBookmarked = bookmarks.has(idx);
    document.querySelectorAll('.row-card').forEach(card => {
      const fi = parseInt(card.dataset.filterIdx, 10);
      if (filteredRows[fi] && filteredRows[fi].idx === idx) {
        card.classList.toggle('bookmarked', isBookmarked);
        const badge = card.querySelector('.bookmark-badge');
        if (badge) badge.textContent = isBookmarked ? '★' : '☆';
      }
    });
  }
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
    const stageData = DATA[currentStage];
    const idCols = stageData.col_info.id_cols || [];
    const idStr = idCols.filter(c => row[c] !== undefined).map(c => `${c}=${row[c]}`).join(', ');
    item.textContent = `#${idx}${idStr ? ' — ' + idStr : ''}`;
    item.addEventListener('click', () => {
      // Find this row in filtered list
      const fi = filteredRows.findIndex(r => r.idx === idx);
      if (fi >= 0) { currentIdx = fi; render(); }
      else {
        // Switch to 'all' filter to find it
        activeFilter = 'all';
        searchQuery = '';
        document.getElementById('search-input').value = '';
        applyFilters();
        const fi2 = filteredRows.findIndex(r => r.idx === idx);
        if (fi2 >= 0) { currentIdx = fi2; render(); }
      }
    });
    list.appendChild(item);
  });
}

function exportBookmarks() {
  if (bookmarks.size === 0) {
    alert('No bookmarks to export.');
    return;
  }
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
  a.download = `bookmarks_${currentStage.replace(/[^a-zA-Z0-9]/g, '_')}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

// ── Keyboard ─────────────────────────────────────────────────────────
function handleKeyboard(e) {
  // Don't capture when typing in input
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
    if (e.key === 'Escape') e.target.blur();
    return;
  }

  switch (e.key) {
    case 'j': case 'ArrowDown':
      e.preventDefault();
      if (currentIdx < filteredRows.length - 1) { currentIdx++; render(); }
      break;
    case 'k': case 'ArrowUp':
      e.preventDefault();
      if (currentIdx > 0) { currentIdx--; render(); }
      break;
    case 'Enter':
      // Already expanded via currentIdx, this is a no-op but feels natural
      break;
    case 'b':
      if (filteredRows[currentIdx]) toggleBookmark(filteredRows[currentIdx].idx);
      break;
    case '/':
      e.preventDefault();
      document.getElementById('search-input').focus();
      break;
    case 'Escape':
      if (bookmarksPanelOpen) toggleBookmarks();
      break;
  }
}

// ── Start ────────────────────────────────────────────────────────────
init();
</script>
</body>
</html>"""


# ── CLI ───────────────────────────────────────────────────────────────────

def parse_run_arg(arg: str) -> tuple[str, str]:
    """Parse a run argument like 'Label=/path/to/run' or just '/path/to/run'."""
    if "=" in arg:
        label, path = arg.split("=", 1)
        return label.strip(), path.strip()
    # Auto-label from directory name
    p = Path(arg)
    return p.name, arg


def parse_row_slice(spec: str, n: int) -> list[int]:
    """Parse a Python-style array index/slice spec into a list of row indices.

    Supports:
        "42"          → [42]
        "10:20"       → [10, 11, ..., 19]
        ":50"         → [0, 1, ..., 49]
        "100:"        → [100, 101, ..., n-1]
        "::2"         → [0, 2, 4, ...]
        "-10:"        → last 10 rows
        "0:100:5"     → [0, 5, 10, ..., 95]
        "0,5,10,42"   → [0, 5, 10, 42]
        "0:10,50:60"  → [0..9, 50..59]

    Args:
        spec: The row specification string.
        n: Total number of available rows.

    Returns:
        Sorted, deduplicated list of valid row indices.

    Raises:
        ValueError: If the spec is malformed or produces no valid indices.
    """
    indices: set[int] = set()

    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue

        if ":" in part:
            # Slice syntax
            pieces = part.split(":")
            if len(pieces) > 3:
                raise ValueError(
                    f"Invalid slice '{part}': too many colons (max format is start:stop:step)"
                )
            try:
                args = [int(p) if p.strip() else None for p in pieces]
            except ValueError:
                raise ValueError(
                    f"Invalid slice '{part}': non-integer component"
                )
            sl = slice(*args)
            resolved = range(*sl.indices(n))
            if len(resolved) == 0:
                raise ValueError(
                    f"Slice '{part}' produces no rows (dataset has {n} rows)"
                )
            indices.update(resolved)
        else:
            # Single index
            try:
                idx = int(part)
            except ValueError:
                raise ValueError(f"Invalid index '{part}': not an integer")
            # Resolve negative indices
            resolved_idx = idx if idx >= 0 else n + idx
            if resolved_idx < 0 or resolved_idx >= n:
                raise ValueError(
                    f"Index {idx} is out of range (dataset has {n} rows, "
                    f"valid range: {-n}..{n - 1})"
                )
            indices.add(resolved_idx)

    if not indices:
        raise ValueError(f"Row spec '{spec}' produced no indices")

    return sorted(indices)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a static HTML completion inspector from eval_all runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--runs", nargs="+", required=True,
        help='Run specifications: "Label=/path/to/run" or just "/path/to/run"',
    )
    parser.add_argument(
        "-o", "--output", default="completion_inspector.html",
        help="Output HTML file path (default: completion_inspector.html)",
    )
    parser.add_argument(
        "--max-rows", type=int, default=None,
        help="Maximum rows per stage (default: all)",
    )
    parser.add_argument(
        "--rows", type=str, default=None, metavar="SPEC",
        help=(
            "Row subset using Python slice syntax. Applied per stage. "
            "Examples: '0:100' (first 100), '50:150' (rows 50-149), "
            "'-50:' (last 50), '::10' (every 10th), '0:10,90:100' "
            "(first 10 + rows 90-99), '42' (single row). "
            "Mutually exclusive with --max-rows."
        ),
    )
    args = parser.parse_args()

    if args.rows and args.max_rows:
        parser.error("--rows and --max-rows are mutually exclusive")

    # Parse run specifications
    runs = {}
    for r in args.runs:
        label, path = parse_run_arg(r)
        runs[label] = resolve_root(path)
        if not runs[label].is_dir():
            print(f"ERROR: Run path does not exist: {runs[label]}", file=sys.stderr)
            sys.exit(1)

    labels = list(runs.keys())
    print(f"Models: {labels}")

    # Discover stages in each run
    per_run_stages = {}
    for label, root in runs.items():
        stages = discover_stages(root)
        per_run_stages[label] = stages
        print(f"  {label}: {len(stages)} stages found at {root}")

    # Find stages present in at least 2 runs (union, not intersection)
    from collections import Counter
    stage_counts = Counter()
    for stages in per_run_stages.values():
        stage_counts.update(stages.keys())
    all_stage_keys = {k for k, c in stage_counts.items() if c >= min(2, len(runs))}
    if not all_stage_keys:
        print("ERROR: No stages found across runs.", file=sys.stderr)
        for label, stages in per_run_stages.items():
            print(f"  {label}: {sorted(stages.keys())}", file=sys.stderr)
        sys.exit(1)

    print(f"\nDiscovered stages ({len(all_stage_keys)}):")
    for k in sorted(all_stage_keys):
        present = [l for l in labels if k in per_run_stages[l]]
        print(f"  {k}  [{', '.join(present)}]")

    # Discover judge stages for each run
    per_run_judges = {}
    for label, root in runs.items():
        per_run_judges[label] = discover_judge_stages(root, per_run_stages[label])

    # Build data for each stage
    data = {}
    for stage_key in sorted(all_stage_keys):
        # Only include models that have this stage
        stage_labels = [l for l in labels if stage_key in per_run_stages[l]]
        print(f"\nProcessing: {stage_key} ({len(stage_labels)} models) ...", end=" ", flush=True)
        label_to_df = {}
        for label in stage_labels:
            pq_path = per_run_stages[label][stage_key]
            df = pd.read_parquet(pq_path)
            label_to_df[label] = df

        # Load related judge stages if available
        # judge_dfs: {judge_display_name: {model_label: DataFrame}}
        judge_dfs: dict[str, dict[str, pd.DataFrame]] = {}
        judge_names_found = []
        for label in stage_labels:
            judges_for_stage = per_run_judges.get(label, {}).get(stage_key, [])
            for jinfo in judges_for_stage:
                jname = jinfo["display_name"]
                if jname not in judge_dfs:
                    judge_dfs[jname] = {}
                    judge_names_found.append(jname)
                judge_dfs[jname][label] = pd.read_parquet(jinfo["path"])

        # Apply --rows slice if specified
        n_total = min(len(df) for df in label_to_df.values())
        if args.rows:
            try:
                row_indices = parse_row_slice(args.rows, n_total)
            except ValueError as e:
                print(f"\nERROR in stage '{stage_key}': {e}", file=sys.stderr)
                sys.exit(1)
            label_to_df = {l: df.iloc[row_indices].reset_index(drop=True)
                           for l, df in label_to_df.items()}
            if judge_dfs:
                judge_dfs = {
                    jname: {l: jdf.iloc[row_indices].reset_index(drop=True)
                            for l, jdf in label_jdfs.items()}
                    for jname, label_jdfs in judge_dfs.items()
                }

        # Use first available model's df to classify columns
        col_info = _classify_columns(label_to_df[stage_labels[0]])
        rows = build_stage_data(
            label_to_df, col_info, max_rows=args.max_rows,
            judge_dfs=judge_dfs if judge_dfs else None,
        )

        stage_data: dict[str, Any] = {
            "rows": rows,
            "col_info": col_info,
            "labels": stage_labels,
            "n_total": n_total,
        }
        if judge_names_found:
            stage_data["judge_names"] = list(dict.fromkeys(judge_names_found))
        data[stage_key] = stage_data

        judge_str = f" + {', '.join(judge_names_found)}" if judge_names_found else ""
        print(f"{len(rows)} rows{judge_str}")

    # Generate HTML
    print(f"\nGenerating HTML ...", end=" ", flush=True)
    data_json = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    labels_json = json.dumps(labels)

    html = HTML_TEMPLATE.replace("__DATA_PLACEHOLDER__", data_json)
    html = html.replace("__LABELS_PLACEHOLDER__", labels_json)

    out_path = Path(args.output)
    out_path.write_text(html, encoding="utf-8")
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"done ({size_mb:.1f} MB)")
    print(f"\nOutput: {out_path.resolve()}")
    print(f"Open in browser: file://{out_path.resolve()}")


if __name__ == "__main__":
    main()
