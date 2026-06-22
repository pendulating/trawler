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
import datetime as _dt
import difflib
import hashlib
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
# `annotation` describes how an expert reviews this judge:
#   verdict_field: judge column to compare expert verdict against (for agree/disagree)
#   options: [{value, label}] shown as radio choices
#   true_value / false_value: which option value maps to a True / False judge verdict
_JUDGE_COLUMNS = {
    "leakage_judge_inference": {
        "text_col": "leak_judge_text",
        "verdict_cols": ["leak_probability", "leak_flag"],
        "display_name": "Leakage Judge",
        "parent_stage": "agent_action_inference",
        "annotation": {
            "verdict_field": "leak_flag",
            "true_value": "leak",
            "false_value": "no-leak",
            "options": [
                {"value": "leak", "label": "Leak"},
                {"value": "no-leak", "label": "No leak"},
                {"value": "skip", "label": "Skip"},
            ],
        },
    },
    "helpfulness_judge_inference": {
        "text_col": "helpfulness_judge_text",
        "verdict_cols": ["helpfulness_score", "helpfulness_binary"],
        "display_name": "Helpfulness Judge",
        "parent_stage": "agent_action_inference",
        "annotation": {
            "verdict_field": "helpfulness_binary",
            "true_value": "helpful",
            "false_value": "not-helpful",
            "options": [
                {"value": "helpful", "label": "Helpful"},
                {"value": "not-helpful", "label": "Not helpful"},
                {"value": "skip", "label": "Skip"},
            ],
        },
    },
}


def _judge_annotation_meta() -> dict[str, dict]:
    """Return {judge_display_name: annotation_meta} for JS consumption."""
    return {
        info["display_name"]: info["annotation"]
        for info in _JUDGE_COLUMNS.values()
        if "annotation" in info
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
                    **{k: v for k, v in judge_info.items()
                       if k not in ("parent_stage", "annotation")},
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
        elif cl in ("vignette", "generate_background", "story", "text", "trajectory"):
            info["context_cols"].append(c)

    # Detect dict-typed context columns with extractable sub-fields
    context_subfields: dict[str, list[tuple[str, str]]] = {}
    for c in info["context_cols"]:
        if c in _CONTEXT_SUBFIELDS and c in cols:
            # Verify column actually contains dicts by checking first non-null
            sample = df[c].dropna().iloc[0] if df[c].notna().any() else None
            if isinstance(sample, dict):
                context_subfields[c] = _CONTEXT_SUBFIELDS[c]
    info["context_subfields"] = context_subfields

    return info


# Sub-fields to extract from dict-typed context columns as separate toggleable entries.
# Maps column_name -> [(sub_key, display_label)]
_CONTEXT_SUBFIELDS = {
    "trajectory": [
        ("sensitive_info_items", "Sensitive Info Items"),
        ("user_instruction", "User Instruction"),
    ],
}


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


def _reformat_prompt_repr(record: dict, col_name: str, col_val: dict) -> None:
    """Replace Python repr of a dict column inside the prompt text with valid JSON.

    The prompt often contains something like:
        Trajectory:\n{'key': 'val', ...}\n\nGenerate the final action...
    We find the repr block and replace it with pretty-printed JSON so that
    the JS formatTextWithJson renderer can parse and syntax-highlight it.
    """
    prompt = record.get("prompt", "")
    if not prompt:
        return

    # Look for the column name (capitalized) followed by the repr.
    # Common patterns: "Trajectory:\n{...}" or "Vignette:\n{...}"
    for marker in (f"{col_name.capitalize()}:\n", f"{col_name}:\n"):
        idx = prompt.find(marker)
        if idx < 0:
            continue
        repr_start = idx + len(marker)

        # The repr is a Python dict starting with '{'.  Find its end by
        # scanning for the matching '}' at the top level, accounting for
        # nesting and string literals (single- and double-quoted).
        depth = 0
        in_str: str | None = None
        esc = False
        end = None
        for j in range(repr_start, len(prompt)):
            ch = prompt[j]
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if in_str:
                if ch == in_str:
                    in_str = None
                continue
            if ch in ("'", '"'):
                in_str = ch
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = j + 1
                    break

        if end is None:
            return  # couldn't find matching brace

        # Build clean JSON from the structured data we already have
        clean = json.dumps(_serialize(col_val), indent=2, ensure_ascii=False)

        record["prompt"] = prompt[:repr_start] + clean + prompt[end:]
        return  # done


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
                val = ref[c]
                record[c] = _serialize(val)
                # Extract sub-fields from dict-typed context columns
                if isinstance(val, dict) and c in _CONTEXT_SUBFIELDS:
                    for sub_key, _label in _CONTEXT_SUBFIELDS[c]:
                        sub_val = val.get(sub_key)
                        if sub_val is not None:
                            record[f"{c}.{sub_key}"] = _serialize(sub_val)

        # Fix prompt: replace Python repr of dict-typed context cols with JSON
        # so the JS formatTextWithJson can parse them properly.
        prompt = record.get("prompt", "")
        if prompt:
            for c in col_info["context_cols"]:
                if c in ref.index and isinstance(ref[c], dict):
                    _reformat_prompt_repr(record, c, ref[c])

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
  display: flex; flex-direction: column;
}
.completion-col .col-header {
  padding: 6px 10px; font-weight: 600; font-size: 12px;
  border-bottom: 1px solid var(--border);
}
.completion-col pre {
  padding: 10px; white-space: pre-wrap; word-break: break-word;
  font-family: var(--mono); font-size: 12px; line-height: 1.5;
  max-height: 500px; overflow-y: auto; margin: 0; flex: 1;
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
  display: flex; flex-direction: column;
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
  flex: 1;
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

/* ── Export modal */
.export-overlay { position:fixed;inset:0;z-index:500;background:rgba(0,0,0,0.5);display:flex;align-items:center;justify-content:center; }
.export-dialog { background:#fff;border-radius:10px;width:92vw;max-width:1000px;max-height:92vh;display:flex;flex-direction:column;box-shadow:0 8px 32px rgba(0,0,0,0.2); }
.export-header { padding:12px 16px;border-bottom:1px solid var(--border);display:flex;justify-content:space-between;align-items:center; }
.export-header h3 { margin:0;font-size:15px; }
.export-controls { padding:10px 16px;border-bottom:1px solid var(--border);display:flex;flex-wrap:wrap;gap:10px 18px;align-items:center;font-size:13px;background:#fafafa; }
.export-controls label { display:inline-flex;align-items:center;gap:4px;cursor:pointer;white-space:nowrap; }
.export-controls .eg { display:inline-flex;align-items:center;gap:2px;border:1px solid var(--border);border-radius:4px;padding:1px 2px;background:#fff; }
.export-controls .eg label { padding:2px 8px;border-radius:3px;font-size:12px; }
.export-controls .eg label.active { background:var(--accent);color:#fff; }
.export-preview { flex:1;overflow:auto;padding:16px;background:#f9f9f9; }
.export-preview [data-export].export-hidden { display:none !important; }
.export-preview .export-frame { background:#fff;border:1px solid var(--border);border-radius:8px;padding:14px;margin:0 auto;font-size:13px; }
.export-preview .annot-strip, .export-frame .annot-strip { display: none !important; }
.export-preview .export-frame .prompt-box details,
.export-preview .export-frame .field-group { }
.export-preview .export-frame details[open] > summary { margin-bottom:4px; }
.export-preview .export-frame .completion-col pre { max-height:none; }
.export-actions { padding:10px 16px;border-top:1px solid var(--border);display:flex;gap:8px;justify-content:flex-end; }
.export-actions button { font-size:13px;padding:6px 16px;border:none;border-radius:4px;cursor:pointer;font-weight:500; }
.export-actions .btn-pdf { background:#1976d2;color:#fff; }
.export-actions .btn-html { background:#2e7d32;color:#fff; }
.export-actions .btn-cancel { background:#757575;color:#fff; }
.export-actions button:hover { opacity:0.9; }

/* ── Annotation UI ──────────────────────────────────────────────────── */
.annot-bar { display: none; gap: 8px; align-items: center; }
body.annotate-on .annot-bar { display: inline-flex; }
.annot-bar .annot-progress {
  font-size: 12px; padding: 3px 10px; background: #f5f5f5; border-radius: 4px;
  border: 1px solid var(--border); color: #333;
}
.annot-bar .annot-progress b { color: var(--accent); }
.annot-bar button.annot-btn {
  background: #6a1b9a; color: #fff; padding: 5px 10px;
  font-size: 12px; border: none; border-radius: 4px; cursor: pointer;
}
.annot-bar button.annot-btn:hover { opacity: 0.9; }
.annot-bar button.annot-btn.danger { background: #b71c1c; }

.badge.annot-status { font-family: var(--mono); font-size: 10px; }
.badge.annot-status.unannot { background: #eee; color: #555; }
.badge.annot-status.partial { background: var(--orange-bg); color: var(--orange); }
.badge.annot-status.complete { background: var(--green-bg); color: var(--green); }
.badge.annot-status.disagree { background: var(--red-bg); color: var(--red); }

/* The annotation strip sits inside each judge-card, below the pre */
.annot-strip {
  border-top: 1px solid #e0d6f5; padding: 6px 10px; background: #fff;
  font-size: 12px; display: flex; flex-direction: column; gap: 5px;
}
.annot-strip .annot-row { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; }
.annot-strip .annot-label {
  font-size: 11px; font-weight: 600; color: #555; text-transform: uppercase;
  letter-spacing: 0.04em;
}
.annot-strip .annot-radios {
  display: inline-flex; gap: 2px; border: 1px solid var(--border); border-radius: 4px;
  background: #fff; overflow: hidden;
}
.annot-strip .annot-radios label {
  font-size: 11px; padding: 3px 8px; cursor: pointer; user-select: none;
  border-right: 1px solid var(--border); transition: background 0.1s;
}
.annot-strip .annot-radios label:last-child { border-right: none; }
.annot-strip .annot-radios label:hover { background: #f0f0f0; }
.annot-strip .annot-radios label.active {
  background: #6a1b9a; color: #fff; font-weight: 600;
}
.annot-strip .annot-agree {
  font-size: 10px; padding: 1px 6px; border-radius: 8px; font-weight: 600;
}
.annot-strip .annot-agree.agree { background: var(--green-bg); color: var(--green); }
.annot-strip .annot-agree.disagree { background: var(--red-bg); color: var(--red); }
.annot-strip .annot-agree.empty { background: #f0f0f0; color: #888; }
.annot-strip input.annot-notes {
  flex: 1; min-width: 200px; font-size: 12px; padding: 3px 6px;
  border: 1px solid var(--border); border-radius: 3px; font-family: var(--sans);
}
.annot-strip.focused {
  background: #fff8e1;
  box-shadow: inset 3px 0 0 #ff9800;
}
.annot-strip.focused .annot-label::before { content: "▶ "; color: #ff9800; }

/* ── Ranking strip ───────────────────────────────────────────────────── */
.rank-strip {
  display: flex; gap: 8px; align-items: center; flex-wrap: wrap;
  padding: 6px 10px; margin: 4px 0 8px;
  background: #fff8e1; border: 1px solid #ffe0b2; border-radius: 6px;
  font-size: 12px;
}
.rank-strip .rank-label {
  font-weight: 600; color: #6d4c00; text-transform: uppercase; font-size: 11px;
  letter-spacing: 0.04em; margin-right: 2px;
}
.rank-strip .rank-dir {
  font-size: 11px; color: #9c6500; font-style: italic;
}
.rank-strip .rank-chips { display: inline-flex; gap: 4px; flex-wrap: wrap; }
.rank-chip {
  display: inline-flex; align-items: center; gap: 4px;
  border: 1px solid #ffcc80; border-radius: 12px;
  padding: 2px 8px; cursor: pointer; background: #fff; user-select: none;
  transition: all 0.1s;
}
.rank-chip:hover { background: #fff3e0; }
.rank-chip .rank-model { font-family: var(--mono); font-size: 11px; color: #333; }
.rank-chip .rank-value {
  display: inline-flex; align-items: center; justify-content: center;
  min-width: 18px; height: 18px; padding: 0 5px;
  background: #ef6c00; color: #fff; border-radius: 9px;
  font-weight: 700; font-size: 11px;
}
.rank-chip .rank-value.empty { background: #e0e0e0; color: #888; font-weight: 500; }
.rank-strip .rank-clear {
  margin-left: auto; font-size: 11px; color: #b71c1c; cursor: pointer;
  text-decoration: underline;
}
.rank-strip .rank-clear:hover { color: #7f0000; }

/* Hide ranking strip in export modal */
.export-preview .rank-strip, .export-frame .rank-strip { display: none !important; }

/* Manifest banner */
.manifest-banner {
  display: none; padding: 6px 16px; background: #e8eaf6; border-bottom: 1px solid #c5cae9;
  font-size: 12px; color: #1a237e;
}
body.annotate-on .manifest-banner { display: block; }
.manifest-banner b { font-family: var(--mono); }
.manifest-banner .mb-warn { color: var(--red); font-weight: 600; }
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
  <div class="filters" id="filter-chips"></div>
  <div class="sep"></div>
  <button id="bookmarks-btn" class="secondary">Bookmarks</button>
  <button id="exportrow-btn" class="secondary">Export Row</button>
  <button id="export-btn" class="secondary">Export BMs</button>
  <div class="annot-bar">
    <div class="sep"></div>
    <span class="annot-progress" id="annot-progress" title="Cells annotated">…</span>
    <button class="annot-btn" id="annot-download-json" title="Download annotations as JSON">Save JSON</button>
    <button class="annot-btn" id="annot-download-csv" title="Download annotations as CSV (one row per cell)">Save CSV</button>
    <button class="annot-btn" id="annot-upload" title="Restore annotations from a previously saved JSON file">Upload</button>
    <input type="file" id="annot-upload-input" accept="application/json,.json" style="display:none">
    <button class="annot-btn danger" id="annot-clear" title="Clear all annotations for this sample">Clear</button>
  </div>
  <div class="info" id="status-info"></div>
  <div class="info">
    <span class="kbd">j</span>/<span class="kbd">k</span> nav
    <span class="kbd">g</span> jump
    <span class="kbd">b</span> bookmark
    <span class="kbd">/</span> search
  </div>
  <div class="info annot-bar">
    <span class="kbd">Tab</span> cell
    <span class="kbd">1</span>/<span class="kbd">2</span>/<span class="kbd">3</span> verdict
    <span class="kbd">n</span> notes
  </div>
</div>

<div class="manifest-banner" id="manifest-banner"></div>
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
const ALL_LABELS = __LABELS_PLACEHOLDER__;
const STAGE_KEYS = Object.keys(DATA);
const MANIFEST = __MANIFEST_PLACEHOLDER__;
const ANNOTATE_MODE = __ANNOTATE_MODE_PLACEHOLDER__;
const JUDGE_META = __JUDGE_META_PLACEHOLDER__;

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
  if (ANNOTATE_MODE) document.body.classList.add('annotate-on');
  _renderManifestBanner();

  const sel = document.getElementById('stage-select');
  STAGE_KEYS.forEach(k => {
    const opt = document.createElement('option');
    opt.value = k;
    opt.textContent = `${k} (${DATA[k].rows.length} rows)`;
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
  document.getElementById('export-btn').addEventListener('click', exportBookmarks);
  document.getElementById('exportrow-btn').addEventListener('click', showExportModal);
  document.getElementById('export-overlay').addEventListener('click', (e) => {
    if (e.target === e.currentTarget) closeExportModal();
  });

  // Annotation controls (no-ops when ANNOTATE_MODE is false)
  if (ANNOTATE_MODE) {
    document.getElementById('annot-download-json').addEventListener('click', exportAnnotationsJSON);
    document.getElementById('annot-download-csv').addEventListener('click', exportAnnotationsCSV);
    document.getElementById('annot-clear').addEventListener('click', clearAllAnnotations);
    const uploadInput = document.getElementById('annot-upload-input');
    document.getElementById('annot-upload').addEventListener('click', () => uploadInput.click());
    uploadInput.addEventListener('change', e => {
      const file = e.target.files && e.target.files[0];
      if (file) importAnnotationsFile(file);
      e.target.value = '';
    });
    // Delegated annotation interactions
    document.getElementById('container').addEventListener('click', _onRankClick);
    document.getElementById('container').addEventListener('click', _onAnnotClick);
    document.getElementById('container').addEventListener('input', _onAnnotNotesInput);
    document.getElementById('container').addEventListener('focusin', _onAnnotFocus);
  }

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
  focusedCell = null;
  document.getElementById('search-input').value = '';
  loadAnnotations();
  buildFilterChips(stageData);
  buildFieldFilters();
  applyFilters();
  _updateAnnotProgress();
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
  if (ANNOTATE_MODE) {
    filters.push(['annot:unannot', 'Unannotated']);
    filters.push(['annot:partial', 'Partial']);
    filters.push(['annot:complete', 'Complete']);
    filters.push(['annot:disagree', 'Disagrees w/ judge']);
  }

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
  if (activeFilter.startsWith('annot:')) {
    const want = activeFilter.slice('annot:'.length);
    return getRowStatus(row) === want;
  }

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
  focusedCell = null;
  render('header');
  updateStatus();
  _updateAnnotProgress();
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

// ── Jump to row ─────────────────────────────────────────────────────
function doJump() {
  const input = document.getElementById('jump-input');
  const target = parseInt(input.value, 10);
  if (isNaN(target)) return;
  const fi = filteredRows.findIndex(r => r.idx === target);
  if (fi >= 0) {
    currentIdx = fi;
    render('center');
  } else {
    const exists = allRows.some(r => r.idx === target);
    if (exists) {
      activeFilter = 'all';
      fieldFilters = {};
      searchQuery = '';
      document.getElementById('search-input').value = '';
      document.getElementById('field-filters').querySelectorAll('select').forEach(s => {
        s.value = '';
        s.classList.remove('active-filter');
      });
      filteredRows = allRows.slice();
      updateStatus();
      const fi2 = filteredRows.findIndex(r => r.idx === target);
      if (fi2 >= 0) { currentIdx = fi2; render('center'); }
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
// scrollMode: 'header' = pin card header to top of viewport (default)
//             'center' = scroll card to center (for jump/search)
//             'none'   = don't scroll
function render(scrollMode) {
  if (!scrollMode) scrollMode = 'header';
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
  const card = document.createElement('div');
  card.className = 'row-card' +
    (filterIdx === currentIdx ? ' expanded current' : '') +
    (bookmarks.has(row.idx) ? ' bookmarked' : '');
  card.dataset.filterIdx = filterIdx;
  card.dataset.rowIdx = row.idx;

  // Header
  const header = document.createElement('div');
  header.className = 'row-header';
  header.addEventListener('click', () => {
    if (currentIdx === filterIdx) return;  // already expanded
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
  // Annotation status badge
  if (ANNOTATE_MODE) {
    const status = getRowStatus(row);
    const annBadge = document.createElement('span');
    annBadge.className = 'badge annot-status ' + status;
    annBadge.textContent = status;
    annBadge.title = 'Annotation status: ' + status;
    badges.appendChild(annBadge);
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
    html += `<div class="prompt-box" data-export="prompt">
      <details${isLong ? '' : ' open'}>
        <summary>Prompt (${row.prompt.length.toLocaleString()} chars)</summary>
        <pre>${highlightSearch(promptFormatted)}</pre>
      </details>
    </div>`;
  }

  // Context columns
  const subfields = colInfo.context_subfields || {};
  (colInfo.context_cols || []).forEach(c => {
    if (row[c]) {
      const raw = stringify(row[c]);
      const formatted = formatTextWithJson(raw);
      html += `<div class="prompt-box" data-export="context-${c}">
        <details><summary>${esc(c.charAt(0).toUpperCase() + c.slice(1).replace(/_/g, ' '))} (${raw.length.toLocaleString()} chars)</summary>
        <pre>${formatted}</pre></details>
      </div>`;
      // Render extracted sub-fields as separate toggleable sections
      if (subfields[c]) {
        subfields[c].forEach(([subKey, subLabel]) => {
          const fullKey = c + '.' + subKey;
          if (row[fullKey] != null) {
            const subRaw = stringify(row[fullKey]);
            const subFormatted = formatTextWithJson(subRaw);
            html += `<div class="prompt-box" data-export="context-${fullKey}" style="border-left:3px solid #ff9800;margin-left:12px;">
              <details open><summary>${esc(subLabel)}</summary>
              <pre>${highlightSearch(subFormatted)}</pre></details>
            </div>`;
          }
        });
      }
    }
  });

  // Completions grid
  const completions = row.completions || {};
  const stageLabels = getStageLabels();
  const nCols = stageLabels.length;
  html += `<div class="completions-grid" data-export="completions" style="grid-template-columns: repeat(${nCols}, 1fr);">`;
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
    html += '<div class="meta-row" data-export="ground-truth">';
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
      html += '<div class="meta-row" data-export="predictions">';
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
      const openByDefault = ANNOTATE_MODE ? ' open' : '';
      html += `<div class="judge-section" data-export="judge">`;
      html += `<div class="judge-toggle${openByDefault}" onclick="
        this.classList.toggle('open');
        document.getElementById('${toggleId}').classList.toggle('open');
      "><span class="arrow">&#9654;</span> ${esc(judgeName)}</div>`;
      const judgeLabels = getStageLabels().filter(l => judgeData[l]);
      const nJCols = judgeLabels.length;
      // Per-judge ranking strip (only when 2+ models are present)
      html += buildRankingStrip(row, judgeName, judgeLabels);
      html += `<div class="judge-grid${openByDefault}" id="${toggleId}" style="grid-template-columns: repeat(${nJCols}, 1fr);">`;
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
            html += `<pre data-export="judge-text">${highlightSearch(formatTextWithJson(String(jEntry[tk])))}</pre>`;
          }
        });
        // Expert annotation strip (visible only when ANNOTATE_MODE)
        html += buildAnnotationStrip(row, label, judgeName, jEntry);
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

// ── Annotation module ────────────────────────────────────────────────
// State shape:
//   annotations[idx][model_label][judge_display_name] = {expert: str, notes: str}
//   rankings[idx][judge_display_name][model_label] = int  (1 = best on the
//     judge's dimension — least leaky / most helpful. Ties allowed.)
let annotations = {};
let rankings = {};
let focusedCell = null;  // {idx, model, judge} or null
let _annotSaveTimer = null;

// Direction labels for ranking (purely cosmetic — shown above the chip row)
const _RANK_DIRECTION = {
  'Leakage Judge': '1 = least leaky',
  'Helpfulness Judge': '1 = most helpful',
};

function _manifestHash() {
  if (!MANIFEST) return 'no-manifest';
  // Canonical key: stage + seed + n_sampled + sampled_indices + models
  const core = {
    stage: MANIFEST.stage,
    seed: MANIFEST.seed,
    n_sampled: MANIFEST.n_sampled,
    sampled_indices: MANIFEST.sampled_indices,
    models: MANIFEST.models,
  };
  const s = JSON.stringify(core);
  // djb2 → unsigned 32-bit hex
  let h = 5381;
  for (let i = 0; i < s.length; i++) h = ((h * 33) ^ s.charCodeAt(i)) >>> 0;
  return h.toString(16).padStart(8, '0');
}

function _annotStorageKey() {
  return 'inspector:annot:v1:' + _manifestHash();
}

function loadAnnotations() {
  annotations = {};
  rankings = {};
  if (!ANNOTATE_MODE) return;
  try {
    const raw = localStorage.getItem(_annotStorageKey());
    if (!raw) return;
    const parsed = JSON.parse(raw);
    // Schema v2: {annotations, rankings}.  v1: raw annotations dict.
    if (parsed && typeof parsed === 'object' && ('annotations' in parsed || 'rankings' in parsed)) {
      annotations = parsed.annotations || {};
      rankings = parsed.rankings || {};
    } else {
      annotations = parsed || {};
    }
  } catch (e) {
    console.warn('Failed to restore annotations:', e);
    annotations = {};
    rankings = {};
  }
}

function _scheduleSave() {
  if (_annotSaveTimer) clearTimeout(_annotSaveTimer);
  _annotSaveTimer = setTimeout(() => {
    try {
      localStorage.setItem(_annotStorageKey(), JSON.stringify({annotations, rankings}));
    } catch (e) {
      console.error('Failed to save annotations:', e);
    }
  }, 250);
}

// ── Rankings ─────────────────────────────────────────────────────────
function getRanking(idx, judge, model) {
  return (rankings[idx] && rankings[idx][judge] && rankings[idx][judge][model]) || null;
}

function setRanking(idx, judge, model, rank) {
  if (!rankings[idx]) rankings[idx] = {};
  if (!rankings[idx][judge]) rankings[idx][judge] = {};
  if (rank == null) {
    delete rankings[idx][judge][model];
    if (Object.keys(rankings[idx][judge]).length === 0) delete rankings[idx][judge];
    if (Object.keys(rankings[idx]).length === 0) delete rankings[idx];
  } else {
    rankings[idx][judge][model] = rank;
  }
  _scheduleSave();
}

// Click on a chip → cycle: null → 1 → 2 → ... → N → null
function cycleRanking(idx, judge, model, nModels) {
  const cur = getRanking(idx, judge, model);
  let next;
  if (cur == null) next = 1;
  else if (cur >= nModels) next = null;
  else next = cur + 1;
  setRanking(idx, judge, model, next);
}

function clearRankingForJudge(idx, judge) {
  if (rankings[idx] && rankings[idx][judge]) {
    delete rankings[idx][judge];
    if (Object.keys(rankings[idx]).length === 0) delete rankings[idx];
    _scheduleSave();
  }
}

function getAnnotation(idx, model, judge) {
  return (annotations[idx] && annotations[idx][model] && annotations[idx][model][judge]) || null;
}

function setAnnotation(idx, model, judge, partial) {
  if (!annotations[idx]) annotations[idx] = {};
  if (!annotations[idx][model]) annotations[idx][model] = {};
  const cur = annotations[idx][model][judge] || {expert: null, notes: ''};
  annotations[idx][model][judge] = {...cur, ...partial};
  _scheduleSave();
}

function clearAnnotation(idx, model, judge) {
  if (annotations[idx] && annotations[idx][model]) {
    delete annotations[idx][model][judge];
    if (Object.keys(annotations[idx][model]).length === 0) delete annotations[idx][model];
    if (Object.keys(annotations[idx]).length === 0) delete annotations[idx];
    _scheduleSave();
  }
}

// Expected expert verdict given the judge's verdict (T/F), or null if undecided.
function _expectedExpertFor(judgeName, jEntry) {
  const meta = JUDGE_META[judgeName];
  if (!meta) return null;
  const v = jEntry ? jEntry[meta.verdict_field] : undefined;
  if (v === true) return meta.true_value;
  if (v === false) return meta.false_value;
  return null;
}

// Returns 'agree' | 'disagree' | 'empty'
function annotAgreement(idx, model, judgeName, jEntry) {
  const annot = getAnnotation(idx, model, judgeName);
  if (!annot || !annot.expert) return 'empty';
  if (annot.expert === 'skip') return 'empty';
  const expected = _expectedExpertFor(judgeName, jEntry);
  if (expected === null) return 'empty';
  return annot.expert === expected ? 'agree' : 'disagree';
}

// Total cells = N rows * sum over each row of (#models with judge present, per judge).
// Returns {annotated, total, disagree, perJudge: {judgeName: {annotated, total}}}
function computeProgress() {
  let total = 0, annotated = 0, disagree = 0;
  const perJudge = {};
  if (!ANNOTATE_MODE) return {annotated, total, disagree, perJudge};

  const stageData = DATA[currentStage];
  if (!stageData) return {annotated, total, disagree, perJudge};
  const judgeNames = stageData.judge_names || [];

  for (const row of allRows) {
    if (!row.judges) continue;
    for (const jn of judgeNames) {
      const jData = row.judges[jn];
      if (!jData) continue;
      if (!perJudge[jn]) perJudge[jn] = {annotated: 0, total: 0};
      for (const label of getStageLabels()) {
        const jEntry = jData[label];
        if (!jEntry) continue;
        total += 1;
        perJudge[jn].total += 1;
        const annot = getAnnotation(row.idx, label, jn);
        if (annot && annot.expert) {
          annotated += 1;
          perJudge[jn].annotated += 1;
          if (annot.expert !== 'skip') {
            const expected = _expectedExpertFor(jn, jEntry);
            if (expected !== null && annot.expert !== expected) disagree += 1;
          }
        }
      }
    }
  }
  return {annotated, total, disagree, perJudge};
}

// Returns 'unannot' | 'partial' | 'complete' | 'disagree'
function getRowStatus(row) {
  if (!row.judges) return 'unannot';
  const stageData = DATA[currentStage];
  const judgeNames = stageData.judge_names || [];
  let cells = 0, done = 0, anyDisagree = false;
  for (const jn of judgeNames) {
    const jData = row.judges[jn];
    if (!jData) continue;
    for (const label of getStageLabels()) {
      const jEntry = jData[label];
      if (!jEntry) continue;
      cells += 1;
      const annot = getAnnotation(row.idx, label, jn);
      if (annot && annot.expert) {
        done += 1;
        if (annot.expert !== 'skip') {
          const expected = _expectedExpertFor(jn, jEntry);
          if (expected !== null && annot.expert !== expected) anyDisagree = true;
        }
      }
    }
  }
  if (cells === 0) return 'unannot';
  if (done === 0) return 'unannot';
  if (anyDisagree && done === cells) return 'disagree';
  if (anyDisagree) return 'partial';  // partially annotated with at least one disagreement
  if (done === cells) return 'complete';
  return 'partial';
}

function buildRankingStrip(row, judgeName, modelsForJudge) {
  if (!ANNOTATE_MODE) return '';
  if (!modelsForJudge || modelsForJudge.length < 2) return '';  // ranking needs >=2 models
  const direction = _RANK_DIRECTION[judgeName] || '1 = best';
  const chips = modelsForJudge.map(m => {
    const r = getRanking(row.idx, judgeName, m);
    const valHtml = r == null ? '—' : String(r);
    const cls = r == null ? 'rank-value empty' : 'rank-value';
    return `<span class="rank-chip" data-model="${esc(m)}">
      <span class="rank-model">${esc(m)}</span>
      <span class="${cls}">${valHtml}</span>
    </span>`;
  }).join('');
  return `
    <div class="rank-strip" data-idx="${row.idx}" data-judge="${esc(judgeName)}"
         data-nmodels="${modelsForJudge.length}">
      <span class="rank-label">Rank</span>
      <span class="rank-dir">(${esc(direction)}; click chips to cycle)</span>
      <span class="rank-chips">${chips}</span>
      <span class="rank-clear" data-action="clear">clear</span>
    </div>
  `;
}

function buildAnnotationStrip(row, model, judgeName, jEntry) {
  if (!ANNOTATE_MODE) return '';
  const meta = JUDGE_META[judgeName];
  if (!meta) return '';
  const annot = getAnnotation(row.idx, model, judgeName) || {expert: null, notes: ''};
  const agree = annotAgreement(row.idx, model, judgeName, jEntry);
  const focused = focusedCell &&
    focusedCell.idx === row.idx &&
    focusedCell.model === model &&
    focusedCell.judge === judgeName;

  const optsHtml = meta.options.map(o =>
    `<label class="${annot.expert === o.value ? 'active' : ''}"
            data-value="${esc(o.value)}">${esc(o.label)}</label>`
  ).join('');

  const agreeHtml = `<span class="annot-agree ${agree}">${
    agree === 'agree' ? '✓ agree with judge'
    : agree === 'disagree' ? '✗ disagree with judge'
    : '—'
  }</span>`;

  return `
    <div class="annot-strip${focused ? ' focused' : ''}"
         data-idx="${row.idx}" data-model="${esc(model)}" data-judge="${esc(judgeName)}">
      <div class="annot-row">
        <span class="annot-label">Expert</span>
        <span class="annot-radios">${optsHtml}</span>
        ${agreeHtml}
      </div>
      <div class="annot-row">
        <input type="text" class="annot-notes" placeholder="Notes (optional)"
               value="${esc(annot.notes || '')}">
      </div>
    </div>
  `;
}

// Delegated click handler for ranking chips and clear control.
function _onRankClick(e) {
  const strip = e.target.closest('.rank-strip');
  if (!strip) return;
  const idx = parseInt(strip.dataset.idx, 10);
  const judge = strip.dataset.judge;
  const nModels = parseInt(strip.dataset.nmodels, 10);

  if (e.target.closest('.rank-clear')) {
    clearRankingForJudge(idx, judge);
    _patchRankStrip(strip);
    return;
  }
  const chip = e.target.closest('.rank-chip');
  if (!chip) return;
  const model = chip.dataset.model;
  cycleRanking(idx, judge, model, nModels);
  _patchRankStrip(strip);
}

function _patchRankStrip(strip) {
  const idx = parseInt(strip.dataset.idx, 10);
  const judge = strip.dataset.judge;
  strip.querySelectorAll('.rank-chip').forEach(chip => {
    const model = chip.dataset.model;
    const r = getRanking(idx, judge, model);
    const valEl = chip.querySelector('.rank-value');
    if (!valEl) return;
    if (r == null) {
      valEl.textContent = '—';
      valEl.className = 'rank-value empty';
    } else {
      valEl.textContent = String(r);
      valEl.className = 'rank-value';
    }
  });
}

// Delegated handler for annotation interactions.  Attaches once at init().
function _onAnnotClick(e) {
  // Ranking strips are handled separately
  if (e.target.closest('.rank-strip')) return;
  const strip = e.target.closest('.annot-strip');
  if (!strip) return;
  const idx = parseInt(strip.dataset.idx, 10);
  const model = strip.dataset.model;
  const judge = strip.dataset.judge;
  focusedCell = {idx, model, judge};

  // Radio click?
  const lbl = e.target.closest('.annot-radios label');
  if (lbl) {
    const value = lbl.dataset.value;
    const cur = getAnnotation(idx, model, judge);
    const newValue = (cur && cur.expert === value) ? null : value;
    if (newValue === null) {
      clearAnnotation(idx, model, judge);
    } else {
      setAnnotation(idx, model, judge, {expert: newValue});
    }
    _patchAnnotStrip(strip);
    _patchRowStatus(idx);
    _updateAnnotProgress();
    _refreshAnnotFocusUI();
    if (activeFilter !== 'all' && activeFilter.startsWith('annot:')) {
      refilter();
    }
    return;
  }

  // Clicking elsewhere on the strip just sets focus
  _refreshAnnotFocusUI();
}

function _onAnnotNotesInput(e) {
  if (!e.target.classList.contains('annot-notes')) return;
  const strip = e.target.closest('.annot-strip');
  if (!strip) return;
  const idx = parseInt(strip.dataset.idx, 10);
  const model = strip.dataset.model;
  const judge = strip.dataset.judge;
  setAnnotation(idx, model, judge, {notes: e.target.value});
  // Don't re-render — input cursor would jump
}

function _onAnnotFocus(e) {
  const strip = e.target.closest('.annot-strip');
  if (!strip) return;
  focusedCell = {
    idx: parseInt(strip.dataset.idx, 10),
    model: strip.dataset.model,
    judge: strip.dataset.judge,
  };
  _refreshAnnotFocusUI();
}

function _patchAnnotStrip(strip) {
  // Re-render this strip in place from current state.
  const idx = parseInt(strip.dataset.idx, 10);
  const model = strip.dataset.model;
  const judge = strip.dataset.judge;
  const row = allRows.find(r => r.idx === idx);
  if (!row || !row.judges || !row.judges[judge]) return;
  const jEntry = row.judges[judge][model];
  if (!jEntry) return;

  const annot = getAnnotation(idx, model, judge) || {expert: null, notes: ''};
  strip.querySelectorAll('.annot-radios label').forEach(l => {
    l.classList.toggle('active', l.dataset.value === annot.expert);
  });
  const ag = annotAgreement(idx, model, judge, jEntry);
  const agreeSpan = strip.querySelector('.annot-agree');
  if (agreeSpan) {
    agreeSpan.className = 'annot-agree ' + ag;
    agreeSpan.textContent = ag === 'agree' ? '✓ agree with judge'
      : ag === 'disagree' ? '✗ disagree with judge' : '—';
  }
  // Don't touch notes input (preserve cursor)
}

function _patchRowStatus(idx) {
  const card = document.querySelector(`.row-card[data-row-idx="${idx}"]`);
  if (!card) return;
  const status = getRowStatus(allRows.find(r => r.idx === idx));
  const badge = card.querySelector('.badge.annot-status');
  if (badge) {
    badge.className = 'badge annot-status ' + status;
    badge.textContent = status;
  }
}

function _refreshAnnotFocusUI() {
  document.querySelectorAll('.annot-strip').forEach(s => {
    const idx = parseInt(s.dataset.idx, 10);
    const isFocus = focusedCell &&
      focusedCell.idx === idx &&
      focusedCell.model === s.dataset.model &&
      focusedCell.judge === s.dataset.judge;
    s.classList.toggle('focused', isFocus);
  });
}

function _updateAnnotProgress() {
  if (!ANNOTATE_MODE) return;
  const p = computeProgress();
  const breakdown = Object.entries(p.perJudge)
    .map(([jn, c]) => `${jn}: ${c.annotated}/${c.total}`).join(' · ');
  const el = document.getElementById('annot-progress');
  if (el) {
    el.innerHTML = `Annotated <b>${p.annotated}</b>/${p.total}`
      + (p.disagree > 0 ? ` · <span style="color:#c62828">${p.disagree} disagree</span>` : '');
    el.title = breakdown || 'No judge cells available';
  }
}

function _getRowCellsInDOMOrder(rowIdx) {
  // Return list of {idx, model, judge} cells visible on the current row card.
  const card = document.querySelector(`.row-card[data-row-idx="${rowIdx}"]`);
  if (!card) return [];
  return [...card.querySelectorAll('.annot-strip')].map(s => ({
    idx: parseInt(s.dataset.idx, 10),
    model: s.dataset.model,
    judge: s.dataset.judge,
  }));
}

function annotFocusNext(dir) {
  const row = filteredRows[currentIdx];
  if (!row) return;
  const cells = _getRowCellsInDOMOrder(row.idx);
  if (cells.length === 0) return;
  let pos = -1;
  if (focusedCell) {
    pos = cells.findIndex(c =>
      c.idx === focusedCell.idx && c.model === focusedCell.model && c.judge === focusedCell.judge);
  }
  if (pos < 0) {
    focusedCell = dir > 0 ? cells[0] : cells[cells.length - 1];
  } else {
    const next = (pos + dir + cells.length) % cells.length;
    focusedCell = cells[next];
  }
  _refreshAnnotFocusUI();
  // Make sure focused strip is visible
  const sel = `.annot-strip[data-idx="${focusedCell.idx}"][data-model="${focusedCell.model}"][data-judge="${focusedCell.judge}"]`;
  const el = document.querySelector(sel);
  if (el) el.scrollIntoView({block: 'nearest', behavior: 'smooth'});
}

function annotSetVerdictByIndex(optIdx) {
  if (!focusedCell) {
    // Auto-focus first cell of current row
    annotFocusNext(1);
    if (!focusedCell) return;
  }
  const meta = JUDGE_META[focusedCell.judge];
  if (!meta || optIdx < 0 || optIdx >= meta.options.length) return;
  const value = meta.options[optIdx].value;
  const cur = getAnnotation(focusedCell.idx, focusedCell.model, focusedCell.judge);
  if (cur && cur.expert === value) {
    clearAnnotation(focusedCell.idx, focusedCell.model, focusedCell.judge);
  } else {
    setAnnotation(focusedCell.idx, focusedCell.model, focusedCell.judge, {expert: value});
  }
  const sel = `.annot-strip[data-idx="${focusedCell.idx}"][data-model="${focusedCell.model}"][data-judge="${focusedCell.judge}"]`;
  const strip = document.querySelector(sel);
  if (strip) _patchAnnotStrip(strip);
  _patchRowStatus(focusedCell.idx);
  _updateAnnotProgress();
  if (activeFilter.startsWith('annot:')) refilter();
}

function annotFocusNotes() {
  if (!focusedCell) return;
  const sel = `.annot-strip[data-idx="${focusedCell.idx}"][data-model="${focusedCell.model}"][data-judge="${focusedCell.judge}"] input.annot-notes`;
  const inp = document.querySelector(sel);
  if (inp) { inp.focus(); inp.select(); }
}

// ── Export / Import annotations ──────────────────────────────────────
function exportAnnotationsJSON() {
  const payload = {manifest: MANIFEST, annotations, rankings};
  const blob = new Blob([JSON.stringify(payload, null, 2)], {type: 'application/json'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = _annotFilenameBase() + '.json';
  a.click();
  URL.revokeObjectURL(url);
}

function exportAnnotationsCSV() {
  // One row per (row_idx, model, judge) cell — annotated or not.
  const lines = ['row_idx,model,judge,judge_verdict,expert_verdict,agree,rank,notes'];
  const stageData = DATA[currentStage];
  const judgeNames = (stageData && stageData.judge_names) || [];
  const labels = getStageLabels();
  for (const row of allRows) {
    if (!row.judges) continue;
    for (const jn of judgeNames) {
      const jData = row.judges[jn];
      if (!jData) continue;
      for (const label of labels) {
        const jEntry = jData[label];
        if (!jEntry) continue;
        const annot = getAnnotation(row.idx, label, jn);
        const meta = JUDGE_META[jn];
        const jv = (meta && jEntry[meta.verdict_field]);
        const judgeVerdict = jv === true ? meta.true_value
          : jv === false ? meta.false_value : '';
        const expert = (annot && annot.expert) || '';
        const ag = annotAgreement(row.idx, label, jn, jEntry);
        const agree = (expert === '' || expert === 'skip' || ag === 'empty') ? ''
          : (ag === 'agree' ? 'true' : 'false');
        const rank = getRanking(row.idx, jn, label);
        const notes = (annot && annot.notes) ? annot.notes : '';
        lines.push([
          row.idx, label, jn, judgeVerdict, expert, agree,
          rank == null ? '' : rank, notes,
        ].map(_csvEsc).join(','));
      }
    }
  }
  const blob = new Blob([lines.join('\n')], {type: 'text/csv'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = _annotFilenameBase() + '.csv';
  a.click();
  URL.revokeObjectURL(url);
}

function _csvEsc(v) {
  const s = String(v == null ? '' : v);
  if (/[",\n\r]/.test(s)) return '"' + s.replace(/"/g, '""') + '"';
  return s;
}

function _annotFilenameBase() {
  const stage = (MANIFEST && MANIFEST.stage) ? MANIFEST.stage.replace(/[^a-zA-Z0-9]+/g, '_') : 'annotations';
  const seed = (MANIFEST && MANIFEST.seed != null) ? `_seed${MANIFEST.seed}` : '';
  const n = (MANIFEST && MANIFEST.n_sampled != null) ? `_n${MANIFEST.n_sampled}` : '';
  return `annot_${stage}${seed}${n}`;
}

function importAnnotationsFile(file) {
  const reader = new FileReader();
  reader.onload = e => {
    try {
      const payload = JSON.parse(e.target.result);
      const incoming = payload.annotations || payload;  // accept raw or wrapped
      const incomingManifest = payload.manifest;

      // Warn on manifest mismatch
      let warn = '';
      if (incomingManifest && MANIFEST) {
        const issues = [];
        if (incomingManifest.stage !== MANIFEST.stage)
          issues.push(`stage (${incomingManifest.stage} != ${MANIFEST.stage})`);
        if (incomingManifest.seed !== MANIFEST.seed)
          issues.push(`seed (${incomingManifest.seed} != ${MANIFEST.seed})`);
        if (incomingManifest.n_sampled !== MANIFEST.n_sampled)
          issues.push(`n_sampled (${incomingManifest.n_sampled} != ${MANIFEST.n_sampled})`);
        if (issues.length) warn = 'Manifest differs from this export: ' + issues.join(', ') + '. Continue anyway?';
      }
      if (warn && !confirm(warn)) return;

      // Merge annotations (incoming overrides)
      for (const [idx, byModel] of Object.entries(incoming)) {
        if (!annotations[idx]) annotations[idx] = {};
        for (const [model, byJudge] of Object.entries(byModel)) {
          if (!annotations[idx][model]) annotations[idx][model] = {};
          for (const [judge, val] of Object.entries(byJudge)) {
            annotations[idx][model][judge] = val;
          }
        }
      }
      // Merge rankings if present (incoming overrides)
      const incomingRankings = payload.rankings || {};
      for (const [idx, byJudge] of Object.entries(incomingRankings)) {
        if (!rankings[idx]) rankings[idx] = {};
        for (const [judge, byModel] of Object.entries(byJudge)) {
          if (!rankings[idx][judge]) rankings[idx][judge] = {};
          for (const [model, val] of Object.entries(byModel)) {
            rankings[idx][judge][model] = val;
          }
        }
      }
      _scheduleSave();
      render('header');
      _updateAnnotProgress();
      alert('Annotations imported.');
    } catch (err) {
      alert('Failed to parse annotations JSON: ' + err.message);
    }
  };
  reader.readAsText(file);
}

function clearAllAnnotations() {
  if (!confirm('Clear ALL annotations AND rankings for this sample? This cannot be undone.')) return;
  annotations = {};
  rankings = {};
  try { localStorage.removeItem(_annotStorageKey()); } catch (e) {}
  render('header');
  _updateAnnotProgress();
}

// Manifest banner
function _renderManifestBanner() {
  const el = document.getElementById('manifest-banner');
  if (!el || !MANIFEST) return;
  const parts = [
    `<b>${esc(MANIFEST.stage)}</b>`,
    `seed=<b>${MANIFEST.seed != null ? MANIFEST.seed : 'n/a'}</b>`,
    `sample=<b>${MANIFEST.n_sampled}</b> of ${MANIFEST.n_total}`,
    `models=<b>${(MANIFEST.models || []).join(', ')}</b>`,
    `generated <b>${esc(MANIFEST.generated_at || '')}</b>`,
  ];
  el.innerHTML = parts.join(' &nbsp;·&nbsp; ');
}

// ── Bookmarks ────────────────────────────────────────────────────────
function toggleBookmark(idx) {
  if (bookmarks.has(idx)) bookmarks.delete(idx);
  else bookmarks.add(idx);

  // If the bookmarked filter is active, the row set changes — must re-render.
  if (activeFilter === 'bookmarked') {
    render('none');
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
      if (fi >= 0) { currentIdx = fi; render('center'); }
      else {
        // Switch to 'all' filter to find it
        activeFilter = 'all';
        searchQuery = '';
        document.getElementById('search-input').value = '';
        applyFilters();
        const fi2 = filteredRows.findIndex(r => r.idx === idx);
        if (fi2 >= 0) { currentIdx = fi2; render('center'); }
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
      if (currentIdx < filteredRows.length - 1) { currentIdx++; render('header'); }
      break;
    case 'k': case 'ArrowUp':
      e.preventDefault();
      if (currentIdx > 0) { currentIdx--; render('header'); }
      break;
    case 'Enter':
      // Already expanded via currentIdx, this is a no-op but feels natural
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
    case 'Tab':
      if (ANNOTATE_MODE) {
        e.preventDefault();
        annotFocusNext(e.shiftKey ? -1 : 1);
      }
      break;
    case '1': case '2': case '3':
      if (ANNOTATE_MODE) {
        e.preventDefault();
        annotSetVerdictByIndex(parseInt(e.key, 10) - 1);
      }
      break;
    case 'n':
      if (ANNOTATE_MODE) {
        e.preventDefault();
        annotFocusNotes();
      }
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
  const colInfo = stageData.col_info;

  document.getElementById('export-title').textContent = `Export Row #${row.idx} — ${currentStage}`;

  // Build toggle controls
  const ctrl = document.getElementById('export-controls');
  ctrl.innerHTML = '';

  const sections = [];

  // Detect which sections exist in this row
  const subfields = colInfo.context_subfields || {};
  if (row.prompt) sections.push({id: 'prompt', label: 'Prompt', on: true});
  (colInfo.context_cols || []).forEach(c => {
    if (row[c]) {
      sections.push({id: `context-${c}`, label: c, on: true});
      // Sub-field toggles
      if (subfields[c]) {
        subfields[c].forEach(([subKey, subLabel]) => {
          const fullKey = c + '.' + subKey;
          if (row[fullKey] != null) {
            sections.push({id: `context-${fullKey}`, label: '  \u2514 ' + subLabel, on: true});
          }
        });
      }
    }
  });
  sections.push({id: 'completions', label: 'Completions', on: true});
  if ((colInfo.ground_truth_cols || []).some(c => row[c] !== undefined && row[c] !== null))
    sections.push({id: 'ground-truth', label: 'Ground Truth', on: true});
  if (row.predictions) sections.push({id: 'predictions', label: 'Predictions', on: true});

  // Checkboxes for each section
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

  // Judge detail radio group (if judges exist)
  if (row.judges && Object.keys(row.judges).length > 0) {
    const sep = document.createElement('span');
    sep.style.cssText = 'width:1px;height:18px;background:#ccc;margin:0 4px;';
    ctrl.appendChild(sep);

    const glbl = document.createElement('span');
    glbl.style.fontWeight = '600';
    glbl.textContent = 'Judges:';
    ctrl.appendChild(glbl);

    const eg = document.createElement('span');
    eg.className = 'eg';
    ['none', 'verdict', 'full'].forEach(val => {
      const lbl = document.createElement('label');
      lbl.textContent = val === 'none' ? 'None' : val === 'verdict' ? 'Verdict' : 'Full';
      lbl.dataset.judgeLevel = val;
      if (val === 'verdict') lbl.classList.add('active');
      lbl.addEventListener('click', () => {
        eg.querySelectorAll('label').forEach(l => l.classList.remove('active'));
        lbl.classList.add('active');
        updateExportPreview();
      });
      eg.appendChild(lbl);
    });
    ctrl.appendChild(eg);
  }

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

  // Render initial preview
  updateExportPreview();

  document.getElementById('export-overlay').style.display = 'flex';
}

function getExportOptions() {
  const opts = {sections: {}, judgeLevel: 'verdict'};
  document.querySelectorAll('#export-controls input[type=checkbox]').forEach(cb => {
    opts.sections[cb.dataset.section] = cb.checked;
  });
  const activeJudge = document.querySelector('#export-controls .eg label.active');
  if (activeJudge) opts.judgeLevel = activeJudge.dataset.judgeLevel;
  return opts;
}

function _getExportWidth() {
  const el = document.getElementById('export-width');
  return el ? parseInt(el.value, 10) || 1100 : 1100;
}

function updateExportPreview() {
  const frame = document.getElementById('export-frame');
  frame.style.maxWidth = _getExportWidth() + 'px';
  // Clone the current row body from the main view
  const currentBody = document.querySelector('.row-card.current .row-body');
  if (!currentBody) return;

  // Clone and clean up
  const clone = currentBody.cloneNode(true);
  clone.style.display = 'block';

  // Force all <details> open in the export
  clone.querySelectorAll('details').forEach(d => d.setAttribute('open', ''));

  // Remove search highlights
  clone.querySelectorAll('mark').forEach(m => {
    m.replaceWith(document.createTextNode(m.textContent));
  });

  // Remove max-height constraints on pre elements
  clone.querySelectorAll('pre').forEach(p => { p.style.maxHeight = 'none'; });
  clone.querySelectorAll('.prompt-box').forEach(p => { p.style.maxHeight = 'none'; });

  const opts = getExportOptions();

  // Apply section visibility
  clone.querySelectorAll('[data-export]').forEach(el => {
    const key = el.getAttribute('data-export');
    if (key in opts.sections) {
      el.classList.toggle('export-hidden', !opts.sections[key]);
    }
  });

  // Apply judge level
  clone.querySelectorAll('[data-export="judge"]').forEach(el => {
    if (opts.judgeLevel === 'none') {
      el.classList.add('export-hidden');
    } else {
      el.classList.remove('export-hidden');
      // Show/hide judge reasoning text
      el.querySelectorAll('[data-export="judge-text"]').forEach(t => {
        t.classList.toggle('export-hidden', opts.judgeLevel !== 'full');
      });
      // Also force judge grids open
      el.querySelectorAll('.judge-grid').forEach(g => g.classList.add('open'));
      el.querySelectorAll('.judge-toggle').forEach(t => t.classList.add('open'));
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
    pre { white-space: pre-wrap; word-break: break-word; font-family: 'SF Mono','Cascadia Code','Fira Code',Consolas,monospace; font-size: 11.5px; line-height: 1.5; margin: 0; }
    .prompt-box { background: #f5f5f5; border: 1px solid #e8e8e8; border-radius: 6px; padding: 10px 14px; margin-bottom: 10px; }
    .completions-grid { display: grid; gap: 8px; margin-bottom: 10px; align-items: stretch; }
    .completion-col { border: 1px solid #e0e0e0; border-radius: 6px; overflow: hidden; display: grid; grid-template-rows: auto 1fr; }
    .completion-col .col-header { padding: 5px 10px; font-weight: 600; font-size: 12px; border-bottom: 1px solid #e0e0e0; }
    .completion-col pre { padding: 8px 10px; }
    .meta-row { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 8px; font-size: 13px; }
    .meta-row .label { font-weight: 600; color: #555; }
    .meta-row .val { font-family: 'SF Mono',Consolas,monospace; }
    .meta-row .val.correct { color: #2e7d32; font-weight: 600; }
    .meta-row .val.wrong { color: #c62828; font-weight: 600; }
    .judge-section { margin-bottom: 10px; }
    .judge-toggle { display: none; }
    .judge-grid { display: grid !important; gap: 8px; margin-top: 0; align-items: stretch; }
    .judge-card { border: 1px solid #e0d6f5; border-radius: 6px; overflow: hidden; display: grid; grid-template-rows: auto 1fr; }
    .judge-card .judge-header { padding: 5px 10px; font-size: 11px; font-weight: 600; background: #f3e5f5; border-bottom: 1px solid #e0d6f5; display: flex; gap: 8px; flex-wrap: wrap; }
    .judge-card .judge-header .verdict { padding: 1px 6px; border-radius: 8px; font-weight: 700; font-size: 11px; }
    .judge-card .judge-header .verdict.leak { background: #ffebee; color: #c62828; }
    .judge-card .judge-header .verdict.no-leak { background: #e8f5e9; color: #2e7d32; }
    .judge-card .judge-header .verdict.helpful { background: #e8f5e9; color: #2e7d32; }
    .judge-card .judge-header .verdict.not-helpful { background: #fff3e0; color: #ef6c00; }
    .judge-card pre { padding: 6px 10px; font-size: 11px; line-height: 1.4; background: #faf5ff; }
    .json-block { background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 4px; padding: 6px 8px; margin: 4px 0; font-size: 11px; }
    .json-key { color: #881391; } .json-str { color: #0b7285; } .json-num { color: #d9480f; }
    .json-bool { color: #5c940d; font-weight: 600; } .json-null { color: #868e96; font-style: italic; }
    .badge { font-size: 11px; padding: 2px 8px; border-radius: 10px; font-weight: 500; }
    .col-header.correct { border-bottom-color: #2e7d32; }
    .col-header.wrong { border-bottom-color: #c62828; }
  `;
}

function _getExportHTML() {
  const frame = document.getElementById('export-frame');
  const clone = frame.cloneNode(true);
  // Remove hidden sections entirely from the DOM
  clone.querySelectorAll('.export-hidden').forEach(el => el.remove());
  // Remove data-export attributes (cleanup)
  clone.querySelectorAll('[data-export]').forEach(el => el.removeAttribute('data-export'));
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
  const win = window.open('', '_blank');
  win.document.write(`<!DOCTYPE html><html><head><meta charset="utf-8"><title>${_exportFilename('pdf')}</title>
<style>
${css}
@media print { @page { margin: 0.5in; size: auto; } body { -webkit-print-color-adjust: exact; print-color-adjust: exact; } }
</style></head><body style="max-width:${w}px;padding:12px;">
${content}
<script>window.onload=function(){window.print();}<\\/script>
</body></html>`);
  win.document.close();
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


def _validate_stage_key(requested: str, available: list[str]) -> str:
    """Resolve a user-supplied --stage value against discovered stage keys.

    Accepts exact matches, whitespace-normalized matches (the canonical key
    is rendered as ``"benchmark / stage"`` but users naturally type
    ``"benchmark/stage"``), or unique substring matches. Raises ValueError
    on miss with up-to-3 close suggestions.
    """
    if requested in available:
        return requested

    # Whitespace-tolerant exact match: collapse all whitespace around slashes
    def _norm(s: str) -> str:
        return "/".join(p.strip() for p in s.split("/"))
    norm_req = _norm(requested)
    norm_matches = [k for k in available if _norm(k) == norm_req]
    if len(norm_matches) == 1:
        return norm_matches[0]
    if len(norm_matches) > 1:
        raise ValueError(
            f"--stage '{requested}' matches {len(norm_matches)} stages (whitespace-normalized): "
            f"{sorted(norm_matches)}. Be more specific."
        )

    # Substring match — only if unique
    subs = [k for k in available if requested in k or norm_req in _norm(k)]
    if len(subs) == 1:
        return subs[0]
    if len(subs) > 1:
        raise ValueError(
            f"--stage '{requested}' matches {len(subs)} stages: {sorted(subs)}. "
            f"Be more specific."
        )

    # No match — suggest close ones
    close = difflib.get_close_matches(requested, available, n=3, cutoff=0.4)
    msg = f"--stage '{requested}' not found among {len(available)} discovered stages."
    if close:
        msg += f" Did you mean: {close}?"
    else:
        msg += f" Available: {sorted(available)}"
    raise ValueError(msg)


def _select_sample(n_total: int, n_sample: int, seed: int) -> list[int]:
    """Return a sorted list of sampled row indices, deterministic given seed.

    Raises ValueError if n_sample > n_total.
    """
    if n_sample > n_total:
        raise ValueError(
            f"--sample {n_sample} exceeds available rows ({n_total})"
        )
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_total, size=n_sample, replace=False)
    return sorted(int(i) for i in idx)


def _build_manifest(
    stage: str,
    seed: int | None,
    sampled_indices: list[int] | None,
    n_total: int,
    runs: dict[str, Path],
    models: list[str],
    n_sampled: int | None = None,
) -> dict:
    """Assemble the per-export manifest embedded in the HTML."""
    return {
        "stage": stage,
        "seed": seed,
        "n_sampled": n_sampled if n_sampled is not None else (
            len(sampled_indices) if sampled_indices is not None else n_total
        ),
        "n_total": n_total,
        "sampled_indices": sampled_indices,
        "models": models,
        "source_runs": {k: str(v) for k, v in runs.items()},
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
    }


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
            "Mutually exclusive with --max-rows and --sample."
        ),
    )
    parser.add_argument(
        "--stage", type=str, default=None, metavar="STAGE_KEY",
        help=(
            "Restrict output to a single discovered stage (e.g. "
            "'privacylens/agent_action_inference'). Required by --annotate."
        ),
    )
    parser.add_argument(
        "--sample", type=int, default=None, metavar="N",
        help=(
            "Randomly sample N rows from the (single) stage. Requires --stage. "
            "Mutually exclusive with --rows / --max-rows."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=0, metavar="S",
        help="RNG seed for --sample (default: 0).",
    )
    parser.add_argument(
        "--annotate", action="store_true",
        help=(
            "Enable the expert annotation UI for judge audit. Requires --stage. "
            "Typically combined with --sample."
        ),
    )
    args = parser.parse_args()

    if args.rows and args.max_rows:
        parser.error("--rows and --max-rows are mutually exclusive")
    if args.sample and args.rows:
        parser.error("--sample and --rows are mutually exclusive")
    if args.sample and args.max_rows:
        parser.error("--sample and --max-rows are mutually exclusive")
    if args.sample is not None and args.sample <= 0:
        parser.error("--sample must be a positive integer")
    if args.sample and not args.stage:
        parser.error("--sample requires --stage (single benchmark only)")
    if args.annotate and not args.stage:
        parser.error("--annotate requires --stage (single benchmark only)")

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

    # Apply --stage filter (single-benchmark mode)
    if args.stage:
        try:
            resolved_stage = _validate_stage_key(args.stage, sorted(all_stage_keys))
        except ValueError as e:
            print(f"\nERROR: {e}", file=sys.stderr)
            sys.exit(1)
        if resolved_stage != args.stage:
            print(f"\n--stage '{args.stage}' resolved to '{resolved_stage}'")
        all_stage_keys = {resolved_stage}
        print(f"\nRestricted to single stage: {resolved_stage}")

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

        # Apply --rows slice or --sample subset if specified
        n_total = min(len(df) for df in label_to_df.values())
        row_indices: list[int] | None = None
        if args.rows:
            try:
                row_indices = parse_row_slice(args.rows, n_total)
            except ValueError as e:
                print(f"\nERROR in stage '{stage_key}': {e}", file=sys.stderr)
                sys.exit(1)
        elif args.sample:
            try:
                row_indices = _select_sample(n_total, args.sample, args.seed)
            except ValueError as e:
                print(f"\nERROR in stage '{stage_key}': {e}", file=sys.stderr)
                sys.exit(1)
        if row_indices is not None:
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
            "sampled_indices": row_indices,  # None unless --sample/--rows applied
        }
        if judge_names_found:
            stage_data["judge_names"] = list(dict.fromkeys(judge_names_found))
        data[stage_key] = stage_data

        judge_str = f" + {', '.join(judge_names_found)}" if judge_names_found else ""
        print(f"{len(rows)} rows{judge_str}")

    # Build manifest (only meaningful in single-stage mode; null otherwise)
    manifest: dict | None = None
    if args.stage:
        single_stage = next(iter(data))
        sd = data[single_stage]
        manifest = _build_manifest(
            stage=single_stage,
            seed=args.seed if args.sample else None,
            sampled_indices=sd.get("sampled_indices"),
            n_total=sd["n_total"],
            runs=runs,
            models=sd["labels"],
            n_sampled=len(sd["rows"]),
        )

    # Generate HTML
    print(f"\nGenerating HTML ...", end=" ", flush=True)
    data_json = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    labels_json = json.dumps(labels)
    manifest_json = json.dumps(manifest, ensure_ascii=False, separators=(",", ":")) if manifest else "null"
    judge_meta_json = json.dumps(_judge_annotation_meta(), ensure_ascii=False)
    annotate_mode_js = "true" if args.annotate else "false"

    html = HTML_TEMPLATE.replace("__DATA_PLACEHOLDER__", data_json)
    html = html.replace("__LABELS_PLACEHOLDER__", labels_json)
    html = html.replace("__MANIFEST_PLACEHOLDER__", manifest_json)
    html = html.replace("__ANNOTATE_MODE_PLACEHOLDER__", annotate_mode_js)
    html = html.replace("__JUDGE_META_PLACEHOLDER__", judge_meta_json)

    out_path = Path(args.output)
    out_path.write_text(html, encoding="utf-8")
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"done ({size_mb:.1f} MB)")
    if args.annotate:
        print("Annotation mode: ON")
        if manifest:
            print(f"  Sample: {manifest['n_sampled']} / {manifest['n_total']} rows"
                  f" (seed={manifest['seed']})")
    print(f"\nOutput: {out_path.resolve()}")
    print(f"Open in browser: file://{out_path.resolve()}")


if __name__ == "__main__":
    main()
