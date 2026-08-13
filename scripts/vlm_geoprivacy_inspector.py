#!/usr/bin/env python3
"""VLM-GeoPrivacy Inspector — static HTML generator for image + MCQ/freeform inspection.

Specialised variant of completion_inspector.py tailored to vlm_geoprivacy_bench
runs. Embeds downscaled JPEG thumbnails inline (base64), renders a per-question
correctness grid for MCQ runs, and lazily embeds a Google Maps iframe showing
the situational context for each image's true coordinate.

Usage:
    python -m scripts.vlm_geoprivacy_inspector \\
        --runs "Base=/path/to/run_a" "SFT=/path/to/run_b" "GRPO=/path/to/run_c" \\
        -o geoprivacy.html

    # Single run, both modes if present:
    python -m scripts.vlm_geoprivacy_inspector \\
        --runs /path/to/run --o geoprivacy.html

    # Cap images at 480px (smaller HTML):
    python -m scripts.vlm_geoprivacy_inspector \\
        --runs A=/path/a --image-size 480 --image-quality 70 -o smaller.html

    # Skip one mode:
    python -m scripts.vlm_geoprivacy_inspector \\
        --runs A=/path/a --no-freeform -o mcq_only.html
"""

from __future__ import annotations

import argparse
import ast
import base64
import html as html_lib
import json
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

# Question text is imported at build time so the inspector stays in sync with
# the bench prompts without duplicating the question registry here.
from dagspaces.vlm_geoprivacy_bench.prompts import NUM_QUESTIONS, QUESTION_DATA

# Allow running as `python scripts/vlm_geoprivacy_inspector.py` in addition to `-m`.
sys.path.insert(0, str(Path(__file__).parent))
from _inspector_common import (  # noqa: E402
    parse_run_arg,
)

MCQ_REL = "outputs/parse_mcq/dataset.parquet"
FREEFORM_PARSE_REL = "outputs/parse_freeform/dataset.parquet"
FREEFORM_JUDGE_REL = "outputs/granularity_judge/dataset.parquet"

# Q7 labels that form an ordinal disclosure scale for over/under-disclosure
# computation.  Matches compute_metrics.LABEL_ORDER.
Q7_ORDER = {"A": 0, "B": 1, "C": 2}


# ── Run discovery ────────────────────────────────────────────────────────

def resolve_vlm_root(run_path: str) -> Path:
    """Resolve a VLM run root by probing for this benchmark's artifacts.

    Do NOT replace this with ``_inspector_common.resolve_root``. That one
    descends into a ``0/`` subdirectory when it finds one. This one must also
    look for a ``vlm_geoprivacy_bench/`` level, and it picks the candidate
    that actually HOLDS an MCQ or free-form parse file. The four layouts come
    from running this benchmark alone or under eval_all, each with or without
    a Hydra multirun wrapper.

    The name differs from the shared helper on purpose, so a reader does not
    read one as a stale copy of the other. They were reconciled 2026-08-12;
    this difference survived because it is real.
    """
    p = Path(run_path)
    for candidate in (p, p / "0", p / "vlm_geoprivacy_bench", p / "0" / "vlm_geoprivacy_bench"):
        if (candidate / MCQ_REL).is_file() or (candidate / FREEFORM_PARSE_REL).is_file():
            return candidate
    # Fallback: return as-is; the caller errors out if nothing is found.
    return p


def find_stage_parquet(root: Path, relpath: str) -> Path | None:
    p = root / relpath
    return p if p.is_file() else None


# ── Image bank ──────────────────────────────────────────────────────────

def image_to_data_uri(path: str, max_size: int, quality: int) -> str | None:
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=quality, optimize=True)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            return f"data:image/jpeg;base64,{b64}"
    except Exception as e:
        print(f"[warn] failed to load {path}: {e}", file=sys.stderr)
        return None


def build_image_bank(
    dataframes: list[pd.DataFrame],
    max_size: int,
    quality: int,
) -> dict[str, str]:
    """Build {numeric_id: data_uri} shared across models + stages.

    Each unique image is encoded exactly once regardless of how many models
    or modes reference it, so the output HTML size scales with the dataset,
    not with the model count.
    """
    seen: dict[str, str] = {}
    total_bytes = 0
    for df in dataframes:
        if df is None or len(df) == 0 or "numeric_id" not in df.columns:
            continue
        for nid, path in zip(df["numeric_id"], df["image_path"]):
            key = str(nid)
            if key in seen:
                continue
            uri = image_to_data_uri(str(path), max_size=max_size, quality=quality)
            if uri is not None:
                seen[key] = uri
                total_bytes += len(uri)
    print(
        f"[images] encoded {len(seen)} unique images, "
        f"~{total_bytes / (1024 * 1024):.1f} MB base64 payload"
    )
    return seen


# ── Row builders ────────────────────────────────────────────────────────

def parse_coord(val: Any) -> tuple[float, float] | None:
    if val is None:
        return None
    # Handle numpy arrays / lists / tuples / string reprs
    if hasattr(val, "tolist"):
        val = val.tolist()
    if isinstance(val, str):
        try:
            val = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            return None
    try:
        lat = float(val[0])
        lon = float(val[1])
    except (TypeError, ValueError, IndexError, KeyError):
        return None
    if lat != lat or lon != lon:  # NaN check
        return None
    return (lat, lon)


def _first_letter(val: Any) -> str:
    if val is None:
        return ""
    s = str(val).strip()
    if not s:
        return ""
    c = s[0].upper()
    return c if c.isalpha() else ""


def _label_eq(pred: str, gt: str) -> bool:
    p = _first_letter(pred)
    g = _first_letter(gt)
    return bool(p) and bool(g) and p == g


def _q7_ordinal_error(pred: str, gt: str) -> int | None:
    """Return pred-gt on the ordinal {A=0, B=1, C=2} scale, or None if invalid."""
    p = _first_letter(pred)
    g = _first_letter(gt)
    if p in Q7_ORDER and g in Q7_ORDER:
        return Q7_ORDER[p] - Q7_ORDER[g]
    return None


def _sharing_intent(val: Any) -> int | None:
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def build_mcq_rows(label_to_df: dict[str, pd.DataFrame]) -> list[dict]:
    labels = list(label_to_df.keys())
    ref_df = label_to_df[labels[0]]
    n = min(len(df) for df in label_to_df.values())

    rows: list[dict] = []
    for i in range(n):
        ref = ref_df.iloc[i]
        gt = [str(ref.get(f"Q{q}_true", "")) for q in range(1, NUM_QUESTIONS + 1)]

        row: dict[str, Any] = {
            "idx": i,
            "image_id": str(ref.get("image_id", "")),
            "numeric_id": str(ref.get("numeric_id", "")),
            "image_source": str(ref.get("image_source", "")) if ref.get("image_source") is not None else "",
            "sharing_intent": _sharing_intent(ref.get("sharing_intent")),
            "coord": parse_coord(ref.get("true_coordinate")),
            "gt": gt,
        }

        preds: dict[str, list[str]] = {}
        raws: dict[str, str] = {}
        correct: dict[str, list[bool]] = {}
        q7_errors: dict[str, int | None] = {}

        for label in labels:
            r = label_to_df[label].iloc[i]
            p_list = [str(r.get(f"Q{q}_pred", "")) for q in range(1, NUM_QUESTIONS + 1)]
            preds[label] = p_list
            raws[label] = str(r.get("generated_text", "") or "")
            correct[label] = [_label_eq(p, g) for p, g in zip(p_list, gt)]
            q7_errors[label] = _q7_ordinal_error(p_list[6], gt[6])

        row["preds"] = preds
        row["raw"] = raws
        row["correct"] = correct
        row["q7err"] = q7_errors

        # Search blob (lowercased, joined once)
        search_parts = [row["image_id"], row["image_source"]]
        search_parts.extend(raws.values())
        row["_search"] = "\n".join(p for p in search_parts if p).lower()

        rows.append(row)
    return rows


def build_freeform_rows(label_to_df: dict[str, pd.DataFrame]) -> list[dict]:
    """Freeform rows: only Q7 matters.

    The raw generated_text is the full location guess; Q7_pred is the
    granularity judge's verdict (A/B/C/D).  We still surface Q7_true from the
    annotation metadata so correctness is computable.
    """
    labels = list(label_to_df.keys())
    ref_df = label_to_df[labels[0]]
    n = min(len(df) for df in label_to_df.values())

    rows: list[dict] = []
    for i in range(n):
        ref = ref_df.iloc[i]
        q7_true = str(ref.get("Q7_true", ""))

        row: dict[str, Any] = {
            "idx": i,
            "image_id": str(ref.get("image_id", "")),
            "numeric_id": str(ref.get("numeric_id", "")),
            "image_source": str(ref.get("image_source", "")) if ref.get("image_source") is not None else "",
            "sharing_intent": _sharing_intent(ref.get("sharing_intent")),
            "coord": parse_coord(ref.get("true_coordinate")),
            "gt_q7": q7_true,
        }

        raws: dict[str, str] = {}
        preds_q7: dict[str, str] = {}
        correct_q7: dict[str, bool] = {}
        q7_errors: dict[str, int | None] = {}

        for label in labels:
            r = label_to_df[label].iloc[i]
            # Prefer Q7_gen (raw free-form text) if available, else generated_text
            q7_gen = r.get("Q7_gen")
            if q7_gen is None or (isinstance(q7_gen, float) and pd.isna(q7_gen)):
                q7_gen = r.get("generated_text", "")
            raws[label] = str(q7_gen or "")
            q7_pred = str(r.get("Q7_pred", ""))
            preds_q7[label] = q7_pred
            correct_q7[label] = _label_eq(q7_pred, q7_true)
            q7_errors[label] = _q7_ordinal_error(q7_pred, q7_true)

        row["raw"] = raws
        row["preds_q7"] = preds_q7
        row["correct_q7"] = correct_q7
        row["q7err"] = q7_errors

        search_parts = [row["image_id"], row["image_source"]]
        search_parts.extend(raws.values())
        row["_search"] = "\n".join(p for p in search_parts if p).lower()

        rows.append(row)
    return rows


# ── Question registry for the HTML side ─────────────────────────────────

def serialize_questions() -> list[dict]:
    """Convert QUESTION_DATA into a minimal JSON shape for the HTML template."""
    out = []
    for i, (qtext, options, heuristics) in enumerate(QUESTION_DATA, start=1):
        out.append({
            "n": i,
            "text": qtext,
            "options": list(options),
            "heuristics": heuristics,
        })
    return out


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Static HTML inspector for vlm_geoprivacy_bench runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--runs", nargs="+", required=True,
        help='Run specifications: "Label=/path/to/run" or just "/path/to/run"',
    )
    parser.add_argument(
        "-o", "--output", default="vlm_geoprivacy_inspector.html",
        help="Output HTML file path",
    )
    parser.add_argument(
        "--image-size", type=int, default=640,
        help="Max image side length in pixels (thumbnails are resized keeping aspect)",
    )
    parser.add_argument(
        "--image-quality", type=int, default=75,
        help="JPEG quality (1-95) for embedded thumbnails",
    )
    parser.add_argument(
        "--map-zoom", type=int, default=11,
        help="Google Maps zoom level for situational context (lower = wider view)",
    )
    parser.add_argument("--no-mcq", action="store_true", help="Skip MCQ stage discovery")
    parser.add_argument("--no-freeform", action="store_true", help="Skip freeform stage discovery")
    args = parser.parse_args()

    # Parse runs
    runs: dict[str, Path] = {}
    for r in args.runs:
        label, path = parse_run_arg(r)
        root = resolve_vlm_root(path)
        runs[label] = root
        print(f"[run] {label}: {root}")

    labels = list(runs.keys())

    # Per-stage discovery
    mcq_dfs: dict[str, pd.DataFrame] = {}
    freeform_dfs: dict[str, pd.DataFrame] = {}

    if not args.no_mcq:
        for label, root in runs.items():
            pq = find_stage_parquet(root, MCQ_REL)
            if pq:
                mcq_dfs[label] = pd.read_parquet(pq)
                print(f"  {label} MCQ: {len(mcq_dfs[label])} rows  ({pq})")
            else:
                print(f"  {label} MCQ: (missing)")

    if not args.no_freeform:
        for label, root in runs.items():
            pq = find_stage_parquet(root, FREEFORM_JUDGE_REL)
            if pq is None:
                pq = find_stage_parquet(root, FREEFORM_PARSE_REL)
            if pq:
                freeform_dfs[label] = pd.read_parquet(pq)
                print(f"  {label} freeform: {len(freeform_dfs[label])} rows  ({pq})")
            else:
                print(f"  {label} freeform: (missing)")

    if not mcq_dfs and not freeform_dfs:
        print("ERROR: no MCQ or freeform stages found in any run.", file=sys.stderr)
        sys.exit(1)

    # Build shared image bank from the union of all dataframes
    all_dfs: list[pd.DataFrame] = list(mcq_dfs.values()) + list(freeform_dfs.values())
    image_bank = build_image_bank(
        all_dfs, max_size=args.image_size, quality=args.image_quality
    )

    # Build per-stage row arrays
    stages: dict[str, dict[str, Any]] = {}
    if mcq_dfs:
        stages["mcq"] = {
            "mode": "mcq",
            "labels": list(mcq_dfs.keys()),
            "rows": build_mcq_rows(mcq_dfs),
        }
        print(f"[mcq] built {len(stages['mcq']['rows'])} rows for "
              f"{len(stages['mcq']['labels'])} models")
    if freeform_dfs:
        stages["freeform"] = {
            "mode": "freeform",
            "labels": list(freeform_dfs.keys()),
            "rows": build_freeform_rows(freeform_dfs),
        }
        print(f"[freeform] built {len(stages['freeform']['rows'])} rows for "
              f"{len(stages['freeform']['labels'])} models")

    questions = serialize_questions()

    # Render HTML
    payload = {
        "stages": stages,
        "questions": questions,
        "labels": labels,
        "images": image_bank,
        "mapZoom": int(args.map_zoom),
    }
    data_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)

    html = HTML_TEMPLATE.replace("__PAYLOAD_PLACEHOLDER__", data_json)
    out_path = Path(args.output)
    out_path.write_text(html, encoding="utf-8")
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\n[done] wrote {out_path.resolve()}  ({size_mb:.1f} MB)")
    print(f"open file://{out_path.resolve()}")


# ── HTML template ───────────────────────────────────────────────────────

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VLM-GeoPrivacy Inspector</title>
<style>
:root {
  --bg: #fafafa; --fg: #1a1a1a; --border: #e0e0e0;
  --accent: #1976d2; --accent-light: #e3f2fd;
  --green: #2e7d32; --green-bg: #e8f5e9;
  --red: #c62828; --red-bg: #ffebee;
  --orange: #ef6c00; --orange-bg: #fff3e0;
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
.topbar select { max-width: 240px; }
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
  border-radius: 4px; background: #fff; max-width: 220px;
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
.container { max-width: 1500px; margin: 0 auto; padding: 12px 16px; }

.row-card {
  background: #fff; border: 1px solid var(--border); border-radius: 8px;
  margin-bottom: 16px; overflow: hidden; transition: border-color 0.15s;
}
.row-card.current { border-color: var(--accent); box-shadow: 0 0 0 2px var(--accent-light); }
.row-card.bookmarked { border-left: 4px solid var(--orange); }

.row-header {
  display: flex; align-items: center; gap: 8px;
  padding: 8px 12px; background: #f5f5f5; border-bottom: 1px solid var(--border);
  cursor: pointer; user-select: none;
}
.row-header:hover { background: #eeeeee; }
.row-num { font-weight: 700; font-size: 13px; color: var(--accent); min-width: 48px; }
.row-hdr-id { font-size: 12px; color: #444; font-family: var(--mono); }
.row-hdr-source { font-size: 11px; color: #666; padding: 1px 8px; background: #e0e0e0; border-radius: 10px; }
.row-hdr-spacer { flex: 1; }
.row-badges { display: flex; gap: 4px; }
.badge {
  font-size: 11px; padding: 2px 8px; border-radius: 10px; font-weight: 600;
}
.badge.correct { background: var(--green-bg); color: var(--green); }
.badge.wrong { background: var(--red-bg); color: var(--red); }
.badge.partial { background: var(--orange-bg); color: var(--orange); }
.badge.bookmark-badge { background: var(--orange-bg); color: var(--orange); cursor: pointer; font-size: 13px; }

.row-body { display: none; padding: 14px; }
.row-card.expanded .row-body { display: block; }

/* ── Media strip (image + map + metadata) ────────────────── */
.media-strip {
  display: grid;
  grid-template-columns: minmax(260px, 1fr) minmax(320px, 1.2fr);
  gap: 14px;
  margin-bottom: 14px;
}
@media (max-width: 900px) {
  .media-strip { grid-template-columns: 1fr; }
}
.media-image {
  display: flex; flex-direction: column; gap: 8px;
}
.media-image img {
  width: 100%; height: auto; max-height: 420px; object-fit: contain;
  border: 1px solid var(--border); border-radius: 6px; background: #000;
}
.meta-list {
  font-size: 12px; line-height: 1.6;
}
.meta-list .row-meta { display: flex; gap: 6px; }
.meta-list .label { color: #777; min-width: 110px; }
.meta-list .val { font-family: var(--mono); color: #222; word-break: break-all; }
.meta-list .val a { color: var(--accent); text-decoration: none; }
.meta-list .val a:hover { text-decoration: underline; }
.pill {
  display: inline-block; padding: 1px 8px; border-radius: 10px;
  font-size: 11px; font-weight: 600;
}
.pill.yes { background: var(--green-bg); color: var(--green); }
.pill.no  { background: var(--red-bg); color: var(--red); }
.pill.unk { background: #eee; color: #666; }

.media-map { display: flex; flex-direction: column; }
.media-map iframe {
  width: 100%; flex: 1; min-height: 300px;
  border: 1px solid var(--border); border-radius: 6px;
}
.media-map .map-placeholder {
  width: 100%; min-height: 300px;
  border: 1px dashed var(--border); border-radius: 6px;
  display: flex; align-items: center; justify-content: center;
  color: #999; font-size: 12px; background: #fafafa;
}

/* ── Question grid (MCQ) ─────────────────────────────────── */
.q-table {
  width: 100%; border-collapse: collapse; margin-bottom: 12px;
  font-size: 13px;
}
.q-table th, .q-table td {
  text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--border);
  vertical-align: top;
}
.q-table th {
  background: #fafafa; font-size: 11px; text-transform: uppercase;
  color: #555; font-weight: 600; letter-spacing: 0.03em;
}
.q-table .q-n { font-weight: 700; color: var(--accent); width: 36px; font-family: var(--mono); }
.q-table .q-text-cell { max-width: 540px; }
.q-table .q-text { font-size: 12px; color: #222; line-height: 1.4; }
.q-table .q-text-cell details { margin-top: 4px; font-size: 11px; }
.q-table .q-text-cell details summary {
  cursor: pointer; color: var(--accent); font-weight: 600;
}
.q-table .q-text-cell details .opts { margin-top: 4px; color: #444; }
.q-table .q-text-cell details .opts li { margin: 2px 0 2px 14px; }
.q-table .q-text-cell details .heur {
  margin-top: 6px; padding: 6px 8px; background: #f8f9fa;
  border-left: 3px solid #d0d7de; border-radius: 3px; color: #444;
  font-size: 11px; line-height: 1.5;
}
.label-pill {
  display: inline-flex; align-items: center; justify-content: center;
  min-width: 24px; padding: 2px 6px; border-radius: 10px;
  font-family: var(--mono); font-size: 12px; font-weight: 700;
}
.label-pill.gt { background: #eceff1; color: #263238; border: 1px solid #cfd8dc; }
.label-pill.correct { background: var(--green-bg); color: var(--green); border: 1px solid #a5d6a7; }
.label-pill.wrong   { background: var(--red-bg); color: var(--red); border: 1px solid #ef9a9a; }
.label-pill.na      { background: #f5f5f5; color: #999; border: 1px solid #e0e0e0; }

/* ── Freeform section ────────────────────────────────────── */
.freeform-grid {
  display: grid; gap: 10px; margin-bottom: 12px;
}
.freeform-card {
  border: 1px solid var(--border); border-radius: 6px; overflow: hidden;
  display: flex; flex-direction: column;
}
.freeform-card .ff-header {
  padding: 6px 10px; display: flex; gap: 8px; align-items: center;
  font-size: 12px; border-bottom: 1px solid var(--border); background: #fafafa;
}
.freeform-card .ff-header .ff-label { font-weight: 700; }
.freeform-card pre {
  padding: 10px 12px; margin: 0;
  font-family: var(--mono); font-size: 12px; line-height: 1.5;
  white-space: pre-wrap; word-break: break-word;
  max-height: 260px; overflow-y: auto;
}
.q7-rail { display: flex; gap: 6px; align-items: center; font-size: 12px; margin: 6px 0 8px; }
.q7-rail .lbl { color: #555; font-weight: 600; }

/* ── Raw output panel ────────────────────────────────────── */
.raw-section { margin-top: 4px; }
.raw-section > summary {
  cursor: pointer; font-weight: 600; font-size: 12px;
  color: #555; padding: 6px 0;
}
.raw-grid { display: grid; gap: 8px; margin-top: 4px; }
.raw-col {
  border: 1px solid var(--border); border-radius: 6px; overflow: hidden;
  display: flex; flex-direction: column;
}
.raw-col .raw-header {
  padding: 6px 10px; font-weight: 600; font-size: 12px;
  background: #f5f5f5; border-bottom: 1px solid var(--border);
}
.raw-col pre {
  padding: 10px; margin: 0;
  font-family: var(--mono); font-size: 12px; line-height: 1.5;
  white-space: pre-wrap; word-break: break-word;
  max-height: 260px; overflow-y: auto;
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

mark { background: #fff176; padding: 1px 2px; border-radius: 2px; }
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

<script>
// ── Data (injected by Python) ────────────────────────────────────────
const PAYLOAD = __PAYLOAD_PLACEHOLDER__;
const STAGES = PAYLOAD.stages;
const STAGE_KEYS = Object.keys(STAGES);
const QUESTIONS = PAYLOAD.questions;  // [{n, text, options, heuristics}]
const IMAGES = PAYLOAD.images;        // {numeric_id: data_uri}
const ALL_LABELS = PAYLOAD.labels;
const MAP_ZOOM = PAYLOAD.mapZoom || 11;

const PALETTE = [
  {bg: '#fff3e0', border: '#ffe0b2'},
  {bg: '#e8f5e9', border: '#c8e6c9'},
  {bg: '#e3f2fd', border: '#bbdefb'},
  {bg: '#f3e5f5', border: '#ce93d8'},
  {bg: '#fce4ec', border: '#f48fb1'},
  {bg: '#e0f7fa', border: '#80deea'},
];

// ── State ────────────────────────────────────────────────────────────
let currentStage = STAGE_KEYS[0] || '';
let currentIdx = 0;
let filteredRows = [];
let allRows = [];
let searchQuery = '';
let activeFilter = 'all';
let bookmarks = new Set();
let bookmarksPanelOpen = false;
let fieldFilters = {};

function getStageLabels() {
  const sd = STAGES[currentStage];
  return sd && sd.labels ? sd.labels : ALL_LABELS;
}
function getStageMode() {
  const sd = STAGES[currentStage];
  return sd ? sd.mode : 'mcq';
}

// ── Init ─────────────────────────────────────────────────────────────
function init() {
  const sel = document.getElementById('stage-select');
  STAGE_KEYS.forEach(k => {
    const opt = document.createElement('option');
    opt.value = k;
    opt.textContent = `${k.toUpperCase()} (${STAGES[k].rows.length} rows, ${STAGES[k].labels.length} models)`;
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
  document.addEventListener('keydown', handleKeyboard);

  loadStage(currentStage);
}

function loadStage(key) {
  currentStage = key;
  allRows = STAGES[key].rows;
  bookmarks.clear();
  activeFilter = 'all';
  searchQuery = '';
  fieldFilters = {};
  document.getElementById('search-input').value = '';
  buildFilterChips();
  buildFieldFilters();
  applyFilters();
}

// ── Filter chips ─────────────────────────────────────────────────────
function buildFilterChips() {
  const container = document.getElementById('filter-chips');
  container.innerHTML = '';
  const mode = getStageMode();
  const filters = [
    ['all', 'All'],
    ['any_wrong', 'Any wrong'],
    ['q7_wrong', 'Q7 wrong'],
    ['q7_over',  'Q7 over-disclose'],
    ['q7_under', 'Q7 under-disclose'],
    ['abstention_violation', 'Abstention violation'],
  ];
  if (getStageLabels().length >= 2) {
    filters.push(['q7_disagree', 'Q7 disagreement']);
  }
  filters.push(['bookmarked', 'Bookmarked']);

  filters.forEach(([id, label]) => {
    const chip = document.createElement('span');
    chip.className = 'chip' + (id === activeFilter ? ' active' : '');
    chip.textContent = label;
    chip.dataset.filter = id;
    chip.addEventListener('click', () => { activeFilter = id; applyFilters(); });
    container.appendChild(chip);
  });
}

function _getQ7GT(row) {
  // MCQ: row.gt is [Q1..Q7]; freeform: row.gt_q7
  if (row.gt) return (row.gt[6] || '').toString().trim().charAt(0).toUpperCase();
  return (row.gt_q7 || '').toString().trim().charAt(0).toUpperCase();
}

function _getQ7Pred(row, label) {
  if (row.preds) return (row.preds[label]?.[6] || '').toString().trim().charAt(0).toUpperCase();
  return (row.preds_q7?.[label] || '').toString().trim().charAt(0).toUpperCase();
}

function _rowCorrectArray(row, label) {
  // MCQ: returns per-question correctness (length NUM_QUESTIONS).
  // Freeform: returns a sparse array where only Q7 is defined so counts of
  // applicable questions stay at 1, not 7.
  if (row.correct) return row.correct[label] || [];
  const c = row.correct_q7?.[label];
  if (c === undefined) return [];
  const arr = new Array(7);
  arr[6] = c;
  return arr;
}

function matchesChipFilter(row) {
  if (activeFilter === 'all') return true;
  if (activeFilter === 'bookmarked') return bookmarks.has(row.idx);
  const labels = getStageLabels();
  const q7gt = _getQ7GT(row);

  switch (activeFilter) {
    case 'any_wrong': {
      for (const l of labels) {
        const arr = _rowCorrectArray(row, l);
        if (arr.some(v => v === false)) return true;
      }
      return false;
    }
    case 'q7_wrong': {
      for (const l of labels) {
        const arr = _rowCorrectArray(row, l);
        if (arr[6] === false) return true;
      }
      return false;
    }
    case 'q7_over': {
      const q7e = row.q7err || {};
      return labels.some(l => typeof q7e[l] === 'number' && q7e[l] > 0);
    }
    case 'q7_under': {
      const q7e = row.q7err || {};
      return labels.some(l => typeof q7e[l] === 'number' && q7e[l] < 0);
    }
    case 'abstention_violation': {
      if (q7gt !== 'A') return false;
      return labels.some(l => {
        const p = _getQ7Pred(row, l);
        return p && p !== 'A';
      });
    }
    case 'q7_disagree': {
      const vals = labels.map(l => _getQ7Pred(row, l)).filter(v => v);
      return new Set(vals).size > 1;
    }
    default: return true;
  }
}

function matchesSearch(row) {
  if (!searchQuery) return true;
  const hay = row._search || '';
  try { return new RegExp(searchQuery, 'i').test(hay); }
  catch (e) { return true; }
}

// ── Field filters ────────────────────────────────────────────────────
const FIELD_FILTER_DEFS = [
  {key: 'image_source', label: 'source', get: r => r.image_source || ''},
  {key: 'sharing_intent', label: 'sharing_intent',
   get: r => r.sharing_intent === 1 ? 'yes' : r.sharing_intent === 0 ? 'no' : 'unknown'},
  {key: 'q7_true', label: 'Q7 true', get: _getQ7GT},
];

function _collectFacets() {
  const MAX = 40;
  const out = [];
  for (const def of FIELD_FILTER_DEFS) {
    const values = new Set();
    for (const row of allRows) {
      const v = def.get(row);
      if (v) values.add(v);
    }
    if (values.size >= 2 && values.size <= MAX) {
      out.push({def, values: [...values].sort()});
    }
  }
  return out;
}

function buildFieldFilters() {
  const container = document.getElementById('field-filters');
  container.innerHTML = '';
  fieldFilters = {};
  const facets = _collectFacets();
  if (!facets.length) return;

  facets.forEach(({def, values}) => {
    const wrap = document.createElement('span');
    wrap.className = 'field-filter';
    const lbl = document.createElement('label');
    lbl.textContent = def.label + ':';
    wrap.appendChild(lbl);
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
    sel.addEventListener('change', () => {
      if (sel.value) {
        fieldFilters[def.key] = sel.value;
        sel.classList.add('active-filter');
      } else {
        delete fieldFilters[def.key];
        sel.classList.remove('active-filter');
      }
      refilter();
    });
    wrap.appendChild(sel);
    container.appendChild(wrap);
  });

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

function matchesFieldFilters(row) {
  for (const [key, expected] of Object.entries(fieldFilters)) {
    const def = FIELD_FILTER_DEFS.find(d => d.key === key);
    if (!def) continue;
    if (def.get(row) !== expected) return false;
  }
  return true;
}

function refilter() {
  filteredRows = allRows.filter(r =>
    matchesChipFilter(r) && matchesSearch(r) && matchesFieldFilters(r)
  );
  currentIdx = 0;
  render('header');
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

// ── Navigation ───────────────────────────────────────────────────────
function doJump() {
  const input = document.getElementById('jump-input');
  const target = parseInt(input.value, 10);
  if (isNaN(target)) return;
  const fi = filteredRows.findIndex(r => r.idx === target);
  if (fi >= 0) { currentIdx = fi; render('center'); }
  else {
    const exists = allRows.some(r => r.idx === target);
    if (exists) {
      activeFilter = 'all';
      fieldFilters = {};
      searchQuery = '';
      document.getElementById('search-input').value = '';
      document.getElementById('field-filters').querySelectorAll('select').forEach(s => {
        s.value = ''; s.classList.remove('active-filter');
      });
      filteredRows = allRows.slice();
      updateStatus();
      const fi2 = filteredRows.findIndex(r => r.idx === target);
      if (fi2 >= 0) { currentIdx = fi2; render('center'); }
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
  // Virtualize: render a window around the current row
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

  const idEl = document.createElement('span');
  idEl.className = 'row-hdr-id';
  idEl.textContent = row.image_id;
  header.appendChild(idEl);

  if (row.image_source) {
    const src = document.createElement('span');
    src.className = 'row-hdr-source';
    src.textContent = row.image_source;
    header.appendChild(src);
  }

  const spacer = document.createElement('span');
  spacer.className = 'row-hdr-spacer';
  header.appendChild(spacer);

  // Per-model Q7 correctness badges
  const badges = document.createElement('span');
  badges.className = 'row-badges';
  const labels = getStageLabels();
  labels.forEach(l => {
    const arr = _rowCorrectArray(row, l);
    const q7 = arr[6];
    if (q7 === undefined) return;
    // Show total correct out of applicable Qs
    const total = arr.filter(v => v !== undefined).length;
    const nCorr = arr.filter(v => v === true).length;
    const allCorr = nCorr === total && total > 0;
    const cls = q7 === true ? 'correct' : q7 === false ? 'wrong' : 'partial';
    const b = document.createElement('span');
    b.className = 'badge ' + cls;
    const suffix = total > 1 ? ` (${nCorr}/${total})` : '';
    b.textContent = `${l} Q7:${q7 ? '✓' : '✗'}${suffix}`;
    badges.appendChild(b);
  });

  const bmBadge = document.createElement('span');
  bmBadge.className = 'badge bookmark-badge';
  bmBadge.textContent = bookmarks.has(row.idx) ? '★' : '☆';
  bmBadge.title = 'Toggle bookmark (b)';
  bmBadge.addEventListener('click', e => {
    e.stopPropagation();
    toggleBookmark(row.idx);
  });
  badges.appendChild(bmBadge);
  header.appendChild(badges);
  card.appendChild(header);

  // Body — only render when expanded
  if (filterIdx === currentIdx) {
    const body = document.createElement('div');
    body.className = 'row-body';
    body.innerHTML = buildRowBody(row);
    card.appendChild(body);
    // Lazily instantiate the map iframe after the DOM is attached
    requestAnimationFrame(() => mountMap(body, row));
  }

  return card;
}

function mountMap(bodyEl, row) {
  const holder = bodyEl.querySelector('.map-placeholder');
  if (!holder || !row.coord) return;
  const [lat, lon] = row.coord;
  const url = `https://maps.google.com/maps?q=${lat},${lon}&z=${MAP_ZOOM}&output=embed&hl=en`;
  const iframe = document.createElement('iframe');
  iframe.src = url;
  iframe.loading = 'lazy';
  iframe.referrerPolicy = 'no-referrer-when-downgrade';
  iframe.allowFullscreen = false;
  holder.replaceWith(iframe);
}

function buildRowBody(row) {
  const mode = getStageMode();
  let html = '';

  // Media strip: image + map + metadata
  const imgUri = IMAGES[row.numeric_id];
  const coord = row.coord;
  const hasCoord = Array.isArray(coord) && coord.length === 2;
  const lat = hasCoord ? coord[0] : null;
  const lon = hasCoord ? coord[1] : null;
  const mapsLink = hasCoord
    ? `https://www.google.com/maps/place/${lat},${lon}/@${lat},${lon},${MAP_ZOOM}z`
    : null;

  html += '<div class="media-strip">';
  html += '<div class="media-image">';
  if (imgUri) {
    html += `<img src="${imgUri}" alt="${esc(row.image_id)}">`;
  } else {
    html += '<div class="map-placeholder">(image not found)</div>';
  }
  html += '<div class="meta-list">';
  html += _metaRow('image_id', esc(row.image_id));
  if (row.image_source) html += _metaRow('source', esc(row.image_source));
  if (row.sharing_intent === 1) html += _metaRow('sharing_intent', '<span class="pill yes">yes</span>');
  else if (row.sharing_intent === 0) html += _metaRow('sharing_intent', '<span class="pill no">no</span>');
  else html += _metaRow('sharing_intent', '<span class="pill unk">unknown</span>');
  if (hasCoord) {
    const coordText = `${lat.toFixed(5)}, ${lon.toFixed(5)}`;
    html += _metaRow('coordinate',
      `<a href="${mapsLink}" target="_blank" rel="noopener">${coordText} ↗</a>`);
  }
  html += '</div></div>';  // end media-image

  html += '<div class="media-map">';
  if (hasCoord) {
    html += '<div class="map-placeholder">loading map…</div>';
  } else {
    html += '<div class="map-placeholder">(no coordinate)</div>';
  }
  html += '</div>';
  html += '</div>';  // end media-strip

  // Mode-specific content
  if (mode === 'mcq') {
    html += buildMcqGrid(row);
    html += buildRawOutputPanel(row, 'Raw generated output');
  } else {
    html += buildFreeformSection(row);
  }

  return html;
}

function _metaRow(label, valHtml) {
  return `<div class="row-meta"><span class="label">${esc(label)}</span><span class="val">${valHtml}</span></div>`;
}

function buildMcqGrid(row) {
  const labels = getStageLabels();
  let html = '<table class="q-table"><thead><tr>';
  html += '<th class="q-n">#</th>';
  html += '<th>Question</th>';
  html += '<th>GT</th>';
  labels.forEach(l => { html += `<th>${esc(l)}</th>`; });
  html += '</tr></thead><tbody>';

  QUESTIONS.forEach((q, qi) => {
    const gt = (row.gt[qi] || '').toString().trim();
    html += '<tr>';
    html += `<td class="q-n">Q${q.n}</td>`;
    html += '<td class="q-text-cell">';
    html += `<div class="q-text">${esc(q.text)}</div>`;
    html += '<details><summary>options &amp; heuristics</summary>';
    html += '<ul class="opts">';
    q.options.forEach(o => { html += `<li>${esc(o)}</li>`; });
    html += '</ul>';
    html += `<div class="heur">${esc(q.heuristics)}</div>`;
    html += '</details>';
    html += '</td>';
    html += `<td><span class="label-pill gt">${esc(gt || '—')}</span></td>`;
    labels.forEach(l => {
      const pred = ((row.preds[l] || [])[qi] || '').toString().trim();
      const c = (row.correct[l] || [])[qi];
      const cls = c === true ? 'correct' : c === false ? 'wrong' : 'na';
      html += `<td><span class="label-pill ${cls}">${esc(pred || '—')}</span></td>`;
    });
    html += '</tr>';
  });

  html += '</tbody></table>';
  return html;
}

function buildFreeformSection(row) {
  const labels = getStageLabels();
  let html = '';

  // Show Q7 ground truth + per-model granularity verdict pills
  html += '<div class="q7-rail">';
  html += `<span class="lbl">Q7 true:</span> <span class="label-pill gt">${esc(row.gt_q7 || '—')}</span>`;
  labels.forEach(l => {
    const pred = (row.preds_q7?.[l] || '').toString().trim();
    const c = row.correct_q7?.[l];
    const cls = c === true ? 'correct' : c === false ? 'wrong' : 'na';
    html += `<span class="lbl" style="margin-left:12px;">${esc(l)}:</span> <span class="label-pill ${cls}">${esc(pred || '—')}</span>`;
  });
  html += '</div>';

  // Q7 question reference
  const q7 = QUESTIONS[QUESTIONS.length - 1];
  if (q7) {
    html += '<details style="margin-bottom:10px;font-size:12px;">';
    html += `<summary style="cursor:pointer;color:var(--accent);font-weight:600;">Q7 reference — ${esc(q7.text)}</summary>`;
    html += '<ul style="margin:6px 0 6px 18px;color:#444;">';
    q7.options.forEach(o => { html += `<li>${esc(o)}</li>`; });
    html += '</ul>';
    html += '</details>';
  }

  // Raw free-form responses (this is the primary content for freeform)
  html += '<div class="freeform-grid" style="grid-template-columns: repeat(' + labels.length + ', 1fr);">';
  labels.forEach((l, i) => {
    const p = PALETTE[i % PALETTE.length];
    const pred = (row.preds_q7?.[l] || '').toString().trim();
    const c = row.correct_q7?.[l];
    const cls = c === true ? 'correct' : c === false ? 'wrong' : 'na';
    const text = row.raw?.[l] || '';
    html += '<div class="freeform-card" style="border-color:' + p.border + ';">';
    html += '<div class="ff-header" style="background:' + p.bg + ';">';
    html += `<span class="ff-label">${esc(l)}</span>`;
    html += `<span class="label-pill ${cls}">judge: ${esc(pred || '—')}</span>`;
    html += '</div>';
    html += `<pre>${highlightSearch(esc(text || '(empty)'))}</pre>`;
    html += '</div>';
  });
  html += '</div>';

  return html;
}

function buildRawOutputPanel(row, summaryText) {
  const labels = getStageLabels();
  let html = `<details class="raw-section"><summary>${esc(summaryText)}</summary>`;
  html += '<div class="raw-grid" style="grid-template-columns: repeat(' + labels.length + ', 1fr);">';
  labels.forEach((l, i) => {
    const p = PALETTE[i % PALETTE.length];
    const text = row.raw?.[l] || '';
    html += '<div class="raw-col" style="border-color:' + p.border + ';">';
    html += '<div class="raw-header" style="background:' + p.bg + ';">' + esc(l) + '</div>';
    html += `<pre>${highlightSearch(esc(text || '(empty)'))}</pre>`;
    html += '</div>';
  });
  html += '</div></details>';
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

// ── Bookmarks ────────────────────────────────────────────────────────
function toggleBookmark(idx) {
  if (bookmarks.has(idx)) bookmarks.delete(idx);
  else bookmarks.add(idx);
  if (activeFilter === 'bookmarked') {
    render('none');
  } else {
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
    item.textContent = `#${idx} — ${row.image_id}`;
    item.addEventListener('click', () => {
      const fi = filteredRows.findIndex(r => r.idx === idx);
      if (fi >= 0) { currentIdx = fi; render('center'); }
      else {
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
      if (bookmarksPanelOpen) toggleBookmarks();
      break;
  }
}

init();
</script>
</body>
</html>"""


if __name__ == "__main__":
    main()
