#!/usr/bin/env python3
"""SFT Inspector — static HTML browser for SFT training data and diagnostics.

Reads a grpo_training run (or specifically an `sft_only` pipeline output) and
generates a self-contained HTML file with:

  - "pairs" stage: per-row cards for every sft_pairs.parquet training example
    (prompt + target completion + parsed CI flows + token counts).
  - "tokenization" stage: first-N tokenized examples sampled by the SFT
    trace callback (shows what actually went into the trainer after masking).
  - Training overview panel: loss / lr / grad_norm curves from sft_traces.jsonl
    and the init/final metadata cards.

Usage:
    python -m scripts.sft_inspector --run <path> -o sft_inspector.html

    # Auto-discovers sft_pairs.parquet and sft_traces.jsonl under the root.
    # Pass an explicit path to override discovery:
    python -m scripts.sft_inspector \\
        --sft-pairs /path/to/sft_pairs.parquet \\
        --sft-traces /path/to/sft_traces.jsonl \\
        -o sft_inspector.html

    # Limit rows (useful for large datasets):
    python -m scripts.sft_inspector --run <path> --max-rows 500 -o sft_inspector.html

    # Row subset with Python slice syntax:
    python -m scripts.sft_inspector --run <path> --rows "0:100" -o first100.html
"""

from __future__ import annotations

import argparse
import html as html_lib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# Allow running as `python scripts/sft_inspector.py` in addition to `-m`.
sys.path.insert(0, str(Path(__file__).parent))
from _inspector_common import (  # noqa: E402
    _serialize,
    extract_assistant_content,
    extract_user_prompt,
    parse_row_slice,
    render_page,
    resolve_root,
)


# ── Discovery ───────────────────────────────────────────────────────────


def discover_artifacts(root: Path) -> dict[str, Path]:
    """Find sft_pairs.parquet and sft_traces.jsonl under a run root.

    Searches common layouts:
      - <root>/outputs/sft_data/sft_pairs.parquet
      - <root>/outputs/sft/checkpoint/sft_traces.jsonl
      - <root>/<anypipeline>/outputs/sft_data/sft_pairs.parquet
      - <root>/<anypipeline>/outputs/sft/checkpoint/sft_traces.jsonl

    Returns a dict with optional keys "pairs" and "traces".
    """
    found: dict[str, Path] = {}

    # Search for sft_pairs.parquet
    pairs_candidates = list(root.rglob("sft_data/sft_pairs.parquet"))
    if pairs_candidates:
        # Prefer the shallowest path when multiple exist.
        pairs_candidates.sort(key=lambda p: len(p.parts))
        found["pairs"] = pairs_candidates[0]

    # Search for sft_traces.jsonl
    traces_candidates = list(root.rglob("sft/checkpoint/sft_traces.jsonl"))
    if traces_candidates:
        traces_candidates.sort(key=lambda p: len(p.parts))
        found["traces"] = traces_candidates[0]

    return found


# ── Parsing ─────────────────────────────────────────────────────────────


def _parse_messages(raw: Any) -> list[dict] | None:
    """Parse the `messages` column (may be a JSON string or already a list)."""
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return None
    if isinstance(raw, list):
        return raw
    return None


def _extract_ci_flows(completion_text: str) -> dict | None:
    """Attempt to parse the assistant JSON completion for CI extraction.

    Returns the parsed dict (with reasoning, has_information_exchange, flows)
    or None when parsing fails.
    """
    if not completion_text:
        return None
    # Strip any leading <think>...</think> block
    text = completion_text
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    text = text.strip()
    # Try direct parse first
    try:
        return json.loads(text)
    except Exception:
        pass
    # Find first {...} block
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if esc:
            esc = False
            continue
        if c == "\\":
            esc = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except Exception:
                    return None
    return None


def build_pairs_rows(
    df: pd.DataFrame,
    row_indices: list[int] | None,
    tok_by_idx: dict[int, dict],
) -> list[dict]:
    """Build row dicts for the 'pairs' stage."""
    n = len(df)
    indices = row_indices if row_indices is not None else range(n)

    rows = []
    for idx in indices:
        if idx >= n:
            continue
        row = df.iloc[idx]
        rec: dict[str, Any] = {"idx": int(idx)}

        # Core fields
        rec["source_id"] = _serialize(row.get("source_id")) if "source_id" in row.index else None
        rec["task_type"] = _serialize(row.get("task_type")) if "task_type" in row.index else None

        # Parse messages
        msgs = _parse_messages(row.get("messages"))
        user_prompt = extract_user_prompt(msgs) or ""
        assistant = extract_assistant_content(msgs) or ""
        rec["user_prompt"] = user_prompt
        rec["assistant"] = assistant
        rec["user_len"] = len(user_prompt)
        rec["assistant_len"] = len(assistant)

        # Parse the assistant JSON — extracts has_information_exchange, flows
        parsed = _extract_ci_flows(assistant)
        if parsed is not None:
            rec["parsed"] = _serialize(parsed)
            flows = parsed.get("flows") or parsed.get("extraction") or []
            rec["n_flows"] = len(flows) if isinstance(flows, list) else 0
            has_ex = parsed.get("has_information_exchange")
            if has_ex is None and isinstance(parsed.get("reasoning"), dict):
                has_ex = parsed["reasoning"].get("has_information_exchange")
            rec["has_exchange"] = bool(has_ex) if has_ex is not None else None
        else:
            rec["parsed"] = None
            rec["n_flows"] = 0
            rec["has_exchange"] = None

        # Token counts from the tokenization_sample trace (matched by dataset idx)
        tok = tok_by_idx.get(int(idx))
        if tok:
            rec["tokens_total"] = tok.get("total_tokens")
            rec["tokens_prompt"] = tok.get("prompt_tokens")
            rec["tokens_completion"] = tok.get("completion_tokens")

        # Facets for filtering
        facets = {}
        if rec["source_id"] is not None:
            facets["source_id"] = str(rec["source_id"])
        if rec["task_type"] is not None:
            facets["task_type"] = str(rec["task_type"])
        if rec["has_exchange"] is not None:
            facets["has_exchange"] = "yes" if rec["has_exchange"] else "no"
        nf = rec["n_flows"]
        if nf == 0:
            facets["n_flows"] = "0"
        elif nf == 1:
            facets["n_flows"] = "1"
        elif nf <= 3:
            facets["n_flows"] = "2-3"
        else:
            facets["n_flows"] = "4+"
        if facets:
            rec["_facets"] = facets

        rows.append(rec)

    return rows


def build_tok_rows(tok_samples: list[dict]) -> list[dict]:
    """Build row dicts for the 'tokenization' stage."""
    rows = []
    for sample in tok_samples:
        idx = sample.get("idx")
        if idx is None:
            continue
        rec = {
            "idx": int(idx),
            "total_tokens": sample.get("total_tokens"),
            "prompt_tokens": sample.get("prompt_tokens"),
            "completion_tokens": sample.get("completion_tokens"),
            "prompt_text": sample.get("prompt_text", ""),
            "completion_text": sample.get("completion_text", ""),
        }
        rows.append(rec)
    return rows


# ── Trace parsing ───────────────────────────────────────────────────────


def load_traces(path: Path) -> dict[str, Any]:
    """Parse sft_traces.jsonl into init / steps / tok_samples / final."""
    init: dict | None = None
    final: dict | None = None
    steps: list[dict] = []
    tok_samples: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            t = d.get("type")
            if t == "init":
                init = d
            elif t == "final":
                final = d
            elif t == "step":
                steps.append(d)
            elif t == "tokenization_sample":
                tok_samples.append(d)
    return {
        "init": init,
        "final": final,
        "steps": steps,
        "tok_samples": tok_samples,
    }


# ── Overview panel HTML ─────────────────────────────────────────────────


def _esc(s: Any) -> str:
    return html_lib.escape(str(s))


def _curve_svg(
    steps: list[dict],
    field: str,
    label: str,
    color: str,
    width: int = 680,
    height: int = 130,
) -> str:
    """Render a simple SVG line chart of one step-trace field vs global_step."""
    pts = [(s.get("global_step"), s.get(field)) for s in steps]
    pts = [(x, y) for x, y in pts if x is not None and y is not None and isinstance(y, (int, float))]
    if len(pts) < 2:
        return f'<div class="curve-chart" style="padding:40px;text-align:center;color:#999;font-size:12px;">{_esc(label)}: insufficient data</div>'
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if ymax == ymin:
        ymax = ymin + 1e-9
    if xmax == xmin:
        xmax = xmin + 1
    pad_l, pad_r, pad_t, pad_b = 44, 10, 12, 26
    w, h = width, height

    def xs2(v):
        return pad_l + (v - xmin) / (xmax - xmin) * (w - pad_l - pad_r)

    def ys2(v):
        return h - pad_b - (v - ymin) / (ymax - ymin) * (h - pad_t - pad_b)

    path = " ".join(
        ("M" if i == 0 else "L") + f"{xs2(x):.1f},{ys2(y):.1f}"
        for i, (x, y) in enumerate(pts)
    )
    # Y-axis ticks (min, mid, max)
    y_ticks = [ymin, (ymin + ymax) / 2, ymax]
    tick_html = "".join(
        f'<text x="{pad_l - 4}" y="{ys2(yv) + 3:.1f}" text-anchor="end" font-size="9" fill="#888">{yv:.3g}</text>'
        f'<line x1="{pad_l}" y1="{ys2(yv):.1f}" x2="{w - pad_r}" y2="{ys2(yv):.1f}" stroke="#f0f0f0" stroke-width="1"/>'
        for yv in y_ticks
    )
    # X-axis labels (start, end)
    x_labels = (
        f'<text x="{pad_l}" y="{h - pad_b + 14}" font-size="9" fill="#888">step {xmin}</text>'
        f'<text x="{w - pad_r}" y="{h - pad_b + 14}" text-anchor="end" font-size="9" fill="#888">step {xmax}</text>'
    )
    return (
        f'<svg class="curve-chart" viewBox="0 0 {w} {h}" preserveAspectRatio="xMidYMid meet">'
        f'<text x="{pad_l}" y="10" font-size="11" font-weight="600" fill="#333">{_esc(label)}</text>'
        f'{tick_html}'
        f'<path d="{path}" stroke="{color}" stroke-width="1.5" fill="none"/>'
        f'{x_labels}'
        f'</svg>'
    )


def build_overview_html(
    traces: dict[str, Any],
    pairs_summary: dict[str, Any],
) -> str:
    """Build the overview panel HTML blob."""
    init = traces.get("init") or {}
    final = traces.get("final") or {}
    steps = traces.get("steps") or []

    # Stats grid from sft_pairs
    stats_rows = [
        ("Total pairs", f"{pairs_summary.get('n_total', 0):,}"),
        ("Sources", str(pairs_summary.get("n_sources", 0))),
        ("With exchange", f"{pairs_summary.get('n_with_exchange', 0):,}"),
        ("No exchange", f"{pairs_summary.get('n_no_exchange', 0):,}"),
        ("Mean flows / pair", f"{pairs_summary.get('mean_flows', 0):.2f}"),
    ]
    if init:
        stats_rows += [
            ("Base model", _short_model(init.get("base_model", "?"))),
            ("Dataset size", f"{init.get('dataset_size', '?')}"),
            ("LoRA rank", f"{init.get('lora_rank', '?')}"),
            ("LoRA alpha", f"{init.get('lora_alpha', '?')}"),
            ("Trainable %", f"{init.get('trainable_pct', '?')}%"),
            ("Epochs", f"{init.get('num_epochs', '?')}"),
            ("Batch size", f"{init.get('per_device_batch_size', '?')}"),
            ("Grad accum", f"{init.get('gradient_accumulation_steps', '?')}"),
            ("Learning rate", f"{init.get('learning_rate', '?')}"),
            ("Max seq len", f"{init.get('max_seq_length', '?')}"),
        ]
    if final:
        wall = final.get("total_wall_seconds")
        if isinstance(wall, (int, float)):
            mins = wall / 60
            stats_rows.append(("Wall time", f"{mins:.1f} min"))
        stats_rows.append(("Final step", str(final.get("global_step", "?"))))

    stats_html = ""
    for label, val in stats_rows:
        stats_html += f'<div class="overview-stat"><div class="stat-label">{_esc(label)}</div><div class="stat-value">{_esc(val)}</div></div>'

    # Curves
    curves_html = ""
    if steps:
        curves_html += '<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:10px;">'
        curves_html += _curve_svg(steps, "loss", "Loss", "#1976d2")
        curves_html += _curve_svg(steps, "learning_rate", "Learning rate", "#2e7d32")
        curves_html += _curve_svg(steps, "grad_norm", "Grad norm", "#e65100")
        curves_html += "</div>"
    else:
        curves_html = '<p style="color:#999;font-size:12px;margin-top:8px;">No step traces found.</p>'

    return (
        '<div class="overview-panel">'
        '<h2>SFT Training Overview</h2>'
        f'<div class="overview-grid">{stats_html}</div>'
        f'{curves_html}'
        '</div>'
    )


def _short_model(name: str) -> str:
    """Trim a model path to its last 2 segments for display."""
    if not name:
        return ""
    parts = str(name).rstrip("/").split("/")
    return "/".join(parts[-2:]) if len(parts) > 1 else parts[0]


# ── SFT-specific JS (row body renderer) ────────────────────────────────

SFT_BODY_JS = r"""
HOOKS.getRowBadges = function(row, stageData) {
  const badges = [];
  if (stageData.stage_type === 'pairs') {
    if (row.task_type) badges.push({text: row.task_type, cls: 'accent'});
    if (row.has_exchange === true) badges.push({text: 'exchange', cls: 'ok'});
    else if (row.has_exchange === false) badges.push({text: 'no exchange', cls: 'warn'});
    if (typeof row.n_flows === 'number') badges.push({text: `${row.n_flows} flow${row.n_flows !== 1 ? 's' : ''}`, cls: 'info'});
    if (row.tokens_total) badges.push({text: `${row.tokens_total} tok`, cls: 'neutral'});
  } else if (stageData.stage_type === 'tokenization') {
    badges.push({text: `${row.total_tokens} tok`, cls: 'info'});
    badges.push({text: `p:${row.prompt_tokens} / c:${row.completion_tokens}`, cls: 'neutral'});
  }
  return badges;
};

HOOKS.getRowIdLine = function(row, stageData) {
  const parts = [];
  if (stageData.stage_type === 'pairs') {
    if (row.source_id != null) parts.push(`<b>source</b>=${esc(row.source_id)}`);
    if (row.user_len != null) parts.push(`<b>prompt</b>=${row.user_len}ch`);
    if (row.assistant_len != null) parts.push(`<b>target</b>=${row.assistant_len}ch`);
  } else {
    parts.push(`<b>sample</b>=${row.idx}`);
  }
  return parts.join('&ensp;');
};

HOOKS.buildRowBody = function(row, stageData) {
  if (stageData.stage_type === 'pairs') return buildPairsBody(row);
  if (stageData.stage_type === 'tokenization') return buildTokBody(row);
  return '<pre>' + esc(JSON.stringify(row, null, 2)) + '</pre>';
};

function buildPairsBody(row) {
  let html = '';

  // User prompt (the CI instruction + article text)
  if (row.user_prompt) {
    const isLong = row.user_prompt.length > 500;
    html += `<div class="text-box user" data-export="user-prompt">
      <details${isLong ? '' : ' open'}>
        <summary>User prompt (${row.user_prompt.length.toLocaleString()} chars${row.tokens_prompt ? ', ' + row.tokens_prompt + ' tokens' : ''})</summary>
        <pre>${highlightSearch(esc(row.user_prompt))}</pre>
      </details>
    </div>`;
  }

  // Target completion (assistant JSON)
  if (row.assistant) {
    html += `<div class="text-box assistant" data-export="target">
      <details open>
        <summary>Target completion (${row.assistant.length.toLocaleString()} chars${row.tokens_completion ? ', ' + row.tokens_completion + ' tokens' : ''})</summary>
        <pre>${highlightSearch(formatTextWithJson(row.assistant))}</pre>
      </details>
    </div>`;
  }

  // Parsed CI flows table
  if (row.parsed && row.parsed.flows && Array.isArray(row.parsed.flows) && row.parsed.flows.length > 0) {
    html += '<div class="field-group open" data-export="flows">';
    html += `<div class="field-group-header">Extracted flows (${row.parsed.flows.length})</div>`;
    html += '<div class="field-group-body">';
    row.parsed.flows.forEach((flow, i) => {
      html += buildFlowCard(flow, i);
    });
    html += '</div></div>';
  } else if (row.parsed && row.parsed.has_information_exchange === false) {
    html += '<div class="field-group open"><div class="field-group-header">No information exchange (negative example)</div></div>';
  }

  // Reasoning (if present as a separate field)
  if (row.parsed && row.parsed.reasoning && typeof row.parsed.reasoning === 'string') {
    html += `<div class="text-box" data-export="reasoning">
      <details><summary>Reasoning trace</summary>
      <pre>${highlightSearch(esc(row.parsed.reasoning))}</pre></details>
    </div>`;
  }

  return html;
}

function buildFlowCard(flow, i) {
  const ciFields = [
    ['Sender', 'sender'],
    ['Recipient', 'recipient'],
    ['Subject', 'subject'],
    ['Info Type', 'information_type'],
    ['Transmission', 'transmission_principle'],
  ];
  let html = `<div style="border-bottom:1px solid #f0f0f0;padding:8px 12px;">`;
  html += `<div style="font-size:11px;color:#888;margin-bottom:6px;">Flow #${i}</div>`;
  html += '<div class="ci-tuple-card">';
  ciFields.forEach(([label, key]) => {
    const val = flow[key];
    html += `<div class="ci-tuple-cell"><div class="cell-label">${esc(label)}</div><div class="cell-value">${esc(val != null ? String(val) : '—')}</div></div>`;
  });
  html += '</div>';
  // Additional fields. Any of these may be absent under SFT pair-format ablations
  // (training/sft={no_context,no_appropriateness,no_norms_meta,no_confidence,minimal_tuple});
  // the `flow[k] != null` guard below silently skips missing keys.
  const extra = ['context', 'appropriateness', 'norms_invoked', 'norm_source', 'is_new_flow', 'confidence'];
  const rows = [];
  extra.forEach(k => {
    if (flow[k] != null && flow[k] !== '') {
      const val = typeof flow[k] === 'object' ? JSON.stringify(flow[k]) : String(flow[k]);
      rows.push(`<div class="field-row"><div class="field-label">${esc(k)}</div><div class="field-value">${highlightSearch(esc(val))}</div></div>`);
    }
  });
  if (rows.length > 0) {
    html += rows.join('');
  }
  html += '</div>';
  return html;
}

function buildTokBody(row) {
  let html = '';
  if (row.prompt_text) {
    html += `<div class="text-box user" data-export="prompt">
      <details open><summary>Prompt tokens (${row.prompt_tokens})</summary>
      <pre>${highlightSearch(esc(row.prompt_text))}</pre></details>
    </div>`;
  }
  if (row.completion_text) {
    html += `<div class="text-box assistant" data-export="completion">
      <details open><summary>Completion tokens (${row.completion_tokens})</summary>
      <pre>${highlightSearch(formatTextWithJson(row.completion_text))}</pre></details>
    </div>`;
  }
  return html;
}
"""


# ── Main ────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Static HTML inspector for SFT training data and diagnostics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--run", type=str, default=None,
        help="Run root (auto-discovers sft_pairs.parquet and sft_traces.jsonl)",
    )
    parser.add_argument(
        "--sft-pairs", type=str, default=None,
        help="Explicit path to sft_pairs.parquet (overrides --run discovery)",
    )
    parser.add_argument(
        "--sft-traces", type=str, default=None,
        help="Explicit path to sft_traces.jsonl (overrides --run discovery)",
    )
    parser.add_argument(
        "-o", "--output", default="sft_inspector.html",
        help="Output HTML file path (default: sft_inspector.html)",
    )
    parser.add_argument(
        "--max-rows", type=int, default=None,
        help="Maximum rows in the pairs stage (default: all)",
    )
    parser.add_argument(
        "--rows", type=str, default=None, metavar="SPEC",
        help="Row slice spec ('0:100', '::10', '-50:', etc). Overrides --max-rows.",
    )
    args = parser.parse_args()

    if args.rows and args.max_rows:
        parser.error("--rows and --max-rows are mutually exclusive")

    # Resolve artifact paths
    pairs_path = Path(args.sft_pairs) if args.sft_pairs else None
    traces_path = Path(args.sft_traces) if args.sft_traces else None
    if args.run and (not pairs_path or not traces_path):
        # Use the raw --run path for rglob (don't descend into 0/).
        # In grpo_training multiruns the outputs live under a pipeline-named
        # subdir like sft_only/, not in the hydra 0/ sweep subdir.
        root = Path(args.run)
        found = discover_artifacts(root)
        if not pairs_path and "pairs" in found:
            pairs_path = found["pairs"]
        if not traces_path and "traces" in found:
            traces_path = found["traces"]

    if not pairs_path:
        print("ERROR: could not locate sft_pairs.parquet. Use --sft-pairs or --run.", file=sys.stderr)
        sys.exit(1)
    if not pairs_path.exists():
        print(f"ERROR: sft_pairs not found at {pairs_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading sft_pairs: {pairs_path}")
    df = pd.read_parquet(pairs_path)
    n_total = len(df)
    print(f"  {n_total} rows, cols: {list(df.columns)}")

    # Load traces (optional)
    traces: dict[str, Any] = {"init": None, "final": None, "steps": [], "tok_samples": []}
    if traces_path and traces_path.exists():
        print(f"Loading sft_traces: {traces_path}")
        traces = load_traces(traces_path)
        print(f"  init: {'yes' if traces['init'] else 'no'}, "
              f"steps: {len(traces['steps'])}, "
              f"tok_samples: {len(traces['tok_samples'])}, "
              f"final: {'yes' if traces['final'] else 'no'}")
    elif args.run:
        print(f"(no sft_traces.jsonl found)")

    # Pairs stage
    row_indices: list[int] | None = None
    if args.rows:
        try:
            row_indices = parse_row_slice(args.rows, n_total)
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)
    elif args.max_rows:
        row_indices = list(range(min(args.max_rows, n_total)))

    tok_by_idx = {s["idx"]: s for s in traces["tok_samples"] if "idx" in s}
    print(f"Building pairs stage ({len(row_indices) if row_indices else n_total} rows)...")
    pairs_rows = build_pairs_rows(df, row_indices, tok_by_idx)

    # Pairs summary stats
    n_with_ex = sum(1 for r in pairs_rows if r.get("has_exchange") is True)
    n_no_ex = sum(1 for r in pairs_rows if r.get("has_exchange") is False)
    all_flows = [r.get("n_flows", 0) for r in pairs_rows]
    mean_flows = (sum(all_flows) / len(all_flows)) if all_flows else 0
    sources = {r.get("source_id") for r in pairs_rows if r.get("source_id") is not None}
    pairs_summary = {
        "n_total": len(pairs_rows),
        "n_sources": len(sources),
        "n_with_exchange": n_with_ex,
        "n_no_exchange": n_no_ex,
        "mean_flows": mean_flows,
    }
    print(f"  {pairs_summary}")

    # Tokenization stage (from sft_traces.jsonl samples)
    tok_rows = build_tok_rows(traces["tok_samples"])
    print(f"Building tokenization stage ({len(tok_rows)} samples)...")

    # Build stages data
    stages_data: dict[str, dict] = {}
    stages_data["pairs"] = {
        "rows": pairs_rows,
        "id_cols": ["source_id"],
        "n_total": n_total,
        "stage_type": "pairs",
    }
    if tok_rows:
        stages_data["tokenization"] = {
            "rows": tok_rows,
            "id_cols": [],
            "n_total": len(tok_rows),
            "stage_type": "tokenization",
        }

    overview_html = build_overview_html(traces, pairs_summary)

    print("Rendering HTML...")
    html = render_page(
        tool_name="sft",
        title="SFT Inspector",
        accent="blue",
        stages_data=stages_data,
        overview_html=overview_html,
        body_renderer_js=SFT_BODY_JS,
    )

    out = Path(args.output)
    out.write_text(html, encoding="utf-8")
    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"\nOutput: {out.resolve()} ({size_mb:.1f} MB)")
    print(f"Open in browser: file://{out.resolve()}")


if __name__ == "__main__":
    main()
