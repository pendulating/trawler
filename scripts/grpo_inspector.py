#!/usr/bin/env python3
"""GRPO Inspector — static HTML browser for GRPO reward traces.

Reads `reward_traces.jsonl` from a GRPO training run and generates a
self-contained HTML file that shows, for each traced reward call:
  - The prompt that was sampled (when persisted).
  - All rollouts in that call, with per-component reward bars.
  - Completion text (JSON-formatted).
  - R_ground flow-level breakdown: retrieved norms + judge scores for
    both the correct and the contrastive wrong universe.

Usage:
    python -m scripts.grpo_inspector --run <path> -o grpo_inspector.html

    # Explicit trace path:
    python -m scripts.grpo_inspector \\
        --reward-traces /path/to/reward_traces.jsonl \\
        -o grpo_inspector.html

    # Limit to the last N sampled calls (large runs can be huge):
    python -m scripts.grpo_inspector --run <path> --max-calls 200 -o recent.html
"""

from __future__ import annotations

import argparse
import html as html_lib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# Allow running as a script.
sys.path.insert(0, str(Path(__file__).parent))
from _inspector_common import _serialize, parse_row_slice, render_page  # noqa: E402


# ── Discovery ───────────────────────────────────────────────────────────


def discover_reward_traces(root: Path) -> Path | None:
    """Find the latest reward_traces.jsonl under a run root."""
    candidates = list(root.rglob("grpo/checkpoint/reward_traces.jsonl"))
    if not candidates:
        return None
    # Prefer shallowest, then largest (most-recent training).
    candidates.sort(key=lambda p: (len(p.parts), -p.stat().st_size))
    return candidates[0]


def load_traces(path: Path) -> list[dict]:
    """Parse a reward_traces.jsonl file into a list of dict entries."""
    entries: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except Exception:
                continue
    return entries


# ── Grouping by call ────────────────────────────────────────────────────


def group_by_call(entries: list[dict]) -> dict[str, list[list[dict]]]:
    """Group trace entries into call-groups, split by task_type.

    Returns:
        {task_type: [ [rollout0, rollout1, ...],  # call 0
                      [rollout0, rollout1, ...],  # call 1
                      ... ]}
    """
    by_task_call: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for e in entries:
        # Default task_type for old traces without the field
        task_type = e.get("task_type") or "ci_extraction"
        call = e.get("call", -1)
        by_task_call[task_type][call].append(e)

    result: dict[str, list[list[dict]]] = {}
    for task_type, by_call in by_task_call.items():
        calls = []
        for call_num in sorted(by_call.keys()):
            rollouts = sorted(by_call[call_num], key=lambda r: r.get("idx", 0))
            calls.append(rollouts)
        result[task_type] = calls
    return result


# ── Per-call row construction ───────────────────────────────────────────


def build_ci_call_rows(calls: list[list[dict]]) -> list[dict]:
    """Build row dicts for a list of ci_extraction call groups."""
    rows = []
    for i, rollouts in enumerate(calls):
        if not rollouts:
            continue
        head = rollouts[0]
        composites = [r.get("composite") for r in rollouts if r.get("composite") is not None]
        rec: dict[str, Any] = {
            "idx": int(head.get("call", i)),
            "n_rollouts": len(rollouts),
            "source_id": head.get("source_id"),
            "prompt_id": head.get("prompt_id"),
            "is_contrastive": head.get("is_contrastive", False),
            "contrastive_source": head.get("contrastive_source"),
            "gold_has_exchange": head.get("gold_has_exchange"),
            "is_no_flow": head.get("is_no_flow", False),
            "enable_thinking_grpo": head.get("enable_thinking_grpo"),
            "rground_mode": head.get("rground_mode"),
            "prompt": head.get("prompt", ""),
            "composite_mean": (sum(composites) / len(composites)) if composites else 0.0,
            "composite_min": min(composites) if composites else 0.0,
            "composite_max": max(composites) if composites else 0.0,
            "composite_spread": (max(composites) - min(composites)) if composites else 0.0,
        }

        # Rollouts list (serialize each for JSON injection)
        rec["rollouts"] = [_serialize(r) for r in rollouts]

        # Facets
        facets: dict[str, str] = {}
        if rec["source_id"]:
            facets["source_id"] = str(rec["source_id"])
        facets["contrastive"] = "yes" if rec["is_contrastive"] else "no"
        if rec["gold_has_exchange"] is True:
            facets["gold_has_exchange"] = "yes"
        elif rec["gold_has_exchange"] is False:
            facets["gold_has_exchange"] = "no"
        facets["no_flow"] = "yes" if rec["is_no_flow"] else "no"
        if rec["rground_mode"]:
            facets["rground_mode"] = str(rec["rground_mode"])
        # Composite bucket
        m = rec["composite_mean"]
        if m < 0.25:
            facets["composite"] = "<0.25"
        elif m < 0.5:
            facets["composite"] = "0.25-0.5"
        elif m < 0.75:
            facets["composite"] = "0.5-0.75"
        else:
            facets["composite"] = ">=0.75"
        rec["_facets"] = facets

        rows.append(rec)
    return rows


def build_judgment_call_rows(calls: list[list[dict]]) -> list[dict]:
    """Build row dicts for norm_judgment call groups."""
    rows = []
    for i, rollouts in enumerate(calls):
        if not rollouts:
            continue
        head = rollouts[0]
        composites = [r.get("composite") for r in rollouts if r.get("composite") is not None]
        rec: dict[str, Any] = {
            "idx": int(head.get("call", i)),
            "n_rollouts": len(rollouts),
            "source_id": head.get("source_id"),
            "gold_judgment": head.get("gold_judgment"),
            "prompt": head.get("prompt", ""),
            "composite_mean": (sum(composites) / len(composites)) if composites else 0.0,
            "composite_spread": (max(composites) - min(composites)) if composites else 0.0,
        }
        rec["rollouts"] = [_serialize(r) for r in rollouts]
        facets = {}
        if rec["source_id"]:
            facets["source_id"] = str(rec["source_id"])
        if rec["gold_judgment"]:
            facets["gold_judgment"] = str(rec["gold_judgment"])
        rec["_facets"] = facets
        rows.append(rec)
    return rows


# ── Overview / aggregate stats ─────────────────────────────────────────


COMPONENT_NAMES = ["r_uncert", "r_complete", "r_consist", "r_context", "r_cohere", "r_ground"]


def _esc(s: Any) -> str:
    return html_lib.escape(str(s))


def _trend_svg(
    series: list[tuple[int, float]],
    label: str,
    color: str,
    width: int = 680,
    height: int = 130,
) -> str:
    """Render a trend line chart: list of (call_number, value)."""
    pts = [(int(x), float(y)) for x, y in series if x is not None and y is not None]
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
    y_ticks = [ymin, (ymin + ymax) / 2, ymax]
    tick_html = "".join(
        f'<text x="{pad_l - 4}" y="{ys2(yv) + 3:.1f}" text-anchor="end" font-size="9" fill="#888">{yv:.3g}</text>'
        f'<line x1="{pad_l}" y1="{ys2(yv):.1f}" x2="{w - pad_r}" y2="{ys2(yv):.1f}" stroke="#f0f0f0" stroke-width="1"/>'
        for yv in y_ticks
    )
    x_labels = (
        f'<text x="{pad_l}" y="{h - pad_b + 14}" font-size="9" fill="#888">call {xmin}</text>'
        f'<text x="{w - pad_r}" y="{h - pad_b + 14}" text-anchor="end" font-size="9" fill="#888">call {xmax}</text>'
    )
    return (
        f'<svg class="curve-chart" viewBox="0 0 {w} {h}" preserveAspectRatio="xMidYMid meet">'
        f'<text x="{pad_l}" y="10" font-size="11" font-weight="600" fill="#333">{_esc(label)}</text>'
        f'{tick_html}'
        f'<path d="{path}" stroke="{color}" stroke-width="1.5" fill="none"/>'
        f'{x_labels}'
        f'</svg>'
    )


def _histogram_svg(
    values: list[float],
    label: str,
    color: str,
    bins: int = 20,
    width: int = 330,
    height: int = 100,
) -> str:
    """Render a fixed-bin histogram (values in [0, 1])."""
    if not values:
        return f'<div class="curve-chart" style="padding:30px;text-align:center;color:#999;font-size:11px;">{_esc(label)}</div>'
    hist = [0] * bins
    for v in values:
        b = min(bins - 1, max(0, int(v * bins)))
        hist[b] += 1
    maxh = max(hist) or 1
    pad_l, pad_r, pad_t, pad_b = 28, 6, 16, 18
    w, h = width, height
    bar_w = (w - pad_l - pad_r) / bins
    rects = []
    for i, c in enumerate(hist):
        bar_h = (c / maxh) * (h - pad_t - pad_b)
        x = pad_l + i * bar_w
        y = h - pad_b - bar_h
        rects.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w - 0.5:.1f}" height="{bar_h:.1f}" fill="{color}" opacity="0.75"/>')
    axis = (
        f'<text x="{pad_l}" y="10" font-size="10" font-weight="600" fill="#333">{_esc(label)} '
        f'<tspan fill="#888" font-weight="400">(n={len(values)}, μ={sum(values) / len(values):.2f})</tspan></text>'
        f'<text x="{pad_l}" y="{h - 4}" font-size="8" fill="#888">0</text>'
        f'<text x="{w - pad_r}" y="{h - 4}" text-anchor="end" font-size="8" fill="#888">1</text>'
    )
    return (
        f'<svg class="curve-chart" style="height:{h}px;" viewBox="0 0 {w} {h}" preserveAspectRatio="xMidYMid meet">'
        f'{axis}{"".join(rects)}</svg>'
    )


def build_overview_html(
    all_entries: list[dict],
    weights: list[float] | None,
    trace_path: Path,
) -> str:
    """Assemble the top-of-page overview for a GRPO run."""
    n_total = len(all_entries)
    by_task: dict[str, list[dict]] = defaultdict(list)
    for e in all_entries:
        by_task[e.get("task_type") or "ci_extraction"].append(e)

    ci = by_task.get("ci_extraction", [])
    jm = by_task.get("norm_judgment", [])

    composites_ci = [r.get("composite", 0) for r in ci if r.get("composite") is not None]
    composites_jm = [r.get("composite", 0) for r in jm if r.get("composite") is not None]

    # Stats grid
    stats_rows = [
        ("Trace file", trace_path.name),
        ("Total rollouts", f"{n_total:,}"),
        ("CI extraction", f"{len(ci):,}"),
    ]
    if jm:
        stats_rows.append(("Norm judgment", f"{len(jm):,}"))
    if composites_ci:
        stats_rows.append(("Mean composite (CI)", f"{sum(composites_ci) / len(composites_ci):.3f}"))
        stats_rows.append(("Max composite (CI)", f"{max(composites_ci):.3f}"))
    if composites_jm:
        stats_rows.append(("Mean composite (judgment)", f"{sum(composites_jm) / len(composites_jm):.3f}"))

    # Count contrastive / no-flow / online
    n_contrastive = sum(1 for r in ci if r.get("is_contrastive"))
    n_no_flow = sum(1 for r in ci if r.get("is_no_flow"))
    n_online = sum(1 for r in ci if r.get("rground_mode") == "online")
    if ci:
        stats_rows.append(("Contrastive", f"{n_contrastive}/{len(ci)}"))
        stats_rows.append(("No-flow rollouts", f"{n_no_flow}/{len(ci)}"))
        if n_online:
            stats_rows.append(("Online R_ground", f"{n_online}/{len(ci)}"))

    # Unique calls sampled
    unique_calls = {e.get("call") for e in ci}
    if unique_calls:
        unique_calls.discard(None)
        stats_rows.append(("CI calls sampled", str(len(unique_calls))))

    if weights:
        for name, w in zip(COMPONENT_NAMES, weights):
            stats_rows.append((f"w[{name}]", f"{w:.2f}"))

    stats_html = ""
    for label, val in stats_rows:
        stats_html += f'<div class="overview-stat"><div class="stat-label">{_esc(label)}</div><div class="stat-value">{_esc(val)}</div></div>'

    # Trends — one series per component averaged by call number
    trends_html = ""
    if ci:
        # Group by call, mean per component
        by_call: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for e in ci:
            call = e.get("call")
            if call is None:
                continue
            comps = e.get("components") or {}
            for name in COMPONENT_NAMES:
                v = comps.get(name)
                if isinstance(v, (int, float)):
                    by_call[call][name].append(float(v))
            comp = e.get("composite")
            if isinstance(comp, (int, float)):
                by_call[call]["composite"].append(float(comp))

        call_nums = sorted(by_call.keys())
        composite_series = [
            (c, sum(by_call[c]["composite"]) / len(by_call[c]["composite"]))
            for c in call_nums if by_call[c]["composite"]
        ]
        rground_series = [
            (c, sum(by_call[c]["r_ground"]) / len(by_call[c]["r_ground"]))
            for c in call_nums if by_call[c].get("r_ground")
        ]
        rcontext_series = [
            (c, sum(by_call[c]["r_context"]) / len(by_call[c]["r_context"]))
            for c in call_nums if by_call[c].get("r_context")
        ]
        rcohere_series = [
            (c, sum(by_call[c]["r_cohere"]) / len(by_call[c]["r_cohere"]))
            for c in call_nums if by_call[c].get("r_cohere")
        ]

        trends_html += '<h3 style="font-size:12px;color:#555;margin-top:12px;margin-bottom:6px;">Trends over training (call-averaged)</h3>'
        trends_html += '<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;">'
        trends_html += _trend_svg(composite_series, "Composite reward", "#6a1b9a")
        trends_html += _trend_svg(rground_series, "R_ground", "#1976d2")
        trends_html += _trend_svg(rcontext_series, "R_context", "#2e7d32")
        trends_html += _trend_svg(rcohere_series, "R_cohere", "#e65100")
        trends_html += "</div>"

        # Component histograms
        trends_html += '<h3 style="font-size:12px;color:#555;margin-top:14px;margin-bottom:6px;">Component distributions (all rollouts)</h3>'
        trends_html += '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;">'
        palette = ["#1976d2", "#2e7d32", "#00796b", "#6a1b9a", "#e65100", "#c62828"]
        for i, name in enumerate(COMPONENT_NAMES):
            values = [
                float((e.get("components") or {}).get(name))
                for e in ci
                if isinstance((e.get("components") or {}).get(name), (int, float))
            ]
            trends_html += _histogram_svg(values, name, palette[i % len(palette)])
        trends_html += "</div>"

    return (
        '<div class="overview-panel">'
        '<h2>GRPO Training Overview</h2>'
        f'<div class="overview-grid">{stats_html}</div>'
        f'{trends_html}'
        '</div>'
    )


def infer_weights(entries: list[dict]) -> list[float] | None:
    """Infer per-component weights from a (components, weighted) pair.

    Takes the first entry that has both non-zero components and weighted
    fields and divides.
    """
    for e in entries:
        comps = e.get("components") or {}
        weighted = e.get("weighted") or {}
        if not comps or not weighted:
            continue
        ws = []
        ok = True
        for name in COMPONENT_NAMES:
            c = comps.get(name)
            w = weighted.get(name)
            if c is None or w is None:
                ok = False
                break
            if abs(c) < 1e-9:
                ws.append(None)  # can't compute
            else:
                ws.append(float(w) / float(c))
        if not ok:
            continue
        # Fill missing weights with another entry if possible
        if any(w is None for w in ws):
            continue
        return ws
    return None


# ── GRPO-specific JS (row body renderer) ──────────────────────────────

GRPO_BODY_JS = r"""
const COMPONENT_NAMES = ["r_uncert", "r_complete", "r_consist", "r_context", "r_cohere", "r_ground"];
const COMPONENT_COLORS = {
  r_uncert:   "#1976d2",
  r_complete: "#2e7d32",
  r_consist:  "#00796b",
  r_context:  "#6a1b9a",
  r_cohere:   "#e65100",
  r_ground:   "#c62828",
};

HOOKS.getRowIdLine = function(row, stageData) {
  const parts = [];
  if (row.source_id != null) parts.push(`<b>source</b>=${esc(row.source_id)}`);
  if (row.prompt_id) parts.push(`<b>prompt</b>=${esc(String(row.prompt_id).slice(0, 10))}…`);
  parts.push(`<b>rollouts</b>=${row.n_rollouts}`);
  if (row.composite_mean != null) parts.push(`<b>μ</b>=${row.composite_mean.toFixed(3)}`);
  if (row.composite_spread != null && row.composite_spread > 0) parts.push(`<b>Δ</b>=${row.composite_spread.toFixed(3)}`);
  return parts.join('&ensp;');
};

HOOKS.getRowBadges = function(row, stageData) {
  const badges = [];
  if (stageData.stage_type === 'ci_calls') {
    if (row.is_contrastive) badges.push({text: 'contrastive', cls: 'warn'});
    if (row.is_no_flow) badges.push({text: 'no-flow', cls: 'neutral'});
    if (row.gold_has_exchange === true) badges.push({text: 'gold: exchange', cls: 'ok'});
    else if (row.gold_has_exchange === false) badges.push({text: 'gold: no exchange', cls: 'warn'});
    if (row.rground_mode) badges.push({text: `rg:${row.rground_mode}`, cls: 'accent'});
    if (row.enable_thinking_grpo === true) badges.push({text: 'think', cls: 'info'});
  } else if (stageData.stage_type === 'judgment_calls') {
    if (row.gold_judgment) badges.push({text: `gold:${row.gold_judgment}`, cls: 'accent'});
  }
  return badges;
};

HOOKS.buildRowBody = function(row, stageData) {
  if (stageData.stage_type === 'ci_calls') return buildCiCallBody(row);
  if (stageData.stage_type === 'judgment_calls') return buildJudgmentCallBody(row);
  return '<pre>' + esc(JSON.stringify(row, null, 2)) + '</pre>';
};

function buildCiCallBody(row) {
  let html = '';

  // Call-level summary bar
  html += '<div style="display:flex;gap:16px;flex-wrap:wrap;margin-bottom:12px;font-size:12px;color:#555;">';
  html += `<span><b>Call #</b>${row.idx}</span>`;
  html += `<span><b>n rollouts:</b> ${row.n_rollouts}</span>`;
  if (row.composite_mean != null) html += `<span><b>composite μ:</b> ${row.composite_mean.toFixed(3)}</span>`;
  if (row.composite_min != null) html += `<span><b>min:</b> ${row.composite_min.toFixed(3)}</span>`;
  if (row.composite_max != null) html += `<span><b>max:</b> ${row.composite_max.toFixed(3)}</span>`;
  if (row.composite_spread != null) html += `<span><b>Δ (advantage):</b> ${row.composite_spread.toFixed(3)}</span>`;
  if (row.is_contrastive) html += `<span><b>contrastive wrong:</b> ${esc(row.contrastive_source || '?')}</span>`;
  html += '</div>';

  // Prompt (from the rewards hook)
  if (row.prompt) {
    const isLong = row.prompt.length > 600;
    html += `<div class="text-box" data-export="prompt">
      <details${isLong ? '' : ' open'}>
        <summary>Prompt (${row.prompt.length.toLocaleString()} chars)</summary>
        <pre>${highlightSearch(esc(row.prompt))}</pre>
      </details>
    </div>`;
  } else {
    html += `<div class="text-box" data-export="prompt">
      <details><summary>Prompt not persisted (older trace) — prompt_id=${esc(row.prompt_id || '?')}</summary>
      <pre>Re-run GRPO after the prompt-persistence hook to see the source text here.</pre></details>
    </div>`;
  }

  // Rollouts
  const rollouts = row.rollouts || [];
  rollouts.forEach((rollout, i) => {
    html += buildCiRolloutCard(rollout, i);
  });

  return html;
}

function buildCiRolloutCard(rollout, pos) {
  const comps = rollout.components || {};
  const weighted = rollout.weighted || {};
  const composite = rollout.composite != null ? rollout.composite.toFixed(4) : '—';

  let html = `<div class="field-group open" data-export="rollout-${pos}">`;
  html += `<div class="field-group-header">
    Rollout #${rollout.idx} &nbsp;·&nbsp; composite = <b style="color:#6a1b9a;">${composite}</b>
    ${rollout.completion_len ? '&nbsp;·&nbsp; ' + rollout.completion_len + ' chars' : ''}
  </div>`;
  html += '<div class="field-group-body" style="padding:10px 14px;">';

  // Reward component bars
  html += '<div style="margin-bottom:12px;">';
  COMPONENT_NAMES.forEach(name => {
    const raw = comps[name];
    const wVal = weighted[name];
    const rawStr = (typeof raw === 'number') ? raw.toFixed(3) : '—';
    const wStr = (typeof wVal === 'number') ? wVal.toFixed(3) : '—';
    const color = COMPONENT_COLORS[name] || '#888';
    const pct = (typeof raw === 'number') ? Math.max(0, Math.min(1, raw)) * 100 : 0;
    html += `<div class="reward-bar-row">
      <div class="reward-bar-label">${name}</div>
      <div class="reward-bar-track">
        <div class="reward-bar-fill" style="width:${pct.toFixed(1)}%;background:${color};"></div>
      </div>
      <div class="reward-bar-value">${rawStr} <span class="weighted">(×w=${wStr})</span></div>
    </div>`;
  });
  html += '</div>';

  // Completion text
  if (rollout.completion) {
    const compLen = rollout.completion.length;
    html += `<div class="text-box completion" data-export="completion-${pos}">
      <details${compLen < 2000 ? ' open' : ''}>
        <summary>Completion (${compLen.toLocaleString()} chars)</summary>
        <pre>${highlightSearch(formatTextWithJson(rollout.completion))}</pre>
      </details>
    </div>`;
  }

  // R_ground flow breakdowns
  if (Array.isArray(rollout.rground_flows) && rollout.rground_flows.length > 0) {
    html += `<div style="margin-top:10px;">`;
    html += `<div style="font-size:11px;font-weight:600;color:#555;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.4px;">R_ground flow breakdown (${rollout.rground_flows.length})</div>`;
    rollout.rground_flows.forEach((flow, fi) => {
      html += buildRgroundFlowCard(flow, fi);
    });
    html += '</div>';
  }

  html += '</div></div>';
  return html;
}

function buildRgroundFlowCard(flow, fi) {
  const correctScore = flow.correct_score;
  const wrongScore = flow.wrong_score;
  const lambdaScale = 0.5;  // default — we'll show the formula, exact λ depends on config
  const contrastive = (typeof correctScore === 'number' && typeof wrongScore === 'number')
    ? Math.max(0, Math.min(1, correctScore - lambdaScale * wrongScore))
    : null;

  // Sign-flip = wrong universe scored higher or equal — an interesting failure mode
  const signFlip = (typeof correctScore === 'number' && typeof wrongScore === 'number' && wrongScore >= correctScore);

  let html = `<div class="rground-flow-card${signFlip ? ' sign-flip' : ''}">`;
  html += `<div class="rground-flow-header">Flow #${fi}${signFlip ? ' — ⚠ wrong universe scored ≥ correct' : ''}</div>`;
  html += '<div class="rground-flow-body">';

  // Query
  if (flow.query) {
    html += `<div class="rground-query">${highlightSearch(esc(flow.query))}</div>`;
  }

  // Two-column layout: correct vs wrong
  html += '<div class="rground-two-col">';
  html += buildRgroundUniverseCol(
    'correct', flow.source_id,
    flow.correct_norm_snippets, flow.correct_retrieval_sims,
    flow.correct_norm_match, flow.correct_governance, flow.correct_appropriateness,
    flow.correct_score
  );
  html += buildRgroundUniverseCol(
    'wrong', flow.wrong_source,
    flow.wrong_norm_snippets, flow.wrong_retrieval_sims,
    flow.wrong_norm_match, flow.wrong_governance, flow.wrong_appropriateness,
    flow.wrong_score
  );
  html += '</div>';

  // Contrastive formula
  if (contrastive != null) {
    html += `<div class="rground-contrastive">
      clamp(${correctScore.toFixed(3)} − λ·${wrongScore.toFixed(3)}, 0, 1) = <b>${contrastive.toFixed(3)}</b> <span style="color:#888;">(assuming λ=0.5)</span>
    </div>`;
  }

  html += '</div></div>';
  return html;
}

function buildRgroundUniverseCol(cls, source, snippets, sims, normMatch, governance, appropriateness, score) {
  let html = `<div class="rground-col ${cls}">`;
  html += `<div class="rground-col-header">${cls === 'correct' ? 'Correct' : 'Wrong'}`;
  if (source) html += ` <span style="color:#666;font-weight:400;">(${esc(source)})</span>`;
  html += '</div>';
  if (Array.isArray(snippets) && snippets.length > 0) {
    html += '<div class="rground-norm-list">';
    snippets.forEach((s, i) => {
      const sim = (sims && sims[i] != null) ? Number(sims[i]).toFixed(3) : '—';
      html += `<div class="rground-norm-item"><span class="sim">${sim}</span> ${esc(s)}</div>`;
    });
    html += '</div>';
  }
  html += '<div class="rground-scores">';
  if (typeof normMatch === 'number') html += `<span class="sc">match: ${normMatch.toFixed(2)}</span>`;
  if (typeof governance === 'number') html += `<span class="sc">gov: ${governance.toFixed(2)}</span>`;
  if (typeof appropriateness === 'number') html += `<span class="sc">appr: ${appropriateness.toFixed(2)}</span>`;
  if (typeof score === 'number') html += `<span class="sc" style="font-weight:600;">score: ${score.toFixed(3)}</span>`;
  html += '</div>';
  html += '</div>';
  return html;
}

function buildJudgmentCallBody(row) {
  let html = '';
  html += '<div style="display:flex;gap:16px;flex-wrap:wrap;margin-bottom:12px;font-size:12px;color:#555;">';
  html += `<span><b>Call #</b>${row.idx}</span>`;
  html += `<span><b>n rollouts:</b> ${row.n_rollouts}</span>`;
  if (row.composite_mean != null) html += `<span><b>composite μ:</b> ${row.composite_mean.toFixed(3)}</span>`;
  if (row.gold_judgment) html += `<span><b>gold:</b> ${esc(row.gold_judgment)}</span>`;
  html += '</div>';

  if (row.prompt) {
    html += `<div class="text-box" data-export="prompt">
      <details open><summary>Prompt (${row.prompt.length.toLocaleString()} chars)</summary>
      <pre>${highlightSearch(esc(row.prompt))}</pre></details>
    </div>`;
  }

  const rollouts = row.rollouts || [];
  rollouts.forEach((rollout, pos) => {
    const composite = rollout.composite != null ? rollout.composite.toFixed(4) : '—';
    html += `<div class="field-group open" data-export="rollout-${pos}">`;
    html += `<div class="field-group-header">Rollout #${rollout.idx} · composite = <b>${composite}</b></div>`;
    html += '<div class="field-group-body" style="padding:10px 14px;">';
    // Judgment component bars
    const comps = rollout.components || {};
    ['r_judgment', 'r_reasoning', 'r_norm_cite'].forEach(name => {
      const v = comps[name];
      const rawStr = (typeof v === 'number') ? v.toFixed(3) : '—';
      const pct = (typeof v === 'number') ? Math.max(0, Math.min(1, v)) * 100 : 0;
      html += `<div class="reward-bar-row">
        <div class="reward-bar-label">${name}</div>
        <div class="reward-bar-track">
          <div class="reward-bar-fill" style="width:${pct.toFixed(1)}%;background:#6a1b9a;"></div>
        </div>
        <div class="reward-bar-value">${rawStr}</div>
      </div>`;
    });
    if (rollout.completion) {
      html += `<div class="text-box completion" data-export="completion-${pos}">
        <details open><summary>Completion (${rollout.completion.length.toLocaleString()} chars)</summary>
        <pre>${highlightSearch(formatTextWithJson(rollout.completion))}</pre></details>
      </div>`;
    }
    html += '</div></div>';
  });

  return html;
}
"""


# ── Main ────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Static HTML inspector for GRPO reward traces.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--run", type=str, default=None,
        help="Run root (auto-discovers grpo/checkpoint/reward_traces.jsonl)",
    )
    parser.add_argument(
        "--reward-traces", type=str, default=None,
        help="Explicit path to reward_traces.jsonl",
    )
    parser.add_argument(
        "-o", "--output", default="grpo_inspector.html",
        help="Output HTML file path (default: grpo_inspector.html)",
    )
    parser.add_argument(
        "--max-calls", type=int, default=None,
        help="Maximum number of call groups per stage (default: all). "
             "Selects the most recent calls.",
    )
    parser.add_argument(
        "--rows", type=str, default=None, metavar="SPEC",
        help="Call-index slice spec ('0:100', '-50:', '::5'). Overrides --max-calls.",
    )
    args = parser.parse_args()

    if args.rows and args.max_calls:
        parser.error("--rows and --max-calls are mutually exclusive")

    traces_path = Path(args.reward_traces) if args.reward_traces else None
    if args.run and not traces_path:
        traces_path = discover_reward_traces(Path(args.run))

    if not traces_path:
        print("ERROR: could not locate reward_traces.jsonl. Use --reward-traces or --run.", file=sys.stderr)
        sys.exit(1)
    if not traces_path.exists():
        print(f"ERROR: reward_traces not found at {traces_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading reward_traces: {traces_path}")
    size_mb = traces_path.stat().st_size / (1024 * 1024)
    print(f"  file size: {size_mb:.1f} MB")
    entries = load_traces(traces_path)
    print(f"  parsed {len(entries):,} rollout entries")

    weights = infer_weights(entries)
    if weights:
        print(f"  inferred weights: {dict(zip(COMPONENT_NAMES, [round(w, 3) for w in weights]))}")

    by_task_calls = group_by_call(entries)
    stages_data: dict[str, dict] = {}

    # Fixed order so ci_extraction is always the default stage
    task_order = ["ci_extraction", "norm_judgment"]
    task_order += [t for t in by_task_calls if t not in task_order]
    for task_type in task_order:
        calls = by_task_calls.get(task_type, [])
        if not calls:
            continue
        n_total_calls = len(calls)

        # Apply call slice
        if args.rows:
            try:
                indices = parse_row_slice(args.rows, n_total_calls)
                calls = [calls[i] for i in indices]
            except ValueError as e:
                print(f"ERROR in {task_type}: {e}", file=sys.stderr)
                sys.exit(1)
        elif args.max_calls and args.max_calls < n_total_calls:
            calls = calls[-args.max_calls:]

        if task_type == "ci_extraction":
            rows = build_ci_call_rows(calls)
            stage_key = "ci_calls"
            stage_type = "ci_calls"
        elif task_type == "norm_judgment":
            rows = build_judgment_call_rows(calls)
            stage_key = "judgment_calls"
            stage_type = "judgment_calls"
        else:
            print(f"(skipping unknown task_type: {task_type})")
            continue

        stages_data[stage_key] = {
            "rows": rows,
            "id_cols": ["source_id"],
            "n_total": n_total_calls,
            "stage_type": stage_type,
            "task_type": task_type,
        }
        print(f"  {stage_key}: {len(rows)} calls (of {n_total_calls} total)")

    if not stages_data:
        print("ERROR: no stages built — reward_traces file may be empty or unparseable.", file=sys.stderr)
        sys.exit(1)

    overview_html = build_overview_html(entries, weights, traces_path)

    print("Rendering HTML...")
    html = render_page(
        tool_name="grpo",
        title="GRPO Inspector",
        accent="purple",
        stages_data=stages_data,
        overview_html=overview_html,
        body_renderer_js=GRPO_BODY_JS,
    )

    out = Path(args.output)
    out.write_text(html, encoding="utf-8")
    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"\nOutput: {out.resolve()} ({size_mb:.1f} MB)")
    print(f"Open in browser: file://{out.resolve()}")


if __name__ == "__main__":
    main()
