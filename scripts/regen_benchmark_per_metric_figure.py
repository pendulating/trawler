#!/usr/bin/env python3
"""Regenerate the main-body per-metric benchmark figure from the FINAL table.

`figures/zero_shot_vs_sft_vs_grpo_per_metric.pdf` (referenced in 04_results.tex)
was previously generated from a stale May-1 W&B cache, which carried two errors:
(1) PrivacyLens panels used the original lenient judge, and (2) the Qwen3.5-9B
GRPO bars were a pre-v9 GRPO run. The v9 evals are not in W&B.

To guarantee figure<->table consistency, this script parses the authoritative
`tables/benchmark_results.tex` (now final: matched contentless-v6 SFT, v9-ckpt100
GRPO, Qwen3.6-27B-judged PrivacyLens) and plots the same 6 metrics the original
figure showed. Single source of truth = the table.

Usage:
  python scripts/regen_benchmark_per_metric_figure.py        # writes into the paper figures/
  python scripts/regen_benchmark_per_metric_figure.py --out /some/dir
"""
import argparse
import re
from math import ceil
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAPER = Path("/share/pierson/matt/UAIR/papers/colm26_normative-simulacra")
TABLE = PAPER / "tables" / "benchmark_results.tex"

# Column index (0-based, after splitting a data row on '&') -> metric.
# Row layout: Model & Cond & Appl & Comp & QA & Lk & AdjLk & Helpful & Help & r & CIRL-Comp & CIRL-Acc & Q7 & MMLU
COL = {"Appl": 2, "Comp": 3, "Helpful": 7, "AdjLk": 6, "r": 9, "Q7": 12}

# The 6 plotted panels: (table-key, panel title, higher_is_better)
PANELS = [
    ("Appl",   "GoldCoin Appl. F1 (%)",        True),
    ("Comp",   "GoldCoin Comp. F1 (%)",        True),
    ("Helpful","PrivacyLens Helpful Rate (%)", True),
    ("AdjLk",  "PrivacyLens Adj Leak (%) ↓", False),
    ("r",      "ConfAIde Pearson r (%)",       True),
    ("Q7",     "VLM-GeoPrivacy Q7 Acc (%)",    True),
]

COND_COLORS = {"0-Shot": "#4C72B0", "SFT": "#DD8452", "GRPO": "#55A868"}
COND_ORDER = ["0-Shot", "SFT", "GRPO"]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8.5,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.alpha": 0.25, "grid.linewidth": 0.5,
})


def _cell(tok):
    """Strip \\textbf{...} and convert '---' to NaN; return float or nan."""
    tok = tok.strip()
    m = re.fullmatch(r"\\textbf\{(.*)\}", tok)
    if m:
        tok = m.group(1).strip()
    if tok in ("---", "--", "", "$\\downarrow$"):
        return np.nan
    try:
        return float(tok)
    except ValueError:
        return np.nan


def parse_table():
    """Return ordered {model: {cond: {metric: value}}} from the table body."""
    rows = []  # (model, cond, {metric: val}) in file order
    cur_model = None
    in_body = False
    for raw in TABLE.read_text().splitlines():
        line = raw.strip()
        if line.startswith(r"\midrule"):
            in_body = True
            continue
        if line.startswith(r"\bottomrule"):
            break
        if not in_body or "&" not in line:
            continue
        # data row ends with '\\' (possibly trailing comment); drop comment + \\
        line = line.split("%")[0].rstrip()
        if not line.endswith(r"\\"):
            continue
        cells = [c for c in line[:-2].split("&")]
        if len(cells) < 12:
            continue
        model_tok = cells[0].strip().replace(r"\textbf{", "").replace("}", "")
        cond = cells[1].strip()
        if model_tok:
            cur_model = model_tok
        if cur_model is None or cond not in COND_ORDER:
            continue
        vals = {k: _cell(cells[idx]) for k, idx in COL.items()}
        rows.append((cur_model, cond, vals))

    # preserve first-seen model order
    data, order = {}, []
    for model, cond, vals in rows:
        if model not in data:
            data[model] = {}
            order.append(model)
        data[model][cond] = vals
    return data, order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(PAPER / "figures"))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    data, models = parse_table()
    n_models = len(models)
    x = np.arange(n_models)
    bar_w = 0.27

    ncols = ceil(len(PANELS) / 2)
    fig, axes = plt.subplots(2, ncols, figsize=(5 * ncols, 9.2), squeeze=False)
    axes = axes.flatten()

    for ax, (key, title, _hb) in zip(axes, PANELS):
        for j, cond in enumerate(COND_ORDER):
            offset = (j - 1) * bar_w  # -1,0,+1 -> ZS,SFT,GRPO
            vals = [data.get(m, {}).get(cond, {}).get(key, np.nan) for m in models]
            ax.bar(x + offset, vals, bar_w, color=COND_COLORS[cond], label=cond,
                   edgecolor="white", linewidth=0.5, zorder=3)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=40, ha="right")
        ax.set_ylim(0, max(5, np.nanmax([
            data.get(m, {}).get(c, {}).get(key, np.nan)
            for m in models for c in COND_ORDER]) * 1.12))

    # one shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    # de-dup labels while keeping order
    seen, h2, l2 = set(), [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l); h2.append(h); l2.append(l)
    fig.legend(h2, l2, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    dest = out / "zero_shot_vs_sft_vs_grpo_per_metric.pdf"
    fig.savefig(dest); plt.close(fig)
    print(f"wrote {dest}")
    print(f"models ({n_models}): {models}")
    # sanity echo of the 9B progression
    nine = data.get("Qwen3.5-9B", {})
    for cond in COND_ORDER:
        print(f"  Qwen3.5-9B {cond:9s}: {nine.get(cond)}")


if __name__ == "__main__":
    main()
