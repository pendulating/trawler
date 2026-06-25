#!/usr/bin/env python3
"""Regenerate the longitudinal GRPO reward plots from the v9 run's reward traces.

Mirrors the style of notebooks/COLM26/grpo_ablations.ipynb (serif, window-15
uniform smoothing, dpi 300) but for the single v9 config (no lambda sweep), and
ADDS the appropriateness-verdict panel that tells the v9 story: the hedge
("ambiguous") fraction collapsing as the direction multiplier takes hold.

Outputs (PDF) to the dir given by --out (default: scratchpad):
  v9_reward_trajectory.pdf   composite reward over training (judge-call index)
  v9_reward_components.pdf    the 6 components over training
  v9_completion_length.pdf    mean completion length over training
  v9_appropriateness.pdf      appropriateness-verdict fractions over training (NEW)
  v9_grpo_panels.pdf          all four as a 2x2 combined panel (preview)

Read-only on the trace; writes only image files.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

TRACE = ("/share/pierson/matt/UAIR/multirun/2026-06-23_grpo_redesign_full_v9/"
         "20-09-18/0/grpo_only_online_external/outputs/grpo/checkpoint/reward_traces.jsonl")
SMOOTH = 15
COMPONENT_ORDER = ["r_uncert", "r_complete", "r_consist", "r_context", "r_cohere", "r_ground"]
COMPONENT_LABEL = {
    "r_uncert": r"$R_{\mathrm{uncert}}$", "r_complete": r"$R_{\mathrm{complete}}$",
    "r_consist": r"$R_{\mathrm{consist}}$", "r_context": r"$R_{\mathrm{context}}$",
    "r_cohere": r"$R_{\mathrm{cohere}}$", "r_ground": r"$R_{\mathrm{ground}}$",
}
COMPONENT_COLOR = {
    "r_uncert": "#D62728", "r_complete": "#1F77B4", "r_consist": "#2CA02C",
    "r_context": "#807DBA", "r_cohere": "#FF7F0E", "r_ground": "#8C564B",
}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.alpha": 0.25, "grid.linewidth": 0.5,
})


def smooth(y):
    y = np.asarray(y, dtype=float)
    return uniform_filter1d(y, size=min(SMOOTH, max(1, len(y)))) if len(y) else y


def verdict_of(rec):
    """Majority appropriateness verdict among the completion's flows, or None."""
    try:
        cj = json.loads(rec["completion"])
    except Exception:
        return None
    labs = []
    for fl in (cj.get("flows") or cj.get("information_flows") or []):
        if isinstance(fl, dict) and fl.get("appropriateness"):
            labs.append(str(fl["appropriateness"]).strip().lower())
    if not labs:
        return None
    # majority
    return max(set(labs), key=labs.count)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default=TRACE)
    ap.add_argument("--out", default="/tmp/claude-1643544/-share-pierson-matt-UAIR/"
                    "b943b9fe-3ff0-432f-8710-13b722eb7953/scratchpad/plots")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    ci = []
    with open(args.trace) as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("task_type") == "ci_extraction":
                ci.append(d)
    ci.sort(key=lambda r: r.get("call", 0))
    calls = np.array([r.get("call", 0) for r in ci], dtype=float)
    print(f"{len(ci)} ci_extraction records, call {calls.min():.0f}->{calls.max():.0f}")

    # group by call (mean per call) for smooth trajectories
    import collections
    by_call = collections.defaultdict(list)
    for r in ci:
        by_call[r.get("call", 0)].append(r)
    ucalls = sorted(by_call)

    def series(fn):
        return np.array([np.mean([fn(r) for r in by_call[c] if fn(r) is not None]
                                 or [np.nan]) for c in ucalls])

    comp_series = {k: series(lambda r, k=k: (r.get("components") or {}).get(k))
                   for k in COMPONENT_ORDER}
    composite = series(lambda r: r.get("composite"))
    complen = series(lambda r: r.get("completion_len"))

    # ---- 1. reward trajectory ----
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.scatter(ucalls, composite, s=3, color="#A63603", alpha=0.18, zorder=2)
    ax.plot(ucalls, smooth(composite), color="#A63603", lw=2, zorder=3, label="composite reward (v9)")
    ax.set_xlabel("training step (judge-call index)"); ax.set_ylabel("composite reward")
    ax.set_title("v9 composite reward over training"); ax.legend()
    fig.savefig(out / "v9_reward_trajectory.pdf"); plt.close(fig)

    # ---- 2. reward components ----
    fig, ax = plt.subplots(figsize=(5, 3))
    for k in COMPONENT_ORDER:
        ax.plot(ucalls, smooth(comp_series[k]), color=COMPONENT_COLOR[k], lw=1.6, label=COMPONENT_LABEL[k])
    ax.set_xlabel("training step (judge-call index)"); ax.set_ylabel("component reward")
    ax.set_ylim(0, 1.02); ax.set_title("v9 reward components over training")
    ax.legend(ncol=3, loc="lower center")
    fig.savefig(out / "v9_reward_components.pdf"); plt.close(fig)

    # ---- 3. completion length ----
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(ucalls, smooth(complen), color="#2171B5", lw=2)
    ax.set_xlabel("training step (judge-call index)"); ax.set_ylabel("mean completion length (chars)")
    ax.set_title("v9 completion length over training")
    fig.savefig(out / "v9_completion_length.pdf"); plt.close(fig)

    # ---- 4. appropriateness verdict fractions (the v9 hedge-collapse story) ----
    # bin by call into deciles for stable fractions
    nbins = 12
    edges = np.linspace(calls.min(), calls.max() + 1e-6, nbins + 1)
    centers, frac_amb, frac_app, frac_inapp = [], [], [], []
    for i in range(nbins):
        lo, hi = edges[i], edges[i + 1]
        recs = [r for r in ci if lo <= r.get("call", 0) < hi]
        verds = [verdict_of(r) for r in recs]
        verds = [v for v in verds if v is not None]
        if len(verds) < 5:
            continue
        n = len(verds)
        centers.append((lo + hi) / 2)
        frac_amb.append(sum(v == "ambiguous" for v in verds) / n)
        frac_app.append(sum(v == "appropriate" for v in verds) / n)
        frac_inapp.append(sum(v == "inappropriate" for v in verds) / n)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(centers, np.array(frac_amb) * 100, "-o", ms=3, color="#7F7F7F", label="ambiguous (hedge)")
    ax.plot(centers, np.array(frac_app) * 100, "-o", ms=3, color="#2CA02C", label="appropriate")
    ax.plot(centers, np.array(frac_inapp) * 100, "-o", ms=3, color="#D62728", label="inappropriate")
    ax.set_xlabel("training step (judge-call index)"); ax.set_ylabel("share of extracted-flow verdicts (%)")
    ax.set_title("v9 appropriateness verdicts over training"); ax.legend()
    fig.savefig(out / "v9_appropriateness.pdf"); plt.close(fig)
    if frac_amb:
        print(f"hedge fraction: {frac_amb[0]*100:.1f}% (early) -> {frac_amb[-1]*100:.1f}% (late)")

    # ---- combined 2x2 preview ----
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    a = axes[0, 0]
    a.scatter(ucalls, composite, s=3, color="#A63603", alpha=0.18)
    a.plot(ucalls, smooth(composite), color="#A63603", lw=2)
    a.set_title("composite reward"); a.set_xlabel("step"); a.set_ylabel("reward")
    a = axes[0, 1]
    for k in COMPONENT_ORDER:
        a.plot(ucalls, smooth(comp_series[k]), color=COMPONENT_COLOR[k], lw=1.4, label=COMPONENT_LABEL[k])
    a.set_ylim(0, 1.02); a.set_title("reward components"); a.set_xlabel("step")
    a.legend(ncol=3, fontsize=6, loc="lower center")
    a = axes[1, 0]
    a.plot(ucalls, smooth(complen), color="#2171B5", lw=2)
    a.set_title("completion length"); a.set_xlabel("step"); a.set_ylabel("chars")
    a = axes[1, 1]
    a.plot(centers, np.array(frac_amb) * 100, "-o", ms=3, color="#7F7F7F", label="ambiguous")
    a.plot(centers, np.array(frac_app) * 100, "-o", ms=3, color="#2CA02C", label="appropriate")
    a.plot(centers, np.array(frac_inapp) * 100, "-o", ms=3, color="#D62728", label="inappropriate")
    a.set_title("appropriateness verdicts (%)"); a.set_xlabel("step"); a.legend(fontsize=6)
    fig.suptitle("v9 GRPO training dynamics (Qwen3.5-9B, contentless-v6 base)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "v9_grpo_panels.pdf"); plt.close(fig)
    print(f"wrote 5 PDFs to {out}")


if __name__ == "__main__":
    main()
