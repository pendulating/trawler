#!/usr/bin/env python3
"""Generate the paper-ready GRPO figures for the v9 run (single config).

Replaces the prior lambda=0.5/1.0 dual-line figures with single-config v9
versions, and adds a held-out GoldCoin compliance breakdown (SFT/v8/v9) — the
figure where the real v9-vs-v8 story lives (the training-trace "hedge collapse"
was a measurement artifact; see wiki field notes).

Outputs to --out (staging dir); copy into the paper figures/ after review:
  reward_trajectory.pdf       composite reward over training (v9)
  reward_components.pdf        6 reward components over training (v9)
  training_diagnostics.pdf     no-flow rate + early/late reward distribution (v9)
  completion_length.pdf        mean completion length over training (v9, chars)
  goldcoin_compliance_breakdown.pdf   NEW: SFT/v8/v9 held-out GoldCoin bars
"""
import argparse, json, collections
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

V9 = ("/share/pierson/matt/UAIR/multirun/2026-06-23_grpo_redesign_full_v9/"
      "20-09-18/0/grpo_only_online_external/outputs/grpo/checkpoint/reward_traces.jsonl")
SMOOTH = 15
COMP = ["r_uncert", "r_complete", "r_consist", "r_context", "r_cohere", "r_ground"]
CLAB = {"r_uncert": r"$R_{\mathrm{uncert}}$", "r_complete": r"$R_{\mathrm{complete}}$",
        "r_consist": r"$R_{\mathrm{consist}}$", "r_context": r"$R_{\mathrm{context}}$",
        "r_cohere": r"$R_{\mathrm{cohere}}$", "r_ground": r"$R_{\mathrm{ground}}$"}
CCOL = {"r_uncert": "#D62728", "r_complete": "#1F77B4", "r_consist": "#2CA02C",
        "r_context": "#807DBA", "r_cohere": "#FF7F0E", "r_ground": "#8C564B"}

# Held-out GoldCoin (run 2026-06-24_goldcoin_v9_vs_v8_vs_sft/11-25-03), verified metrics.json.
GOLDCOIN = {                       # Forbid recall, Forbid prec, Permit recall, Comp macro-F1
    "SFT (contentless-v6)": [0.650, 0.500, 0.851, 0.723],
    "v8-ckpt200":           [0.350, 0.538, 0.931, 0.660],
    "v9-ckpt100 (GRPO)":    [0.550, 0.647, 0.931, 0.755],
}
GC_METRICS = ["Forbid\nrecall", "Forbid\nprecision", "Permit\nrecall", "Compliance\nmacro-F1"]
GC_COLORS = {"SFT (contentless-v6)": "#9E9E9E", "v8-ckpt200": "#D62728", "v9-ckpt100 (GRPO)": "#2171B5"}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.alpha": 0.25, "grid.linewidth": 0.5,
})


def smooth(y):
    y = np.asarray(y, float)
    return uniform_filter1d(y, size=min(SMOOTH, max(1, len(y)))) if len(y) else y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/claude-1643544/-share-pierson-matt-UAIR/"
                    "b943b9fe-3ff0-432f-8710-13b722eb7953/scratchpad/paper_figures")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    ci = []
    with open(V9) as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("task_type") == "ci_extraction":
                ci.append(d)
    by_call = collections.defaultdict(list)
    for r in ci:
        by_call[r.get("call", 0)].append(r)
    ucalls = sorted(by_call)

    def series(fn):
        return np.array([np.mean([fn(r) for r in by_call[c] if fn(r) is not None] or [np.nan])
                         for c in ucalls])

    composite = series(lambda r: r.get("composite"))
    complen = series(lambda r: r.get("completion_len"))
    comp_series = {k: series(lambda r, k=k: (r.get("components") or {}).get(k)) for k in COMP}
    noflow = series(lambda r: 1.0 if r.get("is_no_flow") else 0.0)

    # 1. reward trajectory
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.scatter(ucalls, composite, s=4, color="#A63603", alpha=0.15, zorder=2)
    ax.plot(ucalls, smooth(composite), color="#A63603", lw=2, zorder=3)
    ax.set_xlabel("training step (judge-call index)"); ax.set_ylabel("composite reward")
    ax.set_ylim(0, 0.7)
    fig.savefig(out / "reward_trajectory.pdf"); plt.close(fig)

    # 2. reward components
    fig, ax = plt.subplots(figsize=(8, 3))
    for k in COMP:
        ax.plot(ucalls, smooth(comp_series[k]), color=CCOL[k], lw=1.8, label=CLAB[k])
    ax.set_xlabel("training step (judge-call index)"); ax.set_ylabel("component reward")
    ax.set_ylim(0, 1.02); ax.legend(ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.18), frameon=False)
    fig.savefig(out / "reward_components.pdf"); plt.close(fig)

    # 3. training diagnostics: no-flow rate (left) + early/late reward dist (right)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    axes[0].plot(ucalls, smooth(noflow) * 100, color="#6A51A3", lw=2)
    axes[0].set_xlabel("training step (judge-call index)"); axes[0].set_ylabel("no-flow (abstention) rate (%)")
    axes[0].set_ylim(0, 100); axes[0].set_title("Abstention rate")
    cmax = max(ucalls)
    early = [r.get("composite") for r in ci if r.get("call", 0) <= ucalls[len(ucalls) // 10]]
    late = [r.get("composite") for r in ci if r.get("call", 0) >= ucalls[-len(ucalls) // 10]]
    axes[1].hist([e for e in early if e is not None], bins=20, range=(0, 1), alpha=0.55,
                 color="#FDAE6B", label="early", density=True)
    axes[1].hist([l for l in late if l is not None], bins=20, range=(0, 1), alpha=0.55,
                 color="#A63603", label="late", density=True)
    axes[1].set_xlabel("composite reward"); axes[1].set_ylabel("density"); axes[1].set_title("Reward distribution")
    axes[1].legend()
    fig.tight_layout(); fig.savefig(out / "training_diagnostics.pdf"); plt.close(fig)

    # 4. completion length (chars)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(ucalls, smooth(complen), color="#2171B5", lw=2)
    ax.set_xlabel("training step (judge-call index)"); ax.set_ylabel("mean completion length (chars)")
    fig.savefig(out / "completion_length.pdf"); plt.close(fig)

    # 5. NEW: held-out GoldCoin compliance breakdown (SFT/v8/v9)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    arms = list(GOLDCOIN); x = np.arange(len(GC_METRICS)); bw = 0.8 / len(arms)
    for j, arm in enumerate(arms):
        off = (j - len(arms) / 2 + 0.5) * bw
        bars = ax.bar(x + off, GOLDCOIN[arm], bw, color=GC_COLORS[arm], label=arm,
                      edgecolor="white", linewidth=0.5, zorder=3)
        for b, v in zip(bars, GOLDCOIN[arm]):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(GC_METRICS); ax.set_ylim(0, 1.05)
    ax.set_ylabel("score"); ax.legend(loc="lower right", ncol=1)
    ax.set_title("Held-out GoldCoin-HIPAA compliance (matched, contentless-v6 base)")
    fig.savefig(out / "goldcoin_compliance_breakdown.pdf"); plt.close(fig)

    print(f"wrote 5 paper figures to {out}")
    print(f"no-flow(abstention) rate: {noflow.mean()*100:.1f}% mean over training")


if __name__ == "__main__":
    main()
