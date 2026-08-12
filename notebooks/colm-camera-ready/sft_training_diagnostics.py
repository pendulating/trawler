import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # COLM camera-ready — SFT training diagnostics

    Builds the Appendix figures and tables for the **supervised fine-tuning**
    stage, which the paper reports for eleven models in
    `tab:benchmark_results` without ever showing a training curve, a
    hyperparameter, or a held-out signal. This notebook is the SFT counterpart
    of `grpo_kto_training_diagnostics.py`.

    ### Two protocol eras, and why both appear

    | | reported sweep | later sweep |
    |---|---|---|
    | dir | `2026-07-15_sft_canonical_gemma4{,_gptoss}` | `2026-07-19_sft_canonical_gemma4{,_gptoss}` |
    | loss | TRL stock NLL | DFT (`loss_type: dft`) |
    | held-out split | **none** | grouped, whole novels, 5% |
    | checkpoint kept | last epoch | lowest held-out loss |
    | chat template | Gemma-3 template applied to Gemma-4 | per-family, verified |

    The benchmark table's SFT rows come from the **2026-07-15** sweep, so that
    is what the training-diagnostics figure and table describe. The 07-19 sweep
    exists because the 07-15 protocol had three defects, and it is the only
    source of a held-out curve, so it carries the generalization figure. Its
    losses are DFT-weighted and are an order of magnitude below the 07-15 NLL
    values: **never plot the two loss scales on shared axes.**

    ### The Gemma-4 template defect is visible in the training signal

    `_detect_template_family` matched the bare substring `"gemma"`, so every
    Gemma-4 run in the 07-15 sweep trained under the *Gemma-3* chat template,
    whose turn delimiters (`<start_of_turn>`) are absent from the Gemma-4
    vocabulary and tokenize into seven arbitrary word pieces. The diagnostics
    below detect it independently of that code reading: gemma-4-12b starts at
    loss 3.12 against 0.86--1.38 for every other model, posts a median gradient
    norm of 6.00 against 0.37--0.76, and is clipped at `max_grad_norm=1.0` on
    **all** of its logged steps. This is the strongest argument for printing
    the section at all: the defect was invisible in the benchmark numbers and
    obvious in the training signal.

    ### Sources — everything is read off disk

    | What | Where |
    |---|---|
    | per-model optimizer logs | `multirun/2026-07-{15,19}_sft_canonical_gemma4*/**/trainer_state.json` |
    | per-checkpoint benchmark trajectory | `data/sft_per_checkpoint_longitudinal_2026_07_20/cells.parquet` (11 models x epochs 0--3; epoch 0 = the pre-SFT instruct weights) |
    | measured re-run noise floor | `.../variance_noise_floor.parquet` (judge-free N=3 variance sweep) |
    """)
    return


@app.cell
def _():
    import glob
    import json
    import os
    from pathlib import Path

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import yaml

    PROJECT_ROOT = Path("/share/pierson/matt/UAIR")
    NB_DIR = PROJECT_ROOT / "notebooks/colm-camera-ready"
    FIG_DIR = NB_DIR / "figures/sft_diagnostics"
    TAB_DIR = NB_DIR / "tables/sft_diagnostics"
    for _d in (FIG_DIR, TAB_DIR):
        _d.mkdir(parents=True, exist_ok=True)
    PAPER_DIR = PROJECT_ROOT / "papers/colm26_normative-simulacra"
    PAPER_FIG_DIR = PAPER_DIR / "figures"
    PAPER_TAB_DIR = PAPER_DIR / "tables"

    # The reported sweep and the later, repaired one. gpt-oss trains in its own
    # dir in both eras (harmony chat format needs a separate launcher).
    SWEEPS = {
        "reported": [PROJECT_ROOT / "multirun/2026-07-15_sft_canonical_gemma4",
                     PROJECT_ROOT / "multirun/2026-07-15_sft_canonical_gemma4_gptoss"],
        "repaired": [PROJECT_ROOT / "multirun/2026-07-19_sft_canonical_gemma4",
                     PROJECT_ROOT / "multirun/2026-07-19_sft_canonical_gemma4_gptoss"],
    }

    LONGITUDINAL = (NB_DIR / "data" /
                    "sft_per_checkpoint_longitudinal_2026_07_20")

    # Family colour + within-family line style. Ten curves are unreadable under
    # ten arbitrary colours; grouping by backbone family makes the Gemma-4
    # anomaly legible as a family effect, which is what it is.
    FAMILY = {
        "qwen3.5": "#0072B2",
        "gemma-4": "#D55E00",
        "llama3.1": "#009E73",
        "harc-llama3.1": "#009E73",
        "openthinker3": "#CC79A7",
        "phi-4": "#7570B3",
        "gpt-oss": "#8A8A8A",
    }
    STYLES = ["-", "--", ":"]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
        }
    )
    TEXTWIDTH_IN = 5.5

    def save_fig(fig, name, pad_inches=0.0):
        for ext in ("png", "pdf"):
            fig.savefig(FIG_DIR / f"{name}.{ext}", dpi=300,
                        bbox_inches="tight", pad_inches=pad_inches)
        fig.savefig(PAPER_FIG_DIR / f"{name}.pdf", dpi=300,
                    bbox_inches="tight", pad_inches=pad_inches)
        print(f"[fig]   {PAPER_FIG_DIR / name}.pdf")

    def save_table(df, name, index=False):
        df.to_csv(TAB_DIR / f"{name}.csv", index=index)
        print(f"[table] {TAB_DIR / name}.csv")

    def write_table_tex(name, *, colspec, header, rows, caption, label,
                        tabcolsep="3.2pt", size=r"\small"):
        """Emit a complete \\input-able LaTeX table.

        A complete environment rather than a row fragment: \\input-ing bare rows
        into an open `tabular` breaks, because the last row's `\\\\` scans past
        the file boundary and expands the following `\\bottomrule` where
        alignment material is illegal.
        """
        body = "\n".join([
            r"\begin{table}[ht]", r"\centering", size,
            rf"\setlength{{\tabcolsep}}{{{tabcolsep}}}",
            rf"\caption{{{caption}}}", rf"\label{{{label}}}",
            rf"\begin{{tabular}}{{{colspec}}}",
            r"\toprule", header, r"\midrule", rows, r"\bottomrule",
            r"\end{tabular}", r"\end{table}",
        ])
        for d in (TAB_DIR, PAPER_TAB_DIR):
            (d / f"{name}.tex").write_text(body + "\n")
        print(f"[latex] {PAPER_TAB_DIR / name}.tex")

    return (
        FAMILY,
        LONGITUDINAL,
        STYLES,
        SWEEPS,
        TEXTWIDTH_IN,
        glob,
        json,
        np,
        os,
        pd,
        plt,
        save_fig,
        save_table,
        write_table_tex,
        yaml,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load — per-model optimizer logs for both eras

    One `trainer_state.json` per model per era, resolved to the highest-numbered
    checkpoint so the log is complete. Model identity comes from the job's
    `.hydra/overrides.yaml`, not from directory order, because the sweep
    directories interleave re-queued arms.
    """)
    return


@app.cell
def _(FAMILY, STYLES, SWEEPS, glob, json, os, pd):
    # Presentation order: by family, then by size. Fixed here so every figure
    # and the table agree, and so line styles are stable across re-runs.
    MODEL_ORDER = [
        "qwen3.5-2b", "qwen3.5-4b", "qwen3.5-9b",
        "gemma-4-e2b", "gemma-4-e4b", "gemma-4-12b",
        "llama3.1-8b", "harc-llama3.1-8b",
        "openthinker3-7b", "phi-4", "gpt-oss-20b",
    ]
    LABEL = {
        "qwen3.5-2b": "Qwen3.5-2B", "qwen3.5-4b": "Qwen3.5-4B",
        "qwen3.5-9b": "Qwen3.5-9B", "gemma-4-e2b": "Gemma-4-E2B",
        "gemma-4-e4b": "Gemma-4-E4B", "gemma-4-12b": "Gemma-4-12B",
        "llama3.1-8b": "Llama-3.1-8B", "harc-llama3.1-8b": "HARC-Llama-3.1-8B",
        "openthinker3-7b": "OpenThinker3-7B", "phi-4": "Phi-4",
        "gpt-oss-20b": "GPT-OSS-20B",
    }

    def _family(slug):
        for f in ("harc-llama3.1", "qwen3.5", "gemma-4", "llama3.1",
                  "openthinker3", "phi-4", "gpt-oss"):
            if slug.startswith(f):
                return f
        return slug

    # Style index counts within the *colour* group, so the two Llama variants
    # (which deliberately share a colour) get different dashes.
    _seen = {}
    STYLE_OF = {}
    for _m in MODEL_ORDER:
        _c = FAMILY[_family(_m)]
        STYLE_OF[_m] = STYLES[_seen.get(_c, 0) % len(STYLES)]
        _seen[_c] = _seen.get(_c, 0) + 1
    COLOR_OF = {m: FAMILY[_family(m)] for m in MODEL_ORDER}

    def _load_era(dirs):
        out = {}
        for root in dirs:
            for job in sorted(glob.glob(str(root) + "/*/*/")):
                ov = os.path.join(job, ".hydra/overrides.yaml")
                if not os.path.exists(ov):
                    continue
                hit = [ln for ln in open(ov) if "model=" in ln]
                if not hit:
                    continue
                slug = hit[0].split("model=")[1].strip().split("/")[0]
                states = sorted(
                    glob.glob(job + "**/trainer_state.json", recursive=True),
                    key=lambda p: int(p.split("checkpoint-")[1].split("/")[0])
                    if "checkpoint-" in p else 0)
                if not states:
                    continue
                st = json.loads(open(states[-1]).read())
                df = pd.DataFrame([e for e in st["log_history"] if "loss" in e])
                if df.empty:
                    continue
                # Keep the longest log if an arm was re-queued.
                if slug in out and len(out[slug][0]) >= len(df):
                    continue
                out[slug] = (df.reset_index(drop=True), st)
        return out

    sft_logs = {era: _load_era(dirs) for era, dirs in SWEEPS.items()}

    for _era in ("reported", "repaired"):
        _have = [m for m in MODEL_ORDER if m in sft_logs[_era]]
        print(f"{_era:9s} {len(_have):2d} models: {', '.join(_have)}")
        _missing = [m for m in MODEL_ORDER if m not in sft_logs[_era]]
        if _missing:
            print(f"{'':9s} missing (no trainer_state on disk): {', '.join(_missing)}")
    return COLOR_OF, LABEL, MODEL_ORDER, STYLE_OF, sft_logs


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure — SFT training diagnostics across the canonical set

    Four panels over the **reported** sweep. Panel (b) is the one that matters:
    the gradient norm is on a log axis with the clipping threshold drawn, and
    the Gemma-4 family separates from every other model by more than an order
    of magnitude.
    """)
    return


@app.cell
def _(
    COLOR_OF,
    LABEL,
    MODEL_ORDER,
    STYLE_OF,
    TEXTWIDTH_IN,
    plt,
    save_fig,
    sft_logs,
):
    _fig, _ax = plt.subplots(2, 2, figsize=(TEXTWIDTH_IN, 3.9), sharex=True)
    _p = _ax.ravel()
    _era = sft_logs["reported"]

    for _m in MODEL_ORDER:
        if _m not in _era:
            continue
        _d = _era[_m][0]
        _kw = dict(color=COLOR_OF[_m], ls=STYLE_OF[_m], lw=1.2, label=LABEL[_m])
        _p[0].plot(_d["step"], _d["loss"], **_kw)
        if "grad_norm" in _d:
            _p[1].plot(_d["step"], _d["grad_norm"], **_kw)
        if "mean_token_accuracy" in _d:
            _p[2].plot(_d["step"], _d["mean_token_accuracy"], **_kw)
        if "entropy" in _d:
            _p[3].plot(_d["step"], _d["entropy"], **_kw)

    _p[0].set_ylabel("training loss")
    _p[0].set_title("(a) training loss", loc="left")

    _p[1].set_yscale("log")
    _p[1].axhline(1.0, color="#333333", ls=":", lw=0.9)
    _p[1].text(0.02, 1.15, "clipping threshold",
               transform=_p[1].get_yaxis_transform(),
               ha="left", va="bottom", fontsize=6.0, color="#555555")
    _p[1].set_ylabel("gradient norm")
    _p[1].set_title("(b) gradient norm", loc="left")

    _p[2].set_ylabel("token accuracy")
    _p[2].set_xlabel("training step")
    _p[2].set_title("(c) token accuracy", loc="left")

    _p[3].set_ylabel("token entropy")
    _p[3].set_xlabel("training step")
    _p[3].set_title("(d) token entropy", loc="left")

    for _a in _p:
        _a.margins(x=0.01)
        _a.title.set_fontsize(8.6)
    _h, _l = _p[0].get_legend_handles_labels()
    _fig.legend(_h, _l, ncol=4, frameon=False, fontsize=6.4,
                loc="lower center", bbox_to_anchor=(0.5, -0.075),
                handlelength=1.9, columnspacing=1.2)
    _fig.tight_layout(pad=0.35, w_pad=1.0, h_pad=0.6, rect=(0, 0.085, 1, 1))
    save_fig(_fig, "fig_sft_training_diagnostics")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure — what the repaired protocol adds

    Three panels. (a) is a **within-era** measure: the share of logged steps
    whose gradient norm exceeded the clipping threshold, in the reported sweep.
    It stays inside one era on purpose. A reported-versus-repaired comparison
    of gradient magnitude looks tempting and is confounded, because the
    repaired era also switched the loss to DFT, which rescales gradients for
    *every* model; the cross-era drop is therefore not attributable to the
    template repair. (b) and (c) are the held-out curves, which exist only in
    the repaired era, since the reported sweep ran with no validation split at
    all.
    """)
    return


@app.cell
def _(
    COLOR_OF,
    LABEL,
    MODEL_ORDER,
    STYLE_OF,
    TEXTWIDTH_IN,
    json,
    np,
    plt,
    save_fig,
    sft_logs,
):
    _fig2, _ax2 = plt.subplots(1, 3, figsize=(TEXTWIDTH_IN, 2.0),
                              gridspec_kw={"width_ratios": [1.25, 1, 1]})

    # (a) Share of logged steps at the clipping ceiling, reported era only.
    # Within-era by construction, so it carries no loss-function confound.
    _rows = []
    for _m in MODEL_ORDER:
        _got = sft_logs["reported"].get(_m)
        if _got is None or "grad_norm" not in _got[0]:
            continue
        _gn = _got[0]["grad_norm"]
        _rows.append((_m, float((_gn > 1.0).mean())))
    _ys = np.arange(len(_rows))
    _ax2[0].barh(_ys, [f for _, f in _rows],
                 color=[COLOR_OF[m] for m, _ in _rows], height=0.72)
    _ax2[0].set_yticks(_ys)
    _ax2[0].set_yticklabels(
        [LABEL[m].replace("Gemma-4-", "G4-").replace("HARC-Llama-3.1-8B", "HARC-L3.1")
                 .replace("Llama-3.1-8B", "Llama-3.1").replace("OpenThinker3-7B", "OpenThink3")
                 .replace("GPT-OSS-20B", "GPT-OSS").replace("Qwen3.5-", "Qwen-")
         for m, _ in _rows], fontsize=5.8)
    _ax2[0].invert_yaxis()
    _ax2[0].set_xlim(0, 1.02)
    _ax2[0].set_xlabel("steps clipped (frac.)")
    _ax2[0].grid(False, axis="y")
    _ax2[0].grid(True, axis="x", alpha=0.25, lw=0.5)
    _ax2[0].set_title("(a) gradient clipping", loc="left", fontsize=8.6)

    # (b, c) held-out curves. Repaired era only: the reported sweep logged no
    # eval keys at all, which is itself the finding.
    def _evals(era, key):
        out = {}
        for _m in MODEL_ORDER:
            got = sft_logs[era].get(_m)
            if got is None:
                continue
            hist = got[1]["log_history"]
            pts = [(e["epoch"], e[key]) for e in hist if key in e]
            if len(pts) >= 2:
                out[_m] = np.array(pts)
        return out

    for _i, (_key, _lab, _ttl) in enumerate(
            [("eval_loss", "held-out loss (DFT)", "(b) held-out loss"),
             ("eval_mean_token_accuracy", "held-out token acc.",
              "(c) held-out token acc.")]):
        _axx = _ax2[_i + 1]
        for _m, _pts in _evals("repaired", _key).items():
            _axx.plot(_pts[:, 0], _pts[:, 1], color=COLOR_OF[_m],
                      ls=STYLE_OF[_m], lw=1.1, marker="o", ms=2.6, mew=0)
        _axx.set_xticks([1, 2, 3])
        _axx.set_xlabel("epoch")
        _axx.set_ylabel(_lab)
        _axx.set_title(_ttl, loc="left", fontsize=8.6)

    _fig2.tight_layout(pad=0.35, w_pad=1.1)
    save_fig(_fig2, "fig_sft_protocol_repair")

    print("reported-era eval keys:",
          sorted({k for _m in sft_logs["reported"]
                  for e in sft_logs["reported"][_m][1]["log_history"]
                  for k in e if k.startswith("eval_")}) or "NONE")
    _fig2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure — does more SFT help on the benchmarks?

    The held-out curves above are computed on withheld examples of the **same
    extraction task in the same output format**, so they can detect divergence
    but not format contamination: a model that over-commits to the SFT output
    shape scores better there while doing worse on the ReAct / multiple-choice
    / Likert benchmarks. This figure is the out-of-format counterpart. Epoch 0
    is the pre-SFT instruct checkpoint each adapter was trained from, so each
    trace is a true longitudinal series over the same weights.

    Shaded bands are the measured re-run noise floor for that metric (median
    per-cell standard deviation from the judge-free N=3 variance sweep), so a
    movement inside the band is not a training effect.
    """)
    return


@app.cell
def _(COLOR_OF, LABEL, LONGITUDINAL, MODEL_ORDER, np, pd):
    cells = pd.read_parquet(LONGITUDINAL / "cells.parquet")
    noise = pd.read_parquet(LONGITUDINAL / "variance_noise_floor.parquet")

    # Six metrics with complete 11x4 coverage, spanning all four benchmark
    # families plus the capability control.
    METRICS = [
        ("gc_appl", "GoldCoin Appl.", False),
        ("gc_comp", "GoldCoin Comp.", False),
        ("pl_qa", "PrivacyLens QA", False),
        ("pl_adjlk", "PrivacyLens Leak", True),
        ("ca_2b", "ConfAIde $r$ (2b)", False),
        ("mmlu", "MMLU", False),
    ]
    SLUG_OF = {LABEL[m]: m for m in MODEL_ORDER}

    _c = cells[cells["metric"].isin([m for m, _, _ in METRICS])].copy()
    _c["slug"] = _c["model"].map(SLUG_OF)
    traj = (_c.pivot_table(index=["slug", "metric"], columns="epoch",
                           values="value", aggfunc="first")
              .reset_index())
    # Express every metric as a change from epoch 0, in points, so six panels
    # with different natural ranges share one readable y-axis meaning.
    for _e in (1, 2, 3):
        traj[f"d{_e}"] = 100.0 * (traj[_e] - traj[0])

    NOISE_OF = dict(zip(noise["metric"], noise["median_std"]))
    print("metrics x models with a complete epoch-0..3 series:")
    print(traj.groupby("metric")[[0, 1, 2, 3]].count().to_string())
    print("\nnoise floor (median per-cell sd, points):",
          {m: round(NOISE_OF.get(m, float('nan')), 2) for m, _, _ in METRICS})
    _missing = traj[traj[[0, 1, 2, 3]].isna().any(axis=1)]
    if len(_missing):
        print(f"\nincomplete series dropped from the figure: {len(_missing)}")
        print(_missing[["slug", "metric"]].to_string(index=False))
    return METRICS, NOISE_OF, traj


@app.cell
def _(
    COLOR_OF,
    METRICS,
    NOISE_OF,
    STYLE_OF,
    TEXTWIDTH_IN,
    np,
    plt,
    save_fig,
    traj,
):
    _fig3, _ax3 = plt.subplots(2, 3, figsize=(TEXTWIDTH_IN, 3.1), sharex=True)
    _p3 = _ax3.ravel()

    for _i, (_metric, _title, _lower_better) in enumerate(METRICS):
        _axx = _p3[_i]
        _sub = traj[traj["metric"] == _metric].dropna(subset=[0, 1, 2, 3])
        _band = NOISE_OF.get(_metric)
        # The variance record is judge-free, so the PrivacyLens metrics have no
        # measured floor. Say so in the panel rather than leaving a bandless
        # panel to read as "no noise".
        if _band is not None and not np.isnan(_band):
            # Edge lines as well as the span: at these y-ranges the band is
            # only a few percent of the axis and would otherwise be invisible.
            _axx.axhspan(-_band, _band, color="#333333", alpha=0.10, lw=0,
                         zorder=0)
            for _e in (-_band, _band):
                _axx.axhline(_e, color="#777777", ls=(0, (1, 2)), lw=0.6,
                             zorder=1)
        else:
            _axx.text(0.5, 0.04, "noise floor not measured",
                      transform=_axx.transAxes, ha="center", va="bottom",
                      fontsize=5.6, color="#777777", style="italic")
        _axx.axhline(0, color="#333333", lw=0.6, zorder=1)
        _M = []
        for _, _r in _sub.iterrows():
            _y = [0.0, _r["d1"], _r["d2"], _r["d3"]]
            _M.append(_y)
            _axx.plot([0, 1, 2, 3], _y, color=COLOR_OF[_r["slug"]],
                      ls=STYLE_OF[_r["slug"]], lw=0.7, alpha=0.45, zorder=2)
        if _M:
            _axx.plot([0, 1, 2, 3], np.median(np.array(_M), axis=0),
                      color="#111111", lw=1.8, zorder=5)
        _axx.set_title(f"({chr(97 + _i)}) {_title}"
                       + (r" $\downarrow$" if _lower_better else ""),
                       loc="left", fontsize=8.0)
        _axx.set_xticks([0, 1, 2, 3])
        if _i >= 3:
            _axx.set_xlabel("SFT epoch")
        if _i % 3 == 0:
            _axx.set_ylabel("$\\Delta$ vs. epoch 0 (pts)", fontsize=8)
        _axx.margins(x=0.04)

    _fig3.tight_layout(pad=0.35, w_pad=1.0, h_pad=0.6)
    save_fig(_fig3, "fig_sft_epoch_trajectory")
    _fig3
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## Table — per-model SFT training summary (reported sweep)""")
    return


@app.cell
def _(
    LABEL,
    MODEL_ORDER,
    np,
    pd,
    save_table,
    sft_logs,
    write_table_tex,
):
    _rows = []
    for _m in MODEL_ORDER:
        _got = sft_logs["reported"].get(_m)
        if _got is None:
            continue
        _d, _st = _got
        _gn = _d["grad_norm"] if "grad_norm" in _d else pd.Series(dtype=float)
        _rows.append({
            "model": LABEL[_m],
            "steps": int(_d["step"].max()),
            "epochs": round(float(_st.get("num_train_epochs", np.nan)), 1),
            "loss_first": round(float(_d["loss"].iloc[0]), 3),
            "loss_last": round(float(_d["loss"].iloc[-1]), 3),
            "tok_acc_last": round(float(_d["mean_token_accuracy"].iloc[-1]), 3)
            if "mean_token_accuracy" in _d else None,
            "gn_median": round(float(_gn.median()), 3) if len(_gn) else None,
            "gn_max": round(float(_gn.max()), 1) if len(_gn) else None,
            "gn_clipped": int((_gn > 1.0).sum()) if len(_gn) else None,
            "gn_clipped_frac": round(float((_gn > 1.0).mean()), 3) if len(_gn) else None,
            "n_logged": len(_d),
        })
    sft_summary = pd.DataFrame(_rows)
    save_table(sft_summary, "sft_training_summary")

    def _n(v, fmt):
        return "--" if v is None or (isinstance(v, float) and np.isnan(v)) \
            else format(v, fmt)

    _tex = "\n".join(
        f"{r['model']} & {r['steps']} & {r['loss_first']:.2f} & "
        f"{r['loss_last']:.2f} & {_n(r['tok_acc_last'], '.3f')} & "
        f"{_n(r['gn_median'], '.2f')} & {_n(r['gn_max'], '.1f')} & "
        f"{r['gn_clipped']}/{r['n_logged']} \\\\"
        for _, r in sft_summary.iterrows())

    write_table_tex(
        "sft_training_summary",
        colspec="@{}lrrrrrrr@{}",
        header=(r"& & \multicolumn{2}{c}{loss} & tok. & "
                r"\multicolumn{3}{c}{gradient norm} \\" "\n"
                r"\cmidrule(lr){3-4}\cmidrule(l){6-8}" "\n"
                r"model & steps & first & last & acc. & median & max & "
                r"clipped \\"),
        rows=_tex,
        label="tab:sft-training",
        caption=(
            r"Supervised fine-tuning summary for the reported sweep, one row "
            r"per model, under the hyperparameters of "
            r"\autoref{app:sft-hyperparams}: three epochs at an effective "
            r"batch size of 16, with gradients clipped at a norm of $1.0$. "
            r"Ten of the eleven models complete 540 optimizer steps, i.e.\ "
            r"2{,}880 training examples per epoch; GPT-OSS-20B trains under a "
            r"separate launcher with a 4{,}096-token sequence budget rather "
            r"than 8{,}192 and completes 270. "
            r"\textit{first} and \textit{last} are the first and last logged "
            r"training loss; \textit{tok.\ acc.}\ is the final mean "
            r"next-token accuracy; \textit{clipped} counts logged steps whose "
            r"gradient norm exceeded the clipping threshold. The three "
            r"Gemma-4 rows are the ones affected by the chat-template defect "
            r"of \autoref{app:sft-dynamics}, and they are exactly the three "
            r"rows whose gradient statistics separate from the rest."
        ),
    )
    print(sft_summary.to_string(index=False))
    return (sft_summary,)


@app.cell
def _(METRICS, NOISE_OF, np, sft_summary, traj):
    # Numbers the appendix prose quotes, printed together so the text can be
    # diffed against one block of output.
    print("--- reported sweep, training signal ---")
    _g4 = sft_summary[sft_summary["model"].str.startswith("Gemma-4")]
    _rest = sft_summary[~sft_summary["model"].str.startswith("Gemma-4")]
    print(f"initial loss   non-Gemma-4 {_rest['loss_first'].min():.2f}-"
          f"{_rest['loss_first'].max():.2f} | Gemma-4 "
          f"{_g4['loss_first'].min():.2f}-{_g4['loss_first'].max():.2f}")
    print(f"final loss     all {sft_summary['loss_last'].min():.2f}-"
          f"{sft_summary['loss_last'].max():.2f}")
    print(f"median |g|     non-Gemma-4 {_rest['gn_median'].min():.2f}-"
          f"{_rest['gn_median'].max():.2f} | Gemma-4 "
          f"{_g4['gn_median'].min():.2f}-{_g4['gn_median'].max():.2f}")
    print(f"max |g|        non-Gemma-4 <= {_rest['gn_max'].max():.1f} | "
          f"Gemma-4 up to {_g4['gn_max'].max():.1f}")
    print("clipped steps  " + ", ".join(
        f"{r['model']} {r['gn_clipped']}/{r['n_logged']}"
        for _, r in sft_summary.iterrows()))
    print(f"final token acc {sft_summary['tok_acc_last'].min():.3f}-"
          f"{sft_summary['tok_acc_last'].max():.3f}")

    print("\n--- epoch trajectory, median over models (points vs epoch 0) ---")
    for _metric, _title, _lb in METRICS:
        _s = traj[traj["metric"] == _metric].dropna(subset=[0, 1, 2, 3])
        _med = [float(np.median(_s[f"d{e}"])) for e in (1, 2, 3)]
        _band = NOISE_OF.get(_metric, float("nan"))
        if np.isnan(_band):
            _verdict = "noise floor NOT MEASURED"
        else:
            _verdict = (f"noise +/-{_band:.2f}, epoch 3 "
                        + ("inside noise" if abs(_med[-1]) <= _band
                           else "clears noise"))
        print(f"{_title:20s} e1 {_med[0]:+6.2f}  e2 {_med[1]:+6.2f}  "
              f"e3 {_med[2]:+6.2f}   ({_verdict})")
    return


if __name__ == "__main__":
    app.run()
