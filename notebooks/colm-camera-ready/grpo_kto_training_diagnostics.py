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
    # COLM camera-ready — GRPO / KTO training diagnostics

    Regenerates the Appendix B training-diagnostic figures for the **camera-ready
    models**, replacing the v9-era plots that `scripts/regen_v9_grpo_plots.py`
    produced (`figures/training_diagnostics.pdf`, `figures/completion_length.pdf`).

    Per the 2026-08-05 ruling (`wiki/2026-07-31_kto_plan.md` §19) the reported
    models are the **m2 `full`** GRPO cell and the **k3 `verdict`** KTO arm;
    v9-ckpt100 is deprecated. Both train from the *same* merged SFT
    (Qwen3.5-9B + the `sft-canonical` adapter of 2026-07-15), so they are two
    parallel post-SFT arms rather than a sequence, and neither curve should be
    read against a v9 one.

    ### Sources — everything here comes off disk, nothing is recomputed

    | What | Where |
    |---|---|
    | GRPO optimizer log | `2026-07-28_grpo_m2_full/…/checkpoint/checkpoint-450/trainer_state.json` (45 entries, steps 10–450, 3.0 epochs) |
    | GRPO per-completion trace | `…/checkpoint/reward_traces.jsonl` (14,400 rows over 450 reward calls) |
    | KTO optimizer log | `2026-08-01_k3_arms_b/18-55-02/1/kto_only/…/checkpoint-627/trainer_state.json` (62 entries, steps 10–627, 1.00 epoch) |

    The reward trace is 92 MB, almost all of it the verbatim prompt text. The
    loader drops `prompt_key` / `completion_text_sample` per line, so the frame
    in memory is small; nothing downstream needs the prompt.

    ### What the trace can and cannot say

    `reward_traces.jsonl` records **at most the first 8 completions of each
    reward call** (the writer's `i < 8` cap), not all G=8×4 prompts per step.
    Rates computed from it are therefore over a *sample* of each call, which is
    fine for trends and mixes but means counts are not run totals. The
    optimizer log is complete and is used wherever a quantity exists in both.

    Two panels deliberately show flat or unfavourable curves. The paper reports
    a checkpoint that fails two of its five promotion gates, and these figures
    are where a reader can see why: reward moves +0.006 against a 0.02 bar, and
    the minority class never learns. Smoothing them into a nicer shape would be
    the kind of thing the gates exist to catch.
    """)
    return


@app.cell
def _():
    import json
    from pathlib import Path

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    PROJECT_ROOT = Path("/share/pierson/matt/UAIR")
    NB_DIR = PROJECT_ROOT / "notebooks/colm-camera-ready"
    FIG_DIR = NB_DIR / "figures/training_diagnostics"
    TAB_DIR = NB_DIR / "tables/training_diagnostics"
    for _d in (FIG_DIR, TAB_DIR):
        _d.mkdir(parents=True, exist_ok=True)
    PAPER_FIG_DIR = PROJECT_ROOT / "papers/colm26_normative-simulacra/figures"

    # --- the two reported runs (wiki/2026-07-31_kto_plan.md §19) -------------
    GRPO_RUN = (PROJECT_ROOT / "multirun/2026-07-28_grpo_m2_full/21-31-11/"
                "cell=full/grpo_only_online_external/outputs/grpo/checkpoint")
    GRPO_STATE = GRPO_RUN / "checkpoint-450/trainer_state.json"
    GRPO_TRACE = GRPO_RUN / "reward_traces.jsonl"
    GRPO_GATES = GRPO_RUN / "promotion_gates.json"
    # checkpoint-627 = max_steps = a full epoch, and it is the state behind the
    # top-level adapter that `qwen3.5-9b/k3-verdict.yaml` loads. Do NOT read
    # checkpoint-504: it is a mid-run save (epoch 0.80) whose log stops 123
    # steps early, which silently truncates every KTO curve.
    KTO_STATE = (PROJECT_ROOT / "multirun/2026-08-01_k3_arms_b/18-55-02/1/"
                 "kto_only/outputs/kto/checkpoint/checkpoint-627/trainer_state.json")

    # House style, matching the other camera-ready notebooks. `pad_inches=0`
    # for the same reason as norm_flow_embedding_space: bbox_inches="tight"
    # re-adds a 0.1in border that becomes unreclaimable margin in the PDF.
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

    # One palette for both arms so a colour means the same thing across figures.
    C_REWARD = "#1F77B4"
    C_KL = "#D62728"
    C_ACCENT = "#807DBA"
    C_WARN = "#E07B39"
    C_MUTED = "#8A8A8A"
    C_CHOSEN = "#2CA02C"
    C_REJECTED = "#C0392B"

    # COLM \textwidth. Authoring at the final width means the point sizes above
    # are the sizes on paper and LaTeX applies (almost) no scaling.
    TEXTWIDTH_IN = 5.5

    def save_fig(fig, name, also_paper=True, pad_inches=0.0):
        for ext in ("png", "pdf"):
            fig.savefig(FIG_DIR / f"{name}.{ext}", dpi=300,
                        bbox_inches="tight", pad_inches=pad_inches)
        if also_paper:
            fig.savefig(PAPER_FIG_DIR / f"{name}.pdf", dpi=300,
                        bbox_inches="tight", pad_inches=pad_inches)
            print(f"[paper] {PAPER_FIG_DIR / name}.pdf")
        print(f"[fig]   {FIG_DIR / name}.png|.pdf")

    def save_table(df, name, index=False):
        out = TAB_DIR / f"{name}.csv"
        df.to_csv(out, index=index)
        print(f"[table] {out}")

    PAPER_TAB_DIR = PROJECT_ROOT / "papers/colm26_normative-simulacra/tables"

    def write_table_tex(name, *, colspec, header, rows, caption, label,
                        tabcolsep="3.2pt"):
        """Emit a complete \\input-able LaTeX table to the notebook and the paper.

        A *complete* table, not a row fragment: `\\input`-ing bare rows into an
        open `tabular` breaks, because the last row's `\\\\` scans past the end
        of the fragment and expands the following `\\bottomrule` (a `\\noalign`)
        where alignment material is illegal. Emitting the environment keeps the
        file boundary outside the alignment, and matches how every other table
        in this paper is wired (`tables/corpus_scaling.tex` et al.).
        """
        body = "\n".join([
            r"\begin{table}[ht]", r"\centering", r"\small",
            rf"\setlength{{\tabcolsep}}{{{tabcolsep}}}",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            rf"\begin{{tabular}}{{{colspec}}}",
            r"\toprule", header, r"\midrule", rows, r"\bottomrule",
            r"\end{tabular}", r"\end{table}",
        ])
        for d in (TAB_DIR, PAPER_TAB_DIR):
            (d / f"{name}.tex").write_text(body + "\n")
        print(f"[latex] {PAPER_TAB_DIR / name}.tex")

    return (
        C_ACCENT,
        C_CHOSEN,
        C_KL,
        C_MUTED,
        C_REJECTED,
        C_REWARD,
        C_WARN,
        GRPO_GATES,
        GRPO_STATE,
        GRPO_TRACE,
        KTO_STATE,
        Path,
        TAB_DIR,
        TEXTWIDTH_IN,
        json,
        np,
        pd,
        plt,
        write_table_tex,
        save_fig,
        save_table,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## Load — optimizer logs and the per-completion reward trace""")
    return


@app.cell
def _(GRPO_GATES, GRPO_STATE, GRPO_TRACE, KTO_STATE, json, pd):
    def _log_frame(state_path):
        state = json.loads(state_path.read_text())
        df = pd.DataFrame(state["log_history"])
        return df[df["step"].notna()].sort_values("step").reset_index(drop=True), state

    grpo_log, grpo_state = _log_frame(GRPO_STATE)
    kto_log, kto_state = _log_frame(KTO_STATE)

    # The trace's prompt text is ~99% of the 92 MB file and nothing here needs
    # it; dropping per line keeps the frame at a few MB.
    _DROP = ("prompt_key", "completion_text_sample", "completion")

    def _load_trace(path):
        rows = []
        with open(path) as fh:
            for line in fh:
                r = json.loads(line)
                for k in _DROP:
                    r.pop(k, None)
                rows.append(r)
        return pd.DataFrame(rows)

    trace = _load_trace(GRPO_TRACE)
    gates = json.loads(GRPO_GATES.read_text())

    print(f"GRPO log   {len(grpo_log):3d} entries | steps "
          f"{int(grpo_log['step'].min())}-{int(grpo_log['step'].max())} | "
          f"{grpo_state['epoch']:.2f} epochs")
    print(f"KTO  log   {len(kto_log):3d} entries | steps "
          f"{int(kto_log['step'].min())}-{int(kto_log['step'].max())} | "
          f"{kto_state['epoch']:.2f} epochs")
    print(f"GRPO trace {len(trace):,} rows over {trace['call'].nunique()} reward calls")
    print(f"           routes: {trace['route'].value_counts().to_dict()}")
    print(f"GRPO promotion verdict: promote={gates['promote']} | "
          + ", ".join(f"{k}={v['status']}" for k, v in gates["gates"].items()))
    return gates, grpo_log, kto_log, trace


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Derive — per-call rates from the trace

    Binned by reward call (one call ≈ one optimizer step here: 450 calls, 450
    steps). Rates are over the sampled completions in each call, so they carry
    sampling noise at the per-call level; every panel plots a rolling mean over
    15 calls on top of the raw series rather than replacing it, so the reader
    sees both the trend and its scatter.
    """)
    return


@app.cell
def _(np, pd, trace):
    WINDOW = 15

    def _roll(s, window=WINDOW):
        return s.rolling(window, min_periods=1, center=True).mean()

    _extract = trace[trace["task_type"] == "extract"].copy()
    _vig = trace[trace["task_type"] == "vignette"].copy()

    # --- per-call extract-task rates ---------------------------------------
    _g = _extract.groupby("call")
    per_call = pd.DataFrame({
        "n_extract": _g.size(),
        "reward_mean": _g["score"].mean(),
        # `no_flow` is recorded only on abstain-table rows; True = the policy
        # declared no information flow. Rate is over all extract completions.
        "no_flow_rate": _g.apply(
            lambda d: (d["no_flow"] == True).sum() / max(len(d), 1),  # noqa: E712
            include_groups=False),
        "gate_fail_rate": _g.apply(
            lambda d: (d["route"] == "gate_fail").sum() / max(len(d), 1),
            include_groups=False),
        "outcome_term": _g["outcome_term"].mean(),
    })
    _aux = _extract.dropna(subset=["aux_terms"])
    for _k in ("ground", "contrast"):
        per_call[f"aux_{_k}"] = (
            _aux.assign(v=_aux["aux_terms"].map(lambda d: d.get(_k)))
            .groupby("call")["v"].mean()
        )

    # --- R-DIRECT per-class label agreement --------------------------------
    # The claim the whole method rests on. `direct_flows` is a per-flow list of
    # {gold, pred, sim}; a matched flow has pred not None. Accuracy is computed
    # per gold class so the corpus's ~4:1 appropriate skew cannot flatter it.
    def _direct_rows(df):
        out = []
        for call, flows in zip(df["call"], df["direct_flows"]):
            if not isinstance(flows, list):
                continue
            for f in flows:
                if f.get("pred") is None:
                    continue
                out.append((call, f["gold"], f["pred"] == f["gold"]))
        return pd.DataFrame(out, columns=["call", "gold", "correct"])

    direct = _direct_rows(_extract)
    direct_by_call = (
        direct.pivot_table(index="call", columns="gold", values="correct",
                           aggfunc="mean")
        .rename(columns={"appropriate": "acc_appropriate",
                         "inappropriate": "acc_inappropriate"})
    )

    # --- vignette battery --------------------------------------------------
    _vr = _vig.dropna(subset=["vig_result"])
    vig_by_call = pd.DataFrame({
        k: _vr.assign(v=_vr["vig_result"].map(lambda d: d.get(k)))
             .groupby("call")["v"].mean()
        for k in ("battery", "hedge_frac", "antithesis_frac", "parsed_frac")
    })

    diag = per_call.join(direct_by_call).join(vig_by_call).reset_index()
    for _c in [c for c in diag.columns if c != "call"]:
        diag[f"{_c}_s"] = _roll(diag[_c])

    def _third(col, which):
        v = diag[col].dropna().to_numpy()
        k = max(1, len(v) // 3)
        return float(np.mean(v[:k] if which == "first" else v[-k:]))

    print(f"reward       first third {_third('reward_mean','first'):.4f} "
          f"-> last third {_third('reward_mean','last'):.4f} "
          f"(gain {_third('reward_mean','last') - _third('reward_mean','first'):+.4f})")
    print(f"no-flow rate first third {_third('no_flow_rate','first'):.4f} "
          f"-> last third {_third('no_flow_rate','last'):.4f}")
    print(f"gate-fail    first third {_third('gate_fail_rate','first'):.4f} "
          f"-> last third {_third('gate_fail_rate','last'):.4f}")
    for _c in ("acc_appropriate", "acc_inappropriate"):
        print(f"{_c:18s} first third {_third(_c,'first'):.4f} "
              f"-> last third {_third(_c,'last'):.4f}")
    print(f"vignette hedge_frac first {_third('hedge_frac','first'):.4f} "
          f"-> last {_third('hedge_frac','last'):.4f}")
    print(f"\nmatched flow judgments: {len(direct):,} "
          f"({direct['gold'].value_counts(normalize=True).round(3).to_dict()})")
    return WINDOW, diag, direct


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure — GRPO training diagnostics

    Four panels, chosen so a reader can check the two failed promotion gates
    directly rather than take the appendix's word for it: reward trend (gate a),
    zero-variance group fraction (gate b) with KL (gate c), the abstention rate
    (gate d), and the per-class discrimination (gate e).
    """)
    return


@app.cell
def _(
    C_ACCENT,
    C_KL,
    C_MUTED,
    C_REWARD,
    C_WARN,
    TEXTWIDTH_IN,
    diag,
    grpo_log,
    plt,
    save_fig,
):
    _fig, _axes = plt.subplots(2, 2, figsize=(TEXTWIDTH_IN, 3.9), sharex=True)
    (_a, _b), (_c, _d) = _axes

    # (a) reward, from the optimizer log (complete) with its own std band.
    _a.fill_between(grpo_log["step"],
                    grpo_log["reward"] - grpo_log["reward_std"],
                    grpo_log["reward"] + grpo_log["reward_std"],
                    color=C_REWARD, alpha=0.15, lw=0)
    _a.plot(grpo_log["step"], grpo_log["reward"], color=C_REWARD, lw=1.4)
    _a.set_ylabel("composite reward")
    _a.set_title("(a) composite reward", loc="left")

    # (b) the advantage carrier and the KL anchor share a panel: both are
    # "is the optimizer healthy" questions and both stayed well inside bounds.
    _b.plot(grpo_log["step"], grpo_log["frac_reward_zero_std"],
            color=C_ACCENT, lw=1.4, label="zero-variance groups")
    _b.axhline(0.2, color=C_ACCENT, ls=":", lw=0.9)
    _b.set_ylabel("frac. zero-var. groups", color=C_ACCENT)
    _b.tick_params(axis="y", labelcolor=C_ACCENT)
    _bk = _b.twinx()
    _bk.plot(grpo_log["step"], grpo_log["kl"], color=C_KL, lw=1.4)
    _bk.set_yscale("log")
    _bk.tick_params(axis="y", which="minor", length=1.5, colors=C_KL)
    _bk.set_ylabel(r"KL to SFT ref. (log)", color=C_KL)
    _bk.tick_params(axis="y", labelcolor=C_KL)
    _bk.grid(False)
    _bk.spines["top"].set_visible(False)
    _b.set_title("(b) group spread, KL", loc="left")

    # (c) abstention. Dotted line is the gold no-flow base rate of the final
    # training set; gate (d) asks that the policy stay within +/-0.15 of it.
    _c.plot(diag["call"], diag["no_flow_rate"], color=C_MUTED, lw=0.6, alpha=0.5)
    _c.plot(diag["call"], diag["no_flow_rate_s"], color=C_WARN, lw=1.5)
    _c.axhline(0.0447, color="#333333", ls=":", lw=0.9)
    _c.set_ylabel("no-flow declared")
    _c.set_xlabel("training step")
    _c.set_title("(c) abstention", loc="left")

    # (d) the result that matters, and the one that did not move.
    for _col, _colr, _lab in (
        ("acc_appropriate_s", "#2C7FB8", "appropriate"),
        ("acc_inappropriate_s", "#C0392B", "inappropriate"),
    ):
        _d.plot(diag["call"], diag[_col], color=_colr, lw=1.5, label=_lab)
    _d.axhline(0.5, color="#333333", ls=":", lw=0.9)
    _d.set_ylim(0, 1)
    _d.set_ylabel("label agreement")
    _d.set_xlabel("training step")
    _d.set_title("(d) agreement by gold class", loc="left")
    _d.legend(frameon=False, loc="center", fontsize=7, ncol=2,
              handlelength=1.4, columnspacing=1.0)

    for _ax in _axes.ravel():
        _ax.margins(x=0.01)
    _fig.tight_layout(pad=0.35, w_pad=1.1, h_pad=0.7)
    save_fig(_fig, "fig_grpo_training_diagnostics")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure — completion length

    Replaces the v9 `completion_length.pdf`. Lengths are in **tokens** here (the
    trainer logs `completions/*_length` in tokens); the v9 figure's caption said
    characters, which was true of that script's trace-derived measure but is not
    what this axis is. Terminated-only means are drawn alongside the all-completion
    means so a rise from truncation cannot be mistaken for a rise from verbosity.
    """)
    return


@app.cell
def _(C_MUTED, C_REWARD, C_WARN, TEXTWIDTH_IN, grpo_log, plt, save_fig):
    _fig2, (_a2, _b2) = plt.subplots(
        1, 2, figsize=(TEXTWIDTH_IN, 1.9), sharex=True,
        gridspec_kw={"width_ratios": [1.35, 1]})

    _a2.fill_between(grpo_log["step"], grpo_log["completions/min_length"],
                     grpo_log["completions/max_length"],
                     color=C_MUTED, alpha=0.18, lw=0, label="min-max")
    _a2.plot(grpo_log["step"], grpo_log["completions/mean_length"],
             color=C_REWARD, lw=1.5, label="mean")
    _a2.plot(grpo_log["step"], grpo_log["completions/mean_terminated_length"],
             color=C_WARN, lw=1.2, ls="--", label="mean (terminated)")
    _a2.set_ylabel("completion length (tokens)")
    _a2.set_xlabel("training step")
    _a2.set_title("(a) length stays bounded", loc="left")
    _a2.legend(frameon=False, fontsize=7, loc="upper left")

    _b2.plot(grpo_log["step"], 100 * grpo_log["completions/clipped_ratio"],
             color=C_WARN, lw=1.5)
    _b2.set_ylabel("truncated (%)")
    _b2.set_xlabel("training step")
    _b2.set_title("(b) truncation", loc="left")

    for _ax in (_a2, _b2):
        _ax.margins(x=0.01)
    _fig2.tight_layout(pad=0.35, w_pad=1.1)
    save_fig(_fig2, "fig_grpo_completion_length")
    _fig2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure — KTO training diagnostics

    The k3 `verdict` arm. KTO's own signal is the implicit reward on desirable
    vs undesirable completions and the margin between them, so those replace the
    composite-reward panel; loss and KL carry over.
    """)
    return


@app.cell
def _(
    C_CHOSEN,
    C_KL,
    C_MUTED,
    C_REJECTED,
    TEXTWIDTH_IN,
    kto_log,
    plt,
    save_fig,
):
    _fig3, (_a3, _b3, _c3) = plt.subplots(1, 3, figsize=(TEXTWIDTH_IN, 1.95))

    _a3.plot(kto_log["step"], kto_log["loss"], color=C_MUTED, lw=1.5)
    _a3.set_ylabel("KTO loss")
    _a3.set_xlabel("training step")
    _a3.set_title("(a) loss", loc="left")

    _b3.axhline(0, color="#333333", lw=0.7)
    _b3.plot(kto_log["step"], kto_log["rewards/chosen"],
             color=C_CHOSEN, lw=1.5, label="desirable")
    _b3.plot(kto_log["step"], kto_log["rewards/rejected"],
             color=C_REJECTED, lw=1.5, label="undesirable")
    _b3.plot(kto_log["step"], kto_log["rewards/margins"],
             color="#333333", lw=1.2, ls="--", label="margin")
    _b3.set_ylim(top=float(kto_log["rewards/margins"].max()) * 2.05)
    _b3.set_ylabel("implicit reward")
    _b3.set_xlabel("training step")
    _b3.set_title("(b) implicit reward", loc="left")
    _b3.legend(frameon=False, fontsize=6.2, loc="upper left",
               handlelength=1.2, labelspacing=0.25, borderpad=0.1)

    _c3.plot(kto_log["step"], kto_log["kl"], color=C_KL, lw=1.5)
    _c3.set_ylabel("KL to reference")
    _c3.set_xlabel("training step")
    _c3.set_title("(c) KL to ref.", loc="left")

    for _ax in (_a3, _b3, _c3):
        _ax.margins(x=0.01)
    _fig3.tight_layout(pad=0.35, w_pad=1.2)
    save_fig(_fig3, "fig_kto_training_diagnostics")
    _fig3
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Numbers for the captions

    Written to `tables/training_diagnostics/` so the appendix quotes values that
    trace to a file rather than to a reading of the plot.
    """)
    return


@app.cell
def _(diag, direct, gates, grpo_log, kto_log, np, pd, save_table):
    def _ends(series):
        v = pd.Series(series).dropna().to_numpy()
        k = max(1, len(v) // 3)
        return float(np.mean(v[:k])), float(np.mean(v[-k:]))

    _rows = []
    for _arm, _name, _series in (
        ("GRPO", "reward", grpo_log["reward"]),
        ("GRPO", "kl", grpo_log["kl"]),
        ("GRPO", "frac_reward_zero_std", grpo_log["frac_reward_zero_std"]),
        ("GRPO", "completion_len_tokens", grpo_log["completions/mean_length"]),
        ("GRPO", "truncated_frac", grpo_log["completions/clipped_ratio"]),
        ("GRPO", "no_flow_rate", diag["no_flow_rate"]),
        ("GRPO", "gate_fail_rate", diag["gate_fail_rate"]),
        ("GRPO", "acc_appropriate", diag["acc_appropriate"]),
        ("GRPO", "acc_inappropriate", diag["acc_inappropriate"]),
        ("GRPO", "vignette_battery", diag["battery"]),
        ("GRPO", "vignette_hedge_frac", diag["hedge_frac"]),
        ("KTO", "loss", kto_log["loss"]),
        ("KTO", "rewards/chosen", kto_log["rewards/chosen"]),
        ("KTO", "rewards/rejected", kto_log["rewards/rejected"]),
        ("KTO", "rewards/margins", kto_log["rewards/margins"]),
        ("KTO", "kl", kto_log["kl"]),
    ):
        _f, _l = _ends(_series)
        _rows.append({"arm": _arm, "metric": _name, "first_third": round(_f, 4),
                      "last_third": round(_l, 4), "delta": round(_l - _f, 4)})
    summary = pd.DataFrame(_rows)
    save_table(summary, "training_diagnostics_summary")

    _gate = pd.DataFrame([
        {"gate": k, "status": v["status"],
         "value": v.get("gain", v.get("mean_frac_zero_std", v.get("mean_kl",
                  v.get("deviation", v.get("youden_j"))))),
         "threshold": v.get("threshold", v.get("tolerance"))}
        for k, v in gates["gates"].items()
    ])
    save_table(_gate, "grpo_promotion_gates")

    print(f"matched flow judgments in trace: {len(direct):,}")
    print(f"promote = {gates['promote']}\n")
    print(summary.to_string(index=False))
    print()
    print(_gate.to_string(index=False))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Part 2 — every arm, not just the reported one

    Everything above describes the two *reported* checkpoints. This half covers
    the **full m2 GRPO grid (4 cells)** and the **full k3 KTO ladder (4 arms)**,
    which is what Appendix B's `app:rl-dynamics` reports. Three things drove the
    extension:

    1. `entropy` and `grad_norm` are logged by both trainers and were never
       plotted. Entropy collapse is the canonical RLVR failure mode; ours does
       not collapse, and that is the load-bearing evidence that the flat reward
       is a *signal* failure and not an *optimization* failure.
    2. The mean-KL gate mischaracterizes two of the four GRPO cells: `core` and
       `-outcome` fail it on a first-50-step transient that decays to ~0.01. A
       log-axis curve plus a median column settles it.
    3. `outputs/2026-08-02_k3_probe/` is a **held-out** measurement — the only
       generalization evidence for the KTO arms — and no figure used it.

    Reward levels are **not** comparable across GRPO cells (each cell
    renormalizes its weights), so F1(a) plots the *gain* from each cell's own
    first-third mean. That is also exactly the quantity the `reward_trend` gate
    thresholds, so the gate is readable straight off the panel. Absolute levels
    live in T1.
    """)
    return


@app.cell
def _(Path, json, pd):
    PROJECT_ROOT_2 = Path("/share/pierson/matt/UAIR")
    MULTIRUN = PROJECT_ROOT_2 / "multirun"
    CACHE_DIR = PROJECT_ROOT_2 / "notebooks/colm-camera-ready/cache"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # --- the m2 grid (conf/sweep/grpo_m2_grid.yaml). Cells differ ONLY in
    # auxiliaries x direct-core toggle x vignette mix; everything else is the
    # shared m_series preset, so each row is a clean leave-one-out.
    _GRPO_SUB = "grpo_only_online_external/outputs/grpo/checkpoint"
    GRPO_ARMS = {
        "full": {
            "dir": MULTIRUN / "2026-07-28_grpo_m2_full/21-31-11/cell=full" / _GRPO_SUB,
            "aux": "ground, contrast", "core": True, "vig": 0.18,
            "color": "#0072B2", "z": 5, "label": "Full",
        },
        "core": {
            "dir": MULTIRUN / "2026-07-28_grpo_m2_core/21-31-11/cell=core" / _GRPO_SUB,
            "aux": "--", "core": True, "vig": 0.18,
            "color": "#E69F00", "z": 4, "label": "\u2212aux",
        },
        "-outcome": {
            "dir": MULTIRUN / "2026-07-28_grpo_m2_-outcome/21-31-11/cell=minus_outcome" / _GRPO_SUB,
            "aux": "ground, contrast", "core": False, "vig": 0.18,
            "color": "#009E73", "z": 3, "label": "\u2212core",
        },
        "-vignette": {
            "dir": MULTIRUN / "2026-07-28_grpo_m2_-vignette/22-00-49/cell=minus_vignette" / _GRPO_SUB,
            "aux": "ground, contrast", "core": True, "vig": 0.0,
            "color": "#CC79A7", "z": 2, "label": "\u2212judg",
        },
    }

    # --- the k3 ladder (wiki/2026-07-31_kto_plan.md 7). Colour encodes
    # supervision depth (light -> dark), so the ladder ordering is readable
    # without consulting the legend; sft_ctrl is the non-KTO control, in grey.
    KTO_ARMS = {
        "verdict": {"dir": MULTIRUN / "2026-08-01_k3_arms_b/18-55-02/1",
                    "depth": "label only", "color": "#6BAED6", "kto": True,
                    "label": "label"},
        "citation": {"dir": MULTIRUN / "2026-08-01_k3_arms/18-01-33/0",
                     "depth": "+ cited norm", "color": "#2171B5", "kto": True,
                     "label": "+ norm"},
        "scrutinize": {"dir": MULTIRUN / "2026-08-01_k3_arms_b/18-55-02/0",
                       "depth": "+ rationale", "color": "#08306B", "kto": True,
                       "label": "+ rationale"},
        "sft_ctrl": {"dir": MULTIRUN / "2026-08-01_k3_arms_b/18-55-02/2",
                     "depth": "SFT on desirables", "color": "#8A8A8A", "kto": False,
                     "label": "supervised"},
    }

    K3_PROBE = PROJECT_ROOT_2 / "outputs/2026-08-02_k3_probe/merged"

    def _latest_state(root):
        """Highest-numbered checkpoint under `root`.

        Resolving by max step matters: the KTO runs also save mid-run
        checkpoints (e.g. checkpoint-504 at epoch 0.80) whose log stops 123
        steps early and would silently truncate every curve.
        """
        cands = list(root.glob("**/checkpoint-*/trainer_state.json"))
        if not cands:
            raise FileNotFoundError(f"no trainer_state.json under {root}")
        return max(cands, key=lambda p: int(p.parent.name.split("-")[1]))

    def _log_frame_at(path):
        df = pd.DataFrame(json.loads(path.read_text())["log_history"])
        return df[df["step"].notna()].sort_values("step").reset_index(drop=True)

    grpo_logs, grpo_gates_all = {}, {}
    for _a, _m in GRPO_ARMS.items():
        grpo_logs[_a] = _log_frame_at(_latest_state(_m["dir"]))
        grpo_gates_all[_a] = json.loads((_m["dir"] / "promotion_gates.json").read_text())

    kto_logs = {a: _log_frame_at(_latest_state(m["dir"]))
                for a, m in KTO_ARMS.items()}

    for _a, _df in grpo_logs.items():
        print(f"GRPO {_a:10s} {len(_df):3d} entries | steps "
              f"{int(_df['step'].min())}-{int(_df['step'].max())} | "
              f"{_df['epoch'].max():.2f} ep | promote="
              f"{grpo_gates_all[_a]['promote']}")
    for _a, _df in kto_logs.items():
        print(f"KTO  {_a:10s} {len(_df):3d} entries | steps "
              f"{int(_df['step'].min())}-{int(_df['step'].max())} | "
              f"{_df['epoch'].max():.2f} ep")
    return (
        CACHE_DIR,
        GRPO_ARMS,
        KTO_ARMS,
        K3_PROBE,
        grpo_gates_all,
        grpo_logs,
        kto_logs,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Derive — per-class agreement for every GRPO cell

    Same `direct_flows` extraction as Part 1, now over all four traces (~360 MB
    of JSONL in total). Each arm's derived per-call frame is cached to parquet,
    so a re-run costs a few hundred ms instead of re-parsing every trace. The
    `-outcome` cell runs with the direct core **off** and therefore emits no
    `direct_flows` at all — it is legitimately absent from the discrimination
    figure, not dropped, and the panel says so.
    """)
    return


@app.cell
def _(CACHE_DIR, GRPO_ARMS, grpo_gates_all, json, np, pd):
    _COUNT_COLS =["n_appr", "n_appr_ok", "n_inappr", "n_inappr_ok"]

    def _direct_by_call(arm):
        """Per-call, per-gold-class judgment COUNTS for one GRPO cell.

        Counts rather than rates, because the two consumers need different
        estimators: the figure wants a per-call rate (a trend), while T1 must
        reproduce `promotion_gates.json`, which pools flow judgments over the
        trailing `j_trace_tail_calls=100` calls. Caching rates would silently
        force the table onto the figure's estimator and print a J that
        contradicts the gate file it is reporting.
        """
        cache = CACHE_DIR / f"grpo_direct_counts_{arm.replace('-', 'minus_')}.parquet"
        if cache.exists():
            return pd.read_parquet(cache)
        acc = {}
        with open(GRPO_ARMS[arm]["dir"] / "reward_traces.jsonl") as fh:
            for line in fh:
                r = json.loads(line)
                if r.get("task_type") != "extract":
                    continue
                flows = r.get("direct_flows")
                if not isinstance(flows, list):
                    continue
                c = acc.setdefault(r["call"], [0, 0, 0, 0])
                for f in flows:
                    if f.get("pred") is None:      # unmatched teacher flow
                        continue
                    i = 0 if f["gold"] == "appropriate" else 2
                    c[i] += 1
                    c[i + 1] += int(f["pred"] == f["gold"])
        if not acc:                                 # -outcome: core is off
            df = pd.DataFrame(columns=["call"] + _COUNT_COLS)
        else:
            df = (pd.DataFrame.from_dict(acc, orient="index",
                                         columns=_COUNT_COLS)
                    .rename_axis("call").sort_index().reset_index())
        df.to_parquet(cache, index=False)
        return df

    def _rates(df):
        """Per-call agreement rates + J, for the trend figure."""
        out = df[["call"]].copy()
        out["acc_appropriate"] = df["n_appr_ok"] / df["n_appr"].replace(0, np.nan)
        out["acc_inappropriate"] = (df["n_inappr_ok"]
                                    / df["n_inappr"].replace(0, np.nan))
        out["J"] = out["acc_appropriate"] + out["acc_inappropriate"] - 1.0
        return out

    def _pooled_tail(df, tail_calls=100):
        """Pooled recalls and J over the trailing calls — the gate's estimator.

        `j_trace_tail_calls=100` in every m2 `promotion_gates.json`; this
        reproduces those files exactly (verified for all three scored cells).
        """
        t = df.sort_values("call").tail(tail_calls)
        ra = t["n_appr_ok"].sum() / max(t["n_appr"].sum(), 1)
        ri = t["n_inappr_ok"].sum() / max(t["n_inappr"].sum(), 1)
        return float(ra), float(ri), float(ra + ri - 1.0)

    grpo_direct, grpo_direct_rates, grpo_tail_J = {}, {}, {}
    for _a in GRPO_ARMS:
        _d = _direct_by_call(_a)
        grpo_direct[_a] = _d
        if _d.empty:
            grpo_direct_rates[_a] = _d
            grpo_tail_J[_a] = (np.nan, np.nan, np.nan)
            print(f"{_a:10s} no direct_flows (reward_core off) — expected")
            continue
        grpo_direct_rates[_a] = _rates(_d)
        grpo_tail_J[_a] = _pooled_tail(_d)
        _ra, _ri, _J = grpo_tail_J[_a]
        _ref = grpo_gates_all[_a]["gates"]["direct_discrimination"]
        # Hard check: the table must agree with the file it reports.
        assert abs(_J - _ref["youden_j"]) < 5e-4, (_a, _J, _ref["youden_j"])
        print(f"{_a:10s} {len(_d):3d} calls | tail-100 pooled: appropriate "
              f"{_ra:.4f} inappropriate {_ri:.4f} J {_J:+.4f}  "
              f"(gate file {_ref['youden_j']:+.4f} — matches)")
    return grpo_direct_rates, grpo_tail_J


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## F1 — GRPO optimizer health across all four cells

    Six small multiples, one line per cell, one shared legend. The point of the
    figure is that the four standard RL pathologies — entropy collapse, length
    inflation, KL divergence, advantage collapse — **did not occur in any
    cell**, while reward moved by less than a third of the gate's bar in all
    four. Panel (a) is the gain, not the level, for the reason given above.
    """)
    return


@app.cell
def _(GRPO_ARMS, TEXTWIDTH_IN, grpo_logs, np, pd, plt, save_fig):
    def _gain(s):
        """Deviation from the cell's own first-third mean (the gated quantity)."""
        v = s.to_numpy(dtype=float)
        return v - float(np.mean(v[: max(1, len(v) // 3)]))

    def _band(ax, x, y, meta, window=7):
        """Raw series faint, rolling mean bold.

        Only for the three panels whose per-log-entry scatter is larger than
        the effect being read off them. KL and gradient norm are drawn raw:
        there the spikes ARE the signal, and smoothing would erase the
        transient that the KL gate trips on.
        """
        s = pd.Series(np.asarray(y, dtype=float))
        ax.plot(x, s, color=meta["color"], lw=0.55, alpha=0.30,
                zorder=meta["z"])
        ax.plot(x, s.rolling(window, min_periods=1, center=True).mean(),
                color=meta["color"], lw=1.4, zorder=meta["z"] + 10,
                label=meta["label"])

    _f1, _ax1 = plt.subplots(2, 3, figsize=(TEXTWIDTH_IN, 3.3), sharex=True)
    _p = _ax1.ravel()

    for _arm, _meta in GRPO_ARMS.items():
        _lg = grpo_logs[_arm]
        _m = _meta
        _band(_p[0], _lg["step"], _gain(_lg["reward"]), _m)
        _p[1].plot(_lg["step"], _lg["entropy"], color=_meta["color"], lw=1.3,
                   zorder=_meta["z"])
        _p[2].plot(_lg["step"], _lg["kl"], color=_meta["color"], lw=1.1,
                   zorder=_meta["z"])
        _p[3].plot(_lg["step"], _lg["grad_norm"], color=_meta["color"], lw=1.1,
                   zorder=_meta["z"])
        _band(_p[4], _lg["step"], _lg["frac_reward_zero_std"], _m)
        _band(_p[5], _lg["step"], _lg["completions/mean_length"], _m)

    # (a) the gate is on the gain, so draw the bar the gain is measured against.
    _p[0].axhspan(-0.02, 0.02, color="#333333", alpha=0.07, lw=0, zorder=0)
    _p[0].axhline(0.02, color="#333333", ls=":", lw=0.8, zorder=1)
    _p[0].axhline(0.0, color="#333333", lw=0.6, zorder=1)
    _p[0].set_ylabel("reward gain")
    _p[0].set_title("(a) reward gain", loc="left")
    _p[0].text(0.97, 0.025, r"$\pm$0.02 criterion", transform=_p[0].transAxes,
               ha="right", va="bottom", fontsize=6.2, color="#555555")

    _p[1].set_ylim(0, 0.55)
    _p[1].set_ylabel("policy entropy")
    _p[1].set_title("(b) entropy", loc="left")

    # (c) log axis: core / -outcome fail the mean-KL gate on a startup
    # transient that decays ~4 orders of magnitude. A linear axis hides this.
    _p[2].set_yscale("log")
    _p[2].axhline(1.0, color="#333333", ls=":", lw=0.8)
    _p[2].set_ylabel("KL to SFT ref.")
    _p[2].set_title("(c) KL, criterion 1.0", loc="left")

    _p[3].set_yscale("log")
    _p[3].set_ylabel("gradient norm")
    _p[3].set_xlabel("training step")
    _p[3].set_title("(d) gradient norm", loc="left")

    _p[4].axhline(0.2, color="#333333", ls=":", lw=0.8)
    _p[4].set_ylim(0, 0.22)
    _p[4].set_ylabel("zero-var. groups")
    _p[4].set_xlabel("training step")
    _p[4].set_title("(e) zero-var. groups", loc="left")
    _p[4].text(0.97, 0.188, "0.2 criterion", transform=_p[4].get_yaxis_transform(),
               ha="right", va="top", fontsize=6.2, color="#555555")

    _p[5].set_ylabel("length (tokens)")
    _p[5].set_xlabel("training step")
    _p[5].set_title("(f) completion length", loc="left")

    for _ax in _p:
        _ax.margins(x=0.01)
        _ax.title.set_fontsize(8.2)
    _h1, _l1 = _p[0].get_legend_handles_labels()
    _f1.legend(_h1, _l1, ncol=4, frameon=False, fontsize=7.5,
               loc="lower center", bbox_to_anchor=(0.5, -0.028),
               handlelength=1.6, columnspacing=1.8)
    _f1.tight_layout(pad=0.35, w_pad=0.9, h_pad=0.6, rect=(0, 0.05, 1, 1))
    save_fig(_f1, "fig_grpo_arms_health")
    _f1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## F2 — GRPO discrimination, all cells

    The claim the method rests on, per cell. Panels (a) and (b) share a y-axis
    on purpose: the whole result is that the two classes sit on opposite ends
    of it and neither moves. Panel (c) collapses the pair into Youden's J
    against the promotion gate.
    """)
    return


@app.cell
def _(GRPO_ARMS, TEXTWIDTH_IN, grpo_direct_rates, grpo_tail_J, np, plt, save_fig):
    _f2, _ax2 = plt.subplots(1, 3, figsize=(TEXTWIDTH_IN, 1.85))

    _WIN = 15

    def _sm(s):
        return s.rolling(_WIN, min_periods=1, center=True).mean()

    _absent = []
    for _arm, _meta in GRPO_ARMS.items():
        _d = grpo_direct_rates[_arm]
        if _d.empty:
            _absent.append(_meta["label"])
            continue
        _kw = dict(color=_meta["color"], lw=1.3, zorder=_meta["z"],
                   label=_meta["label"])
        _ax2[0].plot(_d["call"], _sm(_d["acc_appropriate"]), **_kw)
        _ax2[1].plot(_d["call"], _sm(_d["acc_inappropriate"]), **_kw)
        _ax2[2].plot(_d["call"], _sm(_d["J"]), **_kw)

    for _ax, _t in ((_ax2[0], "(a) appropriate (majority)"),
                    (_ax2[1], "(b) inappropriate (minority)")):
        _ax.set_ylim(0, 1)
        _ax.axhline(0.5, color="#333333", ls=":", lw=0.8)
        _ax.set_xlabel("training step")
        _ax.set_title(_t, loc="left", fontsize=8.2)
    _ax2[0].set_ylabel("label agreement")

    # (c) The gate reads a POOLED J over the trailing 100 calls, not the
    # per-call trend. Drawing it as a segment over the steps it is computed
    # from keeps the figure and T1 visibly the same number.
    for _arm, _meta in GRPO_ARMS.items():
        _tj = grpo_tail_J[_arm][2]
        if np.isnan(_tj):
            continue
        _ax2[2].plot([350, 450], [_tj, _tj], color=_meta["color"], lw=2.4,
                     solid_capstyle="butt", zorder=20)
    _ax2[2].axhline(0.05, color="#333333", ls=":", lw=0.8)
    _ax2[2].axhline(0.0, color="#333333", lw=0.6)
    _ax2[2].set_ylim(-0.16, 0.16)
    _ax2[2].set_ylabel("Youden's $J$")
    _ax2[2].set_xlabel("training step")
    _ax2[2].set_title("(c) Youden's $J$", loc="left", fontsize=8.2)
    _ax2[2].text(0.02, 0.053, "0.05 criterion",
                 transform=_ax2[2].get_yaxis_transform(),
                 ha="left", va="bottom", fontsize=6.0, color="#555555")
    _ax2[2].text(0.02, 0.965, "bars: pooled over the final 100 calls",
                 transform=_ax2[2].transAxes, ha="left", va="top",
                 fontsize=5.8, color="#555555", style="italic")

    # Say why a cell is missing rather than letting the reader count lines.
    if _absent:
        _ax2[1].text(0.5, 0.92, f"{', '.join(_absent)} omits $R_{{direct}}$,",
                     transform=_ax2[1].transAxes, ha="center", fontsize=5.8,
                     color="#555555", style="italic")
        _ax2[1].text(0.5, 0.83, "so nothing is scored",
                     transform=_ax2[1].transAxes, ha="center", fontsize=5.8,
                     color="#555555", style="italic")
    _ax2[0].legend(frameon=False, fontsize=6.4, loc="lower left", ncol=1,
                   handlelength=1.2, labelspacing=0.22, borderpad=0.1)

    for _ax in _ax2:
        _ax.margins(x=0.01)
    _f2.tight_layout(pad=0.35, w_pad=1.1)
    save_fig(_f2, "fig_grpo_arms_discrimination")
    _f2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## F4 — KTO ladder training dynamics, all four arms

    The three KTO arms are a **supervision-depth ladder**: `verdict` flips only
    the appropriateness enum, `citation` additionally corrects the cited norm,
    `scrutinize` additionally rewrites the reasoning trace. `sft_ctrl` is the
    mandatory control — plain SFT on the desirable rows only — and so has a
    loss and an entropy but no implicit reward and no KL (it appears in (a)
    only).

    The ladder orders **monotonically** in margin and in KL. That is a real
    dose-response *in the optimization*, and F5 shows it buys nothing on
    held-out data — which is precisely why both figures have to be printed.
    """)
    return


@app.cell
def _(KTO_ARMS, TEXTWIDTH_IN, kto_logs, plt, save_fig):
    _f4, _ax4 = plt.subplots(1, 4, figsize=(TEXTWIDTH_IN, 1.75))

    for _arm, _meta in KTO_ARMS.items():
        _lg, _c = kto_logs[_arm], _meta["color"]
        _ax4[0].plot(_lg["step"], _lg["loss"], color=_c, lw=1.3,
                     label=_meta["label"])
        if not _meta["kto"]:                     # SFT control: no implicit reward
            continue
        _ax4[1].plot(_lg["step"], _lg["rewards/chosen"], color=_c, lw=1.3)
        _ax4[1].plot(_lg["step"], _lg["rewards/rejected"], color=_c, lw=1.3,
                     ls="--")
        _ax4[2].plot(_lg["step"], _lg["rewards/margins"], color=_c, lw=1.3)
        _ax4[3].plot(_lg["step"], _lg["kl"], color=_c, lw=1.3)

    _ax4[0].set_ylabel("loss")
    _ax4[0].set_title("(a) loss", loc="left", fontsize=8.5)
    _ax4[0].legend(frameon=False, fontsize=6.0, loc="upper right",
                   handlelength=1.1, labelspacing=0.2, borderpad=0.1)

    _ax4[1].axhline(0, color="#333333", lw=0.6)
    _ax4[1].set_ylabel("implicit reward")
    _ax4[1].set_title("(b) implicit reward", loc="left", fontsize=8.5)
    # Line style, not colour, separates the two halves here — say so in the
    # panel so the reader never has to hunt for it in the caption.
    _ax4[1].text(0.96, 0.96, "— desirable", transform=_ax4[1].transAxes,
                 ha="right", va="top", fontsize=6.0, color="#555555")
    _ax4[1].text(0.96, 0.04, "- - undesirable", transform=_ax4[1].transAxes,
                 ha="right", va="bottom", fontsize=6.0, color="#555555")

    _ax4[2].set_ylabel("margin")
    _ax4[2].set_title("(c) margin", loc="left", fontsize=8.5)

    _ax4[3].set_ylabel("KL to reference")
    _ax4[3].set_title("(d) KL", loc="left", fontsize=8.5)

    for _ax in _ax4:
        _ax.margins(x=0.01)
        _ax.set_xlabel("training step")
    _f4.tight_layout(pad=0.35, w_pad=1.0)
    save_fig(_f4, "fig_kto_arms_dynamics")
    _f4
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load — the held-out probe

    `outputs/2026-08-02_k3_probe/` — 45 slices x 150 held-out chunks x k=8 =
    54,000 completions, scored through the *production* labeler. This is the
    only generalization measurement the k-series has; every number above is
    train-side.

    `minority_acc` = recall on gold-inappropriate (violation) flows,
    `majority_acc` = recall on gold-appropriate flows, both over `scored`
    completions (gate failures are excluded from J and reported separately).
    The aggregation below reproduces `per_checkpoint_summary.csv` exactly,
    which is the check that licenses bootstrapping on top of it.
    """)
    return


@app.cell
def _(K3_PROBE, np, pd):
    probe_summary = pd.read_csv(K3_PROBE / "per_checkpoint_summary.csv")
    probe_raw = pd.read_parquet(K3_PROBE / "probe_results.parquet")

    _CHUNKS = sorted(probe_raw["chunk_key"].unique())

    def _chunk_counts(slice_name):
        """Per-chunk (n_viol, n_viol_correct, n_appr, n_appr_correct)."""
        g = probe_raw[(probe_raw["slice"] == slice_name)
                      & (probe_raw["status"] == "scored")]
        return (g.groupby("chunk_key")[["n_viol", "n_viol_correct",
                                        "n_appr", "n_appr_correct"]]
                 .sum().reindex(_CHUNKS).fillna(0).to_numpy(dtype=float))

    def _J_from(c):
        minority = c[:, 1].sum() / max(c[:, 0].sum(), 1)
        majority = c[:, 3].sum() / max(c[:, 2].sum(), 1)
        return minority + majority - 1.0

    # Verify the aggregation against the published summary before trusting it.
    for _sl in ("baseline", "verdict/final", "scrutinize/final"):
        _c = _chunk_counts(_sl)
        _ref = probe_summary[probe_summary["slice"] == _sl].iloc[0]
        assert abs(_J_from(_c) - _ref["J"]) < 1e-9, _sl
    print(f"aggregation reproduces per_checkpoint_summary.csv on "
          f"{len(_CHUNKS)} held-out chunks")

    # Paired bootstrap over CHUNKS (not completions): the 8 samples of a chunk
    # are not independent, so resampling completions would understate the CI.
    _rng = np.random.default_rng(20260806)
    _BOOT = 4000
    _idx = _rng.integers(0, len(_CHUNKS), size=(_BOOT, len(_CHUNKS)))
    _base_c = _chunk_counts("baseline")

    def _delta_J_ci(slice_name):
        arm = _chunk_counts(slice_name)
        d = np.empty(_BOOT)
        for i in range(_BOOT):
            s = _idx[i]
            d[i] = _J_from(arm[s]) - _J_from(_base_c[s])
        lo, hi = np.percentile(d, [2.5, 97.5])
        # two-sided bootstrap p: how often the resampled delta crosses zero
        p = 2 * min((d <= 0).mean(), (d >= 0).mean())
        return _J_from(arm) - _J_from(_base_c), lo, hi, p

    probe_final = []
    for _arm in ("verdict", "citation", "scrutinize", "sft_ctrl"):
        _sl = f"{_arm}/final"
        _row = probe_summary[probe_summary["slice"] == _sl].iloc[0]
        _dj, _lo, _hi, _p = _delta_J_ci(_sl)
        probe_final.append({
            "arm": _arm, "minority_acc": _row["minority_acc"],
            "majority_acc": _row["majority_acc"], "J": _row["J"],
            "delta_J": _dj, "ci_lo": _lo, "ci_hi": _hi, "p": _p,
            "gate_fail_rate": _row["gate_fail_rate"],
            "abstain_rate_gold_no": _row["abstain_rate_gold_no"],
            "miss_rate": _row["miss_rate"],
        })
    probe_final = pd.DataFrame(probe_final)
    print(probe_final.round(4).to_string(index=False))
    return probe_final, probe_summary


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## F5 — held-out: a threshold shift, not discrimination

    Panel (a) is the figure this whole section exists for. Each arm's
    checkpoints are plotted in the ROC plane — x = 1 - majority accuracy (false
    alarms), y = minority accuracy (violations caught) — and joined in
    checkpoint order. **Every arm slides along the chance diagonal.** The arms
    catch far more violations than the SFT baseline and pay for it one-for-one
    in false alarms, which is a moved decision threshold, not a better
    detector. Panel (b) is the same fact as J over training, with the paired
    bootstrap CI on the final checkpoint.
    """)
    return


@app.cell
def _(KTO_ARMS, TEXTWIDTH_IN, np, plt, probe_final, probe_summary, save_fig):
    _f5, (_a5, _b5) = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 2.35),
                                   gridspec_kw={"width_ratios": [1, 1.12]})

    _base = probe_summary[probe_summary["slice"] == "baseline"].iloc[0]

    # (a) ROC plane. The diagonal is chance; distance ABOVE it is J.
    _a5.plot([0, 1], [0, 1], color="#333333", ls=":", lw=0.9, zorder=1)
    _a5.text(0.60, 0.545, "chance", fontsize=6.4, color="#555555",
             rotation=45, rotation_mode="anchor", style="italic", zorder=1)

    def _ck_order(df):
        d = df[df["slice"].str.contains("checkpoint-")].copy()
        d["_n"] = d["slice"].str.extract(r"checkpoint-(\d+)")[0].astype(float)
        return d.sort_values("_n")

    for _arm, _meta in KTO_ARMS.items():
        _d = _ck_order(probe_summary[probe_summary["arm"] == _arm])
        _x, _y = 1 - _d["majority_acc"], _d["minority_acc"]
        _a5.plot(_x, _y, color=_meta["color"], lw=1.0, alpha=0.85, zorder=3,
                 marker="o", ms=2.4, mew=0, label=_meta["label"])
        _a5.plot(_x.iloc[-1], _y.iloc[-1], marker="o", ms=5.0, mew=0.7,
                 mfc=_meta["color"], mec="white", zorder=4)

    _a5.plot(1 - _base["majority_acc"], _base["minority_acc"], marker="*",
             ms=11, mfc="#D62728", mec="white", mew=0.7, zorder=5, ls="none",
             label="SFT baseline")
    _a5.set_xlim(0, 0.68)
    _a5.set_ylim(0, 0.68)
    _a5.set_aspect("equal")
    _a5.grid(True, axis="both", alpha=0.25, lw=0.5)
    _a5.set_xlabel("false alarms (1 $-$ majority acc.)")
    _a5.set_ylabel("violations caught (minority acc.)")
    _a5.set_title("(a) held-out ROC plane", loc="left", fontsize=8.5)
    _a5.legend(frameon=False, fontsize=6.2, loc="upper left",
               handlelength=1.2, labelspacing=0.25, borderpad=0.1)

    # (b) J per checkpoint, with the final-checkpoint bootstrap CI.
    for _arm, _meta in KTO_ARMS.items():
        _d = _ck_order(probe_summary[probe_summary["arm"] == _arm])
        _b5.plot(np.arange(1, len(_d) + 1), _d["J"], color=_meta["color"],
                 lw=1.2, marker="o", ms=2.4, mew=0, label=_meta["label"])
    _b5.axhline(_base["J"], color="#D62728", ls="-.", lw=1.0)
    _b5.text(0.5, _base["J"] - 0.004, "SFT baseline (below chance)",
             transform=_b5.get_yaxis_transform(), ha="center", va="top",
             fontsize=6.2, color="#D62728")
    _b5.axhline(0.0, color="#333333", lw=0.6)
    _b5.axhline(0.022, color="#333333", ls=":", lw=0.9)
    _b5.text(0.5, 0.0245, "acceptance threshold",
             transform=_b5.get_yaxis_transform(),
             ha="center", va="bottom", fontsize=6.2, color="#555555")

    # Final-checkpoint J with its paired-bootstrap CI, in a gutter to the right
    # of the trajectories. Same y-axis, so the CIs are read against the same
    # promotion bar and baseline the curves are.
    _x0 = 12.2
    for _i, _r in probe_final.iterrows():
        _J0 = float(_base["J"])
        _b5.errorbar(_x0 + _i * 0.72, _r["J"],
                     yerr=[[max(_r["J"] - (_J0 + _r["ci_lo"]), 0)],
                           [max((_J0 + _r["ci_hi"]) - _r["J"], 0)]],
                     color=KTO_ARMS[_r["arm"]]["color"], lw=1.1, capsize=1.8,
                     marker="o", ms=3.2, mew=0)
    _b5.axvline(11.4, color="#BBBBBB", lw=0.6)
    _b5.set_xlim(0.3, _x0 + 3 * 0.72 + 0.8)
    _b5.set_xticks([1, 4, 7, 10, _x0 + 1.08])
    _b5.set_xticklabels(["1", "4", "7", "10", "final\n95% CI"])
    _b5.set_xlabel("checkpoint")
    _b5.set_ylabel("Youden's $J$")
    _b5.set_title("(b) discrimination over training", loc="left", fontsize=8.5)

    _f5.tight_layout(pad=0.35, w_pad=1.2)
    save_fig(_f5, "fig_kto_heldout_threshold")
    _f5
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## T1 / T2 — the appendix tables

    Written as CSV *and* as `\input`-able LaTeX row files so the appendix never
    transcribes a number by hand. T1 reports mean **and median** KL: the mean is
    the quantity the gate used, and it is dominated by a startup transient in
    two cells. That is a defect in our own gate and is printed as such.
    """)
    return


@app.cell
def _(
    GRPO_ARMS,
    write_table_tex,
    grpo_gates_all,
    grpo_logs,
    grpo_tail_J,
    np,
    pd,
    save_table,
):
    def _thirds(s):
        v = pd.Series(s).dropna().to_numpy(dtype=float)
        k = max(1, len(v) // 3)
        return float(np.mean(v[:k])), float(np.mean(v[-k:]))

    _t1 = []
    for _arm, _meta in GRPO_ARMS.items():
        _lg, _g = grpo_logs[_arm], grpo_gates_all[_arm]["gates"]
        _r0, _r1 = _thirds(_lg["reward"])
        _e0, _e1 = _thirds(_lg["entropy"])
        # Gate estimator: pooled over the trailing 100 reward calls, asserted
        # equal to this cell's promotion_gates.json above.
        _rec_a, _rec_i, _J = grpo_tail_J[_arm]
        _t1.append({
            "cell": _arm,
            "auxiliaries": _meta["aux"],
            "direct_core": "on" if _meta["core"] else "off",
            "vignette_mix": _meta["vig"],
            "reward_first": round(_r0, 4),
            "reward_last": round(_r1, 4),
            "reward_gain": round(_r1 - _r0, 4),
            "kl_mean": round(float(_lg["kl"].mean()), 3),
            "kl_median": round(float(_lg["kl"].median()), 4),
            "kl_max": round(float(_lg["kl"].max()), 1),
            "entropy_first": round(_e0, 4),
            "entropy_last": round(_e1, 4),
            "frac_zero_std_max": round(float(_lg["frac_reward_zero_std"].max()), 4),
            "len_mean_tok": round(float(_lg["completions/mean_length"].mean()), 1),
            "truncated_pct": round(100 * float(_lg["completions/clipped_ratio"].max()), 3),
            "recall_appropriate": None if np.isnan(_rec_a) else round(_rec_a, 4),
            "recall_inappropriate": None if np.isnan(_rec_i) else round(_rec_i, 4),
            "youden_J": None if np.isnan(_J) else round(_J, 4),
            "gate_reward_trend": _g["reward_trend"]["status"],
            "gate_zero_std": _g["zero_std_groups"]["status"],
            "gate_kl": _g["kl_bounded"]["status"],
            "gate_no_flow": _g["no_flow_rate"]["status"],
            "gate_discrimination": _g["direct_discrimination"]["status"],
            "promote": grpo_gates_all[_arm]["promote"],
        })
    t1 = pd.DataFrame(_t1)
    save_table(t1, "t1_grpo_cell_summary")

    _MARK = {"pass": r"\gpass", "fail": r"\gfail", "skipped": r"\gskip"}

    def _gates_str(r):
        return "".join(_MARK[r[f"gate_{k}"]] for k in
                       ("reward_trend", "zero_std", "kl", "no_flow",
                        "discrimination"))

    def _num(v, fmt):
        return "--" if v is None or (isinstance(v, float) and np.isnan(v)) \
            else format(v, fmt)

    # Transposed: four cells against many metrics, so cells become COLUMNS.
    # The row-major form ran ~135pt past \textwidth; this fits without shrinking
    # type or dropping evidence.
    def _sec(title):
        return (rf"\addlinespace[2pt]" "\n"
                rf"\multicolumn{{5}}{{@{{}}l}}{{\textit{{{title}}}}} \\")

    def _row(label, fn):
        return label + " & " + " & ".join(fn(r) for _, r in t1.iterrows()) + r" \\"

    # Reward composition as one row per module rather than a prose
    # "auxiliaries" cell: it lines the table up with the module list of
    # app:reward, and the two-character cells keep the table inside \textwidth.
    def _has(term):
        return lambda r: "yes" if term in r["auxiliaries"] else "--"

    _rows = "\n".join([
        _row(r"$R_{\text{direct}}$\ \ (verifiable core)",
             lambda r: "yes" if r["direct_core"] == "on" else "--"),
        _row(r"$R_{\text{ground}}$\ \ (judged)", _has("ground")),
        _row(r"$R_{\text{contrast}}$\ \ (judged)", _has("contrast")),
        _row(r"judgment task mix", lambda r: f"{r['vignette_mix']:.2f}"),
        _sec("optimization"),
        _row(r"reward, first third", lambda r: f"{r['reward_first']:.3f}"),
        _row(r"\quad gain (criterion $>{+}0.02$)",
             lambda r: f"{r['reward_gain']:+.4f}"),
        _row(r"KL, mean (criterion $<1.0$)", lambda r: f"{r['kl_mean']:.2f}"),
        _row(r"\quad median", lambda r: f"{r['kl_median']:.3f}"),
        _row(r"entropy, first $\to$ last",
             lambda r: f"{r['entropy_first']:.2f}$\\to${r['entropy_last']:.2f}"),
        _row(r"zero-var. groups, max", lambda r: f"{r['frac_zero_std_max']:.3f}"),
        _row(r"completion length (tok)", lambda r: f"{r['len_mean_tok']:.0f}"),
        _row(r"truncated, max (\%)", lambda r: f"{r['truncated_pct']:.2f}"),
        _sec("discrimination (pooled, final 100 reward calls)"),
        _row(r"recall, appropriate",
             lambda r: _num(r["recall_appropriate"], ".3f")),
        _row(r"recall, inappropriate",
             lambda r: _num(r["recall_inappropriate"], ".3f")),
        _row(r"Youden's $J$ (criterion $\geq0.05$)",
             lambda r: _num(r["youden_J"], "+.4f")),
        _sec("acceptance criteria"),
        _row(r"criteria (caption order)", _gates_str),
        _row(r"all satisfied?",
             lambda r: "yes" if r["promote"] else "\\textbf{no}"),
    ])
    write_table_tex(
        "t1_grpo_cell_summary",
        colspec="@{}lcccc@{}",
        header=(
            r"& \multicolumn{4}{c}{reward-ablation configuration} \\" "\n"
            r"\cmidrule(l){2-5}" "\n"
            r"& \textsc{Full} & $-$\textsc{aux} & $-$\textsc{core} & "
            r"$-$\textsc{judg} \\"
        ),
        rows=_rows,
        label="tab:grpo-ablation",
        caption=(
            r"GRPO reward-ablation configurations: definition, optimization "
            r"summary, and acceptance criteria. Configurations are the "
            r"leave-one-out ablation of \autoref{app:reward}: \textsc{Full}; "
            r"$-$\textsc{aux}, which removes both judged auxiliaries; "
            r"$-$\textsc{core}, which removes $R_{\text{direct}}$; and "
            r"$-$\textsc{judg}, which removes the judgment task. "
            r"\textit{first} is the composite reward averaged over the first "
            r"third of logged steps, and \textit{gain} its change to the last "
            r"third; levels are not comparable across configurations, because "
            r"each renormalizes its component weights, so only the gain is "
            r"thresholded. We report KL as both a mean and a median over "
            r"logged steps: the mean is the quantity the criterion reads, and "
            r"it is dominated by a startup transient in $-$\textsc{aux} and "
            r"$-$\textsc{core}, whose medians are ordinary. \textit{entropy} "
            r"runs from the first third to the last third; "
            r"\textit{zero-var.}\ is the maximum fraction of reward groups "
            r"with zero reward variance. Recalls and $J$ are pooled over the "
            r"final 100 reward calls, which is the statistic the criterion "
            r"itself reads. The acceptance criteria are, in order, reward "
            r"trend, zero-variance groups, KL bound, abstention rate, and "
            r"discrimination, with \gpass\ satisfied, \gfail\ failed, and "
            r"\gskip\ not applicable. No configuration satisfies all five; the "
            r"reported model is \textsc{Full} (\autoref{app:grpo-hyperparams})."
        ),
    )
    print(t1.to_string(index=False))
    return (t1,)


@app.cell
def _(KTO_ARMS, kto_logs, np, pd, probe_final, save_table, write_table_tex):
    # Realized K1 dataset composition, fingerprint b27a46f8e7f5
    # (wiki/2026-07-31_kto_plan.md 14). Identical across the three KTO arms --
    # one build emits all three edit depths from the same sampled completions.
    _K1 = {"rows": 20059, "n_D": 11695, "n_U": 8364, "ratio": 1.15}

    _t2 = []
    for _arm, _meta in KTO_ARMS.items():
        _lg = kto_logs[_arm]
        _k = max(1, len(_lg) // 6)
        _p = probe_final[probe_final["arm"] == _arm].iloc[0]
        _kto = _meta["kto"]
        _t2.append({
            "arm": _arm,
            "supervision": _meta["depth"],
            "rows": _K1["rows"] if _kto else _K1["n_D"],
            "n_D": _K1["n_D"] if _kto else None,
            "n_U": _K1["n_U"] if _kto else None,
            "weighted_ratio": _K1["ratio"] if _kto else None,
            "steps": int(_lg["step"].max()),
            "loss_first": round(float(_lg["loss"].iloc[:_k].mean()), 4),
            "loss_last": round(float(_lg["loss"].iloc[-_k:].mean()), 4),
            "margin_final": round(float(_lg["rewards/margins"].iloc[-_k:].mean()), 3)
                            if _kto else None,
            "kl_final": round(float(_lg["kl"].iloc[-_k:].mean()), 2) if _kto else None,
            "entropy_first": round(float(_lg["entropy"].iloc[:_k].mean()), 4),
            "entropy_last": round(float(_lg["entropy"].iloc[-_k:].mean()), 4),
            "heldout_minority": round(float(_p["minority_acc"]), 4),
            "heldout_majority": round(float(_p["majority_acc"]), 4),
            "heldout_J": round(float(_p["J"]), 4),
            "delta_J": round(float(_p["delta_J"]), 4),
            "ci_lo": round(float(_p["ci_lo"]), 4),
            "ci_hi": round(float(_p["ci_hi"]), 4),
            "p": round(float(_p["p"]), 4),
            "gate_fail_rate": round(float(_p["gate_fail_rate"]), 4),
            "abstain_gold_no": round(float(_p["abstain_rate_gold_no"]), 4),
            "miss_rate": round(float(_p["miss_rate"]), 4),
        })
    t2 = pd.DataFrame(_t2)
    save_table(t2, "t2_kto_arm_summary")

    def _n(v, fmt):
        return "--" if v is None or (isinstance(v, float) and np.isnan(v)) \
            else format(v, fmt)

    # Transposed for the same reason as T1: four arms, many metrics.
    def _sec2(title):
        return (rf"\addlinespace[2pt]" "\n"
                rf"\multicolumn{{5}}{{@{{}}l}}{{\textit{{{title}}}}} \\")

    def _row2(label, fn):
        return label + " & " + " & ".join(fn(r) for _, r in t2.iterrows()) + r" \\"

    # No "supervision depth" row: the column headers already carry it, and its
    # widest cell was pushing the table past \textwidth.
    _rows2 = "\n".join([
        _row2(r"preference rows", lambda r: f"{int(r['rows']):,}"),
        _row2(r"optimizer steps", lambda r: f"{int(r['steps'])}"),
        _sec2("training dynamics"),
        _row2(r"loss, first $\to$ last",
              lambda r: f"{r['loss_first']:.3f}$\\to${r['loss_last']:.3f}"),
        _row2(r"implicit-reward margin, final",
              lambda r: _n(r["margin_final"], ".2f")),
        _row2(r"KL to reference, final", lambda r: _n(r["kl_final"], ".1f")),
        _row2(r"entropy, first $\to$ last",
              lambda r: f"{r['entropy_first']:.3f}$\\to${r['entropy_last']:.3f}"),
        _sec2("held-out evaluation, final checkpoint"),
        _row2(r"minority recall", lambda r: f"{r['heldout_minority']:.3f}"),
        _row2(r"majority recall", lambda r: f"{r['heldout_majority']:.3f}"),
        _row2(r"Youden's $J$", lambda r: f"{r['heldout_J']:+.4f}"),
        _row2(r"$\Delta J$ vs SFT", lambda r: f"{r['delta_J']:+.4f}"),
        _row2(r"\quad 95\% CI",
              lambda r: f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]"),
        _sec2("behavioral side-effects"),
        _row2(r"validity-failure rate (ceiling $0.08$)",
              lambda r: f"{r['gate_fail_rate']:.3f}"),
        _row2(r"correct abstention", lambda r: f"{r['abstain_gold_no']:.3f}"),
        _row2(r"miss rate", lambda r: f"{r['miss_rate']:.3f}"),
    ])
    write_table_tex(
        "t2_kto_arm_summary",
        colspec="@{}lcccc@{}",
        tabcolsep="2.6pt",
        header=(
            r"& \multicolumn{3}{c}{KTO, by supervision depth} & control \\" "\n"
            r"\cmidrule(lr){2-4}\cmidrule(l){5-5}" "\n"
            r"& label & $+$\,norm & $+$\,rationale & supervised \\"
        ),
        rows=_rows2,
        label="tab:kto-arms",
        caption=(
            r"KTO arms by supervision depth: definition, optimization summary, "
            r"and held-out evaluation. \textit{label} corrects the "
            r"appropriateness label alone; $+$\,\textit{norm} additionally "
            r"corrects the cited norm; $+$\,\textit{rationale} additionally "
            r"rewrites the reasoning trace. All three train on the same "
            r"$20{,}059$-row preference set ($11{,}695$ desirable, $8{,}364$ "
            r"undesirable, weighted ratio $1.15$, inside the recommended "
            r"$[1, 4/3]$ band), and differ only in supervision depth; the "
            r"supervised control trains with a plain supervised loss on the "
            r"$11{,}695$ desirable rows alone. Margin and KL are means over "
            r"the final sixth of logged steps. Held-out columns report the "
            r"final checkpoint on 150 withheld chunks; $\Delta J$ is "
            r"measured against the SFT baseline ($J = -0.034$, minority "
            r"$0.106$, majority $0.860$) with a paired bootstrap over chunks "
            r"($4{,}000$ resamples). \textit{Validity-failure rate} is the "
            r"share of completions failing the JSON validity check (ceiling "
            r"$0.08$); \textit{correct abstention} is the rate of correct "
            r"abstention on chunks containing no reference flows (baseline "
            r"$0.023$); \textit{miss rate} is the miss rate on chunks that "
            r"do contain flows (baseline $0.208$). No arm satisfies its "
            r"acceptance criteria, and none separates from the supervised "
            r"control."
        ),
    )
    print(t2.to_string(index=False))
    return (t2,)


@app.cell
def _(t1, t2):
    # Sentence-level numbers the appendix prose quotes, printed together so a
    # reviewer can diff the text against one block of output.
    print("--- GRPO, all four m2 cells ---")
    print(f"reward gain      {t1['reward_gain'].min():+.4f} .. "
          f"{t1['reward_gain'].max():+.4f}   (gate > +0.02; 0/4 pass)")
    print(f"entropy first    {t1['entropy_first'].min():.3f} .. "
          f"{t1['entropy_first'].max():.3f}  ->  last "
          f"{t1['entropy_last'].min():.3f} .. {t1['entropy_last'].max():.3f}"
          "   (rises in every cell -- no collapse)")
    print(f"KL mean          {t1['kl_mean'].min():.2f} .. {t1['kl_mean'].max():.2f}"
          f"   BUT median {t1['kl_median'].min():.3f} .. {t1['kl_median'].max():.3f}")
    print(f"zero-var groups  max {t1['frac_zero_std_max'].max():.3f}  (gate 0.2)")
    print(f"length           {t1['len_mean_tok'].min():.0f} .. "
          f"{t1['len_mean_tok'].max():.0f} tok; truncation max "
          f"{t1['truncated_pct'].max():.3f}%")
    _d = t1.dropna(subset=["youden_J"])
    print(f"recall appropriate {_d['recall_appropriate'].min():.3f} .. "
          f"{_d['recall_appropriate'].max():.3f} | inappropriate "
          f"{_d['recall_inappropriate'].min():.3f} .. "
          f"{_d['recall_inappropriate'].max():.3f}")
    print(f"Youden J         {_d['youden_J'].min():+.4f} .. "
          f"{_d['youden_J'].max():+.4f}   (gate 0.05; 0/3 pass)")

    print("\n--- KTO, the supervision-depth ladder ---")
    _k = t2[t2["margin_final"].notna()].sort_values("margin_final")
    print("margin     " + "  <  ".join(f"{r['arm']} {r['margin_final']:.2f}"
                                       for _, r in _k.iterrows()))
    print("KL         " + "  <  ".join(f"{r['arm']} {r['kl_final']:.1f}"
                                       for _, r in _k.iterrows()))
    print("held-out J, same arm order: "
          + " , ".join(f"{r['arm']} {r['heldout_J']:+.4f}"
                       for _, r in _k.iterrows()))
    print(f"\nheld-out minority {t2['heldout_minority'].min():.3f} .. "
          f"{t2['heldout_minority'].max():.3f} (baseline 0.106) | majority "
          f"{t2['heldout_majority'].min():.3f} .. "
          f"{t2['heldout_majority'].max():.3f} (baseline 0.860)")
    print(f"gate-fail {t2['gate_fail_rate'].min():.3f} .. "
          f"{t2['gate_fail_rate'].max():.3f} (ceiling 0.08) | abstain-gold-NO "
          f"{t2['abstain_gold_no'].min():.3f} .. {t2['abstain_gold_no'].max():.3f} "
          "(baseline 0.023)")
    print(f"miss rate {t2['miss_rate'].min():.3f} .. {t2['miss_rate'].max():.3f} "
          "(baseline 0.208)")
    return


if __name__ == "__main__":
    app.run()
