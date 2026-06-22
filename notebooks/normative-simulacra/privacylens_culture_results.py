import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # PrivacyLens — cultural name-perturbation **results**

        Companion to the perturbation *inspector*. This notebook reads a finished
        `eval_all` multirun of `pipeline=privacylens_culture` (models × name-cultures)
        and compares contextual-integrity behaviour across cultures with the model
        held fixed.

        **Metrics** (per `compute_metrics`):

        - `leakage_rate` — fraction of agent actions that leak a sensitive item
          (overall, non-format actions counted as non-leaking). **Lower is safer.**
        - `helpful_rate` / `help_mean` — judge helpfulness (0–3). **Higher is better.**
        - `adjusted_leakage` — leakage **among helpful actions only**. This is the
          headline CI metric: it controls for the engagement confound (a model that
          simply does less also leaks less *and* helps less).
        - `format_rate` — fraction of actions in valid tool-call format (engagement /
          confound proxy).
        - `qa_acc` — comprehension probe accuracy (validity check: should be flat
          across cultures if perturbation preserved meaning).

        Read the loader's coverage note first — this run was **incomplete** (12/18
        cells), so only `grpo` spans all six cultures.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    run_dir = mo.ui.text(
        value="/share/pierson/matt/UAIR/multirun/2026-06-08_eval_all/20-43-32",
        label="multirun dir",
        full_width=True,
    )
    run_dir
    return (run_dir,)


@app.cell
def _():
    # Fixed display order; the western entry is the control everything is diffed against.
    MODEL_ORDER = ["base", "sft-ci", "grpo-l100"]
    CULTURE_ORDER = [
        "western",
        "east_asian",
        "south_asian",
        "arabic_me",
        "african",
        "african_american",
    ]
    CONTROL = "western"
    # eval_all sweep-job index -> model label (from each job's .hydra/overrides.yaml)
    JOB_MODEL = {"0": "base", "1": "sft-ci", "2": "grpo-l100"}
    return CONTROL, CULTURE_ORDER, JOB_MODEL, MODEL_ORDER


@app.cell
def _(CULTURE_ORDER, JOB_MODEL, run_dir):
    import json
    import os

    import numpy as np
    import pandas as pd

    def _load_metrics(base):
        rows = []
        for _job, _model in JOB_MODEL.items():
            for _c in CULTURE_ORDER:
                _f = os.path.join(
                    base,
                    _job,
                    f"privacylens_{_c}",
                    "privacylens_eval",
                    "outputs",
                    "compute_metrics",
                    "metrics.json",
                )
                if not os.path.exists(_f):
                    rows.append({"model": _model, "culture": _c, "present": False})
                    continue
                _d = json.load(open(_f))
                _lk = _d.get("leakage", {})
                _hp = _d.get("helpfulness", {})
                _qa = _d.get("qa_probing", {})
                _adj = _d.get("adjusted_leakage", {})
                _axis = _qa.get("per_axis", {})
                rows.append(
                    {
                        "model": _model,
                        "culture": _c,
                        "present": True,
                        "leakage_rate": _lk.get("leakage_rate_overall_with_default_zero"),
                        "leakage_parseable": _lk.get("leakage_rate_among_parseable"),
                        "help_rate": _hp.get("helpful_rate_overall_with_default_zero"),
                        "help_mean": _hp.get("mean_score_overall_with_default_zero"),
                        "adj_leakage": _adj.get("adjusted_leakage_rate"),
                        "n_helpful_judged": _adj.get("total_helpful_and_judged"),
                        "format_rate": _lk.get("agent_action_format_rate"),
                        "qa_acc": _qa.get("accuracy"),
                        "qa_S": _axis.get("S", {}).get("accuracy"),
                        "qa_T": _axis.get("T", {}).get("accuracy"),
                        "qa_V": _axis.get("V", {}).get("accuracy"),
                    }
                )
        return pd.DataFrame(rows)

    df = _load_metrics(run_dir.value)
    df_ok = df[df["present"]].drop(columns=["present"]).reset_index(drop=True)
    return df, df_ok, np, pd


@app.cell(hide_code=True)
def _(JOB_MODEL, CULTURE_ORDER, df, mo):
    _n_done = int(df["present"].sum())
    _n_tot = len(JOB_MODEL) * len(CULTURE_ORDER)
    _missing = df[~df["present"]][["model", "culture"]].apply(
        lambda r: f"{r['model']}/{r['culture']}", axis=1
    ).tolist()
    mo.md(
        f"""
        **Coverage: {_n_done} / {_n_tot} cells present.**

        Missing: {', '.join(_missing) if _missing else 'none — full matrix.'}

        > Only `grpo-l100` spans all six cultures. For `base`/`sft-ci` the cross-culture
        > comparison is limited to western / east_asian / south_asian. Treat any
        > base/sft-ci "africa/arabic" gap as unavailable, not zero.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 1. Headline table""")
    return


@app.cell
def _(MODEL_ORDER, CULTURE_ORDER, df_ok, mo, pd):
    def _pivot(metric, pct=True):
        _p = df_ok.pivot(index="model", columns="culture", values=metric)
        _p = _p.reindex(index=MODEL_ORDER, columns=CULTURE_ORDER)
        if pct:
            _p = (_p * 100).round(1)
        else:
            _p = _p.round(3)
        return _p

    _tables = {
        "leakage_rate %  (lower=safer)": _pivot("leakage_rate"),
        "adjusted_leakage %  (leak | helpful — headline)": _pivot("adj_leakage"),
        "help_rate %  (higher=better)": _pivot("help_rate"),
        "help_mean (0-3)": _pivot("help_mean", pct=False),
        "format_rate %  (engagement)": _pivot("format_rate"),
        "qa_acc %  (validity)": _pivot("qa_acc"),
    }
    mo.vstack(
        [mo.vstack([mo.md(f"**{_k}**"), mo.ui.table(_v.reset_index(), selection=None)]) for _k, _v in _tables.items()]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 2. Deltas vs. western control

        Each treatment culture minus its own model's western value. Negative leakage
        delta = the model leaks **less** on non-Western names than on Western ones.
        Watch whether helpfulness drops in lockstep (the confound) — if it does,
        lean on `adjusted_leakage` instead of raw leakage.
        """
    )
    return


@app.cell
def _(CONTROL, MODEL_ORDER, CULTURE_ORDER, df_ok, pd):
    def _deltas(metric):
        _w = df_ok[df_ok["culture"] == CONTROL].set_index("model")[metric]
        _rows = []
        for _, _r in df_ok.iterrows():
            if _r["culture"] == CONTROL:
                continue
            _base = _w.get(_r["model"])
            if _base is None or _r[metric] is None:
                continue
            _rows.append(
                {"model": _r["model"], "culture": _r["culture"], "delta": (_r[metric] - _base) * 100}
            )
        _d = pd.DataFrame(_rows)
        if _d.empty:
            return _d
        _treat = [c for c in CULTURE_ORDER if c != CONTROL]
        return _d.pivot(index="model", columns="culture", values="delta").reindex(
            index=MODEL_ORDER, columns=_treat
        ).round(1)

    delta_leak = _deltas("leakage_rate")
    delta_adj = _deltas("adj_leakage")
    delta_help = _deltas("help_rate")
    return delta_adj, delta_help, delta_leak


@app.cell
def _(delta_adj, delta_help, delta_leak, mo):
    mo.vstack(
        [
            mo.md("**Δ leakage_rate vs western (pp)**"),
            mo.ui.table(delta_leak.reset_index(), selection=None),
            mo.md("**Δ adjusted_leakage vs western (pp) — confound-controlled**"),
            mo.ui.table(delta_adj.reset_index(), selection=None),
            mo.md("**Δ help_rate vs western (pp)**"),
            mo.ui.table(delta_help.reset_index(), selection=None),
        ]
    )
    return


@app.cell
def _(MODEL_ORDER, CULTURE_ORDER, df_ok):
    import matplotlib.pyplot as plt

    _treat = [c for c in CULTURE_ORDER if c != "western"]
    _fig, _axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=False)
    for _ax, _metric, _title in zip(
        _axes,
        ["leakage_rate", "adj_leakage", "help_rate"],
        ["Leakage rate", "Adjusted leakage (|helpful)", "Helpful rate"],
    ):
        _piv = df_ok.pivot(index="culture", columns="model", values=_metric).reindex(
            index=CULTURE_ORDER, columns=MODEL_ORDER
        )
        for _m in MODEL_ORDER:
            _ax.plot(
                range(len(CULTURE_ORDER)),
                (_piv[_m].values.astype(float)) * 100,
                marker="o",
                label=_m,
            )
        _ax.set_xticks(range(len(CULTURE_ORDER)))
        _ax.set_xticklabels(CULTURE_ORDER, rotation=40, ha="right", fontsize=8)
        _ax.axvline(0, color="grey", ls=":", lw=1)  # western control at x=0
        _ax.set_title(_title)
        _ax.set_ylabel("%")
        _ax.grid(alpha=0.3)
    _axes[0].legend(fontsize=8)
    _fig.suptitle("CI metrics by name-culture (western = leftmost control)", y=1.02)
    _fig.tight_layout()
    _fig
    return (plt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 3. The engagement confound

        Raw leakage and helpfulness tend to move **together** across cultures — a sign
        the model is simply *less active* on perturbed vignettes rather than reasoning
        differently about privacy. If the points below trend along the diagonal, the raw
        leakage gap is partly an engagement artifact, and `adjusted_leakage` (which
        conditions on helpful actions) is the metric to trust.
        """
    )
    return


@app.cell
def _(df_ok, plt):
    _fig, _ax = plt.subplots(figsize=(6.5, 5.5))
    _markers = {"base": "o", "sft-ci": "s", "grpo-l100": "^"}
    for _m, _mk in _markers.items():
        _sub = df_ok[df_ok["model"] == _m]
        _ax.scatter(
            _sub["help_rate"] * 100,
            _sub["leakage_rate"] * 100,
            marker=_mk,
            s=80,
            label=_m,
            alpha=0.8,
        )
        for _, _r in _sub.iterrows():
            _ax.annotate(
                _r["culture"].replace("_", " "),
                (_r["help_rate"] * 100, _r["leakage_rate"] * 100),
                fontsize=7,
                xytext=(4, 2),
                textcoords="offset points",
            )
    _ax.set_xlabel("helpful rate %")
    _ax.set_ylabel("leakage rate %")
    _ax.set_title("Leakage vs. helpfulness (co-movement = engagement confound)")
    _ax.legend()
    _ax.grid(alpha=0.3)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 4. Perturbation validity — swap coverage

        Sanity that the treatment cultures actually rewrote names (and the western
        control did not). Reads the `perturb_culture` audit columns. `qa_acc` staying
        flat across cultures (table 1) is the complementary check that meaning survived.
        """
    )
    return


@app.cell
def _(JOB_MODEL, CULTURE_ORDER, pd, run_dir):
    import os as _os

    def _swap_coverage(base):
        # Read one model's parquets (audit cols are model-independent); prefer grpo (job 2, full).
        _rows = []
        for _c in CULTURE_ORDER:
            _f = None
            for _job in ("2", "1", "0"):
                _cand = _os.path.join(
                    base, _job, f"privacylens_{_c}", "privacylens_eval",
                    "outputs", "perturb_culture", "dataset.parquet",
                )
                if _os.path.exists(_cand):
                    _f = _cand
                    break
            if _f is None:
                continue
            _d = pd.read_parquet(_f, columns=["n_persons_swapped", "n_locations_swapped"])
            _rows.append(
                {
                    "culture": _c,
                    "rows": len(_d),
                    "mean_persons_swapped": round(_d["n_persons_swapped"].mean(), 2),
                    "mean_locations_swapped": round(_d["n_locations_swapped"].mean(), 2),
                    "frac_rows_with_swap": round((_d["n_persons_swapped"] > 0).mean(), 3),
                }
            )
        return pd.DataFrame(_rows)

    swap_cov = _swap_coverage(run_dir.value)
    return (swap_cov,)


@app.cell
def _(mo, swap_cov):
    mo.vstack(
        [
            mo.md(
                "**Swap coverage** — western should be all-zero (identity passthrough); "
                "treatments should show persons swapped on nearly every row."
            ),
            mo.ui.table(swap_cov, selection=None),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 5. Export tidy table

        The full per-(model, culture) metrics frame, for stats / paper tables.
        """
    )
    return


@app.cell
def _(df_ok, mo):
    mo.ui.table(df_ok.round(4), selection=None)
    return


if __name__ == "__main__":
    app.run()
