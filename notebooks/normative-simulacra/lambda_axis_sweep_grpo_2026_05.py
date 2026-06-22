import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # λ-axis GRPO sweep — fetch & analyze the contrastive-lambda runs

    Two SLURM multiruns varied the GRPO **contrastive_lambda** while holding the rest
    of the composite-CI reward fixed:

    | sweep | multirun dir | λ values |
    |---|---|---|
    | `full (05-13)`    | `multirun/2026-05-13_lambda_axis_sweep/12-31-50`         | 0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0 |
    | `partial (05-15)` | `multirun/2026-05-15_lambda_axis_partial_sweep/21-58-26` | 1.0, 1.5, 2.0 |

    These are **training** runs, logged to W&B project `uair/grpo-ci-training` (not
    `eval-all`, so the repo's `fetch_wandb_runs.py` / `wandb_cache.py` don't cover them —
    those target eval benchmarks). This notebook ships a **self-contained fetch + cache**
    for the training histories, parsing each run's `λ` and its sweep from
    `config.output_dir` (the only trustworthy key for *these* runs — run names collide
    for λ∈{0,0.25,0.5}, and they predate the W&B tag fix so they only carry the old
    `contrastive:0.0` tag, which encoded `contrastive_ratio`, not the swept
    `contrastive_lambda`).

    > Tag fix (`dagspaces/common/wandb_logger.py`): the ambiguous `contrastive:{cr}`
    > tag was renamed to `cratio:{cr}` and a new `clambda:{cl}` tag now records
    > `contrastive_lambda`. It is forward-only — runs cached here predate it, so we
    > still key off `output_dir`. Future λ-axis runs will be distinguishable by the
    > `clambda:` tag directly.

    Cache lives in `wandb_cache/lambda_axis/` next to this notebook:
    `runs.json` (enriched run dicts), `history/<run_id>.parquet` (per-run training
    history), `meta.json`. Press **Fetch / refresh** below to (re)build it; otherwise
    the loader just reads the cache offline.
    """)
    return


@app.cell
def _():
    import json
    import re
    import sys
    from datetime import datetime, timezone
    from pathlib import Path

    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    NB_DIR = Path("/share/pierson/matt/UAIR/notebooks/normative-simulacra")
    CACHE_DIR = NB_DIR / "wandb_cache" / "lambda_axis"
    HIST_DIR = CACHE_DIR / "history"
    TABLES_DIR = NB_DIR / "tables"

    WANDB_ENTITY = "uair"
    WANDB_PROJECT = "grpo-ci-training"

    # The two requested multiruns, keyed by the unique path fragment that appears in
    # each W&B run's config.output_dir.
    SWEEPS = {
        "2026-05-13_lambda_axis_sweep/12-31-50": "full (05-13)",
        "2026-05-15_lambda_axis_partial_sweep/21-58-26": "partial (05-15)",
    }
    SWEEP_ORDER = ["full (05-13)", "partial (05-15)"]

    # Training metrics worth tracking over global_step.
    HISTORY_KEYS = [
        "train/global_step",
        "train/reward",
        "train/reward_std",
        "train/rewards/composite_ci_reward/mean",
        "train/rewards/composite_ci_reward/std",
        "train/loss",
        "train/entropy",
        "train/grad_norm",
        "train/learning_rate",
        "train/frac_reward_zero_std",
        "train/kl",
        "train/completions/mean_length",
        "train/completions/clipped_ratio",
        "train/clip_ratio/region_mean",
        "train/epoch",
    ]

    pd.set_option("display.max_columns", 60)
    pd.set_option("display.float_format", "{:.4f}".format)

    sys.path.insert(0, "/share/pierson/matt/UAIR/notebooks")
    try:
        from font_utils import load_ibm_plex_sans

        load_ibm_plex_sans()
    except Exception as _e:
        print(f"[font_utils] skipped: {_e}")
    return (
        CACHE_DIR,
        HISTORY_KEYS,
        HIST_DIR,
        SWEEPS,
        SWEEP_ORDER,
        TABLES_DIR,
        WANDB_ENTITY,
        WANDB_PROJECT,
        datetime,
        json,
        pd,
        plt,
        re,
        timezone,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · Fetch / refresh the cache
    """)
    return


@app.cell
def _(mo):
    fetch_button = mo.ui.run_button(label="Fetch / refresh from W&B")
    force_switch = mo.ui.switch(label="force overwrite existing cache", value=False)
    mo.hstack([fetch_button, force_switch], justify="start", gap=2)
    return fetch_button, force_switch


@app.cell
def _(
    CACHE_DIR,
    HISTORY_KEYS,
    HIST_DIR,
    SWEEPS,
    WANDB_ENTITY,
    WANDB_PROJECT,
    datetime,
    fetch_button,
    force_switch,
    json,
    re,
    timezone,
):
    def _parse_output_dir(output_dir: str):
        """Return (sweep_label, lambda) parsed from a run's config.output_dir."""
        if not output_dir:
            return None, None
        sweep_label = None
        for frag, label in SWEEPS.items():
            if frag in output_dir:
                sweep_label = label
                break
        if sweep_label is None:
            return None, None
        m = re.search(r"lambda=([0-9]+(?:\.[0-9]+)?)", output_dir)
        lam = float(m.group(1)) if m else None
        return sweep_label, lam

    def _enrich(run, sweep_label, lam):
        cfg = run.config
        grpo = cfg.get("grpo", {}) if isinstance(cfg.get("grpo"), dict) else {}
        summary = {k: v for k, v in run.summary.items() if not k.startswith("_")}
        return {
            "run_id": run.id,
            "run_name": run.name,
            "run_url": run.url,
            "state": run.state,
            "created_at": str(run.created_at),
            "tags": list(run.tags),
            "sweep": sweep_label,
            "lambda": lam,
            "output_dir": cfg.get("output_dir"),
            "reward_weights": grpo.get("reward_weights"),
            "num_generations": grpo.get("num_generations"),
            "learning_rate": grpo.get("learning_rate"),
            "online_rground": grpo.get("online_rground"),
            "max_steps": cfg.get("max_steps"),
            "summary": summary,
            "final_step": summary.get("train/global_step"),
            "final_reward": summary.get("train/reward"),
            "final_reward_std": summary.get("train/reward_std"),
            "final_loss": summary.get("train/loss"),
            "final_entropy": summary.get("train/entropy"),
            "train_runtime_s": summary.get("train_runtime"),
        }

    def _fetch(force: bool):
        import wandb

        HIST_DIR.mkdir(parents=True, exist_ok=True)
        api = wandb.Api()
        path = f"{WANDB_ENTITY}/{WANDB_PROJECT}"
        # grpo_training runs only; orchestrator runs have no training history.
        raw = api.runs(
            path,
            filters={"display_name": {"$regex": "lambda_axis.*grpo_training"}},
        )
        log = [f"scanned {len(raw)} candidate runs in {path}"]

        enriched = []
        for run in raw:
            sweep_label, lam = _parse_output_dir(run.config.get("output_dir", ""))
            if sweep_label is None or lam is None:
                continue
            rec = _enrich(run, sweep_label, lam)
            enriched.append(rec)

            hist_path = HIST_DIR / f"{run.id}.parquet"
            if hist_path.exists() and not force:
                rec["history_rows"] = "cached"
                continue
            try:
                h = run.history(keys=HISTORY_KEYS, pandas=True)
            except Exception:
                h = run.history(pandas=True)
            if h is not None and len(h):
                h.to_parquet(hist_path)
                rec["history_rows"] = len(h)
            else:
                rec["history_rows"] = 0

        enriched.sort(key=lambda r: (r["sweep"], r["lambda"], r["created_at"]))
        with open(CACHE_DIR / "runs.json", "w") as f:
            json.dump(enriched, f, indent=2, default=str)
        meta = {
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "entity": WANDB_ENTITY,
            "project": WANDB_PROJECT,
            "sweeps": SWEEPS,
            "n_runs": len(enriched),
        }
        with open(CACHE_DIR / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        log.append(f"cached {len(enriched)} grpo_training runs -> {CACHE_DIR}")
        return enriched, log

    fetch_log = None
    if fetch_button.value:
        _runs, fetch_log = _fetch(force_switch.value)
        for _line in fetch_log:
            print(_line)
    elif not (CACHE_DIR / "runs.json").exists():
        print(
            "No cache found at "
            f"{CACHE_DIR / 'runs.json'} — press 'Fetch / refresh from W&B' above."
        )
    else:
        print(f"Using existing cache at {CACHE_DIR} (press button to refresh).")
    return (fetch_log,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · Load the cache

    `runs.json` → one row per `(sweep, λ)` training run with final-step scalars;
    per-run `history/<run_id>.parquet` holds the full step-wise training curve.
    """)
    return


@app.cell
def _(CACHE_DIR, HIST_DIR, SWEEP_ORDER, fetch_log, json, pd):
    def _load_runs():
        p = CACHE_DIR / "runs.json"
        if not p.exists():
            return pd.DataFrame()
        with open(p) as f:
            runs = json.load(f)
        rows = []
        for r in runs:
            rows.append(
                {
                    "sweep": r.get("sweep"),
                    "lambda": r.get("lambda"),
                    "run_id": r.get("run_id"),
                    "run_name": r.get("run_name"),
                    "state": r.get("state"),
                    "created_at": r.get("created_at"),
                    "final_step": r.get("final_step"),
                    "final_reward": r.get("final_reward"),
                    "final_reward_std": r.get("final_reward_std"),
                    "final_loss": r.get("final_loss"),
                    "final_entropy": r.get("final_entropy"),
                    "train_runtime_s": r.get("train_runtime_s"),
                    "reward_weights": r.get("reward_weights"),
                    "run_url": r.get("run_url"),
                }
            )
        df = pd.DataFrame(rows)
        if not df.empty:
            df["sweep"] = pd.Categorical(df["sweep"], SWEEP_ORDER, ordered=True)
            df = df.sort_values(["sweep", "lambda"]).reset_index(drop=True)
        return df

    def load_history(run_id: str) -> pd.DataFrame:
        p = HIST_DIR / f"{run_id}.parquet"
        if not p.exists():
            return pd.DataFrame()
        h = pd.read_parquet(p)
        if "train/global_step" in h.columns:
            h = h.dropna(subset=["train/global_step"]).sort_values("train/global_step")
        return h

    # depend on fetch_log so the table refreshes right after a fetch
    _ = fetch_log
    runs_df = _load_runs()
    print(f"Loaded {len(runs_df)} runs across sweeps: {sorted(runs_df['sweep'].dropna().unique()) if len(runs_df) else '—'}")
    runs_df
    return load_history, runs_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · Run inventory

    Sanity check: did every requested `(sweep, λ)` finish, how many optimizer steps,
    and what was the final composite reward? `full (05-13)` should have 7 λ values,
    `partial (05-15)` should have 3 (λ∈{1.0, 1.5, 2.0}).
    """)
    return


@app.cell
def _(runs_df):
    def _inventory(df):
        if df.empty:
            return df
        piv = df.pivot_table(
            index="lambda",
            columns="sweep",
            values="final_reward",
            aggfunc="first",
            observed=True,
        )
        return piv

    inventory = _inventory(runs_df)
    n_unfinished = int((runs_df["state"] != "finished").sum()) if len(runs_df) else 0
    print(f"Runs not in state=finished: {n_unfinished}")
    (
        runs_df[runs_df["state"] != "finished"][
            ["sweep", "lambda", "run_id", "state", "run_url"]
        ]
        if n_unfinished
        else inventory
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · Training curves per λ

    One panel per metric, one line per λ (color = λ), one figure per sweep. This is
    the core diagnostic: how the contrastive term reshapes reward growth, reward
    variance (`reward_std`), entropy collapse, loss, and completion length over
    optimizer steps.
    """)
    return


@app.cell
def _(load_history, pd, runs_df):
    def _stack_histories(df):
        frames = []
        for _, r in df.iterrows():
            h = load_history(r["run_id"])
            if h.empty:
                continue
            h = h.copy()
            h["sweep"] = r["sweep"]
            h["lambda"] = r["lambda"]
            h["run_id"] = r["run_id"]
            frames.append(h)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    hist_all = _stack_histories(runs_df)
    print(
        f"Stacked history: {len(hist_all)} rows, "
        f"{hist_all['run_id'].nunique() if len(hist_all) else 0} runs, "
        f"columns: {[c for c in hist_all.columns if c.startswith('train/')][:8] if len(hist_all) else '—'}"
    )
    hist_all.head()
    return (hist_all,)


@app.cell
def _(SWEEP_ORDER, hist_all, plt):
    def _plot_curves(hist):
        if hist.empty:
            print("No history to plot.")
            return None
        panels = [
            ("train/reward", "composite reward"),
            ("train/reward_std", "reward std (within-group)"),
            ("train/loss", "GRPO loss"),
            ("train/entropy", "policy entropy"),
            ("train/completions/mean_length", "mean completion length"),
            ("train/grad_norm", "grad norm"),
            ("train/frac_reward_zero_std", "frac. groups w/ zero reward std"),
            ("train/clip_ratio/region_mean", "clip ratio (region mean)"),
        ]
        panels = [(c, t) for c, t in panels if c in hist.columns]
        sweeps = [s for s in SWEEP_ORDER if s in set(hist["sweep"])]
        nfig = len(sweeps)
        fig, axgrid = plt.subplots(
            nfig, len(panels), figsize=(3.4 * len(panels), 3.4 * nfig), squeeze=False
        )
        cmap = plt.get_cmap("viridis")
        for row, sweep in enumerate(sweeps):
            sub = hist[hist["sweep"] == sweep]
            lams = sorted(sub["lambda"].dropna().unique())
            lo, hi = (min(lams), max(lams)) if lams else (0, 1)
            for col, (metric, title) in enumerate(panels):
                ax = axgrid[row][col]
                for lam in lams:
                    g = sub[sub["lambda"] == lam].sort_values("train/global_step")
                    if metric not in g or g[metric].dropna().empty:
                        continue
                    c = cmap((lam - lo) / (hi - lo)) if hi > lo else cmap(0.5)
                    ax.plot(
                        g["train/global_step"],
                        g[metric],
                        marker="o",
                        ms=3,
                        lw=1.4,
                        color=c,
                        label=f"λ={lam:g}",
                    )
                ax.set_title(title, fontsize=9)
                ax.grid(True, alpha=0.25)
                if col == 0:
                    ax.set_ylabel(f"{sweep}", fontsize=10)
                if row == nfig - 1:
                    ax.set_xlabel("global step", fontsize=8)
            axgrid[row][0].legend(fontsize=7, frameon=False, ncol=2)
        fig.suptitle(
            "λ-axis GRPO sweep — training curves by contrastive_lambda",
            y=1.005,
            fontsize=13,
        )
        fig.tight_layout()
        return fig

    _plot_curves(hist_all)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 · Reproducibility — overlapping λ across the two sweeps

    λ ∈ {1.0, 1.5, 2.0} were run in **both** sweeps. Overlay the reward curve for
    each shared λ (solid = full 05-13, dashed = partial 05-15) to check the partial
    rerun reproduced the original trajectory.
    """)
    return


@app.cell
def _(hist_all, plt):
    def _plot_overlap(hist):
        if hist.empty:
            print("No history.")
            return None
        shared = sorted(
            set(hist[hist["sweep"] == "full (05-13)"]["lambda"].dropna())
            & set(hist[hist["sweep"] == "partial (05-15)"]["lambda"].dropna())
        )
        if not shared:
            print("No λ overlap between the two sweeps in cache.")
            return None
        fig, axes = plt.subplots(
            1, len(shared), figsize=(5 * len(shared), 4), squeeze=False
        )
        style = {"full (05-13)": "-", "partial (05-15)": "--"}
        color = {"full (05-13)": "#4c78a8", "partial (05-15)": "#e45756"}
        for ax, lam in zip(axes[0], shared):
            for sweep in ("full (05-13)", "partial (05-15)"):
                g = hist[
                    (hist["sweep"] == sweep) & (hist["lambda"] == lam)
                ].sort_values("train/global_step")
                if g.empty:
                    continue
                ax.plot(
                    g["train/global_step"],
                    g["train/reward"],
                    style[sweep],
                    color=color[sweep],
                    marker="o",
                    ms=3,
                    label=sweep,
                )
            ax.set_title(f"λ = {lam:g}", fontsize=10)
            ax.set_xlabel("global step")
            ax.set_ylabel("composite reward")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8, frameon=False)
        fig.suptitle("Reproducibility of shared λ across sweeps", y=1.03, fontsize=12)
        fig.tight_layout()
        return fig

    _plot_overlap(hist_all)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6 · Final-step summary vs λ

    The headline question for the λ-axis: does increasing the contrastive weight
    monotonically raise the final composite reward, and at what cost to entropy /
    reward variance? Solid markers = `full (05-13)`, hollow = `partial (05-15)`.
    """)
    return


@app.cell
def _(plt, runs_df):
    def _plot_final(df):
        if df.empty:
            print("No runs.")
            return None
        panels = [
            ("final_reward", "final composite reward"),
            ("final_reward_std", "final reward std"),
            ("final_entropy", "final policy entropy"),
            ("final_loss", "final GRPO loss"),
        ]
        fig, axes = plt.subplots(1, 4, figsize=(18, 4))
        marker = {"full (05-13)": "o", "partial (05-15)": "s"}
        fill = {"full (05-13)": "full", "partial (05-15)": "none"}
        for ax, (col, title) in zip(axes, panels):
            for sweep in df["sweep"].cat.categories:
                g = df[df["sweep"] == sweep].sort_values("lambda")
                if g.empty or g[col].dropna().empty:
                    continue
                ax.plot(
                    g["lambda"],
                    g[col],
                    marker=marker[sweep],
                    fillstyle=fill[sweep],
                    lw=1.3,
                    label=sweep,
                )
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("contrastive λ")
            ax.grid(True, alpha=0.25)
        axes[0].legend(fontsize=8, frameon=False)
        fig.suptitle("Final-step training metrics vs contrastive λ", y=1.03, fontsize=12)
        fig.tight_layout()
        return fig

    _plot_final(runs_df)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7 · Save consolidated tables

    Drop the run inventory and the stacked training history into `tables/` so the
    paper-table builder and downstream notebooks can consume them without re-hitting
    W&B.
    """)
    return


@app.cell
def _(TABLES_DIR, hist_all, runs_df):
    def _save():
        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        written = []
        if not runs_df.empty:
            p = TABLES_DIR / "lambda_axis_2026_05_runs.csv"
            runs_df.assign(sweep=runs_df["sweep"].astype(str)).to_csv(p, index=False)
            written.append(p.name)
        if not hist_all.empty:
            p = TABLES_DIR / "lambda_axis_2026_05_history.parquet"
            hist_all.to_parquet(p)
            written.append(p.name)
        return written

    print("wrote:", _save())
    return


if __name__ == "__main__":
    app.run()
