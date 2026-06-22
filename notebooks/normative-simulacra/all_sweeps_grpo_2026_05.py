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
    # COLM (λ × ρ) GRPO sweeps — comparative training analysis

    Four SLURM multiruns together cover the 15-cell `(contrastive_lambda × contrastive_ratio)`
    grid described in §A.4 / `app:grpo-ablation-viz`:

    | sweep | multirun dir | cells |
    |---|---|---|
    | `lambda-axis (05-13)`         | `multirun/2026-05-13_lambda_axis_sweep/12-31-50`         | λ ∈ {0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0}, ρ=0 |
    | `lambda-axis-partial (05-15)` | `multirun/2026-05-15_lambda_axis_partial_sweep/21-58-26` | λ ∈ {1.0, 1.5, 2.0}, ρ=0 (rerun of the corrupted cells) |
    | `ratio-axis (05-17)`          | `multirun/2026-05-17_ratio_axis_sweep/10-20-32`          | ρ ∈ {0.05, 0.10, 0.20, 0.50}, λ=1.0 |
    | `offaxis (05-18)`             | `multirun/2026-05-18_offaxis_sweep/10-28-09`             | (λ, ρ) ∈ {0.5, 1.5} × {0.10, 0.50} |

    For λ ∈ {1.0, 1.5, 2.0}, ρ=0 the partial rerun is the **canonical** trace; the
    original `lambda_axis_sweep` numbers for those three cells are kept for
    reproducibility checks (§5) but flagged as corrupted upstream because the judge /
    embedding servers timed out partway through training.

    Self-contained fetch + cache lives in `wandb_cache/all_sweeps/` next to this
    notebook: `runs.json` (one row per W&B run), `history/<run_id>.parquet` (full
    training history), `meta.json`. Press **Fetch / refresh** below to (re)build it;
    otherwise the loader just reads the cache offline.

    The `(λ, ρ)` coordinate for each W&B run is parsed from `config.output_dir` — the
    only key that disambiguates pre-fix runs (the old `contrastive:` tag conflated
    `contrastive_ratio` with `contrastive_lambda`; only the May 17–18 sweeps carry
    the new `clambda:` / `cratio:` tags).
    """)
    return


@app.cell
def _():
    import json
    import re
    import sys
    from datetime import datetime, timezone
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    PROJECT_ROOT = Path("/share/pierson/matt/UAIR")
    NB_DIR = PROJECT_ROOT / "notebooks/normative-simulacra"
    CACHE_DIR = NB_DIR / "wandb_cache" / "all_sweeps"
    HIST_DIR = CACHE_DIR / "history"
    TABLES_DIR = NB_DIR / "tables"

    WANDB_ENTITY = "uair"
    WANDB_PROJECT = "grpo-ci-training"

    # Maps the unique multirun fragment in W&B's config.output_dir to (sweep_label,
    # sweep_kind). sweep_kind drives which axis a cell varies along and how it
    # should be plotted.
    SWEEPS = {
        "2026-05-13_lambda_axis_sweep/12-31-50": (
            "lambda-axis (05-13)",
            "lambda",
        ),
        "2026-05-15_lambda_axis_partial_sweep/21-58-26": (
            "lambda-axis-partial (05-15)",
            "lambda",
        ),
        "2026-05-17_ratio_axis_sweep/10-20-32": (
            "ratio-axis (05-17)",
            "ratio",
        ),
        "2026-05-18_offaxis_sweep/10-28-09": (
            "offaxis (05-18)",
            "offaxis",
        ),
    }
    SWEEP_ORDER = [
        "lambda-axis (05-13)",
        "lambda-axis-partial (05-15)",
        "ratio-axis (05-17)",
        "offaxis (05-18)",
    ]
    SWEEP_KIND_ORDER = ["lambda", "ratio", "offaxis"]
    # λ ∈ {1.0, 1.5, 2.0}, ρ=0 cells in the 05-13 run are corrupted (judge timeouts);
    # the 05-15 partial sweep is the canonical source for those cells.
    CORRUPTED_LAMBDAS_05_13 = {1.0, 1.5, 2.0}

    # Training metrics worth tracking over global_step. Mix of GRPO/TRL aggregate
    # logs and per-component composite-reward breakouts.
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

    # Per-call reward component keys (offline view from reward_traces.jsonl).
    COMPONENT_KEYS = [
        "r_uncert",
        "r_complete",
        "r_consist",
        "r_context",
        "r_cohere",
        "r_ground",
    ]

    pd.set_option("display.max_columns", 60)
    pd.set_option("display.float_format", "{:.4f}".format)

    sys.path.insert(0, str(NB_DIR.parent))
    try:
        from font_utils import load_ibm_plex_sans

        load_ibm_plex_sans()
    except Exception as _e:
        print(f"[font_utils] skipped: {_e}")
    return (
        CACHE_DIR,
        COMPONENT_KEYS,
        CORRUPTED_LAMBDAS_05_13,
        HISTORY_KEYS,
        HIST_DIR,
        LinearSegmentedColormap,
        Normalize,
        PROJECT_ROOT,
        SWEEPS,
        SWEEP_KIND_ORDER,
        SWEEP_ORDER,
        TABLES_DIR,
        WANDB_ENTITY,
        WANDB_PROJECT,
        datetime,
        json,
        np,
        pd,
        plt,
        re,
        timezone,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · Scan local artifacts

    Walk each sweep's multirun dir and pull `(λ, ρ)` straight from
    `training_metadata.json`. This is the authoritative cell inventory (W&B may
    be missing finished-but-not-uploaded runs, or carry extra orchestrator runs);
    we use this to (a) sanity-check that every requested cell finished and
    (b) drive the offline `reward_traces.jsonl` analysis in §6.
    """)
    return


@app.cell
def _(PROJECT_ROOT, SWEEPS, SWEEP_ORDER, json, pd):
    def _scan_local_cells():
        rows = []
        for frag, (sweep_label, sweep_kind) in SWEEPS.items():
            run_dir = PROJECT_ROOT / "multirun" / frag
            if not run_dir.exists():
                continue
            for cell_dir in sorted(run_dir.iterdir()):
                if not cell_dir.is_dir():
                    continue
                ckpt_dir = (
                    cell_dir
                    / "grpo_only_online_external"
                    / "outputs"
                    / "grpo"
                    / "checkpoint"
                )
                meta_path = ckpt_dir / "training_metadata.json"
                traces_path = ckpt_dir / "reward_traces.jsonl"
                if not meta_path.exists():
                    continue
                with meta_path.open() as f:
                    meta = json.load(f)
                rows.append(
                    {
                        "sweep": sweep_label,
                        "sweep_kind": sweep_kind,
                        "cell": cell_dir.name,
                        "lambda": float(meta.get("contrastive_lambda", float("nan"))),
                        "ratio": float(meta.get("contrastive_ratio", float("nan"))),
                        "n_training_rows": meta.get("n_training_rows"),
                        "n_flow_chunks": meta.get("n_flow_chunks"),
                        "n_no_flow_chunks": meta.get("n_no_flow_chunks"),
                        "reward_weights": meta.get("reward_weights"),
                        "judgment_reward_weights": meta.get("judgment_reward_weights"),
                        "no_flow_scoring": meta.get("no_flow_scoring"),
                        "online_rground": meta.get("online_rground"),
                        "enable_thinking": meta.get("enable_thinking_grpo"),
                        "base_model": meta.get("base_model"),
                        "sft_checkpoint": meta.get("sft_checkpoint"),
                        "ckpt_dir": str(ckpt_dir),
                        "traces_path": str(traces_path) if traces_path.exists() else None,
                        "has_merged_sft": (ckpt_dir / "_merged_sft").exists(),
                    }
                )
        df = pd.DataFrame(rows)
        if not df.empty:
            df["sweep"] = pd.Categorical(df["sweep"], SWEEP_ORDER, ordered=True)
            df = df.sort_values(["sweep", "lambda", "ratio"]).reset_index(drop=True)
        return df

    local_cells = _scan_local_cells()
    print(
        f"Local cells: {len(local_cells)} across "
        f"{local_cells['sweep'].nunique() if len(local_cells) else 0} sweeps"
    )
    local_cells
    return (local_cells,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · Fetch / refresh the W&B cache

    Pulls every `grpo_training` run in `uair/grpo-ci-training` whose
    `config.output_dir` belongs to one of the four sweep dirs. One W&B run per cell.
    """)
    return


@app.cell
def _(mo):
    fetch_button = mo.ui.run_button(label="Fetch / refresh from W&B")
    force_switch = mo.ui.switch(label="force overwrite history parquets", value=False)
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
        """Return (sweep_label, sweep_kind, lambda, ratio) parsed from a run's
        config.output_dir."""
        if not output_dir:
            return None, None, None, None
        sweep_label, sweep_kind = None, None
        for frag, (label, kind) in SWEEPS.items():
            if frag in output_dir:
                sweep_label, sweep_kind = label, kind
                break
        if sweep_label is None:
            return None, None, None, None
        m_l = re.search(r"lambda=([0-9]+(?:\.[0-9]+)?)", output_dir)
        m_r = re.search(r"ratio=([0-9]+(?:\.[0-9]+)?)", output_dir)
        lam = float(m_l.group(1)) if m_l else None
        rho = float(m_r.group(1)) if m_r else None
        # The two pure-λ sweeps don't put ratio= in their subdir name; ρ=0 there.
        if sweep_kind == "lambda" and rho is None:
            rho = 0.0
        # The ratio-axis sweep pins λ=1.0 and doesn't put lambda= in the subdir name.
        if sweep_kind == "ratio" and lam is None:
            lam = 1.0
        return sweep_label, sweep_kind, lam, rho

    def _enrich(run, sweep_label, sweep_kind, lam, rho):
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
            "sweep_kind": sweep_kind,
            "lambda": lam,
            "ratio": rho,
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
            "final_kl": summary.get("train/kl"),
            "final_grad_norm": summary.get("train/grad_norm"),
            "final_mean_completion_len": summary.get("train/completions/mean_length"),
            "train_runtime_s": summary.get("train_runtime"),
        }

    def _fetch(force: bool):
        import wandb

        HIST_DIR.mkdir(parents=True, exist_ok=True)
        api = wandb.Api()
        path = f"{WANDB_ENTITY}/{WANDB_PROJECT}"
        # Match the grpo_training stage for any of our four sweeps. Orchestrator
        # runs (no training history) get filtered out by the output_dir parse.
        raw = api.runs(
            path,
            filters={
                "display_name": {
                    "$regex": "(lambda_axis|lambda_axis_partial|ratio_axis|offaxis).*grpo_training"
                }
            },
        )
        log = [f"scanned {len(raw)} candidate runs in {path}"]

        enriched = []
        for run in raw:
            sweep_label, sweep_kind, lam, rho = _parse_output_dir(
                run.config.get("output_dir", "")
            )
            if sweep_label is None or lam is None or rho is None:
                continue
            rec = _enrich(run, sweep_label, sweep_kind, lam, rho)
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

        enriched.sort(
            key=lambda r: (r["sweep"], r["lambda"], r["ratio"], r["created_at"])
        )
        with open(CACHE_DIR / "runs.json", "w") as f:
            json.dump(enriched, f, indent=2, default=str)
        meta = {
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "entity": WANDB_ENTITY,
            "project": WANDB_PROJECT,
            "sweeps": {k: v[0] for k, v in SWEEPS.items()},
            "n_runs": len(enriched),
        }
        with open(CACHE_DIR / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        log.append(f"cached {len(enriched)} grpo_training runs -> {CACHE_DIR}")
        return enriched, log

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
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
    ## 3 · Load the cache + dedupe to canonical cells

    `runs.json` → one row per `(sweep, λ, ρ)` training run with final-step scalars.

    For the four cells where λ ∈ {1.0, 1.5, 2.0}, ρ=0 the original `lambda-axis
    (05-13)` run is **superseded** by the 05-15 partial rerun (the originals
    trained against a stale judge/embedding server). `canonical_runs` picks the
    rerun whenever both are present; `runs_df` keeps everything for §5
    reproducibility plots.
    """)
    return


@app.cell
def _(
    CACHE_DIR,
    CORRUPTED_LAMBDAS_05_13,
    HIST_DIR,
    SWEEP_ORDER,
    fetch_log,
    json,
    pd,
):
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
                    "sweep_kind": r.get("sweep_kind"),
                    "lambda": r.get("lambda"),
                    "ratio": r.get("ratio"),
                    "run_id": r.get("run_id"),
                    "run_name": r.get("run_name"),
                    "state": r.get("state"),
                    "created_at": r.get("created_at"),
                    "final_step": r.get("final_step"),
                    "final_reward": r.get("final_reward"),
                    "final_reward_std": r.get("final_reward_std"),
                    "final_loss": r.get("final_loss"),
                    "final_entropy": r.get("final_entropy"),
                    "final_kl": r.get("final_kl"),
                    "final_grad_norm": r.get("final_grad_norm"),
                    "final_mean_completion_len": r.get("final_mean_completion_len"),
                    "train_runtime_s": r.get("train_runtime_s"),
                    "reward_weights": r.get("reward_weights"),
                    "run_url": r.get("run_url"),
                }
            )
        df = pd.DataFrame(rows)
        if not df.empty:
            df["sweep"] = pd.Categorical(df["sweep"], SWEEP_ORDER, ordered=True)
            df = df.sort_values(["sweep", "lambda", "ratio"]).reset_index(drop=True)
        return df

    def _canonicalize(df):
        """Drop the 05-13 entries for λ ∈ {1.0, 1.5, 2.0}, ρ=0 in favour of the
        05-15 partial-sweep rerun."""
        if df.empty:
            return df
        mask_drop = (
            (df["sweep"] == "lambda-axis (05-13)")
            & (df["ratio"] == 0.0)
            & (df["lambda"].isin(CORRUPTED_LAMBDAS_05_13))
        )
        canon = df[~mask_drop].copy()
        # If for some reason both partial and original were dropped, fall back —
        # not expected in this dataset, but guard anyway.
        canon = canon.drop_duplicates(subset=["lambda", "ratio"], keep="last")
        return canon.sort_values(["lambda", "ratio"]).reset_index(drop=True)

    def load_history(run_id: str) -> pd.DataFrame:
        p = HIST_DIR / f"{run_id}.parquet"
        if not p.exists():
            return pd.DataFrame()
        h = pd.read_parquet(p)
        if "train/global_step" in h.columns:
            h = h.dropna(subset=["train/global_step"]).sort_values(
                "train/global_step"
            )
        return h

    _ = fetch_log  # refresh after a button press
    runs_df = _load_runs()
    canonical_runs = _canonicalize(runs_df)
    print(
        f"All W&B runs: {len(runs_df)}  |  Canonical cells: {len(canonical_runs)}  "
        f"(unique (λ, ρ) tuples after dedup)"
    )
    canonical_runs[
        [
            "sweep",
            "lambda",
            "ratio",
            "state",
            "final_step",
            "final_reward",
            "final_reward_std",
            "final_entropy",
            "train_runtime_s",
        ]
    ]
    return canonical_runs, load_history, runs_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · Cell inventory — (λ, ρ) grid

    Sanity check that every requested cell finished and exists in W&B. Empty
    cells mean either (a) the cell was never run (most non-diagonal entries are
    intentional gaps — the sweep is L-shaped + corners, not a full grid) or
    (b) the run failed to log to W&B.
    """)
    return


@app.cell
def _(canonical_runs, local_cells, pd):
    def _inventory_grid(df, value_col, label):
        if df.empty:
            print(f"No data for {label}.")
            return pd.DataFrame()
        piv = df.pivot_table(
            index="lambda",
            columns="ratio",
            values=value_col,
            aggfunc="first",
        )
        piv.columns.name = "ρ"
        piv.index.name = "λ"
        print(f"\n{label} — values = {value_col}")
        return piv

    print("== Cell coverage from local artifacts (canonical only) ==")
    _local_canon = local_cells.copy()
    # Drop the corrupted 05-13 cells from local view too, to match canonical_runs.
    _local_canon = _local_canon[
        ~(
            (_local_canon["sweep"] == "lambda-axis (05-13)")
            & (_local_canon["ratio"] == 0.0)
            & (_local_canon["lambda"].isin({1.0, 1.5, 2.0}))
        )
    ]
    _inventory_grid(_local_canon, "n_training_rows", "local: n_training_rows per cell")
    return


@app.cell
def _(canonical_runs):
    def _inv(df, col):
        return (
            df.pivot_table(
                index="lambda", columns="ratio", values=col, aggfunc="first"
            ).rename_axis(index="λ", columns="ρ")
            if not df.empty
            else None
        )

    print("== W&B final-step composite reward per canonical cell ==")
    _inv(canonical_runs, "final_reward")
    return


@app.cell
def _(canonical_runs):
    n_unfinished = (
        int((canonical_runs["state"] != "finished").sum())
        if len(canonical_runs)
        else 0
    )
    if n_unfinished:
        print(f"!! {n_unfinished} canonical run(s) not in state=finished:")
        unfinished = canonical_runs[canonical_runs["state"] != "finished"][
            ["sweep", "lambda", "ratio", "run_id", "state", "run_url"]
        ]
    else:
        print("All canonical W&B runs are state=finished.")
        unfinished = None
    unfinished
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 · Reproducibility — 05-13 vs 05-15 partial rerun

    λ ∈ {1.0, 1.5, 2.0}, ρ=0 were trained in both `lambda-axis (05-13)` and
    `lambda-axis-partial (05-15)`. The reward trajectories of the partial rerun
    should diverge from the originals in the regime where the original's judge
    server was timing out (later steps), confirming the rationale for the rerun.
    Solid = corrupted 05-13, dashed = canonical 05-15.
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
            h["sweep_kind"] = r["sweep_kind"]
            h["lambda"] = r["lambda"]
            h["ratio"] = r["ratio"]
            h["run_id"] = r["run_id"]
            frames.append(h)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    hist_all = _stack_histories(runs_df)
    print(
        f"Stacked history (ALL runs incl. corrupted): {len(hist_all)} rows, "
        f"{hist_all['run_id'].nunique() if len(hist_all) else 0} runs"
    )
    hist_all.head()
    return (hist_all,)


@app.cell
def _(hist_all, plt):
    def _plot_repro(hist):
        if hist.empty:
            print("No history to plot.")
            return None
        shared = sorted(
            set(
                hist[hist["sweep"] == "lambda-axis (05-13)"]["lambda"].dropna()
            )
            & set(
                hist[hist["sweep"] == "lambda-axis-partial (05-15)"]["lambda"].dropna()
            )
        )
        # Only the corrupted three should overlap.
        if not shared:
            print("No λ overlap between 05-13 and 05-15 in cache.")
            return None
        fig, axes = plt.subplots(
            2, len(shared), figsize=(4.5 * len(shared), 7), squeeze=False
        )
        style = {"lambda-axis (05-13)": "-", "lambda-axis-partial (05-15)": "--"}
        color = {
            "lambda-axis (05-13)": "#b07a7a",
            "lambda-axis-partial (05-15)": "#3a6da9",
        }
        for col_i, lam in enumerate(shared):
            for row_i, metric in enumerate(
                ["train/reward", "train/rewards/composite_ci_reward/mean"]
            ):
                ax = axes[row_i][col_i]
                plotted = False
                for sweep in (
                    "lambda-axis (05-13)",
                    "lambda-axis-partial (05-15)",
                ):
                    g = hist[
                        (hist["sweep"] == sweep)
                        & (hist["lambda"] == lam)
                        & (hist["ratio"] == 0.0)
                    ].sort_values("train/global_step")
                    if g.empty or metric not in g or g[metric].dropna().empty:
                        continue
                    ax.plot(
                        g["train/global_step"],
                        g[metric],
                        style[sweep],
                        color=color[sweep],
                        marker="o",
                        ms=2.5,
                        lw=1.4,
                        label=sweep.split(" ")[0],
                    )
                    plotted = True
                ax.set_title(
                    f"λ = {lam:g}, ρ = 0  ({metric.rsplit('/', 1)[-1]})",
                    fontsize=9,
                )
                ax.grid(True, alpha=0.25)
                if row_i == 1:
                    ax.set_xlabel("global step")
                if col_i == 0:
                    ax.set_ylabel("reward")
                if plotted and col_i == 0:
                    ax.legend(fontsize=7, frameon=False)
        fig.suptitle(
            "Reproducibility — 05-13 (judge timed out) vs 05-15 (canonical)",
            y=1.0,
            fontsize=12,
        )
        fig.tight_layout()
        return fig

    _plot_repro(hist_all)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6 · Comparative training curves — canonical cells only

    Two figures:

    - **λ axis** (ρ=0): how composite reward, reward variance, entropy and grad
      norm respond to the contrastive weight at ρ=0.
    - **ρ axis** (λ=1.0): the same diagnostics under the canonical λ=1.0 setting
      while ρ grows from 0 to 0.5.
    - **off-axis corners**: how the four bracket cells track the lambda-axis and
      ratio-axis cells they bound.

    Colour encodes the swept variable; line width / style separates the three
    sub-views.
    """)
    return


@app.cell
def _(canonical_runs, hist_all):
    canonical_history = (
        hist_all.merge(
            canonical_runs[["run_id"]].drop_duplicates(),
            on="run_id",
            how="inner",
        )
        if not hist_all.empty
        else hist_all
    )
    print(
        f"Canonical history: {len(canonical_history)} rows, "
        f"{canonical_history['run_id'].nunique() if len(canonical_history) else 0} runs"
    )
    return (canonical_history,)


@app.cell
def _(LinearSegmentedColormap, canonical_history, plt):
    def _grid_plot(hist, axis):
        """axis ∈ {'lambda', 'ratio'} — pick which sub-slice + colour key to use."""
        if hist.empty:
            return None
        if axis == "lambda":
            sub = hist[hist["ratio"] == 0.0].copy()
            color_key = "lambda"
            title = "λ axis (ρ = 0)"
            cmap = plt.get_cmap("viridis")
        else:
            sub = hist[hist["lambda"] == 1.0].copy()
            color_key = "ratio"
            title = "ρ axis (λ = 1.0)"
            cmap = LinearSegmentedColormap.from_list(
                "mako", ["#1c2c5b", "#3a6da9", "#7fb2e0", "#cfe5f5"]
            )
        if sub.empty:
            print(f"No data for {title}")
            return None
        vals = sorted(sub[color_key].dropna().unique())
        lo, hi = (min(vals), max(vals)) if len(vals) > 1 else (vals[0] - 0.5, vals[0] + 0.5)
        panels = [
            ("train/reward", "composite reward"),
            ("train/reward_std", "reward std (within-group)"),
            ("train/entropy", "policy entropy"),
            ("train/loss", "GRPO loss"),
            ("train/completions/mean_length", "mean completion length"),
            ("train/grad_norm", "grad norm"),
            ("train/kl", "KL to ref"),
            ("train/frac_reward_zero_std", "frac. groups w/ zero reward std"),
        ]
        panels = [(c, t) for c, t in panels if c in sub.columns]
        fig, axgrid = plt.subplots(
            2, 4, figsize=(16, 7.5), squeeze=False, sharex=True
        )
        for i, (metric, label) in enumerate(panels):
            ax = axgrid[i // 4][i % 4]
            for v in vals:
                g = sub[sub[color_key] == v].sort_values("train/global_step")
                if metric not in g or g[metric].dropna().empty:
                    continue
                c = cmap((v - lo) / (hi - lo)) if hi > lo else cmap(0.5)
                ax.plot(
                    g["train/global_step"],
                    g[metric],
                    marker="o",
                    ms=2.6,
                    lw=1.3,
                    color=c,
                    label=f"{color_key[0]}={v:g}",
                )
            ax.set_title(label, fontsize=9)
            ax.grid(True, alpha=0.25)
            if i // 4 == 1:
                ax.set_xlabel("global step", fontsize=8)
        axgrid[0][0].legend(fontsize=7, frameon=False, ncol=2)
        fig.suptitle(
            f"{title} — training curves (canonical cells, W&B history)",
            y=1.0,
            fontsize=12,
        )
        fig.tight_layout()
        return fig

    _grid_plot(canonical_history, "lambda")
    return (_grid_plot,)


@app.cell
def _(_grid_plot, canonical_history):
    _grid_plot(canonical_history, "ratio")
    return


@app.cell
def _(canonical_history, plt):
    def _plot_offaxis_curves(hist):
        if hist.empty:
            return None
        off = hist[hist["sweep_kind"] == "offaxis"].copy()
        if off.empty:
            print("No off-axis history in cache.")
            return None
        # Show each off-axis cell against the lambda-axis cell with matching λ and
        # the ratio-axis cell with matching ρ, so the corner sits between its two
        # 1-D references.
        ref_lam = hist[(hist["sweep_kind"] == "lambda")]
        ref_rho = hist[(hist["sweep_kind"] == "ratio")]
        cells = sorted(
            off.groupby(["lambda", "ratio"]).size().index.tolist()
        )
        fig, axes = plt.subplots(1, len(cells), figsize=(4.6 * len(cells), 4.2), squeeze=False)
        for ax, (lam, rho) in zip(axes[0], cells):
            for label, sub, ls, c in [
                (
                    f"λ-axis @ λ={lam:g}, ρ=0",
                    ref_lam[(ref_lam["lambda"] == lam) & (ref_lam["ratio"] == 0.0)],
                    "--",
                    "#888888",
                ),
                (
                    f"ρ-axis @ λ=1.0, ρ={rho:g}",
                    ref_rho[(ref_rho["lambda"] == 1.0) & (ref_rho["ratio"] == rho)],
                    ":",
                    "#888888",
                ),
                (
                    f"off-axis ({lam:g},{rho:g})",
                    off[(off["lambda"] == lam) & (off["ratio"] == rho)],
                    "-",
                    "#c44e52",
                ),
            ]:
                g = sub.sort_values("train/global_step")
                if g.empty or "train/reward" not in g:
                    continue
                ax.plot(
                    g["train/global_step"],
                    g["train/reward"],
                    ls,
                    color=c,
                    marker="o",
                    ms=2.6,
                    lw=1.4,
                    label=label,
                )
            ax.set_title(f"corner (λ={lam:g}, ρ={rho:g})", fontsize=10)
            ax.set_xlabel("global step")
            ax.set_ylabel("composite reward")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=7, frameon=False)
        fig.suptitle(
            "Off-axis corners vs their 1-D references",
            y=1.02,
            fontsize=12,
        )
        fig.tight_layout()
        return fig

    _plot_offaxis_curves(canonical_history)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7 · Final-step landscape over (λ, ρ)

    The headline view: how every canonical cell compares on its final
    composite reward / reward-std / entropy. A heatmap on the (λ, ρ) grid
    highlights the off-axis corners' interaction with the two pinned axes.
    """)
    return


@app.cell
def _(Normalize, canonical_runs, np, plt):
    def _heatmap(df, value, title, cmap_name="viridis"):
        if df.empty:
            return None
        lams = sorted(df["lambda"].dropna().unique())
        rhos = sorted(df["ratio"].dropna().unique())
        grid = np.full((len(lams), len(rhos)), np.nan)
        for _, r in df.iterrows():
            if r["lambda"] in lams and r["ratio"] in rhos:
                i, j = lams.index(r["lambda"]), rhos.index(r["ratio"])
                grid[i, j] = r[value]
        fig, ax = plt.subplots(figsize=(0.9 * len(rhos) + 3, 0.55 * len(lams) + 2))
        finite = grid[np.isfinite(grid)]
        if finite.size == 0:
            print(f"{title}: nothing to plot.")
            plt.close(fig)
            return None
        norm = Normalize(vmin=finite.min(), vmax=finite.max())
        im = ax.imshow(grid, cmap=cmap_name, aspect="auto", norm=norm, origin="lower")
        ax.set_xticks(range(len(rhos)), [f"{r:g}" for r in rhos])
        ax.set_yticks(range(len(lams)), [f"{l:g}" for l in lams])
        ax.set_xlabel("contrastive ratio ρ")
        ax.set_ylabel("contrastive λ")
        for i in range(len(lams)):
            for j in range(len(rhos)):
                v = grid[i, j]
                if np.isfinite(v):
                    ax.text(
                        j,
                        i,
                        f"{v:.3f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white" if norm(v) < 0.5 else "black",
                    )
                else:
                    ax.text(j, i, "·", ha="center", va="center", color="#666", fontsize=10)
        ax.set_title(title, fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
        fig.tight_layout()
        return fig

    _heatmap(canonical_runs, "final_reward", "final composite reward")
    return (_heatmap,)


@app.cell
def _(_heatmap, canonical_runs):
    _heatmap(canonical_runs, "final_reward_std", "final reward std (within-group)", "magma")
    return


@app.cell
def _(_heatmap, canonical_runs):
    _heatmap(canonical_runs, "final_entropy", "final policy entropy", "plasma")
    return


@app.cell
def _(canonical_runs, plt):
    def _line_summary(df):
        if df.empty:
            return None
        panels = [
            ("final_reward", "final composite reward"),
            ("final_reward_std", "final reward std"),
            ("final_entropy", "final policy entropy"),
            ("final_mean_completion_len", "final mean completion length"),
        ]
        fig, axes = plt.subplots(1, 4, figsize=(18, 4))
        # Two 1-D slices on each panel: λ axis at ρ=0, ρ axis at λ=1.0. Off-axis
        # corners overlay as crosses.
        for ax, (col, title) in zip(axes, panels):
            lam_slice = df[df["ratio"] == 0.0].sort_values("lambda")
            if not lam_slice.empty:
                ax.plot(
                    lam_slice["lambda"],
                    lam_slice[col],
                    marker="o",
                    lw=1.4,
                    color="#3a6da9",
                    label="λ axis (ρ=0)",
                )
            rho_slice = df[df["lambda"] == 1.0].sort_values("ratio")
            if not rho_slice.empty:
                # plot vs ρ on a twin x for the ratio axis; for legibility, mark
                # only the values rather than overlay an axis (the off-axis dots
                # use λ on the x).
                pass
            corners = df[df["sweep_kind"] == "offaxis"].reset_index(drop=True)
            for ci, r in corners.iterrows():
                ax.scatter(
                    r["lambda"],
                    r[col],
                    marker="x",
                    s=70,
                    color="#c44e52",
                    label="off-axis" if ci == 0 else None,
                )
                ax.annotate(
                    f"ρ={r['ratio']:g}",
                    (r["lambda"], r[col]),
                    fontsize=7,
                    xytext=(4, 4),
                    textcoords="offset points",
                    color="#c44e52",
                )
            ax.set_xlabel("contrastive λ")
            ax.set_title(title, fontsize=10)
            ax.grid(True, alpha=0.25)
        axes[0].legend(fontsize=8, frameon=False)
        fig.suptitle(
            "Final-step metrics — λ axis line + off-axis corners (red ×)",
            y=1.03,
            fontsize=12,
        )
        fig.tight_layout()
        return fig

    _line_summary(canonical_runs)
    return


@app.cell
def _(canonical_runs, plt):
    def _ratio_summary(df):
        if df.empty:
            return None
        panels = [
            ("final_reward", "final composite reward"),
            ("final_reward_std", "final reward std"),
            ("final_entropy", "final policy entropy"),
            ("final_mean_completion_len", "final mean completion length"),
        ]
        rho_slice = df[df["lambda"] == 1.0].sort_values("ratio")
        if rho_slice.empty:
            print("No ρ-axis cells.")
            return None
        fig, axes = plt.subplots(1, 4, figsize=(18, 4))
        for ax, (col, title) in zip(axes, panels):
            ax.plot(
                rho_slice["ratio"],
                rho_slice[col],
                marker="o",
                lw=1.4,
                color="#7c5295",
                label="ρ axis (λ=1.0)",
            )
            corners = df[df["sweep_kind"] == "offaxis"].reset_index(drop=True)
            for ci, r in corners.iterrows():
                ax.scatter(
                    r["ratio"],
                    r[col],
                    marker="x",
                    s=70,
                    color="#c44e52",
                    label="off-axis" if ci == 0 else None,
                )
                ax.annotate(
                    f"λ={r['lambda']:g}",
                    (r["ratio"], r[col]),
                    fontsize=7,
                    xytext=(4, 4),
                    textcoords="offset points",
                    color="#c44e52",
                )
            ax.set_xlabel("contrastive ratio ρ")
            ax.set_title(title, fontsize=10)
            ax.grid(True, alpha=0.25)
        axes[0].legend(fontsize=8, frameon=False)
        fig.suptitle(
            "Final-step metrics — ρ axis line + off-axis corners (red ×)",
            y=1.03,
            fontsize=12,
        )
        fig.tight_layout()
        return fig

    _ratio_summary(canonical_runs)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8 · Offline per-component view from `reward_traces.jsonl`

    W&B logs aggregate `train/reward` but not the per-call composite
    decomposition. Each cell's `reward_traces.jsonl` records, for every prompt
    seen during GRPO rollout, the six unweighted components (`r_uncert`,
    `r_complete`, `r_consist`, `r_context`, `r_cohere`, `r_ground`) plus the
    weighted/composite values and whether the sample was contrastive. We bin by
    `call` (one logged group of generations per optimizer step) and average
    over the `idx` dimension to get a per-step component trajectory.

    The grounding component (`r_ground`) is the canary the contrastive weight
    is supposed to push on — if λ is doing its job, `r_ground` should rise more
    aggressively for larger λ.
    """)
    return


@app.cell
def _(COMPONENT_KEYS, canonical_runs, json, local_cells, pd):
    def _load_trace_summary(traces_path, smooth_calls=5):
        if not traces_path:
            return pd.DataFrame()
        rows = []
        with open(traces_path) as f:
            for line in f:
                r = json.loads(line)
                comp = r.get("components") or {}
                row = {
                    "call": r.get("call"),
                    "idx": r.get("idx"),
                    "is_contrastive": bool(r.get("is_contrastive")),
                    "is_no_flow": bool(r.get("is_no_flow")),
                    "composite": r.get("composite"),
                    "completion_len": r.get("completion_len"),
                }
                for k in COMPONENT_KEYS:
                    row[k] = comp.get(k)
                rows.append(row)
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        per_call = (
            df.groupby("call", as_index=False)[
                COMPONENT_KEYS + ["composite", "completion_len"]
            ]
            .mean(numeric_only=True)
            .sort_values("call")
        )
        per_call_smooth = per_call.copy()
        for k in COMPONENT_KEYS + ["composite", "completion_len"]:
            per_call_smooth[k] = (
                per_call[k]
                .rolling(window=smooth_calls, min_periods=1, center=True)
                .mean()
            )
        return per_call_smooth

    def _collect_trace_summaries():
        # Join local cells (which have traces_path) with canonical run identity
        # (lambda, ratio, sweep) so the figure colour-keying matches §6.
        cells = local_cells[
            ~(
                (local_cells["sweep"] == "lambda-axis (05-13)")
                & (local_cells["ratio"] == 0.0)
                & (local_cells["lambda"].isin({1.0, 1.5, 2.0}))
            )
        ].copy()
        frames = []
        for _, c in cells.iterrows():
            tp = c["traces_path"]
            if not tp:
                continue
            tsum = _load_trace_summary(tp)
            if tsum.empty:
                continue
            tsum["sweep"] = c["sweep"]
            tsum["sweep_kind"] = c["sweep_kind"]
            tsum["lambda"] = c["lambda"]
            tsum["ratio"] = c["ratio"]
            frames.append(tsum)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    trace_summary = _collect_trace_summaries()
    print(
        f"Per-call component trajectories: {len(trace_summary)} rows across "
        f"{trace_summary.groupby(['lambda', 'ratio']).ngroups if len(trace_summary) else 0} canonical cells"
    )
    # also produce a per-cell final-call snapshot
    if not trace_summary.empty:
        final_components = (
            trace_summary.sort_values("call")
            .groupby(["sweep", "lambda", "ratio"], observed=True)
            .tail(8)  # last 8 calls ≈ tail-window mean
            .groupby(["sweep", "lambda", "ratio"], observed=True, as_index=False)[
                COMPONENT_KEYS + ["composite"]
            ]
            .mean(numeric_only=True)
            .sort_values(["lambda", "ratio"])
        )
    else:
        final_components = pd.DataFrame()
    _ = canonical_runs  # acknowledge dep so cell re-runs after refresh
    final_components
    return final_components, trace_summary


@app.cell
def _(COMPONENT_KEYS, plt, trace_summary):
    def _plot_components(ts, axis):
        if ts.empty:
            return None
        if axis == "lambda":
            sub = ts[ts["ratio"] == 0.0].copy()
            color_key = "lambda"
            title = "λ axis (ρ = 0)"
            cmap = plt.get_cmap("viridis")
        else:
            sub = ts[ts["lambda"] == 1.0].copy()
            color_key = "ratio"
            title = "ρ axis (λ = 1.0)"
            cmap = plt.get_cmap("cividis")
        if sub.empty:
            return None
        vals = sorted(sub[color_key].dropna().unique())
        lo, hi = (min(vals), max(vals)) if len(vals) > 1 else (vals[0] - 0.5, vals[0] + 0.5)
        fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
        for ax, k in zip(axes.flat, COMPONENT_KEYS):
            for v in vals:
                g = sub[sub[color_key] == v].sort_values("call")
                if g.empty or g[k].dropna().empty:
                    continue
                c = cmap((v - lo) / (hi - lo)) if hi > lo else cmap(0.5)
                ax.plot(
                    g["call"],
                    g[k],
                    color=c,
                    lw=1.4,
                    label=f"{color_key[0]}={v:g}",
                )
            ax.set_title(k, fontsize=10)
            ax.grid(True, alpha=0.25)
            ax.set_xlabel("call (≈ optimizer step group)", fontsize=8)
        axes[0][0].legend(fontsize=7, frameon=False, ncol=2)
        fig.suptitle(
            f"{title} — per-component reward (mean over generations per call, "
            f"5-call rolling)",
            y=1.0,
            fontsize=12,
        )
        fig.tight_layout()
        return fig

    _plot_components(trace_summary, "lambda")
    return (_plot_components,)


@app.cell
def _(_plot_components, trace_summary):
    _plot_components(trace_summary, "ratio")
    return


@app.cell
def _(COMPONENT_KEYS, final_components, np, pd, plt):
    def _final_component_bars(df):
        if df.empty:
            return None
        # One panel per component; x = (λ, ρ) labels grouped by sweep kind.
        df = df.copy()
        df["cell"] = df.apply(
            lambda r: f"λ={r['lambda']:g}\nρ={r['ratio']:g}", axis=1
        )
        # Order: λ axis (sorted by λ at ρ=0), ρ axis (sorted by ρ at λ=1.0),
        # off-axis (sorted by λ then ρ). Skip cells outside these slices.
        is_lam = (df["ratio"] == 0.0) & (df["lambda"] != 1.0)
        is_rho = (df["lambda"] == 1.0)
        is_off = ~(is_lam | is_rho)
        ordered = pd.concat(
            [
                df[is_lam].sort_values("lambda"),
                df[is_rho].sort_values("ratio"),
                df[is_off].sort_values(["lambda", "ratio"]),
            ]
        ).reset_index(drop=True)
        fig, axes = plt.subplots(2, 3, figsize=(18, 8), sharex=True)
        bar_colors = ["#3a6da9"] * is_lam.sum() + ["#7c5295"] * is_rho.sum() + [
            "#c44e52"
        ] * is_off.sum()
        x = np.arange(len(ordered))
        for ax, k in zip(axes.flat, COMPONENT_KEYS):
            ax.bar(x, ordered[k].values, color=bar_colors, edgecolor="black", lw=0.4)
            ax.set_title(k, fontsize=10)
            ax.grid(True, alpha=0.25, axis="y")
            ax.set_xticks(x)
            ax.set_xticklabels(ordered["cell"], fontsize=7, rotation=0)
        # legend
        from matplotlib.patches import Patch

        handles = [
            Patch(facecolor="#3a6da9", label="λ-axis (ρ=0)"),
            Patch(facecolor="#7c5295", label="ρ-axis (λ=1.0)"),
            Patch(facecolor="#c44e52", label="off-axis corners"),
        ]
        axes[0][0].legend(handles=handles, fontsize=8, frameon=False, loc="upper left")
        fig.suptitle(
            "Final tail-window (last 8 calls) component means per cell",
            y=1.0,
            fontsize=12,
        )
        fig.tight_layout()
        return fig

    _final_component_bars(final_components)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9 · Save consolidated tables

    Persist the canonical run-level table, the all-runs table (incl. corrupted
    05-13 cells, for repro plots), the stacked W&B history, the offline trace
    summary, and the final-call component means. The downstream eval notebooks /
    paper tables read from these CSV / parquet files instead of re-fetching W&B.
    """)
    return


@app.cell
def _(
    TABLES_DIR,
    canonical_runs,
    final_components,
    hist_all,
    runs_df,
    trace_summary,
):
    def _save():
        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        written = []
        if not runs_df.empty:
            p = TABLES_DIR / "all_sweeps_grpo_2026_05_runs_all.csv"
            runs_df.assign(sweep=runs_df["sweep"].astype(str)).to_csv(p, index=False)
            written.append(p.name)
        if not canonical_runs.empty:
            p = TABLES_DIR / "all_sweeps_grpo_2026_05_runs_canonical.csv"
            canonical_runs.assign(sweep=canonical_runs["sweep"].astype(str)).to_csv(
                p, index=False
            )
            written.append(p.name)
        if not hist_all.empty:
            p = TABLES_DIR / "all_sweeps_grpo_2026_05_history.parquet"
            hist_all.to_parquet(p)
            written.append(p.name)
        if not trace_summary.empty:
            p = TABLES_DIR / "all_sweeps_grpo_2026_05_trace_summary.parquet"
            trace_summary.to_parquet(p)
            written.append(p.name)
        if not final_components.empty:
            p = TABLES_DIR / "all_sweeps_grpo_2026_05_final_components.csv"
            final_components.assign(
                sweep=final_components["sweep"].astype(str)
            ).to_csv(p, index=False)
            written.append(p.name)
        return written

    print("wrote:", _save())
    return


if __name__ == "__main__":
    app.run()
