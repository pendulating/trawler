import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # SFT pair ablation — Qwen3.5-9B base vs instruct

    Sweep over four boolean flags that gate inclusion of CI metadata in the SFT
    target string:

    - `ctx`  → `flow_context`
    - `appr` → `flow_appropriateness`
    - `norms` → `flow_norms_meta`
    - `conf` → `flow_confidence`

    16 ablations × 2 base models = 32 runs.
    Source: `multirun/2026-04-25_sft_pair_ablation/{21-17-40,21-20-27}`.

    Per-run metrics come from each ablation's submitit stdout
    (`.slurm_jobs/sft_training/<jobid>_0_log.out`); the shared
    `sft_traces.jsonl` is overwritten by every run and is not used here.
    """)
    return


@app.cell
def _():
    import ast
    import re
    from pathlib import Path

    import numpy as np
    import pandas as pd

    pd.set_option("display.max_columns", 60)
    pd.set_option("display.width", 200)
    return Path, ast, np, pd, re


@app.cell
def _(Path):
    SWEEP_ROOT = Path(
        "/share/pierson/matt/UAIR/multirun/2026-04-25_sft_pair_ablation"
    )

    SWEEPS = {
        "instruct": SWEEP_ROOT / "21-17-40",
        "base": SWEEP_ROOT / "21-20-27",
    }
    return (SWEEPS,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Parsing
    """)
    return


@app.cell
def _(ast, re):
    _STEP_RE = re.compile(r"^\{'(?:loss|train_runtime)'.*\}$")

    def parse_metric_line(line: str):
        """Parse a TRL/Trainer metric dict line.

        The trainer prints lines like:
            {'loss': '0.7502', 'grad_norm': '0.1413', ..., 'epoch': '0.2073'}
        Values are stringified floats. Returns dict[str, float] or None.
        """
        line = line.strip()
        if not _STEP_RE.match(line):
            return None
        try:
            raw = ast.literal_eval(line)
        except (ValueError, SyntaxError):
            return None
        out = {}
        for k, v in raw.items():
            try:
                out[k] = float(v)
            except (TypeError, ValueError):
                out[k] = v
        return out

    return (parse_metric_line,)


@app.function
def parse_ablation_name(name: str) -> dict:
    """Parse 'ctx-False_appr-True_norms-False_conf-True' → flag dict."""
    flags = {}
    for part in name.split("_"):
        key, _, val = part.partition("-")
        flags[key] = (val == "True")
    return flags


@app.cell
def _(Path, parse_metric_line):
    def collect_run(model: str, ablation_dir: Path) -> dict | None:
        """Read training metrics for a single ablation directory.

        Returns dict with metadata + 'steps' list of per-log-step dicts,
        or None if no slurm stdout was found.
        """
        slurm_dir = ablation_dir / ".slurm_jobs" / "sft_training"
        if not slurm_dir.is_dir():
            return None
        outs = sorted(slurm_dir.glob("*_0_log.out"))
        if not outs:
            return None
        log_path = outs[-1]

        flags = parse_ablation_name(ablation_dir.name)
        steps = []
        final = None
        with log_path.open("r", errors="replace") as f:
            for line in f:
                rec = parse_metric_line(line)
                if rec is None:
                    continue
                if "train_runtime" in rec:
                    final = rec
                else:
                    steps.append(rec)

        return {
            "model": model,
            "ablation": ablation_dir.name,
            "log_path": str(log_path),
            "job_id": log_path.name.split("_")[0],
            **flags,
            "steps": steps,
            "final": final,
        }

    return (collect_run,)


@app.cell
def _(SWEEPS, collect_run):
    runs = []
    for _model, _root in SWEEPS.items():
        for _ablation_dir in sorted(_root.glob("ctx-*")):
            if not _ablation_dir.is_dir():
                continue
            r = collect_run(_model, _ablation_dir)
            if r is not None:
                runs.append(r)

    {
        "n_runs": len(runs),
        "by_model": {m: sum(1 for r in runs if r["model"] == m) for m in {r["model"] for r in runs}},
        "missing_final": [r["ablation"] for r in runs if r["final"] is None][:5],
        "n_steps_distribution": sorted({len(r["steps"]) for r in runs}),
    }
    return (runs,)


@app.cell
def _(pd, runs):
    _step_rows = []
    for _r in runs:
        for _idx, _s in enumerate(_r["steps"]):
            _step_rows.append({
                "model": _r["model"],
                "ablation": _r["ablation"],
                "ctx": _r["ctx"],
                "appr": _r["appr"],
                "norms": _r["norms"],
                "conf": _r["conf"],
                "log_idx": _idx,
                **_s,
            })
    steps_df = pd.DataFrame(_step_rows)
    steps_df["n_flags_on"] = (
        steps_df[["ctx", "appr", "norms", "conf"]].astype(int).sum(axis=1)
    )
    steps_df.head()
    return (steps_df,)


@app.cell
def _(pd, runs):
    _final_rows = []
    for _r in runs:
        _final = _r["final"] or {}
        _last = _r["steps"][-1] if _r["steps"] else {}
        _final_rows.append({
            "model": _r["model"],
            "ablation": _r["ablation"],
            "ctx": _r["ctx"],
            "appr": _r["appr"],
            "norms": _r["norms"],
            "conf": _r["conf"],
            "n_flags_on": int(_r["ctx"]) + int(_r["appr"]) + int(_r["norms"]) + int(_r["conf"]),
            "job_id": _r["job_id"],
            "n_log_steps": len(_r["steps"]),
            "last_step_loss": _last.get("loss"),
            "last_step_grad_norm": _last.get("grad_norm"),
            "last_step_entropy": _last.get("entropy"),
            "last_step_token_acc": _last.get("mean_token_accuracy"),
            "train_loss": _final.get("train_loss"),
            "train_runtime_s": _final.get("train_runtime"),
            "train_samples_per_s": _final.get("train_samples_per_second"),
            "final_entropy": _final.get("entropy"),
            "final_token_acc": _final.get("mean_token_accuracy"),
            "final_num_tokens": _final.get("num_tokens"),
        })
    final_df = pd.DataFrame(_final_rows).sort_values(["model", "ablation"]).reset_index(drop=True)
    final_df
    return (final_df,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Loss curves
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import sys

    sys.path.insert(0, "/share/pierson/matt/UAIR/notebooks")
    try:
        from font_utils import load_ibm_plex_sans

        load_ibm_plex_sans()
    except Exception as _exc:
        print(f"[font] using matplotlib default ({_exc})")

    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.25
    return (plt,)


@app.cell
def _(plt, steps_df):
    def _plot_loss_curves(df, metric="loss", title=None):
        models = sorted(df["model"].unique())
        fig, axes = plt.subplots(
            1, len(models), figsize=(7.5 * len(models), 5.0), sharey=True
        )
        if len(models) == 1:
            axes = [axes]
        cmap = plt.get_cmap("viridis")
        for ax, m in zip(axes, models):
            sub = df[df["model"] == m]
            abls = sorted(sub["ablation"].unique())
            for i, abl in enumerate(abls):
                run = sub[sub["ablation"] == abl].sort_values("epoch")
                ax.plot(
                    run["epoch"],
                    run[metric],
                    color=cmap(i / max(len(abls) - 1, 1)),
                    alpha=0.85,
                    linewidth=1.4,
                    label=abl,
                )
            ax.set_title(f"{m}  ({metric})", fontsize=12)
            ax.set_xlabel("epoch")
            ax.set_ylabel(metric)
        axes[-1].legend(
            bbox_to_anchor=(1.02, 1), loc="upper left",
            fontsize=7, frameon=False, ncol=1,
        )
        if title:
            fig.suptitle(title, y=1.02, fontsize=13)
        fig.tight_layout()
        return fig

    _plot_loss_curves(steps_df, metric="loss", title="Per-step training loss by ablation")
    return


@app.cell
def _(plt, steps_df):
    def _plot_by_flag(df, flag, metric="loss"):
        models = sorted(df["model"].unique())
        fig, axes = plt.subplots(
            1, len(models), figsize=(7.0 * len(models), 4.5), sharey=True
        )
        if len(models) == 1:
            axes = [axes]
        for ax, m in zip(axes, models):
            sub = df[df["model"] == m]
            for val, color in [(False, "#3b82f6"), (True, "#ef4444")]:
                run_grp = sub[sub[flag] == val]
                if run_grp.empty:
                    continue
                grouped = (
                    run_grp.groupby("epoch")[metric]
                    .agg(["mean", "min", "max"])
                    .reset_index()
                )
                ax.fill_between(
                    grouped["epoch"], grouped["min"], grouped["max"],
                    color=color, alpha=0.12,
                )
                ax.plot(
                    grouped["epoch"], grouped["mean"],
                    color=color, linewidth=2.0, label=f"{flag}={val}",
                )
            ax.set_title(f"{m}  —  marginal effect of {flag}", fontsize=11)
            ax.set_xlabel("epoch")
            ax.set_ylabel(metric)
            ax.legend(loc="upper right", fontsize=9, frameon=False)
        fig.tight_layout()
        return fig

    _plot_by_flag(steps_df, "ctx", "loss")
    return


@app.cell
def _(plt, steps_df):
    def _plot_marginal_grid(df, metric="loss"):
        flags = ["ctx", "appr", "norms", "conf"]
        models = sorted(df["model"].unique())
        fig, axes = plt.subplots(
            len(models), len(flags),
            figsize=(3.6 * len(flags), 3.0 * len(models)),
            sharey=True, sharex=True,
        )
        if len(models) == 1:
            axes = axes[None, :]
        for r, m in enumerate(models):
            for c, fl in enumerate(flags):
                ax = axes[r, c]
                sub = df[df["model"] == m]
                for val, color in [(False, "#3b82f6"), (True, "#ef4444")]:
                    grp = (
                        sub[sub[fl] == val]
                        .groupby("epoch")[metric].mean()
                        .reset_index()
                    )
                    if grp.empty:
                        continue
                    ax.plot(
                        grp["epoch"], grp[metric],
                        color=color, linewidth=1.6,
                        label=f"{fl}={val}",
                    )
                ax.set_title(f"{m} | {fl}", fontsize=9)
                if r == len(models) - 1:
                    ax.set_xlabel("epoch")
                if c == 0:
                    ax.set_ylabel(metric)
                ax.legend(fontsize=7, frameon=False, loc="upper right")
        fig.tight_layout()
        return fig

    _plot_marginal_grid(steps_df, metric="loss")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Final-loss summary
    """)
    return


@app.cell
def _(final_df):
    final_df.pivot_table(
        index=["ctx", "appr", "norms", "conf"],
        columns="model",
        values="train_loss",
        aggfunc="first",
    ).round(4)
    return


@app.cell
def _(final_df, np, pd):
    rows = []
    for fl in ["ctx", "appr", "norms", "conf"]:
        for m, sub in final_df.groupby("model"):
            on = sub.loc[sub[fl], "train_loss"].astype(float).to_numpy()
            off = sub.loc[~sub[fl], "train_loss"].astype(float).to_numpy()
            rows.append({
                "flag": fl,
                "model": m,
                "n_on": len(on),
                "n_off": len(off),
                "mean_on": np.nanmean(on) if len(on) else np.nan,
                "mean_off": np.nanmean(off) if len(off) else np.nan,
                "delta_on_minus_off": (np.nanmean(on) - np.nanmean(off)) if len(on) and len(off) else np.nan,
            })
    marginal_effects = (
        pd.DataFrame(rows)
        .pivot_table(index="flag", columns="model", values="delta_on_minus_off")
        .round(5)
    )
    marginal_effects
    return


@app.cell
def _(mo):
    mo.md(r"""
    Each cell is `mean(train_loss | flag=True) - mean(train_loss | flag=False)`,
    averaged over the 8 ablations where that flag is on vs off.
    Negative ⇒ enabling the flag lowers final loss.
    """)
    return


@app.cell
def _(plt, steps_df):
    def _plot_loss_by_flag_count(df, metric="loss"):
        models = sorted(df["model"].unique())
        fig, axes = plt.subplots(
            1, len(models), figsize=(7.0 * len(models), 4.5), sharey=True
        )
        if len(models) == 1:
            axes = [axes]
        cmap = plt.get_cmap("plasma")
        for ax, m in zip(axes, models):
            sub = df[df["model"] == m]
            for k in sorted(sub["n_flags_on"].unique()):
                grp = (
                    sub[sub["n_flags_on"] == k]
                    .groupby("epoch")[metric].mean()
                    .reset_index()
                )
                ax.plot(
                    grp["epoch"], grp[metric],
                    color=cmap(k / 4.0), linewidth=2.0,
                    label=f"{k} flags on",
                )
            ax.set_title(f"{m}  —  loss vs # of flags enabled", fontsize=11)
            ax.set_xlabel("epoch")
            ax.set_ylabel(metric)
            ax.legend(fontsize=8, frameon=False, loc="upper right")
        fig.tight_layout()
        return fig

    _plot_loss_by_flag_count(steps_df, "loss")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Base vs instruct: paired deltas
    """)
    return


@app.cell
def _(final_df):
    paired = (
        final_df.pivot_table(
            index="ablation", columns="model", values="train_loss", aggfunc="first"
        )
        .dropna(how="any")
    )
    paired["instruct_minus_base"] = paired["instruct"] - paired["base"]
    paired = paired.round(5).sort_values("instruct_minus_base")
    paired
    return (paired,)


@app.cell
def _(paired, plt):
    _fig, _ax = plt.subplots(figsize=(8, 6))
    _y = range(len(paired))
    _ax.barh(
        list(_y),
        paired["instruct_minus_base"].to_numpy(),
        color=[
            "#ef4444" if v > 0 else "#3b82f6"
            for v in paired["instruct_minus_base"]
        ],
    )
    _ax.set_yticks(list(_y))
    _ax.set_yticklabels(paired.index, fontsize=7)
    _ax.set_xlabel("instruct.train_loss − base.train_loss")
    _ax.set_title("Per-ablation paired delta (instruct − base)")
    _ax.axvline(0, color="black", linewidth=0.8)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Per-step trajectory deltas
    """)
    return


@app.cell
def _(steps_df):
    def _trajectory_table(df):
        agg = (
            df.groupby(["model", "ablation", "log_idx"])
            .agg(epoch=("epoch", "mean"), loss=("loss", "mean"))
            .reset_index()
        )
        wide = agg.pivot_table(
            index=["ablation", "log_idx"], columns="model", values="loss"
        ).reset_index()
        if {"base", "instruct"}.issubset(wide.columns):
            wide["delta"] = wide["instruct"] - wide["base"]
        return wide

    traj = _trajectory_table(steps_df)
    by_idx = (
        traj.groupby("log_idx")
        .agg(
            mean_delta=("delta", "mean"),
            min_delta=("delta", "min"),
            max_delta=("delta", "max"),
            n=("delta", "count"),
        )
        .reset_index()
    )
    by_idx
    return (by_idx,)


@app.cell
def _(by_idx, plt):
    _fig, _ax = plt.subplots(figsize=(9, 4.5))
    _ax.fill_between(
        by_idx["log_idx"], by_idx["min_delta"], by_idx["max_delta"],
        color="#94a3b8", alpha=0.3, label="min/max across 16 ablations",
    )
    _ax.plot(
        by_idx["log_idx"], by_idx["mean_delta"],
        color="#0f172a", linewidth=2.0, label="mean delta",
    )
    _ax.axhline(0, color="black", linewidth=0.8)
    _ax.set_xlabel("log step index (1..14)")
    _ax.set_ylabel("instruct.loss − base.loss")
    _ax.set_title("Per-step training loss delta (instruct − base) across ablations")
    _ax.legend(fontsize=9, frameon=False)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Final-loss heatmap
    """)
    return


@app.cell
def _(final_df, plt):
    def _heatmap(df, metric="train_loss"):
        models = sorted(df["model"].unique())
        fig, axes = plt.subplots(
            1, len(models), figsize=(6.5 * len(models), 5.5)
        )
        if len(models) == 1:
            axes = [axes]
        all_vals = df[metric].astype(float)
        vmin, vmax = float(all_vals.min()), float(all_vals.max())
        for ax, m in zip(axes, models):
            sub = df[df["model"] == m].copy()
            sub["row"] = sub.apply(
                lambda r: f"ctx={int(r.ctx)} appr={int(r.appr)}", axis=1
            )
            sub["col"] = sub.apply(
                lambda r: f"norms={int(r.norms)} conf={int(r.conf)}", axis=1
            )
            grid = sub.pivot_table(index="row", columns="col", values=metric)
            im = ax.imshow(
                grid.values, cmap="viridis_r", vmin=vmin, vmax=vmax,
                aspect="auto",
            )
            ax.set_xticks(range(len(grid.columns)))
            ax.set_xticklabels(grid.columns, rotation=30, ha="right", fontsize=8)
            ax.set_yticks(range(len(grid.index)))
            ax.set_yticklabels(grid.index, fontsize=8)
            ax.set_title(f"{m} — {metric}", fontsize=11)
            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    v = grid.values[i, j]
                    if v == v:
                        ax.text(
                            j, i, f"{v:.3f}",
                            ha="center", va="center",
                            color="white" if v > (vmin + vmax) / 2 else "black",
                            fontsize=8,
                        )
            fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        fig.tight_layout()
        return fig

    _heatmap(final_df, "train_loss")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Other final metrics
    """)
    return


@app.cell
def _(final_df):
    final_df.pivot_table(
        index=["ctx", "appr", "norms", "conf"],
        columns="model",
        values=["final_token_acc", "final_entropy", "train_runtime_s"],
    ).round(4)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Notes

    - `train_loss` is the trainer's epoch-3 average over all training steps.
    - `last_step_loss` is the loss at the final logged step (one of 14
      buckets at every ~10% of training).
    - The shared `sft_traces.jsonl` in `sft_only/outputs/sft/checkpoint/`
      is overwritten by every ablation in the sweep, so it cannot be used
      to recover per-ablation curves — use the per-ablation submitit logs.
    - 14 step entries (every ~10% of 147 total steps) plus 1 final entry
      per ablation; missing rows in the table indicate a job that did not
      reach `train_runtime`.
    """)
    return


if __name__ == "__main__":
    app.run()
