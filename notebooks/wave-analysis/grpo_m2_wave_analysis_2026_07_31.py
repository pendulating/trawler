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
    # GRPO m2 wave A — post-mortem: a clean negative (2026-07-31)

    Analysis of the four-cell wave-A grid (`sweep/grpo_m2_grid.yaml`,
    launched 2026-07-28 21:31), the m1 re-run on the fully repaired reward
    stack (SFT-aligned prompts, chunk-denominator R-DIRECT at k=1/τ=0.55,
    re-anchored batteries, gates with teeth — see the m1 post-mortem and the
    2026-07-28 pre-launch audit). All four cells COMPLETED 450/450 on
    Qwen3.5-9B sft-canonical.

    | cell | job | elapsed | verdict |
    |---|---|---|---|
    | `core` | 565752 | 31h14m | **HOLD** |
    | `full` | 565751 | 52h28m | **HOLD** |
    | `−outcome` | 565753 | 51h53m | **HOLD** |
    | `−vignette` | 567011 | 57h27m | **HOLD** |

    **Headline: no cell promotes, and this time that is a *finding*, not an
    infrastructure failure.** The reward provably measures normative
    discrimination (validated end-to-end pre-launch; §3 verifies it held in
    production), format noise is gone, the gold is clean, within-group
    gradient exists — and GRPO at the keeper optimizer preset still does not
    lift minority-class discrimination off the blanket floor in 450 steps.
    Consequences: **wave B (−ground/−contrast) is NO-GO** per the
    pre-registration (`full` does not beat `core`), and **the keeper
    v9-ckpt100 remains the camera-ready model**, with m2 as the honest
    ablation/limitations evidence.

    ### Sources of truth — all disk, no W&B

    `promotion_gates.json`, `training_metadata.json` (merged, principle-6
    trail intact this wave), `checkpoint-450/trainer_state.json`, and
    `reward_traces.jsonl` per cell. The R5 trace schema carries per-flow
    gold/pred (`direct_flows`) and per-battery forensics (`vig_result`,
    `model_forces`), so every number here is recomputable offline.

    ### Read-before-quoting caveats

    1. **NOT comparable to m1** in levels: core term (chunk-fixed macro,
       prices recall), battery scale (re-anchored), and gates all changed.
    2. **Small-window checkpoint reads are noise.** The live ⅓/⅔ reads
       flip-flopped ("rising" → "faded") exactly as m1's did; §4 shows the
       windowed illusion against uniform 75-call bins. Quote the bins.
    3. `kl_bounded` gate failures in two cells are **single-step transient
       artifacts** (§5), not drift — the gate means; the medians are ~0.02.
    4. `−vignette`'s canonical run dir is `22-00-49`; the `21-31-11` dir is
       the port-collision false start (EADDRINUSE at engine init, 0 steps).
    """)
    return


@app.cell
def _():
    import json
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    M2_ROOT = Path("/share/pierson/matt/UAIR/multirun")
    M2_CELLS = {
        "core": "2026-07-28_grpo_m2_core/21-31-11/cell=core",
        "full": "2026-07-28_grpo_m2_full/21-31-11/cell=full",
        "-outcome": "2026-07-28_grpo_m2_-outcome/21-31-11/cell=minus_outcome",
        # 21-31-11 for -vignette is the port-collision false start (0 steps).
        "-vignette": "2026-07-28_grpo_m2_-vignette/22-00-49/cell=minus_vignette",
    }
    M2_CKPT = {
        c: M2_ROOT / p / "grpo_only_online_external/outputs/grpo/checkpoint"
        for c, p in M2_CELLS.items()
    }
    M2_CACHE = Path(__file__).parent / "m2_wave_cache"
    M2_CACHE.mkdir(exist_ok=True)
    CELL_ORDER = ["core", "full", "-outcome", "-vignette"]
    CELL_COLOR = {
        "core": "#1f77b4", "full": "#d62728",
        "-outcome": "#ff7f0e", "-vignette": "#2ca02c",
    }
    # Measured references (reward-direct-spec.md; pre-launch audit)
    BASELINE_INAPPR = 0.102   # SFT zero-shot minority accuracy
    HEDGE_FLOOR = 0.043       # perfect-policy hedge_frac (4.3% permitted golds)
    BLANKET_ANTITHESIS = 0.234  # blanket-obligatory signature on batteries
    return (
        BASELINE_INAPPR, BLANKET_ANTITHESIS, CELL_COLOR, CELL_ORDER,
        HEDGE_FLOOR, M2_CACHE, M2_CKPT, json, np, pd, plt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · Promotion gates — the teeth worked

    Every m1 cell was promoted by the old gates; every m2 cell is HELD by
    the new ones, for stated reasons. The one gate firing for the wrong
    reason is `kl_bounded`: it means over the logged KL, and a single-step
    transient (§5) poisons the mean. The medians tell the true story.
    """)
    return


@app.cell
def _(M2_CKPT, json, pd):
    import statistics as _st

    def _gate_row(cell, path):
        g = json.load(open(path / "promotion_gates.json"))
        state = json.load(open(path / "checkpoint-450/trainer_state.json"))
        kl = [e["kl"] for e in state["log_history"] if "kl" in e]
        gt = g["gates"]
        dd = gt.get("direct_discrimination", {})
        rec = dd.get("recalls") or {}
        return {
            "cell": cell,
            "promote": g["promote"],
            "trend_gain": gt["reward_trend"].get("gain"),
            "trend": gt["reward_trend"]["status"],
            "kl_mean": gt["kl_bounded"].get("mean_kl"),
            "kl_median": round(_st.median(kl), 3) if kl else None,
            "kl_gate": gt["kl_bounded"]["status"],
            "label_J": dd.get("youden_j"),
            "inappr_recall": rec.get("inappropriate"),
            "appr_recall": rec.get("appropriate"),
            "miss_frac": dd.get("miss_frac"),
            "J_gate": dd.get("status"),
            "no_flow": gt["no_flow_rate"]["status"],
            "zero_std": gt["zero_std_groups"]["status"],
        }

    gates_df = pd.DataFrame(
        [_gate_row(c, p) for c, p in M2_CKPT.items()]
    ).set_index("cell")
    gates_df
    return (gates_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Readings:

    * **`reward_trend`**: gains −0.003…+0.006 against the 0.02 bar — flat,
      honestly reported. (Gate (a) is intra-cell only; module weights
      differ by cell.)
    * **`direct_discrimination`**: final-tail label-J 0.002 (`core`) /
      0.011 (`full`) / −0.025 (`−vignette`) — the blanket floor. Minority
      recall ends ≈0.14 vs the 0.102 zero-shot baseline: real but small,
      and the mid-run bump above it decayed (§4).
    * **`kl_bounded`**: fails for `core` (mean 23.2) and `−outcome`
      (mean 273) — both are single-step transients over medians of ~0.02
      (§5). Post-wave gate fix: median-robust KL (registered TODO).
    * `no_flow_rate` (live for the first time on a modular run) and
      `zero_std` pass everywhere — no abstention collapse, healthy spread.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · Trace loading (cached)

    Three parquet caches per cell: trace rows, per-flow `direct_flows`
    records (the discrimination substrate), and per-battery `vig_result`.
    """)
    return


@app.cell
def _(M2_CACHE, M2_CKPT, json, pd):
    def _parse(cell, path):
        rc = M2_CACHE / f"rows_{cell.replace('-', 'm_')}.parquet"
        fc = M2_CACHE / f"flows_{cell.replace('-', 'm_')}.parquet"
        vc = M2_CACHE / f"vig_{cell.replace('-', 'm_')}.parquet"
        if rc.exists() and fc.exists() and vc.exists():
            return (pd.read_parquet(rc), pd.read_parquet(fc),
                    pd.read_parquet(vc))
        rows, flows, vigs = [], [], []
        for line in open(path / "reward_traces.jsonl"):
            o = json.loads(line)
            rows.append({
                "cell": cell, "call": o["call"], "route": o["route"],
                "score": o["score"], "task_type": o.get("task_type"),
                "n_flows": o.get("n_flows"),
                "spurious": o.get("direct_spurious"),
                "gate_reason": o.get("gate_reason"),
            })
            for fl in o.get("direct_flows") or []:
                flows.append({
                    "cell": cell, "call": o["call"], "gold": fl["gold"],
                    "matched": fl["pred"] is not None,
                    "correct": fl["pred"] == fl["gold"],
                    "sim": fl.get("sim"),
                })
            vr = o.get("vig_result")
            if vr:
                vigs.append({"cell": cell, "call": o["call"], **{
                    k: float(v) for k, v in vr.items()}})
        dfs = (pd.DataFrame(rows), pd.DataFrame(flows), pd.DataFrame(vigs))
        for df, c in zip(dfs, (rc, fc, vc)):
            df.to_parquet(c)
        return dfs

    _parts = [_parse(c, p) for c, p in M2_CKPT.items()]
    rows_df = pd.concat([p[0] for p in _parts], ignore_index=True)
    flows_df = pd.concat([p[1] for p in _parts if len(p[1])],
                         ignore_index=True)
    vig_df = pd.concat([p[2] for p in _parts if len(p[2])],
                       ignore_index=True)
    (
        rows_df.groupby(["cell", "route"])
        .agg(n=("score", "size"), mean_score=("score", "mean")).round(3)
    )
    return flows_df, rows_df, vig_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · The repaired stack held in production

    The m1 defects, re-measured on the m2 run itself — every fix survived
    contact with 450 live steps:
    """)
    return


@app.cell
def _(flows_df, pd, rows_df):
    def _stack_row(cell):
        r = rows_df[rows_df["cell"] == cell]
        ext = r[r["task_type"] == "extract"]
        sc = r[r["route"] == "scored"]
        f = flows_df[flows_df["cell"] == cell]
        out = {
            "gate_fail (m1: 24.5%)": f"{(ext['route'] == 'gate_fail').mean():.1%}",
            "n_flows/comp first→last": (
                f"{sc[sc['call'] < 75]['n_flows'].mean():.1f}→"
                f"{sc[sc['call'] >= 375]['n_flows'].mean():.1f}"),
            "spurious_rate": (
                f"{sc['spurious'].sum() / sc['n_flows'].sum():.2f}"
                if sc["n_flows"].sum() else "—"),
            "miss_frac": f"{(~f['matched']).mean():.2f}" if len(f) else "—",
            "flow judgments on disk": len(f),
        }
        return out

    pd.DataFrame({c: _stack_row(c)
                  for c in ["core", "full", "-outcome", "-vignette"]}).T
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    * Gate failure ~3% all run (m1: 24.5% and flat) — **R1 held**: format
      is out of the advantage.
    * Flow counts 5.5→5.7, spurious steady 0.22 — the **flow-inflation
      kill criterion never fired** (the audit's accepted-unpriced channel
      stayed quiet; τ's requirement that a match approximate a real
      extraction was apparently guard enough at these settings).
    * Hundreds of thousands of per-flow judgments recoverable from disk —
      **R5 held**: no W&B dependency anywhere in this notebook.

    ## 4 · Discrimination — the result, and the windowing illusion

    The substrate: per-flow label accuracy on MATCHED flows (label-J
    semantics, m1-comparable in kind). Left: uniform 75-call bins — the
    honest trajectory. Right: the same data read the way the live
    checkpoint monitors read it (small trailing windows), reproducing the
    "⅓ rising / ⅔ faded" illusion. The m1 post-mortem warned that
    checkpoint reads flip verdicts on noise; m2's monitoring repeated the
    mistake with new metrics. Uniform bins or nothing.
    """)
    return


@app.cell
def _(BASELINE_INAPPR, CELL_COLOR, flows_df, np, plt):
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(11.5, 4), sharey=True)
    for _c, _g in flows_df.groupby("cell"):
        _m = _g[_g["matched"]]
        _bins = (_m["call"] // 75) * 75
        _tr = _m[_m["gold"] == "inappropriate"].groupby(
            (_m[_m["gold"] == "inappropriate"]["call"] // 75) * 75
        )["correct"].mean()
        _ax1.plot(_tr.index + 37, _tr.values, marker="o", lw=1.4,
                  label=_c, color=CELL_COLOR[_c])
        # windowed illusion: trailing 30-call window at each "checkpoint"
        _xs, _ys = [], []
        for _ck in range(30, 451, 30):
            _w = _m[(_m["call"] >= _ck - 30) & (_m["call"] < _ck)
                    & (_m["gold"] == "inappropriate")]
            if len(_w) > 30:
                _xs.append(_ck)
                _ys.append(_w["correct"].mean())
        _ax2.plot(_xs, _ys, marker=".", lw=1.0, label=_c,
                  color=CELL_COLOR[_c], alpha=0.8)
    for _ax, _t in ((_ax1, "uniform 75-call bins (quote these)"),
                    (_ax2, "trailing 30-call windows (the live-read illusion)")):
        _ax.axhline(BASELINE_INAPPR, color="k", ls="--", lw=0.8,
                    label="SFT zero-shot baseline (0.102)" if _ax is _ax1 else None)
        _ax.set_title(_t, fontsize=10)
        _ax.set_xlabel("call")
        _ax.grid(alpha=0.3)
    _ax1.set_ylabel("minority-class (inappropriate) label accuracy")
    _ax1.legend(fontsize=8)
    _fig.suptitle("Minority-class accuracy: flat-with-wiggle around 0.10–0.14; "
                  "the windowed view manufactures trends", y=1.02)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(flows_df, pd):
    def _binned(cell):
        f = flows_df[(flows_df["cell"] == cell) & flows_df["matched"]]
        out = {}
        for lo in range(0, 451, 75):
            w = f[(f["call"] >= lo) & (f["call"] < lo + 75)]
            if not len(w):
                continue
            acc = w.groupby("gold")["correct"].mean()
            j = acc.get("appropriate", float("nan")) + acc.get(
                "inappropriate", float("nan")) - 1
            out[f"{lo}-{lo+74}"] = round(j, 3)
        return out

    disc_df = pd.DataFrame({c: _binned(c)
                            for c in ["core", "full", "-vignette"]}).T
    disc_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Label-J per uniform bin never leaves **[−0.04, +0.01]** in any cell at
    any point. There was no rise-then-fade; there was noise around the
    blanket floor with a final level of inappr ≈ 0.14 vs the 0.102
    baseline — a ~4-point residual that the trend, the J-gate, and the
    bins all agree is not sustained learning.

    ## 5 · The single-step KL transients

    Two cells detonated for exactly one logged step and recovered to
    ~0.02 by the next log: `core` at step 50 (KL 1038) and `−outcome` at
    step 150 (KL 12,293). Entropy and completion lengths were stable
    through both; no clipping; rewards unperturbed. These are one-batch
    events (plausibly a single pathological rollout group under
    `token_truncate` importance sampling), not drift — but each poisons
    its cell's `kl_bounded` MEAN for the whole run.
    """)
    return


@app.cell
def _(CELL_COLOR, CELL_ORDER, M2_CKPT, json, np, plt):
    _fig, _ax = plt.subplots(figsize=(9, 3.4))
    for _c in CELL_ORDER:
        _s = json.load(open(M2_CKPT[_c] / "checkpoint-450/trainer_state.json"))
        _kl = [(e["step"], max(e["kl"], 1e-3))
               for e in _s["log_history"] if "kl" in e]
        _ax.plot([x for x, _ in _kl], [y for _, y in _kl], marker=".",
                 lw=1.0, label=_c, color=CELL_COLOR[_c])
    _ax.set_yscale("log")
    _ax.axhline(1.0, color="k", ls="--", lw=0.8, label="kl_bounded threshold (on the MEAN)")
    _ax.set_xlabel("step")
    _ax.set_ylabel("KL (log scale)")
    _ax.set_title("KL to SFT reference — two single-step transients, instant recovery", fontsize=10)
    _ax.legend(fontsize=8)
    _ax.grid(alpha=0.3)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Registered fix (post-wave):** make `kl_bounded` median-robust (or
    trimmed-mean), keeping `max_logged_kl` as a reported diagnostic. Until
    then, adjudicate KL fails from this curve. (Worth a look someday:
    what the two pathological batches actually contained — the traces
    hold the completions' scores but not their tokens.)

    ## 6 · The battery term under the re-anchored scale

    The re-anchor guaranteed the term *can* separate policies (hedge
    0.611→0.190 for degenerate strategies, measured pre-launch); whether
    the policy would *move* was explicitly not guaranteed. It did not:
    """)
    return


@app.cell
def _(BLANKET_ANTITHESIS, CELL_COLOR, HEDGE_FLOOR, plt, vig_df):
    _fig, _axes = plt.subplots(1, 3, figsize=(12, 3.4), sharex=True)
    for _c, _g in vig_df.groupby("cell"):
        _b = (_g["call"] // 75) * 75
        for _ax, _k in zip(_axes, ("battery", "hedge_frac", "antithesis_frac")):
            _tr = _g.groupby(_b)[_k].mean()
            _ax.plot(_tr.index + 37, _tr.values, marker="o", lw=1.4,
                     label=_c, color=CELL_COLOR[_c])
    _axes[1].axhline(HEDGE_FLOOR, color="k", ls="--", lw=0.8,
                     label="perfect-policy floor 0.043")
    _axes[2].axhline(BLANKET_ANTITHESIS, color="k", ls=":", lw=0.8,
                     label="blanket-obligatory signature 0.234")
    for _ax, _t in zip(_axes, ("battery mean (re-anchored scale)",
                               "hedge_frac", "antithesis_frac")):
        _ax.set_title(_t, fontsize=10)
        _ax.set_xlabel("call")
        _ax.grid(alpha=0.3)
        _ax.legend(fontsize=7)
    _fig.suptitle("Battery term: flat ~0.58–0.60; hedge flat ~0.22 (did NOT fall); "
                  "antithesis ~0.11 — no blanket collapse either", y=1.05)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Battery ≈ 0.58–0.60 and hedge ≈ 0.22 throughout, in every cell that has
    the task. Two honest conclusions: (a) the audit's causal caveat was
    right — m1's hedge drift was not battery-reward-driven, since fixing
    the hedge economics changed nothing; (b) at 18% of rows the battery
    term again contributed no visible learning, so the `−vignette` LOO
    delta is (again) noise — but this time because the *policy* didn't
    move, not because the scorer couldn't see movement.

    ## 7 · Synthesis

    **The result.** With every m1 defect repaired and verified in
    production, GRPO at the keeper optimizer preset (lr 2e-5, β=0.02, G=8,
    450 steps, 600 prompts × 3 epochs) does not move minority-class
    normative discrimination off the blanket floor for this SFT policy:
    final label-J ≤ 0.011 everywhere, minority recall 0.14 vs 0.102
    baseline, battery term flat. All four cells HOLD; `full` does not beat
    `core`; **wave B is NO-GO**; **keeper v9-ckpt100 stays** as the paper's
    model, and m2 is the limitations/ablation evidence.

    **Why this is a finding and not a failure.** m1's flatness was
    uninterpretable (the reward was measuring format noise against
    collapsed gold). m2's flatness survives every audit the reward stack
    was subjected to: the policy was scored on its training distribution,
    against clean chunk-fixed gold, with live gradient (dead groups ~4%),
    and per-flow evidence on disk. The paper can now say precisely what
    was tried and what the signal did.

    **Candidate explanations for the null (unordered, untested):**
    the minority class is thin *per group* (median one inappropriate flow
    per mixed chunk → the advantage for catching it is diluted across a
    macro that the majority class and the recall channel dominate);
    entropy ~0.4 at temperature 1.0 may not explore label flips on
    specific flows; 450 steps × 600 prompts may be an order short for a
    discrimination signal this sparse; or the SFT policy's zero-shot
    normative grounding (the paper's own motivating claim) may simply not
    be RL-recoverable without process supervision — which would make the
    negative the strongest version of the paper's argument.

    **Registered follow-ups:**
    1. `kl_bounded` → median-robust (gates.py), keep max as diagnostic.
    2. Monitoring protocol: uniform-bin reads only; retire trailing-window
       checkpoint verdicts (this bit us in both waves).
    3. If any future wave is attempted: attack the per-group minority
       dilution directly (e.g. minority-flow advantage weighting or
       mixed-chunk-only curricula) before touching optimizer knobs.
    """)
    return


if __name__ == "__main__":
    app.run()
