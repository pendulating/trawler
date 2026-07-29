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
    # GRPO m1 wave — post-mortem & next-iteration evidence (2026-07-28)

    Analysis of the four-cell m1 ablation grid launched 2026-07-26 00:13
    (`multirun/2026-07-26_grpo_m1_{core,-outcome,-vignette,full}`), all four
    COMPLETED at 450/450 steps on Qwen3.5-9B-SFT (canonical gemma4-teacher
    adapter), G=8, `scale_rewards=none`, β=0.02, R-DIRECT core.

    | cell | job | elapsed | finished |
    |---|---|---|---|
    | `core` | 397536 | 27h48m | 07-27 04:04 |
    | `−outcome` | 397519 | 39h10m | 07-27 15:25 |
    | `full` | 397537 | 39h35m | 07-27 15:52 |
    | `−vignette` | 397518 | 45h21m | 07-27 21:36 |

    **Headline: every promotion gate passed and no cell learned anything.**
    Reward-trend gains are ±0.004 over 450 steps against a threshold of 0.0.
    The optimizer itself is healthy (KL bounded, ~2% dead groups, no
    completion clipping) — the flatness is a property of the *reward*, not
    the trainer. This notebook quantifies where the advantage mass actually
    went, and assembles the evidence for the next iteration's reward design.

    ### Sources of truth

    * `promotion_gates.json`, `training_metadata.json`, and
      `checkpoint-450/trainer_state.json` per cell — trainer-side record,
      survives everything.
    * `reward_traces.jsonl` per cell (14 400 rows = 450 calls × 32
      completions) — per-completion route/score record; the only full-run
      record for `core`, whose **W&B run crashed at ~step 300** (heartbeat
      loss; training was unaffected).
    * W&B `scan_history` for the per-class agreement recalls (J / balanced
      accuracy) — fetched once and cached to `m1_wave_cache/`; traces do not
      carry per-flow gold/pred, so W&B is the only source for discrimination
      (and `core`'s last ~150 steps are therefore unrecoverable — see
      recommendation R5).

    ### Read-before-quoting caveats

    1. The grid predates the `balanced_accuracy`/`youden_j` logging keys —
       both are derived here from the two per-class recalls.
    2. Per-class recalls are conditioned on disjoint gold subsets and do
       **not** sum to 1; any non-discriminating policy sums to exactly 1.0.
    3. `outcome_term` in traces is per-completion macro-EM, which collapses
       to plain accuracy on completions carrying a single gold class
       (§ macro-EM collapse) — it systematically overstates discrimination.
    """)
    return


@app.cell
def _():
    import json
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    M1_ROOT = Path("/share/pierson/matt/UAIR/multirun")
    M1_CELLS = {
        "core": "2026-07-26_grpo_m1_core/00-13-20/cell=core",
        "-outcome": "2026-07-26_grpo_m1_-outcome/00-13-20/cell=minus_outcome",
        "-vignette": "2026-07-26_grpo_m1_-vignette/00-13-20/cell=minus_vignette",
        "full": "2026-07-26_grpo_m1_full/00-13-20/cell=full",
    }
    M1_CKPT = {
        c: M1_ROOT / p / "grpo_only_online_external/outputs/grpo/checkpoint"
        for c, p in M1_CELLS.items()
    }
    M1_CACHE = Path(__file__).parent / "m1_wave_cache"
    M1_CACHE.mkdir(exist_ok=True)
    CELL_ORDER = ["core", "-outcome", "-vignette", "full"]
    CELL_COLOR = {
        "core": "#1f77b4", "-outcome": "#ff7f0e",
        "-vignette": "#2ca02c", "full": "#d62728",
    }
    return (
        CELL_COLOR, CELL_ORDER, M1_CACHE, M1_CKPT, json, np, pd, plt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · Promotion gates

    All four cells promote. Read the `reward_trend` row before treating that
    as success: the gate threshold is `min_reward_gain = 0.0`, so a run that
    ends *exactly where it started* passes. Gains here are two orders of
    magnitude below any meaningful effect (the abstain-table alone moves the
    per-call mean by more between bins).
    """)
    return


@app.cell
def _(M1_CKPT, json, pd):
    def _gate_row(cell, path):
        g = json.load(open(path / "promotion_gates.json"))
        meta = json.load(open(path / "training_metadata.json"))
        gates = g["gates"]
        return {
            "cell": cell,
            "promote": g["promote"],
            "reward_first_third": gates["reward_trend"]["first_third_mean"],
            "reward_last_third": gates["reward_trend"]["last_third_mean"],
            "gain": gates["reward_trend"]["gain"],
            "mean_kl": gates["kl_bounded"]["mean_kl"],
            "max_kl": gates["kl_bounded"]["max_logged_kl"],
            "frac_zero_std": gates["zero_std_groups"]["mean_frac_zero_std"],
            "n_train_rows": meta["n_training_rows"],
            "n_flow_chunks": meta["n_flow_chunks"],
            "n_no_flow_chunks": meta["n_no_flow_chunks"],
        }

    gates_df = pd.DataFrame(
        [_gate_row(c, p) for c, p in M1_CKPT.items()]
    ).set_index("cell")
    gates_df.round(4)
    return (gates_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Two things worth flagging beyond the flat trend:

    * **`core`'s KL is an order of magnitude above the other cells**
      (≈0.62 vs 0.02–0.06 mean, max logged ≈27). With only the direct core
      active, the policy drifted much further from the reference *without
      any reward gain* — reward-free drift, plausibly the KL-vs-noise
      equilibrium of a reward whose within-group ordering is mostly
      format-gate noise (§4). Whatever `core`'s final adapter is, it is the
      *least* anchored of the four.
    * **Gate design**: `min_reward_gain = 0.0` is toothless (R4 below).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · Optimizer health (trainer_state)

    The flat reward is *not* an optimizer pathology: KL stays bounded, dead
    (zero-variance) groups are ~2%, completions never hit the length cap
    (`clipped_ratio ≈ 0` — important later: parse failures are **not**
    truncation), and mean completion length is stable ~730–770 tokens.
    """)
    return


@app.cell
def _(M1_CKPT, json, pd):
    _KEEP = [
        "reward", "reward_std", "kl", "frac_reward_zero_std", "entropy",
        "completions/mean_length", "completions/clipped_ratio",
    ]

    def _ts_rows(cell, path):
        state = json.load(open(path / "checkpoint-450/trainer_state.json"))
        for rec in state["log_history"]:
            if "reward" not in rec:
                continue
            yield {"cell": cell, "step": rec["step"],
                   **{k: rec.get(k) for k in _KEEP}}

    ts_df = pd.DataFrame(
        r for c, p in M1_CKPT.items() for r in _ts_rows(c, p)
    )
    ts_df.groupby("cell")[_KEEP].agg(["first", "last"]).round(3).T
    return (ts_df,)


@app.cell
def _(CELL_COLOR, CELL_ORDER, plt, ts_df):
    _fig, _axes = plt.subplots(2, 2, figsize=(11, 6), sharex=True)
    _panels = [
        ("reward", "reward (per-call mean)"),
        ("kl", "KL to reference"),
        ("frac_reward_zero_std", "frac dead groups"),
        ("entropy", "entropy"),
    ]
    for _ax, (_m, _t) in zip(_axes.flat, _panels):
        for _c in CELL_ORDER:
            _sub = ts_df[ts_df["cell"] == _c]
            _ax.plot(_sub["step"], _sub[_m], label=_c,
                     color=CELL_COLOR[_c], lw=1.2)
        _ax.set_title(_t, fontsize=10)
        _ax.grid(alpha=0.3)
    _axes[0, 0].legend(fontsize=8)
    _axes[1, 0].set_xlabel("step")
    _axes[1, 1].set_xlabel("step")
    _fig.suptitle("m1 wave — optimizer health", y=1.0)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · Reward traces — where the advantage mass actually goes

    Traces are parsed once and cached to parquet (`m1_wave_cache/`). Each
    call has 4 GRPO groups × 8 completions. Groups are classified `battery`
    (any row routed `vignette`) or `extract` (everything else, including
    no-flow chunks routed through the abstain table).
    """)
    return


@app.cell
def _(M1_CACHE, M1_CKPT, json, pd):
    def _parse_traces(cell, path):
        cache = M1_CACHE / f"traces_{cell.replace('-', 'minus_')}.parquet"
        if cache.exists():
            return pd.read_parquet(cache)
        rows = []
        group_ids = {}
        for line in open(path / "reward_traces.jsonl"):
            o = json.loads(line)
            gk = (o["call"], o["prompt_key"])
            gid = group_ids.setdefault(gk, len(group_ids))
            rows.append({
                "cell": cell,
                "call": o["call"],
                "group": gid,
                "route": o["route"],
                "score": o["score"],
                "outcome_term": o.get("outcome_term"),
                "n_flows": o.get("n_flows"),
                "gate_reason": o.get("gate_reason"),
                "chunk_id": o.get("chunk_id"),
                "battery_id": o.get("battery_id"),
            })
        df = pd.DataFrame(rows)
        task = (
            df.groupby("group")["route"]
            .agg(lambda r: "battery" if (r == "vignette").any() else "extract")
            .rename("task")
        )
        df = df.merge(task, on="group")
        df.to_parquet(cache)
        return df

    traces_df = pd.concat(
        [_parse_traces(c, p) for c, p in M1_CKPT.items()], ignore_index=True
    )
    (
        traces_df.groupby(["cell", "route"])
        .agg(n=("score", "size"), mean_score=("score", "mean"))
        .round(3)
    )
    return (traces_df,)


@app.cell
def _(CELL_COLOR, CELL_ORDER, plt, traces_df):
    _fig, _axes = plt.subplots(1, 3, figsize=(12, 3.4), sharex=True)
    _routes = [("scored", "extract scored (R-DIRECT term)"),
               ("vignette", "battery (T-VIGNETTE term)"),
               ("gate_fail", "gate-fail rate")]
    for _ax, (_r, _t) in zip(_axes, _routes):
        for _c in CELL_ORDER:
            _sub = traces_df[traces_df["cell"] == _c].copy()
            _sub["bin"] = _sub["call"] // 45 * 45
            if _r == "gate_fail":
                _y = _sub.groupby("bin")["route"].apply(
                    lambda s: (s == "gate_fail").mean())
            else:
                _m = _sub[_sub["route"] == _r]
                if _m.empty:
                    continue
                _y = _m.groupby("bin")["score"].mean()
            _ax.plot(_y.index, _y.values, label=_c,
                     color=CELL_COLOR[_c], lw=1.4, marker=".")
        _ax.set_title(_t, fontsize=10)
        _ax.grid(alpha=0.3)
        _ax.set_xlabel("call")
    _axes[0].legend(fontsize=8)
    _axes[0].set_ylabel("mean score")
    _fig.suptitle("Per-route reward trajectories — flat on every term, every cell",
                  y=1.05)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · The format gate — the user-flagged suspect, quantified

    Hypothesis to test: *a large share of reward is spent on JSON format
    scaffolding that SFT should already have taught; if saturated it should
    be deweighted.*

    The data says something sharper: the format gate is **not saturated —
    it is noise**, and it dominates the advantage signal while producing
    zero learning.

    * ~24.5–25.1% of extract-task completions score a hard 0.0 at the gate
      (the 16–17% figures earlier drafts quoted included battery rows in
      the denominator)
      (reason ≈87% `parse`, 12% `schema`), **flat across all 450 steps in
      every cell**.
    * It is *not* truncation: `completions/clipped_ratio ≈ 0` throughout.
      The SFT policy genuinely emits unparseable output on a quarter of
      extract rows, and 450 GRPO steps did not move that number.
    * The gate is binary (0 vs ~0.7), so wherever a group mixes passes and
      fails, the within-group spread — the thing GRPO differentiates — is
      mostly the gate, not reasoning quality.

    The decomposition below quantifies how much of the advantage mass the
    gate consumes; the dispersion check then shows the failures are
    approximately *stochastic per completion* — which explains why all
    that gradient bought nothing.
    """)
    return


@app.cell
def _(np, pd, traces_df):
    def _decompose(sub):
        """Within-group variance, total vs with gate-fail spread removed."""
        tot, content, mixed, allfail = 0.0, 0.0, 0, 0
        groups = sub.groupby("group")["score"]
        n_groups = groups.ngroups
        for _, s in groups:
            v = s.to_numpy(dtype=float)
            fail = v == 0.0
            tot += v.var()
            if fail.all():
                allfail += 1
                continue
            if fail.any():
                mixed += 1
            w = v.copy()
            w[fail] = v[~fail].mean()  # counterfactual: gate never fired
            content += w.var()
        return pd.Series({
            "n_groups": n_groups,
            "frac_groups_gate_split": mixed / n_groups,
            "frac_groups_all_fail": allfail / n_groups,
            "advantage_mass_total": tot,
            "advantage_mass_content_only": content,
            "format_share_of_advantage": 1 - content / tot if tot else np.nan,
        })

    fmt_decomp_df = (
        traces_df[traces_df["task"] == "extract"]
        .groupby("cell")
        .apply(_decompose, include_groups=False)
        .round(3)
    )
    fmt_decomp_df
    return


@app.cell
def _(np, pd, traces_df):
    def _overdispersion(sub):
        """Are parse failures random per completion, or prompt-determined?

        If failures were i.i.d. within a group at the observed rate p, the
        per-group fail count ~ Binomial(8, p). Prompt-driven failure would
        show up as excess mass at 0/8 and 8/8. Observed: only mild excess
        at the tails, essentially zero all-fail groups — failures are
        mostly stochastic sampling events, not a per-prompt trait.
        """
        ext = sub[sub["task"] == "extract"]
        fails = ext.groupby("group")["score"].apply(
            lambda s: int((s == 0.0).sum()))
        p = (ext["score"] == 0.0).mean()
        obs = fails.value_counts(normalize=True).sort_index()
        k = np.arange(9)
        from math import comb
        binom = pd.Series(
            [comb(8, int(i)) * p**i * (1 - p) ** (8 - i) for i in k], index=k)
        out = pd.DataFrame({"observed": obs, "binomial(8,p)": binom}).fillna(0)
        out.index.name = "n_gate_fails_in_group"
        return out

    _cell = "core"
    fmt_overdisp_df = _overdispersion(
        traces_df[traces_df["cell"] == _cell]).round(3)
    fmt_overdisp_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Reading the two tables together (core shown for the dispersion check;
    the other cells match):

    * **63–79% of extract advantage mass is the format gate** (core 0.63,
      full 0.77, −outcome 0.79, −vignette 0.78) — variance that
      differentiates "emitted parseable JSON" from "didn't", not better
      from worse CI reasoning. **82–85% of extract groups are split by the
      gate**, so this dominates nearly every group's advantage ordering.
    * Failures are only **mildly overdispersed vs binomial** (small excess
      at 0/8 and ≥5/8; all-fail groups ≈0%). Parse failure is mostly a
      *stochastic sampling event*, not a per-prompt or per-policy trait.
    * That resolves the paradox of "83% of groups carry format gradient,
      yet the rate never moves": the gate is punishing **sampling noise**.
      A near-random 0-vs-0.7 coin flip layered onto 8 completions creates
      large advantages with no consistent direction — the reward's
      dominant component is variance the policy cannot systematically
      reduce, which both drowns the R-DIRECT signal and gives the
      optimizer license to drift (cf. `core`'s KL excursion, §1).
    * So the verdict is stronger than "saturated → deweight": the gate is
      an unlearnable noise source consuming ~¾ of the signal budget. It
      must be softened or removed from the advantage calculation, not
      merely down-weighted (R1).

    ### 4b · Root cause found post-hoc: SFT/GRPO prompt mismatch

    Comparing the prompts (2026-07-28) showed the GRPO rollout prompt
    (`conf/prompt/ci_extraction.yaml`: expert persona, numbered
    pre-reasoning scaffold, explicit JSON schema block, "ONLY a JSON
    object") is a **completely different text** from the SFT training
    instruction (`sft_data_prep._build_ci_instruction()`: one paragraph,
    no schema block, no persona). The assistant *targets* did teach the
    gate's exact schema, but the policy never saw the GRPO instruction
    during training — the whole wave scored the SFT policy
    off-distribution. Worse, the YAML's header comment claims it is "used
    by both SFT data prep and GRPO", which is false (`sft_data_prep.py`
    hardcodes its own instruction) — same failure signature as the
    fiction-prompt-override dead-code incident.

    This also explains the near-binomial dispersion mechanistically: the
    GRPO prompt self-contradicts ("briefly reason about 1/2/3 first" vs
    "respond with ONLY a JSON object"), and an off-distribution policy
    resolves the contradiction per-sample — a prose-preamble mode that
    fails strict parse vs a straight-JSON mode that passes. A per-sample
    mode flip is indistinguishable from a coin flip at group level.

    **A/B probe result** (`scripts/ab_probe_sft_vs_grpo_prompt.py`, job
    527965, `outputs/2026-07-28_ab_prompt_probe/`): merged m1 SFT
    weights, 200 chunks from the core cell's actual extract population,
    8 samples each at the exact rollout params, `valid_gate` verbatim:

    | arm | gate-fail | reasons | chunks w/ 0 fails |
    |---|---|---|---|
    | GRPO prompt | **34.4%** | 513 parse / 32 schema | 25/200 |
    | SFT prompt  | **2.7%**  | 35 parse / 7 core_fields / 1 schema | 161/200 |

    Paired per chunk: SFT prompt strictly better on 165/200, worse on 6.
    Truncation negligible in both arms (0.25%). Same policy, same chunks,
    same sampling — a **12.8× failure reduction from the instruction text
    alone**. The prompt mismatch *is* the format problem; the true
    in-distribution format weakness is ~2.7%.

    (The probe's GRPO-prompt rate, 34.4%, exceeds the m1 in-training
    24.5% — the trained policy had drifted somewhat formatward by the
    steps the traces average over; direction unchanged.)

    Note the SFT-prompt arm also generates *longer* outputs (mean 1107 vs
    788 tokens) — the on-distribution instruction elicits the fuller
    reasoning field the SFT targets contained, so the alignment fix
    plausibly improves the scored content too, not just the parse rate.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 · Macro-EM collapse in the R-DIRECT term

    `outcome_term` is macro-EM over the classes *present in that
    completion's gold*. When a completion carries only one gold class
    (common: ~72/28 appropriate/inappropriate base rate, few flows per
    chunk), macro collapses to plain accuracy — a blanket "appropriate"
    labeler scores 1.0 on every single-class row. The distribution shows
    exactly that: a huge spike at 1.0 and the designed 0.5
    blanket-resistance visible only on the minority of mixed-gold rows.
    """)
    return


@app.cell
def _(CELL_ORDER, plt, traces_df):
    _fig, _axes = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
    for _ax, _c in zip(_axes, CELL_ORDER):
        _s = traces_df[(traces_df["cell"] == _c)
                       & (traces_df["route"] == "scored")]["outcome_term"]
        if _s.notna().sum() == 0:
            _ax.set_title(f"{_c} (no direct core)", fontsize=10)
            continue
        _ax.hist(_s.dropna(), bins=21, color="#1f77b4", edgecolor="white")
        _frac1 = (_s == 1.0).mean()
        _ax.set_title(f"{_c}  P(=1.0)={_frac1:.0%}", fontsize=10)
        _ax.set_xlabel("outcome_term")
    _axes[0].set_ylabel("completions")
    _fig.suptitle("Per-completion macro-EM — the 1.0 spike is single-gold-class"
                  " rows scored as plain accuracy", y=1.05)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Consequence: a cell can hold `outcome_term ≈ 0.73` while pooled
    balanced accuracy is ≈0.56 — the policy is rewarded ~0.73 for
    discrimination barely above a coin flip. The reward's
    blanket-resistance only bites on the ~28% of completions whose gold
    contains both classes (R2).

    ## 6 · Battery (T-VIGNETTE) — never dead, never informative

    Battery groups have **zero** dead groups but ~1/7 the within-group
    spread of extract groups: every completion lands in a tight band around
    0.60. GRPO advantages inside such a group are tiny — batteries occupy
    30% of training rows while contributing almost no gradient, and their
    term is pinned at 0.59–0.61 in every cell for the whole run (§3 plot).
    """)
    return


@app.cell
def _(pd, traces_df):
    def _group_spread(sub):
        stds = sub.groupby("group")["score"].std(ddof=0)
        return pd.Series({
            "n_groups": len(stds),
            "mean_within_group_std": stds.mean(),
            "frac_dead_groups": (stds < 1e-9).mean(),
        })

    battery_spread_df = (
        traces_df.groupby(["cell", "task"])
        .apply(_group_spread, include_groups=False)
        .round(4)
    )
    battery_spread_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7 · Discrimination — balanced accuracy & Youden's J

    Derived from the two per-class agreement recalls in W&B (this grid
    predates the direct `balanced_accuracy`/`youden_j` keys). Fetched once,
    cached to `m1_wave_cache/wandb_direct_history.parquet`.

    **Caveats:** `core` truncates at its W&B crash (~step 300; last ~150
    steps unrecoverable — R5). `−outcome` has no direct core, hence no
    series. J=0 is the blanket floor; ±0.05 wobble is within run noise —
    the `−vignette` cell flipped verdict twice across checkpoints on
    exactly that scale.
    """)
    return


@app.cell
def _(M1_CACHE, pd):
    _APPR = "reward/direct/agreement_by_class/appropriate"
    _INAP = "reward/direct/agreement_by_class/inappropriate"
    _EXTRA = ["reward/direct/agreement_mean", "reward/direct/hedge_frac",
              "reward/direct/unscored_flow_frac", "reward/valid/gate_frac"]

    _cache = M1_CACHE / "wandb_direct_history.parquet"
    if _cache.exists():
        wandb_hist_df = pd.read_parquet(_cache)
    else:
        import wandb

        _api = wandb.Api(timeout=60)
        _runs = [r for r in _api.runs("uair/grpo-ci-training",
                                      order="-created_at", per_page=60)
                 if "grpo_m1_" in r.name and "grpo_training" in r.name]
        _frames = []
        for _r in _runs:
            _rows = list(_r.scan_history(keys=[_APPR, _INAP] + _EXTRA,
                                         page_size=2000))
            _pairs = [x for x in _rows
                      if x.get(_APPR) is not None and x.get(_INAP) is not None]
            if len(_pairs) < 6:
                continue
            _cell = _r.name.split("grpo_m1_")[1].split("-grpo_training")[0]
            _df = pd.DataFrame(_pairs).rename(columns={
                _APPR: "recall_appropriate", _INAP: "recall_inappropriate",
                _EXTRA[0]: "agreement_mean", _EXTRA[1]: "hedge_frac",
                _EXTRA[2]: "unscored_flow_frac", _EXTRA[3]: "valid_gate_frac",
            })
            _df["cell"] = _cell
            _df["idx"] = range(len(_df))
            _frames.append(_df)
        # keep the longest history per cell (crashed core + short aborted
        # restarts both match the name filter)
        _longest = {}
        for _f in _frames:
            _c = _f["cell"].iloc[0]
            if _c not in _longest or len(_f) > len(_longest[_c]):
                _longest[_c] = _f
        wandb_hist_df = pd.concat(_longest.values(), ignore_index=True)
        wandb_hist_df.to_parquet(_cache)

    wandb_hist_df["balanced_accuracy"] = (
        wandb_hist_df["recall_appropriate"]
        + wandb_hist_df["recall_inappropriate"]) / 2
    wandb_hist_df["youden_j"] = 2 * wandb_hist_df["balanced_accuracy"] - 1
    wandb_hist_df.groupby("cell")[
        ["balanced_accuracy", "youden_j", "hedge_frac", "unscored_flow_frac"]
    ].agg(["mean", "last"]).round(3)
    return (wandb_hist_df,)


@app.cell
def _(CELL_COLOR, plt, wandb_hist_df):
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(11, 3.6))
    for _c, _g in wandb_hist_df.groupby("cell"):
        _r = _g["youden_j"].rolling(25, min_periods=5).mean()
        _ax1.plot(_g["idx"], _r, label=_c,
                  color=CELL_COLOR.get(_c, "gray"), lw=1.4)
        _ax2.plot(_g["idx"],
                  _g["unscored_flow_frac"].rolling(25, min_periods=5).mean(),
                  label=_c, color=CELL_COLOR.get(_c, "gray"), lw=1.4)
    _ax1.axhline(0, color="k", lw=0.8, ls="--")
    _ax1.fill_between([0, wandb_hist_df["idx"].max()], -0.05, 0.05,
                      color="gray", alpha=0.15, label="±0.05 noise band")
    _ax1.set_title("Youden's J (rolling 25) — no cell separates from the floor",
                   fontsize=10)
    _ax1.set_ylabel("J")
    _ax1.legend(fontsize=8)
    _ax2.set_title("unscored_flow_frac — ~0.37 of policy flows get a\n"
                   "non-governing top-1 norm (k=1 retrieval noise)", fontsize=10)
    for _ax in (_ax1, _ax2):
        _ax.grid(alpha=0.3)
        _ax.set_xlabel("logged reward call")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8 · Synthesis — what to change before wave 2

    Ranked by expected impact; each item cites the section with the
    evidence.

    **R1 — Align the GRPO rollout prompt with the SFT instruction (§4b),
    then soften what remains of the gate (§4).**
    RESOLVED by the A/B probe: the gate's ~24.5% in-training failure rate
    (63–79% of extract advantage mass) was the policy being scored on a
    prompt it never trained on. Under the SFT instruction the same policy
    fails only 2.7%. Fix, in order: (a) build the GRPO extract user
    message byte-identically to SFT
    (`_build_ci_instruction() + "\n\n" + chunk_text`) — a config/template
    change in the m-series path; (b) correct the false header in
    `conf/prompt/ci_extraction.yaml` claiming SFT and GRPO share it
    (second incident of a config comment asserting a prompt path the code
    does not execute — cf. the fiction-prompt-override finding);
    (c) with the failure base rate at ~2.7% the hard-0.0 gate is far less
    of a noise source, but a multiplicative format discount (score
    content leniently, ×0.8 on repaired parses) is still cleaner than
    zeroing, and cheap; (d) gold-side leak filtering is untouched — this
    concerns the policy-output gate only.

    **R2 — Fix the macro-EM collapse in R-DIRECT (§5).**
    50%+ of scored completions carry one gold class and get plain accuracy;
    the policy holds 0.73 reward at 0.56 balanced accuracy. Options:
    (a) pool per-class agreement across the whole GRPO *group* (all 8
    completions share a chunk, so the group's flows usually cover both
    classes) and score each completion against class-balanced targets;
    (b) reweight flows inversely to gold-class frequency within the
    training batch. Either way the reward, not just the metric, must price
    blanket labeling at 0.5 on *every* row.

    **R3 — Make batteries earn their 30% of rows (§6).**
    Battery groups: zero dead, but within-group std 0.05 (extract: 0.33) —
    uniformly mediocre completions produce ~no advantage, and the term is
    pinned at 0.60 in every cell. Either sharpen scoring (binary
    per-scenario credit rather than graded partial credit compresses less),
    raise difficulty spread per battery, or cut battery ratio and give the
    rows to extract. Running them at current settings is renting GPU time
    for a constant.

    **R4 — Give the promotion gate teeth (§1).**
    `min_reward_gain = 0.0` promoted four flat runs. Set it to a
    noise-calibrated positive threshold (the per-bin wobble here suggests
    ≥0.02 on the per-call mean), and add a discrimination gate: final
    pooled J ≥ some floor above 0, since total reward can rise for
    non-discrimination reasons (format, hedging mass).

    **R5 — Log per-flow gold/pred (+ sampled text) in reward_traces (§7).**
    The W&B crash made `core`'s last 150 steps of discrimination metrics
    unrecoverable because traces only carry scalar terms. Traces should be
    self-sufficient: per-flow gold label, predicted label, and retrieval
    margin per scored completion. That also unlocks offline J at any
    granularity and the teacher-error vs retrieval-noise split.
    (`unscored_flow_frac ≈ 0.37` itself was root-caused and fixed
    2026-07-28: retrieval ran over the full universe — 71% non-flow
    norms — instead of the restricted index the acceptance bar was
    measured on; see reward-direct-spec.md "CORRECTION 2026-07-28".)

    **R6 — Investigate `core`'s KL excursion before promoting its adapter
    (§1).** Mean KL 0.62 (10× siblings, max 27) with zero reward gain =
    maximal drift for minimal signal. If any m1 adapter goes to benchmark
    eval, prefer a mid-training checkpoint of a low-KL cell over
    `core@450`, and check `core`'s completions qualitatively for
    reward-hacking artifacts near the KL spikes.

    ### What the ablation did *not* answer

    The grid was designed to isolate R-GROUND / R-CONTRAST / T-VIGNETTE
    contributions. With every cell flat, cross-cell deltas measure noise:
    **do not** read the m1 grid as evidence that any component is
    useless — the shared core drowned all of them. The ablation needs to be
    re-run after R1–R3 land.
    """)
    return


if __name__ == "__main__":
    app.run()
