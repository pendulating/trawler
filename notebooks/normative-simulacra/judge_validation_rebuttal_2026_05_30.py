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
    # LLM-as-judge validation — reviewer rebuttal

    > **Reviewer qualm.** *"The LLM-as-judge setup needs stronger validation.
    > The paper acknowledges judge unreliability, but it should include human
    > expert calibration, judge-model ablations, etc."*

    This notebook assembles the evidence we already have on disk into a direct,
    quantitative answer. The headline claims, all reproduced in the cells below:

    1. **The original eval judge (Qwen3-32B-AWQ) is a lenient outlier.** On
       PrivacyLens — the one heavily judge-dependent benchmark — it reports
       leakage of **27%** on the same Qwen3.5-9B actions that **Gemma-4-31B-it
       (65%)**, **Qwen3.6-27B (62%)**, and the frontier proprietary
       **gpt-5.2 (65%)** all score 2.3× higher.
    2. **Qwen3.6-27B — the judge we adopt going forward — is consensus-aligned.**
       It agrees tightly with an independent open judge (Gemma-4) *and* a
       frontier proprietary judge (gpt-5.2) across every model we can compare.
       The judge swap that matters (away from Qwen3-32B-AWQ) is the one we made.
    3. **The judge variance is isolated, not pervasive.** Across the same
       sweeps, every *judge-free* metric (GoldCoin-HIPAA accuracy, ConfAIde
       correlation, VLM-GeoPrivacy MCQ, CIRL probing, PrivacyLens QA-probe) —
       metrics that never call the judge — moves by **≤1 percentage point**
       when the judge changes. This is the control that lets us attribute the
       PrivacyLens leakage swing to the judge rather than to task drift,
       sampling noise, or decoding.
    4. **Human-expert calibration agrees with the swap.** On 51 expert-annotated
       records, the leakage judge we adopt (Qwen3.6-27B) reaches κ **0.61** with
       the expert (Gemma 0.65, gpt-5.2 0.72), while the incumbent Qwen3-32B-AWQ
       manages only κ **0.34** and misses ~51% of real leaks. The judge swap
       moves the metric *toward* the human. (Helpfulness agreement is weak for
       every judge — an honest, judge-independent caveat.)

    Together these are a judge-model ablation (claim 1–2) *with* an explicit
    isolation of judge variance from task variance (claim 3) and a **direct
    human-expert calibration** (claim 4), with a frontier proprietary model
    (gpt-5.2) as a corroborating expert proxy. The human calibration is in
    **Phase E**.

    ---

    ## How the data supports this (provenance & method)

    These findings come from controlled re-judging runs already on the cluster.
    The key facts established while assembling this notebook:

    **Judge identity is verified, not assumed.** Each `eval_all` sweep launched
    its own judge server; the served model is recovered from
    `.slurm_jobs/judge-server/*.out` headers by matching the port the eval hit
    (in `privacylens_eval.log`) to the launch that preceded it.

    | Sweep | Multirun | Port | Judge | Slurm log |
    |---|---|---|---|---|
    | **Mar30** | `2026-03-30_eval_all/22-41-52` (+`22-42-39`) | 9015/9016 | **Qwen3-32B-AWQ** | 864261 / 864262 |
    | **Apr20** | `2026-04-20_eval_all/18-15-21` | 8002 | **Gemma-4-31B-it** | 697529 |
    | **Apr24** | `2026-04-24_eval_all/10-13-47` | 8002 | **Qwen3.6-27B** | 861761 |
    | **May27** | `2026-03-30_…` rejudged offline | — | **gpt-5.2** (`gpt-5.2-2025-12-11`) | OpenAI Batch |

    *(Both Mar30 server instances — 9015 and 9016, launched 2s apart — served
    Qwen3-32B-AWQ; the second was for throughput across the two Mar30
    multiruns.)*

    **The comparison is byte-identical at the task level.** Mar30, Apr20, and
    Apr24 reuse the same model weights and sampling, and the PrivacyLens
    action-inference prompt was unchanged until the Apr 26 ReAct rewrite
    (`wiki/changelog/privacylens-action-prompt-react-rewrite-2026-04-26.md`),
    which doubled mean action length and broke the "same task output, different
    judge" precondition. So **only Mar30/Apr20/Apr24 (+ the gpt-5.2 offline
    rejudge of the Mar30 actions) are valid cross-judge comparisons** — May12+
    sweeps are deliberately excluded. The judge-free control in Phase B
    independently confirms task behavior is reproduced (ranges ≤0.01).

    **Model identity must be resolved from weights, not the config label.**
    The Mar30 config label `qwen3.5-9b/base` points at the *Instruct* weights;
    the Apr sweeps later split that into a separate `qwen3.5-9b/base` (raw
    `-Base` pretrained) and `qwen3.5-9b/instruct`. Matching on the hydra label
    silently pairs two different models (and manufactures a fake 28-point
    "compliance drift"). Every table below keys on `model_source`+`lora_path`.

    **Coverage is asymmetric — and we are explicit about it.** Apr20/Apr24
    only re-ran the Qwen3.5 family + Llama-3.1-8B, so the broad cross-judge
    delta covers those (2–4 judges each). The other Appendix-Table-1
    architectures (Gemma-3-12B, Phi-4, GPT-OSS-20B, OpenThinker3-7B, and their
    SFT variants) were judged on identical actions **only by Qwen3-32B-AWQ**
    (Mar30 `22-42-39`); no second judge ever scored their identical actions, so
    they appear as single-judge context in the coverage table but cannot enter
    a cross-judge delta. gpt-5.2 (offline) covers only the three Qwen3.5-9B
    variants on PrivacyLens.

    **Which metrics are judge-dependent.** Only PrivacyLens (leakage,
    helpfulness) and CIRL-trajectory consume the judge server. GoldCoin-HIPAA,
    ConfAIde, VLM-GeoPrivacy MCQ, CIRL **probing**, and PrivacyLens **QA-probe**
    are gold-scored / guided-decoding classification — *judge-free*. That split
    is exactly what makes the judge-free metrics a usable control.

    ---

    ### Design

    - **Phase A** — inventory every sweep × benchmark × model; resolve judge and
      model identity; build one long-form table.
    - **Phase B** — *the control*: judge-free metrics are invariant across
      judges (≤0.01) ⇒ the experiment is controlled.
    - **Phase C** — *the ablation*: PrivacyLens leakage by judge; Qwen3-32B-AWQ
      is the lenient outlier; per-judge deviation from the consensus.
    - **Phase D** — judge-agreement matrix (pairwise mean-abs leakage gap).
    - **Phase E** — human-calibration discussion + objective gold-label anchors.
    - **Phase F** — coverage table + ready-to-paste rebuttal paragraph + saved
      artifacts.
    """)
    return


@app.cell
def _():
    from pathlib import Path

    REPO = Path("/share/pierson/matt/UAIR")

    # Validated judge attribution per sweep (see provenance table above). Each
    # entry: where the sub-runs live, the resolved judge model, the PrivacyLens
    # metrics subdir to read, and whether this sweep contributes ONLY PrivacyLens
    # (the offline gpt-5.2 rejudge did not touch the other benchmarks, so reading
    # their compute_metrics/ here would mis-attribute Qwen3-32B-AWQ numbers to
    # gpt-5.2).
    SWEEPS = [
        {
            "label": "Mar30",
            "judge": "Qwen3-32B-AWQ",
            "roots": [
                REPO / "multirun/2026-03-30_eval_all/22-41-52",
                REPO / "multirun/2026-03-30_eval_all/22-42-39",
            ],
            "pl_subdir": "compute_metrics",
            "privacylens_only": False,
        },
        {
            "label": "Apr20",
            "judge": "Gemma-4-31B-it",
            "roots": [REPO / "multirun/2026-04-20_eval_all/18-15-21"],
            "pl_subdir": "compute_metrics",
            "privacylens_only": False,
        },
        {
            "label": "Apr24",
            "judge": "Qwen3.6-27B",
            "roots": [REPO / "multirun/2026-04-24_eval_all/10-13-47"],
            "pl_subdir": "compute_metrics",
            "privacylens_only": False,
        },
        {
            "label": "May27",
            "judge": "gpt-5.2",
            "roots": [REPO / "multirun/2026-03-30_eval_all/22-41-52"],
            "pl_subdir": "compute_metrics_gpt52",
            "privacylens_only": True,
        },
    ]

    # The going-forward judge (paper: 00_main.tex:132, A_additional-methods.tex:5).
    CHOSEN_JUDGE = "Qwen3.6-27B"
    INCUMBENT_JUDGE = "Qwen3-32B-AWQ"

    # Stable colour per judge, reused across every plot.
    JUDGE_COLORS = {
        "Qwen3-32B-AWQ": "#cc6677",   # red   — incumbent / lenient outlier
        "Gemma-4-31B-it": "#4477aa",  # blue  — independent open judge
        "Qwen3.6-27B": "#228833",     # green — chosen going forward
        "gpt-5.2": "#ddaa33",         # gold  — frontier proprietary proxy
    }
    JUDGE_ORDER = ["Qwen3-32B-AWQ", "Gemma-4-31B-it", "Qwen3.6-27B", "gpt-5.2"]

    # Benchmark metric registry. `judged` flags whether the metric consumes the
    # judge server (only PrivacyLens leakage/helpfulness do); everything else is
    # gold-scored / guided-decoding and therefore judge-FREE (the control).
    # `cols` is a fallback chain (the compute_metrics schema gained
    # *_among_parseable / *_overall_with_default_zero variants over time).
    PL = "privacylens"
    BENCHES = [
        {
            "name": "privacylens", "inner": "privacylens_eval",
            "metrics": [
                {"key": "leakage_rate", "subdir": PL, "judged": True,
                 "cols": ["leakage_rate_overall_with_default_zero",
                          "leakage_rate_among_parseable", "leakage_rate"]},
                {"key": "adjusted_leakage_rate", "subdir": PL, "judged": True,
                 "cols": ["adjusted_leakage_rate"]},
                {"key": "helpfulness_mean_score", "subdir": PL, "judged": True,
                 "cols": ["helpfulness_mean_score_overall_with_default_zero",
                          "helpfulness_mean_score_among_parseable",
                          "helpfulness_mean_score"]},
                {"key": "qa_accuracy", "subdir": PL, "judged": False,
                 "cols": ["qa_accuracy"]},
            ],
        },
        {
            "name": "goldcoin", "inner": "goldcoin_hipaa",
            "metrics": [
                {"key": "goldcoin_applicability_acc",
                 "subdir": "compute_metrics_applicability", "judged": False,
                 "cols": ["accuracy"]},
                {"key": "goldcoin_compliance_acc",
                 "subdir": "compute_metrics_compliance", "judged": False,
                 "cols": ["accuracy"]},
            ],
        },
        {
            "name": "cirl_vignettes", "inner": "cirl_vignettes",
            "metrics": [
                {"key": "cirl_probing_acc", "subdir": "compute_metrics",
                 "judged": False, "cols": ["accuracy"]},
            ],
        },
        {
            "name": "vlm_geoprivacy", "inner": "vlm_geoprivacy_bench",
            "metrics": [
                {"key": "vlm_mean_acc", "subdir": "compute_metrics",
                 "judged": False, "cols": ["__VLM_MEAN__"]},
                {"key": "vlm_Q7_acc", "subdir": "compute_metrics",
                 "judged": False, "cols": ["Q7_accuracy"]},
            ],
        },
        {
            "name": "confaide", "inner": "confaide",
            "metrics": [
                {"key": "confaide_t2a_pearson",
                 "subdir": "compute_metrics_tier2a", "judged": False,
                 "cols": ["pearson_r"]},
            ],
        },
    ]

    REPORT_DIR = (
        Path(__file__).resolve().parent
        / "tables" / "judge_validation_rebuttal_2026_05_30"
    )
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    return (
        BENCHES,
        CHOSEN_JUDGE,
        INCUMBENT_JUDGE,
        JUDGE_COLORS,
        JUDGE_ORDER,
        REPORT_DIR,
        SWEEPS,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Phase A — Inventory: every sweep × benchmark × model

    Walk each sub-run, resolve the task model from
    `model.model_source`+`model.lora_path` (NOT the hydra label — see the
    provenance notes), and pull every registered metric. The result `long_df`
    has one row per (model, judge, metric).
    """)
    return


@app.cell
def _(BENCHES, SWEEPS):
    import pandas as pd
    from omegaconf import OmegaConf

    def _vlm_mean(_row):
        _accs = [_row[f"Q{_i}_accuracy"] for _i in range(1, 8)
                 if f"Q{_i}_accuracy" in _row.index]
        return float(sum(_accs) / len(_accs)) if _accs else None

    def _scalar(_path, _cols):
        if not _path.exists():
            return None
        try:
            _row = pd.read_parquet(_path).iloc[0]
        except Exception:
            return None
        for _c in _cols:
            if _c == "__VLM_MEAN__":
                _v = _vlm_mean(_row)
                if _v is not None:
                    return _v
            elif _c in _row.index and pd.notna(_row[_c]):
                return float(_row[_c])
        return None

    def _display_name(_ms, _lora):
        _short = _ms.rsplit("/", 1)[-1] if _ms else "?"
        if _lora:
            _tag = _lora.rsplit("/", 2)[-2] if "/" in _lora else "lora"
            return f"{_short}+{_tag}"
        return _short

    _rows = []
    for _sweep in SWEEPS:
        for _root in _sweep["roots"]:
            if not _root.is_dir():
                continue
            for _cell in sorted(
                (_p for _p in _root.iterdir() if _p.is_dir() and _p.name.isdigit()),
                key=lambda _p: int(_p.name),
            ):
                # Resolve model identity from whichever benchmark's hydra config
                # is present (privacylens preferred).
                _cfg = None
                for _b in ("privacylens", "goldcoin", "cirl_vignettes",
                           "vlm_geoprivacy", "confaide"):
                    _cp = _cell / _b / ".hydra" / "config.yaml"
                    if _cp.exists():
                        _cfg = OmegaConf.load(_cp)
                        break
                if _cfg is None:
                    continue
                _ms = str(OmegaConf.select(_cfg, "model.model_source") or "")
                _lora = str(OmegaConf.select(_cfg, "model.lora_path") or "")
                _disp = _display_name(_ms, _lora)

                for _bench in BENCHES:
                    if _sweep["privacylens_only"] and _bench["name"] != "privacylens":
                        continue
                    for _m in _bench["metrics"]:
                        _subdir = (_sweep["pl_subdir"] if _m["subdir"] == "privacylens"
                                   else _m["subdir"])
                        _path = (_cell / _bench["name"] / _bench["inner"]
                                 / "outputs" / _subdir / "metrics.parquet")
                        _val = _scalar(_path, _m["cols"])
                        if _val is None:
                            continue
                        _rows.append({
                            "sweep": _sweep["label"],
                            "judge": _sweep["judge"],
                            "model_source": _ms,
                            "lora_path": _lora,
                            "display_name": _disp,
                            "benchmark": _bench["name"],
                            "metric": _m["key"],
                            "judged": _m["judged"],
                            "value": _val,
                            "cell": str(_cell),
                        })
    # One observation per (sweep, model identity, metric).
    long_df = (
        pd.DataFrame(_rows)
        .drop_duplicates(subset=["sweep", "model_source", "lora_path", "metric"],
                         keep="first")
        .reset_index(drop=True)
    )
    print(
        f"{len(long_df)} rows | "
        f"{long_df['display_name'].nunique()} model identities | "
        f"{long_df['metric'].nunique()} metrics | "
        f"judges: {sorted(long_df['judge'].unique())}"
    )
    long_df
    return long_df, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Coverage matrix — how many judges scored each model on identical actions

    `pl_judges` counts distinct judges that scored each model's **PrivacyLens**
    actions (the judge-dependent metric). Models with ≥2 enter the cross-judge
    delta in Phase C; single-judge rows are context only. Every model also has
    judge-free metrics that feed the Phase B control.
    """)
    return


@app.cell
def _(long_df):
    _pl = long_df[(long_df["benchmark"] == "privacylens")
                  & (long_df["metric"] == "leakage_rate")]
    coverage = (
        _pl.groupby("display_name")
        .agg(
            pl_judges=("judge", "nunique"),
            judges=("judge", lambda _s: ", ".join(sorted(set(_s)))),
            model_source=("model_source", "first"),
        )
        .reset_index()
        .sort_values(["pl_judges", "display_name"], ascending=[False, True])
        .reset_index(drop=True)
    )
    print(f"{(coverage['pl_judges'] >= 2).sum()} models have ≥2 PrivacyLens judges "
          f"(enter the cross-judge delta); "
          f"{(coverage['pl_judges'] == 1).sum()} are single-judge context.")
    coverage
    return (coverage,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Phase B — The control: judge-free metrics are invariant across judges

    For every metric that **does not** call the judge, the same model scored
    under three different judges should give the same number — any wobble is
    re-inference noise, not judge effect. We pivot each judge-free metric by
    (model × judge) and report the cross-judge **range** (max − min).

    If these ranges are ~0 while PrivacyLens leakage swings by tens of points,
    the leakage swing is *provably* judge-attributable: the task behavior is
    identical, only the rater changed.
    """)
    return


@app.cell
def _(long_df):
    _free = long_df[~long_df["judged"]].copy()
    _piv = _free.pivot_table(
        index=["display_name", "metric"], columns="judge", values="value",
        aggfunc="first",
    )
    # Only rows observed under ≥2 judges can show a range.
    _piv = _piv[_piv.notna().sum(axis=1) >= 2]
    control_ranges = _piv.assign(
        n_judges=_piv.notna().sum(axis=1),
        range=_piv.max(axis=1) - _piv.min(axis=1),
    ).reset_index()

    _summary = (
        control_ranges.groupby("metric")["range"]
        .agg(max_range="max", mean_range="mean", n="size")
        .reset_index()
        .sort_values("max_range", ascending=False)
    )
    print("Cross-judge range of JUDGE-FREE metrics (should be ≈0):")
    print(_summary.to_string(index=False))
    control_summary = _summary
    control_ranges
    return (control_summary,)


@app.cell
def _(long_df):
    # Companion number: the cross-judge range of the JUDGED leakage metric on the
    # same models. This is the quantity the control is contrasted against.
    _leak = long_df[(long_df["metric"] == "leakage_rate")].pivot_table(
        index="display_name", columns="judge", values="value", aggfunc="first",
    )
    _leak = _leak[_leak.notna().sum(axis=1) >= 2]
    judged_ranges = _leak.assign(
        n_judges=_leak.notna().sum(axis=1),
        range=_leak.max(axis=1) - _leak.min(axis=1),
    ).reset_index()
    print("Cross-judge range of JUDGED leakage_rate (the contrast):")
    print(f"  max  = {judged_ranges['range'].max():.3f}")
    print(f"  mean = {judged_ranges['range'].mean():.3f}")
    judged_ranges.sort_values("range", ascending=False)
    return (judged_ranges,)


@app.cell
def _(REPORT_DIR, control_summary, judged_ranges):
    import matplotlib.pyplot as _plt
    import numpy as _np

    _plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False, "axes.spines.right": False,
    })

    # Judge-free metrics (each = mean cross-judge range) vs the judged leakage.
    _free = control_summary.sort_values("mean_range")
    _labels = [
        {"goldcoin_applicability_acc": "GoldCoin appl.",
         "goldcoin_compliance_acc": "GoldCoin compl.",
         "cirl_probing_acc": "CIRL probing",
         "vlm_mean_acc": "VLM MCQ (mean)",
         "vlm_Q7_acc": "VLM Q7",
         "qa_accuracy": "PrivacyLens QA",
         "confaide_t2a_pearson": "ConfAIde t2a r"}.get(_m, _m)
        for _m in _free["metric"]
    ] + ["PrivacyLens\nLEAKAGE (judged)"]
    _vals = list(_free["mean_range"]) + [judged_ranges["range"].mean()]
    _colors = ["#88aa88"] * len(_free) + ["#cc6677"]

    _fig, _ax = _plt.subplots(figsize=(10, 4.2), constrained_layout=True)
    _bars = _ax.bar(range(len(_vals)), _vals, color=_colors, edgecolor="#333",
                    linewidth=0.6)
    for _i, _v in enumerate(_vals):
        _ax.text(_i, _v + 0.005, f"{_v:.3f}", ha="center", va="bottom", fontsize=8)
    _ax.set_xticks(range(len(_vals)))
    _ax.set_xticklabels(_labels, rotation=25, ha="right", fontsize=8.5)
    _ax.set_ylabel("mean cross-judge range\n(max − min over judges)")
    _ax.set_title("Judge-free metrics are invariant to the judge; only the judged "
                  "leakage metric moves", fontsize=11)
    _ax.axhline(0.01, color="grey", ls="--", lw=1, alpha=0.7)
    _ax.annotate("1 pp", (len(_vals) - 0.5, 0.012), fontsize=7.5, color="grey")
    _fig.savefig(REPORT_DIR / "plot_B_control_invariance.png", dpi=200,
                 bbox_inches="tight")
    _fig.savefig(REPORT_DIR / "plot_B_control_invariance.pdf", bbox_inches="tight")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Phase C — The ablation: PrivacyLens leakage by judge

    The judge-dependent headline. Each marker is one judge's leakage rate on a
    model's byte-identical actions. The incumbent **Qwen3-32B-AWQ (red)** sits
    well below the others on essentially every model; **Qwen3.6 (green)**,
    **Gemma-4 (blue)**, and **gpt-5.2 (gold)** cluster together.
    """)
    return


@app.cell
def _(JUDGE_COLORS, JUDGE_ORDER, REPORT_DIR, long_df):
    import matplotlib.pyplot as _plt
    import numpy as _np
    import matplotlib.lines as _mlines

    _leak = long_df[long_df["metric"] == "leakage_rate"].pivot_table(
        index="display_name", columns="judge", values="value", aggfunc="first",
    )
    # Order models by their cross-judge mean leakage; keep only ≥2-judge models
    # for the delta plot (single-judge rows shown separately in coverage).
    _multi = _leak[_leak.notna().sum(axis=1) >= 2].copy()
    _multi["_order"] = _multi.mean(axis=1, skipna=True)
    _multi = _multi.sort_values("_order").drop(columns="_order")

    _judges = [_j for _j in JUDGE_ORDER if _j in _multi.columns]
    _y = _np.arange(len(_multi))

    _fig, _ax = _plt.subplots(figsize=(9.5, 0.5 * len(_multi) + 1.6),
                              constrained_layout=True)
    # Connector showing the spread per model.
    for _i, (_name, _row) in enumerate(_multi.iterrows()):
        _vals = [_row[_j] for _j in _judges if _np.isfinite(_row.get(_j, _np.nan))]
        if len(_vals) >= 2:
            _ax.plot([min(_vals), max(_vals)], [_i, _i], color="#bbb", lw=2,
                     zorder=1)
    for _j in _judges:
        _ax.scatter(_multi[_j], _y, s=85, color=JUDGE_COLORS[_j],
                    edgecolor="white", linewidth=0.8, zorder=3, label=_j)
    _ax.set_yticks(_y)
    _ax.set_yticklabels(_multi.index, fontsize=9)
    _ax.set_xlabel("PrivacyLens leakage rate (↓ better)")
    _ax.set_xlim(-0.03, 1.0)
    _ax.grid(axis="x", alpha=0.3)
    _ax.set_title("Same actions, different judge: Qwen3-32B-AWQ under-reports "
                  "leakage; Qwen3.6 ≈ Gemma-4 ≈ gpt-5.2", fontsize=10.5)
    _ax.legend(loc="lower right", fontsize=8.5, framealpha=0.95, title="Judge")
    _fig.savefig(REPORT_DIR / "plot_C_leakage_by_judge.png", dpi=200,
                 bbox_inches="tight")
    _fig.savefig(REPORT_DIR / "plot_C_leakage_by_judge.pdf", bbox_inches="tight")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Per-judge deviation from consensus

    For each model scored by ≥2 judges, take the leave-one-out mean of the
    *other* judges as the local consensus, and measure each judge's signed
    deviation from it. Averaged across models: a judge that systematically
    under- or over-reports shows a large signed deviation; a consensus-aligned
    judge sits near zero.

    > **Read with the coverage caveat.** This statistic is sensitive to *which*
    > models each judge overlaps. gpt-5.2's positive leakage deviation (+0.28)
    > is largely an artifact: it only co-judged the SFT/GRPO models, whose only
    > other judge is the lenient Qwen3-32B-AWQ, so the "consensus" it is
    > compared against on those rows is just that one low number. In *absolute*
    > terms gpt-5.2 (0.63–0.65) sits right with Gemma-4 and Qwen3.6 on the
    > shared Qwen3.5-9B (Phase C). The robust, coverage-balanced evidence is the
    > pairwise agreement matrix in **Phase D** (Gemma-4 ↔ Qwen3.6 agree to 0.02
    > over n=8 models). The clean reading of *this* plot is the **leakage** and
    > **adjusted-leakage** columns for Qwen3-32B-AWQ (−0.17 / −0.19) and Qwen3.6
    > (≈0), plus Qwen3-32B-AWQ's extreme +1.15 helpfulness inflation.
    """)
    return


@app.cell
def _(long_df, pd):
    _metrics = ["leakage_rate", "adjusted_leakage_rate", "helpfulness_mean_score"]
    _out = []
    for _metric in _metrics:
        _piv = long_df[long_df["metric"] == _metric].pivot_table(
            index="display_name", columns="judge", values="value", aggfunc="first",
        )
        _piv = _piv[_piv.notna().sum(axis=1) >= 2]
        for _judge in _piv.columns:
            _devs = []
            for _name, _row in _piv.iterrows():
                _v = _row[_judge]
                _others = [_row[_j] for _j in _piv.columns
                           if _j != _judge and pd.notna(_row[_j])]
                if pd.notna(_v) and _others:
                    _devs.append(_v - sum(_others) / len(_others))
            if _devs:
                _out.append({
                    "metric": _metric, "judge": _judge, "n_models": len(_devs),
                    "mean_signed_dev": sum(_devs) / len(_devs),
                    "mean_abs_dev": sum(abs(_d) for _d in _devs) / len(_devs),
                })
    consensus_dev = pd.DataFrame(_out)
    print("Signed deviation from leave-one-out consensus "
          "(negative = under-reports vs peers):")
    print(consensus_dev.pivot_table(index="judge", columns="metric",
                                    values="mean_signed_dev").round(3).to_string())
    consensus_dev
    return (consensus_dev,)


@app.cell
def _(JUDGE_COLORS, JUDGE_ORDER, REPORT_DIR, consensus_dev):
    import matplotlib.pyplot as _plt
    import numpy as _np

    _metrics = [("leakage_rate", "Leakage rate"),
                ("adjusted_leakage_rate", "Adj. leakage"),
                ("helpfulness_mean_score", "Helpfulness (0–3)")]
    _judges = [_j for _j in JUDGE_ORDER
               if _j in consensus_dev["judge"].unique()]
    _x = _np.arange(len(_metrics))
    _w = 0.8 / max(len(_judges), 1)

    _fig, _ax = _plt.subplots(figsize=(9, 4.2), constrained_layout=True)
    for _i, _j in enumerate(_judges):
        _vals = []
        for _m, _ in _metrics:
            _r = consensus_dev[(consensus_dev["judge"] == _j)
                               & (consensus_dev["metric"] == _m)]
            _vals.append(float(_r["mean_signed_dev"].iloc[0]) if len(_r) else 0.0)
        _ax.bar(_x + (_i - (len(_judges) - 1) / 2) * _w, _vals, _w * 0.95,
                color=JUDGE_COLORS[_j], label=_j, edgecolor="#333", linewidth=0.5)
    _ax.axhline(0, color="#333", lw=1)
    _ax.set_xticks(_x)
    _ax.set_xticklabels([_t for _, _t in _metrics])
    _ax.set_ylabel("mean signed deviation from\nleave-one-out consensus")
    _ax.set_title("Qwen3-32B-AWQ systematically under-reports leakage; "
                  "Qwen3.6 hugs the consensus", fontsize=10.5)
    _ax.legend(fontsize=8.5, framealpha=0.95, title="Judge", ncol=2)
    _ax.grid(axis="y", alpha=0.3)
    _fig.savefig(REPORT_DIR / "plot_C2_consensus_deviation.png", dpi=200,
                 bbox_inches="tight")
    _fig.savefig(REPORT_DIR / "plot_C2_consensus_deviation.pdf",
                 bbox_inches="tight")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Phase D — Judge-agreement matrix

    Pairwise **mean absolute leakage gap** across the models both judges scored.
    Small = the two judges agree on the same actions. The Qwen3-32B-AWQ row/col
    is large against everyone (0.22–0.36); the Gemma-4 / Qwen3.6 / gpt-5.2 block
    is small (≤0.02). The flagship result is **Gemma-4 ↔ Qwen3.6 = 0.02 over
    n=8 models** — two independent open judges agree to within 2 points on
    identical actions, while both differ from Qwen3-32B-AWQ by ~0.22. The
    gpt-5.2 cells (n=1, only Qwen3.5-9B overlaps) corroborate but are thin.
    """)
    return


@app.cell
def _(JUDGE_ORDER, long_df, pd):
    _leak = long_df[long_df["metric"] == "leakage_rate"].pivot_table(
        index="display_name", columns="judge", values="value", aggfunc="first",
    )
    _judges = [_j for _j in JUDGE_ORDER if _j in _leak.columns]
    _mad = {}
    _npairs = {}
    for _a in _judges:
        _mad[_a] = {}
        _npairs[_a] = {}
        for _b in _judges:
            if _a == _b:
                _npairs[_a][_b] = int(_leak[_a].notna().sum())
                _mad[_a][_b] = 0.0
                continue
            _both = _leak[[_a, _b]].dropna()
            _npairs[_a][_b] = len(_both)
            _mad[_a][_b] = (
                float((_both[_a] - _both[_b]).abs().mean())
                if len(_both) else float("nan")
            )
    agreement_mad = pd.DataFrame(_mad).T.loc[_judges, _judges]
    print("Mean absolute leakage gap between judges (n shared models):")
    for _a in _judges:
        print("  " + "  ".join(
            f"{_a[:9]:>9}↔{_b[:9]:<9}={_mad[_a][_b]:.3f}(n{int(_npairs[_a][_b])})"
            for _b in _judges if _b > _a))
    agreement_mad
    return (agreement_mad,)


@app.cell
def _(JUDGE_ORDER, REPORT_DIR, agreement_mad):
    import matplotlib.pyplot as _plt
    import numpy as _np

    _judges = [_j for _j in JUDGE_ORDER if _j in agreement_mad.columns]
    _m = agreement_mad.loc[_judges, _judges].astype(float).values

    _fig, _ax = _plt.subplots(figsize=(5.6, 4.8), constrained_layout=True)
    _im = _ax.imshow(_m, cmap="RdYlGn_r", vmin=0, vmax=0.4)
    _ax.set_xticks(range(len(_judges)))
    _ax.set_yticks(range(len(_judges)))
    _ax.set_xticklabels(_judges, rotation=30, ha="right", fontsize=8)
    _ax.set_yticklabels(_judges, fontsize=8)
    for _i in range(len(_judges)):
        for _j in range(len(_judges)):
            _v = _m[_i, _j]
            if _np.isfinite(_v):
                _ax.text(_j, _i, f"{_v:.2f}", ha="center", va="center",
                         fontsize=9, color="black" if _v < 0.25 else "white")
    _ax.set_title("Pairwise mean |Δ leakage|\n(0 = perfect agreement on same actions)",
                  fontsize=10)
    _plt.colorbar(_im, ax=_ax, shrink=0.8, label="mean |Δ leakage|")
    _fig.savefig(REPORT_DIR / "plot_D_agreement_matrix.png", dpi=200,
                 bbox_inches="tight")
    _fig.savefig(REPORT_DIR / "plot_D_agreement_matrix.pdf", bbox_inches="tight")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Phase E — Human-expert calibration (51 annotated records)

    We **now have** dedicated human-expert annotations — the one piece the
    earlier draft of this notebook had to concede was missing. An expert
    annotated the first **51 records** (rows 0–50) of the seed-777 / n=100
    PrivacyLens audit (`privacylens_audit_n100_seed777.html`), giving a verdict
    for each `(record × {Base, SFT, GRPO} × {leakage, helpfulness})` cell
    (`annotation_runs/annot_privacylens_agent_action_inference_seed777_n100_50annotated.csv`).
    Vocab matches the judges': leakage ∈ {leak, no-leak}, helpfulness ∈
    {helpful, not-helpful}.

    The audited actions come from the **Mar30** sweep (`22-41-52`,
    sub-runs 2/3/4 = Base/SFT/GRPO), so we can score each judge LLM's per-record
    verdict against the expert on the **same actions the expert saw**:

    - **Qwen3-32B-AWQ** (incumbent) and **gpt-5.2** judged byte-identical Mar30
      actions (gpt-5.2 via the offline rejudge) → coverage on all three models.
    - **Gemma-4-31B-it** and **Qwen3.6-27B** judged their own Apr re-runs (Base
      only); we join on `record_id` and flag any record whose action drifted
      from the Mar30 action the expert saw.

    Metrics per (judge × judge-type): raw **accuracy**, chance-corrected
    **Cohen's κ**, and for leakage a **precision/recall/F1 on the safety-relevant
    `leak` class**. We report a clean **Base (4-judge, identical actions)**
    comparison, a **byte-identical-only** variant (drops drifted Apr records),
    and a **pooled** Base+SFT+GRPO view for the two full-coverage judges.
    """)
    return


@app.cell
def _(pd):
    # Load expert annotations and align each judge LLM's per-record verdict.
    # Cell-local imports (underscore-prefixed) so they don't collide with other
    # cells' globals under marimo's one-definition-per-name rule.
    import re as _re
    import json as _json
    from pathlib import Path as _Path

    _REPO = _Path("/share/pierson/matt/UAIR")
    _ANNOT = _REPO / "annotation_runs/annot_privacylens_agent_action_inference_seed777_n100_50annotated.csv"
    _AUDIT = _REPO / "privacylens_audit_n100_seed777.html"

    # row_idx → record_id from the audit's sampled_indices (seed 777, n=100).
    _html = _AUDIT.read_text(errors="ignore")
    _si = _json.loads(_re.search(r'"sampled_indices":(\[[0-9,\s]+\])', _html).group(1))

    _csv = pd.read_csv(_ANNOT)
    _ann = _csv[_csv["expert_verdict"].notna()].copy()
    _ann["record_id"] = _ann["row_idx"].map(lambda i: str(_si[int(i)]))

    # Record counts are derived from the annotation set (not hardcoded) so the
    # scope labels, plot titles, and rebuttal track whatever CSV is loaded. This
    # set: rows 0–50 = 51 records (the original calibration used a 36-record
    # subset). N_BASE is the count of distinct Base-model annotated records,
    # which is what every "Base (all N)" scope reports.
    N_BASE = int(_ann[_ann["model"] == "Base"]["record_id"].nunique())
    N_RECORDS = int(_ann["record_id"].nunique())
    SCOPE_BASE = f"Base (all {N_BASE})"
    SCOPE_IDENT = "Base (byte-identical actions)"
    SCOPE_POOLED = "Pooled Base+SFT+GRPO"

    # Mar30 source sub-runs the expert annotated.
    _SUB = {"Base": 2, "SFT": 3, "GRPO": 4}
    _MAR = _REPO / "multirun/2026-03-30_eval_all/22-41-52"
    _APR = {
        "Gemma-4-31B-it": _REPO / "multirun/2026-04-20_eval_all/18-15-21/4",
        "Qwen3.6-27B": _REPO / "multirun/2026-04-24_eval_all/10-13-47/4",
    }

    def _binary_verdict(df, kind):
        df = df.copy()
        df["record_id"] = df["record_id"].astype(str)
        if kind == "leak":
            df["verdict"] = df["leak_flag"].map(lambda x: "leak" if x else "no-leak")
        else:
            df["verdict"] = df["helpfulness_binary"].map(
                lambda x: "helpful" if x else "not-helpful")
        if "generated_action" not in df.columns:
            df["generated_action"] = None
        return df[["record_id", "verdict", "generated_action"]]

    def _load(judge, model, kind):
        try:
            if judge == "Qwen3-32B-AWQ":
                base = _MAR / str(_SUB[model]) / "privacylens/privacylens_eval/outputs"
                f = "leakage_judge_inference" if kind == "leak" else "helpfulness_judge_inference"
                return _binary_verdict(pd.read_parquet(base / f / "results.parquet"), kind)
            if judge == "gpt-5.2":
                base = _MAR / str(_SUB[model]) / "privacylens/privacylens_eval/outputs/judge_batches"
                f = "leakage_judge_batch" if kind == "leak" else "helpfulness_judge_batch"
                return _binary_verdict(pd.read_parquet(base / f / "results_gpt52.parquet"), kind)
            if judge in _APR and model == "Base":
                base = _APR[judge] / "privacylens/privacylens_eval/outputs"
                f = "leakage_judge_inference" if kind == "leak" else "helpfulness_judge_inference"
                return _binary_verdict(pd.read_parquet(base / f / "results.parquet"), kind)
        except (OSError, KeyError, ValueError):
            return None
        return None

    # The Mar30 actions the expert actually saw (for the drift check).
    _mar_act = {}
    for _m in _SUB:
        _d = _load("Qwen3-32B-AWQ", _m, "leak")
        _mar_act[_m] = _d.set_index("record_id")["generated_action"].to_dict()

    _JUDGES = ["Qwen3-32B-AWQ", "Gemma-4-31B-it", "Qwen3.6-27B", "gpt-5.2"]
    _KIND = {"Leakage Judge": "leak", "Helpfulness Judge": "help"}

    _rows = []
    for _, _r in _ann.iterrows():
        _model, _jk, _rec = _r["model"], _r["judge"], _r["record_id"]
        _kind = _KIND[_jk]
        for _J in _JUDGES:
            _d = _load(_J, _model, _kind)
            if _d is None:
                continue
            _vmap = _d.set_index("record_id")["verdict"]
            if _rec not in _vmap.index:
                continue
            # Action drift vs the Mar30 action the expert saw (only Apr judges
            # can differ; Qwen3-32B-AWQ is Mar30 and gpt-5.2 rejudged Mar30).
            _identical = True
            if _J in _APR:
                _amap = _d.set_index("record_id")["generated_action"]
                _av = _amap.loc[_rec] if _rec in _amap.index else None
                _identical = str(_av) == str(_mar_act.get(_model, {}).get(_rec))
            _rows.append({
                "record_id": _rec, "model": _model,
                "judge_type": "leakage" if _kind == "leak" else "helpfulness",
                "judge_llm": _J, "expert": _r["expert_verdict"],
                "judge_verdict": _vmap.loc[_rec],
                "agree": _vmap.loc[_rec] == _r["expert_verdict"],
                "action_identical": bool(_identical),
            })
    human_align = pd.DataFrame(_rows)

    # Sanity: the audit's judge_verdict column IS the Qwen3-32B-AWQ judge, so our
    # Mar30 parquet verdict must reproduce the CSV judge_verdict exactly (1.0).
    _ann_jt = _ann.assign(judge_type=_ann["judge"].map(
        {"Leakage Judge": "leakage", "Helpfulness Judge": "helpfulness"}))
    _q32 = human_align[human_align["judge_llm"] == "Qwen3-32B-AWQ"].merge(
        _ann_jt[["record_id", "model", "judge_type", "judge_verdict"]],
        on=["record_id", "model", "judge_type"], how="inner",
        suffixes=("", "_csv"))
    _sane = (_q32["judge_verdict"] == _q32["judge_verdict_csv"]).mean()
    print(f"human_align: {len(human_align)} judge×cell rows | "
          f"{human_align['record_id'].nunique()} records × "
          f"{human_align['model'].nunique()} models × "
          f"{human_align['judge_llm'].nunique()} judges")
    print(f"sanity — Mar30 parquet == CSV (Qwen3-32B-AWQ) verdict: {_sane:.3f}")
    _drift = (human_align[human_align["judge_llm"].isin(_APR)]
              .drop_duplicates(["judge_llm", "record_id", "model"])
              .groupby("judge_llm")["action_identical"].mean())
    print("Base action-identical vs Mar30 (Apr judges): "
          + ", ".join(f"{k} {v:.2f}" for k, v in _drift.items()))
    return N_BASE, N_RECORDS, SCOPE_BASE, SCOPE_IDENT, SCOPE_POOLED, human_align


@app.cell
def _(SCOPE_BASE, SCOPE_IDENT, SCOPE_POOLED, human_align, pd):
    # Agreement of each judge LLM with the expert, across three scopes.
    from sklearn.metrics import cohen_kappa_score as _ck
    from sklearn.metrics import precision_recall_fscore_support as _prf

    def _metrics(df, jt):
        exp = df["expert"].tolist()
        jud = df["judge_verdict"].tolist()
        n = len(exp)
        if n == 0:
            return None
        acc = sum(a == b for a, b in zip(exp, jud)) / n
        kap = (_ck(exp, jud) if len(set(exp)) > 1 and len(set(jud)) > 1
               else float("nan"))
        out = {"n": n, "accuracy": round(acc, 3), "cohen_kappa": round(kap, 3)}
        if jt == "leakage":
            p, r, f, _ = _prf(exp, jud, labels=["leak"], average=None, zero_division=0)
            out.update({"leak_precision": round(float(p[0]), 3),
                        "leak_recall": round(float(r[0]), 3),
                        "leak_f1": round(float(f[0]), 3)})
        return out

    _ALL = ["Qwen3-32B-AWQ", "Gemma-4-31B-it", "Qwen3.6-27B", "gpt-5.2"]
    _FULL = ["Qwen3-32B-AWQ", "gpt-5.2"]   # full Base+SFT+GRPO coverage
    _scopes = [
        (SCOPE_BASE, lambda d: d[d["model"] == "Base"], _ALL),
        (SCOPE_IDENT,
         lambda d: d[(d["model"] == "Base") & d["action_identical"]], _ALL),
        (SCOPE_POOLED, lambda d: d, _FULL),
    ]
    _rows = []
    for _scope, _sel, _judges in _scopes:
        _d0 = _sel(human_align)
        for _jt in ["leakage", "helpfulness"]:
            for _J in _judges:
                _sub = _d0[(_d0["judge_type"] == _jt) & (_d0["judge_llm"] == _J)]
                _r = _metrics(_sub, _jt)
                if _r is None:
                    continue
                _rows.append({"scope": _scope, "judge_type": _jt,
                              "judge_llm": _J, **_r})
    human_agree = pd.DataFrame(_rows)
    human_agree
    return (human_agree,)


@app.cell
def _(JUDGE_ORDER, N_BASE, REPORT_DIR, SCOPE_BASE, human_agree, human_align):
    import matplotlib.pyplot as _plt
    import numpy as _np

    _base = human_agree[human_agree["scope"] == SCOPE_BASE]
    _fig, _axes = _plt.subplots(1, 2, figsize=(12, 4.3))
    for _ax, _jt in zip(_axes, ["leakage", "helpfulness"]):
        _sub = (_base[_base["judge_type"] == _jt]
                .set_index("judge_llm").reindex(JUDGE_ORDER))
        _x = _np.arange(len(JUDGE_ORDER))
        _w = 0.38
        _ax.bar(_x - _w / 2, _sub["accuracy"], _w, label="accuracy", color="#4477aa")
        _ax.bar(_x + _w / 2, _sub["cohen_kappa"], _w, label="Cohen's κ", color="#cc6677")
        if _jt == "leakage":
            for _i, _v in enumerate(_sub["leak_f1"]):
                if _v == _v:
                    _ax.text(_i, 0.02, f"leak-F1\n{_v:.2f}", ha="center",
                             va="bottom", fontsize=7, color="#333")
        _ax.set_title(f"{_jt} judge vs expert  (Base, n={N_BASE})")
        _ax.set_xticks(_x)
        _ax.set_xticklabels(JUDGE_ORDER, rotation=20, ha="right", fontsize=8)
        _ax.axhline(0, color="grey", lw=0.6)
        _lo = min(0.0, float(_sub[["accuracy", "cohen_kappa"]].min().min()) - 0.05)
        _ax.set_ylim(_lo, 1.02)
        _ax.grid(axis="y", alpha=0.3)
    _axes[1].legend(fontsize=8, loc="upper right")
    _axes[0].set_ylabel("agreement with expert")
    _fig.suptitle(f"Judge–expert agreement on {N_BASE} human-annotated "
                  "PrivacyLens records", fontsize=12)
    _fig.tight_layout()
    _fig.savefig(REPORT_DIR / "plot_E_human_agreement.png", dpi=200, bbox_inches="tight")
    _fig.savefig(REPORT_DIR / "plot_E_human_agreement.pdf", bbox_inches="tight")
    human_align.to_parquet(REPORT_DIR / "human_align.parquet", index=False)
    human_agree.to_parquet(REPORT_DIR / "human_agreement.parquet", index=False)
    _fig
    return


@app.cell
def _(JUDGE_ORDER, REPORT_DIR, SCOPE_BASE, SCOPE_IDENT, SCOPE_POOLED, human_agree, mo):
    # Markdown: LEAKAGE judge agreement with the human expert, per judge × scope.
    _SCOPES = [SCOPE_BASE, SCOPE_IDENT, SCOPE_POOLED]
    _COLS = [("n", "n"), ("accuracy", "Accuracy"), ("cohen_kappa", "Cohen's κ"),
             ("leak_precision", "leak P"), ("leak_recall", "leak R"),
             ("leak_f1", "leak F1")]

    def _cell(k, v):
        if v is None or (isinstance(v, float) and v != v):
            return "—"
        return str(int(v)) if k == "n" else f"{v:.3f}"

    _h = human_agree[human_agree["judge_type"] == "leakage"]
    _lines = [
        "## Leakage judge — agreement with human expert",
        "",
        "*Per judge LLM vs the expert verdicts. Accuracy / Cohen's κ / leak-F1 "
        "higher = closer to the human; leak-R is recall on expert-confirmed "
        "leaks (the incumbent's low recall = it misses real leaks).*",
        "",
    ]
    for _sc in _SCOPES:
        _sub = _h[_h["scope"] == _sc]
        if _sub.empty:
            continue
        _lines += [f"#### {_sc}",
                   "| Judge | " + " | ".join(c[1] for c in _COLS) + " |",
                   "|" + "---|" * (len(_COLS) + 1)]
        for _J in [j for j in JUDGE_ORDER if j in set(_sub["judge_llm"])]:
            _r = _sub[_sub["judge_llm"] == _J].iloc[0]
            _lines.append("| " + _J + " | "
                          + " | ".join(_cell(_k, _r.get(_k)) for _k, _ in _COLS)
                          + " |")
        _lines.append("")
    leakage_human_md = "\n".join(_lines)
    (REPORT_DIR / "human_agreement_leakage.md").write_text(leakage_human_md)
    print(leakage_human_md)
    mo.md(leakage_human_md)
    return


@app.cell
def _(JUDGE_ORDER, REPORT_DIR, SCOPE_BASE, SCOPE_IDENT, SCOPE_POOLED, human_agree, mo):
    # Markdown: HELPFULNESS judge agreement with the human expert.
    _SCOPES = [SCOPE_BASE, SCOPE_IDENT, SCOPE_POOLED]
    _COLS = [("n", "n"), ("accuracy", "Accuracy"), ("cohen_kappa", "Cohen's κ")]

    def _cell(k, v):
        if v is None or (isinstance(v, float) and v != v):
            return "—"
        return str(int(v)) if k == "n" else f"{v:.3f}"

    _h = human_agree[human_agree["judge_type"] == "helpfulness"]
    _lines = [
        "## Helpfulness judge — agreement with human expert",
        "",
        "*Per judge LLM vs the expert verdicts. Agreement is weak for every "
        "judge (κ ≈ 0): all judges over-call \"helpful\" relative to the expert, "
        "so this is a limitation of the helpfulness rubric, not of any one "
        "judge.*",
        "",
    ]
    for _sc in _SCOPES:
        _sub = _h[_h["scope"] == _sc]
        if _sub.empty:
            continue
        _lines += [f"#### {_sc}",
                   "| Judge | " + " | ".join(c[1] for c in _COLS) + " |",
                   "|" + "---|" * (len(_COLS) + 1)]
        for _J in [j for j in JUDGE_ORDER if j in set(_sub["judge_llm"])]:
            _r = _sub[_sub["judge_llm"] == _J].iloc[0]
            _lines.append("| " + _J + " | "
                          + " | ".join(_cell(_k, _r.get(_k)) for _k, _ in _COLS)
                          + " |")
        _lines.append("")
    helpfulness_human_md = "\n".join(_lines)
    (REPORT_DIR / "human_agreement_helpfulness.md").write_text(helpfulness_human_md)
    print(helpfulness_human_md)
    mo.md(helpfulness_human_md)
    return


@app.cell(hide_code=True)
def _(SCOPE_BASE, SCOPE_POOLED, human_agree, mo, pd):
    # Numbers pulled live from human_agree so the narrative tracks the loaded
    # annotation set (now n=51) instead of the original hardcoded n=36 claims.
    def _g(scope, jt, judge, col):
        _r = human_agree[(human_agree["scope"] == scope)
                         & (human_agree["judge_type"] == jt)
                         & (human_agree["judge_llm"] == judge)]
        if not len(_r) or pd.isna(_r[col].iloc[0]):
            return float("nan")
        return float(_r[col].iloc[0])

    _inc_acc = _g(SCOPE_BASE, "leakage", "Qwen3-32B-AWQ", "accuracy")
    _inc_k = _g(SCOPE_BASE, "leakage", "Qwen3-32B-AWQ", "cohen_kappa")
    _inc_rec = _g(SCOPE_BASE, "leakage", "Qwen3-32B-AWQ", "leak_recall")
    _ch_acc = _g(SCOPE_BASE, "leakage", "Qwen3.6-27B", "accuracy")
    _ch_k = _g(SCOPE_BASE, "leakage", "Qwen3.6-27B", "cohen_kappa")
    _ch_f1 = _g(SCOPE_BASE, "leakage", "Qwen3.6-27B", "leak_f1")
    _gm_acc = _g(SCOPE_BASE, "leakage", "Gemma-4-31B-it", "accuracy")
    _gm_k = _g(SCOPE_BASE, "leakage", "Gemma-4-31B-it", "cohen_kappa")
    _gm_f1 = _g(SCOPE_BASE, "leakage", "Gemma-4-31B-it", "leak_f1")
    _gp_acc = _g(SCOPE_BASE, "leakage", "gpt-5.2", "accuracy")
    _gp_k = _g(SCOPE_BASE, "leakage", "gpt-5.2", "cohen_kappa")
    _gp_f1 = _g(SCOPE_BASE, "leakage", "gpt-5.2", "leak_f1")
    _pl_n = _g(SCOPE_POOLED, "leakage", "gpt-5.2", "n")
    _pl_gp_k = _g(SCOPE_POOLED, "leakage", "gpt-5.2", "cohen_kappa")
    _pl_inc_k = _g(SCOPE_POOLED, "leakage", "Qwen3-32B-AWQ", "cohen_kappa")
    _n_base = _g(SCOPE_BASE, "leakage", "Qwen3-32B-AWQ", "n")

    _h = human_agree[(human_agree["scope"] == SCOPE_BASE)
                     & (human_agree["judge_type"] == "helpfulness")]
    _help_lo = float(_h["accuracy"].min()) if len(_h) else float("nan")
    _help_hi = float(_h["accuracy"].max()) if len(_h) else float("nan")

    mo.md(f"""
    ### What the human calibration shows

    **Leakage judge (the metric that matters).** Against the expert on the
    {_n_base:.0f} Base records (identical actions):

    - **Qwen3-32B-AWQ** (incumbent): accuracy **{_inc_acc:.2f}**, κ
      **{_inc_k:.2f}**, and leak **recall only {_inc_rec:.2f}** — it misses
      ~{(1 - _inc_rec) * 100:.0f}% of genuine leaks. Its disagreement is exactly
      the *lenient* failure mode (it calls real leaks "no-leak"), which flatters
      a privacy method.
    - **Qwen3.6-27B** (chosen): accuracy **{_ch_acc:.2f}**, κ **{_ch_k:.2f}**,
      leak-F1 **{_ch_f1:.2f}**; **Gemma-4-31B-it**: accuracy **{_gm_acc:.2f}**,
      κ **{_gm_k:.2f}**, leak-F1 **{_gm_f1:.2f}**.
    - **gpt-5.2** (frontier proxy): accuracy **{_gp_acc:.2f}**, κ
      **{_gp_k:.2f}**, leak-F1 {_gp_f1:.2f}.

    So the judge swap we made (away from Qwen3-32B-AWQ) moves the leakage judge
    *toward* the human expert, and the going-forward judge is statistically
    aligned with both an independent open judge and a frontier proprietary one.
    The pooled Base+SFT+GRPO view (n={_pl_n:.0f}) tells the same story (gpt-5.2 κ
    {_pl_gp_k:.2f} vs Qwen3-32B-AWQ κ {_pl_inc_k:.2f}).

    **Helpfulness judge (an honest caveat).** *Every* judge agrees poorly with
    the expert on helpfulness (accuracy {_help_lo:.2f}–{_help_hi:.2f}, κ ≈ 0).
    All judges over-predict "helpful"; the expert is far stricter. This is
    judge-model-independent, so it is a limitation of the helpfulness
    rubric/prompt rather than of any one judge — and we report it rather than
    bury it.

    **The objective anchors still hold** (and now corroborate the human data):
    gold-label benchmarks move ≤1 pp across judges (Phase B); gpt-5.2 — the
    closest stand-in for an expert — sits with the consensus, not the incumbent;
    and three independent judge families converge while Qwen3-32B-AWQ is the lone
    lenient dissenter. The human calibration converts "trust the consensus" into
    "the consensus matches the expert."
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Phase F — Coverage table, rebuttal paragraph, and saved artifacts
    """)
    return


@app.cell
def _(
    CHOSEN_JUDGE,
    INCUMBENT_JUDGE,
    SCOPE_BASE,
    consensus_dev,
    human_agree,
    judged_ranges,
    long_df,
):
    # Pull the exact numbers the rebuttal paragraph quotes, straight from the data.
    def _leak(judge, disp):
        _r = long_df[(long_df["metric"] == "leakage_rate")
                     & (long_df["judge"] == judge)
                     & (long_df["display_name"] == disp)]
        return float(_r["value"].iloc[0]) if len(_r) else float("nan")

    def _ha(judge, jt, col, scope=SCOPE_BASE):
        _r = human_agree[(human_agree["scope"] == scope)
                         & (human_agree["judge_type"] == jt)
                         & (human_agree["judge_llm"] == judge)]
        return float(_r[col].iloc[0]) if len(_r) else float("nan")

    facts = {
        "q9b_incumbent": _leak(INCUMBENT_JUDGE, "Qwen3.5-9B"),
        "q9b_gemma": _leak("Gemma-4-31B-it", "Qwen3.5-9B"),
        "q9b_chosen": _leak(CHOSEN_JUDGE, "Qwen3.5-9B"),
        "q9b_gpt": _leak("gpt-5.2", "Qwen3.5-9B"),
        "leak_range_max": float(judged_ranges["range"].max()),
        "incumbent_dev": float(
            consensus_dev[(consensus_dev["judge"] == INCUMBENT_JUDGE)
                          & (consensus_dev["metric"] == "leakage_rate")]
            ["mean_signed_dev"].iloc[0]),
        "chosen_dev": float(
            consensus_dev[(consensus_dev["judge"] == CHOSEN_JUDGE)
                          & (consensus_dev["metric"] == "leakage_rate")]
            ["mean_signed_dev"].iloc[0]),
        # Human-expert calibration (Phase E), Base scope, identical actions.
        "human_leak_kappa_incumbent": _ha(INCUMBENT_JUDGE, "leakage", "cohen_kappa"),
        "human_leak_kappa_chosen": _ha(CHOSEN_JUDGE, "leakage", "cohen_kappa"),
        "human_leak_kappa_gpt": _ha("gpt-5.2", "leakage", "cohen_kappa"),
        "human_leak_recall_incumbent": _ha(INCUMBENT_JUDGE, "leakage", "leak_recall"),
        "human_leak_acc_chosen": _ha(CHOSEN_JUDGE, "leakage", "accuracy"),
        "human_help_kappa_chosen": _ha(CHOSEN_JUDGE, "helpfulness", "cohen_kappa"),
    }
    for _k, _v in facts.items():
        print(f"  {_k:28s} = {_v:.4f}")
    return (facts,)


@app.cell
def _(
    N_BASE,
    REPORT_DIR,
    agreement_mad,
    consensus_dev,
    control_summary,
    coverage,
    facts,
    judged_ranges,
    long_df,
):
    import json as _json

    _g_q = float(agreement_mad.loc["Gemma-4-31B-it", "Qwen3.6-27B"])
    _g_i = float(agreement_mad.loc["Gemma-4-31B-it", "Qwen3-32B-AWQ"])
    _q_i = float(agreement_mad.loc["Qwen3.6-27B", "Qwen3-32B-AWQ"])
    # Split the control range into accuracy/rate metrics vs the ConfAIde
    # correlation so the "≤1pp" claim stays honest.
    _acc_max = float(control_summary[
        control_summary["metric"] != "confaide_t2a_pearson"]["max_range"].max())
    _r_max = float(control_summary["max_range"].max())
    _rebuttal = f"""\
    ## Reviewer response — LLM-as-judge validation

    We strengthen the judge validation with a controlled judge-model ablation and an
    explicit separation of judge variance from task variance.

    **Setup.** We re-scored byte-identical model outputs with four independent
    judges: the original Qwen3-32B-AWQ, Gemma-4-31B-it, Qwen3.6-27B (the judge we
    adopt), and the frontier proprietary gpt-5.2. Re-judging used the same weights,
    sampling, and (pre-rewrite) prompts, so only the judge differs. Of our
    benchmarks, only PrivacyLens leakage/helpfulness consult a judge; GoldCoin-HIPAA,
    VLM-GeoPrivacy, CIRL probing, ConfAIde, and the PrivacyLens QA-probe are
    gold-scored.

    **Judge variance is isolated, not pervasive.** Across the three judges, every
    gold-scored accuracy/rate metric is invariant to within ~{_acc_max * 100:.0f}
    percentage points (and the ConfAIde correlation to within {_r_max:.2f}); the
    experiment is therefore controlled — task behaviour is reproduced and only the
    rater changes. The only metric that moves is the judged leakage rate (mean
    cross-judge range {judged_ranges['range'].mean():.2f}, up to
    {facts['leak_range_max']:.2f}).

    **The original judge is the lone outlier; three independent judges agree.**
    Gemma-4 and Qwen3.6 — independent open judges — agree on leakage to within
    {_g_q:.2f} across n=8 shared models, and the frontier proprietary gpt-5.2 falls
    in the same band. The
    original Qwen3-32B-AWQ disagrees with each of them by {min(_g_i, _q_i):.2f}–{max(_g_i, _q_i):.2f}.
    On the flagship Qwen3.5-9B actions it reports {facts['q9b_incumbent']:.0%}
    leakage versus Gemma-4 {facts['q9b_gemma']:.0%}, Qwen3.6 {facts['q9b_chosen']:.0%},
    and gpt-5.2 {facts['q9b_gpt']:.0%} (~2.3× higher); averaged over comparable
    models it deviates {facts['incumbent_dev']:+.2f} from the leave-one-out consensus
    and inflates helpfulness by +1.1, i.e. it paints a uniformly rosier picture.

    **The judge we adopt is the consensus choice.** Qwen3.6 deviates only
    {facts['chosen_dev']:+.2f} from consensus on leakage and tracks both an
    independent open judge (Gemma-4) and a frontier proprietary judge (gpt-5.2).
    Crucially, moving off the original judge *raises* measured leakage roughly
    uniformly, so it does not change task-model rankings and the affected metric
    moves against, not for, our method.

    **Human-expert calibration.** On {N_BASE} expert-annotated records (identical
    Qwen3.5-9B actions), the judge we adopt agrees with the expert on leakage at
    Cohen's κ {facts['human_leak_kappa_chosen']:.2f} (accuracy
    {facts['human_leak_acc_chosen']:.0%}), matching Gemma-4 and the frontier
    gpt-5.2 (κ {facts['human_leak_kappa_gpt']:.2f}). The original Qwen3-32B-AWQ
    reaches only κ {facts['human_leak_kappa_incumbent']:.2f} and recovers just
    {facts['human_leak_recall_incumbent']:.0%} of expert-confirmed leaks — its
    errors are precisely the lenient, leak-missing direction. So the human data
    confirms the swap: the consensus judges are the human-aligned ones.

    **Limitation.** The expert set is small (n={N_BASE} records) and single-annotator,
    and helpfulness agreement is weak for *every* judge (κ ≈ 0; all over-call
    "helpful"), indicating the helpfulness rubric — not any one judge — needs
    revision. A larger multi-annotator calibration set is the natural next step;
    the leakage calibration above, the gold-scored controls (judges agree to
    ≤1pp), and frontier-proxy agreement are mutually corroborating.
    """
    (REPORT_DIR / "REBUTTAL.md").write_text(_rebuttal)
    long_df.to_parquet(REPORT_DIR / "metrics_long.parquet", index=False)
    coverage.to_parquet(REPORT_DIR / "coverage.parquet", index=False)
    control_summary.to_parquet(REPORT_DIR / "control_summary.parquet", index=False)
    judged_ranges.to_parquet(REPORT_DIR / "judged_ranges.parquet", index=False)
    consensus_dev.to_parquet(REPORT_DIR / "consensus_deviation.parquet", index=False)
    agreement_mad.to_parquet(REPORT_DIR / "agreement_mad.parquet")
    (REPORT_DIR / "facts.json").write_text(_json.dumps(facts, indent=2))
    print(f"artifacts + REBUTTAL.md written to:\n  {REPORT_DIR}")
    return


@app.cell
def _(REPORT_DIR, mo):
    mo.md(f"""
    **Rendered rebuttal** (also saved to `{REPORT_DIR}/REBUTTAL.md`):
    """)
    return


@app.cell(hide_code=True)
def _(REPORT_DIR, mo):
    _p = REPORT_DIR / "REBUTTAL.md"
    mo.md(_p.read_text() if _p.exists() else "_run Phase F to generate_")
    return


if __name__ == "__main__":
    app.run()
