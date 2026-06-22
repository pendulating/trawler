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
    # Qwen3.5 eval-all sweep — verify, validate, analyze

    **Source:** `multirun/2026-05-12_eval_all/10-55-35/` — 7 jobs, one per model in the Qwen3.5 size/finetune sweep:

    | idx | model |
    |---|---|
    | 0 | `qwen3.5-2b/base` |
    | 1 | `qwen3.5-2b/instruct` |
    | 2 | `qwen3.5-4b/base` |
    | 3 | `qwen3.5-4b/instruct` |
    | 4 | `qwen3.5-9b/base` |
    | 5 | `qwen3.5-9b/instruct` |
    | 6 | `qwen3.5-9b/sft-ci` |

    Each job ran the `all_benchmarks` pipeline: **CIRL-Vignettes**, **ConfAIde** (tiers 2a/2b + four tier-3 sub-tasks), **GoldCoin-HIPAA** (applicability + compliance), **MMLU**, **PrivacyLens** (QA probing + leakage + helpfulness + adjusted leakage), **VLM-GeoPrivacy** (Q1–Q7).

    Three passes:

    1. **Verify** — every benchmark dir has a `pipeline_manifest.json` and the expected `compute_metrics*/metrics.json`.
    2. **Validate** — row counts, parseable rates, PrivacyLens agent-action format, judge defaulting.
    3. **Analyze** — wide leaderboard, headline plots, and the 9B SFT-CI Δ vs base/instruct.
    """)
    return


@app.cell
def _():
    import json
    import sys
    from pathlib import Path
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    SWEEP_ROOT = Path("/share/pierson/matt/UAIR/multirun/2026-05-12_eval_all/10-55-35")

    JOB_TO_MODEL = {
        0: "qwen3.5-2b/base",
        1: "qwen3.5-2b/instruct",
        2: "qwen3.5-4b/base",
        3: "qwen3.5-4b/instruct",
        4: "qwen3.5-9b/base",
        5: "qwen3.5-9b/instruct",
        6: "qwen3.5-9b/sft-ci",
    }
    MODEL_ORDER = list(JOB_TO_MODEL.values())

    BENCH_INNER = {
        "cirl_vignettes": "cirl_vignettes",
        "confaide": "confaide",
        "goldcoin": "goldcoin_hipaa",
        "mmlu": "mmlu",
        "privacylens": "privacylens_eval",
        "vlm_geoprivacy": "vlm_geoprivacy_bench",
    }
    EXPECTED_METRICS_DIRS = {
        "cirl_vignettes": ["compute_metrics"],
        "confaide": [
            "compute_metrics_tier2a",
            "compute_metrics_tier2b",
            "compute_metrics_tier3_control",
            "compute_metrics_tier3_free",
            "compute_metrics_tier3_info",
            "compute_metrics_tier3_sharing",
        ],
        "goldcoin": ["compute_metrics_applicability", "compute_metrics_compliance"],
        "mmlu": ["compute_metrics"],
        "privacylens": ["compute_metrics"],
        "vlm_geoprivacy": ["compute_metrics"],
    }

    def model_size(m: str) -> int:
        return int(m.split("-")[1].split("/")[0].rstrip("b"))

    def model_variant(m: str) -> str:
        return m.split("/", 1)[1]

    pd.set_option("display.max_columns", 60)
    pd.set_option("display.float_format", "{:.4f}".format)

    sys.path.insert(0, "/share/pierson/matt/UAIR/notebooks")
    try:
        from font_utils import load_ibm_plex_sans
        load_ibm_plex_sans()
    except Exception as _e:
        print(f"[font_utils] skipped: {_e}")
    return (
        BENCH_INNER,
        EXPECTED_METRICS_DIRS,
        JOB_TO_MODEL,
        MODEL_ORDER,
        Path,
        SWEEP_ROOT,
        json,
        model_size,
        model_variant,
        mticker,
        pd,
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · Verify — every job/benchmark wrote its expected artifacts
    """)
    return


@app.cell
def _(BENCH_INNER, EXPECTED_METRICS_DIRS, JOB_TO_MODEL, SWEEP_ROOT, pd):
    def _verify():
        rows = []
        for job, model in JOB_TO_MODEL.items():
            for bench, expected in EXPECTED_METRICS_DIRS.items():
                base = SWEEP_ROOT / str(job) / bench / BENCH_INNER[bench]
                rec = {
                    "job": job,
                    "model": model,
                    "benchmark": bench,
                    "manifest_present": (base / "pipeline_manifest.json").exists(),
                    "missing_metrics_dirs": [],
                    "missing_metrics_json": [],
                }
                for d in expected:
                    mdir = base / "outputs" / d
                    if not mdir.is_dir():
                        rec["missing_metrics_dirs"].append(d)
                    elif not (mdir / "metrics.json").exists():
                        rec["missing_metrics_json"].append(d)
                rows.append(rec)
        return pd.DataFrame(rows)

    verify_df = _verify()
    verify_issues = verify_df[
        (~verify_df["manifest_present"])
        | (verify_df["missing_metrics_dirs"].map(len) > 0)
        | (verify_df["missing_metrics_json"].map(len) > 0)
    ]
    print(f"Total job×benchmark pairs: {len(verify_df)}")
    print(f"With issues: {len(verify_issues)}")
    verify_issues if len(verify_issues) else "All artifacts present."
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · Validate — sanity, row counts, parse rates, judge defaulting

    Load every `metrics.json` into a long-form table keyed by `(model, benchmark, sub_task)`, and read each `pipeline_manifest.json` to surface the `sanity` blocks emitted by `parse_responses` and the PrivacyLens judges.
    """)
    return


@app.cell
def _(BENCH_INNER, EXPECTED_METRICS_DIRS, JOB_TO_MODEL, SWEEP_ROOT, json, pd):
    def _load_metrics_long():
        records = []
        for job, model in JOB_TO_MODEL.items():
            for bench, subs in EXPECTED_METRICS_DIRS.items():
                for sub in subs:
                    p = SWEEP_ROOT / str(job) / bench / BENCH_INNER[bench] / "outputs" / sub / "metrics.json"
                    with open(p) as f:
                        raw = json.load(f)
                    records.append({
                        "job": job,
                        "model": model,
                        "benchmark": bench,
                        "metrics_dir": sub,
                        "raw": raw,
                    })
        return pd.DataFrame(records)

    metrics_long = _load_metrics_long()
    print(f"Loaded {len(metrics_long)} (job, benchmark, sub) metric blobs.")
    metrics_long.head()
    return (metrics_long,)


@app.cell
def _(BENCH_INNER, JOB_TO_MODEL, SWEEP_ROOT, json, pd):
    def _collect_sanity():
        rows = []
        for job, model in JOB_TO_MODEL.items():
            for bench, inner in BENCH_INNER.items():
                mf = SWEEP_ROOT / str(job) / bench / inner / "pipeline_manifest.json"
                if not mf.exists():
                    continue
                with open(mf) as f:
                    data = json.load(f)
                for node_name, node in data["nodes"].items():
                    meta = node.get("metadata") or {}
                    sanity = meta.get("sanity") or {}
                    for stage_key, sb in sanity.items():
                        rows.append({
                            "job": job,
                            "model": model,
                            "benchmark": bench,
                            "node": node_name,
                            "stage_key": stage_key,
                            "halted": sb.get("halted"),
                            "n_warnings": sb.get("n_warnings"),
                            "n_failures": sb.get("n_failures"),
                            "failure_rows": sb.get("n_failure_rows"),
                            "metrics": sb.get("metrics"),
                        })
        return pd.DataFrame(rows)

    sanity_df = _collect_sanity()
    sanity_halted = sanity_df[sanity_df["halted"] == True]
    sanity_failed = sanity_df[
        (sanity_df["n_failures"].fillna(0) > 0)
        | (sanity_df["failure_rows"].fillna(0) > 0)
    ]
    sanity_warned = sanity_df[sanity_df["n_warnings"].fillna(0) > 0]
    print(
        f"sanity rows: {len(sanity_df)} | halted: {len(sanity_halted)} | "
        f"with failures: {len(sanity_failed)} | with warnings: {len(sanity_warned)}"
    )
    sanity_warned if len(sanity_warned) else "No parse/format sanity warnings across the sweep."
    return (sanity_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2a · Parseable / unparseable rates across every metric blob
    """)
    return


@app.cell
def _(metrics_long, pd):
    def _walk_rates(raw):
        out = []
        def rec(node, path):
            if isinstance(node, dict):
                for k, v in node.items():
                    if k == "metric_provenance":
                        continue
                    if isinstance(v, (int, float)) and (
                        "parseable_rate" in k or "unparseable_rate" in k or "format_rate" in k
                    ):
                        out.append((".".join([*path, k]), float(v)))
                    elif isinstance(v, dict):
                        rec(v, [*path, k])
        rec(raw, [])
        return out

    def _build_rates(df):
        rows = []
        for _, r in df.iterrows():
            for path, val in _walk_rates(r["raw"]):
                rows.append({
                    "model": r["model"],
                    "benchmark": r["benchmark"],
                    "metrics_dir": r["metrics_dir"],
                    "field": path,
                    "value": val,
                })
        return pd.DataFrame(rows)

    rates_df = _build_rates(metrics_long)
    parse_problems = rates_df[
        (rates_df["field"].str.contains("parseable_rate") & (rates_df["value"] < 1.0))
        | (rates_df["field"].str.contains("unparseable_rate") & (rates_df["value"] > 0.0))
        | (rates_df["field"].str.contains("format_rate") & (rates_df["value"] < 0.9))
    ].sort_values("value")
    print(f"Total parseable/format rate rows: {len(rates_df)} | flagged: {len(parse_problems)}")
    parse_problems if len(parse_problems) else "All parse/format rates at ceiling."
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2b · Row counts match dataset sizes

    Every job should evaluate the *same* underlying dataset rows per benchmark — drift in `total` across the 7 models signals a partial run or a dataset variant override.
    """)
    return


@app.cell
def _(metrics_long, pd):
    def _get_totals(raw, bench):
        if bench == "privacylens":
            return {
                "privacylens.qa_probing.total": raw["qa_probing"]["total"],
                "privacylens.leakage.total": raw["leakage"]["total"],
                "privacylens.helpfulness.total": raw["helpfulness"]["total"],
            }
        if bench == "vlm_geoprivacy":
            return {"vlm_geoprivacy.n_samples": raw["n_samples"]}
        return {f"{bench}.total": raw.get("total")}

    def _build_totals(df):
        rows = []
        for _, r in df.iterrows():
            for k, v in _get_totals(r["raw"], r["benchmark"]).items():
                rows.append({
                    "model": r["model"],
                    "benchmark": r["benchmark"],
                    "metrics_dir": r["metrics_dir"],
                    "total_key": k,
                    "total": v,
                })
        return pd.DataFrame(rows)

    totals_df = _build_totals(metrics_long)
    totals_drift = totals_df.groupby(["benchmark", "metrics_dir", "total_key"])["total"].nunique()
    totals_drift = totals_drift[totals_drift > 1]
    print(
        f"row-count entries: {len(totals_df)} | benchmarks with cross-model row drift: {len(totals_drift)}"
    )
    (
        totals_drift
        if len(totals_drift)
        else totals_df.groupby(["benchmark", "metrics_dir", "total_key"])["total"].first().reset_index()
    )
    return (totals_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2c · PrivacyLens judge defaulting

    `leakage.leakage_rate_overall_with_default_zero` and `helpfulness.helpful_rate_overall_with_default_zero` substitute `0` for rows where the judge was skipped (agent-action format failure or missing sensitive info). Surface the `defaulted_rate` from `metric_provenance` so we know how much of the headline number is real vs. imputed.
    """)
    return


@app.cell
def _(metrics_long, pd):
    def _build_pl(df):
        rows = []
        for _, r in df.iterrows():
            if r["benchmark"] != "privacylens":
                continue
            raw = r["raw"]
            prov = raw.get("metric_provenance", {})
            rows.append({
                "model": r["model"],
                "qa_acc": raw["qa_probing"]["accuracy"],
                "agent_format_rate": raw["leakage"]["agent_action_format_rate"],
                "leak_rate_parseable": raw["leakage"]["leakage_rate_among_parseable"],
                "leak_rate_defaulted0": raw["leakage"]["leakage_rate_overall_with_default_zero"],
                "leak_default_rate": prov.get(
                    "leakage.leakage_rate_overall_with_default_zero", {}
                ).get("defaulted_rate"),
                "help_rate_parseable": raw["helpfulness"]["helpful_rate_among_parseable"],
                "help_rate_defaulted0": raw["helpfulness"]["helpful_rate_overall_with_default_zero"],
                "help_default_rate": prov.get(
                    "helpfulness.helpful_rate_overall_with_default_zero", {}
                ).get("defaulted_rate"),
                "adj_leak_rate": raw["adjusted_leakage"]["adjusted_leakage_rate"],
                "adj_leak_denom": raw["adjusted_leakage"]["total_helpful_and_judged"],
            })
        return pd.DataFrame(rows)

    pl_df = _build_pl(metrics_long)
    pl_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · Analyze — wide leaderboard

    One row per model, columns are the headline scalar per (sub-)benchmark. Where a benchmark has multiple natural headlines (PrivacyLens, GoldCoin, ConfAIde) we keep them as separate columns rather than collapsing.
    """)
    return


@app.cell
def _(MODEL_ORDER, metrics_long, pd):
    def _headline(bench, metrics_dir, raw):
        if bench == "cirl_vignettes":
            return [
                ("cirl.accuracy", raw["accuracy"]),
                ("cirl.accuracy_seed", raw["per_level"]["seed"]["accuracy"]),
                ("cirl.accuracy_vignette", raw["per_level"]["vignette"]["accuracy"]),
            ]
        if bench == "confaide":
            tier = metrics_dir.replace("compute_metrics_", "")
            return [(f"confaide.{tier}.pearson_r", raw.get("pearson_r"))]
        if bench == "goldcoin":
            sub = metrics_dir.replace("compute_metrics_", "")
            return [
                (f"goldcoin.{sub}.accuracy", raw["accuracy"]),
                (f"goldcoin.{sub}.macro_f1", raw["macro_f1"]),
            ]
        if bench == "mmlu":
            cats = raw["by_category"]
            return [
                ("mmlu.overall_accuracy", raw["overall_accuracy"]),
                ("mmlu.stem", cats["STEM"]["accuracy"]),
                ("mmlu.humanities", cats["humanities"]["accuracy"]),
                ("mmlu.social_sciences", cats["social_sciences"]["accuracy"]),
                ("mmlu.other", cats["other"]["accuracy"]),
            ]
        if bench == "privacylens":
            qa = raw["qa_probing"]
            leak = raw["leakage"]
            hp = raw["helpfulness"]
            adj = raw["adjusted_leakage"]
            return [
                ("pl.qa_accuracy", qa["accuracy"]),
                ("pl.qa_S", qa["per_axis"]["S"]["accuracy"]),
                ("pl.qa_T", qa["per_axis"]["T"]["accuracy"]),
                ("pl.qa_V", qa["per_axis"]["V"]["accuracy"]),
                ("pl.agent_format_rate", leak["agent_action_format_rate"]),
                ("pl.leak_rate_parseable", leak["leakage_rate_among_parseable"]),
                ("pl.leak_rate_default0", leak["leakage_rate_overall_with_default_zero"]),
                ("pl.help_rate_parseable", hp["helpful_rate_among_parseable"]),
                ("pl.help_rate_default0", hp["helpful_rate_overall_with_default_zero"]),
                ("pl.adj_leak_rate", adj["adjusted_leakage_rate"]),
            ]
        if bench == "vlm_geoprivacy":
            cols = [
                (f"vlm.Q{i}.accuracy", raw["per_question"][f"Q{i}"]["accuracy"])
                for i in range(1, 8)
            ]
            q7 = raw["per_question"]["Q7"]
            cols.append(("vlm.Q7.over_disclosure_rate", q7["over_disclosure_rate"]))
            cols.append(("vlm.Q7.under_disclosure_rate", q7["under_disclosure_rate"]))
            cols.append(("vlm.Q7.mae", q7["mae"]))
            return cols
        return []

    def _build_leaderboard(df):
        flat = []
        for _, r in df.iterrows():
            for col, val in _headline(r["benchmark"], r["metrics_dir"], r["raw"]):
                flat.append({"model": r["model"], "metric": col, "value": val})
        wide = (
            pd.DataFrame(flat)
            .pivot_table(index="model", columns="metric", values="value", aggfunc="first")
        )
        return wide.reindex(MODEL_ORDER)

    leaderboard = _build_leaderboard(metrics_long)
    leaderboard
    return (leaderboard,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3a · Headline plot — primary CI metric per benchmark

    - CIRL: `accuracy`
    - ConfAIde: tier-2a / 2b Pearson r (closer to 1 = better aligned with humans)
    - GoldCoin: applicability + compliance accuracy
    - MMLU: overall accuracy (capability anchor — should *not* regress under SFT-CI)
    - PrivacyLens: QA accuracy, helpfulness-default0, adjusted-leak (lower is better)
    - VLM: mean Q1–Q6 accuracy (Q7 has its own MAE/over-disclosure story below)
    """)
    return


@app.cell
def _(leaderboard, pd):
    def _build_headlines(lb):
        out = pd.DataFrame(index=lb.index)
        out["cirl_accuracy"] = lb["cirl.accuracy"]
        out["confaide_pearson_2a"] = lb.get("confaide.tier2a.pearson_r")
        out["confaide_pearson_2b"] = lb.get("confaide.tier2b.pearson_r")
        out["gc_applicability_acc"] = lb["goldcoin.applicability.accuracy"]
        out["gc_compliance_acc"] = lb["goldcoin.compliance.accuracy"]
        out["mmlu_overall"] = lb["mmlu.overall_accuracy"]
        out["pl_qa_accuracy"] = lb["pl.qa_accuracy"]
        out["pl_help_default0"] = lb["pl.help_rate_default0"]
        out["pl_adj_leak_lower_is_better"] = lb["pl.adj_leak_rate"]
        out["vlm_q1_q6_mean_accuracy"] = lb[[f"vlm.Q{i}.accuracy" for i in range(1, 7)]].mean(axis=1)
        return out

    headlines = _build_headlines(leaderboard)
    headlines
    return (headlines,)


@app.cell
def _(MODEL_ORDER, headlines, model_size, model_variant, mticker, plt):
    def _plot_headlines(hl):
        panels = [
            ("CIRL accuracy", "cirl_accuracy", False),
            ("ConfAIde tier-2a Pearson r", "confaide_pearson_2a", False),
            ("ConfAIde tier-2b Pearson r", "confaide_pearson_2b", False),
            ("GoldCoin applicability acc.", "gc_applicability_acc", False),
            ("GoldCoin compliance acc.", "gc_compliance_acc", False),
            ("MMLU overall acc.", "mmlu_overall", False),
            ("PrivacyLens QA acc.", "pl_qa_accuracy", False),
            ("PrivacyLens helpfulness (default-0)", "pl_help_default0", False),
            ("PrivacyLens adjusted leakage (↓ better)", "pl_adj_leak_lower_is_better", True),
            ("VLM-GeoPrivacy Q1–Q6 mean acc.", "vlm_q1_q6_mean_accuracy", False),
        ]
        fig, axes = plt.subplots(2, 5, figsize=(20, 7.5), sharey=False)
        palette = {"base": "#4c78a8", "instruct": "#f58518", "sft-ci": "#54a24b"}
        x_lookup = {m: i for i, m in enumerate(MODEL_ORDER)}
        variants = [model_variant(m) for m in MODEL_ORDER]

        for ax, (title, col, lower_better) in zip(axes.flat, panels):
            for variant in ("base", "instruct", "sft-ci"):
                xs = [x_lookup[m] for m, v in zip(MODEL_ORDER, variants) if v == variant]
                ys = [hl.loc[m, col] for m, v in zip(MODEL_ORDER, variants) if v == variant]
                ax.scatter(xs, ys, color=palette[variant], s=60, label=variant, zorder=3)
            for sz in (2, 4, 9):
                pair = [m for m in MODEL_ORDER if model_size(m) == sz]
                ax.plot(
                    [x_lookup[m] for m in pair],
                    [hl.loc[m, col] for m in pair],
                    color="#bbbbbb", linewidth=1, zorder=1,
                )
            ax.set_xticks(range(len(MODEL_ORDER)))
            ax.set_xticklabels(
                [m.split("/")[-1] + f"\n({model_size(m)}B)" for m in MODEL_ORDER],
                rotation=45, ha="right", fontsize=8,
            )
            ax.set_title(title, fontsize=10)
            ax.grid(True, axis="y", alpha=0.25)
            if not lower_better:
                ax.set_ylim(bottom=min(0, ax.get_ylim()[0]))
            if "Pearson" not in title and "leakage" not in title.lower():
                ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

        axes.flat[0].legend(loc="lower right", fontsize=8, frameon=False)
        fig.suptitle("Qwen3.5 sweep — primary CI metrics (2026-05-12 eval_all)", y=1.02, fontsize=13)
        fig.tight_layout()
        return fig

    _plot_headlines(headlines)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3b · SFT-CI delta vs 9B anchors

    How much does `qwen3.5-9b/sft-ci` move each metric vs. its `base` and `instruct` siblings? Positive Δ = sft-ci is higher; for `pl_adj_leak_lower_is_better` lower is better, so a negative Δ is the win.
    """)
    return


@app.cell
def _(headlines, pd):
    def _delta(hl):
        anchors = hl.loc[["qwen3.5-9b/base", "qwen3.5-9b/instruct", "qwen3.5-9b/sft-ci"]]
        return pd.DataFrame({
            "sft_ci - base": anchors.loc["qwen3.5-9b/sft-ci"] - anchors.loc["qwen3.5-9b/base"],
            "sft_ci - instruct": anchors.loc["qwen3.5-9b/sft-ci"] - anchors.loc["qwen3.5-9b/instruct"],
            "base value": anchors.loc["qwen3.5-9b/base"],
            "instruct value": anchors.loc["qwen3.5-9b/instruct"],
            "sft_ci value": anchors.loc["qwen3.5-9b/sft-ci"],
        })

    sftci_delta = _delta(headlines)
    sftci_delta
    return (sftci_delta,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3c · VLM Q7 — disclosure error decomposition

    Q7 of VLM-GeoPrivacy asks the model to choose how *finely* to disclose a coordinate. The headline `accuracy` for Q7 ignores direction; more diagnostic are `over_disclosure_rate` (model leaks finer than ground truth — privacy failure) and `under_disclosure_rate` (overcautious).
    """)
    return


@app.cell
def _(MODEL_ORDER, leaderboard, plt):
    def _plot_q7(lb):
        fig, ax = plt.subplots(figsize=(9, 4.5))
        xs = range(len(MODEL_ORDER))
        over = [lb.loc[m, "vlm.Q7.over_disclosure_rate"] for m in MODEL_ORDER]
        under = [lb.loc[m, "vlm.Q7.under_disclosure_rate"] for m in MODEL_ORDER]
        ax.bar([x - 0.2 for x in xs], over, width=0.4,
               label="over-disclosure (privacy failure)", color="#e45756")
        ax.bar([x + 0.2 for x in xs], under, width=0.4,
               label="under-disclosure (overcautious)", color="#72b7b2")
        ax.set_xticks(list(xs))
        ax.set_xticklabels([m.split("/")[-1] for m in MODEL_ORDER], rotation=30, ha="right")
        ax.set_ylabel("rate")
        ax.set_title("VLM-GeoPrivacy Q7 — over- vs under-disclosure by model")
        ax.legend(frameon=False)
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        return fig

    _plot_q7(leaderboard)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3d · PrivacyLens — defaulting-aware leakage view

    Plot leakage rate two ways: judge-parseable rows only, and overall with skip→0. The gap = how much of the headline number depends on judge-failure imputation.
    """)
    return


@app.cell
def _(MODEL_ORDER, leaderboard, plt):
    def _plot_pl(lb):
        fig, ax = plt.subplots(figsize=(9, 4.5))
        xs = list(range(len(MODEL_ORDER)))
        leak_p = [lb.loc[m, "pl.leak_rate_parseable"] for m in MODEL_ORDER]
        leak_d = [lb.loc[m, "pl.leak_rate_default0"] for m in MODEL_ORDER]
        fmt = [lb.loc[m, "pl.agent_format_rate"] for m in MODEL_ORDER]
        ax.plot(xs, leak_p, marker="o", label="leak rate (judge-parseable)", color="#4c78a8")
        ax.plot(xs, leak_d, marker="s", label="leak rate (overall, skip→0)", color="#f58518")
        ax.plot(xs, fmt, marker="^", linestyle=":", label="agent-action format rate", color="#54a24b")
        ax.set_xticks(xs)
        ax.set_xticklabels([m.split("/")[-1] for m in MODEL_ORDER], rotation=30, ha="right")
        ax.set_ylabel("rate")
        ax.set_ylim(0, 1.02)
        ax.set_title("PrivacyLens — leakage views + format adherence")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        return fig

    _plot_pl(leaderboard)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3e · MMLU capability anchor

    MMLU is the *non-CI* anchor: SFT-CI should not regress on it relative to the 9B base/instruct. Show overall + per-category.
    """)
    return


@app.cell
def _(MODEL_ORDER, leaderboard, plt):
    def _plot_mmlu(lb):
        cats = [
            ("overall", "mmlu.overall_accuracy"),
            ("STEM", "mmlu.stem"),
            ("humanities", "mmlu.humanities"),
            ("social_sciences", "mmlu.social_sciences"),
            ("other", "mmlu.other"),
        ]
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for label, col in cats:
            ax.plot(
                range(len(MODEL_ORDER)),
                [lb.loc[m, col] for m in MODEL_ORDER],
                marker="o", label=label,
            )
        ax.set_xticks(range(len(MODEL_ORDER)))
        ax.set_xticklabels([m.split("/")[-1] for m in MODEL_ORDER], rotation=30, ha="right")
        ax.set_ylabel("accuracy")
        ax.set_title("MMLU accuracy by category (capability anchor)")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        return fig

    _plot_mmlu(leaderboard)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · Save consolidated tables to disk

    Drop the long-form sanity table, the headline leaderboard, and the SFT-CI Δ into a `tables/` sibling so downstream notebooks and the paper-table builder can consume them without re-walking the sweep.
    """)
    return


@app.cell
def _(Path, headlines, leaderboard, sanity_df, sftci_delta, totals_df):
    def _save():
        out = Path("/share/pierson/matt/UAIR/notebooks/normative-simulacra/tables")
        out.mkdir(parents=True, exist_ok=True)
        leaderboard.to_csv(out / "qwen35_2026_05_12_leaderboard.csv")
        headlines.to_csv(out / "qwen35_2026_05_12_headlines.csv")
        sftci_delta.to_csv(out / "qwen35_2026_05_12_sftci_delta.csv")
        sanity_df.to_parquet(out / "qwen35_2026_05_12_sanity.parquet")
        totals_df.to_parquet(out / "qwen35_2026_05_12_totals.parquet")
        return sorted(p.name for p in out.iterdir())

    print("wrote:", _save())
    return


if __name__ == "__main__":
    app.run()
