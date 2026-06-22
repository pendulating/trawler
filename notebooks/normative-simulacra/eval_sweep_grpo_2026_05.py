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
    # COLM (λ × ρ) GRPO sweeps — benchmark evaluation analysis

    Companion to `all_sweeps_grpo_2026_05.py` (training-side). Where that notebook
    looks at training dynamics (composite reward, per-component evolution), this
    one analyzes downstream **benchmark** performance — how each `(λ, ρ)` cell
    fares on the four CI evaluations + MMLU capability anchor:

    | benchmark | what it measures |
    |---|---|
    | **CIRL-Vignettes** | accept/reject of permitted vs forbidden flows |
    | **ConfAIde** (tiers 2a/2b/3) | judgement-correlation with humans + tier-3 sharing/info errors |
    | **GoldCoin-HIPAA** (applicability + compliance) | healthcare CI classification |
    | **PrivacyLens** | QA-probing accuracy + leakage rate + helpfulness + adjusted leakage |
    | **VLM-GeoPrivacy** (Q1–Q7) | visual geolocation CI; Q7 over/under-disclosure |
    | **MMLU** | capability anchor — must not regress under GRPO |

    Eval roots covered (extend `EVAL_ROOTS` as more sweeps get evaluated):

    | sweep eval root | cells | what it covers |
    |---|---|---|
    | `2026-05-19_eval_all/18-01-58` | 10 | λ axis at ρ=0; λ ∈ {0, 0.25, 0.5, 0.75, 1, 1, 1.5, 1.5, 2, 2} (λ ∈ {1, 1.5, 2} re-evaluated for replicate noise) |
    | `2026-05-20_eval_all/17-12-48` | 4 | ρ axis at λ=1.0; ρ ∈ {0.05, 0.10, 0.20, 0.50} |
    | `2026-05-21_eval_all/01-48-02` | 4 | off-axis corners: (λ=0.5, ρ ∈ {0.10, 0.50}), (λ=1.5, ρ ∈ {0.10, 0.50}) |

    Total: 18 evaluated cells → 13 unique `(λ, ρ)` after replicate dedup.

    `(λ, ρ)` is parsed from each cell's `model=` override
    (`grpo-l{LLL}-r{RRR}` → λ=LLL/100, ρ=RRR/100). Replicates of the same `(λ, ρ)`
    are kept as separate observations and aggregated into mean ± std, which
    gives us a free **sampling-noise band** for cells λ ∈ {1, 1.5, 2}, ρ=0.

    Baseline anchor: `qwen3.5-9b/sft-ci` from `2026-05-12_eval_all/10-55-35`. This
    is the SFT checkpoint that fed *every* GRPO cell; any positive Δ (relative to
    this baseline) is attributable to GRPO with the swept reward weights. Loaded
    from `tables/qwen35_2026_05_12_leaderboard.csv` if present.
    """)
    return


@app.cell
def _():
    import json
    import re
    import sys
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from matplotlib.colors import Normalize

    PROJECT_ROOT = Path("/share/pierson/matt/UAIR")
    NB_DIR = PROJECT_ROOT / "notebooks/normative-simulacra"
    TABLES_DIR = NB_DIR / "tables"

    # Each entry: (multirun root, sweep_label, sweep_kind). The two axes feeding
    # the comparative view; off-axis evals slot in here once they finish.
    EVAL_ROOTS = [
        (
            PROJECT_ROOT / "multirun/2026-05-19_eval_all/18-01-58",
            "lambda-axis-eval (05-19)",
            "lambda",
        ),
        (
            PROJECT_ROOT / "multirun/2026-05-20_eval_all/17-12-48",
            "ratio-axis-eval (05-20)",
            "ratio",
        ),
        (
            PROJECT_ROOT / "multirun/2026-05-21_eval_all/01-48-02",
            "offaxis-eval (05-21)",
            "offaxis",
        ),
    ]
    SWEEP_LABEL_ORDER = [label for _, label, _ in EVAL_ROOTS]
    SWEEP_KIND_ORDER = ["lambda", "ratio", "offaxis"]

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

    BASELINE_LEADERBOARD = TABLES_DIR / "qwen35_2026_05_12_leaderboard.csv"
    BASELINE_MODEL_KEY = "qwen3.5-9b/sft-ci"
    # The training-side cross-reference table — produced by all_sweeps_grpo_2026_05.py.
    TRAIN_CANONICAL_CSV = TABLES_DIR / "all_sweeps_grpo_2026_05_runs_canonical.csv"
    TRAIN_FINAL_COMPONENTS_CSV = TABLES_DIR / "all_sweeps_grpo_2026_05_final_components.csv"

    def parse_grpo_lr(model_str: str):
        """`qwen3.5-9b/grpo-l050-r010` → (0.5, 0.1). Returns (None, None) if it
        isn't one of our grpo cells."""
        m = re.search(r"grpo-l(\d{3})-r(\d{3})", model_str)
        if not m:
            return None, None
        return int(m.group(1)) / 100.0, int(m.group(2)) / 100.0

    pd.set_option("display.max_columns", 80)
    pd.set_option("display.float_format", "{:.4f}".format)

    sys.path.insert(0, str(NB_DIR.parent))
    try:
        from font_utils import load_ibm_plex_sans

        load_ibm_plex_sans()
    except Exception as _e:
        print(f"[font_utils] skipped: {_e}")
    return (
        BASELINE_LEADERBOARD,
        BASELINE_MODEL_KEY,
        BENCH_INNER,
        EVAL_ROOTS,
        EXPECTED_METRICS_DIRS,
        Normalize,
        Path,
        TABLES_DIR,
        TRAIN_CANONICAL_CSV,
        TRAIN_FINAL_COMPONENTS_CSV,
        json,
        mticker,
        np,
        parse_grpo_lr,
        pd,
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · Discover evaluated cells

    Walk every cell under every `EVAL_ROOTS` entry, parse the `model=` override,
    extract `(λ, ρ)`, and tag with its sweep label. Replicate evals of the same
    `(λ, ρ)` are kept as separate rows — see §3 for how they're aggregated.
    """)
    return


@app.cell
def _(EVAL_ROOTS, parse_grpo_lr, pd):
    def _scan_cells():
        rows = []
        for root, label, kind in EVAL_ROOTS:
            if not root.exists():
                continue
            for cell_dir in sorted(root.iterdir()):
                if not cell_dir.is_dir() or not cell_dir.name.isdigit():
                    continue
                ov_path = cell_dir / ".hydra" / "overrides.yaml"
                if not ov_path.exists():
                    continue
                with open(ov_path) as f:
                    overrides = [line.strip() for line in f if line.strip()]
                model_override = next(
                    (
                        o.split("=", 1)[1].strip()
                        for o in overrides
                        if o.lstrip("- ").startswith("model=")
                    ),
                    None,
                )
                if not model_override:
                    continue
                lam, rho = parse_grpo_lr(model_override)
                if lam is None or rho is None:
                    # Not a grpo cell (e.g. baseline) — skip.
                    continue
                rows.append(
                    {
                        "sweep_label": label,
                        "sweep_kind": kind,
                        "eval_root": str(root),
                        "cell_idx": int(cell_dir.name),
                        "cell_dir": str(cell_dir),
                        "model": model_override,
                        "lambda": lam,
                        "ratio": rho,
                    }
                )
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values(["sweep_label", "lambda", "ratio", "cell_idx"]).reset_index(
                drop=True
            )
        return df

    eval_cells = _scan_cells()
    print(
        f"Discovered {len(eval_cells)} evaluated cells across "
        f"{eval_cells['sweep_label'].nunique() if len(eval_cells) else 0} eval roots"
    )
    eval_cells
    return (eval_cells,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · Verify — every cell wrote all expected metric files

    Per-cell completeness check across the 6 benchmarks and their sub-tasks (12
    metric files per fully-complete cell when MMLU is included, 11 otherwise).
    Any flagged row means the eval halted partway and that cell's metrics are
    incomplete — its leaderboard entries should be treated as unreliable.
    """)
    return


@app.cell
def _(BENCH_INNER, EXPECTED_METRICS_DIRS, Path, eval_cells, pd):
    def _verify(cells):
        rows = []
        for _, c in cells.iterrows():
            base_cell = Path(c["cell_dir"])
            for bench, expected in EXPECTED_METRICS_DIRS.items():
                base = base_cell / bench / BENCH_INNER[bench]
                rec = {
                    "sweep_label": c["sweep_label"],
                    "lambda": c["lambda"],
                    "ratio": c["ratio"],
                    "cell_idx": c["cell_idx"],
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

    verify_df = _verify(eval_cells)
    verify_issues = verify_df[
        (~verify_df["manifest_present"])
        | (verify_df["missing_metrics_dirs"].map(len) > 0)
        | (verify_df["missing_metrics_json"].map(len) > 0)
    ]
    print(
        f"Total cell × benchmark pairs: {len(verify_df)}  |  with issues: {len(verify_issues)}"
    )
    verify_issues if len(verify_issues) else "All artifacts present."
    return (verify_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · Load all `metrics.json` blobs

    One row per `(cell, benchmark, metrics_dir)`; raw blob kept as a dict for
    downstream extraction. We tolerate missing files (MMLU isn't in every eval
    root) and skip them silently — the §2 verifier already surfaced anything
    surprising.
    """)
    return


@app.cell
def _(BENCH_INNER, EXPECTED_METRICS_DIRS, Path, eval_cells, json, pd):
    def _load_metrics(cells):
        records = []
        for _, c in cells.iterrows():
            base_cell = Path(c["cell_dir"])
            for bench, subs in EXPECTED_METRICS_DIRS.items():
                for sub in subs:
                    p = (
                        base_cell
                        / bench
                        / BENCH_INNER[bench]
                        / "outputs"
                        / sub
                        / "metrics.json"
                    )
                    if not p.exists():
                        continue
                    with open(p) as f:
                        raw = json.load(f)
                    records.append(
                        {
                            "sweep_label": c["sweep_label"],
                            "sweep_kind": c["sweep_kind"],
                            "lambda": c["lambda"],
                            "ratio": c["ratio"],
                            "cell_idx": c["cell_idx"],
                            "model": c["model"],
                            "benchmark": bench,
                            "metrics_dir": sub,
                            "raw": raw,
                        }
                    )
        return pd.DataFrame(records)

    metrics_long = _load_metrics(eval_cells)
    print(f"Loaded {len(metrics_long)} (cell, benchmark, sub) metric blobs.")
    metrics_long.head()
    return (metrics_long,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · Sanity — parse rates, judge defaulting, total drift

    Pulls from each cell's `pipeline_manifest.json` (parse_responses /
    leakage_judge / helpfulness_judge sanity blocks). Any halted/failed stage or
    sub-ceiling parseable rate gets surfaced. Mirrors the validate pass from
    the qwen3.5 eval notebook.
    """)
    return


@app.cell
def _(BENCH_INNER, Path, eval_cells, json, pd):
    def _collect_sanity(cells):
        rows = []
        for _, c in cells.iterrows():
            base_cell = Path(c["cell_dir"])
            for bench, inner in BENCH_INNER.items():
                mf = base_cell / bench / inner / "pipeline_manifest.json"
                if not mf.exists():
                    continue
                with open(mf) as f:
                    data = json.load(f)
                for node_name, node in (data.get("nodes") or {}).items():
                    meta = node.get("metadata") or {}
                    sanity = meta.get("sanity") or {}
                    for stage_key, sb in sanity.items():
                        rows.append(
                            {
                                "sweep_label": c["sweep_label"],
                                "lambda": c["lambda"],
                                "ratio": c["ratio"],
                                "cell_idx": c["cell_idx"],
                                "benchmark": bench,
                                "node": node_name,
                                "stage_key": stage_key,
                                "halted": sb.get("halted"),
                                "n_warnings": sb.get("n_warnings"),
                                "n_failures": sb.get("n_failures"),
                                "failure_rows": sb.get("n_failure_rows"),
                            }
                        )
        return pd.DataFrame(rows)

    sanity_df = _collect_sanity(eval_cells)
    sanity_halted = sanity_df[sanity_df["halted"] == True] if len(sanity_df) else sanity_df
    sanity_failed = (
        sanity_df[
            (sanity_df["n_failures"].fillna(0) > 0)
            | (sanity_df["failure_rows"].fillna(0) > 0)
        ]
        if len(sanity_df)
        else sanity_df
    )
    sanity_warned = (
        sanity_df[sanity_df["n_warnings"].fillna(0) > 0]
        if len(sanity_df)
        else sanity_df
    )
    print(
        f"sanity rows: {len(sanity_df)} | halted: {len(sanity_halted)} | "
        f"with failures: {len(sanity_failed)} | with warnings: {len(sanity_warned)}"
    )
    sanity_warned if len(sanity_warned) else "No parse/format sanity warnings across the sweep."
    return (sanity_df,)


@app.cell
def _(metrics_long, pd):
    def _build_pl(df):
        rows = []
        for _, r in df.iterrows():
            if r["benchmark"] != "privacylens":
                continue
            raw = r["raw"]
            prov = raw.get("metric_provenance", {})
            rows.append(
                {
                    "sweep_label": r["sweep_label"],
                    "lambda": r["lambda"],
                    "ratio": r["ratio"],
                    "cell_idx": r["cell_idx"],
                    "qa_acc": raw["qa_probing"]["accuracy"],
                    "agent_format_rate": raw["leakage"]["agent_action_format_rate"],
                    "leak_rate_parseable": raw["leakage"][
                        "leakage_rate_among_parseable"
                    ],
                    "leak_rate_default0": raw["leakage"][
                        "leakage_rate_overall_with_default_zero"
                    ],
                    "leak_default_rate": prov.get(
                        "leakage.leakage_rate_overall_with_default_zero", {}
                    ).get("defaulted_rate"),
                    "help_rate_parseable": raw["helpfulness"][
                        "helpful_rate_among_parseable"
                    ],
                    "help_rate_default0": raw["helpfulness"][
                        "helpful_rate_overall_with_default_zero"
                    ],
                    "help_default_rate": prov.get(
                        "helpfulness.helpful_rate_overall_with_default_zero", {}
                    ).get("defaulted_rate"),
                    "adj_leak_rate": raw["adjusted_leakage"]["adjusted_leakage_rate"],
                }
            )
        return pd.DataFrame(rows).sort_values(["lambda", "ratio", "cell_idx"])

    pl_default = _build_pl(metrics_long)
    print("PrivacyLens judge defaulting — gap between *_parseable and *_default0 = imputed share")
    pl_default
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 · Per-cell leaderboard — every replicate as its own row

    Headline scalar per (sub-)benchmark, extracted via the same `_headline`
    function as `eval_sweep_qwen35_2026_05_12.py` so the columns line up with the
    earlier sweep's leaderboard CSV (used as baseline in §7). Index is
    `(sweep_label, λ, ρ, cell_idx)` — duplicates per `(λ, ρ)` show up as
    distinct rows, ready for §6 aggregation.
    """)
    return


@app.cell
def _(metrics_long, pd):
    def _headline(bench, metrics_dir, raw):
        if bench == "cirl_vignettes":
            return [
                ("cirl.accuracy", raw["accuracy"]),
                ("cirl.accuracy_seed", raw["per_level"]["seed"]["accuracy"]),
                ("cirl.accuracy_vignette", raw["per_level"]["vignette"]["accuracy"]),
            ]
        if bench == "confaide":
            tier = metrics_dir.replace("compute_metrics_", "")
            cols = [(f"confaide.{tier}.pearson_r", raw.get("pearson_r"))]
            # tier-3 sharing / info don't have pearson; expose their error rate.
            if "error_rate_among_parseable" in raw:
                cols.append(
                    (f"confaide.{tier}.error_rate", raw["error_rate_among_parseable"])
                )
            return cols
        if bench == "goldcoin":
            sub = metrics_dir.replace("compute_metrics_", "")
            return [
                (f"goldcoin.{sub}.accuracy", raw["accuracy"]),
                (f"goldcoin.{sub}.macro_f1", raw["macro_f1"]),
            ]
        if bench == "mmlu":
            cats = raw.get("by_category", {})
            return [
                ("mmlu.overall_accuracy", raw.get("overall_accuracy")),
                ("mmlu.stem", cats.get("STEM", {}).get("accuracy")),
                ("mmlu.humanities", cats.get("humanities", {}).get("accuracy")),
                ("mmlu.social_sciences", cats.get("social_sciences", {}).get("accuracy")),
                ("mmlu.other", cats.get("other", {}).get("accuracy")),
            ]
        if bench == "privacylens":
            qa = raw["qa_probing"]
            leak = raw["leakage"]
            hp = raw["helpfulness"]
            adj = raw.get("adjusted_leakage", {})
            return [
                ("pl.qa_accuracy", qa["accuracy"]),
                ("pl.qa_S", qa["per_axis"]["S"]["accuracy"]),
                ("pl.qa_T", qa["per_axis"]["T"]["accuracy"]),
                ("pl.qa_V", qa["per_axis"]["V"]["accuracy"]),
                ("pl.agent_format_rate", leak["agent_action_format_rate"]),
                ("pl.leak_rate_parseable", leak["leakage_rate_among_parseable"]),
                (
                    "pl.leak_rate_default0",
                    leak["leakage_rate_overall_with_default_zero"],
                ),
                ("pl.help_rate_parseable", hp["helpful_rate_among_parseable"]),
                (
                    "pl.help_rate_default0",
                    hp["helpful_rate_overall_with_default_zero"],
                ),
                ("pl.adj_leak_rate", adj.get("adjusted_leakage_rate")),
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
                flat.append(
                    {
                        "sweep_label": r["sweep_label"],
                        "sweep_kind": r["sweep_kind"],
                        "lambda": r["lambda"],
                        "ratio": r["ratio"],
                        "cell_idx": r["cell_idx"],
                        "metric": col,
                        "value": val,
                    }
                )
        long = pd.DataFrame(flat)
        if long.empty:
            return long, pd.DataFrame()
        wide = long.pivot_table(
            index=["sweep_label", "sweep_kind", "lambda", "ratio", "cell_idx"],
            columns="metric",
            values="value",
            aggfunc="first",
        ).reset_index()
        wide = wide.sort_values(["lambda", "ratio", "cell_idx"]).reset_index(drop=True)
        return long, wide

    leaderboard_long, leaderboard = _build_leaderboard(metrics_long)
    print(
        f"Leaderboard: {len(leaderboard)} rows × {leaderboard.shape[1] - 5 if not leaderboard.empty else 0} metric columns"
    )
    leaderboard
    return leaderboard, leaderboard_long


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6 · Aggregate replicates → per-(λ, ρ) mean ± std

    For `(λ, ρ)` pairs that were evaluated more than once (here: λ ∈ {1, 1.5, 2},
    ρ=0 in the 05-19 root), collapse to mean and std across replicates. This is
    the "canonical" eval-side leaderboard used by every subsequent plot.

    The std column is the **sampling-noise band** — the spread is what you'd see
    from re-evaluating the same checkpoint without changing anything else.
    """)
    return


@app.cell
def _(leaderboard):
    def _aggregate(df):
        if df.empty:
            return df, df
        metric_cols = [
            c
            for c in df.columns
            if c not in {"sweep_label", "sweep_kind", "lambda", "ratio", "cell_idx"}
        ]
        grp = df.groupby(["lambda", "ratio"], as_index=False)
        means = grp[metric_cols].mean(numeric_only=True)
        stds = grp[metric_cols].std(numeric_only=True).fillna(0.0)
        counts = grp.size().rename(columns={"size": "n_replicates"})
        means = means.merge(counts, on=["lambda", "ratio"])
        stds = stds.merge(counts, on=["lambda", "ratio"])
        # attach back the sweep_kind (single value per (λ, ρ) by construction)
        kind_map = (
            df.drop_duplicates(["lambda", "ratio"])[["lambda", "ratio", "sweep_kind"]]
        )
        means = means.merge(kind_map, on=["lambda", "ratio"])
        stds = stds.merge(kind_map, on=["lambda", "ratio"])
        return (
            means.sort_values(["lambda", "ratio"]).reset_index(drop=True),
            stds.sort_values(["lambda", "ratio"]).reset_index(drop=True),
        )

    leaderboard_mean, leaderboard_std = _aggregate(leaderboard)
    print(
        f"Per-(λ, ρ) means: {len(leaderboard_mean)} cells  |  "
        f"replicate counts: {sorted(leaderboard_mean['n_replicates'].unique()) if len(leaderboard_mean) else '—'}"
    )
    leaderboard_mean
    return leaderboard_mean, leaderboard_std


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7 · Baseline anchor — `qwen3.5-9b/sft-ci` from the 2026-05-12 eval

    Every GRPO cell in the sweep starts from the same SFT checkpoint
    (`qwen3.5-9b/sft-ci`). Its scores on the May-12 eval set the "GRPO did
    nothing" floor; positive Δ relative to it is the value GRPO + (λ, ρ) added.

    If the baseline table is missing, this cell prints a hint and downstream Δ
    plots fall back to the lowest-λ cell instead.
    """)
    return


@app.cell
def _(BASELINE_LEADERBOARD, BASELINE_MODEL_KEY, leaderboard_mean, pd):
    def _load_baseline():
        if not BASELINE_LEADERBOARD.exists():
            print(
                f"Baseline table missing: {BASELINE_LEADERBOARD}\n"
                f"  Run `eval_sweep_qwen35_2026_05_12.py` first to populate it, "
                f"or §8 falls back to using λ=0, ρ=0 as the within-sweep baseline."
            )
            return None
        df = pd.read_csv(BASELINE_LEADERBOARD).set_index("model")
        if BASELINE_MODEL_KEY not in df.index:
            print(
                f"Baseline row '{BASELINE_MODEL_KEY}' not in {BASELINE_LEADERBOARD}; "
                f"using λ=0, ρ=0 fallback."
            )
            return None
        # Restrict to columns shared with our eval leaderboard.
        baseline = df.loc[BASELINE_MODEL_KEY]
        # Drop any non-numeric or NaN-only entries.
        baseline = pd.to_numeric(baseline, errors="coerce").dropna()
        return baseline

    baseline = _load_baseline()
    if baseline is not None:
        shared = [c for c in baseline.index if c in leaderboard_mean.columns]
        baseline_aligned = baseline.loc[shared]
        print(f"Baseline anchor has {len(baseline_aligned)} metric columns aligned with the sweep.")
        baseline_aligned
    else:
        baseline_aligned = None
        print("(no baseline loaded)")
    baseline_aligned
    return (baseline_aligned,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8 · Headline metrics vs λ at ρ=0

    The λ-axis story: how each headline CI metric responds to the contrastive
    weight at ρ=0. Markers = per-cell mean across replicates; thin error bars =
    replicate std; horizontal dashed line = SFT-CI baseline (from §7, where
    available); pink shaded band = within-sweep fallback baseline (λ=0 cell).
    """)
    return


@app.cell
def _(baseline_aligned, leaderboard_mean, leaderboard_std, mticker, plt):
    HEADLINE_PANELS = [
        ("CIRL accuracy", "cirl.accuracy", False, True),
        ("ConfAIde tier-2a Pearson r", "confaide.tier2a.pearson_r", False, False),
        ("ConfAIde tier-2b Pearson r", "confaide.tier2b.pearson_r", False, False),
        ("ConfAIde tier-3 sharing error (↓ better)", "confaide.tier3_sharing.error_rate", True, True),
        ("GoldCoin applicability acc.", "goldcoin.applicability.accuracy", False, True),
        ("GoldCoin compliance acc.", "goldcoin.compliance.accuracy", False, True),
        ("PrivacyLens QA accuracy", "pl.qa_accuracy", False, True),
        ("PrivacyLens helpfulness (default-0)", "pl.help_rate_default0", False, True),
        ("PrivacyLens adjusted leakage (↓ better)", "pl.adj_leak_rate", True, True),
        ("MMLU overall accuracy", "mmlu.overall_accuracy", False, True),
        ("VLM Q1–Q6 mean accuracy", "_vlm_q1_q6_mean", False, True),
        ("VLM Q7 over-disclosure (↓ better)", "vlm.Q7.over_disclosure_rate", True, True),
    ]

    def _vlm_q1_q6(df):
        cols = [f"vlm.Q{i}.accuracy" for i in range(1, 7) if f"vlm.Q{i}.accuracy" in df.columns]
        if not cols:
            return None
        return df[cols].mean(axis=1)

    def _plot_lambda(mean, std):
        sub = mean[mean["ratio"] == 0.0].sort_values("lambda").copy()
        sub_std = std[std["ratio"] == 0.0].sort_values("lambda").copy()
        if sub.empty:
            print("No ρ=0 cells in mean table.")
            return None
        sub["_vlm_q1_q6_mean"] = _vlm_q1_q6(sub)
        sub_std["_vlm_q1_q6_mean"] = _vlm_q1_q6(sub_std)
        nrow, ncol = 3, 4
        fig, axes = plt.subplots(nrow, ncol, figsize=(20, 11), sharex=True)
        for ax, (title, col, lower_better, as_pct) in zip(
            axes.flat, HEADLINE_PANELS
        ):
            if col not in sub.columns:
                ax.set_title(f"{title}\n(missing column)", fontsize=9, color="#888")
                ax.axis("off")
                continue
            ys = sub[col].values
            es = sub_std[col].values
            xs = sub["lambda"].values
            ax.errorbar(
                xs,
                ys,
                yerr=es,
                marker="o",
                ms=5,
                lw=1.4,
                capsize=3,
                color="#3a6da9" if not lower_better else "#c44e52",
                ecolor="#999",
                label="per-(λ, ρ=0) mean ± replicate std",
            )
            # SFT baseline horizontal line
            if baseline_aligned is not None and col in baseline_aligned.index:
                bv = baseline_aligned.loc[col]
                ax.axhline(
                    bv,
                    linestyle="--",
                    color="#555",
                    lw=1.0,
                    label=f"SFT-CI baseline = {bv:.3f}",
                )
            # Within-sweep fallback: λ=0 cell
            if (sub["lambda"] == 0.0).any():
                fv = float(sub[sub["lambda"] == 0.0][col].iloc[0])
                ax.axhspan(
                    fv - 0.002,
                    fv + 0.002,
                    color="#f7d7d7",
                    alpha=0.5,
                    label="within-sweep λ=0 baseline",
                )
            ax.set_title(title, fontsize=9)
            ax.grid(True, axis="y", alpha=0.25)
            ax.set_xlabel("contrastive λ", fontsize=8)
            if as_pct:
                ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        # Combined legend on the first panel.
        handles, labels = axes.flat[0].get_legend_handles_labels()
        seen = set()
        legend_pairs = []
        for h, l in zip(handles, labels):
            if l not in seen:
                seen.add(l)
                legend_pairs.append((h, l))
        if legend_pairs:
            axes.flat[0].legend(
                [h for h, _ in legend_pairs],
                [l for _, l in legend_pairs],
                fontsize=7,
                frameon=False,
                loc="lower right",
            )
        fig.suptitle(
            "λ-axis benchmark headlines at ρ=0 (canonical GRPO cells, replicate-aggregated)",
            y=1.005,
            fontsize=13,
        )
        fig.tight_layout()
        return fig

    _plot_lambda(leaderboard_mean, leaderboard_std)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9 · ρ axis at λ=1.0

    The ρ-axis story (`2026-05-20_eval_all/17-12-48`, 4 cells): how each headline
    CI metric responds to the contrastive ratio with λ pinned at 1.0. The ρ=0
    cell from §8 is overlaid as the natural anchor (dotted line). Off-axis
    corners (λ ∈ {0.5, 1.5}, ρ ∈ {0.10, 0.50}) show up in the §10–§11 heatmaps.
    """)
    return


@app.cell
def _(leaderboard_mean, leaderboard_std, mticker, plt):
    def _plot_ratio(mean, std):
        sub = mean[(mean["lambda"] == 1.0) & (mean["ratio"] > 0.0)].sort_values("ratio").copy()
        if sub.empty:
            print("No ρ-axis cells (λ=1.0, ρ>0) in the eval cache yet.")
            return None
        sub_std = std[(std["lambda"] == 1.0) & (std["ratio"] > 0.0)].sort_values("ratio").copy()
        anchor_row = mean[(mean["lambda"] == 1.0) & (mean["ratio"] == 0.0)]
        panels = [
            ("CIRL accuracy", "cirl.accuracy"),
            ("GoldCoin applicability acc.", "goldcoin.applicability.accuracy"),
            ("GoldCoin compliance acc.", "goldcoin.compliance.accuracy"),
            ("PrivacyLens QA accuracy", "pl.qa_accuracy"),
            ("PrivacyLens adjusted leakage (↓)", "pl.adj_leak_rate"),
            ("VLM Q7 over-disclosure (↓)", "vlm.Q7.over_disclosure_rate"),
        ]
        fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
        for ax, (title, col) in zip(axes.flat, panels):
            if col not in sub.columns:
                ax.set_title(f"{title}\n(missing)", fontsize=9, color="#888")
                ax.axis("off")
                continue
            ax.errorbar(
                sub["ratio"],
                sub[col],
                yerr=sub_std[col],
                marker="o",
                ms=5,
                lw=1.4,
                capsize=3,
                color="#7c5295",
                ecolor="#aaa",
                label="ρ-axis (λ=1.0)",
            )
            if not anchor_row.empty:
                ax.axhline(
                    float(anchor_row[col].iloc[0]),
                    color="#555",
                    linestyle=":",
                    lw=1.0,
                    label="ρ=0 anchor",
                )
            ax.set_title(title, fontsize=9)
            ax.grid(True, axis="y", alpha=0.25)
            ax.set_xlabel("contrastive ratio ρ", fontsize=8)
            if "leakage" not in title.lower() and "disclosure" not in title.lower():
                ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        handles, labels = axes.flat[0].get_legend_handles_labels()
        if handles:
            axes.flat[0].legend(handles, labels, fontsize=7, frameon=False, loc="lower right")
        fig.suptitle("ρ-axis benchmark headlines at λ=1.0 (4 cells; dotted line = ρ=0 anchor)", y=1.0, fontsize=12)
        fig.tight_layout()
        return fig

    _plot_ratio(leaderboard_mean, leaderboard_std)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10 · (λ, ρ) coverage heatmap

    Where evals exist. As more sweeps finish (off-axis corners,
    rest of the ρ axis), the empty cells fill in. Each cell is annotated with
    the mean CIRL accuracy across replicates if present, "·" otherwise.
    """)
    return


@app.cell
def _(Normalize, leaderboard_mean, np, plt):
    def coverage_heatmap(mean, value="cirl.accuracy", title="CIRL accuracy"):
        if mean.empty or value not in mean.columns:
            print(f"{value} not available.")
            return None
        # Pad the grid with the full λ × ρ universe so missing cells are visible.
        all_lams = sorted({0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0})
        all_rhos = sorted({0.0, 0.05, 0.10, 0.20, 0.50})
        grid = np.full((len(all_lams), len(all_rhos)), np.nan)
        for _, r in mean.iterrows():
            if r["lambda"] in all_lams and r["ratio"] in all_rhos:
                i = all_lams.index(r["lambda"])
                j = all_rhos.index(r["ratio"])
                grid[i, j] = r[value]
        fig, ax = plt.subplots(figsize=(0.9 * len(all_rhos) + 3, 0.55 * len(all_lams) + 2))
        finite = grid[np.isfinite(grid)]
        if finite.size == 0:
            print("nothing to plot.")
            plt.close(fig)
            return None
        norm = Normalize(vmin=finite.min(), vmax=finite.max())
        im = ax.imshow(grid, cmap="viridis", aspect="auto", norm=norm, origin="lower")
        ax.set_xticks(range(len(all_rhos)), [f"{r:g}" for r in all_rhos])
        ax.set_yticks(range(len(all_lams)), [f"{l:g}" for l in all_lams])
        ax.set_xlabel("contrastive ratio ρ")
        ax.set_ylabel("contrastive λ")
        for i in range(len(all_lams)):
            for j in range(len(all_rhos)):
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
                    ax.text(
                        j,
                        i,
                        "·",
                        ha="center",
                        va="center",
                        color="#999",
                        fontsize=12,
                    )
        ax.set_title(f"{title} — eval coverage on the (λ, ρ) grid", fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
        fig.tight_layout()
        return fig

    coverage_heatmap(leaderboard_mean, "cirl.accuracy", "CIRL accuracy")
    return (coverage_heatmap,)


@app.cell
def _(coverage_heatmap, leaderboard_mean):
    coverage_heatmap(
        leaderboard_mean, "pl.qa_accuracy", "PrivacyLens QA accuracy"
    )
    return


@app.cell
def _(coverage_heatmap, leaderboard_mean):
    coverage_heatmap(
        leaderboard_mean, "pl.adj_leak_rate", "PrivacyLens adjusted leakage (↓ better)"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11 · Δ vs SFT-CI baseline (per-metric, per-cell)

    Direct attribution: how much did GRPO with each `(λ, ρ)` move each headline
    metric relative to the SFT-CI checkpoint it started from? Positive Δ = GRPO
    helped (or hurt, for ↓-better metrics). Rows = `(λ, ρ)` cells (sorted by
    sweep then λ then ρ); columns = headline metrics.
    """)
    return


@app.cell
def _(baseline_aligned, leaderboard_mean, np, pd, plt):
    def _delta_table(mean, baseline):
        if mean.empty or baseline is None:
            return None
        shared = [c for c in baseline.index if c in mean.columns]
        if not shared:
            return None
        rows = []
        for _, r in mean.sort_values(["lambda", "ratio"]).iterrows():
            rec = {"lambda": r["lambda"], "ratio": r["ratio"]}
            for c in shared:
                rec[c] = float(r[c]) - float(baseline.loc[c])
            rows.append(rec)
        return pd.DataFrame(rows)

    delta_df = _delta_table(leaderboard_mean, baseline_aligned)

    def _plot_delta_heatmap(delta):
        if delta is None or delta.empty:
            print("Δ table unavailable (baseline missing or no shared columns).")
            return None
        # Focus on the headline subset for legibility.
        focus = [
            "cirl.accuracy",
            "goldcoin.applicability.accuracy",
            "goldcoin.compliance.accuracy",
            "pl.qa_accuracy",
            "pl.help_rate_default0",
            "pl.adj_leak_rate",
            "vlm.Q7.over_disclosure_rate",
            "mmlu.overall_accuracy",
            "confaide.tier2a.pearson_r",
            "confaide.tier2b.pearson_r",
        ]
        focus = [c for c in focus if c in delta.columns]
        if not focus:
            print("No focus metrics in Δ table.")
            return None
        cells = delta[["lambda", "ratio"]].copy()
        cells["label"] = cells.apply(
            lambda r: f"λ={r['lambda']:g}, ρ={r['ratio']:g}", axis=1
        )
        mat = delta[focus].values
        vmax = float(np.nanmax(np.abs(mat))) or 0.001
        fig, ax = plt.subplots(figsize=(0.95 * len(focus) + 2, 0.42 * len(cells) + 2))
        im = ax.imshow(mat, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(focus)), focus, rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(len(cells)), cells["label"], fontsize=8)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if np.isfinite(v):
                    ax.text(
                        j,
                        i,
                        f"{v:+.3f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="black" if abs(v) < vmax * 0.5 else "white",
                    )
        ax.set_title(
            "Δ vs SFT-CI baseline (cell − baseline). Red = below baseline, blue = above.",
            fontsize=10,
        )
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.04)
        fig.tight_layout()
        return fig

    _plot_delta_heatmap(delta_df)
    return (delta_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 12 · Cross-reference — does final training reward predict eval scores?

    Joins the per-cell eval leaderboard with the canonical training-side table
    from `all_sweeps_grpo_2026_05.py` (final composite reward + final
    `r_ground`). If those training-side signals track eval performance, the
    composite reward design is calibrated; if they decouple, the reward is
    capturing the wrong axis. Scatter is per `(λ, ρ)` cell, coloured by λ.
    """)
    return


@app.cell
def _(
    TRAIN_CANONICAL_CSV,
    TRAIN_FINAL_COMPONENTS_CSV,
    leaderboard_mean,
    pd,
    plt,
):
    def _load_training_xref():
        runs = (
            pd.read_csv(TRAIN_CANONICAL_CSV)
            if TRAIN_CANONICAL_CSV.exists()
            else pd.DataFrame()
        )
        comps = (
            pd.read_csv(TRAIN_FINAL_COMPONENTS_CSV)
            if TRAIN_FINAL_COMPONENTS_CSV.exists()
            else pd.DataFrame()
        )
        if runs.empty and comps.empty:
            print(
                f"Training cross-ref tables missing.\n"
                f"  Expected: {TRAIN_CANONICAL_CSV}\n"
                f"            {TRAIN_FINAL_COMPONENTS_CSV}\n"
                f"  Run `all_sweeps_grpo_2026_05.py` first to populate them."
            )
            return None
        keep_runs = ["lambda", "ratio", "final_reward", "final_entropy", "final_loss"]
        keep_comp = ["lambda", "ratio", "r_ground", "r_uncert", "composite"]
        df = leaderboard_mean[["lambda", "ratio", "cirl.accuracy", "pl.qa_accuracy", "pl.adj_leak_rate", "goldcoin.compliance.accuracy"]].copy()
        if not runs.empty:
            df = df.merge(runs[[c for c in keep_runs if c in runs.columns]], on=["lambda", "ratio"], how="left")
        if not comps.empty:
            comps_r = comps.rename(
                columns={"r_ground": "train_r_ground", "r_uncert": "train_r_uncert", "composite": "train_composite"}
            )
            df = df.merge(
                comps_r[
                    [c for c in ["lambda", "ratio", "train_r_ground", "train_r_uncert", "train_composite"] if c in comps_r.columns]
                ],
                on=["lambda", "ratio"],
                how="left",
            )
        return df

    xref = _load_training_xref()

    def _plot_xref(df):
        if df is None or df.empty:
            return None
        scatter_pairs = [
            ("final_reward", "cirl.accuracy", "training composite reward", "eval: CIRL acc"),
            ("final_reward", "pl.qa_accuracy", "training composite reward", "eval: PL QA acc"),
            ("final_reward", "pl.adj_leak_rate", "training composite reward", "eval: PL adj-leak (↓)"),
            ("train_r_ground", "pl.qa_accuracy", "training final r_ground", "eval: PL QA acc"),
            ("train_r_ground", "goldcoin.compliance.accuracy", "training final r_ground", "eval: GoldCoin compliance"),
            ("final_entropy", "cirl.accuracy", "training final entropy", "eval: CIRL acc"),
        ]
        pairs = [(x, y, lx, ly) for (x, y, lx, ly) in scatter_pairs if x in df and y in df]
        if not pairs:
            print("No overlapping (training, eval) columns to scatter.")
            return None
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        cmap = plt.get_cmap("viridis")
        lams = df["lambda"].dropna().unique()
        lo, hi = (min(lams), max(lams)) if len(lams) > 1 else (0, 1)
        for ax, (xcol, ycol, lx, ly) in zip(axes.flat, pairs):
            sub = df[[xcol, ycol, "lambda", "ratio"]].dropna()
            if sub.empty:
                ax.set_title(f"{lx} → {ly}\n(no overlap)", fontsize=9, color="#888")
                ax.axis("off")
                continue
            colors = [cmap((v - lo) / (hi - lo)) if hi > lo else cmap(0.5) for v in sub["lambda"]]
            ax.scatter(sub[xcol], sub[ycol], c=colors, s=55, edgecolor="black", lw=0.4)
            for _, r in sub.iterrows():
                ax.annotate(
                    f"({r['lambda']:g},{r['ratio']:g})",
                    (r[xcol], r[ycol]),
                    fontsize=6,
                    xytext=(4, 2),
                    textcoords="offset points",
                    color="#444",
                )
            # Pearson r in the title
            if sub[xcol].std() > 0 and sub[ycol].std() > 0:
                r = float(sub[[xcol, ycol]].corr().iloc[0, 1])
            else:
                r = float("nan")
            ax.set_xlabel(lx, fontsize=8)
            ax.set_ylabel(ly, fontsize=8)
            ax.set_title(f"r = {r:+.2f}", fontsize=9)
            ax.grid(True, alpha=0.25)
        fig.suptitle("Training-side vs eval-side per-cell scatter (colour = λ)", y=1.0, fontsize=12)
        fig.tight_layout()
        return fig

    _plot_xref(xref)
    return (xref,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 13 · Save consolidated tables

    Persist the long-form metric blob index, the per-replicate leaderboard, the
    per-(λ, ρ) aggregated mean/std, the Δ-vs-SFT-CI table, and the training/eval
    cross-reference. Downstream paper-table builders read these instead of
    re-walking the multiruns.
    """)
    return


@app.cell
def _(
    TABLES_DIR,
    delta_df,
    eval_cells,
    leaderboard,
    leaderboard_long,
    leaderboard_mean,
    leaderboard_std,
    sanity_df,
    verify_df,
    xref,
):
    def _save():
        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        written = []

        def _write_csv(df, name):
            if df is None or len(df) == 0:
                return
            p = TABLES_DIR / name
            df.to_csv(p, index=False)
            written.append(p.name)

        def _write_parquet(df, name):
            if df is None or len(df) == 0:
                return
            p = TABLES_DIR / name
            df.to_parquet(p, index=False)
            written.append(p.name)

        _write_csv(eval_cells, "eval_sweep_grpo_2026_05_cells.csv")
        _write_csv(verify_df.assign(
            missing_metrics_dirs=verify_df["missing_metrics_dirs"].map(str),
            missing_metrics_json=verify_df["missing_metrics_json"].map(str),
        ), "eval_sweep_grpo_2026_05_verify.csv")
        _write_parquet(sanity_df, "eval_sweep_grpo_2026_05_sanity.parquet")
        _write_csv(leaderboard, "eval_sweep_grpo_2026_05_leaderboard.csv")
        _write_parquet(leaderboard_long, "eval_sweep_grpo_2026_05_leaderboard_long.parquet")
        _write_csv(leaderboard_mean, "eval_sweep_grpo_2026_05_leaderboard_mean.csv")
        _write_csv(leaderboard_std, "eval_sweep_grpo_2026_05_leaderboard_std.csv")
        _write_csv(delta_df, "eval_sweep_grpo_2026_05_delta_vs_sftci.csv")
        _write_csv(xref, "eval_sweep_grpo_2026_05_train_eval_xref.csv")
        return written

    print("wrote:", _save())
    return


if __name__ == "__main__":
    app.run()
