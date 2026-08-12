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
    # m2 wave-A GRPO arms on external gold

    Reads `multirun/2026-08-03_m2_arms_all_eval/22-40-24` — the 13-cell
    `all_benchmarks` sweep defined by
    `dagspaces/eval_all/conf/sweep/m2_arms_all_2026_08_03.yaml`. **The sweep was
    still running when this notebook was first built (2026-08-04).** Every
    table below distinguishes *pending* from *absent*; nothing is rendered as a
    result until its `metrics.json` exists.

    ## Why the sweep exists

    m2 wave A was closed as a clean negative: all four arms HOLD, trend gains
    −0.003…+0.006 against a 0.02 bar, and the `direct_discrimination` gate
    failed everywhere (core Youden J = 0.0015). Every one of those numbers came
    from the **internal reward instrument**, and wiki §17 (2026-08-03) showed
    that instrument's appropriateness gold is not the construct the policy was
    trained on: `make_direct_chunk_gold`'s top-1 retrieved-norm deontic force
    agrees with the teacher's own `ci_appropriateness` at κ = 0.053, while SFT
    trains on `ci_appropriateness` (`sft_data_prep.py:233`). The m2 HOLD was
    decided by an instrument measuring a different construct.

    External benchmarks never touch that gold. They are the only uncontaminated
    read on what these arms learned. This is the m-series counterpart of the
    K4 revival (`k3_arms_ci_2026_08_03.yaml`).

    ## What this notebook concludes

    The external suite **corroborates the m2 null**, but by a route worth
    reporting: it does *not* simply reproduce flatness. One column — CIRL-729
    net score on scorable rows — shows a large, smooth, checkpoint-ordered
    trajectory that peaks at ckpt-100 (+0.28 over the in-sweep base) and decays
    monotonically back to base by ckpt-450. Four tests decompose it, and
    **96–115% of that gain is attributable to a shift in answer length, with a
    matched-length behavioural residual of ≈0 or negative.** The arms did not
    learn to withhold the disallowed attribute; they learned to emit a short
    answer, and CIRL's leakage metric counts substrings.

    Everything else is flat inside the measured re-run noise floor.

    ## Reading discipline (inherited from the sweep header)

    - **Compare to `k3-base` IN THIS SWEEP.** Never to the paper's v9 numbers:
      different SFT lineage (m2/k-series sit on `sft-canonical`, the Gemma-4
      teacher contentless-v6 rebuild; v9 sits on the qwen-teacher-era SFT) and
      PrivacyLens protocol drift since the keeper era make those cells
      non-comparable. These arms *are* lineage-comparable with the camera-ready
      table's `Qwen3.5-9B / SFT` row and with the K4 cells.
    - **Nine checkpoints of one arm scored on held-out gold is a search over
      the test set.** A best-of-nine here is a DISCOVERY licensing "look
      again", not a camera-ready row. That hazard is what makes the confound
      analysis below load-bearing rather than decorative.
    - **CIRL `*_scorable`, never `*_lenient`** — the lenient rate scores
      silence as perfect privacy. The strict `net_score` is degenerate for this
      whole family (0–47 of 729 rows strict-parseable) and is displayed only as
      a parse-health diagnostic.

    Figures → `figures/m2_external/`, tables → `tables/m2_external/`.
    """)
    return


@app.cell
def _():
    import json
    import sys
    from pathlib import Path

    import matplotlib

    matplotlib.use("Agg")  # headless-safe; marimo renders figures regardless

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.lines import Line2D

    PROJECT_ROOT = Path("/share/pierson/matt/UAIR")
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    NB_DIR = PROJECT_ROOT / "notebooks/colm-camera-ready"
    SWEEP = PROJECT_ROOT / "multirun/2026-08-03_m2_arms_all_eval/22-40-24"
    FIG_DIR = NB_DIR / "figures/m2_external"
    TAB_DIR = NB_DIR / "tables/m2_external"
    for _d in (FIG_DIR, TAB_DIR):
        _d.mkdir(parents=True, exist_ok=True)

    # Exported by benchmark_results.py — the measured re-run noise floor from
    # the judge-free variance record. Read, never recomputed: this notebook
    # must gate against the SAME instrument the camera-ready table gates
    # against, or "flat" here and "flat" there would mean different things.
    NOISE_FLOOR = NB_DIR / "tables/benchmark_results/noise_floor.parquet"

    # Semantics copied verbatim from benchmark_results.py so a cell that reads
    # as "0.95 GoldCoin Appl." means the same number in both places.
    CIRL_SCORABLE_MIN = 0.5
    CONFAIDE_R_SUBDIR = "compute_metrics_tier2b"
    EXPECTED_JUDGE = "Gemma-4-31B-it"

    # (group, column, benchmark dir, dagspace dir, stage subdir, metric key, kind)
    COLUMNS = [
        ("GoldCoin", "Appl.", "goldcoin", "goldcoin_hipaa",
         "compute_metrics_applicability", "accuracy", "gc_acc"),
        ("GoldCoin", "Comp.", "goldcoin", "goldcoin_hipaa",
         "compute_metrics_compliance", "accuracy", "gc_acc"),
        ("PrivacyLens", "QA Acc", "privacylens", "privacylens_eval",
         "compute_metrics", "qa_probing.accuracy", "plain"),
        ("PrivacyLens", "Adj Lk", "privacylens", "privacylens_eval",
         "compute_metrics", "adjusted_leakage.adjusted_leakage_rate", "plain"),
        ("PrivacyLens", "Helpful", "privacylens", "privacylens_eval",
         "compute_metrics", "helpfulness.helpful_rate_among_parseable", "plain"),
        ("ConfAIde", "r", "confaide", "confaide",
         CONFAIDE_R_SUBDIR, "pearson_r", "plain"),
        ("CIRL", "Lk", "cirl", "cirl",
         "compute_metrics", "leakage.leakage_rate_scorable", "cirl_scorable"),
        ("CIRL", "Util", "cirl", "cirl",
         "compute_metrics", "utility.utility_rate_scorable", "cirl_scorable"),
        ("CIRL", "Net(strict)", "cirl", "cirl",
         "compute_metrics", "net_score", "cirl_net"),
        ("VLM", "Q7", "vlm_geoprivacy", "vlm_geoprivacy_bench",
         "compute_metrics", "per_question.Q7.accuracy", "plain"),
        ("MMLU", "Acc", "mmlu", "mmlu",
         "compute_metrics", "overall_accuracy", "plain"),
    ]

    # Display order = sweep order = the `model:` list in the sweep yaml, with
    # the reference cell first.
    CELL_ORDER = [
        "k3-base",
        *[f"m2-core-ckpt{s}" for s in range(50, 500, 50)],
        "m2-full-ckpt450",
        "m2-outcome-ckpt450",
        "m2-vignette-ckpt450",
    ]
    REF_CELL = "k3-base"

    # `core` trajectory steps, for the checkpoint-ordered figures. ckpt-450 is
    # epoch 3.00; each 50 steps is ~0.33 epoch.
    CORE_STEPS = list(range(50, 500, 50))
    STEPS_PER_EPOCH = 150.0

    pd.set_option("display.max_columns", 80)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.4f}".format)

    # COLM camera-ready house style, matching the other camera-ready notebooks.
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
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
        }
    )

    BASE_COLOR = "#4C4C4C"
    CORE_COLOR = "#4C72B0"
    ARM_COLORS = {
        "m2-full-ckpt450": "#DD8452",
        "m2-outcome-ckpt450": "#55A868",
        "m2-vignette-ckpt450": "#C44E52",
    }

    def save_fig(fig, name, also_paper=False):
        for _ext in ("png", "pdf"):
            fig.savefig(FIG_DIR / f"{name}.{_ext}", dpi=300, bbox_inches="tight")
        if also_paper:
            _pf = PROJECT_ROOT / "papers/colm26_normative-simulacra/figures"
            fig.savefig(_pf / f"{name}.pdf", dpi=300, bbox_inches="tight")
            print(f"[paper] {_pf / name}.pdf")
        print(f"[fig] {FIG_DIR / name}.png|.pdf")

    def save_caption(name, title, caption, label, tags):
        _out = FIG_DIR / f"{name}.json"
        _out.write_text(
            json.dumps(
                {
                    "plot-title": title,
                    "plot-caption": caption,
                    "plot-latex-label": label,
                    "plot-tags": tags,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n"
        )
        print(f"[caption] {_out}")

    def save_table(df, name, index=True):
        _out = TAB_DIR / f"{name}.csv"
        df.to_csv(_out, index=index)
        print(f"[table] {_out}")

    print(f"sweep     {SWEEP}")
    print(f"exists    {SWEEP.exists()}")
    return (
        ARM_COLORS,
        BASE_COLOR,
        CELL_ORDER,
        CIRL_SCORABLE_MIN,
        COLUMNS,
        CORE_COLOR,
        CORE_STEPS,
        EXPECTED_JUDGE,
        Line2D,
        NOISE_FLOOR,
        PROJECT_ROOT,
        REF_CELL,
        STEPS_PER_EPOCH,
        SWEEP,
        json,
        np,
        pd,
        plt,
        save_caption,
        save_fig,
        save_table,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Provenance and completeness

    Each numbered subdirectory of the sweep is one cell; its identity comes
    from `.hydra/overrides.yaml` (`model=`), not from directory order. The
    PrivacyLens judge is read from the judge-batch manifest
    (`privacylens/privacylens_eval/outputs/*_judge_batch/manifest.json`
    → `.model`), which is self-attested by the served model at request time —
    the same artifact-level check the camera-ready table applies.

    The completeness table distinguishes three states per benchmark:

    | state | meaning |
    |---|---|
    | `ok` | `metrics.json` exists and parsed |
    | `pending` | cell directory exists, benchmark subdir absent or has no metrics — **still running** |
    | `N/A` | benchmark ran, but the cell is structurally unscoreable on that metric (CIRL below the `scorable_rate` floor) |
    """)
    return


@app.cell
def _(CIRL_SCORABLE_MIN, COLUMNS, CELL_ORDER, SWEEP, json, pd):
    def _dotted(d, path):
        cur = d
        for _p in path.split("."):
            if not isinstance(cur, dict) or _p not in cur:
                return None
            cur = cur[_p]
        return cur

    def _override_model(sub):
        for _ov in sub.glob("*/.hydra/overrides.yaml"):
            for _line in _ov.read_text(errors="ignore").splitlines():
                _line = _line.strip().lstrip("- ").strip()
                if _line.startswith("model="):
                    return _line.split("=", 1)[1]
        return None

    def _served_judge(sub):
        for _man in sorted(
            sub.glob("privacylens/privacylens_eval/outputs/*_judge_batch/manifest.json")
        ):
            try:
                _m = json.loads(_man.read_text()).get("model")
            except Exception:
                continue
            if _m:
                return _m.rstrip("/").split("/")[-1]
        return None

    def _read_metric(sub, bd, inner, subdir, key, kind):
        """→ (value, status, note). status ∈ {ok, pending, N/A}."""
        _mp = sub / bd / inner / "outputs" / subdir / "metrics.json"
        if not _mp.exists():
            return None, "pending", ""
        _data = json.loads(_mp.read_text())
        if kind == "gc_acc":
            # GoldCoin parity: the upstream headline denominator counts
            # unparseable as WRONG. Native runs expose accuracy_among_parseable
            # alongside; older runs need the retro conversion acc x parseable.
            _acc, _pr = _data.get("accuracy"), _data.get("parseable_rate")
            if _acc is None:
                return None, "pending", "no accuracy key"
            if "accuracy_among_parseable" in _data:
                return float(_acc), "ok", "forced-wrong native"
            if _pr is None:
                return None, "N/A", "no denominator semantics"
            return float(_acc) * float(_pr), "ok", f"retro pr={_pr:.3f}"
        if kind == "cirl_net":
            _val = _dotted(_data, key)
            if _val is None:
                return None, "pending", ""
            return (
                float(_val),
                "ok",
                f"strict {_data.get('parseable')}/{_data.get('total')}",
            )
        if kind == "cirl_scorable":
            _sr = _data.get("scorable_rate")
            _val = _dotted(_data, key)
            if _val is None:
                return None, "pending", ""
            if _sr is None:
                return None, "N/A", "pre-rescore (no scorable_rate)"
            if float(_sr) < CIRL_SCORABLE_MIN:
                return None, "N/A", f"scorable={float(_sr):.3f}"
            return float(_val), "ok", f"scorable={float(_sr):.3f}"
        _val = _dotted(_data, key)
        if _val is None:
            return None, "pending", ""
        return float(_val), "ok", ""

    _cell_dirs = sorted(
        (p for p in SWEEP.iterdir() if p.is_dir() and p.name.isdigit()),
        key=lambda p: int(p.name),
    )

    _recs, _meta = [], []
    for _sub in _cell_dirs:
        _model = _override_model(_sub) or _sub.name
        _short = _model.split("/")[-1]
        _meta.append(
            {
                "cell": _short,
                "slot": int(_sub.name),
                "model_override": _model,
                "judge": _served_judge(_sub),
                "path": str(_sub),
            }
        )
        for _grp, _col, _bd, _inner, _subdir, _key, _kind in COLUMNS:
            _v, _st, _note = _read_metric(_sub, _bd, _inner, _subdir, _key, _kind)
            _recs.append(
                {
                    "cell": _short,
                    "slot": int(_sub.name),
                    "benchmark": _grp,
                    "col_id": f"{_grp}::{_col}",
                    "value": _v,
                    "status": _st,
                    "note": _note,
                }
            )

    scan = pd.DataFrame(_recs)
    meta = pd.DataFrame(_meta).set_index("cell").reindex(CELL_ORDER)

    _bad_judge = meta[meta["judge"].notna() & (meta["judge"] != "Gemma-4-31B-it")]
    if len(_bad_judge):
        print("!! judge mismatch — PrivacyLens columns are NOT comparable:")
        print(_bad_judge[["judge"]].to_string())
    else:
        _n = int(meta["judge"].notna().sum())
        print(f"judge attested Gemma-4-31B-it in {_n}/{len(meta)} cells "
              f"(the rest have not reached the judge stage yet)")

    _missing = [c for c in CELL_ORDER if c not in set(scan["cell"])]
    if _missing:
        print(f"!! cells named in CELL_ORDER but absent from the sweep: {_missing}")
    _extra = sorted(set(scan["cell"]) - set(CELL_ORDER))
    if _extra:
        print(f"!! sweep cells not in CELL_ORDER (will not render): {_extra}")

    print()
    print(meta[["slot", "model_override", "judge"]].to_string())
    return meta, scan


@app.cell
def _(CELL_ORDER, save_table, scan):
    completeness = (
        scan.pivot_table(
            index="cell", columns="benchmark", values="status", aggfunc="first"
        )
        .reindex(CELL_ORDER)
    )
    # A benchmark whose columns disagree (some ok, some N/A) is collapsed by
    # `first`; recompute honestly.
    _agg = (
        scan.groupby(["cell", "benchmark"])["status"]
        .agg(lambda s: "ok" if (s == "ok").all() else ("pending" if (s == "pending").all() else "mixed"))
        .unstack()
        .reindex(CELL_ORDER)
    )
    completeness = _agg
    _n_ok = int((scan["status"] == "ok").sum())
    print(f"{_n_ok}/{len(scan)} metric reads complete "
          f"({100 * _n_ok / len(scan):.0f}% of the full 13x11 grid)\n")
    save_table(completeness, "completeness")
    completeness
    return (completeness,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The camera-ready columns

    Same eleven columns and the same semantics as
    `notebooks/colm-camera-ready/benchmark_results.py`, so these cells can be
    laid next to that table's `Qwen3.5-9B / SFT` row without re-deriving
    anything. Blank = pending (the sweep is still running), `N/A` = structural.

    **`CIRL Net(strict)` is a diagnostic column, not a result.** The paper
    protocol scores a response that misses the `<think>`/`<answer>` contract as
    −1; this family emits `<answer>` prose without the strict wrapper, so
    strict-parseable counts run 0–47 of 729 and every cell pins near −1. The
    reportable CIRL quantities are `Lk`/`Util` on scorable rows, and the
    derived `net_scorable = Util − Lk` introduced below.
    """)
    return


@app.cell
def _(CELL_ORDER, COLUMNS, save_table, scan):
    _col_ids = [f"{g}::{c}" for g, c, *_ in COLUMNS]

    def _fmt(row):
        if row["status"] == "N/A":
            return "N/A"
        if row["value"] is None:
            return ""
        return f"{row['value']:.4f}"

    _disp = scan.copy()
    _disp["shown"] = _disp.apply(_fmt, axis=1)
    results = (
        _disp.pivot(index="cell", columns="col_id", values="shown")
        .reindex(CELL_ORDER)[_col_ids]
    )
    values = (
        scan.pivot(index="cell", columns="col_id", values="value")
        .reindex(CELL_ORDER)[_col_ids]
    )
    save_table(results, "results_grid")
    save_table(values, "results_values")
    results
    return results, values


@app.cell
def _(CELL_ORDER, save_table, scan):
    parse_health = (
        scan[scan["note"] != ""]
        .pivot(index="cell", columns="col_id", values="note")
        .reindex(CELL_ORDER)
        .dropna(axis=1, how="all")
    )
    save_table(parse_health, "parse_health")
    parse_health
    return (parse_health,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Gating against the measured re-run noise floor

    Bands are **read** from `tables/benchmark_results/noise_floor.parquet`, the
    judge-free seed/rep variance record (163 arms, 3–8 reps per config) that
    gates the camera-ready table. It supplies *dispersion only, never a value*.
    Per that gate, two cells are separated only when

    $$|a - b| \;\ge\; \tfrac{1}{2}\left(\text{band}_a + \text{band}_b\right)$$

    Here every cell inherits the **column median rep-range** — none of the m2
    arms is in the variance record, so no cell has its own measured band. This
    is the transfer caveat the camera-ready notebook already carries, and it
    makes these bands a floor rather than a full uncertainty budget.

    **Three columns have no band at all.** The variance record is judge-free by
    construction, so PrivacyLens QA Acc / Adj Lk / Helpful get nothing. Their
    deltas below are printed with `band = unmeasured` and **must not** be
    called separated or flat on the basis of this table.
    """)
    return


@app.cell
def _(COLUMNS, NOISE_FLOOR, REF_CELL, np, pd, save_table, values):
    _nf = pd.read_parquet(NOISE_FLOOR)

    # noise_floor.parquet is in DISPLAY units: percentage points for every
    # column except CIRL Net (raw -1..1). Our `values` are raw fractions, so
    # pct-scale bands divide by 100. Column ids differ in one place: the
    # camera-ready table calls leakage "CIRL::Lk↓".
    _NF_ID = {
        "GoldCoin::Appl.": "GoldCoin::Appl.",
        "GoldCoin::Comp.": "GoldCoin::Comp.",
        "ConfAIde::r": "ConfAIde::r",
        "CIRL::Lk": "CIRL::Lk↓",
        "CIRL::Util": "CIRL::Util",
        "CIRL::Net(strict)": "CIRL::Net",
        "VLM::Q7": "VLM::Q7",
        "MMLU::Acc": "MMLU::Acc",
    }
    _RAW_SCALE = {"CIRL::Net(strict)"}  # already raw in the noise file

    _band = {}
    for _our, _theirs in _NF_ID.items():
        _row = _nf[_nf["col_id"] == _theirs]
        if not len(_row):
            continue
        _b = float(_row["median_range"].iloc[0])
        _band[_our] = _b if _our in _RAW_SCALE else _b / 100.0

    _rows = []
    for _g, _c, *_ in COLUMNS:
        _cid = f"{_g}::{_c}"
        _ref = values.loc[REF_CELL, _cid] if _cid in values.columns else np.nan
        for _cell in values.index:
            _v = values.loc[_cell, _cid]
            if pd.isna(_v) or pd.isna(_ref) or _cell == REF_CELL:
                continue
            _d = float(_v) - float(_ref)
            _bd = _band.get(_cid)
            _rows.append(
                {
                    "cell": _cell,
                    "col_id": _cid,
                    "base": float(_ref),
                    "value": float(_v),
                    "delta": _d,
                    "band": _bd if _bd is not None else np.nan,
                    "separated": (abs(_d) >= _bd) if _bd is not None else None,
                }
            )
    deltas = pd.DataFrame(_rows)

    _unbanded = sorted(set(deltas.loc[deltas["band"].isna(), "col_id"]))
    print(f"columns with NO measured noise floor (never call these flat): {_unbanded}\n")

    sep_summary = (
        deltas[deltas["band"].notna()]
        .groupby("col_id")
        .agg(
            n=("delta", "size"),
            max_abs_delta=("delta", lambda s: s.abs().max()),
            band=("band", "first"),
            n_separated=("separated", "sum"),
        )
        .sort_values("n_separated", ascending=False)
    )
    save_table(deltas, "deltas_vs_base", index=False)
    save_table(sep_summary, "separation_summary")
    sep_summary
    return deltas, sep_summary


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Read that table by **magnitude first, gate second**. The gate answers "is
    this bigger than re-run dispersion?", and for benchmarks whose reruns are
    nearly deterministic the band is small enough that trivial differences
    clear it — VLM Q7 marks 10/10 arms "separated" on a largest absolute move
    of 1.3 points, and MMLU marks 4/9 on a largest move of **0.13 points**.
    That is the gate working correctly and telling us very little: a tight band
    means a real but negligible difference is detectable, not that it matters.

    With that read:

    - **MMLU is the capability control and it is intact.** These arms took
      three full epochs of RL at lr 2e-5; if that had cratered baseline
      knowledge, MMLU would show it. The largest move across the whole sweep is
      0.20 points, and 6/11 arms clear a band of 0.08 points — detectable,
      irrelevant.
    - **GoldCoin, ConfAIde and VLM Q7 move by ≤ 2.8, ≤ 2.9 and ≤ 1.3 points**
      respectively, with no checkpoint ordering — noise-scale wobble, and
      GoldCoin sub-1pt gaps are known noise at temp 0.2 regardless of the band.
    - **CIRL is the exception and it is not subtle:** leakage moves 40 points
      and utility 12, both separated in every arm.

    So one column carries the entire signal, and it carries it in a very
    particular shape.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## CIRL-729: a large, checkpoint-ordered, monotone effect

    CIRL-729 asks the model to carry out a task using a set of attributes, of
    which some are *allowed* and some *disallowed* in that context. Utility =
    fraction of allowed attributes that appear in the answer; leakage =
    fraction of disallowed attributes that appear. Both are **substring
    membership tests over the emitted text**, so the metric is monotone in how
    much the model says.

    Everything from here down is scored **row-level**, by importing the
    production scorer directly:

    ```python
    from dagspaces.cirl.stages.compute_metrics import _pairs, _score_row
    ```

    so the per-row numbers aggregate exactly to the `*_scorable` rates in the
    table above. A row is scorable when its lenient answer is non-empty and did
    not hit the token cap (`finish_reason != "length"`).

    We report the derived **`net_scorable` = utility_rate − leakage_rate on
    scorable rows** — a Youden-J on item inclusion. It is a *notebook-local*
    quantity: the camera-ready `CIRL Net` column is the strict paper-protocol
    `net_score`, which is degenerate here. A conservative band for
    `net_scorable` is the sum of the Lk and Util column median ranges,
    0.0083 + 0.0079 = **0.0162**; the effects below are 15–30× that.
    """)
    return


@app.cell
def _(SWEEP, np, pd):
    # Imported through the module rather than `from ... import _pairs`: marimo
    # mangles underscore-prefixed cell-local names, and the mangling does not
    # follow the binding into a nested function.
    from dagspaces.cirl.stages import compute_metrics as cirl_metrics

    pair_fn, score_fn = cirl_metrics._pairs, cirl_metrics._score_row

    def _cirl_rows(sub):
        _p = sub / "cirl/cirl/outputs/parse_responses/dataset.parquet"
        if not _p.exists():
            return None
        _df = pd.read_parquet(_p)
        _out = []
        for _, _r in _df.iterrows():
            _allowed = pair_fn(_r.get("allowed_scored"))
            _dis = pair_fn(_r.get("disallowed_scored"))
            _ans = str(_r.get("answer_lenient", "") or "")
            _u, _l, _nlk, _ndisc = score_fn(_ans, _allowed, _dis)
            _trunc = str(_r.get("finish_reason", "stop")) == "length"
            _out.append(
                {
                    "row_id": _r["row_id"],
                    "util": _u,
                    "leak": _l,
                    "n_leaked": _nlk,
                    "n_disclosed": _ndisc,
                    "n_allowed": len(_allowed),
                    "n_disallowed": len(_dis),
                    "chars": len(_ans),
                    "scorable": bool(_ans.strip()) and not _trunc,
                }
            )
        return pd.DataFrame(_out)

    def _name(sub):
        for _ov in sub.glob("*/.hydra/overrides.yaml"):
            for _line in _ov.read_text(errors="ignore").splitlines():
                _line = _line.strip().lstrip("- ").strip()
                if _line.startswith("model="):
                    return _line.split("=", 1)[1].split("/")[-1]
        return sub.name

    _frames = []
    for _sub in sorted(
        (p for p in SWEEP.iterdir() if p.is_dir() and p.name.isdigit()),
        key=lambda p: int(p.name),
    ):
        _d = _cirl_rows(_sub)
        if _d is None:
            continue
        _d["cell"] = _name(_sub)
        _frames.append(_d)

    cirl_rows = pd.concat(_frames, ignore_index=True)
    cirl_rows["net"] = cirl_rows["util"] - cirl_rows["leak"]
    cirl_scorable = cirl_rows[cirl_rows["scorable"]].copy()

    print(f"{cirl_rows['cell'].nunique()} cells with CIRL responses, "
          f"{len(cirl_rows)} rows, {len(cirl_scorable)} scorable "
          f"({100 * len(cirl_scorable) / len(cirl_rows):.1f}%)")
    return cirl_rows, cirl_scorable


@app.cell
def _(CELL_ORDER, cirl_scorable, pd, save_table):
    _rows = []
    for _cell, _g in cirl_scorable.groupby("cell"):
        _disc, _lk = _g["n_disclosed"].sum(), _g["n_leaked"].sum()
        _rows.append(
            {
                "cell": _cell,
                "n_scorable": len(_g),
                "util": _g["util"].mean(),
                "leak": _g["leak"].mean(),
                "net_scorable": _g["util"].mean() - _g["leak"].mean(),
                "precision": _disc / (_disc + _lk) if (_disc + _lk) else float("nan"),
                "items_per_row": (_disc + _lk) / len(_g),
                "median_chars": _g["chars"].median(),
            }
        )
    cirl_summary = (
        pd.DataFrame(_rows).set_index("cell").reindex(CELL_ORDER).dropna(how="all")
    )
    save_table(cirl_summary, "cirl_summary")
    cirl_summary
    return (cirl_summary,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    That table is the whole finding in miniature. Along the `core` trajectory,
    `net_scorable` rises from the base's 0.185 to **0.465 at ckpt-100**, then
    decays monotonically — 0.432, 0.414, 0.392, 0.362, 0.307, 0.250 — back to
    0.230 at ckpt-450, essentially base. `m2-full-ckpt450` (0.470) and
    `m2-outcome-ckpt450` (0.454) retain it at the same step where `core` has
    lost it.

    Read the last two columns alongside. `items_per_row` — total attributes
    mentioned, allowed plus disallowed — tracks the effect inversely: 4.22 at
    base, 2.85 at ckpt-100, back to 4.07 at ckpt-450. So does median answer
    length: **2,057 characters at base, 547 at ckpt-100, 2,054 at ckpt-450.**
    The gain and the terseness are the same curve.

    That is the alarm. The next four cells are the tests.
    """)
    return


@app.cell
def _(
    ARM_COLORS,
    BASE_COLOR,
    CORE_COLOR,
    CORE_STEPS,
    STEPS_PER_EPOCH,
    cirl_summary,
    plt,
    save_caption,
    save_fig,
):
    _core = [(s, f"m2-core-ckpt{s}") for s in CORE_STEPS]
    _core = [(s, c) for s, c in _core if c in cirl_summary.index]
    _x = [s / STEPS_PER_EPOCH for s, _ in _core]

    fig_traj, _axes = plt.subplots(1, 2, figsize=(7.0, 2.7))

    _ax = _axes[0]
    _ax.plot(_x, [cirl_summary.loc[c, "net_scorable"] for _, c in _core],
             "o-", color=CORE_COLOR, lw=1.4, ms=4, label="m2-core trajectory")
    if "k3-base" in cirl_summary.index:
        _ax.axhline(cirl_summary.loc["k3-base", "net_scorable"], color=BASE_COLOR,
                    ls="--", lw=1.0, label="k3-base (in-sweep SFT)")
    for _c, _col in ARM_COLORS.items():
        if _c in cirl_summary.index:
            _ax.plot([3.0], [cirl_summary.loc[_c, "net_scorable"]], "D",
                     color=_col, ms=5, label=_c.replace("m2-", "").replace("-ckpt450", " @450"))
    _ax.set_xlabel("epoch")
    _ax.set_ylabel("CIRL net (scorable)")
    _ax.set_title("Net score peaks early, decays to base")
    # The legend goes under the figure, not inside an axes: an unframed legend
    # drawn over this panel puts its marker handles in the middle of the data
    # region, where they read as extra points.
    _handles, _labels = _ax.get_legend_handles_labels()

    _ax = _axes[1]
    _ax.plot(_x, [cirl_summary.loc[c, "median_chars"] for _, c in _core],
             "o-", color=CORE_COLOR, lw=1.4, ms=4)
    if "k3-base" in cirl_summary.index:
        _ax.axhline(cirl_summary.loc["k3-base", "median_chars"], color=BASE_COLOR,
                    ls="--", lw=1.0)
    for _c, _col in ARM_COLORS.items():
        if _c in cirl_summary.index:
            _ax.plot([3.0], [cirl_summary.loc[_c, "median_chars"]], "D", color=_col, ms=5)
    _ax.set_xlabel("epoch")
    _ax.set_ylabel("median answer length (chars)")
    _ax.set_title("Answer length traces the same curve, inverted")

    fig_traj.tight_layout()
    fig_traj.legend(
        _handles,
        _labels,
        frameon=False,
        loc="lower center",
        ncol=len(_labels),
        bbox_to_anchor=(0.5, -0.10),
    )
    save_fig(fig_traj, "fig_m2_cirl_trajectory")
    save_caption(
        "fig_m2_cirl_trajectory",
        "CIRL-729 net score and answer length along the m2-core trajectory",
        "CIRL-729 net score on scorable rows (left) and median answer length "
        "(right) across the nine m2-core checkpoints, with the in-sweep SFT "
        "base as a dashed line and the three other wave-A arms at epoch 3. The "
        "net-score gain peaks at ckpt-100 and decays monotonically back to "
        "base; median answer length falls from 2,057 characters to 547 and "
        "returns. The two panels are the same curve.",
        "fig:m2-cirl-trajectory",
        ["m2", "cirl", "confound"],
    )
    fig_traj
    return (fig_traj,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Test 1 — the proportional-omission null

    The cheapest explanation for "fewer items mentioned, higher net" is that
    the model dropped items *at random*. If a cell mentions a fraction
    $\rho$ of the base's items and the omissions are indifferent to
    allowed/disallowed status, then both rates scale by $\rho$ and

    $$\text{net}' = \rho \cdot \text{net}_{\text{base}} \le \text{net}_{\text{base}}$$

    So random truncation can only ever *lower* net. Any positive excess over
    $\rho \cdot J_0$ is selective omission — the model preferentially dropping
    disallowed attributes. This test cannot be passed by verbosity alone, and
    the arms pass it decisively.
    """)
    return


@app.cell
def _(REF_CELL, cirl_summary, save_table):
    _J0 = cirl_summary.loc[REF_CELL, "net_scorable"]
    _rho = cirl_summary["items_per_row"] / cirl_summary.loc[REF_CELL, "items_per_row"]
    omission_null = cirl_summary[["items_per_row", "net_scorable"]].copy()
    omission_null["rho"] = _rho
    omission_null["null_net"] = _rho * _J0
    omission_null["excess"] = omission_null["net_scorable"] - omission_null["null_net"]
    save_table(omission_null, "omission_null")
    print(f"base net J0 = {_J0:.4f}; proportional omission can only give rho*J0 <= J0\n")
    omission_null
    return (omission_null,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Excess of **+0.339** at ckpt-100 (+0.333 for `full`, +0.328 for
    `outcome`). The omission is emphatically *not* random: precision on
    mentioned items rises from 0.588 to 0.747.

    Which is exactly why Test 1 is not enough. "Selective with respect to
    allowed/disallowed" is compatible with two very different mechanisms:

    1. the policy learned a contextual-integrity rule and withholds the
       disallowed attribute; or
    2. the policy became terse, and *terse answers on this benchmark are
       selective by construction* — a short answer names the few attributes the
       task literally requires (mostly allowed) and never gets around to the
       incidental ones (disproportionately disallowed).

    Mechanism 2 predicts that the effect lives entirely in the length shift.
    Tests 2–4 target that prediction.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Test 2 — the length-matched paired comparison

    Join each arm to the base **row by row** (same CIRL item, both scorable),
    then restrict to rows where the arm's answer length barely moved
    ($|\Delta \text{chars}| < 200$). On those rows the arm and the base are
    saying roughly as much as each other, so any remaining difference is
    behaviour rather than verbosity.
    """)
    return


@app.cell
def _(REF_CELL, cirl_scorable, pd, save_table):
    _base = cirl_scorable[cirl_scorable["cell"] == REF_CELL]

    _rows = []
    for _cell, _g in cirl_scorable.groupby("cell"):
        if _cell == REF_CELL:
            continue
        _m = _base.merge(_g, on="row_id", suffixes=("_b", "_x"))
        _m = _m.assign(
            dlen=_m["chars_x"] - _m["chars_b"],
            dnet=(_m["util_x"] - _m["leak_x"]) - (_m["util_b"] - _m["leak_b"]),
            dleak=_m["leak_x"] - _m["leak_b"],
            dutil=_m["util_x"] - _m["util_b"],
        )
        _same = _m[_m["dlen"].abs() < 200]
        _rows.append(
            {
                "cell": _cell,
                "n_paired": len(_m),
                "dnet_all": _m["dnet"].mean(),
                "median_dlen": _m["dlen"].median(),
                "n_matched": len(_same),
                "dnet_matched": _same["dnet"].mean() if len(_same) else float("nan"),
                "dleak_matched": _same["dleak"].mean() if len(_same) else float("nan"),
                "dutil_matched": _same["dutil"].mean() if len(_same) else float("nan"),
            }
        )
    length_matched = pd.DataFrame(_rows).set_index("cell")
    save_table(length_matched, "length_matched")
    length_matched
    return (length_matched,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **The result reverses.** `m2-core-ckpt100` gains +0.278 net over all paired
    rows and **−0.096** on the length-matched subset (n = 147).
    `m2-full-ckpt450`: +0.286 overall, −0.017 matched. `m2-core-ckpt450`:
    +0.044 overall, −0.009 matched. On rows where the arm did not shorten its
    answer, it is no better than the base and mostly slightly worse.

    A caveat worth stating rather than hiding: the matched subset is not a
    random sample of items. Rows whose length barely moved are disproportionately
    rows that were *already short* under the base (base Q1 median is ~490
    characters, close to ckpt-100's global median), where the base already
    scores well (+0.571). So Test 2 alone is vulnerable to a ceiling argument.
    Test 3 removes that objection by conditioning on the arm's *own* length.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Test 3 — conditioning on the model's own answer length

    Bin every response by its **own** character count, using one set of edges
    shared across all cells, and read leakage and net within each bin. If the
    arms learned a privacy behaviour, an arm's leakage curve should sit *below*
    the base's at matched length. If length is the whole story, the cells share
    one curve and differ only in where their mass sits along it.

    Two things to check, and they point the same way:

    1. **The curve itself is steep and shared.** Base net falls 0.577 → 0.626 →
       0.380 → 0.111 → 0.067 → 0.027 → 0.054 across the bins; leakage climbs
       0.151 → 0.848. Every cell reproduces that shape. Length is not a nuisance
       covariate on this benchmark, it is the dominant one.
    2. **In the bins holding the arms' mass, the arms are at or slightly below
       the base.** ckpt-100 puts 85% of its rows under 1,000 characters and
       scores 0.514 / 0.554 / 0.357 there against the base's 0.577 / 0.626 /
       0.380. It is not better at matched length; it is worse, consistent with
       Test 2.

    **Where the arms do beat the base at matched length** — the 1k–2.2k bins
    (ckpt-100: 0.277 and 0.348 vs 0.111 and 0.067) — read that with the mass
    column. The base has 2.1% of its rows in 1k–1.5k; ckpt-100 has 9.1%. Those
    bins hold different item populations under the two policies (a row that is
    1.2k characters for a terse model is a hard, long-answer item; for the base
    it is an unusually short one), so this is a selection artifact of
    conditioning on an outcome variable, not a matched comparison. Test 4
    prices it in: reweighting the base's curve by each arm's mass gives the
    counterfactual directly.
    """)
    return


@app.cell
def _(cirl_scorable, np, pd, save_table):
    LEN_EDGES = [0, 500, 700, 1000, 1500, 2200, 3000, np.inf]
    LEN_LABELS = ["<500", "500-700", "700-1k", "1k-1.5k", "1.5k-2.2k", "2.2k-3k", ">3k"]

    _b = cirl_scorable.copy()
    _b["lbin"] = pd.cut(_b["chars"], bins=LEN_EDGES, labels=LEN_LABELS, right=False)

    length_bins = (
        _b.groupby(["cell", "lbin"], observed=True)
        .agg(n=("net", "size"), leak=("leak", "mean"), util=("util", "mean"),
             net=("net", "mean"))
        .reset_index()
    )
    # A bin with a handful of rows is noise, not a curve.
    length_bins = length_bins[length_bins["n"] >= 15]

    net_by_bin = length_bins.pivot(index="cell", columns="lbin", values="net")
    leak_by_bin = length_bins.pivot(index="cell", columns="lbin", values="leak")
    mass_by_bin = (
        _b.groupby(["cell", "lbin"], observed=True).size().unstack(fill_value=0)
    )
    mass_by_bin = mass_by_bin.div(mass_by_bin.sum(axis=1), axis=0)

    save_table(net_by_bin, "net_by_own_length_bin")
    save_table(leak_by_bin, "leak_by_own_length_bin")
    save_table(mass_by_bin, "length_mass_by_bin")

    print("net score within own-length bin (spread across cells, per bin):")
    print((net_by_bin.max() - net_by_bin.min()).round(3).to_string())
    net_by_bin
    return LEN_EDGES, LEN_LABELS, leak_by_bin, length_bins, mass_by_bin, net_by_bin


@app.cell
def _(
    ARM_COLORS,
    BASE_COLOR,
    CORE_COLOR,
    Line2D,
    REF_CELL,
    leak_by_bin,
    mass_by_bin,
    net_by_bin,
    plt,
    save_caption,
    save_fig,
):
    def _style(cell):
        if cell == REF_CELL:
            return BASE_COLOR, 1.8, "--", 5
        if cell in ARM_COLORS:
            return ARM_COLORS[cell], 1.1, "-", 3
        return CORE_COLOR, 0.9, "-", 2.5

    fig_collapse, _axes = plt.subplots(1, 3, figsize=(9.2, 2.8))

    for _ax, _df, _lab in (
        (_axes[0], net_by_bin, "net score"),
        (_axes[1], leak_by_bin, "leakage rate"),
    ):
        for _cell in _df.index:
            _c, _lw, _ls, _ms = _style(_cell)
            # Plot against the bin's own position, NOT against a dropna'd
            # index: a cell missing an interior bin would otherwise be drawn
            # shifted left. matplotlib breaks the line at NaN, which is what
            # we want.
            _s = _df.loc[_cell]
            _ax.plot(range(len(_df.columns)), _s.values, marker="o", color=_c,
                     lw=_lw, ls=_ls, ms=_ms,
                     alpha=0.85 if _cell == REF_CELL else 0.6)
        _ax.set_xticks(range(len(_df.columns)))
        _ax.set_xticklabels(_df.columns, rotation=45, ha="right")
        _ax.set_ylabel(_lab)
        _ax.set_xlabel("own answer length (chars)")
    _axes[0].set_title("Net falls steeply with length")
    _axes[1].set_title("Leakage climbs with length")

    _ax = _axes[2]
    for _cell in mass_by_bin.index:
        _c, _lw, _ls, _ms = _style(_cell)
        _ax.plot(range(mass_by_bin.shape[1]), mass_by_bin.loc[_cell].values,
                 marker="o", color=_c, lw=_lw, ls=_ls, ms=_ms,
                 alpha=0.85 if _cell == REF_CELL else 0.6)
    _ax.set_xticks(range(mass_by_bin.shape[1]))
    _ax.set_xticklabels(mass_by_bin.columns, rotation=45, ha="right")
    _ax.set_ylabel("share of rows")
    _ax.set_xlabel("own answer length (chars)")
    _ax.set_title("Only the mass moves")

    fig_collapse.tight_layout()
    _lh = [
        Line2D([], [], color=BASE_COLOR, ls="--", lw=1.8, marker="o", ms=5,
               label="k3-base (in-sweep SFT)"),
        Line2D([], [], color=CORE_COLOR, lw=0.9, marker="o", ms=2.5,
               label="m2-core ckpt50…450"),
        *[
            Line2D([], [], color=_c, lw=1.1, marker="o", ms=3,
                   label=_k.replace("m2-", "").replace("-ckpt450", " @450"))
            for _k, _c in ARM_COLORS.items()
            if _k in net_by_bin.index
        ],
    ]
    fig_collapse.legend(
        handles=_lh,
        frameon=False,
        loc="lower center",
        ncol=len(_lh),
        bbox_to_anchor=(0.5, -0.22),
    )
    save_fig(fig_collapse, "fig_m2_cirl_length_collapse")
    save_caption(
        "fig_m2_cirl_length_collapse",
        "CIRL-729 scores conditioned on the model's own answer length",
        "Net score (left) and leakage rate (centre) within bins of the "
        "model's own answer length, and the share of each cell's rows falling "
        "in each bin (right). Dashed grey is the in-sweep SFT base, blue the "
        "nine m2-core checkpoints, coloured markers the other three wave-A "
        "arms. Net score falls and leakage climbs steeply with length for "
        "every cell, and in the short bins where the terse arms concentrate "
        "their mass they score at or slightly below the base. What moves "
        "between cells is the mass, not the curve.",
        "fig:m2-cirl-length-collapse",
        ["m2", "cirl", "confound"],
    )
    fig_collapse
    return (fig_collapse,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Test 4 — decomposing the gain into length and behaviour

    An Oaxaca-style counterfactual makes the accounting explicit. For cell $c$
    with length-bin mass $w_c(b)$ and the base's within-bin net score
    $J_0(b)$:

    $$\underbrace{J_c}_{\text{observed}} \;=\;
      \underbrace{\textstyle\sum_b w_c(b) J_0(b)}_{\text{CF}_{\text{len}}
      \;=\;\text{base behaviour, arm's lengths}}
      \;+\; \underbrace{\big(J_c - \text{CF}_{\text{len}}\big)}_{\text{behaviour}}$$

    and the gain over base splits as

    $$J_c - J_0 \;=\; \underbrace{\text{CF}_{\text{len}} - J_0}_{\text{length}}
      \;+\; \underbrace{J_c - \text{CF}_{\text{len}}}_{\text{behaviour}}.$$

    Cells are dropped from the reweighting where the base has no bin mass, so
    `CF_len` is only defined on the support the base actually covers; the
    coverage column reports how much of each arm's mass that is.
    """)
    return


@app.cell
def _(REF_CELL, mass_by_bin, net_by_bin, pd):
    _base_net = net_by_bin.loc[REF_CELL]
    _support = _base_net.dropna().index

    _rows = []
    for _cell in mass_by_bin.index:
        _w = mass_by_bin.loc[_cell, _support]
        _cov = float(_w.sum())
        if _cov <= 0:
            continue
        # Renormalise over the base's support: CF_len is the base's behaviour
        # curve reweighted by the arm's lengths, and it is only defined where
        # the base has a curve at all.
        _cf = float((_w * _base_net[_support]).sum() / _cov)
        _rows.append({"cell": _cell, "CF_len": _cf, "coverage": _cov})

    decomp = pd.DataFrame(_rows).set_index("cell")
    return (decomp,)


@app.cell
def _(REF_CELL, cirl_summary, decomp, np, save_table):
    _d = decomp.copy()
    _d["observed"] = cirl_summary["net_scorable"].reindex(_d.index)
    _J0 = _d.loc[REF_CELL, "CF_len"]
    _d["length_part"] = _d["CF_len"] - _J0
    _d["behaviour_part"] = _d["observed"] - _d["CF_len"]
    _d["total_gain"] = _d["observed"] - _d.loc[REF_CELL, "observed"]
    _d["pct_length"] = np.where(
        _d["total_gain"].abs() > 0.02,
        100 * _d["length_part"] / _d["total_gain"],
        np.nan,
    )
    decomposition = _d[
        ["observed", "CF_len", "total_gain", "length_part", "behaviour_part",
         "pct_length", "coverage"]
    ]
    save_table(decomposition, "length_behaviour_decomposition")
    print("pct_length = share of the gain over base explained by the length "
          "distribution shift alone (blank where the gain is within noise)\n")
    decomposition
    return (decomposition,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **The length term accounts for 96–115% of the gain in every arm that shows
    one, and the behavioural residual is ≈0 or negative:**

    | cell | observed | CF (base behaviour, arm's lengths) | gain | length | behaviour | % length |
    |---|---|---|---|---|---|---|
    | `m2-core-ckpt100` | 0.465 | 0.488 | +0.280 | +0.303 | −0.023 | 108% |
    | `m2-core-ckpt150` | 0.432 | 0.438 | +0.247 | +0.253 | −0.007 | 103% |
    | `m2-core-ckpt200` | 0.414 | 0.432 | +0.229 | +0.247 | −0.018 | 108% |
    | `m2-core-ckpt250` | 0.392 | 0.402 | +0.207 | +0.217 | −0.011 | 105% |
    | `m2-core-ckpt300` | 0.362 | 0.363 | +0.177 | +0.178 | −0.001 | 101% |
    | `m2-core-ckpt350` | 0.307 | 0.303 | +0.122 | +0.118 | +0.004 | 96% |
    | `m2-full-ckpt450` | 0.470 | 0.469 | +0.284 | +0.283 | +0.001 | 100% |
    | `m2-outcome-ckpt450` | 0.454 | 0.493 | +0.269 | +0.308 | −0.040 | 115% |

    Reading `m2-full-ckpt450`: had the base produced answers with that arm's
    length distribution, and kept its own behaviour at every length, it would
    have scored 0.469 against the arm's observed 0.470. The behaviour term is
    +0.001 — a thousandth of a point, against a band of 0.016.

    The two late `core` checkpoints (ckpt-400, ckpt-450) fall to 89% and 78%,
    but their total gains are 0.064 and 0.045 and their behaviour terms are
    +0.007 and +0.010 — inside the 0.016 band. There is no checkpoint anywhere
    on this trajectory whose behavioural residual is distinguishable from zero
    in the favourable direction.

    Taken together, the four tests say the same thing from four directions.
    The arms did not learn a contextual-integrity rule that CIRL detects. They
    learned to be terse, and CIRL's substring-membership leakage metric rewards
    terseness — precisely because a shorter answer touches fewer attributes,
    and the incidental attributes it drops first are disproportionately the
    disallowed ones.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Does the terseness shift generalise?

    If the arms had globally become terse, we would expect it on every
    generative benchmark, and the CIRL result would be one visible symptom of a
    general behaviour change. If instead the shift is CIRL-specific, it is a
    prompt-format interaction with CIRL's long attribute-list task, and even
    less generalisable than the confound analysis already implies.

    Median generated characters per benchmark, read from each benchmark's
    inference-stage parquet.
    """)
    return


@app.cell
def _(CELL_ORDER, SWEEP, pd, save_table):
    LENGTH_SOURCES = {
        "CIRL": "cirl/cirl/outputs/llm_inference/dataset.parquet",
        "GoldCoin": "goldcoin/goldcoin_hipaa/outputs/llm_inference_applicability/dataset.parquet",
        "ConfAIde": "confaide/confaide/outputs/llm_inference_tier2b/dataset.parquet",
        "PrivacyLens": "privacylens/privacylens_eval/outputs/agent_action_inference/results.parquet",
        "MMLU": "mmlu/mmlu/outputs/llm_inference/dataset.parquet",
    }

    def _cell_name(sub):
        for _ov in sub.glob("*/.hydra/overrides.yaml"):
            for _line in _ov.read_text(errors="ignore").splitlines():
                _line = _line.strip().lstrip("- ").strip()
                if _line.startswith("model="):
                    return _line.split("=", 1)[1].split("/")[-1]
        return sub.name

    _rows = []
    for _sub in sorted(
        (p for p in SWEEP.iterdir() if p.is_dir() and p.name.isdigit()),
        key=lambda p: int(p.name),
    ):
        _rec = {"cell": _cell_name(_sub)}
        for _bench, _rel in LENGTH_SOURCES.items():
            _p = _sub / _rel
            if not _p.exists():
                _rec[_bench] = None
                continue
            try:
                _df = pd.read_parquet(_p, columns=["generated_text"])
            except Exception:
                _rec[_bench] = None
                continue
            _rec[_bench] = float(_df["generated_text"].fillna("").str.len().median())
        _rows.append(_rec)

    lengths = (
        pd.DataFrame(_rows).set_index("cell").reindex(CELL_ORDER).dropna(how="all")
    )
    save_table(lengths, "median_generated_chars")
    print("median generated characters (blank = benchmark not yet run for that cell)\n")
    lengths
    return (lengths,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **The shift is CIRL-specific.** CIRL swings 4× across the trajectory
    (594 → 2,374 characters) while every other benchmark holds a narrow band:
    GoldCoin 566–589, ConfAIde 3 (a bare number at `max_tokens=32`), MMLU 15,
    PrivacyLens agent actions 927–1,000. Whatever the RL changed, it is not a
    global length policy — it surfaces on the one benchmark whose metric is
    monotone in output length, and it is invisible on the four that are not.

    That also disposes of the most charitable reading of the CIRL result. A
    genuine privacy behaviour would be expected to show *somewhere* else in a
    five-benchmark CI suite. It shows nowhere, and the one place it appears to
    show is fully accounted for by verbosity.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What this licenses

    1. **The m2 null survives external measurement, and is now better
       supported than it was.** The original HOLD rested on an instrument later
       shown to score a different construct (κ = 0.053). This sweep replaces
       that evidence with uncontaminated external gold and reaches the same
       verdict — not by finding flatness everywhere, but by finding one large
       effect and showing it to be a verbosity artifact. That is a dissociation,
       and it is stronger evidence than flatness would have been.

    2. **No arm is promotable.** `m2-full-ckpt450`, `m2-outcome-ckpt450` and
       `m2-core-ckpt100` post the best CIRL numbers in the sweep, and their
       behavioural residuals are +0.001, −0.040 and −0.023. The
       selection hazard in the sweep header — best-of-nine on held-out gold is a
       search over the test set — never even has to be invoked; the candidates
       fail on mechanism before they reach the multiplicity question.

    3. **The keeper stands.** Nothing here displaces v9-ckpt100, and nothing
       here is comparable to it in the first place (different SFT lineage,
       different PrivacyLens protocol). These cells are comparable to the
       camera-ready `Qwen3.5-9B / SFT` row and to the K4 cells.

    4. **CIRL-729 net score needs a length control whenever it is used to
       compare policies.** This is a general instrument finding, not an m2
       finding: a substring-membership leakage rate is monotone in output
       length, so any intervention that changes verbosity moves it for free. The
       own-length-bin collapse (Test 3) is the cheap version of that control and
       should accompany any future CIRL comparison across training arms.

    5. **What is still owed.** Twelve of the thirteen cells are complete;
       `m2-vignette-ckpt450` has GoldCoin only, and `m2-core-ckpt250` is
       missing GoldCoin. The completeness table at the top of this notebook is
       the authority — re-run and read it there rather than trusting this
       paragraph. Neither gap can change conclusions 1–4, which rest on the
       full nine-checkpoint `core` trajectory and on two of the three finals,
       but the table should read clean before any of this is quoted.

    ### Caveats stated rather than implied

    - **The PrivacyLens trio has no measured noise floor.** The variance record
      is judge-free by construction. QA Acc / Adj Lk / Helpful deltas are
      reported ungated and must not be described as flat *or* as separated on
      the strength of this notebook.
    - **No m2 cell has its own measured band.** Every band is the column median
      transferred from the variance record's 44 arms. Treat it as a floor.
    - **Single inference run per cell.** The bands describe re-run dispersion
      measured elsewhere on other checkpoints, not repeated draws of these
      cells.
    - **`net_scorable` is a notebook-local metric.** The camera-ready `CIRL
      Net` column remains the strict paper-protocol `net_score`; do not
      substitute one for the other in the table.
    """)
    return


@app.cell
def _():
    print("done — figures/ and tables/ under notebooks/colm-camera-ready/*/m2_external")
    return


if __name__ == "__main__":
    app.run()
