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
    # Norm-distribution shift: top100 (qwen3.6-27b) vs. fiction10 (qwen2.5-72b-awq)

    Built 2026-06-01.

    Two extractions of Raz-style norms from fiction, produced by different
    extractor models on different corpora:

    | corpus | extractor | books | path |
    |---|---|---|---|
    | **top100** | `Qwen3.6-27B` | 97 | `multirun/2026-05-26_historical_norms/05-00-38/.../role_abstraction/abstracted_norms.parquet` |
    | **fiction10** | `Qwen2.5-72B-AWQ` | 10 | `n2s4cir/data/fiction10/abstracted_norms.parquet` |

    Both files are the **role-abstraction stage output** (`abstracted_norms.parquet`,
    one row per extracted norm, identical schema), so the categorical norm
    fields are directly comparable.

    This notebook asks: **how do the *distributions* of norm attributes differ
    between the two extractions?** It is the distributional companion to
    `norm_yield_gap_qwen36_2026_05.py` (which compares norm *counts* / the
    `has_norms` gate, not the shape of the extracted norms).

    Axes compared (the categorical/ordinal fields the schema attaches to every
    norm):

    - `raz_normative_force` — deontic modality (obligatory / prohibited / permitted / recommended / discouraged)
    - `raz_norm_source` — explicit vs. implicit statement in text
    - `raz_governs_info_flow` — does the norm regulate information transmission? (the CI-relevant subset)
    - `raz_context` — societal domain
    - `raz_confidence_qual` / `raz_confidence_quant` — extractor confidence
    - `norm_quality_passed` — generalizability gate (no character/plot-specific leakage)

    > **Two confounds, stated up front.** (1) The corpora differ — 97 books vs
    > 10, with only partial overlap — so a distribution shift can reflect *what
    > was read* as much as *who read it*. (2) The models differ. This notebook
    > cannot separate the two; it characterizes the *combined* shift and, where
    > useful, reports a per-book-averaged variant to blunt the
    > composition confound.
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    PROJECT_ROOT = Path("/share/pierson/matt/UAIR")
    NB_DIR = PROJECT_ROOT / "notebooks/normative-simulacra"

    # fiction10 — qwen2.5-72b-awq, canonical reference corpus (10 books)
    COLM_ABSTRACTED = Path(
        "/share/pierson/matt/n2s4cir/data/fiction10/abstracted_norms.parquet"
    )
    # top100 — qwen3.6-27b, 2026-05-26 standalone role-abstraction run (97 books)
    TOP100_ABSTRACTED = Path(
        "/share/pierson/matt/UAIR/multirun/2026-05-26_historical_norms/05-00-38/0"
        "/role_abstraction_standalone_qwen36/outputs/role_abstraction/abstracted_norms.parquet"
    )

    # Consistent corpus colours throughout (match the sibling yield-gap notebook:
    # qwen2.5-72b = blue, qwen3.6-27b = orange).
    COLOR = {"fiction10": "#1f77b4", "top100": "#ff7f0e"}
    LABEL = {
        "fiction10": "fiction10 (qwen2.5-72b-awq, 10 books)",
        "top100": "top100 (qwen3.6-27b, 97 books)",
    }

    pd.set_option("display.max_columns", 80)
    pd.set_option("display.float_format", "{:.3f}".format)

    sys.path.insert(0, str(NB_DIR.parent))
    try:
        from font_utils import load_ibm_plex_sans

        load_ibm_plex_sans()
    except Exception as _e:
        print(f"[font_utils] skipped: {_e}")
    return COLM_ABSTRACTED, COLOR, LABEL, TOP100_ABSTRACTED, np, pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · Load and align

    Read both parquets, tag each with a `corpus` label, drop the handful of
    null-`raz_normative_force` rows (chunks that produced no usable norm — 315
    in fiction10, 33 in top100), and concatenate on the shared columns.
    """)
    return


@app.cell
def _(COLM_ABSTRACTED, TOP100_ABSTRACTED, pd):
    def _load(path, corpus):
        df = pd.read_parquet(path)
        df["corpus"] = corpus
        df["gutenberg_id"] = df["gutenberg_id"].astype(str)
        return df

    colm_raw = _load(COLM_ABSTRACTED, "fiction10")
    top_raw = _load(TOP100_ABSTRACTED, "top100")

    shared_cols = [
        c
        for c in colm_raw.columns
        if c in set(top_raw.columns) and c != "corpus"
    ]
    combined_raw = pd.concat(
        [colm_raw[shared_cols + ["corpus"]], top_raw[shared_cols + ["corpus"]]],
        ignore_index=True,
    )

    # A "valid norm" = one with a deontic force assigned. Null force marks an
    # empty / failed extraction row; exclude from distribution stats.
    norms = combined_raw[combined_raw["raz_normative_force"].notna()].copy()

    print("only-in-top100 columns dropped from comparison:",
          sorted(set(top_raw.columns) - set(colm_raw.columns)))
    for c in ("fiction10", "top100"):
        sub_raw = combined_raw[combined_raw["corpus"] == c]
        sub = norms[norms["corpus"] == c]
        print(
            f"{c:>10}: {len(sub_raw):>6,} rows  ->  {len(sub):>6,} valid norms  "
            f"({sub['gutenberg_id'].nunique()} books, "
            f"{len(sub) / sub['gutenberg_id'].nunique():.0f} norms/book)"
        )
    return colm_raw, norms, top_raw


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · Distribution-comparison helpers

    For a categorical axis we want, per corpus, the **proportion** of norms in
    each category (proportions, not counts — the corpora are very different
    sizes). To quantify how far apart two proportion vectors are we report:

    - **TVD** (total variation distance) — `½·Σ|p−q|`, the single largest
      probability mass you'd have to move; ranges 0 (identical) to 1.
    - **JSD** (Jensen–Shannon divergence, base 2) — symmetric, bounded [0, 1],
      sensitive to the whole distribution rather than just the max gap.

    A χ² test of independence would be significant for almost any axis here (N
    is tens of thousands), so we lead with these effect-size measures and treat
    the p-value as a footnote.

    `axis_table(field)` returns a tidy proportion table; `divergence(field)`
    returns `(tvd, jsd, n_fiction10, n_top100)` on the categories present in
    either corpus.
    """)
    return


@app.cell
def _(norms, np, pd):
    def axis_table(field, dropna=True, data=None):
        """Per-corpus count + proportion for each category of `field`.

        `data` defaults to the full `norms` frame; pass a subset (e.g. the
        overlapping-books slice in Part C) to recompute on that slice.
        """
        src = norms if data is None else data
        sub = src[[field, "corpus"]].copy()
        if dropna:
            sub = sub[sub[field].notna()]
        else:
            sub[field] = sub[field].fillna("(none)")
        ct = (
            sub.groupby(["corpus", field]).size().unstack("corpus", fill_value=0)
        )
        for c in ("fiction10", "top100"):
            if c not in ct.columns:
                ct[c] = 0
        ct = ct[["fiction10", "top100"]]
        prop = ct.div(ct.sum(axis=0), axis=1)
        out = pd.DataFrame(
            {
                "fiction10_n": ct["fiction10"],
                "top100_n": ct["top100"],
                "fiction10_p": prop["fiction10"],
                "top100_p": prop["top100"],
            }
        )
        out["delta_p"] = out["top100_p"] - out["fiction10_p"]
        return out.sort_values("fiction10_p", ascending=False)

    def divergence(field, dropna=True, data=None):
        t = axis_table(field, dropna=dropna, data=data)
        p = t["fiction10_p"].to_numpy()
        q = t["top100_p"].to_numpy()
        tvd = 0.5 * np.abs(p - q).sum()

        def _kl(a, b):
            mask = a > 0
            return np.sum(a[mask] * np.log2(a[mask] / b[mask]))

        m = 0.5 * (p + q)
        jsd = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
        return dict(
            field=field,
            tvd=tvd,
            jsd=jsd,
            n_fiction10=int(t["fiction10_n"].sum()),
            n_top100=int(t["top100_n"].sum()),
        )

    def plot_axis(ax, field, title, color, label, dropna=True, order=None, data=None):
        t = axis_table(field, dropna=dropna, data=data)
        if order is not None:
            t = t.reindex([c for c in order if c in t.index])
        cats = [str(c) for c in t.index]
        y = np.arange(len(cats))
        bar_h = 0.38
        ax.barh(y - bar_h / 2, t["fiction10_p"], bar_h,
                label=label["fiction10"], color=color["fiction10"])
        ax.barh(y + bar_h / 2, t["top100_p"], bar_h,
                label=label["top100"], color=color["top100"])
        ax.set_yticks(y)
        ax.set_yticklabels(cats, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("proportion of norms")
        d = divergence(field, dropna=dropna, data=data)
        ax.set_title(f"{title}\nTVD={d['tvd']:.3f}  JSD={d['jsd']:.3f}", fontsize=10)
        ax.grid(True, axis="x", alpha=0.3)
        return t

    return axis_table, divergence, plot_axis


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · Deontic modality — `raz_normative_force`

    What *kind* of "ought" does each extractor surface? The five Raz forces
    span obligation through prohibition to mere permission.
    """)
    return


@app.cell
def _(COLOR, LABEL, axis_table, plot_axis, plt):
    FORCE_ORDER = ["obligatory", "recommended", "permitted", "discouraged", "prohibited"]
    _fig, _ax = plt.subplots(figsize=(8, 4.2))
    _t = plot_axis(_ax, "raz_normative_force", "Normative force", COLOR, LABEL,
                   order=FORCE_ORDER)
    _ax.legend(loc="lower right", fontsize=8)
    _fig.tight_layout()
    print(axis_table("raz_normative_force").to_string())
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · Information-flow governance — `raz_governs_info_flow`

    The fraction of norms that regulate *information transmission* is the
    CI-relevant slice and the one most load-bearing for the paper's thesis. A
    large gap here means the two extractions disagree about how much of fiction's
    normative content is about who-may-tell-whom-what.
    """)
    return


@app.cell
def _(COLOR, LABEL, axis_table, divergence, plot_axis, plt):
    _fig, _ax = plt.subplots(figsize=(7, 2.8))
    plot_axis(_ax, "raz_governs_info_flow", "Governs information flow", COLOR, LABEL)
    _ax.legend(loc="lower right", fontsize=8)
    _fig.tight_layout()
    _t = axis_table("raz_governs_info_flow")
    print(_t.to_string())
    _d = divergence("raz_governs_info_flow")
    print(
        f"\ninfo-flow norm rate:  fiction10 = {_t.loc[True, 'fiction10_p']:.1%}   "
        f"top100 = {_t.loc[True, 'top100_p']:.1%}   "
        f"(x{_t.loc[True, 'top100_p'] / _t.loc[True, 'fiction10_p']:.2f})"
    )
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 · Explicit vs. implicit — `raz_norm_source`

    Does the model read norms off the surface text (`explicit`) or infer them
    from the scene (`implicit`)? A higher implicit share means more
    interpretive extraction.
    """)
    return


@app.cell
def _(COLOR, LABEL, axis_table, plot_axis, plt):
    _fig, _ax = plt.subplots(figsize=(7, 2.8))
    plot_axis(_ax, "raz_norm_source", "Norm source", COLOR, LABEL)
    _ax.legend(loc="lower right", fontsize=8)
    _fig.tight_layout()
    print(axis_table("raz_norm_source").to_string())
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6 · Societal domain — `raz_context`

    Which spheres of life the norms govern. This axis is the most exposed to
    the **corpus** confound (different books → different settings), so read it
    as a joint corpus×model fingerprint rather than a pure model effect.
    """)
    return


@app.cell
def _(COLOR, LABEL, axis_table, plot_axis, plt):
    _t_full = axis_table("raz_context")
    _top_ctx = (
        (_t_full["fiction10_p"] + _t_full["top100_p"]).sort_values(ascending=False).head(12).index
    )
    _fig, _ax = plt.subplots(figsize=(8, 5.5))
    plot_axis(_ax, "raz_context", "Societal domain (top 12 by pooled share)",
              COLOR, LABEL, order=list(_top_ctx))
    _ax.legend(loc="lower right", fontsize=8)
    _fig.tight_layout()
    print(_t_full.to_string())
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7 · Extractor confidence — `raz_confidence_qual` / `raz_confidence_quant`

    Two views of how sure the extractor was. Left: the ordinal qualitative
    label. Right: the 0–10 numeric score as a normalized histogram.
    """)
    return


@app.cell
def _(COLOR, LABEL, norms, np, plot_axis, plt):
    _fig, _axes = plt.subplots(1, 2, figsize=(13, 4))

    CONF_ORDER = ["very_uncertain", "uncertain", "somewhat_certain", "certain", "very_certain"]
    plot_axis(_axes[0], "raz_confidence_qual", "Confidence (qualitative)",
              COLOR, LABEL, order=CONF_ORDER)
    _axes[0].legend(loc="lower right", fontsize=8)

    _ax = _axes[1]
    _bins = np.arange(0.5, 11.5, 1.0)
    for _c in ("fiction10", "top100"):
        _v = norms.loc[norms["corpus"] == _c, "raz_confidence_quant"].dropna()
        _ax.hist(_v, bins=_bins, density=True, alpha=0.55,
                 label=f"{LABEL[_c]}  (μ={_v.mean():.2f})", color=COLOR[_c])
    _ax.set_xlabel("raz_confidence_quant (0–10)")
    _ax.set_ylabel("density")
    _ax.set_title("Confidence (numeric)")
    _ax.legend(loc="upper left", fontsize=8)
    _ax.grid(True, axis="y", alpha=0.3)

    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8 · Generalizability gate — `norm_quality_passed`

    `norm_quality_passed=True` means the norm carried no character/plot-specific
    leakage (no character names or titled names in the norm fields) — i.e. it
    generalizes. A large gap here is a **quality** signal: the cleaner extractor
    produces more reusable norms.
    """)
    return


@app.cell
def _(COLOR, LABEL, axis_table, plot_axis, plt):
    _fig, _ax = plt.subplots(figsize=(7, 2.8))
    plot_axis(_ax, "norm_quality_passed", "Quality gate passed", COLOR, LABEL)
    _ax.legend(loc="lower right", fontsize=8)
    _fig.tight_layout()
    _t = axis_table("norm_quality_passed")
    print(_t.to_string())
    print(
        f"\npass rate:  fiction10 = {_t.loc[True, 'fiction10_p']:.1%}   "
        f"top100 = {_t.loc[True, 'top100_p']:.1%}"
    )
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9 · Divergence summary

    All axes ranked by Jensen–Shannon divergence — a single-glance answer to
    *which attribute shifted most* between the two extractions. χ² is reported
    for completeness but, given N, is significant essentially everywhere.
    """)
    return


@app.cell
def _(axis_table, divergence, np, pd):
    AXES = [
        "raz_normative_force",
        "raz_governs_info_flow",
        "raz_norm_source",
        "raz_context",
        "raz_confidence_qual",
        "norm_quality_passed",
    ]

    def _chi2(field):
        t = axis_table(field)
        obs = t[["fiction10_n", "top100_n"]].to_numpy(dtype=float)
        obs = obs[obs.sum(axis=1) > 0]
        row = obs.sum(axis=1, keepdims=True)
        col = obs.sum(axis=0, keepdims=True)
        exp = row @ col / obs.sum()
        stat = np.sum((obs - exp) ** 2 / exp)
        dof = (obs.shape[0] - 1) * (obs.shape[1] - 1)
        return stat, dof

    _rows = []
    for _f in AXES:
        _d = divergence(_f)
        _stat, _dof = _chi2(_f)
        _d["chi2"] = _stat
        _d["dof"] = _dof
        _rows.append(_d)
    summary = (
        pd.DataFrame(_rows)
        .sort_values("jsd", ascending=False)
        .reset_index(drop=True)
    )
    print(summary.to_string(index=False))
    return AXES, summary


@app.cell
def _(AXES, divergence, plt):
    _d = sorted([divergence(f) for f in AXES], key=lambda r: r["jsd"])
    _fig, _ax = plt.subplots(figsize=(8, 4))
    _y = range(len(_d))
    _ax.barh(list(_y), [r["jsd"] for r in _d], color="#555555")
    _ax.barh([i + 0.0 for i in _y], [r["tvd"] for r in _d], height=0.0)  # keep TVD off the bar
    for _i, _r in enumerate(_d):
        _ax.text(_r["jsd"] + 0.005, _i, f"JSD {_r['jsd']:.3f} · TVD {_r['tvd']:.3f}",
                 va="center", fontsize=8)
    _ax.set_yticks(list(_y))
    _ax.set_yticklabels([r["field"] for r in _d], fontsize=9)
    _ax.set_xlabel("Jensen–Shannon divergence (base 2)")
    _ax.set_title("Distribution shift by axis: top100 vs fiction10")
    _ax.set_xlim(0, max(r["jsd"] for r in _d) * 1.35)
    _ax.grid(True, axis="x", alpha=0.3)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10 · Composition control — per-book-averaged proportions

    The pooled proportions above weight each *norm* equally, so the 97-book
    top100 set is dominated by a few prolific books and the corpora's different
    book mixes leak into every axis. As a partial control, recompute each
    corpus's distribution as the **mean of per-book proportions** (each book
    contributes equally), and compare the resulting JSD to the pooled JSD.

    If per-book JSD ≈ pooled JSD, the shift is not an artifact of a handful of
    dominant books. If it collapses, the pooled view was composition-driven.
    """)
    return


@app.cell
def _(AXES, norms, np, pd):
    def _perbook_dist(field):
        sub = norms[[field, "corpus", "gutenberg_id"]].dropna(subset=[field])
        # count per (corpus, book, category)
        cnt = (
            sub.groupby(["corpus", "gutenberg_id", field], observed=True)
            .size()
            .rename("n")
            .reset_index()
        )
        # within-book proportion
        cnt["book_tot"] = cnt.groupby(["corpus", "gutenberg_id"])["n"].transform("sum")
        cnt["p"] = cnt["n"] / cnt["book_tot"]
        # average the per-book proportion across books in each corpus
        avg = (
            cnt.groupby(["corpus", field])["p"].mean().unstack("corpus", fill_value=0.0)
        )
        for c in ("fiction10", "top100"):
            if c not in avg.columns:
                avg[c] = 0.0
        # renormalize (a category absent in some books shrinks its mean)
        avg = avg / avg.sum(axis=0)
        return avg

    def _jsd_from_props(p, q):
        p, q = np.asarray(p), np.asarray(q)
        m = 0.5 * (p + q)

        def _kl(a, b):
            mask = a > 0
            return np.sum(a[mask] * np.log2(a[mask] / b[mask]))

        return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)

    _rows = []
    for _f in AXES:
        avg = _perbook_dist(_f)
        _rows.append(
            dict(
                field=_f,
                jsd_perbook=_jsd_from_props(avg["fiction10"], avg["top100"]),
            )
        )
    perbook_summary = pd.DataFrame(_rows)
    print(perbook_summary.to_string(index=False))
    return (perbook_summary,)


@app.cell
def _(perbook_summary, summary):
    cmp = summary[["field", "jsd"]].rename(columns={"jsd": "jsd_pooled"}).merge(
        perbook_summary, on="field"
    )
    cmp["ratio_perbook_over_pooled"] = cmp["jsd_perbook"] / cmp["jsd_pooled"]
    print("Pooled vs per-book-averaged JSD (ratio ~1 ⇒ not a composition artifact):")
    print(cmp.sort_values("jsd_pooled", ascending=False).to_string(index=False))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11 · Read-out

    Run §9 and §10. Headline numbers observed at authorship (2026-06-01,
    11,554 valid fiction10 norms vs 21,517 top100):

    - **`norm_quality_passed`** is the largest shift: fiction10 passes the
      generalizability gate ~41% of the time vs ~91% for top100. The qwen3.6
      extraction leaks far less character/plot specificity — a genuine quality
      gap, not a corpus effect (it survives the per-book control in §10).
    - **`raz_governs_info_flow`** roughly **doubles** in top100 (~26% of norms
      vs ~11%). More of the qwen3.6 norms are CI-relevant — the slice the paper
      cares about — but check §10 before attributing this to the model vs the
      broader 97-book corpus.
    - **`raz_normative_force`** shifts toward *prohibition*: top100 surfaces a
      markedly larger `prohibited` share, fiction10 leans `recommended`.
    - **`raz_norm_source`** shifts toward *implicit*: top100 infers more norms
      from scene rather than reading them off the surface.
    - **`raz_confidence_*`**: top100's numeric confidence is slightly higher on
      average but with a heavier tail of low-confidence norms (more
      `uncertain`/`very_certain` mass at both ends).

    **Caveats.** (1) Corpus ≠ model: 97 books vs 10, partial overlap — §10's
    per-book control mitigates but does not eliminate this. (2) Both files are
    post role-abstraction; the quality gate reflects the *combined*
    extract→abstract pipeline, not extraction alone. (3) Null-force rows (315
    fiction10 / 33 top100) are excluded; they are not part of any distribution.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Part B · Underlying corpus comparison

    Everything above compares the *extracted norms*. This part compares the
    *books themselves* — the difference in what each model was asked to read.
    Analysis here is **book-level** (one row per book, unweighted by norm
    count), so it characterizes corpus composition rather than norm density.

    Metadata source: the Gutenberg catalog snapshot
    `gutenberg_cache/catalog/catalog_latest.parquet` (6,400 books), joined on
    `gutenberg_id`.

    - **Genre** — the `bookshelves` field's `Category: …` labels (multi-label;
      a book can carry several).
    - **Author** — first listed author from the catalog `authors` field.
    - **Publication era** — *proxy only.* Gutenberg records no publication
      year, so we use the **author's birth and death years** as an era stand-in.
      Read these as "when the author lived," not "when the book came out."

    > **Coverage caveat.** All 97 top100 books are in the catalog. Of the 10
    > fiction10 books, 9 are; *Nineteen Eighty-Four* (id 1984) is a custom add
    > absent from the Gutenberg snapshot — its metadata is supplied manually
    > below and flagged. Several books belong to **both** corpora (the fiction10
    > set partly overlaps top100); such a book contributes one row to each
    > corpus, which is correct for per-corpus distributions.
    """)
    return


@app.cell
def _(colm_raw, np, pd, top_raw):
    import json as _json

    CATALOG = (
        "/share/pierson/matt/zoo/datasets/gutenberg_cache/catalog/catalog_latest.parquet"
    )
    cat = pd.read_parquet(CATALOG)
    cat["gid"] = cat["gutenberg_id"].astype(str)

    # Manual supplement for books absent from the catalog snapshot.
    MANUAL_META = {
        "1984": dict(
            title="Nineteen Eighty-Four",
            author="Orwell, George",
            birth_year=1903,
            death_year=1950,
            genres=["British Literature", "Science-Fiction & Fantasy", "Novels"],
        ),
    }

    def _first_author(a):
        try:
            lst = _json.loads(a) if isinstance(a, str) else a
            return lst[0] if lst else {}
        except Exception:
            return {}

    def _genres(b):
        try:
            lst = _json.loads(b) if isinstance(b, str) else b
            return [
                x.replace("Category: ", "")
                for x in lst
                if isinstance(x, str) and x.startswith("Category:")
            ]
        except Exception:
            return []

    _cat_ix = cat.set_index("gid")

    def _meta_for(gid):
        gid = str(gid)
        if gid in _cat_ix.index:
            row = _cat_ix.loc[gid]
            au = _first_author(row["authors"])
            return dict(
                gid=gid,
                title=str(row["title"]),
                author=au.get("name"),
                birth_year=au.get("birth_year"),
                death_year=au.get("death_year"),
                genres=_genres(row["bookshelves"]),
                in_catalog=True,
            )
        if gid in MANUAL_META:
            mm = MANUAL_META[gid]
            return dict(
                gid=gid,
                title=mm["title"],
                author=mm["author"],
                birth_year=mm["birth_year"],
                death_year=mm["death_year"],
                genres=mm["genres"],
                in_catalog=False,
            )
        return dict(
            gid=gid, title=None, author=None, birth_year=None,
            death_year=None, genres=[], in_catalog=False,
        )

    def _book_meta(df, corpus):
        ids = sorted(df["gutenberg_id"].astype(str).unique(), key=int)
        recs = [{**_meta_for(g), "corpus": corpus} for g in ids]
        return pd.DataFrame(recs)

    book_meta = pd.concat(
        [_book_meta(colm_raw, "fiction10"), _book_meta(top_raw, "top100")],
        ignore_index=True,
    )

    for _c in ("fiction10", "top100"):
        _sub = book_meta[book_meta["corpus"] == _c]
        print(
            f"{_c:>10}: {len(_sub):>3} books  "
            f"({int(_sub['in_catalog'].sum())} from catalog, "
            f"{int((~_sub['in_catalog']).sum())} manual/missing)  "
            f"| {_sub['author'].nunique()} unique authors  "
            f"| death-year median {np.nanmedian(_sub['death_year'].astype(float)):.0f}"
        )
    return (book_meta,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 13 · Genre distribution

    Genres are multi-label, so two complementary views:

    - **% of books carrying each genre tag** (bars) — interpretable; columns do
      not sum to 1 since a book has several tags.
    - **JSD on tag-share** — normalize each corpus's total genre-tag counts to a
      probability vector and measure divergence, for a single shift number.
    """)
    return


@app.cell
def _(COLOR, LABEL, book_meta, np, pd, plt):
    def _genre_long(bm):
        recs = []
        for _, r in bm.iterrows():
            for g in (r["genres"] or []):
                recs.append({"corpus": r["corpus"], "genre": g})
        return pd.DataFrame(recs)

    _gl = _genre_long(book_meta)
    _n_books = book_meta.groupby("corpus")["gid"].nunique()
    # tag counts per corpus
    _pct = _gl.groupby(["corpus", "genre"]).size().unstack("corpus", fill_value=0)
    for _c in ("fiction10", "top100"):
        if _c not in _pct.columns:
            _pct[_c] = 0
    # share of books carrying each tag (multi-label; columns need not sum to 1)
    _pct_books = _pct.copy().astype(float)
    for _c in ("fiction10", "top100"):
        _pct_books[_c] = _pct_books[_c] / _n_books[_c]

    _order = (
        (_pct_books["fiction10"] + _pct_books["top100"])
        .sort_values(ascending=False)
        .head(14)
        .index
    )
    _pb = _pct_books.loc[_order]

    _fig, _ax = plt.subplots(figsize=(8, 6))
    _y = np.arange(len(_pb))
    _bar_h = 0.38
    _ax.barh(_y - _bar_h / 2, _pb["fiction10"], _bar_h,
             label=LABEL["fiction10"], color=COLOR["fiction10"])
    _ax.barh(_y + _bar_h / 2, _pb["top100"], _bar_h,
             label=LABEL["top100"], color=COLOR["top100"])
    _ax.set_yticks(_y)
    _ax.set_yticklabels(_pb.index, fontsize=9)
    _ax.invert_yaxis()
    _ax.set_xlabel("share of books carrying genre tag")

    # JSD on normalized tag-share
    _sh = _pct.astype(float)
    _p = (_sh["fiction10"] / _sh["fiction10"].sum()).to_numpy()
    _q = (_sh["top100"] / _sh["top100"].sum()).to_numpy()
    _mm = 0.5 * (_p + _q)

    def _kl(a, b):
        mask = a > 0
        return np.sum(a[mask] * np.log2(a[mask] / b[mask]))

    _jsd_genre = 0.5 * _kl(_p, _mm) + 0.5 * _kl(_q, _mm)
    _ax.set_title(
        f"Genre (top 14 by pooled book share)\n"
        f"tag-share JSD = {_jsd_genre:.3f}  |  "
        f"{int(_sh['fiction10'].astype(bool).sum())} vs "
        f"{int(_sh['top100'].astype(bool).sum())} distinct genres",
        fontsize=10,
    )
    _ax.legend(loc="lower right", fontsize=8)
    _ax.grid(True, axis="x", alpha=0.3)
    _fig.tight_layout()
    print(
        _pct_books.loc[_order]
        .rename(columns={"fiction10": "fiction10_book_share",
                         "top100": "top100_book_share"})
        .to_string()
    )
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 14 · Author distribution

    Books-per-author concentration, the most prolific authors in each corpus,
    and the author overlap between the two sets.
    """)
    return


@app.cell
def _(book_meta):
    auth = book_meta.dropna(subset=["author"])
    _per_corpus = (
        auth.groupby("corpus")
        .agg(n_books=("gid", "nunique"), n_authors=("author", "nunique"))
        .assign(books_per_author=lambda d: d["n_books"] / d["n_authors"])
    )
    print("Author concentration:")
    print(_per_corpus.to_string())

    _f10 = set(auth.loc[auth["corpus"] == "fiction10", "author"])
    _t100 = set(auth.loc[auth["corpus"] == "top100", "author"])
    print(
        f"\nAuthor-set overlap: {len(_f10 & _t100)} shared "
        f"(of {len(_f10)} fiction10, {len(_t100)} top100)"
    )
    print("fiction10 authors NOT in top100:", sorted(_f10 - _t100))

    print("\nMost prolific top100 authors (by #books):")
    print(
        auth[auth["corpus"] == "top100"]["author"].value_counts().head(12).to_string()
    )

    print("\nfiction10 authors (by #books):")
    print(
        auth[auth["corpus"] == "fiction10"]["author"].value_counts().to_string()
    )
    return (auth,)


@app.cell
def _(COLOR, LABEL, auth, np, plt):
    # Distribution of books-per-author within each corpus
    _bpa = auth.groupby(["corpus", "author"]).size().rename("n").reset_index()
    _fig, _ax = plt.subplots(figsize=(7, 3.5))
    _bins = np.arange(0.5, _bpa["n"].max() + 1.5, 1.0)
    for _c in ("fiction10", "top100"):
        _v = _bpa.loc[_bpa["corpus"] == _c, "n"]
        _ax.hist(_v, bins=_bins, density=True, alpha=0.55,
                 label=f"{LABEL[_c]}", color=COLOR[_c])
    _ax.set_xlabel("books per author")
    _ax.set_ylabel("density")
    _ax.set_title("Author productivity within corpus")
    _ax.legend(fontsize=8)
    _ax.grid(True, axis="y", alpha=0.3)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 15 · Publication-era distribution (author birth/death proxy)

    Gutenberg has no publication year, so this uses the **author's life span**
    as an era proxy. Left: author *death* year (the better single proxy — most
    works land in the latter part of a career). Right: author *birth* year.
    Overlaid normalized histograms; medians annotated.
    """)
    return


@app.cell
def _(COLOR, LABEL, book_meta, np, plt):
    _fig, _axes = plt.subplots(1, 2, figsize=(13, 4))
    for _ax, _field, _name in [
        (_axes[0], "death_year", "author death year"),
        (_axes[1], "birth_year", "author birth year"),
    ]:
        _vals = {}
        for _c in ("fiction10", "top100"):
            _vals[_c] = (
                book_meta.loc[book_meta["corpus"] == _c, _field]
                .astype("float64")
                .dropna()
            )
        _lo = int(min(s.min() for s in _vals.values()) // 25 * 25)
        _hi = int(max(s.max() for s in _vals.values()) // 25 * 25 + 25)
        _bins = np.arange(_lo, _hi + 1, 25)
        for _c in ("fiction10", "top100"):
            _v = _vals[_c]
            _ax.hist(_v, bins=_bins, density=True, alpha=0.55,
                     label=f"{LABEL[_c]}  (med {_v.median():.0f}, n={len(_v)})",
                     color=COLOR[_c])
        _ax.set_xlabel(_name + "  (publication-era proxy)")
        _ax.set_ylabel("density")
        _ax.set_title(_name.title())
        _ax.legend(fontsize=8)
        _ax.grid(True, axis="y", alpha=0.3)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(book_meta, pd):
    # Numeric era summary
    _recs = []
    for _c in ("fiction10", "top100"):
        _sub = book_meta[book_meta["corpus"] == _c]
        for _field in ("birth_year", "death_year"):
            _v = _sub[_field].astype("float64").dropna()
            _recs.append(
                dict(
                    corpus=_c,
                    field=_field,
                    n=len(_v),
                    min=_v.min(),
                    median=_v.median(),
                    max=_v.max(),
                )
            )
    print(pd.DataFrame(_recs).to_string(index=False))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 16 · Corpus-comparison read-out

    Book-level differences between the two reading lists (genre, author, era)
    are the *upstream* explanation for any norm-distribution shift in Part A
    that survives the §10 per-book control. In particular the §6 / §13
    domain-vs-genre story should be read together: a genre mix that leans more
    toward (say) adventure, military, or religious works will mechanically move
    the `raz_context` and `raz_governs_info_flow` distributions regardless of
    extractor behaviour.

    **Caveats restated.** Genre tags are multi-label and Gutenberg-curated
    (coarse). The era axis is an author-lifespan proxy, *not* publication year.
    *Nineteen Eighty-Four* is supplied manually; all other metadata is from the
    catalog snapshot.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Part C · Same-book head-to-head (overlapping corpus only)

    Parts A–B mix two effects: the **extractor model** (qwen3.6-27b vs
    qwen2.5-72b-awq) and the **book set** (97 vs 10 books). To isolate the
    *model* effect, restrict to the books that appear in **both** corpora — then
    the book set is held fixed and the only thing that varies is the extractor.

    The two corpora overlap on **7 books**: *Alice's Adventures in Wonderland*,
    *Les Misérables*, *Middlemarch*, *Bleak House*, *The Count of Monte Cristo*,
    *Pride and Prejudice*, *Anna Karenina* (the 3 fiction10 books missing from
    top100 are *Nineteen Eighty-Four*, *The Picture of Dorian Gray*, and *The
    Age of Innocence* — see Part A's sibling yield-gap notebook).

    Same books, same 6,000-char chunking (established in the yield-gap
    notebook), different extractor. Any distribution gap that **survives** this
    restriction is a model effect; any gap that **collapses** relative to the
    pooled Part-A numbers was driven by corpus composition.

    > Caveat: proportions still differ in support — qwen2.5 extracts ~2.8× more
    > norms per book than qwen3.6 here (10,309 vs 3,748 on the 7 books) — but
    > the distributions are over proportions, so the count gap (covered by the
    > yield-gap notebook) does not distort the shape comparison.
    """)
    return


@app.cell
def _(norms, pd):
    _f10_books = set(norms.loc[norms["corpus"] == "fiction10", "gutenberg_id"])
    _top_books = set(norms.loc[norms["corpus"] == "top100", "gutenberg_id"])
    overlap_ids = sorted(_f10_books & _top_books, key=int)
    norms_overlap = norms[norms["gutenberg_id"].isin(overlap_ids)].copy()

    _titles = (
        norms.drop_duplicates("gutenberg_id").set_index("gutenberg_id")["book_title"]
    )
    _rows = []
    for _g in overlap_ids:
        _sub = norms_overlap[norms_overlap["gutenberg_id"] == _g]
        _rows.append(
            dict(
                gid=_g,
                title=str(_titles.get(_g))[:34],
                fiction10_norms=int((_sub["corpus"] == "fiction10").sum()),
                top100_norms=int((_sub["corpus"] == "top100").sum()),
            )
        )
    overlap_books = pd.DataFrame(_rows)
    print(f"Overlapping books: {len(overlap_ids)}")
    print(overlap_books.to_string(index=False))
    print(
        f"\nTotal overlap norms:  fiction10 = "
        f"{overlap_books['fiction10_norms'].sum():,}  |  "
        f"top100 = {overlap_books['top100_norms'].sum():,}"
    )
    return (norms_overlap,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 17 · Model effect vs pooled shift, per axis

    For each axis, compare the **overlap JSD** (same 7 books → pure model
    effect) against the **pooled JSD** from §9 (all books → model + corpus). The
    ratio `jsd_overlap / jsd_pooled` reads as:

    - **≈ 1** — the Part-A shift is essentially a model effect; the books didn't
      matter.
    - **< 1** — part of the Part-A shift was corpus composition; it shrinks once
      the book set is held fixed.
    - **> 1** — the shift is *larger* on the shared books (the extra top100
      books happened to dampen it).
    """)
    return


@app.cell
def _(AXES, divergence, norms_overlap, pd, summary):
    _rows = []
    for _f in AXES:
        _d = divergence(_f, data=norms_overlap)
        _rows.append(
            dict(
                field=_f,
                jsd_overlap=_d["jsd"],
                tvd_overlap=_d["tvd"],
                n_fiction10_ov=_d["n_fiction10"],
                n_top100_ov=_d["n_top100"],
            )
        )
    ov_summary = (
        pd.DataFrame(_rows)
        .merge(
            summary[["field", "jsd", "tvd"]].rename(
                columns={"jsd": "jsd_pooled", "tvd": "tvd_pooled"}
            ),
            on="field",
        )
    )
    ov_summary["jsd_ratio_ov_over_pooled"] = (
        ov_summary["jsd_overlap"] / ov_summary["jsd_pooled"]
    )
    ov_summary = ov_summary.sort_values("jsd_overlap", ascending=False).reset_index(
        drop=True
    )
    print(
        "Overlap (model-only) vs pooled (model+corpus) divergence, per axis:\n"
    )
    print(
        ov_summary[
            [
                "field",
                "jsd_overlap",
                "jsd_pooled",
                "jsd_ratio_ov_over_pooled",
                "n_fiction10_ov",
                "n_top100_ov",
            ]
        ].to_string(index=False)
    )
    return (ov_summary,)


@app.cell
def _(np, ov_summary, plt):
    _o = ov_summary.sort_values("jsd_overlap")
    _y = np.arange(len(_o))
    _bar_h = 0.38
    _fig, _ax = plt.subplots(figsize=(8.5, 4.2))
    _ax.barh(_y - _bar_h / 2, _o["jsd_pooled"], _bar_h,
             label="pooled (model + corpus)", color="#bbbbbb")
    _ax.barh(_y + _bar_h / 2, _o["jsd_overlap"], _bar_h,
             label="overlap (model only, 7 books)", color="#d62728")
    _ax.set_yticks(_y)
    _ax.set_yticklabels(_o["field"], fontsize=9)
    _ax.set_xlabel("Jensen–Shannon divergence (base 2)")
    _ax.set_title("Model effect (same books) vs pooled shift, per axis")
    _ax.legend(loc="lower right", fontsize=8)
    _ax.grid(True, axis="x", alpha=0.3)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 18 · Per-axis distributions on the shared books

    The same axes as Part A, recomputed on the 7 overlapping books only. Bars
    are proportions; titles carry the same-book TVD/JSD (pure model effect).
    """)
    return


@app.cell
def _(COLOR, LABEL, axis_table, norms_overlap, plot_axis, plt):
    _fig, _axes = plt.subplots(2, 3, figsize=(16, 9))

    _FORCE = ["obligatory", "recommended", "permitted", "discouraged", "prohibited"]
    _CONF = ["very_uncertain", "uncertain", "somewhat_certain", "certain", "very_certain"]
    _ctx_t = axis_table("raz_context", data=norms_overlap)
    _ctx_order = list(
        (_ctx_t["fiction10_p"] + _ctx_t["top100_p"]).sort_values(ascending=False).head(8).index
    )

    plot_axis(_axes[0, 0], "raz_normative_force", "Normative force",
              COLOR, LABEL, order=_FORCE, data=norms_overlap)
    plot_axis(_axes[0, 1], "raz_governs_info_flow", "Governs information flow",
              COLOR, LABEL, data=norms_overlap)
    plot_axis(_axes[0, 2], "raz_norm_source", "Norm source",
              COLOR, LABEL, data=norms_overlap)
    plot_axis(_axes[1, 0], "raz_confidence_qual", "Confidence (qualitative)",
              COLOR, LABEL, order=_CONF, data=norms_overlap)
    plot_axis(_axes[1, 1], "norm_quality_passed", "Quality gate passed",
              COLOR, LABEL, data=norms_overlap)
    plot_axis(_axes[1, 2], "raz_context", "Societal domain (top 8)",
              COLOR, LABEL, order=_ctx_order, data=norms_overlap)

    _axes[0, 0].legend(loc="lower right", fontsize=8)
    _fig.tight_layout()

    _ifr = axis_table("raz_governs_info_flow", data=norms_overlap)
    _qp = axis_table("norm_quality_passed", data=norms_overlap)
    print(
        "On the 7 shared books:\n"
        f"  info-flow norm rate:  fiction10 = {_ifr.loc[True, 'fiction10_p']:.1%}   "
        f"top100 = {_ifr.loc[True, 'top100_p']:.1%}\n"
        f"  quality-pass rate:    fiction10 = {_qp.loc[True, 'fiction10_p']:.1%}   "
        f"top100 = {_qp.loc[True, 'top100_p']:.1%}"
    )
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 19 · Same-book read-out

    Read §17 first. The overlap JSD is the **clean model comparison** — corpus
    composition is held fixed across the 7 shared books, so whatever divergence
    remains is attributable to the extractor (qwen3.6-27b vs qwen2.5-72b-awq).

    **Headline (observed at authorship):** *every* axis keeps **0.82–1.02× of
    its pooled JSD** on the shared books. Nothing collapses. So the Part-A
    distribution shifts are **predominantly a model effect**, not corpus
    composition — restricting to identical books barely moves the divergences.

    Axis by axis:

    - **`norm_quality_passed`** — the largest gap and it persists (JSD 0.19 vs
      pooled 0.22; pass rate **38.7% → 86.4%** on identical books). The
      generalizability gap is the extractor, full stop. Corroborates §10.
    - **`raz_governs_info_flow`** — the info-flow rate stays at **10.8% →
      25.0%** (~2.3×) on the shared books. The CI-relevant enrichment is a
      property of the qwen3.6 extractor that **travels to any corpus**, not a
      genre artifact of the 97-book set. This is the load-bearing result for the
      paper.
    - **`raz_context`** — JSD is **essentially unchanged** (ratio ≈ 1.02).
      Contrary to the Part-A §6 expectation that domain was corpus-driven, the
      divergence is the two models using **different context-label
      vocabularies** (qwen3.6's long free-text tail), not the genre mix. Note
      this differs from §10's per-book metric, which is inflated by per-book
      singleton labels; the same-book overlap here is the cleaner read.
    - **`raz_normative_force` / `raz_norm_source` / `raz_confidence_qual`** —
      small but stable shifts (ratios 0.82–0.92): modest, genuine model
      differences.

    Caveat unchanged from Part A: both files are post role-abstraction, so the
    quality gate reflects the combined extract→abstract pipeline.
    """)
    return


if __name__ == "__main__":
    app.run()
