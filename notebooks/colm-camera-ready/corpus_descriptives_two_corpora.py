"""Two-corpus supplemental descriptives for the COLM 2026 camera-ready.

Regenerates the appendix "normative simulacra" descriptive figure suite
(`B_additional-results.tex` sections `app:norm-descriptives` and
`app:corpus-scaling`) with a **comparative lens over both text corpora**:

    fiction10  — 10 setting-heavy public-domain novels (the training corpus)
    top100     — the 100 most-downloaded public-domain fiction works on
                 Project Gutenberg (the book-level holdout / scaling corpus)

Both corpora were extracted by the **same teacher under the same prompts**
(Gemma-4-31B-it, `norm_extraction_fiction` / `ci_extraction_fiction`, both
post-dating the 2026-07-12 prompt-wiring fix). The extractor is therefore held
fixed and every distribution shift measured here is a **corpus-composition
effect** — what was read, not who read it. This is the property that makes the
old appendix figures unusable: they were fiction10-only, drawn from the
Qwen2.5-72B-AWQ lineage, and the one existing two-corpus comparison
(`top100_norm_attributes`, `top100_divergence_axes`) confounded corpus with
extractor.

Companion to `norm_distribution_top100_vs_fiction10.py`, which covers the
categorical norm-attribute axes (deontic force, norm source, governs-info-flow,
quality gate, confidence). This script covers what that notebook does not:
per-chunk extraction yield, the societal-context vocabulary, contextual
entropy, deontic force conditioned on context, and the CI flow side
(flow contexts, flow appropriateness).

Comparing the corpora
    Every corpus comparison here is a two-sided Mann--Whitney U test on
    **per-book** category shares (10 books against 100), Benjamini--Hochberg
    corrected within an axis, with Hodges--Lehmann shift estimates and
    distribution-free intervals for effect size. See `book_level_test`.

    This replaced a JSD/TVD pair reported against a "noise floor" built from
    the seven books the corpora share. Those are descriptive distances with no
    sampling distribution, so the floor was standing in for a test; and both
    they and a pooled chi-square treat 63,526 norms as independent when norms
    are nested in books. Respecting the nesting changes the answer, not just
    the presentation: pooled chi-square calls deontic force (p=6e-5) and norm
    source (p=4e-19) significant, and neither survives.

    Caveat that no test fixes: 7 of the 10 fiction10 books (1023, 11, 1184,
    1342, 135, 1399, 145) are also in top100, so the two samples are not
    independent and every comparison below is conservative.

Outputs
    figures/corpus_descriptives/*.{pdf,png,json}
    tables/corpus_descriptives/*.csv

Run:
    .venv-vllm025cu129/bin/python notebooks/colm-camera-ready/corpus_descriptives_two_corpora.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats

# --------------------------------------------------------------------------
# Provenance — canonical Gemma-4 lineage artifacts (see module docstring).
# --------------------------------------------------------------------------
ROOT = Path("/share/pierson/matt/UAIR")

SOURCES = {
    "fiction10": {
        "norms": ROOT
        / "outputs/2026-07-12_fiction10_norms_gemma4/18-36-28"
        / "COLM_norms_fiction_gemma4/outputs/extraction/structured_norms.parquet",
        "flows": ROOT
        / "outputs/2026-07-12_fiction10_flows_gemma4/23-14-17"
        / "COLM_flows_fiction_gemma4/outputs/ci_extraction/ci_flows.parquet",
        # One row per chunk, whether or not it yielded a flow — the denominator
        # for every per-chunk yield statistic below.
        "chunks": ROOT
        / "outputs/2026-07-12_fiction10_flows_gemma4/23-14-17"
        / "COLM_flows_fiction_gemma4/outputs/ci_reasoning/reasoning.parquet",
    },
    "top100": {
        "norms": ROOT
        / "outputs/2026-07-13_top100_norms_extraction_gemma4/16-23-09"
        / "COLM_norms_fiction_gemma4/outputs/extraction/structured_norms.parquet",
        "flows": ROOT
        / "outputs/2026-07-13_top100_flows_gemma4/16-23-09"
        / "COLM_flows_fiction_gemma4/outputs/ci_extraction/ci_flows.parquet",
        "chunks": ROOT
        / "outputs/2026-07-13_top100_flows_gemma4/16-23-09"
        / "COLM_flows_fiction_gemma4/outputs/ci_reasoning/reasoning.parquet",
    },
}

HERE = Path(__file__).resolve().parent
FIG_DIR = HERE / "figures" / "corpus_descriptives"
TAB_DIR = HERE / "tables" / "corpus_descriptives"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

# Every figure is authored at the paper's text width so \includegraphics
# never has to scale it down (which would shrink the type below legibility).
TW = 6.9  # inches

CORPORA = ("fiction10", "top100")

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
# Muted fills with thin black edges, sampled from the reference figure style the
# paper adopts. Every colour below comes from that sample; nothing is invented,
# so the supplement reads as one system with the rest of the paper.
PAL = {
    "teal": "#498573",
    "mint": "#96cebf",
    "cream": "#e5d2bb",
    "amber": "#f3b14f",
    "coral": "#e57264",
    "blue": "#5674b3",
    "periwinkle": "#a6c0f4",
    "tan": "#a08a6f",
    "green": "#83bd7b",
    "slate": "#a7b3d2",
    "warmgrey": "#7a6f63",
}
EDGE = "#1a1a1a"      # bar/patch outline
EDGE_LW = 0.5

# Corpus identity, reserved so it never collides with any other encoding.
# Blue vs amber keeps the luminance and hue separation of the Okabe-Ito pair it
# replaces, and both are in the sampled palette.
COLOR = {"fiction10": PAL["blue"], "top100": PAL["amber"]}
LABEL = {
    "fiction10": "fiction10 (10 books)",
    "top100": "top100 (100 books)",
}

FORCE_ORDER = ["obligatory", "recommended", "permitted", "discouraged", "prohibited"]
# Ordinal data gets an ordered ramp, not categorical hues: the deontic axis runs
# from positive prescription (teal) through neutral (cream) to negative (coral).
FORCE_COLOR = {
    "obligatory": PAL["teal"],
    "recommended": PAL["mint"],
    "permitted": PAL["cream"],
    "discouraged": PAL["amber"],
    "prohibited": PAL["coral"],
}

APPR_ORDER = ["appropriate", "ambiguous", "inappropriate"]
APPR_COLOR = {
    "appropriate": PAL["mint"],
    "ambiguous": PAL["amber"],
    "inappropriate": PAL["coral"],
}

# Continuous ramp for the heatmaps, built from the same swatches so the
# categorical and continuous marks belong to one palette.
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

CMAP = LinearSegmentedColormap.from_list(
    "normsim",
    [PAL["blue"], PAL["periwinkle"], PAL["cream"], PAL["amber"], PAL["coral"]],
)
CMAP_SEQ = LinearSegmentedColormap.from_list(
    "normsim_seq", ["#f7f3ec", PAL["cream"], PAL["mint"], PAL["teal"]]
)

# COLM camera-ready house style (matches the other camera-ready notebooks).
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.5,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "patch.edgecolor": EDGE,
        "patch.linewidth": EDGE_LW,
        "patch.force_edgecolor": True,
        "hatch.linewidth": 0.5,
    }
)


def _lum(rgb):
    def lin(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = rgb[:3]
    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)


def text_on(fill):
    """Black or white label text, chosen from the fill's relative luminance.

    A hard-coded rule ("white on the bar, black outside") breaks the moment the
    palette changes: several of these fills are light enough that white text on
    them is unreadable. Luminance does not care which palette is in use.
    """
    return EDGE if _lum(mcolors.to_rgb(fill)) > 0.42 else "white"


def on_color(cmap, norm_value):
    """text_on() for a colormap cell, given its normalised value."""
    return text_on(cmap(float(np.clip(norm_value, 0, 1))))


def save_fig(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig]   {FIG_DIR / name}.pdf")


def save_caption(name, title, caption, label, tags):
    (FIG_DIR / f"{name}.json").write_text(
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


def save_table(df, name, index=True):
    df.to_csv(TAB_DIR / f"{name}.csv", index=index)
    print(f"[table] {TAB_DIR / name}.csv")


# --------------------------------------------------------------------------
# Context canonicalisation
# --------------------------------------------------------------------------
# The fiction extraction prompt leaves `raz_context` / `ci_context` as free
# text, and the teacher frequently emits slash-joined composites
# ("gender/social propriety"). Raw strings are therefore long-tailed (802
# distinct in fiction10, 3,813 in top100) and not directly comparable across
# corpora. We canonicalise by splitting composites into facets and attributing
# the norm/flow to each facet (multi-label), after protecting the one composite
# the teacher uses as an *atomic* label.
_ATOMIC = {"class/status": "class-status"}
_ALIAS = {
    # The teacher writes the class facet four ways; they are one concept and
    # splitting them would understate a top-3 context in both corpora.
    "class status": "class-status",
    "class-status": "class-status",
    "class": "class-status",
    "status": "class-status",
    "legal conduct": "legal",
    "moral conduct": "morality",
    "military conduct": "military",
    "religious conduct": "religious observance",
}
_SPLIT = re.compile(r"[/;,]|\band\b")


def facets(value: str) -> list[str]:
    """Canonical context facets for one free-text context string."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    s = str(value).strip().lower()
    if not s:
        return []
    for atomic, token in _ATOMIC.items():
        s = s.replace(atomic, token)
    out, seen = [], set()
    for part in _SPLIT.split(s):
        part = re.sub(r"\s+", " ", part).strip(" -")
        if not part:
            continue
        part = _ALIAS.get(part, part)
        if part not in seen:
            seen.add(part)
            out.append(part)
    return out


def pretty(facet: str) -> str:
    """Back-compat alias; display() is the single labelling entry point."""
    return display(facet)


# --------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------
NORM_COLS = [
    "gutenberg_id",
    "chunk_id",
    "book_title",
    "book_author",
    "raz_normative_force",
    "raz_context",
    "raz_governs_info_flow",
    "raz_norm_source",
    "raz_confidence_qual",
    "norm_quality_passed",
    "prompt_name",
]
FLOW_COLS = [
    "gutenberg_id",
    "chunk_id",
    "book_title",
    "ci_context",
    "ci_appropriateness",
    "prompt_name",
]
CHUNK_COLS = ["gutenberg_id", "chunk_id", "book_title", "has_information_exchange"]


def _load(kind: str, cols: list[str], prompt_expect: str | None) -> pd.DataFrame:
    frames = []
    for corpus in CORPORA:
        df = pd.read_parquet(SOURCES[corpus][kind], columns=cols)
        df["corpus"] = corpus
        df["gutenberg_id"] = df["gutenberg_id"].astype(str)
        if prompt_expect is not None:
            seen = set(df["prompt_name"].dropna().unique())
            assert seen == {prompt_expect}, f"{corpus}/{kind}: prompt_name={seen}"
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


print("loading …")
norms_raw = _load("norms", NORM_COLS, "norm_extraction_fiction")
# A "valid norm" = one with a deontic force assigned; a null force marks an
# empty/failed extraction row.
norms = norms_raw[norms_raw["raz_normative_force"].notna()].copy()
flows = _load("flows", FLOW_COLS, "ci_extraction_fiction")
chunks = _load("chunks", CHUNK_COLS, None)

# Per-chunk keys are book-local indices, so the corpus-wide key is the pair.
for _df in (norms, flows, chunks):
    _df["book_key"] = _df["gutenberg_id"] + ":" + _df["chunk_id"].astype(str)

for c in CORPORA:
    n, f, k = (d[d.corpus == c] for d in (norms, flows, chunks))
    print(
        f"  {c:>10}: {k.gutenberg_id.nunique():>3} books  "
        f"{len(k):>6,} chunks  {len(n):>6,} norms  {len(f):>6,} flows"
    )


# --------------------------------------------------------------------------
# Per-book yield table (also the source for tab:source-texts and the top-100
# corpus-composition table)
# --------------------------------------------------------------------------
def per_book_table() -> pd.DataFrame:
    ch = (
        chunks.groupby(["corpus", "gutenberg_id"])
        .agg(
            book_title=("book_title", "first"),
            chunks=("chunk_id", "nunique"),
            flow_chunks=("has_information_exchange", "sum"),
        )
        .reset_index()
    )
    nm = (
        norms.groupby(["corpus", "gutenberg_id"])
        .agg(norms=("raz_normative_force", "size"))
        .reset_index()
    )
    fl = (
        flows.groupby(["corpus", "gutenberg_id"])
        .agg(flows=("ci_appropriateness", "size"))
        .reset_index()
    )
    ci = (
        norms[norms["raz_governs_info_flow"] == True]  # noqa: E712
        .groupby(["corpus", "gutenberg_id"])
        .size()
        .rename("ci_norms")
        .reset_index()
    )
    out = (
        ch.merge(nm, on=["corpus", "gutenberg_id"], how="left")
        .merge(fl, on=["corpus", "gutenberg_id"], how="left")
        .merge(ci, on=["corpus", "gutenberg_id"], how="left")
        .fillna({"norms": 0, "flows": 0, "ci_norms": 0})
    )
    out["norms_per_chunk"] = out["norms"] / out["chunks"]
    out["flows_per_chunk"] = out["flows"] / out["chunks"]
    out["flow_chunk_rate"] = out["flow_chunks"] / out["chunks"]
    out["ci_norm_share"] = out["ci_norms"] / out["norms"].replace(0, np.nan)
    return out


books = per_book_table()
save_table(books.sort_values(["corpus", "gutenberg_id"]), "per_book_yield", index=False)

F10_ORDER = (
    books[books.corpus == "fiction10"].sort_values("norms_per_chunk").gutenberg_id.tolist()
)
F10_TITLE = dict(
    zip(
        books[books.corpus == "fiction10"].gutenberg_id,
        books[books.corpus == "fiction10"].book_title,
    )
)
# Short display names for the ten training books.
SHORT = {
    "1984": "1984",
    "541": "Age of Innocence",
    "11": "Alice's Adventures in Wonderland",
    "1399": "Anna Karenina",
    "1023": "Bleak House",
    "1184": "Monte Cristo",
    "135": "Les Misérables",
    "145": "Middlemarch",
    "4078": "Dorian Gray",
    "1342": "Pride and Prejudice",
}


def short(gid: str) -> str:
    return SHORT.get(gid, F10_TITLE.get(gid, gid))


# --------------------------------------------------------------------------
# Display names
# --------------------------------------------------------------------------
# Schema field names and snake_case category values are internal keys and are
# never what a reader should see. Everything that reaches a figure label, a
# panel title or a table header goes through display() / axis_label() first.
AXIS_LABEL = {
    "raz_normative_force": "Normative force",
    "raz_governs_info_flow": "Governs information flow",
    "raz_norm_source": "Norm source",
    "raz_confidence_qual": "Extractor confidence",
    "norm_quality_passed": "Generalizability gate",
    "raz_context": "Societal context",
    "ci_appropriateness": "Flow appropriateness",
    "ci_context": "Flow context",
    "facet": "Context facet",
}
CATEGORY_LABEL = {
    True: "Yes",
    False: "No",
    "class-status": "Class/status",
}


def display(value) -> str:
    """Reader-facing label for one category value or context facet.

    Falls through to sentence case, which is right for every free-text facet
    the teacher emits ("social propriety") and for the snake_case confidence
    levels ("somewhat_certain" -> "Somewhat certain").
    """
    try:
        if value in CATEGORY_LABEL:
            return CATEGORY_LABEL[value]
    except TypeError:  # unhashable
        pass
    s = str(value).replace("_", " ").strip()
    return s[:1].upper() + s[1:]


def axis_label(field: str) -> str:
    return AXIS_LABEL.get(field, display(field))


# --------------------------------------------------------------------------
# Book-level comparison
# --------------------------------------------------------------------------
def book_level_test(
    df: pd.DataFrame,
    field: str,
    categories=None,
    *,
    min_count: int = 0,
    min_books: int = 3,
) -> pd.DataFrame:
    """Compare the two corpora on one categorical axis, book by book.

    The unit of analysis is the **book**, not the norm. Norms are nested in
    books, so a test over pooled norms treats 63k correlated observations as
    independent and will call almost any axis significant: pooled chi-square
    puts deontic force at p=6e-5 and norm source at p=4e-19, both of which are
    indistinguishable from a random 10/100 split of the same books once the
    nesting is respected. Per-book shares are 10 values against 100 and need no
    clustering correction of their own.

    Mann--Whitney U, two-sided -- the quantity is a bounded share on 10 books,
    where normality is not worth assuming. Multiplicity is controlled across
    the categories of a single axis with Benjamini--Hochberg. The effect size
    is the rank-biserial correlation (a monotone transform of U, so it inherits
    the test's assumptions and adds none): +1 means every fiction10 book sits
    above every top100 book on that category's share, -1 the reverse.

    `min_count` drops books contributing fewer than that many rows on the axis,
    which matters only for the within-context slices where a book may carry a
    handful of norms and its share would otherwise be 0 or 1 by accident.
    """
    d = df[df[field].notna()]
    counts = (
        d.groupby(["corpus", "gutenberg_id", field]).size().unstack(field, fill_value=0)
    )
    if min_count:
        counts = counts[counts.sum(axis=1) >= min_count]
    share = counts.div(counts.sum(axis=1), axis=0)

    cats = list(categories) if categories is not None else list(share.columns)
    rows = []
    for cat in cats:
        if cat not in share.columns:
            continue
        a = share.loc["fiction10", cat].to_numpy(float) if "fiction10" in share.index.get_level_values(0) else np.array([])
        b = share.loc["top100", cat].to_numpy(float) if "top100" in share.index.get_level_values(0) else np.array([])
        if len(a) < min_books or len(b) < min_books:
            continue
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        rows.append(
            {
                "category": cat,
                "label": display(cat),
                "fiction10_median": float(np.median(a)),
                "top100_median": float(np.median(b)),
                "delta_median": float(np.median(b) - np.median(a)),
                "rank_biserial": float(2 * u / (len(a) * len(b)) - 1),
                "u": float(u),
                "p": float(p),
                "n_books_fiction10": int(len(a)),
                "n_books_top100": int(len(b)),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out.assign(q=pd.Series(dtype=float))
    out["q"] = (
        stats.false_discovery_control(out["p"].to_numpy(), method="bh")
        if len(out) > 1
        else out["p"]
    )
    return out


def run_effect_test(df: pd.DataFrame, field: str, categories=None) -> pd.DataFrame:
    """Paired same-book control for the extraction run.

    The two corpora were extracted on different days (2026-07-12 and -13), so
    corpus and extraction run are confounded: a between-corpus difference can
    be a property of the reading list or of the run. Seven books appear in
    both, which gives a paired estimate of the run effect alone -- same book,
    same teacher, same prompts, two independent generations.

    Wilcoxon signed-rank on the 7 paired per-book shares. Read it against the
    between-corpus shift: where the paired shift matches it, the reading list
    is not what moved.
    """
    source = norm_facets if field == "facet" else df
    d = source[source[field].notna()]
    counts = (
        d.groupby(["corpus", "gutenberg_id", field]).size().unstack(field, fill_value=0)
    )
    share = counts.div(counts.sum(axis=1), axis=0)
    shared = sorted(set(share.loc["fiction10"].index) & set(share.loc["top100"].index))
    cats = list(categories) if categories is not None else list(share.columns)
    rows = []
    for cat in cats:
        if cat not in share.columns:
            continue
        a = share.loc["fiction10", cat].reindex(shared).to_numpy(float)
        b = share.loc["top100", cat].reindex(shared).to_numpy(float)
        paired = float(np.median(b - a))
        if np.allclose(a, b):
            p = 1.0
        else:
            p = float(stats.wilcoxon(a, b).pvalue)
        rows.append({"category": cat, "run_shift": paired, "run_p": p,
                     "n_shared_books": len(shared)})
    return pd.DataFrame(rows)


def test_headline(res: pd.DataFrame, unit: str = "") -> str:
    """Panel-subtitle summary of a book_level_test result.

    Deliberately short. These land under a panel title in a three-across grid,
    where naming the test and the correction in every panel both overruns the
    neighbouring panel and repeats what the caption says once.
    """
    if res.empty:
        return "not estimable"
    k = int((res["q"] < 0.05).sum())
    return f"{k} of {len(res)}{' ' + unit if unit else ''} differ ($q<0.05$)"


def pct(x: float, dp: int = 1) -> str:
    """Percentage for a LaTeX caption. A bare `%` from an f-string `:.1%` is a
    comment character and silently eats the rest of the caption line."""
    return f"{x * 100:.{dp}f}\\%"


def pp(x: float, dp: int = 1) -> str:
    """A *difference* between two shares, in percentage points. Rendering it
    with a per-cent sign invites reading a 4.4-point gap as a 4.4\\% change."""
    return f"{x * 100:.{dp}f}~pp"


def fmt_q(q: float) -> str:
    """Render a q-value for display. Two decimals never reach the interesting
    range here -- the surviving axes land near 1e-5 -- and `q=0.0000` reads as
    zero, so anything below the floor is shown as an inequality."""
    if not np.isfinite(q):
        return "$q$ n/a"
    if q < 0.001:
        return "$q<0.001$"
    return f"$q$={q:.3f}" if q < 0.01 else f"$q$={q:.2f}"


def star(q: float) -> str:
    """Significance mark. One threshold only -- a three-star ladder invites
    reading the count as an effect size, which is exactly what it is not."""
    return "*" if q is not None and np.isfinite(q) and q < 0.05 else ""


def shannon(counts: pd.Series) -> float:
    p = counts[counts > 0].to_numpy(float)
    p = p / p.sum()
    return float(-(p * np.log2(p)).sum())


def facet_frame(df: pd.DataFrame, col: str, extra: list[str] | None = None) -> pd.DataFrame:
    """Explode a context column into canonical facets, one row per (row, facet).

    `extra` columns are carried along so downstream cross-tabs never have to
    re-join against the parent frame (whose index is gapped by the valid-norm
    filter, which would silently misalign an index merge).
    """
    keep = ["corpus", "gutenberg_id", col] + list(extra or [])
    out = df[keep].copy()
    out["facet"] = out[col].map(facets)
    out = out.explode("facet")
    return out[out["facet"].notna()]


print("canonicalising contexts …")
norm_facets = facet_frame(norms, "raz_context", ["raz_normative_force"])
flow_facets = facet_frame(flows, "ci_context", ["ci_appropriateness"])


def facet_shares(fdf: pd.DataFrame) -> pd.DataFrame:
    ct = fdf.groupby(["corpus", "facet"]).size().unstack("corpus", fill_value=0)
    for c in CORPORA:
        if c not in ct.columns:
            ct[c] = 0
    ct = ct[list(CORPORA)]
    share = ct.div(ct.sum(axis=0), axis=1)
    share.columns = [f"{c}_p" for c in CORPORA]
    out = pd.concat([ct.add_suffix("_n"), share], axis=1)
    out["delta_p"] = out["top100_p"] - out["fiction10_p"]
    out["pooled_p"] = (out["fiction10_n"] + out["top100_n"]) / (
        out["fiction10_n"].sum() + out["top100_n"].sum()
    )
    return out.sort_values("pooled_p", ascending=False)


norm_ctx = facet_shares(norm_facets)
flow_ctx = facet_shares(flow_facets)
save_table(norm_ctx, "norm_context_facets")
save_table(flow_ctx, "flow_context_facets")

TOP_N_CTX = 16
NORM_CTX_TOP = norm_ctx.head(TOP_N_CTX).index.tolist()
FLOW_CTX_TOP = flow_ctx.head(TOP_N_CTX).index.tolist()


# ==========================================================================
# Figure 1 — per-chunk extraction yield
# ==========================================================================
def fig_extraction_yield():
    fig, axes = plt.subplots(1, 3, figsize=(TW, 2.5))
    metrics = [
        ("norms_per_chunk", "norms per chunk"),
        ("flows_per_chunk", "CI flow tuples per chunk"),
        ("flow_chunk_rate", "chunks with $\\geq$1 flow"),
    ]
    rng = np.random.default_rng(0)
    for ax, (col, title) in zip(axes, metrics):
        f10 = books[books.corpus == "fiction10"][col].to_numpy()
        t100 = books[books.corpus == "top100"][col].to_numpy()
        parts = ax.violinplot(
            [f10, t100], positions=[0, 1], widths=0.7, showextrema=False
        )
        for body, c in zip(parts["bodies"], CORPORA):
            body.set_facecolor(COLOR[c])
            body.set_alpha(0.20)
            body.set_edgecolor("none")
        for i, (vals, c) in enumerate(zip([f10, t100], CORPORA)):
            jitter = rng.uniform(-0.10, 0.10, size=len(vals))
            ax.scatter(
                np.full(len(vals), i) + jitter,
                vals,
                s=16 if c == "fiction10" else 9,
                color=COLOR[c],
                alpha=0.85 if c == "fiction10" else 0.45,
                linewidths=0,
                zorder=3,
            )
            ax.hlines(
                np.median(vals), i - 0.30, i + 0.30, color=COLOR[c], lw=2, zorder=4
            )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(list(CORPORA), fontsize=8)
        fmt = "{:.2f}" if col != "flow_chunk_rate" else "{:.3f}"
        ax.set_title(
            f"{title}\nmedian " + " / ".join(fmt.format(np.median(v)) for v in (f10, t100))
        )
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        ax.set_ylim(bottom=0)
        ax.set_xlim(-0.72, 1.72)
    axes[0].set_ylabel("Per-book value")
    fig.tight_layout()
    save_fig(fig, "corpus_extraction_yield")

    rows = []
    for col, title in metrics:
        r = {"metric": col}
        for c in CORPORA:
            v = books[books.corpus == c][col]
            r[f"{c}_median"] = v.median()
            r[f"{c}_iqr_lo"] = v.quantile(0.25)
            r[f"{c}_iqr_hi"] = v.quantile(0.75)
        rows.append(r)
    save_table(pd.DataFrame(rows), "extraction_yield_summary", index=False)


# ==========================================================================
# Figure 2 — deontic force: pooled + per-book composition
# ==========================================================================
def fig_deontic():
    fig = plt.figure(figsize=(TW, 3.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.30, 0.85], wspace=0.62)

    # (a) pooled proportions, grouped bars
    ax = fig.add_subplot(gs[0, 0])
    ct = (
        norms.groupby(["corpus", "raz_normative_force"])
        .size()
        .unstack("corpus")
        .reindex(FORCE_ORDER)
    )
    prop = ct.div(ct.sum(axis=0), axis=1)
    y = np.arange(len(FORCE_ORDER))
    for i, c in enumerate(CORPORA):
        ax.barh(
            y + (i - 0.5) * 0.38,
            prop[c],
            0.38,
            color=COLOR[c],
            label=LABEL[c],
        )
    # Bars are the pooled description; the mark beside each is the book-level
    # inference. Keeping the two visually distinct (grey delta, black star) is
    # deliberate -- the pooled delta is not what the test is computed on.
    res = book_level_test(norms, "raz_normative_force", FORCE_ORDER).set_index("category")
    for j, f in enumerate(FORCE_ORDER):
        ax.annotate(
            f"{prop['top100'][f] - prop['fiction10'][f]:+.3f}{star(res.loc[f, 'q'])}",
            (max(prop.loc[f]) + 0.012, y[j]),
            va="center",
            fontsize=7,
            color="#555555",
        )
    ax.set_yticks(y)
    ax.set_yticklabels([display(f) for f in FORCE_ORDER])
    ax.invert_yaxis()
    ax.set_xlim(0, 0.78)
    ax.set_xlabel("Proportion of norms")
    ax.set_title(f"(a) Pooled deontic force\n{test_headline(res, 'forces')}")
    # Label the two bars in place rather than with a legend box: at this size a
    # box lands on top of the short-category bars whichever corner it takes.
    for i, c in enumerate(CORPORA):
        ax.annotate(
            LABEL[c].split(" (")[0],
            (prop[c]["obligatory"] - 0.015, y[0] + (i - 0.5) * 0.38),
            ha="right", va="center", fontsize=6.5, color=text_on(COLOR[c]),
        )
    ax.grid(True, axis="x", alpha=0.25)
    ax.set_axisbelow(True)

    # (b) per-book composition: the ten training books, stacked
    ax = fig.add_subplot(gs[0, 1])
    f10 = norms[norms.corpus == "fiction10"]
    comp = (
        f10.groupby(["gutenberg_id", "raz_normative_force"])
        .size()
        .unstack("raz_normative_force")
        .reindex(columns=FORCE_ORDER)
        .fillna(0)
    )
    comp = comp.div(comp.sum(axis=1), axis=0)
    comp = comp.sort_values("obligatory")
    left = np.zeros(len(comp))
    yy = np.arange(len(comp))
    for f in FORCE_ORDER:
        ax.barh(yy, comp[f], 0.68, left=left, color=FORCE_COLOR[f], label=display(f))
        left += comp[f].to_numpy()
    # Titles go inside the (dark, always-widest) obligatory segment: as
    # y-tick labels they overrun into panel (a).
    ax.set_yticks([])
    for j, g in enumerate(comp.index):
        ax.annotate(short(g), (0.015, yy[j]), ha="left", va="center",
                    fontsize=6.5, color=text_on(FORCE_COLOR["obligatory"]))
    ax.set_xlim(0, 1)
    ax.set_xlabel("Proportion of the book's norms")
    ax.set_title("(b) fiction10 per-book composition")
    ax.legend(
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        frameon=False,
        fontsize=7,
    )
    ax.grid(False)

    # (c) per-book spread of the two poles, both corpora
    ax = fig.add_subplot(gs[0, 2])
    pb = (
        norms.groupby(["corpus", "gutenberg_id", "raz_normative_force"])
        .size()
        .unstack("raz_normative_force")
        .reindex(columns=FORCE_ORDER)
        .fillna(0)
    )
    pb = pb.div(pb.sum(axis=1), axis=0).reset_index()
    rng = np.random.default_rng(1)
    for k, force in enumerate(["obligatory", "prohibited"]):
        for i, c in enumerate(CORPORA):
            v = pb[pb.corpus == c][force].to_numpy()
            x = k * 1.0 + (i - 0.5) * 0.36
            ax.scatter(
                x + rng.uniform(-0.07, 0.07, len(v)),
                v,
                s=15 if c == "fiction10" else 8,
                color=COLOR[c],
                alpha=0.85 if c == "fiction10" else 0.45,
                linewidths=0,
                zorder=3,
            )
            ax.hlines(np.median(v), x - 0.15, x + 0.15, color=COLOR[c], lw=2, zorder=4)
    # This panel *is* the tested quantity -- 10 dots against 100, one per book --
    # so the q-value belongs here rather than only in the pooled panel.
    for k, force in enumerate(["obligatory", "prohibited"]):
        q = float(res.loc[force, "q"])
        ax.annotate(
            f"{fmt_q(q)}{star(q)}",
            (k, 0.015),
            ha="center", va="bottom", fontsize=6.5, color="#555555",
        )
    ax.set_xlim(-0.5, 1.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([display(f) for f in ("obligatory", "prohibited")])
    ax.set_ylabel("Per-book share of norms")
    ax.set_ylim(0, 0.95)
    ax.set_title("(c) Per-book spread")
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    # Annotate the obligatory group's two median rules rather than adding a
    # legend box: at this panel width any box lands on the data.
    for i, c in enumerate(CORPORA):
        v = pb[pb.corpus == c]["obligatory"].to_numpy()
        ax.annotate(
            c,
            ((i - 0.5) * 0.36, 0.90 if i == 0 else 0.83),
            ha="center", va="center", fontsize=6.5, color=COLOR[c],
        )
    save_fig(fig, "corpus_deontic_force")
    tightest = res.loc[res["q"].idxmin()]
    proh = res.loc["prohibited"]
    save_caption(
        "corpus_deontic_force",
        "Deontic force of extracted norms, fiction10 versus top100",
        f"Panel (a) gives the pooled share of each deontic force in the two "
        f"corpora ({len(norms[norms.corpus == 'fiction10']):,} and "
        f"{len(norms[norms.corpus == 'top100']):,} norms), with the top100 "
        f"minus fiction10 difference beside each pair. Panels (b) and (c) break "
        f"the same quantity out by book. Norms are nested in books, so the test "
        f"takes the book as its unit -- a two-sided Mann--Whitney U on per-book "
        f"shares, 10 books against 100, Benjamini--Hochberg corrected across the "
        f"five forces. No force clears $q<0.05$; the tightest is "
        f"{display(tightest.name).lower()} at {fmt_q(tightest['q'])}. "
        f"Prohibited norms are {pct(proh['fiction10_median'])} of a median "
        f"fiction10 book and {pct(proh['top100_median'])} of a median top100 "
        f"book. Seven of the ten fiction10 books are also in top100, so the "
        f"comparison understates whatever difference exists.",
        "fig:corpus-deontic-force",
        ["corpus-descriptives", "deontic-modality", "fiction10-vs-top100",
         "mann-whitney", "camera-ready"],
    )
    save_table(prop.reindex(FORCE_ORDER), "deontic_force_pooled")
    save_table(comp, "deontic_force_by_book_fiction10")
    save_table(res.reset_index(), "deontic_force_book_level_test", index=False)


# ==========================================================================
# Figure 3 — societal-context vocabulary (norms and flows)
# ==========================================================================
def _ctx_panel(ax, table, top, title, unit, res=None):
    t = table.loc[top]
    y = np.arange(len(t))
    for i, c in enumerate(CORPORA):
        ax.barh(y + (i - 0.5) * 0.38, t[f"{c}_p"], 0.38, color=COLOR[c], label=LABEL[c])
    # A star marks the facets that survive the book-level test. Marking them on
    # the axis rather than in a separate panel keeps the reader's eye on the one
    # comparison the figure is about (Tufte: a second panel is a third picture).
    if res is not None:
        q = res.set_index("category")["q"]
        for j, f in enumerate(t.index):
            if f in q.index and star(q[f]):
                ax.annotate(
                    "*",
                    (max(t.loc[f, f"{c}_p"] for c in CORPORA) + 0.006, y[j]),
                    va="center", fontsize=9, color=EDGE,
                )
    ax.set_yticks(y)
    ax.set_yticklabels([display(f) for f in t.index], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel(f"Share of {unit} (multi-label facets)")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.25)
    ax.set_axisbelow(True)


def _coverage_panel(ax, fdf, title):
    """Draw the cumulative-coverage curve; return {corpus: (distinct, n80)} so
    the caption can quote the same numbers the panel plots."""
    stats_out = {}
    for c in CORPORA:
        vc = fdf[fdf.corpus == c].facet.value_counts()
        cum = vc.cumsum() / vc.sum()
        ax.plot(
            np.arange(1, len(cum) + 1),
            cum.to_numpy(),
            color=COLOR[c],
            lw=1.6,
            label=f"{LABEL[c]} — {len(vc):,} distinct",
        )
        n80 = int(np.searchsorted(cum.to_numpy(), 0.80) + 1)
        stats_out[c] = (len(vc), n80)
        ax.scatter([n80], [cum.to_numpy()[n80 - 1]], s=22, color=COLOR[c], zorder=4)
        # A white halo keeps these readable where they cross the curves.
        ax.annotate(
            f"{n80} facets → 80%",
            (n80, 0.80),
            textcoords="offset points",
            xytext=(12, 26 if c == "fiction10" else -22),
            fontsize=7,
            color=COLOR[c],
            path_effects=[pe.withStroke(linewidth=2.2, foreground="white")],
        )
    ax.axhline(0.80, color="#999999", lw=0.7, ls="--", zorder=1)
    ax.set_xscale("log")
    ax.set_xlabel("Context facets, ranked by frequency (log scale)")
    ax.set_ylabel("Cumulative share")
    ax.set_ylim(0, 1.02)
    ax.set_title(title)
    ax.legend(loc="lower right", frameon=False)
    return stats_out


def fig_context_vocabulary():
    fig, axes = plt.subplots(2, 2, figsize=(TW, 6.4), width_ratios=[1.25, 1.0])
    # Tested on the top facets only: a book-level test over all 1,907 facets is
    # dominated by singletons that most books never use, and correcting across
    # them would bury the facets the panel actually shows.
    res_n = book_level_test(norm_facets, "facet", NORM_CTX_TOP)
    res_f = book_level_test(flow_facets, "facet", FLOW_CTX_TOP)
    _ctx_panel(
        axes[0, 0],
        norm_ctx,
        NORM_CTX_TOP,
        f"(a) Norm contexts, top {TOP_N_CTX}\n{test_headline(res_n, 'facets')}",
        "norms",
        res_n,
    )
    axes[0, 0].legend(loc="lower right", frameon=False)
    cov_n = _coverage_panel(axes[0, 1], norm_facets, "(b) Norm-context vocabulary breadth")
    _ctx_panel(
        axes[1, 0],
        flow_ctx,
        FLOW_CTX_TOP,
        f"(c) Flow contexts, top {TOP_N_CTX}\n{test_headline(res_f, 'facets')}",
        "flows",
        res_f,
    )
    _coverage_panel(axes[1, 1], flow_facets, "(d) Flow-context vocabulary breadth")
    fig.tight_layout()
    save_fig(fig, "corpus_context_vocabulary")
    (d_f10, n80_f10), (d_t100, n80_t100) = cov_n["fiction10"], cov_n["top100"]
    save_caption(
        "corpus_context_vocabulary",
        "Societal-context vocabulary of the two corpora",
        f"Panels (a) and (c) give the share of norms and of CI flow tuples "
        f"carrying each of the {TOP_N_CTX} most frequent context facets. "
        f"Contexts are free text, so composites are split into facets and "
        f"counted multi-label; the shares do not sum to one. Stars mark facets "
        f"that separate the corpora under a two-sided Mann--Whitney U test on "
        f"per-book shares, 10 books against 100, Benjamini--Hochberg corrected "
        f"across the {TOP_N_CTX} facets shown "
        f"({int((res_n['q'] < 0.05).sum())} norm facets and "
        f"{int((res_f['q'] < 0.05).sum())} flow facets qualify). Panels (b) and "
        f"(d) rank the full vocabulary by frequency and plot cumulative "
        f"coverage. The two corpora agree on the head and separate in the tail. "
        f"Reaching 80\\% of the mass takes {n80_f10} facets in fiction10 and "
        f"{n80_t100} in top100, but the full vocabularies run to {d_f10:,} and "
        f"{d_t100:,} distinct facets. Ten setting-heavy novels therefore cover "
        f"the same frequent societal domains as a hundred, and what the "
        f"additional ninety books add is rare-facet coverage.",
        "fig:corpus-context-vocabulary",
        ["corpus-descriptives", "societal-context", "fiction10-vs-top100",
         "mann-whitney", "camera-ready"],
    )
    save_table(res_n.assign(axis="norm_context"), "norm_context_book_level_test", index=False)
    save_table(res_f.assign(axis="flow_context"), "flow_context_book_level_test", index=False)


# ==========================================================================
# Figure 4 — deontic force conditioned on context
# ==========================================================================
def fig_deontic_by_context():
    merged = norm_facets[norm_facets.facet.isin(NORM_CTX_TOP)]
    fig, axes = plt.subplots(
        1, 3, figsize=(TW, 3.9), width_ratios=[1, 1, 0.55], sharey=True
    )
    mats = {}
    for panel, (ax, c) in zip("ab", zip(axes[:2], CORPORA)):
        sub = merged[merged.corpus == c]
        m = (
            sub.groupby(["facet", "raz_normative_force"])
            .size()
            .unstack("raz_normative_force")
            .reindex(index=NORM_CTX_TOP, columns=FORCE_ORDER)
            .fillna(0)
        )
        m = m.div(m.sum(axis=1), axis=0)
        mats[c] = m
        im = ax.imshow(m.to_numpy(), cmap=CMAP, vmin=0, vmax=0.75, aspect="auto")
        ax.set_xticks(range(len(FORCE_ORDER)))
        ax.set_xticklabels([display(f) for f in FORCE_ORDER],
                           rotation=35, ha="right", fontsize=7.5)
        ax.set_yticks(range(len(NORM_CTX_TOP)))
        ax.set_yticklabels([display(f) for f in NORM_CTX_TOP], fontsize=7.5)
        ax.set_title(f"({panel}) {LABEL[c]}")
        ax.grid(False)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                v = m.iat[i, j]
                if v >= 0.05:
                    ax.text(
                        j,
                        i,
                        f"{v:.2f}".lstrip("0"),
                        ha="center",
                        va="center",
                        fontsize=6.5,
                        color=on_color(CMAP, v / 0.75),
                    )
    fig.colorbar(im, ax=axes, orientation="horizontal", fraction=0.045,
                 pad=0.20, aspect=45, label="Share of the context's norms")

    # (c) Within each context, test the five forces book by book and keep the
    # force that moves most. Books carrying fewer than five norms in a context
    # are dropped: their share is 0 or 1 by accident of rounding, not by
    # composition. Correction runs within a context, across its five forces.
    ax = axes[2]
    per_ctx = []
    for f in NORM_CTX_TOP:
        r = book_level_test(
            merged[merged.facet == f], "raz_normative_force", FORCE_ORDER, min_count=5
        )
        if r.empty:
            per_ctx.append({"facet": f, "force": None, "delta": 0.0, "q": np.nan})
            continue
        top = r.loc[r["delta_median"].abs().idxmax()]
        per_ctx.append({"facet": f, "force": top["category"],
                        "delta": abs(top["delta_median"]), "q": top["q"]})
    pc = pd.DataFrame(per_ctx).set_index("facet").loc[NORM_CTX_TOP]
    order = pc["delta"].sort_values(ascending=False)
    ax.barh(
        [NORM_CTX_TOP.index(f) for f in order.index],
        order.to_numpy(),
        0.7,
        color=PAL["warmgrey"],
    )
    for f in order.index:
        if star(pc.loc[f, "q"]):
            ax.annotate("*", (pc.loc[f, "delta"] + 0.004, NORM_CTX_TOP.index(f)),
                        va="center", fontsize=9, color=EDGE)
    ax.set_xlabel("Largest per-book shift")
    ax.set_title("(c) Per-context shift")
    ax.grid(True, axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    save_fig(fig, "corpus_deontic_by_context")
    n_sig = int((pc["q"] < 0.05).sum())
    save_caption(
        "corpus_deontic_by_context",
        "Deontic force conditioned on societal context",
        f"Rows are the {TOP_N_CTX} most frequent norm-context facets, columns "
        f"the five deontic forces; each cell is that force's share of the "
        f"context's norms, so rows sum to one. The two heatmaps hold the "
        f"extractor and the prompts fixed and vary only the reading list. Panel "
        f"(c) reports, for each context, the force whose per-book median share "
        f"moves most between the corpora, with a star where a two-sided "
        f"Mann--Whitney U test on per-book shares clears $q<0.05$ after "
        f"Benjamini--Hochberg correction within the context "
        f"({n_sig} of {TOP_N_CTX} contexts). Books contributing fewer than five "
        f"norms to a context are excluded from that context's test.",
        "fig:corpus-deontic-by-context",
        ["corpus-descriptives", "deontic-modality", "societal-context",
         "mann-whitney", "camera-ready"],
    )
    save_table(mats["fiction10"], "deontic_by_context_fiction10")
    save_table(mats["top100"], "deontic_by_context_top100")
    save_table(pc.reset_index(), "deontic_by_context_book_level_test", index=False)


# ==========================================================================
# Figure 5 — contextual entropy per book
# ==========================================================================
def fig_context_entropy():
    rows = []
    for kind, fdf in (("norm", norm_facets), ("flow", flow_facets)):
        for (c, gid), grp in fdf.groupby(["corpus", "gutenberg_id"]):
            rows.append(
                {
                    "kind": kind,
                    "corpus": c,
                    "gutenberg_id": gid,
                    "entropy": shannon(grp.facet.value_counts()),
                    "n_facets": grp.facet.nunique(),
                }
            )
    ent = pd.DataFrame(rows)
    save_table(ent, "context_entropy_per_book", index=False)

    pooled = {
        (kind, c): shannon(
            fdf[fdf.corpus == c].facet.value_counts()
        )
        for kind, fdf in (("norm", norm_facets), ("flow", flow_facets))
        for c in CORPORA
    }

    fig, axes = plt.subplots(1, 2, figsize=(TW, 3.1), width_ratios=[1.5, 1])

    # (a) per-book entropy: fiction10 books named, top100 as a distribution
    ax = axes[0]
    f10 = (
        ent[(ent.corpus == "fiction10")]
        .pivot(index="gutenberg_id", columns="kind", values="entropy")
        .sort_values("norm")
    )
    yy = np.arange(len(f10))
    ax.scatter(f10["norm"], yy, s=26, color=COLOR["fiction10"], label="Norm contexts", zorder=3)
    ax.scatter(
        f10["flow"],
        yy,
        s=26,
        facecolors="none",
        edgecolors=COLOR["fiction10"],
        linewidths=1.1,
        label="Flow contexts",
        zorder=3,
    )
    ax.hlines(yy, f10[["norm", "flow"]].min(axis=1), f10[["norm", "flow"]].max(axis=1),
              color=COLOR["fiction10"], alpha=0.35, lw=1, zorder=2)
    ax.set_yticks(yy)
    ax.set_yticklabels([short(g) for g in f10.index], fontsize=7.5)
    ax.set_xlabel("Shannon entropy over context facets (bits)")
    ax.set_title("(a) fiction10 per-book contextual diversity")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(True, axis="x", alpha=0.25)
    ax.set_axisbelow(True)

    # (b) distribution comparison + pooled corpus entropy
    ax = axes[1]
    rng = np.random.default_rng(2)
    for k, kind in enumerate(("norm", "flow")):
        for i, c in enumerate(CORPORA):
            v = ent[(ent.kind == kind) & (ent.corpus == c)].entropy.to_numpy()
            x = k + (i - 0.5) * 0.36
            ax.scatter(
                x + rng.uniform(-0.07, 0.07, len(v)),
                v,
                s=15 if c == "fiction10" else 8,
                color=COLOR[c],
                alpha=0.85 if c == "fiction10" else 0.4,
                linewidths=0,
                zorder=3,
            )
            ax.hlines(np.median(v), x - 0.15, x + 0.15, color=COLOR[c], lw=2, zorder=4)
            ax.scatter(
                [x], [pooled[(kind, c)]], marker="D", s=30, color=COLOR[c],
                edgecolors="white", linewidths=0.8, zorder=5,
            )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Norm contexts", "Flow contexts"])
    ax.set_ylabel("Entropy (bits)")
    ax.set_title("(b) Per-book (dots) vs. pooled corpus (diamonds)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(
        handles=[Patch(facecolor=COLOR[c], label=LABEL[c]) for c in CORPORA]
        + [
            Line2D([], [], marker="D", ls="none", color=EDGE, label="Pooled corpus"),
        ],
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        frameon=False,
        fontsize=6.5,
    )
    fig.tight_layout()
    save_fig(fig, "corpus_context_entropy")
    save_table(
        pd.DataFrame(
            [{"kind": k, "corpus": c, "pooled_entropy": v} for (k, c), v in pooled.items()]
        ),
        "context_entropy_pooled",
        index=False,
    )


# ==========================================================================
# Figure 6 — flow appropriateness
# ==========================================================================
def fig_flow_appropriateness():
    fig, axes = plt.subplots(1, 2, figsize=(TW, 3.6), width_ratios=[0.75, 1])

    # (a) pooled verdict mix
    ax = axes[0]
    ct = (
        flows.groupby(["corpus", "ci_appropriateness"])
        .size()
        .unstack("ci_appropriateness")
        .reindex(columns=APPR_ORDER)
        .fillna(0)
    )
    prop = ct.div(ct.sum(axis=1), axis=0)
    left = np.zeros(len(CORPORA))
    yy = np.arange(len(CORPORA))
    for a in APPR_ORDER:
        vals = prop.loc[list(CORPORA), a].to_numpy()
        ax.barh(yy, vals, 0.42, left=left, color=APPR_COLOR[a], label=display(a))
        # Stagger the two small segments' labels; side by side they collide.
        for i, v in enumerate(vals):
            if a == "appropriate":       # wide segment: label sits inside it
                ax.annotate(f"{v:.3f}", (left[i] + v / 2, yy[i]),
                            ha="center", va="center", fontsize=7,
                            color=text_on(APPR_COLOR[a]))
            else:                        # thin segments: right-hand gutter
                dy = -0.13 if a == "ambiguous" else 0.13
                ax.annotate(f"{display(a)[:5]}. {v:.3f}", (1.04, yy[i] + dy),
                            ha="left", va="center", fontsize=6.5, color=APPR_COLOR[a])
        left += vals
    res_appr = book_level_test(flows, "ci_appropriateness", APPR_ORDER)
    ax.set_yticks(yy)
    ax.set_yticklabels([LABEL[c] for c in CORPORA], fontsize=7)
    ax.set_xlim(0, 1.42)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_ylim(1.7, -0.7)
    ax.set_xlabel("Share of flows")
    ax.set_title(f"(a) Pooled appropriateness\n{test_headline(res_appr, 'verdicts')}")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.20),
              frameon=False, fontsize=6.5)
    ax.grid(False)

    # (b) the tail of interest: inappropriate share by context, both corpora.
    # A stacked 0--1 bar per context hides a 5--30% signal; a grouped bar on the
    # inappropriate share alone is what the prose actually claims.
    merged = flow_facets[flow_facets.facet.isin(FLOW_CTX_TOP)]
    mats = {}
    for c in CORPORA:
        m = (
            merged[merged.corpus == c]
            .groupby(["facet", "ci_appropriateness"])
            .size()
            .unstack("ci_appropriateness")
            .reindex(index=FLOW_CTX_TOP, columns=APPR_ORDER)
            .fillna(0)
        )
        mats[c] = m.div(m.sum(axis=1), axis=0)
        save_table(mats[c], f"flow_appropriateness_by_context_{c}")
    order = (
        (mats["fiction10"]["inappropriate"] + mats["top100"]["inappropriate"])
        .sort_values(ascending=False)
        .index
    )
    ax = axes[1]
    y = np.arange(len(order))
    for i, c in enumerate(CORPORA):
        ax.barh(
            y + (i - 0.5) * 0.38,
            mats[c].loc[order, "inappropriate"],
            0.38,
            color=COLOR[c],
            label=LABEL[c],
        )
    pooled_rate = prop["inappropriate"].mean()
    ax.axvline(pooled_rate, color="#666666", lw=0.8, ls="--", zorder=1)
    ax.annotate(
        f"pooled {pooled_rate:.3f}",
        (pooled_rate, -0.75),
        textcoords="offset points",
        xytext=(4, 0),
        fontsize=6.5,
        color="#666666",
        annotation_clip=False,
    )
    ax.set_yticks(y)
    ax.set_yticklabels([display(f) for f in order], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Share of the context's flows judged inappropriate")
    ax.set_title("(b) Inappropriate flows by context")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(True, axis="x", alpha=0.25)
    ax.set_axisbelow(True)

    fig.tight_layout()
    save_fig(fig, "corpus_flow_appropriateness")
    inappr = res_appr.set_index("category").loc["inappropriate"]
    save_caption(
        "corpus_flow_appropriateness",
        "Appropriateness of extracted CI flow tuples",
        f"Panel (a) gives the pooled verdict mix over "
        f"{len(flows[flows.corpus == 'fiction10']):,} fiction10 and "
        f"{len(flows[flows.corpus == 'top100']):,} top100 flow tuples. Panel "
        f"(b) isolates the inappropriate share within each of the "
        f"{TOP_N_CTX} most frequent flow contexts, since a stacked bar over the "
        f"full verdict range hides a signal that never exceeds 30\\%. "
        f"Significance is a two-sided Mann--Whitney U test on per-book shares, "
        f"10 books against 100, Benjamini--Hochberg corrected across the three "
        f"verdicts. Inappropriate flows are {pct(inappr['fiction10_median'])} of "
        f"a median fiction10 book and {pct(inappr['top100_median'])} of a median "
        f"top100 book ({fmt_q(inappr['q'])}).",
        "fig:corpus-flow-appropriateness",
        ["corpus-descriptives", "contextual-integrity", "fiction10-vs-top100",
         "mann-whitney", "camera-ready"],
    )
    save_table(prop, "flow_appropriateness_pooled")
    save_table(res_appr, "flow_appropriateness_book_level_test", index=False)


# ==========================================================================
# Figure 7 — CI-relevant norm share (the slice the method depends on)
# ==========================================================================
def fig_ci_relevance():
    fig, axes = plt.subplots(1, 2, figsize=(TW, 2.7), width_ratios=[1, 1.1])

    ax = axes[0]
    rng = np.random.default_rng(3)
    pooled_by_corpus = {}
    for i, c in enumerate(CORPORA):
        v = books[books.corpus == c]["ci_norm_share"].dropna().to_numpy()
        ax.scatter(
            np.full(len(v), i) + rng.uniform(-0.11, 0.11, len(v)),
            v,
            s=17 if c == "fiction10" else 9,
            color=COLOR[c],
            alpha=0.85 if c == "fiction10" else 0.45,
            linewidths=0,
            zorder=3,
        )
        ax.hlines(np.median(v), i - 0.28, i + 0.28, color=COLOR[c], lw=2, zorder=4)
        pooled = (
            norms[(norms.corpus == c) & (norms.raz_governs_info_flow == True)].shape[0]  # noqa: E712
            / norms[norms.corpus == c].shape[0]
        )
        ax.scatter([i], [pooled], marker="D", s=34, color=COLOR[c],
                   edgecolors=EDGE, linewidths=0.6, zorder=5)
        pooled_by_corpus[c] = pooled
    res_ci = book_level_test(norms, "raz_governs_info_flow", [True, False])
    q_ci = float(res_ci.set_index("category").loc[True, "q"])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(list(CORPORA), fontsize=8)
    ax.set_ylabel("Share of norms governing information flow")
    ax.set_ylim(0, 0.55)
    ax.set_xlim(-0.7, 1.7)
    ax.set_title(
        "(a) CI-relevant norm share\nPooled "
        f"{pooled_by_corpus['fiction10']:.3f} / {pooled_by_corpus['top100']:.3f}"
        f"  ({fmt_q(q_ci)}{star(q_ci)})"
    )
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    # (b) quality gate
    ax = axes[1]
    ct = (
        norms.groupby(["corpus", "norm_quality_passed"])
        .size()
        .unstack("norm_quality_passed")
        .fillna(0)
    )
    rate = ct[True] / ct.sum(axis=1)
    yy = np.arange(len(CORPORA))
    ax.barh(yy, [rate[c] for c in CORPORA], 0.42, color=[COLOR[c] for c in CORPORA])
    for i, c in enumerate(CORPORA):
        ax.text(rate[c] - 0.02, i, f"{rate[c]:.3f}", ha="right", va="center",
                color=text_on(COLOR[c]), fontsize=8)
    ax.set_yticks(yy)
    ax.set_yticklabels([LABEL[c] for c in CORPORA], fontsize=8)
    ax.invert_yaxis()
    ax.set_ylim(1.7, -0.7)
    ax.set_xlim(0, 1.02)
    res_gate = book_level_test(norms, "norm_quality_passed", [True, False])
    q_gate = float(res_gate.set_index("category").loc[True, "q"])
    gate = res_gate.set_index("category").loc[True]
    _run = run_effect_test(norms, "norm_quality_passed", [True]).iloc[0]
    gate_run, gate_run_p = float(_run["run_shift"]), float(_run["run_p"])
    ax.set_xlabel("Share passing the generalizability gate")
    ax.set_title("(b) Generalizability gate\n"
                 f"{fmt_q(q_gate)}{star(q_gate)}")
    ax.grid(True, axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save_fig(fig, "corpus_ci_relevance")
    save_caption(
        "corpus_ci_relevance",
        "CI-relevant norm share and the generalizability gate",
        f"Panel (a) gives the share of each book's norms that the extractor "
        f"marks as governing information flow, one dot per book, with the "
        f"pooled corpus rate as a diamond. This is the slice the reward "
        f"indexes, and it holds across the two corpora at "
        f"{pct(pooled_by_corpus['fiction10'])} of fiction10 norms against "
        f"{pct(pooled_by_corpus['top100'])} of top100 norms, with no book-level "
        f"difference ({fmt_q(q_ci)}). Panel (b) gives the share passing the "
        f"no-character, no-plot-leakage gate. "
        f"{pct(gate['fiction10_median'])} of a median fiction10 book passes "
        f"against {pct(gate['top100_median'])} of a median top100 book, and the "
        f"between-corpus test does separate them ({fmt_q(q_gate)}) -- but the "
        f"seven books present in both corpora shift by "
        f"{pp(float(gate_run))} between the two extraction runs on their own "
        f"(Wilcoxon signed-rank $p$={gate_run_p:.3f}), which is the whole "
        f"difference. The corpora were extracted a day apart, so this axis "
        f"tracks the run and not the reading list. Both between-corpus tests "
        f"are two-sided Mann--Whitney U on per-book shares, 10 books against "
        f"100.",
        "fig:corpus-ci-relevance",
        ["corpus-descriptives", "contextual-integrity", "quality-gate",
         "mann-whitney", "camera-ready"],
    )
    save_table(
        books[["corpus", "gutenberg_id", "book_title", "ci_norms", "norms", "ci_norm_share"]],
        "ci_norm_share_per_book",
        index=False,
    )
    save_table(pd.concat([res_ci.assign(axis="raz_governs_info_flow"),
                          res_gate.assign(axis="norm_quality_passed")]),
               "ci_relevance_book_level_test", index=False)


# ==========================================================================
# Figure 8 — pooled norm-attribute grid (all six schema axes at once)
# ==========================================================================
def fig_norm_attributes():
    """Pooled per-corpus proportions on every categorical norm attribute.

    Distinct from the camera-ready notebook's `fig_overlap_axes_grid`, which
    computes the same panels on the seven shared books only. This one covers
    both corpora in full.
    """
    N_CTX_PANEL = 8
    AXES = [
        ("raz_normative_force", FORCE_ORDER),
        ("raz_governs_info_flow", [True, False]),
        ("raz_norm_source", ["implicit", "explicit", "both"]),
        ("raz_confidence_qual",
         ["very_certain", "certain", "somewhat_certain", "uncertain"]),
        ("norm_quality_passed", [True, False]),
        ("raz_context", None),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(TW, 5.6))
    results = []
    for ax, (field, order) in zip(axes.ravel(), AXES):
        if field == "raz_context":
            cats = norm_ctx.head(N_CTX_PANEL).index.tolist()
            t = norm_ctx.loc[cats]
            p, q = t["fiction10_p"].to_numpy(), t["top100_p"].to_numpy()
            res = book_level_test(norm_facets, "facet", cats)
            title = f"{axis_label(field)} (top {N_CTX_PANEL} facets)"
        else:
            sub = norms[norms[field].notna()]
            ct = sub.groupby(["corpus", field]).size().unstack("corpus", fill_value=0)
            ct = ct.reindex([c for c in order if c in ct.index])
            prop = ct.div(ct.sum(axis=0), axis=1)
            cats = list(prop.index)
            p, q = prop["fiction10"].to_numpy(), prop["top100"].to_numpy()
            res = book_level_test(norms, field, cats)
            title = axis_label(field)
        results.append(res.assign(axis=field))

        qmap = res.set_index("category")["q"]
        y = np.arange(len(cats))
        ax.barh(y - 0.19, p, 0.38, color=COLOR["fiction10"])
        ax.barh(y + 0.19, q, 0.38, color=COLOR["top100"])
        for j, cat in enumerate(cats):
            if cat in qmap.index and star(qmap[cat]):
                ax.annotate("*", (max(p[j], q[j]) + 0.012, y[j]),
                            va="center", fontsize=9, color=EDGE)
        ax.set_yticks(y)
        ax.set_yticklabels([display(c) for c in cats], fontsize=7.5)
        ax.invert_yaxis()
        ax.set_xlabel("Proportion of norms")
        ax.set_title(f"{title}\n{test_headline(res)}", fontsize=9.5)
        ax.grid(True, axis="x", alpha=0.25)
        ax.set_axisbelow(True)
    handles = [Patch(facecolor=COLOR[c], label=LABEL[c]) for c in CORPORA]
    fig.legend(handles=handles, ncol=2, loc="lower center",
               bbox_to_anchor=(0.5, -0.03), frameon=False, fontsize=7.5)
    fig.tight_layout()
    save_fig(fig, "corpus_norm_attributes")
    allres = pd.concat(results, ignore_index=True)
    moved = allres[allres["q"] < 0.05]
    save_caption(
        "corpus_norm_attributes",
        "Every categorical norm attribute, fiction10 versus top100",
        f"Pooled proportions on the six categorical axes of the norm schema. "
        f"Both corpora use the same Gemma-4-31B-it teacher and the same "
        f"prompts, though not the same extraction run. "
        f"Stars mark categories that separate the corpora under a two-sided "
        f"Mann--Whitney U test on per-book shares, 10 books against 100, "
        f"Benjamini--Hochberg corrected within each axis; "
        f"{len(moved)} of {len(allres)} categories qualify, and they fall on "
        f"{moved['axis'].nunique()} of the six axes "
        f"({', '.join(axis_label(a).lower() for a in moved['axis'].unique())}). "
        f"Both of those are extractor self-report rather than normative "
        f"content, and \\autoref{{fig:corpus-divergence}} shows the paired "
        f"same-book control that attributes the gate shift, and about half the "
        f"confidence shift, to the extraction run rather than to the reading "
        f"list. Deontic force, norm source and CI relevance are flat, which is "
        f"the condition training on fiction10 and reporting on held-out books "
        f"relies on.",
        "fig:corpus-norm-attributes",
        ["corpus-descriptives", "norm-schema", "fiction10-vs-top100",
         "mann-whitney", "camera-ready"],
    )
    save_table(allres, "norm_attribute_book_level_test", index=False)


# ==========================================================================
# Figure 9 — corpus composition (genre tags and author era)
# ==========================================================================
def fig_composition():
    """Genre and era composition of the two reading lists.

    Redrawn here rather than reused from `norm_distribution_*.py`, whose
    versions are laid out for a wide notebook canvas and become illegible when
    scaled into a single-column figure.
    """
    meta = pd.read_csv(HERE / "tables/norm_distribution/book_meta.csv")
    fig, axes = plt.subplots(1, 2, figsize=(TW, 3.3), width_ratios=[1.35, 1])

    # (a) genre-tag prevalence (a book carries several tags)
    tags = meta.assign(genre=meta.genres.map(eval)).explode("genre")
    ct = tags.groupby(["corpus", "genre"]).size().unstack("corpus", fill_value=0)
    nb = meta.groupby("corpus").size()
    share = pd.DataFrame({c: ct.get(c, 0) / nb[c] for c in CORPORA})
    share = share.assign(pooled=share.mean(axis=1)).sort_values(
        "pooled", ascending=False
    ).head(14)
    ax = axes[0]
    y = np.arange(len(share))
    for i, c in enumerate(CORPORA):
        ax.barh(y + (i - 0.5) * 0.38, share[c], 0.38, color=COLOR[c], label=LABEL[c])
    ax.set_yticks(y)
    ax.set_yticklabels(share.index, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("Share of books carrying the genre tag")
    ax.set_title(
        f"(a) genre tags, top 14 by pooled share\n"
        f"{ct[ct['fiction10'] > 0].shape[0]} vs {ct[ct['top100'] > 0].shape[0]} distinct tags"
    )
    ax.legend(loc="lower right", frameon=False)
    ax.grid(True, axis="x", alpha=0.25)
    ax.set_axisbelow(True)

    # (b) author era, as a publication-era proxy
    ax = axes[1]
    bins = np.arange(1540, 1961, 20)
    meds = {}
    for c in CORPORA:
        v = meta.loc[meta.corpus == c, "birth_year"].dropna()
        meds[c] = v.median()
        ax.hist(v, bins=bins, density=True, color=COLOR[c], alpha=0.75,
                edgecolor=EDGE, linewidth=EDGE_LW, label=LABEL[c])
    for c in CORPORA:
        ax.axvline(meds[c], color=COLOR[c], lw=1.1, ls="--", zorder=5)
    ax.set_xlabel("Author birth year (publication-era proxy)")
    ax.set_ylabel("Density")
    ax.set_ylim(top=ax.get_ylim()[1] * 1.12)
    ax.set_title(
        f"(b) author era  (medians {meds['fiction10']:.0f} / {meds['top100']:.0f})"
    )
    ax.legend(loc="upper left", frameon=False, fontsize=6.5)
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save_fig(fig, "corpus_composition")
    save_table(share, "genre_share")




# ==========================================================================
# Figure 10 — norm embedding space (fiction10 only)
# ==========================================================================
# Scope note: the Gemma-4 norm-universe build succeeded for fiction10
# (2026-07-25) but not for top100, so per-book embeddings exist for the
# training corpus only. This figure is therefore fiction10-scoped by data
# availability, and the appendix says so rather than pairing it with a stale
# Qwen-era top100 counterpart.
EMB_DIR = ROOT / "outputs/2026-07-25_universe_fiction10_polarity/embeddings"


def fig_norm_centroids():
    if not EMB_DIR.is_dir():
        print(f"[skip] no embeddings at {EMB_DIR}")
        return
    gids, cents = [], []
    for f in sorted(EMB_DIR.glob("*.npy")):
        a = np.load(f)
        a = a / np.linalg.norm(a, axis=1, keepdims=True)
        gids.append(f.stem)
        cents.append(a.mean(axis=0))
    C = np.vstack(cents)
    C = C / np.linalg.norm(C, axis=1, keepdims=True)
    sim = C @ C.T
    names = [short(g) for g in gids]
    order = np.argsort([-books.set_index(["corpus", "gutenberg_id"])
                        .loc[("fiction10", g), "norms"] for g in gids])
    sim, names = sim[np.ix_(order, order)], [names[i] for i in order]

    fig, ax = plt.subplots(figsize=(TW * 0.72, TW * 0.62))
    m = sim.copy()
    np.fill_diagonal(m, np.nan)
    vmin, vmax = np.nanmin(m), np.nanmax(m)
    im = ax.imshow(m, cmap=CMAP_SEQ, aspect="equal", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=40, ha="right", fontsize=7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.grid(False)
    for i in range(len(names)):
        for j in range(len(names)):
            if i != j:
                ax.text(j, i, f"{sim[i, j]:.2f}".lstrip("0"), ha="center",
                        va="center", fontsize=5.8,
                        color=on_color(CMAP_SEQ, (sim[i, j] - vmin) / (vmax - vmin)))
    fig.colorbar(im, ax=ax, fraction=0.042, pad=0.03,
                 label="Cosine similarity of per-book norm centroids")
    fig.tight_layout()
    save_fig(fig, "corpus_norm_centroids")
    save_table(pd.DataFrame(sim, index=names, columns=names), "norm_centroid_similarity")
    off = m[~np.isnan(m)]
    print(f"  centroid similarity: min {off.min():.3f} max {off.max():.3f} mean {off.mean():.3f}")
    iu = np.triu_indices(len(names), 1)
    pairs = sorted(zip(sim[iu], [(names[i], names[j]) for i, j in zip(*iu)]))
    print("  most distant:", pairs[:3])
    print("  most similar:", pairs[-3:])


# ==========================================================================
# Figure 11 — per-category shift on every axis, as estimates with intervals
# ==========================================================================


def hodges_lehmann(a: np.ndarray, b: np.ndarray, alpha: float = 0.05):
    """Median of all pairwise differences b - a, with a distribution-free CI.

    The location-shift estimate that pairs with Mann--Whitney: it is built from
    the same ranks, so it adds no assumption the test does not already make,
    and it reports in the units the reader is looking at (share of a book's
    norms) rather than as a rank correlation. The interval is the standard
    order-statistic one, with the rank offset from the normal approximation to
    U -- exact enough at 10 x 100 = 1,000 pairwise differences.
    """
    diffs = np.sort((b[:, None] - a[None, :]).ravel())
    n1, n2 = len(a), len(b)
    w = n1 * n2
    est = float(np.median(diffs))
    mu = w / 2
    sd = np.sqrt(w * (n1 + n2 + 1) / 12)
    k = int(np.floor(mu - stats.norm.ppf(1 - alpha / 2) * sd))
    if k < 1:
        return est, np.nan, np.nan
    return est, float(diffs[k - 1]), float(diffs[w - k])


def axis_shift_table(field: str, categories=None) -> pd.DataFrame:
    """Per-category book-level shift on one axis, with Hodges--Lehmann CIs."""
    source = norm_facets if field == "facet" else norms
    res = book_level_test(source, field, categories)
    counts = (
        source[source[field].notna()]
        .groupby(["corpus", "gutenberg_id", field])
        .size()
        .unstack(field, fill_value=0)
    )
    share = counts.div(counts.sum(axis=1), axis=0)
    est, lo, hi = [], [], []
    for cat in res["category"]:
        e, l, h = hodges_lehmann(
            share.loc["fiction10", cat].to_numpy(float),
            share.loc["top100", cat].to_numpy(float),
        )
        est.append(e), lo.append(l), hi.append(h)
    out = res.assign(axis=field, shift=est, shift_lo=lo, shift_hi=hi)
    return out.merge(run_effect_test(source, field, out["category"]),
                     on="category", how="left")


def fig_divergence():
    """Per-category shift on every axis, as estimates with confidence intervals.

    Replaces the JSD-versus-noise-floor pair this figure used to carry. The
    floor was a workaround for having no test: it asked whether a pooled
    divergence exceeded the divergence between the seven books the corpora
    share. Testing per-book shares answers the same question directly, and
    answers it differently -- pooled chi-square calls deontic force (p=6e-5)
    and norm source (p=4e-19) significant, and neither survives once the
    nesting of norms inside books is respected.
    """
    # Binary axes carry each category twice as exact mirror images; plotting
    # both doubles the ink and says nothing. Keep the affirmative level only.
    frames = [
        axis_shift_table("raz_normative_force", FORCE_ORDER),
        axis_shift_table("raz_governs_info_flow", [True]),
        axis_shift_table("raz_norm_source", ["implicit", "explicit", "both"]),
        axis_shift_table(
            "raz_confidence_qual",
            ["very_certain", "certain", "somewhat_certain", "uncertain"],
        ),
        axis_shift_table("norm_quality_passed", [True]),
        axis_shift_table("facet", norm_ctx.head(8).index.tolist()),
    ]
    d = pd.concat(frames, ignore_index=True)
    d["axis"] = d["axis"].replace({"facet": "raz_context"})
    save_table(d, "axis_shift_book_level", index=False)
    print(d[["axis", "label", "shift", "shift_lo", "shift_hi", "q"]]
          .round(4).to_string(index=False))

    # A category no book ever uses (uncertain: 0 in all 110) plots as a dot on
    # zero with no interval and reads as a measured null rather than an absence.
    plot_d = d[~((d["shift"] == 0) & (d["shift_lo"] == d["shift_hi"]))]
    dropped = d[~d.index.isin(plot_d.index)]["label"].tolist()

    # Forest-plot layout: the axis name takes its own row above its categories,
    # left-aligned with them, so nothing has to share horizontal space. An axis
    # reduced to a single level needs no header -- the axis name is the label.
    ticks, labels, weights, rows = [], [], [], []
    pos = 0.0
    for axis_name, grp in plot_d.groupby("axis", sort=False):
        if len(grp) == 1:
            rows.append((pos, grp.iloc[0]))
            ticks.append(pos), labels.append(axis_label(axis_name)), weights.append("bold")
            pos += 1.6
            continue
        ticks.append(pos), labels.append(axis_label(axis_name)), weights.append("bold")
        pos += 1
        for _, r in grp.iterrows():
            rows.append((pos, r))
            ticks.append(pos), labels.append(display(r["label"])), weights.append("normal")
            pos += 1
        pos += 0.6

    fig, ax = plt.subplots(figsize=(TW, 0.185 * pos + 1.35))
    ax.axvline(0, color=EDGE, lw=0.8, zorder=1)
    for y, r in rows:
        sig = star(r["q"])
        colour = PAL["teal"] if sig else PAL["warmgrey"]
        ax.hlines(y, r["shift_lo"], r["shift_hi"], color=colour, lw=1.4, zorder=3)
        ax.scatter([r["shift"]], [y], s=26, zorder=4,
                   color=colour if sig else "white",
                   edgecolors=colour, linewidths=1.1)
        # The paired same-book run effect, on the same scale. Where this cross
        # sits on top of the dot, the reading list is not what moved.
        ax.scatter([r["run_shift"]], [y], marker="|", s=90, zorder=5,
                   color=PAL["coral"], linewidths=1.4)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=7)
    for lab, w in zip(ax.get_yticklabels(), weights):
        lab.set_fontweight(w)
        lab.set_horizontalalignment("left")
    ax.tick_params(axis="y", pad=92, length=0)
    ax.set_ylim(pos - 0.6, -1)
    ax.set_xlabel("Shift in per-book share, top100 − fiction10 "
                  "(Hodges–Lehmann estimate, 95% CI)")
    ax.set_title("Where the two reading lists diverge")
    ax.grid(True, axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(
        handles=[
            Line2D([], [], marker="o", ls="none", color=PAL["teal"],
                   markeredgecolor=PAL["teal"], label="Differs at BH $q<0.05$"),
            Line2D([], [], marker="o", ls="none", color="white",
                   markeredgecolor=PAL["warmgrey"], label="No detected difference"),
            Line2D([], [], marker="|", ls="none", color=PAL["coral"],
                   markersize=8, markeredgewidth=1.4,
                   label="Same-book run effect (7 paired books)"),
        ],
        loc="lower right", frameon=False, fontsize=6.5,
    )
    fig.tight_layout()
    save_fig(fig, "corpus_divergence")
    if dropped:
        print(f"  dropped from the shift figure (no book uses them): {dropped}")
    sig = d[d["q"] < 0.05]
    gate_run = d.loc[d["axis"] == "norm_quality_passed", "run_shift"].iloc[0]
    save_caption(
        "corpus_divergence",
        "Per-category shift between the two corpora, with confidence intervals",
        f"Each row is one category of one norm-schema axis. The point is the "
        f"Hodges--Lehmann estimate of the shift in that category's per-book "
        f"share (top100 minus fiction10) and the bar its 95\\% distribution-free "
        f"interval; filled points differ at $q<0.05$ under a two-sided "
        f"Mann--Whitney U test with Benjamini--Hochberg correction within the "
        f"axis. Norms are nested in books, so the book is the unit of analysis; "
        f"a test over the {len(norms):,} pooled norms treats correlated "
        f"observations as independent and calls deontic force ($p$=6e-5) and "
        f"norm source ($p$=4e-19) significant when neither separates the "
        f"corpora book by book. {len(sig)} of {len(d)} categories move, on "
        f"{sig['axis'].nunique()} axes "
        f"({', '.join(sorted(axis_label(a).lower() for a in sig['axis'].unique()))}). "
        f"The vertical rule on each row is the paired same-book run effect, "
        f"estimated on the seven books present in both corpora (median "
        f"within-book difference, Wilcoxon signed-rank). The two corpora were "
        f"extracted on different days, so corpus and extraction run are "
        f"confounded, and a row whose rule sits on its dot has moved by an "
        f"amount the extractor reproduces on identical books. That is the case "
        f"for the generalizability gate, where the paired shift "
        f"({pp(float(gate_run))}) accounts for the whole between-corpus "
        f"shift, and about half the case for extractor confidence. The "
        f"intervals are marginal and uncorrected while the fill status is "
        f"corrected, so an interval may exclude zero on a category the test "
        f"does not call. Binary axes are drawn at their affirmative level "
        f"only, the other level being its mirror image.",
        "fig:corpus-divergence",
        ["corpus-descriptives", "fiction10-vs-top100", "mann-whitney",
         "hodges-lehmann", "camera-ready"],
    )


if __name__ == "__main__":
    fig_composition()
    fig_norm_attributes()
    fig_extraction_yield()
    fig_deontic()
    fig_context_vocabulary()
    fig_deontic_by_context()
    fig_context_entropy()
    fig_flow_appropriateness()
    fig_ci_relevance()
    fig_norm_centroids()
    fig_divergence()
    print("done.")
