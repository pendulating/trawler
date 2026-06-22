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
    # Top-100 fiction corpus — inspection & EDA

    The corpus produced by

    ```bash
    python -m dagspaces.common.gutenberg.cli select top-k \
      --k 100 --languages en --only-fiction \
      --out $GUTENBERG_CACHE_ROOT/selections/top100_fiction_en.yaml

    python -m dagspaces.common.gutenberg.cli materialize \
      --selection $GUTENBERG_CACHE_ROOT/selections/top100_fiction_en.yaml \
      --chunk-size 6000 --overlap 1000 \
      --out $GUTENBERG_CACHE_ROOT/chunks_top100_fiction_en.parquet
    ```

    Three artefacts feed this notebook:

    | path | what it has |
    |---|---|
    | `chunks_top100_fiction_en.parquet` | 16,367 chunks × `{gutenberg_id, chunk_id, article_text, chunk_size, book_title, book_author, book_summary}` |
    | `chunks_top100_fiction_en.manifest.json` | materialize summary + selection blob (download_count, languages) |
    | `catalog/catalog_latest.parquet` | full Gutendex snapshot (subjects, bookshelves, author birth/death, formats) — joined on `gutenberg_id` for the 100 selected books |

    Sections:

    1. Load the three artefacts and join into a per-book frame
    2. Corpus totals (books, chunks, characters, ≈tokens)
    3. Per-book contribution — chunks and characters
    4. Chunk-size distribution — verify the 6000-char target and tail
    5. Author coverage — who dominates the voice
    6. Subject / bookshelf distribution — fiction filter quality
    7. Author era — birth-year coverage
    8. Popularity — `download_count` head and distribution
    9. Sample chunk inspection — eyes-on sanity check
    """)
    return


@app.cell
def _():
    import json
    import random
    import sys
    from collections import Counter
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    PROJECT_ROOT = Path("/share/pierson/matt/UAIR")
    NB_DIR = PROJECT_ROOT / "notebooks/normative-simulacra"

    CACHE_ROOT = Path("/share/pierson/matt/zoo/datasets/gutenberg_cache")
    CHUNKS_PARQUET = CACHE_ROOT / "chunks_top100_fiction_en.parquet"
    MANIFEST_JSON = CACHE_ROOT / "chunks_top100_fiction_en.manifest.json"
    CATALOG_PARQUET = CACHE_ROOT / "catalog" / "catalog_latest.parquet"

    pd.set_option("display.max_columns", 80)
    pd.set_option("display.float_format", "{:.2f}".format)

    sys.path.insert(0, str(NB_DIR.parent))
    try:
        from font_utils import load_ibm_plex_sans

        load_ibm_plex_sans()
    except Exception as _e:
        print(f"[font_utils] skipped: {_e}")
    return (
        CATALOG_PARQUET,
        CHUNKS_PARQUET,
        Counter,
        MANIFEST_JSON,
        json,
        mticker,
        np,
        pd,
        plt,
        random,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · Load chunks, manifest, and catalog

    `chunks` is the row-per-chunk frame; `book_meta` is the row-per-book frame
    joined from the selection manifest and Gutendex catalog snapshot. The catalog
    is what carries the genre / subject / author-era metadata that the
    chunks parquet drops for size.
    """)
    return


@app.cell
def _(CATALOG_PARQUET, CHUNKS_PARQUET, MANIFEST_JSON, json, pd):
    chunks = pd.read_parquet(CHUNKS_PARQUET)
    chunks["gutenberg_id"] = chunks["gutenberg_id"].astype(str)
    chunks["n_chars"] = chunks["article_text"].str.len()

    with open(MANIFEST_JSON) as f:
        manifest = json.load(f)

    selection_books = pd.DataFrame(manifest["selection_books"])
    selection_books["gutenberg_id"] = selection_books["gutenberg_id"].astype(str)

    catalog = pd.read_parquet(CATALOG_PARQUET)
    catalog["gutenberg_id"] = catalog["gutenberg_id"].astype(str)
    for _col in ("authors", "subjects", "bookshelves", "languages", "formats"):
        if _col in catalog.columns and catalog[_col].dtype == object:
            catalog[_col] = catalog[_col].map(
                lambda v: json.loads(v) if isinstance(v, str) else v
            )

    selected_ids = set(selection_books["gutenberg_id"])
    catalog_selected = catalog[catalog["gutenberg_id"].isin(selected_ids)].copy()

    book_meta = selection_books.merge(
        catalog_selected[
            ["gutenberg_id", "authors", "subjects", "bookshelves", "copyright", "media_type"]
        ].rename(columns={"authors": "authors_meta"}),
        on="gutenberg_id",
        how="left",
    )

    print(
        f"chunks: {len(chunks):,} rows × {chunks.shape[1]} cols  |  "
        f"books in selection: {len(selection_books)}  |  "
        f"catalog rows joined: {book_meta['subjects'].notna().sum()} / {len(book_meta)}"
    )
    return book_meta, chunks, manifest, selection_books


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · Corpus totals

    Headline numbers: book count, chunk count, total characters, and a rough
    token estimate at `chars/4` (a fine-enough rule-of-thumb for English
    prose — actual BPE tokenisation will vary by tokenizer).
    """)
    return


@app.cell
def _(chunks, manifest, pd):
    _total_chars = int(chunks["n_chars"].sum())
    totals = pd.Series(
        {
            "books_requested": manifest["books_requested"],
            "books_cached": manifest["books_cached"],
            "books_fetch_failed": len(manifest["books_fetch_failed"]),
            "books_failed": len(manifest["books_failed"]),
            "n_chunks": len(chunks),
            "n_unique_books_in_chunks": chunks["gutenberg_id"].nunique(),
            "total_chars": _total_chars,
            "total_tokens_approx": _total_chars // 4,
            "chunk_size_target": manifest["chunk_size"],
            "overlap": manifest["overlap"],
        }
    )
    totals.to_frame("value")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · Per-book contribution

    Two views of the same fact: a few long novels carry far more weight than
    short ones. Top-20 by chunk count + per-book chunk-count histogram. If
    norm extraction is sensitive to over-representation by any single book,
    this is the slice to inspect.
    """)
    return


@app.cell
def _(chunks, mticker, np, plt):
    per_book = (
        chunks.groupby("gutenberg_id")
        .agg(
            book_title=("book_title", "first"),
            book_author=("book_author", "first"),
            n_chunks=("chunk_id", "count"),
            total_chars=("n_chars", "sum"),
            mean_chunk_chars=("n_chars", "mean"),
        )
        .reset_index()
        .sort_values("n_chunks", ascending=False)
    )

    _fig, _axes = plt.subplots(1, 2, figsize=(13, 4.5))

    _ax = _axes[0]
    _top20 = per_book.head(20).iloc[::-1]
    _labels = [
        f"{t[:48]} — {a.split(',')[0]}"
        for t, a in zip(_top20["book_title"], _top20["book_author"])
    ]
    _ax.barh(_labels, _top20["n_chunks"], color="#4c72b0")
    _ax.set_xlabel("chunks (6000 chars, 1000 overlap)")
    _ax.set_title("top 20 books by chunk count")
    _ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    _ax = _axes[1]
    _bins = np.arange(0, per_book["n_chunks"].max() + 25, 25)
    _ax.hist(per_book["n_chunks"], bins=_bins, color="#55a868", edgecolor="white")
    _ax.axvline(
        per_book["n_chunks"].median(),
        ls="--", c="k", lw=1,
        label=f"median = {per_book['n_chunks'].median():.0f}",
    )
    _ax.set_xlabel("chunks per book")
    _ax.set_ylabel("# books")
    _ax.set_title("chunk count distribution (per book)")
    _ax.legend()

    plt.tight_layout()
    plt.gca()
    return (per_book,)


@app.cell
def _(per_book):
    per_book.head(100).reset_index(drop=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · Chunk size distribution

    Target is 6000 characters with 1000-char overlap (paragraph-aware in
    `chunking.py`). Most chunks should be ~6000; the long tail toward smaller
    values is the trailing chunk of each book and paragraph-boundary slack.
    """)
    return


@app.cell
def _(chunks, np, plt):
    _fig, _axes = plt.subplots(1, 2, figsize=(13, 4))

    _ax = _axes[0]
    _ax.hist(chunks["n_chars"], bins=80, color="#4c72b0", edgecolor="white")
    _ax.axvline(6000, ls="--", c="k", lw=1, label="target = 6000")
    _ax.set_xlabel("chunk character length")
    _ax.set_ylabel("# chunks")
    _ax.set_title(f"chunk size — all {len(chunks):,} chunks")
    _ax.legend()

    _ax = _axes[1]
    _trailing = chunks.sort_values(["gutenberg_id", "chunk_id"]).groupby("gutenberg_id").tail(1)
    _interior = chunks.drop(_trailing.index)
    _ax.hist(
        [_interior["n_chars"], _trailing["n_chars"]],
        bins=np.linspace(0, chunks["n_chars"].max(), 60),
        stacked=True,
        color=["#4c72b0", "#dd8452"],
        label=[f"interior ({len(_interior):,})", f"trailing ({len(_trailing):,})"],
        edgecolor="white",
    )
    _ax.set_xlabel("chunk character length")
    _ax.set_title("interior vs. per-book trailing chunk")
    _ax.legend()

    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(chunks):
    chunks["n_chars"].describe().to_frame("char_length_stats")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 · Author coverage

    Which voices dominate the corpus? Books are credited to their primary
    author string from the chunks frame (`book_author`); per-author totals
    sum across all that author's books.
    """)
    return


@app.cell
def _(chunks, plt):
    by_author = (
        chunks.groupby("book_author")
        .agg(
            n_books=("gutenberg_id", "nunique"),
            n_chunks=("chunk_id", "count"),
            total_chars=("n_chars", "sum"),
        )
        .reset_index()
        .sort_values(["n_books", "n_chunks"], ascending=False)
    )

    _fig, _ax = plt.subplots(figsize=(9, 5))
    _top_auth = by_author.head(15).iloc[::-1]
    _ax.barh(_top_auth["book_author"], _top_auth["n_books"], color="#8172b3")
    _ax.set_xlabel("# books in top-100")
    _ax.set_title("top 15 authors by book count")
    plt.tight_layout()
    plt.gca()
    return (by_author,)


@app.cell
def _(by_author):
    by_author.head(20).reset_index(drop=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6 · Subject & bookshelf distribution

    Sanity-checks the `--only-fiction` filter. Subjects use the Library of
    Congress Subject Headings (LCSH); bookshelves are Gutendex's editorial
    categories. Top tags should be saturated with fiction-indicating strings.
    """)
    return


@app.cell
def _(Counter, book_meta, plt):
    def _explode_tags(series):
        c = Counter()
        for tags in series.dropna():
            for t in tags:
                c[t] += 1
        return c

    subj_counts = _explode_tags(book_meta["subjects"])
    shelf_counts = _explode_tags(book_meta["bookshelves"])

    fiction_hits = sum(
        1
        for tags in book_meta["subjects"].dropna()
        if any("fiction" in t.lower() for t in tags)
    )
    print(
        f"books with at least one 'fiction'-tagged subject: "
        f"{fiction_hits} / {len(book_meta)}"
    )

    _fig, _axes = plt.subplots(1, 2, figsize=(14, 6))
    for _ax, _counts, _title, _color in [
        (_axes[0], subj_counts, "top 20 subjects (LCSH)", "#4c72b0"),
        (_axes[1], shelf_counts, "top 20 bookshelves (Gutendex)", "#55a868"),
    ]:
        _items = _counts.most_common(20)[::-1]
        _labels = [k[:55] for k, _ in _items]
        _ax.barh(_labels, [v for _, v in _items], color=_color)
        _ax.set_xlabel("# books")
        _ax.set_title(_title)

    plt.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7 · Author era

    Birth-year histogram from the Gutendex `authors[].birth_year` field
    (multiple authors per book → first author's year used). Skews 19th-century
    by construction: that's where the popular-EN-fiction public-domain mass
    lives.
    """)
    return


@app.cell
def _(book_meta, np, plt):
    def _first_birth_year(authors):
        if authors is None or len(authors) == 0:
            return None
        first = authors[0]
        if isinstance(first, dict):
            return first.get("birth_year")
        return None

    birth_years = book_meta["authors_meta"].map(_first_birth_year).dropna().astype(int)
    print(
        f"birth-year coverage: {len(birth_years)} / {len(book_meta)} books  |  "
        f"range {birth_years.min()}–{birth_years.max()}  |  median {birth_years.median():.0f}"
    )

    _fig, _ax = plt.subplots(figsize=(9, 4))
    _ax.hist(
        birth_years,
        bins=np.arange(birth_years.min() - 25, birth_years.max() + 50, 25),
        color="#c44e52",
        edgecolor="white",
    )
    _ax.set_xlabel("first-author birth year")
    _ax.set_ylabel("# books")
    _ax.set_title("author era distribution")
    plt.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8 · Popularity (`download_count`)

    Selection was top-100 by descending `download_count`, so the head is
    monotone and the tail tells us where the cutoff sat. Log-y on the right
    panel to show the heavy-tailed shape.
    """)
    return


@app.cell
def _(plt, selection_books):
    sb = selection_books.sort_values("download_count", ascending=False).reset_index(drop=True)
    sb["rank"] = sb.index + 1

    _fig, _axes = plt.subplots(1, 2, figsize=(13, 4.5))
    _ax = _axes[0]
    _ax.plot(sb["rank"], sb["download_count"], marker=".", c="#4c72b0")
    _ax.set_xlabel("rank")
    _ax.set_ylabel("download_count")
    _ax.set_title("popularity rank curve")
    _ax.grid(True, alpha=0.3)

    _ax = _axes[1]
    _ax.semilogy(sb["rank"], sb["download_count"], marker=".", c="#dd8452")
    _ax.set_xlabel("rank")
    _ax.set_ylabel("download_count (log)")
    _ax.set_title("same, log-y")
    _ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.gca()
    return (sb,)


@app.cell
def _(sb):
    sb[["rank", "gutenberg_id", "title", "authors", "download_count"]].head(20)
    return


@app.cell
def _(sb):
    sb[["rank", "gutenberg_id", "title", "authors", "download_count"]].tail(10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9 · Sample chunk — eyes-on sanity

    Pick a random chunk and print the first 1500 characters. Watch for:
    boilerplate that escaped the cleaner ("Project Gutenberg" header /
    license footer), broken paragraph boundaries, OCR noise, language drift.
    Re-run the cell to draw a different sample.
    """)
    return


@app.cell
def _(chunks, mo, random):
    _sample = chunks.sample(1, random_state=random.randint(0, 10_000)).iloc[0]
    _preview = _sample["article_text"][:1500]
    mo.md(
        f"""
    **book**: *{_sample['book_title']}* — {_sample['book_author']}
    (`gutenberg_id={_sample['gutenberg_id']}`, chunk {_sample['chunk_id']},
    {_sample['n_chars']} chars)

    ---

    ```
    {_preview}{'…' if len(_sample['article_text']) > 1500 else ''}
    ```
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10 · Quick anomaly checks

    Last pass: anything that should never happen in a clean corpus. If any
    counter is non-zero, the materialize step needs revisiting.
    """)
    return


@app.cell
def _(chunks, pd, selection_books):
    checks = pd.Series(
        {
            "chunks_with_empty_text": int((chunks["n_chars"] == 0).sum()),
            "chunks_under_500_chars": int((chunks["n_chars"] < 500).sum()),
            "chunks_over_target_plus_overlap": int((chunks["n_chars"] > 7100).sum()),
            "chunks_missing_book_title": int(chunks["book_title"].isna().sum()),
            "chunks_missing_book_author": int(chunks["book_author"].isna().sum()),
            "chunks_with_summary": int((chunks["book_summary"].fillna("") != "").sum()),
            "books_not_english_only": int(
                selection_books["languages"].map(lambda L: list(L) != ["en"]).sum()
            ),
            "duplicate_chunk_ids_within_book": int(
                chunks.duplicated(subset=["gutenberg_id", "chunk_id"]).sum()
            ),
        }
    )
    checks.to_frame("count")
    return


if __name__ == "__main__":
    app.run()
