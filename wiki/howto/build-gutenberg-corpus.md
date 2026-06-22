# How to build a Gutenberg corpus

The `dagspaces.common.gutenberg` module pulls books from Project Gutenberg
**once** into a durable on-disk cache, then materializes arbitrary subsets
(top-K by popularity, top-K authors × N books each, or explicit ID lists)
as a `chunks.parquet` ready for the historical_norms pipelines.

It replaces the per-run HTTP fetch in
`dagspaces/historical_norms/stages/fetch_gutenberg.py` for any new corpus
work. The legacy stage still works for the curated 10-book COLM list.

## Cache layout

Rooted at `$GUTENBERG_CACHE_ROOT` (default `/share/pierson/matt/zoo/datasets/gutenberg_cache`):

```
catalog/catalog_latest.parquet    # Gutendex snapshot, sorted by download_count
raw/{id}.txt                      # boilerplate-stripped full text
raw/{id}.meta.json                # source_url, sha256, fetched_at
chunks/cs{cs}_o{ov}/{id}.parquet  # per-book chunks (one parquet per book)
selections/{name}.yaml            # saved selection specs
```

Per-book parquets are keyed by chunking parameters, so changing
`chunk_size` / `overlap` rebuilds chunks but reuses cached raw text.

## Recipes

### One-time setup

```bash
# Add to server.env
GUTENBERG_CACHE_ROOT=/share/pierson/matt/zoo/datasets/gutenberg_cache

# Snapshot Gutendex catalog (default: 160 pages = ~5000 most-popular EN books)
python -m dagspaces.common.gutenberg.cli refresh-catalog

# Catalog is reused for 30 days; --force or --max-age-days N to rebuild sooner.
```

### Top-K most-popular books

```bash
python -m dagspaces.common.gutenberg.cli select top-k \
  --k 50 --languages en \
  --out $GUTENBERG_CACHE_ROOT/selections/top50_en.yaml

python -m dagspaces.common.gutenberg.cli materialize \
  --selection $GUTENBERG_CACHE_ROOT/selections/top50_en.yaml \
  --chunk-size 6000 --overlap 1000 \
  --out $GUTENBERG_CACHE_ROOT/chunks_top50_en.parquet
```

### Fiction-only top-K

Pass `--only-fiction` to drop textbooks, reference works, history,
philosophy, and government documents. Audiobook releases (`media_type=Sound`
in Gutendex) are dropped unconditionally — those are LibriVox recordings
whose only HTML format is an audio-track directory listing, not source text.

```bash
python -m dagspaces.common.gutenberg.cli select top-k \
  --k 1000 --languages en --only-fiction \
  --out $GUTENBERG_CACHE_ROOT/selections/top1000_fiction_en.yaml
```

A book is classified as fiction if any subject contains "fiction"
(LCSH "-- Fiction" suffix, "Domestic fiction", etc.) or any bookshelf
membership keyword matches (`novel`, `fiction`, `romance`, `mystery`,
`adventure`, `fantasy`, `horror`, `science fiction`, `children's lit`,
`gothic`, `detective`, `fairy tale`, `short stories`, `best books ever`).
Drama and poetry surface when their subjects are explicitly fiction-tagged.

Fiction makes up ~22% of the Gutendex popular tail, so for top-K fiction
you need a catalog roughly K/0.22 books wide — for K=1000, refresh with
`--max-pages 200` (≈6400 books).

### Top-K authors × N books each

A book co-authored by multiple selected authors contributes its full
`download_count` to each author's score, then is included once in the
final book set.

```bash
python -m dagspaces.common.gutenberg.cli select top-authors \
  --k-authors 10 --n 5 --languages en \
  --out $GUTENBERG_CACHE_ROOT/selections/top10authors_5books.yaml

python -m dagspaces.common.gutenberg.cli materialize \
  --selection $GUTENBERG_CACHE_ROOT/selections/top10authors_5books.yaml \
  --chunk-size 6000 --overlap 1000 \
  --out $GUTENBERG_CACHE_ROOT/chunks_top10authors_5books.parquet
```

### Wire into the historical_norms pipeline

Point `FICTION_CHUNKS_PATH` at the materialized parquet, then run the
prefetched pipeline:

```bash
export FICTION_CHUNKS_PATH=$GUTENBERG_CACHE_ROOT/chunks_top50_en.parquet
python -m dagspaces.historical_norms.cli pipeline=COLM_norms_fiction_prefetched
```

The output schema is a superset of the legacy `chunks.parquet` —
`gutenberg_id, chunk_id, article_text, chunk_size, book_title,
book_author, book_summary` — so `norm_reasoning`, `norm_extraction`,
`norm_role_abstraction`, `ci_reasoning`, and `ci_extraction` consume it
unchanged. `book_summary` is left blank by default; pass
`--summaries-json data/fiction_novel_summaries.json` to `materialize` to
populate it from a Wikipedia-derived JSON map.

## Notes

- **Polite to gutenberg.org**: 1s sleep between fetches. A 50-book cold
  build takes ~1 minute; subsequent runs are pure-cache reads (~1s).
- **In-copyright texts**: 1984 and similar are not on gutenberg.org. The
  `Selection.books` schema accepts a `source_url` per book, which the
  fetcher uses verbatim instead of probing `gutenberg.org`. (TODO: expose
  this in the CLI when the use case comes up.)
- **Catalog scope**: Gutendex's `?sort=popular` returns books in
  descending `download_count`. The default 160 pages cover the popular
  tail well; raise `--max-pages` to widen.
- **Author scoring** explodes co-authors before grouping, so a book with
  three listed authors counts once toward each. To group only by first
  author, edit `select.top_k_authors_n_books` (`_author_names` returns
  the full list).
