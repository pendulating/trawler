"""Concatenate per-book chunk parquets into the canonical chunks.parquet."""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import pandas as pd

from .fetch import BookFetchError, ensure_chunks
from .paths import chunks_dir
from .select import Selection

log = logging.getLogger(__name__)


def materialize_dataset(
    selection: Selection,
    chunk_size: int,
    overlap: int,
    out_path: Path,
    book_summaries: Optional[dict[str, str]] = None,
) -> dict:
    """Build the chunks parquet downstream stages consume.

    Schema: gutenberg_id, chunk_id, article_text, chunk_size, book_title,
    book_author, book_summary.

    Returns a summary dict (counts of selected/cached/missing/written rows).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    summaries = book_summaries or {}
    cached: list[Path] = []
    fetch_failed: list[str] = []
    failed: list[tuple[str, str]] = []

    title_by_id: dict[str, str] = {}
    author_by_id: dict[str, str] = {}
    summary_by_id: dict[str, str] = {}

    for ref in selection.books:
        gid = ref.gutenberg_id
        try:
            cached.append(ensure_chunks(gid, chunk_size=chunk_size, overlap=overlap))
        except BookFetchError as e:
            fetch_failed.append(gid)
            log.warning("fetch failed for %s: %s", gid, e)
            continue
        except Exception as e:  # noqa: BLE001
            failed.append((gid, str(e)))
            continue

        title_by_id[gid] = ref.title
        author_by_id[gid] = "; ".join(ref.authors)
        summary_by_id[gid] = summaries.get(gid, "")

    if not cached:
        raise RuntimeError(
            f"materialize: no books cached. fetch_failed={len(fetch_failed)}, "
            f"failed={len(failed)}"
        )

    parts = [pd.read_parquet(p) for p in cached]
    df = pd.concat(parts, ignore_index=True)
    df["book_title"] = df["gutenberg_id"].map(lambda g: title_by_id.get(str(g), ""))
    df["book_author"] = df["gutenberg_id"].map(lambda g: author_by_id.get(str(g), ""))
    df["book_summary"] = df["gutenberg_id"].map(lambda g: summary_by_id.get(str(g), ""))

    df.to_parquet(out_path, index=False)

    summary = {
        "out_path": str(out_path),
        "chunk_size": chunk_size,
        "overlap": overlap,
        "books_requested": len(selection.books),
        "books_cached": len(cached),
        "books_fetch_failed": fetch_failed,
        "books_failed": failed,
        "rows": int(len(df)),
        "selection_strategy": selection.strategy,
        "selection_params": selection.params,
        "selection_books": [asdict(b) for b in selection.books],
    }
    log.info(
        "materialized %s: %d rows from %d/%d books (fetch_failed=%d, failed=%d)",
        out_path, len(df), len(cached), len(selection.books),
        len(fetch_failed), len(failed),
    )
    return summary


def chunks_dir_for(chunk_size: int, overlap: int) -> Path:
    return chunks_dir(chunk_size, overlap)
