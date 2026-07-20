"""Gutendex catalog snapshotter.

Pulls book metadata (title, authors, languages, subjects, bookshelves,
download_count) from https://gutendex.com — a JSON API over the official
Project Gutenberg catalog — and caches it as a single parquet snapshot.

Default sort is descending download_count, which means a small `max_pages`
covers the popular tail useful for top-K and top-author selection.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .paths import catalog_path

log = logging.getLogger(__name__)

GUTENDEX_BASE = "https://gutendex.com/books/"
DEFAULT_MAX_PAGES = 160         # ~5000 books at 32/page
DEFAULT_PAGE_SIZE = 32          # Gutendex returns 32/page; not configurable upstream
REQUEST_TIMEOUT = 30
RETRY_BACKOFF_S = (1, 3, 8)


def _get(url: str, params: dict | None = None) -> dict:
    last_err: Exception | None = None
    for delay in (0, *RETRY_BACKOFF_S):
        if delay:
            time.sleep(delay)
        try:
            r = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            if r.status_code == 200:
                return r.json()
            last_err = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise RuntimeError(f"gutendex request failed: {url} ({last_err})")


def _flatten_book(book: dict[str, Any]) -> dict[str, Any]:
    authors = book.get("authors") or []
    return {
        "gutenberg_id": str(book.get("id")),
        "title": book.get("title") or "",
        "authors": [
            {
                "name": a.get("name") or "",
                "birth_year": a.get("birth_year"),
                "death_year": a.get("death_year"),
            }
            for a in authors
        ],
        "languages": list(book.get("languages") or []),
        "subjects": list(book.get("subjects") or []),
        "bookshelves": list(book.get("bookshelves") or []),
        "download_count": int(book.get("download_count") or 0),
        "copyright": book.get("copyright"),
        "media_type": book.get("media_type") or "",
        "formats": dict(book.get("formats") or {}),
    }


def refresh_catalog(
    languages: Iterable[str] = ("en",),
    max_pages: int = DEFAULT_MAX_PAGES,
    max_age_days: float = 30.0,
    force: bool = False,
    out_path: Path | None = None,
) -> Path:
    """Snapshot Gutendex by descending download_count to ``catalog_latest.parquet``.

    If a snapshot newer than ``max_age_days`` already exists and ``force=False``,
    this is a no-op.
    """
    out = Path(out_path) if out_path else catalog_path()
    out.parent.mkdir(parents=True, exist_ok=True)

    if not force and out.is_file():
        age_days = (time.time() - out.stat().st_mtime) / 86400.0
        if age_days < max_age_days:
            log.info("catalog %s is %.1f days old (< %.1f); skipping refresh", out, age_days, max_age_days)
            return out

    rows: list[dict] = []
    params = {
        "languages": ",".join(languages),
        "sort": "popular",   # Gutendex: descending download_count
    }
    next_url: str | None = GUTENDEX_BASE
    page = 0
    while next_url and page < max_pages:
        page += 1
        log.info("fetching page %d: %s", page, next_url)
        payload = _get(next_url, params=params if page == 1 else None)
        for book in payload.get("results") or []:
            rows.append(_flatten_book(book))
        next_url = payload.get("next")

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("gutendex returned 0 books")

    df["fetched_at"] = datetime.now(timezone.utc).isoformat()
    # JSON-encode list/dict columns so parquet stays portable across engines.
    for col in ("authors", "languages", "subjects", "bookshelves", "formats"):
        df[col] = df[col].map(lambda v: json.dumps(v, ensure_ascii=False))

    df.to_parquet(out, index=False)
    log.info("wrote %s: %d books across %d pages", out, len(df), page)
    return out


def load_catalog() -> pd.DataFrame:
    p = catalog_path()
    if not p.is_file():
        raise FileNotFoundError(
            f"catalog not found at {p}. Run: "
            "python -m dagspaces.common.gutenberg.cli refresh-catalog"
        )
    df = pd.read_parquet(p)
    for col in ("authors", "languages", "subjects", "bookshelves", "formats"):
        if col in df.columns:
            df[col] = df[col].map(json.loads)
    return df
