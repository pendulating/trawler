"""Per-book text and chunk fetch from gutenberg.org with on-disk cache."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

from .chunking import chunk_text, clean_gutenberg_boilerplate
from .paths import chunks_dir, raw_dir

log = logging.getLogger(__name__)

GUTENBERG_URL_TEMPLATES = (
    "https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt",
    "https://www.gutenberg.org/files/{gid}/{gid}-0.txt",
    "https://www.gutenberg.org/files/{gid}/{gid}.txt",
)
# HTML fallbacks for the rare books where only HTML editions exist.
# We deliberately exclude `{gid}_index.html` / `{gid}-index.html` — those are
# LibriVox audiobook directory listings (track names + mp3/ogg filenames),
# not source text, and were the entire reason the legacy .txt-only fetcher
# 404'd on these ids.
GUTENBERG_HTML_URL_TEMPLATES = (
    "https://www.gutenberg.org/cache/epub/{gid}/pg{gid}-images.html",
    "https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.html",
    "https://www.gutenberg.org/files/{gid}/{gid}-h/{gid}-h.htm",
)
USER_AGENT = "TrawlerPipeline/1.0 (COLM 2026 research; mwf62@cornell.edu)"
REQUEST_TIMEOUT = 60
INTER_REQUEST_DELAY_S = 1.0  # be polite to gutenberg.org
# HTML-derived books that come back smaller than this are almost certainly
# audiobook directory listings or rights pages, not a real source text.
MIN_HTML_TEXT_CHARS = 8000
# Books that produce less than this after boilerplate stripping are stubs —
# typically video/film releases (e.g. id 23053 "Night of the Living Dead",
# whose only "text" is a paragraph describing the .mpeg release) or other
# non-text gutenberg objects. Below this we treat as fetch_failed.
MIN_BOOK_TEXT_CHARS = 2000


class BookFetchError(Exception):
    """All known URLs for this Gutenberg id failed."""


def _http_get(url: str) -> str | None:
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT)
        if r.status_code == 200:
            r.encoding = r.apparent_encoding or r.encoding
            return r.text
        if r.status_code == 404:
            return None
        log.warning("HTTP %d on %s", r.status_code, url)
    except Exception as e:  # noqa: BLE001
        log.warning("error fetching %s: %s", url, e)
    return None


def _html_to_text(html: str) -> str:
    """Strip HTML tags + scripts/styles, preserving paragraph breaks."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "head"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _raw_paths(gid: str) -> tuple[Path, Path]:
    base = raw_dir() / gid
    return base.with_suffix(".txt"), base.with_suffix(".meta.json")


def fetch_gutenberg_text(gid: str) -> tuple[str, str]:
    """Try each URL template; return (raw_text, source_url) or raise.

    Plain `.txt` URLs are tried first. If all 404, falls back to HTML
    editions and converts to plain text — needed for audiobook releases
    where the HTML is the only text-bearing format. HTML results shorter
    than ``MIN_HTML_TEXT_CHARS`` are treated as audio-directory placeholders
    and rejected so the caller can register them as fetch_failed.
    """
    last_url = None
    for tmpl in GUTENBERG_URL_TEMPLATES:
        url = tmpl.format(gid=gid)
        last_url = url
        text = _http_get(url)
        if text:
            return text, url
        time.sleep(0.2)

    for tmpl in GUTENBERG_HTML_URL_TEMPLATES:
        url = tmpl.format(gid=gid)
        last_url = url
        html = _http_get(url)
        if not html:
            time.sleep(0.2)
            continue
        text = _html_to_text(html)
        if len(text) < MIN_HTML_TEXT_CHARS:
            log.info(
                "html at %s yielded only %d chars (<%d); treating as no-text",
                url, len(text), MIN_HTML_TEXT_CHARS,
            )
            time.sleep(0.2)
            continue
        return text, url

    raise BookFetchError(f"no URL succeeded for gutenberg_id={gid} (last={last_url})")


def fetch_text_from_url(url: str) -> str:
    """For non-gutenberg.org sources (e.g. Gutenberg Australia)."""
    text = _http_get(url)
    if not text:
        raise BookFetchError(f"failed to fetch {url}")
    return text


def ensure_text(
    gutenberg_id: str | int,
    source_url: str | None = None,
    force: bool = False,
    polite_delay: float = INTER_REQUEST_DELAY_S,
) -> Path:
    """Materialize cleaned full text at ``raw/{id}.txt`` (idempotent).

    If ``source_url`` is given, fetches that URL directly (e.g. Gutenberg
    Australia for in-copyright texts); otherwise probes gutenberg.org.
    """
    gid = str(gutenberg_id)
    txt_path, meta_path = _raw_paths(gid)
    if txt_path.is_file() and meta_path.is_file() and not force:
        return txt_path

    if source_url:
        raw, used_url = fetch_text_from_url(source_url), source_url
    else:
        raw, used_url = fetch_gutenberg_text(gid)
    if polite_delay:
        time.sleep(polite_delay)

    cleaned = clean_gutenberg_boilerplate(raw)
    if not cleaned:
        raise RuntimeError(f"cleaned text empty for {gid} (raw was {len(raw)} chars from {used_url})")
    if len(cleaned) < MIN_BOOK_TEXT_CHARS:
        # Stub releases (video/film descriptions, placeholder pages). Surface
        # as fetch_failed so the manifest records them without polluting the
        # raw cache.
        raise BookFetchError(
            f"text for {gid} too short ({len(cleaned)} < {MIN_BOOK_TEXT_CHARS} chars) "
            f"from {used_url}; likely a non-text release"
        )

    txt_path.write_text(cleaned, encoding="utf-8")
    meta_path.write_text(
        json.dumps(
            {
                "gutenberg_id": gid,
                "source_url": used_url,
                "raw_text_chars": len(raw),
                "clean_text_chars": len(cleaned),
                "sha256": hashlib.sha256(cleaned.encode("utf-8")).hexdigest(),
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log.info("cached %s (%d chars) from %s", txt_path.name, len(cleaned), used_url)
    return txt_path


def ensure_chunks(
    gutenberg_id: str | int,
    chunk_size: int = 6000,
    overlap: int = 1000,
    force: bool = False,
    source_url: str | None = None,
) -> Path:
    """Materialize per-book chunks parquet, return its path.

    Schema: gutenberg_id (str), chunk_id (int), article_text (str), chunk_size (int).
    """
    gid = str(gutenberg_id)
    out = chunks_dir(chunk_size, overlap) / f"{gid}.parquet"
    if out.is_file() and not force:
        return out

    txt_path = ensure_text(gid, source_url=source_url, force=force)
    text = txt_path.read_text(encoding="utf-8")
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        raise RuntimeError(f"chunking produced 0 chunks for {gid}")

    df = pd.DataFrame(
        {
            "gutenberg_id": gid,
            "chunk_id": list(range(len(chunks))),
            "article_text": chunks,
            "chunk_size": [len(c) for c in chunks],
        }
    )
    df.to_parquet(out, index=False)
    log.info("cached %s (%d chunks)", out.name, len(chunks))
    return out
