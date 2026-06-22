"""Project Gutenberg corpus tooling.

Builds and queries a durable on-disk cache so we fetch each book once and
reuse it across runs. Pulls text bodies directly from gutenberg.org and
metadata (popularity, authors) from Gutendex.

Cache layout under ``$GUTENBERG_CACHE_ROOT``::

    catalog/
        catalog_latest.parquet              # Gutendex snapshot
    raw/{id}.txt                            # cleaned full text
    raw/{id}.meta.json                      # {title, authors, language, sha256, source_url, fetched_at}
    chunks/cs{cs}_o{ov}/{id}.parquet        # per-book chunks
    selections/{name}.yaml                  # saved selection specs

See ``cli.py`` for the user-facing commands.
"""

from .paths import cache_root

__all__ = ["cache_root"]
