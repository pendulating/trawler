"""Cache path resolution for the Gutenberg tooling."""

from __future__ import annotations

import os
from pathlib import Path

from ..stage_utils import ensure_dotenv

DEFAULT_CACHE_ROOT = "/share/pierson/matt/zoo/datasets/gutenberg_cache"


def cache_root() -> Path:
    ensure_dotenv()
    root = Path(os.environ.get("GUTENBERG_CACHE_ROOT", DEFAULT_CACHE_ROOT))
    root.mkdir(parents=True, exist_ok=True)
    return root


def catalog_path() -> Path:
    p = cache_root() / "catalog"
    p.mkdir(parents=True, exist_ok=True)
    return p / "catalog_latest.parquet"


def raw_dir() -> Path:
    p = cache_root() / "raw"
    p.mkdir(parents=True, exist_ok=True)
    return p


def chunks_dir(chunk_size: int, overlap: int) -> Path:
    p = cache_root() / "chunks" / f"cs{chunk_size}_o{overlap}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def selections_dir() -> Path:
    p = cache_root() / "selections"
    p.mkdir(parents=True, exist_ok=True)
    return p
