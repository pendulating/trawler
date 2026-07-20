"""Selection strategies over the Gutendex catalog."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass, field

import pandas as pd

from .catalog import load_catalog

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class BookRef:
    gutenberg_id: str
    title: str
    authors: tuple[str, ...]
    download_count: int
    languages: tuple[str, ...]


@dataclass
class Selection:
    """A selection of books plus the parameters that produced it."""
    strategy: str                                        # "top_k" | "top_authors_n" | "ids"
    params: dict                                         # human-readable params
    books: list[BookRef]
    author_rankings: list[dict] = field(default_factory=list)  # only for top_authors_n


# Library-of-Congress-style subject strings that mark a work as prose
# narrative fiction. LCSH "-- Fiction" suffix is the primary signal; the
# explicit set below catches works tagged only with colloquial genre subjects.
# Poetry and drama are deliberately excluded — `only_fiction` means *prose
# narrative fiction* (novels, novellas, short stories).
_FICTION_SUBJECT_LITERALS = frozenset({
    "Adventure stories", "Detective and mystery stories", "Fairy tales",
    "Fantasy fiction", "Ghost stories", "Historical fiction", "Horror tales",
    "Love stories", "Science fiction", "Sea stories", "Short stories",
    "Western stories", "Bildungsromans", "Domestic fiction",
})
# Substrings that, when found in any subject, mark a work as prose narrative
# fiction or folklore (prose folk tales).
_FICTION_SUBJECT_SUBSTRINGS = (
    "fiction",        # LCSH "-- Fiction" suffix
    "fairy tales",    # "Fairy tales -- Germany"
    "folklore",
)
# Subject markers that DISQUALIFY a book even if it appears on a
# fiction-flavored shelf. Catches biographies, real correspondence, travel
# memoirs, economics treatises, religious texts, poetry, drama, etc.
_NON_FICTION_SUBJECT_SUBSTRINGS = (
    "biography",
    "autobiography",
    "memoirs",
    "correspondence",
    "diaries",
    "description and travel",
    "economics",
    "bible.",          # "Bible. Genesis -- ...", "Bible. Exodus -- ..."
    "poetry",          # "American poetry", "Epic poetry, Greek", "-- Poetry"
    "drama",           # "Marriage -- Drama", "Tragedies (Drama)"
    "tragedies",
    "comedies",
)
_FICTION_SHELF_KEYWORDS = (
    "novel", "fiction", "romance", "mystery", "adventure", "fantasy",
    "horror", "science fiction", "children's lit",
    "gothic", "detective", "fairy tale", "short stories",
)
# Shelf markers that DISQUALIFY a book — catches works mis-shelved on
# fiction-named shelves (e.g. Kama Sutra on "Erotic Fiction"), poetry and
# play collections, and works classified as philosophy / sociology / biography
# / religion / scholarly studies.
_NON_FICTION_SHELF_SUBSTRINGS = (
    "biograph",
    "philosophy",
    "sociology",
    "travel writing",
    "essays, letters",
    "economics",
    "religion/spirituality",   # drops biblical paraphrase + religious-themed canon
    "archaeology",             # drops anthropology / "study in survivals" works
    "poetry",                  # poetry collections (Whitman, Dante, Ovid, etc.)
    "plays",                   # "Category: Plays/Films/Dramas", "Plays"
    "drama",
)

# Title patterns that identify heterogeneous compilations (multi-volume
# editions, "Complete Works of X", etc.). Even if their subject classification
# looks like prose fiction, a compilation may mix poetry, drama, essays, and
# fiction, so we exclude them as a class. Use single-novel editions instead.
_COMPILATION_TITLE_PATTERNS = (
    re.compile(r"(?i)\bcomplete works\b"),
    re.compile(r"(?i)\bthe works of\b"),
    re.compile(r"(?i)\bcollected (?:works|poems|stories|essays|letters|plays)\b"),
    re.compile(r"(?i)\bvol\.?\s*\d+\b"),
    re.compile(r"(?i)\bvolume\s+\d+\b"),
    re.compile(r"\[[^\]]*?[Ee]dition[^\]]*?\]"),  # "[Cambridge Edition]"
)


def _is_fiction_novel(subjects: list[str], bookshelves: list[str]) -> bool:
    """Heuristic: does this Gutendex record describe prose narrative fiction?

    Two-stage filter:
      1. Disqualify if any subject or shelf carries a non-fiction marker
         (biography, correspondence, travel writing, economics, philosophy,
         sociology, religion, poetry, drama). These override any positive
         shelf match.
      2. Otherwise, accept if any subject contains a fiction substring,
         matches a fiction literal, or any shelf carries a fiction-flavored
         keyword.

    "Narrative fiction" means prose novels, novellas, and short story
    collections — *not* poetry or drama (which surface elsewhere in the
    catalog and would need their own selection path).
    """
    subj_low = [s.lower() for s in (subjects or [])]
    shelf_low = [sh.lower() for sh in (bookshelves or [])]

    for s in subj_low:
        if any(nf in s for nf in _NON_FICTION_SUBJECT_SUBSTRINGS):
            return False
    for sh in shelf_low:
        if any(nf in sh for nf in _NON_FICTION_SHELF_SUBSTRINGS):
            return False

    for s, raw in zip(subj_low, subjects or []):
        if any(fs in s for fs in _FICTION_SUBJECT_SUBSTRINGS):
            return True
        if raw in _FICTION_SUBJECT_LITERALS:
            return True
    for sh in shelf_low:
        if any(k in sh for k in _FICTION_SHELF_KEYWORDS):
            return True
    return False


def _is_compilation_title(title: str) -> bool:
    """True if `title` looks like a heterogeneous multi-work compilation
    (e.g. "The Works of X", "Complete Works of X", "[Cambridge Edition]
    [Vol. 3 of 9]")."""
    if not title:
        return False
    return any(p.search(title) for p in _COMPILATION_TITLE_PATTERNS)


def _normalize_title_for_dedup(title: str) -> str:
    """Normalize a title for near-duplicate detection within an author.

    Strips subtitles after ";" / ":" / "—", bracketed edition tags, volume
    markers, and punctuation. Used to fold e.g. "Frankenstein; or, The
    Modern Prometheus" and "Frankenstein; Or, The Modern Prometheus" into
    a single key.
    """
    if not title:
        return ""
    t = title.lower()
    t = re.sub(r"[;:—–-].*$", "", t)              # strip subtitles
    t = re.sub(r"\s*\[.*?\]\s*", " ", t)          # "[Cambridge Edition]"
    t = re.sub(r"\bvol\.?\s*\d+(?:\s*of\s*\d+)?\b", "", t)
    t = re.sub(r"\bvolume\s+\d+\b", "", t)
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _dedupe_by_normalized_title(df: pd.DataFrame) -> pd.DataFrame:
    """Within each first-author group, fold rows with the same normalized
    title down to the highest-download row. Preserves the input sort order
    among surviving rows."""
    if df.empty:
        return df
    work = df.copy()
    work["_first_author"] = work["authors"].map(
        lambda a: (a[0].get("name", "") if a and isinstance(a[0], dict) else "")
    )
    work["_norm_title"] = work["title"].map(_normalize_title_for_dedup)
    # idxmax over download_count per group → row to keep
    keep_idx = (
        work.reset_index()
        .sort_values("download_count", ascending=False)
        .drop_duplicates(subset=["_first_author", "_norm_title"], keep="first")
        ["index"]
    )
    surviving = df.loc[df.index.isin(keep_idx)]
    return surviving


def _filter_catalog(
    df: pd.DataFrame,
    languages: Iterable[str],
    min_downloads: int,
    *,
    text_only: bool = True,
    only_fiction: bool = False,
) -> pd.DataFrame:
    langs = set(languages)
    mask = df["languages"].map(lambda xs: bool(langs.intersection(xs or [])))
    out = df[mask & (df["download_count"] >= min_downloads)].copy()
    if text_only and "media_type" in out.columns:
        # Drops `Sound` (LibriVox audiobook releases — no source text)
        # and any future non-text media types.
        out = out[out["media_type"] == "Text"]
    if only_fiction:
        keep = out.apply(
            lambda r: _is_fiction_novel(r.get("subjects") or [], r.get("bookshelves") or []),
            axis=1,
        )
        out = out[keep]
        # Also drop heterogeneous compilation titles ("Complete Works of X",
        # "[Vol. 3 of 9]", etc.) — they mix poetry/drama with prose fiction.
        compilations = out["title"].map(_is_compilation_title)
        out = out[~compilations]
        out = _dedupe_by_normalized_title(out)
    return out.sort_values("download_count", ascending=False)


def _author_names(authors: list[dict]) -> list[str]:
    return [a.get("name", "") for a in (authors or []) if a.get("name")]


def _to_book_ref(row: pd.Series) -> BookRef:
    return BookRef(
        gutenberg_id=str(row["gutenberg_id"]),
        title=row["title"],
        authors=tuple(_author_names(row["authors"])),
        download_count=int(row["download_count"]),
        languages=tuple(row["languages"]),
    )


def top_k_by_popularity(
    k: int,
    languages: Iterable[str] = ("en",),
    min_downloads: int = 0,
    catalog: pd.DataFrame | None = None,
    *,
    only_fiction: bool = False,
) -> Selection:
    df = catalog if catalog is not None else load_catalog()
    df = _filter_catalog(df, languages, min_downloads, only_fiction=only_fiction)
    head = df.head(k)
    books = [_to_book_ref(r) for _, r in head.iterrows()]
    return Selection(
        strategy="top_k",
        params={
            "k": k,
            "languages": list(languages),
            "min_downloads": min_downloads,
            "only_fiction": only_fiction,
        },
        books=books,
    )


def top_k_authors_n_books(
    k_authors: int,
    n_per_author: int,
    languages: Iterable[str] = ("en",),
    min_downloads: int = 0,
    catalog: pd.DataFrame | None = None,
    *,
    only_fiction: bool = False,
) -> Selection:
    """Pick the K most-popular authors (by summed download_count across their
    books), then for each take their top-N books. A book co-authored by
    multiple selected authors is included once.

    Co-authored books contribute their full download_count to *each* author's
    score (per the explode-on-author rule).
    """
    df = catalog if catalog is not None else load_catalog()
    df = _filter_catalog(df, languages, min_downloads, only_fiction=only_fiction)
    if df.empty:
        return Selection(
            strategy="top_authors_n",
            params={"k_authors": k_authors, "n_per_author": n_per_author,
                    "languages": list(languages), "min_downloads": min_downloads,
                    "only_fiction": only_fiction},
            books=[],
        )

    df = df.assign(_author_names=df["authors"].map(_author_names))
    exploded = df.explode("_author_names").rename(columns={"_author_names": "author"})
    exploded = exploded[exploded["author"].astype(bool)]

    author_scores = (
        exploded.groupby("author")["download_count"]
        .sum()
        .sort_values(ascending=False)
    )
    top_authors = author_scores.head(k_authors)
    rankings = [
        {"author": a, "summed_downloads": int(s)}
        for a, s in top_authors.items()
    ]

    selected_ids: dict[str, BookRef] = {}
    for author in top_authors.index:
        author_books = (
            exploded[exploded["author"] == author]
            .sort_values("download_count", ascending=False)
            .head(n_per_author)
        )
        for _, row in author_books.iterrows():
            gid = str(row["gutenberg_id"])
            if gid not in selected_ids:
                selected_ids[gid] = _to_book_ref(row)

    return Selection(
        strategy="top_authors_n",
        params={"k_authors": k_authors, "n_per_author": n_per_author,
                "languages": list(languages), "min_downloads": min_downloads,
                "only_fiction": only_fiction},
        books=list(selected_ids.values()),
        author_rankings=rankings,
    )


def select_by_ids(
    ids: Iterable[str | int],
    catalog: pd.DataFrame | None = None,
) -> Selection:
    df = catalog if catalog is not None else load_catalog()
    wanted = {str(x) for x in ids}
    sub = df[df["gutenberg_id"].astype(str).isin(wanted)].copy()
    found = {str(r["gutenberg_id"]): _to_book_ref(r) for _, r in sub.iterrows()}
    missing = sorted(wanted - set(found))
    if missing:
        log.warning("ids missing from catalog (will be skipped): %s", missing[:10])
    return Selection(
        strategy="ids",
        params={"ids": sorted(wanted), "missing_in_catalog": missing},
        books=list(found.values()),
    )
