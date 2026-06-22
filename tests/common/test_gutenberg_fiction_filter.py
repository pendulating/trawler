"""Regression tests for `_is_fiction_novel`, compilation detection, and
title-based dedup in dagspaces.common.gutenberg.select.

Each case below is a real (gutenberg_id, subjects, bookshelves) triple pulled
from the Gutendex catalog snapshot on 2026-04-28. The selection target is
*prose narrative fiction* — novels, novellas, short story collections.
Poetry, drama, and heterogeneous compilations ("Complete Works of X",
multi-volume editions) are explicitly out of scope.
"""
from __future__ import annotations

import pandas as pd
import pytest

from dagspaces.common.gutenberg.select import (
    _dedupe_by_normalized_title,
    _is_compilation_title,
    _is_fiction_novel,
    _normalize_title_for_dedup,
)


# (label, subjects, bookshelves) — works that MUST classify as fiction.
KEEP_CASES = [
    (
        "frankenstein-prose-fiction",
        ["Frankenstein's monster (Fictitious character) -- Fiction", "Science fiction"],
        ["Category: Gothic Fiction", "Category: Science-Fiction & Fantasy"],
    ),
    (
        "grimms-fairy-tales-folklore",
        ["Fairy tales -- Germany"],
        ["Category: Children & Young Adult Reading", "Category: Short Stories"],
    ),
    (
        "thousand-and-one-nights-folklore",
        ["Fairy tales -- Arab countries", "Folklore -- Arab countries"],
        ["Category: Adventure", "Category: Mythology, Legends & Folklore"],
    ),
    (
        "le-morte-darthur-romance-shelf",
        ["Arthur, King -- Legends", "Arthurian romances"],
        ["Arthurian Legends", "Category: Historical Novels", "Fantasy"],
    ),
    (
        "poe-short-fiction-volume",
        ["American fiction -- 19th century", "Fantasy fiction",
         "Horror tales, American", "Short stories"],
        ["Category: American Literature", "Category: Crime, Thrillers and Mystery",
         "Category: Short Stories", "Detective Fiction", "Gothic Fiction",
         "Horror", "Mystery Fiction"],
    ),
    (
        "sherlock-holmes-prose-detective-fiction",
        ["Detective and mystery stories",
         "Holmes, Sherlock (Fictitious character) -- Fiction",
         "Private investigators -- England -- Fiction"],
        ["Category: Crime, Thrillers and Mystery", "Detective Fiction"],
    ),
]

# (label, subjects, bookshelves) — works that must NOT classify as fiction.
REJECT_CASES = [
    (
        "wealth-of-nations-economics",
        ["Economics"],
        ["Best Books Ever Listings", "Category: Economics", "Harvard Classics"],
    ),
    (
        "my-friends-the-savages-travel-memoir",
        ["Malay Peninsula -- Description and travel", "Senoi (Southeast Asian people)"],
        ["Category: Adventure", "Category: Archaeology & Anthropology", "Category: Travel Writing"],
    ),
    (
        "kama-sutra-philosophy-mis-shelved",
        ["Love", "Sex"],
        ["Category: Philosophy & Ethics", "Category: Sexuality & Erotica", "Erotic Fiction", "Sociology"],
    ),
    (
        "wollstonecraft-love-letters-correspondence",
        ["Authors, English -- 18th century -- Correspondence",
         "Imlay, Gilbert, 1754?-1828? -- Correspondence", "Love-letters"],
        ["Category: Biographies", "Category: Essays, Letters & Speeches", "Category: Romance"],
    ),
    (
        "bidwell-criminal-autobiography",
        ["Bank fraud -- England", "Bidwell, Austin", "Criminals -- Biography"],
        ["Category: American Literature", "Category: Biographies",
         "Category: Crime, Thrillers and Mystery"],
    ),
    (
        "pure-popularity-shelf-no-genre-signal",
        ["Some neutral subject"],
        ["Best Books Ever Listings"],
    ),
    (
        "genesis-exodus-middle-english-bible-paraphrase",
        ["Bible. Exodus -- History of Biblical events -- Poetry",
         "Bible. Genesis -- History of Biblical events -- Poetry",
         "Christian poetry, English (Middle)"],
        ["Category: British Literature", "Category: Poetry", "Category: Religion/Spirituality"],
    ),
    (
        "paradise-lost-bible-themed-canon",
        ["Adam (Biblical figure) -- Poetry", "Bible. Genesis -- History of Biblical events -- Poetry",
         "Eve (Biblical figure) -- Poetry", "Fall of man -- Poetry"],
        ["Category: British Literature", "Category: Classics of Literature",
         "Category: Poetry", "Category: Religion/Spirituality"],
    ),
    (
        "greek-folklore-scholarly-study",
        ["Folklore -- Greece", "Greece -- Religion", "Mythology, Greek"],
        ["Category: Archaeology & Anthropology", "Category: History - Other",
         "Category: Mythology, Legends & Folklore", "Category: Religion/Spirituality"],
    ),
    # Drama — not narrative fiction. All explicitly rejected.
    (
        "romeo-and-juliet-drama",
        ["Conflict of generations -- Drama", "Tragedies (Drama)"],
        ["Category: Plays/Films/Dramas", "Category: Romance"],
    ),
    (
        "hamlet-drama",
        ["Denmark -- Drama", "Hamlet (Legendary character) -- Drama"],
        ["Category: British Literature", "Category: Plays/Films/Dramas", "Plays"],
    ),
    (
        "dolls-house-drama",
        ["Man-woman relationships -- Drama", "Marriage -- Drama"],
        ["Category: Classics of Literature", "Category: Plays/Films/Dramas"],
    ),
    (
        "pygmalion-drama",
        ["English drama -- 20th century", "London (England) -- Drama"],
        ["Category: British Literature", "Category: Plays/Films/Dramas"],
    ),
    # Poetry — not narrative fiction. All explicitly rejected.
    (
        "leaves-of-grass-poetry",
        ["American poetry -- 19th century"],
        ["Category: American Literature", "Category: Poetry", "Poetry"],
    ),
    (
        "divine-comedy-epic-poetry",
        ["Epic poetry, Italian -- Translations into English"],
        ["Category: Classics of Literature", "Category: Poetry"],
    ),
    (
        "metamorphoses-epic-poetry",
        ["Classical literature", "Latin poetry -- Translations into English",
         "Metamorphosis -- Mythology -- Poetry"],
        ["Category: Mythology, Legends & Folklore", "Category: Poetry"],
    ),
    (
        "odyssey-classical-epic-poetry",
        ["Epic poetry, Greek -- Translations into English",
         "Odysseus, King of Ithaca (Mythological character)"],
        ["Category: Classics of Literature", "Category: Mythology, Legends & Folklore",
         "Category: Poetry", "Classical Antiquity"],
    ),
    (
        "ramayana-epic-poetry",
        ["Epic poetry, Sanskrit -- Translations into English",
         "Folklore -- India", "Rama (Hindu deity) -- Poetry"],
        ["Category: Mythology, Legends & Folklore", "Category: Poetry", "India"],
    ),
    (
        "beowulf-epic-poetry",
        ["Dragons -- Poetry", "Epic poetry, English (Old)", "Monsters -- Poetry"],
        ["Category: British Literature", "Category: Mythology, Legends & Folklore",
         "Category: Poetry", "Poetry"],
    ),
]


@pytest.mark.parametrize("label,subjects,shelves", KEEP_CASES, ids=[c[0] for c in KEEP_CASES])
def test_fiction_keep(label, subjects, shelves):
    assert _is_fiction_novel(subjects, shelves), (
        f"{label}: expected fiction=True but got False\n"
        f"  subjects={subjects}\n  shelves={shelves}"
    )


@pytest.mark.parametrize("label,subjects,shelves", REJECT_CASES, ids=[c[0] for c in REJECT_CASES])
def test_fiction_reject(label, subjects, shelves):
    assert not _is_fiction_novel(subjects, shelves), (
        f"{label}: expected fiction=False but got True\n"
        f"  subjects={subjects}\n  shelves={shelves}"
    )


def test_empty_inputs_rejected():
    """A record with no subjects and no shelves should not be classified as fiction."""
    assert not _is_fiction_novel([], [])
    assert not _is_fiction_novel(None, None)


# --- compilation title detection ---

COMPILATION_TITLES = [
    "The Complete Works of William Shakespeare",
    "The Works of William Shakespeare [Cambridge Edition] [Vol. 3 of 9]",
    "The Works of Edgar Allan Poe — Volume 2",
    "Collected Poems",
    "Collected Stories of Anton Chekhov",
    "Some Novel [Centennial Edition]",
    "Anything Vol. 5",
    "Something Volume 3",
]
NON_COMPILATION_TITLES = [
    "Frankenstein; or, the modern prometheus",
    "Pride and Prejudice",
    "A Study in Scarlet",
    "Of Human Bondage",
    "The House of Mirth",
    "Anne of Green Gables",
    "Adventures of Huckleberry Finn",
    "War and Peace",
]


@pytest.mark.parametrize("title", COMPILATION_TITLES)
def test_compilation_title_detected(title):
    assert _is_compilation_title(title), f"expected compilation: {title!r}"


@pytest.mark.parametrize("title", NON_COMPILATION_TITLES)
def test_non_compilation_title_passes(title):
    assert not _is_compilation_title(title), f"expected NOT compilation: {title!r}"


def test_compilation_title_handles_empty():
    assert not _is_compilation_title("")
    assert not _is_compilation_title(None)


# --- title normalization + dedup ---

def test_normalize_title_collapses_frankenstein_variants():
    a = _normalize_title_for_dedup("Frankenstein; or, the modern prometheus")
    b = _normalize_title_for_dedup("Frankenstein; Or, The Modern Prometheus")
    assert a == b == "frankenstein"


def test_normalize_title_collapses_little_women_variants():
    a = _normalize_title_for_dedup("Little Women")
    b = _normalize_title_for_dedup("Little Women; Or, Meg, Jo, Beth, and Amy")
    assert a == b == "little women"


def test_normalize_title_strips_volume_marker_and_brackets():
    assert _normalize_title_for_dedup(
        "The Works of William Shakespeare [Cambridge Edition] [Vol. 3 of 9]"
    ) == "the works of william shakespeare"


def test_dedupe_keeps_highest_download_within_author_title_group():
    df = pd.DataFrame([
        {"gutenberg_id": "84",    "title": "Frankenstein; or, the modern prometheus",
         "authors": [{"name": "Shelley, Mary Wollstonecraft"}], "download_count": 178271},
        {"gutenberg_id": "41445", "title": "Frankenstein; Or, The Modern Prometheus",
         "authors": [{"name": "Shelley, Mary Wollstonecraft"}], "download_count": 14673},
        {"gutenberg_id": "514",   "title": "Little Women",
         "authors": [{"name": "Alcott, Louisa May"}], "download_count": 13056},
        {"gutenberg_id": "37106", "title": "Little Women; Or, Meg, Jo, Beth, and Amy",
         "authors": [{"name": "Alcott, Louisa May"}], "download_count": 49414},
        {"gutenberg_id": "1342",  "title": "Pride and Prejudice",
         "authors": [{"name": "Austen, Jane"}], "download_count": 107502},
    ])
    out = _dedupe_by_normalized_title(df).sort_values("gutenberg_id").reset_index(drop=True)
    kept_ids = set(out["gutenberg_id"])
    # Highest-download winner per (author, normalized title) group:
    assert kept_ids == {"84", "37106", "1342"}, f"unexpected survivors: {kept_ids}"


def test_dedupe_empty_frame_is_noop():
    empty = pd.DataFrame(columns=["gutenberg_id", "title", "authors", "download_count"])
    assert _dedupe_by_normalized_title(empty).empty


def test_dedupe_preserves_distinct_works_by_same_author():
    df = pd.DataFrame([
        {"gutenberg_id": "98",   "title": "A Tale of Two Cities",
         "authors": [{"name": "Dickens, Charles"}], "download_count": 50000},
        {"gutenberg_id": "1400", "title": "Great Expectations",
         "authors": [{"name": "Dickens, Charles"}], "download_count": 45000},
        {"gutenberg_id": "730",  "title": "Oliver Twist",
         "authors": [{"name": "Dickens, Charles"}], "download_count": 40000},
    ])
    out = _dedupe_by_normalized_title(df)
    assert len(out) == 3, "distinct novels by same author should NOT be deduped"
