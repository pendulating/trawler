"""Fetch plot summaries from Wikipedia for the COLM fiction novels.

Saves a static JSON file at data/fiction_novel_summaries.json with
title, author, and truncated plot summary for each novel. Run once;
the JSON is committed and used by the norm extraction pipeline.

Usage:
    python scripts/fetch_novel_summaries.py
"""

import json
import os
import sys
import textwrap

import wikipediaapi

# Gutenberg ID → (Wikipedia page title, author)
NOVELS = {
    "1342": ("Pride and Prejudice", "Jane Austen"),
    "11": ("Alice's Adventures in Wonderland", "Lewis Carroll"),
    "145": ("Middlemarch", "George Eliot"),
    "541": ("The Age of Innocence", "Edith Wharton"),
    "1023": ("Bleak House", "Charles Dickens"),
    "135": ("Les Misérables", "Victor Hugo"),
    "1399": ("Anna Karenina", "Leo Tolstoy"),
    "4078": ("The Picture of Dorian Gray", "Oscar Wilde"),
    "1184": ("The Count of Monte Cristo", "Alexandre Dumas"),
    "1984": ("Nineteen Eighty-Four", "George Orwell"),
}

SUMMARY_WARN_WORDS = 2000


def _find_plot_section(page):
    """Find the plot/synopsis section in a Wikipedia page."""
    candidates = ("plot", "plot summary", "synopsis", "story", "summary",
                  "plot introduction", "narrative")

    def _search(sections):
        for section in sections:
            title_lower = section.title.lower().strip()
            if title_lower in candidates:
                # Prefer section with subsections (concatenated) for longer novels
                full_text = section.full_text() if hasattr(section, "full_text") else section.text
                return full_text
            # Recurse into subsections
            result = _search(section.sections)
            if result:
                return result
        return None

    return _search(page.sections)


def _strip_wiki_headers(text):
    """Remove Wikipedia section headers from the start of plot summaries."""
    import re
    # Strip leading headers like "Plot", "Synopsis", "Summary",
    # "Plot summary", "Plot introduction", "Volume I: Fantine", etc.
    text = re.sub(
        r"^(?:Plot summary|Plot introduction|Synopsis|Summary|Plot)\s*",
        "", text, count=1
    )
    # Strip sub-section headers like "Volume I: Fantine" or "Marseille and Château d'If"
    # (lines that are short, title-cased, and followed by a longer sentence)
    text = re.sub(r"^[A-Z][^\n]{0,60}\n", "", text, count=1)
    return text.strip()


def _warn_if_long(title, text, threshold=SUMMARY_WARN_WORDS):
    """Log a warning if the summary exceeds the word threshold."""
    n = len(text.split())
    if n > threshold:
        print(f"  WARNING: '{title}' summary is {n} words (>{threshold})", file=sys.stderr)


def main():
    wiki = wikipediaapi.Wikipedia(
        user_agent="TrawlerPipeline/1.0 (COLM 2026 research; mwf62@cornell.edu)",
        language="en",
    )

    summaries = {}
    for gid, (title, author) in NOVELS.items():
        page = wiki.page(title)
        if not page.exists():
            print(f"  WARNING: Wikipedia page not found for '{title}'", file=sys.stderr)
            summaries[gid] = {"title": title, "author": author, "summary": ""}
            continue

        plot_text = _find_plot_section(page)
        if not plot_text:
            # Fallback: use the page summary (intro paragraphs)
            print(f"  No plot section for '{title}', using page summary")
            plot_text = page.summary

        plot_text = _strip_wiki_headers(plot_text)
        summary = plot_text
        _warn_if_long(title, summary)
        summaries[gid] = {
            "title": title,
            "author": author,
            "summary": summary,
        }
        print(f"  {title}: {len(summary.split())} words")

    out_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "fiction_novel_summaries.json",
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(summaries)} summaries to {out_path}")


if __name__ == "__main__":
    main()
