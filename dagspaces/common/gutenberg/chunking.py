"""Boilerplate stripping + paragraph-aware chunking.

Lifted from ``dagspaces/historical_norms/stages/fetch_gutenberg.py`` so the
Gutenberg cache and the legacy fetch stage can share one implementation.
"""

from __future__ import annotations

import re

START_MARKERS = (
    r"\*\*\* START OF THIS PROJECT GUTENBERG EBOOK .* \*\*\*",
    r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK .* \*\*\*",
)
END_MARKERS = (
    r"\*\*\* END OF THIS PROJECT GUTENBERG EBOOK .* \*\*\*",
    r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK .* \*\*\*",
)


def clean_gutenberg_boilerplate(text: str) -> str:
    """Strip Gutenberg headers/footers, illustrations, and normalize whitespace."""
    start_idx = 0
    for marker in START_MARKERS:
        m = re.search(marker, text, re.IGNORECASE)
        if m:
            start_idx = m.end()
            break

    end_idx = len(text)
    for marker in END_MARKERS:
        m = re.search(marker, text, re.IGNORECASE)
        if m:
            end_idx = m.start()
            break

    text = text[start_idx:end_idx].strip()
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\[Illustration[^\]]*\]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 6000, overlap: int = 1000) -> list[str]:
    """Paragraph-aware chunking with character overlap; matches the legacy stage.

    Invariant: no emitted chunk exceeds *chunk_size* characters.  Content
    that does not fit alongside the overlap seed is packed sentence-by-
    sentence, and sentences that cannot fit on a freshly seeded chunk are
    hard-split at the character level.  A chunk is only emitted once it
    holds content beyond its overlap seed (no duplicate-only chunks).
    """
    paragraphs = re.split(r"\n\s*\n", text)
    chunks: list[str] = []
    current = ""
    seed_len = 0  # length of the overlap seed at the head of `current`

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current) + len(para) + 2 < chunk_size:
            current += para + "\n\n"
            continue

        if len(current) > seed_len:
            chunks.append(current.strip())
            prev = current.strip()
            current = prev[-overlap:] + "\n\n" if overlap and len(prev) > overlap else ""
            seed_len = len(current)

        if len(current) + len(para) + 2 < chunk_size:
            current += para + "\n\n"
            continue

        # Paragraph doesn't fit even on a freshly seeded chunk: pack it
        # sentence-by-sentence so no chunk exceeds chunk_size.
        sentences = re.split(r"(?<=[.!?])\s+", para)
        for sentence in sentences:
            if len(current) + len(sentence) + 1 < chunk_size:
                current += sentence + " "
                continue
            if len(current) > seed_len:
                chunks.append(current.strip())
                prev = current.strip()
                current = prev[-overlap:] + " " if overlap and len(prev) > overlap else ""
                seed_len = len(current)
            # Sentence doesn't fit even on a fresh seed: hard-split it.
            while len(current) + len(sentence) + 1 >= chunk_size:
                room = max(1, chunk_size - len(current) - 1)
                chunks.append((current + sentence[:room]).strip())
                sentence = sentence[room:]
                prev = chunks[-1]
                current = prev[-overlap:] + " " if overlap and len(prev) > overlap else ""
                seed_len = len(current)
            current += sentence + " "

    if current.strip() and len(current) > seed_len:
        chunks.append(current.strip())
    return chunks
