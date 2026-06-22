"""Regression tests for paragraph-aware chunking (2026-06-09 review, F3).

The pre-fix chunker seeded the next chunk with the trailing overlap and then
appended the next paragraph without re-checking the size budget, producing
chunks up to ``chunk_size + overlap`` characters (~7000 on the COLM 6000/1000
settings). Both implementations — the legacy stage and the shared
``dagspaces.common.gutenberg`` copy — carried the same bug and share these
tests.
"""

import pytest

from dagspaces.common.gutenberg.chunking import chunk_text as common_chunk_text
from dagspaces.historical_norms.stages.fetch_gutenberg import (
    chunk_text as legacy_chunk_text,
)

CHUNK_SIZE = 6000
OVERLAP = 1000


@pytest.mark.parametrize(
    "chunk_text",
    [common_chunk_text, legacy_chunk_text],
    ids=["common_gutenberg", "historical_norms_stage"],
)
class TestChunkSizeInvariant:
    def test_overlap_seed_plus_paragraph_respects_budget(self, chunk_text):
        # Pre-fix repro: two near-budget paragraphs produced sizes [5998, 7000].
        text = "A" * 5998 + "\n\n" + "B" * 5998
        chunks = chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP)
        assert all(len(c) <= CHUNK_SIZE for c in chunks)
        # No content lost (overlap may duplicate characters, never drop them)
        assert sum(c.count("A") for c in chunks) >= 5998
        assert sum(c.count("B") for c in chunks) >= 5998

    def test_unbreakable_sentence_is_hard_split(self, chunk_text):
        # Single paragraph with no sentence boundaries at all.
        text = "X" * 20000
        chunks = chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP)
        assert all(len(c) <= CHUNK_SIZE for c in chunks)
        assert sum(c.count("X") for c in chunks) >= 20000

    def test_giant_paragraph_split_on_sentences(self, chunk_text):
        sentence = "The quick brown fox jumps over the lazy dog. "
        text = sentence * 400  # one ~18k-char paragraph
        chunks = chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP)
        assert len(chunks) > 1
        assert all(len(c) <= CHUNK_SIZE for c in chunks)

    def test_normal_paragraphs_chunk_with_overlap(self, chunk_text):
        paras = [f"Paragraph {i}: " + ("word " * 200) for i in range(20)]
        text = "\n\n".join(paras)
        chunks = chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP)
        assert len(chunks) > 1
        assert all(len(c) <= CHUNK_SIZE for c in chunks)
        # Every paragraph survives in some chunk
        joined = "\n".join(chunks)
        for i in range(20):
            assert f"Paragraph {i}:" in joined
        # Each chunk begins with the trailing overlap of its predecessor
        for prev, nxt in zip(chunks, chunks[1:]):
            assert nxt.startswith(prev[-OVERLAP:].lstrip())

    def test_no_duplicate_only_chunks(self, chunk_text):
        # A chunk must never consist solely of its overlap seed.
        text = "A" * 5998 + "\n\n" + "B" * 5998 + "\n\n" + "C" * 5998
        chunks = chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP)
        for prev, nxt in zip(chunks, chunks[1:]):
            assert not prev.endswith(nxt), "chunk is a pure overlap duplicate"

    def test_short_text_single_chunk(self, chunk_text):
        text = "One small paragraph."
        assert chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP) == [text]
