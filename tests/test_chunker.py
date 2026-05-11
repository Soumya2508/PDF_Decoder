"""Sanity test for the chunker.

This test doesn't load any models, so it runs in milliseconds and is safe
to wire into CI later.
"""

from src.rag.chunker import chunk_pages
from src.rag.pdf_loader import PageText


def _make_pages(num_pages: int, words_per_page: int) -> list[PageText]:
    return [
        PageText(page_number=i + 1, text=" ".join([f"word{i}_{j}" for j in range(words_per_page)]))
        for i in range(num_pages)
    ]


def test_chunks_are_produced_with_overlap():
    pages = _make_pages(num_pages=3, words_per_page=400)
    chunks = chunk_pages(pages, chunk_size_tokens=200, overlap_tokens=20)

    assert len(chunks) > 1, "expected multiple chunks for a multi-page input"
    for c in chunks:
        assert c.text.strip(), "chunks should not be blank"
        assert 1 <= c.page_number <= 3

    # Overlap means consecutive chunks should share some characters.
    a, b = chunks[0].text, chunks[1].text
    overlap_hits = sum(1 for token in a.split()[-5:] if token in b)
    assert overlap_hits > 0, "expected overlap between consecutive chunks"


def test_overlap_must_be_smaller_than_chunk_size():
    pages = _make_pages(1, 50)
    try:
        chunk_pages(pages, chunk_size_tokens=100, overlap_tokens=100)
    except ValueError:
        return
    raise AssertionError("expected ValueError when overlap >= chunk size")
