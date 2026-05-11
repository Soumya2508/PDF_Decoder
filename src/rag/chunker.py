"""Chunking: split long page text into overlapping, retrieval-friendly pieces.

Why chunks at all:
  - LLM context windows are bounded; we can't paste the whole report.
  - Smaller chunks make embedding similarity more discriminative — a query
    matches the *specific* paragraph that answers it, not a whole page of
    unrelated text.

Why overlap:
  - A sentence at a chunk boundary can split a key fact in two. Overlap
    ensures every neighborhood of tokens appears intact in at least one chunk.

Sizing: we approximate tokens as `len(text) / 4` (a common rule of thumb for
English) so beginners don't need to install a tokenizer just to chunk.
"""

from __future__ import annotations

from dataclasses import dataclass

from .pdf_loader import PageText

# 1 token ~ 4 characters for English text.
CHARS_PER_TOKEN = 4


@dataclass
class Chunk:
    chunk_id: int
    page_number: int  # page where the chunk starts
    text: str


def chunk_pages(
    pages: list[PageText],
    chunk_size_tokens: int = 500,
    overlap_tokens: int = 50,
) -> list[Chunk]:
    """Concatenate pages and split into overlapping windows.

    Returns chunks tagged with the starting page number so the UI can cite
    where each retrieved passage came from.
    """
    if chunk_size_tokens <= overlap_tokens:
        raise ValueError("chunk_size must be larger than overlap")

    size = chunk_size_tokens * CHARS_PER_TOKEN
    step = (chunk_size_tokens - overlap_tokens) * CHARS_PER_TOKEN

    # Build a single string but remember which character index starts each page.
    full_text_parts: list[str] = []
    page_starts: list[tuple[int, int]] = []  # (char_index, page_number)
    cursor = 0
    for p in pages:
        page_starts.append((cursor, p.page_number))
        full_text_parts.append(p.text)
        cursor += len(p.text) + 1  # +1 for the newline separator we add below
    full_text = "\n".join(full_text_parts)

    def page_for(char_idx: int) -> int:
        page = page_starts[0][1] if page_starts else 1
        for start, pno in page_starts:
            if start <= char_idx:
                page = pno
            else:
                break
        return page

    chunks: list[Chunk] = []
    i = 0
    chunk_id = 0
    while i < len(full_text):
        piece = full_text[i : i + size].strip()
        if piece:
            chunks.append(
                Chunk(chunk_id=chunk_id, page_number=page_for(i), text=piece)
            )
            chunk_id += 1
        i += step
    return chunks
