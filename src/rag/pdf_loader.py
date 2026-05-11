"""PDF text extraction using PyMuPDF (fitz).

Why PyMuPDF: fast, pure-Python wheel on all platforms, good at text-order
extraction. We keep the page number with each block so the UI can show
citations like 'p. 14'.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import BinaryIO


@dataclass
class PageText:
    page_number: int  # 1-indexed for human-friendly citations
    text: str


def extract_pages(file: BinaryIO | bytes | str) -> list[PageText]:
    """Extract text page-by-page from a PDF.

    Accepts a filesystem path, raw bytes, or a file-like object (e.g. the
    Streamlit UploadedFile). Empty pages are skipped.
    """
    # Lazy import: keeps the PageText dataclass importable in environments
    # (e.g. unit tests) where PyMuPDF isn't installed.
    import fitz  # PyMuPDF

    if isinstance(file, (bytes, bytearray)):
        doc = fitz.open(stream=bytes(file), filetype="pdf")
    elif isinstance(file, str):
        doc = fitz.open(file)
    else:
        data = file.read()
        doc = fitz.open(stream=data, filetype="pdf")

    pages: list[PageText] = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if text:
            pages.append(PageText(page_number=i, text=text))
    doc.close()
    return pages
