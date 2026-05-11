"""End-to-end RAG pipeline: ingest a PDF, then answer questions about it.

The pipeline owns the embedder, the FAISS store, and the LLM client. The
Streamlit layer keeps a single cached pipeline instance per session.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import BinaryIO

from .chunker import chunk_pages
from .embedder import Embedder
from .llm import Answer, GroqClient
from .pdf_loader import extract_pages
from .vector_store import FaissStore, SearchHit


@dataclass
class IngestResult:
    num_pages: int
    num_chunks: int


@dataclass
class QueryResult:
    answer: Answer
    hits: list[SearchHit]


class RAGPipeline:
    def __init__(self, api_key: str | None = None) -> None:
        self.embedder = Embedder()
        self.store: FaissStore | None = None
        self.llm = GroqClient(api_key=api_key)

    def ingest(self, file: BinaryIO | bytes | str) -> IngestResult:
        pages = extract_pages(file)
        if not pages:
            raise ValueError(
                "No extractable text found. This PDF may be scanned images; "
                "OCR is not yet supported."
            )
        chunks = chunk_pages(pages)
        vectors = self.embedder.encode([c.text for c in chunks])
        store = FaissStore(dim=self.embedder.dim)
        store.add(chunks, vectors)
        self.store = store
        return IngestResult(num_pages=len(pages), num_chunks=len(chunks))

    def ask(self, question: str, k: int = 4) -> QueryResult:
        if self.store is None:
            raise RuntimeError("No document indexed yet. Call ingest() first.")
        q_vec = self.embedder.encode([question])
        hits = self.store.search(q_vec, k=k)
        answer = self.llm.answer(question, hits)
        return QueryResult(answer=answer, hits=hits)
