"""FAISS vector store: fast nearest-neighbor search over chunk embeddings.

Why FAISS:
  - Open-source, battle-tested, in-process (no server, no API key).
  - Many index types; we use `IndexFlatL2` (exact brute-force) because:
      * Annual reports yield a few hundred to a few thousand chunks.
      * Flat is exact and trivially correct; approximate indexes (IVF, HNSW)
        only pay off above ~100k vectors.
"""

from __future__ import annotations

from dataclasses import dataclass

import faiss
import numpy as np

from .chunker import Chunk


@dataclass
class SearchHit:
    chunk: Chunk
    score: float  # smaller L2 distance = more similar (because vectors are normalized)


class FaissStore:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if vectors.shape[0] != len(chunks):
            raise ValueError("vectors and chunks length mismatch")
        if vectors.shape[1] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {vectors.shape[1]}")
        self.index.add(vectors.astype("float32"))
        self.chunks.extend(chunks)

    def search(self, query_vec: np.ndarray, k: int = 4) -> list[SearchHit]:
        if query_vec.ndim == 1:
            query_vec = query_vec[None, :]
        distances, indices = self.index.search(query_vec.astype("float32"), k)
        hits: list[SearchHit] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            hits.append(SearchHit(chunk=self.chunks[idx], score=float(dist)))
        return hits

    def __len__(self) -> int:
        return len(self.chunks)
