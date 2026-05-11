"""Embeddings: text -> fixed-size vector that captures meaning.

Model: `sentence-transformers/all-MiniLM-L6-v2`
  - 384-dim vectors, ~80 MB, runs comfortably on CPU.
  - Strong on semantic similarity benchmarks for its size.
  - Free and fully local; no API call per chunk.

We L2-normalize vectors so that L2 distance in FAISS is monotonic with
cosine similarity (i.e. smaller distance = more similar).
"""

from __future__ import annotations

import os

import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = os.environ.get(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)


class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode a batch of strings into a (N, dim) float32 matrix."""
        vectors = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.asarray(vectors, dtype="float32")
