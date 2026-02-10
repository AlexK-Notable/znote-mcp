"""Fake embedding and reranking providers for testing.

These produce deterministic, controlled outputs without loading real models.
FakeEmbeddingProvider uses 8-dimensional vectors derived from text hashing,
so identical inputs always produce identical vectors and similar inputs
produce somewhat similar vectors. FakeRerankerProvider scores by keyword
overlap between query and document.

Design principles (from test strategy):
- Never mock sqlite-vec — always use real in-memory SQLite
- Fake providers produce real numpy arrays of controlled dimensionality
- Deterministic: same input → same output, always
- Inspectable: test code can predict exact outputs
"""
import hashlib
from typing import List, Sequence, Tuple

import numpy as np


class FakeEmbeddingProvider:
    """Deterministic 8-dimensional embedding provider for testing.

    Produces L2-normalized vectors derived from a hash of the input text.
    This ensures:
    - Same text → same vector (deterministic)
    - Different texts → different vectors (discriminative)
    - Vectors are properly normalized (unit length)
    """

    def __init__(self, dim: int = 8) -> None:
        self._dim = dim
        self._loaded = False
        self.load_count = 0
        self.unload_count = 0
        self.embed_count = 0

    @property
    def dimension(self) -> int:
        return self._dim

    def load(self) -> None:
        self._loaded = True
        self.load_count += 1

    def unload(self) -> None:
        self._loaded = False
        self.unload_count += 1

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def embed(self, text: str) -> np.ndarray:
        """Produce a deterministic vector from text via hashing.

        For dimensions > 32, the hash is extended by iteratively hashing
        the previous digest, producing an unlimited stream of deterministic
        bytes.  This ensures large-dim vectors are still deterministic and
        discriminative.
        """
        self.embed_count += 1
        # Build enough bytes for the requested dimension
        chunks: list[bytes] = []
        needed = self._dim
        seed = text.encode("utf-8")
        while needed > 0:
            seed = hashlib.sha256(seed).digest()
            chunks.append(seed)
            needed -= len(seed)
        all_bytes = b"".join(chunks)[: self._dim]

        raw_bytes = np.frombuffer(all_bytes, dtype=np.uint8).astype(np.float64)
        # Map [0, 255] -> [-1, 1]
        raw = (raw_bytes / 127.5) - 1.0
        # L2-normalize
        norm = np.linalg.norm(raw)
        if norm > 0:
            raw = raw / norm
        return raw.astype(np.float32)

    def embed_batch(
        self, texts: Sequence[str], batch_size: int = 32
    ) -> List[np.ndarray]:
        """Embed multiple texts, processing all at once (no real batching needed)."""
        return [self.embed(text) for text in texts]


class FakeRerankerProvider:
    """Keyword-overlap reranker for testing.

    Scores each document by the fraction of query words that appear in
    the document text (case-insensitive). This gives predictable, intuitive
    scores: a document containing all query terms scores 1.0, one containing
    none scores 0.0.
    """

    def __init__(self) -> None:
        self._loaded = False
        self.load_count = 0
        self.unload_count = 0
        self.rerank_count = 0

    def load(self) -> None:
        self._loaded = True
        self.load_count += 1

    def unload(self) -> None:
        self._loaded = False
        self.unload_count += 1

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def rerank(
        self, query: str, documents: Sequence[str], top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """Score documents by keyword overlap with query."""
        self.rerank_count += 1
        query_words = set(query.lower().split())
        if not query_words:
            return [(i, 0.0) for i in range(min(top_k, len(documents)))]

        scored = []
        for i, doc in enumerate(documents):
            doc_lower = doc.lower()
            matches = sum(1 for w in query_words if w in doc_lower)
            score = matches / len(query_words)
            scored.append((i, score))

        # Sort by score descending, then by original index for stability
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:top_k]
