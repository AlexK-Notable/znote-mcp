"""Type protocols for embedding and reranking providers.

Defines the structural contracts that both production ONNX providers
and test fakes must satisfy. Uses Protocol (PEP 544) for structural
subtyping — implementations don't need to inherit from these.

This module is importable without numpy installed (annotations are
deferred via __future__). Actual providers require numpy at runtime.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Protocol, Sequence, Tuple, runtime_checkable

if TYPE_CHECKING:
    import numpy as np


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Contract for embedding text into dense vectors."""

    @property
    def dimension(self) -> int:
        """Dimensionality of produced vectors."""
        ...

    def load(self) -> None:
        """Load model into memory. May be called multiple times (idempotent)."""
        ...

    def unload(self) -> None:
        """Release model from memory. May be called multiple times (idempotent)."""
        ...

    @property
    def is_loaded(self) -> bool:
        """Whether the model is currently loaded in memory."""
        ...

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text into a dense vector.

        Args:
            text: Input text to embed.

        Returns:
            1-D numpy array of shape (dimension,), L2-normalized.
        """
        ...

    def embed_batch(
        self, texts: Sequence[str], batch_size: int = 32
    ) -> List[np.ndarray]:
        """Embed multiple texts in batches.

        Args:
            texts: Sequence of input texts.
            batch_size: Number of texts per inference batch.

        Returns:
            List of 1-D numpy arrays, each of shape (dimension,).
        """
        ...


@runtime_checkable
class RerankerProvider(Protocol):
    """Contract for reranking query-document pairs."""

    def load(self) -> None:
        """Load model into memory. May be called multiple times (idempotent)."""
        ...

    def unload(self) -> None:
        """Release model from memory. May be called multiple times (idempotent)."""
        ...

    @property
    def is_loaded(self) -> bool:
        """Whether the model is currently loaded in memory."""
        ...

    def rerank(
        self, query: str, documents: Sequence[str], top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """Rerank documents against a query.

        Args:
            query: The search query.
            documents: Sequence of document texts to rerank.
            top_k: Number of top results to return.

        Returns:
            List of (original_index, score) tuples, sorted by score descending.
        """
        ...
