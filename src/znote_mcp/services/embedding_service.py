"""Embedding service for semantic search.

Orchestrates embedding and reranking providers with lazy loading,
thread-safe model management, and idle timeout for the reranker.

The embedder is loaded on first use and kept warm (no timeout).
The reranker is loaded lazily and unloaded after an idle period
to conserve memory when not actively searching.

Usage:
    service = EmbeddingService(embedder=embedder, reranker=reranker)
    vector = service.embed("some text")
    vectors = service.embed_batch(["text1", "text2"])
    ranked = service.rerank("query", ["doc1", "doc2"])
    service.shutdown()  # Clean up on server exit
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple

from znote_mcp.exceptions import EmbeddingError, ErrorCode

if TYPE_CHECKING:
    import numpy as np

    from znote_mcp.services.embedding_types import EmbeddingProvider, RerankerProvider

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Manages embedding and reranking with lazy loading and idle timeout.

    Thread-safe: all model load/unload operations are guarded by locks.
    The embedder is loaded eagerly on first embed() call and kept warm.
    The reranker is loaded lazily on first rerank() call and unloaded
    after `reranker_idle_timeout` seconds of inactivity.

    Args:
        embedder: An EmbeddingProvider implementation.
        reranker: An optional RerankerProvider implementation.
        reranker_idle_timeout: Seconds before idle reranker is unloaded.
            Set to 0 to disable idle unloading.
    """

    def __init__(
        self,
        embedder: EmbeddingProvider,
        reranker: Optional[RerankerProvider] = None,
        reranker_idle_timeout: int = 600,
    ) -> None:
        self._embedder = embedder
        self._reranker = reranker
        self._reranker_idle_timeout = reranker_idle_timeout

        # Thread safety
        self._embedder_lock = threading.Lock()
        self._reranker_lock = threading.Lock()
        self._idle_timer: Optional[threading.Timer] = None
        self._shutdown = False

    @property
    def dimension(self) -> int:
        """Embedding dimensionality (delegates to provider)."""
        return self._embedder.dimension

    @property
    def embedder_loaded(self) -> bool:
        """Whether the embedding model is currently in memory."""
        return self._embedder.is_loaded

    @property
    def reranker_loaded(self) -> bool:
        """Whether the reranker model is currently in memory."""
        return self._reranker is not None and self._reranker.is_loaded

    @property
    def has_reranker(self) -> bool:
        """Whether a reranker provider is configured."""
        return self._reranker is not None

    def _ensure_embedder(self) -> None:
        """Load the embedder if not already loaded. Thread-safe."""
        if self._embedder.is_loaded:
            return
        with self._embedder_lock:
            if self._embedder.is_loaded:
                return  # Double-check after acquiring lock
            try:
                self._embedder.load()
                logger.info("Embedding model loaded")
            except Exception as e:
                raise EmbeddingError(
                    f"Failed to load embedding model: {e}",
                    code=ErrorCode.EMBEDDING_MODEL_LOAD_FAILED,
                    operation="embedder_load",
                    original_error=e,
                )

    def _ensure_reranker(self) -> None:
        """Load the reranker if not already loaded. Thread-safe."""
        if self._reranker is None:
            raise EmbeddingError(
                "No reranker provider configured",
                code=ErrorCode.RERANKER_FAILED,
                operation="reranker_load",
            )
        if self._reranker.is_loaded:
            self._reset_idle_timer()
            return
        with self._reranker_lock:
            if self._reranker.is_loaded:
                self._reset_idle_timer()
                return
            try:
                self._reranker.load()
                logger.info("Reranker model loaded")
                self._reset_idle_timer()
            except Exception as e:
                raise EmbeddingError(
                    f"Failed to load reranker model: {e}",
                    code=ErrorCode.EMBEDDING_MODEL_LOAD_FAILED,
                    operation="reranker_load",
                    original_error=e,
                )

    def _reset_idle_timer(self) -> None:
        """Reset (or start) the reranker idle unload timer."""
        if self._reranker_idle_timeout <= 0:
            return

        # Cancel any existing timer
        if self._idle_timer is not None:
            self._idle_timer.cancel()

        self._idle_timer = threading.Timer(
            self._reranker_idle_timeout, self._idle_unload_reranker
        )
        self._idle_timer.daemon = True  # Don't block process exit
        self._idle_timer.start()

    def _idle_unload_reranker(self) -> None:
        """Called by timer to unload idle reranker."""
        if self._shutdown:
            return
        with self._reranker_lock:
            if self._reranker is not None and self._reranker.is_loaded:
                self._reranker.unload()
                logger.info(
                    f"Reranker unloaded after {self._reranker_idle_timeout}s idle"
                )

    def embed(self, text: str) -> "np.ndarray":
        """Embed a single text into a dense vector.

        Loads the embedding model on first call (lazy initialization).

        Args:
            text: Input text to embed.

        Returns:
            1-D numpy array of shape (dimension,), L2-normalized.

        Raises:
            EmbeddingError: If model loading or inference fails.
        """
        self._ensure_embedder()
        try:
            return self._embedder.embed(text)
        except EmbeddingError:
            raise
        except Exception as e:
            raise EmbeddingError(
                f"Embedding inference failed: {e}",
                code=ErrorCode.EMBEDDING_INFERENCE_FAILED,
                operation="embed",
                original_error=e,
            )

    def embed_batch(
        self, texts: Sequence[str], batch_size: int = 32
    ) -> List["np.ndarray"]:
        """Embed multiple texts in batches.

        Args:
            texts: Sequence of input texts.
            batch_size: Number of texts per inference batch.

        Returns:
            List of 1-D numpy arrays, each of shape (dimension,).

        Raises:
            EmbeddingError: If model loading or inference fails.
        """
        self._ensure_embedder()
        try:
            return self._embedder.embed_batch(texts, batch_size)
        except EmbeddingError:
            raise
        except Exception as e:
            raise EmbeddingError(
                f"Batch embedding failed: {e}",
                code=ErrorCode.EMBEDDING_INFERENCE_FAILED,
                operation="embed_batch",
                original_error=e,
            )

    def embed_batch_adaptive(
        self, texts: Sequence[str], memory_budget_gb: float = 4.0
    ) -> List["np.ndarray"]:
        """Embed texts with dynamic batch sizes based on text length.

        Groups texts by token count and uses larger batches for shorter
        texts.  Gives full-coverage embeddings (no truncation) while
        staying within the memory budget.

        Args:
            texts: Sequence of input texts.
            memory_budget_gb: Max attention memory in GB (default 4.0).

        Returns:
            List of 1-D numpy arrays, each of shape (dimension,).

        Raises:
            EmbeddingError: If model loading or inference fails.
        """
        self._ensure_embedder()
        if not hasattr(self._embedder, "embed_batch_adaptive"):
            # Fallback to fixed batching if provider doesn't support adaptive
            return self.embed_batch(texts, batch_size=8)
        try:
            return self._embedder.embed_batch_adaptive(texts, memory_budget_gb)
        except EmbeddingError:
            raise
        except Exception as e:
            raise EmbeddingError(
                f"Adaptive batch embedding failed: {e}",
                code=ErrorCode.EMBEDDING_INFERENCE_FAILED,
                operation="embed_batch_adaptive",
                original_error=e,
            )

    def rerank(
        self, query: str, documents: Sequence[str], top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """Rerank documents by relevance to query.

        Loads the reranker model on first call (lazy initialization).
        Resets the idle timeout on each call.

        Args:
            query: The search query.
            documents: Sequence of document texts to rerank.
            top_k: Number of top results to return.

        Returns:
            List of (original_index, score) tuples, sorted by score descending.

        Raises:
            EmbeddingError: If no reranker configured, or loading/inference fails.
        """
        self._ensure_reranker()
        try:
            return self._reranker.rerank(query, documents, top_k)
        except EmbeddingError:
            raise
        except Exception as e:
            raise EmbeddingError(
                f"Reranking failed: {e}",
                code=ErrorCode.RERANKER_FAILED,
                operation="rerank",
                original_error=e,
            )

    def shutdown(self) -> None:
        """Clean up: cancel timers, unload all models.

        Call this when the server is shutting down.
        """
        self._shutdown = True

        # Cancel idle timer
        if self._idle_timer is not None:
            self._idle_timer.cancel()
            self._idle_timer = None

        # Unload models
        with self._embedder_lock:
            if self._embedder.is_loaded:
                self._embedder.unload()

        with self._reranker_lock:
            if self._reranker is not None and self._reranker.is_loaded:
                self._reranker.unload()

        logger.info("EmbeddingService shut down")
