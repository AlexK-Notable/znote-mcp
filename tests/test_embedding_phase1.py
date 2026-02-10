"""Phase 1 tests for embedding infrastructure.

Tests cover:
- FakeEmbeddingProvider correctness (determinism, normalization, dimension)
- FakeRerankerProvider correctness (keyword overlap scoring, ranking)
- EmbeddingService lifecycle (lazy load, shutdown, error propagation)
- EmbeddingService reranker idle timeout
- Config embedding fields
- Error code ranges

These tests use ONLY fake providers — no real models.
"""
import time

import numpy as np
import pytest

from tests.fakes import FakeEmbeddingProvider, FakeRerankerProvider
from znote_mcp.config import ZettelkastenConfig
from znote_mcp.exceptions import EmbeddingError, ErrorCode
from znote_mcp.services.embedding_service import EmbeddingService


# =============================================================================
# FakeEmbeddingProvider Tests
# =============================================================================


class TestFakeEmbeddingProvider:
    """Verify the test fake produces well-formed, deterministic vectors."""

    def test_dimension(self):
        provider = FakeEmbeddingProvider(dim=8)
        assert provider.dimension == 8

    def test_custom_dimension(self):
        provider = FakeEmbeddingProvider(dim=16)
        assert provider.dimension == 16

    def test_not_loaded_initially(self):
        provider = FakeEmbeddingProvider()
        assert not provider.is_loaded

    def test_load_unload_lifecycle(self):
        provider = FakeEmbeddingProvider()
        provider.load()
        assert provider.is_loaded
        assert provider.load_count == 1
        provider.unload()
        assert not provider.is_loaded
        assert provider.unload_count == 1

    def test_deterministic_same_input(self):
        """Same text always produces the same vector."""
        provider = FakeEmbeddingProvider()
        v1 = provider.embed("hello world")
        v2 = provider.embed("hello world")
        np.testing.assert_array_equal(v1, v2)

    def test_different_inputs_different_vectors(self):
        """Different texts produce different vectors."""
        provider = FakeEmbeddingProvider()
        v1 = provider.embed("hello")
        v2 = provider.embed("goodbye")
        assert not np.array_equal(v1, v2)

    def test_output_shape(self):
        provider = FakeEmbeddingProvider(dim=8)
        vec = provider.embed("test")
        assert vec.shape == (8,)

    def test_l2_normalized(self):
        """Output vectors should have unit L2 norm."""
        provider = FakeEmbeddingProvider()
        vec = provider.embed("test normalization")
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-6

    def test_embed_batch(self):
        provider = FakeEmbeddingProvider(dim=8)
        texts = ["alpha", "beta", "gamma"]
        vectors = provider.embed_batch(texts)
        assert len(vectors) == 3
        for v in vectors:
            assert v.shape == (8,)
        # Each should match individual embed calls
        for text, vec in zip(texts, vectors):
            np.testing.assert_array_equal(vec, provider.embed(text))

    def test_embed_count_tracking(self):
        provider = FakeEmbeddingProvider()
        provider.embed("a")
        provider.embed("b")
        assert provider.embed_count == 2


# =============================================================================
# FakeRerankerProvider Tests
# =============================================================================


class TestFakeRerankerProvider:
    """Verify the test fake scores by keyword overlap."""

    def test_not_loaded_initially(self):
        provider = FakeRerankerProvider()
        assert not provider.is_loaded

    def test_load_unload(self):
        provider = FakeRerankerProvider()
        provider.load()
        assert provider.is_loaded
        provider.unload()
        assert not provider.is_loaded

    def test_full_overlap_scores_one(self):
        """Document containing all query words should score 1.0."""
        provider = FakeRerankerProvider()
        results = provider.rerank("hello world", ["hello world is great"])
        assert len(results) == 1
        assert results[0][0] == 0  # Index of the document
        assert results[0][1] == 1.0  # Perfect overlap

    def test_no_overlap_scores_zero(self):
        """Document with no query words should score 0.0."""
        provider = FakeRerankerProvider()
        results = provider.rerank("hello world", ["foo bar baz"])
        assert results[0][1] == 0.0

    def test_partial_overlap(self):
        """Document with some query words should score proportionally."""
        provider = FakeRerankerProvider()
        results = provider.rerank("hello world", ["hello there"])
        # 1 of 2 query words present
        assert results[0][1] == 0.5

    def test_ranking_order(self):
        """Documents should be ranked by descending score."""
        provider = FakeRerankerProvider()
        docs = [
            "nothing related",          # 0.0
            "hello there",              # 0.5 (1 of 2)
            "hello world is here",      # 1.0 (2 of 2)
        ]
        results = provider.rerank("hello world", docs, top_k=3)
        assert results[0][0] == 2  # Best match first
        assert results[1][0] == 1  # Partial match second
        assert results[2][0] == 0  # No match last

    def test_top_k_limits_results(self):
        provider = FakeRerankerProvider()
        docs = ["a", "b", "c", "d", "e"]
        results = provider.rerank("x", docs, top_k=2)
        assert len(results) == 2

    def test_empty_query(self):
        provider = FakeRerankerProvider()
        results = provider.rerank("", ["some doc"], top_k=1)
        assert len(results) == 1
        assert results[0][1] == 0.0


# =============================================================================
# EmbeddingService Lifecycle Tests
# =============================================================================


class TestEmbeddingServiceLifecycle:
    """Test lazy loading, shutdown, and error handling."""

    def test_embedder_not_loaded_on_init(self):
        embedder = FakeEmbeddingProvider()
        service = EmbeddingService(embedder=embedder)
        assert not service.embedder_loaded

    def test_embedder_loaded_on_first_embed(self):
        embedder = FakeEmbeddingProvider()
        service = EmbeddingService(embedder=embedder)
        service.embed("trigger load")
        assert service.embedder_loaded
        assert embedder.load_count == 1
        service.shutdown()

    def test_embedder_stays_loaded(self):
        """Embedder should remain loaded between calls (kept warm)."""
        embedder = FakeEmbeddingProvider()
        service = EmbeddingService(embedder=embedder)
        service.embed("first")
        service.embed("second")
        assert embedder.load_count == 1  # Only loaded once
        service.shutdown()

    def test_reranker_loaded_on_first_rerank(self):
        embedder = FakeEmbeddingProvider()
        reranker = FakeRerankerProvider()
        service = EmbeddingService(embedder=embedder, reranker=reranker)
        service.rerank("query", ["doc"])
        assert service.reranker_loaded
        assert reranker.load_count == 1
        service.shutdown()

    def test_rerank_without_reranker_raises(self):
        embedder = FakeEmbeddingProvider()
        service = EmbeddingService(embedder=embedder, reranker=None)
        with pytest.raises(EmbeddingError) as exc_info:
            service.rerank("query", ["doc"])
        assert exc_info.value.code == ErrorCode.RERANKER_FAILED
        service.shutdown()

    def test_has_reranker_flag(self):
        embedder = FakeEmbeddingProvider()
        svc_no = EmbeddingService(embedder=embedder, reranker=None)
        assert not svc_no.has_reranker

        svc_yes = EmbeddingService(
            embedder=embedder, reranker=FakeRerankerProvider()
        )
        assert svc_yes.has_reranker
        svc_no.shutdown()
        svc_yes.shutdown()

    def test_dimension_delegates_to_embedder(self):
        embedder = FakeEmbeddingProvider(dim=8)
        service = EmbeddingService(embedder=embedder)
        assert service.dimension == 8
        service.shutdown()

    def test_shutdown_unloads_all(self):
        embedder = FakeEmbeddingProvider()
        reranker = FakeRerankerProvider()
        service = EmbeddingService(embedder=embedder, reranker=reranker)
        service.embed("load embedder")
        service.rerank("query", ["doc"])
        assert service.embedder_loaded
        assert service.reranker_loaded

        service.shutdown()
        assert not service.embedder_loaded
        assert not service.reranker_loaded

    def test_embed_batch_works(self):
        embedder = FakeEmbeddingProvider(dim=8)
        service = EmbeddingService(embedder=embedder)
        vecs = service.embed_batch(["a", "b", "c"])
        assert len(vecs) == 3
        for v in vecs:
            assert v.shape == (8,)
        service.shutdown()


# =============================================================================
# EmbeddingService Error Propagation Tests
# =============================================================================


class _BrokenEmbedder:
    """Embedder that fails on load for error propagation testing."""

    @property
    def dimension(self) -> int:
        return 8

    def load(self) -> None:
        raise RuntimeError("GPU out of memory")

    def unload(self) -> None:
        pass

    @property
    def is_loaded(self) -> bool:
        return False

    def embed(self, text: str) -> np.ndarray:
        raise RuntimeError("Not loaded")

    def embed_batch(self, texts, batch_size=32):
        raise RuntimeError("Not loaded")


class _LoadableButBrokenEmbedder:
    """Embedder that loads but fails on inference."""

    def __init__(self) -> None:
        self._loaded = False

    @property
    def dimension(self) -> int:
        return 8

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def embed(self, text: str) -> np.ndarray:
        raise ValueError("Tokenizer overflow")

    def embed_batch(self, texts, batch_size=32):
        raise ValueError("Tokenizer overflow")


class TestEmbeddingServiceErrors:
    """Test that errors are properly wrapped in EmbeddingError."""

    def test_load_failure_wraps_in_embedding_error(self):
        service = EmbeddingService(embedder=_BrokenEmbedder())
        with pytest.raises(EmbeddingError) as exc_info:
            service.embed("trigger")
        assert exc_info.value.code == ErrorCode.EMBEDDING_MODEL_LOAD_FAILED
        assert "GPU out of memory" in str(exc_info.value)

    def test_inference_failure_wraps_in_embedding_error(self):
        service = EmbeddingService(embedder=_LoadableButBrokenEmbedder())
        with pytest.raises(EmbeddingError) as exc_info:
            service.embed("trigger")
        assert exc_info.value.code == ErrorCode.EMBEDDING_INFERENCE_FAILED
        assert "Tokenizer overflow" in str(exc_info.value)

    def test_batch_inference_failure(self):
        service = EmbeddingService(embedder=_LoadableButBrokenEmbedder())
        with pytest.raises(EmbeddingError) as exc_info:
            service.embed_batch(["a", "b"])
        assert exc_info.value.code == ErrorCode.EMBEDDING_INFERENCE_FAILED


# =============================================================================
# Reranker Idle Timeout Tests
# =============================================================================


class TestRerankerIdleTimeout:
    """Test that the reranker unloads after idle timeout."""

    def test_reranker_unloads_after_timeout(self):
        """Reranker should be unloaded after the idle timeout expires."""
        embedder = FakeEmbeddingProvider()
        reranker = FakeRerankerProvider()
        # Use a very short timeout for testing (0.3 seconds)
        service = EmbeddingService(
            embedder=embedder,
            reranker=reranker,
            reranker_idle_timeout=1,  # 1 second
        )
        service.rerank("query", ["doc"])
        assert service.reranker_loaded

        # Wait for timeout + buffer
        time.sleep(1.5)
        assert not service.reranker_loaded
        assert reranker.unload_count == 1
        service.shutdown()

    def test_reranker_timeout_resets_on_use(self):
        """Using the reranker should reset the idle timer."""
        embedder = FakeEmbeddingProvider()
        reranker = FakeRerankerProvider()
        service = EmbeddingService(
            embedder=embedder,
            reranker=reranker,
            reranker_idle_timeout=1,
        )
        service.rerank("query", ["doc"])

        # Use it again before timeout
        time.sleep(0.5)
        service.rerank("query2", ["doc2"])

        # Wait — the timer should have been reset
        time.sleep(0.7)
        # Should still be loaded because 0.7s < 1s timeout from last use
        assert service.reranker_loaded

        # Now wait the full timeout
        time.sleep(0.5)
        assert not service.reranker_loaded
        service.shutdown()

    def test_reranker_reloads_after_idle_unload(self):
        """After idle unload, reranker should reload on next use."""
        embedder = FakeEmbeddingProvider()
        reranker = FakeRerankerProvider()
        service = EmbeddingService(
            embedder=embedder,
            reranker=reranker,
            reranker_idle_timeout=1,
        )
        service.rerank("q1", ["d1"])
        time.sleep(1.5)
        assert not service.reranker_loaded

        # Should reload
        service.rerank("q2", ["d2"])
        assert service.reranker_loaded
        assert reranker.load_count == 2
        service.shutdown()

    def test_no_timeout_when_disabled(self):
        """With timeout=0, reranker should stay loaded."""
        embedder = FakeEmbeddingProvider()
        reranker = FakeRerankerProvider()
        service = EmbeddingService(
            embedder=embedder,
            reranker=reranker,
            reranker_idle_timeout=0,
        )
        service.rerank("query", ["doc"])
        time.sleep(0.5)
        assert service.reranker_loaded
        service.shutdown()


# =============================================================================
# Config Tests
# =============================================================================


class TestEmbeddingConfig:
    """Test embedding-related config fields."""

    def test_defaults(self):
        cfg = ZettelkastenConfig()
        assert cfg.embeddings_enabled is True
        assert cfg.embedding_model == "Alibaba-NLP/gte-modernbert-base"
        assert cfg.reranker_model == "Alibaba-NLP/gte-reranker-modernbert-base"
        assert cfg.embedding_dim == 768
        assert cfg.embedding_max_tokens == 8192
        assert cfg.reranker_idle_timeout == 600
        assert cfg.embedding_batch_size == 32
        assert cfg.embedding_model_cache_dir is None

    def test_custom_values(self):
        cfg = ZettelkastenConfig(
            embeddings_enabled=True,
            embedding_dim=384,
            embedding_batch_size=16,
            reranker_idle_timeout=300,
        )
        assert cfg.embeddings_enabled is True
        assert cfg.embedding_dim == 384
        assert cfg.embedding_batch_size == 16
        assert cfg.reranker_idle_timeout == 300


# =============================================================================
# Error Code Tests
# =============================================================================


class TestEmbeddingErrorCodes:
    """Verify embedding error codes are in the 8xxx range."""

    def test_error_codes_exist(self):
        assert ErrorCode.EMBEDDING_UNAVAILABLE.value == 8001
        assert ErrorCode.EMBEDDING_MODEL_LOAD_FAILED.value == 8002
        assert ErrorCode.EMBEDDING_INFERENCE_FAILED.value == 8003
        assert ErrorCode.VECTOR_STORE_ERROR.value == 8004
        assert ErrorCode.RERANKER_FAILED.value == 8005

    def test_embedding_error_inherits_from_zettelkasten_error(self):
        from znote_mcp.exceptions import ZettelkastenError

        err = EmbeddingError("test error")
        assert isinstance(err, ZettelkastenError)

    def test_embedding_error_to_dict(self):
        err = EmbeddingError(
            "Model not found",
            code=ErrorCode.EMBEDDING_MODEL_LOAD_FAILED,
            operation="load",
        )
        d = err.to_dict()
        assert d["code"] == 8002
        assert d["code_name"] == "EMBEDDING_MODEL_LOAD_FAILED"
        assert "Model not found" in d["message"]
        assert d["details"]["operation"] == "load"
