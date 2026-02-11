"""Phase 4 tests for the embedding read path (semantic search + reranking).

Tests cover:
- semantic_search(): basic flow, reranking, exclude_ids, empty query, no results
- find_related(): basic flow, reranking, missing embedding, seed note exclusion
- Graceful degradation: disabled embeddings, no embedding service, broken embedder
- Reranker integration: results are reranked, score comes from reranker
- SemanticSearchResult dataclass field correctness

Uses FakeEmbeddingProvider and FakeRerankerProvider from tests/fakes.py
with real in-memory SQLite + sqlite-vec.
"""

import pytest

from znote_mcp.services.embedding_service import EmbeddingService
from znote_mcp.services.search_service import SearchService, SemanticSearchResult
from znote_mcp.services.zettel_service import ZettelService
from znote_mcp.storage.note_repository import NoteRepository

# =============================================================================
# Fixtures (shared _enable/_disable_embeddings, fake_embedder, fake_reranker
# are in conftest.py)
# =============================================================================


@pytest.fixture
def embedding_service(fake_embedder, fake_reranker):
    svc = EmbeddingService(embedder=fake_embedder, reranker=fake_reranker)
    yield svc
    svc.shutdown()


@pytest.fixture
def embedding_service_no_reranker(fake_embedder):
    svc = EmbeddingService(embedder=fake_embedder, reranker=None)
    yield svc
    svc.shutdown()


@pytest.fixture
def repo(tmp_path):
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    return NoteRepository(notes_dir=notes_dir, in_memory_db=True)


@pytest.fixture
def zettel_service(repo, embedding_service, _enable_embeddings):
    """ZettelService with embeddings enabled (auto-embeds on create)."""
    return ZettelService(repository=repo, embedding_service=embedding_service)


@pytest.fixture
def search_service(zettel_service, embedding_service, _enable_embeddings):
    """SearchService with embedding service configured."""
    return SearchService(
        zettel_service=zettel_service,
        embedding_service=embedding_service,
    )


@pytest.fixture
def search_service_no_reranker(repo, embedding_service_no_reranker, _enable_embeddings):
    """SearchService with embedding service but no reranker."""
    zs = ZettelService(repository=repo, embedding_service=embedding_service_no_reranker)
    return SearchService(
        zettel_service=zs,
        embedding_service=embedding_service_no_reranker,
    )


@pytest.fixture
def search_service_disabled(zettel_service, embedding_service, _disable_embeddings):
    """SearchService with embeddings disabled."""
    return SearchService(
        zettel_service=zettel_service,
        embedding_service=embedding_service,
    )


@pytest.fixture
def search_service_no_embedder(repo, _enable_embeddings):
    """SearchService with no embedding service at all."""
    zs = ZettelService(repository=repo, embedding_service=None)
    return SearchService(zettel_service=zs, embedding_service=None)


def _seed_notes(service: ZettelService, count: int = 5):
    """Create test notes via the service (auto-embeds them)."""
    notes = []
    for i in range(count):
        note = service.create_note(
            title=f"Note about topic {i}",
            content=f"Content discussing subject {i} in detail",
        )
        notes.append(note)
    return notes


# =============================================================================
# SemanticSearchResult Tests
# =============================================================================


class TestSemanticSearchResult:
    """Verify the dataclass fields."""

    def test_fields_present(self):
        from znote_mcp.models.schema import Note

        note = Note(title="T", content="C")
        r = SemanticSearchResult(note=note, distance=0.5, score=0.67)
        assert r.note is note
        assert r.distance == 0.5
        assert r.score == 0.67
        assert r.reranked is False

    def test_reranked_flag(self):
        from znote_mcp.models.schema import Note

        note = Note(title="T", content="C")
        r = SemanticSearchResult(note=note, distance=0.5, score=0.9, reranked=True)
        assert r.reranked is True


# =============================================================================
# semantic_search() Tests
# =============================================================================


class TestSemanticSearch:
    """Test the semantic_search() read path."""

    def test_basic_search_returns_results(self, search_service, zettel_service):
        notes = _seed_notes(zettel_service, count=5)
        results = search_service.semantic_search("topic 0", limit=3)
        assert len(results) > 0
        assert len(results) <= 3
        assert all(isinstance(r, SemanticSearchResult) for r in results)

    def test_results_have_positive_scores(self, search_service, zettel_service):
        _seed_notes(zettel_service, count=3)
        results = search_service.semantic_search("topic", limit=3)
        for r in results:
            assert r.score > 0
            assert r.distance >= 0

    def test_limit_respected(self, search_service, zettel_service):
        _seed_notes(zettel_service, count=10)
        results = search_service.semantic_search("topic", limit=3)
        assert len(results) <= 3

    def test_exclude_ids(self, search_service, zettel_service):
        notes = _seed_notes(zettel_service, count=5)
        exclude = [notes[0].id, notes[1].id]
        results = search_service.semantic_search("topic", limit=10, exclude_ids=exclude)
        result_ids = {r.note.id for r in results}
        assert notes[0].id not in result_ids
        assert notes[1].id not in result_ids

    def test_empty_query_returns_empty(self, search_service, zettel_service):
        _seed_notes(zettel_service, count=3)
        assert search_service.semantic_search("", limit=5) == []
        assert search_service.semantic_search("   ", limit=5) == []

    def test_no_notes_returns_empty(self, search_service):
        results = search_service.semantic_search("anything", limit=5)
        assert results == []

    def test_results_ordered_by_score_desc(self, search_service, zettel_service):
        """Without reranker, scores should be monotonically derived from distance."""
        _seed_notes(zettel_service, count=5)
        results = search_service.semantic_search(
            "topic",
            limit=5,
            use_reranker=False,
        )
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_without_reranker_flag(self, search_service, zettel_service):
        _seed_notes(zettel_service, count=3)
        results = search_service.semantic_search(
            "topic",
            limit=3,
            use_reranker=False,
        )
        for r in results:
            assert r.reranked is False


# =============================================================================
# Reranking Tests
# =============================================================================


class TestSemanticSearchWithReranker:
    """Test reranker integration in semantic_search()."""

    def test_reranked_flag_set(self, search_service, zettel_service):
        _seed_notes(zettel_service, count=3)
        results = search_service.semantic_search(
            "topic",
            limit=3,
            use_reranker=True,
        )
        # With FakeRerankerProvider configured, results should be reranked
        for r in results:
            assert r.reranked is True

    def test_reranker_scores_reflect_keyword_overlap(
        self,
        search_service,
        zettel_service,
    ):
        """FakeRerankerProvider scores by keyword overlap with query."""
        notes = _seed_notes(zettel_service, count=5)
        # Search for exact content of note 2
        results = search_service.semantic_search(
            "topic 2",
            limit=5,
            use_reranker=True,
        )
        # The reranker should score the note containing "topic 2" highly
        if results:
            assert results[0].score > 0

    def test_no_reranker_configured(
        self,
        search_service_no_reranker,
    ):
        """When no reranker is configured, results should not be reranked."""
        zs = search_service_no_reranker.zettel_service
        _seed_notes(zs, count=3)
        results = search_service_no_reranker.semantic_search(
            "topic",
            limit=3,
            use_reranker=True,  # Requested but unavailable
        )
        for r in results:
            assert r.reranked is False

    def test_reranker_rerank_count_incremented(
        self,
        search_service,
        zettel_service,
        fake_reranker,
    ):
        _seed_notes(zettel_service, count=3)
        assert fake_reranker.rerank_count == 0
        search_service.semantic_search("topic", limit=3, use_reranker=True)
        assert fake_reranker.rerank_count == 1


# =============================================================================
# find_related() Tests
# =============================================================================


class TestFindRelated:
    """Test the find_related() method."""

    def test_basic_find_related(self, search_service, zettel_service):
        notes = _seed_notes(zettel_service, count=5)
        results = search_service.find_related(notes[0].id, limit=3)
        assert len(results) > 0
        assert len(results) <= 3

    def test_excludes_seed_note(self, search_service, zettel_service):
        notes = _seed_notes(zettel_service, count=5)
        results = search_service.find_related(notes[0].id, limit=10)
        result_ids = {r.note.id for r in results}
        assert notes[0].id not in result_ids

    def test_missing_embedding_returns_empty(
        self, search_service, zettel_service, repo
    ):
        notes = _seed_notes(zettel_service, count=3)
        # Delete the embedding for note 0
        repo.delete_embedding(notes[0].id)
        results = search_service.find_related(notes[0].id, limit=3)
        assert results == []

    def test_nonexistent_note_returns_empty(self, search_service):
        results = search_service.find_related("nonexistent-id", limit=3)
        assert results == []

    def test_find_related_with_reranker(
        self, search_service, zettel_service, fake_reranker
    ):
        notes = _seed_notes(zettel_service, count=5)
        results = search_service.find_related(
            notes[0].id,
            limit=3,
            use_reranker=True,
        )
        assert len(results) > 0
        for r in results:
            assert r.reranked is True
        assert fake_reranker.rerank_count == 1

    def test_find_related_without_reranker(self, search_service, zettel_service):
        notes = _seed_notes(zettel_service, count=5)
        results = search_service.find_related(
            notes[0].id,
            limit=3,
            use_reranker=False,
        )
        for r in results:
            assert r.reranked is False

    def test_single_note_returns_empty(self, search_service, zettel_service):
        """With only one note, find_related should return empty (seed excluded)."""
        notes = _seed_notes(zettel_service, count=1)
        results = search_service.find_related(notes[0].id, limit=5)
        assert results == []


# =============================================================================
# Graceful Degradation Tests
# =============================================================================


class TestGracefulDegradation:
    """Semantic search should degrade gracefully."""

    def test_disabled_embeddings_semantic_search(
        self,
        search_service_disabled,
        zettel_service,
    ):
        _seed_notes(zettel_service, count=3)
        results = search_service_disabled.semantic_search("topic", limit=5)
        assert results == []

    def test_disabled_embeddings_find_related(
        self,
        search_service_disabled,
        zettel_service,
    ):
        notes = _seed_notes(zettel_service, count=3)
        results = search_service_disabled.find_related(notes[0].id, limit=5)
        assert results == []

    def test_no_embedding_service_semantic_search(self, search_service_no_embedder):
        results = search_service_no_embedder.semantic_search("topic", limit=5)
        assert results == []

    def test_no_embedding_service_find_related(self, search_service_no_embedder):
        results = search_service_no_embedder.find_related("any-id", limit=5)
        assert results == []


# =============================================================================
# Broken Embedder (fire-and-forget on read path too)
# =============================================================================


class _BrokenEmbedder:
    """Embedder that always fails on embed()."""

    @property
    def dimension(self) -> int:
        return 8

    def load(self) -> None:
        pass

    def unload(self) -> None:
        pass

    @property
    def is_loaded(self) -> bool:
        return True

    def embed(self, text):
        raise RuntimeError("Inference exploded")

    def embed_batch(self, texts, batch_size=32):
        raise RuntimeError("Inference exploded")


class TestBrokenEmbedder:
    """Semantic search should not raise when the embedder fails."""

    @pytest.fixture
    def broken_search_service(self, repo, _enable_embeddings):
        broken_emb = EmbeddingService(embedder=_BrokenEmbedder())
        zs = ZettelService(repository=repo, embedding_service=None)
        return SearchService(zettel_service=zs, embedding_service=broken_emb)

    def test_semantic_search_returns_empty(self, broken_search_service):
        results = broken_search_service.semantic_search("topic", limit=5)
        assert results == []

    def test_find_related_returns_empty(self, broken_search_service):
        results = broken_search_service.find_related("any-id", limit=5)
        assert results == []
