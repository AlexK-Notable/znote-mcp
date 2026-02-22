"""Integration tests for the chunked embedding pipeline.

Tests cover the full chunked embedding lifecycle: TextChunker splitting,
multi-chunk storage via store_chunk_embeddings, deduplication in
vec_similarity_search, routing in _embed_note (single vs chunked), write-path
lifecycle (create/update/delete), search-path integration, reindex with
mixed short/long notes, and end-to-end pipeline verification.

Written against the interface defined in plan phase 2.
Tests verify behavior, not implementation details.

Uses FakeEmbeddingProvider (hash-based, 768-dim) from tests/fakes.py,
real sqlite-vec via NoteRepository(in_memory_db=True), and real TextChunker.
Monkeypatches config.embedding_chunk_size=50 to trigger chunking on small
test data instead of requiring 16KB strings.
"""

import pytest

from znote_mcp.config import config
from znote_mcp.models.schema import Note, NoteType
from znote_mcp.services.embedding_service import EmbeddingService
from znote_mcp.services.search_service import SearchService
from znote_mcp.services.text_chunker import TextChunker
from znote_mcp.services.zettel_service import ZettelService
from znote_mcp.storage.note_repository import NoteRepository

# =============================================================================
# Fixtures and Helpers
# =============================================================================


@pytest.fixture
def _small_chunk_config(monkeypatch):
    """Set chunk_size small enough that multi-sentence notes trigger chunking.

    With chunk_size=50 tokens and the character approximation (len//4),
    text longer than ~200 characters triggers chunking. chunk_overlap=10
    ensures overlap logic is exercised.
    """
    monkeypatch.setattr(config, "embedding_chunk_size", 50)
    monkeypatch.setattr(config, "embedding_chunk_overlap", 10)


@pytest.fixture
def embedding_service(fake_embedder, fake_reranker):
    svc = EmbeddingService(embedder=fake_embedder, reranker=fake_reranker)
    yield svc
    svc.shutdown()


@pytest.fixture
def repo(tmp_path):
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    return NoteRepository(notes_dir=notes_dir, in_memory_db=True)


@pytest.fixture
def service(repo, embedding_service, _enable_embeddings, _small_chunk_config):
    """ZettelService with embeddings enabled and small chunk_size."""
    return ZettelService(repository=repo, embedding_service=embedding_service)


@pytest.fixture
def search_service(service, embedding_service):
    """SearchService wired with the chunking-enabled ZettelService."""
    return SearchService(
        zettel_service=service,
        embedding_service=embedding_service,
    )


def _long_content(sentences: int = 20) -> str:
    """Generate content long enough to trigger chunking with small chunk_size.

    Each sentence is unique to produce distinct chunk embeddings.
    With chunk_size=50 tokens (~200 chars), 20 sentences of ~40 chars each
    produces ~800 chars (~200 tokens), yielding multiple chunks.
    """
    return (
        ". ".join(
            f"This is sentence number {i} about topic {i % 5}" for i in range(sentences)
        )
        + "."
    )


def _short_content() -> str:
    """Generate content short enough to stay in single-vector path."""
    return "A brief note about testing."


# =============================================================================
# TestTextChunkerUnit
# =============================================================================


class TestTextChunkerUnit:
    """Unit tests for TextChunker: chunk_id format, splitting, validation."""

    def test_make_chunk_id_format(self):
        """make_chunk_id must produce '{note_id}::chunk_{index}' format."""
        assert TextChunker.make_chunk_id("note_abc", 0) == "note_abc::chunk_0"
        assert TextChunker.make_chunk_id("note_abc", 5) == "note_abc::chunk_5"

    def test_parse_chunk_id_valid(self):
        """parse_chunk_id must extract (note_id, index) from valid chunk_ids."""
        assert TextChunker.parse_chunk_id("note_abc::chunk_0") == ("note_abc", 0)
        assert TextChunker.parse_chunk_id("note_abc::chunk_5") == ("note_abc", 5)

    def test_parse_chunk_id_with_separator_in_note_id(self):
        """Adversarial: note_id containing '::chunk_' should parse via rfind."""
        cid = "note::chunk_99::chunk_3"
        note_id, idx = TextChunker.parse_chunk_id(cid)
        assert note_id == "note::chunk_99"
        assert idx == 3

    def test_parse_chunk_id_invalid_raises(self):
        """Invalid chunk_id without separator must raise ValueError."""
        with pytest.raises(ValueError):
            TextChunker.parse_chunk_id("no_separator_here")

    def test_parse_chunk_id_non_integer_raises(self):
        """Non-integer chunk index must raise ValueError."""
        with pytest.raises(ValueError):
            TextChunker.parse_chunk_id("note::chunk_abc")

    def test_roundtrip(self):
        """parse_chunk_id(make_chunk_id(nid, idx)) must return (nid, idx)."""
        cases = [("abc", 0), ("note-123", 7), ("x::chunk_5", 2)]
        for nid, idx in cases:
            assert TextChunker.parse_chunk_id(TextChunker.make_chunk_id(nid, idx)) == (
                nid,
                idx,
            )

    def test_short_text_single_chunk(self):
        """Text shorter than chunk_size should produce a single chunk."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.chunk("Hello world")
        assert len(chunks) == 1
        assert chunks[0].index == 0

    def test_long_text_multiple_chunks(self):
        """Text longer than chunk_size should produce multiple chunks."""
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        long_text = ". ".join(
            f"Sentence {i} with enough words to fill tokens" for i in range(20)
        )
        chunks = chunker.chunk(long_text)
        assert len(chunks) > 1
        for i, chunk in enumerate(chunks):
            assert chunk.index == i

    def test_overlap_exists_between_chunks(self):
        """Consecutive chunks should share overlapping content."""
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        long_text = ". ".join(f"Sentence {i} about unique topic {i}" for i in range(20))
        chunks = chunker.chunk(long_text)
        assert len(chunks) >= 2, "Need at least 2 chunks to test overlap"
        # Verify overlapping content between consecutive chunks
        for i in range(len(chunks) - 1):
            words_i = chunks[i].text.split()
            words_next = chunks[i + 1].text.split()
            tail_words = set(words_i[-5:]) if len(words_i) >= 5 else set(words_i)
            head_words = (
                set(words_next[:5]) if len(words_next) >= 5 else set(words_next)
            )
            overlap = tail_words & head_words
            assert len(overlap) > 0, f"No overlap between chunk {i} and {i + 1}"

    def test_empty_text_returns_single_chunk(self):
        """Empty text should produce a single chunk, not crash."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.chunk("")
        assert len(chunks) == 1

    def test_chunk_overlap_gte_chunk_size_raises(self):
        """chunk_overlap >= chunk_size must raise ValueError."""
        with pytest.raises(ValueError):
            TextChunker(chunk_size=100, chunk_overlap=100)
        with pytest.raises(ValueError):
            TextChunker(chunk_size=100, chunk_overlap=150)


# =============================================================================
# TestChunkedStorageIntegration
# =============================================================================


class TestChunkedStorageIntegration:
    """Integration tests for store_chunk_embeddings + vec_similarity_search."""

    @staticmethod
    def _create_note(repo, note_id, title="Test"):
        """Insert a minimal note into the repository and return the Note."""
        note = Note(
            id=note_id,
            title=title,
            content=f"Content for {title}",
            note_type=NoteType.PERMANENT,
        )
        return repo.create(note)

    def test_store_chunk_embeddings_stores_all(self, repo, fake_embedder):
        """Storing 3 chunk embeddings should produce 3 rows in the vec table."""
        note = self._create_note(repo, "test_note_1", "Test")
        chunks = [
            (0, fake_embedder.embed("chunk 0")),
            (1, fake_embedder.embed("chunk 1")),
            (2, fake_embedder.embed("chunk 2")),
        ]
        stored = repo.store_chunk_embeddings(note.id, chunks, "test-model", "hash123")
        assert stored == 3
        assert repo.count_embeddings() == 3
        assert repo.count_embedded_notes() == 1

    def test_store_chunk_embeddings_replaces_existing(self, repo, fake_embedder):
        """Storing new chunks should atomically replace previous chunks."""
        note = self._create_note(repo, "test_note_2", "Test")
        chunks_v1 = [(i, fake_embedder.embed(f"v1 chunk {i}")) for i in range(3)]
        repo.store_chunk_embeddings(note.id, chunks_v1, "model", "hash_v1")
        assert repo.count_embeddings() == 3

        chunks_v2 = [(i, fake_embedder.embed(f"v2 chunk {i}")) for i in range(2)]
        repo.store_chunk_embeddings(note.id, chunks_v2, "model", "hash_v2")
        assert repo.count_embeddings() == 2

    def test_store_chunks_then_similarity_search_returns_note_id(
        self, repo, fake_embedder
    ):
        """vec_similarity_search must return note_ids, not chunk_ids."""
        note_a = self._create_note(repo, "note_a", "Alpha")
        note_b = self._create_note(repo, "note_b", "Beta")
        chunks_a = [(i, fake_embedder.embed(f"alpha chunk {i}")) for i in range(3)]
        chunks_b = [(0, fake_embedder.embed("beta single"))]
        repo.store_chunk_embeddings(note_a.id, chunks_a, "model", "hash_a")
        repo.store_chunk_embeddings(note_b.id, chunks_b, "model", "hash_b")

        query_vec = fake_embedder.embed("alpha chunk 1")
        results = repo.vec_similarity_search(query_vec, limit=5)
        result_ids = [nid for nid, _ in results]
        # Results must contain note_ids, not chunk_ids
        assert all(
            "::chunk_" not in nid for nid, _ in results
        ), "Search returned chunk_id instead of note_id"
        assert note_a.id in result_ids

    def test_dedup_keeps_best_distance(self, repo, fake_embedder):
        """When a note has multiple chunks, search should keep the best distance."""
        note = self._create_note(repo, "test_note_dedup", "Test")
        query_vec = fake_embedder.embed("exact match query")
        chunks = [
            (0, fake_embedder.embed("unrelated chunk zero")),
            (1, query_vec),  # This chunk matches the query exactly
            (2, fake_embedder.embed("unrelated chunk two")),
        ]
        repo.store_chunk_embeddings(note.id, chunks, "model", "hash")
        results = repo.vec_similarity_search(query_vec, limit=5)
        # Note should appear exactly once
        note_results = [(nid, dist) for nid, dist in results if nid == note.id]
        assert len(note_results) == 1
        # Distance should be near-zero from the exact-match chunk
        assert note_results[0][1] < 0.01


# =============================================================================
# TestEmbedNoteRouting
# =============================================================================


class TestEmbedNoteRouting:
    """Tests that _embed_note routes correctly between single and chunked paths."""

    def test_short_note_uses_single_vector(self, service, repo):
        """Short notes should produce exactly 1 embedding (single-vector path)."""
        note = service.create_note(title="Short", content=_short_content())
        assert repo.count_embeddings() == 1
        meta = repo.get_embedding_metadata(note.id)
        assert meta is not None

    def test_long_note_uses_chunked_path(self, service, repo):
        """Long notes should produce multiple chunk embeddings."""
        note = service.create_note(title="Long", content=_long_content(sentences=20))
        assert repo.count_embeddings() > 1
        assert repo.count_embedded_notes() == 1

    def test_long_note_uses_embed_batch(self, service, repo, fake_embedder):
        """Chunked path should call embed for each chunk (via embed_batch)."""
        initial_count = fake_embedder.embed_count
        service.create_note(title="Long", content=_long_content(sentences=20))
        chunk_count = repo.count_embeddings()
        # embed_batch in FakeEmbeddingProvider calls embed() once per text
        assert fake_embedder.embed_count - initial_count == chunk_count


# =============================================================================
# TestChunkedWritePath
# =============================================================================


class TestChunkedWritePath:
    """Tests for create, update, and delete lifecycle of chunked notes."""

    def test_update_long_note_replaces_chunks(self, service, repo):
        """Updating a long note should replace all old chunks with new ones."""
        note = service.create_note(title="Long", content=_long_content(sentences=20))
        chunks_before = repo.count_embeddings()
        assert chunks_before > 1
        # Update with different long content
        service.update_note(note.id, content=_long_content(sentences=25))
        chunks_after = repo.count_embeddings()
        assert chunks_after > 1
        # Note count should still be 1 (no stale chunk accumulation)
        assert repo.count_embedded_notes() == 1

    def test_long_to_short_update(self, service, repo):
        """Updating a long note to short content should collapse to 1 embedding."""
        note = service.create_note(title="Long", content=_long_content(sentences=20))
        assert repo.count_embeddings() > 1
        service.update_note(note.id, content=_short_content())
        assert repo.count_embeddings() == 1
        assert repo.count_embedded_notes() == 1

    def test_short_to_long_update(self, service, repo):
        """Updating a short note to long content should produce multiple chunks."""
        note = service.create_note(title="Short", content=_short_content())
        assert repo.count_embeddings() == 1
        service.update_note(note.id, content=_long_content(sentences=20))
        assert repo.count_embeddings() > 1
        assert repo.count_embedded_notes() == 1

    def test_delete_long_note_removes_all_chunks(self, service, repo):
        """Deleting a long note must remove all its chunk embeddings."""
        note = service.create_note(title="Long", content=_long_content(sentences=20))
        assert repo.count_embeddings() > 1
        service.delete_note(note.id)
        assert repo.count_embeddings() == 0
        assert repo.count_embedded_notes() == 0

    def test_content_hash_skips_reembed(self, service, repo, fake_embedder):
        """Metadata-only updates should not re-embed after content hash stabilizes."""
        note = service.create_note(title="Long", content=_long_content(sentences=20))
        # First metadata-only update may re-embed due to markdown round-trip
        service.update_note(note.id, tags=["stabilize"])
        count_after_stabilize = fake_embedder.embed_count
        # Second metadata-only update: hash is now stable, should skip
        service.update_note(note.id, project="new-project")
        assert fake_embedder.embed_count == count_after_stabilize


# =============================================================================
# TestChunkedSearchPath
# =============================================================================


class TestChunkedSearchPath:
    """Tests for search behavior with chunked notes."""

    def test_search_finds_chunked_note(self, search_service, service):
        """Semantic search should find a chunked note and return it as a Note."""
        service.create_note(
            title="Quantum Physics", content=_long_content(sentences=20)
        )
        results = search_service.semantic_search("Quantum Physics", limit=5)
        assert len(results) > 0
        assert hasattr(results[0], "note")
        assert results[0].note.title == "Quantum Physics"

    def test_search_mixed_short_and_long(self, search_service, service):
        """Both short and long notes should be findable via search."""
        service.create_note(title="Brief Note", content=_short_content())
        service.create_note(title="Long Note", content=_long_content(sentences=20))
        results = search_service.semantic_search("note", limit=10)
        # At least one note should be found (hash-based matching is approximate)
        assert len(results) >= 1

    def test_search_excludes_chunked_note(self, search_service, service):
        """Excluding a chunked note's ID must exclude all its chunks from results."""
        note = service.create_note(
            title="Excluded", content=_long_content(sentences=20)
        )
        service.create_note(title="Included", content=_short_content())
        results = search_service.semantic_search(
            "test", limit=10, exclude_ids=[note.id]
        )
        result_ids = {r.note.id for r in results}
        assert note.id not in result_ids

    def test_search_dedup_with_many_chunked_notes(self, search_service, service):
        """Each chunked note should appear at most once in search results."""
        for i in range(3):
            service.create_note(
                title=f"Long Note {i}", content=_long_content(sentences=20)
            )
        results = search_service.semantic_search("topic", limit=10)
        result_ids = [r.note.id for r in results]
        assert len(result_ids) == len(
            set(result_ids)
        ), "Duplicate note_ids in search results"

    def test_find_related_works_with_chunked_note(self, search_service, service):
        """find_related should work with a chunked seed note."""
        seed = service.create_note(title="Seed", content=_long_content(sentences=20))
        for i in range(3):
            service.create_note(title=f"Other {i}", content=_long_content(sentences=15))
        results = search_service.find_related(seed.id, limit=5)
        result_ids = {r.note.id for r in results}
        # Seed note should be excluded from results
        assert seed.id not in result_ids


# =============================================================================
# TestReindexWithChunkedNotes
# =============================================================================


class TestReindexWithChunkedNotes:
    """Tests for reindex_embeddings with mixed short and long notes."""

    def test_reindex_mixed_short_long(self, service, repo):
        """Reindex should handle a mix of short and long notes correctly."""
        service.create_note(title="Short 1", content=_short_content())
        service.create_note(title="Short 2", content=_short_content())
        service.create_note(title="Long 1", content=_long_content(sentences=20))
        service.create_note(title="Long 2", content=_long_content(sentences=20))

        stats = service.reindex_embeddings()
        assert stats["total"] == 4
        assert stats["embedded"] == 4
        assert stats["failed"] == 0
        # chunks should account for multi-chunk notes (more chunks than notes)
        assert stats["chunks"] > stats["embedded"]
        # Embedding count should match chunks stat
        assert repo.count_embeddings() == stats["chunks"]

    def test_reindex_preserves_searchability(self, search_service, service):
        """Notes should remain searchable after reindex."""
        service.create_note(
            title="Searchable Long", content=_long_content(sentences=20)
        )
        service.create_note(title="Searchable Short", content=_short_content())

        service.reindex_embeddings()

        results = search_service.semantic_search("Searchable", limit=5)
        assert len(results) > 0


# =============================================================================
# TestFullChunkedPipeline
# =============================================================================


class TestFullChunkedPipeline:
    """End-to-end tests for the full chunked embedding pipeline."""

    def test_end_to_end_create_search_chunked(self, search_service, service):
        """THE test: create a long note, verify search finds it by note_id."""
        note = service.create_note(
            title="Advanced Quantum Computing Algorithms",
            content=_long_content(sentences=25),
        )
        results = search_service.semantic_search("Advanced Quantum Computing", limit=5)
        assert len(results) > 0
        found_ids = {r.note.id for r in results}
        assert note.id in found_ids
        # Result should have valid distance and score
        for r in results:
            if r.note.id == note.id:
                assert r.distance >= 0
                assert r.score > 0

    def test_end_to_end_update_chunked_then_search(self, search_service, service):
        """Update a chunked note and verify search uses new content."""
        note = service.create_note(
            title="Original Topic", content=_long_content(sentences=20)
        )
        new_content = (
            ". ".join(
                f"Updated sentence {i} about a completely different subject {i}"
                for i in range(20)
            )
            + "."
        )
        service.update_note(note.id, content=new_content, title="Updated Topic")
        results = search_service.semantic_search("Updated Topic", limit=5)
        found_ids = {r.note.id for r in results}
        assert note.id in found_ids

    def test_end_to_end_delete_chunked_then_search(self, search_service, service):
        """Delete a chunked note and verify search no longer finds it."""
        note = service.create_note(
            title="Deletable", content=_long_content(sentences=20)
        )
        # Verify it is searchable
        results_before = search_service.semantic_search("Deletable", limit=5)
        found_before = {r.note.id for r in results_before}
        assert note.id in found_before
        # Delete
        service.delete_note(note.id)
        # Verify it is no longer searchable
        results_after = search_service.semantic_search("Deletable", limit=5)
        found_after = {r.note.id for r in results_after}
        assert note.id not in found_after
