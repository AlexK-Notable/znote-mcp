"""Phase 2 tests for vector storage (sqlite-vec integration).

Tests cover:
- sqlite-vec table initialization (vec0 + metadata)
- Embedding CRUD: store, get, update, delete
- Similarity search (KNN via vec0 MATCH)
- Graceful degradation when sqlite-vec is unavailable
- Bulk operations (count, clear)
- Edge cases (empty DB, missing notes, duplicate stores)

These tests use the REAL sqlite-vec extension against in-memory databases.
No mocking of the database layer — we test actual SQL execution.
"""

import hashlib

import numpy as np
import pytest
from sqlalchemy import text

from znote_mcp.models.db_models import init_sqlite_vec
from znote_mcp.storage.note_repository import NoteRepository

# =============================================================================
# Helpers
# =============================================================================


def _make_embedding(seed: str, dim: int = 768) -> np.ndarray:
    """Create a deterministic L2-normalised embedding from a seed string."""
    rng = np.random.RandomState(
        int(hashlib.sha256(seed.encode()).hexdigest()[:8], 16) % (2**31)
    )
    vec = rng.randn(dim).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return vec


def _create_test_note(repo: NoteRepository, note_id: str, title: str = "Test") -> str:
    """Insert a minimal note into the repository and return its ID."""
    from znote_mcp.models.schema import Note, NoteType

    note = Note(
        id=note_id,
        title=title,
        content=f"Content for {title}",
        note_type=NoteType.PERMANENT,
    )
    repo.create(note)
    return note_id


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def vec_repo(tmp_path):
    """Create a NoteRepository with sqlite-vec enabled (in-memory DB)."""
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    repo = NoteRepository(notes_dir=notes_dir, in_memory_db=True)
    # Verify vec is available in dev environment
    assert (
        repo._vec_available
    ), "sqlite-vec not available — install with: pip install sqlite-vec"
    yield repo


@pytest.fixture
def no_vec_repo(tmp_path):
    """Create a NoteRepository where _vec_available is forced False."""
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    repo = NoteRepository(notes_dir=notes_dir, in_memory_db=True)
    repo._vec_available = False  # Simulate missing extension
    yield repo


# =============================================================================
# Table Initialization Tests
# =============================================================================


class TestSqliteVecInit:
    """Verify that init_sqlite_vec creates the expected schema."""

    def test_vec0_table_exists(self, vec_repo):
        """The note_embeddings vec0 virtual table should exist."""
        with vec_repo.engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT name FROM sqlite_master "
                    "WHERE type='table' AND name='note_embeddings'"
                )
            ).fetchone()
        assert row is not None

    def test_metadata_table_exists(self, vec_repo):
        """The embedding_metadata regular table should exist."""
        with vec_repo.engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT name FROM sqlite_master "
                    "WHERE type='table' AND name='embedding_metadata'"
                )
            ).fetchone()
        assert row is not None

    def test_metadata_index_exists(self, vec_repo):
        """An index on content_hash should exist in embedding_metadata."""
        with vec_repo.engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT name FROM sqlite_master "
                    "WHERE type='index' AND name='idx_embedding_metadata_hash'"
                )
            ).fetchone()
        assert row is not None

    def test_vec_available_flag_set(self, vec_repo):
        assert vec_repo._vec_available is True

    def test_init_sqlite_vec_returns_true(self, vec_repo):
        """Calling init_sqlite_vec on an engine with the extension should succeed."""
        result = init_sqlite_vec(vec_repo.engine)
        assert result is True

    def test_init_idempotent(self, vec_repo):
        """Calling init_sqlite_vec twice should not fail (IF NOT EXISTS)."""
        assert init_sqlite_vec(vec_repo.engine) is True
        assert init_sqlite_vec(vec_repo.engine) is True


# =============================================================================
# Store & Retrieve Tests
# =============================================================================


class TestStoreAndRetrieve:
    """Test storing, retrieving, and updating embeddings."""

    def test_store_returns_true(self, vec_repo):
        _create_test_note(vec_repo, "note-1", "Test Note")
        emb = _make_embedding("note-1")
        assert vec_repo.store_embedding("note-1", emb, "test-model", "abc123") is True

    def test_retrieve_matches_stored(self, vec_repo):
        _create_test_note(vec_repo, "note-1")
        emb = _make_embedding("note-1")
        vec_repo.store_embedding("note-1", emb, "test-model", "abc123")

        retrieved = vec_repo.get_embedding("note-1")
        assert retrieved is not None
        np.testing.assert_array_almost_equal(retrieved, emb, decimal=6)

    def test_retrieve_nonexistent_returns_none(self, vec_repo):
        assert vec_repo.get_embedding("nonexistent") is None

    def test_retrieve_shape_and_dtype(self, vec_repo):
        _create_test_note(vec_repo, "note-1")
        emb = _make_embedding("note-1")
        vec_repo.store_embedding("note-1", emb, "test-model", "abc123")

        retrieved = vec_repo.get_embedding("note-1")
        assert retrieved.shape == (768,)
        assert retrieved.dtype == np.float32

    def test_overwrite_embedding(self, vec_repo):
        """Storing a new embedding for the same note should replace the old one."""
        _create_test_note(vec_repo, "note-1")
        emb1 = _make_embedding("version-1")
        emb2 = _make_embedding("version-2")

        vec_repo.store_embedding("note-1", emb1, "model-a", "hash1")
        vec_repo.store_embedding("note-1", emb2, "model-b", "hash2")

        retrieved = vec_repo.get_embedding("note-1")
        np.testing.assert_array_almost_equal(retrieved, emb2, decimal=6)

        # Metadata should also be updated
        meta = vec_repo.get_embedding_metadata("note-1")
        assert meta["model_name"] == "model-b"
        assert meta["content_hash"] == "hash2"

    def test_store_multiple_notes(self, vec_repo):
        for i in range(5):
            nid = f"note-{i}"
            _create_test_note(vec_repo, nid)
            emb = _make_embedding(nid)
            vec_repo.store_embedding(nid, emb, "model", f"hash-{i}")

        assert vec_repo.count_embeddings() == 5


# =============================================================================
# Metadata Tests
# =============================================================================


class TestEmbeddingMetadata:
    """Test embedding metadata storage and retrieval."""

    def test_metadata_stored_correctly(self, vec_repo):
        _create_test_note(vec_repo, "note-1")
        emb = _make_embedding("note-1")
        vec_repo.store_embedding("note-1", emb, "gte-modernbert-base", "deadbeef")

        meta = vec_repo.get_embedding_metadata("note-1")
        assert meta is not None
        assert meta["model_name"] == "gte-modernbert-base"
        assert meta["content_hash"] == "deadbeef"
        assert meta["dimension"] == 768
        assert meta["created_at"] is not None

    def test_metadata_nonexistent_returns_none(self, vec_repo):
        assert vec_repo.get_embedding_metadata("nonexistent") is None

    def test_metadata_updated_on_overwrite(self, vec_repo):
        _create_test_note(vec_repo, "note-1")
        emb = _make_embedding("note-1")
        vec_repo.store_embedding("note-1", emb, "model-v1", "hash-v1")
        vec_repo.store_embedding("note-1", emb, "model-v2", "hash-v2")

        meta = vec_repo.get_embedding_metadata("note-1")
        assert meta["model_name"] == "model-v2"
        assert meta["content_hash"] == "hash-v2"


# =============================================================================
# Delete Tests
# =============================================================================


class TestDeleteEmbedding:
    """Test embedding deletion."""

    def test_delete_existing(self, vec_repo):
        _create_test_note(vec_repo, "note-1")
        emb = _make_embedding("note-1")
        vec_repo.store_embedding("note-1", emb, "model", "hash")

        assert vec_repo.delete_embedding("note-1") is True
        assert vec_repo.get_embedding("note-1") is None
        assert vec_repo.get_embedding_metadata("note-1") is None

    def test_delete_nonexistent(self, vec_repo):
        assert vec_repo.delete_embedding("nonexistent") is False

    def test_delete_does_not_affect_other_notes(self, vec_repo):
        _create_test_note(vec_repo, "note-1")
        _create_test_note(vec_repo, "note-2")
        vec_repo.store_embedding("note-1", _make_embedding("note-1"), "m", "h1")
        vec_repo.store_embedding("note-2", _make_embedding("note-2"), "m", "h2")

        vec_repo.delete_embedding("note-1")

        assert vec_repo.get_embedding("note-1") is None
        assert vec_repo.get_embedding("note-2") is not None
        assert vec_repo.count_embeddings() == 1


# =============================================================================
# Similarity Search Tests
# =============================================================================


class TestSimilaritySearch:
    """Test KNN similarity search via sqlite-vec."""

    def _seed_notes(self, repo, count=5):
        """Insert N notes with embeddings. Returns list of note IDs."""
        ids = []
        for i in range(count):
            nid = f"note-{i}"
            _create_test_note(repo, nid, f"Note {i}")
            emb = _make_embedding(nid)
            repo.store_embedding(nid, emb, "model", f"hash-{i}")
            ids.append(nid)
        return ids

    def test_basic_search_returns_results(self, vec_repo):
        self._seed_notes(vec_repo, count=5)
        query = _make_embedding("note-0")
        results = vec_repo.vec_similarity_search(query, limit=3)
        assert len(results) > 0
        assert len(results) <= 3

    def test_closest_match_is_self(self, vec_repo):
        """Searching with a stored embedding should return itself first."""
        self._seed_notes(vec_repo, count=5)
        query = _make_embedding("note-2")  # Same seed = same vector
        results = vec_repo.vec_similarity_search(query, limit=1)
        assert results[0][0] == "note-2"
        assert results[0][1] < 0.01  # Near-zero distance

    def test_results_ordered_by_distance(self, vec_repo):
        self._seed_notes(vec_repo, count=10)
        query = _make_embedding("note-0")
        results = vec_repo.vec_similarity_search(query, limit=10)
        distances = [r[1] for r in results]
        assert distances == sorted(distances)

    def test_limit_respected(self, vec_repo):
        self._seed_notes(vec_repo, count=10)
        query = _make_embedding("note-0")
        results = vec_repo.vec_similarity_search(query, limit=3)
        assert len(results) == 3

    def test_exclude_ids(self, vec_repo):
        self._seed_notes(vec_repo, count=5)
        query = _make_embedding("note-0")
        results = vec_repo.vec_similarity_search(query, limit=5, exclude_ids=["note-0"])
        result_ids = [r[0] for r in results]
        assert "note-0" not in result_ids

    def test_exclude_multiple_ids(self, vec_repo):
        self._seed_notes(vec_repo, count=5)
        query = _make_embedding("note-0")
        results = vec_repo.vec_similarity_search(
            query, limit=5, exclude_ids=["note-0", "note-1"]
        )
        result_ids = [r[0] for r in results]
        assert "note-0" not in result_ids
        assert "note-1" not in result_ids

    def test_empty_database(self, vec_repo):
        query = _make_embedding("anything")
        results = vec_repo.vec_similarity_search(query, limit=5)
        assert results == []

    def test_returns_tuples_of_id_and_float(self, vec_repo):
        self._seed_notes(vec_repo, count=3)
        query = _make_embedding("note-0")
        results = vec_repo.vec_similarity_search(query, limit=3)
        for nid, dist in results:
            assert isinstance(nid, str)
            assert isinstance(dist, float)


# =============================================================================
# Count & Clear Tests
# =============================================================================


class TestCountAndClear:
    """Test count_embeddings and clear_all_embeddings."""

    def test_count_empty(self, vec_repo):
        assert vec_repo.count_embeddings() == 0

    def test_count_after_inserts(self, vec_repo):
        for i in range(3):
            nid = f"note-{i}"
            _create_test_note(vec_repo, nid)
            vec_repo.store_embedding(nid, _make_embedding(nid), "m", f"h{i}")
        assert vec_repo.count_embeddings() == 3

    def test_clear_returns_count(self, vec_repo):
        for i in range(4):
            nid = f"note-{i}"
            _create_test_note(vec_repo, nid)
            vec_repo.store_embedding(nid, _make_embedding(nid), "m", f"h{i}")

        cleared = vec_repo.clear_all_embeddings()
        assert cleared == 4
        assert vec_repo.count_embeddings() == 0

    def test_clear_empty_returns_zero(self, vec_repo):
        assert vec_repo.clear_all_embeddings() == 0

    def test_clear_removes_metadata(self, vec_repo):
        _create_test_note(vec_repo, "note-1")
        vec_repo.store_embedding("note-1", _make_embedding("n1"), "m", "h")
        vec_repo.clear_all_embeddings()
        assert vec_repo.get_embedding_metadata("note-1") is None


# =============================================================================
# Graceful Degradation Tests
# =============================================================================


class TestGracefulDegradation:
    """Test that all vec operations degrade gracefully when unavailable."""

    def test_store_returns_false(self, no_vec_repo):
        emb = _make_embedding("x")
        assert no_vec_repo.store_embedding("note-1", emb, "m", "h") is False

    def test_get_returns_none(self, no_vec_repo):
        assert no_vec_repo.get_embedding("note-1") is None

    def test_get_metadata_returns_none(self, no_vec_repo):
        assert no_vec_repo.get_embedding_metadata("note-1") is None

    def test_delete_returns_false(self, no_vec_repo):
        assert no_vec_repo.delete_embedding("note-1") is False

    def test_search_returns_empty(self, no_vec_repo):
        emb = _make_embedding("x")
        assert no_vec_repo.vec_similarity_search(emb) == []

    def test_count_returns_zero(self, no_vec_repo):
        assert no_vec_repo.count_embeddings() == 0

    def test_clear_returns_zero(self, no_vec_repo):
        assert no_vec_repo.clear_all_embeddings() == 0
