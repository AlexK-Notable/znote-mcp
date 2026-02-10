"""Phase 3 tests for the embedding write path.

Tests cover:
- Content hashing (determinism, sensitivity to title/content changes)
- Auto-embed on create (create_note, create_note_versioned, bulk_create)
- Auto-embed on update (update_note, update_note_versioned)
- Skip re-embed when content hash unchanged (metadata-only update)
- Embedding cleanup on delete (delete_note, delete_note_versioned, bulk_delete)
- reindex_embeddings() bulk operation
- Fire-and-forget: embedding failures don't break CRUD
- Disabled embeddings: no embedding calls when config.embeddings_enabled=False

Uses FakeEmbeddingProvider from tests/fakes.py with real in-memory SQLite.
"""
import pytest

from tests.fakes import FakeEmbeddingProvider
from znote_mcp.config import config
from znote_mcp.services.embedding_service import EmbeddingService
from znote_mcp.services.zettel_service import ZettelService
from znote_mcp.storage.note_repository import NoteRepository


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def _enable_embeddings():
    """Temporarily enable embeddings in global config."""
    original = config.embeddings_enabled
    config.embeddings_enabled = True
    yield
    config.embeddings_enabled = original


@pytest.fixture
def _disable_embeddings():
    """Temporarily disable embeddings in global config."""
    original = config.embeddings_enabled
    config.embeddings_enabled = False
    yield
    config.embeddings_enabled = original


@pytest.fixture
def fake_embedder():
    # Must match the vec0 table dimension (768 by default in init_sqlite_vec)
    return FakeEmbeddingProvider(dim=768)


@pytest.fixture
def embedding_service(fake_embedder):
    svc = EmbeddingService(embedder=fake_embedder)
    yield svc
    svc.shutdown()


@pytest.fixture
def repo(tmp_path):
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    return NoteRepository(notes_dir=notes_dir, in_memory_db=True)


@pytest.fixture
def service(repo, embedding_service, _enable_embeddings):
    """ZettelService with embeddings enabled and a FakeEmbeddingProvider."""
    return ZettelService(repository=repo, embedding_service=embedding_service)


@pytest.fixture
def service_disabled(repo, embedding_service, _disable_embeddings):
    """ZettelService with embeddings disabled."""
    return ZettelService(repository=repo, embedding_service=embedding_service)


@pytest.fixture
def service_no_embedder(repo, _enable_embeddings):
    """ZettelService with no embedding service at all."""
    return ZettelService(repository=repo, embedding_service=None)


# =============================================================================
# Content Hash Tests
# =============================================================================


class TestContentHash:
    """Verify content hash determinism and sensitivity."""

    def test_deterministic(self):
        h1 = ZettelService._content_hash("Title", "Body")
        h2 = ZettelService._content_hash("Title", "Body")
        assert h1 == h2

    def test_different_title(self):
        h1 = ZettelService._content_hash("Title A", "Body")
        h2 = ZettelService._content_hash("Title B", "Body")
        assert h1 != h2

    def test_different_content(self):
        h1 = ZettelService._content_hash("Title", "Body A")
        h2 = ZettelService._content_hash("Title", "Body B")
        assert h1 != h2

    def test_hex_string_64_chars(self):
        h = ZettelService._content_hash("T", "C")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


# =============================================================================
# Auto-Embed on Create Tests
# =============================================================================


class TestAutoEmbedOnCreate:
    """Test that create operations trigger embedding."""

    def test_create_note_stores_embedding(self, service, repo, fake_embedder):
        note = service.create_note(title="Alpha", content="First note")
        assert repo.get_embedding(note.id) is not None
        assert fake_embedder.embed_count == 1

    def test_create_note_stores_metadata(self, service, repo):
        note = service.create_note(title="Alpha", content="First note")
        meta = repo.get_embedding_metadata(note.id)
        assert meta is not None
        assert meta["content_hash"] == ZettelService._content_hash("Alpha", "First note")

    def test_create_note_versioned_stores_embedding(self, service, repo, fake_embedder):
        result = service.create_note_versioned(title="Beta", content="Versioned note")
        assert repo.get_embedding(result.note.id) is not None
        assert fake_embedder.embed_count == 1

    def test_bulk_create_stores_all_embeddings(self, service, repo, fake_embedder):
        notes_data = [
            {"title": f"Note {i}", "content": f"Content {i}"}
            for i in range(5)
        ]
        created = service.bulk_create_notes(notes_data)
        assert len(created) == 5
        assert fake_embedder.embed_count == 5
        for note in created:
            assert repo.get_embedding(note.id) is not None


# =============================================================================
# Auto-Embed on Update Tests
# =============================================================================


class TestAutoEmbedOnUpdate:
    """Test that update operations re-embed when content changes."""

    def test_update_content_re_embeds(self, service, repo, fake_embedder):
        note = service.create_note(title="Test", content="Original")
        old_hash = repo.get_embedding_metadata(note.id)["content_hash"]

        service.update_note(note.id, content="Updated content")
        new_hash = repo.get_embedding_metadata(note.id)["content_hash"]

        assert old_hash != new_hash
        assert fake_embedder.embed_count == 2  # create + update

    def test_update_title_re_embeds(self, service, repo, fake_embedder):
        note = service.create_note(title="Original Title", content="Body")
        service.update_note(note.id, title="New Title")
        assert fake_embedder.embed_count == 2  # create + update

    def test_update_metadata_only_skips_embed_after_stabilization(
        self, service, repo, fake_embedder
    ):
        """After content hash stabilizes, metadata-only updates skip embedding.

        The first update after create always re-embeds because the markdown
        round-trip adds a title header to content (e.g. "Body" -> "# Test\\n\\nBody").
        After that, the hash is stable and metadata-only updates skip.
        """
        note = service.create_note(title="Test", content="Body")
        assert fake_embedder.embed_count == 1

        # First metadata-only update: re-embeds due to round-trip content change
        service.update_note(note.id, tags=["tag-1"])
        assert fake_embedder.embed_count == 2  # Content hash changed after round-trip

        # Second metadata-only update: hash is now stable, should skip
        service.update_note(note.id, tags=["tag-2"])
        assert fake_embedder.embed_count == 2  # No re-embed!

    def test_update_project_only_skips_embed_after_stabilization(
        self, service, repo, fake_embedder
    ):
        note = service.create_note(title="Test", content="Body")
        # First update stabilizes the content hash
        service.update_note(note.id, tags=["stabilize"])
        count_after_stabilize = fake_embedder.embed_count

        service.update_note(note.id, project="new-project")
        assert fake_embedder.embed_count == count_after_stabilize  # No re-embed

    def test_update_versioned_re_embeds(self, service, repo, fake_embedder):
        result = service.create_note_versioned(title="V", content="One")
        service.update_note_versioned(result.note.id, content="Two")
        assert fake_embedder.embed_count == 2


# =============================================================================
# Embedding Cleanup on Delete Tests
# =============================================================================


class TestDeleteCleansUpEmbedding:
    """Test that delete operations remove embeddings."""

    def test_delete_note_removes_embedding(self, service, repo):
        note = service.create_note(title="Temp", content="To be deleted")
        assert repo.get_embedding(note.id) is not None

        service.delete_note(note.id)
        assert repo.get_embedding(note.id) is None

    def test_bulk_delete_removes_embeddings(self, service, repo):
        notes = service.bulk_create_notes([
            {"title": "A", "content": "Content A"},
            {"title": "B", "content": "Content B"},
        ])
        ids = [n.id for n in notes]

        service.bulk_delete_notes(ids)
        for nid in ids:
            assert repo.get_embedding(nid) is None


# =============================================================================
# Reindex Tests
# =============================================================================


class TestReindexEmbeddings:
    """Test the reindex_embeddings bulk operation."""

    def test_reindex_embeds_all_notes(self, service, repo, fake_embedder):
        # Create notes (auto-embeds them)
        for i in range(3):
            service.create_note(title=f"Note {i}", content=f"Content {i}")
        assert fake_embedder.embed_count == 3

        # Reindex: clears and re-embeds all
        stats = service.reindex_embeddings()
        assert stats["total"] == 3
        assert stats["embedded"] == 3
        assert stats["failed"] == 0
        # 3 original + 3 reindex = 6 total embed calls
        assert fake_embedder.embed_count == 6

    def test_reindex_clears_first(self, service, repo, fake_embedder):
        note = service.create_note(title="Test", content="Body")
        assert repo.count_embeddings() == 1

        stats = service.reindex_embeddings()
        # Should still have exactly 1 embedding (cleared + re-created)
        assert repo.count_embeddings() == 1
        assert stats["embedded"] == 1

    def test_reindex_returns_stats(self, service):
        stats = service.reindex_embeddings()
        assert "total" in stats
        assert "embedded" in stats
        assert "skipped" in stats
        assert "failed" in stats

    def test_reindex_without_embedding_service_raises(self, service_no_embedder):
        from znote_mcp.exceptions import EmbeddingError
        with pytest.raises(EmbeddingError):
            service_no_embedder.reindex_embeddings()


# =============================================================================
# Fire-and-Forget (Embedding Failure) Tests
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
        raise RuntimeError("GPU exploded")

    def embed_batch(self, texts, batch_size=32):
        raise RuntimeError("GPU exploded")


class TestFireAndForget:
    """Embedding failures must not break CRUD operations."""

    @pytest.fixture
    def broken_service(self, repo, _enable_embeddings):
        emb_svc = EmbeddingService(embedder=_BrokenEmbedder())
        return ZettelService(repository=repo, embedding_service=emb_svc)

    def test_create_succeeds_despite_embed_failure(self, broken_service):
        note = broken_service.create_note(title="OK", content="Still works")
        assert note.id is not None
        assert note.title == "OK"

    def test_update_succeeds_despite_embed_failure(self, broken_service):
        note = broken_service.create_note(title="OK", content="Original")
        updated = broken_service.update_note(note.id, content="New content")
        assert updated.content == "New content"

    def test_bulk_create_succeeds_despite_embed_failure(self, broken_service):
        notes = broken_service.bulk_create_notes([
            {"title": "A", "content": "Body A"},
            {"title": "B", "content": "Body B"},
        ])
        assert len(notes) == 2


# =============================================================================
# Disabled Embeddings Tests
# =============================================================================


class TestDisabledEmbeddings:
    """When embeddings_enabled=False, no embedding calls should happen."""

    def test_create_does_not_embed(self, service_disabled, repo, fake_embedder):
        note = service_disabled.create_note(title="X", content="Y")
        assert repo.get_embedding(note.id) is None
        assert fake_embedder.embed_count == 0

    def test_update_does_not_embed(self, service_disabled, repo, fake_embedder):
        note = service_disabled.create_note(title="X", content="Y")
        service_disabled.update_note(note.id, content="Z")
        assert fake_embedder.embed_count == 0

    def test_delete_does_not_touch_embeddings(self, service_disabled, repo, fake_embedder):
        note = service_disabled.create_note(title="X", content="Y")
        service_disabled.delete_note(note.id)
        # No error, no embed calls
        assert fake_embedder.embed_count == 0


# =============================================================================
# No Embedding Service Tests
# =============================================================================


class TestNoEmbeddingService:
    """When no embedding_service is provided, everything works normally."""

    def test_create_works(self, service_no_embedder):
        note = service_no_embedder.create_note(title="Hello", content="World")
        assert note.id is not None

    def test_update_works(self, service_no_embedder):
        note = service_no_embedder.create_note(title="Hello", content="World")
        updated = service_no_embedder.update_note(note.id, content="Updated")
        assert updated.content == "Updated"

    def test_delete_works(self, service_no_embedder):
        note = service_no_embedder.create_note(title="Hello", content="World")
        service_no_embedder.delete_note(note.id)
        assert service_no_embedder.get_note(note.id) is None
