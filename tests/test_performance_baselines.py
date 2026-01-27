"""Performance baseline tests.

These tests establish performance baselines for critical operations.
They can be used to detect performance regressions in future changes.

Note: These tests use reasonable thresholds that should pass on most systems.
Actual performance will vary based on hardware and system load.
"""
import time
import pytest

from znote_mcp.models.schema import Note, NoteType, Tag
from znote_mcp.services.zettel_service import ZettelService
from znote_mcp.storage.note_repository import NoteRepository


class TestNoteOperationPerformance:
    """Performance baselines for note CRUD operations."""

    def test_single_note_create_performance(self, zettel_service):
        """Single note creation should complete in reasonable time."""
        start = time.perf_counter()

        note = zettel_service.create_note(
            title="Performance Test Note",
            content="This is a performance test note with some content.",
            tags=["perf", "test"]
        )

        elapsed = time.perf_counter() - start

        # Should complete in under 100ms
        assert elapsed < 0.1, f"Note creation took {elapsed:.3f}s (expected < 0.1s)"
        assert note.id is not None

    def test_single_note_read_performance(self, zettel_service):
        """Single note read should complete in reasonable time."""
        # Create a note first
        note = zettel_service.create_note(
            title="Read Test",
            content="Content"
        )

        start = time.perf_counter()
        retrieved = zettel_service.get_note(note.id)
        elapsed = time.perf_counter() - start

        # Should complete in under 50ms
        assert elapsed < 0.05, f"Note read took {elapsed:.3f}s (expected < 0.05s)"
        assert retrieved is not None

    def test_single_note_update_performance(self, zettel_service):
        """Single note update should complete in reasonable time."""
        note = zettel_service.create_note(
            title="Update Test",
            content="Original content"
        )

        start = time.perf_counter()
        updated = zettel_service.update_note(
            note.id,
            content="Updated content"
        )
        elapsed = time.perf_counter() - start

        # Should complete in under 100ms
        assert elapsed < 0.1, f"Note update took {elapsed:.3f}s (expected < 0.1s)"
        assert updated is not None

    def test_single_note_delete_performance(self, zettel_service):
        """Single note deletion should complete in reasonable time."""
        note = zettel_service.create_note(
            title="Delete Test",
            content="Content"
        )

        start = time.perf_counter()
        zettel_service.delete_note(note.id)
        elapsed = time.perf_counter() - start

        # Should complete in under 100ms
        assert elapsed < 0.1, f"Note delete took {elapsed:.3f}s (expected < 0.1s)"


class TestBulkOperationPerformance:
    """Performance baselines for bulk operations."""

    def test_bulk_create_10_notes(self, zettel_service):
        """Bulk creation of 10 notes should be efficient."""
        notes_data = [
            {"title": f"Bulk Note {i}", "content": f"Content {i}"}
            for i in range(10)
        ]

        start = time.perf_counter()
        created = zettel_service.bulk_create_notes(notes_data)
        elapsed = time.perf_counter() - start

        # 10 notes should complete in under 500ms
        assert elapsed < 0.5, f"Bulk create 10 took {elapsed:.3f}s (expected < 0.5s)"
        assert len(created) == 10

    def test_bulk_create_should_be_faster_than_individual(self, zettel_service):
        """Bulk create should be faster than creating notes individually."""
        # Time individual creates
        individual_notes = []
        start_individual = time.perf_counter()
        for i in range(5):
            note = zettel_service.create_note(
                title=f"Individual Note {i}",
                content=f"Content {i}"
            )
            individual_notes.append(note)
        individual_time = time.perf_counter() - start_individual

        # Time bulk create
        bulk_data = [
            {"title": f"Bulk Note {i}", "content": f"Content {i}"}
            for i in range(5)
        ]
        start_bulk = time.perf_counter()
        bulk_notes = zettel_service.bulk_create_notes(bulk_data)
        bulk_time = time.perf_counter() - start_bulk

        # Bulk should be faster (or at least not significantly slower)
        # Allow some variance since both are fast operations
        assert bulk_time <= individual_time * 1.5, (
            f"Bulk ({bulk_time:.3f}s) should be similar or faster than individual ({individual_time:.3f}s)"
        )

    def test_bulk_delete_performance(self, zettel_service):
        """Bulk deletion should be efficient."""
        # Create notes to delete
        notes = []
        for i in range(10):
            note = zettel_service.create_note(
                title=f"To Delete {i}",
                content=f"Content {i}"
            )
            notes.append(note)

        note_ids = [n.id for n in notes]

        start = time.perf_counter()
        deleted = zettel_service.bulk_delete_notes(note_ids)
        elapsed = time.perf_counter() - start

        # 10 deletes should complete in under 500ms
        assert elapsed < 0.5, f"Bulk delete 10 took {elapsed:.3f}s (expected < 0.5s)"
        assert deleted == 10

    def test_bulk_add_tags_performance(self, zettel_service):
        """Bulk tag addition should be efficient."""
        # Create notes
        notes = []
        for i in range(10):
            note = zettel_service.create_note(
                title=f"Tag Test {i}",
                content=f"Content {i}"
            )
            notes.append(note)

        note_ids = [n.id for n in notes]

        start = time.perf_counter()
        updated = zettel_service.bulk_add_tags(note_ids, ["bulk-tag", "performance"])
        elapsed = time.perf_counter() - start

        # Should complete in under 500ms
        assert elapsed < 0.5, f"Bulk add tags took {elapsed:.3f}s (expected < 0.5s)"
        assert updated == 10


class TestSearchPerformance:
    """Performance baselines for search operations."""

    def test_search_with_small_dataset(self, zettel_service):
        """Search should be fast with small dataset."""
        # Create some notes
        for i in range(20):
            zettel_service.create_note(
                title=f"Search Test {i}",
                content=f"This is note number {i} with searchable content.",
                tags=["searchable"] if i % 2 == 0 else ["other"]
            )

        start = time.perf_counter()
        results = zettel_service.search_notes(content="searchable")
        elapsed = time.perf_counter() - start

        # Search should complete in under 100ms
        assert elapsed < 0.1, f"Search took {elapsed:.3f}s (expected < 0.1s)"
        assert len(results) > 0

    def test_search_by_tag_performance(self, zettel_service):
        """Tag-based search should be efficient."""
        # Create tagged notes
        for i in range(20):
            zettel_service.create_note(
                title=f"Tagged {i}",
                content=f"Content {i}",
                tags=["python"] if i % 3 == 0 else ["other"]
            )

        start = time.perf_counter()
        results = zettel_service.search_notes(tag="python")
        elapsed = time.perf_counter() - start

        # Should complete in under 100ms
        assert elapsed < 0.1, f"Tag search took {elapsed:.3f}s (expected < 0.1s)"

    def test_full_text_search_performance(self, zettel_service):
        """Full-text search should be reasonably fast."""
        # Create notes with varied content
        for i in range(20):
            zettel_service.create_note(
                title=f"FTS Note {i}",
                content=f"This note discusses topic {i} in depth with various keywords."
            )

        start = time.perf_counter()
        results = zettel_service.fts_search("discusses topic")
        elapsed = time.perf_counter() - start

        # FTS should complete in under 200ms
        assert elapsed < 0.2, f"FTS took {elapsed:.3f}s (expected < 0.2s)"


class TestLinkOperationPerformance:
    """Performance baselines for link operations."""

    def test_create_link_performance(self, zettel_service):
        """Link creation should be fast."""
        note1 = zettel_service.create_note(title="Source", content="Content")
        note2 = zettel_service.create_note(title="Target", content="Content")

        start = time.perf_counter()
        link = zettel_service.create_link(note1.id, note2.id)
        elapsed = time.perf_counter() - start

        # Should complete in under 50ms
        assert elapsed < 0.05, f"Link create took {elapsed:.3f}s (expected < 0.05s)"
        assert link is not None

    def test_get_linked_notes_performance(self, zettel_service):
        """Getting linked notes should be fast."""
        # Create a hub with many spokes
        hub = zettel_service.create_note(title="Hub", content="Central node")
        for i in range(10):
            spoke = zettel_service.create_note(title=f"Spoke {i}", content="Spoke")
            zettel_service.create_link(hub.id, spoke.id)

        start = time.perf_counter()
        linked = zettel_service.get_linked_notes(hub.id, direction="outgoing")
        elapsed = time.perf_counter() - start

        # Should complete in under 100ms
        assert elapsed < 0.1, f"Get linked took {elapsed:.3f}s (expected < 0.1s)"
        assert len(linked) == 10


class TestDatabaseHealthPerformance:
    """Performance baselines for database operations."""

    def test_health_check_performance(self, zettel_service):
        """Health check should complete quickly."""
        # Create some data
        for i in range(10):
            zettel_service.create_note(title=f"Health {i}", content="Content")

        start = time.perf_counter()
        health = zettel_service.check_database_health()
        elapsed = time.perf_counter() - start

        # Should complete in under 200ms
        assert elapsed < 0.2, f"Health check took {elapsed:.3f}s (expected < 0.2s)"
        assert "sqlite_ok" in health

    def test_get_all_notes_performance(self, zettel_service):
        """Getting all notes should be efficient for small datasets."""
        # Create some notes
        for i in range(30):
            zettel_service.create_note(title=f"All {i}", content=f"Content {i}")

        start = time.perf_counter()
        all_notes = zettel_service.get_all_notes()
        elapsed = time.perf_counter() - start

        # Should complete in under 500ms for 30 notes
        assert elapsed < 0.5, f"Get all took {elapsed:.3f}s (expected < 0.5s)"
        assert len(all_notes) == 30
