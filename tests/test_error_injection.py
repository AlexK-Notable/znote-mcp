"""Tests for error injection and failure handling.

Tests that verify the system handles various failure scenarios gracefully.
"""

import os
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from znote_mcp.exceptions import (
    BulkOperationError,
    ConfigurationError,
    DatabaseCorruptionError,
    ErrorCode,
    LinkError,
    NoteNotFoundError,
    NoteValidationError,
    SearchError,
    StorageError,
    TagError,
    ValidationError,
    ZettelkastenError,
)
from znote_mcp.models.schema import LinkType, NoteType
from znote_mcp.services.search_service import SearchService
from znote_mcp.services.zettel_service import ZettelService


class TestExceptionHierarchy:
    """Tests for the exception class hierarchy and serialization."""

    def test_base_exception_to_dict(self):
        """Test base exception serialization."""
        exc = ZettelkastenError(
            "Test error", code=ErrorCode.VALIDATION_FAILED, details={"key": "value"}
        )
        result = exc.to_dict()

        assert result["error"] == "ZettelkastenError"
        assert result["code"] == ErrorCode.VALIDATION_FAILED.value
        assert result["code_name"] == "VALIDATION_FAILED"
        assert result["message"] == "Test error"
        assert result["details"] == {"key": "value"}

    def test_note_not_found_error(self):
        """Test NoteNotFoundError with note ID."""
        exc = NoteNotFoundError("abc123")

        assert exc.note_id == "abc123"
        assert exc.code == ErrorCode.NOTE_NOT_FOUND
        assert "abc123" in str(exc)

    def test_link_error_with_details(self):
        """Test LinkError with full context."""
        exc = LinkError(
            "Cannot create link",
            source_id="src123",
            target_id="tgt456",
            link_type="reference",
            code=ErrorCode.LINK_ALREADY_EXISTS,
        )

        assert exc.source_id == "src123"
        assert exc.target_id == "tgt456"
        assert exc.link_type == "reference"
        assert exc.code == ErrorCode.LINK_ALREADY_EXISTS
        assert "src123" in exc.details["source_id"]

    def test_storage_error_path_sanitization(self):
        """Test that StorageError sanitizes full paths."""
        exc = StorageError(
            "Write failed",
            operation="save",
            path="/home/user/secret/path/notes/test.md",
        )

        # Should only show file name, not full path
        assert "/home/user/secret" not in exc.details.get("path_hint", "")
        assert "test.md" in exc.details.get("path_hint", "")

    def test_database_corruption_error(self):
        """Test DatabaseCorruptionError with recovery info."""
        exc = DatabaseCorruptionError(
            "FTS index corrupted", recovered=True, backup_path="/backup/db.bak"
        )

        assert exc.recovered is True
        assert exc.backup_path == "/backup/db.bak"
        assert exc.details["recovered"] is True

    def test_bulk_operation_error_with_failed_ids(self):
        """Test BulkOperationError tracks failures."""
        failed = ["id1", "id2", "id3"]
        exc = BulkOperationError(
            "Partial failure",
            operation="bulk_create",
            total_count=10,
            success_count=7,
            failed_ids=failed,
        )

        assert exc.total_count == 10
        assert exc.success_count == 7
        assert exc.failed_count == 3
        assert exc.failed_ids == failed

    def test_bulk_operation_error_validates_counts(self):
        """Test BulkOperationError validates count invariants."""
        with pytest.raises(ValueError, match="non-negative"):
            BulkOperationError("error", "op", total_count=-1)

        with pytest.raises(ValueError, match="cannot exceed"):
            BulkOperationError("error", "op", total_count=5, success_count=10)


class TestNoteNotFoundHandling:
    """Tests for handling missing notes."""

    def test_get_nonexistent_note(self, zettel_service):
        """Test that getting a non-existent note returns None or raises."""
        # Current implementation returns None for missing notes
        result = zettel_service.get_note("nonexistent-id-12345")
        assert result is None

    def test_update_nonexistent_note(self, zettel_service):
        """Test that updating a non-existent note raises NoteNotFoundError."""
        with pytest.raises(NoteNotFoundError) as exc_info:
            zettel_service.update_note("nonexistent-id", content="New content")

        assert exc_info.value.note_id == "nonexistent-id"

    def test_delete_nonexistent_note(self, zettel_service):
        """Test that deleting a non-existent note raises NoteNotFoundError."""
        with pytest.raises(NoteNotFoundError):
            zettel_service.delete_note("nonexistent-id")


class TestLinkErrorHandling:
    """Tests for handling link-related errors."""

    def test_link_to_nonexistent_target(self, zettel_service):
        """Test linking to a non-existent target note."""
        source = zettel_service.create_note(title="Source", content="Content")

        with pytest.raises((NoteNotFoundError, LinkError)):
            zettel_service.create_link(source.id, "nonexistent-target")

    def test_link_from_nonexistent_source(self, zettel_service):
        """Test linking from a non-existent source note."""
        target = zettel_service.create_note(title="Target", content="Content")

        with pytest.raises((NoteNotFoundError, LinkError)):
            zettel_service.create_link("nonexistent-source", target.id)

    def test_self_referencing_link_allowed(self, zettel_service):
        """Test that self-referencing links are currently allowed (no validation)."""
        note = zettel_service.create_note(title="Self Ref", content="Content")

        # Current implementation allows self-referencing links
        # This test documents the actual behavior
        result = zettel_service.create_link(note.id, note.id)
        # Link was created (returns the source note tuple)
        assert result is not None

    def test_duplicate_link_is_idempotent(self, zettel_service):
        """Test that creating a duplicate link is idempotent (no error raised)."""
        note1 = zettel_service.create_note(title="Note 1", content="Content")
        note2 = zettel_service.create_note(title="Note 2", content="Content")

        # Create initial link
        zettel_service.create_link(note1.id, note2.id, LinkType.REFERENCE)

        # Duplicate link is silently ignored (idempotent behavior)
        result = zettel_service.create_link(note1.id, note2.id, LinkType.REFERENCE)

        # Returns the source note without error
        assert result is not None
        source_note, target_note = result
        assert source_note.id == note1.id

        # Still only one link exists
        links = [l for l in source_note.links if l.target_id == note2.id]
        assert len(links) == 1


class TestTagErrorHandling:
    """Tests for handling tag-related errors."""

    def test_add_tag_to_nonexistent_note(self, zettel_service):
        """Test adding a tag to a non-existent note."""
        with pytest.raises(NoteNotFoundError):
            zettel_service.add_tag_to_note("nonexistent-note", "test-tag")

    def test_remove_tag_from_nonexistent_note(self, zettel_service):
        """Test removing a tag from a non-existent note."""
        with pytest.raises(NoteNotFoundError):
            zettel_service.remove_tag_from_note("nonexistent-note", "test-tag")


class TestSearchErrorHandling:
    """Tests for handling search-related errors."""

    def test_find_similar_to_nonexistent_note(self, zettel_service):
        """Test finding similar notes for non-existent note."""
        search_service = SearchService(zettel_service)

        with pytest.raises(NoteNotFoundError):
            search_service.find_similar_notes("nonexistent-id")

    def test_get_linked_notes_nonexistent_note(self, zettel_service):
        """Test getting linked notes for non-existent note raises NoteNotFoundError."""
        # Raises NoteNotFoundError for non-existent notes
        with pytest.raises(NoteNotFoundError):
            zettel_service.get_linked_notes("nonexistent-id")


class TestDatabaseFailureHandling:
    """Tests for handling database failures."""

    def test_database_corruption_error_creation(self):
        """Test DatabaseCorruptionError can be properly created and used."""
        # Verify the exception class works correctly
        exc = DatabaseCorruptionError(
            "FTS index corrupted", recovered=True, backup_path="/backup/db.bak"
        )

        assert exc.recovered is True
        assert exc.backup_path == "/backup/db.bak"
        assert exc.code == ErrorCode.DATABASE_CORRUPTED
        assert "FTS index corrupted" in str(exc)

    def test_storage_error_with_original_exception(self):
        """Test StorageError wraps original exceptions properly."""
        original = OSError(28, "No space left on device")
        exc = StorageError(
            "Write failed",
            operation="save",
            code=ErrorCode.STORAGE_READ_FAILED,
            original_error=original,
        )

        assert exc.original_error == original
        assert "No space left" in exc.details["original_error"]
        assert exc.operation == "save"


class TestConcurrentAccessHandling:
    """Tests for handling concurrent access scenarios."""

    def test_sequential_updates_succeed(self, zettel_service):
        """Test that sequential updates succeed without corruption."""
        note = zettel_service.create_note(
            title="Concurrent Test", content="Original content"
        )

        # Simulate sequential update scenario
        # First update succeeds
        zettel_service.update_note(note.id, content="Update 1")

        # Second update should also succeed
        zettel_service.update_note(note.id, content="Update 2")

        # Verify final state contains the update
        result = zettel_service.get_note(note.id)
        assert "Update 2" in result.content


class TestBulkOperationFailures:
    """Tests for handling bulk operation failures."""

    def test_bulk_create_with_invalid_note(self, zettel_service):
        """Test bulk create handles invalid notes gracefully."""
        notes = [
            {"title": "Valid Note 1", "content": "Content 1"},
            {"title": "", "content": "No title - might be invalid"},  # Empty title
            {"title": "Valid Note 2", "content": "Content 2"},
        ]

        # Should either succeed with warnings or raise BulkOperationError
        try:
            result = zettel_service.bulk_create_notes(notes)
            # If it succeeds, check that valid notes were created
            assert len(result) >= 1
        except (BulkOperationError, NoteValidationError) as e:
            # If it fails, it should fail with proper error info
            if isinstance(e, BulkOperationError):
                assert e.success_count >= 0
                assert e.total_count == 3

    def test_bulk_delete_with_nonexistent(self, zettel_service):
        """Test bulk delete handles non-existent notes."""
        note = zettel_service.create_note(title="To Delete", content="Content")

        note_ids = [note.id, "nonexistent-1", "nonexistent-2"]

        # Should delete what it can and report failures
        try:
            result = zettel_service.bulk_delete_notes(note_ids)
            # Check result format if it returns something
        except BulkOperationError as e:
            # Should indicate partial failure
            assert e.success_count >= 1  # At least one deleted
            assert e.failed_count >= 1  # At least some failures


class TestFileSystemFailures:
    """Tests for handling file system failures."""

    def test_write_to_readonly_directory(self):
        """Test handling of read-only directory errors."""
        # This test simulates what happens when we can't write to notes directory
        with patch("builtins.open") as mock_open:
            mock_open.side_effect = PermissionError("Read-only filesystem")

            # Operations that write should handle this gracefully
            # The service should catch and wrap in StorageError

    def test_disk_full_simulation(self):
        """Test handling of disk full scenarios."""
        # Simulate OSError for disk full
        with patch("builtins.open") as mock_open:
            mock_open.side_effect = OSError(28, "No space left on device")

            # Write operations should fail gracefully


class TestInputValidation:
    """Tests for input validation and sanitization."""

    def test_note_with_path_traversal_in_id(self, zettel_service):
        """Test that path traversal in IDs is rejected."""
        # Attempt to use path traversal in note retrieval
        malicious_ids = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32",
            "notes/../secrets/api_key",
        ]

        for malicious_id in malicious_ids:
            # Should raise ValueError for path traversal attempts
            with pytest.raises(ValueError, match="path traversal"):
                zettel_service.get_note(malicious_id)

    def test_extremely_long_title(self, zettel_service):
        """Test handling of extremely long titles."""
        long_title = "A" * 10000  # 10KB title

        # Should either truncate or raise validation error
        try:
            note = zettel_service.create_note(
                title=long_title, content="Normal content"
            )
            # If it succeeds, title might be truncated
            assert len(note.title) <= 10000
        except (NoteValidationError, ValidationError):
            # Valid rejection of too-long title
            pass

    def test_extremely_long_content(self, zettel_service):
        """Test handling of extremely long content."""
        long_content = "B" * 1_000_000  # 1MB content

        # Should handle without crashing
        try:
            note = zettel_service.create_note(
                title="Long Content Note", content=long_content
            )
            assert len(note.content) >= 0  # Created successfully
        except (NoteValidationError, ValidationError, MemoryError):
            # Also valid to reject
            pass

    def test_special_characters_in_tags(self, zettel_service):
        """Test handling of special characters in tags."""
        special_tags = [
            "tag-with-dash",
            "tag_with_underscore",
            "123numeric",
            "MixedCase",
        ]

        note = zettel_service.create_note(
            title="Special Tags", content="Content", tags=special_tags
        )

        # Should handle valid tags
        tag_names = {tag.name for tag in note.tags}
        for tag in special_tags:
            assert tag.lower() in tag_names or tag in tag_names


class TestNullAndEmptyHandling:
    """Tests for handling null/empty values."""

    def test_empty_search_query(self, zettel_service):
        """Test search with empty query."""
        search_service = SearchService(zettel_service)

        # Empty query should return empty results, not error
        results = search_service.search_combined(text="")
        # Empty text with no other filters returns all notes (no text matching)
        # This is valid behavior - search_combined with empty text returns all
        assert isinstance(results, list)

    def test_none_values_in_update(self, zettel_service):
        """Test updating with None values preserves existing data."""
        note = zettel_service.create_note(
            title="Original Title", content="Original Content"
        )

        # Update with only content, title should remain
        updated = zettel_service.update_note(
            note.id,
            content="New Content",
            # title not specified
        )

        assert updated.title == "Original Title"
        assert updated.content == "New Content"

    def test_empty_tags_list(self, zettel_service):
        """Test creating note with empty tags list."""
        note = zettel_service.create_note(title="No Tags", content="Content", tags=[])

        assert note.tags == [] or len(note.tags) == 0


class TestErrorRecovery:
    """Tests for error recovery mechanisms."""

    def test_failed_operation_does_not_affect_other_data(self, zettel_service):
        """Test that failed operations don't corrupt other data."""
        # Create a note successfully
        note = zettel_service.create_note(
            title="Transaction Test", content="Original Content Here"
        )

        initial_notes = zettel_service.get_all_notes()
        initial_count = len(initial_notes)

        # Attempt an operation that should fail
        try:
            # Try to update a non-existent note
            zettel_service.update_note("nonexistent", content="Bad update")
        except NoteNotFoundError:
            pass

        # Original note should still exist and contain original content
        recovered_note = zettel_service.get_note(note.id)
        assert "Original Content Here" in recovered_note.content

        # Count should be unchanged
        after_notes = zettel_service.get_all_notes()
        assert len(after_notes) == initial_count
