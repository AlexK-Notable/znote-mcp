"""Tests for bulk operations in the service layer.

These tests cover:
- bulk_add_tags: Adding tags to multiple notes at once
- bulk_remove_tags: Removing tags from multiple notes at once
- bulk_update_project: Moving multiple notes to a different project
- bulk_delete_notes: Deleting multiple notes at once
"""
import pytest

from znote_mcp.exceptions import BulkOperationError, ErrorCode
from znote_mcp.models.schema import Note, Tag, NoteType, NotePurpose
from znote_mcp.services.zettel_service import ZettelService
from znote_mcp.storage.note_repository import NoteRepository


class TestBulkAddTags:
    """Tests for bulk_add_tags operation."""

    def test_bulk_add_tags_adds_to_all_notes(self, zettel_service):
        """Tags should be added to all specified notes."""
        # Create some notes
        notes = []
        for i in range(3):
            note = zettel_service.create_note(
                title=f"Note {i}",
                content=f"Content {i}"
            )
            notes.append(note)

        note_ids = [n.id for n in notes]
        tags_to_add = ["important", "review"]

        # Bulk add tags
        updated_count = zettel_service.bulk_add_tags(note_ids, tags_to_add)

        assert updated_count == 3

        # Verify tags were added to all notes
        for note_id in note_ids:
            note = zettel_service.get_note(note_id)
            tag_names = {t.name for t in note.tags}
            assert "important" in tag_names
            assert "review" in tag_names

    def test_bulk_add_tags_with_existing_tags(self, zettel_service):
        """Adding tags that already exist should not duplicate them."""
        # Create a note with existing tag
        note = zettel_service.create_note(
            title="Tagged Note",
            content="Content",
            tags=["existing"]
        )

        # Add tags including existing one
        updated_count = zettel_service.bulk_add_tags([note.id], ["existing", "new"])

        assert updated_count == 1

        # Verify no duplicate tags
        retrieved = zettel_service.get_note(note.id)
        tag_names = [t.name for t in retrieved.tags]
        assert tag_names.count("existing") == 1
        assert "new" in tag_names

    def test_bulk_add_tags_with_empty_list_raises_error(self, zettel_service):
        """Empty note_ids list should raise BulkOperationError."""
        with pytest.raises(BulkOperationError) as exc_info:
            zettel_service.bulk_add_tags([], ["tag1"])

        assert exc_info.value.code == ErrorCode.BULK_OPERATION_EMPTY_INPUT

    def test_bulk_add_tags_with_nonexistent_note_raises_error(self, zettel_service):
        """Nonexistent notes should raise an error."""
        # Create one real note
        note = zettel_service.create_note(
            title="Real Note",
            content="Content"
        )

        # Mix real and fake note IDs - the bulk operation validates all IDs
        note_ids = [note.id, "nonexistent-id"]

        # The operation should raise an error for invalid IDs
        with pytest.raises(BulkOperationError):
            zettel_service.bulk_add_tags(note_ids, ["tag"])


class TestBulkRemoveTags:
    """Tests for bulk_remove_tags operation."""

    def test_bulk_remove_tags_removes_from_all_notes(self, zettel_service):
        """Tags should be removed from all specified notes."""
        # Create notes with tags
        notes = []
        for i in range(3):
            note = zettel_service.create_note(
                title=f"Note {i}",
                content=f"Content {i}",
                tags=["common", f"unique{i}"]
            )
            notes.append(note)

        note_ids = [n.id for n in notes]

        # Bulk remove the common tag
        updated_count = zettel_service.bulk_remove_tags(note_ids, ["common"])

        assert updated_count == 3

        # Verify tag was removed from all notes
        for i, note_id in enumerate(note_ids):
            note = zettel_service.get_note(note_id)
            tag_names = {t.name for t in note.tags}
            assert "common" not in tag_names
            assert f"unique{i}" in tag_names  # Other tags preserved

    def test_bulk_remove_tags_nonexistent_tag(self, zettel_service):
        """Removing nonexistent tags should not cause errors."""
        note = zettel_service.create_note(
            title="Test Note",
            content="Content",
            tags=["existing"]
        )

        # Try to remove a tag that doesn't exist
        updated_count = zettel_service.bulk_remove_tags([note.id], ["nonexistent"])

        # Should still "update" the note (no-op, but not an error)
        assert updated_count >= 0

        # Original tag should remain
        retrieved = zettel_service.get_note(note.id)
        tag_names = {t.name for t in retrieved.tags}
        assert "existing" in tag_names

    def test_bulk_remove_tags_with_empty_list_raises_error(self, zettel_service):
        """Empty note_ids list should raise BulkOperationError."""
        with pytest.raises(BulkOperationError) as exc_info:
            zettel_service.bulk_remove_tags([], ["tag1"])

        assert exc_info.value.code == ErrorCode.BULK_OPERATION_EMPTY_INPUT


class TestBulkUpdateProject:
    """Tests for bulk_update_project operation."""

    def test_bulk_update_project_moves_all_notes(self, zettel_service):
        """All notes should be moved to the target project."""
        # Create notes in different projects
        notes = []
        for i in range(3):
            note = zettel_service.create_note(
                title=f"Note {i}",
                content=f"Content {i}",
                project=f"project-{i}"
            )
            notes.append(note)

        note_ids = [n.id for n in notes]

        # Bulk move to new project
        updated_count = zettel_service.bulk_update_project(note_ids, "consolidated")

        assert updated_count == 3

        # Verify all notes are in new project
        for note_id in note_ids:
            note = zettel_service.get_note(note_id)
            assert note.project == "consolidated"

    def test_bulk_update_project_preserves_other_fields(self, zettel_service):
        """Moving notes should not affect other note fields."""
        note = zettel_service.create_note(
            title="Original Title",
            content="Original content",
            project="original-project",
            note_type=NoteType.PERMANENT,  # Use enum directly
            tags=["tag1", "tag2"]
        )

        # Get the actual stored content (may include title header from file)
        stored_note = zettel_service.get_note(note.id)
        original_content = stored_note.content

        # Move to new project
        zettel_service.bulk_update_project([note.id], "new-project")

        # Verify other fields are preserved
        retrieved = zettel_service.get_note(note.id)
        assert retrieved.title == "Original Title"
        assert retrieved.content == original_content  # Content preserved exactly
        assert retrieved.note_type == NoteType.PERMANENT
        assert {t.name for t in retrieved.tags} == {"tag1", "tag2"}
        assert retrieved.project == "new-project"

    def test_bulk_update_project_with_empty_list_raises_error(self, zettel_service):
        """Empty note_ids list should raise BulkOperationError."""
        with pytest.raises(BulkOperationError) as exc_info:
            zettel_service.bulk_update_project([], "project")

        assert exc_info.value.code == ErrorCode.BULK_OPERATION_EMPTY_INPUT

    def test_bulk_update_project_with_nonexistent_notes_raises_error(self, zettel_service):
        """Nonexistent notes should raise an error."""
        note = zettel_service.create_note(
            title="Real Note",
            content="Content",
            project="old-project"
        )

        # Mix real and fake note IDs - operation validates all IDs
        note_ids = [note.id, "fake-id-1", "fake-id-2"]

        # The operation should raise an error for invalid IDs
        with pytest.raises(BulkOperationError):
            zettel_service.bulk_update_project(note_ids, "new-project")


class TestBulkDeleteNotes:
    """Tests for bulk_delete_notes operation."""

    def test_bulk_delete_notes_removes_all(self, zettel_service):
        """All specified notes should be deleted."""
        # Create notes
        notes = []
        for i in range(3):
            note = zettel_service.create_note(
                title=f"Note {i}",
                content=f"Content {i}"
            )
            notes.append(note)

        note_ids = [n.id for n in notes]

        # Bulk delete
        deleted_count = zettel_service.bulk_delete_notes(note_ids)

        assert deleted_count == 3

        # Verify all notes are gone
        for note_id in note_ids:
            assert zettel_service.get_note(note_id) is None

    def test_bulk_delete_notes_with_links(self, zettel_service):
        """Deleting notes should handle links correctly."""
        # Create two notes and link them
        note1 = zettel_service.create_note(
            title="Note 1",
            content="Content 1"
        )
        note2 = zettel_service.create_note(
            title="Note 2",
            content="Content 2"
        )

        # Create link from note1 to note2
        zettel_service.create_link(note1.id, note2.id)

        # Delete note1 (which has outgoing link)
        deleted_count = zettel_service.bulk_delete_notes([note1.id])

        assert deleted_count == 1
        assert zettel_service.get_note(note1.id) is None
        # note2 should still exist
        assert zettel_service.get_note(note2.id) is not None

    def test_bulk_delete_notes_with_empty_list_raises_error(self, zettel_service):
        """Empty note_ids list should raise BulkOperationError."""
        with pytest.raises(BulkOperationError) as exc_info:
            zettel_service.bulk_delete_notes([])

        assert exc_info.value.code == ErrorCode.BULK_OPERATION_EMPTY_INPUT

    def test_bulk_delete_notes_with_nonexistent_notes(self, zettel_service):
        """Nonexistent notes should be skipped (not cause errors)."""
        note = zettel_service.create_note(
            title="Real Note",
            content="Content"
        )

        # Mix real and fake note IDs
        note_ids = [note.id, "fake-id"]
        deleted_count = zettel_service.bulk_delete_notes(note_ids)

        # Should delete the real note at minimum
        assert deleted_count >= 1
        assert zettel_service.get_note(note.id) is None


class TestBulkCreateNotes:
    """Tests for bulk_create_notes operation."""

    def test_bulk_create_notes_creates_all(self, zettel_service):
        """All notes should be created."""
        notes_data = [
            {"title": "Note 1", "content": "Content 1"},
            {"title": "Note 2", "content": "Content 2"},
            {"title": "Note 3", "content": "Content 3"},
        ]

        created = zettel_service.bulk_create_notes(notes_data)

        assert len(created) == 3
        for i, note in enumerate(created):
            assert note.title == f"Note {i+1}"
            assert note.content == f"Content {i+1}"

    def test_bulk_create_notes_with_all_fields(self, zettel_service):
        """bulk_create should preserve all optional fields."""
        notes_data = [
            {
                "title": "Full Note",
                "content": "Content with all fields",
                "project": "test-project",
                "note_type": "permanent",
                "note_purpose": "research",
                "plan_id": "plan-123",
                "tags": ["tag1", "tag2"],
                "metadata": {"key": "value"}
            }
        ]

        created = zettel_service.bulk_create_notes(notes_data)

        assert len(created) == 1
        note = created[0]
        assert note.title == "Full Note"
        assert note.project == "test-project"
        assert note.note_type == NoteType.PERMANENT
        assert note.note_purpose == NotePurpose.RESEARCH
        assert note.plan_id == "plan-123"
        assert {t.name for t in note.tags} == {"tag1", "tag2"}

    def test_bulk_create_notes_empty_list_raises_error(self, zettel_service):
        """Empty list should raise BulkOperationError."""
        with pytest.raises(BulkOperationError) as exc_info:
            zettel_service.bulk_create_notes([])

        assert exc_info.value.code == ErrorCode.BULK_OPERATION_EMPTY_INPUT

    def test_bulk_create_notes_default_values(self, zettel_service):
        """Notes should get default values when not specified."""
        notes_data = [
            {"title": "Minimal Note", "content": "Just content"}
        ]

        created = zettel_service.bulk_create_notes(notes_data)

        assert len(created) == 1
        note = created[0]
        assert note.project == "general"  # Default project
        assert note.note_type == NoteType.PERMANENT  # Default type (permanent is default)
        assert note.note_purpose == NotePurpose.GENERAL  # Default purpose
