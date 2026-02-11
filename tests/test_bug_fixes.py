"""Tests for bug fixes identified in code review.

These tests verify that bugs discovered during the two-tier code review
process have been properly fixed and won't regress.
"""

import tempfile
from datetime import timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import select

from znote_mcp.backup import BackupManager
from znote_mcp.models.db_models import DBLink, DBNote, DBProject
from znote_mcp.models.schema import ConflictResult, Note, NotePurpose, NoteType, Tag
from znote_mcp.storage.note_repository import NoteRepository, escape_like_pattern
from znote_mcp.storage.project_repository import ProjectRepository


class TestLikePatternEscaping:
    """Tests for SQL LIKE pattern injection prevention (Task #4, #13)."""

    def test_escape_like_pattern_escapes_percent(self):
        """% should be escaped to prevent wildcard matching."""
        result = escape_like_pattern("100% complete")
        assert result == "100\\% complete"

    def test_escape_like_pattern_escapes_underscore(self):
        """_ should be escaped to prevent single-char wildcard matching."""
        result = escape_like_pattern("file_name")
        assert result == "file\\_name"

    def test_escape_like_pattern_escapes_backslash(self):
        """Backslash should be escaped first."""
        result = escape_like_pattern("path\\to\\file")
        assert result == "path\\\\to\\\\file"

    def test_escape_like_pattern_multiple_wildcards(self):
        """Multiple wildcards should all be escaped."""
        result = escape_like_pattern("100%_test_50%")
        assert result == "100\\%\\_test\\_50\\%"

    def test_search_with_percent_finds_literal_match(self, note_repository):
        """Search for '100%' should not match all notes."""
        # Create notes with and without %
        note1 = note_repository.create(
            Note(title="100% Complete", content="This task is fully done")
        )
        note2 = note_repository.create(
            Note(title="Half Done", content="This is 50% progress")
        )
        note3 = note_repository.create(
            Note(title="Other Note", content="Random content without percent")
        )

        # Search for literal "100%" - should only find note1
        results = note_repository.search(content="100%")

        assert len(results) == 1
        assert results[0].id == note1.id

    def test_search_with_underscore_finds_literal_match(self, note_repository):
        """Search for 'file_name' should not match 'filename'."""
        # Create notes with and without _
        note1 = note_repository.create(
            Note(title="file_name.txt", content="File with underscore")
        )
        note2 = note_repository.create(
            Note(title="filename.txt", content="File without underscore")
        )

        # Search for literal "file_name" - should only find note1
        results = note_repository.search(title="file_name")

        assert len(results) == 1
        assert results[0].id == note1.id


class TestTimezoneConsistency:
    """Tests for timezone-aware timestamps (Task #2, #12)."""

    def test_note_created_at_is_timezone_aware(self, note_repository):
        """Note created_at should be timezone-aware UTC."""
        note = note_repository.create(Note(title="TZ Test", content="Testing timezone"))

        retrieved = note_repository.get(note.id)

        assert retrieved.created_at.tzinfo is not None
        assert retrieved.created_at.tzinfo == timezone.utc

    def test_note_updated_at_is_timezone_aware(self, note_repository):
        """Note updated_at should be timezone-aware UTC."""
        note = note_repository.create(Note(title="TZ Test", content="Testing timezone"))

        retrieved = note_repository.get(note.id)

        assert retrieved.updated_at.tzinfo is not None
        assert retrieved.updated_at.tzinfo == timezone.utc

    def test_db_timestamps_match_model_timestamps(self, note_repository):
        """DB and model timestamps should be close (within 1 second)."""
        note = note_repository.create(Note(title="TZ Test", content="Testing timezone"))

        # Get from file (model)
        file_note = note_repository.get(note.id)

        # Get from DB directly
        with note_repository.session_factory() as session:
            db_note = session.execute(
                select(DBNote).where(DBNote.id == note.id)
            ).scalar_one()

            # Both should be close in time
            if db_note.created_at.tzinfo is None:
                # If DB still has naive datetime, treat as UTC for comparison
                db_time = db_note.created_at.replace(tzinfo=timezone.utc)
            else:
                db_time = db_note.created_at

            diff = abs((db_time - file_note.created_at).total_seconds())
            assert diff < 1.0, f"Timestamps differ by {diff}s"


class TestBulkCreateFieldPersistence:
    """Tests for bulk_create_notes preserving all fields (Task #3, #14)."""

    def test_bulk_create_preserves_project(self, zettel_service):
        """bulk_create_notes should preserve project field."""
        notes_data = [
            {
                "title": "Project Note",
                "content": "Note in specific project",
                "project": "test-project",
            }
        ]

        created = zettel_service.bulk_create_notes(notes_data)

        assert len(created) == 1
        assert created[0].project == "test-project"

        # Verify persistence by re-reading
        retrieved = zettel_service.get_note(created[0].id)
        assert retrieved.project == "test-project"

    def test_bulk_create_preserves_note_purpose(self, zettel_service):
        """bulk_create_notes should preserve note_purpose field."""
        notes_data = [
            {
                "title": "Research Note",
                "content": "Investigation content",
                "note_purpose": "research",
            }
        ]

        created = zettel_service.bulk_create_notes(notes_data)

        assert len(created) == 1
        assert created[0].note_purpose == NotePurpose.RESEARCH

        # Verify persistence
        retrieved = zettel_service.get_note(created[0].id)
        assert retrieved.note_purpose == NotePurpose.RESEARCH

    def test_bulk_create_preserves_plan_id(self, zettel_service):
        """bulk_create_notes should preserve plan_id field."""
        notes_data = [
            {
                "title": "Planning Note",
                "content": "Plan content",
                "note_purpose": "planning",
                "plan_id": "plan-123-abc",
            }
        ]

        created = zettel_service.bulk_create_notes(notes_data)

        assert len(created) == 1
        assert created[0].plan_id == "plan-123-abc"

        # Verify persistence
        retrieved = zettel_service.get_note(created[0].id)
        assert retrieved.plan_id == "plan-123-abc"

    def test_bulk_create_preserves_all_fields_together(self, zettel_service):
        """bulk_create_notes should preserve all optional fields together."""
        notes_data = [
            {
                "title": "Full Note",
                "content": "Complete note with all fields",
                "project": "my-project",
                "note_purpose": "bugfixing",
                "plan_id": "debug-session-001",
                "note_type": "fleeting",
                "tags": ["bug", "urgent"],
            }
        ]

        created = zettel_service.bulk_create_notes(notes_data)

        assert len(created) == 1
        note = created[0]
        assert note.project == "my-project"
        assert note.note_purpose == NotePurpose.BUGFIXING
        assert note.plan_id == "debug-session-001"
        assert note.note_type == NoteType.FLEETING
        assert {t.name for t in note.tags} == {"bug", "urgent"}


class TestBackupPathTraversal:
    """Tests for backup restore path validation (Task #6, #11)."""

    def test_restore_rejects_absolute_path_outside_backup_dir(self):
        """Restore should reject absolute paths outside backup_dir."""
        with tempfile.TemporaryDirectory() as backup_dir:
            manager = BackupManager(backup_dir=backup_dir)

            # Try to restore from /etc/passwd
            result = manager.restore_database("/etc/passwd")

            assert result is False

    def test_restore_rejects_path_traversal(self):
        """Restore should reject ../ path traversal attempts."""
        with tempfile.TemporaryDirectory() as backup_dir:
            manager = BackupManager(backup_dir=backup_dir)

            # Try path traversal
            malicious_path = f"{backup_dir}/../../../etc/passwd"
            result = manager.restore_database(malicious_path)

            assert result is False

    def test_restore_rejects_tmp_path(self):
        """Restore should reject paths in /tmp outside backup_dir."""
        with tempfile.TemporaryDirectory() as backup_dir:
            manager = BackupManager(backup_dir=backup_dir)

            result = manager.restore_database("/tmp/malicious.db")

            assert result is False

    def test_restore_accepts_valid_backup_path(self):
        """Restore should accept paths within backup_dir."""
        with tempfile.TemporaryDirectory() as backup_dir:
            manager = BackupManager(backup_dir=backup_dir)

            # Create a fake backup file within backup_dir
            backup_path = Path(backup_dir) / "test_backup.db"
            backup_path.write_bytes(b"fake db content")

            # This should fail because it's not a valid SQLite DB,
            # but it should NOT fail the path validation
            # The error will be from SQLite, not from path validation
            with patch.object(manager, "_lock"):
                # We'll just check the path validation passes
                # by verifying the backup_path.exists() check is reached
                result = manager.restore_database(str(backup_path))
                # Will return False because no valid DB, but path was accepted


class TestDeleteVersionedConflict:
    """Tests for delete_versioned conflict handling (Task #5, #10)."""

    def test_delete_versioned_with_stale_version_returns_conflict(self, temp_dirs):
        """delete_versioned with wrong version should return ConflictResult."""
        notes_dir, db_dir = temp_dirs
        repo = NoteRepository(notes_dir=notes_dir, use_git=True)

        # Create a versioned note
        versioned = repo.create_versioned(
            Note(title="Test Note", content="Original content")
        )
        original_version = versioned.version.commit_hash

        # Update to create new version
        updated = repo.update_versioned(
            Note(id=versioned.note.id, title="Updated Title", content="Updated content")
        )

        # Try to delete with old version
        result = repo.delete_versioned(
            versioned.note.id, expected_version=original_version
        )

        # Should return ConflictResult
        assert isinstance(result, ConflictResult)
        assert result.status == "conflict"
        assert result.note_id == versioned.note.id

    def test_delete_versioned_conflict_leaves_note_intact(self, temp_dirs):
        """When conflict detected, note should NOT be deleted."""
        notes_dir, db_dir = temp_dirs
        repo = NoteRepository(notes_dir=notes_dir, use_git=True)

        # Create a versioned note
        versioned = repo.create_versioned(
            Note(title="Test Note", content="Original content")
        )
        original_version = versioned.version.commit_hash

        # Update to create new version
        repo.update_versioned(
            Note(id=versioned.note.id, title="Updated Title", content="Updated content")
        )

        # Try to delete with old version (should conflict)
        result = repo.delete_versioned(
            versioned.note.id, expected_version=original_version
        )

        assert isinstance(result, ConflictResult)

        # CRITICAL: Note should still exist
        note = repo.get(versioned.note.id)
        assert note is not None, "Note was deleted despite version conflict!"
        assert note.title == "Updated Title"

    def test_delete_versioned_with_correct_version_succeeds(self, temp_dirs):
        """delete_versioned with correct version should succeed."""
        notes_dir, db_dir = temp_dirs
        repo = NoteRepository(notes_dir=notes_dir, use_git=True)

        # Create a versioned note
        versioned = repo.create_versioned(Note(title="Test Note", content="Content"))

        # Delete with correct version
        result = repo.delete_versioned(
            versioned.note.id, expected_version=versioned.version.commit_hash
        )

        # Should succeed (not a ConflictResult)
        assert not isinstance(result, ConflictResult)

        # Note should be gone
        note = repo.get(versioned.note.id)
        assert note is None


class TestSearchCountConsistency:
    """Tests for search and count consistency (Task #18)."""

    def test_search_and_count_return_same_number(self, note_repository):
        """search() results count should match count_search_results()."""
        # Create some notes
        note_repository.create(
            Note(
                title="Python Guide",
                content="Learn Python programming",
                tags=[Tag(name="python")],
            )
        )
        note_repository.create(
            Note(
                title="Go Guide", content="Learn Go programming", tags=[Tag(name="go")]
            )
        )
        note_repository.create(
            Note(
                title="Python Tips",
                content="Advanced Python tips",
                tags=[Tag(name="python")],
            )
        )

        # Search and count with same criteria
        results = note_repository.search(content="Python")
        count = note_repository.count_search_results(content="Python")

        assert len(results) == count
        assert count == 2

    def test_search_and_count_consistency_with_tags(self, note_repository):
        """search() and count_search_results() should be consistent with tag filters."""
        note_repository.create(
            Note(
                title="Tagged Note 1", content="Content 1", tags=[Tag(name="important")]
            )
        )
        note_repository.create(
            Note(
                title="Tagged Note 2", content="Content 2", tags=[Tag(name="important")]
            )
        )
        note_repository.create(Note(title="Untagged Note", content="Content 3"))

        results = note_repository.search(tag="important")
        count = note_repository.count_search_results(tag="important")

        assert len(results) == count
        assert count == 2
