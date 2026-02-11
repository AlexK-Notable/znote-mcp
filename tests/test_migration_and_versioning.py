"""Tests for migration methods and git versioned operations.

These tests cover:
- migrate_notes_add_purpose() migration method
- Git versioned CRUD operations (create, update, delete)
- Version conflict detection and handling
"""

import tempfile
from pathlib import Path

import pytest

from znote_mcp.models.schema import (
    ConflictResult,
    Note,
    NotePurpose,
    NoteType,
    Tag,
    VersionedNote,
    VersionInfo,
)
from znote_mcp.services.zettel_service import ZettelService
from znote_mcp.storage.note_repository import NoteRepository


class TestMigrationAddPurpose:
    """Tests for migrate_notes_add_purpose() method.

    The migration method works on notes already in the database.
    It reads files from disk and adds the 'purpose:' field if missing.
    We test by creating notes normally, then modifying the files to
    simulate legacy notes without the purpose field.
    """

    def test_migration_adds_purpose_to_note_without_purpose(self, temp_dirs):
        """Notes without purpose key get GENERAL purpose added."""
        notes_dir, db_dir = temp_dirs

        # Create a note through the repository (adds to DB + file)
        repo = NoteRepository(notes_dir=notes_dir)
        note = repo.create(
            Note(title="Old Note", content="This is an old note without purpose.")
        )

        # Manually rewrite the file to simulate legacy note without purpose
        note_path = notes_dir / f"{note.id}.md"
        legacy_content = f"""---
id: {note.id}
title: Old Note
note_type: permanent
project: general
tags: []
links: []
---
This is an old note without purpose."""
        note_path.write_text(legacy_content)

        # Run migration
        service = ZettelService(repository=repo)
        stats = service.migrate_notes_add_purpose()

        assert stats["migrated"] == 1
        assert stats["skipped"] == 0
        assert len(stats["errors"]) == 0

        # Verify purpose was added
        new_content = note_path.read_text()
        assert "purpose:" in new_content or "note_purpose:" in new_content

    def test_migration_skips_note_with_existing_purpose(self, temp_dirs):
        """Notes with existing purpose are skipped."""
        notes_dir, db_dir = temp_dirs

        # Create a note with purpose through the repository
        repo = NoteRepository(notes_dir=notes_dir)
        note = repo.create(
            Note(
                title="New Note",
                content="This note already has a purpose.",
                note_purpose=NotePurpose.RESEARCH,
            )
        )

        # Verify purpose exists in file
        note_path = notes_dir / f"{note.id}.md"
        content = note_path.read_text()
        assert "purpose:" in content

        # Run migration
        service = ZettelService(repository=repo)
        stats = service.migrate_notes_add_purpose()

        assert stats["migrated"] == 0
        assert stats["skipped"] == 1

    def test_migration_with_custom_default_purpose(self, temp_dirs):
        """Migration can set a custom default purpose."""
        notes_dir, db_dir = temp_dirs

        # Create a note through the repository
        repo = NoteRepository(notes_dir=notes_dir)
        note = repo.create(Note(title="Research Note", content="Content here."))

        # Manually rewrite the file to simulate legacy note without purpose
        note_path = notes_dir / f"{note.id}.md"
        legacy_content = f"""---
id: {note.id}
title: Research Note
note_type: permanent
project: general
tags: []
links: []
---
Content here."""
        note_path.write_text(legacy_content)

        # Run migration with RESEARCH as default
        service = ZettelService(repository=repo)
        stats = service.migrate_notes_add_purpose(default_purpose=NotePurpose.RESEARCH)

        assert stats["migrated"] == 1

        # Verify purpose was added with correct value
        new_content = note_path.read_text()
        assert "research" in new_content.lower()


class TestGitVersionedCreate:
    """Tests for create_versioned() method."""

    def test_create_versioned_returns_versioned_note(self, temp_dirs):
        """create_versioned returns VersionedNote with version info."""
        notes_dir, db_dir = temp_dirs
        repo = NoteRepository(notes_dir=notes_dir, use_git=True)

        result = repo.create_versioned(
            Note(title="Versioned Note", content="Content here")
        )

        assert isinstance(result, VersionedNote)
        assert result.note.title == "Versioned Note"
        assert result.version is not None
        assert len(result.version.commit_hash) == 7  # Short SHA

    def test_create_versioned_creates_git_commit(self, temp_dirs):
        """create_versioned should create a git commit."""
        notes_dir, db_dir = temp_dirs
        repo = NoteRepository(notes_dir=notes_dir, use_git=True)

        result = repo.create_versioned(Note(title="Git Note", content="Content"))

        # Verify git commit exists
        assert result.version.commit_hash != "0000000"
        assert result.version.timestamp is not None

    def test_create_versioned_without_git_returns_dummy_version(self, temp_dirs):
        """create_versioned without git returns dummy version."""
        notes_dir, db_dir = temp_dirs
        repo = NoteRepository(notes_dir=notes_dir, use_git=False)

        result = repo.create_versioned(Note(title="Non-Git Note", content="Content"))

        assert isinstance(result, VersionedNote)
        assert result.version.commit_hash == "0000000"


class TestGitVersionedUpdate:
    """Tests for update_versioned() method."""

    def test_update_versioned_returns_new_version(self, temp_dirs):
        """update_versioned returns updated note with new version."""
        notes_dir, db_dir = temp_dirs
        repo = NoteRepository(notes_dir=notes_dir, use_git=True)

        # Create initial version
        created = repo.create_versioned(
            Note(title="Original", content="Original content")
        )
        original_hash = created.version.commit_hash

        # Update
        result = repo.update_versioned(
            Note(id=created.note.id, title="Updated", content="Updated content")
        )

        assert isinstance(result, VersionedNote)
        assert result.note.title == "Updated"
        assert result.version.commit_hash != original_hash

    def test_update_versioned_with_stale_version_returns_conflict(self, temp_dirs):
        """update_versioned with stale version returns ConflictResult."""
        notes_dir, db_dir = temp_dirs
        repo = NoteRepository(notes_dir=notes_dir, use_git=True)

        # Create initial version
        v1 = repo.create_versioned(Note(title="Version 1", content="Content v1"))
        original_hash = v1.version.commit_hash

        # Update to v2
        repo.update_versioned(
            Note(id=v1.note.id, title="Version 2", content="Content v2")
        )

        # Try to update with stale v1 hash
        result = repo.update_versioned(
            Note(id=v1.note.id, title="Conflict", content="This should conflict"),
            expected_version=original_hash,
        )

        assert isinstance(result, ConflictResult)
        assert result.status == "conflict"
        assert result.note_id == v1.note.id

    def test_update_versioned_with_correct_version_succeeds(self, temp_dirs):
        """update_versioned with correct version succeeds."""
        notes_dir, db_dir = temp_dirs
        repo = NoteRepository(notes_dir=notes_dir, use_git=True)

        # Create initial version
        v1 = repo.create_versioned(Note(title="Version 1", content="Content v1"))

        # Update with correct version
        result = repo.update_versioned(
            Note(id=v1.note.id, title="Version 2", content="Content v2"),
            expected_version=v1.version.commit_hash,
        )

        assert isinstance(result, VersionedNote)
        assert result.note.title == "Version 2"


class TestGitVersionedDelete:
    """Tests for delete_versioned() method."""

    def test_delete_versioned_returns_version_info(self, temp_dirs):
        """delete_versioned returns VersionInfo on success."""
        notes_dir, db_dir = temp_dirs
        repo = NoteRepository(notes_dir=notes_dir, use_git=True)

        # Create note
        created = repo.create_versioned(Note(title="To Delete", content="Delete me"))

        # Delete with correct version
        result = repo.delete_versioned(
            created.note.id, expected_version=created.version.commit_hash
        )

        assert isinstance(result, VersionInfo)
        assert len(result.commit_hash) == 7

    def test_delete_versioned_removes_note(self, temp_dirs):
        """delete_versioned actually removes the note."""
        notes_dir, db_dir = temp_dirs
        repo = NoteRepository(notes_dir=notes_dir, use_git=True)

        # Create note
        created = repo.create_versioned(Note(title="To Delete", content="Delete me"))
        note_id = created.note.id

        # Delete
        repo.delete_versioned(note_id)

        # Verify note is gone
        assert repo.get(note_id) is None

    def test_delete_versioned_with_stale_version_returns_conflict(self, temp_dirs):
        """delete_versioned with stale version returns ConflictResult."""
        notes_dir, db_dir = temp_dirs
        repo = NoteRepository(notes_dir=notes_dir, use_git=True)

        # Create and update to get two versions
        v1 = repo.create_versioned(Note(title="Original", content="Content"))
        original_hash = v1.version.commit_hash

        repo.update_versioned(
            Note(id=v1.note.id, title="Updated", content="New content")
        )

        # Try to delete with stale version
        result = repo.delete_versioned(v1.note.id, expected_version=original_hash)

        assert isinstance(result, ConflictResult)
        assert result.status == "conflict"

    def test_delete_versioned_conflict_preserves_note(self, temp_dirs):
        """When delete conflicts, note is NOT deleted."""
        notes_dir, db_dir = temp_dirs
        repo = NoteRepository(notes_dir=notes_dir, use_git=True)

        # Create and update
        v1 = repo.create_versioned(Note(title="Original", content="Content"))
        original_hash = v1.version.commit_hash

        repo.update_versioned(
            Note(id=v1.note.id, title="Updated", content="New content")
        )

        # Try to delete with stale version (should conflict)
        result = repo.delete_versioned(v1.note.id, expected_version=original_hash)

        assert isinstance(result, ConflictResult)

        # Note should still exist
        note = repo.get(v1.note.id)
        assert note is not None
        assert note.title == "Updated"


class TestGitVersionedGet:
    """Tests for get_versioned() method."""

    def test_get_versioned_returns_note_with_version(self, temp_dirs):
        """get_versioned returns VersionedNote with current version."""
        notes_dir, db_dir = temp_dirs
        repo = NoteRepository(notes_dir=notes_dir, use_git=True)

        # Create note
        created = repo.create_versioned(Note(title="Test Note", content="Content"))

        # Get versioned
        result = repo.get_versioned(created.note.id)

        assert isinstance(result, VersionedNote)
        assert result.note.title == "Test Note"
        assert result.version.commit_hash == created.version.commit_hash

    def test_get_versioned_returns_none_for_nonexistent(self, temp_dirs):
        """get_versioned returns None for nonexistent note."""
        notes_dir, db_dir = temp_dirs
        repo = NoteRepository(notes_dir=notes_dir, use_git=True)

        result = repo.get_versioned("nonexistent-id")

        assert result is None

    def test_get_versioned_reflects_latest_version(self, temp_dirs):
        """get_versioned returns latest version after updates."""
        notes_dir, db_dir = temp_dirs
        repo = NoteRepository(notes_dir=notes_dir, use_git=True)

        # Create and update
        v1 = repo.create_versioned(Note(title="V1", content="Content v1"))
        v2 = repo.update_versioned(
            Note(id=v1.note.id, title="V2", content="Content v2")
        )

        # Get should return v2
        result = repo.get_versioned(v1.note.id)

        assert result.note.title == "V2"
        assert result.version.commit_hash == v2.version.commit_hash
