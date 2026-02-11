"""Tests for versioned CRUD operations."""

import tempfile
from pathlib import Path

import pytest

from znote_mcp.models.schema import (
    ConflictResult,
    Note,
    NoteType,
    Tag,
    VersionedNote,
    VersionInfo,
)
from znote_mcp.services.zettel_service import ZettelService
from znote_mcp.storage.note_repository import NoteRepository


class TestVersionedRepository:
    """Tests for versioned repository operations."""

    @pytest.fixture
    def temp_env(self, monkeypatch):
        """Set up temporary directories for testing."""
        with tempfile.TemporaryDirectory() as notes_dir:
            with tempfile.TemporaryDirectory() as db_dir:
                monkeypatch.setenv("ZETTELKASTEN_NOTES_DIR", notes_dir)
                monkeypatch.setenv("ZETTELKASTEN_DATABASE_PATH", f"{db_dir}/test.db")
                monkeypatch.setenv("ZETTELKASTEN_GIT_ENABLED", "true")
                monkeypatch.setenv("ZETTELKASTEN_IN_MEMORY_DB", "true")
                yield {"notes_dir": notes_dir, "db_dir": db_dir}

    @pytest.fixture
    def repository(self, temp_env):
        """Create a repository with git enabled."""
        return NoteRepository(
            notes_dir=Path(temp_env["notes_dir"]), use_git=True, in_memory_db=True
        )

    def test_create_versioned_returns_version(self, repository):
        """Test that create_versioned returns a VersionedNote with version info."""
        note = Note(
            title="Test Note",
            content="Test content",
            note_type=NoteType.PERMANENT,
        )

        result = repository.create_versioned(note)

        assert isinstance(result, VersionedNote)
        assert result.note.title == "Test Note"
        assert result.version.commit_hash is not None
        assert len(result.version.commit_hash) >= 7

    def test_get_versioned_returns_version(self, repository):
        """Test that get_versioned returns version info."""
        note = Note(
            title="Test Note",
            content="Content",
            note_type=NoteType.PERMANENT,
        )
        created = repository.create_versioned(note)

        result = repository.get_versioned(created.note.id)

        assert isinstance(result, VersionedNote)
        assert result.version.commit_hash == created.version.commit_hash

    def test_get_versioned_not_found(self, repository):
        """Test get_versioned returns None for non-existent note."""
        result = repository.get_versioned("nonexistent")
        assert result is None

    def test_update_versioned_no_conflict(self, repository):
        """Test update_versioned succeeds without expected_version."""
        note = Note(
            title="Original",
            content="Original content",
            note_type=NoteType.PERMANENT,
        )
        created = repository.create_versioned(note)

        # Update without version check
        updated_note = Note(
            id=created.note.id,
            title="Updated",
            content="Updated content",
            note_type=NoteType.PERMANENT,
        )
        result = repository.update_versioned(updated_note)

        assert isinstance(result, VersionedNote)
        assert result.note.title == "Updated"
        assert result.version.commit_hash != created.version.commit_hash

    def test_update_versioned_with_correct_version(self, repository):
        """Test update_versioned succeeds with correct expected_version."""
        note = Note(
            title="Original",
            content="Original content",
            note_type=NoteType.PERMANENT,
        )
        created = repository.create_versioned(note)

        # Update with correct version
        updated_note = Note(
            id=created.note.id,
            title="Updated",
            content="Updated content",
            note_type=NoteType.PERMANENT,
        )
        result = repository.update_versioned(
            updated_note, expected_version=created.version.commit_hash
        )

        assert isinstance(result, VersionedNote)
        assert result.note.title == "Updated"

    def test_update_versioned_conflict(self, repository):
        """Test update_versioned returns ConflictResult when version mismatch."""
        note = Note(
            title="Original",
            content="Original content",
            note_type=NoteType.PERMANENT,
        )
        created = repository.create_versioned(note)

        # Make a change to create a new version
        updated_once = Note(
            id=created.note.id,
            title="First Update",
            content="First update content",
            note_type=NoteType.PERMANENT,
        )
        repository.update_versioned(updated_once)

        # Try to update with old version
        updated_twice = Note(
            id=created.note.id,
            title="Second Update",
            content="Second update content",
            note_type=NoteType.PERMANENT,
        )
        result = repository.update_versioned(
            updated_twice, expected_version=created.version.commit_hash  # Old version
        )

        assert isinstance(result, ConflictResult)
        assert result.status == "conflict"
        assert result.expected_version == created.version.commit_hash

    def test_delete_versioned_success(self, repository):
        """Test delete_versioned returns VersionInfo on success."""
        note = Note(
            title="To Delete",
            content="Content",
            note_type=NoteType.PERMANENT,
        )
        created = repository.create_versioned(note)

        result = repository.delete_versioned(created.note.id)

        assert isinstance(result, VersionInfo)

    def test_delete_versioned_with_correct_version(self, repository):
        """Test delete_versioned succeeds with correct expected_version."""
        note = Note(
            title="To Delete",
            content="Content",
            note_type=NoteType.PERMANENT,
        )
        created = repository.create_versioned(note)

        result = repository.delete_versioned(
            created.note.id, expected_version=created.version.commit_hash
        )

        assert isinstance(result, VersionInfo)

    def test_delete_versioned_conflict(self, repository):
        """Test delete_versioned returns ConflictResult on version mismatch."""
        note = Note(
            title="To Delete",
            content="Content",
            note_type=NoteType.PERMANENT,
        )
        created = repository.create_versioned(note)

        # Update to create new version
        updated_note = Note(
            id=created.note.id,
            title="Updated",
            content="Updated content",
            note_type=NoteType.PERMANENT,
        )
        repository.update_versioned(updated_note)

        # Try to delete with old version
        result = repository.delete_versioned(
            created.note.id, expected_version=created.version.commit_hash  # Old version
        )

        assert isinstance(result, ConflictResult)
        assert result.status == "conflict"


class TestVersionedService:
    """Tests for versioned service layer operations."""

    @pytest.fixture
    def temp_env(self, monkeypatch):
        """Set up temporary directories for testing."""
        with tempfile.TemporaryDirectory() as notes_dir:
            with tempfile.TemporaryDirectory() as db_dir:
                monkeypatch.setenv("ZETTELKASTEN_NOTES_DIR", notes_dir)
                monkeypatch.setenv("ZETTELKASTEN_DATABASE_PATH", f"{db_dir}/test.db")
                monkeypatch.setenv("ZETTELKASTEN_GIT_ENABLED", "true")
                monkeypatch.setenv("ZETTELKASTEN_IN_MEMORY_DB", "true")
                yield {"notes_dir": notes_dir, "db_dir": db_dir}

    @pytest.fixture
    def service(self, temp_env):
        """Create a ZettelService with git-enabled repository."""
        repo = NoteRepository(
            notes_dir=Path(temp_env["notes_dir"]), use_git=True, in_memory_db=True
        )
        return ZettelService(repository=repo)

    def test_create_note_versioned(self, service):
        """Test create_note_versioned returns VersionedNote."""
        result = service.create_note_versioned(
            title="Service Test",
            content="Test content",
            note_type=NoteType.PERMANENT,
        )

        assert isinstance(result, VersionedNote)
        assert result.note.title == "Service Test"
        assert result.version.commit_hash is not None

    def test_update_note_versioned_success(self, service):
        """Test update_note_versioned succeeds without conflict."""
        created = service.create_note_versioned(
            title="Original",
            content="Original content",
        )

        result = service.update_note_versioned(
            note_id=created.note.id,
            title="Updated",
            expected_version=created.version.commit_hash,
        )

        assert isinstance(result, VersionedNote)
        assert result.note.title == "Updated"

    def test_update_note_versioned_conflict(self, service):
        """Test update_note_versioned returns conflict on version mismatch."""
        created = service.create_note_versioned(
            title="Original",
            content="Original content",
        )

        # Update to create new version
        service.update_note_versioned(
            note_id=created.note.id,
            content="First update",
        )

        # Try with old version
        result = service.update_note_versioned(
            note_id=created.note.id,
            content="Second update",
            expected_version=created.version.commit_hash,
        )

        assert isinstance(result, ConflictResult)
        assert result.status == "conflict"

    def test_delete_note_versioned_success(self, service):
        """Test delete_note_versioned succeeds with correct version."""
        created = service.create_note_versioned(
            title="To Delete",
            content="Content",
        )

        result = service.delete_note_versioned(
            note_id=created.note.id,
            expected_version=created.version.commit_hash,
        )

        assert isinstance(result, VersionInfo)

    def test_delete_note_versioned_conflict(self, service):
        """Test delete_note_versioned returns conflict on version mismatch."""
        created = service.create_note_versioned(
            title="To Delete",
            content="Content",
        )

        # Update to create new version
        service.update_note_versioned(
            note_id=created.note.id,
            content="Updated",
        )

        # Try delete with old version
        result = service.delete_note_versioned(
            note_id=created.note.id,
            expected_version=created.version.commit_hash,
        )

        assert isinstance(result, ConflictResult)
        assert result.status == "conflict"
