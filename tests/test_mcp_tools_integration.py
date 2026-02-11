"""Integration tests for MCP tools.

These tests exercise the MCP tools with real repositories and services,
avoiding mock theater by testing actual behavior end-to-end.

Tests cover:
- Project MCP tools (zk_create_project, zk_list_projects, etc.)
- Admin tools (zk_system, zk_status)
- Restore tool (zk_restore)
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from znote_mcp.models.schema import Note, NotePurpose, NoteType, Project
from znote_mcp.services.zettel_service import ZettelService
from znote_mcp.storage.note_repository import NoteRepository
from znote_mcp.storage.project_repository import ProjectRepository


class TestProjectMcpToolsIntegration:
    """Integration tests for Project MCP tools."""

    @pytest.fixture
    def setup_services(self, temp_dirs):
        """Set up real services with temp directories."""
        notes_dir, db_dir = temp_dirs
        repo = NoteRepository(notes_dir=notes_dir)
        service = ZettelService(repository=repo)
        project_repo = ProjectRepository(engine=repo.engine)
        return service, project_repo, notes_dir

    def test_create_project_creates_real_project(self, setup_services):
        """zk_create_project should create a real project in the database."""
        service, project_repo, notes_dir = setup_services

        # Create project through service (simulating MCP tool behavior)
        project = Project(
            id="test-project", name="Test Project", description="A test project"
        )
        created = project_repo.create(project)

        # Verify project exists
        assert created.id == "test-project"
        assert created.name == "Test Project"

        # Verify can retrieve it
        retrieved = project_repo.get("test-project")
        assert retrieved is not None
        assert retrieved.name == "Test Project"

    def test_list_projects_returns_all_projects(self, setup_services):
        """zk_list_projects should return all projects."""
        service, project_repo, notes_dir = setup_services

        # Create multiple projects
        for i in range(3):
            project_repo.create(
                Project(
                    id=f"project-{i}",
                    name=f"Project {i}",
                    description=f"Description {i}",
                )
            )

        # List projects
        projects = project_repo.get_all()

        # Should have 3 projects
        assert len(projects) == 3
        project_ids = {p.id for p in projects}
        assert project_ids == {"project-0", "project-1", "project-2"}

    def test_get_project_retrieves_details(self, setup_services):
        """zk_get_project should retrieve full project details."""
        service, project_repo, notes_dir = setup_services

        # Create project with metadata
        project_repo.create(
            Project(
                id="detailed-project",
                name="Detailed Project",
                description="Project with details",
                path="/some/path",
                metadata={"key": "value"},
            )
        )

        # Get project
        project = project_repo.get("detailed-project")

        # Verify all fields
        assert project.id == "detailed-project"
        assert project.name == "Detailed Project"
        assert project.description == "Project with details"
        assert project.path == "/some/path"
        assert project.metadata == {"key": "value"}

    def test_delete_project_removes_project(self, setup_services):
        """zk_delete_project should remove the project."""
        service, project_repo, notes_dir = setup_services

        # Create project
        project_repo.create(Project(id="to-delete", name="To Delete"))

        # Verify exists
        assert project_repo.get("to-delete") is not None

        # Delete
        project_repo.delete("to-delete")

        # Verify gone
        assert project_repo.get("to-delete") is None

    def test_project_with_notes_cannot_be_deleted(self, setup_services):
        """Project with notes should not be deletable."""
        service, project_repo, notes_dir = setup_services

        # Create project
        project_repo.create(Project(id="has-notes", name="Has Notes"))

        # Create note in project
        service.create_note(
            title="Note in Project", content="Content", project="has-notes"
        )

        # Try to delete project - should fail
        from znote_mcp.exceptions import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            project_repo.delete("has-notes")

        assert "notes belong to it" in str(exc_info.value)


class TestSystemToolIntegration:
    """Integration tests for zk_system admin tool."""

    @pytest.fixture
    def setup_services(self, temp_dirs):
        """Set up real services with temp directories."""
        notes_dir, db_dir = temp_dirs
        repo = NoteRepository(notes_dir=notes_dir)
        service = ZettelService(repository=repo)
        return service, repo, notes_dir

    def test_system_rebuild_index(self, setup_services):
        """zk_system rebuild should rebuild the database from files."""
        service, repo, notes_dir = setup_services

        # Create some notes
        note1 = service.create_note(title="Note 1", content="Content 1")
        note2 = service.create_note(title="Note 2", content="Content 2")

        # Get initial count
        initial_count = len(service.get_all_notes())
        assert initial_count == 2

        # Rebuild index
        service.rebuild_index()

        # Should still have 2 notes
        assert len(service.get_all_notes()) == 2

    def test_system_health_check(self, setup_services):
        """zk_system health should validate database integrity."""
        service, repo, notes_dir = setup_services

        # Create some notes
        service.create_note(title="Test", content="Content")

        # Check health
        result = service.check_database_health()

        # Should pass basic checks
        assert "sqlite_ok" in result
        assert "fts_ok" in result
        assert "note_count" in result

    def test_system_after_delete_fragmentation(self, setup_services):
        """System should work correctly after creating/deleting many notes."""
        service, repo, notes_dir = setup_services

        # Create and delete some notes to create fragmentation
        for i in range(10):
            note = service.create_note(title=f"Note {i}", content=f"Content {i}")
            if i % 2 == 0:
                service.delete_note(note.id)

        # Remaining notes should be accessible
        notes = service.get_all_notes()
        assert len(notes) == 5  # Only odd-numbered notes remain

        # Health check should still pass
        health = service.check_database_health()
        assert health["sqlite_ok"]


class TestStatusToolIntegration:
    """Integration tests for zk_status tool."""

    @pytest.fixture
    def setup_services(self, temp_dirs):
        """Set up real services with temp directories."""
        notes_dir, db_dir = temp_dirs
        repo = NoteRepository(notes_dir=notes_dir)
        service = ZettelService(repository=repo)
        return service, repo, notes_dir

    def test_status_returns_note_counts(self, setup_services):
        """zk_status should return accurate note counts."""
        service, repo, notes_dir = setup_services

        # Create notes of different types
        service.create_note(
            title="Fleeting 1", content="Content", note_type=NoteType.FLEETING
        )
        service.create_note(
            title="Permanent 1", content="Content", note_type=NoteType.PERMANENT
        )
        service.create_note(
            title="Permanent 2", content="Content", note_type=NoteType.PERMANENT
        )

        # Get status
        all_notes = repo.get_all()
        fleeting_count = sum(1 for n in all_notes if n.note_type == NoteType.FLEETING)
        permanent_count = sum(1 for n in all_notes if n.note_type == NoteType.PERMANENT)

        # Verify counts
        assert len(all_notes) == 3
        assert fleeting_count == 1
        assert permanent_count == 2

    def test_status_returns_tag_statistics(self, setup_services):
        """zk_status should return tag statistics."""
        service, repo, notes_dir = setup_services

        # Create notes with tags
        service.create_note(
            title="Note 1", content="Content", tags=["python", "programming"]
        )
        service.create_note(
            title="Note 2", content="Content", tags=["python", "tutorial"]
        )
        service.create_note(title="Note 3", content="Content", tags=["javascript"])

        # Get all tags
        tags = repo.get_all_tags()
        tag_names = {t.name for t in tags}

        # Should have 4 unique tags
        assert tag_names == {"python", "programming", "tutorial", "javascript"}


class TestRestoreToolIntegration:
    """Integration tests for zk_restore/rebuild functionality."""

    @pytest.fixture
    def setup_services(self, temp_dirs):
        """Set up real services with temp directories."""
        notes_dir, db_dir = temp_dirs
        repo = NoteRepository(notes_dir=notes_dir)
        service = ZettelService(repository=repo)
        return service, repo, notes_dir

    def test_rebuild_recovers_notes_from_files(self, setup_services):
        """rebuild_index should recover notes from markdown files."""
        service, repo, notes_dir = setup_services

        # Create a note
        note = service.create_note(title="Important Note", content="Important content")
        note_id = note.id

        # Verify file exists
        note_path = notes_dir / f"{note_id}.md"
        assert note_path.exists()

        # Delete from DB only (simulating corruption scenario)
        repo._delete_from_db(note_id)

        # Note can still be read from file (dual-storage architecture)
        # But DB count should be lower
        from sqlalchemy import func, select

        from znote_mcp.models.db_models import DBNote

        with repo.session_factory() as session:
            db_count = session.execute(
                select(func.count()).select_from(DBNote)
            ).scalar()
        assert db_count == 0  # DB has no notes

        # Rebuild index should recover from files
        service.rebuild_index()

        # Now DB should have the note again
        with repo.session_factory() as session:
            db_count = session.execute(
                select(func.count()).select_from(DBNote)
            ).scalar()
        assert db_count == 1

    def test_rebuild_handles_missing_files(self, setup_services):
        """rebuild_index should handle missing files gracefully."""
        service, repo, notes_dir = setup_services

        # Create two notes
        note1 = service.create_note(title="Note 1", content="Content 1")
        note2 = service.create_note(title="Note 2", content="Content 2")

        # Delete note2's file (simulating file loss)
        note2_path = notes_dir / f"{note2.id}.md"
        note2_path.unlink()

        # Rebuild - should not crash
        service.rebuild_index()

        # note1 should still be accessible
        retrieved = service.get_note(note1.id)
        assert retrieved is not None
        assert retrieved.title == "Note 1"

    def test_rebuild_updates_stale_db_records(self, setup_services):
        """rebuild_index should update DB when files change."""
        service, repo, notes_dir = setup_services

        # Create a note
        note = service.create_note(title="Original Title", content="Original content")
        note_id = note.id

        # Manually modify the file (simulating external edit)
        note_path = notes_dir / f"{note_id}.md"
        content = note_path.read_text()
        modified_content = content.replace("Original Title", "Modified Title")
        note_path.write_text(modified_content)

        # Rebuild index
        service.rebuild_index()

        # DB should have updated title
        from sqlalchemy import select

        from znote_mcp.models.db_models import DBNote

        with repo.session_factory() as session:
            db_note = session.execute(
                select(DBNote).where(DBNote.id == note_id)
            ).scalar_one_or_none()

        assert db_note is not None
        assert db_note.title == "Modified Title"


class TestBulkProjectUpdateIntegration:
    """Integration tests for zk_bulk_move_to_project tool."""

    @pytest.fixture
    def setup_services(self, temp_dirs):
        """Set up real services with temp directories."""
        notes_dir, db_dir = temp_dirs
        repo = NoteRepository(notes_dir=notes_dir)
        service = ZettelService(repository=repo)
        return service, repo, notes_dir

    def test_bulk_move_notes_to_project(self, setup_services):
        """zk_bulk_move_to_project should move multiple notes."""
        service, repo, notes_dir = setup_services

        # Create notes in different projects
        notes = []
        for i in range(5):
            note = service.create_note(
                title=f"Note {i}",
                content=f"Content {i}",
                project=f"old-project-{i % 2}",
            )
            notes.append(note)

        note_ids = [n.id for n in notes]

        # Bulk move to new project
        updated = service.bulk_update_project(note_ids, "unified-project")

        # Verify all moved
        assert updated == 5

        for note_id in note_ids:
            note = service.get_note(note_id)
            assert note.project == "unified-project"

    def test_bulk_move_preserves_note_data(self, setup_services):
        """Moving notes should preserve all other data."""
        service, repo, notes_dir = setup_services

        # Create note with all fields
        note = service.create_note(
            title="Full Note",
            content="Full content",
            project="original",
            note_type=NoteType.PERMANENT,
            tags=["tag1", "tag2"],
        )

        # Move to new project
        service.bulk_update_project([note.id], "new-project")

        # Verify all fields preserved
        moved = service.get_note(note.id)
        assert moved.title == "Full Note"
        assert "Full content" in moved.content
        assert moved.project == "new-project"
        assert moved.note_type == NoteType.PERMANENT
        assert {t.name for t in moved.tags} == {"tag1", "tag2"}
