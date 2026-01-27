"""Comprehensive tests for ProjectRepository.

This test file addresses the 0% test coverage gap identified
in the code review for project management functionality.
"""
import json
import tempfile
from pathlib import Path

import pytest
from sqlalchemy import create_engine

from znote_mcp.config import config
from znote_mcp.models.db_models import Base, DBProject
from znote_mcp.models.schema import Project
from znote_mcp.storage.project_repository import ProjectRepository, escape_like_pattern
from znote_mcp.exceptions import ValidationError, ErrorCode


@pytest.fixture
def project_repository(temp_dirs):
    """Create a test project repository."""
    notes_dir, db_dir = temp_dirs
    database_path = db_dir / "test_projects.db"

    # Update config for tests
    original_notes_dir = config.notes_dir
    original_database_path = config.database_path
    config.notes_dir = notes_dir
    config.database_path = database_path

    # Create engine and tables
    engine = create_engine(f"sqlite:///{database_path}")
    Base.metadata.create_all(engine)

    # Create repository
    repo = ProjectRepository(engine=engine)

    yield repo

    # Cleanup
    engine.dispose()
    config.notes_dir = original_notes_dir
    config.database_path = original_database_path


class TestProjectCreate:
    """Tests for ProjectRepository.create()."""

    def test_create_project_basic(self, project_repository):
        """Create a basic project."""
        project = Project(
            id="test-project",
            name="Test Project",
            description="A test project"
        )

        created = project_repository.create(project)

        assert created.id == "test-project"
        assert created.name == "Test Project"
        assert created.description == "A test project"

    def test_create_project_with_path(self, project_repository):
        """Create a project with filesystem path."""
        project = Project(
            id="repo-project",
            name="Repository Project",
            path="/home/user/repos/my-project"
        )

        created = project_repository.create(project)

        assert created.path == "/home/user/repos/my-project"

    def test_create_project_with_metadata(self, project_repository):
        """Create a project with custom metadata."""
        project = Project(
            id="meta-project",
            name="Metadata Project",
            metadata={"git_remote": "git@github.com:user/repo.git", "language": "python"}
        )

        created = project_repository.create(project)

        assert created.metadata["git_remote"] == "git@github.com:user/repo.git"
        assert created.metadata["language"] == "python"

    def test_create_project_with_parent(self, project_repository):
        """Create a sub-project with parent."""
        # First create parent
        parent = Project(id="parent", name="Parent Project")
        project_repository.create(parent)

        # Create child
        child = Project(
            id="parent/child",
            name="Child Project",
            parent_id="parent"
        )
        created = project_repository.create(child)

        assert created.parent_id == "parent"

    def test_create_duplicate_project_fails(self, project_repository):
        """Creating a project with existing ID should fail."""
        project = Project(id="duplicate", name="First")
        project_repository.create(project)

        duplicate = Project(id="duplicate", name="Second")
        with pytest.raises(ValidationError) as exc_info:
            project_repository.create(duplicate)

        assert exc_info.value.code == ErrorCode.VALIDATION_FAILED
        assert "already exists" in str(exc_info.value)

    def test_create_with_nonexistent_parent_fails(self, project_repository):
        """Creating a project with nonexistent parent should fail."""
        project = Project(
            id="orphan",
            name="Orphan Project",
            parent_id="nonexistent"
        )

        with pytest.raises(ValidationError) as exc_info:
            project_repository.create(project)

        assert "Parent project" in str(exc_info.value)
        assert "not found" in str(exc_info.value)

    def test_create_self_referencing_project_fails(self, project_repository):
        """Creating a project that is its own parent should fail."""
        project = Project(
            id="self-ref",
            name="Self Reference",
            parent_id="self-ref"
        )

        with pytest.raises(ValidationError) as exc_info:
            project_repository.create(project)

        assert "cannot be its own parent" in str(exc_info.value)


class TestProjectGet:
    """Tests for ProjectRepository.get()."""

    def test_get_existing_project(self, project_repository):
        """Get an existing project by ID."""
        project = Project(id="get-test", name="Get Test")
        project_repository.create(project)

        retrieved = project_repository.get("get-test")

        assert retrieved is not None
        assert retrieved.id == "get-test"
        assert retrieved.name == "Get Test"

    def test_get_nonexistent_project(self, project_repository):
        """Get a nonexistent project returns None."""
        result = project_repository.get("nonexistent")

        assert result is None

    def test_get_preserves_metadata(self, project_repository):
        """Get should preserve all metadata fields."""
        project = Project(
            id="full-project",
            name="Full Project",
            description="Description here",
            path="/some/path",
            metadata={"key": "value"}
        )
        project_repository.create(project)

        retrieved = project_repository.get("full-project")

        assert retrieved.description == "Description here"
        assert retrieved.path == "/some/path"
        assert retrieved.metadata == {"key": "value"}


class TestProjectGetAll:
    """Tests for ProjectRepository.get_all()."""

    def test_get_all_empty(self, project_repository):
        """get_all on empty repository returns empty list."""
        result = project_repository.get_all()

        assert result == []

    def test_get_all_returns_all_projects(self, project_repository):
        """get_all returns all created projects."""
        project_repository.create(Project(id="project-a", name="A"))
        project_repository.create(Project(id="project-b", name="B"))
        project_repository.create(Project(id="project-c", name="C"))

        result = project_repository.get_all()

        assert len(result) == 3
        ids = {p.id for p in result}
        assert ids == {"project-a", "project-b", "project-c"}

    def test_get_all_ordered_by_id(self, project_repository):
        """get_all returns projects ordered by ID."""
        project_repository.create(Project(id="zebra", name="Z"))
        project_repository.create(Project(id="alpha", name="A"))
        project_repository.create(Project(id="middle", name="M"))

        result = project_repository.get_all()

        ids = [p.id for p in result]
        assert ids == ["alpha", "middle", "zebra"]


class TestProjectUpdate:
    """Tests for ProjectRepository.update()."""

    def test_update_project_name(self, project_repository):
        """Update project name."""
        project = Project(id="update-test", name="Original")
        project_repository.create(project)

        project.name = "Updated"
        updated = project_repository.update(project)

        assert updated.name == "Updated"
        retrieved = project_repository.get("update-test")
        assert retrieved.name == "Updated"

    def test_update_project_description(self, project_repository):
        """Update project description."""
        project = Project(id="desc-test", name="Test", description="Original")
        project_repository.create(project)

        project.description = "Updated description"
        project_repository.update(project)

        retrieved = project_repository.get("desc-test")
        assert retrieved.description == "Updated description"

    def test_update_project_metadata(self, project_repository):
        """Update project metadata."""
        project = Project(id="meta-test", name="Test", metadata={"old": "value"})
        project_repository.create(project)

        project.metadata = {"new": "data", "count": 42}
        project_repository.update(project)

        retrieved = project_repository.get("meta-test")
        assert retrieved.metadata == {"new": "data", "count": 42}

    def test_update_nonexistent_project_fails(self, project_repository):
        """Updating nonexistent project should fail."""
        project = Project(id="nonexistent", name="Ghost")

        with pytest.raises(ValidationError) as exc_info:
            project_repository.update(project)

        assert exc_info.value.code == ErrorCode.PROJECT_NOT_FOUND

    def test_update_parent_id(self, project_repository):
        """Update project parent."""
        # Create parent projects
        project_repository.create(Project(id="parent-a", name="Parent A"))
        project_repository.create(Project(id="parent-b", name="Parent B"))

        # Create child under parent-a
        child = Project(id="child", name="Child", parent_id="parent-a")
        project_repository.create(child)

        # Move to parent-b
        child.parent_id = "parent-b"
        project_repository.update(child)

        retrieved = project_repository.get("child")
        assert retrieved.parent_id == "parent-b"

    def test_update_to_circular_reference_fails(self, project_repository):
        """Setting parent to create circular reference should fail."""
        # Create A -> B -> C hierarchy
        project_repository.create(Project(id="a", name="A"))
        project_repository.create(Project(id="b", name="B", parent_id="a"))
        project_repository.create(Project(id="c", name="C", parent_id="b"))

        # Try to make A a child of C (would create cycle: A -> B -> C -> A)
        project_a = project_repository.get("a")
        project_a.parent_id = "c"

        with pytest.raises(ValidationError) as exc_info:
            project_repository.update(project_a)

        assert "circular reference" in str(exc_info.value).lower()

    def test_update_self_reference_fails(self, project_repository):
        """Setting project as its own parent should fail."""
        project = Project(id="self", name="Self")
        project_repository.create(project)

        project.parent_id = "self"

        with pytest.raises(ValidationError) as exc_info:
            project_repository.update(project)

        assert "cannot be its own parent" in str(exc_info.value)


class TestProjectDelete:
    """Tests for ProjectRepository.delete()."""

    def test_delete_project(self, project_repository):
        """Delete an existing project."""
        project = Project(id="delete-me", name="Delete Me")
        project_repository.create(project)

        project_repository.delete("delete-me")

        assert project_repository.get("delete-me") is None

    def test_delete_nonexistent_project_fails(self, project_repository):
        """Deleting nonexistent project should fail."""
        with pytest.raises(ValidationError) as exc_info:
            project_repository.delete("nonexistent")

        assert exc_info.value.code == ErrorCode.PROJECT_NOT_FOUND

    def test_delete_project_with_children_fails(self, project_repository):
        """Deleting project with child projects should fail."""
        project_repository.create(Project(id="parent", name="Parent"))
        project_repository.create(Project(id="child", name="Child", parent_id="parent"))

        with pytest.raises(ValidationError) as exc_info:
            project_repository.delete("parent")

        assert "has child projects" in str(exc_info.value)


class TestProjectSearch:
    """Tests for ProjectRepository.search()."""

    def test_search_by_name(self, project_repository):
        """Search projects by name."""
        project_repository.create(Project(id="py-proj", name="Python Project"))
        project_repository.create(Project(id="go-proj", name="Go Project"))
        project_repository.create(Project(id="py-lib", name="Python Library"))

        results = project_repository.search(name="Python")

        assert len(results) == 2
        ids = {p.id for p in results}
        assert ids == {"py-proj", "py-lib"}

    def test_search_by_parent_id(self, project_repository):
        """Search projects by parent."""
        project_repository.create(Project(id="parent", name="Parent"))
        project_repository.create(Project(id="child-a", name="A", parent_id="parent"))
        project_repository.create(Project(id="child-b", name="B", parent_id="parent"))
        project_repository.create(Project(id="orphan", name="Orphan"))

        results = project_repository.search(parent_id="parent")

        assert len(results) == 2
        ids = {p.id for p in results}
        assert ids == {"child-a", "child-b"}

    def test_search_case_insensitive(self, project_repository):
        """Search should be case-insensitive."""
        project_repository.create(Project(id="mixed", name="MixedCase"))

        results = project_repository.search(name="mixedcase")

        assert len(results) == 1
        assert results[0].id == "mixed"

    def test_search_escapes_wildcards(self, project_repository):
        """Search should escape LIKE wildcards."""
        project_repository.create(Project(id="percent", name="100% Complete"))
        project_repository.create(Project(id="other", name="Other Project"))

        # Search for literal "100%" should not match everything
        results = project_repository.search(name="100%")

        assert len(results) == 1
        assert results[0].id == "percent"


class TestProjectHierarchy:
    """Tests for ProjectRepository.get_hierarchy()."""

    def test_get_hierarchy_empty(self, project_repository):
        """get_hierarchy on empty repository returns empty list."""
        result = project_repository.get_hierarchy()

        assert result == []

    def test_get_hierarchy_flat(self, project_repository):
        """get_hierarchy with no parent-child relationships."""
        project_repository.create(Project(id="a", name="A"))
        project_repository.create(Project(id="b", name="B"))

        result = project_repository.get_hierarchy()

        assert len(result) == 2

    def test_get_hierarchy_nested(self, project_repository):
        """get_hierarchy returns proper tree structure."""
        # Create root
        project_repository.create(Project(id="root", name="Root"))
        # Create children
        project_repository.create(Project(id="child1", name="Child 1", parent_id="root"))
        project_repository.create(Project(id="child2", name="Child 2", parent_id="root"))
        # Create grandchild
        project_repository.create(Project(id="grandchild", name="Grandchild", parent_id="child1"))

        result = project_repository.get_hierarchy()

        # Should have one root
        assert len(result) == 1
        root = result[0]
        assert root["project"].id == "root"

        # Root should have 2 children
        assert len(root["children"]) == 2

        # Find child1 and verify it has grandchild
        child1 = next(c for c in root["children"] if c["project"].id == "child1")
        assert len(child1["children"]) == 1
        assert child1["children"][0]["project"].id == "grandchild"


class TestProjectExists:
    """Tests for ProjectRepository.exists()."""

    def test_exists_returns_true_for_existing(self, project_repository):
        """exists() returns True for existing project."""
        project_repository.create(Project(id="exists", name="Exists"))

        assert project_repository.exists("exists") is True

    def test_exists_returns_false_for_nonexistent(self, project_repository):
        """exists() returns False for nonexistent project."""
        assert project_repository.exists("nonexistent") is False


class TestProjectNoteCount:
    """Tests for ProjectRepository.get_note_count()."""

    def test_get_note_count_empty(self, project_repository, note_repository):
        """get_note_count returns 0 for project with no notes."""
        project_repository.create(Project(id="empty-proj", name="Empty"))

        count = project_repository.get_note_count("empty-proj")

        assert count == 0


class TestProjectExportImport:
    """Tests for ProjectRepository export/import to JSON."""

    def test_export_to_json(self, project_repository, temp_dirs):
        """export_to_json creates valid JSON file."""
        notes_dir, db_dir = temp_dirs

        project_repository.create(Project(id="export-a", name="A"))
        project_repository.create(Project(id="export-b", name="B", description="Desc"))

        # Create the .znote directory first
        znote_dir = notes_dir / ".znote"
        znote_dir.mkdir(parents=True, exist_ok=True)

        export_path = znote_dir / "projects.json"
        result_path = project_repository.export_to_json(export_path)

        assert result_path.exists()

        with open(result_path) as f:
            data = json.load(f)

        assert "projects" in data
        assert len(data["projects"]) == 2

    def test_import_from_json(self, project_repository, temp_dirs):
        """import_from_json loads projects from JSON file."""
        notes_dir, db_dir = temp_dirs
        json_path = notes_dir / "import.json"

        # Create JSON file
        data = {
            "projects": [
                {"id": "import-a", "name": "Import A"},
                {"id": "import-b", "name": "Import B", "parent_id": "import-a"}
            ]
        }
        with open(json_path, "w") as f:
            json.dump(data, f)

        count = project_repository.import_from_json(json_path)

        assert count == 2
        assert project_repository.exists("import-a")
        assert project_repository.exists("import-b")

    def test_import_skips_existing_projects(self, project_repository, temp_dirs):
        """import_from_json skips projects that already exist."""
        notes_dir, db_dir = temp_dirs
        json_path = notes_dir / "import.json"

        # Create existing project
        project_repository.create(Project(id="existing", name="Existing"))

        # Create JSON with same ID
        data = {
            "projects": [
                {"id": "existing", "name": "Different Name"},
                {"id": "new", "name": "New Project"}
            ]
        }
        with open(json_path, "w") as f:
            json.dump(data, f)

        count = project_repository.import_from_json(json_path)

        assert count == 1  # Only new project imported
        assert project_repository.get("existing").name == "Existing"  # Not overwritten


class TestEscapeLikePattern:
    """Tests for the escape_like_pattern utility."""

    def test_escape_percent(self):
        """% is escaped."""
        assert escape_like_pattern("50%") == "50\\%"

    def test_escape_underscore(self):
        """_ is escaped."""
        assert escape_like_pattern("a_b") == "a\\_b"

    def test_escape_backslash(self):
        """Backslash is escaped."""
        assert escape_like_pattern("a\\b") == "a\\\\b"
