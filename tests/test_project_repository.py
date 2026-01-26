"""Tests for the ProjectRepository class."""
import json
import pytest
import tempfile
from pathlib import Path

from znote_mcp.models.schema import Project
from znote_mcp.storage.project_repository import ProjectRepository
from znote_mcp.models.db_models import init_db, Base
from znote_mcp.config import config
from sqlalchemy import create_engine


@pytest.fixture
def project_repository(test_config):
    """Create a test project repository."""
    # Create tables
    database_path = test_config.get_absolute_path(test_config.database_path)
    engine = create_engine(f"sqlite:///{database_path}")
    Base.metadata.create_all(engine)

    # Initialize with proper DB setup
    from znote_mcp.models.db_models import init_db
    engine = init_db()

    repository = ProjectRepository(engine=engine)
    yield repository


class TestProjectRepository:
    """Tests for ProjectRepository CRUD operations."""

    def test_create_project(self, project_repository):
        """Test creating a new project."""
        project = Project(
            id="test-project",
            name="Test Project",
            description="A test project"
        )
        created = project_repository.create(project)

        assert created.id == "test-project"
        assert created.name == "Test Project"
        assert created.description == "A test project"

    def test_create_duplicate_project_fails(self, project_repository):
        """Test that creating a duplicate project raises an error."""
        project = Project(id="dup-project", name="Duplicate")
        project_repository.create(project)

        from znote_mcp.exceptions import ValidationError
        with pytest.raises(ValidationError):
            project_repository.create(project)

    def test_get_project(self, project_repository):
        """Test retrieving a project by ID."""
        project = Project(id="get-test", name="Get Test")
        project_repository.create(project)

        retrieved = project_repository.get("get-test")
        assert retrieved is not None
        assert retrieved.id == "get-test"
        assert retrieved.name == "Get Test"

    def test_get_nonexistent_project(self, project_repository):
        """Test that getting a nonexistent project returns None."""
        result = project_repository.get("nonexistent")
        assert result is None

    def test_get_all_projects(self, project_repository):
        """Test retrieving all projects."""
        project_repository.create(Project(id="proj-a", name="Project A"))
        project_repository.create(Project(id="proj-b", name="Project B"))
        project_repository.create(Project(id="proj-c", name="Project C"))

        all_projects = project_repository.get_all()
        assert len(all_projects) == 3
        ids = [p.id for p in all_projects]
        assert "proj-a" in ids
        assert "proj-b" in ids
        assert "proj-c" in ids

    def test_update_project(self, project_repository):
        """Test updating a project."""
        project = Project(id="update-test", name="Original Name")
        project_repository.create(project)

        project.name = "Updated Name"
        project.description = "New description"
        project_repository.update(project)

        retrieved = project_repository.get("update-test")
        assert retrieved.name == "Updated Name"
        assert retrieved.description == "New description"

    def test_delete_project(self, project_repository):
        """Test deleting a project."""
        project = Project(id="delete-test", name="Delete Me")
        project_repository.create(project)

        assert project_repository.exists("delete-test")
        project_repository.delete("delete-test")
        assert not project_repository.exists("delete-test")

    def test_delete_project_with_children_fails(self, project_repository):
        """Test that deleting a project with children fails."""
        parent = Project(id="parent", name="Parent")
        child = Project(id="child", name="Child", parent_id="parent")

        project_repository.create(parent)
        project_repository.create(child)

        from znote_mcp.exceptions import ValidationError
        with pytest.raises(ValidationError, match="child projects"):
            project_repository.delete("parent")

    def test_subproject_hierarchy(self, project_repository):
        """Test creating sub-projects with hierarchy."""
        root = Project(id="monorepo", name="Monorepo Root")
        frontend = Project(
            id="monorepo/frontend",
            name="Frontend App",
            parent_id="monorepo"
        )
        backend = Project(
            id="monorepo/backend",
            name="Backend API",
            parent_id="monorepo"
        )

        project_repository.create(root)
        project_repository.create(frontend)
        project_repository.create(backend)

        # Verify hierarchy
        hierarchy = project_repository.get_hierarchy()
        assert len(hierarchy) == 1  # One root
        assert hierarchy[0]["project"].id == "monorepo"
        assert len(hierarchy[0]["children"]) == 2

    def test_project_exists(self, project_repository):
        """Test checking if a project exists."""
        assert not project_repository.exists("check-test")

        project_repository.create(Project(id="check-test", name="Check"))
        assert project_repository.exists("check-test")

    def test_search_by_name(self, project_repository):
        """Test searching projects by name."""
        project_repository.create(Project(id="search-1", name="Frontend App"))
        project_repository.create(Project(id="search-2", name="Backend API"))
        project_repository.create(Project(id="search-3", name="Frontend Tests"))

        results = project_repository.search(name="Frontend")
        assert len(results) == 2
        ids = [p.id for p in results]
        assert "search-1" in ids
        assert "search-3" in ids

    def test_project_with_metadata(self, project_repository):
        """Test project with custom metadata."""
        project = Project(
            id="meta-test",
            name="Metadata Test",
            metadata={"git_remote": "https://github.com/user/repo.git", "language": "python"}
        )
        project_repository.create(project)

        retrieved = project_repository.get("meta-test")
        assert retrieved.metadata["git_remote"] == "https://github.com/user/repo.git"
        assert retrieved.metadata["language"] == "python"


class TestProjectExportImport:
    """Tests for project export/import to JSON."""

    def test_export_to_json(self, project_repository, tmp_path):
        """Test exporting projects to JSON file."""
        project_repository.create(Project(id="export-1", name="Export One"))
        project_repository.create(Project(id="export-2", name="Export Two"))

        output_path = tmp_path / "projects.json"
        result_path = project_repository.export_to_json(output_path)

        assert result_path.exists()
        with open(result_path) as f:
            data = json.load(f)

        assert len(data["projects"]) == 2
        ids = [p["id"] for p in data["projects"]]
        assert "export-1" in ids
        assert "export-2" in ids

    def test_import_from_json(self, project_repository, tmp_path):
        """Test importing projects from JSON file."""
        # Create JSON file
        data = {
            "root_project": "imported",
            "projects": [
                {"id": "imported", "name": "Imported Project", "description": "From JSON"},
                {"id": "imported/sub", "name": "Sub Project", "parent_id": "imported"}
            ]
        }
        json_path = tmp_path / "projects.json"
        with open(json_path, "w") as f:
            json.dump(data, f)

        # Import
        count = project_repository.import_from_json(json_path)
        assert count == 2

        # Verify
        assert project_repository.exists("imported")
        assert project_repository.exists("imported/sub")

        sub = project_repository.get("imported/sub")
        assert sub.parent_id == "imported"

    def test_import_skips_existing(self, project_repository, tmp_path):
        """Test that import skips already existing projects."""
        # Create existing project
        project_repository.create(Project(id="existing", name="Existing"))

        # Create JSON with same ID
        data = {
            "projects": [
                {"id": "existing", "name": "Should Skip"},
                {"id": "new-one", "name": "Should Import"}
            ]
        }
        json_path = tmp_path / "projects.json"
        with open(json_path, "w") as f:
            json.dump(data, f)

        # Import
        count = project_repository.import_from_json(json_path)
        assert count == 1  # Only new-one imported

        # Verify existing wasn't overwritten
        existing = project_repository.get("existing")
        assert existing.name == "Existing"  # Original name preserved
