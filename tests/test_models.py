# tests/test_models.py
"""Tests for the data models used in the Zettelkasten MCP server."""
import datetime
import time
import re
import pytest
from unittest.mock import patch
from pydantic import ValidationError
from znote_mcp.models.schema import (
    Link, LinkType, Note, NoteType, NotePurpose, Tag, Project,
    generate_id, validate_project_path
)

class TestNoteModel:
    """Tests for the Note model."""
    def test_note_creation(self):
        """Test creating a note with valid values."""
        note = Note(
            title="Test Note",
            content="This is a test note.",
            note_type=NoteType.PERMANENT,
            tags=[Tag(name="test"), Tag(name="example")]
        )
        assert note.id is not None
        assert note.title == "Test Note"
        assert note.content == "This is a test note."
        assert note.note_type == NoteType.PERMANENT
        assert len(note.tags) == 2
        assert note.links == []
        assert isinstance(note.created_at, datetime.datetime)
        assert isinstance(note.updated_at, datetime.datetime)

    def test_note_validation(self):
        """Test note validation for required fields."""
        # Empty title
        with pytest.raises(ValidationError):
            Note(title="", content="Content")
        # Title with only whitespace
        with pytest.raises(ValidationError):
            Note(title="   ", content="Content")
        # Without content - should fail
        with pytest.raises(ValidationError):
            Note(title="Title")

    def test_note_tag_operations(self):
        """Test adding and removing tags."""
        note = Note(
            title="Tag Test",
            content="Testing tag operations.",
            tags=[Tag(name="initial")]
        )
        assert len(note.tags) == 1
        # Add tag as string
        note.add_tag("test")
        assert len(note.tags) == 2
        assert any(tag.name == "test" for tag in note.tags)
        # Add tag as Tag object
        note.add_tag(Tag(name="another"))
        assert len(note.tags) == 3
        assert any(tag.name == "another" for tag in note.tags)
        # Add duplicate tag (should be ignored)
        note.add_tag("test")
        assert len(note.tags) == 3
        # Remove tag
        note.remove_tag("test")
        assert len(note.tags) == 2
        assert all(tag.name != "test" for tag in note.tags)
        # Remove tag that doesn't exist (should not error)
        note.remove_tag("nonexistent")
        assert len(note.tags) == 2

    def test_note_link_operations(self):
        """Test adding and removing links."""
        note = Note(
            title="Link Test",
            content="Testing link operations.",
            id="source123"
        )
        # Add link
        note.add_link("target456", LinkType.REFERENCE, "Test link")
        assert len(note.links) == 1
        assert note.links[0].source_id == "source123"
        assert note.links[0].target_id == "target456"
        assert note.links[0].link_type == LinkType.REFERENCE
        assert note.links[0].description == "Test link"
        # Add duplicate link (should be ignored)
        note.add_link("target456", LinkType.REFERENCE)
        assert len(note.links) == 1
        # Add link with different type
        note.add_link("target456", LinkType.EXTENDS)
        assert len(note.links) == 2
        # Remove specific link type
        note.remove_link("target456", LinkType.REFERENCE)
        assert len(note.links) == 1
        assert note.links[0].link_type == LinkType.EXTENDS
        # Remove all links to target
        note.remove_link("target456")
        assert len(note.links) == 0

    def test_note_to_markdown(self):
        """Test converting a note to markdown format."""
        note = Note(
            id="202501010000",
            title="Markdown Test",
            content="Testing markdown conversion.",
            note_type=NoteType.PERMANENT,
            tags=[Tag(name="test"), Tag(name="markdown")]
        )
        note.add_link("target123", LinkType.REFERENCE, "Reference link")
        markdown = note.to_markdown()
        # Check basic structure
        assert "# Markdown Test" in markdown
        assert "Testing markdown conversion." in markdown
        assert "test" in markdown
        assert "markdown" in markdown
        assert "Reference link" in markdown
        assert "target123" in markdown


class TestLinkModel:
    """Tests for the Link model."""
    def test_link_creation(self):
        """Test creating a link with valid values."""
        link = Link(
            source_id="source123",
            target_id="target456",
            link_type=LinkType.REFERENCE,
            description="Test description"
        )
        assert link.source_id == "source123"
        assert link.target_id == "target456"
        assert link.link_type == LinkType.REFERENCE
        assert link.description == "Test description"
        assert isinstance(link.created_at, datetime.datetime)

    def test_link_validation(self):
        """Test link validation for required fields."""
        # Missing source_id
        with pytest.raises(ValidationError):
            Link(target_id="target456")
        # Missing target_id
        with pytest.raises(ValidationError):
            Link(source_id="source123")
        # Invalid link_type
        with pytest.raises(ValidationError):
            Link(source_id="source123", target_id="target456", link_type="invalid")

    def test_link_immutability(self):
        """Test that Link objects are immutable."""
        link = Link(
            source_id="source123", 
            target_id="target456"
        )
        # Attempt to modify link should fail
        with pytest.raises(Exception):
            link.source_id = "newsource"


class TestTagModel:
    """Tests for the Tag model."""
    def test_tag_creation(self):
        """Test creating a tag with valid values."""
        tag = Tag(name="test")
        assert tag.name == "test"
        assert str(tag) == "test"

    def test_tag_immutability(self):
        """Test that Tag objects are immutable."""
        tag = Tag(name="test")
        # Attempt to modify tag should fail
        with pytest.raises(Exception):
            tag.name = "newname"


class TestHelperFunctions:
    """Tests for helper functions in the schema module."""

    def test_iso8601_id_format(self):
        """Test that generated IDs follow the correct ISO 8601 format with microsecond + counter.

        Format: YYYYMMDDTHHMMSSsssssscccccc (27 characters)
        - YYYYMMDD: 8-digit date
        - T: ISO 8601 separator
        - HHMMSS: 6-digit time
        - ssssss: 6-digit microseconds
        - cccccc: 6-digit counter for uniqueness within same microsecond
        """
        # Generate an ID
        id_str = generate_id()

        # Verify it matches the expected format: YYYYMMDDTHHMMSSsssssscccccc (27 chars)
        pattern = r'^\d{8}T\d{6}\d{12}$'  # 8 + T + 6 + 12 = 27 chars
        assert re.match(pattern, id_str), f"ID {id_str} does not match expected ISO 8601 format"

        # Verify the parts make sense
        date_part = id_str[:8]
        separator = id_str[8]
        time_part = id_str[9:15]
        microseconds_part = id_str[15:21]
        counter_part = id_str[21:]

        assert len(date_part) == 8, "Date part should be 8 digits (YYYYMMDD)"
        assert separator == 'T', "Date/time separator should be 'T' per ISO 8601"
        assert len(time_part) == 6, "Time part should be 6 digits (HHMMSS)"
        assert len(microseconds_part) == 6, "Microseconds part should be 6 digits"
        assert len(counter_part) == 6, "Counter part should be 6 digits"
        assert len(id_str) == 27, f"Total ID length should be 27 chars, got {len(id_str)}"

    def test_iso8601_uniqueness(self):
        """Test that ISO 8601 IDs with nanosecond precision are unique even in rapid succession."""
        # Generate multiple IDs as quickly as possible
        ids = [generate_id() for _ in range(1000)]
        
        # Verify they are all unique
        unique_ids = set(ids)
        assert len(unique_ids) == 1000, "Generated IDs should all be unique"

    def test_iso8601_chronological_sorting(self):
        """Test that ISO 8601 IDs sort chronologically without artificial delays."""
        # Generate multiple IDs in the fastest possible succession
        ids = [generate_id() for _ in range(5)]

        # Verify they're all unique
        assert len(set(ids)) == 5

        # Verify chronological order matches lexicographical sorting
        sorted_ids = sorted(ids)
        assert sorted_ids == ids, "ISO 8601 IDs should sort chronologically"


class TestNotePurpose:
    """Tests for the NotePurpose enum and related Note fields."""

    def test_note_purpose_enum_values(self):
        """Test that NotePurpose has expected values."""
        assert NotePurpose.RESEARCH.value == "research"
        assert NotePurpose.PLANNING.value == "planning"
        assert NotePurpose.BUGFIXING.value == "bugfixing"
        assert NotePurpose.GENERAL.value == "general"

    def test_note_with_purpose(self):
        """Test creating a note with note_purpose field."""
        note = Note(
            title="Planning Note",
            content="This is a planning note.",
            note_purpose=NotePurpose.PLANNING
        )
        assert note.note_purpose == NotePurpose.PLANNING
        assert note.note_purpose.value == "planning"

    def test_note_purpose_default(self):
        """Test that note_purpose defaults to GENERAL."""
        note = Note(title="Default Note", content="Content")
        assert note.note_purpose == NotePurpose.GENERAL

    def test_note_with_plan_id(self):
        """Test creating a note with plan_id field."""
        note = Note(
            title="Plan Revision",
            content="Updated plan details.",
            note_purpose=NotePurpose.PLANNING,
            plan_id="plan-20240126-auth"
        )
        assert note.plan_id == "plan-20240126-auth"

    def test_plan_id_validation_rejects_path_traversal(self):
        """Test that plan_id rejects path traversal attempts."""
        with pytest.raises(ValidationError):
            Note(title="Bad Plan", content="Content", plan_id="../escape")

    def test_plan_id_default_none(self):
        """Test that plan_id defaults to None."""
        note = Note(title="No Plan", content="Content")
        assert note.plan_id is None


class TestProjectPaths:
    """Tests for project path validation including sub-projects."""

    def test_simple_project_name(self):
        """Test note with simple project name."""
        note = Note(title="Test", content="Content", project="my-project")
        assert note.project == "my-project"

    def test_subproject_path(self):
        """Test note with sub-project path using slash separator."""
        note = Note(title="Test", content="Content", project="monorepo/frontend")
        assert note.project == "monorepo/frontend"

    def test_deep_subproject_path(self):
        """Test note with deeply nested sub-project path."""
        note = Note(title="Test", content="Content", project="org/team/project/module")
        assert note.project == "org/team/project/module"

    def test_project_defaults_to_general(self):
        """Test that empty project defaults to 'general'."""
        note = Note(title="Test", content="Content", project="")
        assert note.project == "general"

        note2 = Note(title="Test", content="Content", project="   ")
        assert note2.project == "general"

    def test_project_rejects_path_traversal(self):
        """Test that project rejects path traversal attempts."""
        with pytest.raises(ValidationError):
            Note(title="Test", content="Content", project="../escape")

        with pytest.raises(ValidationError):
            Note(title="Test", content="Content", project="a/../b")

    def test_project_rejects_backslash(self):
        """Test that project rejects Windows-style backslashes."""
        with pytest.raises(ValidationError):
            Note(title="Test", content="Content", project="a\\b")

    def test_project_rejects_leading_slash(self):
        """Test that project rejects leading slashes."""
        with pytest.raises(ValidationError):
            Note(title="Test", content="Content", project="/absolute")

    def test_project_rejects_trailing_slash(self):
        """Test that project rejects trailing slashes."""
        with pytest.raises(ValidationError):
            Note(title="Test", content="Content", project="project/")

    def test_project_rejects_double_slash(self):
        """Test that project rejects double slashes."""
        with pytest.raises(ValidationError):
            Note(title="Test", content="Content", project="a//b")

    def test_project_rejects_special_characters(self):
        """Test that project rejects special characters in segments."""
        with pytest.raises(ValidationError):
            Note(title="Test", content="Content", project="project:name")

        with pytest.raises(ValidationError):
            Note(title="Test", content="Content", project="project name")  # spaces


class TestValidateProjectPath:
    """Tests for the validate_project_path function directly."""

    def test_valid_simple_path(self):
        """Test validation of simple project path."""
        assert validate_project_path("my-project") == "my-project"
        assert validate_project_path("project_name") == "project_name"
        assert validate_project_path("Project123") == "Project123"

    def test_valid_hierarchical_path(self):
        """Test validation of hierarchical project paths."""
        assert validate_project_path("a/b") == "a/b"
        assert validate_project_path("org/team/project") == "org/team/project"

    def test_rejects_empty(self):
        """Test that empty paths are rejected."""
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_project_path("")

    def test_rejects_path_traversal(self):
        """Test rejection of path traversal."""
        with pytest.raises(ValueError, match="path traversal"):
            validate_project_path("..")
        with pytest.raises(ValueError, match="path traversal"):
            validate_project_path("a/../b")

    def test_rejects_empty_segments(self):
        """Test rejection of empty segments."""
        with pytest.raises(ValueError, match="empty segments"):
            validate_project_path("/leading")
        with pytest.raises(ValueError, match="empty segments"):
            validate_project_path("trailing/")
        with pytest.raises(ValueError, match="empty segments"):
            validate_project_path("a//b")

    def test_rejects_invalid_characters(self):
        """Test rejection of invalid characters in segments."""
        with pytest.raises(ValueError, match="invalid characters"):
            validate_project_path("has space")
        with pytest.raises(ValueError, match="invalid characters"):
            validate_project_path("has:colon")
        with pytest.raises(ValueError, match="invalid characters"):
            validate_project_path("has.dot")


class TestProjectModel:
    """Tests for the Project model."""

    def test_project_creation(self):
        """Test creating a project with valid values."""
        project = Project(
            id="my-project",
            name="My Project",
            description="A test project"
        )
        assert project.id == "my-project"
        assert project.name == "My Project"
        assert project.description == "A test project"
        assert project.parent_id is None
        assert project.path is None
        assert isinstance(project.created_at, datetime.datetime)
        assert project.metadata == {}

    def test_project_with_subproject(self):
        """Test creating a sub-project with parent reference."""
        project = Project(
            id="monorepo/frontend",
            name="Frontend App",
            description="React web application",
            parent_id="monorepo",
            path="/home/user/monorepo/frontend"
        )
        assert project.id == "monorepo/frontend"
        assert project.parent_id == "monorepo"
        assert project.path == "/home/user/monorepo/frontend"

    def test_project_with_metadata(self):
        """Test project with custom metadata."""
        project = Project(
            id="my-repo",
            name="My Repo",
            metadata={"git_remote": "https://github.com/user/repo.git"}
        )
        assert project.metadata["git_remote"] == "https://github.com/user/repo.git"

    def test_project_id_validation(self):
        """Test that project ID follows path validation rules."""
        # Valid hierarchical ID
        project = Project(id="org/team/project", name="Project")
        assert project.id == "org/team/project"

        # Invalid: path traversal
        with pytest.raises(ValidationError):
            Project(id="../escape", name="Bad")

        # Invalid: special characters
        with pytest.raises(ValidationError):
            Project(id="has space", name="Bad")

    def test_project_name_required(self):
        """Test that project name is required and non-empty."""
        with pytest.raises(ValidationError):
            Project(id="test", name="")

        with pytest.raises(ValidationError):
            Project(id="test", name="   ")

    def test_project_parent_id_validation(self):
        """Test that parent_id follows path validation rules."""
        # Valid parent
        project = Project(id="child", name="Child", parent_id="parent/path")
        assert project.parent_id == "parent/path"

        # Invalid parent: path traversal
        with pytest.raises(ValidationError):
            Project(id="child", name="Child", parent_id="../escape")
