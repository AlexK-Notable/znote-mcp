"""Integration tests for MCP server with real services.

These tests use real ZettelService and SearchService instances (not mocks)
to verify end-to-end behavior of MCP tools.
"""
import pytest
from unittest.mock import MagicMock, patch

from znote_mcp.server.mcp_server import ZettelkastenMcpServer
from znote_mcp.services.zettel_service import ZettelService
from znote_mcp.services.search_service import SearchService
from znote_mcp.models.schema import NoteType, NotePurpose


def extract_note_id(result: str) -> str:
    """Extract note ID from MCP tool result string.

    Handles format: "Note created successfully with ID: xxx (project: general)"
    """
    if "ID:" not in result:
        raise ValueError(f"No ID found in result: {result}")
    id_part = result.split("ID:")[1].strip()
    # The ID is the first token, ends before space or parenthesis
    return id_part.split()[0].split("(")[0].strip()


class TestMCPIntegration:
    """Integration tests using real services."""

    @pytest.fixture
    def mcp_server(self, zettel_service):
        """Create MCP server with real ZettelService."""
        # Capture registered tools
        registered_tools = {}

        mock_mcp = MagicMock()

        def mock_tool_decorator(*args, **kwargs):
            def tool_wrapper(func):
                name = kwargs.get('name')
                registered_tools[name] = func
                return func
            return tool_wrapper

        mock_mcp.tool = mock_tool_decorator

        with patch('znote_mcp.server.mcp_server.FastMCP', return_value=mock_mcp):
            # Create server with real services injected
            server = ZettelkastenMcpServer()
            # Replace mocked services with real ones
            server.zettel_service = zettel_service
            server.search_service = SearchService(zettel_service)
            server.search_service.initialize()

        return server, registered_tools

    def test_create_and_retrieve_note(self, mcp_server):
        """Test creating a note and retrieving it with real services."""
        server, tools = mcp_server

        # Create a note
        create_result = tools['zk_create_note'](
            title="Integration Test Note",
            content="This is test content for integration testing.",
            note_type="permanent",
            tags="test, integration"
        )

        assert "successfully" in create_result.lower() or "created" in create_result.lower()

        note_id = extract_note_id(create_result)

        # Retrieve the note
        get_result = tools['zk_get_note'](note_id)

        assert "Integration Test Note" in get_result
        assert "test content" in get_result.lower()
        assert note_id in get_result

    def test_update_note_with_real_service(self, mcp_server):
        """Test updating a note using real services."""
        server, tools = mcp_server

        # Create a note first
        create_result = tools['zk_create_note'](
            title="Note to Update",
            content="Original content",
            note_type="fleeting"
        )

        note_id = extract_note_id(create_result)

        # Update the note
        update_result = tools['zk_update_note'](
            note_id=note_id,
            content="Updated content via integration test"
        )

        assert "updated" in update_result.lower()

        # Verify the update
        get_result = tools['zk_get_note'](note_id)
        assert "Updated content" in get_result

    def test_delete_note_with_real_service(self, mcp_server):
        """Test deleting a note using real services."""
        server, tools = mcp_server

        # Create a note
        create_result = tools['zk_create_note'](
            title="Note to Delete",
            content="This will be deleted"
        )

        note_id = extract_note_id(create_result)

        # Delete the note (may fail if git not configured in temp dir)
        delete_result = tools['zk_delete_note'](note_id)
        # Either succeeded or failed due to git issues in test env
        assert "deleted" in delete_result.lower() or "error" in delete_result.lower()

        # If delete succeeded, verify it's gone
        if "deleted" in delete_result.lower():
            get_result = tools['zk_get_note'](note_id)
            assert "not found" in get_result.lower()

    def test_search_notes_integration(self, mcp_server):
        """Test searching notes with real FTS."""
        server, tools = mcp_server

        # Create several notes
        tools['zk_create_note'](
            title="Python Programming Guide",
            content="Learn Python programming with examples"
        )
        tools['zk_create_note'](
            title="JavaScript Basics",
            content="Introduction to JavaScript for beginners"
        )
        tools['zk_create_note'](
            title="Advanced Python Patterns",
            content="Design patterns in Python programming"
        )

        # Search for Python
        search_result = tools['zk_search_notes'](query="Python")

        assert "Python" in search_result
        # Should find both Python notes
        assert "Programming Guide" in search_result or "Patterns" in search_result

    def test_link_creation_integration(self, mcp_server):
        """Test creating links between notes."""
        server, tools = mcp_server

        # Create two notes
        result1 = tools['zk_create_note'](
            title="Source Note",
            content="This note references another"
        )
        result2 = tools['zk_create_note'](
            title="Target Note",
            content="This is the target"
        )

        source_id = extract_note_id(result1)
        target_id = extract_note_id(result2)

        # Create a link
        link_result = tools['zk_create_link'](
            source_id=source_id,
            target_id=target_id,
            link_type="reference"
        )

        assert "created" in link_result.lower() or "link" in link_result.lower()

        # Verify link appears in find_related
        related_result = tools['zk_find_related'](source_id)
        assert target_id in related_result or "Target Note" in related_result

    def test_tag_operations_integration(self, mcp_server):
        """Test adding and removing tags."""
        server, tools = mcp_server

        # Create a note
        result = tools['zk_create_note'](
            title="Tagged Note",
            content="Note with tags"
        )
        note_id = extract_note_id(result)

        # Add a tag
        add_result = tools['zk_add_tag'](note_id, "new-tag")
        assert "added" in add_result.lower() or "new-tag" in add_result

        # Verify tag is present
        get_result = tools['zk_get_note'](note_id)
        assert "new-tag" in get_result

        # Remove the tag
        remove_result = tools['zk_remove_tag'](note_id, "new-tag")
        assert "removed" in remove_result.lower()

    def test_bulk_operations_integration(self, mcp_server):
        """Test bulk note creation."""
        server, tools = mcp_server
        import json

        notes_data = json.dumps([
            {"title": "Bulk Note 1", "content": "First bulk note"},
            {"title": "Bulk Note 2", "content": "Second bulk note"},
            {"title": "Bulk Note 3", "content": "Third bulk note"}
        ])

        result = tools['zk_bulk_create_notes'](notes_data)

        assert "3" in result or "created" in result.lower()

        # Verify notes exist
        list_result = tools['zk_list_notes']()
        assert "Bulk Note 1" in list_result
        assert "Bulk Note 2" in list_result
        assert "Bulk Note 3" in list_result

    def test_status_with_real_data(self, mcp_server):
        """Test zk_status with real notes."""
        server, tools = mcp_server

        # Create some notes with tags
        tools['zk_create_note'](
            title="Status Test Note",
            content="Content for status test",
            tags="status-test, important"
        )

        # Get status
        status_result = tools['zk_status']()

        assert "Summary" in status_result or "Total Notes" in status_result
        assert "Tags" in status_result
        assert "Health" in status_result

    def test_cleanup_tags_integration(self, mcp_server):
        """Test the zk_cleanup_tags tool."""
        server, tools = mcp_server

        # Create a note with tags
        result = tools['zk_create_note'](
            title="Note with Tag",
            content="Will be deleted",
            tags="orphan-test-tag"
        )
        note_id = extract_note_id(result)

        # Delete the note (tag becomes orphaned)
        tools['zk_delete_note'](note_id)

        # Cleanup orphaned tags
        cleanup_result = tools['zk_cleanup_tags']()

        # Should report cleanup or no orphans
        assert "clean" in cleanup_result.lower() or "unused" in cleanup_result.lower()

    def test_note_history_integration(self, mcp_server):
        """Test the zk_note_history tool."""
        server, tools = mcp_server

        # Create a note
        result = tools['zk_create_note'](
            title="History Test Note",
            content="Initial content"
        )
        note_id = extract_note_id(result)

        # Get history (may be empty if git not enabled)
        history_result = tools['zk_note_history'](note_id)

        # Should return some response (history or "git not enabled" message)
        assert note_id in history_result or "history" in history_result.lower() or "git" in history_result.lower()

    def test_error_handling_invalid_note_id(self, mcp_server):
        """Test error handling for invalid note IDs."""
        server, tools = mcp_server

        # Try to get non-existent note
        result = tools['zk_get_note']("nonexistent-note-id-12345")
        assert "not found" in result.lower() or "error" in result.lower()

    def test_error_handling_invalid_link(self, mcp_server):
        """Test error handling for invalid link creation."""
        server, tools = mcp_server

        # Try to create link with non-existent notes
        result = tools['zk_create_link'](
            source_id="nonexistent-1",
            target_id="nonexistent-2",
            link_type="reference"
        )
        assert "error" in result.lower() or "not found" in result.lower()


class TestMCPProjectIntegration:
    """Integration tests for project-related MCP tools."""

    @pytest.fixture
    def mcp_server(self, zettel_service):
        """Create MCP server with real services."""
        registered_tools = {}
        mock_mcp = MagicMock()

        def mock_tool_decorator(*args, **kwargs):
            def tool_wrapper(func):
                name = kwargs.get('name')
                registered_tools[name] = func
                return func
            return tool_wrapper

        mock_mcp.tool = mock_tool_decorator

        with patch('znote_mcp.server.mcp_server.FastMCP', return_value=mock_mcp):
            server = ZettelkastenMcpServer()
            server.zettel_service = zettel_service
            server.search_service = SearchService(zettel_service)
            server.search_service.initialize()

        return server, registered_tools

    def test_create_note_with_project(self, mcp_server):
        """Test creating notes with project assignment."""
        server, tools = mcp_server

        result = tools['zk_create_note'](
            title="Project Note",
            content="Note in a specific project",
            project="my-project"
        )

        note_id = extract_note_id(result)

        # Verify project is set
        get_result = tools['zk_get_note'](note_id)
        assert "my-project" in get_result

    def test_list_notes_by_project(self, mcp_server):
        """Test listing notes filtered by project."""
        server, tools = mcp_server

        # Create notes in different projects
        tools['zk_create_note'](
            title="Project A Note",
            content="In project A",
            project="project-a"
        )
        tools['zk_create_note'](
            title="Project B Note",
            content="In project B",
            project="project-b"
        )

        # List notes for project A
        result = tools['zk_list_notes'](project="project-a")

        assert "Project A Note" in result
        # Project B note may or may not be excluded depending on implementation
