# tests/test_mcp_server.py
"""Tests for the MCP server implementation."""
import datetime
import pytest
from unittest.mock import patch, MagicMock, call

from znote_mcp.server.mcp_server import ZettelkastenMcpServer
from znote_mcp.models.schema import LinkType, NoteType

class TestMcpServer:
    """Tests for the ZettelkastenMcpServer class."""
    
    def setup_method(self):
        """Set up test environment before each test."""
        # Capture the tool decorator functions when registering
        self.registered_tools = {}
        
        # Create a mock for FastMCP
        self.mock_mcp = MagicMock()
        
        # Mock the tool decorator to capture registered functions BEFORE server creation
        def mock_tool_decorator(*args, **kwargs):
            def tool_wrapper(func):
                # Store the function with its name
                name = kwargs.get('name')
                self.registered_tools[name] = func
                return func
            return tool_wrapper
        self.mock_mcp.tool = mock_tool_decorator
        
        # Mock the ZettelService and SearchService
        self.mock_zettel_service = MagicMock()
        self.mock_search_service = MagicMock()
        
        # Create patchers for FastMCP, ZettelService, and SearchService
        self.mcp_patcher = patch('znote_mcp.server.mcp_server.FastMCP', return_value=self.mock_mcp)
        self.zettel_patcher = patch('znote_mcp.server.mcp_server.ZettelService', return_value=self.mock_zettel_service)
        self.search_patcher = patch('znote_mcp.server.mcp_server.SearchService', return_value=self.mock_search_service)
        
        # Start the patchers
        self.mcp_patcher.start()
        self.zettel_patcher.start()
        self.search_patcher.start()
        
        # Create a server instance AFTER setting up the mocks
        self.server = ZettelkastenMcpServer()

    def teardown_method(self):
        """Clean up after each test."""
        self.mcp_patcher.stop()
        self.zettel_patcher.stop()
        self.search_patcher.stop()

    def test_server_initialization(self):
        """Test server initialization."""
        # Check services are initialized
        assert self.mock_zettel_service.initialize.called
        assert self.mock_search_service.initialize.called
        
    def test_create_note_tool(self):
        """Test the zk_create_note tool."""
        # Check the tool is registered
        assert 'zk_create_note' in self.registered_tools
        # Set up return value for create_note
        mock_note = MagicMock()
        mock_note.id = "test123"
        self.mock_zettel_service.create_note.return_value = mock_note
        # Call the tool function directly
        create_note_func = self.registered_tools['zk_create_note']
        result = create_note_func(
            title="Test Note",
            content="Test content",
            note_type="permanent",
            tags="tag1, tag2"
        )
        # Verify result
        assert "successfully" in result
        assert mock_note.id in result
        # Verify service call
        self.mock_zettel_service.create_note.assert_called_with(
            title="Test Note",
            content="Test content",
            note_type=NoteType.PERMANENT,
            project="general",
            tags=["tag1", "tag2"]
        )

    def test_get_note_tool(self):
        """Test the zk_get_note tool."""
        # Check the tool is registered
        assert 'zk_get_note' in self.registered_tools
        
        # Set up mock note
        mock_note = MagicMock()
        mock_note.id = "test123"
        mock_note.title = "Test Note"
        mock_note.content = "Test content"
        mock_note.note_type = NoteType.PERMANENT
        mock_note.created_at.isoformat.return_value = "2023-01-01T12:00:00"
        mock_note.updated_at.isoformat.return_value = "2023-01-01T12:30:00"
        mock_tag1 = MagicMock()
        mock_tag1.name = "tag1"
        mock_tag2 = MagicMock()
        mock_tag2.name = "tag2"
        mock_note.tags = [mock_tag1, mock_tag2]
        mock_note.links = []
        
        # Set up return value for get_note
        self.mock_zettel_service.get_note.return_value = mock_note
        
        # Call the tool function directly
        get_note_func = self.registered_tools['zk_get_note']
        result = get_note_func(identifier="test123")
        
        # Verify result
        assert "# Test Note" in result
        assert "ID: test123" in result
        assert "Test content" in result
        
        # Verify service call
        self.mock_zettel_service.get_note.assert_called_with("test123")

    def test_create_link_tool(self):
        """Test the zk_create_link tool."""
        # Check the tool is registered
        assert 'zk_create_link' in self.registered_tools
        
        # Set up mock notes
        source_note = MagicMock()
        source_note.id = "source123"
        target_note = MagicMock()
        target_note.id = "target456"
        
        # Set up return value for create_link
        self.mock_zettel_service.create_link.return_value = (source_note, target_note)
        
        # Call the tool function directly
        create_link_func = self.registered_tools['zk_create_link']
        result = create_link_func(
            source_id="source123",
            target_id="target456",
            link_type="extends",
            description="Test link",
            bidirectional=True
        )
        
        # Verify result
        assert "Bidirectional link created" in result
        assert "source123" in result
        assert "target456" in result
        
        # Verify service call
        self.mock_zettel_service.create_link.assert_called_with(
            source_id="source123",
            target_id="target456",
            link_type=LinkType.EXTENDS,
            description="Test link",
            bidirectional=True
        )

    def test_search_notes_tool(self):
        """Test the zk_search_notes tool."""
        # Check the tool is registered
        assert 'zk_search_notes' in self.registered_tools
        
        # Set up mock notes
        mock_note1 = MagicMock()
        mock_note1.id = "note1"
        mock_note1.title = "Note 1"
        mock_note1.content = "This is note 1 content"
        mock_tag1 = MagicMock()
        mock_tag1.name = "tag1"
        mock_tag2 = MagicMock()
        mock_tag2.name = "tag2"
        mock_note1.tags = [mock_tag1, mock_tag2]
        mock_note1.created_at.strftime.return_value = "2023-01-01"
        
        mock_note2 = MagicMock()
        mock_note2.id = "note2"
        mock_note2.title = "Note 2"
        mock_note2.content = "This is note 2 content"
        # mock_note2.tags = [MagicMock(name="tag1")]
        mock_tag1 = MagicMock()
        mock_tag1.name = "tag1"
        mock_note2.tags = [mock_tag1]
        mock_note2.created_at.strftime.return_value = "2023-01-02"
        
        # Set up mock search results
        mock_result1 = MagicMock()
        mock_result1.note = mock_note1
        mock_result2 = MagicMock()
        mock_result2.note = mock_note2
        
        self.mock_search_service.search_combined.return_value = [mock_result1, mock_result2]
        
        # Call the tool function directly
        search_notes_func = self.registered_tools['zk_search_notes']
        result = search_notes_func(
            query="test query",
            tags="tag1, tag2",
            note_type="permanent",
            limit=10
        )
        
        # Verify result
        assert "Found 2 matching notes" in result
        assert "Note 1" in result
        assert "Note 2" in result
        
        # Verify service call
        self.mock_search_service.search_combined.assert_called_with(
            text="test query",
            tags=["tag1", "tag2"],
            note_type=NoteType.PERMANENT
        )

    def test_error_handling(self):
        """Test error handling in the server."""
        # Test ValueError handling
        value_error = ValueError("Invalid input")
        result = self.server.format_error_response(value_error)
        assert "Error: Invalid input" in result

        # Test IOError handling
        io_error = IOError("File not found")
        result = self.server.format_error_response(io_error)
        assert "Error: File not found" in result

        # Test general exception handling
        general_error = Exception("Something went wrong")
        result = self.server.format_error_response(general_error)
        assert "Error: Something went wrong" in result

    def test_list_notes_tool(self):
        """Test the zk_list_notes tool with mode='all'."""
        # Check the tool is registered
        assert 'zk_list_notes' in self.registered_tools

        # Set up mock notes with actual datetime values for sorting
        mock_note1 = MagicMock()
        mock_note1.id = "note1"
        mock_note1.title = "Alpha Note"
        mock_note1.note_type.value = "permanent"
        mock_note1.project = "general"
        mock_note1.updated_at = datetime.datetime(2023, 1, 2, 14, 0)
        mock_note1.created_at = datetime.datetime(2023, 1, 2, 12, 0)
        mock_note1.tags = []

        mock_note2 = MagicMock()
        mock_note2.id = "note2"
        mock_note2.title = "Beta Note"
        mock_note2.note_type.value = "fleeting"
        mock_note2.project = "general"
        mock_note2.updated_at = datetime.datetime(2023, 1, 1, 10, 0)
        mock_note2.created_at = datetime.datetime(2023, 1, 1, 9, 0)
        mock_tag = MagicMock()
        mock_tag.name = "test"
        mock_note2.tags = [mock_tag]

        self.mock_zettel_service.get_all_notes.return_value = [mock_note1, mock_note2]

        # Call the tool function directly with mode="all"
        list_notes_func = self.registered_tools['zk_list_notes']
        result = list_notes_func(mode="all", limit=50, offset=0, sort_by="updated_at", descending=True)

        # Verify result
        assert "Notes (1-2 of 2)" in result
        assert "Alpha Note" in result
        assert "Beta Note" in result
        assert "test" in result  # tag

        # Verify service call
        self.mock_zettel_service.get_all_notes.assert_called_once()

    def test_list_notes_pagination(self):
        """Test pagination in zk_list_notes with mode='all'."""
        assert 'zk_list_notes' in self.registered_tools

        # Create 5 mock notes with actual datetime values for sorting
        mock_notes = []
        for i in range(5):
            note = MagicMock()
            note.id = f"note{i}"
            note.title = f"Note {i}"
            note.note_type.value = "permanent"
            note.project = "general"
            note.updated_at = datetime.datetime(2023, 1, i + 1, 10, 0)
            note.created_at = datetime.datetime(2023, 1, i + 1, 9, 0)
            note.tags = []
            mock_notes.append(note)

        self.mock_zettel_service.get_all_notes.return_value = mock_notes

        list_notes_func = self.registered_tools['zk_list_notes']

        # Request only first 2 notes
        result = list_notes_func(mode="all", limit=2, offset=0)
        assert "Notes (1-2 of 5)" in result
        assert "Use offset=2 to see more notes" in result

    def test_list_notes_empty(self):
        """Test zk_list_notes with no notes."""
        assert 'zk_list_notes' in self.registered_tools

        self.mock_zettel_service.get_all_notes.return_value = []

        list_notes_func = self.registered_tools['zk_list_notes']
        result = list_notes_func(mode="all")

        assert "No notes found" in result

    def test_add_tag_tool(self):
        """Test the zk_add_tag tool."""
        assert 'zk_add_tag' in self.registered_tools

        mock_note = MagicMock()
        mock_note.id = "test123"
        mock_note.title = "Test Note"

        self.mock_zettel_service.add_tag_to_note.return_value = mock_note

        add_tag_func = self.registered_tools['zk_add_tag']
        result = add_tag_func(note_id="test123", tag="new-tag")

        assert "Tag 'new-tag' added" in result
        assert "Test Note" in result

        self.mock_zettel_service.add_tag_to_note.assert_called_with("test123", "new-tag")

    def test_add_tag_empty_tag(self):
        """Test zk_add_tag with empty tag."""
        assert 'zk_add_tag' in self.registered_tools

        add_tag_func = self.registered_tools['zk_add_tag']
        result = add_tag_func(note_id="test123", tag="")

        assert "Error: Tag cannot be empty" in result

    def test_remove_tag_tool(self):
        """Test the zk_remove_tag tool."""
        assert 'zk_remove_tag' in self.registered_tools

        mock_note = MagicMock()
        mock_note.id = "test123"
        mock_note.title = "Test Note"

        self.mock_zettel_service.remove_tag_from_note.return_value = mock_note

        remove_tag_func = self.registered_tools['zk_remove_tag']
        result = remove_tag_func(note_id="test123", tag="old-tag")

        assert "Tag 'old-tag' removed" in result
        assert "Test Note" in result

        self.mock_zettel_service.remove_tag_from_note.assert_called_with("test123", "old-tag")

    def test_remove_tag_empty_tag(self):
        """Test zk_remove_tag with empty tag."""
        assert 'zk_remove_tag' in self.registered_tools

        remove_tag_func = self.registered_tools['zk_remove_tag']
        result = remove_tag_func(note_id="test123", tag="  ")

        assert "Error: Tag cannot be empty" in result

    def test_export_note_tool(self):
        """Test the zk_export_note tool."""
        assert 'zk_export_note' in self.registered_tools

        markdown_content = """# Test Note

Some content here.

## Links
- [[Other Note]]
"""
        self.mock_zettel_service.export_note.return_value = markdown_content

        export_func = self.registered_tools['zk_export_note']
        result = export_func(note_id="test123", format="markdown")

        assert "# Test Note" in result
        assert "Some content here" in result

        self.mock_zettel_service.export_note.assert_called_with("test123", "markdown")

    def test_export_note_not_found(self):
        """Test zk_export_note with non-existent note."""
        assert 'zk_export_note' in self.registered_tools

        self.mock_zettel_service.export_note.side_effect = ValueError("Note with ID notfound not found")

        export_func = self.registered_tools['zk_export_note']
        result = export_func(note_id="notfound", format="markdown")

        assert "Error:" in result
        assert "not found" in result

    def test_status_metrics_tool(self):
        """Test the zk_status tool with metrics section."""
        assert 'zk_status' in self.registered_tools

        status_func = self.registered_tools['zk_status']
        result = status_func(sections="metrics")

        # Verify result contains metrics info
        assert "Server Metrics" in result
        assert "Uptime:" in result
        assert "Operations:" in result
        assert "Success Rate:" in result

    def test_status_all_sections(self):
        """Test zk_status with all sections."""
        assert 'zk_status' in self.registered_tools

        status_func = self.registered_tools['zk_status']
        result = status_func(sections="all")

        # Verify result contains status info
        assert "Zettelkasten Status" in result
        assert "Uptime:" in result

    def test_fts_search_tool(self):
        """Test the zk_fts_search tool."""
        assert 'zk_fts_search' in self.registered_tools

        # Set up mock search results
        self.mock_zettel_service.fts_search.return_value = [
            {"id": "note1", "title": "Python Guide", "rank": -1.5, "snippet": "...about <mark>python</mark>..."},
            {"id": "note2", "title": "Async Python", "rank": -1.2, "snippet": "...<mark>python</mark> async..."},
        ]

        fts_search_func = self.registered_tools['zk_fts_search']
        result = fts_search_func(query="python", limit=10, highlight=True)

        # Verify result
        assert "Found 2 notes" in result
        assert "Python Guide" in result
        assert "Async Python" in result
        assert "Relevance:" in result

        # Verify service call
        self.mock_zettel_service.fts_search.assert_called_with(
            query="python",
            limit=10,
            highlight=True
        )

    def test_fts_search_empty_query(self):
        """Test zk_fts_search with empty query."""
        assert 'zk_fts_search' in self.registered_tools

        fts_search_func = self.registered_tools['zk_fts_search']
        result = fts_search_func(query="", limit=10, highlight=True)

        assert "Error:" in result
        assert "query is required" in result

    def test_fts_search_no_results(self):
        """Test zk_fts_search with no matching results."""
        assert 'zk_fts_search' in self.registered_tools

        self.mock_zettel_service.fts_search.return_value = []

        fts_search_func = self.registered_tools['zk_fts_search']
        result = fts_search_func(query="nonexistent", limit=10, highlight=True)

        assert "No notes found" in result

    # ========== Bulk Operations Tests ==========

    def test_bulk_create_notes_tool(self):
        """Test zk_bulk_create_notes tool."""
        assert 'zk_bulk_create_notes' in self.registered_tools

        # Create mock notes
        mock_notes = [
            MagicMock(id="note1", title="Note 1"),
            MagicMock(id="note2", title="Note 2"),
        ]
        self.mock_zettel_service.bulk_create_notes.return_value = mock_notes

        bulk_create_func = self.registered_tools['zk_bulk_create_notes']
        json_input = '[{"title": "Note 1", "content": "Content 1"}, {"title": "Note 2", "content": "Content 2"}]'
        result = bulk_create_func(notes=json_input)

        assert "Successfully created 2 notes" in result
        assert "Note 1" in result
        assert "Note 2" in result

    def test_bulk_create_notes_invalid_json(self):
        """Test zk_bulk_create_notes with invalid JSON."""
        assert 'zk_bulk_create_notes' in self.registered_tools

        bulk_create_func = self.registered_tools['zk_bulk_create_notes']
        result = bulk_create_func(notes="not valid json")

        assert "Error: Invalid JSON" in result

    def test_bulk_delete_notes_tool(self):
        """Test zk_bulk_delete_notes tool."""
        assert 'zk_bulk_delete_notes' in self.registered_tools

        self.mock_zettel_service.bulk_delete_notes.return_value = 3

        bulk_delete_func = self.registered_tools['zk_bulk_delete_notes']
        result = bulk_delete_func(note_ids="note1, note2, note3")

        assert "Successfully deleted 3 notes" in result
        self.mock_zettel_service.bulk_delete_notes.assert_called_once()

    def test_bulk_delete_notes_empty_ids(self):
        """Test zk_bulk_delete_notes with empty IDs."""
        assert 'zk_bulk_delete_notes' in self.registered_tools

        bulk_delete_func = self.registered_tools['zk_bulk_delete_notes']
        result = bulk_delete_func(note_ids="")

        assert "Error:" in result
        assert "No note IDs" in result

    def test_bulk_add_tags_tool(self):
        """Test zk_bulk_add_tags tool."""
        assert 'zk_bulk_add_tags' in self.registered_tools

        self.mock_zettel_service.bulk_add_tags.return_value = 2

        bulk_add_tags_func = self.registered_tools['zk_bulk_add_tags']
        result = bulk_add_tags_func(note_ids="note1, note2", tags="python, programming")

        assert "Added tags" in result
        assert "python" in result
        assert "programming" in result
        assert "2 notes" in result

    def test_bulk_remove_tags_tool(self):
        """Test zk_bulk_remove_tags tool."""
        assert 'zk_bulk_remove_tags' in self.registered_tools

        self.mock_zettel_service.bulk_remove_tags.return_value = 2

        bulk_remove_tags_func = self.registered_tools['zk_bulk_remove_tags']
        result = bulk_remove_tags_func(note_ids="note1, note2", tags="outdated")

        assert "Removed tags" in result
        assert "outdated" in result
        assert "2 notes" in result

    def test_bulk_move_to_project_tool(self):
        """Test zk_bulk_move_to_project tool."""
        assert 'zk_bulk_move_to_project' in self.registered_tools

        self.mock_zettel_service.bulk_update_project.return_value = 3

        bulk_move_func = self.registered_tools['zk_bulk_move_to_project']
        result = bulk_move_func(note_ids="note1, note2, note3", project="research")

        assert "Moved 3 notes" in result
        assert "research" in result

    def test_bulk_move_to_project_empty_project(self):
        """Test zk_bulk_move_to_project with empty project."""
        assert 'zk_bulk_move_to_project' in self.registered_tools

        bulk_move_func = self.registered_tools['zk_bulk_move_to_project']
        result = bulk_move_func(note_ids="note1", project="")

        assert "Error:" in result
        assert "Project name is required" in result
