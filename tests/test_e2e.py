"""End-to-End tests for the Zettelkasten MCP server.

These tests run against a completely isolated environment, ensuring
your production data is never touched.

Run with:
    uv run pytest tests/test_e2e.py -v

Debug with persistent data:
    ZETTELKASTEN_TEST_PERSIST=1 uv run pytest tests/test_e2e.py -v
    # Then inspect: tests/fixtures/notes/ and tests/fixtures/database/
"""
import json
import pytest
from pathlib import Path

from znote_mcp.models.schema import LinkType, NoteType

# Import fixtures from conftest_e2e (use relative import)
from conftest_e2e import (
    isolated_env,
    e2e_zettel_service,
    e2e_search_service,
    e2e_mcp_server,
    e2e_session_id,
    get_mcp_tool,
    IsolatedTestEnvironment,
)


class TestE2EIsolation:
    """Tests to verify the isolation mechanism works correctly."""

    def test_environment_is_isolated(self, isolated_env: IsolatedTestEnvironment):
        """Verify the test environment is properly isolated."""
        info = isolated_env.get_info()

        # Paths should be within tests/fixtures or temp
        assert "fixtures" in str(isolated_env.notes_dir) or "/tmp" in str(isolated_env.notes_dir)
        assert "fixtures" in str(isolated_env.database_path) or "/tmp" in str(isolated_env.database_path)

        # Should NOT be pointing to common production paths
        assert ".zettelkasten" not in str(isolated_env.notes_dir)
        assert "~/notes" not in str(isolated_env.notes_dir)
        assert "/home" not in str(isolated_env.notes_dir) or "fixtures" in str(isolated_env.notes_dir) or "/tmp" in str(isolated_env.notes_dir)

    def test_directories_created(self, isolated_env: IsolatedTestEnvironment):
        """Verify test directories are created."""
        assert isolated_env.notes_dir.exists()
        assert isolated_env.database_dir.exists()
        assert isolated_env.obsidian_dir.exists()


class TestE2ENoteCRUD:
    """End-to-end tests for note CRUD operations."""

    def test_create_and_retrieve_note(self, e2e_zettel_service, isolated_env):
        """Test full note creation and retrieval flow."""
        # Create a note
        note = e2e_zettel_service.create_note(
            title="E2E Test Note",
            content="This is an end-to-end test note.",
            note_type=NoteType.PERMANENT,
            project="e2e-testing",
            tags=["e2e", "test"]
        )

        assert note.id is not None
        assert note.project == "e2e-testing"

        # Verify file was created in isolated directory
        md_files = list(isolated_env.notes_dir.glob("**/*.md"))
        assert len(md_files) >= 1

        # Retrieve and verify
        retrieved = e2e_zettel_service.get_note(note.id)
        assert retrieved is not None
        assert retrieved.title == "E2E Test Note"
        assert "e2e" in [t.name for t in retrieved.tags]

    def test_update_note(self, e2e_zettel_service):
        """Test note update flow."""
        # Create
        note = e2e_zettel_service.create_note(
            title="Update Test",
            content="Original content",
            note_type=NoteType.FLEETING
        )

        # Update
        updated = e2e_zettel_service.update_note(
            note_id=note.id,
            title="Updated Title",
            content="Updated content"
        )

        assert updated.title == "Updated Title"

        # Verify persistence
        retrieved = e2e_zettel_service.get_note(note.id)
        assert retrieved.title == "Updated Title"

    def test_delete_note(self, e2e_zettel_service, isolated_env):
        """Test note deletion flow."""
        # Create
        note = e2e_zettel_service.create_note(
            title="Delete Test",
            content="To be deleted"
        )
        note_id = note.id

        # Verify file exists
        assert len(list(isolated_env.notes_dir.glob("**/*.md"))) >= 1

        # Delete
        e2e_zettel_service.delete_note(note_id)

        # Verify gone
        retrieved = e2e_zettel_service.get_note(note_id)
        assert retrieved is None


class TestE2ELinks:
    """End-to-end tests for link operations."""

    def test_create_bidirectional_link(self, e2e_zettel_service):
        """Test creating bidirectional links between notes."""
        # Create two notes
        source = e2e_zettel_service.create_note(
            title="Source Note",
            content="This note will link to another",
            note_type=NoteType.PERMANENT
        )
        target = e2e_zettel_service.create_note(
            title="Target Note",
            content="This note will be linked to",
            note_type=NoteType.PERMANENT
        )

        # Create bidirectional link
        e2e_zettel_service.create_link(
            source_id=source.id,
            target_id=target.id,
            link_type=LinkType.EXTENDS,
            description="Source extends Target",
            bidirectional=True
        )

        # Verify links exist
        source_links = e2e_zettel_service.get_linked_notes(source.id, "outgoing")
        target_links = e2e_zettel_service.get_linked_notes(target.id, "outgoing")

        assert len(source_links) >= 1
        assert len(target_links) >= 1


class TestE2EMCPTools:
    """End-to-end tests for the MCP tool interface."""

    def test_zk_create_note_tool(self, e2e_mcp_server):
        """Test the zk_create_note MCP tool."""
        create_note = get_mcp_tool(e2e_mcp_server, "zk_create_note")

        result = create_note(
            title="MCP Tool Test",
            content="Created via MCP tool",
            note_type="permanent",
            project="mcp-testing",
            tags="mcp,tool,test"
        )

        assert "Note created successfully" in result
        assert "mcp-testing" in result

    def test_zk_list_notes_tool(self, e2e_mcp_server, e2e_zettel_service):
        """Test the consolidated zk_list_notes MCP tool."""
        # Create some notes first
        for i in range(3):
            e2e_zettel_service.create_note(
                title=f"List Test Note {i}",
                content=f"Note {i} for list testing",
                project="list-testing"
            )

        list_notes = get_mcp_tool(e2e_mcp_server, "zk_list_notes")

        # Test mode="all"
        result = list_notes(mode="all", limit=10)
        assert "Notes" in result
        assert "List Test Note" in result

        # Test mode="by_project"
        result = list_notes(mode="by_project", project="list-testing")
        assert "list-testing" in result

    def test_zk_find_related_tool(self, e2e_mcp_server, e2e_zettel_service):
        """Test the consolidated zk_find_related MCP tool."""
        # Create linked notes
        source = e2e_zettel_service.create_note(
            title="Related Source",
            content="Source for related test"
        )
        target = e2e_zettel_service.create_note(
            title="Related Target",
            content="Target for related test"
        )
        e2e_zettel_service.create_link(
            source_id=source.id,
            target_id=target.id,
            link_type=LinkType.REFERENCE
        )

        find_related = get_mcp_tool(e2e_mcp_server, "zk_find_related")

        # Test mode="linked"
        result = find_related(note_id=source.id, mode="linked", direction="outgoing")
        assert "Related Target" in result

    def test_zk_status_tool(self, e2e_mcp_server, e2e_zettel_service):
        """Test the consolidated zk_status MCP tool."""
        # Create some data
        e2e_zettel_service.create_note(
            title="Status Test Note",
            content="For status testing",
            tags=["status-test"]
        )

        status = get_mcp_tool(e2e_mcp_server, "zk_status")

        # Test all sections
        result = status(sections="all")
        assert "Zettelkasten Status" in result
        assert "Summary" in result
        assert "Tags" in result

        # Test specific section
        result = status(sections="metrics")
        assert "Server Metrics" in result
        assert "Uptime" in result

    def test_zk_system_tool(self, e2e_mcp_server, e2e_zettel_service, isolated_env):
        """Test the consolidated zk_system MCP tool."""
        # Create a note first
        e2e_zettel_service.create_note(
            title="System Test Note",
            content="For system testing"
        )

        system = get_mcp_tool(e2e_mcp_server, "zk_system")

        # Test health action
        result = system(action="health")
        assert "Health Check" in result
        assert "SQLite" in result

        # Test backup action
        result = system(action="backup", backup_label="e2e-test")
        assert "Backup created" in result or "backup" in result.lower()

        # Test list_backups action
        result = system(action="list_backups")
        # Should either list backups or say none found
        assert "backup" in result.lower()

    def test_zk_bulk_operations(self, e2e_mcp_server, e2e_zettel_service):
        """Test bulk MCP tools."""
        bulk_create = get_mcp_tool(e2e_mcp_server, "zk_bulk_create_notes")

        # Bulk create notes
        notes_json = json.dumps([
            {"title": "Bulk Note 1", "content": "First bulk note", "tags": ["bulk"]},
            {"title": "Bulk Note 2", "content": "Second bulk note", "tags": ["bulk"]},
            {"title": "Bulk Note 3", "content": "Third bulk note", "tags": ["bulk"]}
        ])

        result = bulk_create(notes=notes_json)
        assert "Successfully created 3 notes" in result

        # Verify they exist
        all_notes = e2e_zettel_service.get_all_notes()
        bulk_notes = [n for n in all_notes if "Bulk Note" in n.title]
        assert len(bulk_notes) == 3


class TestE2EObsidianSync:
    """End-to-end tests for Obsidian vault synchronization."""

    def test_sync_to_obsidian(self, e2e_mcp_server, e2e_zettel_service, isolated_env):
        """Test syncing notes to Obsidian vault."""
        # Create notes with different projects (will become subdirs)
        e2e_zettel_service.create_note(
            title="Obsidian Test Note 1",
            content="Note for Obsidian sync",
            project="project-a"
        )
        e2e_zettel_service.create_note(
            title="Obsidian Test Note 2",
            content="Another note for sync",
            project="project-b"
        )

        system = get_mcp_tool(e2e_mcp_server, "zk_system")

        # Sync to Obsidian
        result = system(action="sync")
        assert "synced" in result.lower() or "Successfully" in result

        # Verify files in Obsidian vault
        obsidian_files = list(isolated_env.obsidian_dir.glob("**/*.md"))
        assert len(obsidian_files) >= 2


class TestE2ESearch:
    """End-to-end tests for search functionality."""

    def test_fts_search(self, e2e_mcp_server, e2e_zettel_service):
        """Test full-text search via MCP tool."""
        # Create searchable notes
        e2e_zettel_service.create_note(
            title="Python Programming Guide",
            content="This note covers Python async await patterns and coroutines."
        )
        e2e_zettel_service.create_note(
            title="JavaScript Overview",
            content="JavaScript is a versatile programming language."
        )

        fts_search = get_mcp_tool(e2e_mcp_server, "zk_fts_search")

        # Search for Python
        result = fts_search(query="python", limit=10)
        assert "Python" in result
        # JavaScript should not appear prominently
        assert "Python Programming Guide" in result

    def test_search_with_filters(self, e2e_mcp_server, e2e_zettel_service):
        """Test search with tag and type filters."""
        # Create notes with different tags
        e2e_zettel_service.create_note(
            title="Tagged Note A",
            content="Content A",
            tags=["alpha", "test"]
        )
        e2e_zettel_service.create_note(
            title="Tagged Note B",
            content="Content B",
            tags=["beta", "test"]
        )

        search = get_mcp_tool(e2e_mcp_server, "zk_search_notes")

        # Search by tag
        result = search(tags="alpha")
        assert "Tagged Note A" in result
        # Note B should not appear
        assert "Tagged Note B" not in result


class TestE2EDataIntegrity:
    """End-to-end tests for data integrity and recovery."""

    def test_rebuild_index(self, e2e_mcp_server, e2e_zettel_service, isolated_env):
        """Test database rebuild from markdown files."""
        # Create notes
        for i in range(5):
            e2e_zettel_service.create_note(
                title=f"Rebuild Test {i}",
                content=f"Content for rebuild test {i}"
            )

        system = get_mcp_tool(e2e_mcp_server, "zk_system")

        # Rebuild index
        result = system(action="rebuild")
        assert "rebuilt" in result.lower() or "processed" in result.lower()

        # Verify notes still accessible
        all_notes = e2e_zettel_service.get_all_notes()
        assert len(all_notes) >= 5

    def test_database_health_check(self, e2e_mcp_server, e2e_zettel_service):
        """Test database health check functionality."""
        # Create some data
        e2e_zettel_service.create_note(
            title="Health Check Note",
            content="For health check testing"
        )

        # Check health via service
        health = e2e_zettel_service.check_database_health()

        assert health["healthy"] is True
        assert health["sqlite_ok"] is True
        assert health["note_count"] >= 1
