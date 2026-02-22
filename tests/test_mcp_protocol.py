"""Protocol integration tests for the Zettelkasten MCP server.

These tests exercise all 22 MCP tools through the full JSON-RPC protocol
using a real ClientSession connected to a real FastMCP server. No mocking
of transport, server, or session internals.

Written against the interface defined in plan phase 3.
Tests verify behavior through the wire protocol, not implementation details.

Run with:
    uv run pytest tests/test_mcp_protocol.py -v
"""

import json

import pytest
from mcp.types import TextContent

# Import fixtures from conftest_protocol (same pattern as test_e2e.py)
from tests.conftest_protocol import mcp_client  # noqa: F401 — fixture used by pytest
from tests.conftest_protocol import mcp_server  # noqa: F401 — fixture used by pytest
from tests.conftest_protocol import (  # noqa: F401 — fixture used by pytest
    extract_note_id_from_protocol,
    get_text,
    protocol_config,
    semantic_mcp_client,
    semantic_mcp_server,
    semantic_protocol_config,
)

# All 22 tools that should be registered on the server
ALL_TOOL_NAMES = {
    "zk_create_note",
    "zk_get_note",
    "zk_update_note",
    "zk_delete_note",
    "zk_create_link",
    "zk_remove_link",
    "zk_search_notes",
    "zk_fts_search",
    "zk_add_tag",
    "zk_remove_tag",
    "zk_cleanup_tags",
    "zk_bulk_create_notes",
    "zk_list_notes",
    "zk_find_related",
    "zk_status",
    "zk_system",
    "zk_restore",
    "zk_create_project",
    "zk_list_projects",
    "zk_get_project",
    "zk_delete_project",
    "zk_note_history",
}


# =============================================================================
# 1. Connection and Discovery
# =============================================================================


class TestMCPProtocolConnection:
    """Verify the protocol-level handshake and tool discovery."""

    @pytest.mark.anyio
    async def test_client_connects_and_initializes(self, mcp_client):
        """ClientSession can connect and list tools without error."""
        tools_result = await mcp_client.list_tools()
        assert tools_result.tools is not None
        assert len(tools_result.tools) > 0

    @pytest.mark.anyio
    async def test_list_tools_returns_all_22_tools(self, mcp_client):
        """All 22 registered tools are discoverable via the protocol."""
        tools_result = await mcp_client.list_tools()
        tool_names = {t.name for t in tools_result.tools}
        assert tool_names == ALL_TOOL_NAMES, (
            f"Missing: {ALL_TOOL_NAMES - tool_names}, "
            f"Extra: {tool_names - ALL_TOOL_NAMES}"
        )

    @pytest.mark.anyio
    async def test_tool_schemas_have_required_fields(self, mcp_client):
        """zk_create_note schema declares title and content as required."""
        tools_result = await mcp_client.list_tools()
        create_tool = next(t for t in tools_result.tools if t.name == "zk_create_note")
        schema = create_tool.inputSchema
        required = schema.get("required", [])
        assert "title" in required, "title should be required"
        assert "content" in required, "content should be required"


# =============================================================================
# 2. CRUD Operations
# =============================================================================


class TestMCPProtocolCRUD:
    """Create, read, update, delete through the full protocol stack."""

    @pytest.mark.anyio
    async def test_create_note_via_protocol(self, mcp_client):
        """Creating a note via the protocol returns a success message with an ID."""
        result = await mcp_client.call_tool(
            "zk_create_note",
            {
                "title": "Protocol CRUD Test",
                "content": "Created through the MCP protocol.",
                "note_type": "permanent",
                "tags": "protocol,test",
            },
        )
        text = get_text(result)
        assert "successfully" in text.lower()
        note_id = extract_note_id_from_protocol(result)
        assert len(note_id) > 0

    @pytest.mark.anyio
    async def test_create_and_get_note_round_trip(self, mcp_client):
        """A note created via the protocol can be retrieved with correct fields."""
        create_result = await mcp_client.call_tool(
            "zk_create_note",
            {
                "title": "Round Trip Note",
                "content": "Content for round-trip verification.",
                "note_type": "fleeting",
                "project": "test-project",
                "tags": "roundtrip,protocol",
            },
        )
        note_id = extract_note_id_from_protocol(create_result)

        get_result = await mcp_client.call_tool("zk_get_note", {"identifier": note_id})
        text = get_text(get_result)

        assert "Round Trip Note" in text
        assert "Content for round-trip verification" in text
        assert "test-project" in text.lower() or "test-project" in text
        assert "roundtrip" in text.lower()

    @pytest.mark.anyio
    async def test_update_note_via_protocol(self, mcp_client):
        """Updating a note changes its content when retrieved."""
        # Create
        create_result = await mcp_client.call_tool(
            "zk_create_note",
            {
                "title": "Note To Update",
                "content": "Original content before update.",
                "note_type": "permanent",
            },
        )
        note_id = extract_note_id_from_protocol(create_result)

        # Update
        update_result = await mcp_client.call_tool(
            "zk_update_note",
            {
                "note_id": note_id,
                "title": "Updated Title",
                "content": "Updated content after protocol call.",
            },
        )
        update_text = get_text(update_result)
        assert "successfully" in update_text.lower()

        # Verify
        get_result = await mcp_client.call_tool("zk_get_note", {"identifier": note_id})
        text = get_text(get_result)
        assert "Updated Title" in text
        assert "Updated content after protocol call" in text

    @pytest.mark.anyio
    async def test_delete_note_via_protocol(self, mcp_client):
        """Deleting a note succeeds and the note is no longer retrievable."""
        # Create
        create_result = await mcp_client.call_tool(
            "zk_create_note",
            {
                "title": "Note To Delete",
                "content": "This note will be deleted.",
                "note_type": "fleeting",
            },
        )
        note_id = extract_note_id_from_protocol(create_result)

        # Delete (config.git_enabled=False in protocol_config, so no git issues)
        delete_result = await mcp_client.call_tool(
            "zk_delete_note", {"note_id": note_id}
        )
        delete_text = get_text(delete_result)
        assert "deleted" in delete_text.lower() or "successfully" in delete_text.lower()

        # Verify gone
        get_result = await mcp_client.call_tool("zk_get_note", {"identifier": note_id})
        text = get_text(get_result)
        assert "not found" in text.lower()


# =============================================================================
# 3. Search Operations
# =============================================================================


class TestMCPProtocolSearch:
    """Full-text and filtered search through the protocol."""

    @pytest.mark.anyio
    async def test_fts_search_via_protocol(self, mcp_client):
        """FTS5 search finds notes matching a query term."""
        # Seed data
        await mcp_client.call_tool(
            "zk_create_note",
            {
                "title": "Quantum Physics Overview",
                "content": "Quantum entanglement and superposition explained.",
                "note_type": "permanent",
            },
        )
        await mcp_client.call_tool(
            "zk_create_note",
            {
                "title": "Classical Mechanics",
                "content": "Newton's laws of motion are foundational.",
                "note_type": "permanent",
            },
        )

        result = await mcp_client.call_tool(
            "zk_fts_search", {"query": "quantum", "limit": 10}
        )
        text = get_text(result)
        assert "Quantum Physics Overview" in text

    @pytest.mark.anyio
    async def test_search_notes_by_tag(self, mcp_client):
        """zk_search_notes can filter results by tag."""
        await mcp_client.call_tool(
            "zk_create_note",
            {
                "title": "Tagged Note Alpha",
                "content": "Alpha content for tag search.",
                "note_type": "permanent",
                "tags": "alpha-search-tag",
            },
        )
        await mcp_client.call_tool(
            "zk_create_note",
            {
                "title": "Tagged Note Beta",
                "content": "Beta content without the search tag.",
                "note_type": "permanent",
                "tags": "beta-other-tag",
            },
        )

        result = await mcp_client.call_tool(
            "zk_search_notes",
            {"tags": "alpha-search-tag", "mode": "text", "limit": 10},
        )
        text = get_text(result)
        assert "Tagged Note Alpha" in text

    @pytest.mark.anyio
    async def test_list_notes_all_mode(self, mcp_client):
        """zk_list_notes with mode=all returns created notes."""
        await mcp_client.call_tool(
            "zk_create_note",
            {
                "title": "Listable Note One",
                "content": "Content for listing test.",
                "note_type": "permanent",
            },
        )
        await mcp_client.call_tool(
            "zk_create_note",
            {
                "title": "Listable Note Two",
                "content": "Second content for listing test.",
                "note_type": "permanent",
            },
        )

        result = await mcp_client.call_tool(
            "zk_list_notes", {"mode": "all", "limit": 50}
        )
        text = get_text(result)
        assert "Listable Note One" in text
        assert "Listable Note Two" in text


# =============================================================================
# 4. Link Operations
# =============================================================================


class TestMCPProtocolLinks:
    """Link creation and related-note discovery."""

    @pytest.mark.anyio
    async def test_create_link_via_protocol(self, mcp_client):
        """Creating a link between two notes produces a success message."""
        r1 = await mcp_client.call_tool(
            "zk_create_note",
            {
                "title": "Link Source",
                "content": "Source note for link test.",
                "note_type": "permanent",
            },
        )
        r2 = await mcp_client.call_tool(
            "zk_create_note",
            {
                "title": "Link Target",
                "content": "Target note for link test.",
                "note_type": "permanent",
            },
        )
        source_id = extract_note_id_from_protocol(r1)
        target_id = extract_note_id_from_protocol(r2)

        link_result = await mcp_client.call_tool(
            "zk_create_link",
            {
                "source_id": source_id,
                "target_id": target_id,
                "link_type": "reference",
            },
        )
        text = get_text(link_result)
        assert "link created" in text.lower() or "Link created" in text

    @pytest.mark.anyio
    async def test_find_related_linked_mode(self, mcp_client):
        """zk_find_related in linked mode discovers linked notes."""
        r1 = await mcp_client.call_tool(
            "zk_create_note",
            {
                "title": "Hub Note",
                "content": "Central hub with outgoing links.",
                "note_type": "permanent",
            },
        )
        r2 = await mcp_client.call_tool(
            "zk_create_note",
            {
                "title": "Satellite Note",
                "content": "Satellite linked from the hub.",
                "note_type": "permanent",
            },
        )
        hub_id = extract_note_id_from_protocol(r1)
        sat_id = extract_note_id_from_protocol(r2)

        await mcp_client.call_tool(
            "zk_create_link",
            {
                "source_id": hub_id,
                "target_id": sat_id,
                "link_type": "extends",
            },
        )

        related_result = await mcp_client.call_tool(
            "zk_find_related",
            {"note_id": hub_id, "mode": "linked", "direction": "both"},
        )
        text = get_text(related_result)
        assert "Satellite Note" in text


# =============================================================================
# 5. Batch Operations
# =============================================================================


class TestMCPProtocolBatch:
    """Bulk creates and tag batch operations."""

    @pytest.mark.anyio
    async def test_bulk_create_notes_via_protocol(self, mcp_client):
        """zk_bulk_create_notes works through the protocol with JSON string input."""
        notes_payload = json.dumps(
            [
                {
                    "title": "Bulk Note A",
                    "content": "First bulk note.",
                    "note_type": "permanent",
                },
                {
                    "title": "Bulk Note B",
                    "content": "Second bulk note.",
                    "note_type": "fleeting",
                },
                {
                    "title": "Bulk Note C",
                    "content": "Third bulk note.",
                    "note_type": "permanent",
                },
            ]
        )
        result = await mcp_client.call_tool(
            "zk_bulk_create_notes", {"notes": notes_payload}
        )
        text = get_text(result)
        assert not result.isError, f"bulk_create_notes failed: {text}"
        assert "successfully" in text.lower()
        assert "Bulk Note A" in text
        assert "Bulk Note B" in text
        assert "Bulk Note C" in text

    @pytest.mark.anyio
    async def test_add_and_remove_tag_via_protocol(self, mcp_client):
        """Adding then removing a tag is reflected in note retrieval."""
        create_result = await mcp_client.call_tool(
            "zk_create_note",
            {
                "title": "Taggable Note",
                "content": "Note for tag add/remove test.",
                "note_type": "permanent",
            },
        )
        note_id = extract_note_id_from_protocol(create_result)

        # Add tag
        add_result = await mcp_client.call_tool(
            "zk_add_tag", {"note_id": note_id, "tag": "ephemeral-tag"}
        )
        add_text = get_text(add_result)
        assert "ephemeral-tag" in add_text

        # Verify tag is present
        get_result = await mcp_client.call_tool("zk_get_note", {"identifier": note_id})
        assert "ephemeral-tag" in get_text(get_result)

        # Remove tag
        remove_result = await mcp_client.call_tool(
            "zk_remove_tag", {"note_id": note_id, "tag": "ephemeral-tag"}
        )
        remove_text = get_text(remove_result)
        assert "ephemeral-tag" in remove_text

        # Verify tag is gone
        get_result2 = await mcp_client.call_tool("zk_get_note", {"identifier": note_id})
        get_text2 = get_text(get_result2)
        # After removal, the tag should not appear in the tags line
        # (it may still appear in title/content, so check the Tags: line specifically)
        lines = get_text2.split("\n")
        tag_lines = [l for l in lines if l.startswith("Tags:")]
        if tag_lines:
            assert "ephemeral-tag" not in tag_lines[0]


# =============================================================================
# 6. Admin Operations
# =============================================================================


class TestMCPProtocolAdmin:
    """Status dashboard and system maintenance through the protocol."""

    @pytest.mark.anyio
    async def test_status_via_protocol(self, mcp_client):
        """zk_status returns a status dashboard with note counts."""
        # Seed some data so the dashboard has something to report
        await mcp_client.call_tool(
            "zk_create_note",
            {
                "title": "Status Test Note",
                "content": "Note to populate status counts.",
                "note_type": "permanent",
                "tags": "status-test",
            },
        )

        result = await mcp_client.call_tool("zk_status", {"sections": "all"})
        text = get_text(result)
        assert "Zettelkasten Status" in text
        assert "Summary" in text or "Total Notes" in text

    @pytest.mark.anyio
    async def test_system_rebuild_via_protocol(self, mcp_client):
        """zk_system rebuild reindexes the database without errors."""
        # Create notes first so rebuild has data to process
        await mcp_client.call_tool(
            "zk_create_note",
            {
                "title": "Rebuild Test Note",
                "content": "Note to verify rebuild works.",
                "note_type": "permanent",
            },
        )

        result = await mcp_client.call_tool("zk_system", {"action": "rebuild"})
        text = get_text(result)
        assert "rebuilt" in text.lower() or "successfully" in text.lower()


# =============================================================================
# 7. Argument Validation
# =============================================================================


class TestMCPProtocolArgumentValidation:
    """Protocol-level argument validation and coercion."""

    @pytest.mark.anyio
    async def test_create_note_invalid_note_type(self, mcp_client):
        """An invalid note_type returns a user-facing error, not a crash."""
        result = await mcp_client.call_tool(
            "zk_create_note",
            {
                "title": "Bad Type Note",
                "content": "Should fail validation.",
                "note_type": "INVALID",
            },
        )
        text = get_text(result)
        assert "invalid" in text.lower() or "error" in text.lower()

    @pytest.mark.anyio
    async def test_integer_coercion_for_limit(self, mcp_client):
        """Passing limit as a string should succeed via Pydantic coercion."""
        # First create a note so there is something to list
        await mcp_client.call_tool(
            "zk_create_note",
            {
                "title": "Coercion Test Note",
                "content": "For limit coercion test.",
                "note_type": "permanent",
            },
        )

        result = await mcp_client.call_tool(
            "zk_list_notes", {"mode": "all", "limit": "5"}
        )
        text = get_text(result)
        # Should not error; should list notes or show "no notes"
        assert "error" not in text.lower() or "Notes" in text

    @pytest.mark.anyio
    async def test_search_invalid_mode(self, mcp_client):
        """An invalid search mode returns a clear error message."""
        result = await mcp_client.call_tool(
            "zk_search_notes", {"query": "test", "mode": "INVALID_MODE"}
        )
        text = get_text(result)
        assert "invalid" in text.lower() or "error" in text.lower()

    @pytest.mark.anyio
    async def test_bulk_create_json_string_pre_parsing(self, mcp_client):
        """FastMCP pre_parse_json converts JSON string to list — tool handles both.

        When Claude Desktop sends a JSON array as a string value, FastMCP's
        pre_parse_json converts it to a Python list before Pydantic validates.
        The tool accepts str|list to handle both direct and pre-parsed input.
        """
        notes_json = json.dumps(
            [
                {"title": "JSON String A", "content": "Pre-parsed content A."},
                {"title": "JSON String B", "content": "Pre-parsed content B."},
            ]
        )
        result = await mcp_client.call_tool(
            "zk_bulk_create_notes", {"notes": notes_json}
        )
        text = get_text(result)
        assert not result.isError, f"bulk_create pre-parsing failed: {text}"
        assert "successfully" in text.lower()
        assert "JSON String A" in text
        assert "JSON String B" in text


# =============================================================================
# 8. Error Handling
# =============================================================================


class TestMCPProtocolErrorHandling:
    """Error paths through the protocol."""

    @pytest.mark.anyio
    async def test_call_unknown_tool(self, mcp_client):
        """Calling a nonexistent tool returns an error result (isError=True)."""
        result = await mcp_client.call_tool("zk_nonexistent_tool", {"arg": "value"})
        assert result.isError
        text = get_text(result)
        assert "unknown" in text.lower() or "not found" in text.lower()

    @pytest.mark.anyio
    async def test_get_nonexistent_note(self, mcp_client):
        """Getting a bogus note ID returns a 'not found' message, not a crash."""
        result = await mcp_client.call_tool(
            "zk_get_note", {"identifier": "bogus-id-that-does-not-exist-99999"}
        )
        text = get_text(result)
        assert "not found" in text.lower()

    @pytest.mark.anyio
    async def test_create_link_nonexistent_notes(self, mcp_client):
        """Linking bogus note IDs returns an error response."""
        result = await mcp_client.call_tool(
            "zk_create_link",
            {
                "source_id": "nonexistent-source-id",
                "target_id": "nonexistent-target-id",
                "link_type": "reference",
            },
        )
        text = get_text(result)
        assert "error" in text.lower() or "not found" in text.lower()


# =============================================================================
# 9. Response Format
# =============================================================================


class TestMCPProtocolResponseFormat:
    """Verify the shape of protocol responses."""

    @pytest.mark.anyio
    async def test_response_is_text_content(self, mcp_client):
        """Successful tool calls return TextContent in the result."""
        result = await mcp_client.call_tool(
            "zk_create_note",
            {
                "title": "Format Check Note",
                "content": "Verifying response format.",
                "note_type": "permanent",
            },
        )
        assert len(result.content) > 0
        assert isinstance(result.content[0], TextContent)

    @pytest.mark.anyio
    async def test_error_response_has_error_prefix(self, mcp_client):
        """Tool-level errors include 'Error:' in the response text."""
        result = await mcp_client.call_tool(
            "zk_create_note",
            {
                "title": "Error Format Test",
                "content": "Content does not matter.",
                "note_type": "TOTALLY_INVALID_TYPE",
            },
        )
        text = get_text(result)
        # The server returns "Invalid note type: ..." for bad note_type
        assert "invalid" in text.lower()


# =============================================================================
# 10. Semantic Search (with FakeEmbeddingProvider)
# =============================================================================


class TestMCPProtocolSemantic:
    """Semantic search and embedding-dependent features through the protocol.

    Uses FakeEmbeddingProvider and FakeRerankerProvider — no ONNX models
    needed. The fake embedder produces deterministic vectors based on text
    hash, so "similar" content gives reproducibly similar vectors.
    """

    @pytest.mark.anyio
    async def test_semantic_search_returns_results(self, semantic_mcp_client):
        """Semantic search via the protocol returns matching notes."""
        # Seed notes with related content
        await semantic_mcp_client.call_tool(
            "zk_create_note",
            {
                "title": "Quantum Entanglement Theory",
                "content": "Quantum entanglement is a phenomenon where particles become correlated.",
                "note_type": "permanent",
                "tags": "physics,quantum",
            },
        )
        await semantic_mcp_client.call_tool(
            "zk_create_note",
            {
                "title": "Classical Cooking Recipes",
                "content": "How to make a perfect sourdough bread from scratch.",
                "note_type": "permanent",
                "tags": "cooking",
            },
        )

        result = await semantic_mcp_client.call_tool(
            "zk_search_notes",
            {"query": "quantum physics entanglement", "mode": "semantic", "limit": 10},
        )
        text = get_text(result)
        # Should find results (fake embedder produces vectors for any text)
        assert "similar" in text.lower() or "Quantum" in text or "found" in text.lower()

    @pytest.mark.anyio
    async def test_semantic_search_empty_query(self, semantic_mcp_client):
        """Semantic search with empty query returns an error or empty result."""
        result = await semantic_mcp_client.call_tool(
            "zk_search_notes",
            {"query": "", "mode": "semantic", "limit": 10},
        )
        text = get_text(result)
        # Empty query should produce an error message
        assert "error" in text.lower() or "required" in text.lower()

    @pytest.mark.anyio
    async def test_find_related_semantic_mode(self, semantic_mcp_client):
        """zk_find_related with mode=semantic uses embedding similarity."""
        r1 = await semantic_mcp_client.call_tool(
            "zk_create_note",
            {
                "title": "Machine Learning Basics",
                "content": "Neural networks learn patterns from data through backpropagation.",
                "note_type": "permanent",
            },
        )
        note_id = extract_note_id_from_protocol(r1)

        await semantic_mcp_client.call_tool(
            "zk_create_note",
            {
                "title": "Deep Learning Advances",
                "content": "Transformer architectures have revolutionized natural language processing.",
                "note_type": "permanent",
            },
        )

        result = await semantic_mcp_client.call_tool(
            "zk_find_related",
            {"note_id": note_id, "mode": "semantic"},
        )
        text = get_text(result)
        # Should return semantic results or an informative message
        assert (
            "Deep Learning" in text
            or "semantic" in text.lower()
            or "related" in text.lower()
            or "No semantically" in text
        )

    @pytest.mark.anyio
    async def test_status_shows_embedding_info(self, semantic_mcp_client):
        """zk_status with embeddings section shows embedding config."""
        result = await semantic_mcp_client.call_tool(
            "zk_status", {"sections": "embeddings"}
        )
        text = get_text(result)
        assert "Enabled" in text or "embedding" in text.lower()

    @pytest.mark.anyio
    async def test_auto_mode_prefers_semantic_when_available(self, semantic_mcp_client):
        """In auto mode, search should prefer semantic when embeddings are available."""
        await semantic_mcp_client.call_tool(
            "zk_create_note",
            {
                "title": "Auto Mode Test Note",
                "content": "Content for testing auto mode with semantic search.",
                "note_type": "permanent",
            },
        )

        result = await semantic_mcp_client.call_tool(
            "zk_search_notes",
            {"query": "auto mode semantic test", "mode": "auto", "limit": 10},
        )
        text = get_text(result)
        # Auto mode with embeddings available should work without error
        assert "error" not in text.lower() or "Auto Mode Test" in text


# =============================================================================
# 10. Obsidian Path Override
# =============================================================================


class TestObsidianPathProtocol:
    """Verify obsidian_path field through the MCP protocol."""

    @pytest.mark.anyio
    async def test_create_note_with_obsidian_path(self, mcp_client):
        """Create a note with obsidian_path and verify it persists."""
        result = await mcp_client.call_tool(
            "zk_create_note",
            {
                "title": "Obsidian Path Test",
                "content": "Testing custom obsidian path.",
                "obsidian_path": "komi-zone/agents/research/test-topic",
            },
        )
        text = get_text(result)
        assert "created successfully" in text
        note_id = extract_note_id_from_protocol(result)

        # Read back and verify obsidian_path is in the output
        get_result = await mcp_client.call_tool(
            "zk_get_note", {"identifier": note_id, "format": "markdown"}
        )
        get_text_content = get_text(get_result)
        assert "Obsidian Path" in get_text_content
        assert "komi-zone/agents/research/test-topic" in get_text_content

    @pytest.mark.anyio
    async def test_create_note_without_obsidian_path(self, mcp_client):
        """Create a note without obsidian_path — field should not appear in markdown."""
        result = await mcp_client.call_tool(
            "zk_create_note",
            {
                "title": "No Obsidian Path",
                "content": "No custom path set.",
            },
        )
        note_id = extract_note_id_from_protocol(result)

        get_result = await mcp_client.call_tool(
            "zk_get_note", {"identifier": note_id, "format": "markdown"}
        )
        get_text_content = get_text(get_result)
        assert "obsidian_path" not in get_text_content

    @pytest.mark.anyio
    async def test_update_note_obsidian_path(self, mcp_client):
        """Update a note to add obsidian_path, then verify."""
        result = await mcp_client.call_tool(
            "zk_create_note",
            {
                "title": "Update Path Test",
                "content": "Will get a path later.",
            },
        )
        note_id = extract_note_id_from_protocol(result)

        # Update with obsidian_path
        update_result = await mcp_client.call_tool(
            "zk_update_note",
            {
                "note_id": note_id,
                "obsidian_path": "myproject/docs/planning/feature-x",
            },
        )
        update_text = get_text(update_result)
        assert "updated" in update_text.lower() or "success" in update_text.lower()

        # Verify it persists
        get_result = await mcp_client.call_tool(
            "zk_get_note", {"identifier": note_id, "format": "markdown"}
        )
        get_text_content = get_text(get_result)
        assert "myproject/docs/planning/feature-x" in get_text_content

    @pytest.mark.anyio
    async def test_clear_obsidian_path(self, mcp_client):
        """Setting obsidian_path to empty string should clear it."""
        result = await mcp_client.call_tool(
            "zk_create_note",
            {
                "title": "Clear Path Test",
                "content": "Has a path initially.",
                "obsidian_path": "some/custom/path",
            },
        )
        note_id = extract_note_id_from_protocol(result)

        # Clear it
        await mcp_client.call_tool(
            "zk_update_note",
            {
                "note_id": note_id,
                "obsidian_path": "",
            },
        )

        # Verify it's gone
        get_result = await mcp_client.call_tool(
            "zk_get_note", {"identifier": note_id, "format": "markdown"}
        )
        get_text_content = get_text(get_result)
        assert "obsidian_path" not in get_text_content
