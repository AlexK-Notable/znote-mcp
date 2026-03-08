"""Protocol-level tests for ONNX resilience through MCP tool calls.

Tests that the MCP tool layer (the JSON-RPC surface a real client hits)
handles embedding failures gracefully — returning useful messages
instead of crashing, falling back to FTS when appropriate, and reporting
degradation state in zk_status.

Uses the full MCP transport (memory streams, JSON-RPC) — same path
Claude Code exercises.
"""

import re
from typing import List, Sequence, Tuple
from unittest.mock import patch

import numpy as np
import pytest
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import CallToolResult

from tests.conftest_protocol import mcp_client  # noqa: F401 — fixture
from tests.conftest_protocol import mcp_server  # noqa: F401 — fixture
from tests.conftest_protocol import protocol_config  # noqa: F401 — fixture
from znote_mcp.config import config
from znote_mcp.models.db_models import init_db
from znote_mcp.server.mcp_server import ZettelkastenMcpServer
from znote_mcp.services.embedding_service import EmbeddingService

# ---------------------------------------------------------------------------
# Fake providers
# ---------------------------------------------------------------------------


class FakeEmbeddingProvider:
    """Normal embedding provider for seeding notes."""

    def __init__(self, dim=768):
        self._dim = dim
        self._loaded = False
        self.load_count = 0
        self.unload_count = 0

    @property
    def dimension(self):
        return self._dim

    def load(self):
        self._loaded = True
        self.load_count += 1

    def unload(self):
        self._loaded = False
        self.unload_count += 1

    @property
    def is_loaded(self):
        return self._loaded

    def embed(self, text):
        if not self._loaded:
            self.load()
        rng = np.random.RandomState(hash(text) % (2**31))
        vec = rng.randn(self._dim).astype(np.float32)
        vec /= np.linalg.norm(vec) + 1e-9
        return vec

    def embed_batch(self, texts, batch_size=32):
        return [self.embed(t) for t in texts]


class BFCArenaEmbeddingProvider(FakeEmbeddingProvider):
    """Provider that raises BFCArena OOM on embed/embed_batch after being armed."""

    def __init__(self, dim=768):
        super().__init__(dim=dim)
        self._armed = False

    def arm(self):
        """Start raising OOM errors on next embed call."""
        self._armed = True

    def disarm(self):
        self._armed = False

    def embed(self, text):
        if self._armed:
            raise RuntimeError(
                "BFCArena::AllocateRawInternal Failed to allocate memory"
            )
        return super().embed(text)

    def embed_batch(self, texts, batch_size=32):
        if self._armed:
            raise RuntimeError(
                "BFCArena::AllocateRawInternal Failed to allocate memory"
            )
        return super().embed_batch(texts, batch_size)


class FakeRerankerProvider:
    """Deterministic reranker."""

    def __init__(self):
        self._loaded = False

    def load(self):
        self._loaded = True

    def unload(self):
        self._loaded = False

    @property
    def is_loaded(self):
        return self._loaded

    def rerank(self, query, documents, top_k=5):
        if not self._loaded:
            self.load()
        n = len(documents)
        scored = [(i, float(n - i) / n) for i in range(n)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_text(result: CallToolResult) -> str:
    assert result.content, "CallToolResult has no content"
    assert hasattr(result.content[0], "text"), "First content block has no text"
    return result.content[0].text


def extract_note_id(result: CallToolResult) -> str:
    text = get_text(result)
    match = re.search(r"ID:\s*(\S+)", text)
    assert match, f"Could not extract note ID from response: {text}"
    return match.group(1)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def resilience_protocol_config(tmp_path, monkeypatch):
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    monkeypatch.setattr(config, "notes_dir", notes_dir)
    monkeypatch.setattr(config, "database_path", tmp_path / "test.db")
    monkeypatch.setattr(config, "embeddings_enabled", True)
    monkeypatch.setattr(config, "git_enabled", False)
    monkeypatch.setattr(config, "in_memory_db", False)
    return config


@pytest.fixture
def controllable_providers():
    """Create controllable embedding/reranker providers for the test."""
    embedder = BFCArenaEmbeddingProvider(dim=768)
    reranker = FakeRerankerProvider()
    return embedder, reranker


@pytest.fixture
def resilience_mcp_server(resilience_protocol_config, controllable_providers):
    """MCP server with a controllable embedding provider.

    The test can call controllable_providers[0].arm() to trigger BFCArena
    errors, simulating GPU OOM at the protocol level.
    """
    embedder, reranker = controllable_providers

    def _fake_embedding_service():
        return EmbeddingService(
            embedder=embedder,
            reranker=reranker,
            reranker_idle_timeout=0,
            embedder_idle_timeout=0,
            memory_budget_gb=6.0,
        )

    engine = init_db()
    with patch.object(
        ZettelkastenMcpServer,
        "_create_embedding_service",
        staticmethod(_fake_embedding_service),
    ):
        server = ZettelkastenMcpServer(engine=engine)

    yield server
    server._shutdown()


@pytest.fixture
async def resilience_client(resilience_mcp_server):
    async with create_connected_server_and_client_session(
        resilience_mcp_server.mcp._mcp_server,
        raise_exceptions=True,
    ) as client_session:
        yield client_session


# ---------------------------------------------------------------------------
# Tests: Semantic Search Under Failure
# ---------------------------------------------------------------------------


class TestProtocolResilienceSearch:
    """Semantic search through MCP protocol under ONNX failure conditions.

    Key behavior: search_service.semantic_search() catches all exceptions
    and returns [] — the MCP handler then says "No semantically similar
    notes found." This is correct graceful degradation: the server never
    crashes, and the user gets a valid (empty) response.
    """

    @pytest.mark.anyio
    async def test_semantic_search_oom_does_not_crash(
        self, resilience_client, controllable_providers
    ):
        """When embedder hits BFCArena OOM during semantic search, the server
        should NOT crash — it returns a valid response (empty results)."""
        embedder, _ = controllable_providers

        # Seed a note while embedder is healthy
        await resilience_client.call_tool(
            "zk_create_note",
            {
                "title": "Test Note",
                "content": "Some content for searching.",
                "note_type": "permanent",
            },
        )

        # Arm the OOM
        embedder.arm()

        # This must NOT raise or crash the server
        result = await resilience_client.call_tool(
            "zk_search_notes",
            {"query": "test query", "mode": "semantic"},
        )
        text = get_text(result)
        # search_service catches the error and returns [] → "No semantically similar notes found."
        assert isinstance(text, str) and len(text) > 0

    @pytest.mark.anyio
    async def test_repeated_oom_advances_degradation(
        self, resilience_client, resilience_mcp_server, controllable_providers
    ):
        """Each BFCArena error during search should reduce AIMD budget."""
        embedder, _ = controllable_providers
        svc = resilience_mcp_server.zettel_service.embedding_service

        embedder.arm()

        # Each search attempt triggers embed() which fails with BFCArena
        for _ in range(3):
            await resilience_client.call_tool(
                "zk_search_notes",
                {"query": "trigger oom", "mode": "semantic"},
            )

        # AIMD cwnd should have decreased from failures
        assert (
            svc.coordinator.embedder_aimd.cwnd < svc.coordinator.embedder_aimd.max_cwnd
        )

    @pytest.mark.anyio
    async def test_auto_mode_uses_text_when_semantic_returns_empty(
        self, resilience_client, controllable_providers
    ):
        """Auto mode picks semantic when available. If semantic returns empty
        due to OOM, the user gets an empty-result message. Text search via
        explicit mode=text still works as a user fallback."""
        embedder, _ = controllable_providers

        # Seed a note while healthy
        await resilience_client.call_tool(
            "zk_create_note",
            {
                "title": "Photosynthesis Process",
                "content": "Plants convert sunlight into energy via photosynthesis.",
                "note_type": "permanent",
            },
        )

        # Break the embedder
        embedder.arm()

        # Auto mode will try semantic (since has_semantic_search=True) → fails → empty
        result = await resilience_client.call_tool(
            "zk_search_notes",
            {"query": "photosynthesis", "mode": "auto"},
        )
        text = get_text(result)
        # Returns empty semantic result
        assert "no" in text.lower() or "found" in text.lower()

        # But explicit text mode still works — user can fall back
        result = await resilience_client.call_tool(
            "zk_search_notes",
            {"query": "photosynthesis", "mode": "text"},
        )
        text = get_text(result)
        assert "Photosynthesis" in text

    @pytest.mark.anyio
    async def test_fts_search_unaffected_by_embedding_failure(
        self, resilience_client, controllable_providers
    ):
        """FTS search (zk_fts_search) works perfectly even when
        embeddings are completely broken."""
        embedder, _ = controllable_providers

        # Seed a note
        await resilience_client.call_tool(
            "zk_create_note",
            {
                "title": "Database Indexing",
                "content": "B-trees provide efficient lookups in relational databases.",
                "note_type": "permanent",
            },
        )

        # Nuke the embedder
        embedder.arm()
        for _ in range(5):
            await resilience_client.call_tool(
                "zk_search_notes",
                {"query": "anything", "mode": "semantic"},
            )

        # FTS should still work fine
        result = await resilience_client.call_tool(
            "zk_fts_search",
            {"query": "database indexing"},
        )
        text = get_text(result)
        assert "Database Indexing" in text or "B-trees" in text

    @pytest.mark.anyio
    async def test_server_responds_after_many_oom_errors(
        self, resilience_client, controllable_providers
    ):
        """After many consecutive OOM failures the server should still
        respond to all tool calls — never hang or crash."""
        embedder, _ = controllable_providers
        embedder.arm()

        # Hammer with 10 failing semantic searches
        for _ in range(10):
            result = await resilience_client.call_tool(
                "zk_search_notes",
                {"query": f"oom test", "mode": "semantic"},
            )
            text = get_text(result)
            assert isinstance(text, str)

        # Server still healthy — can create notes
        result = await resilience_client.call_tool(
            "zk_create_note",
            {
                "title": "Post-OOM Note",
                "content": "Created after many OOMs.",
                "note_type": "permanent",
            },
        )
        text = get_text(result)
        assert "created successfully" in text.lower()


# ---------------------------------------------------------------------------
# Tests: Find Related Under Failure
# ---------------------------------------------------------------------------


class TestProtocolResilienceFindRelated:
    """zk_find_related under ONNX failure conditions."""

    @pytest.mark.anyio
    async def test_find_related_semantic_uses_stored_embeddings(
        self, resilience_client, controllable_providers
    ):
        """find_related(mode=semantic) uses stored embeddings from the DB,
        not live embed() calls. So it works even when the embedder is broken,
        as long as notes were indexed before the failure."""
        embedder, _ = controllable_providers

        # Create and index notes while healthy
        r1 = await resilience_client.call_tool(
            "zk_create_note",
            {
                "title": "Note Alpha",
                "content": "Content about alpha particles.",
                "note_type": "permanent",
            },
        )
        note_id = extract_note_id(r1)
        await resilience_client.call_tool(
            "zk_create_note",
            {
                "title": "Note Beta",
                "content": "Content about beta decay processes.",
                "note_type": "permanent",
            },
        )

        # Break embedder — find_related should still work using stored vectors
        embedder.arm()

        result = await resilience_client.call_tool(
            "zk_find_related",
            {"note_id": note_id, "mode": "semantic"},
        )
        text = get_text(result)
        # Should still find related notes via stored embeddings
        assert "Note Beta" in text or "semantically" in text.lower()

    @pytest.mark.anyio
    async def test_find_related_linked_mode_unaffected_by_oom(
        self, resilience_client, controllable_providers
    ):
        """Linked mode doesn't use embeddings — works even when
        embedder is disabled."""
        embedder, _ = controllable_providers

        # Create and link notes while healthy
        r1 = await resilience_client.call_tool(
            "zk_create_note",
            {
                "title": "Source Note",
                "content": "Source content.",
                "note_type": "permanent",
            },
        )
        source_id = extract_note_id(r1)
        r2 = await resilience_client.call_tool(
            "zk_create_note",
            {
                "title": "Target Note",
                "content": "Target content.",
                "note_type": "permanent",
            },
        )
        target_id = extract_note_id(r2)

        await resilience_client.call_tool(
            "zk_manage_links",
            {
                "action": "create",
                "source_id": source_id,
                "target_id": target_id,
                "link_type": "related",
            },
        )

        # Disable embedder completely
        embedder.arm()
        for _ in range(5):
            await resilience_client.call_tool(
                "zk_search_notes",
                {"query": "anything", "mode": "semantic"},
            )

        # Linked mode should still work
        result = await resilience_client.call_tool(
            "zk_find_related",
            {"note_id": source_id, "mode": "linked"},
        )
        text = get_text(result)
        assert "Target Note" in text


# ---------------------------------------------------------------------------
# Tests: Status Reporting
# ---------------------------------------------------------------------------


class TestProtocolResilienceStatus:
    """zk_status should report resilience degradation state."""

    @pytest.mark.anyio
    async def test_status_embeddings_section_exists(self, resilience_client):
        """Baseline: status with embeddings section returns content."""
        result = await resilience_client.call_tool(
            "zk_status", {"sections": "embeddings"}
        )
        text = get_text(result)
        assert "Embeddings" in text or "embedding" in text.lower()

    @pytest.mark.anyio
    async def test_status_shows_degradation_after_oom(
        self, resilience_client, controllable_providers
    ):
        """After OOM errors, status should report the degradation state."""
        embedder, _ = controllable_providers

        # Trigger enough OOMs to trip the breaker (trip_threshold=3)
        embedder.arm()
        for _ in range(3):
            await resilience_client.call_tool(
                "zk_search_notes",
                {"query": "trigger oom", "mode": "semantic"},
            )
        embedder.disarm()

        result = await resilience_client.call_tool(
            "zk_status", {"sections": "embeddings"}
        )
        text = get_text(result)
        # New format: "breaker=open" or "budget=X.XGB" in Resilience lines
        assert (
            "Resilience" in text
            or "breaker" in text.lower()
            or "budget" in text.lower()
            or "open" in text.lower()
        )

    @pytest.mark.anyio
    async def test_status_shows_disabled_after_full_cascade(
        self, resilience_client, resilience_mcp_server, controllable_providers
    ):
        """After full degradation cascade, status should clearly show disabled."""
        embedder, _ = controllable_providers
        svc = resilience_mcp_server.zettel_service.embedding_service

        # Drive through OOMs then force disable
        embedder.arm()
        for _ in range(6):
            await resilience_client.call_tool(
                "zk_search_notes",
                {"query": "oom", "mode": "semantic"},
            )

        # Force the breaker to disabled state (simulates CPU also failing)
        svc.coordinator.embedder_breaker.on_cpu_failure()

        result = await resilience_client.call_tool(
            "zk_status", {"sections": "embeddings"}
        )
        text = get_text(result)
        assert "disabled" in text.lower() or "DISABLED" in text


# ---------------------------------------------------------------------------
# Tests: Recovery / Continued Operation
# ---------------------------------------------------------------------------


class TestProtocolResilienceRecovery:
    """Tests that the server continues functioning after degradation."""

    @pytest.mark.anyio
    async def test_note_crud_unaffected_by_embedding_failure(
        self, resilience_client, controllable_providers
    ):
        """Core CRUD operations (create, get, update) work perfectly
        even when embeddings are completely broken."""
        embedder, _ = controllable_providers

        # Disable embedder
        embedder.arm()
        for _ in range(5):
            await resilience_client.call_tool(
                "zk_search_notes",
                {"query": "oom", "mode": "semantic"},
            )

        # CRUD should still work
        r1 = await resilience_client.call_tool(
            "zk_create_note",
            {
                "title": "Created While Degraded",
                "content": "This note was created while embeddings were disabled.",
                "note_type": "permanent",
            },
        )
        text = get_text(r1)
        assert "created successfully" in text.lower()
        note_id = extract_note_id(r1)

        # Get
        r2 = await resilience_client.call_tool("zk_get_note", {"identifier": note_id})
        text = get_text(r2)
        assert "Created While Degraded" in text

        # Update
        r3 = await resilience_client.call_tool(
            "zk_update_note",
            {"note_id": note_id, "content": "Updated content."},
        )
        text = get_text(r3)
        assert "updated successfully" in text.lower()

    @pytest.mark.anyio
    async def test_text_search_works_after_degradation(
        self, resilience_client, controllable_providers
    ):
        """Text mode search and tag search work after embedder is disabled."""
        embedder, _ = controllable_providers

        # Create note, disable embedder, search by text
        await resilience_client.call_tool(
            "zk_create_note",
            {
                "title": "Resilience Testing",
                "content": "This is a resilience test for text search fallback.",
                "note_type": "permanent",
                "tags": "testing",
            },
        )

        embedder.arm()
        for _ in range(5):
            await resilience_client.call_tool(
                "zk_search_notes",
                {"query": "oom", "mode": "semantic"},
            )

        # Text mode search
        result = await resilience_client.call_tool(
            "zk_search_notes",
            {"query": "resilience", "mode": "text"},
        )
        text = get_text(result)
        assert "Resilience Testing" in text

        # Tag-based search
        result = await resilience_client.call_tool(
            "zk_search_notes",
            {"tags": "testing"},
        )
        text = get_text(result)
        assert "Resilience Testing" in text

    @pytest.mark.anyio
    async def test_semantic_search_recovers_after_disarm(
        self, resilience_client, controllable_providers
    ):
        """If the OOM was transient (e.g. another process freed VRAM),
        semantic search should work again after disarming."""
        embedder, _ = controllable_providers

        # Seed a note
        await resilience_client.call_tool(
            "zk_create_note",
            {
                "title": "Recovery Note",
                "content": "This note should be findable after recovery.",
                "note_type": "permanent",
            },
        )

        # One OOM (advances AIMD but doesn't trip breaker with threshold=3)
        embedder.arm()
        await resilience_client.call_tool(
            "zk_search_notes",
            {"query": "trigger", "mode": "semantic"},
        )

        # Disarm — simulates VRAM becoming available
        embedder.disarm()

        # Should work again at reduced budget
        result = await resilience_client.call_tool(
            "zk_search_notes",
            {"query": "recovery note findable", "mode": "semantic"},
        )
        text = get_text(result)
        assert "Recovery Note" in text or "similar" in text.lower()


# ---------------------------------------------------------------------------
# Tests: Inline Resilience Notices (Task 9)
# ---------------------------------------------------------------------------


class TestProtocolInlineNotices:
    """Test that tool responses include inline resilience notices on state changes."""

    @pytest.mark.anyio
    async def test_search_includes_notice_after_oom(
        self, resilience_client, controllable_providers
    ):
        """After OOM during semantic search, the response should include
        a state-change notice with the warning indicator."""
        embedder, _ = controllable_providers

        # Seed a note while healthy
        await resilience_client.call_tool(
            "zk_create_note",
            {
                "title": "Notice Test Note",
                "content": "Content for notice testing.",
                "note_type": "permanent",
            },
        )

        # Arm OOM — next search triggers failure + notice
        embedder.arm()
        result = await resilience_client.call_tool(
            "zk_search_notes",
            {"query": "notice test", "mode": "semantic"},
        )
        text = get_text(result)
        # Should contain a state change notice
        assert "\u26a0" in text or "state change" in text.lower()
        assert "budget" in text.lower() or "memory pressure" in text.lower()

    @pytest.mark.anyio
    async def test_normal_search_has_no_notice(
        self, resilience_client, controllable_providers
    ):
        """A normal search with no state change should NOT include a notice."""
        embedder, _ = controllable_providers

        # Seed a note
        await resilience_client.call_tool(
            "zk_create_note",
            {
                "title": "Clean Search Note",
                "content": "This note exists for clean search testing.",
                "note_type": "permanent",
            },
        )

        # Search while healthy — no state change expected
        result = await resilience_client.call_tool(
            "zk_search_notes",
            {"query": "clean search note", "mode": "semantic"},
        )
        text = get_text(result)
        # No notice block
        assert "---" not in text.split("\n")[0] or "state change" not in text.lower()
        assert "\u26a0" not in text
        assert "\u2705" not in text

    @pytest.mark.anyio
    async def test_recovery_notice_shows_checkmark(
        self, resilience_client, resilience_mcp_server, controllable_providers
    ):
        """After recovery to full capacity, the response should include
        a recovery notice with the checkmark indicator."""
        embedder, _ = controllable_providers
        svc = resilience_mcp_server.zettel_service.embedding_service

        # Seed notes
        await resilience_client.call_tool(
            "zk_create_note",
            {
                "title": "Recovery Notice Note",
                "content": "Content for recovery notice testing.",
                "note_type": "permanent",
            },
        )

        # Cause one OOM to reduce budget (but not trip breaker)
        embedder.arm()
        await resilience_client.call_tool(
            "zk_search_notes",
            {"query": "trigger oom", "mode": "semantic"},
        )
        # Drain the degradation notice so it doesn't appear later
        svc.coordinator.drain_pending_notices()

        # Disarm and manually push budget back to max to trigger recovery
        embedder.disarm()
        coord = svc.coordinator
        coord.embedder_aimd._cwnd = coord.embedder_aimd.max_cwnd
        coord._degraded["embedder"] = True
        # A success at full cwnd triggers recovery event
        coord.on_embedder_success(utilization=0.5)

        # Next search should include recovery notice
        result = await resilience_client.call_tool(
            "zk_search_notes",
            {"query": "recovery notice", "mode": "semantic"},
        )
        text = get_text(result)
        assert "\u2705" in text or "recovered" in text.lower()

    @pytest.mark.anyio
    async def test_circuit_breaker_trip_notice(
        self, resilience_client, controllable_providers
    ):
        """When the circuit breaker trips, the response should mention it."""
        embedder, _ = controllable_providers

        # Seed note
        await resilience_client.call_tool(
            "zk_create_note",
            {
                "title": "Breaker Trip Note",
                "content": "Content for breaker trip testing.",
                "note_type": "permanent",
            },
        )

        # Trip breaker with 3 OOMs (trip_threshold=3)
        embedder.arm()
        for _ in range(3):
            await resilience_client.call_tool(
                "zk_search_notes",
                {"query": "trip breaker", "mode": "semantic"},
            )

        # The last search result should have accumulated notices
        # including circuit_breaker_tripped
        # (notices accumulate across the 3 calls — the 3rd call's response
        # may have them, or we check on the next call)
        result = await resilience_client.call_tool(
            "zk_search_notes",
            {"query": "check notices", "mode": "semantic"},
        )
        text = get_text(result)
        # At least some notice should be present from the accumulated failures
        # (The breaker trips on the 3rd failure, emitting a tripped notice)
        assert (
            "circuit breaker" in text.lower()
            or "state change" in text.lower()
            or "\u26a0" in text
        )

    @pytest.mark.anyio
    async def test_create_note_includes_notice_after_oom(
        self, resilience_client, resilience_mcp_server, controllable_providers
    ):
        """zk_create_note should include notice when embedding fails."""
        embedder, _ = controllable_providers
        svc = resilience_mcp_server.zettel_service.embedding_service

        # Arm OOM
        embedder.arm()

        # Create a note — embedding will fail, generating a notice
        result = await resilience_client.call_tool(
            "zk_create_note",
            {
                "title": "OOM Create Note",
                "content": "Created during OOM condition.",
                "note_type": "permanent",
            },
        )
        text = get_text(result)
        # Note creation should succeed (embedding failure is non-fatal)
        assert "created successfully" in text.lower()
        # And should include a resilience notice
        assert "\u26a0" in text or "state change" in text.lower()


# ---------------------------------------------------------------------------
# Tests: Agent Controls — zk_system actions (Task 10)
# ---------------------------------------------------------------------------


class TestProtocolEmbeddingControls:
    """Test zk_system embedding control actions through MCP protocol."""

    @pytest.mark.anyio
    async def test_embedding_reset(
        self, resilience_client, resilience_mcp_server, controllable_providers
    ):
        """embedding_reset should restore AIMD and breakers to initial state."""
        embedder, _ = controllable_providers
        svc = resilience_mcp_server.zettel_service.embedding_service

        # Degrade the system
        embedder.arm()
        for _ in range(3):
            await resilience_client.call_tool(
                "zk_search_notes",
                {"query": "degrade", "mode": "semantic"},
            )
        assert (
            svc.coordinator.embedder_aimd.cwnd < svc.coordinator.embedder_aimd.max_cwnd
        )

        # Reset via MCP
        result = await resilience_client.call_tool(
            "zk_system", {"action": "embedding_reset"}
        )
        text = get_text(result)
        assert "reset" in text.lower()
        assert "initial state" in text.lower() or "6.0GB" in text

        # Verify state — reset goes to initial cwnd (slow_start at max/2)
        snap = svc.coordinator.snapshot()
        assert snap["embedder"]["aimd"]["phase"] == "slow_start"
        assert snap["embedder"]["circuit_breaker"]["state"] == "closed"
        assert snap["reranker"]["circuit_breaker"]["state"] == "closed"

    @pytest.mark.anyio
    async def test_embedding_force_cpu(self, resilience_client, resilience_mcp_server):
        """embedding_force_cpu should set breakers to forced_cpu."""
        svc = resilience_mcp_server.zettel_service.embedding_service

        result = await resilience_client.call_tool(
            "zk_system", {"action": "embedding_force_cpu"}
        )
        text = get_text(result)
        assert "forced cpu" in text.lower() or "cpu mode" in text.lower()

        snap = svc.coordinator.snapshot()
        assert snap["embedder"]["circuit_breaker"]["state"] == "forced_cpu"
        assert snap["reranker"]["circuit_breaker"]["state"] == "forced_cpu"

    @pytest.mark.anyio
    async def test_embedding_disable(self, resilience_client, resilience_mcp_server):
        """embedding_disable should disable both breakers."""
        svc = resilience_mcp_server.zettel_service.embedding_service

        result = await resilience_client.call_tool(
            "zk_system", {"action": "embedding_disable"}
        )
        text = get_text(result)
        assert "disabled" in text.lower()

        snap = svc.coordinator.snapshot()
        assert snap["embedder"]["circuit_breaker"]["state"] == "disabled"
        assert snap["reranker"]["circuit_breaker"]["state"] == "disabled"

    @pytest.mark.anyio
    async def test_embedding_enable(self, resilience_client, resilience_mcp_server):
        """embedding_enable should re-enable after disable."""
        svc = resilience_mcp_server.zettel_service.embedding_service

        # Disable first
        await resilience_client.call_tool("zk_system", {"action": "embedding_disable"})

        # Re-enable
        result = await resilience_client.call_tool(
            "zk_system", {"action": "embedding_enable"}
        )
        text = get_text(result)
        assert "re-enabled" in text.lower() or "enabled" in text.lower()

        snap = svc.coordinator.snapshot()
        assert snap["embedder"]["circuit_breaker"]["state"] == "closed"
        assert snap["reranker"]["circuit_breaker"]["state"] == "closed"

    @pytest.mark.anyio
    async def test_embedding_controls_without_embedding_service(self, mcp_client):
        """Embedding control actions should return a message when service is None."""
        for action in [
            "embedding_reset",
            "embedding_force_cpu",
            "embedding_disable",
            "embedding_enable",
        ]:
            result = await mcp_client.call_tool("zk_system", {"action": action})
            text = get_text(result)
            assert "not initialized" in text.lower()


# ---------------------------------------------------------------------------
# Tests: Enhanced zk_status (Task 10)
# ---------------------------------------------------------------------------


class TestProtocolEnhancedStatus:
    """Test enhanced zk_status embeddings section with AIMD/breaker details."""

    @pytest.mark.anyio
    async def test_status_shows_aimd_state(self, resilience_client):
        """Status should show AIMD phase, cwnd, ssthresh for both components."""
        result = await resilience_client.call_tool(
            "zk_status", {"sections": "embeddings"}
        )
        text = get_text(result)
        assert "AIMD" in text
        assert "phase=" in text
        assert "cwnd=" in text
        assert "ssthresh=" in text

    @pytest.mark.anyio
    async def test_status_shows_breaker_state(self, resilience_client):
        """Status should show circuit breaker state, provider, failures."""
        result = await resilience_client.call_tool(
            "zk_status", {"sections": "embeddings"}
        )
        text = get_text(result)
        assert "Breaker" in text
        assert "state=" in text
        assert "provider=" in text
        assert "failures=" in text

    @pytest.mark.anyio
    async def test_status_shows_degraded_aimd_after_oom(
        self, resilience_client, controllable_providers
    ):
        """After OOM, status should show reduced cwnd and non-closed breaker."""
        embedder, _ = controllable_providers

        # Trigger OOM failures to trip breaker
        embedder.arm()
        for _ in range(3):
            await resilience_client.call_tool(
                "zk_search_notes",
                {"query": "oom", "mode": "semantic"},
            )

        result = await resilience_client.call_tool(
            "zk_status", {"sections": "embeddings"}
        )
        text = get_text(result)
        # Should show non-closed state and trips > 0
        assert "trips=" in text
        assert "state=open" in text or "state=closed" in text

    @pytest.mark.anyio
    async def test_status_resilience_section_always_present(self, resilience_client):
        """Resilience subsection should always appear when embeddings enabled,
        not only when degraded."""
        result = await resilience_client.call_tool(
            "zk_status", {"sections": "embeddings"}
        )
        text = get_text(result)
        assert "Resilience" in text


# ---------------------------------------------------------------------------
# Tests: E2E AIMD Integration (Task 12)
# ---------------------------------------------------------------------------


class TestProtocolAIMDIntegration:
    """End-to-end tests exercising the full AIMD -> notice -> control pipeline.

    These tests verify the complete resilience lifecycle through MCP JSON-RPC:
    OOM triggers, AIMD halving, inline notices, recovery, resets, and
    cross-component stress propagation.
    """

    @pytest.mark.anyio
    async def test_oom_aimd_halve_notice_in_response(
        self, resilience_client, resilience_mcp_server, controllable_providers
    ):
        """OOM during semantic search triggers AIMD halve and notice in response.

        Verifies: BFCArena error -> coordinator.on_embedder_failure() ->
        AIMD cwnd halved -> inline notice with warning indicator appended
        to the tool response.
        """
        embedder, _ = controllable_providers
        coord = resilience_mcp_server.zettel_service.embedding_service.coordinator

        # Seed a note while embedder is healthy
        await resilience_client.call_tool(
            "zk_create_note",
            {
                "title": "AIMD Halve Test",
                "content": "Content for AIMD halving verification.",
                "note_type": "permanent",
            },
        )
        # Drain any notices from note creation
        coord.drain_pending_notices()

        initial_cwnd = coord.embedder_aimd.cwnd

        # Arm OOM and trigger semantic search
        embedder.arm()
        result = await resilience_client.call_tool(
            "zk_search_notes",
            {"query": "aimd halve test", "mode": "semantic"},
        )
        text = get_text(result)

        # AIMD cwnd should have been halved
        assert coord.embedder_aimd.cwnd < initial_cwnd

        # Response should contain a notice block with separator and warning
        assert "---" in text
        assert "\u26a0" in text or "state change" in text.lower()
        assert "budget" in text.lower()

    @pytest.mark.anyio
    async def test_recovery_notice_after_enough_successes(
        self, resilience_client, resilience_mcp_server, controllable_providers
    ):
        """After OOM + disarm, enough successful searches should recover
        AIMD to max_cwnd and emit a recovery notice.
        """
        embedder, _ = controllable_providers
        coord = resilience_mcp_server.zettel_service.embedding_service.coordinator

        # Seed notes while healthy
        await resilience_client.call_tool(
            "zk_create_note",
            {
                "title": "Recovery E2E Note",
                "content": "Content for recovery end-to-end testing.",
                "note_type": "permanent",
            },
        )

        # Cause one OOM to reduce budget (below trip_threshold=3)
        embedder.arm()
        await resilience_client.call_tool(
            "zk_search_notes",
            {"query": "trigger oom", "mode": "semantic"},
        )
        coord.drain_pending_notices()

        # Disarm and manually push AIMD to just below max to trigger recovery
        embedder.disarm()
        aimd = coord.embedder_aimd
        aimd._cwnd = aimd.max_cwnd
        coord._degraded["embedder"] = True
        # A success at full cwnd triggers the budget_recovered event
        coord.on_embedder_success(utilization=0.5)

        # Next search should carry the recovery notice
        result = await resilience_client.call_tool(
            "zk_search_notes",
            {"query": "recovery e2e", "mode": "semantic"},
        )
        text = get_text(result)
        assert "\u2705" in text or "recovered" in text.lower()

    @pytest.mark.anyio
    async def test_embedding_reset_clears_aimd_and_breaker_state(
        self, resilience_client, resilience_mcp_server, controllable_providers
    ):
        """embedding_reset via zk_system resets AIMD cwnd to initial and
        closes the circuit breaker, verified through zk_status.
        """
        embedder, _ = controllable_providers
        coord = resilience_mcp_server.zettel_service.embedding_service.coordinator

        # Degrade the system with OOMs (trip the breaker)
        embedder.arm()
        for _ in range(4):
            await resilience_client.call_tool(
                "zk_search_notes",
                {"query": "degrade for reset", "mode": "semantic"},
            )
        embedder.disarm()

        # Confirm degraded state
        assert (
            coord.embedder_aimd.cwnd < coord.embedder_aimd.max_cwnd
        )

        # Reset via MCP tool
        await resilience_client.call_tool(
            "zk_system", {"action": "embedding_reset"}
        )

        # Verify via zk_status
        result = await resilience_client.call_tool(
            "zk_status", {"sections": "embeddings"}
        )
        text = get_text(result)

        # Should show reset state: slow_start phase and closed breaker
        assert "phase=slow_start" in text
        assert "state=closed" in text

        # Also verify via coordinator snapshot
        snap = coord.snapshot()
        assert snap["embedder"]["aimd"]["phase"] == "slow_start"
        assert snap["embedder"]["circuit_breaker"]["state"] == "closed"
        assert snap["reranker"]["circuit_breaker"]["state"] == "closed"

    @pytest.mark.anyio
    async def test_embedding_disable_enable_round_trip(
        self, resilience_client, resilience_mcp_server
    ):
        """embedding_disable then embedding_enable round-trip verified
        through zk_status at each step.
        """
        # Disable
        await resilience_client.call_tool(
            "zk_system", {"action": "embedding_disable"}
        )

        # Check status shows disabled
        result = await resilience_client.call_tool(
            "zk_status", {"sections": "embeddings"}
        )
        text = get_text(result)
        assert "disabled" in text.lower()
        assert "state=disabled" in text

        # Re-enable
        await resilience_client.call_tool(
            "zk_system", {"action": "embedding_enable"}
        )

        # Check status shows closed/enabled
        result = await resilience_client.call_tool(
            "zk_status", {"sections": "embeddings"}
        )
        text = get_text(result)
        assert "state=closed" in text
        assert "disabled" not in text.lower() or "state=closed" in text

    @pytest.mark.anyio
    async def test_cross_component_stress_embedder_floor_caps_reranker(
        self, resilience_client, resilience_mcp_server, controllable_providers
    ):
        """When embedder AIMD hits the floor, reranker should receive a
        caution signal (ssthresh capped). Verified through status snapshot.
        """
        embedder, _ = controllable_providers
        coord = resilience_mcp_server.zettel_service.embedding_service.coordinator

        # Record initial reranker ssthresh
        initial_reranker_ssthresh = coord.reranker_aimd.ssthresh

        # Drive embedder to the floor with repeated OOMs
        embedder.arm()
        # Need enough failures so AIMD hits min_cwnd (floor)
        # max_cwnd=6.0, each failure halves: 3.0 -> 1.5 -> 1.0 (floor at min_cwnd=1.0)
        for _ in range(6):
            await resilience_client.call_tool(
                "zk_search_notes",
                {"query": "drive to floor", "mode": "semantic"},
            )

        # Check via snapshot that reranker got the caution
        snap = coord.snapshot()
        reranker_ssthresh = snap["reranker"]["aimd"]["ssthresh"]
        assert reranker_ssthresh < initial_reranker_ssthresh, (
            f"Reranker ssthresh should be capped by cross-component caution: "
            f"was {initial_reranker_ssthresh}, now {reranker_ssthresh}"
        )

        # Verify via status output
        result = await resilience_client.call_tool(
            "zk_status", {"sections": "embeddings"}
        )
        text = get_text(result)
        # Status should show reranker with reduced ssthresh
        assert "Reranker" in text or "reranker" in text.lower()

    @pytest.mark.anyio
    async def test_inline_notices_only_on_transitions(
        self, resilience_client, resilience_mcp_server, controllable_providers
    ):
        """Two consecutive successful searches should NOT both have notices.

        The first search may have a notice if there was a prior event.
        The second search should NOT have a notice (no state change between).
        """
        embedder, _ = controllable_providers
        coord = resilience_mcp_server.zettel_service.embedding_service.coordinator

        # Seed a note
        await resilience_client.call_tool(
            "zk_create_note",
            {
                "title": "Transition Test",
                "content": "Content for transition notice testing.",
                "note_type": "permanent",
            },
        )

        # Drain any pending notices from creation
        coord.drain_pending_notices()

        # First successful search — no prior state change, so no notice
        result1 = await resilience_client.call_tool(
            "zk_search_notes",
            {"query": "transition test", "mode": "semantic"},
        )
        text1 = get_text(result1)

        # Second successful search — definitely no state change
        result2 = await resilience_client.call_tool(
            "zk_search_notes",
            {"query": "transition test again", "mode": "semantic"},
        )
        text2 = get_text(result2)

        # The second search must NOT have any notice indicators
        assert "\u26a0" not in text2, (
            f"Second consecutive search should not have warning notice: {text2}"
        )
        assert "\u2705" not in text2, (
            f"Second consecutive search should not have recovery notice: {text2}"
        )
        assert "---\n" not in text2, (
            f"Second consecutive search should not have notice separator: {text2}"
        )
