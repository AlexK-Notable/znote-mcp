"""Protocol-level test fixtures for MCP integration testing.

Provides fixtures that exercise the full MCP pipeline using in-process
memory transport. Tests using these fixtures communicate with the server
through the actual MCP protocol (JSON-RPC over memory streams), not by
calling Python methods directly.

Usage:
    @pytest.mark.anyio
    async def test_tool_via_protocol(mcp_client):
        result = await mcp_client.call_tool("zk_create_note", {...})
        assert not result.isError
        text = get_text(result)

Semantic fixtures (requires numpy + sqlite-vec):
    @pytest.mark.anyio
    async def test_semantic_via_protocol(semantic_mcp_client):
        # Uses FakeEmbeddingProvider — no ONNX models needed
        result = await semantic_mcp_client.call_tool(
            "zk_search_notes", {"query": "quantum", "mode": "semantic"}
        )
"""

import re
from typing import List, Sequence, Tuple
from unittest.mock import patch

import numpy as np
import pytest
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import CallToolResult

from znote_mcp.config import config
from znote_mcp.models.db_models import init_db
from znote_mcp.server.mcp_server import ZettelkastenMcpServer
from znote_mcp.services.embedding_service import EmbeddingService


@pytest.fixture
def protocol_config(tmp_path, monkeypatch):
    """Configure the global config for protocol-level testing.

    Creates an isolated environment with a temp notes directory and
    file-based SQLite database. Embeddings and git are disabled for
    speed and determinism.
    """
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()

    monkeypatch.setattr(config, "notes_dir", notes_dir)
    monkeypatch.setattr(config, "database_path", tmp_path / "test.db")
    monkeypatch.setattr(config, "embeddings_enabled", False)
    monkeypatch.setattr(config, "git_enabled", False)
    monkeypatch.setattr(config, "in_memory_db", False)

    return config


@pytest.fixture
def mcp_server(protocol_config):
    """Create a real ZettelkastenMcpServer backed by a full-schema database.

    Initialises the database (tables, FTS5, migrations) and passes the
    engine to the server so all repositories share one connection pool.
    """
    engine = init_db()
    server = ZettelkastenMcpServer(engine=engine)

    yield server

    server._shutdown()


@pytest.fixture
async def mcp_client(mcp_server):
    """Provide a ClientSession connected to the server via memory transport.

    The session is fully initialised (handshake complete) and ready for
    ``call_tool``, ``list_tools``, etc.
    """
    async with create_connected_server_and_client_session(
        mcp_server.mcp._mcp_server,
        raise_exceptions=True,
    ) as client_session:
        yield client_session


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def get_text(result: CallToolResult) -> str:
    """Extract the text payload from a CallToolResult.

    Asserts that the result contains at least one content block with text.
    """
    assert result.content, "CallToolResult has no content"
    assert hasattr(result.content[0], "text"), "First content block has no text"
    return result.content[0].text


def extract_note_id_from_protocol(result: CallToolResult) -> str:
    """Extract the note ID from a create_note protocol response.

    Parses the standard success message format:
        "Note created successfully with ID: <id> (project: <project>)"
    """
    text = get_text(result)
    match = re.search(r"ID:\s*(\S+)", text)
    assert match, f"Could not extract note ID from response: {text}"
    return match.group(1)


# ---------------------------------------------------------------------------
# Fake providers for semantic search testing (no ONNX models needed)
# ---------------------------------------------------------------------------


class FakeEmbeddingProvider:
    """Deterministic embedding provider for protocol tests.

    Produces reproducible 768-dim vectors based on text hash.
    Satisfies the EmbeddingProvider protocol via structural subtyping.
    """

    def __init__(self, dim: int = 768):
        self._dim = dim
        self._loaded = False

    @property
    def dimension(self) -> int:
        return self._dim

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def embed(self, text: str) -> np.ndarray:
        self.load()
        rng = np.random.RandomState(hash(text) % (2**31))
        vec = rng.randn(self._dim).astype(np.float32)
        vec /= np.linalg.norm(vec) + 1e-9
        return vec

    def embed_batch(
        self, texts: Sequence[str], batch_size: int = 32
    ) -> List[np.ndarray]:
        return [self.embed(t) for t in texts]


class FakeRerankerProvider:
    """Deterministic reranker for protocol tests.

    Returns documents in reverse order (last doc scores highest) to
    provide a visible reranking effect in tests.
    """

    def __init__(self):
        self._loaded = False

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def rerank(
        self, query: str, documents: Sequence[str], top_k: int = 5
    ) -> List[Tuple[int, float]]:
        self.load()
        n = len(documents)
        scored = [(i, float(n - i) / n) for i in range(n)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


# ---------------------------------------------------------------------------
# Semantic protocol fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def semantic_protocol_config(tmp_path, monkeypatch):
    """Configure the global config for semantic protocol testing.

    Like protocol_config but with embeddings_enabled=True.
    """
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()

    monkeypatch.setattr(config, "notes_dir", notes_dir)
    monkeypatch.setattr(config, "database_path", tmp_path / "test.db")
    monkeypatch.setattr(config, "embeddings_enabled", True)
    monkeypatch.setattr(config, "git_enabled", False)
    monkeypatch.setattr(config, "in_memory_db", False)

    return config


@pytest.fixture
def semantic_mcp_server(semantic_protocol_config):
    """Create a ZettelkastenMcpServer with fake embedding providers.

    Patches _create_embedding_service to inject FakeEmbeddingProvider
    and FakeRerankerProvider into a real EmbeddingService.
    """

    def _fake_embedding_service():
        return EmbeddingService(
            embedder=FakeEmbeddingProvider(dim=768),
            reranker=FakeRerankerProvider(),
            reranker_idle_timeout=0,
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
async def semantic_mcp_client(semantic_mcp_server):
    """Provide a ClientSession connected to the semantic-enabled server."""
    async with create_connected_server_and_client_session(
        semantic_mcp_server.mcp._mcp_server,
        raise_exceptions=True,
    ) as client_session:
        yield client_session
