"""E2E test fixtures with explicit isolation from production data.

This module provides fixtures that guarantee complete isolation from
any production Zettelkasten data. Test data is stored in tests/fixtures/
and is completely separate from your actual notes.

Usage:
    # Default: uses temporary directories (cleaned up automatically)
    uv run pytest tests/test_e2e.py -v

    # Persist test data for debugging:
    ZETTELKASTEN_TEST_PERSIST=1 uv run pytest tests/test_e2e.py -v
"""

import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Generator

import pytest

from znote_mcp.config import config
from znote_mcp.models.db_models import init_db
from znote_mcp.server.mcp_server import ZettelkastenMcpServer
from znote_mcp.services.search_service import SearchService
from znote_mcp.services.zettel_service import ZettelService

# Fixture paths - explicitly within the test directory
FIXTURES_DIR = Path(__file__).parent / "fixtures"
PERSIST_MODE = os.environ.get("ZETTELKASTEN_TEST_PERSIST", "").lower() in (
    "1",
    "true",
    "yes",
)


class IsolatedTestEnvironment:
    """Manages a completely isolated test environment.

    Ensures test data never touches production data by using
    explicit paths within the tests/fixtures/ directory.
    """

    def __init__(self, persist: bool = False, session_id: str = ""):
        """Initialize isolated environment.

        Args:
            persist: If True, keep test data after teardown (for debugging)
            session_id: Unique ID for this test session
        """
        self.persist = persist
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")

        if persist:
            # Use persistent paths in fixtures directory
            self.base_dir = FIXTURES_DIR
            self.notes_dir = self.base_dir / "notes" / self.session_id
            self.database_dir = self.base_dir / "database"
            self.obsidian_dir = self.base_dir / "obsidian_vault" / self.session_id
        else:
            # Use truly temporary directories (auto-cleaned)
            self._temp_base = tempfile.mkdtemp(prefix="zk_e2e_")
            self.base_dir = Path(self._temp_base)
            self.notes_dir = self.base_dir / "notes"
            self.database_dir = self.base_dir / "database"
            self.obsidian_dir = self.base_dir / "obsidian_vault"

        self.database_path = self.database_dir / f"test_{self.session_id}.db"

    def get_info(self) -> dict:
        """Get information about the test environment."""
        return {
            "session_id": self.session_id,
            "persist": self.persist,
            "notes_dir": str(self.notes_dir),
            "database_path": str(self.database_path),
            "obsidian_dir": str(self.obsidian_dir),
            "notes_count": (
                len(list(self.notes_dir.glob("**/*.md")))
                if self.notes_dir.exists()
                else 0
            ),
            "db_exists": self.database_path.exists(),
        }


@pytest.fixture(scope="session")
def e2e_session_id() -> str:
    """Generate a unique session ID for this test run."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


@pytest.fixture(scope="function")
def isolated_env(
    e2e_session_id, monkeypatch
) -> Generator[IsolatedTestEnvironment, None, None]:
    """Provide a completely isolated test environment.

    Uses monkeypatch for config mutation so values are auto-restored
    on teardown, even if the test crashes.

    This fixture guarantees that:
    - All test data is in tests/fixtures/ (not your production data)
    - Each test gets a fresh database
    - Obsidian vault sync is tested in isolation
    - Environment is cleaned up after tests (unless PERSIST=1)

    Usage:
        def test_something(isolated_env):
            # isolated_env.notes_dir, isolated_env.database_path, etc.
            # are all set up and config is pointing to them
            pass
    """
    env = IsolatedTestEnvironment(
        persist=PERSIST_MODE, session_id=f"{e2e_session_id}_{os.getpid()}"
    )

    # Create directories
    env.notes_dir.mkdir(parents=True, exist_ok=True)
    env.database_dir.mkdir(parents=True, exist_ok=True)
    env.obsidian_dir.mkdir(parents=True, exist_ok=True)

    # Use monkeypatch for automatic restoration on teardown
    monkeypatch.setattr(config, "notes_dir", env.notes_dir)
    monkeypatch.setattr(config, "database_path", env.database_path)
    monkeypatch.setattr(config, "obsidian_vault_path", env.obsidian_dir)
    monkeypatch.setenv("ZETTELKASTEN_OBSIDIAN_VAULT", str(env.obsidian_dir))

    yield env

    # Clean up temp files unless persist mode
    if not env.persist and hasattr(env, "_temp_base"):
        shutil.rmtree(env._temp_base, ignore_errors=True)


@pytest.fixture(scope="function")
def e2e_zettel_service(isolated_env) -> Generator[ZettelService, None, None]:
    """Provide an isolated ZettelService for E2E testing."""
    # Initialize database with full schema
    init_db()

    # Create service
    service = ZettelService()

    yield service


@pytest.fixture(scope="function")
def e2e_search_service(e2e_zettel_service) -> SearchService:
    """Provide an isolated SearchService for E2E testing."""
    return SearchService(e2e_zettel_service)


@pytest.fixture(scope="function")
def e2e_mcp_server(e2e_zettel_service) -> ZettelkastenMcpServer:
    """Provide an isolated MCP server for E2E testing."""
    server = ZettelkastenMcpServer()
    # Replace service with our isolated one
    server.zettel_service = e2e_zettel_service
    server.search_service = SearchService(e2e_zettel_service)
    return server


def get_mcp_tool(server: ZettelkastenMcpServer, tool_name: str):
    """Helper to get a tool function from the MCP server."""
    tool = server.mcp._tool_manager.get_tool(tool_name)
    if tool is None:
        raise ValueError(f"Tool '{tool_name}' not found")
    return tool.fn
