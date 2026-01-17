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
from typing import Generator, Tuple

import pytest
from sqlalchemy import create_engine

from znote_mcp.config import config
from znote_mcp.models.db_models import Base, init_db
from znote_mcp.server.mcp_server import ZettelkastenMcpServer
from znote_mcp.services.search_service import SearchService
from znote_mcp.services.zettel_service import ZettelService
from znote_mcp.storage.note_repository import NoteRepository


# Fixture paths - explicitly within the test directory
FIXTURES_DIR = Path(__file__).parent / "fixtures"
PERSIST_MODE = os.environ.get("ZETTELKASTEN_TEST_PERSIST", "").lower() in ("1", "true", "yes")


class IsolatedTestEnvironment:
    """Manages a completely isolated test environment.

    Ensures test data never touches production data by using
    explicit paths within the tests/fixtures/ directory.
    """

    def __init__(self, persist: bool = False, session_id: str = None):
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

        # Track original config for restoration
        self._original_notes_dir = None
        self._original_database_path = None
        self._original_obsidian_vault = None

    def setup(self) -> "IsolatedTestEnvironment":
        """Set up the isolated environment."""
        # Create directories
        self.notes_dir.mkdir(parents=True, exist_ok=True)
        self.database_dir.mkdir(parents=True, exist_ok=True)
        self.obsidian_dir.mkdir(parents=True, exist_ok=True)

        # Save original config
        self._original_notes_dir = config.notes_dir
        self._original_database_path = config.database_path
        self._original_obsidian_vault = os.environ.get("ZETTELKASTEN_OBSIDIAN_VAULT")
        self._original_obsidian_vault_path = config.obsidian_vault_path

        # Configure for isolated testing
        config.notes_dir = self.notes_dir
        config.database_path = self.database_path
        # Set both the env var AND the config attribute (config is already loaded)
        os.environ["ZETTELKASTEN_OBSIDIAN_VAULT"] = str(self.obsidian_dir)
        config.obsidian_vault_path = self.obsidian_dir

        return self

    def teardown(self) -> None:
        """Tear down the isolated environment."""
        # Restore original config
        if self._original_notes_dir is not None:
            config.notes_dir = self._original_notes_dir
        if self._original_database_path is not None:
            config.database_path = self._original_database_path
        if hasattr(self, "_original_obsidian_vault_path"):
            config.obsidian_vault_path = self._original_obsidian_vault_path
        if self._original_obsidian_vault is not None:
            os.environ["ZETTELKASTEN_OBSIDIAN_VAULT"] = self._original_obsidian_vault
        elif "ZETTELKASTEN_OBSIDIAN_VAULT" in os.environ:
            del os.environ["ZETTELKASTEN_OBSIDIAN_VAULT"]

        # Clean up unless persist mode
        if not self.persist and hasattr(self, "_temp_base"):
            try:
                shutil.rmtree(self._temp_base, ignore_errors=True)
            except Exception:
                pass

    def __enter__(self) -> "IsolatedTestEnvironment":
        return self.setup()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.teardown()

    def get_info(self) -> dict:
        """Get information about the test environment."""
        return {
            "session_id": self.session_id,
            "persist": self.persist,
            "notes_dir": str(self.notes_dir),
            "database_path": str(self.database_path),
            "obsidian_dir": str(self.obsidian_dir),
            "notes_count": len(list(self.notes_dir.glob("**/*.md"))) if self.notes_dir.exists() else 0,
            "db_exists": self.database_path.exists(),
        }


@pytest.fixture(scope="session")
def e2e_session_id() -> str:
    """Generate a unique session ID for this test run."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


@pytest.fixture(scope="function")
def isolated_env(e2e_session_id) -> Generator[IsolatedTestEnvironment, None, None]:
    """Provide a completely isolated test environment.

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
        persist=PERSIST_MODE,
        session_id=f"{e2e_session_id}_{os.getpid()}"
    )
    with env:
        yield env


@pytest.fixture(scope="function")
def e2e_zettel_service(isolated_env) -> Generator[ZettelService, None, None]:
    """Provide an isolated ZettelService for E2E testing."""
    # Initialize database with full schema
    init_db()

    # Create service
    service = ZettelService()
    service.initialize()

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
