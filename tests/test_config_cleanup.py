"""Tests for config cleanup: user env loading and zk_status config section.

Verifies:
1. User env at ~/.zettelkasten/.env is picked up by config
2. zk_status sections="config" shows current settings and env file status
3. Server works fine when ~/.zettelkasten/.env doesn't exist
"""

from pathlib import Path
from unittest.mock import patch

from tests.conftest_e2e import e2e_mcp_server  # noqa: F401 (pytest fixture)
from tests.conftest_e2e import e2e_session_id  # noqa: F401 (pytest fixture)
from tests.conftest_e2e import e2e_zettel_service  # noqa: F401 (pytest fixture)
from tests.conftest_e2e import isolated_env  # noqa: F401 (pytest fixture)
from tests.conftest_e2e import (
    get_mcp_tool,
)


class TestUserEnvLoading:
    """Test that ~/.zettelkasten/.env is loaded as a config source."""

    def test_user_env_path_is_correct(self):
        """_USER_ENV points to ~/.zettelkasten/.env."""
        from znote_mcp.config import _USER_ENV

        expected = Path.home() / ".zettelkasten" / ".env"
        assert _USER_ENV == expected

    def test_user_env_picked_up_when_present(self, tmp_path, monkeypatch):
        """Config picks up values from a user .env file."""
        from dotenv import load_dotenv

        from znote_mcp.config import ZettelkastenConfig

        # Write a test .env with an obsidian vault path
        user_env = tmp_path / ".env"
        user_env.write_text("ZETTELKASTEN_OBSIDIAN_VAULT=/tmp/test-vault\n")

        # load_dotenv won't override existing env, so clear it first
        monkeypatch.delenv("ZETTELKASTEN_OBSIDIAN_VAULT", raising=False)
        load_dotenv(user_env)

        # Build a fresh config — it reads os.getenv at construction time
        cfg = ZettelkastenConfig()
        assert cfg.obsidian_vault_path == Path("/tmp/test-vault")

    def test_config_works_without_user_env(self, monkeypatch):
        """Config uses defaults when no user .env exists."""
        from znote_mcp.config import ZettelkastenConfig

        # Ensure no obsidian vault is set in env
        monkeypatch.delenv("ZETTELKASTEN_OBSIDIAN_VAULT", raising=False)

        cfg = ZettelkastenConfig()
        assert cfg.obsidian_vault_path is None
        # Core defaults still work
        assert cfg.git_enabled is True
        assert cfg.in_memory_db is True

    def test_load_dotenv_does_not_override_existing(self, tmp_path, monkeypatch):
        """Process env (.mcp.json) takes priority over user .env."""
        from dotenv import load_dotenv

        from znote_mcp.config import ZettelkastenConfig

        # Simulate .mcp.json setting the notes dir via process env
        monkeypatch.setenv("ZETTELKASTEN_NOTES_DIR", "/from/mcp-json")

        # User .env tries to set a different value
        user_env = tmp_path / ".env"
        user_env.write_text("ZETTELKASTEN_NOTES_DIR=/from/user-env\n")
        load_dotenv(user_env)  # won't override

        cfg = ZettelkastenConfig()
        assert cfg.notes_dir == Path("/from/mcp-json")


class TestZkStatusConfigSection:
    """Test the config section of zk_status."""

    def test_config_section_shows_paths(self, e2e_mcp_server):
        """zk_status sections="config" shows notes dir, database, obsidian."""
        status = get_mcp_tool(e2e_mcp_server, "zk_status")
        result = status(sections="config")

        assert "## Configuration" in result
        assert "**Notes Dir:**" in result
        assert "**Database:**" in result
        assert "**Obsidian Vault:**" in result

    def test_config_section_shows_settings(self, e2e_mcp_server):
        """zk_status sections="config" shows git and in-memory settings."""
        status = get_mcp_tool(e2e_mcp_server, "zk_status")
        result = status(sections="config")

        assert "**Git Versioning:**" in result
        assert "**In-Memory DB:**" in result

    def test_config_section_shows_user_env_status(self, e2e_mcp_server):
        """zk_status sections="config" reports user config file status."""
        status = get_mcp_tool(e2e_mcp_server, "zk_status")
        result = status(sections="config")

        assert "**User Config:**" in result
        # Should show either the path or "not found" guidance
        assert ".zettelkasten/.env" in result

    @patch("znote_mcp.server.mcp_server._USER_ENV")
    def test_config_section_when_user_env_missing(self, mock_user_env, e2e_mcp_server):
        """When user .env doesn't exist, status shows 'not found' with guidance."""
        mock_user_env.exists.return_value = False
        mock_user_env.__str__ = lambda _: "~/.zettelkasten/.env"
        mock_user_env.__fspath__ = lambda _: str(Path.home() / ".zettelkasten" / ".env")

        status = get_mcp_tool(e2e_mcp_server, "zk_status")
        result = status(sections="config")

        assert "not found" in result
        assert ".env.example" in result

    @patch("znote_mcp.server.mcp_server._USER_ENV")
    def test_config_section_when_user_env_exists(
        self, mock_user_env, e2e_mcp_server, tmp_path
    ):
        """When user .env exists, status shows the path without 'not found'."""
        user_env = tmp_path / ".env"
        user_env.write_text("# test config\n")

        # Make _USER_ENV behave like a real Path pointing to our temp file
        mock_user_env.exists.return_value = True
        mock_user_env.__str__ = lambda _: str(user_env)
        mock_user_env.__fspath__ = lambda _: str(user_env)

        status = get_mcp_tool(e2e_mcp_server, "zk_status")
        result = status(sections="config")

        assert "**User Config:**" in result
        # Should NOT contain the "not found" message
        assert "not found" not in result.split("**User Config:**")[1].split("\n")[0]

    def test_config_section_included_in_all(self, e2e_mcp_server):
        """zk_status sections="all" includes the config section."""
        status = get_mcp_tool(e2e_mcp_server, "zk_status")
        result = status(sections="all")

        assert "## Configuration" in result
        assert "**Notes Dir:**" in result

    def test_config_section_shows_config_sources(self, e2e_mcp_server):
        """zk_status sections="config" lists which env files were loaded."""
        status = get_mcp_tool(e2e_mcp_server, "zk_status")
        result = status(sections="config")

        assert "**Config Sources:**" in result
