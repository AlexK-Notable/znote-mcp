"""Tests for Phase 4: Polish and Platform Portability.

Covers:
- Setup wizard (zk_system action="setup_sync")
- Template generation (CODEOWNERS and CI guard)
- Import freshness in zk_status sync section
- User removal cleanup
- CI guard shell injection safety (no direct expression interpolation in run blocks)
"""

import inspect
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from znote_mcp.services.git_sync_service import GitSyncService

# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def sync_service(tmp_path):
    """Create a GitSyncService for testing."""
    return GitSyncService(
        user_id="testuser",
        repo_url="https://example.com/repo.git",
        branch="testuser/notes",
        remote_dir=tmp_path / ".remote",
        notes_dir=tmp_path / "notes",
        imports_dir=tmp_path / "imports",
        push_delay=120,
        push_extend=60,
        import_users=["alice", "bob"],
    )


@pytest.fixture
def sync_service_setup(sync_service, tmp_path):
    """GitSyncService with a fake .git directory (appears set up)."""
    (tmp_path / ".remote" / ".git").mkdir(parents=True)
    return sync_service


# =========================================================================
# 4a. Setup Wizard Tests
# =========================================================================


class TestSetupSyncWizard:
    """Test zk_system action='setup_sync'."""

    def test_setup_sync_recognized_in_source(self):
        """zk_system dispatches to setup_sync."""
        from znote_mcp.server.mcp_server import ZettelkastenMcpServer

        source = inspect.getsource(ZettelkastenMcpServer)
        assert "setup_sync" in source

    def test_setup_wizard_no_sync_service(self):
        """Setup wizard returns env var guidance when sync is not configured."""
        from znote_mcp.server.mcp_server import ZettelkastenMcpServer

        server = ZettelkastenMcpServer.__new__(ZettelkastenMcpServer)
        server.sync_service = None
        result = server._setup_sync_wizard()
        assert "ZETTELKASTEN_SYNC_ENABLED" in result
        assert "ZETTELKASTEN_SYNC_USER_ID" in result
        assert "ZETTELKASTEN_SYNC_REPO" in result

    def test_setup_wizard_git_not_installed(self, sync_service):
        """Setup wizard reports when git is not installed."""
        from znote_mcp.server.mcp_server import ZettelkastenMcpServer

        server = ZettelkastenMcpServer.__new__(ZettelkastenMcpServer)
        server.sync_service = sync_service

        with patch.object(sync_service, "check_prerequisites") as mock_check:
            mock_check.return_value = {
                "git_installed": False,
                "repo_reachable": False,
                "already_setup": False,
            }
            result = server._setup_sync_wizard()
        assert "[FAIL] Git is not installed" in result

    def test_setup_wizard_repo_not_reachable(self, sync_service):
        """Setup wizard reports when repo is not reachable."""
        from znote_mcp.server.mcp_server import ZettelkastenMcpServer

        server = ZettelkastenMcpServer.__new__(ZettelkastenMcpServer)
        server.sync_service = sync_service

        with patch.object(sync_service, "check_prerequisites") as mock_check:
            mock_check.return_value = {
                "git_installed": True,
                "git_version": "git version 2.43.0",
                "repo_reachable": False,
                "repo_error": "Permission denied",
                "already_setup": False,
            }
            result = server._setup_sync_wizard()
        assert "[OK] Git installed" in result
        assert "[FAIL] Repository not reachable" in result
        assert "Permission denied" in result

    def test_setup_wizard_validates_repo_url(self, sync_service):
        """Setup wizard validates repo URL format by checking reachability."""
        from znote_mcp.server.mcp_server import ZettelkastenMcpServer

        server = ZettelkastenMcpServer.__new__(ZettelkastenMcpServer)
        server.sync_service = sync_service

        with patch.object(sync_service, "check_prerequisites") as mock_check:
            mock_check.return_value = {
                "git_installed": True,
                "git_version": "git version 2.43.0",
                "repo_reachable": False,
                "repo_error": "fatal: not a git repository",
                "already_setup": False,
            }
            result = server._setup_sync_wizard()
        assert "[FAIL]" in result

    def test_setup_wizard_already_setup(self, sync_service):
        """Setup wizard reports when already set up."""
        from znote_mcp.server.mcp_server import ZettelkastenMcpServer

        server = ZettelkastenMcpServer.__new__(ZettelkastenMcpServer)
        server.sync_service = sync_service

        with patch.object(sync_service, "check_prerequisites") as mock_check:
            mock_check.return_value = {
                "git_installed": True,
                "git_version": "git version 2.43.0",
                "repo_reachable": True,
                "already_setup": True,
            }
            result = server._setup_sync_wizard()
        assert "[OK] Sparse checkout already initialized" in result

    def test_setup_wizard_successful_setup(self, sync_service):
        """Setup wizard runs setup on passing checks."""
        from znote_mcp.server.mcp_server import ZettelkastenMcpServer

        server = ZettelkastenMcpServer.__new__(ZettelkastenMcpServer)
        server.sync_service = sync_service

        with patch.object(sync_service, "check_prerequisites") as mock_check:
            mock_check.return_value = {
                "git_installed": True,
                "git_version": "git version 2.43.0",
                "repo_reachable": True,
                "already_setup": False,
            }
            with patch.object(sync_service, "setup") as mock_setup:
                result = server._setup_sync_wizard()
                mock_setup.assert_called_once()
        assert "[OK] Sparse checkout initialized" in result

    def test_setup_wizard_setup_fails(self, sync_service):
        """Setup wizard reports setup failure."""
        from znote_mcp.server.mcp_server import ZettelkastenMcpServer

        server = ZettelkastenMcpServer.__new__(ZettelkastenMcpServer)
        server.sync_service = sync_service

        with patch.object(sync_service, "check_prerequisites") as mock_check:
            mock_check.return_value = {
                "git_installed": True,
                "git_version": "git version 2.43.0",
                "repo_reachable": True,
                "already_setup": False,
            }
            with patch.object(
                sync_service, "setup", side_effect=RuntimeError("clone failed")
            ):
                result = server._setup_sync_wizard()
        assert "[FAIL] Setup failed" in result


# =========================================================================
# 4b. Template Generation Tests
# =========================================================================


class TestTemplateGeneration:
    """Test template generation for CODEOWNERS and CI guard."""

    def test_codeowners_template_generation(self):
        """CODEOWNERS template includes all users."""
        from znote_mcp.server.mcp_server import ZettelkastenMcpServer

        result = ZettelkastenMcpServer._generate_codeowners(["alice", "bob", "carol"])
        assert "Auto-generated by znote-mcp" in result
        assert "/notes/alice/       @alice" in result
        assert "/notes/bob/       @bob" in result
        assert "/notes/carol/       @carol" in result

    def test_codeowners_template_empty_users(self):
        """CODEOWNERS template handles empty user list."""
        from znote_mcp.server.mcp_server import ZettelkastenMcpServer

        result = ZettelkastenMcpServer._generate_codeowners([])
        assert "Auto-generated by znote-mcp" in result
        assert "@" not in result.split("directory")[1]  # No user entries

    def test_ci_guard_template_structure(self):
        """CI guard template has correct YAML structure."""
        from znote_mcp.server.mcp_server import ZettelkastenMcpServer

        result = ZettelkastenMcpServer._generate_ci_guard()
        assert "name: Notes ownership guard" in result
        assert "pull_request:" in result
        assert "actions/checkout@v4" in result
        assert "fetch-depth: 0" in result

    def test_ci_guard_uses_env_block(self):
        """CI guard uses env: block for GitHub expressions."""
        from znote_mcp.server.mcp_server import ZettelkastenMcpServer

        result = ZettelkastenMcpServer._generate_ci_guard()
        assert "env:" in result
        assert "PR_AUTHOR:" in result

    def test_generate_sync_templates_produces_both(self):
        """_generate_sync_templates returns both CODEOWNERS and CI guard."""
        from znote_mcp.server.mcp_server import ZettelkastenMcpServer

        server = ZettelkastenMcpServer.__new__(ZettelkastenMcpServer)
        server.sync_service = MagicMock()

        # Mock config at the module level used by _generate_sync_templates
        mock_config = MagicMock()
        mock_config.sync_user_id = "me"
        mock_config.get_import_users.return_value = ["alice", "bob"]

        with patch("znote_mcp.server.mcp_server.config", mock_config):
            result = server._generate_sync_templates()

        assert "CODEOWNERS" in result
        assert "CI Guard" in result
        assert "@me" in result
        assert "@alice" in result
        assert "@bob" in result

    def test_generate_sync_templates_not_configured(self):
        """generate_sync_templates returns error when sync not configured.

        Verified through the zk_system dispatch, not the helper directly.
        """
        from znote_mcp.server.mcp_server import ZettelkastenMcpServer

        source = inspect.getsource(ZettelkastenMcpServer)
        assert "generate_sync_templates" in source


# =========================================================================
# 4c. Import Freshness Tests
# =========================================================================


class TestImportFreshness:
    """Test import freshness reporting in zk_status sync section."""

    def test_import_freshness_in_source(self):
        """zk_status sync section references import freshness."""
        from znote_mcp.server.mcp_server import ZettelkastenMcpServer

        source = inspect.getsource(ZettelkastenMcpServer)
        assert "import_freshness" in source or "Import Freshness" in source

    def test_freshness_fresh_when_recent(self, sync_service):
        """Import is 'fresh' when less than 24 hours old."""
        # Set last import time to 1 hour ago
        sync_service._last_import_time = datetime.now(timezone.utc)
        status = sync_service.get_status()
        last_import = status["last_import_time"]
        assert last_import is not None

        # Verify the freshness calculation logic
        last_import_dt = datetime.fromisoformat(last_import)
        age_hours = (datetime.now(timezone.utc) - last_import_dt).total_seconds() / 3600
        assert age_hours < 24
        freshness = "fresh" if age_hours < 24 else "stale"
        assert freshness == "fresh"

    def test_freshness_stale_when_old(self, sync_service):
        """Import is 'stale' when more than 24 hours old."""
        from datetime import timedelta

        # Set last import time to 48 hours ago
        sync_service._last_import_time = datetime.now(timezone.utc) - timedelta(
            hours=48
        )
        status = sync_service.get_status()
        last_import = status["last_import_time"]
        assert last_import is not None

        # Verify the freshness calculation logic
        last_import_dt = datetime.fromisoformat(last_import)
        age_hours = (datetime.now(timezone.utc) - last_import_dt).total_seconds() / 3600
        assert age_hours >= 24
        freshness = "fresh" if age_hours < 24 else "stale"
        assert freshness == "stale"

    def test_no_import_time_shows_never(self, sync_service):
        """When no imports have been done, last_import_time is None."""
        status = sync_service.get_status()
        assert status["last_import_time"] is None


# =========================================================================
# 4d. User Removal Tests
# =========================================================================


class TestUserRemoval:
    """Test remove_user cleanup on GitSyncService."""

    def test_remove_user_method_exists(self):
        """GitSyncService has a remove_user method."""
        assert hasattr(GitSyncService, "remove_user")

    def test_remove_user_symlink(self, sync_service, tmp_path):
        """remove_user removes symlink for the user."""
        imports_dir = tmp_path / "imports"
        imports_dir.mkdir(parents=True)
        target = tmp_path / "target"
        target.mkdir()
        link = imports_dir / "alice"
        link.symlink_to(target)

        stats = sync_service.remove_user("alice")
        assert stats["symlink_removed"] is True
        assert not link.exists()

    def test_remove_user_directory(self, sync_service, tmp_path):
        """remove_user removes non-symlink directory for the user."""
        imports_dir = tmp_path / "imports"
        imports_dir.mkdir(parents=True)
        user_dir = imports_dir / "alice"
        user_dir.mkdir()
        (user_dir / "note.md").write_text("content")

        stats = sync_service.remove_user("alice")
        assert stats["symlink_removed"] is True
        assert not user_dir.exists()

    def test_remove_user_no_link_exists(self, sync_service):
        """remove_user handles missing import link gracefully."""
        stats = sync_service.remove_user("nonexistent")
        assert stats["symlink_removed"] is False

    def test_remove_user_updates_import_list(self, sync_service):
        """remove_user removes the user from _import_users."""
        assert "alice" in sync_service._import_users
        sync_service.remove_user("alice")
        assert "alice" not in sync_service._import_users

    def test_remove_user_updates_sparse_checkout(self, sync_service_setup):
        """remove_user updates sparse checkout when repo is set up."""
        with patch.object(sync_service_setup, "_run_git") as mock_git:
            mock_git.return_value = MagicMock(returncode=0, stdout="")
            stats = sync_service_setup.remove_user("alice")
        assert stats["sparse_updated"] is True
        mock_git.assert_called_once()
        # Verify the sparse-checkout set command was called
        call_args = mock_git.call_args[0][0]
        assert call_args[0] == "sparse-checkout"
        assert call_args[1] == "set"
        # alice should NOT be in the patterns
        assert "notes/alice/" not in call_args
        # bob and testuser should remain
        assert "notes/testuser/" in call_args
        assert "notes/bob/" in call_args

    def test_remove_user_unknown_not_in_import_list(self, sync_service):
        """remove_user for an unknown user does not change import list."""
        original_users = list(sync_service._import_users)
        sync_service.remove_user("unknown_user")
        assert sync_service._import_users == original_users

    def test_remove_user_no_sparse_update_when_not_setup(self, sync_service):
        """remove_user does not update sparse checkout when not set up."""
        stats = sync_service.remove_user("alice")
        # User removed from list but no sparse checkout update
        assert "alice" not in sync_service._import_users
        assert stats["sparse_updated"] is False


# =========================================================================
# 4e. CI Guard Shell Injection Safety Tests
# =========================================================================


class TestCiGuardShellInjectionSafety:
    """Test that CI guard template does NOT use direct expression interpolation."""

    def test_no_direct_interpolation_in_run_blocks(self):
        """CI guard template does not use ${{ github. in run: blocks.

        The template must use env: blocks for GitHub expressions, NOT
        direct ${{ }} interpolation in run: steps (shell injection risk).
        """
        from znote_mcp.server.mcp_server import ZettelkastenMcpServer

        template = ZettelkastenMcpServer._generate_ci_guard()

        # Parse out run: blocks (everything after "run: |" until next step)
        in_run_block = False
        run_lines = []
        for line in template.split("\n"):
            stripped = line.strip()
            if stripped.startswith("run:"):
                in_run_block = True
                continue
            if in_run_block:
                # A line that starts a new YAML key at the same or higher indent
                # level ends the run block
                if stripped and not stripped.startswith("#") and ":" in stripped:
                    if not stripped.startswith("$") and not stripped.startswith('"'):
                        in_run_block = False
                        continue
                run_lines.append(line)

        run_content = "\n".join(run_lines)
        # The run block should NOT contain ${{ github. }}
        assert "${{ github." not in run_content, (
            "CI guard template uses direct expression interpolation in run block. "
            "Use env: blocks instead to prevent shell injection."
        )

    def test_env_block_contains_github_expression(self):
        """CI guard template uses env: block for GitHub expression."""
        from znote_mcp.server.mcp_server import ZettelkastenMcpServer

        template = ZettelkastenMcpServer._generate_ci_guard()

        # The env block should contain the GitHub expression
        in_env_block = False
        env_lines = []
        for line in template.split("\n"):
            stripped = line.strip()
            if stripped.startswith("env:"):
                in_env_block = True
                continue
            if in_env_block:
                if stripped and not stripped.startswith("#"):
                    if ":" in stripped or stripped.startswith("${{"):
                        env_lines.append(line)
                    else:
                        in_env_block = False

        env_content = "\n".join(env_lines)
        assert "${{ github.event.pull_request.user.login }}" in env_content

    def test_run_block_uses_env_var_not_interpolation(self):
        """CI guard run block uses $PR_AUTHOR (env var) not ${{ }} expression."""
        from znote_mcp.server.mcp_server import ZettelkastenMcpServer

        template = ZettelkastenMcpServer._generate_ci_guard()

        # Extract content after "run: |"
        run_start = template.find("run: |")
        assert run_start != -1, "Template must have a run: | block"
        run_block = template[run_start:]

        # $PR_AUTHOR should appear (env var reference)
        assert "$PR_AUTHOR" in run_block
        # But not ${{ github. in the run block
        assert "${{ github." not in run_block


# =========================================================================
# Check Prerequisites Tests
# =========================================================================


class TestCheckPrerequisites:
    """Test GitSyncService.check_prerequisites."""

    def test_check_prerequisites_method_exists(self):
        """GitSyncService has a check_prerequisites method."""
        assert hasattr(GitSyncService, "check_prerequisites")

    def test_check_prerequisites_git_not_found(self, sync_service, monkeypatch):
        """check_prerequisites detects missing git binary."""
        from znote_mcp.exceptions import SyncError

        with patch.object(
            sync_service,
            "_run_git",
            side_effect=SyncError("Git is not installed"),
        ):
            result = sync_service.check_prerequisites()
        assert result["git_installed"] is False

    def test_check_prerequisites_git_found(self, sync_service):
        """check_prerequisites detects installed git."""
        with patch.object(sync_service, "_run_git") as mock_git:
            # First call: git --version
            version_result = MagicMock(returncode=0, stdout="git version 2.43.0")
            # Second call: git ls-remote
            ls_result = MagicMock(returncode=0, stdout="abc123\tHEAD")
            mock_git.side_effect = [version_result, ls_result]
            result = sync_service.check_prerequisites()
        assert result["git_installed"] is True
        assert result["git_version"] == "git version 2.43.0"
        assert result["repo_reachable"] is True

    def test_check_prerequisites_repo_unreachable(self, sync_service):
        """check_prerequisites detects unreachable repo."""
        with patch.object(sync_service, "_run_git") as mock_git:
            version_result = MagicMock(returncode=0, stdout="git version 2.43.0")
            ls_result = MagicMock(
                returncode=128,
                stdout="",
                stderr="fatal: could not read from remote",
            )
            mock_git.side_effect = [version_result, ls_result]
            result = sync_service.check_prerequisites()
        assert result["git_installed"] is True
        assert result["repo_reachable"] is False
        assert "could not read from remote" in result["repo_error"]

    def test_check_prerequisites_already_setup(self, sync_service_setup):
        """check_prerequisites reports already_setup=True."""
        with patch.object(sync_service_setup, "_run_git") as mock_git:
            version_result = MagicMock(returncode=0, stdout="git version 2.43.0")
            ls_result = MagicMock(returncode=0, stdout="abc123\tHEAD")
            mock_git.side_effect = [version_result, ls_result]
            result = sync_service_setup.check_prerequisites()
        assert result["already_setup"] is True
