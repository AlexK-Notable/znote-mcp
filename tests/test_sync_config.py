"""Tests for sync config fields, SyncError, and GitSyncService importability.

Covers Phase 1 gate checks:
- 8 new config fields with correct defaults
- Model validator: sync_enabled=True requires user_id and repo_url
- sync_user_id validated against SAFE_ID_PATTERN
- Auto-derived sync_branch
- SyncError hierarchy and error codes
- GitSyncService import and basic construction
- Helper properties: sync_remote_dir, sync_imports_dir, get_import_users()
"""

from pathlib import Path

import pytest


class TestSyncConfigFields:
    """Test the 8 new sync configuration fields."""

    def test_all_sync_fields_exist(self, monkeypatch):
        """All 8 sync fields exist on ZettelkastenConfig."""
        monkeypatch.delenv("ZETTELKASTEN_SYNC_ENABLED", raising=False)
        monkeypatch.delenv("ZETTELKASTEN_SYNC_USER_ID", raising=False)
        monkeypatch.delenv("ZETTELKASTEN_USER_ID", raising=False)
        monkeypatch.delenv("ZETTELKASTEN_SYNC_REPO", raising=False)
        monkeypatch.delenv("ZETTELKASTEN_SYNC_BRANCH", raising=False)
        monkeypatch.delenv("ZETTELKASTEN_GIT_PUSH_DELAY", raising=False)
        monkeypatch.delenv("ZETTELKASTEN_GIT_PUSH_EXTEND", raising=False)
        monkeypatch.delenv("ZETTELKASTEN_IMPORT_USERS", raising=False)
        monkeypatch.delenv("ZETTELKASTEN_IMPORT_ON_STARTUP", raising=False)

        from znote_mcp.config import ZettelkastenConfig

        c = ZettelkastenConfig()
        attrs = [
            "sync_enabled",
            "sync_user_id",
            "sync_repo_url",
            "sync_branch",
            "sync_push_delay",
            "sync_push_extend",
            "sync_import_users",
            "sync_import_on_startup",
        ]
        missing = [a for a in attrs if not hasattr(c, a)]
        assert not missing, f"Missing: {missing}"

    def test_sync_enabled_defaults_to_false(self, monkeypatch):
        """sync_enabled defaults to False when env var is not set."""
        monkeypatch.delenv("ZETTELKASTEN_SYNC_ENABLED", raising=False)

        from znote_mcp.config import ZettelkastenConfig

        c = ZettelkastenConfig()
        assert c.sync_enabled is False

    def test_sync_user_id_defaults_to_none(self, monkeypatch):
        """sync_user_id defaults to None when env var is not set."""
        monkeypatch.delenv("ZETTELKASTEN_SYNC_USER_ID", raising=False)
        monkeypatch.delenv("ZETTELKASTEN_USER_ID", raising=False)

        from znote_mcp.config import ZettelkastenConfig

        c = ZettelkastenConfig()
        assert c.sync_user_id is None

    def test_sync_push_delay_defaults_to_120(self, monkeypatch):
        """sync_push_delay defaults to 120 seconds."""
        monkeypatch.delenv("ZETTELKASTEN_GIT_PUSH_DELAY", raising=False)

        from znote_mcp.config import ZettelkastenConfig

        c = ZettelkastenConfig()
        assert c.sync_push_delay == 120

    def test_sync_push_extend_defaults_to_60(self, monkeypatch):
        """sync_push_extend defaults to 60 seconds."""
        monkeypatch.delenv("ZETTELKASTEN_GIT_PUSH_EXTEND", raising=False)

        from znote_mcp.config import ZettelkastenConfig

        c = ZettelkastenConfig()
        assert c.sync_push_extend == 60

    def test_sync_import_on_startup_defaults_to_true(self, monkeypatch):
        """sync_import_on_startup defaults to True."""
        monkeypatch.delenv("ZETTELKASTEN_IMPORT_ON_STARTUP", raising=False)

        from znote_mcp.config import ZettelkastenConfig

        c = ZettelkastenConfig()
        assert c.sync_import_on_startup is True


class TestSyncConfigValidation:
    """Test model validators for sync configuration."""

    def test_sync_enabled_requires_user_id(self, monkeypatch):
        """sync_enabled=True without user_id raises ValueError."""
        monkeypatch.delenv("ZETTELKASTEN_SYNC_USER_ID", raising=False)
        monkeypatch.delenv("ZETTELKASTEN_USER_ID", raising=False)

        from znote_mcp.config import ZettelkastenConfig

        with pytest.raises(ValueError, match="sync_user_id is required"):
            ZettelkastenConfig(
                sync_enabled=True,
                sync_repo_url="https://example.com/repo.git",
            )

    def test_sync_enabled_requires_repo_url(self, monkeypatch):
        """sync_enabled=True without repo_url raises ValueError."""
        monkeypatch.delenv("ZETTELKASTEN_SYNC_REPO", raising=False)

        from znote_mcp.config import ZettelkastenConfig

        with pytest.raises(ValueError, match="sync_repo_url is required"):
            ZettelkastenConfig(
                sync_enabled=True,
                sync_user_id="alice",
                sync_repo_url=None,
            )

    def test_sync_user_id_validated_against_safe_pattern(self):
        """sync_user_id with path traversal chars is rejected."""
        from znote_mcp.config import ZettelkastenConfig

        with pytest.raises(ValueError, match="invalid characters"):
            ZettelkastenConfig(
                sync_enabled=True,
                sync_user_id="../bad",
                sync_repo_url="https://example.com/repo.git",
            )

    def test_sync_user_id_rejects_spaces(self):
        """sync_user_id with spaces is rejected."""
        from znote_mcp.config import ZettelkastenConfig

        with pytest.raises(ValueError, match="invalid characters"):
            ZettelkastenConfig(
                sync_enabled=True,
                sync_user_id="bad user",
                sync_repo_url="https://example.com/repo.git",
            )

    def test_sync_user_id_validated_even_when_disabled(self):
        """sync_user_id with bad chars is rejected even when sync is disabled."""
        from znote_mcp.config import ZettelkastenConfig

        with pytest.raises(ValueError, match="invalid characters"):
            ZettelkastenConfig(
                sync_enabled=False,
                sync_user_id="../bad",
            )

    def test_valid_sync_config_accepted(self):
        """Valid sync configuration is accepted."""
        from znote_mcp.config import ZettelkastenConfig

        c = ZettelkastenConfig(
            sync_enabled=True,
            sync_user_id="alice",
            sync_repo_url="https://example.com/repo.git",
        )
        assert c.sync_enabled is True
        assert c.sync_user_id == "alice"

    def test_sync_branch_auto_derived(self):
        """sync_branch is auto-derived as '{user_id}/notes' when not set."""
        from znote_mcp.config import ZettelkastenConfig

        c = ZettelkastenConfig(
            sync_enabled=True,
            sync_user_id="alice",
            sync_repo_url="https://example.com/repo.git",
        )
        assert c.sync_branch == "alice/notes"

    def test_sync_branch_explicit_override(self):
        """Explicit sync_branch is preserved."""
        from znote_mcp.config import ZettelkastenConfig

        c = ZettelkastenConfig(
            sync_enabled=True,
            sync_user_id="alice",
            sync_repo_url="https://example.com/repo.git",
            sync_branch="custom/branch",
        )
        assert c.sync_branch == "custom/branch"

    def test_sync_user_id_accepts_nanoid(self):
        """sync_user_id accepts NanoID-style IDs."""
        from znote_mcp.config import ZettelkastenConfig

        c = ZettelkastenConfig(
            sync_enabled=True,
            sync_user_id="V1StGXR8_Z5jdHi6B-myT",
            sync_repo_url="https://example.com/repo.git",
        )
        assert c.sync_user_id == "V1StGXR8_Z5jdHi6B-myT"

    def test_sync_user_id_accepts_old_format(self):
        """sync_user_id accepts old timestamp-style IDs."""
        from znote_mcp.config import ZettelkastenConfig

        c = ZettelkastenConfig(
            sync_enabled=True,
            sync_user_id="20260226T042704869187408807",
            sync_repo_url="https://example.com/repo.git",
        )
        assert c.sync_user_id == "20260226T042704869187408807"


class TestSyncConfigHelpers:
    """Test helper properties and methods for sync configuration."""

    def test_sync_remote_dir(self):
        """sync_remote_dir returns notes_dir parent / .remote."""
        from znote_mcp.config import ZettelkastenConfig

        c = ZettelkastenConfig(notes_dir=Path("/data/notes"))
        assert c.sync_remote_dir == Path("/data/.remote")

    def test_sync_imports_dir(self):
        """sync_imports_dir returns notes_dir parent / imports."""
        from znote_mcp.config import ZettelkastenConfig

        c = ZettelkastenConfig(notes_dir=Path("/data/notes"))
        assert c.sync_imports_dir == Path("/data/imports")

    def test_get_import_users_empty(self):
        """get_import_users returns empty list when not configured."""
        from znote_mcp.config import ZettelkastenConfig

        c = ZettelkastenConfig()
        assert c.get_import_users() == []

    def test_get_import_users_single(self):
        """get_import_users parses a single user."""
        from znote_mcp.config import ZettelkastenConfig

        c = ZettelkastenConfig(sync_import_users="bob")
        assert c.get_import_users() == ["bob"]

    def test_get_import_users_multiple(self):
        """get_import_users parses comma-separated users."""
        from znote_mcp.config import ZettelkastenConfig

        c = ZettelkastenConfig(sync_import_users="bob, carol, dave")
        assert c.get_import_users() == ["bob", "carol", "dave"]

    def test_get_import_users_strips_whitespace(self):
        """get_import_users strips whitespace from user IDs."""
        from znote_mcp.config import ZettelkastenConfig

        c = ZettelkastenConfig(sync_import_users=" bob , carol ")
        assert c.get_import_users() == ["bob", "carol"]

    def test_get_import_users_ignores_empty(self):
        """get_import_users ignores empty segments from trailing commas."""
        from znote_mcp.config import ZettelkastenConfig

        c = ZettelkastenConfig(sync_import_users="bob,,carol,")
        assert c.get_import_users() == ["bob", "carol"]


class TestSyncErrorAndCodes:
    """Test SyncError class and error codes."""

    def test_sync_error_hierarchy(self):
        """SyncError inherits from ZettelkastenError."""
        from znote_mcp.exceptions import SyncError, ZettelkastenError

        assert issubclass(SyncError, ZettelkastenError)

    def test_error_codes_9001_to_9007(self):
        """Error codes 9001-9007 exist in ErrorCode enum."""
        from znote_mcp.exceptions import ErrorCode

        codes = [ErrorCode(v) for v in range(9001, 9008)]
        assert len(codes) == 7

    def test_sync_error_code_names(self):
        """Sync error codes have expected names."""
        from znote_mcp.exceptions import ErrorCode

        assert ErrorCode.SYNC_NOT_CONFIGURED.value == 9001
        assert ErrorCode.SYNC_REMOTE_FAILED.value == 9002
        assert ErrorCode.SYNC_PUSH_FAILED.value == 9003
        assert ErrorCode.SYNC_PULL_FAILED.value == 9004
        assert ErrorCode.SYNC_SPARSE_CHECKOUT_FAILED.value == 9005
        assert ErrorCode.SYNC_IMPORT_FAILED.value == 9006
        assert ErrorCode.SYNC_WRITE_REJECTED.value == 9007

    def test_sync_error_with_operation(self):
        """SyncError stores operation in details."""
        from znote_mcp.exceptions import ErrorCode, SyncError

        err = SyncError("test error", operation="push", code=ErrorCode.SYNC_PUSH_FAILED)
        assert err.operation == "push"
        assert err.details["operation"] == "push"
        assert err.code == ErrorCode.SYNC_PUSH_FAILED

    def test_sync_error_with_original_error(self):
        """SyncError stores original error (truncated to 200 chars)."""
        from znote_mcp.exceptions import SyncError

        original = ValueError("x" * 300)
        err = SyncError("wrapped", original_error=original)
        assert err.original_error is original
        assert len(err.details["original_error"]) <= 200

    def test_sync_error_default_code(self):
        """SyncError defaults to SYNC_REMOTE_FAILED."""
        from znote_mcp.exceptions import ErrorCode, SyncError

        err = SyncError("test")
        assert err.code == ErrorCode.SYNC_REMOTE_FAILED

    def test_sync_error_to_dict(self):
        """SyncError serializes to dict."""
        from znote_mcp.exceptions import SyncError

        err = SyncError("test error", operation="clone")
        d = err.to_dict()
        assert d["error"] == "SyncError"
        assert d["code"] == 9002
        assert d["code_name"] == "SYNC_REMOTE_FAILED"
        assert d["message"] == "test error"
        assert d["details"]["operation"] == "clone"


class TestGitSyncServiceImport:
    """Test that GitSyncService is importable and constructible."""

    def test_importable(self):
        """GitSyncService can be imported."""
        from znote_mcp.services.git_sync_service import GitSyncService

        assert GitSyncService is not None

    def test_constructible(self, tmp_path):
        """GitSyncService can be constructed with valid args."""
        from znote_mcp.services.git_sync_service import GitSyncService

        svc = GitSyncService(
            user_id="alice",
            repo_url="https://example.com/repo.git",
            branch="alice/notes",
            remote_dir=tmp_path / ".remote",
            notes_dir=tmp_path / "notes",
            imports_dir=tmp_path / "imports",
        )
        assert svc is not None

    def test_has_sync_remote_dir(self, tmp_path):
        """GitSyncService has sync_remote_dir property."""
        from znote_mcp.services.git_sync_service import GitSyncService

        remote = tmp_path / ".remote"
        svc = GitSyncService(
            user_id="alice",
            repo_url="https://example.com/repo.git",
            branch="alice/notes",
            remote_dir=remote,
            notes_dir=tmp_path / "notes",
            imports_dir=tmp_path / "imports",
        )
        assert svc.sync_remote_dir == remote

    def test_is_setup_false_initially(self, tmp_path):
        """is_setup is False when .git directory doesn't exist."""
        from znote_mcp.services.git_sync_service import GitSyncService

        svc = GitSyncService(
            user_id="alice",
            repo_url="https://example.com/repo.git",
            branch="alice/notes",
            remote_dir=tmp_path / ".remote",
            notes_dir=tmp_path / "notes",
            imports_dir=tmp_path / "imports",
        )
        assert svc.is_setup is False

    def test_is_setup_true_with_git_dir(self, tmp_path):
        """is_setup is True when .git directory exists."""
        from znote_mcp.services.git_sync_service import GitSyncService

        remote = tmp_path / ".remote"
        (remote / ".git").mkdir(parents=True)
        svc = GitSyncService(
            user_id="alice",
            repo_url="https://example.com/repo.git",
            branch="alice/notes",
            remote_dir=remote,
            notes_dir=tmp_path / "notes",
            imports_dir=tmp_path / "imports",
        )
        assert svc.is_setup is True

    def test_get_status(self, tmp_path):
        """get_status returns expected dict structure."""
        from znote_mcp.services.git_sync_service import GitSyncService

        svc = GitSyncService(
            user_id="alice",
            repo_url="https://example.com/repo.git",
            branch="alice/notes",
            remote_dir=tmp_path / ".remote",
            notes_dir=tmp_path / "notes",
            imports_dir=tmp_path / "imports",
            import_users=["bob", "carol"],
        )
        status = svc.get_status()
        assert status["enabled"] is True
        assert status["user_id"] == "alice"
        assert status["branch"] == "alice/notes"
        assert status["is_setup"] is False
        assert status["last_push_time"] is None
        assert status["last_import_time"] is None
        assert status["pending_writes"] == 0
        assert status["import_users"] == ["bob", "carol"]

    def test_get_status_does_not_expose_repo_url(self, tmp_path):
        """get_status omits repo_url for security."""
        from znote_mcp.services.git_sync_service import GitSyncService

        svc = GitSyncService(
            user_id="alice",
            repo_url="https://token@example.com/repo.git",
            branch="alice/notes",
            remote_dir=tmp_path / ".remote",
            notes_dir=tmp_path / "notes",
            imports_dir=tmp_path / "imports",
        )
        status = svc.get_status()
        assert "repo_url" not in status

    def test_shutdown_cancels_timer(self, tmp_path):
        """shutdown() cancels any pending push timer."""
        import threading

        from znote_mcp.services.git_sync_service import GitSyncService

        svc = GitSyncService(
            user_id="alice",
            repo_url="https://example.com/repo.git",
            branch="alice/notes",
            remote_dir=tmp_path / ".remote",
            notes_dir=tmp_path / "notes",
            imports_dir=tmp_path / "imports",
        )
        # Create a real timer (not started) to simulate a pending push
        timer = threading.Timer(9999, lambda: None)
        svc._push_timer = timer
        svc.shutdown()
        assert svc._push_timer is None

    def test_default_import_users_empty(self, tmp_path):
        """import_users defaults to empty list."""
        from znote_mcp.services.git_sync_service import GitSyncService

        svc = GitSyncService(
            user_id="alice",
            repo_url="https://example.com/repo.git",
            branch="alice/notes",
            remote_dir=tmp_path / ".remote",
            notes_dir=tmp_path / "notes",
            imports_dir=tmp_path / "imports",
        )
        assert svc.get_status()["import_users"] == []


class TestGitSyncServiceRunGit:
    """Test _run_git error handling (without real git operations)."""

    def test_run_git_raises_on_missing_git(self, tmp_path, monkeypatch):
        """_run_git raises SyncError when git binary not found."""
        from znote_mcp.exceptions import ErrorCode, SyncError
        from znote_mcp.services.git_sync_service import GitSyncService

        svc = GitSyncService(
            user_id="alice",
            repo_url="https://example.com/repo.git",
            branch="alice/notes",
            remote_dir=tmp_path,
            notes_dir=tmp_path / "notes",
            imports_dir=tmp_path / "imports",
        )

        # Point PATH to an empty dir so git binary is not found
        monkeypatch.setenv("PATH", str(tmp_path / "empty"))
        with pytest.raises(SyncError) as exc_info:
            svc._run_git(["status"])
        assert exc_info.value.code == ErrorCode.SYNC_NOT_CONFIGURED
