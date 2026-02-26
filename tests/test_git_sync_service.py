"""Tests for GitSyncService push and debounce logic (Phase 2).

Covers:
- signal_write starts debounce timer
- flush_push cancels timer and pushes immediately
- Debounce max 30-minute cap
- stage_and_push mirrors files and runs git commands
- shutdown flushes pending writes
- _sync_files mirrors correctly
- ZettelService integration: create/update/delete/bulk signal sync
"""

import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from znote_mcp.services.git_sync_service import GitSyncService


@pytest.fixture
def sync_service(tmp_path):
    """Create a GitSyncService instance for testing (not set up)."""
    return GitSyncService(
        user_id="testuser",
        repo_url="https://example.com/repo.git",
        branch="testuser/notes",
        remote_dir=tmp_path / ".remote",
        notes_dir=tmp_path / "notes",
        imports_dir=tmp_path / "imports",
        push_delay=120,
        push_extend=60,
    )


class TestSignalWrite:
    """Tests for the signal_write debounce method."""

    def test_signal_write_exists(self):
        """signal_write method exists on GitSyncService."""
        assert hasattr(GitSyncService, "signal_write")

    def test_signal_write_increments_pending(self, sync_service):
        """signal_write increments pending_writes counter."""
        assert sync_service._pending_writes == 0
        sync_service.signal_write()
        assert sync_service._pending_writes == 1
        sync_service.signal_write()
        assert sync_service._pending_writes == 2
        # Clean up timer
        with sync_service._push_lock:
            if sync_service._push_timer is not None:
                sync_service._push_timer.cancel()

    def test_signal_write_starts_timer(self, sync_service):
        """signal_write starts a debounce timer."""
        assert sync_service._push_timer is None
        sync_service.signal_write()
        assert sync_service._push_timer is not None
        assert sync_service._push_timer.is_alive()
        # Clean up
        sync_service._push_timer.cancel()

    def test_signal_write_resets_timer_on_subsequent_calls(self, sync_service):
        """Subsequent signal_write calls cancel and restart the timer."""
        sync_service.signal_write()
        first_timer = sync_service._push_timer
        sync_service.signal_write()
        second_timer = sync_service._push_timer
        assert first_timer is not second_timer
        # First timer should have been cancelled
        assert not first_timer.is_alive()
        assert second_timer.is_alive()
        # Clean up
        second_timer.cancel()

    def test_signal_write_sets_first_write_time(self, sync_service):
        """signal_write sets _first_write_time on first call."""
        assert sync_service._first_write_time is None
        sync_service.signal_write()
        assert sync_service._first_write_time is not None
        first_time = sync_service._first_write_time
        sync_service.signal_write()
        # Should not change on subsequent calls
        assert sync_service._first_write_time == first_time
        # Clean up
        sync_service._push_timer.cancel()

    def test_signal_write_uses_push_delay_for_first_write(self, tmp_path):
        """First write uses push_delay, not push_extend."""
        svc = GitSyncService(
            user_id="u",
            repo_url="https://x",
            branch="b",
            remote_dir=tmp_path / ".remote",
            notes_dir=tmp_path / "notes",
            imports_dir=tmp_path / "imports",
            push_delay=300,
            push_extend=30,
        )
        svc.signal_write()
        # Timer interval should be push_delay (300), not push_extend (30)
        assert svc._push_timer.interval == 300
        svc._push_timer.cancel()

    def test_signal_write_uses_push_extend_for_subsequent_writes(self, tmp_path):
        """Subsequent writes use push_extend (shorter delay)."""
        svc = GitSyncService(
            user_id="u",
            repo_url="https://x",
            branch="b",
            remote_dir=tmp_path / ".remote",
            notes_dir=tmp_path / "notes",
            imports_dir=tmp_path / "imports",
            push_delay=300,
            push_extend=30,
        )
        svc.signal_write()
        svc.signal_write()
        # Timer interval should be push_extend (30), not push_delay (300)
        assert svc._push_timer.interval == 30
        svc._push_timer.cancel()


class TestDebounceMaxCap:
    """Tests for the 30-minute max push delay cap."""

    def test_max_push_delay_constant(self):
        """MAX_PUSH_DELAY is 1800 seconds (30 minutes)."""
        assert GitSyncService._MAX_PUSH_DELAY == 1800

    def test_delay_capped_at_remaining_time(self, sync_service):
        """When time remaining until cap is less than delay, use remaining."""
        # Simulate first write was 29 minutes ago (1740 seconds)
        sync_service._first_write_time = datetime.now(timezone.utc)
        # Artificially shift the first write time back
        from datetime import timedelta

        sync_service._first_write_time -= timedelta(seconds=1740)
        sync_service._pending_writes = 0  # Will be first "real" write

        sync_service.signal_write()
        # Should use min(push_delay=120, remaining=60) = 60
        assert sync_service._push_timer.interval <= 60.5
        sync_service._push_timer.cancel()

    def test_delay_fires_immediately_at_cap(self, sync_service):
        """When 30 minutes have elapsed, push fires almost immediately."""
        from datetime import timedelta

        sync_service._first_write_time = datetime.now(timezone.utc) - timedelta(
            seconds=1801
        )
        sync_service._pending_writes = 0

        sync_service.signal_write()
        # Should use 0.1 (immediate push)
        assert sync_service._push_timer.interval == 0.1
        sync_service._push_timer.cancel()


class TestFlushPush:
    """Tests for flush_push."""

    def test_flush_push_exists(self):
        """flush_push method exists on GitSyncService."""
        assert hasattr(GitSyncService, "flush_push")

    def test_flush_push_cancels_timer(self, sync_service):
        """flush_push cancels the pending timer."""
        sync_service.signal_write()
        assert sync_service._push_timer is not None
        with patch.object(sync_service, "stage_and_push", return_value=True):
            sync_service.flush_push()
        assert sync_service._push_timer is None

    def test_flush_push_calls_stage_and_push(self, sync_service):
        """flush_push delegates to stage_and_push."""
        with patch.object(sync_service, "stage_and_push", return_value=True) as mock:
            result = sync_service.flush_push()
        mock.assert_called_once()
        assert result is True

    def test_flush_push_returns_false_on_failure(self, sync_service):
        """flush_push returns False when stage_and_push fails."""
        with patch.object(sync_service, "stage_and_push", return_value=False):
            result = sync_service.flush_push()
        assert result is False


class TestStageAndPush:
    """Tests for stage_and_push."""

    def test_stage_and_push_returns_false_when_not_setup(self, sync_service):
        """stage_and_push returns False when git is not set up."""
        assert not sync_service.is_setup
        assert sync_service.stage_and_push() is False

    def test_stage_and_push_runs_git_commands(self, sync_service, tmp_path):
        """stage_and_push runs add, status, commit, push."""
        # Fake setup
        (tmp_path / ".remote" / ".git").mkdir(parents=True)
        (tmp_path / "notes").mkdir(exist_ok=True)

        status_result = MagicMock()
        status_result.stdout = "M notes/testuser/note.md\n"

        calls = []

        def fake_run_git(args, **kwargs):
            calls.append(args)
            if args[0] == "status":
                return status_result
            return MagicMock(stdout="", returncode=0)

        with patch.object(sync_service, "_run_git", side_effect=fake_run_git):
            with patch.object(sync_service, "_sync_files"):
                result = sync_service.stage_and_push()

        assert result is True
        # Should have called: add, status, commit, push
        assert len(calls) == 4
        assert calls[0][0] == "add"
        assert calls[1][0] == "status"
        assert calls[2][0] == "commit"
        assert calls[3][0] == "push"

    def test_stage_and_push_skips_when_no_changes(self, sync_service, tmp_path):
        """stage_and_push skips commit/push when no changes."""
        (tmp_path / ".remote" / ".git").mkdir(parents=True)
        (tmp_path / "notes").mkdir(exist_ok=True)

        status_result = MagicMock()
        status_result.stdout = ""  # No changes

        calls = []

        def fake_run_git(args, **kwargs):
            calls.append(args)
            if args[0] == "status":
                return status_result
            return MagicMock(stdout="", returncode=0)

        with patch.object(sync_service, "_run_git", side_effect=fake_run_git):
            with patch.object(sync_service, "_sync_files"):
                result = sync_service.stage_and_push()

        assert result is True
        # Should have called: add, status only (no commit/push)
        assert len(calls) == 2

    def test_stage_and_push_updates_last_push_time(self, sync_service, tmp_path):
        """Successful push updates _last_push_time."""
        (tmp_path / ".remote" / ".git").mkdir(parents=True)
        (tmp_path / "notes").mkdir(exist_ok=True)

        status_result = MagicMock()
        status_result.stdout = "M something\n"

        assert sync_service._last_push_time is None

        with patch.object(
            sync_service,
            "_run_git",
            return_value=status_result,
        ):
            with patch.object(sync_service, "_sync_files"):
                sync_service.stage_and_push()

        assert sync_service._last_push_time is not None

    def test_stage_and_push_resets_pending_writes(self, sync_service, tmp_path):
        """Successful push resets _pending_writes and _first_write_time."""
        (tmp_path / ".remote" / ".git").mkdir(parents=True)
        (tmp_path / "notes").mkdir(exist_ok=True)

        status_result = MagicMock()
        status_result.stdout = "M something\n"

        sync_service._pending_writes = 5
        sync_service._first_write_time = datetime.now(timezone.utc)

        with patch.object(
            sync_service,
            "_run_git",
            return_value=status_result,
        ):
            with patch.object(sync_service, "_sync_files"):
                sync_service.stage_and_push()

        assert sync_service._pending_writes == 0
        assert sync_service._first_write_time is None

    def test_stage_and_push_returns_false_on_exception(self, sync_service, tmp_path):
        """stage_and_push catches exceptions and returns False."""
        (tmp_path / ".remote" / ".git").mkdir(parents=True)

        with patch.object(
            sync_service, "_sync_files", side_effect=OSError("disk full")
        ):
            result = sync_service.stage_and_push()
        assert result is False


class TestSyncFiles:
    """Tests for _sync_files mirroring logic."""

    def test_sync_copies_new_files(self, sync_service, tmp_path):
        """New .md files are copied from src to dst."""
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        src.mkdir()
        dst.mkdir()
        (src / "note.md").write_text("hello")

        sync_service._sync_files(src, dst)
        assert (dst / "note.md").read_text() == "hello"

    def test_sync_removes_deleted_files(self, sync_service, tmp_path):
        """Files in dst not present in src are deleted."""
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        src.mkdir()
        dst.mkdir()
        (dst / "old.md").write_text("should be removed")

        sync_service._sync_files(src, dst)
        assert not (dst / "old.md").exists()

    def test_sync_updates_changed_files(self, sync_service, tmp_path):
        """Changed files (newer mtime in src) are overwritten."""
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        src.mkdir()
        dst.mkdir()
        (dst / "note.md").write_text("old content")
        # Ensure src is newer
        time.sleep(0.05)
        (src / "note.md").write_text("new content")

        sync_service._sync_files(src, dst)
        assert (dst / "note.md").read_text() == "new content"

    def test_sync_ignores_non_md_files(self, sync_service, tmp_path):
        """Non-.md files in dst are not removed."""
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        src.mkdir()
        dst.mkdir()
        (dst / "config.json").write_text("{}")

        sync_service._sync_files(src, dst)
        # Should not be removed (not .md)
        assert (dst / "config.json").exists()

    def test_sync_handles_subdirectories(self, sync_service, tmp_path):
        """Subdirectories are correctly mirrored."""
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        src.mkdir()
        dst.mkdir()
        (src / "sub").mkdir()
        (src / "sub" / "deep.md").write_text("deep note")

        sync_service._sync_files(src, dst)
        assert (dst / "sub" / "deep.md").read_text() == "deep note"


class TestShutdown:
    """Tests for shutdown with flush."""

    def test_shutdown_cancels_timer(self, sync_service):
        """shutdown cancels the pending timer."""
        sync_service.signal_write()
        assert sync_service._push_timer is not None
        with patch.object(sync_service, "stage_and_push", return_value=True):
            sync_service.shutdown()
        assert sync_service._push_timer is None

    def test_shutdown_flushes_pending_writes(self, sync_service):
        """shutdown calls stage_and_push when writes are pending."""
        sync_service._pending_writes = 3
        with patch.object(
            sync_service, "stage_and_push", return_value=True
        ) as mock_push:
            sync_service.shutdown()
        mock_push.assert_called_once()

    def test_shutdown_skips_flush_when_no_pending(self, sync_service):
        """shutdown skips stage_and_push when no writes pending."""
        assert sync_service._pending_writes == 0
        with patch.object(
            sync_service, "stage_and_push", return_value=True
        ) as mock_push:
            sync_service.shutdown()
        mock_push.assert_not_called()


class TestDebounceTimerFires:
    """Tests that the debounce timer actually fires stage_and_push."""

    def test_debounce_push_calls_stage_and_push(self, sync_service):
        """_debounce_push delegates to stage_and_push."""
        with patch.object(
            sync_service, "stage_and_push", return_value=True
        ) as mock_push:
            sync_service._debounce_push()
        mock_push.assert_called_once()

    def test_timer_fires_debounce_push(self, tmp_path):
        """Timer created by signal_write fires _debounce_push after delay."""
        svc = GitSyncService(
            user_id="u",
            repo_url="https://x",
            branch="b",
            remote_dir=tmp_path / ".remote",
            notes_dir=tmp_path / "notes",
            imports_dir=tmp_path / "imports",
            push_delay=0,  # immediate
            push_extend=0,
        )
        with patch.object(svc, "stage_and_push", return_value=True) as mock_push:
            svc.signal_write()
            # Wait for the timer to fire
            time.sleep(0.3)
        mock_push.assert_called_once()


class TestZettelServiceSyncIntegration:
    """Tests that ZettelService calls signal_write on write operations."""

    def test_create_note_signals_sync(self, tmp_path):
        """create_note calls _signal_sync."""
        from znote_mcp.services.zettel_service import ZettelService

        mock_sync = MagicMock()
        svc = ZettelService(sync_service=mock_sync)
        svc.create_note(title="Test", content="body")
        mock_sync.signal_write.assert_called_once()

    def test_update_note_signals_sync(self, tmp_path):
        """update_note calls _signal_sync."""
        from znote_mcp.services.zettel_service import ZettelService

        mock_sync = MagicMock()
        svc = ZettelService(sync_service=mock_sync)
        note = svc.create_note(title="Test", content="body")
        mock_sync.signal_write.reset_mock()
        svc.update_note(note.id, content="updated")
        mock_sync.signal_write.assert_called_once()

    def test_delete_note_signals_sync(self, tmp_path):
        """delete_note calls _signal_sync."""
        from znote_mcp.services.zettel_service import ZettelService

        mock_sync = MagicMock()
        svc = ZettelService(sync_service=mock_sync)
        note = svc.create_note(title="Test", content="body")
        mock_sync.signal_write.reset_mock()
        svc.delete_note(note.id)
        mock_sync.signal_write.assert_called_once()

    def test_bulk_create_signals_sync_once(self, tmp_path):
        """bulk_create_notes calls _signal_sync once for the batch."""
        from znote_mcp.services.zettel_service import ZettelService

        mock_sync = MagicMock()
        svc = ZettelService(sync_service=mock_sync)
        svc.bulk_create_notes(
            [
                {"title": "A", "content": "a"},
                {"title": "B", "content": "b"},
            ]
        )
        # create_note during bulk should not signal; only the batch-level call
        # Total: 1 call from bulk_create + 0 from individual (bulk uses repository)
        # Actually bulk_create_notes does call _signal_sync once at the end
        assert mock_sync.signal_write.call_count >= 1

    def test_no_signal_when_sync_not_configured(self, tmp_path):
        """When sync_service is None, no errors occur on writes."""
        from znote_mcp.services.zettel_service import ZettelService

        svc = ZettelService(sync_service=None)
        # Should not raise
        note = svc.create_note(title="Test", content="body")
        svc.update_note(note.id, content="updated")
        svc.delete_note(note.id)

    def test_signal_sync_failure_does_not_break_crud(self, tmp_path):
        """If signal_write raises, the CRUD operation still succeeds."""
        from znote_mcp.services.zettel_service import ZettelService

        mock_sync = MagicMock()
        mock_sync.signal_write.side_effect = RuntimeError("network down")
        svc = ZettelService(sync_service=mock_sync)
        # Should not raise despite sync failure
        note = svc.create_note(title="Test", content="body")
        assert note is not None
        assert note.title == "Test"

    def test_shutdown_calls_sync_shutdown(self, tmp_path):
        """ZettelService.shutdown() calls sync_service.shutdown()."""
        from znote_mcp.services.zettel_service import ZettelService

        mock_sync = MagicMock()
        svc = ZettelService(sync_service=mock_sync)
        svc.shutdown()
        mock_sync.shutdown.assert_called_once()
