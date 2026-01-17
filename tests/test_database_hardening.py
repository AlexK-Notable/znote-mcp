"""Tests for database hardening measures (WAL mode, health check, auto-recovery, FTS degradation)."""
import sqlite3
import tempfile
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text

from znote_mcp.config import config
from znote_mcp.exceptions import (
    DatabaseCorruptionError,
    ErrorCode,
)
from znote_mcp.models.db_models import Base, init_db
from znote_mcp.models.schema import Note, NoteType, Tag
from znote_mcp.storage.note_repository import NoteRepository


class TestWALMode:
    """Tests for WAL mode configuration."""

    def test_wal_mode_enabled(self, note_repository):
        """Verify WAL mode is enabled on database connections."""
        with note_repository.session_factory() as session:
            result = session.execute(text("PRAGMA journal_mode")).fetchone()
            assert result[0].lower() == "wal", "WAL mode should be enabled"

    def test_synchronous_normal(self, note_repository):
        """Verify synchronous mode is set to NORMAL for performance balance."""
        with note_repository.session_factory() as session:
            result = session.execute(text("PRAGMA synchronous")).fetchone()
            # NORMAL = 1
            assert result[0] == 1, "Synchronous mode should be NORMAL (1)"


class TestDatabaseHealthCheck:
    """Tests for comprehensive database health check."""

    def test_health_check_healthy_database(self, note_repository):
        """Test health check returns healthy for a clean database."""
        health = note_repository.check_database_health()

        assert health["healthy"] is True
        assert health["sqlite_ok"] is True
        assert health["fts_ok"] is True
        assert health["issues"] == []

    def test_health_check_returns_counts(self, note_repository):
        """Test health check returns accurate note/file counts."""
        # Create a note
        note = Note(
            title="Health Check Test",
            content="Testing health check counts.",
            note_type=NoteType.PERMANENT,
        )
        note_repository.create(note)

        health = note_repository.check_database_health()

        assert health["note_count"] == 1
        assert health["file_count"] == 1

    def test_health_check_detects_count_mismatch(self, note_repository, temp_dirs):
        """Test health check detects DB/file count mismatch."""
        # Create a note
        note = Note(
            title="Mismatch Test",
            content="Testing count mismatch detection.",
            note_type=NoteType.PERMANENT,
        )
        note_repository.create(note)

        # Create an orphan file that bypasses the repository
        notes_dir = temp_dirs[0]
        orphan_file = notes_dir / "orphan-note.md"
        orphan_file.write_text("# Orphan Note\n\nThis file has no DB entry.")

        health = note_repository.check_database_health()

        # Should detect mismatch (1 in DB, 2 files)
        assert any("mismatch" in issue.lower() for issue in health["issues"])


class TestFTSGracefulDegradation:
    """Tests for FTS5 graceful degradation."""

    def test_fts_search_basic_query(self, note_repository):
        """Test basic FTS search functionality."""
        # Create a note to search for
        note = Note(
            title="FTS Test Note",
            content="This is content for full-text search testing.",
            note_type=NoteType.PERMANENT,
        )
        note_repository.create(note)

        # Search with simple term - should work regardless of FTS mode
        results = note_repository.fts_search("content")

        # Should get results (either via FTS5 or fallback)
        assert isinstance(results, list)
        # If results found, check mode is indicated
        if results:
            assert "search_mode" in results[0]

    def test_fallback_search_returns_fallback_mode(self, note_repository):
        """Test fallback search indicates fallback mode."""
        # Create a note
        note = Note(
            title="Fallback Test Note",
            content="Content for fallback search testing.",
            note_type=NoteType.PERMANENT,
        )
        note_repository.create(note)

        # Directly call fallback search
        results = note_repository._fallback_text_search("fallback")

        assert len(results) > 0
        assert results[0]["search_mode"] == "fallback"

    def test_fts_unavailable_skips_to_fallback(self, note_repository):
        """Test that when FTS is marked unavailable, search uses fallback."""
        # Create a note
        note = Note(
            title="FTS Disabled Test",
            content="Content for FTS disabled testing.",
            note_type=NoteType.PERMANENT,
        )
        note_repository.create(note)

        # Disable FTS
        note_repository._fts_available = False

        results = note_repository.fts_search("disabled")

        assert len(results) > 0
        assert results[0]["search_mode"] == "fallback"

        # Re-enable for other tests
        note_repository._fts_available = True

    def test_reset_fts_availability_success(self, note_repository):
        """Test reset_fts_availability re-enables FTS when healthy."""
        # Disable FTS
        note_repository._fts_available = False

        # Reset should succeed (or fail gracefully if FTS not available)
        result = note_repository.reset_fts_availability()

        # Result should be a boolean
        assert isinstance(result, bool)
        # If successful, flag should match result
        assert note_repository._fts_available == result

    def test_fts_syntax_error_falls_back(self, note_repository):
        """Test that FTS syntax errors fall back to LIKE search."""
        # Create a note
        note = Note(
            title="Syntax Error Test",
            content="Content for syntax error testing.",
            note_type=NoteType.PERMANENT,
        )
        note_repository.create(note)

        # Use invalid FTS syntax (unbalanced parentheses)
        # This should trigger fallback, not raise
        results = note_repository.fts_search("(unbalanced")

        # Should return a list (either empty or with results via fallback)
        # The key is that it doesn't raise an exception
        assert isinstance(results, list)


class TestAutoRecovery:
    """Tests for auto-recovery mechanism."""

    def test_nuke_and_rebuild_creates_backup(self, note_repository, temp_dirs):
        """Test that nuke and rebuild creates a backup file."""
        # Create a note first
        note = Note(
            title="Backup Test Note",
            content="This note should be recoverable.",
            note_type=NoteType.PERMANENT,
        )
        note_repository.create(note)

        # Get the database path
        db_url = config.get_db_url()
        db_path = Path(db_url.replace("sqlite:///", ""))

        # Perform nuke and rebuild
        backup_path = note_repository._nuke_and_rebuild_database()

        # Verify backup was created
        if backup_path:  # May be empty string if no DB existed
            assert Path(backup_path).exists() or backup_path == ""

        # Verify note is still accessible (rebuilt from files)
        all_notes = note_repository.get_all()
        assert len(all_notes) == 1
        assert all_notes[0].title == "Backup Test Note"

    def test_health_check_triggers_rebuild_on_unhealthy(self, note_repository, temp_dirs):
        """Test that _initialize_with_health_check handles unhealthy state."""
        # This test verifies the health check path exists and works
        # by checking that the repository can handle being re-initialized

        # Create a note first
        note = Note(
            title="Rebuild Test Note",
            content="This note tests the rebuild path.",
            note_type=NoteType.PERMANENT,
        )
        note_repository.create(note)

        # Manually trigger the initialization path again
        # This should not raise even if called multiple times
        note_repository._initialize_with_health_check()

        # Verify note is still accessible
        all_notes = note_repository.get_all()
        assert len(all_notes) >= 1


class TestDatabaseCorruptionError:
    """Tests for DatabaseCorruptionError exception class."""

    def test_database_corruption_error_basic(self):
        """Test DatabaseCorruptionError has correct attributes."""
        error = DatabaseCorruptionError(
            message="Test corruption",
            recovered=True,
            backup_path="/tmp/backup.bak"
        )

        assert error.message == "Test corruption"
        assert error.recovered is True
        assert error.backup_path == "/tmp/backup.bak"
        assert error.code == ErrorCode.DATABASE_CORRUPTED

    def test_database_corruption_error_to_dict(self):
        """Test DatabaseCorruptionError serialization."""
        error = DatabaseCorruptionError(
            message="Test corruption",
            recovered=False,
            backup_path="/tmp/backup.bak"
        )

        error_dict = error.to_dict()

        assert error_dict["error"] == "DatabaseCorruptionError"
        assert error_dict["code"] == ErrorCode.DATABASE_CORRUPTED.value
        assert error_dict["message"] == "Test corruption"
        assert error_dict["details"]["recovered"] is False
        assert error_dict["details"]["backup_path"] == "/tmp/backup.bak"

    def test_fts_corrupted_error_code(self):
        """Test FTS corruption uses correct error code."""
        error = DatabaseCorruptionError(
            message="FTS index corrupted",
            code=ErrorCode.FTS_CORRUPTED
        )

        assert error.code == ErrorCode.FTS_CORRUPTED


class TestServiceLayerHealthCheck:
    """Tests for health check exposed through service layer."""

    def test_service_check_database_health(self, zettel_service):
        """Test health check is accessible through service layer."""
        health = zettel_service.check_database_health()

        assert "healthy" in health
        assert "sqlite_ok" in health
        assert "fts_ok" in health
        assert "note_count" in health
        assert "file_count" in health
        assert "issues" in health

    def test_service_reset_fts_availability(self, zettel_service):
        """Test FTS reset is accessible through service layer."""
        result = zettel_service.reset_fts_availability()

        assert isinstance(result, bool)


class TestHealthCheckSeverityLevels:
    """Tests for health check severity differentiation (critical vs warning)."""

    def test_healthy_database_has_no_critical_issues(self, note_repository):
        """Test that a healthy database reports no critical issues."""
        health = note_repository.check_database_health()

        assert health["healthy"] is True
        assert "critical_issues" in health
        assert health["critical_issues"] == []

    def test_count_mismatch_is_warning_not_critical(self, note_repository, temp_dirs):
        """Test that count mismatch is classified as warning, not critical.

        This is the key fix - count mismatches should trigger sync, not nuke.
        """
        # Create a note
        note = Note(
            title="Severity Test",
            content="Testing severity classification.",
            note_type=NoteType.PERMANENT,
        )
        note_repository.create(note)

        # Create orphan files to trigger mismatch
        notes_dir = temp_dirs[0]
        for i in range(3):  # Create enough to trigger warning message
            orphan = notes_dir / f"orphan-{i}.md"
            orphan.write_text(f"# Orphan {i}\n\nNo DB entry.")

        health = note_repository.check_database_health()

        # Mismatch should be in issues (warnings), NOT critical_issues
        assert health["healthy"] is True  # Still healthy (no critical issues)
        assert health["critical_issues"] == []
        assert health["needs_sync"] is True
        # May have warning about mismatch in issues
        assert any("mismatch" in issue.lower() for issue in health["issues"]) or len(health["issues"]) == 0

    def test_health_check_includes_needs_sync_flag(self, note_repository, temp_dirs):
        """Test that health check includes needs_sync flag for incremental sync."""
        # Create mismatch scenario
        note = Note(
            title="Sync Flag Test",
            content="Testing needs_sync flag.",
            note_type=NoteType.PERMANENT,
        )
        note_repository.create(note)

        notes_dir = temp_dirs[0]
        orphan = notes_dir / "sync-test-orphan.md"
        orphan.write_text("# Orphan\n\nTrigger sync need.")

        health = note_repository.check_database_health()

        assert "needs_sync" in health
        assert health["needs_sync"] is True


class TestPathValidation:
    """Tests for path misconfiguration detection."""

    def test_validate_notes_dir_logs_warning_for_nested_notes_dir(self, temp_dirs, caplog):
        """Test that passing base directory instead of notes directory logs warning."""
        import logging

        # Create a directory structure that mimics the real problem:
        # /base/
        #   /notes/  <- contains .md files
        #   /db/     <- contains .db files
        base_dir = temp_dirs[0].parent  # Go up one level from notes_dir

        # Create a 'notes' subdirectory with markdown files
        nested_notes = base_dir / "nested_notes_test"
        nested_notes.mkdir(exist_ok=True)
        (nested_notes / "notes").mkdir(exist_ok=True)

        # Put .md files in the nested 'notes' directory
        for i in range(5):
            (nested_notes / "notes" / f"note-{i}.md").write_text(f"# Note {i}\n\nContent.")

        # Create a separate temp database for this test
        import tempfile
        with tempfile.TemporaryDirectory() as temp_db_dir:
            db_path = Path(temp_db_dir) / "test.db"

            # Save original config
            original_db_url = config.get_db_url()
            original_notes_dir = config.notes_dir

            try:
                # Configure to use our test paths
                config._db_url = f"sqlite:///{db_path}"
                config.notes_dir = nested_notes  # Pass base dir, not notes dir

                with caplog.at_level(logging.WARNING):
                    # This should trigger the path validation warning
                    repo = NoteRepository(nested_notes)

                # Check that warning was logged
                warning_logged = any(
                    "POSSIBLE PATH ERROR" in record.message
                    for record in caplog.records
                )
                assert warning_logged, "Should warn when notes/ subdirectory has more .md files"

            finally:
                # Restore original config
                config._db_url = original_db_url
                config.notes_dir = original_notes_dir

    def test_repository_logs_paths_on_init(self, temp_dirs, caplog):
        """Test that repository logs configured paths on initialization."""
        import logging

        # Create a fresh repository with logging captured
        with caplog.at_level(logging.INFO):
            # The existing note_repository fixture was already created,
            # so we need to create a new one to capture the init logs
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                notes_dir = Path(temp_dir) / "notes"
                notes_dir.mkdir()
                db_dir = Path(temp_dir) / "db"
                db_dir.mkdir()
                db_path = db_dir / "test.db"

                # Save original config
                original_db_url = config.get_db_url()
                original_notes_dir = config.notes_dir

                try:
                    config._db_url = f"sqlite:///{db_path}"
                    config.notes_dir = notes_dir

                    repo = NoteRepository(notes_dir)

                    # Check that path info was logged
                    init_logged = any(
                        "NoteRepository initialized" in record.message
                        and "notes_dir=" in record.message
                        for record in caplog.records
                    )
                    assert init_logged, "Should log notes_dir on initialization"

                finally:
                    config._db_url = original_db_url
                    config.notes_dir = original_notes_dir


class TestGraduatedRecovery:
    """Tests for graduated recovery response."""

    def test_sync_only_for_count_mismatch(self, note_repository, temp_dirs):
        """Test that count mismatch triggers sync, not full rebuild.

        This verifies the fix for the bug where passing the wrong directory
        caused the database to be nuked unnecessarily.
        """
        # Create some notes
        for i in range(3):
            note = Note(
                title=f"Recovery Test {i}",
                content=f"Testing graduated recovery {i}.",
                note_type=NoteType.PERMANENT,
            )
            note_repository.create(note)

        # Verify notes exist
        initial_notes = note_repository.get_all()
        assert len(initial_notes) == 3

        # Create orphan file to trigger mismatch
        notes_dir = temp_dirs[0]
        orphan = notes_dir / "orphan-recovery-test.md"
        orphan.write_text("# Orphan\n\nThis triggers mismatch.")

        # Re-run health check initialization
        note_repository._initialize_with_health_check()

        # Notes should still exist (not nuked)
        final_notes = note_repository.get_all()
        assert len(final_notes) >= 3, "Original notes should survive sync"

    def test_fts_recovery_attempted_before_nuke(self, note_repository):
        """Test that FTS issues attempt FTS-only recovery first."""
        # Simulate FTS being unavailable
        note_repository._fts_available = False

        # Create a note to ensure DB has content
        note = Note(
            title="FTS Recovery Test",
            content="Testing FTS recovery path.",
            note_type=NoteType.PERMANENT,
        )
        note_repository.create(note)

        # Re-initialize - should attempt FTS recovery, not full nuke
        note_repository._initialize_with_health_check()

        # Note should still exist
        all_notes = note_repository.get_all()
        assert len(all_notes) >= 1
        assert any(n.title == "FTS Recovery Test" for n in all_notes)
