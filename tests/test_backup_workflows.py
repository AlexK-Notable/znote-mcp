"""Tests for backup workflows.

Tests for database backup, restore, and rotation functionality.
"""

import os
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from znote_mcp.backup import BackupManager, backup_now


class TestBackupManager:
    """Tests for BackupManager class."""

    @pytest.fixture
    def temp_backup_dir(self):
        """Create a temporary backup directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def temp_db_file(self):
        """Create a temporary SQLite database with test data."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)

        # Create a simple database with test data
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE test_table (id INTEGER PRIMARY KEY, name TEXT)")
        conn.execute("INSERT INTO test_table (name) VALUES ('test_value')")
        conn.commit()
        conn.close()

        yield db_path

        if db_path.exists():
            db_path.unlink()

    @pytest.fixture
    def temp_notes_dir(self):
        """Create a temporary notes directory with test files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            notes_path = Path(temp_dir)
            # Create some test note files
            (notes_path / "note1.md").write_text("# Note 1\nContent 1")
            (notes_path / "note2.md").write_text("# Note 2\nContent 2")
            yield notes_path

    @pytest.fixture
    def backup_manager(self, temp_backup_dir):
        """Create a BackupManager with temp directory."""
        return BackupManager(backup_dir=temp_backup_dir, max_backups=5, max_age_days=30)

    def test_backup_manager_creates_directory(self, temp_backup_dir):
        """Test that BackupManager creates backup directory if needed."""
        new_dir = temp_backup_dir / "subdir" / "backups"
        manager = BackupManager(backup_dir=new_dir)
        assert new_dir.exists()

    def test_backup_database(self, backup_manager, temp_db_file):
        """Test backing up a SQLite database."""
        with patch("znote_mcp.backup.config") as mock_config:
            mock_config.get_db_url.return_value = f"sqlite:///{temp_db_file}"

            backup_path = backup_manager.backup_database(compress=False)

            assert backup_path is not None
            assert backup_path.exists()
            assert backup_path.suffix == ".db"
            # Verify backup contains valid data
            conn = sqlite3.connect(str(backup_path))
            cursor = conn.execute("SELECT name FROM test_table")
            result = cursor.fetchone()
            conn.close()
            assert result[0] == "test_value"

    def test_backup_database_compressed(self, backup_manager, temp_db_file):
        """Test backing up a database with compression."""
        with patch("znote_mcp.backup.config") as mock_config:
            mock_config.get_db_url.return_value = f"sqlite:///{temp_db_file}"

            backup_path = backup_manager.backup_database(compress=True)

            assert backup_path is not None
            assert backup_path.exists()
            assert backup_path.suffix == ".gz"
            # File should be smaller than original due to compression
            assert backup_path.stat().st_size > 0

    def test_backup_database_with_label(self, backup_manager, temp_db_file):
        """Test backup with custom label."""
        with patch("znote_mcp.backup.config") as mock_config:
            mock_config.get_db_url.return_value = f"sqlite:///{temp_db_file}"

            backup_path = backup_manager.backup_database(label="pre-migration")

            assert backup_path is not None
            assert "pre-migration" in backup_path.name

    def test_backup_notes(self, backup_manager, temp_notes_dir):
        """Test backing up notes directory."""
        with patch("znote_mcp.backup.config") as mock_config:
            mock_config.get_absolute_path.return_value = temp_notes_dir
            mock_config.notes_dir = temp_notes_dir

            backup_path = backup_manager.backup_notes(compress=True)

            assert backup_path is not None
            assert backup_path.exists()
            assert backup_path.suffix == ".gz"

    def test_list_backups(self, backup_manager, temp_db_file):
        """Test listing available backups."""
        with patch("znote_mcp.backup.config") as mock_config:
            mock_config.get_db_url.return_value = f"sqlite:///{temp_db_file}"

            # Create multiple backups
            backup_manager.backup_database(label="first")
            backup_manager.backup_database(label="second")

            backups = backup_manager.list_backups()

            assert len(backups) >= 2
            # Check backup metadata
            assert all("path" in b for b in backups)
            assert all("name" in b for b in backups)
            assert all("type" in b for b in backups)
            assert all("size_bytes" in b for b in backups)

    def test_restore_database(self, temp_backup_dir, temp_db_file):
        """Test restoring a database from backup."""
        # Create manager with backup_dir
        manager = BackupManager(backup_dir=temp_backup_dir)

        with patch("znote_mcp.backup.config") as mock_config:
            mock_config.get_db_url.return_value = f"sqlite:///{temp_db_file}"

            # Create backup
            backup_path = manager.backup_database(compress=False)

            # Modify the original database
            conn = sqlite3.connect(str(temp_db_file))
            conn.execute("INSERT INTO test_table (name) VALUES ('new_value')")
            conn.commit()
            conn.close()

            # Verify modification
            conn = sqlite3.connect(str(temp_db_file))
            cursor = conn.execute("SELECT COUNT(*) FROM test_table")
            count_before = cursor.fetchone()[0]
            conn.close()
            assert count_before == 2

            # Patch backup_database to skip pre-restore backup (avoids infinite recursion)
            with patch.object(manager, "backup_database", return_value=None):
                # Restore from backup
                result = manager.restore_database(backup_path)

            assert result is True

            # Verify restoration
            conn = sqlite3.connect(str(temp_db_file))
            cursor = conn.execute("SELECT COUNT(*) FROM test_table")
            count_after = cursor.fetchone()[0]
            conn.close()
            assert count_after == 1  # Original data only

    def test_restore_rejects_path_traversal(self, backup_manager, temp_backup_dir):
        """Test that restore rejects paths outside backup directory."""
        # Try to restore from outside backup_dir
        outside_path = Path("/tmp/malicious_file.db")

        with patch("znote_mcp.backup.config") as mock_config:
            mock_config.get_db_url.return_value = "sqlite:///test.db"

            result = backup_manager.restore_database(outside_path)

            assert result is False


class TestSymlinkProtection:
    """Tests for symlink protection in restore."""

    @pytest.fixture
    def backup_setup(self):
        """Set up backup directories and files."""
        with tempfile.TemporaryDirectory() as backup_dir:
            with tempfile.TemporaryDirectory() as db_dir:
                backup_path = Path(backup_dir)
                db_path = Path(db_dir) / "test.db"

                # Create a test database
                conn = sqlite3.connect(str(db_path))
                conn.execute("CREATE TABLE t (id INTEGER)")
                conn.commit()
                conn.close()

                manager = BackupManager(backup_dir=backup_path)

                yield {
                    "backup_dir": backup_path,
                    "db_path": db_path,
                    "manager": manager,
                }

    def test_restore_rejects_symlinks(self, backup_setup):
        """Test that restore rejects symlink backup paths."""
        backup_dir = backup_setup["backup_dir"]
        manager = backup_setup["manager"]

        # Create a real backup file
        real_backup = backup_dir / "real_backup.db"
        real_backup.write_text("dummy")

        # Create a symlink to external file
        symlink_path = backup_dir / "symlink_backup.db"
        external_file = Path("/tmp/external_file")
        external_file.write_text("external")

        try:
            symlink_path.symlink_to(external_file)

            with patch("znote_mcp.backup.config") as mock_config:
                mock_config.get_db_url.return_value = (
                    f"sqlite:///{backup_setup['db_path']}"
                )

                # Should reject symlink
                result = manager.restore_database(symlink_path)
                assert result is False
        finally:
            if symlink_path.exists():
                symlink_path.unlink()
            if external_file.exists():
                external_file.unlink()


class TestBackupRotation:
    """Tests for backup rotation functionality."""

    @pytest.fixture
    def rotation_manager(self):
        """Create a BackupManager with low rotation limits."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = BackupManager(
                backup_dir=Path(temp_dir), max_backups=3, max_age_days=1
            )
            yield manager

    def test_rotate_by_count(self, rotation_manager):
        """Test that old backups are rotated by count."""
        backup_dir = rotation_manager.backup_dir

        # Create more backups than max_backups
        for i in range(5):
            backup_file = backup_dir / f"zettelkasten_test{i}.db.gz"
            backup_file.write_text(f"backup {i}")

        # Force rotation
        removed = rotation_manager._rotate_backups()

        # Should have removed 2 (5 - 3 = 2)
        remaining = list(backup_dir.glob("zettelkasten_*.db*"))
        assert len(remaining) == 3


class TestBackupNow:
    """Tests for the backup_now convenience function."""

    def test_backup_now_returns_dict(self):
        """Test that backup_now returns a dictionary."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("znote_mcp.backup.backup_manager") as mock_manager:
                mock_manager.create_full_backup.return_value = {
                    "database": Path(temp_dir) / "db.gz",
                    "notes": Path(temp_dir) / "notes.tar.gz",
                }

                result = backup_now(label="test")

                assert isinstance(result, dict)
                mock_manager.create_full_backup.assert_called_once_with(label="test")
