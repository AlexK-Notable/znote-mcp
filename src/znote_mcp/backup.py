"""Backup utilities for the Zettelkasten MCP server.

Provides automated and manual backup capabilities for:
- SQLite database
- Markdown note files
- Configuration files
"""
import gzip
import logging
import os
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Union

from znote_mcp.config import config

logger = logging.getLogger(__name__)

# Default backup directory
DEFAULT_BACKUP_DIR = Path.home() / ".zettelkasten" / "backups"

# Backup retention settings
DEFAULT_MAX_BACKUPS = 10  # Keep last N backups
DEFAULT_MAX_AGE_DAYS = 30  # Delete backups older than N days


class BackupManager:
    """Manages database and file backups with rotation.

    Features:
    - SQLite online backup (safe during writes)
    - Gzip compression for space efficiency
    - Automatic rotation by count and age
    - Manifest tracking for easy restore
    """

    def __init__(
        self,
        backup_dir: Optional[Union[str, Path]] = None,
        max_backups: int = DEFAULT_MAX_BACKUPS,
        max_age_days: int = DEFAULT_MAX_AGE_DAYS,
    ):
        """Initialize the backup manager.

        Args:
            backup_dir: Directory for backups. Defaults to ~/.zettelkasten/backups/
            max_backups: Maximum number of backups to keep
            max_age_days: Delete backups older than this many days
        """
        self.backup_dir = Path(backup_dir) if backup_dir else DEFAULT_BACKUP_DIR
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.max_backups = max_backups
        self.max_age_days = max_age_days
        self._lock = Lock()

    def backup_database(
        self,
        compress: bool = True,
        label: Optional[str] = None,
    ) -> Optional[Path]:
        """Create a backup of the SQLite database.

        Uses SQLite's online backup API for a consistent snapshot,
        even if the database is being written to.

        Args:
            compress: Gzip compress the backup (default: True)
            label: Optional label to include in filename

        Returns:
            Path to the backup file, or None if backup failed.

        Example:
            backup_path = manager.backup_database(label="pre-migration")
        """
        with self._lock:
            try:
                # Get database path from config
                db_url = config.get_db_url()
                if not db_url.startswith("sqlite:///"):
                    logger.error("Backup only supports SQLite databases")
                    return None

                db_path = Path(db_url.replace("sqlite:///", ""))
                if not db_path.exists():
                    logger.warning(f"Database not found: {db_path}")
                    return None

                # Generate backup filename
                timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
                label_part = f"_{label}" if label else ""
                ext = ".db.gz" if compress else ".db"
                backup_name = f"zettelkasten_{timestamp}{label_part}{ext}"
                backup_path = self.backup_dir / backup_name

                # Use SQLite online backup API for consistency
                if compress:
                    # Backup to temp file, then compress
                    temp_path = backup_path.with_suffix("")
                    self._sqlite_backup(db_path, temp_path)
                    self._gzip_file(temp_path, backup_path)
                    temp_path.unlink()
                else:
                    self._sqlite_backup(db_path, backup_path)

                # Get backup size
                size_mb = backup_path.stat().st_size / (1024 * 1024)
                logger.info(f"Database backup created: {backup_path} ({size_mb:.2f} MB)")

                # Rotate old backups
                self._rotate_backups()

                return backup_path

            except Exception as e:
                logger.error(f"Database backup failed: {e}", exc_info=True)
                return None

    def _sqlite_backup(self, source: Path, dest: Path) -> None:
        """Perform SQLite online backup.

        This is safe even during concurrent writes.
        """
        source_conn = sqlite3.connect(str(source))
        dest_conn = sqlite3.connect(str(dest))

        try:
            source_conn.backup(dest_conn)
        finally:
            dest_conn.close()
            source_conn.close()

    def _gzip_file(self, source: Path, dest: Path) -> None:
        """Compress a file with gzip."""
        with open(source, "rb") as f_in:
            with gzip.open(dest, "wb", compresslevel=6) as f_out:
                shutil.copyfileobj(f_in, f_out)

    def backup_notes(
        self,
        compress: bool = True,
        label: Optional[str] = None,
    ) -> Optional[Path]:
        """Create a backup of all markdown note files.

        Creates a compressed archive of the notes directory.

        Args:
            compress: Create a tar.gz archive (default: True)
            label: Optional label to include in filename

        Returns:
            Path to the backup archive, or None if backup failed.
        """
        with self._lock:
            try:
                notes_dir = config.get_absolute_path(config.notes_dir)
                if not notes_dir.exists():
                    logger.warning(f"Notes directory not found: {notes_dir}")
                    return None

                # Generate backup filename
                timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
                label_part = f"_{label}" if label else ""
                ext = ".tar.gz" if compress else ".tar"
                backup_name = f"notes_{timestamp}{label_part}{ext}"
                backup_path = self.backup_dir / backup_name

                # Create archive
                fmt = "gztar" if compress else "tar"
                archive_base = str(backup_path).replace(".tar.gz", "").replace(".tar", "")
                shutil.make_archive(
                    archive_base,
                    fmt,
                    root_dir=notes_dir.parent,
                    base_dir=notes_dir.name,
                )

                # Get backup size
                size_mb = backup_path.stat().st_size / (1024 * 1024)
                logger.info(f"Notes backup created: {backup_path} ({size_mb:.2f} MB)")

                return backup_path

            except Exception as e:
                logger.error(f"Notes backup failed: {e}", exc_info=True)
                return None

    def create_full_backup(
        self,
        label: Optional[str] = None,
    ) -> Dict[str, Optional[Path]]:
        """Create a complete backup of database and notes.

        Args:
            label: Optional label for the backup

        Returns:
            Dictionary with paths to created backups.
        """
        return {
            "database": self.backup_database(compress=True, label=label),
            "notes": self.backup_notes(compress=True, label=label),
        }

    def _rotate_backups(self) -> int:
        """Remove old backups based on count and age limits.

        Returns:
            Number of backups removed.
        """
        removed = 0
        now = datetime.now(timezone.utc)

        # Get all backup files, sorted by modification time (newest first)
        db_backups = sorted(
            self.backup_dir.glob("zettelkasten_*.db*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        notes_backups = sorted(
            self.backup_dir.glob("notes_*.tar*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        for backups in [db_backups, notes_backups]:
            # Remove by count (keep newest N)
            for backup in backups[self.max_backups:]:
                try:
                    backup.unlink()
                    removed += 1
                    logger.debug(f"Removed old backup (count limit): {backup}")
                except OSError:
                    pass

            # Remove by age
            max_age_seconds = self.max_age_days * 24 * 60 * 60
            for backup in backups[:self.max_backups]:
                try:
                    age = now.timestamp() - backup.stat().st_mtime
                    if age > max_age_seconds:
                        backup.unlink()
                        removed += 1
                        logger.debug(f"Removed old backup (age limit): {backup}")
                except OSError:
                    pass

        if removed > 0:
            logger.info(f"Rotated {removed} old backup(s)")

        return removed

    def list_backups(self) -> List[Dict[str, any]]:
        """List all available backups.

        Returns:
            List of backup metadata dictionaries.
        """
        backups = []

        for pattern in ["zettelkasten_*.db*", "notes_*.tar*"]:
            for path in self.backup_dir.glob(pattern):
                stat = path.stat()
                backups.append({
                    "path": str(path),
                    "name": path.name,
                    "type": "database" if "zettelkasten" in path.name else "notes",
                    "size_bytes": stat.st_size,
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "created_at": datetime.fromtimestamp(
                        stat.st_mtime, tz=timezone.utc
                    ).isoformat(),
                })

        # Sort by creation time (newest first)
        backups.sort(key=lambda b: b["created_at"], reverse=True)
        return backups

    def restore_database(self, backup_path: Union[str, Path]) -> bool:
        """Restore database from a backup.

        WARNING: This will overwrite the current database.

        Args:
            backup_path: Path to the backup file

        Returns:
            True if restore succeeded, False otherwise.
        """
        with self._lock:
            try:
                backup_path = Path(backup_path)
                if not backup_path.exists():
                    logger.error(f"Backup not found: {backup_path}")
                    return False

                # Get database path
                db_url = config.get_db_url()
                if not db_url.startswith("sqlite:///"):
                    logger.error("Restore only supports SQLite databases")
                    return False

                db_path = Path(db_url.replace("sqlite:///", ""))

                # Create backup of current database before restore
                if db_path.exists():
                    self.backup_database(label="pre-restore")

                # Decompress if needed
                if backup_path.suffix == ".gz":
                    with gzip.open(backup_path, "rb") as f_in:
                        with open(db_path, "wb") as f_out:
                            shutil.copyfileobj(f_in, f_out)
                else:
                    shutil.copy2(backup_path, db_path)

                logger.info(f"Database restored from: {backup_path}")
                return True

            except Exception as e:
                logger.error(f"Database restore failed: {e}", exc_info=True)
                return False


# Global backup manager instance
backup_manager = BackupManager()


def backup_now(label: Optional[str] = None) -> Dict[str, Optional[Path]]:
    """Convenience function to create a full backup.

    Args:
        label: Optional label for the backup

    Returns:
        Dictionary with paths to created backups.

    Example:
        from znote_mcp.backup import backup_now
        result = backup_now(label="before-migration")
    """
    return backup_manager.create_full_backup(label=label)
