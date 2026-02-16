"""Repository for note storage and retrieval."""

import datetime
import logging
import os
import re
import shutil
import sqlite3
import threading
import weakref
from datetime import timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml
from sqlalchemy import and_, func, or_, select, text
from sqlalchemy.exc import DatabaseError as SQLAlchemyDatabaseError
from sqlalchemy.orm import Session, joinedload

from znote_mcp.config import config
from znote_mcp.exceptions import (
    BulkOperationError,
    DatabaseCorruptionError,
    ErrorCode,
    NoteNotFoundError,
    StorageError,
    ValidationError,
)
from znote_mcp.models.db_models import (
    DBLink,
    DBNote,
    DBTag,
    get_session_factory,
    init_db,
    note_tags,
)
from znote_mcp.models.schema import (
    ConflictResult,
    Link,
    LinkType,
    Note,
    NotePurpose,
    NoteType,
    Tag,
    VersionedNote,
    VersionInfo,
    ensure_timezone_aware,
    utc_now,
    validate_project_path,
    validate_safe_path_component,
)
from znote_mcp.storage.base import Repository
from znote_mcp.storage.fts_index import FtsIndex
from znote_mcp.storage.git_wrapper import GitConflictError, GitWrapper
from znote_mcp.storage.markdown_parser import MarkdownParser
from znote_mcp.storage.obsidian_mirror import ObsidianMirror
from znote_mcp.utils import escape_like_pattern, sanitize_for_terminal

logger = logging.getLogger(__name__)


def _sanitize_commit_message(title: str, max_length: int = 100) -> str:
    """Sanitize note title for use in git commit message.

    Prevents potential issues from malicious or malformed titles:
    - Truncates to reasonable length
    - Removes newlines (could corrupt git log parsing)
    - Prefixes titles starting with dash (could be confused for git flags)

    Args:
        title: The note title to sanitize.
        max_length: Maximum length for the sanitized title.

    Returns:
        Sanitized title safe for use in git commit messages.
    """
    # Truncate to reasonable length
    sanitized = title[:max_length]
    # Remove/replace newlines and carriage returns
    sanitized = sanitized.replace("\n", " ").replace("\r", " ")
    # Ensure doesn't start with dash (git flag confusion)
    if sanitized.startswith("-"):
        sanitized = "_" + sanitized
    return sanitized


class NoteRepository(Repository[Note]):
    """Repository for note storage and retrieval.
    This implements a dual storage approach:
    1. Notes are stored as Markdown files on disk for human readability and editing
    2. SQLite database (with WAL mode) is used for indexing and efficient querying
    The file system is the source of truth - database is rebuilt from files if needed.
    """

    def __init__(
        self,
        notes_dir: Optional[Path] = None,
        database_path: Optional[Path] = None,
        obsidian_vault_path: Optional[Path] = None,
        use_git: bool = True,
        in_memory_db: bool = True,
        engine: Optional[Any] = None,
    ):
        """Initialize the repository.

        Args:
            notes_dir: Path to directory containing note markdown files.
                       If None, uses config.notes_dir.
            database_path: Path to SQLite database file.
                          Ignored when in_memory_db=True or engine is provided.
            obsidian_vault_path: Path to Obsidian vault for mirroring.
                                If None, uses config setting.
            use_git: If True, enable git versioning for notes. Default True.
            in_memory_db: If True, use in-memory SQLite database. Default True.
                         When True, database is rebuilt from files on startup.
                         Ignored when engine is provided.
            engine: Pre-configured SQLAlchemy engine. When provided, the
                    repository uses this engine directly instead of calling
                    init_db(). This allows sharing a single engine across
                    all repositories.

        Note:
            The notes_dir should be the directory CONTAINING the .md files,
            not the base zettelkasten directory. If you pass the base directory
            by mistake, the health check will find 0 files and may incorrectly
            trigger recovery.
        """
        self.notes_dir = (
            config.get_absolute_path(notes_dir)
            if notes_dir
            else config.get_absolute_path(config.notes_dir)
        )

        # Store configuration options
        self.use_git = use_git
        # If engine is provided, in_memory_db is derived from the engine URL
        self.in_memory_db = (
            in_memory_db
            if engine is None
            else (
                str(getattr(engine, "url", "")) == "sqlite:///:memory:"
                or ":memory:" in str(getattr(engine, "url", ""))
            )
        )

        # Ensure directories exist
        self.notes_dir.mkdir(parents=True, exist_ok=True)

        # PATH VALIDATION: Warn about common misconfigurations
        self._validate_notes_dir()

        # Log the configuration being used
        logger.info(
            f"NoteRepository initialized: notes_dir={self.notes_dir}, "
            f"db_url={config.get_db_url() if not self.in_memory_db else ':memory:'}, "
            f"use_git={use_git}, in_memory_db={self.in_memory_db}"
        )

        # Initialize Obsidian vault mirror (optional)
        self.obsidian_vault_path = (
            obsidian_vault_path
            if obsidian_vault_path
            else config.get_obsidian_vault_path()
        )
        if self.obsidian_vault_path:
            logger.info(f"Obsidian vault mirror enabled: {self.obsidian_vault_path}")

        # Initialize GitWrapper for version control (if enabled)
        self._git: Optional[GitWrapper] = None
        if use_git:
            self._git = GitWrapper(self.notes_dir)
            logger.info(f"Git versioning enabled for {self.notes_dir}")

        # Initialize database — use provided engine or create one
        if engine is not None:
            self.engine = engine
        else:
            self.engine = init_db(in_memory=in_memory_db)
        self.session_factory = get_session_factory(self.engine)

        # Extracted subsystems
        self._parser = MarkdownParser()
        self._fts = FtsIndex(self.engine, self.session_factory)
        # ObsidianMirror is created lazily — only when vault path is configured
        self._obsidian: Optional[ObsidianMirror] = None
        if self.obsidian_vault_path:
            self._obsidian = ObsidianMirror(
                self.obsidian_vault_path,
                note_resolver=self.get,
            )

        # File access lock (for bulk operations on multiple files)
        self.file_lock = threading.RLock()

        # Per-note locks to prevent update-delete races (uses WeakValueDictionary
        # so locks are garbage collected when no longer held by any thread)
        self._note_locks: weakref.WeakValueDictionary[str, threading.RLock] = (
            weakref.WeakValueDictionary()
        )
        self._note_locks_lock = threading.Lock()  # Protects _note_locks dict access

        # sqlite-vec availability tracking - allows graceful degradation
        # Detected from engine: if init_sqlite_vec() succeeded during init_db(),
        # the vec0 table exists and the extension is loaded on every connection.
        self._vec_available: bool = self._detect_vec_available()

        # Clean up any orphaned staging files from previous failed operations
        self._cleanup_staging()

        # Check database health and recover if needed
        # For in-memory databases, always rebuild from files
        if in_memory_db:
            self.rebuild_index()
        else:
            self._initialize_with_health_check()

    @property
    def _fts_available(self) -> bool:
        """Whether FTS5 is available (delegates to FtsIndex subsystem)."""
        return self._fts.available

    @_fts_available.setter
    def _fts_available(self, value: bool) -> None:
        """Set FTS5 availability (delegates to FtsIndex subsystem)."""
        self._fts.available = value

    def _detect_vec_available(self) -> bool:
        """Check if sqlite-vec virtual table exists and is usable."""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT count(*) FROM note_embeddings WHERE 0"))
                return True
        except Exception:
            return False

    def _validate_notes_dir(self) -> None:
        """Validate that notes_dir looks like a valid notes directory.

        Logs warnings for common misconfigurations that could lead to
        incorrect health check results or data loss during recovery.
        """
        # Check if this looks like a base directory rather than notes directory
        potential_notes_subdir = self.notes_dir / "notes"
        if potential_notes_subdir.is_dir():
            md_files_in_subdir = len(list(potential_notes_subdir.glob("*.md")))
            md_files_here = len(list(self.notes_dir.glob("*.md")))
            if md_files_in_subdir > md_files_here:
                logger.warning(
                    f"POSSIBLE PATH ERROR: notes_dir={self.notes_dir} contains a "
                    f"'notes' subdirectory with {md_files_in_subdir} .md files, "
                    f"but only {md_files_here} .md files in the configured directory. "
                    "Did you mean to use the 'notes' subdirectory?"
                )

        # Check if there's a db directory that suggests this is base dir
        potential_db_subdir = self.notes_dir / "db"
        if potential_db_subdir.is_dir():
            db_files = list(potential_db_subdir.glob("*.db"))
            if db_files:
                logger.warning(
                    f"POSSIBLE PATH ERROR: notes_dir={self.notes_dir} contains a "
                    f"'db' subdirectory with database files. This suggests the "
                    f"configured path is the base directory, not the notes directory."
                )

        # Validate database path consistency
        db_url = config.get_db_url()
        db_path = Path(db_url.replace("sqlite:///", ""))
        try:
            # Check if notes_dir and db_path share a common ancestor
            notes_parts = self.notes_dir.resolve().parts
            db_parts = db_path.resolve().parent.parts

            # They should share at least some common path structure
            common_depth = sum(1 for a, b in zip(notes_parts, db_parts) if a == b)
            if common_depth < 3 and len(notes_parts) > 3 and len(db_parts) > 3:
                logger.warning(
                    f"POSSIBLE PATH MISMATCH: notes_dir={self.notes_dir} and "
                    f"database path={db_path} don't appear to be related. "
                    "The health check compares DB contents against files in notes_dir. "
                    "If these paths are mismatched, this could cause incorrect recovery."
                )
        except (ValueError, OSError) as e:
            # Path resolution failed, skip this check
            logger.debug(f"Could not validate path consistency: {e}")

    def _cleanup_staging(self) -> None:
        """Clean up orphaned staging files from previous failed operations.

        Called during __init__ to ensure the staging directory is clean.
        This handles the case where a previous bulk operation crashed after
        writing staging files but before moving them to final locations.

        The staging directory (.staging) is used by bulk_create_notes() to
        achieve atomic file operations - files are written to staging first,
        then atomically moved to their final location after DB commit succeeds.
        """
        staging_dir = self.notes_dir / ".staging"
        if not staging_dir.exists():
            return

        orphaned_files = list(staging_dir.glob("*.md"))
        if orphaned_files:
            logger.warning(
                f"Found {len(orphaned_files)} orphaned staging files from previous "
                f"failed operation. Cleaning up..."
            )
            for file_path in orphaned_files:
                try:
                    file_path.unlink()
                    logger.debug(f"Removed orphaned staging file: {file_path.name}")
                except OSError as e:
                    logger.warning(
                        f"Failed to remove orphaned staging file {file_path.name}: {e}"
                    )

            # Try to remove the staging directory if empty
            try:
                staging_dir.rmdir()
            except OSError:
                # Directory not empty or other error - ignore
                pass

    def _initialize_with_health_check(self) -> None:
        """Initialize database with health check and auto-recovery.

        Performs a comprehensive health check on startup with graduated response:
        - CRITICAL issues (corruption): Backup and rebuild entire database
        - FTS issues: Attempt FTS-only rebuild
        - Sync issues: Just sync index from files

        This prevents over-aggressive recovery that could destroy a valid database
        due to path misconfiguration or minor sync issues.
        """
        try:
            health = self.check_database_health()

            # Log health check results for debugging
            logger.info(
                f"Database health check: healthy={health['healthy']}, "
                f"sqlite_ok={health['sqlite_ok']}, fts_ok={health['fts_ok']}, "
                f"notes={health['note_count']}, files={health['file_count']}"
            )

            if health.get("critical_issues"):
                # CRITICAL: Actual database corruption - must rebuild
                logger.error(
                    f"Critical database issues detected: {health['critical_issues']}. "
                    f"Notes dir: {self.notes_dir}, DB: {config.get_db_url()}"
                )
                logger.warning("Initiating database rebuild from source files...")
                self._nuke_and_rebuild_database()
                logger.info("Database auto-recovery completed successfully")
                self._fts_available = True
            else:
                # Database structure is OK
                self._fts_available = health.get("fts_ok", True)

                # Handle FTS issues (not critical, just rebuild FTS)
                if not self._fts_available:
                    logger.warning(
                        "FTS5 index unhealthy but SQLite OK. "
                        "Attempting FTS rebuild..."
                    )
                    try:
                        self.rebuild_fts()
                        self._fts_available = True
                    except Exception as fts_err:
                        logger.error(f"FTS rebuild failed: {fts_err}")

                # Handle sync issues (just sync, don't nuke)
                if health.get("needs_sync"):
                    if health.get("issues"):
                        logger.info(f"Sync needed: {health['issues']}")
                    self.rebuild_index_if_needed()

        except sqlite3.DatabaseError as e:
            if "malformed" in str(e).lower() or "corrupt" in str(e).lower():
                logger.error(
                    f"Database corruption detected during health check: {e}. "
                    f"Notes dir: {self.notes_dir}, DB: {config.get_db_url()}"
                )
                self._nuke_and_rebuild_database()
                self._fts_available = True
            else:
                raise

    def check_database_health(self) -> Dict[str, Any]:
        """Perform comprehensive database health check.

        Checks both SQLite integrity and FTS5 index integrity.
        Distinguishes between CRITICAL issues (corruption) and WARNING issues
        (sync mismatches) to prevent over-aggressive recovery.

        Returns:
            Dict with keys:
                - healthy: bool indicating overall health (no critical issues)
                - sqlite_ok: bool for SQLite integrity
                - fts_ok: bool for FTS5 integrity
                - note_count: int of notes in database
                - file_count: int of markdown files
                - issues: list of issue descriptions (warnings, not critical)
                - critical_issues: list of critical issues requiring recovery
                - needs_sync: bool indicating if sync is needed
        """
        issues = []  # Non-critical issues (warnings)
        critical_issues = []  # Issues requiring database rebuild
        sqlite_ok = False
        fts_ok = False
        note_count = 0
        file_count = len(list(self.notes_dir.glob("*.md")))

        # Log exactly what paths we're checking (crucial for debugging)
        logger.debug(
            f"Health check: notes_dir={self.notes_dir}, "
            f"db_url={config.get_db_url()}"
        )

        try:
            with self.session_factory() as session:
                # Check SQLite integrity - CRITICAL if fails
                result = session.execute(text("PRAGMA integrity_check")).fetchone()
                sqlite_ok = result[0] == "ok"
                if not sqlite_ok:
                    critical_issues.append(
                        f"SQLite integrity check failed: {result[0]}"
                    )

                # Get note count
                note_count = session.scalar(select(func.count(DBNote.id)))

                # Check FTS5 integrity - NOT critical, can be rebuilt separately
                try:
                    session.execute(
                        text(
                            "INSERT INTO notes_fts(notes_fts) VALUES('integrity-check')"
                        )
                    )
                    fts_ok = True
                except (sqlite3.DatabaseError, SQLAlchemyDatabaseError) as e:
                    issues.append(f"FTS5 integrity check failed: {e}")
                    fts_ok = False

        except sqlite3.DatabaseError as e:
            critical_issues.append(f"Database access error: {e}")
            if "malformed" in str(e).lower():
                critical_issues.append("Database file appears to be corrupted")

        # Check count mismatch - this is a WARNING, not critical
        # Small mismatches are normal during operations
        needs_sync = False
        if note_count != file_count:
            mismatch_pct = (
                abs(note_count - file_count) / max(note_count, file_count, 1) * 100
            )
            needs_sync = True
            # Only log as issue if significant mismatch
            if mismatch_pct > 10 or abs(note_count - file_count) > 5:
                issues.append(
                    f"Note count mismatch: {note_count} in DB vs {file_count} files "
                    f"({mismatch_pct:.1f}% difference) - will sync"
                )
            else:
                logger.debug(
                    f"Minor count mismatch: {note_count} in DB vs {file_count} files"
                )

        # Only critical issues make the database "unhealthy" (requiring rebuild)
        # Regular issues just need sync or FTS rebuild
        healthy = sqlite_ok and not critical_issues

        return {
            "healthy": healthy,
            "sqlite_ok": sqlite_ok,
            "fts_ok": fts_ok,
            "note_count": note_count,
            "file_count": file_count,
            "issues": issues,
            "critical_issues": critical_issues,
            "needs_sync": needs_sync,
        }

    def _nuke_and_rebuild_database(self) -> str:
        """Nuclear recovery option: backup corrupted DB and rebuild from scratch.

        Creates a timestamped backup of the corrupted database, deletes it,
        reinitializes the schema, and rebuilds the index from markdown files.

        CAUTION: This deletes the database! Only call when actual corruption
        is detected (SQLite integrity check fails, malformed database error).
        Do NOT call for simple count mismatches - use rebuild_index() instead.

        Returns:
            Path to the backup file.

        Raises:
            DatabaseCorruptionError: If recovery fails.
        """
        db_url = config.get_db_url()
        db_path = Path(db_url.replace("sqlite:///", ""))
        file_count = len(list(self.notes_dir.glob("*.md")))

        # Safety check: warn if rebuilding from very few files
        if file_count < 5:
            logger.warning(
                f"REBUILD SAFETY WARNING: Only {file_count} markdown files found in "
                f"{self.notes_dir}. If you expected more files, the notes_dir path "
                "may be misconfigured. Database will be rebuilt from these files."
            )

        logger.info(
            f"Starting database rebuild: db_path={db_path}, "
            f"notes_dir={self.notes_dir}, source_files={file_count}"
        )

        backup_path = None
        if db_path.exists():
            # Create timestamped backup (labeled as backup, not "corrupted")
            timestamp = utc_now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{db_path.stem}.backup.{timestamp}.bak"
            backup_path = db_path.parent / backup_name

            try:
                shutil.copy(db_path, backup_path)
                logger.info(f"Backed up database to: {backup_path}")
            except IOError as e:
                logger.warning(f"Could not backup database: {e}")

            # Delete the database and WAL files
            try:
                db_path.unlink()
                # Also remove WAL and SHM files if they exist
                wal_path = db_path.with_suffix(".db-wal")
                shm_path = db_path.with_suffix(".db-shm")
                if wal_path.exists():
                    wal_path.unlink()
                if shm_path.exists():
                    shm_path.unlink()
                logger.info("Deleted old database files")
            except IOError as e:
                raise DatabaseCorruptionError(
                    f"Failed to delete database: {e}",
                    recovered=False,
                    backup_path=str(backup_path) if backup_path else None,
                    code=ErrorCode.DATABASE_RECOVERY_FAILED,
                    original_error=e,
                )

        # Reinitialize database
        try:
            self.engine = init_db()
            self.session_factory = get_session_factory(self.engine)

            # Rebuild index from markdown files
            self.rebuild_index()
            final_count = len(list(self.notes_dir.glob("*.md")))
            logger.info(
                f"Database rebuilt successfully: indexed {final_count} files "
                f"from {self.notes_dir}"
            )
        except Exception as e:
            raise DatabaseCorruptionError(
                f"Failed to rebuild database: {e}",
                recovered=False,
                backup_path=str(backup_path) if backup_path else None,
                code=ErrorCode.DATABASE_RECOVERY_FAILED,
                original_error=e,
            )

        return str(backup_path) if backup_path else ""

    def _get_note_lock(self, note_id: str) -> threading.RLock:
        """Get or create a lock for a specific note.

        Uses WeakValueDictionary so locks are garbage collected when no longer
        held. This prevents memory leaks while providing per-note synchronization.

        Args:
            note_id: The ID of the note to lock.

        Returns:
            A reentrant lock for the specified note.
        """
        with self._note_locks_lock:
            lock = self._note_locks.get(note_id)
            if lock is None:
                lock = threading.RLock()
                self._note_locks[note_id] = lock
            return lock

    def rebuild_index_if_needed(self) -> None:
        """Rebuild the database index from files if needed."""
        # Count notes in database
        with self.session_factory() as session:
            db_count = session.scalar(select(text("COUNT(*)")).select_from(DBNote))

        # Count note files
        file_count = len(list(self.notes_dir.glob("*.md")))

        # Rebuild if counts don't match
        if db_count != file_count:
            self.rebuild_index()

    def rebuild_index(self) -> None:
        """Rebuild the database index from markdown files using incremental sync.

        This method is crash-safe: instead of deleting all records and rebuilding
        from scratch, it performs an incremental sync:

        1. Identifies orphaned DB entries (notes in DB but not on disk) and removes them
        2. Upserts all notes from disk files (updates existing, inserts new)
        3. All operations happen in a single transaction for atomicity

        If a crash occurs during rebuild, the database remains in a consistent state
        (either pre-rebuild or post-rebuild, never half-rebuilt with empty tables).

        Note:
            Files that fail to parse are logged and skipped, but do not abort the
            entire rebuild operation.
        """
        with self.session_factory() as session:
            # Step 1: Get existing note IDs from database
            db_ids = {row[0] for row in session.execute(text("SELECT id FROM notes"))}

            # Step 2: Get note IDs from files on disk
            file_ids = {p.stem for p in self.notes_dir.glob("*.md")}

            # Step 3: Delete orphaned DB entries (notes in DB but files deleted)
            orphaned = db_ids - file_ids
            if orphaned:
                logger.info(f"Removing {len(orphaned)} orphaned database entries")
                # SQLite doesn't support tuple binding directly for IN clauses
                # Use individual deletes in a batch for safety
                for orphan_id in orphaned:
                    session.execute(
                        text(
                            "DELETE FROM links WHERE source_id = :id OR target_id = :id"
                        ),
                        {"id": orphan_id},
                    )
                    session.execute(
                        text("DELETE FROM note_tags WHERE note_id = :id"),
                        {"id": orphan_id},
                    )
                    session.execute(
                        text("DELETE FROM notes WHERE id = :id"), {"id": orphan_id}
                    )

            # Step 4: Upsert from files (process in batches for memory efficiency)
            note_files = list(self.notes_dir.glob("*.md"))
            batch_size = 100
            total_processed = 0
            failed_files: List[str] = []

            for i in range(0, len(note_files), batch_size):
                batch = note_files[i : i + batch_size]

                for file_path in batch:
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                        note = self._parse_note_from_markdown(content)
                        # Upsert within the same session (no individual commits)
                        self._upsert_note_in_session(session, note)
                        total_processed += 1
                    except (IOError, OSError) as e:
                        # File access errors - log and continue
                        logger.error(f"Cannot read file {file_path.name}: {e}")
                        failed_files.append(file_path.name)
                    except (ValueError, yaml.YAMLError) as e:
                        # Malformed frontmatter or missing required fields
                        logger.error(f"Invalid note format in {file_path.name}: {e}")
                        failed_files.append(file_path.name)
                    # Let other exceptions (MemoryError, KeyboardInterrupt, bugs) propagate

            if failed_files:
                logger.warning(
                    f"Failed to process {len(failed_files)} files: "
                    f"{failed_files[:5]}{'...' if len(failed_files) > 5 else ''}"
                )

            # Step 5: Commit all changes atomically
            session.commit()
            logger.info(
                f"Incremental rebuild complete: {total_processed} notes indexed, "
                f"{len(orphaned)} orphans removed, {len(failed_files)} files failed"
            )

        # Rebuild FTS5 index to ensure full-text search is in sync
        fts_count = self.rebuild_fts()
        logger.info(f"Rebuilt FTS5 index with {fts_count} notes")

    def _sync_note_to_db(self, session: Session, note: Note) -> None:
        """Synchronise a Note model into the database within an existing session.

        This is the **single authoritative write path** for note ↔ DB
        synchronisation.  All tag and link rows are cleared and rebuilt
        from the Note model so that the DB always reflects the file.

        The caller controls the session and transaction boundary (commit).

        Args:
            session: Active SQLAlchemy session (caller commits).
            note: The Note domain object to persist.
        """
        db_note = session.scalar(select(DBNote).where(DBNote.id == note.id))

        if db_note:
            db_note.title = note.title
            db_note.content = note.content
            db_note.note_type = note.note_type.value
            db_note.note_purpose = (
                note.note_purpose.value if note.note_purpose else "general"
            )
            db_note.plan_id = note.plan_id
            db_note.obsidian_path = note.obsidian_path
            db_note.updated_at = note.updated_at
            db_note.project = note.project
        else:
            db_note = DBNote(
                id=note.id,
                title=note.title,
                content=note.content,
                note_type=note.note_type.value,
                note_purpose=(
                    note.note_purpose.value if note.note_purpose else "general"
                ),
                plan_id=note.plan_id,
                obsidian_path=note.obsidian_path,
                created_at=note.created_at,
                updated_at=note.updated_at,
                project=note.project,
            )
            session.add(db_note)

        session.flush()

        # --- Tags: clear + rebuild -----------------------------------------
        session.execute(
            text("DELETE FROM note_tags WHERE note_id = :nid"),
            {"nid": note.id},
        )
        for tag in note.tags:
            db_tag = self._get_or_create_tag(session, tag.name)
            db_note.tags.append(db_tag)

        # --- Links: clear + rebuild ----------------------------------------
        session.execute(
            text("DELETE FROM links WHERE source_id = :nid"),
            {"nid": note.id},
        )
        for link in note.links:
            session.add(
                DBLink(
                    source_id=link.source_id,
                    target_id=link.target_id,
                    link_type=link.link_type.value,
                    description=link.description,
                    created_at=link.created_at,
                )
            )

    # -- Legacy aliases that delegate to _sync_note_to_db ------------------

    def _upsert_note_in_session(self, session: Session, note: Note) -> None:
        """Upsert a note within an existing session (no commit).

        Delegates to _sync_note_to_db.
        """
        self._sync_note_to_db(session, note)

    def _parse_note_from_markdown(self, content: str) -> Note:
        """Parse a note from markdown content (delegates to MarkdownParser)."""
        return self._parser.parse_note(content)

    def _index_note(self, note: Note) -> None:
        """Index a note in the database.

        Delegates to _sync_note_to_db within a fresh session + commit.
        """
        with self.session_factory() as session:
            self._sync_note_to_db(session, note)
            session.commit()

    def _get_or_create_tag(self, session, tag_name: str) -> DBTag:
        """Atomically get or create a tag to handle concurrent creation race.

        Uses INSERT OR IGNORE followed by SELECT to safely handle the case
        where two transactions try to create the same tag simultaneously.

        Args:
            session: The SQLAlchemy session.
            tag_name: The name of the tag.

        Returns:
            The DBTag object (either existing or newly created).
        """
        # Use INSERT OR IGNORE to avoid integrity errors from race conditions
        session.execute(
            text("INSERT OR IGNORE INTO tags (name) VALUES (:name)"), {"name": tag_name}
        )
        # Now SELECT - the tag definitely exists (either we created it or it existed)
        db_tag = session.scalar(select(DBTag).where(DBTag.name == tag_name))
        return db_tag

    def _note_to_markdown(self, note: Note) -> str:
        """Convert a note to markdown with frontmatter (delegates to MarkdownParser)."""
        return self._parser.render_to_markdown(note)

    @staticmethod
    def _db_note_to_model(db_note: DBNote) -> Note:
        """Convert a SQLAlchemy DBNote to a domain Note without file I/O.

        Constructs the Note directly from database columns and eager-loaded
        relationships.  This is the batch-friendly alternative to get() which
        reads and parses a file per note.

        Limitations:
            - ``metadata`` will be empty (extra frontmatter keys are not stored
              in the DB).  Use get() when metadata is needed.
        """
        tags = [Tag(name=t.name) for t in (db_note.tags or [])]
        links = [
            Link(
                source_id=lnk.source_id,
                target_id=lnk.target_id,
                link_type=LinkType(lnk.link_type),
                description=lnk.description,
                created_at=(
                    ensure_timezone_aware(lnk.created_at)
                    if lnk.created_at
                    else utc_now()
                ),
            )
            for lnk in (db_note.outgoing_links or [])
        ]

        return Note(
            id=db_note.id,
            title=db_note.title,
            content=db_note.content,
            note_type=NoteType(db_note.note_type),
            note_purpose=(
                NotePurpose(db_note.note_purpose)
                if db_note.note_purpose
                else NotePurpose.GENERAL
            ),
            project=db_note.project or "general",
            plan_id=db_note.plan_id,
            obsidian_path=db_note.obsidian_path,
            tags=tags,
            links=links,
            created_at=(
                ensure_timezone_aware(db_note.created_at)
                if db_note.created_at
                else utc_now()
            ),
            updated_at=(
                ensure_timezone_aware(db_note.updated_at)
                if db_note.updated_at
                else utc_now()
            ),
            metadata={},
        )

    def _get_obsidian_mirror(self) -> Optional[ObsidianMirror]:
        """Get the ObsidianMirror instance, creating/updating if vault path changed."""
        if not self.obsidian_vault_path:
            return None
        # Create or recreate if vault path was changed after init
        if (
            self._obsidian is None
            or self._obsidian.vault_path != self.obsidian_vault_path
        ):
            self._obsidian = ObsidianMirror(
                self.obsidian_vault_path,
                note_resolver=self.get,
            )
        return self._obsidian

    def _mirror_to_obsidian(self, note: Note, markdown: str) -> None:
        """Mirror a note to the Obsidian vault (delegates to ObsidianMirror)."""
        mirror = self._get_obsidian_mirror()
        if mirror:
            mirror.mirror_note(note, markdown)

    def _cascade_obsidian_remirror(self, note_id: str) -> None:
        """Re-mirror notes linking to this note (delegates to ObsidianMirror)."""
        mirror = self._get_obsidian_mirror()
        if not mirror:
            return
        try:
            incoming_notes = self.find_linked_notes(note_id, "incoming")
            self._obsidian.cascade_remirror(
                note_id,
                incoming_notes,
                note_to_markdown=self._note_to_markdown,
            )
        except Exception as e:
            logger.warning(f"Obsidian link cascade failed for {note_id}: {e}")

    def _delete_from_db(self, note_id: str) -> None:
        """Delete a note and its relationships from the database.

        This is a low-level helper that only handles DB cleanup.
        Does NOT delete the file or Obsidian mirror.

        Args:
            note_id: The note ID to delete from database
        """
        with self.session_factory() as session:
            # Delete note and its relationships (parameterized to prevent SQL injection)
            session.execute(
                text(
                    "DELETE FROM links WHERE source_id = :note_id OR target_id = :note_id"
                ),
                {"note_id": note_id},
            )
            session.execute(
                text("DELETE FROM note_tags WHERE note_id = :note_id"),
                {"note_id": note_id},
            )
            session.execute(
                text("DELETE FROM notes WHERE id = :note_id"), {"note_id": note_id}
            )
            session.commit()

    def _delete_from_obsidian(
        self,
        note_id: str,
        title: Optional[str] = None,
        project: Optional[str] = None,
        note_purpose: Optional["NotePurpose"] = None,
        obsidian_path: Optional[str] = None,
    ) -> None:
        """Delete a note's mirror from the Obsidian vault if configured.

        Searches for files with ID suffix pattern: "Title_id_suffix.md"
        Uses recursive glob to find files in project/purpose subdirectories.

        Args:
            note_id: The note's ID (used for ID suffix matching).
            title: The note's title (not currently used but available for future).
            project: The project name for targeted directory search.
            note_purpose: The note's purpose for targeted directory search.
            obsidian_path: Custom Obsidian path override (searched first).
        """
        if not self.obsidian_vault_path:
            return

        # Get ID suffix used in filename
        id_suffix = note_id[-8:] if len(note_id) >= 8 else note_id

        # Determine which directories to search
        search_dirs: List[Path] = []

        # Check obsidian_path first (custom override)
        if obsidian_path:
            custom_dir = self.obsidian_vault_path / obsidian_path
            if custom_dir.exists():
                search_dirs.append(custom_dir)

        if not search_dirs and project:
            safe_project = sanitize_for_terminal(project) or "general"
            project_dir = self.obsidian_vault_path / safe_project

            # If purpose is also specified, search more precisely
            if note_purpose and project_dir.exists():
                safe_purpose = sanitize_for_terminal(note_purpose.value)
                purpose_dir = project_dir / safe_purpose
                if purpose_dir.exists():
                    search_dirs.append(purpose_dir)
                else:
                    # Fall back to project dir if purpose dir doesn't exist
                    search_dirs.append(project_dir)
            elif project_dir.exists():
                search_dirs.append(project_dir)

        # If no project specified or project dir doesn't exist, search all subdirs
        if not search_dirs:
            search_dirs = [d for d in self.obsidian_vault_path.iterdir() if d.is_dir()]

        # Search for file with matching ID suffix pattern (recursive for project/purpose structure)
        for search_dir in search_dirs:
            # Recursive glob to find in purpose subdirectories (new format: *_id_suffix.md)
            for file_path in search_dir.glob(f"**/*_{id_suffix}.md"):
                try:
                    file_path.unlink()
                    logger.debug(f"Deleted Obsidian mirror: {file_path}")
                    return  # Found and deleted
                except OSError as e:
                    logger.warning(f"Failed to delete Obsidian mirror {file_path}: {e}")

            # Fallback: Check for legacy filename formats
            # Old format with parentheses: "Title (id_suffix).md"
            for file_path in search_dir.glob(f"**/*({id_suffix}).md"):
                try:
                    file_path.unlink()
                    logger.debug(
                        f"Deleted legacy Obsidian mirror (parens format): {file_path}"
                    )
                    return
                except OSError as e:
                    logger.warning(
                        f"Failed to delete legacy Obsidian mirror {file_path}: {e}"
                    )

            # Very old format: just ID as filename
            legacy_path = search_dir / f"{note_id}.md"
            if legacy_path.exists():
                try:
                    legacy_path.unlink()
                    logger.debug(f"Deleted legacy Obsidian mirror: {legacy_path}")
                    return
                except OSError as e:
                    logger.warning(
                        f"Failed to delete legacy Obsidian mirror {legacy_path}: {e}"
                    )

    def create(self, note: Note) -> Note:
        """Create a new note."""
        # Ensure the note has an ID
        if not note.id:
            from znote_mcp.models.schema import generate_id

            note.id = generate_id()

        # Convert note to markdown
        markdown = self._note_to_markdown(note)

        # Write to file
        file_path = self.notes_dir / f"{note.id}.md"
        try:
            with self.file_lock:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(markdown)
        except IOError as e:
            raise StorageError(
                f"Failed to write note {note.id}",
                operation="create",
                path=str(file_path),
                original_error=e,
            ) from e

        # Mirror to Obsidian vault if configured
        self._mirror_to_obsidian(note, markdown)

        # Index in database
        self._index_note(note)
        return note

    def get(self, id: str) -> Optional[Note]:
        """Get a note by ID.

        Args:
            id: The ISO 8601 formatted identifier of the note

        Returns:
            Note object if found, None otherwise

        Raises:
            ValueError: If the ID contains invalid characters
        """
        # Validate ID to prevent path traversal attacks
        validate_safe_path_component(id, "Note ID")

        file_path = self.notes_dir / f"{id}.md"
        if not file_path.exists():
            return None
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            return self._parse_note_from_markdown(content)
        except Exception as e:
            raise StorageError(
                f"Failed to read note {id}",
                operation="read",
                path=f"{id}.md",
                original_error=e,
            ) from e

    def get_by_title(self, title: str) -> Optional[Note]:
        """Get a note by title."""
        with self.session_factory() as session:
            db_note = session.scalar(select(DBNote).where(DBNote.title == title))
            if not db_note:
                return None
            return self.get(db_note.id)

    def get_by_ids(self, ids: List[str]) -> List[Note]:
        """Get multiple notes by their IDs in a single DB query.

        Much more efficient than calling get() in a loop: one SQL query
        instead of N file reads + YAML parses.

        Args:
            ids: List of note IDs to retrieve.

        Returns:
            List of Note objects for notes that were found.
            Notes that don't exist are silently skipped.

        Raises:
            ValueError: If any ID contains invalid characters.
        """
        if not ids:
            return []

        # Validate all IDs first
        for note_id in ids:
            validate_safe_path_component(note_id, "Note ID")

        with self.session_factory() as session:
            query = (
                select(DBNote)
                .where(DBNote.id.in_(ids))
                .options(
                    joinedload(DBNote.tags),
                    joinedload(DBNote.outgoing_links),
                    joinedload(DBNote.incoming_links),
                )
            )
            db_notes = session.execute(query).unique().scalars().all()

        # Preserve request order
        id_to_note = {}
        for db_note in db_notes:
            try:
                id_to_note[db_note.id] = self._db_note_to_model(db_note)
            except Exception as e:
                logger.warning(f"Failed to convert note {db_note.id}: {e}")
        return [id_to_note[nid] for nid in ids if nid in id_to_note]

    def get_all(self, limit: Optional[int] = None, offset: int = 0) -> List[Note]:
        """Get all notes with optional pagination.

        Args:
            limit: Maximum number of notes to return. None for all notes.
            offset: Number of notes to skip (for pagination).

        Returns:
            List of Note objects.

        Note:
            For large vaults, use limit/offset to avoid memory issues.
            Use count_notes() to get total count for pagination UI.
        """
        with self.session_factory() as session:
            query = (
                select(DBNote)
                .options(
                    joinedload(DBNote.tags),
                    joinedload(DBNote.outgoing_links),
                    joinedload(DBNote.incoming_links),
                )
                .order_by(DBNote.created_at.desc())
            )

            if offset > 0:
                query = query.offset(offset)
            if limit is not None:
                query = query.limit(limit)

            db_notes = session.execute(query).unique().scalars().all()

        notes = []
        failed_ids: List[str] = []
        for db_note in db_notes:
            try:
                notes.append(self._db_note_to_model(db_note))
            except Exception as e:
                logger.error(f"Error converting note {db_note.id}: {e}")
                failed_ids.append(db_note.id)

        if failed_ids:
            logger.warning(
                f"Failed to convert {len(failed_ids)} of {len(db_notes)} notes: "
                f"{failed_ids[:5]}{'...' if len(failed_ids) > 5 else ''}"
            )
        return notes

    def count_notes(self) -> int:
        """Get total count of notes in the repository.

        Useful for pagination UI when using get_all() with limit/offset.

        Returns:
            Total number of notes.
        """
        with self.session_factory() as session:
            result = session.execute(select(func.count(DBNote.id)))
            return result.scalar() or 0

    def count_notes_by_type(self) -> Dict[str, int]:
        """Get note counts grouped by note_type using SQL GROUP BY.

        Returns:
            Dict mapping note_type string to count.
        """
        with self.session_factory() as session:
            rows = (
                session.query(DBNote.note_type, func.count(DBNote.id))
                .group_by(DBNote.note_type)
                .all()
            )
            return {row[0]: row[1] for row in rows}

    def count_notes_by_project(self) -> Dict[str, int]:
        """Get note counts grouped by project using SQL GROUP BY.

        Returns:
            Dict mapping project string to count.
        """
        with self.session_factory() as session:
            rows = (
                session.query(DBNote.project, func.count(DBNote.id))
                .group_by(DBNote.project)
                .all()
            )
            return {row[0]: row[1] for row in rows}

    def list_notes(
        self,
        sort_by: str = "updated_at",
        sort_order: str = "desc",
        limit: int = 20,
        offset: int = 0,
        project: Optional[str] = None,
    ) -> List[Note]:
        """List notes with SQL-level sorting and pagination.

        Args:
            sort_by: Column to sort by (created_at, updated_at, title).
            sort_order: Sort direction ('asc' or 'desc').
            limit: Maximum number of notes to return.
            offset: Number of notes to skip.
            project: Optional project filter.

        Returns:
            List of Note objects.
        """
        with self.session_factory() as session:
            query = select(DBNote).options(
                joinedload(DBNote.tags),
                joinedload(DBNote.outgoing_links),
                joinedload(DBNote.incoming_links),
            )

            if project:
                query = query.where(DBNote.project == project)

            # Map sort_by to column
            col = getattr(DBNote, sort_by, DBNote.updated_at)
            if sort_order == "asc":
                query = query.order_by(col.asc())
            else:
                query = query.order_by(col.desc())

            query = query.offset(offset).limit(limit)
            db_notes = session.execute(query).unique().scalars().all()

        return [self._db_note_to_model(n) for n in db_notes]

    def get_by_project(self, project: str) -> List[Note]:
        """Get all notes for a specific project using SQL-level filtering.

        Args:
            project: The project name to filter by.

        Returns:
            List of Note objects belonging to the specified project.
        """
        with self.session_factory() as session:
            query = (
                select(DBNote)
                .where(DBNote.project == project)
                .options(
                    joinedload(DBNote.tags),
                    joinedload(DBNote.outgoing_links),
                    joinedload(DBNote.incoming_links),
                )
            )
            db_notes = session.execute(query).unique().scalars().all()

        notes = []
        for db_note in db_notes:
            try:
                notes.append(self._db_note_to_model(db_note))
            except Exception as e:
                logger.error(f"Error converting note {db_note.id}: {e}")
        return notes

    def update(self, note: Note) -> Note:
        """Update a note.

        Uses per-note locking to prevent race conditions with concurrent
        delete operations on the same note.
        """
        # Acquire per-note lock to prevent race with delete
        note_lock = self._get_note_lock(note.id)
        with note_lock:
            # Check if note exists
            existing_note = self.get(note.id)
            if not existing_note:
                raise NoteNotFoundError(note.id)

            # If title, project, purpose, or obsidian_path changed, delete old Obsidian mirror (will be recreated)
            if (
                existing_note.title != note.title
                or existing_note.project != note.project
                or existing_note.note_purpose != note.note_purpose
                or existing_note.obsidian_path != note.obsidian_path
            ):
                self._delete_from_obsidian(
                    note.id,
                    existing_note.title,
                    existing_note.project,
                    existing_note.note_purpose,
                    existing_note.obsidian_path,
                )

            # Update timestamp
            note.updated_at = utc_now()

            # Convert note to markdown
            markdown = self._note_to_markdown(note)

            # Write to file
            file_path = self.notes_dir / f"{note.id}.md"
            try:
                with self.file_lock:
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(markdown)
            except IOError as e:
                raise StorageError(
                    f"Failed to write note {note.id}",
                    operation="update",
                    path=str(file_path),
                    original_error=e,
                ) from e

            # Mirror to Obsidian vault if configured
            self._mirror_to_obsidian(note, markdown)

            # Cascade: re-mirror notes that link TO this note if title changed
            if existing_note.title != note.title and self.obsidian_vault_path:
                try:
                    self._cascade_obsidian_remirror(note.id)
                except Exception as e:
                    logger.warning(f"Obsidian cascade failed (best-effort): {e}")

            try:
                # Re-index in database via consolidated write path
                with self.session_factory() as session:
                    self._sync_note_to_db(session, note)
                    session.commit()
            except Exception as e:
                # Log and re-raise the exception
                logger.error(f"Failed to update note in database: {e}")
                raise

            return note

    def delete(self, id: str) -> None:
        """Delete a note by ID.

        Uses per-note locking to prevent race conditions with concurrent
        update operations on the same note.

        Raises:
            ValueError: If the ID is invalid or the note doesn't exist
        """
        # Validate ID to prevent path traversal attacks
        validate_safe_path_component(id, "Note ID")

        # Acquire per-note lock to prevent race with update
        note_lock = self._get_note_lock(id)
        with note_lock:
            # Check if note exists and get its title for Obsidian mirror deletion
            file_path = self.notes_dir / f"{id}.md"
            if not file_path.exists():
                raise NoteNotFoundError(id)

            # Get the note's title, project, and purpose before deleting (for Obsidian mirror)
            note_title = None
            note_project = None
            note_purpose = None
            try:
                note = self.get(id)
                if note:
                    note_title = note.title
                    note_project = note.project
                    note_purpose = note.note_purpose
            except (IOError, OSError, ValueError, yaml.YAMLError) as e:
                # If we can't read the note, log and continue with deletion
                # The note file exists but may be corrupted/unreadable
                logger.warning(f"Cannot read note {id} for Obsidian cleanup: {e}")
            # Let system errors and bugs propagate

            # Delete from file system
            try:
                with self.file_lock:
                    os.remove(file_path)
            except IOError as e:
                raise StorageError(
                    f"Failed to delete note {id}",
                    operation="delete",
                    path=f"{id}.md",
                    original_error=e,
                ) from e

            # Delete from Obsidian vault if configured
            self._delete_from_obsidian(id, note_title, note_project, note_purpose)

            # Delete from database (using parameterized queries to prevent SQL injection)
            with self.session_factory() as session:
                # Delete note and its relationships
                session.execute(
                    text(
                        "DELETE FROM links WHERE source_id = :note_id OR target_id = :note_id"
                    ),
                    {"note_id": id},
                )
                session.execute(
                    text("DELETE FROM note_tags WHERE note_id = :note_id"),
                    {"note_id": id},
                )
                session.execute(
                    text("DELETE FROM notes WHERE id = :note_id"), {"note_id": id}
                )
                session.commit()

    # =========================================================================
    # Versioned CRUD Operations (with git tracking)
    # =========================================================================

    def get_versioned(self, id: str) -> Optional[VersionedNote]:
        """Get a note with its version information.

        Args:
            id: The note ID

        Returns:
            VersionedNote if found, None otherwise
        """
        note = self.get(id)
        if not note:
            return None

        if self._git:
            file_path = self.notes_dir / f"{id}.md"
            version = self._git.get_file_version(file_path)
            if version:
                version_info = VersionInfo.from_git_commit(
                    version.commit_hash, version.timestamp
                )
            else:
                version_info = VersionInfo(
                    commit_hash="0000000", timestamp=note.created_at
                )
        else:
            version_info = VersionInfo(commit_hash="0000000", timestamp=note.updated_at)

        return VersionedNote(note=note, version=version_info)

    def create_versioned(self, note: Note) -> VersionedNote:
        """Create a note with version tracking.

        Args:
            note: The note to create

        Returns:
            VersionedNote with the created note and its version info
        """
        created_note = self.create(note)

        if self._git:
            file_path = self.notes_dir / f"{created_note.id}.md"
            version = self._git.commit_file(
                file_path,
                f"Create note: {_sanitize_commit_message(created_note.title)}",
            )
            version_info = VersionInfo.from_git_commit(
                version.commit_hash, version.timestamp
            )
        else:
            version_info = VersionInfo(
                commit_hash="0000000", timestamp=created_note.created_at
            )

        return VersionedNote(note=created_note, version=version_info)

    def update_versioned(
        self, note: Note, expected_version: Optional[str] = None
    ) -> Union[VersionedNote, ConflictResult]:
        """Update a note with version checking.

        Args:
            note: The note to update
            expected_version: If provided, check this matches current version before updating

        Returns:
            VersionedNote on success, ConflictResult if version conflict detected
        """
        file_path = self.notes_dir / f"{note.id}.md"

        # Check for conflict if expected version provided
        if expected_version and self._git:
            matches, actual = self._git.check_version_match(file_path, expected_version)
            if not matches and actual:
                return ConflictResult(
                    status="conflict",
                    note_id=note.id,
                    expected_version=expected_version,
                    actual_version=actual,
                    message=f"Note was modified by another process. Expected version {expected_version}, found {actual}",
                )

        # Perform the update
        updated_note = self.update(note)

        # Commit to git
        if self._git:
            try:
                version = self._git.commit_file(
                    file_path,
                    f"Update note: {_sanitize_commit_message(updated_note.title)}",
                    expected_version=expected_version,
                )
                version_info = VersionInfo.from_git_commit(
                    version.commit_hash, version.timestamp
                )
            except GitConflictError as e:
                return ConflictResult(
                    status="conflict",
                    note_id=note.id,
                    expected_version=e.expected_version,
                    actual_version=e.actual_version,
                    message=str(e),
                )
        else:
            version_info = VersionInfo(
                commit_hash="0000000", timestamp=updated_note.updated_at
            )

        return VersionedNote(note=updated_note, version=version_info)

    def delete_versioned(
        self, id: str, expected_version: Optional[str] = None
    ) -> Union[VersionInfo, ConflictResult]:
        """Delete a note with version checking.

        Uses per-note locking to prevent race conditions between version check
        and deletion. When git is enabled, version checking and file deletion
        are handled atomically by git_wrapper.delete_file().

        Args:
            id: The note ID to delete
            expected_version: If provided, check this matches current version before deleting

        Returns:
            VersionInfo on success, ConflictResult if version conflict detected

        Raises:
            ValueError: If note doesn't exist
        """
        # Validate ID to prevent path traversal
        validate_safe_path_component(id, "Note ID")
        file_path = self.notes_dir / f"{id}.md"

        # Acquire per-note lock for the entire operation
        # This prevents race condition between version check and deletion
        note_lock = self._get_note_lock(id)
        with note_lock:
            # Get note info before any deletion (for commit message and cleanup)
            note = self.get(id)
            if not note:
                raise NoteNotFoundError(id)
            title = note.title
            project = note.project
            purpose = note.note_purpose

            if self._git:
                # Git handles version check + file deletion atomically via git rm
                try:
                    version = self._git.delete_file(
                        file_path,
                        f"Delete note: {title}",
                        expected_version=expected_version,
                    )
                    # Git succeeded (file deleted by git rm) - now cleanup DB and Obsidian
                    self._delete_from_obsidian(id, title, project, purpose)
                    self._delete_from_db(id)

                    return VersionInfo.from_git_commit(
                        version.commit_hash, version.timestamp
                    )
                except GitConflictError as e:
                    # Version conflict - note was NOT deleted (git rm never ran)
                    return ConflictResult(
                        status="conflict",
                        note_id=id,
                        expected_version=e.expected_version,
                        actual_version=e.actual_version,
                        message="Version conflict during delete. Note was not deleted.",
                    )
            else:
                # No git - use regular delete (handles file + DB + Obsidian)
                self.delete(id)
                return VersionInfo(commit_hash="0000000", timestamp=utc_now())

    @staticmethod
    def _apply_search_filters(query: Any, kwargs: Dict[str, Any]) -> Any:
        """Apply common search filter criteria to a SQLAlchemy query.

        Used by both search() and count_search_results() to avoid
        duplicating filter logic.

        Args:
            query: SQLAlchemy select query to add WHERE clauses to.
            kwargs: Search criteria dict.

        Returns:
            The query with filters applied.
        """
        if "content" in kwargs:
            search_term = escape_like_pattern(kwargs["content"])
            query = query.where(
                or_(
                    DBNote.content.like(f"%{search_term}%", escape="\\"),
                    DBNote.title.like(f"%{search_term}%", escape="\\"),
                )
            )
        if "title" in kwargs:
            search_title = escape_like_pattern(kwargs["title"])
            query = query.where(
                func.lower(DBNote.title).like(f"%{search_title.lower()}%", escape="\\")
            )
        if "note_type" in kwargs:
            note_type = (
                kwargs["note_type"].value
                if isinstance(kwargs["note_type"], NoteType)
                else kwargs["note_type"]
            )
            query = query.where(DBNote.note_type == note_type)
        if "tag" in kwargs:
            tag_name = kwargs["tag"]
            query = query.join(DBNote.tags).where(DBTag.name == tag_name)
        if "tags" in kwargs:
            tag_names = kwargs["tags"]
            if isinstance(tag_names, list):
                query = query.join(DBNote.tags).where(DBTag.name.in_(tag_names))
        if "linked_to" in kwargs:
            target_id = kwargs["linked_to"]
            query = query.join(DBNote.outgoing_links).where(
                DBLink.target_id == target_id
            )
        if "linked_from" in kwargs:
            source_id = kwargs["linked_from"]
            query = query.join(DBNote.incoming_links).where(
                DBLink.source_id == source_id
            )
        if "created_after" in kwargs:
            query = query.where(DBNote.created_at >= kwargs["created_after"])
        if "created_before" in kwargs:
            query = query.where(DBNote.created_at <= kwargs["created_before"])
        if "updated_after" in kwargs:
            query = query.where(DBNote.updated_at >= kwargs["updated_after"])
        if "updated_before" in kwargs:
            query = query.where(DBNote.updated_at <= kwargs["updated_before"])
        return query

    def search(
        self, limit: Optional[int] = None, offset: int = 0, **kwargs: Any
    ) -> List[Note]:
        """Search for notes based on criteria with optional pagination.

        Args:
            limit: Maximum number of results to return. None for all matches.
            offset: Number of results to skip (for pagination).
            **kwargs: Search criteria (content, title, note_type, tag, tags,
                     linked_to, linked_from, created_after, created_before,
                     updated_after, updated_before).

        Returns:
            List of matching Note objects.
        """
        with self.session_factory() as session:
            query = select(DBNote).options(
                joinedload(DBNote.tags),
                joinedload(DBNote.outgoing_links),
                joinedload(DBNote.incoming_links),
            )
            query = self._apply_search_filters(query, kwargs)

            # Order by creation date (newest first) for consistent pagination
            query = query.order_by(DBNote.created_at.desc())

            # Apply pagination at SQL level
            if offset > 0:
                query = query.offset(offset)
            if limit is not None:
                query = query.limit(limit)

            db_notes = session.execute(query).unique().scalars().all()

        return [self._db_note_to_model(db) for db in db_notes]

    def count_search_results(self, **kwargs: Any) -> int:
        """Count notes matching search criteria without loading them.

        Useful for pagination UI when using search() with limit/offset.

        Args:
            **kwargs: Same search criteria as search().

        Returns:
            Total count of matching notes.
        """
        with self.session_factory() as session:
            query = select(func.count(DBNote.id.distinct()))
            query = self._apply_search_filters(query, kwargs)

            result = session.execute(query)
            return result.scalar() or 0

    def fts_search(
        self,
        query: str,
        limit: int = 50,
        highlight: bool = False,
        literal: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """Full-text search using FTS5 (delegates to FtsIndex subsystem)."""
        return self._fts.search(
            query, limit=limit, highlight=highlight, literal=literal
        )

    def reset_fts_availability(self) -> bool:
        """Reset FTS5 availability after manual repair (delegates to FtsIndex)."""
        return self._fts.reset_availability()

    def rebuild_fts(self) -> int:
        """Rebuild the FTS5 index from the notes table (delegates to FtsIndex)."""
        return self._fts.rebuild()

    def find_by_tag(self, tag: Union[str, Tag]) -> List[Note]:
        """Find notes by tag."""
        tag_name = tag.name if isinstance(tag, Tag) else tag
        return self.search(tag=tag_name)

    def find_linked_notes(
        self, note_id: str, direction: str = "outgoing"
    ) -> List[Note]:
        """Find notes linked to/from this note."""
        with self.session_factory() as session:
            if direction == "outgoing":
                # Find notes that this note links to
                query = (
                    select(DBNote)
                    .join(DBLink, DBNote.id == DBLink.target_id)
                    .where(DBLink.source_id == note_id)
                    .options(
                        joinedload(DBNote.tags),
                        joinedload(DBNote.outgoing_links),
                        joinedload(DBNote.incoming_links),
                    )
                )
            elif direction == "incoming":
                # Find notes that link to this note
                query = (
                    select(DBNote)
                    .join(DBLink, DBNote.id == DBLink.source_id)
                    .where(DBLink.target_id == note_id)
                    .options(
                        joinedload(DBNote.tags),
                        joinedload(DBNote.outgoing_links),
                        joinedload(DBNote.incoming_links),
                    )
                )
            elif direction == "both":
                # Find both directions
                query = (
                    select(DBNote)
                    .join(
                        DBLink,
                        or_(
                            and_(
                                DBNote.id == DBLink.target_id,
                                DBLink.source_id == note_id,
                            ),
                            and_(
                                DBNote.id == DBLink.source_id,
                                DBLink.target_id == note_id,
                            ),
                        ),
                    )
                    .options(
                        joinedload(DBNote.tags),
                        joinedload(DBNote.outgoing_links),
                        joinedload(DBNote.incoming_links),
                    )
                )
            else:
                raise ValueError(
                    f"Invalid direction: {direction}. Use 'outgoing', 'incoming', or 'both'"
                )

            result = session.execute(query)
            # Apply unique() to handle the duplicate rows from eager loading
            db_notes = result.unique().scalars().all()

        return [self._db_note_to_model(db) for db in db_notes]

    def find_similarity_candidates(self, note_id: str) -> List[Note]:
        """Find candidate notes for similarity comparison using SQL filtering.

        Returns notes that share at least one tag with the given note OR have
        a direct link to/from it. This reduces O(N) comparisons to O(C) where
        C is the number of candidates.

        Args:
            note_id: The ID of the reference note.

        Returns:
            List of candidate Note objects (excludes the reference note itself).
        """
        with self.session_factory() as session:
            # Get the reference note's tag IDs
            note_tag_ids = (
                session.execute(
                    select(note_tags.c.tag_id).where(note_tags.c.note_id == note_id)
                )
                .scalars()
                .all()
            )

            # Find notes with shared tags
            shared_tag_note_ids = set()
            if note_tag_ids:
                result = session.execute(
                    select(note_tags.c.note_id.distinct())
                    .where(note_tags.c.tag_id.in_(note_tag_ids))
                    .where(note_tags.c.note_id != note_id)
                )
                shared_tag_note_ids = {row[0] for row in result.all()}

            # Find notes with direct links to/from this note
            linked_note_ids = set()
            # Outgoing links (notes this note links to)
            outgoing = (
                session.execute(
                    select(DBLink.target_id).where(DBLink.source_id == note_id)
                )
                .scalars()
                .all()
            )
            linked_note_ids.update(outgoing)

            # Incoming links (notes that link to this note)
            incoming = (
                session.execute(
                    select(DBLink.source_id).where(DBLink.target_id == note_id)
                )
                .scalars()
                .all()
            )
            linked_note_ids.update(incoming)

            # Notes that share common link targets (linked to same notes)
            if outgoing:
                common_target_notes = (
                    session.execute(
                        select(DBLink.source_id.distinct())
                        .where(DBLink.target_id.in_(outgoing))
                        .where(DBLink.source_id != note_id)
                    )
                    .scalars()
                    .all()
                )
                linked_note_ids.update(common_target_notes)

            # Combine all candidate IDs and batch-load from DB
            candidate_ids = shared_tag_note_ids | linked_note_ids
            if not candidate_ids:
                return []

            db_notes = (
                session.execute(
                    select(DBNote)
                    .where(DBNote.id.in_(candidate_ids))
                    .options(
                        joinedload(DBNote.tags),
                        joinedload(DBNote.outgoing_links),
                        joinedload(DBNote.incoming_links),
                    )
                )
                .unique()
                .scalars()
                .all()
            )

        candidates = []
        for db_note in db_notes:
            try:
                candidates.append(self._db_note_to_model(db_note))
            except Exception as e:
                logger.warning(f"Failed to convert candidate note {db_note.id}: {e}")

        return candidates

    def get_all_tags(self) -> List[Tag]:
        """Get all tags in the system."""
        with self.session_factory() as session:
            result = session.execute(select(DBTag))
            db_tags = result.scalars().all()
        return [Tag(name=tag.name) for tag in db_tags]

    def get_tags_with_counts(self) -> Dict[str, int]:
        """Get all tags with their usage counts.

        Returns:
            Dictionary mapping tag names to their note counts.
        """
        with self.session_factory() as session:
            result = session.execute(
                select(DBTag.name, func.count(note_tags.c.note_id))
                .select_from(DBTag)
                .outerjoin(note_tags, DBTag.id == note_tags.c.tag_id)
                .group_by(DBTag.name)
            ).all()

            return {name: count for name, count in result}

    def delete_unused_tags(self) -> int:
        """Delete tags that are not associated with any notes.

        Cleans up orphaned tags that were left behind when notes were deleted
        or had their tags removed.

        Returns:
            Number of tags deleted.
        """
        with self.session_factory() as session:
            unused_tags = session.scalars(
                select(DBTag)
                .outerjoin(note_tags, DBTag.id == note_tags.c.tag_id)
                .where(note_tags.c.note_id.is_(None))
            ).all()

            count = len(unused_tags)
            for tag in unused_tags:
                session.delete(tag)

            session.commit()
            return count

    def find_orphaned_note_ids(self) -> List[str]:
        """Find note IDs that have no incoming or outgoing links.

        Returns:
            List of note IDs for orphaned notes.
        """
        with self.session_factory() as session:
            # Subquery for notes with links
            notes_with_links = (
                select(DBNote.id)
                .outerjoin(
                    DBLink,
                    or_(DBNote.id == DBLink.source_id, DBNote.id == DBLink.target_id),
                )
                .where(or_(DBLink.source_id != None, DBLink.target_id != None))
                .subquery()
            )

            # Query for orphaned note IDs only (lightweight)
            query = select(DBNote.id).where(DBNote.id.not_in(select(notes_with_links)))

            return list(session.scalars(query).all())

    def find_central_note_ids_with_counts(
        self, limit: int = 10
    ) -> List[Tuple[str, int]]:
        """Find note IDs with the most connections (incoming + outgoing links).

        Args:
            limit: Maximum number of notes to return.

        Returns:
            List of (note_id, connection_count) tuples, ordered by count descending.
        """
        with self.session_factory() as session:
            query = text(
                """
            WITH outgoing AS (
                SELECT source_id as note_id, COUNT(*) as outgoing_count
                FROM links
                GROUP BY source_id
            ),
            incoming AS (
                SELECT target_id as note_id, COUNT(*) as incoming_count
                FROM links
                GROUP BY target_id
            )
            SELECT n.id,
                (COALESCE(o.outgoing_count, 0) + COALESCE(i.incoming_count, 0)) as total
            FROM notes n
            LEFT JOIN outgoing o ON n.id = o.note_id
            LEFT JOIN incoming i ON n.id = i.note_id
            WHERE (COALESCE(o.outgoing_count, 0) + COALESCE(i.incoming_count, 0)) > 0
            ORDER BY total DESC
            LIMIT :limit
            """
            )

            results = session.execute(query, {"limit": limit}).all()
            return [(row[0], row[1]) for row in results]

    def get_note_history(self, note_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get version history for a note.

        Requires git versioning to be enabled.

        Args:
            note_id: The note ID
            limit: Maximum number of versions to return

        Returns:
            List of version dictionaries with 'commit_hash', 'short_hash',
            and 'timestamp' fields, most recent first. Empty list if git
            is not enabled or note has no history.
        """
        validate_safe_path_component(note_id, "Note ID")

        if not self._git:
            return []

        file_path = self.notes_dir / f"{note_id}.md"
        if not file_path.exists():
            return []

        versions = self._git.get_history(file_path, limit)
        return [
            {
                "commit_hash": v.commit_hash,
                "short_hash": v.short_hash,
                "timestamp": v.timestamp.isoformat(),
            }
            for v in versions
        ]

    def sync_to_obsidian(self) -> int:
        """Sync all notes to the Obsidian vault (delegates to ObsidianMirror).

        Returns:
            Number of notes synced.

        Raises:
            ValueError: If Obsidian vault is not configured.
        """
        mirror = self._get_obsidian_mirror()
        if not mirror:
            raise ValueError(
                "Obsidian vault not configured. "
                "Set ZETTELKASTEN_OBSIDIAN_VAULT in your .env file."
            )

        # Load notes from disk (NoteRepository responsibility)
        note_files = list(self.notes_dir.glob("*.md"))
        notes: List[Note] = []
        failed_files: List[str] = []

        for file_path in note_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                notes.append(self._parse_note_from_markdown(content))
            except (IOError, OSError, PermissionError) as e:
                logger.warning(f"Cannot access {file_path.name} for sync: {e}")
                failed_files.append(file_path.name)
            except (ValueError, yaml.YAMLError) as e:
                logger.warning(f"Invalid note format in {file_path.name}: {e}")
                failed_files.append(file_path.name)

        # Build in-memory map so link rewriting avoids per-link disk reads
        note_map = {n.id: n for n in notes}
        synced_count = mirror.sync_all(notes, self._note_to_markdown, note_map=note_map)

        if failed_files:
            logger.warning(
                f"Obsidian sync completed with {len(failed_files)} failures: "
                f"{failed_files[:5]}{'...' if len(failed_files) > 5 else ''}"
            )
        logger.info(
            f"Synced {synced_count} of {len(note_files)} notes to Obsidian vault"
        )
        return synced_count

    # ========== Bulk Operations ==========

    def bulk_create_notes(self, notes: List[Note]) -> List[Note]:
        """Create multiple notes in a single atomic batch operation.

        Uses a staged file write pattern for true atomicity:
        1. Write files to a .staging subdirectory first
        2. Commit all database changes in a single transaction
        3. Atomically move files from staging to final location
        4. On any failure, only staging files need cleanup (DB auto-rollbacks)

        This ensures that either ALL notes are created successfully (files exist
        AND database records exist), or NONE are (clean rollback). The staging
        directory pattern prevents partial states where files exist but DB
        records don't, or vice versa.

        Args:
            notes: List of Note objects to create.

        Returns:
            List of created Note objects with assigned IDs.

        Raises:
            BulkOperationError: If the operation fails or input is empty.
            IOError: If any file operation fails.
            ValidationError: If validation fails for any note.
        """
        if not notes:
            raise BulkOperationError(
                "No notes provided for creation",
                operation="bulk_create",
                code=ErrorCode.BULK_OPERATION_EMPTY_INPUT,
            )

        from znote_mcp.models.schema import generate_id

        created_notes = []
        staged_files: List[Tuple[Path, Path]] = []  # (staging_path, final_path) pairs
        staging_dir = self.notes_dir / ".staging"

        # Use a single lock for the entire bulk operation to prevent race conditions
        with self.file_lock:
            try:
                # === Phase 1: Write files to staging directory ===
                staging_dir.mkdir(exist_ok=True)

                for note in notes:
                    # Ensure the note has an ID
                    if not note.id:
                        note.id = generate_id()

                    # Convert note to markdown
                    markdown = self._note_to_markdown(note)

                    # Write to staging directory first
                    staging_path = staging_dir / f"{note.id}.md"
                    final_path = self.notes_dir / f"{note.id}.md"

                    with open(staging_path, "w", encoding="utf-8") as f:
                        f.write(markdown)

                    staged_files.append((staging_path, final_path))
                    created_notes.append(note)

                # === Phase 2: Commit all database changes atomically ===
                with self.session_factory() as session:
                    for note in created_notes:
                        # Create database record with all fields
                        db_note = DBNote(
                            id=note.id,
                            title=note.title,
                            content=note.content,
                            note_type=note.note_type.value,
                            note_purpose=(
                                note.note_purpose.value if note.note_purpose else None
                            ),
                            project=note.project,
                            plan_id=note.plan_id,
                            obsidian_path=note.obsidian_path,
                            created_at=note.created_at,
                            updated_at=note.updated_at,
                        )
                        session.add(db_note)
                        session.flush()

                        # Add tags (using atomic get-or-create to handle race conditions)
                        for tag in note.tags:
                            db_tag = self._get_or_create_tag(session, tag.name)
                            db_note.tags.append(db_tag)

                        # Add links
                        for link in note.links:
                            db_link = DBLink(
                                source_id=link.source_id,
                                target_id=link.target_id,
                                link_type=link.link_type.value,
                                description=link.description,
                                created_at=link.created_at,
                            )
                            session.add(db_link)

                    # This is the commit point - if this fails, DB auto-rollbacks
                    session.commit()

                # === Phase 3: Atomically move staged files to final location ===
                # DB commit succeeded, now move files (this should rarely fail)
                for staging_path, final_path in staged_files:
                    try:
                        # POSIX atomic rename (same filesystem)
                        staging_path.rename(final_path)
                    except OSError:
                        # Cross-filesystem fallback
                        import shutil

                        shutil.move(str(staging_path), str(final_path))

                # === Phase 4: Mirror to Obsidian (best-effort, non-critical) ===
                for note in created_notes:
                    try:
                        markdown = self._note_to_markdown(note)
                        self._mirror_to_obsidian(note, markdown)
                    except Exception as obsidian_error:
                        # Log but don't fail - Obsidian mirroring is optional
                        logger.warning(
                            f"Failed to mirror note {note.id} to Obsidian: {obsidian_error}"
                        )

                # Clean up staging directory if empty
                try:
                    staging_dir.rmdir()
                except OSError:
                    pass  # Not empty or other issue - ignore

                logger.info(f"Bulk created {len(created_notes)} notes atomically")
                return created_notes

            except Exception as e:
                # Rollback: only need to clean staging files
                # (DB auto-rollbacks if commit didn't succeed)
                for staging_path, _ in staged_files:
                    try:
                        if staging_path.exists():
                            staging_path.unlink()
                    except OSError as cleanup_error:
                        logger.warning(
                            f"Failed to cleanup staging file {staging_path}: {cleanup_error}"
                        )

                # Try to remove staging directory
                try:
                    if staging_dir.exists():
                        staging_dir.rmdir()
                except OSError:
                    pass  # Not empty or other issue - ignore

                logger.error(f"Bulk create failed, rolled back: {e}")
                raise BulkOperationError(
                    f"Bulk create failed: {e}",
                    operation="bulk_create",
                    total_count=len(notes),
                    success_count=0,
                    original_error=e,
                )

    def bulk_delete_notes(self, note_ids: List[str]) -> int:
        """Delete multiple notes in a single batch operation.

        Deletes from database first (atomic), then files. This ensures
        consistency: if database deletion fails, no files are touched.

        Args:
            note_ids: List of note IDs to delete.

        Returns:
            Number of notes successfully deleted.

        Raises:
            BulkOperationError: If the operation fails or input is empty.
            ValueError: If any note ID contains unsafe path characters.
        """
        if not note_ids:
            raise BulkOperationError(
                "No note IDs provided for deletion",
                operation="bulk_delete",
                code=ErrorCode.BULK_OPERATION_EMPTY_INPUT,
            )

        # Validate all IDs first
        for note_id in note_ids:
            validate_safe_path_component(note_id, "Note ID")

        # Collect note info for Obsidian deletion and file paths
        notes_info = []
        file_paths = []
        for note_id in note_ids:
            file_path = self.notes_dir / f"{note_id}.md"
            if file_path.exists():
                file_paths.append(file_path)
            try:
                note = self.get(note_id)
                if note:
                    notes_info.append(
                        {
                            "id": note_id,
                            "title": note.title,
                            "project": note.project,
                            "purpose": note.note_purpose,
                        }
                    )
            except Exception:
                notes_info.append(
                    {"id": note_id, "title": None, "project": None, "purpose": None}
                )

        # Delete from database FIRST (atomic transaction)
        try:
            with self.session_factory() as session:
                for note_id in note_ids:
                    session.execute(
                        text(
                            "DELETE FROM links WHERE source_id = :note_id OR target_id = :note_id"
                        ),
                        {"note_id": note_id},
                    )
                    session.execute(
                        text("DELETE FROM note_tags WHERE note_id = :note_id"),
                        {"note_id": note_id},
                    )
                    session.execute(
                        text("DELETE FROM notes WHERE id = :note_id"),
                        {"note_id": note_id},
                    )
                session.commit()
        except Exception as e:
            logger.error(f"Bulk delete database operation failed: {e}")
            raise BulkOperationError(
                f"Database deletion failed: {e}",
                operation="bulk_delete",
                total_count=len(note_ids),
                success_count=0,
                failed_ids=note_ids,
                original_error=e,
            )

        # Now delete files (database is already consistent)
        # Wrap entire file deletion loop in single lock to prevent interleaving
        deleted_count = 0
        failed_files = []
        with self.file_lock:
            for file_path in file_paths:
                try:
                    os.remove(file_path)
                    deleted_count += 1
                except IOError as e:
                    logger.warning(f"Failed to delete note file {file_path.name}: {e}")
                    failed_files.append(file_path.stem)

        # Delete from Obsidian
        for info in notes_info:
            self._delete_from_obsidian(
                info["id"], info["title"], info["project"], info["purpose"]
            )

        if failed_files:
            logger.warning(f"Some files could not be deleted: {failed_files}")

        logger.info(f"Bulk deleted {deleted_count} notes")
        return deleted_count

    def _bulk_operation(
        self,
        note_ids: List[str],
        operation_name: str,
        per_note_fn: Any,
        **kwargs: Any,
    ) -> int:
        """Template method for bulk note operations.

        Handles: ID validation, session management, per-note DB query,
        file rewriting, error collection, BulkOperationError assembly.

        Args:
            note_ids: List of note IDs to operate on.
            operation_name: Name for error reporting (e.g. "bulk_add_tags").
            per_note_fn: Callable(session, db_note, **kwargs) -> bool.
                Returns True if the note was modified and needs file rewrite.
            **kwargs: Passed through to per_note_fn.

        Returns:
            Number of notes successfully updated.

        Raises:
            BulkOperationError: On empty input, total failure, or partial failure.
        """
        if not note_ids:
            raise BulkOperationError(
                f"No note IDs provided for {operation_name}",
                operation=operation_name,
                code=ErrorCode.BULK_OPERATION_EMPTY_INPUT,
            )

        for note_id in note_ids:
            validate_safe_path_component(note_id, "Note ID")

        updated_count = 0
        failed_ids: List[str] = []
        notes_to_update: List[Tuple[str, Note]] = []

        try:
            with self.session_factory() as session:
                for note_id in note_ids:
                    db_note = session.scalar(select(DBNote).where(DBNote.id == note_id))
                    if not db_note:
                        failed_ids.append(note_id)
                        logger.warning(f"Note {note_id} not found for {operation_name}")
                        continue

                    modified = per_note_fn(session, db_note, **kwargs)
                    updated_count += 1
                    if modified:
                        session.flush()
                        note = self._db_note_to_model(db_note)
                        notes_to_update.append((note_id, note))

                session.commit()

            with self.file_lock:
                for note_id, note in notes_to_update:
                    markdown = self._note_to_markdown(note)
                    file_path = self.notes_dir / f"{note_id}.md"
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(markdown)
                    self._mirror_to_obsidian(note, markdown)

        except BulkOperationError:
            raise
        except Exception as e:
            logger.error(f"{operation_name} failed: {e}")
            raise BulkOperationError(
                f"Operation failed: {e}",
                operation=operation_name,
                total_count=len(note_ids),
                success_count=0,
                failed_ids=note_ids,
                code=ErrorCode.BULK_OPERATION_FAILED,
                original_error=e,
            )

        logger.info(f"{operation_name}: updated {updated_count} notes")

        if failed_ids:
            code = (
                ErrorCode.BULK_OPERATION_FAILED
                if updated_count == 0
                else ErrorCode.BULK_OPERATION_PARTIAL
            )
            raise BulkOperationError(
                f"{'Failed' if updated_count == 0 else 'Partial success'}: "
                f"{updated_count} of {len(note_ids)} notes",
                operation=operation_name,
                total_count=len(note_ids),
                success_count=updated_count,
                failed_ids=failed_ids,
                code=code,
            )

        return updated_count

    def bulk_add_tags(self, note_ids: List[str], tags: List[str]) -> int:
        """Add tags to multiple notes in a single atomic transaction."""
        if not tags:
            raise BulkOperationError(
                "No tags provided to add",
                operation="bulk_add_tags",
                code=ErrorCode.BULK_OPERATION_EMPTY_INPUT,
            )

        def _add_tags(session: Session, db_note: DBNote, **kw: Any) -> bool:
            existing = {t.name for t in db_note.tags}
            added = False
            for tag_name in tags:
                if tag_name not in existing:
                    db_note.tags.append(self._get_or_create_tag(session, tag_name))
                    existing.add(tag_name)
                    added = True
            if added:
                db_note.updated_at = utc_now()
            return added

        return self._bulk_operation(note_ids, "bulk_add_tags", _add_tags)

    def bulk_remove_tags(self, note_ids: List[str], tags: List[str]) -> int:
        """Remove tags from multiple notes in a single atomic transaction."""
        if not tags:
            raise BulkOperationError(
                "No tags provided to remove",
                operation="bulk_remove_tags",
                code=ErrorCode.BULK_OPERATION_EMPTY_INPUT,
            )

        tags_to_remove = set(tags)

        def _remove_tags(session: Session, db_note: DBNote, **kw: Any) -> bool:
            original_count = len(db_note.tags)
            db_note.tags = [t for t in db_note.tags if t.name not in tags_to_remove]
            removed = len(db_note.tags) < original_count
            if removed:
                db_note.updated_at = utc_now()
            return removed

        return self._bulk_operation(note_ids, "bulk_remove_tags", _remove_tags)

    def bulk_update_project(self, note_ids: List[str], project: str) -> int:
        """Move multiple notes to a different project in a single atomic transaction.

        Uses pre-commit file writes with rollback for atomicity — if any file
        write fails, all are rolled back and the DB transaction is aborted.
        """
        if not note_ids:
            raise BulkOperationError(
                "No note IDs provided for project update",
                operation="bulk_update_project",
                code=ErrorCode.BULK_OPERATION_EMPTY_INPUT,
            )
        if not project or not project.strip():
            raise BulkOperationError(
                "Project name is required",
                operation="bulk_update_project",
                code=ErrorCode.BULK_OPERATION_EMPTY_INPUT,
            )

        for note_id in note_ids:
            validate_safe_path_component(note_id, "Note ID")
        validate_project_path(project)

        updated_count = 0
        failed_ids: List[str] = []
        notes_to_update: List[Dict[str, Any]] = []

        try:
            with self.session_factory() as session:
                for note_id in note_ids:
                    db_note = session.scalar(select(DBNote).where(DBNote.id == note_id))
                    if not db_note:
                        failed_ids.append(note_id)
                        logger.warning(f"Note {note_id} not found for project update")
                        continue

                    old_project = db_note.project
                    if old_project != project:
                        old_purpose = (
                            NotePurpose(db_note.note_purpose)
                            if db_note.note_purpose
                            else NotePurpose.GENERAL
                        )
                        old_title = db_note.title
                        session.flush()
                        note = self._db_note_to_model(db_note)
                        notes_to_update.append(
                            {
                                "note": note,
                                "old_project": old_project,
                                "old_title": old_title,
                                "old_purpose": old_purpose,
                            }
                        )
                        db_note.project = project
                        db_note.updated_at = utc_now()

                    updated_count += 1

                # Pre-commit file writes with rollback
                file_write_errors: List[Tuple[str, str]] = []
                written_files: List[Dict[str, Any]] = []

                with self.file_lock:
                    for info in notes_to_update:
                        note = info["note"]
                        try:
                            note.project = project
                            note.updated_at = utc_now()
                            markdown = self._note_to_markdown(note)
                            file_path = self.notes_dir / f"{note.id}.md"
                            original = (
                                file_path.read_text(encoding="utf-8")
                                if file_path.exists()
                                else None
                            )
                            with open(file_path, "w", encoding="utf-8") as f:
                                f.write(markdown)
                            written_files.append(
                                {**info, "path": file_path, "original": original}
                            )
                        except Exception as e:
                            file_write_errors.append((note.id, str(e)))
                            logger.error(
                                f"Failed to write file for note {note.id}: {e}"
                            )

                if file_write_errors:
                    for written in written_files:
                        try:
                            if written["original"]:
                                written["path"].write_text(
                                    written["original"], encoding="utf-8"
                                )
                        except Exception as rollback_err:
                            logger.error(
                                f"Rollback failed for {written['path']}: {rollback_err}"
                            )
                    raise BulkOperationError(
                        f"File write failed for {len(file_write_errors)} notes, rolled back",
                        operation="bulk_update_project",
                        total_count=len(note_ids),
                        success_count=0,
                        failed_ids=[e[0] for e in file_write_errors],
                        code=ErrorCode.BULK_OPERATION_FAILED,
                    )

                session.commit()

                for written in written_files:
                    try:
                        self._delete_from_obsidian(
                            written["note"].id,
                            written["old_title"],
                            written["old_project"],
                            written["old_purpose"],
                        )
                        markdown = self._note_to_markdown(written["note"])
                        self._mirror_to_obsidian(written["note"], markdown)
                    except Exception as mirror_err:
                        logger.warning(f"Obsidian mirror update failed: {mirror_err}")

        except BulkOperationError:
            raise
        except Exception as e:
            logger.error(f"Bulk update project failed: {e}")
            raise BulkOperationError(
                f"Operation failed: {e}",
                operation="bulk_update_project",
                total_count=len(note_ids),
                success_count=0,
                failed_ids=note_ids,
                code=ErrorCode.BULK_OPERATION_FAILED,
                original_error=e,
            )

        logger.info(f"Bulk moved {updated_count} notes to project '{project}'")

        if failed_ids:
            code = (
                ErrorCode.BULK_OPERATION_FAILED
                if updated_count == 0
                else ErrorCode.BULK_OPERATION_PARTIAL
            )
            raise BulkOperationError(
                f"{'Failed' if updated_count == 0 else 'Partial success'}: "
                f"{updated_count} of {len(note_ids)} notes",
                operation="bulk_update_project",
                total_count=len(note_ids),
                success_count=updated_count,
                failed_ids=failed_ids,
                code=code,
            )

        return updated_count

    # =========================================================================
    # Vector / Embedding Operations (sqlite-vec)
    # =========================================================================

    def store_embedding(
        self,
        note_id: str,
        embedding: "np.ndarray",
        model_name: str,
        content_hash: str,
    ) -> bool:
        """Store or update a single embedding for a note (chunk_index=0).

        Uses UPSERT semantics: deletes any existing chunks for the note,
        then inserts a single chunk_0 embedding.

        Args:
            note_id: The note ID.
            embedding: A 1-D numpy float32 array (must match configured dimension).
            model_name: Model that produced the embedding.
            content_hash: SHA-256 hex digest of the content that was embedded.

        Returns:
            True on success, False if sqlite-vec is unavailable.
        """
        if not self._vec_available:
            return False

        import numpy as np

        chunk_id = f"{note_id}::chunk_0"
        blob = embedding.astype(np.float32).tobytes()

        with self.engine.connect() as conn:
            # Delete any existing chunks for this note (handles transition
            # from multi-chunk to single-chunk when note is shortened)
            self._delete_chunks_for_note_conn(conn, note_id)

            conn.execute(
                text(
                    "INSERT INTO note_embeddings(chunk_id, embedding) "
                    "VALUES (:cid, :emb)"
                ),
                {"cid": chunk_id, "emb": blob},
            )

            conn.execute(
                text(
                    "INSERT OR REPLACE INTO embedding_metadata"
                    "(chunk_id, note_id, chunk_index, model_name, "
                    "content_hash, dimension, created_at) "
                    "VALUES (:cid, :nid, 0, :model, :hash, :dim, datetime('now'))"
                ),
                {
                    "cid": chunk_id,
                    "nid": note_id,
                    "model": model_name,
                    "hash": content_hash,
                    "dim": embedding.shape[0],
                },
            )
            conn.commit()

        return True

    def store_embeddings_batch(
        self,
        embeddings: "List[Tuple[str, np.ndarray, str, str]]",
    ) -> int:
        """Store multiple single-chunk embeddings in a single transaction.

        Same semantics as store_embedding() but batched: one commit for
        all note embeddings instead of one commit per note.  Each note
        gets a single chunk_0 entry.

        Args:
            embeddings: List of (note_id, embedding, model_name, content_hash).

        Returns:
            Number of embeddings stored, or 0 if sqlite-vec is unavailable.
        """
        if not self._vec_available or not embeddings:
            return 0

        import numpy as np

        with self.engine.connect() as conn:
            for note_id, embedding, model_name, content_hash in embeddings:
                chunk_id = f"{note_id}::chunk_0"
                blob = embedding.astype(np.float32).tobytes()
                self._delete_chunks_for_note_conn(conn, note_id)
                conn.execute(
                    text(
                        "INSERT INTO note_embeddings(chunk_id, embedding) "
                        "VALUES (:cid, :emb)"
                    ),
                    {"cid": chunk_id, "emb": blob},
                )
                conn.execute(
                    text(
                        "INSERT OR REPLACE INTO embedding_metadata"
                        "(chunk_id, note_id, chunk_index, model_name, "
                        "content_hash, dimension, created_at) "
                        "VALUES (:cid, :nid, 0, :model, :hash, :dim, datetime('now'))"
                    ),
                    {
                        "cid": chunk_id,
                        "nid": note_id,
                        "model": model_name,
                        "hash": content_hash,
                        "dim": embedding.shape[0],
                    },
                )
            conn.commit()

        return len(embeddings)

    def get_embedding(self, note_id: str) -> "Optional[np.ndarray]":
        """Retrieve the first (chunk_0) embedding vector for a note.

        Args:
            note_id: The note ID.

        Returns:
            A 1-D float32 numpy array, or None if not found / vec unavailable.
        """
        if not self._vec_available:
            return None

        import numpy as np

        chunk_id = f"{note_id}::chunk_0"
        with self.engine.connect() as conn:
            row = conn.execute(
                text("SELECT embedding FROM note_embeddings WHERE chunk_id = :cid"),
                {"cid": chunk_id},
            ).fetchone()

        if row is None:
            return None

        return np.frombuffer(row[0], dtype=np.float32).copy()

    def get_embedding_metadata(self, note_id: str) -> "Optional[dict]":
        """Retrieve embedding metadata for a note (from chunk_0 or any first chunk).

        Args:
            note_id: The note ID.

        Returns:
            A dict with keys model_name, content_hash, dimension, created_at,
            or None if not found.
        """
        if not self._vec_available:
            return None

        with self.engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT model_name, content_hash, dimension, created_at "
                    "FROM embedding_metadata WHERE note_id = :nid "
                    "ORDER BY chunk_index LIMIT 1"
                ),
                {"nid": note_id},
            ).fetchone()

        if row is None:
            return None

        return {
            "model_name": row[0],
            "content_hash": row[1],
            "dimension": row[2],
            "created_at": row[3],
        }

    def delete_embedding(self, note_id: str) -> bool:
        """Delete all embeddings (all chunks and metadata) for a note.

        Returns:
            True if rows were deleted, False if nothing existed or vec unavailable.
        """
        if not self._vec_available:
            return False

        with self.engine.connect() as conn:
            deleted = self._delete_chunks_for_note_conn(conn, note_id)
            conn.commit()

        return deleted > 0

    def vec_similarity_search(
        self,
        query_embedding: "np.ndarray",
        limit: int = 10,
        exclude_ids: "Optional[list[str]]" = None,
    ) -> "list[tuple[str, float]]":
        """Find notes whose embeddings are closest to *query_embedding*.

        sqlite-vec uses L2 distance.  Because our embeddings are L2-normalised,
        the ranking is equivalent to cosine similarity:
            cosine_sim = 1 - (l2_dist² / 2)

        Results are deduplicated by note_id: when a note has multiple chunks,
        only the best (lowest distance) chunk is kept.

        Args:
            query_embedding: 1-D float32 numpy array.
            limit: Maximum number of results.
            exclude_ids: Note IDs to exclude from results.

        Returns:
            List of (note_id, distance) tuples ordered by ascending distance
            (most similar first).  Empty list if vec unavailable.
        """
        if not self._vec_available:
            return []

        import numpy as np

        blob = query_embedding.astype(np.float32).tobytes()

        # Over-fetch to compensate for chunk dedup and exclusions.
        # Multiple chunks from the same note may appear in KNN results;
        # fetching extra ensures enough unique notes after grouping.
        exclusion_extra = len(exclude_ids) if exclude_ids else 0
        fetch_limit = limit * 3 + exclusion_extra

        with self.engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT chunk_id, distance "
                    "FROM note_embeddings "
                    "WHERE embedding MATCH :qvec "
                    "AND k = :k "
                    "ORDER BY distance"
                ),
                {"qvec": blob, "k": fetch_limit},
            ).fetchall()

        # Dedup: group by note_id, keep best (first/lowest) distance per note
        exclude_set = set(exclude_ids) if exclude_ids else set()
        seen_notes: dict[str, float] = {}
        for row in rows:
            chunk_id = row[0]
            distance = float(row[1])
            # Parse note_id from chunk_id ("{note_id}::chunk_{index}")
            sep_idx = chunk_id.rfind("::chunk_")
            if sep_idx == -1:
                note_id = chunk_id  # Fallback for legacy data
            else:
                note_id = chunk_id[:sep_idx]

            if note_id in exclude_set:
                continue
            if note_id not in seen_notes:
                seen_notes[note_id] = distance

        # Sort by distance and return top `limit`
        results = sorted(seen_notes.items(), key=lambda x: x[1])
        return results[:limit]

    def count_embeddings(self) -> int:
        """Return the number of stored embedding chunks.

        Returns:
            Count of chunk embeddings, or 0 if vec unavailable.
        """
        if not self._vec_available:
            return 0

        with self.engine.connect() as conn:
            row = conn.execute(
                text("SELECT count(*) FROM embedding_metadata")
            ).fetchone()

        return row[0] if row else 0

    def count_embedded_notes(self) -> int:
        """Return the number of distinct notes that have embeddings.

        Returns:
            Count of unique note_ids with embeddings, or 0 if vec unavailable.
        """
        if not self._vec_available:
            return 0

        with self.engine.connect() as conn:
            row = conn.execute(
                text("SELECT count(DISTINCT note_id) FROM embedding_metadata")
            ).fetchone()

        return row[0] if row else 0

    def clear_all_embeddings(self) -> int:
        """Delete all embeddings and metadata.  Used for reindexing.

        Returns:
            Number of embedding chunks deleted, or 0 if vec unavailable.
        """
        if not self._vec_available:
            return 0

        with self.engine.connect() as conn:
            count_row = conn.execute(
                text("SELECT count(*) FROM embedding_metadata")
            ).fetchone()
            count = count_row[0] if count_row else 0

            conn.execute(text("DELETE FROM note_embeddings"))
            conn.execute(text("DELETE FROM embedding_metadata"))
            conn.commit()

        return count

    def _delete_chunks_for_note_conn(self, conn, note_id: str) -> int:
        """Delete all chunks for a note within an existing connection.

        Returns the number of vec0 rows deleted.
        """
        # Find all chunk_ids for this note from metadata
        rows = conn.execute(
            text("SELECT chunk_id FROM embedding_metadata WHERE note_id = :nid"),
            {"nid": note_id},
        ).fetchall()

        for row in rows:
            conn.execute(
                text("DELETE FROM note_embeddings WHERE chunk_id = :cid"),
                {"cid": row[0]},
            )

        result = conn.execute(
            text("DELETE FROM embedding_metadata WHERE note_id = :nid"),
            {"nid": note_id},
        )
        return result.rowcount

    def delete_chunks_for_note(self, note_id: str) -> bool:
        """Delete all embedding chunks for a note.

        Returns:
            True if rows were deleted, False otherwise.
        """
        if not self._vec_available:
            return False

        with self.engine.connect() as conn:
            deleted = self._delete_chunks_for_note_conn(conn, note_id)
            conn.commit()

        return deleted > 0

    def store_chunk_embeddings(
        self,
        note_id: str,
        chunks: "List[Tuple[int, np.ndarray]]",
        model_name: str,
        content_hash: str,
    ) -> int:
        """Store multiple chunk embeddings for a single note.

        Deletes any existing chunks for the note first, then inserts all new chunks
        in a single transaction.

        Args:
            note_id: The note ID.
            chunks: List of (chunk_index, embedding_vector) tuples.
            model_name: Model that produced the embeddings.
            content_hash: SHA-256 hex digest of the whole note content.

        Returns:
            Number of chunks stored, or 0 if sqlite-vec is unavailable.
        """
        if not self._vec_available or not chunks:
            return 0

        import numpy as np

        with self.engine.connect() as conn:
            # Remove existing chunks for this note
            self._delete_chunks_for_note_conn(conn, note_id)

            for chunk_index, embedding in chunks:
                chunk_id = f"{note_id}::chunk_{chunk_index}"
                blob = embedding.astype(np.float32).tobytes()

                conn.execute(
                    text(
                        "INSERT INTO note_embeddings(chunk_id, embedding) "
                        "VALUES (:cid, :emb)"
                    ),
                    {"cid": chunk_id, "emb": blob},
                )
                conn.execute(
                    text(
                        "INSERT OR REPLACE INTO embedding_metadata"
                        "(chunk_id, note_id, chunk_index, model_name, "
                        "content_hash, dimension, created_at) "
                        "VALUES (:cid, :nid, :idx, :model, :hash, :dim, "
                        "datetime('now'))"
                    ),
                    {
                        "cid": chunk_id,
                        "nid": note_id,
                        "idx": chunk_index,
                        "model": model_name,
                        "hash": content_hash,
                        "dim": embedding.shape[0],
                    },
                )

            conn.commit()

        return len(chunks)
