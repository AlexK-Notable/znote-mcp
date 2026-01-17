"""Repository for note storage and retrieval."""
import datetime
from datetime import timezone
import logging
import os
import re
import shutil
import sqlite3
import threading
import weakref
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import frontmatter
import yaml
from sqlalchemy import and_, create_engine, func, or_, select, text
from sqlalchemy.exc import DatabaseError as SQLAlchemyDatabaseError
from sqlalchemy.exc import OperationalError as SQLAlchemyOperationalError
from sqlalchemy.orm import Session, joinedload

from znote_mcp.config import config
from znote_mcp.models.db_models import (Base, DBLink, DBNote, DBTag,
                                            get_session_factory, init_db,
                                            note_tags, rebuild_fts_index)
from znote_mcp.models.schema import (
    Link, LinkType, Note, NoteType, Tag, validate_safe_path_component,
    utc_now, ensure_timezone_aware
)
from znote_mcp.storage.base import Repository
from znote_mcp.exceptions import (
    BulkOperationError,
    DatabaseCorruptionError,
    ErrorCode,
    SearchError,
    ValidationError,
)

logger = logging.getLogger(__name__)

class NoteRepository(Repository[Note]):
    """Repository for note storage and retrieval.
    This implements a dual storage approach:
    1. Notes are stored as Markdown files on disk for human readability and editing
    2. SQLite database (with WAL mode) is used for indexing and efficient querying
    The file system is the source of truth - database is rebuilt from files if needed.
    """
    
    def __init__(self, notes_dir: Optional[Path] = None):
        """Initialize the repository.

        Args:
            notes_dir: Path to directory containing note markdown files.
                       If None, uses config.notes_dir.

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

        # Ensure directories exist
        self.notes_dir.mkdir(parents=True, exist_ok=True)

        # PATH VALIDATION: Warn about common misconfigurations
        self._validate_notes_dir()

        # Log the configuration being used
        logger.info(
            f"NoteRepository initialized: notes_dir={self.notes_dir}, "
            f"db_url={config.get_db_url()}"
        )

        # Initialize Obsidian vault mirror (optional)
        self.obsidian_vault_path = config.get_obsidian_vault_path()
        if self.obsidian_vault_path:
            logger.info(f"Obsidian vault mirror enabled: {self.obsidian_vault_path}")

        # Initialize database
        self.engine = init_db()
        self.session_factory = get_session_factory(self.engine)

        # File access lock (for bulk operations on multiple files)
        self.file_lock = threading.RLock()

        # Per-note locks to prevent update-delete races (uses WeakValueDictionary
        # so locks are garbage collected when no longer held by any thread)
        self._note_locks: weakref.WeakValueDictionary[str, threading.RLock] = (
            weakref.WeakValueDictionary()
        )
        self._note_locks_lock = threading.Lock()  # Protects _note_locks dict access

        # FTS5 availability tracking - allows graceful degradation when corrupted
        self._fts_available: bool = True

        # Clean up any orphaned staging files from previous failed operations
        self._cleanup_staging()

        # Check database health and recover if needed
        self._initialize_with_health_check()

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
                    logger.warning(f"Failed to remove orphaned staging file {file_path.name}: {e}")

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
                    if self._attempt_fts_recovery():
                        self._fts_available = True

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
                        text("INSERT INTO notes_fts(notes_fts) VALUES('integrity-check')")
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
                    original_error=e
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
                original_error=e
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
                        text("DELETE FROM links WHERE source_id = :id OR target_id = :id"),
                        {"id": orphan_id}
                    )
                    session.execute(
                        text("DELETE FROM note_tags WHERE note_id = :id"),
                        {"id": orphan_id}
                    )
                    session.execute(
                        text("DELETE FROM notes WHERE id = :id"),
                        {"id": orphan_id}
                    )

            # Step 4: Upsert from files (process in batches for memory efficiency)
            note_files = list(self.notes_dir.glob("*.md"))
            batch_size = 100
            total_processed = 0
            failed_files: List[str] = []

            for i in range(0, len(note_files), batch_size):
                batch = note_files[i:i + batch_size]

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

    def _upsert_note_in_session(self, session: Session, note: Note) -> None:
        """Upsert a note within an existing session (no commit).

        This is used by rebuild_index() to batch multiple upserts into a single
        transaction. Unlike _index_note(), this method does not create its own
        session or commit - it operates within the provided session.

        Args:
            session: The SQLAlchemy session to use.
            note: The Note object to upsert.
        """
        # Check if note exists
        db_note = session.scalar(select(DBNote).where(DBNote.id == note.id))

        if db_note:
            # Update existing note
            db_note.title = note.title
            db_note.content = note.content
            db_note.note_type = note.note_type.value
            db_note.updated_at = note.updated_at
            db_note.project = note.project
            # Clear existing links and tags to rebuild them
            session.execute(
                text("DELETE FROM links WHERE source_id = :note_id"),
                {"note_id": note.id}
            )
            session.execute(
                text("DELETE FROM note_tags WHERE note_id = :note_id"),
                {"note_id": note.id}
            )
        else:
            # Create new note
            db_note = DBNote(
                id=note.id,
                title=note.title,
                content=note.content,
                note_type=note.note_type.value,
                created_at=note.created_at,
                updated_at=note.updated_at,
                project=note.project
            )
            session.add(db_note)

        session.flush()  # Flush to ensure note exists before adding relationships

        # Add tags (using atomic get-or-create)
        for tag in note.tags:
            db_tag = self._get_or_create_tag(session, tag.name)
            if db_tag not in db_note.tags:
                db_note.tags.append(db_tag)

        # Add links
        for link in note.links:
            db_link = DBLink(
                source_id=link.source_id,
                target_id=link.target_id,
                link_type=link.link_type.value,
                description=link.description,
                created_at=link.created_at
            )
            session.add(db_link)
    
    def _parse_note_from_markdown(self, content: str) -> Note:
        """Parse a note from markdown content."""
        # Parse frontmatter
        post = frontmatter.loads(content)
        metadata = post.metadata
        
        # Extract ID from metadata or filename
        note_id = metadata.get("id")
        if not note_id:
            raise ValueError("Note ID missing from frontmatter")
        
        # Extract title from metadata or first heading
        title = metadata.get("title")
        if not title:
            # Try to extract from content
            lines = post.content.strip().split("\n")
            for line in lines:
                if line.startswith("# "):
                    title = line[2:].strip()
                    break
        if not title:
            raise ValueError("Note title missing from frontmatter or content")
        
        # Extract note type
        note_type_str = metadata.get("type", NoteType.PERMANENT.value)
        try:
            note_type = NoteType(note_type_str)
        except ValueError:
            logger.warning(
                f"Unknown note type '{note_type_str}' in note {note_id}, "
                "defaulting to PERMANENT"
            )
            note_type = NoteType.PERMANENT

        # Extract project (default to "general" for backwards compatibility)
        project = metadata.get("project", "general")
        
        # Extract tags
        tags_str = metadata.get("tags", "")
        if isinstance(tags_str, str):
            tag_names = [t.strip() for t in tags_str.split(",") if t.strip()]
        elif isinstance(tags_str, list):
            tag_names = [str(t).strip() for t in tags_str if str(t).strip()]
        else:
            tag_names = []
        tags = [Tag(name=name) for name in tag_names]
        
        # Extract links
        links = []
        links_section = False
        for line in post.content.split("\n"):
            line = line.strip()
            # Check if we're in the links section
            if line.startswith("## Links"):
                links_section = True
                continue
            if links_section and line.startswith("## "):
                # We've reached the next section
                links_section = False
                continue
            if links_section and line.startswith("- "):
                # Parse link line
                try:
                    # Example format: - reference [[202101010000]] Optional description
                    line_content = line.strip()
                    if "[[" in line_content and "]]" in line_content:
                        # Split the line at the [[ delimiter
                        parts = line_content.split("[[", 1)
                        # Extract the link type from before [[
                        link_type_str = parts[0].strip()
                        # Remove the leading "- " from the link type string
                        if link_type_str.startswith("- "):
                            link_type_str = link_type_str[2:].strip()
                        # Extract target ID and description
                        id_and_description = parts[1].split("]]", 1)
                        target_id = id_and_description[0].strip()
                        description = None
                        if len(id_and_description) > 1:
                            description = id_and_description[1].strip()
                        # Validate link type
                        try:
                            link_type = LinkType(link_type_str)
                        except ValueError:
                            # If not a valid type, default to reference with warning
                            logger.warning(
                                f"Unknown link type '{link_type_str}' in note {note_id}, "
                                "defaulting to REFERENCE"
                            )
                            link_type = LinkType.REFERENCE
                        links.append(
                            Link(
                                source_id=note_id,
                                target_id=target_id,
                                link_type=link_type,
                                description=description,
                                created_at=utc_now()
                            )
                        )
                except (ValueError, IndexError) as e:
                    # Malformed link line - log and skip this link
                    logger.warning(f"Skipping malformed link in note {note_id}: {line} - {e}")
                # Let other exceptions (bugs in Link constructor, etc.) propagate
        
        # Extract timestamps
        created_str = metadata.get("created")
        created_at = (
            datetime.datetime.fromisoformat(created_str)
            if created_str
            else utc_now()
        )
        updated_str = metadata.get("updated")
        updated_at = (
            datetime.datetime.fromisoformat(updated_str)
            if updated_str
            else created_at
        )
        
        # Create note object
        return Note(
            id=note_id,
            title=title,
            content=post.content,
            note_type=note_type,
            project=project,
            tags=tags,
            links=links,
            created_at=created_at,
            updated_at=updated_at,
            metadata={k: v for k, v in metadata.items()
                     if k not in ["id", "title", "type", "project", "tags", "created", "updated"]}
        )
    
    def _index_note(self, note: Note) -> None:
        """Index a note in the database."""
        with self.session_factory() as session:
            # Create or update note
            db_note = session.scalar(select(DBNote).where(DBNote.id == note.id))
            if db_note:
                # Update existing note
                db_note.title = note.title
                db_note.content = note.content
                db_note.note_type = note.note_type.value
                db_note.updated_at = note.updated_at
                db_note.project = note.project
                # Clear existing links and tags to rebuild them (parameterized queries)
                session.execute(text("DELETE FROM links WHERE source_id = :note_id"), {"note_id": note.id})
                session.execute(text("DELETE FROM note_tags WHERE note_id = :note_id"), {"note_id": note.id})
            else:
                # Create new note
                db_note = DBNote(
                    id=note.id,
                    title=note.title,
                    content=note.content,
                    note_type=note.note_type.value,
                    created_at=note.created_at,
                    updated_at=note.updated_at,
                    project=note.project
                )
                session.add(db_note)
                
            session.flush()  # Flush to get the note ID
            
            # Add tags (using atomic get-or-create to handle race conditions)
            for tag in note.tags:
                db_tag = self._get_or_create_tag(session, tag.name)
                db_note.tags.append(db_tag)
            
            # Add links
            for link in note.links:
                # Check if this link already exists in the database
                existing_link = session.scalar(
                    select(DBLink).where(
                        (DBLink.source_id == link.source_id) &
                        (DBLink.target_id == link.target_id) &
                        (DBLink.link_type == link.link_type.value)
                    )
                )
                
                if not existing_link:
                    db_link = DBLink(
                        source_id=link.source_id,
                        target_id=link.target_id,
                        link_type=link.link_type.value,
                        description=link.description,
                        created_at=link.created_at
                    )
                    session.add(db_link)
            
            # Commit changes
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
            text("INSERT OR IGNORE INTO tags (name) VALUES (:name)"),
            {"name": tag_name}
        )
        # Now SELECT - the tag definitely exists (either we created it or it existed)
        db_tag = session.scalar(select(DBTag).where(DBTag.name == tag_name))
        return db_tag

    def _note_to_markdown(self, note: Note) -> str:
        """Convert a note to markdown with frontmatter."""
        # Create frontmatter
        metadata = {
            "id": note.id,
            "title": note.title,
            "type": note.note_type.value,
            "project": note.project,
            "tags": [tag.name for tag in note.tags],
            "created": note.created_at.isoformat(),
            "updated": note.updated_at.isoformat()
        }
        # Add any custom metadata
        metadata.update(note.metadata)
        
        # Check if content already starts with the title
        title_heading = f"# {note.title}"
        if note.content.strip().startswith(title_heading):
            content = note.content
        else:
            content = f"{title_heading}\n\n{note.content}"
        
        # Remove existing Links section(s)
        content_parts = []
        skip_section = False
        for line in content.split("\n"):
            if line.strip() == "## Links":
                skip_section = True
                continue
            elif skip_section and line.startswith("## "):
                skip_section = False
            
            if not skip_section:
                content_parts.append(line)
        
        # Reconstruct the content without the Links sections
        content = "\n".join(content_parts).rstrip()
        
        # Add links section (with deduplication)
        if note.links:
            unique_links = {}  # Use dict to deduplicate
            for link in note.links:
                key = f"{link.target_id}:{link.link_type.value}"
                unique_links[key] = link
            content += "\n\n## Links\n"
            for link in unique_links.values():
                desc = f" {link.description}" if link.description else ""
                content += f"- {link.link_type.value} [[{link.target_id}]]{desc}\n"
        
        # Create markdown with frontmatter
        post = frontmatter.Post(content, **metadata)
        return frontmatter.dumps(post)

    def _mirror_to_obsidian(self, note: Note, markdown: str) -> None:
        """Mirror a note to the Obsidian vault if configured.

        Creates a copy of the note in the Obsidian vault using the note's project
        as subdirectory. Filename includes sanitized title + ID suffix to prevent
        collisions (e.g., "Hello: World" and "Hello; World" both sanitize to
        "Hello_ World" but get unique filenames via ID suffix).
        """
        if not self.obsidian_vault_path:
            return

        # Sanitize the project name for use as a directory
        safe_project = "".join(
            c if c.isalnum() or c in " -_" else "_"
            for c in note.project
        ).strip() or "general"

        # Sanitize the title for use as a filename
        safe_title = "".join(
            c if c.isalnum() or c in " -_" else "_"
            for c in note.title
        ).strip()

        # Create unique filename with ID suffix to prevent title collisions
        # Use last 8 chars of ID for brevity while maintaining uniqueness
        id_suffix = note.id[-8:] if len(note.id) >= 8 else note.id
        if safe_title:
            safe_filename = f"{safe_title} ({id_suffix})"
        else:
            safe_filename = note.id

        # Create project subdirectory
        project_dir = self.obsidian_vault_path / safe_project
        project_dir.mkdir(parents=True, exist_ok=True)

        obsidian_file_path = project_dir / f"{safe_filename}.md"

        try:
            with open(obsidian_file_path, "w", encoding="utf-8") as f:
                f.write(markdown)
            logger.debug(f"Mirrored note to Obsidian: {obsidian_file_path}")
        except IOError as e:
            # Log but don't fail - Obsidian mirror is secondary
            logger.warning(f"Failed to mirror note to Obsidian vault: {e}")

    def _delete_from_obsidian(self, note_id: str, title: Optional[str] = None,
                               project: Optional[str] = None) -> None:
        """Delete a note's mirror from the Obsidian vault if configured.

        Searches for files with ID suffix pattern: "Title (id_suffix).md"
        Uses project directory if known, otherwise searches all subdirectories.
        """
        if not self.obsidian_vault_path:
            return

        # Get ID suffix used in filename
        id_suffix = note_id[-8:] if len(note_id) >= 8 else note_id

        # Determine which directories to search
        search_dirs: List[Path] = []
        if project:
            safe_project = "".join(
                c if c.isalnum() or c in " -_" else "_"
                for c in project
            ).strip() or "general"
            project_dir = self.obsidian_vault_path / safe_project
            if project_dir.exists():
                search_dirs.append(project_dir)
        
        # If no project specified or project dir doesn't exist, search all subdirs
        if not search_dirs:
            search_dirs = [
                d for d in self.obsidian_vault_path.iterdir() 
                if d.is_dir()
            ]

        # Search for file with matching ID suffix pattern
        for search_dir in search_dirs:
            # Primary: Find by ID suffix glob pattern
            for file_path in search_dir.glob(f"*({id_suffix}).md"):
                try:
                    file_path.unlink()
                    logger.debug(f"Deleted Obsidian mirror: {file_path}")
                    return  # Found and deleted
                except OSError as e:
                    logger.warning(f"Failed to delete Obsidian mirror {file_path}: {e}")

            # Fallback: Check for legacy filename (just ID, no title)
            legacy_path = search_dir / f"{note_id}.md"
            if legacy_path.exists():
                try:
                    legacy_path.unlink()
                    logger.debug(f"Deleted legacy Obsidian mirror: {legacy_path}")
                    return
                except OSError as e:
                    logger.warning(f"Failed to delete legacy Obsidian mirror {legacy_path}: {e}")

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
            raise IOError(f"Failed to write note to {file_path}: {e}")

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
            raise IOError(f"Failed to read note {id}: {e}")
    
    def get_by_title(self, title: str) -> Optional[Note]:
        """Get a note by title."""
        with self.session_factory() as session:
            db_note = session.scalar(
                select(DBNote).where(DBNote.title == title)
            )
            if not db_note:
                return None
            return self.get(db_note.id)
    
    def get_all(
        self,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[Note]:
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
            # Build query with eager loading
            query = select(DBNote).options(
                joinedload(DBNote.tags),
                joinedload(DBNote.outgoing_links),
                joinedload(DBNote.incoming_links)
            ).order_by(DBNote.created_at.desc())

            # Apply pagination at SQL level
            if offset > 0:
                query = query.offset(offset)
            if limit is not None:
                query = query.limit(limit)

            result = session.execute(query)
            # Apply unique() to handle the duplicate rows from eager loading
            db_notes = result.unique().scalars().all()

            # Process notes in batches to reduce memory usage
            batch_size = 50
            all_notes = []
            failed_ids: List[str] = []
            # Create batches of note IDs
            note_ids = [note.id for note in db_notes]
            for i in range(0, len(note_ids), batch_size):
                batch_ids = note_ids[i:i + batch_size]
                note_batch = []
                # Process each note in the batch
                for note_id in batch_ids:
                    try:
                        note = self.get(note_id)
                        if note:
                            note_batch.append(note)
                    except (IOError, OSError, ValueError, yaml.YAMLError) as e:
                        # File/parsing errors - log and continue
                        logger.error(f"Error loading note {note_id}: {e}")
                        failed_ids.append(note_id)
                    # Let system errors (MemoryError, etc.) and bugs propagate
                all_notes.extend(note_batch)

            if failed_ids:
                logger.warning(
                    f"Failed to load {len(failed_ids)} of {len(note_ids)} notes: "
                    f"{failed_ids[:5]}{'...' if len(failed_ids) > 5 else ''}"
                )
            return all_notes

    def count_notes(self) -> int:
        """Get total count of notes in the repository.

        Useful for pagination UI when using get_all() with limit/offset.

        Returns:
            Total number of notes.
        """
        with self.session_factory() as session:
            result = session.execute(select(func.count(DBNote.id)))
            return result.scalar() or 0

    def get_by_project(self, project: str) -> List[Note]:
        """Get all notes for a specific project using SQL-level filtering.

        This method queries the database directly for notes matching the project,
        avoiding full file scans. Much faster than filtering get_all() results.

        Args:
            project: The project name to filter by.

        Returns:
            List of Note objects belonging to the specified project.
        """
        with self.session_factory() as session:
            query = select(DBNote).where(DBNote.project == project).options(
                joinedload(DBNote.tags),
                joinedload(DBNote.outgoing_links),
                joinedload(DBNote.incoming_links)
            )
            result = session.execute(query)
            db_notes = result.unique().scalars().all()

            notes = []
            for db_note in db_notes:
                try:
                    note = self.get(db_note.id)
                    if note:
                        notes.append(note)
                except (IOError, OSError, ValueError, yaml.YAMLError) as e:
                    logger.error(f"Error loading note {db_note.id}: {e}")

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
                raise ValueError(f"Note with ID {note.id} does not exist")

            # If title or project changed, delete old Obsidian mirror (will be recreated)
            if existing_note.title != note.title or existing_note.project != note.project:
                self._delete_from_obsidian(note.id, existing_note.title, existing_note.project)

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
                raise IOError(f"Failed to write note to {file_path}: {e}")

            # Mirror to Obsidian vault if configured
            self._mirror_to_obsidian(note, markdown)

            try:
                # Re-index in database
                with self.session_factory() as session:
                    # Get the existing note from the database
                    db_note = session.scalar(select(DBNote).where(DBNote.id == note.id))
                    if db_note:
                        # Update the note fields
                        db_note.title = note.title
                        db_note.content = note.content
                        db_note.note_type = note.note_type.value
                        db_note.updated_at = note.updated_at

                        # Clear existing tags
                        db_note.tags = []

                        # Add tags (using atomic get-or-create to handle race conditions)
                        for tag in note.tags:
                            db_tag = self._get_or_create_tag(session, tag.name)
                            db_note.tags.append(db_tag)

                        # For links, we'll delete existing links and add the new ones (parameterized)
                        session.execute(text("DELETE FROM links WHERE source_id = :note_id"), {"note_id": note.id})

                        # Add new links
                        for link in note.links:
                            db_link = DBLink(
                                source_id=link.source_id,
                                target_id=link.target_id,
                                link_type=link.link_type.value,
                                description=link.description,
                                created_at=link.created_at
                            )
                            session.add(db_link)

                        session.commit()
                    else:
                        # This would be unusual, but handle it by creating a new database record
                        self._index_note(note)
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
                raise ValueError(f"Note with ID {id} does not exist")

            # Get the note's title and project before deleting (for Obsidian mirror)
            note_title = None
            note_project = None
            try:
                note = self.get(id)
                if note:
                    note_title = note.title
                    note_project = note.project
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
                raise IOError(f"Failed to delete note {id}: {e}")

            # Delete from Obsidian vault if configured
            self._delete_from_obsidian(id, note_title, note_project)

            # Delete from database (using parameterized queries to prevent SQL injection)
            with self.session_factory() as session:
                # Delete note and its relationships
                session.execute(text("DELETE FROM links WHERE source_id = :note_id OR target_id = :note_id"), {"note_id": id})
                session.execute(text("DELETE FROM note_tags WHERE note_id = :note_id"), {"note_id": id})
                session.execute(text("DELETE FROM notes WHERE id = :note_id"), {"note_id": id})
                session.commit()
    
    def search(
        self,
        limit: Optional[int] = None,
        offset: int = 0,
        **kwargs: Any
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
                joinedload(DBNote.incoming_links)
            )
            # Process search criteria
            if "content" in kwargs:
                search_term = kwargs['content']
                # Search in both content and title since content might include the title
                query = query.where(
                    or_(
                        DBNote.content.like(f"%{search_term}%"),
                        DBNote.title.like(f"%{search_term}%")
                    )
                )
            if "title" in kwargs:
                search_title = kwargs['title']
                # Use case-insensitive search with func.lower()
                query = query.where(func.lower(DBNote.title).like(f"%{search_title.lower()}%"))
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
                query = query.join(DBNote.outgoing_links).where(DBLink.target_id == target_id)
            if "linked_from" in kwargs:
                source_id = kwargs["linked_from"]
                query = query.join(DBNote.incoming_links).where(DBLink.source_id == source_id)
            if "created_after" in kwargs:
                query = query.where(DBNote.created_at >= kwargs["created_after"])
            if "created_before" in kwargs:
                query = query.where(DBNote.created_at <= kwargs["created_before"])
            if "updated_after" in kwargs:
                query = query.where(DBNote.updated_at >= kwargs["updated_after"])
            if "updated_before" in kwargs:
                query = query.where(DBNote.updated_at <= kwargs["updated_before"])

            # Order by creation date (newest first) for consistent pagination
            query = query.order_by(DBNote.created_at.desc())

            # Apply pagination at SQL level
            if offset > 0:
                query = query.offset(offset)
            if limit is not None:
                query = query.limit(limit)

            # Execute query and apply unique() to handle duplicates from joins
            result = session.execute(query)
            db_notes = result.unique().scalars().all()

        # Load notes from file system (hybrid approach - file is source of truth)
        notes = []
        for db_note in db_notes:
            note = self.get(db_note.id)
            if note:
                notes.append(note)
        return notes

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

            # Apply same filters as search()
            if "content" in kwargs:
                search_term = kwargs['content']
                query = query.where(
                    or_(
                        DBNote.content.like(f"%{search_term}%"),
                        DBNote.title.like(f"%{search_term}%")
                    )
                )
            if "title" in kwargs:
                search_title = kwargs['title']
                query = query.where(func.lower(DBNote.title).like(f"%{search_title.lower()}%"))
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
                query = query.join(DBNote.outgoing_links).where(DBLink.target_id == target_id)
            if "linked_from" in kwargs:
                source_id = kwargs["linked_from"]
                query = query.join(DBNote.incoming_links).where(DBLink.source_id == source_id)
            if "created_after" in kwargs:
                query = query.where(DBNote.created_at >= kwargs["created_after"])
            if "created_before" in kwargs:
                query = query.where(DBNote.created_at <= kwargs["created_before"])
            if "updated_after" in kwargs:
                query = query.where(DBNote.updated_at >= kwargs["updated_after"])
            if "updated_before" in kwargs:
                query = query.where(DBNote.updated_at <= kwargs["updated_before"])

            result = session.execute(query)
            return result.scalar() or 0

    def _should_escape_fts5_query(self, query: str) -> bool:
        """Auto-detect if query should be escaped for literal matching.

        Analyzes the query to determine if it appears to use intentional
        FTS5 syntax (operators, phrases, wildcards) or if it should be
        treated as a literal search string.

        Args:
            query: The search query to analyze.

        Returns:
            True if the query should be escaped (literal matching),
            False if it appears to use intentional FTS5 syntax.
        """
        FTS5_KEYWORDS = {'AND', 'OR', 'NOT', 'NEAR'}

        # Check for intentional FTS5 boolean operators
        words = query.upper().split()
        if any(kw in words for kw in FTS5_KEYWORDS):
            return False

        # Check for phrase quotes (at least a pair of quotes)
        if query.count('"') >= 2:
            return False

        # Check for prefix wildcards (word ending with *)
        if re.search(r'\b\w+\*', query):
            return False

        # Check for column filters (e.g., title:python)
        if re.search(r'\b\w+:', query):
            return False

        # Default: escape for literal matching
        return True

    def _escape_fts5_query(self, query: str) -> str:
        """Escape FTS5 special characters for literal matching.

        Transforms the query so that FTS5 operators and special characters
        are treated as literal text rather than query syntax.

        Args:
            query: The search query to escape.

        Returns:
            Escaped query safe for FTS5 literal matching.
        """
        # Escape double quotes
        result = query.replace('"', '""')

        # Remove wildcards and boost operators that could cause syntax errors
        result = re.sub(r'[*^]', '', result)

        # Wrap FTS5 keywords in quotes to literalize them
        for kw in ['AND', 'OR', 'NOT', 'NEAR']:
            result = re.sub(rf'\b{kw}\b', f'"{kw}"', result, flags=re.IGNORECASE)

        return result

    def fts_search(
        self,
        query: str,
        limit: int = 50,
        highlight: bool = False,
        literal: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """Full-text search using SQLite FTS5 with graceful degradation.

        Performs efficient full-text search with BM25 ranking. If FTS5 is
        unavailable or corrupted, automatically falls back to LIKE-based search.

        Args:
            query: Search query. Supports basic FTS5 syntax:
                   - Simple terms: "python async"
                   - Phrases: '"async await"'
                   - Boolean: "python AND NOT java"
                   - Prefix: "program*"
                   - Column filter: "title:python"
                   Note: Complex syntax (NEAR, parentheses) may require escaping.
            limit: Maximum number of results to return.
            highlight: If True, include highlighted snippets in results.
            literal: Controls query escaping behavior:
                     - None (default): Auto-detect based on query content.
                       Escapes unless FTS5 operators/syntax detected.
                     - True: Force literal matching (escape all operators).
                     - False: Preserve FTS5 syntax (minimal escaping).

        Returns:
            List of dicts with keys:
                - id: Note ID
                - title: Note title
                - rank: BM25 relevance score (lower is better)
                - snippet: Highlighted content snippet (if highlight=True)
                - search_mode: "fts5" or "fallback" indicating search method used
        """
        # Skip FTS5 entirely if known to be unavailable (graceful degradation)
        if not self._fts_available:
            logger.debug("FTS5 unavailable, using fallback search")
            return self._fallback_text_search(query, limit)

        results = []

        with self.session_factory() as session:
            # Smart query escaping based on literal parameter
            # Auto-detect if not explicitly specified
            if literal is None:
                literal = self._should_escape_fts5_query(query)

            if literal:
                # Escape FTS5 operators for literal matching
                safe_query = self._escape_fts5_query(query)
            else:
                # Minimal escaping - preserve FTS5 syntax
                safe_query = query.replace('"', '""')

            if highlight:
                # Query with highlighted snippets
                sql = text("""
                    SELECT
                        id,
                        title,
                        bm25(notes_fts) as rank,
                        snippet(notes_fts, 2, '<mark>', '</mark>', '...', 32) as snippet
                    FROM notes_fts
                    WHERE notes_fts MATCH :query
                    ORDER BY rank
                    LIMIT :limit
                """)
            else:
                # Query without snippets (faster)
                sql = text("""
                    SELECT
                        id,
                        title,
                        bm25(notes_fts) as rank
                    FROM notes_fts
                    WHERE notes_fts MATCH :query
                    ORDER BY rank
                    LIMIT :limit
                """)

            try:
                result = session.execute(sql, {"query": safe_query, "limit": limit})
                rows = result.fetchall()

                for row in rows:
                    entry = {
                        "id": row[0],
                        "title": row[1],
                        "rank": row[2],
                        "search_mode": "fts5"  # Indicate FTS5 was used
                    }
                    if highlight and len(row) > 3:
                        entry["snippet"] = row[3]
                    results.append(entry)

            except (sqlite3.OperationalError, SQLAlchemyOperationalError) as e:
                # FTS5 syntax errors - fall back to LIKE search
                logger.warning(
                    f"FTS5 query failed for '{query}': {e}. Using fallback search."
                )
                return self._fallback_text_search(query, limit)

            except (sqlite3.DatabaseError, SQLAlchemyDatabaseError) as e:
                # Corruption or malformed database - attempt recovery
                error_msg = str(e).lower()
                if "malformed" in error_msg or "corrupt" in error_msg:
                    logger.error(
                        f"FTS5 corruption detected: {e}. Attempting auto-rebuild..."
                    )
                    if self._attempt_fts_recovery():
                        # Retry search after successful rebuild
                        logger.info("FTS5 rebuilt successfully, retrying search")
                        return self.fts_search(query, limit, highlight)
                    else:
                        # Recovery failed, disable FTS and use fallback
                        logger.error(
                            "FTS5 recovery failed. Disabling FTS5 for this session. "
                            "Call reset_fts_availability() after manual repair."
                        )
                        self._fts_available = False
                        return self._fallback_text_search(query, limit)
                else:
                    # Non-corruption database error - still fall back
                    logger.error(f"FTS5 database error: {e}. Using fallback search.")
                    return self._fallback_text_search(query, limit)

        return results

    def _attempt_fts_recovery(self) -> bool:
        """Attempt to recover FTS5 index by rebuilding it.

        Returns:
            True if recovery succeeded, False otherwise.
        """
        try:
            count = self.rebuild_fts()
            logger.info(f"FTS5 index rebuilt with {count} notes")
            return True
        except Exception as e:
            logger.error(f"FTS5 rebuild failed: {e}")
            return False

    def reset_fts_availability(self) -> bool:
        """Reset FTS5 availability flag and verify FTS5 works.

        Call this after manually repairing the FTS5 index to re-enable
        FTS5 search. Performs an integrity check before re-enabling.

        Returns:
            True if FTS5 is now available, False if still broken.
        """
        try:
            with self.session_factory() as session:
                # Test FTS5 with integrity check
                session.execute(
                    text("INSERT INTO notes_fts(notes_fts) VALUES('integrity-check')")
                )
            self._fts_available = True
            logger.info("FTS5 availability reset - FTS5 is now enabled")
            return True
        except Exception as e:
            logger.error(f"FTS5 still unavailable: {e}")
            self._fts_available = False
            return False

    def _fallback_text_search(
        self,
        query: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Fallback text search using LIKE when FTS5 fails.

        Used when FTS5 query syntax is invalid or FTS is unavailable.
        Results include search_mode='fallback' to indicate the search mode.

        Raises:
            SearchError: If the fallback search also fails.
        """
        results = []
        search_term = f"%{query}%"

        try:
            with self.session_factory() as session:
                sql = text("""
                    SELECT id, title, content
                    FROM notes
                    WHERE title LIKE :term OR content LIKE :term
                    LIMIT :limit
                """)
                result = session.execute(sql, {"term": search_term, "limit": limit})
                rows = result.fetchall()

                for row in rows:
                    # Simple relevance: title match scores higher
                    title_match = query.lower() in row[1].lower() if row[1] else False
                    rank = -2.0 if title_match else -1.0

                    results.append({
                        "id": row[0],
                        "title": row[1],
                        "rank": rank,
                        "search_mode": "fallback"  # Indicate fallback was used
                    })
        except Exception as e:
            # Wrap database errors in SearchError as documented
            raise SearchError(
                f"Fallback text search failed: {e}",
                query=query,
                code=ErrorCode.SEARCH_FAILED
            ) from e

        logger.debug(f"Fallback search returned {len(results)} results for query '{query}'")
        return results

    def rebuild_fts(self) -> int:
        """Rebuild the FTS5 index from the notes table.

        Returns:
            Number of notes indexed.
        """
        return rebuild_fts_index(self.engine)

    def find_by_tag(self, tag: Union[str, Tag]) -> List[Note]:
        """Find notes by tag."""
        tag_name = tag.name if isinstance(tag, Tag) else tag
        return self.search(tag=tag_name)
    
    def find_linked_notes(self, note_id: str, direction: str = "outgoing") -> List[Note]:
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
                        joinedload(DBNote.incoming_links)
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
                        joinedload(DBNote.incoming_links)
                    )
                )
            elif direction == "both":
                # Find both directions
                query = (
                    select(DBNote)
                    .join(
                        DBLink,
                        or_(
                            and_(DBNote.id == DBLink.target_id, DBLink.source_id == note_id),
                            and_(DBNote.id == DBLink.source_id, DBLink.target_id == note_id)
                        )
                    )
                    .options(
                        joinedload(DBNote.tags),
                        joinedload(DBNote.outgoing_links),
                        joinedload(DBNote.incoming_links)
                    )
                )
            else:
                raise ValueError(f"Invalid direction: {direction}. Use 'outgoing', 'incoming', or 'both'")
            
            result = session.execute(query)
            # Apply unique() to handle the duplicate rows from eager loading
            db_notes = result.unique().scalars().all()
            
            # Convert to model Notes
            notes = []
            for db_note in db_notes:
                note = self.get(db_note.id)
                if note:
                    notes.append(note)
            return notes

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
            note_tag_ids = session.execute(
                select(note_tags.c.tag_id).where(note_tags.c.note_id == note_id)
            ).scalars().all()

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
            outgoing = session.execute(
                select(DBLink.target_id).where(DBLink.source_id == note_id)
            ).scalars().all()
            linked_note_ids.update(outgoing)

            # Incoming links (notes that link to this note)
            incoming = session.execute(
                select(DBLink.source_id).where(DBLink.target_id == note_id)
            ).scalars().all()
            linked_note_ids.update(incoming)

            # Notes that share common link targets (linked to same notes)
            if outgoing:
                common_target_notes = session.execute(
                    select(DBLink.source_id.distinct())
                    .where(DBLink.target_id.in_(outgoing))
                    .where(DBLink.source_id != note_id)
                ).scalars().all()
                linked_note_ids.update(common_target_notes)

            # Combine all candidate IDs
            candidate_ids = shared_tag_note_ids | linked_note_ids

        # Load candidate notes from file system
        candidates = []
        for cid in candidate_ids:
            try:
                note = self.get(cid)
                if note:
                    candidates.append(note)
            except (IOError, OSError, ValueError, yaml.YAMLError) as e:
                logger.warning(f"Failed to load candidate note {cid}: {e}")

        return candidates
    
    def get_all_tags(self) -> List[Tag]:
        """Get all tags in the system."""
        with self.session_factory() as session:
            result = session.execute(select(DBTag))
            db_tags = result.scalars().all()
        return [Tag(name=tag.name) for tag in db_tags]

    def sync_to_obsidian(self) -> int:
        """Sync all notes to the Obsidian vault.

        Re-mirrors all existing notes to the configured Obsidian vault directory.
        Uses note titles as filenames, so existing files are overwritten (no duplicates).

        Returns:
            Number of notes synced, or 0 if Obsidian vault is not configured.

        Raises:
            ValueError: If Obsidian vault is not configured.
        """
        if not self.obsidian_vault_path:
            raise ValueError(
                "Obsidian vault not configured. "
                "Set ZETTELKASTEN_OBSIDIAN_VAULT in your .env file."
            )

        # Get all notes from the file system
        note_files = list(self.notes_dir.glob("*.md"))
        synced_count = 0
        failed_files: List[str] = []

        for file_path in note_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                note = self._parse_note_from_markdown(content)
                markdown = self._note_to_markdown(note)
                self._mirror_to_obsidian(note, markdown)
                synced_count += 1
            except (IOError, OSError, PermissionError) as e:
                # File access or permission errors
                logger.warning(f"Cannot access {file_path.name} for sync: {e}")
                failed_files.append(file_path.name)
            except (ValueError, yaml.YAMLError) as e:
                # Malformed note content
                logger.warning(f"Invalid note format in {file_path.name}: {e}")
                failed_files.append(file_path.name)
            # Let system errors and bugs propagate

        if failed_files:
            logger.warning(
                f"Obsidian sync completed with {len(failed_files)} failures: "
                f"{failed_files[:5]}{'...' if len(failed_files) > 5 else ''}"
            )
        logger.info(f"Synced {synced_count} of {len(note_files)} notes to Obsidian vault")
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
                code=ErrorCode.BULK_OPERATION_EMPTY_INPUT
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
                        # Create database record
                        db_note = DBNote(
                            id=note.id,
                            title=note.title,
                            content=note.content,
                            note_type=note.note_type.value,
                            created_at=note.created_at,
                            updated_at=note.updated_at
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
                                created_at=link.created_at
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
                        logger.warning(f"Failed to cleanup staging file {staging_path}: {cleanup_error}")

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
                    original_error=e
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
                code=ErrorCode.BULK_OPERATION_EMPTY_INPUT
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
                    notes_info.append({
                        "id": note_id,
                        "title": note.title,
                        "project": note.project
                    })
            except Exception:
                notes_info.append({"id": note_id, "title": None, "project": None})

        # Delete from database FIRST (atomic transaction)
        try:
            with self.session_factory() as session:
                for note_id in note_ids:
                    session.execute(
                        text("DELETE FROM links WHERE source_id = :note_id OR target_id = :note_id"),
                        {"note_id": note_id}
                    )
                    session.execute(
                        text("DELETE FROM note_tags WHERE note_id = :note_id"),
                        {"note_id": note_id}
                    )
                    session.execute(
                        text("DELETE FROM notes WHERE id = :note_id"),
                        {"note_id": note_id}
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
                original_error=e
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
            self._delete_from_obsidian(info["id"], info["title"], info["project"])

        if failed_files:
            logger.warning(f"Some files could not be deleted: {failed_files}")

        logger.info(f"Bulk deleted {deleted_count} notes")
        return deleted_count

    def bulk_add_tags(self, note_ids: List[str], tags: List[str]) -> int:
        """Add tags to multiple notes in a single atomic transaction.

        Args:
            note_ids: List of note IDs to update.
            tags: List of tag names to add.

        Returns:
            Number of notes successfully updated.

        Raises:
            BulkOperationError: If input is empty or operation fails.
            ValidationError: If any note ID fails path validation.
        """
        if not note_ids:
            raise BulkOperationError(
                "No note IDs provided for adding tags",
                operation="bulk_add_tags",
                code=ErrorCode.BULK_OPERATION_EMPTY_INPUT
            )
        if not tags:
            raise BulkOperationError(
                "No tags provided to add",
                operation="bulk_add_tags",
                code=ErrorCode.BULK_OPERATION_EMPTY_INPUT
            )

        # Validate all IDs first (security: prevent path traversal)
        for note_id in note_ids:
            validate_safe_path_component(note_id, "Note ID")

        updated_count = 0
        failed_ids = []
        # Track notes that need file updates (for atomic file operations)
        notes_to_update: List[Tuple[str, Note]] = []

        try:
            with self.session_factory() as session:
                for note_id in note_ids:
                    # Get the note from database
                    db_note = session.scalar(select(DBNote).where(DBNote.id == note_id))
                    if not db_note:
                        failed_ids.append(note_id)
                        logger.warning(f"Note {note_id} not found for adding tags")
                        continue

                    # Get existing tag names for this note
                    existing_tag_names = {tag.name for tag in db_note.tags}

                    # Add new tags (avoiding duplicates, using atomic get-or-create)
                    tags_added = False
                    for tag_name in tags:
                        if tag_name not in existing_tag_names:
                            db_tag = self._get_or_create_tag(session, tag_name)
                            db_note.tags.append(db_tag)
                            existing_tag_names.add(tag_name)
                            tags_added = True

                    if tags_added:
                        db_note.updated_at = utc_now()
                        updated_count += 1

                        # Get note for file update (defer file write until after commit)
                        note = self.get(note_id)
                        if note:
                            for tag_name in tags:
                                if not any(t.name == tag_name for t in note.tags):
                                    note.tags.append(Tag(name=tag_name))
                            notes_to_update.append((note_id, note))
                    else:
                        # Note already had all tags - still counts as "processed"
                        updated_count += 1

                # Commit all database changes atomically
                session.commit()

            # Now update files (database is already committed)
            # Wrap all file operations in a single lock acquisition
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
            logger.error(f"Bulk add tags operation failed: {e}")
            raise BulkOperationError(
                f"Database operation failed: {e}",
                operation="bulk_add_tags",
                total_count=len(note_ids),
                success_count=0,
                failed_ids=note_ids,
                code=ErrorCode.BULK_OPERATION_FAILED,
                original_error=e
            )

        logger.info(f"Bulk added tags {tags} to {updated_count} notes")

        # Raise partial failure if some notes failed
        if failed_ids:
            if updated_count == 0:
                raise BulkOperationError(
                    "Failed to add tags to any notes",
                    operation="bulk_add_tags",
                    total_count=len(note_ids),
                    success_count=0,
                    failed_ids=failed_ids,
                    code=ErrorCode.BULK_OPERATION_FAILED
                )
            else:
                raise BulkOperationError(
                    f"Partial success: added tags to {updated_count} of {len(note_ids)} notes",
                    operation="bulk_add_tags",
                    total_count=len(note_ids),
                    success_count=updated_count,
                    failed_ids=failed_ids,
                    code=ErrorCode.BULK_OPERATION_PARTIAL
                )

        return updated_count

    def bulk_remove_tags(self, note_ids: List[str], tags: List[str]) -> int:
        """Remove tags from multiple notes in a single atomic transaction.

        Args:
            note_ids: List of note IDs to update.
            tags: List of tag names to remove.

        Returns:
            Number of notes successfully updated.

        Raises:
            BulkOperationError: If input is empty or operation fails.
            ValidationError: If any note ID fails path validation.
        """
        if not note_ids:
            raise BulkOperationError(
                "No note IDs provided for removing tags",
                operation="bulk_remove_tags",
                code=ErrorCode.BULK_OPERATION_EMPTY_INPUT
            )
        if not tags:
            raise BulkOperationError(
                "No tags provided to remove",
                operation="bulk_remove_tags",
                code=ErrorCode.BULK_OPERATION_EMPTY_INPUT
            )

        # Validate all IDs first (security: prevent path traversal)
        for note_id in note_ids:
            validate_safe_path_component(note_id, "Note ID")

        updated_count = 0
        failed_ids = []
        tags_to_remove = set(tags)
        # Track notes that need file updates (for atomic file operations)
        notes_to_update: List[Tuple[str, Note]] = []

        try:
            with self.session_factory() as session:
                for note_id in note_ids:
                    # Get the note from database
                    db_note = session.scalar(select(DBNote).where(DBNote.id == note_id))
                    if not db_note:
                        failed_ids.append(note_id)
                        logger.warning(f"Note {note_id} not found for removing tags")
                        continue

                    # Remove specified tags
                    original_count = len(db_note.tags)
                    db_note.tags = [tag for tag in db_note.tags if tag.name not in tags_to_remove]

                    # Only update if tags were actually removed
                    if len(db_note.tags) < original_count:
                        db_note.updated_at = utc_now()
                        updated_count += 1

                        # Get note for file update (defer file write until after commit)
                        note = self.get(note_id)
                        if note:
                            note.tags = [t for t in note.tags if t.name not in tags_to_remove]
                            notes_to_update.append((note_id, note))
                    else:
                        # Note didn't have any of the tags - still counts as "processed"
                        updated_count += 1

                # Commit all database changes atomically
                session.commit()

            # Now update files (database is already committed)
            # Wrap all file operations in a single lock acquisition
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
            logger.error(f"Bulk remove tags operation failed: {e}")
            raise BulkOperationError(
                f"Database operation failed: {e}",
                operation="bulk_remove_tags",
                total_count=len(note_ids),
                success_count=0,
                failed_ids=note_ids,
                code=ErrorCode.BULK_OPERATION_FAILED,
                original_error=e
            )

        logger.info(f"Bulk removed tags {tags} from {updated_count} notes")

        # Raise partial failure if some notes failed
        if failed_ids:
            if updated_count == 0:
                raise BulkOperationError(
                    "Failed to remove tags from any notes",
                    operation="bulk_remove_tags",
                    total_count=len(note_ids),
                    success_count=0,
                    failed_ids=failed_ids,
                    code=ErrorCode.BULK_OPERATION_FAILED
                )
            else:
                raise BulkOperationError(
                    f"Partial success: removed tags from {updated_count} of {len(note_ids)} notes",
                    operation="bulk_remove_tags",
                    total_count=len(note_ids),
                    success_count=updated_count,
                    failed_ids=failed_ids,
                    code=ErrorCode.BULK_OPERATION_PARTIAL
                )

        return updated_count

    def bulk_update_project(self, note_ids: List[str], project: str) -> int:
        """Move multiple notes to a different project in a single atomic transaction.

        Args:
            note_ids: List of note IDs to update.
            project: Target project name.

        Returns:
            Number of notes successfully updated.

        Raises:
            BulkOperationError: If input is empty or operation fails.
            ValidationError: If any note ID fails path validation.
        """
        if not note_ids:
            raise BulkOperationError(
                "No note IDs provided for project update",
                operation="bulk_update_project",
                code=ErrorCode.BULK_OPERATION_EMPTY_INPUT
            )
        if not project or not project.strip():
            raise BulkOperationError(
                "Project name is required",
                operation="bulk_update_project",
                code=ErrorCode.BULK_OPERATION_EMPTY_INPUT
            )

        # Validate all IDs first (security: prevent path traversal)
        for note_id in note_ids:
            validate_safe_path_component(note_id, "Note ID")

        # Also validate the project name
        validate_safe_path_component(project, "Project name")

        updated_count = 0
        failed_ids = []
        # Track notes that need file updates (for atomic file operations)
        notes_to_update = []

        try:
            with self.session_factory() as session:
                for note_id in note_ids:
                    # Get the note from database
                    db_note = session.scalar(select(DBNote).where(DBNote.id == note_id))
                    if not db_note:
                        failed_ids.append(note_id)
                        logger.warning(f"Note {note_id} not found for project update")
                        continue

                    # Only update if project is different
                    old_project = db_note.project
                    if old_project != project:
                        # Get the full note for file operations
                        note = self.get(note_id)
                        if note:
                            notes_to_update.append({
                                "note": note,
                                "old_project": old_project,
                                "old_title": note.title
                            })

                        # Update database
                        db_note.project = project
                        db_note.updated_at = utc_now()
                        updated_count += 1
                    else:
                        # Note already in target project - still counts as "processed"
                        updated_count += 1

                # Commit all database changes atomically
                session.commit()

            # Now update files (database is already committed)
            # Wrap all file operations in a single lock acquisition
            with self.file_lock:
                for update_info in notes_to_update:
                    note = update_info["note"]
                    old_project = update_info["old_project"]
                    old_title = update_info["old_title"]

                    # Delete old Obsidian mirror
                    self._delete_from_obsidian(note.id, old_title, old_project)

                    # Update note's project
                    note.project = project
                    note.updated_at = utc_now()

                    # Write updated markdown file
                    markdown = self._note_to_markdown(note)
                    file_path = self.notes_dir / f"{note.id}.md"
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(markdown)

                    # Mirror to new Obsidian location
                    self._mirror_to_obsidian(note, markdown)

        except BulkOperationError:
            raise
        except Exception as e:
            logger.error(f"Bulk update project operation failed: {e}")
            raise BulkOperationError(
                f"Operation failed: {e}",
                operation="bulk_update_project",
                total_count=len(note_ids),
                success_count=0,
                failed_ids=note_ids,
                code=ErrorCode.BULK_OPERATION_FAILED,
                original_error=e
            )

        logger.info(f"Bulk moved {updated_count} notes to project '{project}'")

        # Raise partial failure if some notes failed
        if failed_ids:
            if updated_count == 0:
                raise BulkOperationError(
                    "Failed to update project for any notes",
                    operation="bulk_update_project",
                    total_count=len(note_ids),
                    success_count=0,
                    failed_ids=failed_ids,
                    code=ErrorCode.BULK_OPERATION_FAILED
                )
            else:
                raise BulkOperationError(
                    f"Partial success: moved {updated_count} of {len(note_ids)} notes to project",
                    operation="bulk_update_project",
                    total_count=len(note_ids),
                    success_count=updated_count,
                    failed_ids=failed_ids,
                    code=ErrorCode.BULK_OPERATION_PARTIAL
                )

        return updated_count
