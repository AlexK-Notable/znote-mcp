"""SQLAlchemy database models for the Zettelkasten MCP server."""

import datetime
import logging
from typing import Optional

logger = logging.getLogger(__name__)

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Table,
    Text,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy.pool import QueuePool, StaticPool

from znote_mcp.config import config
from znote_mcp.models.schema import LinkType, NotePurpose, NoteType, utc_now

# Create base class for SQLAlchemy models
Base = declarative_base()

# Association table for tags and notes
note_tags = Table(
    "note_tags",
    Base.metadata,
    Column("note_id", String(255), ForeignKey("notes.id"), primary_key=True),
    Column("tag_id", Integer, ForeignKey("tags.id"), primary_key=True),
)


class DBNote(Base):
    """Database model for a note."""

    __tablename__ = "notes"
    id = Column(String(255), primary_key=True, index=True)
    title = Column(String(255), nullable=False, index=True)
    content = Column(Text, nullable=False)
    note_type = Column(
        String(50), default=NoteType.PERMANENT.value, nullable=False, index=True
    )
    note_purpose = Column(
        String(50), default=NotePurpose.GENERAL.value, nullable=False, index=True
    )
    created_at = Column(DateTime, default=utc_now, nullable=False)
    updated_at = Column(DateTime, default=utc_now, nullable=False)
    project = Column(String(255), default="general", nullable=False, index=True)
    plan_id = Column(String(255), nullable=True, index=True)

    # Relationships
    tags = relationship("DBTag", secondary=note_tags, back_populates="notes")
    outgoing_links = relationship(
        "DBLink",
        foreign_keys="DBLink.source_id",
        back_populates="source",
        cascade="all, delete-orphan",
    )
    incoming_links = relationship(
        "DBLink",
        foreign_keys="DBLink.target_id",
        back_populates="target",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        """Return string representation of note."""
        return f"<Note(id='{self.id}', title='{self.title}')>"


class DBTag(Base):
    """Database model for a tag."""

    __tablename__ = "tags"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), unique=True, nullable=False)

    # Relationships
    notes = relationship("DBNote", secondary=note_tags, back_populates="tags")

    def __repr__(self) -> str:
        """Return string representation of tag."""
        return f"<Tag(id={self.id}, name='{self.name}')>"


class DBLink(Base):
    """Database model for a link between notes."""

    __tablename__ = "links"
    id = Column(Integer, primary_key=True, autoincrement=True)
    source_id = Column(String(255), ForeignKey("notes.id"), nullable=False, index=True)
    target_id = Column(String(255), ForeignKey("notes.id"), nullable=False, index=True)
    link_type = Column(String(50), default=LinkType.REFERENCE.value, nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utc_now, nullable=False)

    # Relationships
    source = relationship(
        "DBNote", foreign_keys=[source_id], back_populates="outgoing_links"
    )
    target = relationship(
        "DBNote", foreign_keys=[target_id], back_populates="incoming_links"
    )

    # Add a unique constraint to prevent duplicate links of the same type
    __table_args__ = (
        UniqueConstraint(
            "source_id", "target_id", "link_type", name="unique_link_type"
        ),
    )

    def __repr__(self) -> str:
        """Return string representation of link."""
        return (
            f"<Link(id={self.id}, source='{self.source_id}', "
            f"target='{self.target_id}', type='{self.link_type}')>"
        )


class DBProject(Base):
    """Database model for a project in the registry."""

    __tablename__ = "projects"
    id = Column(String(255), primary_key=True, index=True)  # e.g., "monorepo/frontend"
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    parent_id = Column(
        String(255), ForeignKey("projects.id"), nullable=True, index=True
    )
    path = Column(String(1024), nullable=True)  # Filesystem path
    created_at = Column(DateTime, default=utc_now, nullable=False)
    metadata_json = Column(Text, nullable=True)  # JSON string for flexible metadata

    # Self-referential relationship for parent/children
    parent = relationship("DBProject", remote_side=[id], back_populates="children")
    children = relationship("DBProject", back_populates="parent")

    def __repr__(self) -> str:
        """Return string representation of project."""
        return f"<Project(id='{self.id}', name='{self.name}')>"


def init_db(in_memory: bool = False) -> None:
    """Initialize the database with hardened configuration.

    Args:
        in_memory: If True, use in-memory SQLite for process isolation.
                   Each process gets its own isolated database that is
                   rebuilt from markdown files on startup. This eliminates
                   cross-process coordination overhead.

    For persistent mode (in_memory=False):
    - WAL (Write-Ahead Logging) mode for atomic writes
    - NORMAL synchronous mode (good balance of safety vs speed)
    - QueuePool for connection reuse with size limits
    - Pool pre-ping to detect stale connections
    - Connection timeout to prevent deadlocks

    For in-memory mode (in_memory=True):
    - StaticPool keeps single connection alive for process lifetime
    - No WAL mode (not applicable to in-memory)
    - Faster startup, no disk I/O for index operations
    - Index must be rebuilt from markdown files on startup
    """
    from sqlalchemy import event, text

    if in_memory:
        # In-memory SQLite for process isolation
        # StaticPool keeps the single connection alive
        engine = create_engine(
            "sqlite:///:memory:",
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )

        # Apply optimizations for in-memory (no WAL needed)
        @event.listens_for(engine, "connect")
        def set_sqlite_pragma_memory(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            # Use memory journal for in-memory DB
            cursor.execute("PRAGMA journal_mode=MEMORY")
            # Synchronous off for in-memory (no durability needed)
            cursor.execute("PRAGMA synchronous=OFF")
            # Large cache for in-memory performance
            cursor.execute("PRAGMA cache_size=-64000")  # 64MB cache
            cursor.close()

    else:
        # Persistent SQLite with connection pooling
        engine = create_engine(
            config.get_db_url(),
            poolclass=QueuePool,
            pool_size=5,  # Base pool size (concurrent reads)
            max_overflow=10,  # Allow up to 15 total connections under load
            pool_timeout=30,  # Wait up to 30s for a connection
            pool_recycle=3600,  # Recycle connections after 1 hour
            pool_pre_ping=True,  # Validate connections before use
        )

        # Apply WAL mode and other PRAGMA settings on every connection
        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            # WAL mode: writes go to separate journal, preventing corruption on crash
            cursor.execute("PRAGMA journal_mode=WAL")
            # NORMAL sync: flush WAL to disk at critical moments (good balance)
            cursor.execute("PRAGMA synchronous=NORMAL")
            # Increase cache size for better performance (negative = KB)
            cursor.execute("PRAGMA cache_size=-64000")  # 64MB cache
            cursor.close()

    Base.metadata.create_all(engine)

    # Run migrations only for persistent databases
    # (in-memory DBs are created fresh each time)
    if not in_memory:
        _migrate_add_project_column(engine)
        _migrate_add_note_purpose_columns(engine)

    # Create FTS5 virtual table for full-text search
    # (works for both in-memory and persistent)
    init_fts5(engine)

    # Initialize sqlite-vec for vector search (optional, graceful fallback)
    vec_ok = init_sqlite_vec(engine)
    if vec_ok:
        logger.info("sqlite-vec initialized successfully")
    else:
        logger.info("sqlite-vec not available — semantic search disabled")

    return engine


def _migrate_add_project_column(engine) -> None:
    """Migration: Add project column to existing databases.

    SQLite doesn't support IF NOT EXISTS for ADD COLUMN, so we check
    the schema first. This is idempotent and safe to run multiple times.
    """
    from sqlalchemy import inspect, text

    inspector = inspect(engine)
    columns = [col["name"] for col in inspector.get_columns("notes")]

    if "project" not in columns:
        with engine.connect() as conn:
            conn.execute(
                text(
                    "ALTER TABLE notes ADD COLUMN project VARCHAR(255) "
                    "NOT NULL DEFAULT 'general'"
                )
            )
            conn.commit()


def _migrate_add_note_purpose_columns(engine) -> None:
    """Migration: Add note_purpose and plan_id columns to existing databases.

    These columns support the enhanced Obsidian organization feature.
    """
    from sqlalchemy import inspect, text

    inspector = inspect(engine)
    columns = [col["name"] for col in inspector.get_columns("notes")]

    with engine.connect() as conn:
        if "note_purpose" not in columns:
            conn.execute(
                text(
                    "ALTER TABLE notes ADD COLUMN note_purpose VARCHAR(50) "
                    "NOT NULL DEFAULT 'general'"
                )
            )

        if "plan_id" not in columns:
            conn.execute(text("ALTER TABLE notes ADD COLUMN plan_id VARCHAR(255)"))

        conn.commit()


def init_fts5(engine) -> None:
    """Initialize FTS5 full-text search virtual table.

    Creates an FTS5 virtual table that mirrors the notes table for
    efficient full-text search with BM25 ranking.
    """
    from sqlalchemy import text

    with engine.connect() as conn:
        # Create FTS5 virtual table for notes (if not exists)
        # We use content="" for an external content table - we'll manually sync
        conn.execute(
            text(
                """
            CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(
                id UNINDEXED,
                title,
                content,
                content='notes',
                content_rowid='rowid'
            )
        """
            )
        )

        # Create triggers to keep FTS in sync with notes table
        # Note: We need to handle inserts, updates, and deletes

        # Trigger for INSERT - add to FTS
        conn.execute(
            text(
                """
            CREATE TRIGGER IF NOT EXISTS notes_ai AFTER INSERT ON notes BEGIN
                INSERT INTO notes_fts(rowid, id, title, content)
                VALUES (NEW.rowid, NEW.id, NEW.title, NEW.content);
            END
        """
            )
        )

        # Trigger for DELETE - remove from FTS
        conn.execute(
            text(
                """
            CREATE TRIGGER IF NOT EXISTS notes_ad AFTER DELETE ON notes BEGIN
                INSERT INTO notes_fts(notes_fts, rowid, id, title, content)
                VALUES ('delete', OLD.rowid, OLD.id, OLD.title, OLD.content);
            END
        """
            )
        )

        # Trigger for UPDATE - update FTS
        conn.execute(
            text(
                """
            CREATE TRIGGER IF NOT EXISTS notes_au AFTER UPDATE ON notes BEGIN
                INSERT INTO notes_fts(notes_fts, rowid, id, title, content)
                VALUES ('delete', OLD.rowid, OLD.id, OLD.title, OLD.content);
                INSERT INTO notes_fts(rowid, id, title, content)
                VALUES (NEW.rowid, NEW.id, NEW.title, NEW.content);
            END
        """
            )
        )

        conn.commit()


def rebuild_fts_index(engine) -> int:
    """Rebuild the FTS5 index from existing notes.

    This populates the FTS virtual table with data from the notes table.
    Useful after database migrations or when FTS gets out of sync.

    Returns:
        Number of notes indexed.
    """
    from sqlalchemy import text

    with engine.connect() as conn:
        # Clear existing FTS data
        conn.execute(text("DELETE FROM notes_fts"))

        # Repopulate from notes table
        result = conn.execute(
            text(
                """
            INSERT INTO notes_fts(rowid, id, title, content)
            SELECT rowid, id, title, content FROM notes
        """
            )
        )

        conn.commit()

        # Count indexed notes
        count_result = conn.execute(text("SELECT COUNT(*) FROM notes_fts"))
        count = count_result.scalar()

    return count


def init_sqlite_vec(engine, dimension: int = 768) -> bool:
    """Initialize sqlite-vec extension for vector search.

    Loads the sqlite-vec extension and creates the vec0 virtual table
    for storing note embeddings, plus a metadata table for tracking
    model info and content hashes.

    This follows the same graceful degradation pattern as FTS5:
    if the extension isn't available, the system works without it.

    Args:
        engine: SQLAlchemy engine.
        dimension: Embedding vector dimensionality (default 768 for gte-modernbert-base).

    Returns:
        True if sqlite-vec was initialized successfully, False otherwise.
    """
    from sqlalchemy import event, text

    try:
        import sqlite_vec
    except ImportError:
        logger.debug("sqlite-vec package not installed — vector search disabled")
        return False

    # Register extension loading on every new connection
    @event.listens_for(engine, "connect")
    def _load_sqlite_vec(dbapi_connection, connection_record):
        try:
            try:
                dbapi_connection.enable_load_extension(True)
                sqlite_vec.load(dbapi_connection)
            finally:
                dbapi_connection.enable_load_extension(False)
        except Exception as e:
            logger.warning(f"Failed to load sqlite-vec extension: {e}")

    # Create the vec0 virtual table and metadata table
    try:
        with engine.connect() as conn:
            # Force-load extension on this connection (the event listener
            # handles future connections, but this one may already be open)
            raw_conn = conn.connection.dbapi_connection
            try:
                raw_conn.enable_load_extension(True)
                sqlite_vec.load(raw_conn)
            finally:
                raw_conn.enable_load_extension(False)

            # vec0 virtual table for vector storage + KNN search
            conn.execute(
                text(
                    f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS note_embeddings
                USING vec0(
                    note_id TEXT PRIMARY KEY,
                    embedding float[{dimension}]
                )
            """
                )
            )

            # Metadata table for tracking model/hash info per embedding
            conn.execute(
                text(
                    """
                CREATE TABLE IF NOT EXISTS embedding_metadata (
                    note_id TEXT PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    dimension INTEGER NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (note_id) REFERENCES notes(id) ON DELETE CASCADE
                )
            """
                )
            )

            # Index on content_hash for fast change detection
            conn.execute(
                text(
                    """
                CREATE INDEX IF NOT EXISTS idx_embedding_metadata_hash
                ON embedding_metadata(content_hash)
            """
                )
            )

            conn.commit()
            return True

    except Exception as e:
        logger.warning(f"Failed to initialize sqlite-vec tables: {e}")
        return False


def get_session_factory(engine=None):
    """Get a session factory for the database."""
    if engine is None:
        engine = create_engine(config.get_db_url())
    return sessionmaker(bind=engine)
