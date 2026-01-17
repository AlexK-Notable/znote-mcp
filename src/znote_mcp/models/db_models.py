"""SQLAlchemy database models for the Zettelkasten MCP server."""
import datetime
from typing import List, Optional

from sqlalchemy import (Column, DateTime, ForeignKey, Integer, String, Table,
                       Text, UniqueConstraint, create_engine)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Mapped, Session, declarative_base, relationship, sessionmaker
from sqlalchemy.pool import QueuePool

from znote_mcp.config import config
from znote_mcp.models.schema import LinkType, NoteType

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
    note_type = Column(String(50), default=NoteType.PERMANENT.value, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.datetime.now, nullable=False)
    updated_at = Column(DateTime, default=datetime.datetime.now, nullable=False)
    project = Column(String(255), default="general", nullable=False, index=True)

    # Relationships
    tags = relationship(
        "DBTag", secondary=note_tags, back_populates="notes"
    )
    outgoing_links = relationship(
        "DBLink", 
        foreign_keys="DBLink.source_id",
        back_populates="source",
        cascade="all, delete-orphan"
    )
    incoming_links = relationship(
        "DBLink", 
        foreign_keys="DBLink.target_id",
        back_populates="target",
        cascade="all, delete-orphan"
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
    notes = relationship(
        "DBNote", secondary=note_tags, back_populates="tags"
    )
    
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
    created_at = Column(DateTime, default=datetime.datetime.now, nullable=False)
    
    # Relationships
    source = relationship(
        "DBNote", foreign_keys=[source_id], back_populates="outgoing_links"
    )
    target = relationship(
        "DBNote", foreign_keys=[target_id], back_populates="incoming_links"
    )
    
    # Add a unique constraint to prevent duplicate links of the same type
    __table_args__ = (
        UniqueConstraint('source_id', 'target_id', 'link_type', 
                         name='unique_link_type'),
    )
    
    def __repr__(self) -> str:
        """Return string representation of link."""
        return (
            f"<Link(id={self.id}, source='{self.source_id}', "
            f"target='{self.target_id}', type='{self.link_type}')>"
        )

def init_db() -> None:
    """Initialize the database with hardened configuration.

    Applies SQLite best practices for crash resilience:
    - WAL (Write-Ahead Logging) mode for atomic writes
    - NORMAL synchronous mode (good balance of safety vs speed)
    - QueuePool for connection reuse with size limits
    - Pool pre-ping to detect stale connections
    - Connection timeout to prevent deadlocks
    """
    from sqlalchemy import text, event

    # Create engine with connection pooling optimized for SQLite
    # SQLite is single-writer, so a small pool is ideal
    engine = create_engine(
        config.get_db_url(),
        poolclass=QueuePool,
        pool_size=5,           # Base pool size (concurrent reads)
        max_overflow=10,       # Allow up to 15 total connections under load
        pool_timeout=30,       # Wait up to 30s for a connection
        pool_recycle=3600,     # Recycle connections after 1 hour
        pool_pre_ping=True,    # Validate connections before use
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

    # Run migrations for schema updates
    _migrate_add_project_column(engine)

    # Create FTS5 virtual table for full-text search
    init_fts5(engine)

    return engine


def _migrate_add_project_column(engine) -> None:
    """Migration: Add project column to existing databases.

    SQLite doesn't support IF NOT EXISTS for ADD COLUMN, so we check
    the schema first. This is idempotent and safe to run multiple times.
    """
    from sqlalchemy import text, inspect

    inspector = inspect(engine)
    columns = [col['name'] for col in inspector.get_columns('notes')]

    if 'project' not in columns:
        with engine.connect() as conn:
            conn.execute(text(
                "ALTER TABLE notes ADD COLUMN project VARCHAR(255) "
                "NOT NULL DEFAULT 'general'"
            ))
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
        conn.execute(text("""
            CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(
                id UNINDEXED,
                title,
                content,
                content='notes',
                content_rowid='rowid'
            )
        """))

        # Create triggers to keep FTS in sync with notes table
        # Note: We need to handle inserts, updates, and deletes

        # Trigger for INSERT - add to FTS
        conn.execute(text("""
            CREATE TRIGGER IF NOT EXISTS notes_ai AFTER INSERT ON notes BEGIN
                INSERT INTO notes_fts(rowid, id, title, content)
                VALUES (NEW.rowid, NEW.id, NEW.title, NEW.content);
            END
        """))

        # Trigger for DELETE - remove from FTS
        conn.execute(text("""
            CREATE TRIGGER IF NOT EXISTS notes_ad AFTER DELETE ON notes BEGIN
                INSERT INTO notes_fts(notes_fts, rowid, id, title, content)
                VALUES ('delete', OLD.rowid, OLD.id, OLD.title, OLD.content);
            END
        """))

        # Trigger for UPDATE - update FTS
        conn.execute(text("""
            CREATE TRIGGER IF NOT EXISTS notes_au AFTER UPDATE ON notes BEGIN
                INSERT INTO notes_fts(notes_fts, rowid, id, title, content)
                VALUES ('delete', OLD.rowid, OLD.id, OLD.title, OLD.content);
                INSERT INTO notes_fts(rowid, id, title, content)
                VALUES (NEW.rowid, NEW.id, NEW.title, NEW.content);
            END
        """))

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
        result = conn.execute(text("""
            INSERT INTO notes_fts(rowid, id, title, content)
            SELECT rowid, id, title, content FROM notes
        """))

        conn.commit()

        # Count indexed notes
        count_result = conn.execute(text("SELECT COUNT(*) FROM notes_fts"))
        count = count_result.scalar()

    return count


def get_session_factory(engine=None):
    """Get a session factory for the database."""
    if engine is None:
        engine = create_engine(config.get_db_url())
    return sessionmaker(bind=engine)
