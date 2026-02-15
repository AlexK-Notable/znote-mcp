"""Service layer for Zettelkasten operations."""

import datetime
import hashlib
import logging
from datetime import timezone
from typing import Any, Dict, List, Optional, Tuple, Union

from znote_mcp.config import config
from znote_mcp.exceptions import (
    ErrorCode,
    NoteNotFoundError,
    NoteValidationError,
    ValidationError,
)
from znote_mcp.models.schema import (
    ConflictResult,
    LinkType,
    Note,
    NotePurpose,
    NoteType,
    Tag,
    VersionedNote,
    VersionInfo,
)
from znote_mcp.storage.note_repository import NoteRepository

logger = logging.getLogger(__name__)

# Keywords for auto-purpose inference (checked against title, tags, and first 200 chars of content)
_PURPOSE_KEYWORDS: Dict[NotePurpose, List[str]] = {
    NotePurpose.BUGFIXING: [
        "bug",
        "fix",
        "debug",
        "error",
        "issue",
        "crash",
        "broken",
        "regression",
        "patch",
        "hotfix",
        "traceback",
        "exception",
        "stack trace",
        "segfault",
        "failing",
        "failure",
    ],
    NotePurpose.PLANNING: [
        "plan",
        "design",
        "architect",
        "proposal",
        "rfc",
        "spec",
        "blueprint",
        "roadmap",
        "milestone",
        "phase",
        "implementation plan",
        "sprint",
        "epic",
        "strategy",
        "scope",
    ],
    NotePurpose.RESEARCH: [
        "research",
        "analysis",
        "investigation",
        "study",
        "exploration",
        "comparison",
        "evaluate",
        "assessment",
        "survey",
        "literature review",
        "findings",
        "benchmark",
        "experiment",
        "hypothesis",
    ],
}


def _infer_purpose(
    title: str, content: str, tags: Optional[List[str]] = None
) -> NotePurpose:
    """Infer note purpose from title, content, and tags using keyword matching.

    Only the first 200 characters of content are checked to keep inference fast
    and focused on the note's introductory framing rather than deep body text.

    Args:
        title: The note title.
        content: The note content (only first 200 chars examined).
        tags: Optional list of tag names.

    Returns:
        Inferred NotePurpose, or GENERAL if no strong signal found.
    """
    # Build search text: title + tags + content prefix (all lowercased)
    search_parts = [title.lower()]
    if tags:
        search_parts.extend(t.lower() for t in tags)
    search_parts.append(content[:200].lower())
    search_text = " ".join(search_parts)

    # Score each purpose by keyword matches
    best_purpose = NotePurpose.GENERAL
    best_score = 0

    for purpose, keywords in _PURPOSE_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in search_text)
        if score > best_score:
            best_score = score
            best_purpose = purpose

    return best_purpose


class ZettelService:
    """Service for managing Zettelkasten notes."""

    def __init__(
        self,
        repository: Optional[NoteRepository] = None,
        embedding_service: Optional["EmbeddingService"] = None,
        engine: Optional[Any] = None,
    ):
        """Initialize the service.

        Args:
            repository: Note storage backend. Created with defaults if None.
            embedding_service: Optional embedding service for semantic search.
                When provided and config.embeddings_enabled is True, notes
                are automatically embedded on create/update and embeddings
                are cleaned up on delete.
            engine: Pre-configured SQLAlchemy engine to pass to NoteRepository.
                Only used when repository is None.
        """
        if repository is not None:
            self.repository = repository
        elif engine is not None:
            self.repository = NoteRepository(engine=engine)
        else:
            self.repository = NoteRepository()
        self._embedding_service = embedding_service

    # =========================================================================
    # Embedding Helpers (fire-and-forget — never fail the main operation)
    # =========================================================================

    @staticmethod
    def _content_hash(title: str, content: str) -> str:
        """Compute a deterministic SHA-256 hash of embeddable content.

        Used for change detection: if the hash hasn't changed since the last
        embedding, we skip re-embedding to save compute.  Hash is computed
        on the whole note (not per-chunk) so any edit triggers re-embedding.

        Args:
            title: Note title.
            content: Note body content.

        Returns:
            Hex digest string (64 characters).
        """
        return hashlib.sha256(f"{title}\n{content}".encode("utf-8")).hexdigest()

    @staticmethod
    def _prepare_embedding_text(title: str, content: str) -> str:
        """Combine title and content into the text that gets embedded.

        Both single-vector and chunked paths use this to ensure consistent
        text preparation.
        """
        return f"{title}\n{content}"

    def _embed_note(self, note: Note) -> None:
        """Embed a note and store the vector(s).  Fire-and-forget.

        Short notes (under embedding_chunk_size tokens) get a single vector.
        Long notes are split into overlapping chunks, each embedded separately.

        Skips embedding if:
        - No embedding service is configured
        - embeddings_enabled is False
        - The content hash hasn't changed since the last embedding
        - sqlite-vec is unavailable in the repository

        Errors are logged but never propagated — embedding failures must
        not break normal CRUD operations.
        """
        if self._embedding_service is None or not config.embeddings_enabled:
            return

        try:
            content_hash = self._content_hash(note.title, note.content)

            # Check if we already have an up-to-date embedding
            meta = self.repository.get_embedding_metadata(note.id)
            if meta and meta["content_hash"] == content_hash:
                logger.debug(f"Embedding unchanged for note {note.id}, skipping")
                return

            text = self._prepare_embedding_text(note.title, note.content)

            # Decide: single vector or chunked
            from znote_mcp.services.text_chunker import TextChunker

            chunker = TextChunker(
                chunk_size=config.embedding_chunk_size,
                chunk_overlap=config.embedding_chunk_overlap,
            )
            chunks = chunker.chunk(text)

            if len(chunks) <= 1:
                # Single vector path (most notes)
                vector = self._embedding_service.embed(text)
                self.repository.store_embedding(
                    note_id=note.id,
                    embedding=vector,
                    model_name=config.embedding_model,
                    content_hash=content_hash,
                )
                logger.debug(f"Embedded note {note.id} (single vector)")
            else:
                self._embed_note_chunked(note.id, chunks, content_hash)
        except Exception as e:
            logger.warning(f"Failed to embed note {note.id}: {e}")

    def _embed_note_chunked(
        self,
        note_id: str,
        chunks: list,
        content_hash: str,
    ) -> None:
        """Embed multiple chunks for a single note and store them.

        Args:
            note_id: The note ID.
            chunks: List of TextChunk instances from TextChunker.
            content_hash: SHA-256 digest of the whole note content.
        """
        # Embed each chunk
        chunk_texts = [c.text for c in chunks]
        vectors = self._embedding_service.embed_batch(
            chunk_texts, batch_size=config.embedding_batch_size
        )

        # Build (chunk_index, vector) pairs
        chunk_data = list(zip([c.index for c in chunks], vectors))

        self.repository.store_chunk_embeddings(
            note_id=note_id,
            chunks=chunk_data,
            model_name=config.embedding_model,
            content_hash=content_hash,
        )
        logger.debug(f"Embedded note {note_id} ({len(chunks)} chunks)")

    def _delete_embedding(self, note_id: str) -> None:
        """Delete all embedding chunks for a note.  Fire-and-forget."""
        if self._embedding_service is None or not config.embeddings_enabled:
            return
        try:
            self.repository.delete_embedding(note_id)
        except Exception as e:
            logger.warning(f"Failed to delete embedding for note {note_id}: {e}")

    def shutdown(self) -> None:
        """Shut down the service and release resources."""
        if self._embedding_service is not None:
            self._embedding_service.shutdown()
            logger.info("Embedding service shut down")

    def reindex_embeddings(self) -> Dict[str, int]:
        """Rebuild all note embeddings from scratch.

        Clears existing embeddings, then re-embeds every note.  Short notes
        get a single vector; long notes are split into overlapping chunks.

        Returns:
            Dict with keys: total, embedded, skipped, failed, chunks.

        Raises:
            EmbeddingError: If no embedding service is configured.
        """
        from znote_mcp.exceptions import EmbeddingError
        from znote_mcp.services.text_chunker import TextChunker

        if self._embedding_service is None:
            raise EmbeddingError(
                "No embedding service configured",
                code=ErrorCode.EMBEDDING_UNAVAILABLE,
                operation="reindex_embeddings",
            )

        stats = {"total": 0, "embedded": 0, "skipped": 0, "failed": 0, "chunks": 0}

        # Clear existing embeddings
        cleared = self.repository.clear_all_embeddings()
        logger.info(f"Cleared {cleared} existing embeddings")

        # Get all notes
        all_notes = self.repository.get_all()
        stats["total"] = len(all_notes)

        if not all_notes:
            return stats

        chunker = TextChunker(
            chunk_size=config.embedding_chunk_size,
            chunk_overlap=config.embedding_chunk_overlap,
        )

        # Separate notes into short (single-vector) and long (chunked)
        short_notes = []
        short_texts = []
        long_notes = []  # (note, chunks_list)

        for note in all_notes:
            text = self._prepare_embedding_text(note.title, note.content)
            chunks = chunker.chunk(text)
            if len(chunks) <= 1:
                short_notes.append(note)
                short_texts.append(text)
            else:
                long_notes.append((note, chunks))

        # Batch embed short notes
        if short_texts:
            try:
                vectors = self._embedding_service.embed_batch(
                    short_texts, batch_size=config.embedding_batch_size
                )
                batch = []
                for note, vector in zip(short_notes, vectors):
                    try:
                        content_hash = self._content_hash(note.title, note.content)
                        batch.append(
                            (note.id, vector, config.embedding_model, content_hash)
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to prepare embedding for note {note.id}: {e}"
                        )
                        stats["failed"] += 1

                if batch:
                    stored = self.repository.store_embeddings_batch(batch)
                    stats["embedded"] += stored
                    stats["chunks"] += stored
            except Exception as e:
                logger.error(f"Batch embedding failed for short notes: {e}")
                stats["failed"] += len(short_notes)

        # Embed long notes (chunked)
        for note, chunks in long_notes:
            try:
                content_hash = self._content_hash(note.title, note.content)
                self._embed_note_chunked(note.id, chunks, content_hash)
                stats["embedded"] += 1
                stats["chunks"] += len(chunks)
            except Exception as e:
                logger.warning(f"Failed to embed chunked note {note.id}: {e}")
                stats["failed"] += 1

        logger.info(
            f"Reindex complete: {stats['embedded']} notes embedded "
            f"({stats['chunks']} chunks), "
            f"{stats['skipped']} skipped, {stats['failed']} failed "
            f"out of {stats['total']} total"
        )
        return stats

    def create_note(
        self,
        title: str,
        content: str,
        note_type: NoteType = NoteType.PERMANENT,
        project: str = "general",
        note_purpose: NotePurpose = NotePurpose.GENERAL,
        tags: Optional[List[str]] = None,
        plan_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Note:
        """Create a new note.

        Args:
            title: Note title (required).
            content: Note content (required).
            note_type: Zettelkasten note type.
            project: Project this note belongs to.
            note_purpose: Workflow purpose (research, planning, bugfixing, general).
            tags: List of tag names.
            plan_id: Optional ID of associated plan/task.
            metadata: Additional metadata dict.

        Returns:
            Created Note object.
        """
        if not title:
            raise NoteValidationError(
                "Title is required", field="title", code=ErrorCode.NOTE_TITLE_REQUIRED
            )
        if not content:
            raise NoteValidationError(
                "Content is required",
                field="content",
                code=ErrorCode.NOTE_CONTENT_REQUIRED,
            )

        # Auto-infer purpose when left at default
        if note_purpose == NotePurpose.GENERAL:
            inferred = _infer_purpose(title, content, tags)
            if inferred != NotePurpose.GENERAL:
                note_purpose = inferred
                logger.debug(
                    f"Auto-inferred purpose '{note_purpose.value}' for note '{title}'"
                )

        # Create note object
        note = Note(
            title=title,
            content=content,
            note_type=note_type,
            project=project,
            note_purpose=note_purpose,
            tags=[Tag(name=tag) for tag in (tags or [])],
            plan_id=plan_id,
            metadata=metadata or {},
        )

        # Save to repository
        created = self.repository.create(note)

        # Auto-embed (fire-and-forget)
        self._embed_note(created)

        return created

    def get_note(self, note_id: str) -> Optional[Note]:
        """Retrieve a note by ID."""
        return self.repository.get(note_id)

    def get_note_by_title(self, title: str) -> Optional[Note]:
        """Retrieve a note by title."""
        return self.repository.get_by_title(title)

    def update_note(
        self,
        note_id: str,
        title: Optional[str] = None,
        content: Optional[str] = None,
        note_type: Optional[NoteType] = None,
        project: Optional[str] = None,
        note_purpose: Optional[NotePurpose] = None,
        tags: Optional[List[str]] = None,
        plan_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Note:
        """Update an existing note.

        Args:
            note_id: ID of the note to update.
            title: New title (optional).
            content: New content (optional).
            note_type: New Zettelkasten type (optional).
            project: New project (optional).
            note_purpose: New purpose (optional).
            tags: New list of tag names (optional).
            plan_id: New plan ID (optional, use empty string to clear).
            metadata: New metadata dict (optional).

        Returns:
            Updated Note object.
        """
        note = self.repository.get(note_id)
        if not note:
            raise NoteNotFoundError(note_id)

        # Update fields
        if title is not None:
            note.title = title
        if content is not None:
            note.content = content
        if note_type is not None:
            note.note_type = note_type
        if project is not None:
            note.project = project
        if note_purpose is not None:
            note.note_purpose = note_purpose
        if tags is not None:
            note.tags = [Tag(name=tag) for tag in tags]
        if plan_id is not None:
            note.plan_id = plan_id if plan_id else None  # Empty string clears
        if metadata is not None:
            note.metadata = metadata

        note.updated_at = datetime.datetime.now(timezone.utc)

        # Save to repository
        updated = self.repository.update(note)

        # Re-embed if content changed (fire-and-forget)
        self._embed_note(updated)

        return updated

    def delete_note(self, note_id: str) -> None:
        """Delete a note."""
        self.repository.delete(note_id)
        self._delete_embedding(note_id)

    def get_all_notes(self, limit: Optional[int] = None, offset: int = 0) -> List[Note]:
        """Get all notes with optional pagination.

        Args:
            limit: Maximum number of notes to return. None for all notes.
            offset: Number of notes to skip (for pagination).

        Returns:
            List of Note objects, ordered by creation date (newest first).
        """
        return self.repository.get_all(limit=limit, offset=offset)

    def count_notes(self) -> int:
        """Get total count of notes in the repository.

        Useful for pagination when using get_all_notes() with limit/offset.

        Returns:
            Total number of notes.
        """
        return self.repository.count_notes()

    def get_notes_by_project(self, project: str) -> List[Note]:
        """Get all notes for a specific project using SQL-level filtering.

        Args:
            project: The project name to filter by.

        Returns:
            List of Note objects belonging to the specified project.
        """
        return self.repository.get_by_project(project)

    def search_notes(self, **kwargs: Any) -> List[Note]:
        """Search for notes based on criteria."""
        return self.repository.search(**kwargs)

    def count_search_results(self, **kwargs: Any) -> int:
        """Count notes matching search criteria without loading them.

        Useful for pagination UI when using search_notes() with limit/offset.

        Args:
            **kwargs: Same search criteria as search_notes().

        Returns:
            Total count of matching notes.
        """
        return self.repository.count_search_results(**kwargs)

    def get_notes_by_tag(self, tag: str) -> List[Note]:
        """Get notes by tag."""
        return self.repository.find_by_tag(tag)

    def add_tag_to_note(self, note_id: str, tag: str) -> Note:
        """Add a tag to a note."""
        note = self.repository.get(note_id)
        if not note:
            raise NoteNotFoundError(note_id)
        note.add_tag(tag)
        return self.repository.update(note)

    def remove_tag_from_note(self, note_id: str, tag: str) -> Note:
        """Remove a tag from a note."""
        note = self.repository.get(note_id)
        if not note:
            raise NoteNotFoundError(note_id)
        note.remove_tag(tag)
        return self.repository.update(note)

    def get_all_tags(self) -> List[Tag]:
        """Get all tags in the system."""
        return self.repository.get_all_tags()

    def get_tags_with_counts(self) -> Dict[str, int]:
        """Get all tags with their usage counts.

        Returns:
            Dictionary mapping tag names to their note counts.
        """
        return self.repository.get_tags_with_counts()

    def delete_unused_tags(self) -> int:
        """Delete tags that are not associated with any notes.

        Returns:
            Number of tags deleted.
        """
        return self.repository.delete_unused_tags()

    def get_note_history(self, note_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get version history for a note.

        Args:
            note_id: The note ID
            limit: Maximum number of versions to return

        Returns:
            List of version info dictionaries, most recent first.
            Empty list if git versioning is not enabled.
        """
        return self.repository.get_note_history(note_id, limit)

    def create_link(
        self,
        source_id: str,
        target_id: str,
        link_type: LinkType = LinkType.REFERENCE,
        description: Optional[str] = None,
        bidirectional: bool = False,
        bidirectional_type: Optional[LinkType] = None,
    ) -> Tuple[Note, Optional[Note]]:
        """Create a link between notes with proper bidirectional semantics.

        Args:
            source_id: ID of the source note
            target_id: ID of the target note
            link_type: Type of link from source to target
            description: Optional description of the link
            bidirectional: Whether to create a link in both directions
            bidirectional_type: Optional custom link type for the reverse direction
                If not provided, an appropriate inverse relation will be used

        Returns:
            Tuple of (source_note, target_note or None)
        """
        source_note = self.repository.get(source_id)
        if not source_note:
            raise NoteNotFoundError(
                source_id, f"Source note with ID '{source_id}' not found"
            )
        target_note = self.repository.get(target_id)
        if not target_note:
            raise NoteNotFoundError(
                target_id, f"Target note with ID '{target_id}' not found"
            )

        # Check if this link already exists before attempting to add it
        for link in source_note.links:
            if link.target_id == target_id and link.link_type == link_type:
                # Link already exists, no need to add it again
                if not bidirectional:
                    return source_note, None
                break
        else:
            # Only add the link if it doesn't exist
            source_note.add_link(target_id, link_type, description)
            source_note = self.repository.update(source_note)

        # If bidirectional, add link from target to source with appropriate semantics
        reverse_note = None
        if bidirectional:
            # If no explicit bidirectional type is provided, determine appropriate inverse
            if bidirectional_type is None:
                # Map link types to their semantic inverses
                inverse_map = {
                    LinkType.REFERENCE: LinkType.REFERENCE,
                    LinkType.EXTENDS: LinkType.EXTENDED_BY,
                    LinkType.EXTENDED_BY: LinkType.EXTENDS,
                    LinkType.REFINES: LinkType.REFINED_BY,
                    LinkType.REFINED_BY: LinkType.REFINES,
                    LinkType.CONTRADICTS: LinkType.CONTRADICTED_BY,
                    LinkType.CONTRADICTED_BY: LinkType.CONTRADICTS,
                    LinkType.QUESTIONS: LinkType.QUESTIONED_BY,
                    LinkType.QUESTIONED_BY: LinkType.QUESTIONS,
                    LinkType.SUPPORTS: LinkType.SUPPORTED_BY,
                    LinkType.SUPPORTED_BY: LinkType.SUPPORTS,
                    LinkType.RELATED: LinkType.RELATED,
                }
                bidirectional_type = inverse_map.get(link_type, link_type)

            # Check if the reverse link already exists before adding it
            for link in target_note.links:
                if link.target_id == source_id and link.link_type == bidirectional_type:
                    # Reverse link already exists, no need to add it again
                    return source_note, target_note

            # Only add the reverse link if it doesn't exist
            target_note.add_link(source_id, bidirectional_type, description)
            reverse_note = self.repository.update(target_note)

        return source_note, reverse_note

    def remove_link(
        self,
        source_id: str,
        target_id: str,
        link_type: Optional[LinkType] = None,
        bidirectional: bool = False,
    ) -> Tuple[Note, Optional[Note]]:
        """Remove a link between notes."""
        source_note = self.repository.get(source_id)
        if not source_note:
            raise NoteNotFoundError(
                source_id, f"Source note with ID '{source_id}' not found"
            )

        # Remove link from source to target
        source_note.remove_link(target_id, link_type)
        source_note = self.repository.update(source_note)

        # If bidirectional, remove link from target to source
        reverse_note = None
        if bidirectional:
            target_note = self.repository.get(target_id)
            if target_note:
                target_note.remove_link(source_id, link_type)
                reverse_note = self.repository.update(target_note)

        return source_note, reverse_note

    def get_linked_notes(self, note_id: str, direction: str = "outgoing") -> List[Note]:
        """Get notes linked to/from a note."""
        note = self.repository.get(note_id)
        if not note:
            raise NoteNotFoundError(note_id)
        return self.repository.find_linked_notes(note_id, direction)

    def rebuild_index(self) -> None:
        """Rebuild the database index from files."""
        self.repository.rebuild_index()

    def export_note(self, note_id: str, format: str = "markdown") -> str:
        """Export a note in the specified format."""
        note = self.repository.get(note_id)
        if not note:
            raise NoteNotFoundError(note_id)

        if format.lower() == "markdown":
            return note.to_markdown()
        else:
            raise ValidationError(
                f"Unsupported export format: {format}", field="format", value=format
            )

    def sync_to_obsidian(self) -> int:
        """Sync all notes to the Obsidian vault.

        Returns:
            Number of notes synced.

        Raises:
            ValueError: If Obsidian vault is not configured.
        """
        return self.repository.sync_to_obsidian()

    def fts_search(
        self, query: str, limit: int = 50, highlight: bool = False
    ) -> List[Dict[str, Any]]:
        """Full-text search using FTS5.

        Args:
            query: Search query (supports FTS5 syntax).
            limit: Maximum results to return.
            highlight: Include highlighted snippets.

        Returns:
            List of search results with id, title, rank, and optionally snippet.
        """
        return self.repository.fts_search(query, limit, highlight)

    def has_fts(self) -> bool:
        """Check whether FTS5 full-text search is available."""
        return getattr(self.repository, "_fts_available", False)

    def get_notes_by_ids(self, ids: List[str]) -> List[Note]:
        """Retrieve multiple notes by their IDs in a batch."""
        return self.repository.get_by_ids(ids)

    def find_orphaned_note_ids(self) -> List[str]:
        """Find IDs of notes with no incoming or outgoing links."""
        return self.repository.find_orphaned_note_ids()

    def find_central_note_ids_with_counts(
        self, limit: int = 10
    ) -> List[Tuple[str, int]]:
        """Find note IDs with the most connections.

        Returns:
            List of (note_id, connection_count) tuples, sorted by count desc.
        """
        return self.repository.find_central_note_ids_with_counts(limit)

    def vec_similarity_search(
        self,
        query_vector: List[float],
        limit: int = 10,
        exclude_ids: Optional[List[str]] = None,
    ) -> List[Tuple[str, float]]:
        """KNN vector search using sqlite-vec."""
        return self.repository.vec_similarity_search(
            query_vector,
            limit=limit,
            exclude_ids=exclude_ids,
        )

    def get_embedding(self, note_id: str) -> Optional[List[float]]:
        """Retrieve the stored embedding vector for a note."""
        return self.repository.get_embedding(note_id)

    def rebuild_fts(self) -> int:
        """Rebuild the FTS5 index.

        Returns:
            Number of notes indexed.
        """
        return self.repository.rebuild_fts()

    def check_database_health(self) -> Dict[str, Any]:
        """Perform comprehensive database health check.

        Returns:
            Dict with keys:
                - healthy: bool indicating overall health
                - sqlite_ok: bool for SQLite integrity
                - fts_ok: bool for FTS5 integrity
                - note_count: int of notes in database
                - file_count: int of markdown files
                - issues: list of issue descriptions
        """
        return self.repository.check_database_health()

    def reset_fts_availability(self) -> bool:
        """Reset FTS5 availability after manual repair.

        Call this if FTS5 was disabled due to corruption and you've
        manually repaired the database.

        Returns:
            True if FTS5 is now available, False if still broken.
        """
        return self.repository.reset_fts_availability()

    def find_similar_notes(
        self, note_id: str, threshold: float = 0.5
    ) -> List[Tuple[Note, float]]:
        """Find notes similar to the given note based on shared tags and links.

        Uses SQL-optimized candidate selection to avoid O(N) file reads.
        Only notes with at least one shared tag or direct link are considered.
        """
        note = self.repository.get(note_id)
        if not note:
            raise NoteNotFoundError(note_id)

        # Get candidates with shared tags or direct links (SQL-optimized)
        candidates = self.repository.find_similarity_candidates(note_id)
        results = []

        # Set of this note's tags and links
        note_tags = {tag.name for tag in note.tags}
        note_links = {link.target_id for link in note.links}

        # Build set of note IDs that link to this note
        incoming_notes = self.repository.find_linked_notes(note_id, "incoming")
        note_incoming = {n.id for n in incoming_notes}

        # For each candidate, calculate similarity
        for other_note in candidates:
            if other_note.id == note_id:
                continue

            # Calculate tag overlap
            other_tags = {tag.name for tag in other_note.tags}
            tag_overlap = len(note_tags.intersection(other_tags))

            # Calculate link overlap (outgoing)
            other_links = {link.target_id for link in other_note.links}
            link_overlap = len(note_links.intersection(other_links))

            # Check if other note links to this note
            incoming_overlap = 1 if other_note.id in note_incoming else 0

            # Check if this note links to other note
            outgoing_overlap = 1 if other_note.id in note_links else 0

            # Calculate similarity score
            # Weight: 40% tags, 20% outgoing links, 20% incoming links, 20% direct connections
            total_possible = (
                max(len(note_tags), len(other_tags)) * 0.4
                + max(len(note_links), len(other_links)) * 0.2
                + 1 * 0.2  # Possible incoming link
                + 1 * 0.2  # Possible outgoing link
            )

            # Avoid division by zero
            if total_possible == 0:
                similarity = 0.0
            else:
                similarity = (
                    (tag_overlap * 0.4)
                    + (link_overlap * 0.2)
                    + (incoming_overlap * 0.2)
                    + (outgoing_overlap * 0.2)
                ) / total_possible

            if similarity >= threshold:
                results.append((other_note, similarity))

        # Sort by similarity (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    # ========== Bulk Operations ==========

    def bulk_create_notes(self, notes_data: List[Dict[str, Any]]) -> List[Note]:
        """Create multiple notes in a single batch operation.

        Args:
            notes_data: List of dictionaries with note data.
                Each dict should have: title, content, note_type (optional),
                project (optional), tags (optional list of strings).

        Returns:
            List of created Note objects.

        Raises:
            NoteValidationError: If any note fails validation.
        """
        notes = []
        for data in notes_data:
            if not data.get("title"):
                raise NoteValidationError(
                    "Title is required for all notes",
                    field="title",
                    code=ErrorCode.NOTE_TITLE_REQUIRED,
                )
            if not data.get("content"):
                raise NoteValidationError(
                    "Content is required for all notes",
                    field="content",
                    code=ErrorCode.NOTE_CONTENT_REQUIRED,
                )

            note_type = data.get("note_type", NoteType.PERMANENT)
            if isinstance(note_type, str):
                note_type = NoteType(note_type)

            # Extract note_purpose (default to GENERAL)
            note_purpose = data.get("note_purpose", NotePurpose.GENERAL)
            if isinstance(note_purpose, str):
                note_purpose = NotePurpose(note_purpose)

            # Auto-infer purpose when left at default
            if note_purpose == NotePurpose.GENERAL:
                inferred = _infer_purpose(
                    data["title"], data["content"], data.get("tags")
                )
                if inferred != NotePurpose.GENERAL:
                    note_purpose = inferred

            note = Note(
                title=data["title"],
                content=data["content"],
                note_type=note_type,
                note_purpose=note_purpose,
                project=data.get("project", "general"),
                plan_id=data.get("plan_id"),
                tags=[Tag(name=tag) for tag in data.get("tags", [])],
                metadata=data.get("metadata", {}),
            )
            notes.append(note)

        created = self.repository.bulk_create_notes(notes)

        # Auto-embed all created notes (fire-and-forget per note)
        for note in created:
            self._embed_note(note)

        return created

    def bulk_delete_notes(self, note_ids: List[str]) -> int:
        """Delete multiple notes in a single batch operation.

        Args:
            note_ids: List of note IDs to delete.

        Returns:
            Number of notes successfully deleted.
        """
        count = self.repository.bulk_delete_notes(note_ids)

        # Clean up embeddings (fire-and-forget per note)
        for nid in note_ids:
            self._delete_embedding(nid)

        return count

    def bulk_add_tags(self, note_ids: List[str], tags: List[str]) -> int:
        """Add tags to multiple notes.

        Args:
            note_ids: List of note IDs to update.
            tags: List of tag names to add.

        Returns:
            Number of notes successfully updated.
        """
        return self.repository.bulk_add_tags(note_ids, tags)

    def bulk_remove_tags(self, note_ids: List[str], tags: List[str]) -> int:
        """Remove tags from multiple notes.

        Args:
            note_ids: List of note IDs to update.
            tags: List of tag names to remove.

        Returns:
            Number of notes successfully updated.
        """
        return self.repository.bulk_remove_tags(note_ids, tags)

    def bulk_update_project(self, note_ids: List[str], project: str) -> int:
        """Move multiple notes to a different project.

        Args:
            note_ids: List of note IDs to update.
            project: Target project name.

        Returns:
            Number of notes successfully updated.
        """
        return self.repository.bulk_update_project(note_ids, project)

    # =========================================================================
    # Migration Methods
    # =========================================================================

    def migrate_notes_add_purpose(
        self, default_purpose: NotePurpose = NotePurpose.GENERAL
    ) -> Dict[str, Any]:
        """Migrate existing notes to include note_purpose field.

        This checks the actual markdown file for the 'purpose:' key and adds it
        if missing. Safe to run multiple times (idempotent).

        Args:
            default_purpose: Default purpose to assign to notes missing it.

        Returns:
            Dict with migration stats: {migrated: int, skipped: int, errors: []}
        """
        stats = {"migrated": 0, "skipped": 0, "errors": []}
        all_notes = self.repository.get_all()

        for note in all_notes:
            try:
                # Check the actual markdown file to see if 'purpose:' exists
                note_path = self.repository.notes_dir / f"{note.id}.md"
                if not note_path.exists():
                    stats["errors"].append(
                        {"id": note.id, "error": "Markdown file not found"}
                    )
                    continue

                content = note_path.read_text(encoding="utf-8")

                # Check if 'purpose:' exists in the frontmatter
                # Frontmatter is between --- markers
                if content.startswith("---"):
                    end_marker = content.find("---", 3)
                    if end_marker > 0:
                        frontmatter = content[3:end_marker]
                        if "purpose:" in frontmatter:
                            stats["skipped"] += 1
                            continue

                # Need to migrate - set purpose and re-save
                note.note_purpose = default_purpose
                self.repository.update(note)
                stats["migrated"] += 1

            except Exception as e:
                stats["errors"].append({"id": note.id, "error": str(e)})
                logger.warning(f"Failed to migrate note {note.id}: {e}")

        logger.info(
            f"Migration complete: {stats['migrated']} migrated, "
            f"{stats['skipped']} skipped, {len(stats['errors'])} errors"
        )
        return stats

    # =========================================================================
    # Versioned CRUD Operations (with git conflict detection)
    # =========================================================================

    def get_note_versioned(self, note_id: str) -> Optional[VersionedNote]:
        """Retrieve a note by ID with its version information.

        Args:
            note_id: The note ID.

        Returns:
            VersionedNote if found, None otherwise.
        """
        return self.repository.get_versioned(note_id)

    def create_note_versioned(
        self,
        title: str,
        content: str,
        note_type: NoteType = NoteType.PERMANENT,
        note_purpose: NotePurpose = NotePurpose.GENERAL,
        project: str = "general",
        plan_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> VersionedNote:
        """Create a new note with version tracking.

        Args:
            title: Note title.
            content: Note content.
            note_type: Type of note (default: PERMANENT).
            note_purpose: Workflow purpose (default: GENERAL).
            project: Project name (default: "general").
            plan_id: Optional plan ID for grouping related planning notes.
            tags: Optional list of tag names.
            metadata: Optional metadata dictionary.

        Returns:
            VersionedNote with the created note and its version info.

        Raises:
            NoteValidationError: If title or content is missing.
        """
        if not title:
            raise NoteValidationError(
                "Title is required", field="title", code=ErrorCode.NOTE_TITLE_REQUIRED
            )
        if not content:
            raise NoteValidationError(
                "Content is required",
                field="content",
                code=ErrorCode.NOTE_CONTENT_REQUIRED,
            )

        # Auto-infer purpose when left at default
        if note_purpose == NotePurpose.GENERAL:
            inferred = _infer_purpose(title, content, tags)
            if inferred != NotePurpose.GENERAL:
                note_purpose = inferred
                logger.debug(
                    f"Auto-inferred purpose '{note_purpose.value}' for note '{title}'"
                )

        # Create note object
        note = Note(
            title=title,
            content=content,
            note_type=note_type,
            note_purpose=note_purpose,
            project=project,
            plan_id=plan_id,
            tags=[Tag(name=tag) for tag in (tags or [])],
            metadata=metadata or {},
        )

        # Save with version tracking
        result = self.repository.create_versioned(note)

        # Auto-embed (fire-and-forget)
        self._embed_note(result.note)

        return result

    def update_note_versioned(
        self,
        note_id: str,
        title: Optional[str] = None,
        content: Optional[str] = None,
        note_type: Optional[NoteType] = None,
        project: Optional[str] = None,
        note_purpose: Optional[NotePurpose] = None,
        tags: Optional[List[str]] = None,
        plan_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        expected_version: Optional[str] = None,
    ) -> Union[VersionedNote, ConflictResult]:
        """Update an existing note with version conflict detection.

        If expected_version is provided and doesn't match the current version,
        returns a ConflictResult instead of updating.

        Args:
            note_id: The note ID to update.
            title: New title (optional).
            content: New content (optional).
            note_type: New note type (optional).
            project: New project (optional).
            note_purpose: New purpose (optional).
            tags: New tags list (optional).
            plan_id: New plan ID (optional, empty string clears).
            metadata: New metadata (optional).
            expected_version: Expected version hash for conflict detection.

        Returns:
            VersionedNote on success, ConflictResult if version conflict.

        Raises:
            NoteNotFoundError: If the note doesn't exist.
        """
        note = self.repository.get(note_id)
        if not note:
            raise NoteNotFoundError(note_id)

        # Update fields
        if title is not None:
            note.title = title
        if content is not None:
            note.content = content
        if note_type is not None:
            note.note_type = note_type
        if project is not None:
            note.project = project
        if note_purpose is not None:
            note.note_purpose = note_purpose
        if tags is not None:
            note.tags = [Tag(name=tag) for tag in tags]
        if plan_id is not None:
            note.plan_id = plan_id if plan_id else None  # Empty string clears
        if metadata is not None:
            note.metadata = metadata

        note.updated_at = datetime.datetime.now(timezone.utc)

        # Update with version checking
        result = self.repository.update_versioned(note, expected_version)

        # Re-embed if update succeeded (not a conflict)
        if isinstance(result, VersionedNote):
            self._embed_note(result.note)

        return result

    def delete_note_versioned(
        self, note_id: str, expected_version: Optional[str] = None
    ) -> Union[VersionInfo, ConflictResult]:
        """Delete a note with version conflict detection.

        If expected_version is provided and doesn't match the current version,
        returns a ConflictResult instead of deleting.

        Args:
            note_id: The note ID to delete.
            expected_version: Expected version hash for conflict detection.

        Returns:
            VersionInfo on success, ConflictResult if version conflict.
        """
        result = self.repository.delete_versioned(note_id, expected_version)

        # Clean up embedding if delete succeeded (not a conflict)
        if isinstance(result, VersionInfo):
            self._delete_embedding(note_id)

        return result
