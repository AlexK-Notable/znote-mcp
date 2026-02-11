"""MCP server implementation for the Zettelkasten."""

import atexit
import json
import logging
import uuid
from datetime import datetime
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

from znote_mcp.backup import backup_manager
from znote_mcp.config import config
from znote_mcp.exceptions import ValidationError, ZettelkastenError
from znote_mcp.models.schema import (
    ConflictResult,
    LinkType,
    NotePurpose,
    NoteType,
    Project,
)
from znote_mcp.observability import metrics, timed_operation
from znote_mcp.services.search_service import SearchService
from znote_mcp.services.zettel_service import ZettelService
from znote_mcp.storage.project_repository import ProjectRepository

logger = logging.getLogger(__name__)

MAX_TITLE_LENGTH = 500
MAX_CONTENT_LENGTH = 1_000_000  # 1 MB


def _validate_input_lengths(
    title: Optional[str] = None, content: Optional[str] = None
) -> None:
    """Validate input string lengths at the MCP boundary."""
    if title and len(title) > MAX_TITLE_LENGTH:
        raise ValueError(
            f"Title exceeds maximum length of {MAX_TITLE_LENGTH} characters"
        )
    if content and len(content) > MAX_CONTENT_LENGTH:
        raise ValueError(
            f"Content exceeds maximum length of {MAX_CONTENT_LENGTH} characters"
        )


class ZettelkastenMcpServer:
    """MCP server for Zettelkasten."""

    def __init__(self, engine=None):
        """Initialize the MCP server.

        Args:
            engine: Pre-configured SQLAlchemy engine. When provided, all
                    repositories share this single engine. When None, each
                    repository creates its own (legacy behavior).
        """
        self.mcp = FastMCP(config.server_name, version=config.server_version)
        # Conditionally create embedding service
        embedding_service = self._create_embedding_service()

        # Services — share a single database engine when provided
        self.zettel_service = ZettelService(
            embedding_service=embedding_service,
            engine=engine,
        )
        self.search_service = SearchService(
            self.zettel_service,
            embedding_service=embedding_service,
        )
        self.project_repository = ProjectRepository(engine=engine)
        # Initialize services
        self.initialize()
        # Register shutdown hook for resource cleanup
        atexit.register(self._shutdown)
        # Register tools
        self._register_tools()
        self._register_resources()
        self._register_prompts()

    def initialize(self) -> None:
        """Initialize services."""
        logger.info("Zettelkasten MCP server initialized")

    def _shutdown(self) -> None:
        """Clean up resources on server exit."""
        self.zettel_service.shutdown()

    def format_error_response(self, error: Exception) -> str:
        """Format an error response in a consistent way.

        Args:
            error: The exception that occurred

        Returns:
            Formatted error message with appropriate level of detail
        """
        # Generate a unique error ID for traceability in logs
        error_id = str(uuid.uuid4())[:8]

        if isinstance(error, ZettelkastenError):
            # Structured domain errors - use the error code and message
            logger.error(
                f"[{error.code.name}] [{error_id}]: {error.message}",
                extra={"error_details": error.details},
            )
            return f"Error: {error.message}"
        elif isinstance(error, ValueError):
            # Log full detail but return generic ref to avoid leaking internals
            logger.error(f"Validation error [{error_id}]: {str(error)}")
            return f"Error: Invalid input (ref: {error_id})"
        elif isinstance(error, (IOError, OSError)):
            # File system errors - don't expose paths or detailed error messages
            logger.error(f"File system error [{error_id}]: {str(error)}", exc_info=True)
            return f"Error: A file system error occurred (ref: {error_id})"
        else:
            # Unexpected errors - log with full stack trace but return generic message
            logger.error(f"Unexpected error [{error_id}]: {str(error)}", exc_info=True)
            return f"Error: An unexpected error occurred (ref: {error_id})"

    @staticmethod
    def _create_embedding_service():
        """Create an EmbeddingService if embeddings are enabled and deps are available.

        Returns None (gracefully) if:
        - config.embeddings_enabled is False
        - The [semantic] optional dependencies are not installed
        """
        if not config.embeddings_enabled:
            logger.info("Embeddings disabled (ZETTELKASTEN_EMBEDDINGS_ENABLED=false)")
            return None

        try:
            from znote_mcp.services.embedding_service import EmbeddingService
            from znote_mcp.services.onnx_providers import (
                OnnxEmbeddingProvider,
                OnnxRerankerProvider,
            )

            embedder = OnnxEmbeddingProvider(
                model_id=config.embedding_model,
                max_length=config.embedding_max_tokens,
                cache_dir=config.embedding_model_cache_dir,
            )
            reranker = OnnxRerankerProvider(
                model_id=config.reranker_model,
                max_length=config.embedding_max_tokens,
                cache_dir=config.embedding_model_cache_dir,
            )
            service = EmbeddingService(
                embedder=embedder,
                reranker=reranker,
                reranker_idle_timeout=config.reranker_idle_timeout,
            )
            logger.info(
                f"Embedding service created (model={config.embedding_model}, "
                f"dim={config.embedding_dim})"
            )
            return service
        except ImportError as e:
            logger.warning(
                f"Embedding dependencies not available: {e}. "
                "Install with: pip install znote-mcp[semantic]"
            )
            return None
        except Exception as e:
            logger.warning(f"Failed to create embedding service: {e}")
            return None

    def _register_tools(self) -> None:
        """Register MCP tools."""

        # Create a new note
        @self.mcp.tool(name="zk_create_note")
        def zk_create_note(
            title: str,
            content: str,
            note_type: str = "permanent",
            project: str = "general",
            note_purpose: str = "general",
            tags: Optional[str] = None,
            plan_id: Optional[str] = None,
        ) -> str:
            """Create a new Zettelkasten note.
            Args:
                title: The title of the note
                content: The main content of the note
                note_type: Type of note (fleeting, literature, permanent, structure, hub)
                project: Project this note belongs to (organizes notes into project directories)
                note_purpose: Workflow category (research, planning, bugfixing, general). Auto-inferred from content if left as 'general' - set explicitly only when the auto-inference would be wrong
                tags: Comma-separated list of tags (optional)
                plan_id: Optional ID to group related planning notes
            """
            with timed_operation("zk_create_note", title=title[:30]) as op:
                try:
                    _validate_input_lengths(title=title, content=content)
                    # Convert note_type string to enum
                    try:
                        note_type_enum = NoteType(note_type.lower())
                    except ValueError:
                        return f"Invalid note type: {note_type}. Valid types are: {', '.join(t.value for t in NoteType)}"

                    # Convert note_purpose string to enum
                    try:
                        purpose_enum = NotePurpose(note_purpose.lower())
                    except ValueError:
                        return f"Invalid note purpose: {note_purpose}. Valid purposes are: {', '.join(p.value for p in NotePurpose)}"

                    # Convert tags string to list
                    tag_list = []
                    if tags:
                        tag_list = [t.strip() for t in tags.split(",") if t.strip()]

                    # Create the note
                    note = self.zettel_service.create_note(
                        title=title,
                        content=content,
                        note_type=note_type_enum,
                        project=project,
                        note_purpose=purpose_enum,
                        tags=tag_list,
                        plan_id=plan_id,
                    )
                    op["note_id"] = note.id
                    return f"Note created successfully with ID: {note.id} (project: {note.project})"
                except Exception as e:
                    return self.format_error_response(e)

        # Get a note by ID or title
        @self.mcp.tool(name="zk_get_note")
        def zk_get_note(identifier: str, format: str = "summary") -> str:
            """Retrieve a note by ID or title.
            Args:
                identifier: The ID or title of the note
                format: Output format:
                    - "summary" (default): Structured overview with metadata
                    - "markdown": Raw markdown with YAML frontmatter (for export/backup)
            Returns:
                Note content with version hash (use version in zk_update_note/zk_delete_note
                for conflict detection when multiple processes may edit the same note).
            """
            with timed_operation("zk_get_note", identifier=identifier[:30]) as op:
                try:
                    identifier = str(identifier)
                    # Try to get by ID first (with version info)
                    versioned = self.zettel_service.get_note_versioned(identifier)
                    # If not found, try by title
                    if not versioned:
                        note = self.zettel_service.get_note_by_title(identifier)
                        if note:
                            versioned = self.zettel_service.get_note_versioned(note.id)
                    if not versioned:
                        op["found"] = False
                        return f"Note not found: {identifier}"

                    note = versioned.note
                    version = versioned.version
                    op["found"] = True
                    op["note_id"] = note.id

                    # Markdown export format
                    if format == "markdown":
                        return self.zettel_service.export_note(note.id, "markdown")

                    # Default summary format
                    result = f"# {note.title}\n"
                    result += f"ID: {note.id}\n"
                    result += f"Version: {version.commit_hash}\n"
                    result += f"Type: {note.note_type.value}\n"
                    result += f"Project: {note.project}\n"
                    result += f"Purpose: {note.note_purpose.value if note.note_purpose else 'general'}\n"
                    if note.plan_id:
                        result += f"Plan ID: {note.plan_id}\n"
                    result += f"Created: {note.created_at.isoformat()}\n"
                    result += f"Updated: {note.updated_at.isoformat()}\n"
                    if note.tags:
                        result += f"Tags: {', '.join(tag.name for tag in note.tags)}\n"
                    # Add note content, including the Links section added by _note_to_markdown()
                    result += f"\n{note.content}\n"
                    return result
                except Exception as e:
                    return self.format_error_response(e)

        @self.mcp.tool(name="zk_note_history")
        def zk_note_history(note_id: str, limit: int = 10) -> str:
            """Get version history for a note.

            Shows commit history for a note, useful for tracking changes over time.
            Requires git versioning to be enabled in the notes directory.

            Args:
                note_id: The ID of the note
                limit: Maximum number of versions to return (default: 10)
            """
            with timed_operation("zk_note_history", note_id=note_id) as op:
                try:
                    history = self.zettel_service.get_note_history(str(note_id), limit)
                    op["version_count"] = len(history)

                    if not history:
                        return f"No version history for note '{note_id}'. Git versioning may be disabled."

                    result = f"# Version History for {note_id}\n\n"
                    result += f"**{len(history)} version(s)**\n\n"
                    result += "| # | Commit | Date |\n"
                    result += "|---|--------|------|\n"
                    for i, version in enumerate(history, 1):
                        result += f"| {i} | {version['short_hash']} | {version['timestamp']} |\n"

                    return result
                except Exception as e:
                    return self.format_error_response(e)

        # Update a note (supports batch project move with comma-separated IDs)
        @self.mcp.tool(name="zk_update_note")
        def zk_update_note(
            note_id: str,
            title: Optional[str] = None,
            content: Optional[str] = None,
            note_type: Optional[str] = None,
            project: Optional[str] = None,
            note_purpose: Optional[str] = None,
            tags: Optional[str] = None,
            plan_id: Optional[str] = None,
            expected_version: Optional[str] = None,
        ) -> str:
            """Update an existing note, or batch-move multiple notes to a project.

            Supports batch mode: pass comma-separated note IDs to move multiple
            notes to a different project at once. In batch mode, only the
            `project` parameter is allowed — all other fields must be omitted.

            Args:
                note_id: The ID of the note to update, or comma-separated IDs for batch project move
                title: New title (optional, single-note only)
                content: New content (optional, single-note only)
                note_type: New note type (optional, single-note only)
                project: New project (optional, moves note to different project directory)
                note_purpose: New workflow category (research, planning, bugfixing, general) - optional, single-note only
                tags: New comma-separated list of tags (optional, single-note only)
                plan_id: New plan ID (optional, use empty string to clear, single-note only)
                expected_version: Version hash from zk_get_note for conflict detection (single-note only).
                    If provided and doesn't match current version, returns CONFLICT error
                    instead of overwriting (recommended for multi-process safety).
            Returns:
                Success message with new version, or CONFLICT error if version mismatch.
            """
            try:
                _validate_input_lengths(title=title, content=content)
                ids = [id.strip() for id in note_id.split(",") if id.strip()]
                if not ids:
                    return "Error: No note IDs provided."

                # Batch mode: multiple IDs — only project move allowed
                if len(ids) > 1:
                    has_other_fields = any(
                        [
                            title,
                            content,
                            note_type,
                            note_purpose,
                            tags,
                            plan_id,
                            expected_version,
                        ]
                    )
                    if has_other_fields:
                        return "Error: Batch mode only supports project moves. Pass comma-separated IDs with only the 'project' parameter."
                    if not project or not project.strip():
                        return (
                            "Error: 'project' parameter is required for batch update."
                        )
                    updated = self.zettel_service.bulk_update_project(
                        ids, project.strip()
                    )
                    return f"Moved {updated} notes to project '{project.strip()}'."

                # Single mode: existing behavior
                single_id = ids[0]
                note = self.zettel_service.get_note(str(single_id))
                if not note:
                    return f"Note not found: {single_id}"

                # Convert note_type string to enum if provided
                note_type_enum = None
                if note_type:
                    try:
                        note_type_enum = NoteType(note_type.lower())
                    except ValueError:
                        return f"Invalid note type: {note_type}. Valid types are: {', '.join(t.value for t in NoteType)}"

                # Convert note_purpose string to enum if provided
                purpose_enum = None
                if note_purpose:
                    try:
                        purpose_enum = NotePurpose(note_purpose.lower())
                    except ValueError:
                        return f"Invalid note purpose: {note_purpose}. Valid purposes are: {', '.join(p.value for p in NotePurpose)}"

                # Convert tags string to list if provided
                tag_list = None
                if tags is not None:  # Allow empty string to clear tags
                    tag_list = [t.strip() for t in tags.split(",") if t.strip()]

                # Update the note with version checking
                result = self.zettel_service.update_note_versioned(
                    note_id=single_id,
                    title=title,
                    content=content,
                    note_type=note_type_enum,
                    project=project,
                    note_purpose=purpose_enum,
                    tags=tag_list,
                    plan_id=plan_id,
                    expected_version=expected_version,
                )

                # Check for conflict
                if isinstance(result, ConflictResult):
                    return (
                        f"CONFLICT: {result.message}\n"
                        f"Expected version: {result.expected_version}\n"
                        f"Actual version: {result.actual_version}\n"
                        f"Re-read the note with zk_get_note to get the latest version."
                    )

                # Success - return new version
                versioned = result
                return (
                    f"Note updated successfully: {versioned.note.id}\n"
                    f"New version: {versioned.version.commit_hash}\n"
                    f"Project: {versioned.note.project}"
                )
            except Exception as e:
                return self.format_error_response(e)

        # Delete a note (supports batch mode with comma-separated IDs)
        @self.mcp.tool(name="zk_delete_note")
        def zk_delete_note(note_id: str, expected_version: Optional[str] = None) -> str:
            """Delete one or more notes.

            Supports batch mode: pass comma-separated IDs to delete multiple notes
            at once (e.g. "id1, id2, id3"). Version conflict detection is only
            available for single-note deletes.

            Args:
                note_id: The ID of the note to delete, or comma-separated IDs for batch delete
                expected_version: Version hash from zk_get_note for conflict detection.
                    Only valid for single-note deletes. If provided and doesn't match
                    current version, returns CONFLICT error instead of deleting.
            Returns:
                Success message, or CONFLICT error if version mismatch.
            """
            try:
                ids = [id.strip() for id in note_id.split(",") if id.strip()]
                if not ids:
                    return "Error: No note IDs provided."

                # Batch mode: multiple IDs
                if len(ids) > 1:
                    if expected_version:
                        return "Error: expected_version cannot be used with batch delete. Delete notes individually for version checking."
                    deleted = self.zettel_service.bulk_delete_notes(ids)
                    return f"Successfully deleted {deleted} notes."

                # Single mode: existing behavior
                single_id = ids[0]
                note = self.zettel_service.get_note(single_id)
                if not note:
                    return f"Note not found: {single_id}"

                result = self.zettel_service.delete_note_versioned(
                    note_id=str(single_id), expected_version=expected_version
                )

                if isinstance(result, ConflictResult):
                    return (
                        f"CONFLICT: {result.message}\n"
                        f"Expected version: {result.expected_version}\n"
                        f"Actual version: {result.actual_version}\n"
                        f"Re-read the note with zk_get_note to verify before deleting."
                    )

                return f"Note deleted successfully: {single_id}"
            except Exception as e:
                return self.format_error_response(e)

        # Add a link between notes
        @self.mcp.tool(name="zk_create_link")
        def zk_create_link(
            source_id: str,
            target_id: str,
            link_type: str = "reference",
            description: Optional[str] = None,
            bidirectional: bool = False,
        ) -> str:
            """Create a link between two notes.
            Args:
                source_id: ID of the source note
                target_id: ID of the target note
                link_type: Type of link (reference, extends, refines, contradicts, questions, supports, related)
                description: Optional description of the link
                bidirectional: Whether to create a link in both directions
            """
            try:
                # Convert link_type string to enum
                try:
                    link_type_enum = LinkType(link_type.lower())
                except ValueError:
                    return f"Invalid link type: {link_type}. Valid types are: {', '.join(t.value for t in LinkType)}"

                # Create the link
                source_note, target_note = self.zettel_service.create_link(
                    source_id=source_id,
                    target_id=target_id,
                    link_type=link_type_enum,
                    description=description,
                    bidirectional=bidirectional,
                )
                if bidirectional:
                    return f"Bidirectional link created between {source_id} and {target_id}"
                else:
                    return f"Link created from {source_id} to {target_id}"
            except Exception as e:
                if "UNIQUE constraint failed" in str(e):
                    return f"A link of this type already exists between these notes. Try a different link type."
                return self.format_error_response(e)

        # Remove a link between notes
        @self.mcp.tool(name="zk_remove_link")
        def zk_remove_link(
            source_id: str, target_id: str, bidirectional: bool = False
        ) -> str:
            """Remove a link between two notes.
            Args:
                source_id: ID of the source note
                target_id: ID of the target note
                bidirectional: Whether to remove the link in both directions
            """
            try:
                # Remove the link
                source_note, target_note = self.zettel_service.remove_link(
                    source_id=str(source_id),
                    target_id=str(target_id),
                    bidirectional=bidirectional,
                )
                if bidirectional:
                    return f"Bidirectional link removed between {source_id} and {target_id}"
                else:
                    return f"Link removed from {source_id} to {target_id}"
            except Exception as e:
                return self.format_error_response(e)

        # Search for notes
        @self.mcp.tool(name="zk_search_notes")
        def zk_search_notes(
            query: Optional[str] = None,
            tags: Optional[str] = None,
            note_type: Optional[str] = None,
            mode: str = "auto",
            limit: int = 10,
        ) -> str:
            """Search for notes by text, tags, or type.
            Args:
                query: Text to search for in titles and content
                tags: Comma-separated list of tags to filter by
                note_type: Type of note to filter by
                mode: Search mode:
                    - "auto": Automatically picks the best strategy (default).
                      Uses semantic search when embeddings are available and no
                      tag/type filters are set; otherwise uses text search.
                    - "semantic": Embedding-based similarity search.
                      Finds conceptually related notes even without exact keyword matches.
                      Does not support tag/type filtering.
                    - "text": Keyword matching in titles/content with tag/type filters.
                limit: Maximum number of results to return
            """
            with timed_operation(
                "zk_search_notes", query=query[:30] if query else None, mode=mode
            ) as op:
                try:
                    if mode == "auto":
                        has_filters = tags or note_type
                        if (
                            self.search_service.has_semantic_search
                            and query
                            and query.strip()
                            and not has_filters
                        ):
                            mode = "semantic"
                        else:
                            mode = "text"

                    if mode == "semantic":
                        if tags or note_type:
                            return (
                                "Error: tag and note_type filters are not yet supported "
                                "in semantic mode. Use mode='text' or mode='auto'."
                            )
                        # Fall back to text mode when embeddings unavailable
                        if not self.search_service.has_semantic_search:
                            mode = "text"
                        elif not query or not query.strip():
                            return "Error: query is required for semantic search."

                    if mode == "semantic":
                        results = self.search_service.semantic_search(
                            query=query.strip(),
                            limit=limit,
                            use_reranker=True,
                        )
                        op["result_count"] = len(results)

                        if not results:
                            return "No semantically similar notes found."

                        output = f"Found {len(results)} semantically similar notes:\n\n"
                        for i, result in enumerate(results, 1):
                            note = result.note
                            output += f"{i}. {note.title} (ID: {note.id})\n"
                            output += f"   Score: {result.score:.3f}"
                            if result.reranked:
                                output += " (reranked)"
                            output += "\n"
                            if note.tags:
                                output += f"   Tags: {', '.join(tag.name for tag in note.tags)}\n"
                            content_preview = note.content[:150].replace("\n", " ")
                            if len(note.content) > 150:
                                content_preview += "..."
                            output += f"   Preview: {content_preview}\n\n"
                        return output

                    elif mode == "text":
                        # Convert tags string to list if provided
                        tag_list = None
                        if tags:
                            tag_list = [t.strip() for t in tags.split(",") if t.strip()]

                        # Convert note_type string to enum if provided
                        note_type_enum = None
                        if note_type:
                            try:
                                note_type_enum = NoteType(note_type.lower())
                            except ValueError:
                                return f"Invalid note type: {note_type}. Valid types are: {', '.join(t.value for t in NoteType)}"

                        # Perform search
                        results = self.search_service.search_combined(
                            text=query, tags=tag_list, note_type=note_type_enum
                        )

                        # Limit results
                        results = results[:limit]
                        op["result_count"] = len(results)
                        if not results:
                            return "No matching notes found."

                        # Format results
                        output = f"Found {len(results)} matching notes:\n\n"
                        for i, result in enumerate(results, 1):
                            note = result.note
                            output += f"{i}. {note.title} (ID: {note.id})\n"
                            if note.tags:
                                output += f"   Tags: {', '.join(tag.name for tag in note.tags)}\n"
                            output += (
                                f"   Created: {note.created_at.strftime('%Y-%m-%d')}\n"
                            )
                            # Add a snippet of content (first 150 chars)
                            content_preview = note.content[:150].replace("\n", " ")
                            if len(note.content) > 150:
                                content_preview += "..."
                            output += f"   Preview: {content_preview}\n\n"
                        return output

                    else:
                        return (
                            f"Invalid mode: '{mode}'. Valid modes: auto, text, semantic"
                        )

                except Exception as e:
                    return self.format_error_response(e)

        # Full-text search with FTS5
        @self.mcp.tool(name="zk_fts_search")
        def zk_fts_search(query: str, limit: int = 20, highlight: bool = True) -> str:
            """Full-text search using FTS5 with advanced query syntax.
            Args:
                query: Search query. Supports:
                       - Simple terms: "python async"
                       - Phrases: '"async await"'
                       - Boolean: "python AND NOT java"
                       - Prefix: "program*"
                       - Column filter: "title:python"
                limit: Maximum number of results to return
                highlight: Include highlighted snippets in results
            """
            with timed_operation(
                "zk_fts_search", query=query[:30] if query else None
            ) as op:
                try:
                    if not query or not query.strip():
                        return "Error: Search query is required."

                    results = self.zettel_service.fts_search(
                        query=query.strip(), limit=limit, highlight=highlight
                    )

                    op["result_count"] = len(results)

                    if not results:
                        return f"No notes found matching '{query}'."

                    # Check if fallback mode was used and warn the user
                    fallback_warning = ""
                    if results and results[0].get("search_mode") == "fallback":
                        fallback_warning = (
                            "⚠️ Note: FTS5 search failed, using basic text matching. "
                            "Results may be less accurate and slower.\n\n"
                        )
                        op["search_mode"] = "fallback"
                    else:
                        op["search_mode"] = "fts5"

                    # Format results
                    output = f"{fallback_warning}Found {len(results)} notes matching '{query}':\n\n"
                    for i, result in enumerate(results, 1):
                        output += f"{i}. {result['title']} (ID: {result['id']})\n"
                        output += f"   Relevance: {abs(result['rank']):.2f}\n"
                        if highlight and "snippet" in result:
                            snippet = result["snippet"].replace("\n", " ")
                            output += f"   Match: {snippet}\n"
                        output += "\n"

                    return output
                except Exception as e:
                    return self.format_error_response(e)

        # Add tags to notes (supports batch mode with comma-separated values)
        @self.mcp.tool(name="zk_add_tag")
        def zk_add_tag(note_id: str, tag: str) -> str:
            """Add one or more tags to one or more notes.

            Supports batch mode: pass comma-separated note IDs and/or
            comma-separated tags to operate on multiple items at once.

            Args:
                note_id: The ID of the note, or comma-separated IDs for batch mode
                tag: The tag to add, or comma-separated tags for batch mode
            """
            try:
                ids = [id.strip() for id in note_id.split(",") if id.strip()]
                tags = [t.strip() for t in tag.split(",") if t.strip()]

                if not ids:
                    return "Error: No note IDs provided."
                if not tags:
                    return "Error: Tag cannot be empty"

                # Batch mode: multiple IDs or multiple tags
                if len(ids) > 1 or len(tags) > 1:
                    updated = self.zettel_service.bulk_add_tags(ids, tags)
                    return f"Added tags [{', '.join(tags)}] to {updated} notes."

                # Single mode: existing behavior
                note = self.zettel_service.add_tag_to_note(str(ids[0]), tags[0])
                return f"Tag '{tags[0]}' added to note '{note.title}' (ID: {note.id})"
            except Exception as e:
                return self.format_error_response(e)

        # Remove tags from notes (supports batch mode with comma-separated values)
        @self.mcp.tool(name="zk_remove_tag")
        def zk_remove_tag(note_id: str, tag: str) -> str:
            """Remove one or more tags from one or more notes.

            Supports batch mode: pass comma-separated note IDs and/or
            comma-separated tags to operate on multiple items at once.

            Args:
                note_id: The ID of the note, or comma-separated IDs for batch mode
                tag: The tag to remove, or comma-separated tags for batch mode
            """
            try:
                ids = [id.strip() for id in note_id.split(",") if id.strip()]
                tags = [t.strip() for t in tag.split(",") if t.strip()]

                if not ids:
                    return "Error: No note IDs provided."
                if not tags:
                    return "Error: Tag cannot be empty"

                # Batch mode: multiple IDs or multiple tags
                if len(ids) > 1 or len(tags) > 1:
                    updated = self.zettel_service.bulk_remove_tags(ids, tags)
                    return f"Removed tags [{', '.join(tags)}] from {updated} notes."

                # Single mode: existing behavior
                note = self.zettel_service.remove_tag_from_note(str(ids[0]), tags[0])
                return (
                    f"Tag '{tags[0]}' removed from note '{note.title}' (ID: {note.id})"
                )
            except Exception as e:
                return self.format_error_response(e)

        @self.mcp.tool(name="zk_cleanup_tags")
        def zk_cleanup_tags() -> str:
            """Delete tags that are not associated with any notes.

            Cleans up orphaned tags left behind when notes were deleted
            or had their tags removed. This is a maintenance operation.

            Returns a summary of deleted tags.
            """
            with timed_operation("zk_cleanup_tags") as op:
                try:
                    count = self.zettel_service.delete_unused_tags()
                    op["deleted_count"] = count
                    if count == 0:
                        return "No unused tags found. Tag database is clean."
                    return f"Cleaned up {count} unused tag(s)."
                except Exception as e:
                    return self.format_error_response(e)

        # ========== Batch Create ==========

        @self.mcp.tool(name="zk_bulk_create_notes")
        def zk_bulk_create_notes(notes: str) -> str:
            """Create multiple notes in a single batch operation.

            All notes are created atomically - if any fails, all are rolled back.

            Args:
                notes: JSON array of note objects. Each must have:
                    - title (required): Note title
                    - content (required): Note content
                    - note_type (optional): fleeting, literature, permanent, structure, hub
                    - project (optional): Project name (default: "general")
                    - tags (optional): Array of tag names
            """
            with timed_operation("zk_bulk_create_notes") as op:
                try:
                    notes_data = json.loads(notes)
                    if not isinstance(notes_data, list):
                        return "Error: Input must be a JSON array of note objects."

                    op["note_count"] = len(notes_data)
                    created = self.zettel_service.bulk_create_notes(notes_data)

                    output = f"Successfully created {len(created)} notes:\n\n"
                    for note in created:
                        output += f"- {note.title} (ID: {note.id})\n"

                    return output
                except json.JSONDecodeError as e:
                    return f"Error: Invalid JSON - {e}"
                except Exception as e:
                    return self.format_error_response(e)

        # ========== Consolidated Tools ==========

        @self.mcp.tool(name="zk_list_notes")
        def zk_list_notes(
            mode: str = "all",
            project: Optional[str] = None,
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            use_updated: bool = False,
            sort_by: str = "updated_at",
            descending: bool = True,
            limit: int = 20,
            offset: int = 0,
        ) -> str:
            """List and discover notes with flexible filtering.

            Args:
                mode: Discovery mode:
                    - "all": List all notes with pagination
                    - "by_date": Filter by date range
                    - "by_project": Filter by project
                    - "central": Most connected notes (most links)
                    - "orphans": Notes with no connections
                project: Project name (required for mode="by_project")
                start_date: Start date in ISO format YYYY-MM-DD (for mode="by_date")
                end_date: End date in ISO format YYYY-MM-DD (for mode="by_date")
                use_updated: Use updated_at instead of created_at for date filtering
                sort_by: Sort field (created_at, updated_at, title) - for mode="all"
                descending: Sort order (True for newest first)
                limit: Maximum results to return
                offset: Skip this many results (for pagination)
            """
            with timed_operation("zk_list_notes", mode=mode) as op:
                try:
                    limit = min(max(1, limit), 200)
                    offset = max(0, offset)

                    if mode == "all":
                        total_count = self.zettel_service.repository.count_notes()
                        if total_count == 0:
                            return "No notes found in the Zettelkasten."

                        sort_order = "desc" if descending else "asc"
                        notes = self.zettel_service.repository.list_notes(
                            sort_by=sort_by,
                            sort_order=sort_order,
                            limit=limit,
                            offset=offset,
                        )
                        op["result_count"] = len(notes)

                        if not notes:
                            return f"No notes found at offset {offset}. Total notes: {total_count}"

                        output = f"Notes ({offset + 1}-{offset + len(notes)} of {total_count}):\n\n"
                        for i, note in enumerate(notes, offset + 1):
                            output += f"{i}. {note.title} (ID: {note.id})\n"
                            output += f"   Type: {note.note_type.value} | Project: {note.project}\n"
                            output += f"   Updated: {note.updated_at.strftime('%Y-%m-%d %H:%M')}\n"
                            if note.tags:
                                output += f"   Tags: {', '.join(tag.name for tag in note.tags)}\n"
                            output += "\n"

                        if offset + limit < total_count:
                            output += (
                                f"\n(Use offset={offset + limit} to see more notes)"
                            )
                        return output

                    elif mode == "by_date":
                        start_datetime = None
                        if start_date:
                            start_datetime = datetime.fromisoformat(
                                f"{start_date}T00:00:00+00:00"
                            )
                        end_datetime = None
                        if end_date:
                            end_datetime = datetime.fromisoformat(
                                f"{end_date}T23:59:59+00:00"
                            )

                        notes = self.search_service.find_notes_by_date_range(
                            start_date=start_datetime,
                            end_date=end_datetime,
                            use_updated=use_updated,
                        )[:limit]
                        op["result_count"] = len(notes)

                        if not notes:
                            return "No notes found in the specified date range."

                        date_type = "updated" if use_updated else "created"
                        output = f"Notes {date_type} in date range ({len(notes)} results):\n\n"
                        for i, note in enumerate(notes, 1):
                            date = note.updated_at if use_updated else note.created_at
                            output += f"{i}. {note.title} (ID: {note.id})\n"
                            output += f"   {date_type.capitalize()}: {date.strftime('%Y-%m-%d %H:%M')}\n\n"
                        return output

                    elif mode == "by_project":
                        if not project:
                            return "Error: 'project' parameter is required for mode='by_project'"
                        notes = self.zettel_service.get_notes_by_project(project)
                        op["result_count"] = len(notes)

                        if not notes:
                            return f"No notes found in project '{project}'."

                        output = (
                            f"Notes in project '{project}' ({len(notes)} results):\n\n"
                        )
                        for i, note in enumerate(notes, 1):
                            output += f"{i}. {note.title} (ID: {note.id})\n"
                            output += f"   Type: {note.note_type.value}\n"
                            if note.tags:
                                output += f"   Tags: {', '.join(tag.name for tag in note.tags)}\n"
                            output += "\n"
                        return output

                    elif mode == "central":
                        central_notes = self.search_service.find_central_notes(limit)
                        op["result_count"] = len(central_notes)

                        if not central_notes:
                            return "No notes found with connections."

                        output = "Most connected notes (central hubs):\n\n"
                        for i, (note, conn_count) in enumerate(central_notes, 1):
                            output += f"{i}. {note.title} (ID: {note.id})\n"
                            output += f"   Connections: {conn_count}\n"
                            if note.tags:
                                output += f"   Tags: {', '.join(tag.name for tag in note.tags)}\n"
                            output += "\n"
                        return output

                    elif mode == "orphans":
                        orphans = self.search_service.find_orphaned_notes()
                        op["result_count"] = len(orphans)

                        if not orphans:
                            return (
                                "No orphaned notes found. All notes have connections!"
                            )

                        output = (
                            f"Orphaned notes ({len(orphans)} with no connections):\n\n"
                        )
                        for i, note in enumerate(orphans, 1):
                            output += f"{i}. {note.title} (ID: {note.id})\n"
                            output += f"   Type: {note.note_type.value} | Project: {note.project}\n"
                            if note.tags:
                                output += f"   Tags: {', '.join(tag.name for tag in note.tags)}\n"
                            output += "\n"
                        return output

                    else:
                        return f"Invalid mode: '{mode}'. Valid modes: all, by_date, by_project, central, orphans"

                except ValueError as e:
                    return f"Error parsing date: {str(e)}"
                except Exception as e:
                    return self.format_error_response(e)

        @self.mcp.tool(name="zk_find_related")
        def zk_find_related(
            note_id: str,
            mode: str = "linked",
            direction: str = "both",
            threshold: float = 0.3,
            limit: int = 10,
        ) -> str:
            """Find notes related to a specific note.

            Args:
                note_id: ID of the reference note
                mode: Relation type:
                    - "linked": Notes connected via links
                    - "similar": Notes with shared tags/links (similarity score)
                    - "semantic": Notes with similar meaning via embeddings (requires embeddings enabled).
                      Finds conceptually related notes regardless of shared tags or links.
                direction: Link direction for mode="linked" (outgoing, incoming, both)
                threshold: Similarity threshold 0.0-1.0 for mode="similar"
                limit: Maximum results to return
            """
            with timed_operation(
                "zk_find_related", mode=mode, note_id=note_id[:20]
            ) as op:
                try:
                    note_id = str(note_id)

                    # Verify note exists
                    note = self.zettel_service.get_note(note_id)
                    if not note:
                        return f"Note not found: {note_id}"

                    if mode == "linked":
                        if direction not in ["outgoing", "incoming", "both"]:
                            return f"Invalid direction: '{direction}'. Use: outgoing, incoming, both"

                        linked_notes = self.zettel_service.get_linked_notes(
                            note_id, direction
                        )
                        op["result_count"] = len(linked_notes)

                        if not linked_notes:
                            return f"No {direction} links found for note '{note.title}'"

                        output = f"Notes linked to '{note.title}' ({direction}):\n\n"
                        for i, linked_note in enumerate(linked_notes, 1):
                            output += (
                                f"{i}. {linked_note.title} (ID: {linked_note.id})\n"
                            )
                            if linked_note.tags:
                                output += f"   Tags: {', '.join(tag.name for tag in linked_note.tags)}\n"

                            # Try to show link type
                            if direction in ["outgoing", "both"]:
                                for link in note.links:
                                    if str(link.target_id) == str(linked_note.id):
                                        output += (
                                            f"   Link type: {link.link_type.value}\n"
                                        )
                                        if link.description:
                                            output += (
                                                f"   Description: {link.description}\n"
                                            )
                                        break
                            output += "\n"
                        return output

                    elif mode == "similar":
                        similar_notes = self.zettel_service.find_similar_notes(
                            note_id, threshold
                        )[:limit]
                        op["result_count"] = len(similar_notes)

                        if not similar_notes:
                            return f"No similar notes found for '{note.title}' with threshold {threshold}"

                        output = f"Notes similar to '{note.title}':\n\n"
                        for i, (sim_note, score) in enumerate(similar_notes, 1):
                            output += f"{i}. {sim_note.title} (ID: {sim_note.id})\n"
                            output += f"   Similarity: {score:.2f}\n"
                            if sim_note.tags:
                                output += f"   Tags: {', '.join(tag.name for tag in sim_note.tags)}\n"
                            output += "\n"
                        return output

                    elif mode == "semantic":
                        results = self.search_service.find_related(
                            note_id,
                            limit=limit,
                            use_reranker=True,
                        )
                        op["result_count"] = len(results)

                        if not results:
                            if not config.embeddings_enabled:
                                return (
                                    "Semantic search unavailable: embeddings are disabled.\n"
                                    "Set ZETTELKASTEN_EMBEDDINGS_ENABLED=true and install "
                                    "the [semantic] extra to enable."
                                )
                            return f"No semantically related notes found for '{note.title}'"

                        output = f"Notes semantically related to '{note.title}':\n\n"
                        for i, result in enumerate(results, 1):
                            r_note = result.note
                            output += f"{i}. {r_note.title} (ID: {r_note.id})\n"
                            output += f"   Score: {result.score:.3f}"
                            if result.reranked:
                                output += " (reranked)"
                            output += "\n"
                            if r_note.tags:
                                output += f"   Tags: {', '.join(tag.name for tag in r_note.tags)}\n"
                            output += "\n"
                        return output

                    else:
                        return f"Invalid mode: '{mode}'. Valid modes: linked, similar, semantic"

                except Exception as e:
                    return self.format_error_response(e)

        @self.mcp.tool(name="zk_status")
        def zk_status(sections: str = "all") -> str:
            """Get comprehensive Zettelkasten status dashboard.

            Args:
                sections: Comma-separated sections to include:
                    - "summary": Note counts by type and project
                    - "tags": All tags with usage counts
                    - "health": Database integrity status
                    - "embeddings": Embedding system status
                    - "metrics": Server performance metrics
                    - "all": Include all sections (default)
            """
            with timed_operation("zk_status") as op:
                try:
                    requested = set(s.strip().lower() for s in sections.split(","))
                    include_all = "all" in requested

                    output = "# Zettelkasten Status\n\n"

                    # Summary section
                    if include_all or "summary" in requested:
                        repo = self.zettel_service.repository
                        total = repo.count_notes()
                        by_type = repo.count_notes_by_type()
                        by_project = repo.count_notes_by_project()

                        output += f"## Summary\n"
                        output += f"**Total Notes:** {total}\n\n"

                        if by_type:
                            output += "**By Type:**\n"
                            for t, count in sorted(
                                by_type.items(), key=lambda x: -x[1]
                            ):
                                output += f"  - {t}: {count}\n"
                            output += "\n"

                        if by_project:
                            output += "**By Project:**\n"
                            for p, count in sorted(
                                by_project.items(), key=lambda x: -x[1]
                            ):
                                output += f"  - {p}: {count}\n"
                            output += "\n"

                        # Link stats
                        central = self.search_service.find_central_notes(5)
                        orphans = self.search_service.find_orphaned_notes()
                        output += f"**Connections:** {len(central)} notes with links, {len(orphans)} orphans\n\n"

                    # Tags section
                    if include_all or "tags" in requested:
                        tag_counts = self.zettel_service.get_tags_with_counts()
                        output += f"## Tags ({len(tag_counts)} total)\n"
                        if tag_counts:
                            # Sort by usage count (descending), then name (ascending)
                            sorted_tags = sorted(
                                tag_counts.items(), key=lambda x: (-x[1], x[0].lower())
                            )
                            output += "| Tag | Notes |\n"
                            output += "|-----|-------|\n"
                            for tag_name, count in sorted_tags:
                                output += f"| {tag_name} | {count} |\n"
                            output += "\n"
                        else:
                            output += "No tags defined.\n\n"

                    # Health section
                    if include_all or "health" in requested:
                        health = self.zettel_service.check_database_health()
                        output += "## Health\n"
                        status_icon = "✅" if health["healthy"] else "❌"
                        output += f"**Overall:** {status_icon} {'Healthy' if health['healthy'] else 'Issues detected'}\n"
                        output += (
                            f"**SQLite:** {'OK' if health['sqlite_ok'] else 'ERROR'}\n"
                        )
                        output += (
                            f"**FTS5:** {'OK' if health['fts_ok'] else 'Degraded'}\n"
                        )
                        output += f"**DB Notes:** {health['note_count']} | **Files:** {health['file_count']}\n"
                        output += f"**Sync Needed:** {'Yes' if health.get('needs_sync') else 'No'}\n"
                        if health.get("issues"):
                            output += f"**Issues:** {', '.join(health['issues'])}\n"
                        if health.get("critical_issues"):
                            output += "**Critical Issues:**\n"
                            for issue in health["critical_issues"]:
                                output += f"  - {issue}\n"
                        output += "\n"

                    # Embeddings section
                    if include_all or "embeddings" in requested:
                        output += "## Embeddings\n"
                        output += f"**Enabled:** {'Yes' if config.embeddings_enabled else 'No'}\n"
                        if config.embeddings_enabled:
                            output += f"**Model:** {config.embedding_model}\n"
                            output += f"**Dimension:** {config.embedding_dim}\n"
                            emb_count = (
                                self.zettel_service.repository.count_embeddings()
                            )
                            total_notes = self.zettel_service.count_notes()
                            output += f"**Indexed:** {emb_count}/{total_notes} notes\n"
                            if total_notes > 0:
                                pct = emb_count / total_notes * 100
                                output += f"**Coverage:** {pct:.0f}%\n"
                        output += "\n"

                    # Metrics section
                    if include_all or "metrics" in requested:
                        summary = metrics.get_summary()
                        output += "## Server Metrics\n"
                        output += (
                            f"**Uptime:** {summary['uptime_seconds']:.0f} seconds\n"
                        )
                        output += f"**Operations:** {summary['total_operations']}\n"
                        output += (
                            f"**Success Rate:** {summary['overall_success_rate']:.1%}\n"
                        )
                        output += f"**Errors:** {summary['total_errors']}\n\n"

                    # Backup status
                    if include_all or "summary" in requested:
                        backups = backup_manager.list_backups()
                        db_backups = [b for b in backups if b["type"] == "database"]
                        if db_backups:
                            latest = db_backups[0]
                            output += f"**Last Backup:** {latest['created_at'][:10]} ({latest['size_mb']} MB)\n"
                        else:
                            output += "**Backups:** None found\n"

                    op["sections"] = sections
                    return output

                except Exception as e:
                    return self.format_error_response(e)

        @self.mcp.tool(name="zk_system")
        def zk_system(action: str, backup_label: Optional[str] = None) -> str:
            """System administration operations.

            Args:
                action: Operation to perform:
                    - "rebuild": Rebuild database index from markdown files
                    - "sync": Sync notes to Obsidian vault
                    - "backup": Create database and notes backup
                    - "list_backups": List available backups
                    - "reset_fts": Reset FTS5 availability after manual repair
                    - "reindex_embeddings": Rebuild all note embeddings from scratch
                      (use after model changes or to fix inconsistencies)
                backup_label: Optional label for backup (e.g., "pre-migration")
            """
            with timed_operation("zk_system", action=action) as op:
                try:
                    action = action.lower().strip()

                    if action == "rebuild":
                        note_count_before = len(self.zettel_service.get_all_notes())
                        self.zettel_service.rebuild_index()
                        note_count_after = len(self.zettel_service.get_all_notes())

                        return (
                            f"Database index rebuilt successfully.\n"
                            f"Notes processed: {note_count_after}\n"
                            f"Change in count: {note_count_after - note_count_before}"
                        )

                    elif action == "sync":
                        synced_count = self.zettel_service.sync_to_obsidian()
                        return (
                            f"Successfully synced {synced_count} notes to Obsidian vault.\n"
                            f"Old mirror files were cleaned before re-sync to prevent duplicates."
                        )

                    elif action == "backup":
                        result = backup_manager.create_full_backup(label=backup_label)
                        output = "Backup created:\n"
                        if result.get("database"):
                            db_size = result["database"].stat().st_size / (1024 * 1024)
                            output += f"  Database: {result['database'].name} ({db_size:.2f} MB)\n"
                        if result.get("notes"):
                            notes_size = result["notes"].stat().st_size / (1024 * 1024)
                            output += f"  Notes: {result['notes'].name} ({notes_size:.2f} MB)\n"
                        return output

                    elif action == "list_backups":
                        backups = backup_manager.list_backups()
                        if not backups:
                            return "No backups found."

                        output = f"Available backups ({len(backups)} total):\n\n"
                        for i, b in enumerate(backups, 1):
                            output += f"{i}. {b['name']}\n"
                            output += (
                                f"   Type: {b['type']} | Size: {b['size_mb']} MB\n"
                            )
                            output += f"   Created: {b['created_at']}\n\n"
                        return output

                    elif action == "reset_fts":
                        success = self.zettel_service.reset_fts_availability()
                        if success:
                            return "FTS5 availability reset. Full-text search is now enabled."
                        else:
                            return (
                                "FTS5 reset failed. The index may still be corrupted."
                            )

                    elif action == "reindex_embeddings":
                        if not config.embeddings_enabled:
                            return (
                                "Embeddings are disabled.\n"
                                "Set ZETTELKASTEN_EMBEDDINGS_ENABLED=true and install "
                                "the [semantic] extra to enable."
                            )
                        try:
                            stats = self.zettel_service.reindex_embeddings()
                            return (
                                f"Embedding reindex complete.\n"
                                f"Total notes: {stats['total']}\n"
                                f"Embedded: {stats['embedded']}\n"
                                f"Skipped: {stats['skipped']}\n"
                                f"Failed: {stats['failed']}"
                            )
                        except Exception as reindex_err:
                            return f"Reindex failed: {reindex_err}"

                    else:
                        return (
                            f"Invalid action: '{action}'. Valid actions: "
                            "rebuild, sync, backup, list_backups, "
                            "reset_fts, reindex_embeddings"
                        )

                except Exception as e:
                    return self.format_error_response(e)

        @self.mcp.tool(name="zk_restore")
        def zk_restore(backup_path: str, confirm: bool = False) -> str:
            """Restore database from a backup.

            WARNING: This is a destructive operation that will overwrite
            the current database. A safety backup is created automatically
            before restoration.

            Args:
                backup_path: Full path to the backup file to restore
                confirm: Must be True to proceed (safety check)
            """
            with timed_operation("zk_restore") as op:
                try:
                    if not confirm:
                        return (
                            "⚠️ DESTRUCTIVE OPERATION\n\n"
                            "This will overwrite your current database with the backup.\n"
                            "A safety backup will be created first.\n\n"
                            "To proceed, call again with confirm=True"
                        )

                    # List available backups to help user
                    if not backup_path or backup_path.lower() == "list":
                        backups = backup_manager.list_backups()
                        db_backups = [b for b in backups if b["type"] == "database"]
                        if not db_backups:
                            return "No database backups found."

                        output = "Available database backups:\n\n"
                        for b in db_backups:
                            output += f"  {b['path']}\n"
                        output += "\nUse the full path as backup_path parameter."
                        return output

                    success = backup_manager.restore_database(backup_path)
                    if success:
                        op["restored"] = True
                        return (
                            f"✅ Database restored from: {backup_path}\n"
                            "Note: A safety backup was created before restoration.\n"
                            "You may need to restart the MCP server for changes to take effect."
                        )
                    else:
                        op["restored"] = False
                        return f"❌ Restore failed. Check that the backup file exists: {backup_path}"

                except Exception as e:
                    return self.format_error_response(e)

        # ========== Project Management Tools ==========

        @self.mcp.tool(name="zk_create_project")
        def zk_create_project(
            project_id: str,
            name: str,
            description: Optional[str] = None,
            parent_id: Optional[str] = None,
            path: Optional[str] = None,
        ) -> str:
            """Create a new project in the registry.

            Projects organize notes and can be hierarchical (sub-projects).
            Use '/' in project_id for hierarchy: "monorepo/frontend".

            Args:
                project_id: Unique project ID (use '/' for sub-projects, e.g., "monorepo/frontend")
                name: Human-readable display name
                description: Brief description for LLM context (helps route notes correctly)
                parent_id: Parent project ID for sub-projects (optional)
                path: Filesystem path associated with project (optional)
            """
            with timed_operation("zk_create_project", project_id=project_id) as op:
                try:
                    project = Project(
                        id=project_id,
                        name=name,
                        description=description,
                        parent_id=parent_id,
                        path=path,
                    )
                    created = self.project_repository.create(project)
                    op["created"] = True
                    return (
                        f"Project created: {created.id}\n"
                        f"   Name: {created.name}\n"
                        f"   Description: {created.description or '(none)'}\n"
                        f"   Parent: {created.parent_id or '(root project)'}"
                    )
                except ValidationError as e:
                    return f"Error: {e.message}"
                except Exception as e:
                    return self.format_error_response(e)

        @self.mcp.tool(name="zk_list_projects")
        def zk_list_projects(include_note_counts: bool = True) -> str:
            """List all projects in the registry.

            Use this to see available projects before creating notes.

            Args:
                include_note_counts: Include count of notes per project
            """
            with timed_operation("zk_list_projects") as op:
                try:
                    projects = self.project_repository.get_all()
                    op["count"] = len(projects)

                    if not projects:
                        return (
                            "No projects registered yet.\n\n"
                            "Use zk_create_project to create one."
                        )

                    output = f"Projects ({len(projects)}):\n\n"
                    for p in projects:
                        indent = "  " * p.id.count("/")
                        note_count = ""
                        if include_note_counts:
                            count = self.project_repository.get_note_count(p.id)
                            note_count = f" ({count} notes)"

                        output += f"{indent}* {p.id}{note_count}\n"
                        output += f"{indent}  Name: {p.name}\n"
                        if p.description:
                            output += f"{indent}  Description: {p.description}\n"
                        output += "\n"

                    return output.rstrip()
                except Exception as e:
                    return self.format_error_response(e)

        @self.mcp.tool(name="zk_get_project")
        def zk_get_project(project_id: str) -> str:
            """Get details of a specific project.

            Args:
                project_id: The project ID to look up
            """
            with timed_operation("zk_get_project", project_id=project_id) as op:
                try:
                    project = self.project_repository.get(project_id)
                    if not project:
                        return f"Project '{project_id}' not found."

                    note_count = self.project_repository.get_note_count(project_id)
                    children = self.project_repository.search(parent_id=project_id)

                    output = f"Project: {project.id}\n\n"
                    output += f"Name: {project.name}\n"
                    output += f"Description: {project.description or '(none)'}\n"
                    output += f"Parent: {project.parent_id or '(root project)'}\n"
                    output += f"Path: {project.path or '(not set)'}\n"
                    output += f"Notes: {note_count}\n"
                    output += f"Created: {project.created_at.isoformat()}\n"

                    if children:
                        output += f"\nSub-projects ({len(children)}):\n"
                        for child in children:
                            output += f"  * {child.id}\n"

                    if project.metadata:
                        output += (
                            f"\nMetadata: {json.dumps(project.metadata, indent=2)}\n"
                        )

                    return output
                except Exception as e:
                    return self.format_error_response(e)

        @self.mcp.tool(name="zk_delete_project")
        def zk_delete_project(project_id: str, confirm: bool = False) -> str:
            """Delete a project from the registry.

            Cannot delete projects that have notes or sub-projects.
            Move or delete notes first, then delete child projects.

            Args:
                project_id: The project ID to delete
                confirm: Must be True to proceed (safety check)
            """
            with timed_operation("zk_delete_project", project_id=project_id) as op:
                try:
                    project = self.project_repository.get(project_id)
                    if not project:
                        return f"Project '{project_id}' not found."

                    if not confirm:
                        note_count = self.project_repository.get_note_count(project_id)
                        children = self.project_repository.search(parent_id=project_id)
                        return (
                            f"Delete project '{project_id}'?\n\n"
                            f"Notes in project: {note_count}\n"
                            f"Sub-projects: {len(children)}\n\n"
                            "To proceed, call again with confirm=True"
                        )

                    self.project_repository.delete(project_id)
                    op["deleted"] = True
                    return f"Project '{project_id}' deleted."
                except ValidationError as e:
                    return f"Error: {e.message}"
                except Exception as e:
                    return self.format_error_response(e)

    def _register_resources(self) -> None:
        """Register MCP resources."""
        # Currently, we don't define resources for the Zettelkasten server
        pass

    def _register_prompts(self) -> None:
        """Register MCP prompts."""
        # Currently, we don't define prompts for the Zettelkasten server
        pass

    def run(self) -> None:
        """Run the MCP server."""
        self.mcp.run()
