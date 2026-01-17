"""MCP server implementation for the Zettelkasten."""
import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from sqlalchemy import exc as sqlalchemy_exc
from mcp.server.fastmcp import Context, FastMCP
from znote_mcp.config import config
from znote_mcp.exceptions import (
    ZettelkastenError,
    NoteNotFoundError,
    NoteValidationError,
    LinkError,
    StorageError,
    ValidationError,
)
from znote_mcp.models.schema import LinkType, Note, NoteType, Tag
from znote_mcp.backup import backup_manager
from znote_mcp.observability import metrics, timed_operation
from znote_mcp.services.search_service import SearchService
from znote_mcp.services.zettel_service import ZettelService

logger = logging.getLogger(__name__)

class ZettelkastenMcpServer:
    """MCP server for Zettelkasten."""
    def __init__(self):
        """Initialize the MCP server."""
        self.mcp = FastMCP(
            config.server_name,
            version=config.server_version
        )
        # Services
        self.zettel_service = ZettelService()
        self.search_service = SearchService(self.zettel_service)
        # Initialize services
        self.initialize()
        # Register tools
        self._register_tools()
        self._register_resources()
        self._register_prompts()

    def initialize(self) -> None:
        """Initialize services."""
        self.zettel_service.initialize()
        self.search_service.initialize()
        logger.info("Zettelkasten MCP server initialized")

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
                extra={"error_details": error.details}
            )
            return f"Error: {error.message}"
        elif isinstance(error, ValueError):
            # Legacy domain validation errors - typically safe to show to users
            logger.error(f"Validation error [{error_id}]: {str(error)}")
            return f"Error: {str(error)}"
        elif isinstance(error, (IOError, OSError)):
            # File system errors - don't expose paths or detailed error messages
            logger.error(f"File system error [{error_id}]: {str(error)}", exc_info=True)
            return f"Error: {str(error)}"
        else:
            # Unexpected errors - log with full stack trace but return generic message
            logger.error(f"Unexpected error [{error_id}]: {str(error)}", exc_info=True)
            return f"Error: {str(error)}"

    def _register_tools(self) -> None:
        """Register MCP tools."""
        # Create a new note
        @self.mcp.tool(name="zk_create_note")
        def zk_create_note(
            title: str,
            content: str,
            note_type: str = "permanent",
            project: str = "general",
            tags: Optional[str] = None
        ) -> str:
            """Create a new Zettelkasten note.
            Args:
                title: The title of the note
                content: The main content of the note
                note_type: Type of note (fleeting, literature, permanent, structure, hub)
                project: Project this note belongs to (used for Obsidian subdirectory organization)
                tags: Comma-separated list of tags (optional)
            """
            with timed_operation("zk_create_note", title=title[:30]) as op:
                try:
                    # Convert note_type string to enum
                    try:
                        note_type_enum = NoteType(note_type.lower())
                    except ValueError:
                        return f"Invalid note type: {note_type}. Valid types are: {', '.join(t.value for t in NoteType)}"

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
                        tags=tag_list,
                    )
                    op["note_id"] = note.id
                    return f"Note created successfully with ID: {note.id} (project: {note.project})"
                except Exception as e:
                    return self.format_error_response(e)

        # Get a note by ID or title
        @self.mcp.tool(name="zk_get_note")
        def zk_get_note(identifier: str) -> str:
            """Retrieve a note by ID or title.
            Args:
                identifier: The ID or title of the note
            """
            with timed_operation("zk_get_note", identifier=identifier[:30]) as op:
                try:
                    identifier = str(identifier)
                    # Try to get by ID first
                    note = self.zettel_service.get_note(identifier)
                    # If not found, try by title
                    if not note:
                        note = self.zettel_service.get_note_by_title(identifier)
                    if not note:
                        op["found"] = False
                        return f"Note not found: {identifier}"

                    op["found"] = True
                    op["note_id"] = note.id
                    # Format the note
                    result = f"# {note.title}\n"
                    result += f"ID: {note.id}\n"
                    result += f"Type: {note.note_type.value}\n"
                    result += f"Created: {note.created_at.isoformat()}\n"
                    result += f"Updated: {note.updated_at.isoformat()}\n"
                    if note.tags:
                        result += f"Tags: {', '.join(tag.name for tag in note.tags)}\n"
                    # Add note content, including the Links section added by _note_to_markdown()
                    result += f"\n{note.content}\n"
                    return result
                except Exception as e:
                    return self.format_error_response(e)

        # Update a note
        @self.mcp.tool(name="zk_update_note")
        def zk_update_note(
            note_id: str,
            title: Optional[str] = None,
            content: Optional[str] = None,
            note_type: Optional[str] = None,
            project: Optional[str] = None,
            tags: Optional[str] = None
        ) -> str:
            """Update an existing note.
            Args:
                note_id: The ID of the note to update
                title: New title (optional)
                content: New content (optional)
                note_type: New note type (optional)
                project: New project (optional, moves note to different Obsidian subdirectory)
                tags: New comma-separated list of tags (optional)
            """
            try:
                # Get the note
                note = self.zettel_service.get_note(str(note_id))
                if not note:
                    return f"Note not found: {note_id}"

                # Convert note_type string to enum if provided
                note_type_enum = None
                if note_type:
                    try:
                        note_type_enum = NoteType(note_type.lower())
                    except ValueError:
                        return f"Invalid note type: {note_type}. Valid types are: {', '.join(t.value for t in NoteType)}"

                # Convert tags string to list if provided
                tag_list = None
                if tags is not None:  # Allow empty string to clear tags
                    tag_list = [t.strip() for t in tags.split(",") if t.strip()]

                # Update the note
                updated_note = self.zettel_service.update_note(
                    note_id=note_id,
                    title=title,
                    content=content,
                    note_type=note_type_enum,
                    project=project,
                    tags=tag_list
                )
                return f"Note updated successfully: {updated_note.id} (project: {updated_note.project})"
            except Exception as e:
                return self.format_error_response(e)

        # Delete a note
        @self.mcp.tool(name="zk_delete_note")
        def zk_delete_note(note_id: str) -> str:
            """Delete a note.
            Args:
                note_id: The ID of the note to delete
            """
            try:
                # Check if note exists
                note = self.zettel_service.get_note(note_id)
                if not note:
                    return f"Note not found: {note_id}"
                
                # Delete the note
                self.zettel_service.delete_note(str(note_id))
                return f"Note deleted successfully: {note_id}"
            except Exception as e:
                return self.format_error_response(e)

        # Add a link between notes
        @self.mcp.tool(name="zk_create_link")
        def zk_create_link(
            source_id: str,
            target_id: str,
            link_type: str = "reference",
            description: Optional[str] = None,
            bidirectional: bool = False
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
                    source_id_str = str(source_id)
                    target_id_str = str(target_id)
                    link_type_enum = LinkType(link_type.lower())
                except ValueError:
                    return f"Invalid link type: {link_type}. Valid types are: {', '.join(t.value for t in LinkType)}"
                
                # Create the link
                source_note, target_note = self.zettel_service.create_link(
                    source_id=source_id,
                    target_id=target_id,
                    link_type=link_type_enum,
                    description=description,
                    bidirectional=bidirectional
                )
                if bidirectional:
                    return f"Bidirectional link created between {source_id} and {target_id}"
                else:
                    return f"Link created from {source_id} to {target_id}"
            except (Exception, sqlalchemy_exc.IntegrityError) as e:
                if "UNIQUE constraint failed" in str(e):
                    return f"A link of this type already exists between these notes. Try a different link type."
                return self.format_error_response(e)

        # Remove a link between notes
        @self.mcp.tool(name="zk_remove_link")
        def zk_remove_link(
            source_id: str,
            target_id: str,
            bidirectional: bool = False
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
                    bidirectional=bidirectional
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
            limit: int = 10
        ) -> str:
            """Search for notes by text, tags, or type.
            Args:
                query: Text to search for in titles and content
                tags: Comma-separated list of tags to filter by
                note_type: Type of note to filter by
                limit: Maximum number of results to return
            """
            with timed_operation("zk_search_notes", query=query[:30] if query else None) as op:
                try:
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
                        text=query,
                        tags=tag_list,
                        note_type=note_type_enum
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
                        output += f"   Created: {note.created_at.strftime('%Y-%m-%d')}\n"
                        # Add a snippet of content (first 150 chars)
                        content_preview = note.content[:150].replace("\n", " ")
                        if len(note.content) > 150:
                            content_preview += "..."
                        output += f"   Preview: {content_preview}\n\n"
                    return output
                except Exception as e:
                    return self.format_error_response(e)

        # Full-text search with FTS5
        @self.mcp.tool(name="zk_fts_search")
        def zk_fts_search(
            query: str,
            limit: int = 20,
            highlight: bool = True
        ) -> str:
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
            with timed_operation("zk_fts_search", query=query[:30] if query else None) as op:
                try:
                    if not query or not query.strip():
                        return "Error: Search query is required."

                    results = self.zettel_service.fts_search(
                        query=query.strip(),
                        limit=limit,
                        highlight=highlight
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
                        if highlight and 'snippet' in result:
                            snippet = result['snippet'].replace('\n', ' ')
                            output += f"   Match: {snippet}\n"
                        output += "\n"

                    return output
                except Exception as e:
                    return self.format_error_response(e)

        # Add a tag to a note
        @self.mcp.tool(name="zk_add_tag")
        def zk_add_tag(note_id: str, tag: str) -> str:
            """Add a tag to an existing note.
            Args:
                note_id: The ID of the note
                tag: The tag to add
            """
            try:
                if not tag or not tag.strip():
                    return "Error: Tag cannot be empty"

                tag = tag.strip()
                note = self.zettel_service.add_tag_to_note(str(note_id), tag)
                return f"Tag '{tag}' added to note '{note.title}' (ID: {note.id})"
            except Exception as e:
                return self.format_error_response(e)

        # Remove a tag from a note
        @self.mcp.tool(name="zk_remove_tag")
        def zk_remove_tag(note_id: str, tag: str) -> str:
            """Remove a tag from a note.
            Args:
                note_id: The ID of the note
                tag: The tag to remove
            """
            try:
                if not tag or not tag.strip():
                    return "Error: Tag cannot be empty"

                tag = tag.strip()
                note = self.zettel_service.remove_tag_from_note(str(note_id), tag)
                return f"Tag '{tag}' removed from note '{note.title}' (ID: {note.id})"
            except Exception as e:
                return self.format_error_response(e)

        # Export a note
        @self.mcp.tool(name="zk_export_note")
        def zk_export_note(note_id: str, format: str = "markdown") -> str:
            """Export a note in the specified format.
            Args:
                note_id: The ID of the note to export
                format: Export format (currently only 'markdown' is supported)
            """
            try:
                content = self.zettel_service.export_note(str(note_id), format)
                return content
            except Exception as e:
                return self.format_error_response(e)

        # ========== Bulk Operations ==========

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
            import json
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

        @self.mcp.tool(name="zk_bulk_delete_notes")
        def zk_bulk_delete_notes(note_ids: str) -> str:
            """Delete multiple notes in a single batch operation.

            Args:
                note_ids: Comma-separated list of note IDs to delete.
            """
            with timed_operation("zk_bulk_delete_notes") as op:
                try:
                    ids = [id.strip() for id in note_ids.split(",") if id.strip()]
                    if not ids:
                        return "Error: No note IDs provided."

                    op["note_count"] = len(ids)
                    deleted = self.zettel_service.bulk_delete_notes(ids)

                    return f"Successfully deleted {deleted} notes."
                except Exception as e:
                    return self.format_error_response(e)

        @self.mcp.tool(name="zk_bulk_add_tags")
        def zk_bulk_add_tags(note_ids: str, tags: str) -> str:
            """Add tags to multiple notes at once.

            Args:
                note_ids: Comma-separated list of note IDs.
                tags: Comma-separated list of tags to add.
            """
            with timed_operation("zk_bulk_add_tags") as op:
                try:
                    ids = [id.strip() for id in note_ids.split(",") if id.strip()]
                    tag_list = [t.strip() for t in tags.split(",") if t.strip()]

                    if not ids:
                        return "Error: No note IDs provided."
                    if not tag_list:
                        return "Error: No tags provided."

                    op["note_count"] = len(ids)
                    op["tag_count"] = len(tag_list)
                    updated = self.zettel_service.bulk_add_tags(ids, tag_list)

                    return f"Added tags [{', '.join(tag_list)}] to {updated} notes."
                except Exception as e:
                    return self.format_error_response(e)

        @self.mcp.tool(name="zk_bulk_remove_tags")
        def zk_bulk_remove_tags(note_ids: str, tags: str) -> str:
            """Remove tags from multiple notes at once.

            Args:
                note_ids: Comma-separated list of note IDs.
                tags: Comma-separated list of tags to remove.
            """
            with timed_operation("zk_bulk_remove_tags") as op:
                try:
                    ids = [id.strip() for id in note_ids.split(",") if id.strip()]
                    tag_list = [t.strip() for t in tags.split(",") if t.strip()]

                    if not ids:
                        return "Error: No note IDs provided."
                    if not tag_list:
                        return "Error: No tags provided."

                    op["note_count"] = len(ids)
                    op["tag_count"] = len(tag_list)
                    updated = self.zettel_service.bulk_remove_tags(ids, tag_list)

                    return f"Removed tags [{', '.join(tag_list)}] from {updated} notes."
                except Exception as e:
                    return self.format_error_response(e)

        @self.mcp.tool(name="zk_bulk_move_to_project")
        def zk_bulk_move_to_project(note_ids: str, project: str) -> str:
            """Move multiple notes to a different project.

            Args:
                note_ids: Comma-separated list of note IDs.
                project: Target project name.
            """
            with timed_operation("zk_bulk_move_to_project") as op:
                try:
                    ids = [id.strip() for id in note_ids.split(",") if id.strip()]

                    if not ids:
                        return "Error: No note IDs provided."
                    if not project or not project.strip():
                        return "Error: Project name is required."

                    op["note_count"] = len(ids)
                    op["project"] = project.strip()
                    updated = self.zettel_service.bulk_update_project(ids, project.strip())

                    return f"Moved {updated} notes to project '{project}'."
                except Exception as e:
                    return self.format_error_response(e)

        # ========== NEW CONSOLIDATED TOOLS ==========

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
            offset: int = 0
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
                        all_notes = self.zettel_service.get_all_notes()
                        total_count = len(all_notes)
                        if not all_notes:
                            return "No notes found in the Zettelkasten."

                        # Sort notes
                        if sort_by == "created_at":
                            all_notes.sort(key=lambda n: n.created_at, reverse=descending)
                        elif sort_by == "title":
                            all_notes.sort(key=lambda n: n.title.lower(), reverse=descending)
                        else:
                            all_notes.sort(key=lambda n: n.updated_at, reverse=descending)

                        notes = all_notes[offset:offset + limit]
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
                            output += f"\n(Use offset={offset + limit} to see more notes)"
                        return output

                    elif mode == "by_date":
                        start_datetime = None
                        if start_date:
                            start_datetime = datetime.fromisoformat(f"{start_date}T00:00:00")
                        end_datetime = None
                        if end_date:
                            end_datetime = datetime.fromisoformat(f"{end_date}T23:59:59")

                        notes = self.search_service.find_notes_by_date_range(
                            start_date=start_datetime,
                            end_date=end_datetime,
                            use_updated=use_updated
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

                        output = f"Notes in project '{project}' ({len(notes)} results):\n\n"
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
                            return "No orphaned notes found. All notes have connections!"

                        output = f"Orphaned notes ({len(orphans)} with no connections):\n\n"
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
            limit: int = 10
        ) -> str:
            """Find notes related to a specific note.

            Args:
                note_id: ID of the reference note
                mode: Relation type:
                    - "linked": Notes connected via links
                    - "similar": Notes with shared tags/links (similarity score)
                direction: Link direction for mode="linked" (outgoing, incoming, both)
                threshold: Similarity threshold 0.0-1.0 for mode="similar"
                limit: Maximum results to return
            """
            with timed_operation("zk_find_related", mode=mode, note_id=note_id[:20]) as op:
                try:
                    note_id = str(note_id)

                    # Verify note exists
                    note = self.zettel_service.get_note(note_id)
                    if not note:
                        return f"Note not found: {note_id}"

                    if mode == "linked":
                        if direction not in ["outgoing", "incoming", "both"]:
                            return f"Invalid direction: '{direction}'. Use: outgoing, incoming, both"

                        linked_notes = self.zettel_service.get_linked_notes(note_id, direction)
                        op["result_count"] = len(linked_notes)

                        if not linked_notes:
                            return f"No {direction} links found for note '{note.title}'"

                        output = f"Notes linked to '{note.title}' ({direction}):\n\n"
                        for i, linked_note in enumerate(linked_notes, 1):
                            output += f"{i}. {linked_note.title} (ID: {linked_note.id})\n"
                            if linked_note.tags:
                                output += f"   Tags: {', '.join(tag.name for tag in linked_note.tags)}\n"

                            # Try to show link type
                            if direction in ["outgoing", "both"]:
                                for link in note.links:
                                    if str(link.target_id) == str(linked_note.id):
                                        output += f"   Link type: {link.link_type.value}\n"
                                        if link.description:
                                            output += f"   Description: {link.description}\n"
                                        break
                            output += "\n"
                        return output

                    elif mode == "similar":
                        similar_notes = self.zettel_service.find_similar_notes(note_id, threshold)[:limit]
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

                    else:
                        return f"Invalid mode: '{mode}'. Valid modes: linked, similar"

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
                        all_notes = self.zettel_service.get_all_notes()
                        total = len(all_notes)

                        # Count by type
                        by_type = {}
                        by_project = {}
                        for note in all_notes:
                            t = note.note_type.value
                            by_type[t] = by_type.get(t, 0) + 1
                            p = note.project
                            by_project[p] = by_project.get(p, 0) + 1

                        output += f"## Summary\n"
                        output += f"**Total Notes:** {total}\n\n"

                        if by_type:
                            output += "**By Type:**\n"
                            for t, count in sorted(by_type.items(), key=lambda x: -x[1]):
                                output += f"  - {t}: {count}\n"
                            output += "\n"

                        if by_project:
                            output += "**By Project:**\n"
                            for p, count in sorted(by_project.items(), key=lambda x: -x[1]):
                                output += f"  - {p}: {count}\n"
                            output += "\n"

                        # Link stats
                        central = self.search_service.find_central_notes(5)
                        orphans = self.search_service.find_orphaned_notes()
                        output += f"**Connections:** {len(central)} notes with links, {len(orphans)} orphans\n\n"

                    # Tags section
                    if include_all or "tags" in requested:
                        tags = self.zettel_service.get_all_tags()
                        output += f"## Tags ({len(tags)} total)\n"
                        if tags:
                            tags.sort(key=lambda t: t.name.lower())
                            tag_names = [tag.name for tag in tags]
                            output += ", ".join(tag_names) + "\n\n"
                        else:
                            output += "No tags defined.\n\n"

                    # Health section
                    if include_all or "health" in requested:
                        health = self.zettel_service.check_database_health()
                        output += "## Health\n"
                        status_icon = "✅" if health["healthy"] else "❌"
                        output += f"**Overall:** {status_icon} {'Healthy' if health['healthy'] else 'Issues detected'}\n"
                        output += f"**SQLite:** {'OK' if health['sqlite_ok'] else 'ERROR'}\n"
                        output += f"**FTS5:** {'OK' if health['fts_ok'] else 'Degraded'}\n"
                        output += f"**DB Notes:** {health['note_count']} | **Files:** {health['file_count']}\n"
                        if health.get("issues"):
                            output += f"**Issues:** {', '.join(health['issues'])}\n"
                        output += "\n"

                    # Metrics section
                    if include_all or "metrics" in requested:
                        summary = metrics.get_summary()
                        output += "## Server Metrics\n"
                        output += f"**Uptime:** {summary['uptime_seconds']:.0f} seconds\n"
                        output += f"**Operations:** {summary['total_operations']}\n"
                        output += f"**Success Rate:** {summary['overall_success_rate']:.1%}\n"
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
        def zk_system(
            action: str,
            backup_label: Optional[str] = None
        ) -> str:
            """System administration operations.

            Args:
                action: Operation to perform:
                    - "rebuild": Rebuild database index from markdown files
                    - "sync": Sync notes to Obsidian vault
                    - "backup": Create database and notes backup
                    - "list_backups": List available backups
                    - "health": Detailed health check
                    - "reset_fts": Reset FTS5 availability after manual repair
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
                        return f"Successfully synced {synced_count} notes to Obsidian vault."

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
                            output += f"   Type: {b['type']} | Size: {b['size_mb']} MB\n"
                            output += f"   Created: {b['created_at']}\n\n"
                        return output

                    elif action == "health":
                        health = self.zettel_service.check_database_health()
                        output = "## Detailed Health Check\n\n"
                        output += f"**SQLite Integrity:** {'PASS' if health['sqlite_ok'] else 'FAIL'}\n"
                        output += f"**FTS5 Integrity:** {'PASS' if health['fts_ok'] else 'FAIL'}\n"
                        output += f"**Notes in DB:** {health['note_count']}\n"
                        output += f"**Markdown Files:** {health['file_count']}\n"
                        output += f"**Sync Needed:** {'Yes' if health.get('needs_sync') else 'No'}\n"

                        if health.get("issues"):
                            output += f"\n**Issues:**\n"
                            for issue in health["issues"]:
                                output += f"  - {issue}\n"

                        if health.get("critical_issues"):
                            output += f"\n**Critical Issues:**\n"
                            for issue in health["critical_issues"]:
                                output += f"  - ⚠️ {issue}\n"

                        return output

                    elif action == "reset_fts":
                        success = self.zettel_service.reset_fts_availability()
                        if success:
                            return "FTS5 availability reset. Full-text search is now enabled."
                        else:
                            return "FTS5 reset failed. The index may still be corrupted."

                    else:
                        return (
                            f"Invalid action: '{action}'. Valid actions: "
                            "rebuild, sync, backup, list_backups, health, reset_fts"
                        )

                except ValueError as e:
                    return str(e)
                except Exception as e:
                    return self.format_error_response(e)

        @self.mcp.tool(name="zk_restore")
        def zk_restore(
            backup_path: str,
            confirm: bool = False
        ) -> str:
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
