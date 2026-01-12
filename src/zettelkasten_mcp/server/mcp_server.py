"""MCP server implementation for the Zettelkasten."""
import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from sqlalchemy import exc as sqlalchemy_exc
from mcp.server.fastmcp import Context, FastMCP
from zettelkasten_mcp.config import config
from zettelkasten_mcp.exceptions import (
    ZettelkastenError,
    NoteNotFoundError,
    NoteValidationError,
    LinkError,
    StorageError,
    ValidationError,
)
from zettelkasten_mcp.models.schema import LinkType, Note, NoteType, Tag
from zettelkasten_mcp.observability import metrics, timed_operation, get_logger
from zettelkasten_mcp.services.search_service import SearchService
from zettelkasten_mcp.services.zettel_service import ZettelService

logger = logging.getLogger(__name__)
obs_logger = get_logger("mcp_server")

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
        self.zk_create_link = zk_create_link

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

        # Get linked notes
        @self.mcp.tool(name="zk_get_linked_notes")
        def zk_get_linked_notes(
            note_id: str,
            direction: str = "both"
        ) -> str:
            """Get notes linked to/from a note.
            Args:
                note_id: ID of the note
                direction: Direction of links (outgoing, incoming, both)
            """
            try:
                if direction not in ["outgoing", "incoming", "both"]:
                    return f"Invalid direction: {direction}. Use 'outgoing', 'incoming', or 'both'."
                # Get linked notes
                linked_notes = self.zettel_service.get_linked_notes(str(note_id), direction)
                if not linked_notes:
                    return f"No {direction} links found for note {note_id}."
                # Format results
                output = f"Found {len(linked_notes)} {direction} linked notes for {note_id}:\n\n"
                for i, note in enumerate(linked_notes, 1):
                    output += f"{i}. {note.title} (ID: {note.id})\n"
                    if note.tags:
                        output += f"   Tags: {', '.join(tag.name for tag in note.tags)}\n"
                    # Try to determine link type
                    if direction in ["outgoing", "both"]:
                        # Check source note's outgoing links
                        source_note = self.zettel_service.get_note(str(note_id))
                        if source_note:
                            for link in source_note.links:
                                if str(link.target_id) == str(note.id):  # Explicit string conversion for comparison
                                    output += f"   Link type: {link.link_type.value}\n"
                                    if link.description:
                                        output += f"   Description: {link.description}\n"
                                    break
                    if direction in ["incoming", "both"]:
                        # Check target note's outgoing links
                        for link in note.links:
                            if str(link.target_id) == str(note_id):  # Explicit string conversion for comparison
                                output += f"   Incoming link type: {link.link_type.value}\n"
                                if link.description:
                                    output += f"   Description: {link.description}\n"
                                break
                    output += "\n"
                return output
            except Exception as e:
                return self.format_error_response(e)
        self.zk_get_linked_notes = zk_get_linked_notes

        # Get all tags
        @self.mcp.tool(name="zk_get_all_tags")
        def zk_get_all_tags() -> str:
            """Get all tags in the Zettelkasten."""
            try:
                tags = self.zettel_service.get_all_tags()
                if not tags:
                    return "No tags found in the Zettelkasten."
                
                # Format results
                output = f"Found {len(tags)} tags:\n\n"
                # Sort alphabetically
                tags.sort(key=lambda t: t.name.lower())
                for i, tag in enumerate(tags, 1):
                    output += f"{i}. {tag.name}\n"
                return output
            except Exception as e:
                return self.format_error_response(e)

        # Find similar notes
        @self.mcp.tool(name="zk_find_similar_notes")
        def zk_find_similar_notes(
            note_id: str,
            threshold: float = 0.3,
            limit: int = 5
        ) -> str:
            """Find notes similar to a given note.
            Args:
                note_id: ID of the reference note
                threshold: Similarity threshold (0.0-1.0)
                limit: Maximum number of results to return
            """
            try:
                # Get similar notes
                similar_notes = self.zettel_service.find_similar_notes(str(note_id), threshold)
                # Limit results
                similar_notes = similar_notes[:limit]
                if not similar_notes:
                    return f"No similar notes found for {note_id} with threshold {threshold}."
                
                # Format results
                output = f"Found {len(similar_notes)} similar notes for {note_id}:\n\n"
                for i, (note, similarity) in enumerate(similar_notes, 1):
                    output += f"{i}. {note.title} (ID: {note.id})\n"
                    output += f"   Similarity: {similarity:.2f}\n"
                    if note.tags:
                        output += f"   Tags: {', '.join(tag.name for tag in note.tags)}\n"
                    # Add a snippet of content (first 100 chars)
                    content_preview = note.content[:100].replace("\n", " ")
                    if len(note.content) > 100:
                        content_preview += "..."
                    output += f"   Preview: {content_preview}\n\n"
                return output
            except Exception as e:
                return self.format_error_response(e)

        # Find central notes
        @self.mcp.tool(name="zk_find_central_notes")
        def zk_find_central_notes(limit: int = 10) -> str:
            """Find notes with the most connections (incoming + outgoing links).
            Notes are ranked by their total number of connections, determining
            their centrality in the knowledge network. Due to database constraints,
            only one link of each type is counted between any pair of notes.

            Args:
                limit: Maximum number of results to return (default: 10)
            """
            try:
                # Get central notes
                central_notes = self.search_service.find_central_notes(limit)
                if not central_notes:
                    return "No notes found with connections."
                
                # Format results
                output = "Central notes in the Zettelkasten (most connected):\n\n"
                for i, (note, connection_count) in enumerate(central_notes, 1):
                    output += f"{i}. {note.title} (ID: {note.id})\n"
                    output += f"   Connections: {connection_count}\n"
                    if note.tags:
                        output += f"   Tags: {', '.join(tag.name for tag in note.tags)}\n"
                    # Add a snippet of content (first 100 chars)
                    content_preview = note.content[:100].replace("\n", " ")
                    if len(note.content) > 100:
                        content_preview += "..."
                    output += f"   Preview: {content_preview}\n\n"
                return output
            except Exception as e:
                return self.format_error_response(e)

        # Find orphaned notes
        @self.mcp.tool(name="zk_find_orphaned_notes")
        def zk_find_orphaned_notes() -> str:
            """Find notes with no connections to other notes."""
            try:
                # Get orphaned notes
                orphans = self.search_service.find_orphaned_notes()
                if not orphans:
                    return "No orphaned notes found."
                
                # Format results
                output = f"Found {len(orphans)} orphaned notes:\n\n"
                for i, note in enumerate(orphans, 1):
                    output += f"{i}. {note.title} (ID: {note.id})\n"
                    if note.tags:
                        output += f"   Tags: {', '.join(tag.name for tag in note.tags)}\n"
                    # Add a snippet of content (first 100 chars)
                    content_preview = note.content[:100].replace("\n", " ")
                    if len(note.content) > 100:
                        content_preview += "..."
                    output += f"   Preview: {content_preview}\n\n"
                return output
            except Exception as e:
                return self.format_error_response(e)

        # List notes by date range
        @self.mcp.tool(name="zk_list_notes_by_date")
        def zk_list_notes_by_date(
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            use_updated: bool = False,
            limit: int = 10
        ) -> str:
            """List notes created or updated within a date range.
            Args:
                start_date: Start date in ISO format (YYYY-MM-DD)
                end_date: End date in ISO format (YYYY-MM-DD)
                use_updated: Whether to use updated_at instead of created_at
                limit: Maximum number of results to return
            """
            try:
                # Parse dates
                start_datetime = None
                if start_date:
                    start_datetime = datetime.fromisoformat(f"{start_date}T00:00:00")
                end_datetime = None
                if end_date:
                    end_datetime = datetime.fromisoformat(f"{end_date}T23:59:59")
                
                # Get notes
                notes = self.search_service.find_notes_by_date_range(
                    start_date=start_datetime,
                    end_date=end_datetime,
                    use_updated=use_updated
                )
                
                # Limit results
                notes = notes[:limit]
                if not notes:
                    date_type = "updated" if use_updated else "created"
                    date_range = ""
                    if start_date and end_date:
                        date_range = f" between {start_date} and {end_date}"
                    elif start_date:
                        date_range = f" after {start_date}"
                    elif end_date:
                        date_range = f" before {end_date}"
                    return f"No notes found {date_type}{date_range}."
                
                # Format results
                date_type = "updated" if use_updated else "created"
                output = f"Notes {date_type}"
                if start_date or end_date:
                    if start_date and end_date:
                        output += f" between {start_date} and {end_date}"
                    elif start_date:
                        output += f" after {start_date}"
                    elif end_date:
                        output += f" before {end_date}"
                output += f" (showing {len(notes)} results):\n\n"
                for i, note in enumerate(notes, 1):
                    date = note.updated_at if use_updated else note.created_at
                    output += f"{i}. {note.title} (ID: {note.id})\n"
                    output += f"   {date_type.capitalize()}: {date.strftime('%Y-%m-%d %H:%M')}\n"
                    if note.tags:
                        output += f"   Tags: {', '.join(tag.name for tag in note.tags)}\n"
                    # Add a snippet of content (first 100 chars)
                    content_preview = note.content[:100].replace("\n", " ")
                    if len(note.content) > 100:
                        content_preview += "..."
                    output += f"   Preview: {content_preview}\n\n"
                return output
            except ValueError as e:
                # Special handling for date parsing errors
                logger.error(f"Date parsing error: {str(e)}")
                return f"Error parsing date: {str(e)}"
            except Exception as e:
                return self.format_error_response(e)

        # Rebuild the index
        @self.mcp.tool(name="zk_rebuild_index")
        def zk_rebuild_index() -> str:
            """Rebuild the database index and FTS5 search index from files."""
            try:
                # Get count before rebuild
                note_count_before = len(self.zettel_service.get_all_notes())

                # Perform the rebuild (includes FTS5 index rebuild)
                self.zettel_service.rebuild_index()

                # Get count after rebuild
                note_count_after = len(self.zettel_service.get_all_notes())

                # Return a detailed success message
                return (
                    f"Database index rebuilt successfully.\n"
                    f"Notes processed: {note_count_after}\n"
                    f"FTS5 search index: rebuilt\n"
                    f"Change in note count: {note_count_after - note_count_before}"
                )
            except Exception as e:
                # Provide a detailed error message
                logger.error(f"Failed to rebuild index: {e}", exc_info=True)
                return self.format_error_response(e)

        # List all notes with pagination
        @self.mcp.tool(name="zk_list_all_notes")
        def zk_list_all_notes(
            limit: int = 50,
            offset: int = 0,
            sort_by: str = "updated_at",
            descending: bool = True
        ) -> str:
            """List all notes in the Zettelkasten with pagination.
            Args:
                limit: Maximum number of notes to return (default: 50, max: 200)
                offset: Number of notes to skip (for pagination)
                sort_by: Field to sort by (created_at, updated_at, title)
                descending: Sort in descending order (newest first)
            """
            try:
                # Validate limit
                limit = min(max(1, limit), 200)
                offset = max(0, offset)

                # Get all notes
                all_notes = self.zettel_service.get_all_notes()
                total_count = len(all_notes)

                if not all_notes:
                    return "No notes found in the Zettelkasten."

                # Sort notes
                if sort_by == "created_at":
                    all_notes.sort(key=lambda n: n.created_at, reverse=descending)
                elif sort_by == "updated_at":
                    all_notes.sort(key=lambda n: n.updated_at, reverse=descending)
                elif sort_by == "title":
                    all_notes.sort(key=lambda n: n.title.lower(), reverse=descending)
                else:
                    all_notes.sort(key=lambda n: n.updated_at, reverse=descending)

                # Apply pagination
                notes = all_notes[offset:offset + limit]

                if not notes:
                    return f"No notes found at offset {offset}. Total notes: {total_count}"

                # Format results
                output = f"Notes ({offset + 1}-{offset + len(notes)} of {total_count}):\n\n"
                for i, note in enumerate(notes, offset + 1):
                    output += f"{i}. {note.title} (ID: {note.id})\n"
                    output += f"   Type: {note.note_type.value}\n"
                    output += f"   Updated: {note.updated_at.strftime('%Y-%m-%d %H:%M')}\n"
                    if note.tags:
                        output += f"   Tags: {', '.join(tag.name for tag in note.tags)}\n"
                    output += "\n"

                # Add pagination hint if there are more notes
                if offset + limit < total_count:
                    output += f"\n(Use offset={offset + limit} to see more notes)"

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

        # Sync notes to Obsidian vault
        @self.mcp.tool(name="zk_sync_to_obsidian")
        def zk_sync_to_obsidian() -> str:
            """Sync all notes to the configured Obsidian vault.

            Re-mirrors all existing zettelkasten notes to the Obsidian vault.
            Uses note titles as filenames, so existing files are overwritten.
            Requires ZETTELKASTEN_OBSIDIAN_VAULT to be set in the environment.
            """
            try:
                synced_count = self.zettel_service.sync_to_obsidian()
                return (
                    f"Successfully synced {synced_count} notes to Obsidian vault.\n"
                    f"Notes are saved with their titles as filenames."
                )
            except ValueError as e:
                # Configuration error (vault not set)
                return str(e)
            except Exception as e:
                logger.error(f"Failed to sync to Obsidian: {e}", exc_info=True)
                return self.format_error_response(e)

        # Get server metrics and health status
        @self.mcp.tool(name="zk_get_metrics")
        def zk_get_metrics(include_details: bool = False) -> str:
            """Get server metrics and health status.

            Returns operation counts, timing statistics, and error rates
            for monitoring server performance.

            Args:
                include_details: Include per-operation breakdown (default: False)
            """
            try:
                summary = metrics.get_summary()

                output = "## Zettelkasten Server Health\n\n"
                output += f"**Uptime:** {summary['uptime_seconds']:.1f} seconds\n"
                output += f"**Total Operations:** {summary['total_operations']}\n"
                output += f"**Success Rate:** {summary['overall_success_rate']:.1%}\n"
                output += f"**Errors:** {summary['total_errors']}\n"

                if include_details and summary['operations_tracked']:
                    output += "\n### Operation Details\n\n"
                    op_metrics = metrics.get_metrics()

                    for op_name in sorted(op_metrics.keys()):
                        m = op_metrics[op_name]
                        output += f"**{op_name}**\n"
                        output += f"  - Count: {m['count']} ({m['success_count']} success, {m['error_count']} errors)\n"
                        output += f"  - Avg Duration: {m['avg_duration_ms']:.2f}ms\n"
                        output += f"  - Range: {m['min_duration_ms']:.2f}ms - {m['max_duration_ms']:.2f}ms\n"
                        if m['last_error']:
                            output += f"  - Last Error: {m['last_error']} at {m['last_error_time']}\n"
                        output += "\n"

                return output
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
