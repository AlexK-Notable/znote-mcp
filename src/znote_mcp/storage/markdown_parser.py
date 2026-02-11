"""Markdown parsing and serialization for Zettelkasten notes.

Handles conversion between Note domain objects and markdown files
with YAML frontmatter. This is extracted from NoteRepository to
keep parsing logic cohesive and independently testable.
"""
import datetime
import logging
import re
from typing import Dict, List, Optional

import frontmatter

from znote_mcp.models.schema import (
    Link,
    LinkType,
    Note,
    NoteType,
    NotePurpose,
    Tag,
    ensure_timezone_aware,
    utc_now,
)

logger = logging.getLogger(__name__)


class MarkdownParser:
    """Parses and serializes Zettelkasten notes as markdown with frontmatter."""

    def parse_note(self, content: str) -> Note:
        """Parse a note from markdown content with YAML frontmatter.

        Args:
            content: Raw markdown string with ``---`` frontmatter delimiters.

        Returns:
            A fully populated Note domain object.

        Raises:
            ValueError: If required fields (id, title) are missing.
        """
        post = frontmatter.loads(content)
        metadata = post.metadata

        # Extract ID
        note_id = metadata.get("id")
        if not note_id:
            raise ValueError("Note ID missing from frontmatter")

        # Extract title
        title = metadata.get("title")
        if not title:
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

        # Extract project
        project = metadata.get("project", "general")

        # Extract purpose
        purpose_str = metadata.get("purpose", NotePurpose.GENERAL.value)
        try:
            note_purpose = NotePurpose(purpose_str)
        except ValueError:
            logger.warning(
                f"Unknown note purpose '{purpose_str}' in note {note_id}, "
                "defaulting to GENERAL"
            )
            note_purpose = NotePurpose.GENERAL

        # Extract plan_id
        plan_id = metadata.get("plan_id")

        # Extract tags
        tags_str = metadata.get("tags", "")
        if isinstance(tags_str, str):
            tag_names = [t.strip() for t in tags_str.split(",") if t.strip()]
        elif isinstance(tags_str, list):
            tag_names = [str(t).strip() for t in tags_str if str(t).strip()]
        else:
            tag_names = []
        tags = [Tag(name=name) for name in tag_names]

        # Extract links from ## Links section
        links = self._parse_links_section(post.content, note_id)

        # Extract timestamps
        created_str = metadata.get("created")
        created_at = (
            ensure_timezone_aware(datetime.datetime.fromisoformat(created_str))
            if created_str
            else utc_now()
        )
        updated_str = metadata.get("updated")
        updated_at = (
            ensure_timezone_aware(datetime.datetime.fromisoformat(updated_str))
            if updated_str
            else created_at
        )

        return Note(
            id=note_id,
            title=title,
            content=post.content,
            note_type=note_type,
            note_purpose=note_purpose,
            project=project,
            plan_id=plan_id,
            tags=tags,
            links=links,
            created_at=created_at,
            updated_at=updated_at,
            metadata={
                k: v
                for k, v in metadata.items()
                if k not in [
                    "id", "title", "type", "purpose", "project",
                    "plan_id", "tags", "created", "updated",
                ]
            },
        )

    def render_to_markdown(self, note: Note) -> str:
        """Convert a Note domain object to markdown with frontmatter.

        Args:
            note: The note to serialize.

        Returns:
            Markdown string with YAML frontmatter.
        """
        metadata: Dict = {
            "id": note.id,
            "title": note.title,
            "type": note.note_type.value,
            "purpose": (
                note.note_purpose.value
                if note.note_purpose
                else NotePurpose.GENERAL.value
            ),
            "project": note.project,
            "tags": [tag.name for tag in note.tags],
            "created": note.created_at.isoformat(),
            "updated": note.updated_at.isoformat(),
        }
        if note.plan_id:
            metadata["plan_id"] = note.plan_id
        metadata.update(note.metadata)

        # Ensure content starts with title heading
        title_heading = f"# {note.title}"
        if note.content.strip().startswith(title_heading):
            content = note.content
        else:
            content = f"{title_heading}\n\n{note.content}"

        # Strip existing Links section
        content = self._strip_links_section(content)

        # Append links section (deduplicated)
        if note.links:
            unique_links: Dict[str, Link] = {}
            for link in note.links:
                key = f"{link.target_id}:{link.link_type.value}"
                unique_links[key] = link
            content += "\n\n## Links\n"
            for link in unique_links.values():
                desc = f" {link.description}" if link.description else ""
                content += f"- {link.link_type.value} [[{link.target_id}]]{desc}\n"

        post = frontmatter.Post(content, **metadata)
        return frontmatter.dumps(post)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_links_section(body: str, note_id: str) -> List[Link]:
        """Extract links from the ``## Links`` section of note content."""
        links: List[Link] = []
        links_section = False

        for line in body.split("\n"):
            line = line.strip()
            if line.startswith("## Links"):
                links_section = True
                continue
            if links_section and line.startswith("## "):
                links_section = False
                continue
            if links_section and line.startswith("- "):
                try:
                    if "[[" in line and "]]" in line:
                        parts = line.split("[[", 1)
                        link_type_str = parts[0].strip()
                        if link_type_str.startswith("- "):
                            link_type_str = link_type_str[2:].strip()
                        id_and_description = parts[1].split("]]", 1)
                        target_id = id_and_description[0].strip()
                        description = None
                        if len(id_and_description) > 1:
                            description = id_and_description[1].strip()
                        try:
                            link_type = LinkType(link_type_str)
                        except ValueError:
                            logger.warning(
                                f"Unknown link type '{link_type_str}' in note "
                                f"{note_id}, defaulting to REFERENCE"
                            )
                            link_type = LinkType.REFERENCE
                        links.append(
                            Link(
                                source_id=note_id,
                                target_id=target_id,
                                link_type=link_type,
                                description=description,
                                created_at=utc_now(),
                            )
                        )
                except (ValueError, IndexError) as e:
                    logger.warning(
                        f"Skipping malformed link in note {note_id}: {line} - {e}"
                    )
        return links

    @staticmethod
    def _strip_links_section(content: str) -> str:
        """Remove all ``## Links`` sections from markdown content."""
        content_parts: List[str] = []
        skip_section = False
        for line in content.split("\n"):
            if line.strip() == "## Links":
                skip_section = True
                continue
            elif skip_section and line.startswith("## "):
                skip_section = False
            if not skip_section:
                content_parts.append(line)
        return "\n".join(content_parts).rstrip()
