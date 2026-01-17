"""Data models for the Zettelkasten MCP server."""
import sys
import time
import datetime
from datetime import datetime as dt, timezone
import random
import inspect
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union
from pydantic import BaseModel, Field, field_validator
import threading
import re

# Regex pattern for valid note IDs (alphanumeric, underscores, hyphens, T separator)
# Matches format: YYYYMMDDTHHMMSSssssssccc or similar safe patterns
SAFE_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_\-T]+$')

def validate_safe_path_component(value: str, field_name: str = "value") -> str:
    """Validate that a value is safe to use as a filesystem path component.

    Prevents path traversal attacks by rejecting:
    - Path separators (/, \\)
    - Parent directory references (..)
    - Current directory references (single .)
    - Any characters outside alphanumeric, underscore, hyphen, T

    Args:
        value: The string to validate
        field_name: Name of the field for error messages

    Returns:
        The validated value (unchanged)

    Raises:
        ValueError: If the value contains unsafe characters
    """
    if not value:
        raise ValueError(f"{field_name} cannot be empty")

    # Check for path traversal attempts
    if '..' in value:
        raise ValueError(f"{field_name} cannot contain '..' (path traversal)")

    if '/' in value or '\\' in value:
        raise ValueError(f"{field_name} cannot contain path separators")

    # Validate against safe pattern
    if not SAFE_ID_PATTERN.match(value):
        raise ValueError(
            f"{field_name} contains invalid characters. "
            "Only alphanumeric characters, underscores, hyphens, and 'T' are allowed."
        )

    return value


def utc_now() -> datetime.datetime:
    """Get current UTC time as timezone-aware datetime.

    Returns:
        Current time with UTC timezone info attached.
    """
    return datetime.datetime.now(timezone.utc)


def ensure_timezone_aware(dt_value: datetime.datetime) -> datetime.datetime:
    """Ensure a datetime is timezone-aware, treating naive datetimes as UTC.

    This is used for migration compatibility: existing naive datetimes in the
    database are assumed to be UTC.

    Args:
        dt_value: A datetime that may or may not have timezone info.

    Returns:
        The same datetime with UTC timezone if it was naive, otherwise unchanged.
    """
    if dt_value is None:
        return utc_now()
    if dt_value.tzinfo is None:
        return dt_value.replace(tzinfo=timezone.utc)
    return dt_value


# Thread-safe counter for uniqueness
_id_lock = threading.Lock()
_last_timestamp = 0
_counter = 0

def generate_id() -> str:
    """Generate an ISO 8601 compliant timestamp-based ID with guaranteed uniqueness.

    Returns:
        A string in format "YYYYMMDDTHHMMSSsssssscccccc" where:
        - YYYYMMDD is the date
        - T is the ISO 8601 date/time separator
        - HHMMSS is the time (hours, minutes, seconds)
        - ssssss is the 6-digit microsecond component
        - cccccc is a 6-digit counter for uniqueness within the same microsecond

    The format follows ISO 8601 basic format with extended precision,
    allowing up to 1 trillion unique IDs per second (1 million per microsecond).
    """
    global _last_timestamp, _counter

    with _id_lock:
        # Get current timestamp with microsecond precision (UTC for consistency)
        now = utc_now()
        # Create a timestamp in microseconds
        current_timestamp = int(now.timestamp() * 1_000_000)

        # If multiple IDs generated in same microsecond, increment counter
        if current_timestamp == _last_timestamp:
            _counter += 1
        else:
            _last_timestamp = current_timestamp
            _counter = 0

        # Ensure counter doesn't overflow our 6 digits (supports 1M IDs/microsecond)
        _counter %= 1_000_000

        # Format as ISO 8601 basic format with microseconds and counter
        date_time = now.strftime('%Y%m%dT%H%M%S')
        microseconds = now.microsecond

        return f"{date_time}{microseconds:06d}{_counter:06d}"

class LinkType(str, Enum):
    """Types of links between notes."""
    REFERENCE = "reference"        # Simple reference to another note
    EXTENDS = "extends"            # Current note extends another note
    EXTENDED_BY = "extended_by"    # Current note is extended by another note
    REFINES = "refines"            # Current note refines another note
    REFINED_BY = "refined_by"      # Current note is refined by another note
    CONTRADICTS = "contradicts"    # Current note contradicts another note
    CONTRADICTED_BY = "contradicted_by"  # Current note is contradicted by another note
    QUESTIONS = "questions"        # Current note questions another note
    QUESTIONED_BY = "questioned_by"  # Current note is questioned by another note
    SUPPORTS = "supports"          # Current note supports another note
    SUPPORTED_BY = "supported_by"  # Current note is supported by another note
    RELATED = "related"            # Notes are related in some way

class Link(BaseModel):
    """A link between two notes."""
    source_id: str = Field(..., description="ID of the source note")
    target_id: str = Field(..., description="ID of the target note")
    link_type: LinkType = Field(default=LinkType.REFERENCE, description="Type of link")
    description: Optional[str] = Field(default=None, description="Optional description of the link")
    created_at: datetime.datetime = Field(
        default_factory=utc_now,
        description="When the link was created (UTC)"
    )
    
    model_config = {
        "validate_assignment": True,
        "extra": "forbid",
        "frozen": True  # Links are immutable
    }

class NoteType(str, Enum):
    """Types of notes in a Zettelkasten."""
    FLEETING = "fleeting"    # Quick, temporary notes
    LITERATURE = "literature"  # Notes from reading material
    PERMANENT = "permanent"  # Permanent, well-formulated notes
    STRUCTURE = "structure"  # Structure/index notes that organize other notes
    HUB = "hub"              # Hub notes that serve as entry points

class Tag(BaseModel):
    """A tag for categorizing notes."""
    name: str = Field(..., description="Tag name")
    
    model_config = {
        "validate_assignment": True,
        "frozen": True
    }
    
    def __str__(self) -> str:
        """Return string representation of tag."""
        return self.name

class Note(BaseModel):
    """A Zettelkasten note."""
    id: str = Field(default_factory=generate_id, description="Unique ID of the note")
    title: str = Field(..., description="Title of the note")
    content: str = Field(..., description="Content of the note")
    note_type: NoteType = Field(default=NoteType.PERMANENT, description="Type of note")
    project: str = Field(default="general", description="Project this note belongs to")
    tags: List[Tag] = Field(default_factory=list, description="Tags for categorization")
    links: List[Link] = Field(default_factory=list, description="Links to other notes")
    created_at: datetime.datetime = Field(
        default_factory=utc_now,
        description="When the note was created (UTC)"
    )
    updated_at: datetime.datetime = Field(
        default_factory=utc_now,
        description="When the note was last updated (UTC)"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for the note"
    )
    
    model_config = {
        "validate_assignment": True,
        "extra": "forbid"
    }
    
    @field_validator("id")
    @classmethod
    def validate_id(cls, v: str) -> str:
        """Validate that the ID is safe for filesystem use."""
        return validate_safe_path_component(v, "Note ID")

    @field_validator("project")
    @classmethod
    def validate_project(cls, v: str) -> str:
        """Validate that the project name is safe for filesystem use."""
        # Allow empty string to default to "general"
        if not v or not v.strip():
            return "general"
        return validate_safe_path_component(v, "Project name")

    @field_validator("title")
    @classmethod
    def validate_title(cls, v: str) -> str:
        """Validate that the title is not empty."""
        if not v.strip():
            raise ValueError("Title cannot be empty")
        return v
    
    def add_tag(self, tag: Union[str, Tag]) -> None:
        """Add a tag to the note."""
        if isinstance(tag, str):
            tag = Tag(name=tag)
        # Check if tag already exists
        tag_names = {t.name for t in self.tags}
        if tag.name not in tag_names:
            self.tags.append(tag)
            self.updated_at = utc_now()
    
    def remove_tag(self, tag: Union[str, Tag]) -> None:
        """Remove a tag from the note."""
        tag_name = tag.name if isinstance(tag, Tag) else tag
        self.tags = [t for t in self.tags if t.name != tag_name]
        self.updated_at = utc_now()
    
    def add_link(self, target_id: str, link_type: LinkType = LinkType.REFERENCE, 
                description: Optional[str] = None) -> None:
        """Add a link to another note."""
        # Check if link already exists
        for link in self.links:
            if link.target_id == target_id and link.link_type == link_type:
                return  # Link already exists
        link = Link(
            source_id=self.id,
            target_id=target_id,
            link_type=link_type,
            description=description
        )
        self.links.append(link)
        self.updated_at = utc_now()
    
    def remove_link(self, target_id: str, link_type: Optional[LinkType] = None) -> None:
        """Remove a link to another note."""
        if link_type:
            self.links = [
                link for link in self.links 
                if not (link.target_id == target_id and link.link_type == link_type)
            ]
        else:
            self.links = [link for link in self.links if link.target_id != target_id]
        self.updated_at = utc_now()
    
    def get_linked_note_ids(self) -> Set[str]:
        """Get all note IDs that this note links to."""
        return {link.target_id for link in self.links}
    
    def to_markdown(self) -> str:
        """Convert the note to a markdown formatted string."""
        from znote_mcp.config import config
        # Format tags
        tags_str = ", ".join([tag.name for tag in self.tags])
        # Format links
        links_str = ""
        if self.links:
            links_str = "\n".join([
                f"- [{link.link_type}] [[{link.target_id}]] {link.description or ''}"
                for link in self.links
            ])
        # Apply template
        return config.default_note_template.format(
            title=self.title,
            content=self.content,
            created_at=self.created_at.isoformat(),
            tags=tags_str,
            links=links_str
        )
