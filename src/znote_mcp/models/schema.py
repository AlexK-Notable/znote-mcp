"""Data models for the Zettelkasten MCP server."""

import datetime
import os
import re
import threading
from dataclasses import dataclass
from datetime import timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Set, Union

from pydantic import BaseModel, Field, field_validator

# Regex pattern for valid note IDs (alphanumeric, underscores, hyphens, T separator)
# Matches format: YYYYMMDDTHHMMSSssssssccc or similar safe patterns
SAFE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_\-T]+$")

# Regex pattern for valid project path segments
SAFE_PROJECT_SEGMENT_PATTERN = re.compile(r"^[a-zA-Z0-9_\-]+$")


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
    if ".." in value:
        raise ValueError(f"{field_name} cannot contain '..' (path traversal)")

    if "/" in value or "\\" in value:
        raise ValueError(f"{field_name} cannot contain path separators")

    # Validate against safe pattern
    if not SAFE_ID_PATTERN.match(value):
        raise ValueError(
            f"{field_name} contains invalid characters. "
            "Only alphanumeric characters, underscores, hyphens, and 'T' are allowed."
        )

    return value


def validate_project_path(value: str) -> str:
    """Validate a project path that may contain sub-projects.

    Project paths can use '/' to denote hierarchy (e.g., "monorepo/frontend").
    Each segment must be safe for filesystem use.

    Args:
        value: The project path to validate

    Returns:
        The validated path (unchanged)

    Raises:
        ValueError: If the path is invalid
    """
    if not value:
        raise ValueError("Project path cannot be empty")

    # Check for path traversal attempts
    if ".." in value:
        raise ValueError("Project path cannot contain '..' (path traversal)")

    # Check for backslashes (Windows path separators not allowed)
    if "\\" in value:
        raise ValueError("Project path cannot contain backslashes")

    # Split into segments and validate each
    segments = value.split("/")

    # Check for empty segments (leading/trailing/double slashes)
    if any(not seg for seg in segments):
        raise ValueError(
            "Project path cannot have empty segments (no leading/trailing/double slashes)"
        )

    # Validate each segment
    for segment in segments:
        if not SAFE_PROJECT_SEGMENT_PATTERN.match(segment):
            raise ValueError(
                f"Project segment '{segment}' contains invalid characters. "
                "Only alphanumeric characters, underscores, and hyphens are allowed."
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


# Thread-safe counter for uniqueness (seeded from PID for cross-process safety)
_id_lock = threading.Lock()
_last_timestamp = 0
_counter = (
    os.getpid() * 7
) % 1_000_000  # PID-based seed prevents multiprocess collisions


def generate_id() -> str:
    """Generate an ISO 8601 compliant timestamp-based ID with guaranteed uniqueness.

    Returns:
        A string in format "YYYYMMDDTHHMMSSsssssscccccc" where:
        - YYYYMMDD is the date
        - T is the ISO 8601 date/time separator
        - HHMMSS is the time (hours, minutes, seconds)
        - ssssss is the 6-digit microsecond component
        - cccccc is a 6-digit counter for cross-process and same-microsecond uniqueness

    The counter is seeded from the process ID so that separate processes
    (e.g. multiprocessing.Pool workers) produce different IDs even when
    they call generate_id() at the exact same microsecond.
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
            _counter = (
                os.getpid() * 7
            ) % 1_000_000  # Re-seed from PID on new microsecond

        # Ensure counter doesn't overflow our 6 digits (supports 1M IDs/microsecond)
        _counter %= 1_000_000

        # Format as ISO 8601 basic format with microseconds and counter
        date_time = now.strftime("%Y%m%dT%H%M%S")
        microseconds = now.microsecond

        return f"{date_time}{microseconds:06d}{_counter:06d}"


class LinkType(str, Enum):
    """Types of links between notes."""

    REFERENCE = "reference"  # Simple reference to another note
    EXTENDS = "extends"  # Current note extends another note
    EXTENDED_BY = "extended_by"  # Current note is extended by another note
    REFINES = "refines"  # Current note refines another note
    REFINED_BY = "refined_by"  # Current note is refined by another note
    CONTRADICTS = "contradicts"  # Current note contradicts another note
    CONTRADICTED_BY = "contradicted_by"  # Current note is contradicted by another note
    QUESTIONS = "questions"  # Current note questions another note
    QUESTIONED_BY = "questioned_by"  # Current note is questioned by another note
    SUPPORTS = "supports"  # Current note supports another note
    SUPPORTED_BY = "supported_by"  # Current note is supported by another note
    RELATED = "related"  # Notes are related in some way


class Link(BaseModel):
    """A link between two notes."""

    source_id: str = Field(..., description="ID of the source note")
    target_id: str = Field(..., description="ID of the target note")
    link_type: LinkType = Field(default=LinkType.REFERENCE, description="Type of link")
    description: Optional[str] = Field(
        default=None, description="Optional description of the link"
    )
    created_at: datetime.datetime = Field(
        default_factory=utc_now, description="When the link was created (UTC)"
    )

    model_config = {
        "validate_assignment": True,
        "extra": "forbid",
        "frozen": True,  # Links are immutable
    }


class NoteType(str, Enum):
    """Types of notes in a Zettelkasten."""

    FLEETING = "fleeting"  # Quick, temporary notes
    LITERATURE = "literature"  # Notes from reading material
    PERMANENT = "permanent"  # Permanent, well-formulated notes
    STRUCTURE = "structure"  # Structure/index notes that organize other notes
    HUB = "hub"  # Hub notes that serve as entry points


class NotePurpose(str, Enum):
    """Workflow-oriented purpose of a note (for Obsidian organization)."""

    RESEARCH = "research"  # Investigation, learning, exploration
    PLANNING = "planning"  # Plans, designs, architecture decisions
    BUGFIXING = "bugfixing"  # Debugging sessions, fixes, investigations
    GENERAL = "general"  # Default for uncategorized notes


class Tag(BaseModel):
    """A tag for categorizing notes."""

    name: str = Field(..., description="Tag name")

    model_config = {"validate_assignment": True, "frozen": True}

    def __str__(self) -> str:
        """Return string representation of tag."""
        return self.name


@dataclass(frozen=True)
class VersionInfo:
    """Git commit metadata for version tracking.

    Attributes:
        commit_hash: Short SHA (7 characters) of the git commit.
        timestamp: When the commit was created.
        author: Who created the commit (defaults to "znote-mcp").
    """

    commit_hash: str
    timestamp: datetime.datetime
    author: str = "znote-mcp"

    @classmethod
    def from_git_commit(
        cls, commit_hash: str, timestamp: datetime.datetime
    ) -> "VersionInfo":
        """Create a VersionInfo from git commit data.

        Args:
            commit_hash: Full or short SHA of the git commit.
            timestamp: When the commit was created.

        Returns:
            VersionInfo with the hash truncated to 7 characters.
        """
        return cls(
            commit_hash=commit_hash[:7],
            timestamp=timestamp,
        )


@dataclass
class ConflictResult:
    """Response indicating a version conflict during note update.

    Attributes:
        status: Always "conflict" for this result type.
        note_id: ID of the note that had a conflict.
        expected_version: The version the client expected.
        actual_version: The actual current version of the note.
        message: Human-readable description of the conflict.
    """

    status: Literal["conflict"]
    note_id: str
    expected_version: str
    actual_version: str
    message: str

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with all fields.
        """
        return {
            "status": self.status,
            "note_id": self.note_id,
            "expected_version": self.expected_version,
            "actual_version": self.actual_version,
            "message": self.message,
        }


@dataclass
class VersionedNote:
    """A note with its version information.

    Wraps a Note with its corresponding VersionInfo for tracking
    which version of the note is being worked with.

    Attributes:
        note: The note data.
        version: Git version metadata for this note.
    """

    note: "Note"
    version: VersionInfo

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with 'note' (as dict) and 'version' (as dict) keys.
        """
        return {
            "note": self.note.model_dump(),
            "version": {
                "commit_hash": self.version.commit_hash,
                "timestamp": self.version.timestamp.isoformat(),
                "author": self.version.author,
            },
        }


class Note(BaseModel):
    """A Zettelkasten note."""

    id: str = Field(default_factory=generate_id, description="Unique ID of the note")
    title: str = Field(..., description="Title of the note")
    content: str = Field(..., description="Content of the note")
    note_type: NoteType = Field(default=NoteType.PERMANENT, description="Type of note")
    note_purpose: NotePurpose = Field(
        default=NotePurpose.GENERAL,
        description="Workflow purpose (research, planning, bugfixing)",
    )
    project: str = Field(
        default="general",
        description="Project this note belongs to (use '/' for sub-projects)",
    )
    plan_id: Optional[str] = Field(
        default=None,
        description="Groups related planning notes (auto-generated for planning purpose)",
    )
    tags: List[Tag] = Field(default_factory=list, description="Tags for categorization")
    links: List[Link] = Field(default_factory=list, description="Links to other notes")
    created_at: datetime.datetime = Field(
        default_factory=utc_now, description="When the note was created (UTC)"
    )
    updated_at: datetime.datetime = Field(
        default_factory=utc_now, description="When the note was last updated (UTC)"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata for the note"
    )

    model_config = {"validate_assignment": True, "extra": "forbid"}

    @field_validator("id")
    @classmethod
    def validate_id(cls, v: str) -> str:
        """Validate that the ID is safe for filesystem use."""
        return validate_safe_path_component(v, "Note ID")

    @field_validator("project")
    @classmethod
    def validate_project(cls, v: str) -> str:
        """Validate project path (supports sub-projects with '/' separator)."""
        # Allow empty string to default to "general"
        if not v or not v.strip():
            return "general"
        return validate_project_path(v)

    @field_validator("plan_id")
    @classmethod
    def validate_plan_id(cls, v: Optional[str]) -> Optional[str]:
        """Validate that the plan_id is safe for filesystem use."""
        if v is None:
            return None
        if not v.strip():
            return None
        return validate_safe_path_component(v, "Plan ID")

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

    def add_link(
        self,
        target_id: str,
        link_type: LinkType = LinkType.REFERENCE,
        description: Optional[str] = None,
    ) -> None:
        """Add a link to another note."""
        # Check if link already exists
        for link in self.links:
            if link.target_id == target_id and link.link_type == link_type:
                return  # Link already exists
        link = Link(
            source_id=self.id,
            target_id=target_id,
            link_type=link_type,
            description=description,
        )
        self.links.append(link)
        self.updated_at = utc_now()

    def remove_link(self, target_id: str, link_type: Optional[LinkType] = None) -> None:
        """Remove a link to another note."""
        if link_type:
            self.links = [
                link
                for link in self.links
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
            links_str = "\n".join(
                [
                    f"- [{link.link_type}] [[{link.target_id}]] {link.description or ''}"
                    for link in self.links
                ]
            )
        # Apply template
        return config.default_note_template.format(
            title=self.title,
            content=self.content,
            created_at=self.created_at.isoformat(),
            tags=tags_str,
            links=links_str,
        )


class Project(BaseModel):
    """A project in the Zettelkasten system.

    Projects organize notes and can be hierarchical (sub-projects).
    Project IDs use '/' for hierarchy: "monorepo/frontend".
    """

    id: str = Field(..., description="Unique project ID (use '/' for sub-projects)")
    name: str = Field(..., description="Human-readable display name")
    description: Optional[str] = Field(
        default=None, description="Brief project description for LLM context"
    )
    parent_id: Optional[str] = Field(
        default=None, description="Parent project ID for sub-projects"
    )
    path: Optional[str] = Field(
        default=None, description="Filesystem path associated with project"
    )
    created_at: datetime.datetime = Field(
        default_factory=utc_now, description="When the project was created (UTC)"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata (git remote, etc.)"
    )

    model_config = {"validate_assignment": True, "extra": "forbid"}

    @field_validator("id")
    @classmethod
    def validate_id(cls, v: str) -> str:
        """Validate project ID (supports sub-projects with '/' separator)."""
        return validate_project_path(v)

    @field_validator("parent_id")
    @classmethod
    def validate_parent_id(cls, v: Optional[str]) -> Optional[str]:
        """Validate parent project ID."""
        if v is None:
            return None
        return validate_project_path(v)

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate that the name is not empty."""
        if not v.strip():
            raise ValueError("Project name cannot be empty")
        return v
