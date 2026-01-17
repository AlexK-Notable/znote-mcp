"""Repository for tag storage and retrieval."""
import logging
from typing import Dict, List, Optional

from sqlalchemy import func, select, text

from znote_mcp.models.db_models import DBNote, DBTag, note_tags
from znote_mcp.models.schema import Tag

logger = logging.getLogger(__name__)


class TagRepository:
    """Repository for managing tags.

    Provides CRUD operations for tags in the database.
    Tags are also persisted in markdown files via the NoteRepository.
    """

    def __init__(self, session_factory):
        """Initialize the tag repository.

        Args:
            session_factory: SQLAlchemy session factory for database operations.
        """
        self.session_factory = session_factory

    def get_or_create(self, tag_name: str) -> Tag:
        """Get an existing tag or create a new one.

        Args:
            tag_name: The name of the tag.

        Returns:
            The Tag object.
        """
        with self.session_factory() as session:
            # Use INSERT OR IGNORE to handle concurrent creation race
            session.execute(
                text("INSERT OR IGNORE INTO tags (name) VALUES (:name)"),
                {"name": tag_name}
            )
            session.commit()
            # Now SELECT - the tag definitely exists
            db_tag = session.scalar(select(DBTag).where(DBTag.name == tag_name))
            return Tag(name=db_tag.name)

    def get(self, tag_name: str) -> Optional[Tag]:
        """Get a tag by name.

        Args:
            tag_name: The name of the tag.

        Returns:
            The Tag object if found, None otherwise.
        """
        with self.session_factory() as session:
            db_tag = session.scalar(
                select(DBTag).where(DBTag.name == tag_name)
            )
            if not db_tag:
                return None

            return Tag(name=db_tag.name)

    def get_all(self) -> List[Tag]:
        """Get all tags in the system.

        Returns:
            List of all Tag objects.
        """
        with self.session_factory() as session:
            db_tags = session.scalars(select(DBTag)).all()

            return [Tag(name=tag.name) for tag in db_tags]

    def get_with_counts(self) -> Dict[str, int]:
        """Get all tags with their usage counts.

        Returns:
            Dictionary mapping tag names to their note counts.
        """
        with self.session_factory() as session:
            # Query to count notes per tag
            result = session.execute(
                select(DBTag.name, func.count(note_tags.c.note_id))
                .select_from(DBTag)
                .outerjoin(note_tags, DBTag.id == note_tags.c.tag_id)
                .group_by(DBTag.name)
            ).all()

            return {name: count for name, count in result}

    def find_note_ids_by_tag(self, tag_name: str) -> List[str]:
        """Find all note IDs that have a specific tag.

        Args:
            tag_name: The name of the tag.

        Returns:
            List of note IDs.
        """
        with self.session_factory() as session:
            result = session.execute(
                select(note_tags.c.note_id)
                .select_from(note_tags)
                .join(DBTag, note_tags.c.tag_id == DBTag.id)
                .where(DBTag.name == tag_name)
            ).all()

            return [row[0] for row in result]

    def find_note_ids_by_tags(self, tag_names: List[str], match_all: bool = False) -> List[str]:
        """Find note IDs that have any or all of the specified tags.

        Args:
            tag_names: List of tag names.
            match_all: If True, only return notes that have ALL tags.
                      If False, return notes that have ANY of the tags.

        Returns:
            List of note IDs.
        """
        with self.session_factory() as session:
            if match_all:
                # Find notes that have ALL specified tags
                # This uses a subquery for each tag and intersects them
                base_query = None
                for tag_name in tag_names:
                    subquery = (
                        select(note_tags.c.note_id)
                        .select_from(note_tags)
                        .join(DBTag, note_tags.c.tag_id == DBTag.id)
                        .where(DBTag.name == tag_name)
                    )
                    if base_query is None:
                        base_query = subquery
                    else:
                        base_query = base_query.intersect(subquery)

                if base_query is None:
                    return []

                result = session.execute(base_query).all()
            else:
                # Find notes that have ANY of the specified tags
                result = session.execute(
                    select(note_tags.c.note_id)
                    .select_from(note_tags)
                    .join(DBTag, note_tags.c.tag_id == DBTag.id)
                    .where(DBTag.name.in_(tag_names))
                    .distinct()
                ).all()

            return [row[0] for row in result]

    def get_tags_for_note(self, note_id: str) -> List[Tag]:
        """Get all tags for a specific note.

        Args:
            note_id: The note ID.

        Returns:
            List of Tag objects.
        """
        with self.session_factory() as session:
            result = session.execute(
                select(DBTag.name)
                .select_from(DBTag)
                .join(note_tags, DBTag.id == note_tags.c.tag_id)
                .where(note_tags.c.note_id == note_id)
            ).all()

            return [Tag(name=row[0]) for row in result]

    def add_tag_to_note(self, note_id: str, tag_name: str) -> None:
        """Add a tag to a note.

        Args:
            note_id: The note ID.
            tag_name: The tag name.
        """
        with self.session_factory() as session:
            # Get or create the tag (using INSERT OR IGNORE to handle race conditions)
            session.execute(
                text("INSERT OR IGNORE INTO tags (name) VALUES (:name)"),
                {"name": tag_name}
            )
            db_tag = session.scalar(select(DBTag).where(DBTag.name == tag_name))

            # Get the note
            db_note = session.scalar(
                select(DBNote).where(DBNote.id == note_id)
            )
            if not db_note:
                raise ValueError(f"Note {note_id} not found")

            # Check if association already exists
            if db_tag not in db_note.tags:
                db_note.tags.append(db_tag)
                session.commit()

    def remove_tag_from_note(self, note_id: str, tag_name: str) -> bool:
        """Remove a tag from a note.

        Args:
            note_id: The note ID.
            tag_name: The tag name.

        Returns:
            True if the tag was removed, False if it wasn't present.
        """
        with self.session_factory() as session:
            # Get the tag
            db_tag = session.scalar(
                select(DBTag).where(DBTag.name == tag_name)
            )
            if not db_tag:
                return False

            # Get the note
            db_note = session.scalar(
                select(DBNote).where(DBNote.id == note_id)
            )
            if not db_note:
                raise ValueError(f"Note {note_id} not found")

            # Remove association if it exists
            if db_tag in db_note.tags:
                db_note.tags.remove(db_tag)
                session.commit()
                return True

            return False

    def delete_unused(self) -> int:
        """Delete tags that are not associated with any notes.

        Returns:
            Number of tags deleted.
        """
        with self.session_factory() as session:
            # Find tags with no notes
            unused_tags = session.scalars(
                select(DBTag)
                .outerjoin(note_tags, DBTag.id == note_tags.c.tag_id)
                .where(note_tags.c.note_id.is_(None))
            ).all()

            count = len(unused_tags)
            for tag in unused_tags:
                session.delete(tag)

            session.commit()
            return count
