"""Repository for link storage and retrieval."""
import logging
from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from znote_mcp.models.db_models import DBLink, DBNote
from znote_mcp.models.schema import Link, LinkType

logger = logging.getLogger(__name__)


class LinkRepository:
    """Repository for managing links between notes.

    Provides CRUD operations for links in the database.
    Links are also persisted in markdown files via the NoteRepository.
    """

    def __init__(self, session_factory):
        """Initialize the link repository.

        Args:
            session_factory: SQLAlchemy session factory for database operations.
        """
        self.session_factory = session_factory

    def create(self, link: Link) -> Link:
        """Create a new link in the database.

        Args:
            link: The Link object to create.

        Returns:
            The created Link object.

        Raises:
            ValueError: If a link with the same source, target, and type already exists.
        """
        with self.session_factory() as session:
            # Check if link already exists
            existing = session.scalar(
                select(DBLink).where(
                    (DBLink.source_id == link.source_id) &
                    (DBLink.target_id == link.target_id) &
                    (DBLink.link_type == link.link_type.value)
                )
            )
            if existing:
                raise ValueError(
                    f"Link already exists: {link.source_id} -> {link.target_id} ({link.link_type.value})"
                )

            db_link = DBLink(
                source_id=link.source_id,
                target_id=link.target_id,
                link_type=link.link_type.value,
                description=link.description,
                created_at=link.created_at
            )
            session.add(db_link)
            session.commit()

        return link

    def get(self, source_id: str, target_id: str, link_type: Optional[LinkType] = None) -> Optional[Link]:
        """Get a link by source, target, and optionally type.

        Args:
            source_id: The source note ID.
            target_id: The target note ID.
            link_type: Optional link type to filter by.

        Returns:
            The Link object if found, None otherwise.
        """
        with self.session_factory() as session:
            query = select(DBLink).where(
                (DBLink.source_id == source_id) &
                (DBLink.target_id == target_id)
            )
            if link_type:
                query = query.where(DBLink.link_type == link_type.value)

            db_link = session.scalar(query)
            if not db_link:
                return None

            return Link(
                source_id=db_link.source_id,
                target_id=db_link.target_id,
                link_type=LinkType(db_link.link_type),
                description=db_link.description,
                created_at=db_link.created_at
            )

    def get_outgoing(self, note_id: str) -> List[Link]:
        """Get all outgoing links from a note.

        Args:
            note_id: The source note ID.

        Returns:
            List of Link objects.
        """
        with self.session_factory() as session:
            db_links = session.scalars(
                select(DBLink).where(DBLink.source_id == note_id)
            ).all()

            return [
                Link(
                    source_id=link.source_id,
                    target_id=link.target_id,
                    link_type=LinkType(link.link_type),
                    description=link.description,
                    created_at=link.created_at
                )
                for link in db_links
            ]

    def get_incoming(self, note_id: str) -> List[Link]:
        """Get all incoming links to a note.

        Args:
            note_id: The target note ID.

        Returns:
            List of Link objects.
        """
        with self.session_factory() as session:
            db_links = session.scalars(
                select(DBLink).where(DBLink.target_id == note_id)
            ).all()

            return [
                Link(
                    source_id=link.source_id,
                    target_id=link.target_id,
                    link_type=LinkType(link.link_type),
                    description=link.description,
                    created_at=link.created_at
                )
                for link in db_links
            ]

    def get_all_for_note(self, note_id: str) -> List[Link]:
        """Get all links (incoming and outgoing) for a note.

        Args:
            note_id: The note ID.

        Returns:
            List of Link objects.
        """
        outgoing = self.get_outgoing(note_id)
        incoming = self.get_incoming(note_id)

        # Deduplicate (a link could be both incoming and outgoing in theory)
        seen = set()
        result = []
        for link in outgoing + incoming:
            key = (link.source_id, link.target_id, link.link_type.value)
            if key not in seen:
                seen.add(key)
                result.append(link)

        return result

    def delete(self, source_id: str, target_id: str, link_type: Optional[LinkType] = None) -> bool:
        """Delete a link.

        Args:
            source_id: The source note ID.
            target_id: The target note ID.
            link_type: Optional link type. If None, deletes all links between the notes.

        Returns:
            True if any links were deleted, False otherwise.
        """
        with self.session_factory() as session:
            query = select(DBLink).where(
                (DBLink.source_id == source_id) &
                (DBLink.target_id == target_id)
            )
            if link_type:
                query = query.where(DBLink.link_type == link_type.value)

            db_links = session.scalars(query).all()
            if not db_links:
                return False

            for db_link in db_links:
                session.delete(db_link)
            session.commit()

            return True

    def delete_all_for_note(self, note_id: str) -> int:
        """Delete all links (incoming and outgoing) for a note.

        Args:
            note_id: The note ID.

        Returns:
            Number of links deleted.
        """
        with self.session_factory() as session:
            # Count before deletion
            outgoing = session.scalars(
                select(DBLink).where(DBLink.source_id == note_id)
            ).all()
            incoming = session.scalars(
                select(DBLink).where(DBLink.target_id == note_id)
            ).all()

            count = len(outgoing) + len(incoming)

            # Delete
            for link in outgoing + incoming:
                session.delete(link)
            session.commit()

            return count

    def count_connections(self, note_id: str) -> int:
        """Count total connections (incoming + outgoing) for a note.

        Args:
            note_id: The note ID.

        Returns:
            Total number of connections.
        """
        with self.session_factory() as session:
            outgoing_count = session.scalar(
                select(DBLink).where(DBLink.source_id == note_id).with_only_columns(
                    DBLink.id
                ).count()
            ) or 0
            incoming_count = session.scalar(
                select(DBLink).where(DBLink.target_id == note_id).with_only_columns(
                    DBLink.id
                ).count()
            ) or 0

            return outgoing_count + incoming_count

    def find_orphaned_note_ids(self) -> List[str]:
        """Find note IDs that have no links (neither incoming nor outgoing).

        Returns:
            List of orphaned note IDs.
        """
        with self.session_factory() as session:
            # Get all note IDs
            all_note_ids = set(
                session.scalars(select(DBNote.id)).all()
            )

            # Get note IDs that have links
            linked_as_source = set(
                session.scalars(select(DBLink.source_id).distinct()).all()
            )
            linked_as_target = set(
                session.scalars(select(DBLink.target_id).distinct()).all()
            )

            linked_ids = linked_as_source | linked_as_target
            orphaned_ids = all_note_ids - linked_ids

            return list(orphaned_ids)
