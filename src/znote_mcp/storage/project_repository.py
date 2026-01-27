"""Repository for project storage and retrieval."""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from znote_mcp.config import config
from znote_mcp.models.db_models import DBProject, get_session_factory, init_db
from znote_mcp.models.schema import Project, utc_now
from znote_mcp.storage.base import Repository
from znote_mcp.exceptions import ErrorCode, ValidationError

logger = logging.getLogger(__name__)


def escape_like_pattern(value: str) -> str:
    """Escape SQL LIKE wildcards to treat them as literals.

    Args:
        value: User input string that may contain LIKE wildcards

    Returns:
        String with '%', '_', and '\\' escaped for safe use in LIKE clauses
    """
    escape_table = str.maketrans({
        '\\': '\\\\',
        '%': '\\%',
        '_': '\\_',
    })
    return value.translate(escape_table)


class ProjectRepository(Repository[Project]):
    """Repository for project storage and retrieval.

    Projects are stored in SQLite database and can be exported/imported
    from .znote/projects.json for version control and portability.
    """

    def __init__(self, engine=None):
        """Initialize the repository.

        Args:
            engine: SQLAlchemy engine. If None, uses default from config.
        """
        self.engine = engine or init_db()
        self.session_factory = get_session_factory(self.engine)
        logger.info("ProjectRepository initialized")

    def create(self, project: Project) -> Project:
        """Create a new project.

        Args:
            project: Project to create.

        Returns:
            Created project.

        Raises:
            ValidationError: If project with same ID already exists.
        """
        with self.session_factory() as session:
            # Check if project already exists
            existing = session.get(DBProject, project.id)
            if existing:
                raise ValidationError(
                    f"Project '{project.id}' already exists",
                    code=ErrorCode.VALIDATION_FAILED
                )

            # Validate parent exists if specified
            if project.parent_id:
                # Check for self-reference
                if project.parent_id == project.id:
                    raise ValidationError(
                        f"Project '{project.id}' cannot be its own parent",
                        code=ErrorCode.VALIDATION_FAILED
                    )
                parent = session.get(DBProject, project.parent_id)
                if not parent:
                    raise ValidationError(
                        f"Parent project '{project.parent_id}' not found",
                        code=ErrorCode.VALIDATION_FAILED
                    )

            # Create DB record
            db_project = DBProject(
                id=project.id,
                name=project.name,
                description=project.description,
                parent_id=project.parent_id,
                path=project.path,
                created_at=project.created_at,
                metadata_json=json.dumps(project.metadata) if project.metadata else None
            )
            session.add(db_project)
            session.commit()

            logger.info(f"Created project: {project.id}")
            return project

    def get(self, id: str) -> Optional[Project]:
        """Get a project by ID.

        Args:
            id: Project ID.

        Returns:
            Project if found, None otherwise.
        """
        with self.session_factory() as session:
            db_project = session.get(DBProject, id)
            if not db_project:
                return None
            return self._db_to_model(db_project)

    def get_all(self) -> List[Project]:
        """Get all projects.

        Returns:
            List of all projects.
        """
        with self.session_factory() as session:
            result = session.execute(select(DBProject).order_by(DBProject.id))
            return [self._db_to_model(db) for db in result.scalars().all()]

    def update(self, project: Project) -> Project:
        """Update an existing project.

        Args:
            project: Project with updated fields.

        Returns:
            Updated project.

        Raises:
            ValidationError: If project not found.
        """
        with self.session_factory() as session:
            db_project = session.get(DBProject, project.id)
            if not db_project:
                raise ValidationError(
                    f"Project '{project.id}' not found",
                    code=ErrorCode.PROJECT_NOT_FOUND
                )

            # Validate parent exists if specified and check for cycles
            if project.parent_id and project.parent_id != db_project.parent_id:
                # Check for self-reference
                if project.parent_id == project.id:
                    raise ValidationError(
                        f"Project '{project.id}' cannot be its own parent",
                        code=ErrorCode.VALIDATION_FAILED
                    )
                parent = session.get(DBProject, project.parent_id)
                if not parent:
                    raise ValidationError(
                        f"Parent project '{project.parent_id}' not found",
                        code=ErrorCode.VALIDATION_FAILED
                    )
                # Check for circular reference by walking up the parent chain
                # If we encounter project.id, it would create a cycle
                visited = {project.id}
                current = parent
                while current and current.parent_id:
                    if current.parent_id in visited:
                        raise ValidationError(
                            f"Setting parent to '{project.parent_id}' would create a circular reference",
                            code=ErrorCode.VALIDATION_FAILED
                        )
                    visited.add(current.parent_id)
                    current = session.get(DBProject, current.parent_id)

            # Update fields
            db_project.name = project.name
            db_project.description = project.description
            db_project.parent_id = project.parent_id
            db_project.path = project.path
            db_project.metadata_json = json.dumps(project.metadata) if project.metadata else None

            session.commit()
            logger.info(f"Updated project: {project.id}")
            return project

    def delete(self, id: str) -> None:
        """Delete a project by ID.

        Args:
            id: Project ID to delete.

        Raises:
            ValidationError: If project not found or has notes/children.
        """
        with self.session_factory() as session:
            db_project = session.get(DBProject, id)
            if not db_project:
                raise ValidationError(
                    f"Project '{id}' not found",
                    code=ErrorCode.PROJECT_NOT_FOUND
                )

            # Check for child projects
            children = session.execute(
                select(DBProject).where(DBProject.parent_id == id)
            ).scalars().all()
            if children:
                child_ids = [c.id for c in children]
                raise ValidationError(
                    f"Cannot delete project '{id}': has child projects {child_ids}. "
                    "Delete children first.",
                    code=ErrorCode.VALIDATION_FAILED
                )

            # Check for notes using this project
            note_count = session.execute(
                text("SELECT COUNT(*) FROM notes WHERE project = :project_id"),
                {"project_id": id}
            ).scalar()
            if note_count > 0:
                raise ValidationError(
                    f"Cannot delete project '{id}': {note_count} notes belong to it. "
                    "Move or delete notes first.",
                    code=ErrorCode.VALIDATION_FAILED
                )

            session.delete(db_project)
            session.commit()
            logger.info(f"Deleted project: {id}")

    def search(self, **kwargs) -> List[Project]:
        """Search for projects.

        Args:
            name: Search by name (partial match).
            parent_id: Filter by parent project.

        Returns:
            List of matching projects.
        """
        with self.session_factory() as session:
            query = select(DBProject)

            if "name" in kwargs:
                escaped_name = escape_like_pattern(kwargs['name'])
                query = query.where(DBProject.name.ilike(f"%{escaped_name}%", escape='\\'))
            if "parent_id" in kwargs:
                query = query.where(DBProject.parent_id == kwargs["parent_id"])

            query = query.order_by(DBProject.id)
            result = session.execute(query)
            return [self._db_to_model(db) for db in result.scalars().all()]

    def get_hierarchy(self) -> List[Dict[str, Any]]:
        """Get projects as a hierarchical tree structure.

        Returns:
            List of root projects with nested children.
        """
        all_projects = self.get_all()

        # Build lookup and find roots
        by_id = {p.id: {"project": p, "children": []} for p in all_projects}
        roots = []

        for p in all_projects:
            if p.parent_id and p.parent_id in by_id:
                by_id[p.parent_id]["children"].append(by_id[p.id])
            else:
                roots.append(by_id[p.id])

        return roots

    def exists(self, id: str) -> bool:
        """Check if a project exists.

        Args:
            id: Project ID to check.

        Returns:
            True if project exists.
        """
        with self.session_factory() as session:
            return session.get(DBProject, id) is not None

    def get_note_count(self, id: str) -> int:
        """Get number of notes in a project.

        Args:
            id: Project ID.

        Returns:
            Number of notes using this project.
        """
        with self.session_factory() as session:
            count = session.execute(
                text("SELECT COUNT(*) FROM notes WHERE project = :project_id"),
                {"project_id": id}
            ).scalar()
            return count or 0

    # ========== Import/Export for .znote/projects.json ==========

    def export_to_json(self, path: Optional[Path] = None) -> Path:
        """Export projects to JSON file for version control.

        Args:
            path: Output path. Defaults to .znote/projects.json in notes_dir.

        Returns:
            Path to exported file.
        """
        if path is None:
            notes_dir = config.get_absolute_path(config.notes_dir)
            znote_dir = notes_dir.parent / ".znote"
            znote_dir.mkdir(parents=True, exist_ok=True)
            path = znote_dir / "projects.json"

        projects = self.get_all()

        # Find root project (if any)
        roots = [p for p in projects if not p.parent_id]
        root_project = roots[0].id if len(roots) == 1 else None

        data = {
            "root_project": root_project,
            "projects": [
                {
                    "id": p.id,
                    "name": p.name,
                    "description": p.description,
                    "path": p.path,
                    "parent_id": p.parent_id,
                    "metadata": p.metadata
                }
                for p in projects
            ]
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported {len(projects)} projects to {path}")
        return path

    def import_from_json(self, path: Optional[Path] = None) -> int:
        """Import projects from JSON file.

        Args:
            path: Input path. Defaults to .znote/projects.json in notes_dir.

        Returns:
            Number of projects imported.
        """
        if path is None:
            notes_dir = config.get_absolute_path(config.notes_dir)
            path = notes_dir.parent / ".znote" / "projects.json"

        if not path.exists():
            logger.warning(f"Projects file not found: {path}")
            return 0

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        imported = 0
        projects_data = data.get("projects", [])

        # Sort by depth (parents first) to handle dependencies
        def depth(p):
            return p.get("id", "").count("/")
        projects_data.sort(key=depth)

        for p_data in projects_data:
            project = Project(
                id=p_data["id"],
                name=p_data["name"],
                description=p_data.get("description"),
                path=p_data.get("path"),
                parent_id=p_data.get("parent_id"),
                metadata=p_data.get("metadata", {})
            )

            # Skip if already exists
            if self.exists(project.id):
                logger.debug(f"Project {project.id} already exists, skipping")
                continue

            try:
                self.create(project)
                imported += 1
            except Exception as e:
                logger.warning(f"Failed to import project {project.id}: {e}")

        logger.info(f"Imported {imported} projects from {path}")
        return imported

    def _db_to_model(self, db_project: DBProject) -> Project:
        """Convert DBProject to Project model."""
        return Project(
            id=db_project.id,
            name=db_project.name,
            description=db_project.description,
            parent_id=db_project.parent_id,
            path=db_project.path,
            created_at=db_project.created_at,
            metadata=json.loads(db_project.metadata_json) if db_project.metadata_json else {}
        )
