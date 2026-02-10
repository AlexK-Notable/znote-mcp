"""Configuration module for the Zettelkasten MCP server."""
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from znote_mcp import __version__

# Load environment variables
load_dotenv()

class ZettelkastenConfig(BaseModel):
    """Configuration for the Zettelkasten server."""
    # Base directory for the project
    base_dir: Path = Field(
        default_factory=lambda: Path(os.getenv("ZETTELKASTEN_BASE_DIR", "."))
    )
    # Storage configuration
    notes_dir: Path = Field(
        default_factory=lambda: Path(
            os.getenv("ZETTELKASTEN_NOTES_DIR", "data/notes")
        )
    )
    # Database configuration
    database_path: Path = Field(
        default_factory=lambda: Path(
            os.getenv("ZETTELKASTEN_DATABASE_PATH", "data/db/zettelkasten.db")
        )
    )
    # Obsidian mirror configuration (optional)
    # When set, notes will be copied to this directory as well
    obsidian_vault_path: Optional[Path] = Field(
        default_factory=lambda: (
            Path(os.getenv("ZETTELKASTEN_OBSIDIAN_VAULT"))
            if os.getenv("ZETTELKASTEN_OBSIDIAN_VAULT")
            else None
        )
    )
    # Git versioning configuration
    # When True, notes are version-controlled with git and include commit hashes
    git_enabled: bool = Field(
        default_factory=lambda: os.getenv(
            "ZETTELKASTEN_GIT_ENABLED", "true"
        ).lower() in ("true", "1", "yes")
    )
    # In-memory database configuration
    # When True, uses in-memory SQLite for process isolation (recommended for
    # multi-process environments). Index is rebuilt from markdown files on startup.
    in_memory_db: bool = Field(
        default_factory=lambda: os.getenv(
            "ZETTELKASTEN_IN_MEMORY_DB", "true"
        ).lower() in ("true", "1", "yes")
    )
    # Server configuration
    server_name: str = Field(
        default=os.getenv("ZETTELKASTEN_SERVER_NAME", "znote-mcp")
    )
    server_version: str = Field(default=__version__)
    # Embedding / semantic search configuration
    embeddings_enabled: bool = Field(
        default_factory=lambda: os.getenv(
            "ZETTELKASTEN_EMBEDDINGS_ENABLED", "false"
        ).lower() in ("true", "1", "yes")
    )
    embedding_model: str = Field(
        default=os.getenv(
            "ZETTELKASTEN_EMBEDDING_MODEL",
            "Alibaba-NLP/gte-modernbert-base"
        )
    )
    reranker_model: str = Field(
        default=os.getenv(
            "ZETTELKASTEN_RERANKER_MODEL",
            "Alibaba-NLP/gte-reranker-modernbert-base"
        )
    )
    embedding_dim: int = Field(
        default_factory=lambda: int(os.getenv(
            "ZETTELKASTEN_EMBEDDING_DIM", "768"
        ))
    )
    embedding_max_tokens: int = Field(
        default_factory=lambda: int(os.getenv(
            "ZETTELKASTEN_EMBEDDING_MAX_TOKENS", "8192"
        ))
    )
    reranker_idle_timeout: int = Field(
        default_factory=lambda: int(os.getenv(
            "ZETTELKASTEN_RERANKER_IDLE_TIMEOUT", "600"
        ))
    )
    embedding_batch_size: int = Field(
        default_factory=lambda: int(os.getenv(
            "ZETTELKASTEN_EMBEDDING_BATCH_SIZE", "32"
        ))
    )
    embedding_model_cache_dir: Optional[Path] = Field(
        default_factory=lambda: (
            Path(os.getenv("ZETTELKASTEN_EMBEDDING_CACHE_DIR"))
            if os.getenv("ZETTELKASTEN_EMBEDDING_CACHE_DIR")
            else None
        )
    )

    # Date format for ID generation (using ISO format for timestamps)
    id_date_format: str = Field(default="%Y%m%dT%H%M%S")
    # Default note template
    default_note_template: str = Field(
        default=(
            "# {title}\n\n"
            "## Metadata\n"
            "- Created: {created_at}\n"
            "- Tags: {tags}\n\n"
            "## Content\n\n"
            "{content}\n\n"
            "## Links\n"
            "{links}\n"
        )
    )
    
    def get_absolute_path(self, path: Path) -> Path:
        """Convert a relative path to an absolute path based on base_dir."""
        if path.is_absolute():
            return path
        return self.base_dir / path
    
    def get_db_url(self) -> str:
        """Get the database URL for SQLite."""
        db_path = self.get_absolute_path(self.database_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{db_path}"

    def get_obsidian_vault_path(self) -> Optional[Path]:
        """Get the absolute path to the Obsidian vault mirror directory.

        Returns None if not configured. When configured, ensures the directory exists.
        """
        if self.obsidian_vault_path is None:
            return None
        vault_path = self.get_absolute_path(self.obsidian_vault_path)
        vault_path.mkdir(parents=True, exist_ok=True)
        return vault_path

# Create a global config instance
config = ZettelkastenConfig()
