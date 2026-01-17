"""Configuration module for the Zettelkasten MCP server."""
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

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
    # Server configuration
    server_name: str = Field(
        default=os.getenv("ZETTELKASTEN_SERVER_NAME", "znote-mcp")
    )
    server_version: str = Field(default="1.2.1")
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
