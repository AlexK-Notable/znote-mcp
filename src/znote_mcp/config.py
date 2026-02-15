"""Configuration module for the Zettelkasten MCP server."""

import logging
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator

from znote_mcp import __version__

# Load environment variables from the project root .env file.
# Anchored to __file__ so it works regardless of the process CWD
# (e.g. when launched as a Claude Code plugin subprocess).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# User-level config: survives plugin updates, lives alongside notes
_USER_ENV = Path.home() / ".zettelkasten" / ".env"
load_dotenv(_USER_ENV)


logger = logging.getLogger(__name__)

# gte-modernbert-base: 12 layers, 12 heads, 768 hidden_dim
_MODEL_NUM_HEADS = 12
_BYTES_PER_ELEMENT = 4  # float32

# Warn if estimated peak memory exceeds this threshold (bytes)
_MEMORY_WARN_BYTES = 4 * 1024**3  # 4 GB


def estimate_embedding_peak_memory(batch_size: int, max_tokens: int) -> float:
    """Estimate peak memory (bytes) for one embedding batch.

    The attention mechanism allocates memory proportional to
    batch_size × max_tokens² × num_heads × bytes_per_element per layer.
    ONNX runtime doesn't materialize all layers simultaneously, so we
    use a 3x multiplier (empirical upper-bound) on the single-layer cost.
    """
    per_layer = batch_size * (max_tokens**2) * _MODEL_NUM_HEADS * _BYTES_PER_ELEMENT
    return per_layer * 3.0


def get_available_memory_bytes() -> Optional[float]:
    """Detect available system RAM in bytes. Returns None if unavailable."""
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
        if page_size > 0 and page_count > 0:
            return float(page_size * page_count)
    except (ValueError, OSError, AttributeError):
        pass
    return None


class ZettelkastenConfig(BaseModel):
    """Configuration for the Zettelkasten server."""

    # Base directory for the project
    base_dir: Path = Field(
        default_factory=lambda: Path(os.getenv("ZETTELKASTEN_BASE_DIR", "."))
    )
    # Storage configuration
    notes_dir: Path = Field(
        default_factory=lambda: Path(os.getenv("ZETTELKASTEN_NOTES_DIR", "data/notes"))
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
        default_factory=lambda: os.getenv("ZETTELKASTEN_GIT_ENABLED", "true").lower()
        in ("true", "1", "yes")
    )
    # In-memory database configuration
    # When True, uses in-memory SQLite for process isolation (recommended for
    # multi-process environments). Index is rebuilt from markdown files on startup.
    in_memory_db: bool = Field(
        default_factory=lambda: os.getenv("ZETTELKASTEN_IN_MEMORY_DB", "true").lower()
        in ("true", "1", "yes")
    )
    # Server configuration
    server_name: str = Field(default=os.getenv("ZETTELKASTEN_SERVER_NAME", "znote-mcp"))
    server_version: str = Field(default=__version__)
    # Embedding / semantic search configuration
    embeddings_enabled: bool = Field(
        default_factory=lambda: os.getenv(
            "ZETTELKASTEN_EMBEDDINGS_ENABLED", "false"
        ).lower()
        in ("true", "1", "yes")
    )
    embedding_model: str = Field(
        default=os.getenv(
            "ZETTELKASTEN_EMBEDDING_MODEL", "Alibaba-NLP/gte-modernbert-base"
        )
    )
    reranker_model: str = Field(
        default=os.getenv(
            "ZETTELKASTEN_RERANKER_MODEL", "Alibaba-NLP/gte-reranker-modernbert-base"
        )
    )
    embedding_dim: int = Field(
        default_factory=lambda: int(os.getenv("ZETTELKASTEN_EMBEDDING_DIM", "768"))
    )
    embedding_max_tokens: int = Field(
        default_factory=lambda: int(
            os.getenv("ZETTELKASTEN_EMBEDDING_MAX_TOKENS", "2048")
        )
    )
    reranker_idle_timeout: int = Field(
        default_factory=lambda: int(
            os.getenv("ZETTELKASTEN_RERANKER_IDLE_TIMEOUT", "600")
        )
    )
    embedding_batch_size: int = Field(
        default_factory=lambda: int(
            os.getenv("ZETTELKASTEN_EMBEDDING_BATCH_SIZE", "8")
        )
    )
    embedding_model_cache_dir: Optional[Path] = Field(
        default_factory=lambda: (
            Path(os.getenv("ZETTELKASTEN_EMBEDDING_CACHE_DIR"))
            if os.getenv("ZETTELKASTEN_EMBEDDING_CACHE_DIR")
            else None
        )
    )
    # ONNX execution provider preference: "auto" (detect GPU/CPU), or
    # comma-separated list like "CUDAExecutionProvider,CPUExecutionProvider"
    onnx_providers: str = Field(
        default_factory=lambda: os.getenv("ZETTELKASTEN_ONNX_PROVIDERS", "auto")
    )
    # Chunking: notes longer than this (in tokens) get split into overlapping chunks
    embedding_chunk_size: int = Field(
        default_factory=lambda: int(
            os.getenv("ZETTELKASTEN_EMBEDDING_CHUNK_SIZE", "4096")
        )
    )
    embedding_chunk_overlap: int = Field(
        default_factory=lambda: int(
            os.getenv("ZETTELKASTEN_EMBEDDING_CHUNK_OVERLAP", "256")
        )
    )

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

    @model_validator(mode="after")
    def _validate_embedding_config(self) -> "ZettelkastenConfig":
        """Validate embedding settings and warn about high memory usage."""
        if self.embedding_batch_size < 1:
            raise ValueError("embedding_batch_size must be >= 1")
        if self.embedding_max_tokens < 128:
            raise ValueError("embedding_max_tokens must be >= 128")

        if self.embeddings_enabled:
            peak = estimate_embedding_peak_memory(
                self.embedding_batch_size, self.embedding_max_tokens
            )
            if peak > _MEMORY_WARN_BYTES:
                peak_gb = peak / (1024**3)
                logger.warning(
                    "Embedding config (batch_size=%d, max_tokens=%d) may require "
                    "~%.1fGB peak RAM. Consider reducing batch_size or max_tokens. "
                    "See .env.example for guidance.",
                    self.embedding_batch_size,
                    self.embedding_max_tokens,
                    peak_gb,
                )
        return self

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
