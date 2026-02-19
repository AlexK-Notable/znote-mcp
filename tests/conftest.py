"""Common test fixtures for the Zettelkasten MCP server."""

import tempfile
from pathlib import Path

import pytest
from sqlalchemy import create_engine

from tests.fakes import FakeEmbeddingProvider, FakeRerankerProvider
from znote_mcp.config import config
from znote_mcp.models.db_models import Base
from znote_mcp.services.zettel_service import ZettelService
from znote_mcp.storage.note_repository import NoteRepository


@pytest.fixture(params=["asyncio"])
def anyio_backend(request):
    """Restrict anyio tests to asyncio only (trio is not installed)."""
    return request.param


@pytest.fixture
def temp_dirs():
    """Create temporary directories for notes and database."""
    with tempfile.TemporaryDirectory() as notes_dir:
        with tempfile.TemporaryDirectory() as db_dir:
            yield Path(notes_dir), Path(db_dir)


@pytest.fixture
def test_config(temp_dirs, monkeypatch):
    """Configure with test paths (auto-restored even on crash)."""
    notes_dir, db_dir = temp_dirs
    database_path = db_dir / "test_zettelkasten.db"
    monkeypatch.setattr(config, "notes_dir", notes_dir)
    monkeypatch.setattr(config, "database_path", database_path)
    yield config


@pytest.fixture
def note_repository(test_config):
    """Create a test note repository."""
    # Create tables
    database_path = test_config.get_absolute_path(test_config.database_path)
    # Create sync engine to initialize tables
    engine = create_engine(f"sqlite:///{database_path}")
    Base.metadata.create_all(engine)
    engine.dispose()
    # Create repository
    repository = NoteRepository(notes_dir=test_config.notes_dir)
    # Initialize is handled in constructor
    yield repository


@pytest.fixture
def zettel_service(note_repository):
    """Create a test ZettelService."""
    service = ZettelService(repository=note_repository)
    # Initialize is handled in constructor
    yield service


# =============================================================================
# Shared Embedding Test Fixtures
# =============================================================================


@pytest.fixture
def _enable_embeddings(monkeypatch):
    """Temporarily enable embeddings in global config."""
    monkeypatch.setattr(config, "embeddings_enabled", True)


@pytest.fixture
def _disable_embeddings(monkeypatch):
    """Temporarily disable embeddings in global config."""
    monkeypatch.setattr(config, "embeddings_enabled", False)


@pytest.fixture
def fake_embedder():
    """Create a FakeEmbeddingProvider matching the default vec0 dimension (768)."""
    return FakeEmbeddingProvider(dim=768)


@pytest.fixture
def fake_reranker():
    """Create a FakeRerankerProvider for testing."""
    return FakeRerankerProvider()
