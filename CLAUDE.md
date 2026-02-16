# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

znote-mcp is an MCP (Model Context Protocol) server implementing Zettelkasten knowledge management. It provides 22 tools for creating, linking, searching, and synthesizing atomic notes through Claude and other MCP-compatible clients. Version 1.4.0 adds optional semantic search via ONNX-based embeddings with sqlite-vec vector storage.

## Common Commands

```bash
# Run the MCP server
python -m znote_mcp.main
python -m znote_mcp.main --notes-dir ./data/notes --database-path ./data/db/zettelkasten.db

# Run all tests
uv run pytest -v tests/

# Run with coverage
uv run pytest --cov=znote_mcp --cov-report=term-missing tests/

# Run single test file
uv run pytest -v tests/test_models.py

# Run single test class/function
uv run pytest -v tests/test_models.py::TestNoteModel
uv run pytest -v tests/test_models.py::TestNoteModel::test_note_validation

# Run E2E tests (isolated environment)
uv run pytest tests/test_e2e.py -v

# Debug E2E with persistent data
ZETTELKASTEN_TEST_PERSIST=1 uv run pytest tests/test_e2e.py -v

# Run protocol integration tests (full MCP pipeline, no mocking)
uv run pytest tests/test_mcp_protocol.py -v

# Run embedding/semantic search tests (phases 1-5)
uv run pytest tests/test_embedding_phase1.py tests/test_embedding_phase2.py tests/test_embedding_phase3.py tests/test_embedding_phase4.py tests/test_embedding_phase5.py -v

# Run chunked embedding integration tests
uv run pytest tests/test_chunked_embedding_integration.py -v

# Linting and formatting
uv run black src/ tests/
uv run isort src/ tests/
uv run mypy src/

# Generate API documentation
./scripts/generate-docs.sh
# Or manually:
ZETTELKASTEN_DATABASE_PATH=":memory:" uv run pdoc src/znote_mcp -o docs/api --docformat google

# Database migrations (Alembic)
uv run alembic current           # Check current migration
uv run alembic upgrade head      # Apply all migrations
uv run alembic revision --autogenerate -m "Description"  # Create new migration
```

## Architecture

### Dual Storage System

The system uses markdown files as source of truth with SQLite as an indexing layer:

1. **Markdown Files** (`notes_dir`): Human-readable notes with YAML frontmatter. Version-controllable, editable externally.
2. **SQLite Database** (`database_path`): WAL-mode database with FTS5 full-text search. Rebuilt via `zk_system(action="rebuild")`.

### Layer Structure

```
MCP Tools (server/mcp_server.py)
    ↓
Services (services/zettel_service.py, search_service.py)
    ↓                          ↓
Repositories                Embedding Layer (optional)
  storage/note_repository.py    services/embedding_service.py
  storage/tag_repository.py     services/onnx_providers.py
  storage/link_repository.py    services/text_chunker.py
    ↓                          ↓
Models                      Vector Storage
  models/schema.py (Pydantic)   sqlite-vec (vec0 virtual table)
  models/db_models.py (SQLAlchemy)
```

### Semantic Search Architecture

When `[semantic]` deps are installed, embeddings auto-enable on startup:

1. **Setup** (`setup_manager.py`): Auto-installs semantic deps into venv, pre-downloads ONNX models in background thread
2. **Providers** (`onnx_providers.py`): Direct ONNX Runtime embedding (gte-modernbert-base, 768-dim) and reranking (gte-reranker-modernbert-base) with lazy loading
3. **Types** (`embedding_types.py`): Protocol interfaces for `EmbeddingProvider` and `RerankerProvider` (structural subtyping, no inheritance required)
4. **Chunking** (`text_chunker.py`): Splits long notes into overlapping token-aware chunks respecting sentence boundaries
5. **Service** (`embedding_service.py`): Thread-safe orchestrator with idle timeout for reranker, warm-loaded embedder
6. **Storage**: sqlite-vec `vec0` virtual table for KNN vector search; chunked embeddings stored with note-level and chunk-level granularity
7. **Integration**: `zettel_service.py` embeds on create/update, `search_service.py` uses vector KNN + optional reranking for `mode="semantic"`

### Key Files

| File | Purpose |
|------|---------|
| `src/znote_mcp/main.py` | Entry point - parses CLI args, initializes DB, auto-enables embeddings, starts server |
| `src/znote_mcp/config.py` | Pydantic configuration with env var support (including all embedding config) |
| `src/znote_mcp/server/mcp_server.py` | MCP server with 22 tools registered via decorators |
| `src/znote_mcp/services/zettel_service.py` | Business logic for CRUD, links, tags, bulk ops; embeds notes on create/update |
| `src/znote_mcp/services/search_service.py` | Search by text, tags, links, semantic vectors; find orphans/central notes |
| `src/znote_mcp/services/embedding_service.py` | Thread-safe embedding/reranking orchestrator with lazy loading and idle timeout |
| `src/znote_mcp/services/embedding_types.py` | Protocol interfaces for EmbeddingProvider and RerankerProvider |
| `src/znote_mcp/services/onnx_providers.py` | ONNX Runtime implementations for embedding and reranking models |
| `src/znote_mcp/services/text_chunker.py` | Token-aware text chunking with sentence-boundary-respecting overlap |
| `src/znote_mcp/setup_manager.py` | Auto-installs semantic deps, pre-downloads ONNX models in background |
| `src/znote_mcp/storage/note_repository.py` | Dual storage implementation (markdown files + SQLite + sqlite-vec vectors) |
| `src/znote_mcp/exceptions.py` | Custom exception hierarchy with error codes |
| `tests/conftest_protocol.py` | Protocol test fixtures using `mcp.shared.memory` transport + FakeEmbeddingProvider |
| `tests/test_mcp_protocol.py` | 30 protocol integration tests (CRUD, search, links, batch, semantic, validation) |

### Domain Model

**Note Types**: `fleeting`, `literature`, `permanent`, `structure`, `hub`

**Link Types** (with inverse): `reference`↔`reference`, `extends`↔`extended_by`, `refines`↔`refined_by`, `contradicts`↔`contradicted_by`, `questions`↔`questioned_by`, `supports`↔`supported_by`, `related`↔`related`

### Environment Variables

**Core:**
```bash
ZETTELKASTEN_NOTES_DIR=~/.zettelkasten/notes
ZETTELKASTEN_DATABASE_PATH=~/.zettelkasten/db/zettelkasten.db
ZETTELKASTEN_LOG_LEVEL=INFO
ZETTELKASTEN_OBSIDIAN_VAULT=/path/to/obsidian/vault  # Optional
ZETTELKASTEN_GIT_ENABLED=true        # Git versioning for conflict detection
ZETTELKASTEN_IN_MEMORY_DB=true       # Per-process in-memory SQLite (recommended)
```

**Semantic search / embeddings** (requires `pip install znote-mcp[semantic]`):
```bash
ZETTELKASTEN_EMBEDDINGS_ENABLED=false          # Auto-enables when deps installed; set false to force off
ZETTELKASTEN_EMBEDDING_MODEL=Alibaba-NLP/gte-modernbert-base
ZETTELKASTEN_RERANKER_MODEL=Alibaba-NLP/gte-reranker-modernbert-base
ZETTELKASTEN_EMBEDDING_DIM=768                 # Must match model output dimension
ZETTELKASTEN_EMBEDDING_MAX_TOKENS=2048         # Max tokens per embedding input
ZETTELKASTEN_EMBEDDING_BATCH_SIZE=8            # Higher = faster reindex, more memory
ZETTELKASTEN_EMBEDDING_CHUNK_SIZE=4096         # Notes longer than this get split
ZETTELKASTEN_EMBEDDING_CHUNK_OVERLAP=256       # Token overlap between chunks
ZETTELKASTEN_EMBEDDING_CACHE_DIR=              # Custom model cache dir (default: HF cache)
ZETTELKASTEN_ONNX_PROVIDERS=auto               # "auto", "cpu", or explicit provider list
ZETTELKASTEN_RERANKER_IDLE_TIMEOUT=600         # Seconds before idle reranker unloads (0 = never)
```

See `.env.example` for full documentation including memory usage guidance per batch_size/max_tokens combination.

## Testing

36 test files covering unit, integration, E2E, protocol, and semantic search.

- **Unit tests**: `tests/conftest.py` provides fixtures with temp directories
- **E2E tests**: `tests/conftest_e2e.py` provides `IsolatedTestEnvironment` class ensuring complete isolation from production data
- **Protocol tests**: `tests/conftest_protocol.py` + `tests/test_mcp_protocol.py` — 30 tests exercising all 22 tools through the full MCP JSON-RPC pipeline using `mcp.shared.memory.create_connected_server_and_client_session` (no mocking). Includes semantic search tests with `FakeEmbeddingProvider`/`FakeRerankerProvider`.
- **Embedding tests**: 5 phased test files (`test_embedding_phase1.py` through `test_embedding_phase5.py`) covering providers, service, chunking, search integration, and reranker
- **Chunked embedding tests**: `test_chunked_embedding_integration.py` for long-note splitting and multi-chunk vector storage
- **Concurrency tests**: `test_concurrency.py` and `test_multiprocess_concurrency.py` for thread and process safety

E2E tests use isolated environments that:
- Never touch production notes
- Create fresh databases per test
- Auto-cleanup (unless `ZETTELKASTEN_TEST_PERSIST=1`)

## Code Style

- **Formatter**: Black (line length 88)
- **Import sorting**: isort (black profile)
- **Type checking**: mypy (strict mode - `disallow_untyped_defs`, `disallow_incomplete_defs`)
- **Python**: 3.10+
- **Optional deps**: Semantic search packages use lazy imports with clear error messages (see `onnx_providers.py` pattern)
- **Protocols**: Embedding interfaces use `typing.Protocol` (PEP 544) for structural subtyping -- no base class inheritance required
