# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

znote-mcp is an MCP (Model Context Protocol) server implementing Zettelkasten knowledge management. It provides 27 tools for creating, linking, searching, and synthesizing atomic notes through Claude and other MCP-compatible clients.

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
    ↓
Repositories (storage/note_repository.py, tag_repository.py, link_repository.py)
    ↓
Models (models/schema.py - Pydantic, models/db_models.py - SQLAlchemy)
```

### Key Files

| File | Purpose |
|------|---------|
| `src/znote_mcp/main.py` | Entry point - parses CLI args, initializes DB, starts server |
| `src/znote_mcp/config.py` | Pydantic configuration with env var support |
| `src/znote_mcp/server/mcp_server.py` | MCP server with 27 tools registered via decorators |
| `src/znote_mcp/services/zettel_service.py` | Business logic for CRUD, links, tags, bulk ops |
| `src/znote_mcp/services/search_service.py` | Search by text, tags, links; find orphans/central notes |
| `src/znote_mcp/storage/note_repository.py` | Dual storage implementation |
| `src/znote_mcp/exceptions.py` | Custom exception hierarchy with error codes |

### Domain Model

**Note Types**: `fleeting`, `literature`, `permanent`, `structure`, `hub`

**Link Types** (with inverse): `reference`↔`reference`, `extends`↔`extended_by`, `refines`↔`refined_by`, `contradicts`↔`contradicted_by`, `questions`↔`questioned_by`, `supports`↔`supported_by`, `related`↔`related`

### Environment Variables

```bash
ZETTELKASTEN_NOTES_DIR=~/.zettelkasten/notes
ZETTELKASTEN_DATABASE_PATH=~/.zettelkasten/db/zettelkasten.db
ZETTELKASTEN_LOG_LEVEL=INFO
ZETTELKASTEN_OBSIDIAN_VAULT=/path/to/obsidian/vault  # Optional
```

## Testing

- **Unit tests**: `tests/conftest.py` provides fixtures with temp directories
- **E2E tests**: `tests/conftest_e2e.py` provides `IsolatedTestEnvironment` class ensuring complete isolation from production data
- **Test data**: Stored in `tests/fixtures/` when persisted

E2E tests use isolated environments that:
- Never touch production notes
- Create fresh databases per test
- Auto-cleanup (unless `ZETTELKASTEN_TEST_PERSIST=1`)

## Code Style

- **Formatter**: Black (line length 88)
- **Import sorting**: isort (black profile)
- **Type checking**: mypy (strict mode - `disallow_untyped_defs`, `disallow_incomplete_defs`)
- **Python**: 3.10+
