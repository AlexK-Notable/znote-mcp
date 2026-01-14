# Test Fixtures

This directory contains isolated test data that is **completely separate** from your production Zettelkasten data.

## Directory Structure

```
fixtures/
├── notes/           # Isolated notes directory for E2E tests
├── obsidian_vault/  # Mock Obsidian vault for sync testing
├── database/        # SQLite databases for E2E tests
└── README.md        # This file
```

## Important

- **All data in this directory is for testing only**
- **Never point your production config at these directories**
- These directories are git-ignored except for seed files
- Tests clean up after themselves by default

## Usage

### Running E2E Tests with Isolated Fixtures

```bash
# Run all E2E tests (uses temporary directories by default)
uv run pytest tests/test_e2e.py -v

# Run with persistent fixtures (for debugging)
ZETTELKASTEN_TEST_PERSIST=1 uv run pytest tests/test_e2e.py -v

# Inspect fixtures after test (when PERSIST=1)
ls -la tests/fixtures/notes/
ls -la tests/fixtures/database/
```

### Environment Variables

- `ZETTELKASTEN_TEST_PERSIST=1` - Keep test data after tests complete
- `ZETTELKASTEN_TEST_FIXTURES_DIR` - Override fixtures directory location

## Seed Data

Sample seed data can be placed in `_seed/` subdirectories and will be copied
into the test environment at the start of each test session.
