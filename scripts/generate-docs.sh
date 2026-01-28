#!/bin/bash
# Generate API documentation using pdoc

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Use in-memory database to avoid initialization issues
export ZETTELKASTEN_DATABASE_PATH=":memory:"
export ZETTELKASTEN_NOTES_DIR="/tmp/znote_docs_temp"

echo "Generating API documentation..."
uv run pdoc src/znote_mcp -o docs/api --docformat google

echo "Documentation generated in docs/api/"
echo "Open docs/api/index.html to view"
