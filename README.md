# znote-mcp

A Model Context Protocol (MCP) server that implements the Zettelkasten knowledge management methodology, allowing you to create, link, explore and synthesize atomic notes through Claude and other MCP-compatible clients.

> **Note**: This is a fork of [entanglr/zettelkasten-mcp](https://github.com/entanglr/zettelkasten-mcp), renamed to znote-mcp.

## What is Zettelkasten?

The Zettelkasten method is a knowledge management system developed by German sociologist Niklas Luhmann, who used it to produce over 70 books and hundreds of articles. It consists of three core principles:

1. **Atomicity**: Each note contains exactly one idea, making it a discrete unit of knowledge
2. **Connectivity**: Notes are linked together to create a network of knowledge, with meaningful relationships between ideas
3. **Emergence**: As the network grows, new patterns and insights emerge that weren't obvious when the individual notes were created

What makes the Zettelkasten approach powerful is how it enables exploration in multiple ways:

- **Vertical exploration**: dive deeper into specific topics by following connections within a subject area.
- **Horizontal exploration**: discover unexpected relationships between different fields by traversing links that cross domains.

This structure invites serendipitous discoveries as you follow trails of thought from note to note, all while keeping each piece of information easily accessible through its unique identifier. Luhmann called his system his "second brain" or "communication partner" - this digital implementation aims to provide similar benefits through modern technology.

## Features

- **Atomic notes** with unique timestamp-based IDs and YAML frontmatter metadata
- **Semantic linking** with 7 typed link types and automatic inverse links
- **Tag system** with batch operations and orphan cleanup
- **Full-text search** via SQLite FTS5 with boolean operators, phrases, and prefix matching
- **Semantic search** via ONNX-based embeddings with optional GPU acceleration
- **Token-aware chunking** for embedding long notes into overlapping segments
- **Reranking** for improved search result quality using cross-encoder models
- **Project organization** with hierarchical sub-projects and note routing
- **Bulk operations** for batch note creation, tag management, and project moves
- **Obsidian vault mirroring** for browsing notes in a rich editor
- **Dual storage** with markdown files as source of truth and SQLite for indexing
- **Multi-process concurrency** with per-process in-memory databases and git-based conflict detection
- **Backup and restore** with labeled snapshots and safety checks
- **Hardware-aware auto-configuration** that detects GPU/RAM and tunes batch size, max tokens, and memory budget (env var overrides always win)
- **Auto-setup** of semantic dependencies and model pre-download on first startup

## Examples

- Knowledge creation: [A small Zettelkasten knowledge network about the Zettelkasten method itself](https://github.com/entanglr/znote-mcp/discussions/5)

## Available MCP Tools

All tools are prefixed with `zk_` for organization. The server exposes 22 tools:

### Notes

| Tool | Description |
|------|-------------|
| `zk_create_note` | Create a note with title, content, type, project, tags, and optional plan ID |
| `zk_get_note` | Retrieve a note by ID or title (summary or raw markdown format) |
| `zk_update_note` | Update a note's content or metadata; supports batch project moves via comma-separated IDs |
| `zk_delete_note` | Delete one or more notes; supports batch delete via comma-separated IDs |
| `zk_note_history` | View git commit history for a note (requires git versioning enabled) |
| `zk_bulk_create_notes` | Create multiple notes atomically from a JSON array (all-or-nothing) |

### Links

| Tool | Description |
|------|-------------|
| `zk_create_link` | Create a typed link between two notes (optionally bidirectional) |
| `zk_remove_link` | Remove a link between two notes (optionally bidirectional) |

### Search & Discovery

| Tool | Description |
|------|-------------|
| `zk_search_notes` | Search by text, tags, type, or semantically; auto-selects best strategy |
| `zk_fts_search` | Full-text search with FTS5 syntax (boolean, phrases, prefix, column filters) |
| `zk_list_notes` | List notes by date, project, connectivity (central/orphan), with pagination |
| `zk_find_related` | Find linked, similar, or semantically related notes |

### Tags

| Tool | Description |
|------|-------------|
| `zk_add_tag` | Add tags to notes (batch: comma-separated IDs and/or tags) |
| `zk_remove_tag` | Remove tags from notes (batch: comma-separated IDs and/or tags) |
| `zk_cleanup_tags` | Delete orphaned tags not associated with any notes |

### Projects

| Tool | Description |
|------|-------------|
| `zk_create_project` | Create a project (supports hierarchical sub-projects via `/`) |
| `zk_list_projects` | List all projects with optional note counts |
| `zk_get_project` | Get project details including note count and children |
| `zk_delete_project` | Delete an empty project (must have no notes or sub-projects) |

### System

| Tool | Description |
|------|-------------|
| `zk_status` | Dashboard with note counts, tags, health, embeddings, metrics, and config |
| `zk_system` | Admin operations: rebuild index, sync to Obsidian, backup, reindex embeddings |
| `zk_restore` | Restore database from a backup (creates safety backup first) |

## Note Types

| Type | Handle | Description |
|------|--------|-------------|
| **Fleeting notes** | `fleeting` | Quick, temporary notes for capturing ideas |
| **Literature notes** | `literature` | Notes from reading material |
| **Permanent notes** | `permanent` | Well-formulated, evergreen notes |
| **Structure notes** | `structure` | Index or outline notes that organize other notes |
| **Hub notes** | `hub` | Entry points to the Zettelkasten on key topics |

## Link Types

Each link type has a semantic inverse, creating a rich multi-dimensional knowledge graph:

| Primary Link Type | Inverse Link Type | Relationship Description |
|-------------------|-------------------|--------------------------|
| `reference` | `reference` | Simple reference to related information (symmetric) |
| `extends` | `extended_by` | One note builds upon or develops concepts from another |
| `refines` | `refined_by` | One note clarifies or improves upon another |
| `contradicts` | `contradicted_by` | One note presents opposing views to another |
| `questions` | `questioned_by` | One note poses questions about another |
| `supports` | `supported_by` | One note provides evidence for another |
| `related` | `related` | Generic relationship (symmetric) |

## Semantic Search

Semantic search uses ONNX-based embedding models to find conceptually related notes, even without shared keywords. This is an optional feature requiring extra dependencies.

### Setup

```bash
# CPU-only (default)
pip install znote-mcp[semantic]

# GPU acceleration (NVIDIA CUDA 12.x, x86_64 Linux/Windows only)
pip install znote-mcp[semantic-gpu]
```

> **Note**: `onnxruntime` and `onnxruntime-gpu` are mutually exclusive packages. Do not install both `[semantic]` and `[semantic-gpu]`.

When semantic dependencies are installed, embeddings auto-enable on startup. The server auto-downloads models on first run.

### How It Works

1. **Embedding**: Notes are embedded using [Alibaba-NLP/gte-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-modernbert-base) (768-dim, Apache-2.0)
2. **Chunking**: Notes longer than `ZETTELKASTEN_EMBEDDING_CHUNK_SIZE` tokens are split into overlapping chunks, each embedded separately
3. **Vector search**: Queries are embedded and matched against note vectors using sqlite-vec (L2 distance on L2-normalized vectors = cosine similarity)
4. **Reranking**: Optional cross-encoder reranker ([Alibaba-NLP/gte-reranker-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base)) improves result quality
5. **Deduplication**: Multi-chunk notes are deduplicated in results, keeping the best-matching chunk

### Configuration

All embedding settings are in `.env.example`. Key variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `ZETTELKASTEN_EMBEDDINGS_ENABLED` | `false` | Auto-enables when deps are installed; set `false` to force off |
| `ZETTELKASTEN_EMBEDDING_MODEL` | `Alibaba-NLP/gte-modernbert-base` | HuggingFace model ID |
| `ZETTELKASTEN_EMBEDDING_DIM` | `768` | Must match model output dimension |
| `ZETTELKASTEN_EMBEDDING_MAX_TOKENS` | `2048` | Max tokens per embedding input |
| `ZETTELKASTEN_EMBEDDING_BATCH_SIZE` | `8` | Batch size for reindex operations |
| `ZETTELKASTEN_EMBEDDING_CHUNK_SIZE` | `2048` | Tokens per chunk for long notes |
| `ZETTELKASTEN_EMBEDDING_CHUNK_OVERLAP` | `256` | Overlap tokens between chunks |
| `ZETTELKASTEN_ONNX_PROVIDERS` | `auto` | `auto`, `cpu`, or explicit provider list |
| `ZETTELKASTEN_ONNX_QUANTIZED` | `false` | Use INT8 quantized ONNX models (~4x smaller, ~97% quality) |
| `ZETTELKASTEN_EMBEDDING_MEMORY_BUDGET_GB` | `6.0` | Memory budget in GB for adaptive batching (auto-tuned) |
| `ZETTELKASTEN_EMBEDDING_CACHE_DIR` | HF default | Custom model cache directory |
| `ZETTELKASTEN_RERANKER_MODEL` | `Alibaba-NLP/gte-reranker-modernbert-base` | Reranker model ID |
| `ZETTELKASTEN_RERANKER_MAX_TOKENS` | `2048` | Max input tokens for reranker model (auto-tuned) |
| `ZETTELKASTEN_RERANKER_IDLE_TIMEOUT` | `600` | Seconds before idle reranker is unloaded |

### Hardware Auto-Tuning

On startup, the server detects your GPU (via `nvidia-smi`) and system RAM, then selects optimal defaults for batch size, max tokens, and memory budget. Explicit environment variable overrides always take priority over auto-detected values.

| Tier | Batch Size | Embed Max Tokens | Rerank Max Tokens | Memory Budget |
|------|-----------|------------------|-------------------|---------------|
| gpu-16gb+ | 64 | 8192 | 8192 | 10 GB |
| gpu-8gb+ | 32 | 4096 | 4096 | 6 GB |
| gpu-small | 16 | 2048 | 2048 | 3 GB |
| cpu-32gb+ | 16 | 8192 | 4096 | 8 GB |
| cpu-16gb+ | 8 | 4096 | 2048 | 4 GB |
| cpu-8gb+ | 4 | 2048 | 1024 | 2 GB |
| cpu-small | 2 | 512 | 512 | 1 GB |

The detected tier is logged at startup (e.g., `device_label: gpu-8gb+ (NVIDIA RTX 4070)`). To override any auto-tuned value, set the corresponding environment variable.

### Model Selection

The default embedding and reranker models were selected through benchmarking on a real 961-note Zettelkasten knowledge base.

**Embedding model**: [Alibaba-NLP/gte-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-modernbert-base) was chosen after evaluating 9 models across 12 configurations (FP32 and INT8, varying chunk sizes). Models benchmarked: all-MiniLM-L6-v2, bge-small-en-v1.5, bge-base-en-v1.5, gte-modernbert-base, nomic-embed-text-v1.5, snowflake-arctic-embed-m-v2.0, snowflake-arctic-embed-l-v2.0, mxbai-embed-large-v1, and embeddinggemma-300m. Quality was measured by link prediction MRR (mean reciprocal rank) and tag coherence; performance was measured by throughput (chunks/s), peak memory, and GPU vs CPU scaling. gte-modernbert-base offers the best quality-to-performance tradeoff with native long-context support (8192 tokens).

**Reranker model**: [Alibaba-NLP/gte-reranker-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base) was chosen after evaluating 10 reranker models: ms-marco-MiniLM-L-6-v2, ms-marco-MiniLM-L-12-v2, bge-reranker-base, bge-reranker-large, gte-reranker-modernbert-base, bge-reranker-v2-m3, jina-reranker-v2-base-multilingual, stsb-distilroberta-base, stsb-roberta-base, and stsb-roberta-large. The gte-reranker is the only model that consistently improves retrieval quality across all semantic challenge categories while supporting long contexts (8192 tokens).

Full benchmark data is available in the `benchmarks/` directory. Benchmark scripts are in `scripts/` (`benchmark_embed.py`, `benchmark_quality.py`, `benchmark_rerank.py`).

## Storage Architecture

This system uses a dual storage approach:

1. **Markdown Files**: All notes are stored as human-readable Markdown files with YAML frontmatter for metadata. These files are the **source of truth** and can be:
   - Edited directly in any text editor
   - Placed under version control (Git, etc.)
   - Backed up using standard file backup procedures
   - Shared or transferred like any other text files

2. **SQLite Database**: Functions as an indexing layer that:
   - Facilitates efficient querying and search operations via FTS5
   - Enables Claude to quickly traverse the knowledge graph
   - Stores embedding vectors for semantic search (via sqlite-vec)
   - Is automatically rebuilt from Markdown files when needed

If you edit Markdown files directly outside the system, run `zk_system(action="rebuild")` to update the database index. The database can be deleted at any time — it will be regenerated from your Markdown files.

## Concurrency Model

This system is designed for safe multi-process access, enabling multiple Claude Code instances or other MCP clients to work with your notes simultaneously.

### How It Works

1. **Per-Process In-Memory Database**: Each process maintains its own in-memory SQLite database, eliminating lock contention. The database is rebuilt from markdown files on startup.

2. **Git-Based Version Control**: Notes are versioned using git commits. Each create/update operation produces a commit hash that serves as a version identifier.

3. **Optimistic Concurrency Control**: When updating or deleting notes, you can provide an `expected_version` parameter. If another process modified the note, you'll receive a `CONFLICT` response instead of silently overwriting changes.

### Usage Example

```
# Read a note (returns version hash)
zk_get_note("my-note-id")
# Output includes: Version: abc1234

# Update with version check
zk_update_note(
    note_id="my-note-id",
    content="New content",
    expected_version="abc1234"  # From zk_get_note
)

# If another process updated the note, you get:
# CONFLICT: Note was modified by another process...
# Re-read the note with zk_get_note to get the latest version.
```

### Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ZETTELKASTEN_GIT_ENABLED` | `true` | Enable git versioning for conflict detection |
| `ZETTELKASTEN_IN_MEMORY_DB` | `true` | Use per-process in-memory SQLite (recommended) |

See `.env.example` for full documentation.

## Installation

```bash
# Clone the repository
git clone https://github.com/AlexK-Notable/znote-mcp.git
cd znote-mcp

# Create a virtual environment with uv
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv sync

# (Optional) Install semantic search dependencies
uv pip install -e ".[semantic]"

# (Optional) Install with GPU support instead
uv pip install -e ".[semantic-gpu]"

# Install dev dependencies
uv sync --dev
```

## Configuration

Create a `.env` file by copying the example:

```bash
cp .env.example .env
```

The configuration file can be placed at:
- `~/.zettelkasten/.env` (recommended — survives updates)
- `<project-root>/.env` (development use)

Priority: process env (`.mcp.json`) > project `.env` > `~/.zettelkasten/.env` > defaults.

Core settings:

```bash
ZETTELKASTEN_NOTES_DIR=~/.zettelkasten/notes
ZETTELKASTEN_DATABASE_PATH=~/.zettelkasten/db/zettelkasten.db
ZETTELKASTEN_LOG_LEVEL=INFO
```

See `.env.example` for all available settings including semantic search, GPU, and Obsidian vault mirroring.

## Usage

### Starting the Server

```bash
python -m znote_mcp.main
```

Or with explicit configuration:

```bash
python -m znote_mcp.main --notes-dir ./data/notes --database-path ./data/db/zettelkasten.db
```

### Connecting to Claude Desktop

Add the following configuration to your Claude Desktop:

```json
{
  "mcpServers": {
    "znote": {
      "command": "/absolute/path/to/znote-mcp/.venv/bin/python",
      "args": [
        "-m",
        "znote_mcp.main"
      ],
      "env": {
        "ZETTELKASTEN_NOTES_DIR": "/absolute/path/to/znote-mcp/data/notes",
        "ZETTELKASTEN_DATABASE_PATH": "/absolute/path/to/znote-mcp/data/db/zettelkasten.db",
        "ZETTELKASTEN_LOG_LEVEL": "INFO"
      }
    }
  }
}
```

## Prompting

To ensure maximum effectiveness, we recommend using a system prompt ("project instructions"), project knowledge, and an appropriate chat prompt when asking the LLM to process information, or explore or synthesize your Zettelkasten notes. The `docs` directory in this repository contains the necessary files to get you started:

### System prompts

Pick one:

- [system-prompt.md](docs/prompts/system/system-prompt.md)
- [system-prompt-with-protocol.md](docs/prompts/system/system-prompt-with-protocol.md)

### Project knowledge

For end users:

- [zettelkasten-methodology-technical.md](docs/project-knowledge/user/zettelkasten-methodology-technical.md)
- [link-types-in-zettelkasten-mcp-server.md](docs/project-knowledge/user/link-types-in-zettelkasten-mcp-server.md)
- (more info relevant to your project)

### Chat Prompts

- [chat-prompt-knowledge-creation.md](docs/prompts/chat/chat-prompt-knowledge-creation.md)
- [chat-prompt-knowledge-creation-batch.md](docs/prompts/chat/chat-prompt-knowledge-creation-batch.md)
- [chat-prompt-knowledge-exploration.md](docs/prompts/chat/chat-prompt-knowledge-exploration.md)
- [chat-prompt-knowledge-synthesis.md](docs/prompts/chat/chat-prompt-knowledge-synthesis.md)

### Project knowledge (dev)

For developers and contributors:

- [Example - A simple MCP server.md](docs/project-knowledge/dev/Example%20-%20A%20simple%20MCP%20server%20that%20exposes%20a%20website%20fetching%20tool.md)
- [MCP Python SDK-README.md](docs/project-knowledge/dev/MCP%20Python%20SDK-README.md)
- [llms-full.txt](docs/project-knowledge/dev/llms-full.txt)

NB: Optionally include the source code with a tool like [repomix](https://github.com/yamadashy/repomix).

## Project Structure

```
znote-mcp/
├── src/
│   └── znote_mcp/
│       ├── models/              # Pydantic schemas and SQLAlchemy ORM models
│       ├── storage/             # Repository layer
│       │   ├── base.py              # Abstract repository interface
│       │   ├── note_repository.py   # Dual storage (markdown + SQLite + embeddings)
│       │   ├── tag_repository.py    # Tag CRUD and batch operations
│       │   ├── link_repository.py   # Semantic link management
│       │   ├── project_repository.py # Hierarchical project registry
│       │   ├── fts_index.py         # FTS5 full-text search index
│       │   ├── git_wrapper.py       # Git versioning for concurrency
│       │   ├── obsidian_mirror.py   # Obsidian vault sync
│       │   └── markdown_parser.py   # Frontmatter parsing
│       ├── services/            # Business logic
│       │   ├── zettel_service.py    # Core CRUD, links, tags, bulk ops
│       │   ├── search_service.py    # Text, FTS5, and semantic search
│       │   ├── embedding_service.py # Embedding lifecycle and reindexing
│       │   ├── embedding_types.py   # Provider interfaces
│       │   ├── onnx_providers.py    # ONNX Runtime embedding + reranker
│       │   └── text_chunker.py      # Token-aware note chunking
│       ├── server/              # MCP server with 22 tools
│       ├── config.py            # Pydantic config with env var support
│       ├── hardware.py          # Hardware detection and auto-tuning
│       ├── setup_manager.py     # Auto-install semantic deps + model warmup
│       ├── backup.py            # Backup and restore operations
│       ├── observability.py     # Structured logging and metrics
│       ├── exceptions.py        # Error hierarchy with codes
│       └── main.py              # Entry point
├── tests/                       # 36 test files, 700+ tests
├── alembic/                     # Database migrations
├── docs/                        # Prompts, project knowledge, design docs
├── scripts/                     # Utility scripts
├── .env.example                 # Full configuration reference
├── CLAUDE.md                    # Claude Code project context
└── README.md
```

## Tests

Comprehensive test suite with 780+ tests covering all layers from models to MCP server integration.

### Running Tests

```bash
# Run all tests
uv run pytest -v tests/

# With coverage report
uv run pytest --cov=znote_mcp --cov-report=term-missing tests/

# Run a specific test file
uv run pytest -v tests/test_models.py

# Run E2E tests (isolated environment)
uv run pytest tests/test_e2e.py -v

# Debug E2E with persistent data
ZETTELKASTEN_TEST_PERSIST=1 uv run pytest tests/test_e2e.py -v
```

### Test Categories

| Category | Files | Description |
|----------|-------|-------------|
| Unit | `test_models.py`, `test_note_repository.py`, `test_search_service.py`, `test_zettel_service.py`, `test_git_wrapper.py`, `test_config_cleanup.py`, `test_hardware.py` | Individual component tests |
| Integration | `test_integration.py`, `test_mcp_integration.py`, `test_mcp_tools_integration.py`, `test_mcp_protocol.py` | Cross-layer integration |
| E2E | `test_e2e.py`, `test_e2e_workflows.py` | Full system workflows with isolated environments |
| Embeddings | `test_embedding_phase1.py` through `test_embedding_phase5.py`, `test_chunked_embedding_integration.py` | Semantic search pipeline (chunking, storage, search, reindex) |
| Resilience | `test_error_injection.py`, `test_failure_recovery.py`, `test_database_hardening.py` | Error handling and recovery |
| Concurrency | `test_concurrency.py`, `test_multiprocess_concurrency.py`, `test_versioned_operations.py` | Multi-process safety |
| Features | `test_semantic_links.py`, `test_bulk_operations.py`, `test_project_repository.py`, `test_obsidian_enhancements.py`, `test_unicode_edge_cases.py` | Feature-specific tests |
| Operations | `test_backup_workflows.py`, `test_migration_and_versioning.py`, `test_setup_manager.py`, `test_performance_baselines.py`, `test_observability.py`, `test_bug_fixes.py` | Operational concerns |

## Important Notice

This software is experimental and provided as-is without warranty of any kind. While efforts have been made to ensure data integrity, it may contain bugs that could potentially lead to data loss or corruption. Always back up your notes regularly and use caution when testing with important information.

## Acknowledgments

- [Peter J. Herrel (@diggy)](https://github.com/diggy) and [Entanglr](https://github.com/entanglr) - Original creators of [zettelkasten-mcp](https://github.com/entanglr/zettelkasten-mcp)
- This MCP server was crafted with the assistance of Claude, who helped organize the atomic thoughts of this project into a coherent knowledge graph.

## License

MIT License - see the [LICENSE](LICENSE) file for details.
