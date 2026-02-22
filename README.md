# znote-mcp

A Zettelkasten knowledge management server for the [Model Context Protocol](https://modelcontextprotocol.io). Create, link, search, and synthesize atomic notes through Claude and other MCP-compatible clients.

> **Origin**: This project began as a fork of [entanglr/zettelkasten-mcp](https://github.com/entanglr/zettelkasten-mcp) by [Peter J. Herrel](https://github.com/diggy) and has since diverged substantially.

## What It Does

znote-mcp gives AI assistants a persistent, structured memory. Instead of losing context between conversations, your notes accumulate into a knowledge network where ideas connect, patterns emerge, and past work informs future thinking.

You write notes. The system links them, indexes them, and makes them searchable — by keyword, by concept, or by meaning. Over time, a few hundred notes become a knowledge graph that surfaces connections you wouldn't have found on your own.

### Quick Start

```bash
# Install
git clone https://github.com/AlexK-Notable/znote-mcp.git && cd znote-mcp
uv venv && source .venv/bin/activate
uv sync

# Run
python -m znote_mcp.main

# Optional: semantic search (find notes by meaning, not just keywords)
uv pip install -e ".[semantic]"       # CPU
uv pip install -e ".[semantic-gpu]"   # NVIDIA GPU
```

Add to Claude Desktop:

```json
{
  "mcpServers": {
    "znote": {
      "command": "/path/to/znote-mcp/.venv/bin/python",
      "args": ["-m", "znote_mcp.main"],
      "env": {
        "ZETTELKASTEN_NOTES_DIR": "~/.zettelkasten/notes",
        "ZETTELKASTEN_DATABASE_PATH": "~/.zettelkasten/db/zettelkasten.db"
      }
    }
  }
}
```

Configuration lives in `~/.zettelkasten/.env` (survives updates) or `<project>/.env`. See `.env.example` for all options.

### The Zettelkasten Method

The system follows the Zettelkasten method developed by Niklas Luhmann, who used it to produce over 70 books and hundreds of articles. Three principles:

1. **Atomicity** — each note holds one idea
2. **Connectivity** — notes link to each other with typed relationships
3. **Emergence** — the network reveals patterns the individual notes don't

The result is both vertical depth (following a thread deeper into a topic) and horizontal breadth (discovering unexpected connections across domains). Luhmann called his system a "communication partner" — this is a digital implementation of that idea.

---

## Features

### 22 MCP Tools

The server exposes 22 tools, all prefixed with `zk_`:

**Notes** — `zk_create_note`, `zk_get_note`, `zk_update_note`, `zk_delete_note`, `zk_note_history`, `zk_bulk_create_notes`

**Links** — `zk_create_link`, `zk_remove_link`

**Search** — `zk_search_notes` (auto-selects strategy), `zk_fts_search` (FTS5 with boolean/phrase/prefix), `zk_list_notes` (by date, project, connectivity), `zk_find_related` (linked, similar, or semantic)

**Tags** — `zk_add_tag`, `zk_remove_tag`, `zk_cleanup_tags`

**Projects** — `zk_create_project`, `zk_list_projects`, `zk_get_project`, `zk_delete_project`

**System** — `zk_status` (dashboard), `zk_system` (rebuild, sync, backup, reindex), `zk_restore`

### Note Types and Link Types

Five note types capture different stages of thought:

| Type | Purpose |
|------|---------|
| `fleeting` | Quick captures — ideas, observations, things to revisit |
| `literature` | Notes from reading material |
| `permanent` | Refined, evergreen notes — the core of the system |
| `structure` | Index or outline notes that organize others |
| `hub` | Entry points to major topics |

Seven link types with semantic inverses create a multi-dimensional knowledge graph:

| Link | Inverse | Meaning |
|------|---------|---------|
| `reference` | `reference` | Related information (symmetric) |
| `extends` | `extended_by` | Builds on concepts from another note |
| `refines` | `refined_by` | Clarifies or improves another note |
| `contradicts` | `contradicted_by` | Presents opposing views |
| `questions` | `questioned_by` | Poses questions about another note |
| `supports` | `supported_by` | Provides evidence for another note |
| `related` | `related` | Generic relationship (symmetric) |

### Semantic Search

Beyond keyword matching, semantic search finds notes by *meaning*. A search for "distributed consensus algorithms" will surface your notes about Raft, Paxos, and Byzantine fault tolerance — even if those exact words never appear in the query.

How it works:

1. Notes are embedded into 768-dimensional vectors using a local ONNX model
2. Long notes are split into overlapping chunks, each embedded separately
3. Queries are embedded and matched against note vectors via cosine similarity (sqlite-vec)
4. A cross-encoder reranker rescores the top candidates for higher precision
5. Multi-chunk notes are deduplicated, keeping the best-matching chunk

Everything runs locally. No API calls, no data leaves your machine.

When the `[semantic]` extra is installed, embeddings auto-enable on startup and models are downloaded in the background on first run.

### Hardware-Aware Auto-Tuning

The server detects your hardware on startup and configures itself accordingly. GPU memory, system RAM, and CPU architecture determine batch sizes, token limits, and memory budgets. You don't need to tune anything — but if you want to, every auto-detected value can be overridden with an environment variable.

| Tier | Batch Size | Embed Tokens | Rerank Tokens | Memory Budget |
|------|-----------|-------------|--------------|--------------|
| GPU 16GB+ | 64 | 8192 | 8192 | 10 GB |
| GPU 8GB+ | 32 | 4096 | 4096 | 6 GB |
| GPU small | 16 | 2048 | 2048 | 3 GB |
| CPU 32GB+ | 16 | 8192 | 4096 | 8 GB |
| CPU 16GB+ | 8 | 4096 | 2048 | 4 GB |
| CPU 8GB+ | 4 | 2048 | 1024 | 2 GB |
| CPU small | 2 | 512 | 512 | 1 GB |

The detected tier is logged at startup (e.g., `Hardware auto-tune: gpu-8gb+ (NVIDIA RTX 4070)`).

### Why These Models

The default embedding and reranker models weren't chosen arbitrarily — they were selected through benchmarking on a real 961-note Zettelkasten knowledge base.

**Embedding model**: [Alibaba-NLP/gte-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-modernbert-base) — selected after evaluating 9 models across 12 configurations (FP32 and INT8, varying chunk sizes). Quality was measured by link prediction MRR and tag coherence; performance by throughput and memory usage. gte-modernbert-base offers the best quality-to-performance ratio with native long-context support (8192 tokens). Models evaluated: all-MiniLM-L6-v2, bge-small-en-v1.5, bge-base-en-v1.5, gte-modernbert-base, nomic-embed-text-v1.5, snowflake-arctic-embed-m-v2.0, snowflake-arctic-embed-l-v2.0, mxbai-embed-large-v1, and Alibaba-NLP/gte-embedding-gemma-300m.

**Reranker model**: [Alibaba-NLP/gte-reranker-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base) — selected after evaluating 10 cross-encoder models across semantic challenge categories (synonym matching, conceptual relationships, negation handling, specificity ranking). It's the only model that consistently improves retrieval quality across all categories while supporting long contexts. Models evaluated: ms-marco-MiniLM-L-6-v2, ms-marco-MiniLM-L-12-v2, bge-reranker-base, bge-reranker-large, gte-reranker-modernbert-base, bge-reranker-v2-m3, jina-reranker-v2-base-multilingual, stsb-distilroberta-base, stsb-roberta-base, and stsb-roberta-large.

Full benchmark methodology and results are in [`docs/MODEL_SELECTION.md`](docs/MODEL_SELECTION.md). Raw data lives in `benchmarks/`. Reproduction scripts are in `scripts/`.

### Dual Storage

Notes are stored as plain Markdown files with YAML frontmatter. That's the source of truth — you can edit them in any text editor, version them with Git, back them up however you like. They're just files.

SQLite sits alongside as an indexing layer: FTS5 for full-text search, sqlite-vec for vector search, and standard tables for the link graph. The database is disposable — delete it and it rebuilds from your Markdown files.

### Multi-Process Safety

Multiple Claude Code instances (or other MCP clients) can safely work with the same notes simultaneously:

- Each process gets its own in-memory SQLite database — no lock contention
- Git commits on every write provide version hashes for optimistic concurrency control
- Pass `expected_version` on updates to detect conflicts instead of silently overwriting

### Projects and Organization

Notes are organized into projects (with hierarchical sub-projects via `/`). Project moves are batch-capable. Obsidian vault mirroring copies notes into a vault directory for browsing in Obsidian, with wikilinks rewritten to match.

---

## Configuration Reference

Create `~/.zettelkasten/.env` (recommended) or `<project>/.env`. Priority: process env > project `.env` > `~/.zettelkasten/.env` > defaults.

### Core

| Variable | Default | Description |
|----------|---------|-------------|
| `ZETTELKASTEN_NOTES_DIR` | `~/.zettelkasten/notes` | Markdown file storage |
| `ZETTELKASTEN_DATABASE_PATH` | `~/.zettelkasten/db/zettelkasten.db` | SQLite database path |
| `ZETTELKASTEN_LOG_LEVEL` | `INFO` | Logging level |
| `ZETTELKASTEN_GIT_ENABLED` | `true` | Git versioning for conflict detection |
| `ZETTELKASTEN_IN_MEMORY_DB` | `true` | Per-process in-memory SQLite |
| `ZETTELKASTEN_OBSIDIAN_VAULT` | *(unset)* | Obsidian vault path for mirroring |

### Semantic Search

Requires `pip install znote-mcp[semantic]` (CPU) or `znote-mcp[semantic-gpu]` (NVIDIA CUDA 12.x). These are mutually exclusive — don't install both.

| Variable | Default | Description |
|----------|---------|-------------|
| `ZETTELKASTEN_EMBEDDINGS_ENABLED` | `false` | Auto-enables when deps installed; set `false` to force off |
| `ZETTELKASTEN_EMBEDDING_MODEL` | `Alibaba-NLP/gte-modernbert-base` | Embedding model (HuggingFace ID) |
| `ZETTELKASTEN_RERANKER_MODEL` | `Alibaba-NLP/gte-reranker-modernbert-base` | Reranker model |
| `ZETTELKASTEN_EMBEDDING_DIM` | `768` | Must match model output dimension |
| `ZETTELKASTEN_EMBEDDING_MAX_TOKENS` | `2048` | Max tokens per embedding input (auto-tuned) |
| `ZETTELKASTEN_RERANKER_MAX_TOKENS` | `2048` | Max tokens for reranker input (auto-tuned) |
| `ZETTELKASTEN_EMBEDDING_BATCH_SIZE` | `8` | Batch size for reindex (auto-tuned) |
| `ZETTELKASTEN_EMBEDDING_CHUNK_SIZE` | `2048` | Token threshold for splitting long notes |
| `ZETTELKASTEN_EMBEDDING_CHUNK_OVERLAP` | `256` | Overlap tokens between chunks |
| `ZETTELKASTEN_EMBEDDING_MEMORY_BUDGET_GB` | `6.0` | Memory budget for adaptive batching (auto-tuned) |
| `ZETTELKASTEN_ONNX_PROVIDERS` | `auto` | `auto`, `cpu`, or explicit provider list |
| `ZETTELKASTEN_ONNX_QUANTIZED` | `false` | INT8 quantized models (~4x smaller, ~97% quality) |
| `ZETTELKASTEN_EMBEDDING_CACHE_DIR` | HF default | Custom model cache directory |
| `ZETTELKASTEN_RERANKER_IDLE_TIMEOUT` | `600` | Seconds before idle reranker unloads (0 = never) |

See `.env.example` for full documentation including memory usage guidance.

## Prompting

System prompts, project knowledge, and chat prompts are provided in the `docs/` directory:

**System prompts** — [`system-prompt.md`](docs/prompts/system/system-prompt.md), [`system-prompt-with-protocol.md`](docs/prompts/system/system-prompt-with-protocol.md)

**Project knowledge** — [`zettelkasten-methodology-technical.md`](docs/project-knowledge/user/zettelkasten-methodology-technical.md), [`link-types-in-zettelkasten-mcp-server.md`](docs/project-knowledge/user/link-types-in-zettelkasten-mcp-server.md)

**Chat prompts** — [`knowledge-creation.md`](docs/prompts/chat/chat-prompt-knowledge-creation.md), [`knowledge-creation-batch.md`](docs/prompts/chat/chat-prompt-knowledge-creation-batch.md), [`knowledge-exploration.md`](docs/prompts/chat/chat-prompt-knowledge-exploration.md), [`knowledge-synthesis.md`](docs/prompts/chat/chat-prompt-knowledge-synthesis.md)

**Developer docs** — [`Example MCP server`](docs/project-knowledge/dev/Example%20-%20A%20simple%20MCP%20server%20that%20exposes%20a%20website%20fetching%20tool.md), [`MCP Python SDK`](docs/project-knowledge/dev/MCP%20Python%20SDK-README.md)

- [Example: A Zettelkasten about the Zettelkasten method](https://github.com/entanglr/znote-mcp/discussions/5)

---

## Architecture

### Layer Structure

```
MCP Tools (server/mcp_server.py) ─── 22 registered tools
    │
Services ─── business logic
    ├── zettel_service.py ─── CRUD, links, tags, bulk ops, embedding on write
    ├── search_service.py ─── text, FTS5, semantic, graph traversal
    └── embedding_service.py ─── thread-safe embedding/reranking lifecycle
    │
Repositories ─── storage abstraction
    ├── note_repository.py ─── dual storage (markdown + SQLite + vectors)
    ├── tag_repository.py ─── tag CRUD and batch ops
    ├── link_repository.py ─── semantic link graph
    ├── project_repository.py ─── hierarchical project registry
    └── fts_index.py ─── FTS5 full-text index
    │
Infrastructure
    ├── hardware.py ─── GPU/RAM detection, 7-tier auto-tuning
    ├── onnx_providers.py ─── ONNX Runtime embedding + reranker
    ├── text_chunker.py ─── token-aware chunking with sentence boundaries
    ├── git_wrapper.py ─── git versioning for concurrency
    ├── obsidian_mirror.py ─── vault sync with wikilink rewriting
    └── setup_manager.py ─── auto-install deps, model warmup
```

### Project Structure

```
znote-mcp/
├── src/znote_mcp/
│   ├── models/                  # Pydantic schemas + SQLAlchemy ORM
│   ├── storage/                 # Repository layer
│   │   ├── base.py                  # Abstract repository interface
│   │   ├── note_repository.py       # Markdown + SQLite + sqlite-vec
│   │   ├── tag_repository.py        # Tag operations
│   │   ├── link_repository.py       # Link graph
│   │   ├── project_repository.py    # Hierarchical projects
│   │   ├── fts_index.py             # FTS5 index
│   │   ├── git_wrapper.py           # Git versioning
│   │   ├── obsidian_mirror.py       # Obsidian sync
│   │   └── markdown_parser.py       # Frontmatter parsing
│   ├── services/                # Business logic
│   │   ├── zettel_service.py        # Core CRUD + embedding on write
│   │   ├── search_service.py        # All search strategies
│   │   ├── embedding_service.py     # Embedding lifecycle
│   │   ├── embedding_types.py       # Protocol interfaces (PEP 544)
│   │   ├── onnx_providers.py        # ONNX Runtime providers
│   │   └── text_chunker.py          # Token-aware chunking
│   ├── server/mcp_server.py     # MCP server (22 tools)
│   ├── config.py                # Pydantic config with env var support
│   ├── hardware.py              # Hardware detection + auto-tuning
│   ├── setup_manager.py         # Semantic dep auto-install + model warmup
│   ├── backup.py                # Snapshot backup/restore
│   ├── observability.py         # Structured logging and metrics
│   ├── exceptions.py            # Error hierarchy with codes
│   └── main.py                  # Entry point
├── tests/                       # 36 test files, 782 tests
├── benchmarks/                  # Embedding + reranker benchmark data
├── scripts/                     # Benchmark and utility scripts
├── alembic/                     # Database migrations
├── docs/                        # Prompts, design docs, model selection guide
└── .env.example                 # Full configuration reference
```

### Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Protocol | [MCP](https://modelcontextprotocol.io) (Python SDK) | Standard for AI tool integration |
| Database | SQLite (WAL mode) + [sqlite-vec](https://github.com/asg017/sqlite-vec) | Zero-config, embedded, vector-capable |
| Search | FTS5 + sqlite-vec KNN | Full-text and semantic in one database |
| Embeddings | [ONNX Runtime](https://onnxruntime.ai/) | Local inference, no API dependency |
| Embedding model | [gte-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-modernbert-base) (768-dim) | Best quality/performance ratio ([benchmarked](docs/MODEL_SELECTION.md)) |
| Reranker | [gte-reranker-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base) | Only model that improves all categories ([benchmarked](docs/MODEL_SELECTION.md)) |
| Tokenizer | HuggingFace `tokenizers` | Fast, model-aligned tokenization |
| Config | Pydantic + python-dotenv | Typed config with env var support |
| ORM | SQLAlchemy 2.0 | Database abstraction |
| Migrations | Alembic | Schema versioning |
| Versioning | Git (subprocess) | Optimistic concurrency control |
| Python | 3.10+ | Type hints, structural pattern matching |

### Embedding Interfaces

Embedding and reranker providers use `typing.Protocol` (PEP 544) for structural subtyping. Any object with the right method signatures works — no base class inheritance required. This makes testing straightforward: swap in a fake provider that returns deterministic vectors.

### Adaptive Batching

Rather than fixed batch sizes, the embedding system uses greedy adaptive batching. Given a memory budget and the actual token lengths of pending texts, it packs as many items as possible into each batch without exceeding the budget. Short notes get large batches (fast); long notes get small batches (safe). This replaces the earlier fixed-bucket approach and improves reindex throughput significantly on mixed-length corpora.

### INT8 Quantization

Quantized ONNX models are supported for both embedding and reranking. INT8 models are roughly 4x smaller (143MB vs 569MB) with approximately 97% quality retention. Performance impact is hardware-dependent — faster on some platforms, slower on others. Set `ZETTELKASTEN_ONNX_QUANTIZED=true` to enable. Falls back to FP32 if quantized model files aren't found.

### Tests

782 tests across 36 files covering unit, integration, E2E, protocol, embedding, concurrency, and resilience:

```bash
uv run pytest -v tests/                                    # All tests
uv run pytest --cov=znote_mcp --cov-report=term-missing    # With coverage
uv run pytest tests/test_e2e.py -v                         # E2E (isolated)
uv run pytest tests/test_mcp_protocol.py -v                # MCP protocol (34 tests, no mocking)
```

| Category | What it covers |
|----------|---------------|
| Unit | Models, repositories, services, config, hardware detection |
| Integration | Cross-layer, MCP tool wiring, protocol round-trips |
| E2E | Full system workflows in isolated environments |
| Embeddings | Provider, service, chunking, search, reindex (5 phased files) |
| Concurrency | Thread safety, multi-process, versioned operations |
| Resilience | Error injection, failure recovery, database hardening |

---

## Important Notice

This software is experimental and provided as-is without warranty of any kind. While efforts have been made to ensure data integrity, it may contain bugs that could potentially lead to data loss or corruption. Always back up your notes regularly.

## Acknowledgments

- [Peter J. Herrel (@diggy)](https://github.com/diggy) and [Entanglr](https://github.com/entanglr) — original creators of [zettelkasten-mcp](https://github.com/entanglr/zettelkasten-mcp)
- [Alibaba NLP](https://huggingface.co/Alibaba-NLP) — gte-modernbert-base and gte-reranker-modernbert-base models (Apache-2.0)
- This MCP server was built with the assistance of Claude, who helped organize the atomic thoughts of this project into a coherent knowledge graph.

## License

MIT License — see [LICENSE](LICENSE) for details.
