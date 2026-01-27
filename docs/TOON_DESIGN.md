# TOON Integration Design Document

## znote-mcp Zettelkasten Knowledge Management Server

**Document Status:** Design Rationale
**Scope:** Conceptual analysis of TOON adoption for MCP tool responses

---

## Executive Summary

znote-mcp presents a **selective adoption** case for TOON. Unlike flat data systems, Zettelkasten notes are inherently relational—each note contains nested `tags[]` and `links[]` arrays that break pure tabular eligibility. However, specific tools and data views exhibit strong TOON fit, particularly full-text search results and bulk operation inputs.

Expected token savings: **15-25%** overall, with **35%+** on targeted high-fit endpoints.

---

## 1. The Zettelkasten Data Model Challenge

### Why Notes Are Different

A Zettelkasten note is not a flat record. It's a node in a knowledge graph:

```
Note
├── id (primitive)
├── title (primitive)
├── content (primitive - but potentially large)
├── note_type (enum)
├── project (primitive)
├── tags[] ← nested array of Tag objects
├── links[] ← nested array of Link objects with relationship metadata
├── created_at (primitive)
└── updated_at (primitive)
```

This nesting is fundamental to the Zettelkasten philosophy—notes derive meaning from their connections. Flattening `tags[]` and `links[]` into primitives would lose semantic information.

### The Tabular Eligibility Problem

TOON's tabular format requires:
1. Homogeneous arrays (every element has identical fields)
2. Primitive values (strings, numbers, booleans, null)
3. No nested objects or arrays within rows

A full note object violates requirement #2. The `tags[]` and `links[]` fields are arrays containing objects:

```json
{
  "tags": [{"name": "philosophy"}, {"name": "epistemology"}],
  "links": [
    {"source_id": "abc", "target_id": "def", "link_type": "extends", "description": "builds on concept"}
  ]
}
```

TOON cannot represent this as a simple tabular row without losing structure.

### The Strategic Question

Given this constraint, where does TOON provide value in znote-mcp?

**Answer:** In the *views* and *projections* of note data, not in full note representations.

---

## 2. Data View Analysis

### Tier 1: Excellent TOON Fit

These data views are naturally tabular and should adopt TOON.

#### Full-Text Search Results (`zk_fts_search`)

**Current structure:**
```json
[
  {"id": "20240115T103045...", "title": "Epistemology of Practice", "rank": 2.45, "snippet": "...the knowing-in-action...", "search_mode": "fts5"},
  {"id": "20240112T091230...", "title": "Tacit Knowledge", "rank": 1.89, "snippet": "...we know more than we can tell...", "search_mode": "fts5"},
  ...
]
```

**Why it fits TOON perfectly:**
- Homogeneous array: every result has identical fields
- All primitive values (strings, numbers)
- No nested structures
- Designed for scanning, not deep inspection
- Typical result count: 10-50 items

**TOON representation:**
```
results[N]{id,title,rank,snippet,search_mode}:
  20240115T103045...,Epistemology of Practice,2.45,...the knowing-in-action...,fts5
  20240112T091230...,Tacit Knowledge,1.89,...we know more than we can tell...,fts5
  ...
```

**Token impact:** ~35-40% reduction. This is the single best TOON target in znote-mcp.

**Rationale:** FTS results are *discovery* data—the LLM scans to find relevant notes, then fetches full notes separately. The tabular view optimizes the common case (scanning many results) without compromising the detailed case (reading one note).

---

#### Bulk Create Input (`zk_bulk_create_notes`)

**Current structure:**
```json
[
  {"title": "Note 1", "content": "...", "note_type": "permanent", "project": "philosophy", "tags": ["epistemology", "practice"]},
  {"title": "Note 2", "content": "...", "note_type": "literature", "project": "philosophy", "tags": ["tacit-knowledge"]},
  ...
]
```

**Why it fits TOON (with adaptation):**
- Homogeneous array
- Most fields are primitives
- `tags` can be represented as comma-joined string: `"epistemology,practice"`

**TOON representation:**
```
notes[N]{title,content,note_type,project,tags}:
  Note 1,...content...,permanent,philosophy,"epistemology,practice"
  Note 2,...content...,literature,philosophy,tacit-knowledge
  ...
```

**Token impact:** ~30% reduction on input.

**Rationale:** Bulk creation is an *efficiency* operation—the user wants to create many notes quickly. TOON reduces the token cost of expressing the batch, making larger bulk operations feasible within context limits.

**Trade-off:** Tags become a comma-joined string rather than a proper array. The server must parse this back. This is acceptable because:
1. Tags are simple strings without special characters
2. The parsing logic is trivial
3. The token savings justify the minor complexity

---

#### Note List Summaries (`zk_list_notes` - metadata only mode)

**Current structure (simplified view):**
```json
[
  {"id": "20240115T103045...", "title": "Epistemology of Practice", "note_type": "permanent", "project": "philosophy", "updated_at": "2024-01-15T10:30:45Z"},
  ...
]
```

**Why it fits TOON:**
- When returning *metadata only* (no tags, links, or content), notes become flat records
- All primitive values
- Consistent schema

**TOON representation:**
```
notes[N]{id,title,note_type,project,updated_at}:
  20240115T103045...,Epistemology of Practice,permanent,philosophy,2024-01-15T10:30:45Z
  ...
```

**Token impact:** ~30% reduction.

**Condition:** This requires a *metadata-only* response mode that excludes nested fields. If the full note (with tags/links) is returned, TOON benefit diminishes.

---

### Tier 2: Conditional TOON Fit

These views benefit from TOON under specific conditions.

#### Search Results with Tag Filter (`zk_search_notes`)

**Structure depends on query:**
- If returning note summaries (id, title, score): Excellent fit
- If returning full notes with tags/links: Poor fit

**Recommendation:** Implement two response modes:
1. **Summary mode** (default): TOON-encoded `{id, title, score, matched_context}`
2. **Full mode** (on request): JSON-encoded complete notes

The LLM workflow is typically:
1. Search → scan summaries (TOON)
2. Identify interesting notes → fetch full details (JSON via `zk_get_note`)

This maps naturally to TOON's strengths.

---

#### Bulk Tag Operations (`zk_bulk_add_tags`, `zk_bulk_remove_tags`)

**Current structure (input):**
- `note_ids`: comma-separated string
- `tags`: comma-separated string

**Analysis:** Already uses comma-separated primitives—essentially a simplified TOON-like encoding. No change needed; the format is already efficient.

---

### Tier 3: Poor TOON Fit

These structures should remain JSON-encoded.

#### Full Note Objects (`zk_get_note`)

**Structure:**
```json
{
  "id": "20240115T103045...",
  "title": "Epistemology of Practice",
  "content": "Long markdown content...\n\nWith multiple paragraphs...",
  "tags": [{"name": "philosophy"}, {"name": "epistemology"}],
  "links": [
    {"source_id": "...", "target_id": "...", "link_type": "extends", "description": "..."}
  ],
  ...
}
```

**Why TOON struggles:**
- `tags[]` and `links[]` are nested arrays of objects
- `content` is often multi-line with special characters
- Single note retrieval doesn't benefit from tabular format (N=1)

**Recommendation:** Keep as JSON. Full note retrieval is about *depth*, not *breadth*—the opposite of TOON's optimization target.

---

#### Link Structures

**Structure:**
```json
{
  "source_id": "abc",
  "target_id": "def",
  "link_type": "extends",
  "description": "Builds on the concept of...",
  "created_at": "..."
}
```

**Why TOON provides limited benefit:**
- Links are relational data—they define edges in a knowledge graph
- Query patterns vary: "links from note X", "links to note X", "links of type Y"
- Description field may contain arbitrary text

**Recommendation:** Keep as JSON. Consider graph-specific formats if link queries become a bottleneck.

---

#### Related Notes (`zk_find_related`)

**Structure:** Array of notes with relationship metadata

**Why TOON struggles:**
- Each result may include different relationship types
- Notes include nested tags/links
- Relationship context is as important as the note itself

**Recommendation:** Keep as JSON. The relational context matters more than tabular efficiency here.

---

## 3. Tool-by-Tool Recommendations

| Tool | TOON Adoption | Rationale |
|------|---------------|-----------|
| `zk_fts_search` | **Yes** | Perfect tabular fit. Primary TOON target. |
| `zk_search_notes` | **Partial** | TOON for summary mode; JSON for full mode. |
| `zk_list_notes` | **Partial** | TOON for metadata-only; JSON when including tags/links. |
| `zk_bulk_create_notes` | **Yes** | TOON input with comma-joined tags. |
| `zk_bulk_add_tags` | **No change** | Already uses efficient comma-separated format. |
| `zk_bulk_remove_tags` | **No change** | Already uses efficient comma-separated format. |
| `zk_bulk_delete_notes` | **No change** | Comma-separated IDs already efficient. |
| `zk_get_note` | **No** | Single object with nested structures. |
| `zk_create_note` | **No** | Single object input/output. |
| `zk_update_note` | **No** | Single object operations. |
| `zk_find_related` | **No** | Relational context requires full structure. |
| `zk_create_link` | **No** | Single link creation. |
| `zk_status` | **No** | Heterogeneous status object. |

---

## 4. The Two-Mode Pattern

A recurring theme: tools benefit from TOON when operating in **summary/discovery mode**, but not in **detail/retrieval mode**.

### Proposed Pattern

```
Tool Request:
  zk_search_notes(query="epistemology", response_mode="summary")

Response (TOON):
  results[3]{id,title,score}:
    20240115T103045...,Epistemology of Practice,0.92
    20240112T091230...,Tacit Knowledge,0.87
    20240110T142015...,Knowing-in-Action,0.81

---

Tool Request:
  zk_get_note(id="20240115T103045...")

Response (JSON):
  {
    "id": "20240115T103045...",
    "title": "Epistemology of Practice",
    "content": "...",
    "tags": [...],
    "links": [...]
  }
```

This mirrors how humans use search engines: scan many results quickly, then dive deep into selected items. TOON optimizes the scan; JSON preserves detail for the dive.

---

## 5. Content Field Considerations

### The Content Problem

Note `content` fields are markdown text that may contain:
- Multiple paragraphs
- Code blocks with special characters
- Embedded commas, quotes, newlines
- Large text (500-5000 characters typical)

TOON can encode these using quoted strings with escape sequences, but:
1. Escape overhead reduces savings
2. Multi-line content in tabular rows is awkward
3. Content is often the *reason* for the query—truncating it defeats the purpose

### Recommendation

For tools that return content:
- **Short content** (snippets, <200 chars): Include in TOON with quoting
- **Full content**: Use JSON or hybrid format

Example hybrid approach for search results:
```
results[N]{id,title,score,snippet}:
  20240115T103045...,Epistemology of Practice,0.92,"...knowing-in-action reveals..."
  ...

# Full content available via zk_get_note(id)
```

The snippet provides enough context for decision-making; full content is a separate retrieval.

---

## 6. Architectural Considerations

### Backward Compatibility

znote-mcp currently returns formatted strings (human-readable), not raw JSON. Any TOON adoption must consider:

1. **Existing clients**: May expect current string format
2. **Human readability**: Formatted strings are debuggable; TOON is less intuitive

**Recommendation:** Implement format negotiation:
```
zk_fts_search(query="...", output_format="toon")  # TOON response
zk_fts_search(query="...", output_format="text")  # Current format (default)
zk_fts_search(query="...", output_format="json")  # Raw JSON
```

### LLM System Prompt Integration

If TOON responses are enabled, the MCP server's system prompt should include:

```markdown
## Response Formats

Some tools return data in TOON (Token-Oriented Object Notation) format for efficiency.

TOON arrays look like:
  results[3]{id,title,score}:
    abc,First Result,0.95
    def,Second Result,0.87
    ghi,Third Result,0.72

This declares an array of 3 items with fields id, title, score. Each line is one item.

Parse by:
1. Header line: `name[count]{field1,field2,...}:`
2. Data lines: comma-separated values in field order
```

---

## 7. Expected Outcomes

### Token Savings by Tool

| Tool | Typical Response | JSON Tokens | TOON Tokens | Savings |
|------|-----------------|-------------|-------------|---------|
| `zk_fts_search` (20 results) | Search results | ~1,800 | ~1,100 | 39% |
| `zk_list_notes` (20 notes, metadata) | Note summaries | ~1,500 | ~1,000 | 33% |
| `zk_search_notes` (10 results, summary) | Search matches | ~1,200 | ~800 | 33% |
| `zk_bulk_create_notes` (10 notes, input) | Batch input | ~2,000 | ~1,400 | 30% |

### Aggregate Impact

Assuming a typical session involves:
- 3-5 search/list operations (TOON-eligible)
- 5-10 single note operations (JSON)
- 1-2 bulk operations (TOON-eligible)

Estimated session-level savings: **15-20%** on MCP traffic tokens.

### Qualitative Benefits

1. **Larger search result sets**: Can return 50+ results within reasonable token budgets
2. **Faster bulk operations**: More notes per batch creation
3. **Cleaner separation**: Summary views (TOON) vs. detail views (JSON) is a natural UX pattern

---

## 8. Decision Summary

### Adopt TOON For

| Data View | Rationale |
|-----------|-----------|
| FTS search results | Perfect tabular fit, high volume, discovery-oriented |
| Note list summaries (metadata only) | Flat records when excluding nested fields |
| Bulk create inputs | Batch efficiency with comma-joined tags |
| Search result summaries | Discovery before detail retrieval |

### Keep JSON For

| Data View | Rationale |
|-----------|-----------|
| Full note objects | Nested tags[] and links[] break tabular format |
| Link structures | Relational data with variable context |
| Related note results | Relationship metadata matters |
| Single-item operations | N=1 doesn't benefit from tabular optimization |
| Status/system responses | Heterogeneous structures |

### Key Design Principle

**TOON for breadth, JSON for depth.**

- Scanning many items → TOON
- Inspecting one item → JSON
- Discovery phase → TOON
- Detailed work → JSON

This aligns with TOON's design philosophy (optimizing repeated structures) and Zettelkasten's usage patterns (search → explore → connect).

---

## 9. Non-Goals

This design explicitly does **not** cover:

1. **Forcing TOON onto nested structures**: No awkward flattening of tags/links
2. **Replacing JSON entirely**: JSON remains the right choice for complex, nested data
3. **Changing the Zettelkasten data model**: Notes keep their relational structure
4. **Client-side complexity**: LLMs should parse TOON naturally without custom code

---

*This document captures design rationale. Implementation details, encoding functions, and API changes are out of scope.*
