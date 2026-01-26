# Plan: Enhanced Obsidian Organization & Project Management

## Overview

Enhance znote-mcp to provide better organization of notes in Obsidian vaults through:
1. Project registry with enforcement
2. New `note_purpose` field for workflow organization
3. Hierarchical folder structure in Obsidian
4. Configurable organization strategies

## Problem Statement

Currently:
- `project` field defaults to "general" - LLMs take path of least resistance
- Notes dumped into flat project folders with no workflow organization
- No concept of sub-projects for monorepos
- Human users need folder structure; agents need links

## Design

### 1. Schema Changes

**Add `note_purpose` field (alongside existing `note_type`):**

```python
class NotePurpose(str, Enum):
    RESEARCH = "research"      # Investigation, learning, exploration
    PLANNING = "planning"      # Plans, designs, architecture decisions
    BUGFIXING = "bugfixing"    # Debugging sessions, fixes, investigations
    GENERAL = "general"        # Default for uncategorized notes
```

**Note model changes:**
```python
class Note(BaseModel):
    # ... existing fields ...
    note_purpose: NotePurpose = Field(default=NotePurpose.GENERAL)
    plan_id: Optional[str] = Field(default=None)  # Groups related planning notes
```

### 2. Project Registry

**New `Project` model:**
```python
class Project(BaseModel):
    id: str                          # e.g., "znote-mcp" or "monorepo/frontend"
    name: str                        # Display name
    description: Optional[str]
    parent_id: Optional[str]         # For sub-projects: "monorepo" -> "monorepo/frontend"
    created_at: datetime
    metadata: Dict[str, Any]         # Git remote URL, etc.
```

**New MCP tools:**
- `zk_create_project(id, name, description?, parent_id?)` - Create/register project
- `zk_list_projects()` - List all projects (with hierarchy)
- `zk_get_project(id)` - Get project details
- `zk_delete_project(id)` - Delete project (optionally cascade notes)

**Enforcement:**
- `zk_create_note` validates `project` exists in registry
- Returns clear error: "Project 'xyz' not found. Create it first with zk_create_project or use zk_list_projects to see available projects."

### 3. Obsidian Folder Structure

**New configurable structure:**
```
vault/
├── {project}/
│   ├── research/
│   │   └── {title} ({id_suffix}).md
│   ├── planning/
│   │   ├── {plan_id}/                    # Grouped by plan_id
│   │   │   ├── {title} ({id_suffix}).md  # Initial plan
│   │   │   └── {title} ({id_suffix}).md  # Revisions
│   │   └── {title} ({id_suffix}).md      # Ungrouped plans
│   ├── bugfixing/
│   │   └── {title} ({id_suffix}).md
│   └── general/
│       └── {title} ({id_suffix}).md
└── _unassigned/                          # Notes without valid project
    └── ...
```

**Sub-projects use path convention:**
```
vault/
├── monorepo/
│   ├── frontend/
│   │   ├── research/
│   │   └── planning/
│   └── backend/
│       ├── research/
│       └── planning/
```

### 4. Configuration

**New environment variables:**
```bash
# Organization strategy: "purpose" (new default), "flat" (current), "type", "date"
ZETTELKASTEN_OBSIDIAN_ORGANIZATION=purpose

# Date format for date-based organization
ZETTELKASTEN_OBSIDIAN_DATE_FORMAT=%Y/%m

# Whether to enforce project registry (default: true)
ZETTELKASTEN_ENFORCE_PROJECT_REGISTRY=true
```

### 5. Project Discovery & Configuration

**`zk_configure_projects` - One-time repo survey tool:**

Instead of detecting context every call, do a one-time intelligent survey:

1. **LLM surveys repo structure** (directories, package.json, pyproject.toml, etc.)
2. **Proposes project structure** to user with descriptions
3. **User confirms/adjusts**
4. **Registry created** with rich metadata

**Stored in repo:** `.znote/projects.json`
```json
{
  "root_project": "monorepo",
  "projects": [
    {
      "id": "monorepo",
      "name": "Monorepo Root",
      "description": "Shared tooling, CI/CD, cross-cutting concerns",
      "path": "/"
    },
    {
      "id": "monorepo/frontend",
      "name": "Frontend App",
      "description": "React-based web application with TypeScript",
      "path": "/frontend",
      "parent_id": "monorepo"
    },
    {
      "id": "monorepo/backend",
      "name": "Backend API",
      "description": "FastAPI Python service with PostgreSQL",
      "path": "/backend",
      "parent_id": "monorepo"
    }
  ]
}
```

**Workflow:**
1. User runs `zk_configure_projects` (or Claude suggests on first note attempt)
2. Claude analyzes repo, proposes structure
3. User confirms → registry saved to `.znote/projects.json` AND database
4. Future `zk_list_projects` returns this registry with summaries
5. Claude uses summaries to intelligently route notes

**Benefits:**
- One-time setup cost, ongoing benefit
- Rich context (summaries) for intelligent routing
- Version-controllable (stored in repo)
- Portable across machines

### 6. Enforcement Strategy (LLM Reliability)

**Multi-layer approach:**

1. **Tool-level enforcement** (MCP server):
   - `zk_create_note` rejects if project not in registry
   - Clear error messages: "Project 'xyz' not found. Run zk_configure_projects to set up this repo, or use zk_list_projects to see available projects."

2. **Rich project context** (from registry):
   - `zk_list_projects` returns project summaries
   - LLM can intelligently match task to project based on description

3. **Enhanced tool descriptions** (MCP prompts):
   - Tool docstrings explicitly instruct project requirements
   - Reference `zk_configure_projects` for new repos

## Implementation Phases

### Phase 1: Schema & Registry (Foundation) ✅ COMPLETED
1. ✅ Add `NotePurpose` enum to `models/schema.py`
2. ✅ Add `note_purpose` and `plan_id` fields to `Note` model
3. ✅ Create `Project` model
4. ✅ Add `projects` table to database schema
5. ✅ Create `ProjectRepository` in storage layer
6. ✅ Add project CRUD methods to `ZettelService`

### Phase 2: MCP Tools ✅ COMPLETED
1. ✅ Add `zk_create_project`, `zk_list_projects`, `zk_get_project`, `zk_delete_project`
2. ✅ Add `zk_migration_status`, `zk_bulk_update_project`, `zk_bulk_update_purpose`
3. ✅ Modify `zk_create_note` to accept `note_purpose` and `plan_id`
4. ✅ Add project validation to `zk_create_note` (soft enforcement with warning)
5. ✅ Update tool docstrings with enforcement guidance
6. ⏳ `zk_configure_projects` - future enhancement for intelligent repo survey

### Phase 3: Obsidian Organization ✅ COMPLETED
1. ✅ Refactor `_mirror_to_obsidian` to use new `project/purpose/` folder structure
2. ✅ Refactor `_delete_from_obsidian` to handle new paths (with backward compatibility)
3. ✅ Update tests for new folder structure

### Phase 4: Migration & Polish ⏳ PARTIALLY COMPLETED
1. ✅ Add migration status tool (`zk_migration_status`)
2. ✅ Add bulk update tools (`zk_bulk_update_project`, `zk_bulk_update_purpose`)
3. ✅ Update tests for all new functionality (176 tests passing)
4. ⏳ Documentation updates (CLAUDE.md)
5. ⏳ `zk_configure_projects` for intelligent repo survey
6. ⏳ `zk_reorganize_obsidian` for vault restructuring

## Files to Modify

| File | Changes |
|------|---------|
| `src/znote_mcp/models/schema.py` | Add `NotePurpose`, `Project`, update `Note` |
| `src/znote_mcp/models/db_models.py` | Add `DBProject`, update `DBNote` |
| `src/znote_mcp/storage/project_repository.py` | New file - project CRUD |
| `src/znote_mcp/services/zettel_service.py` | Add project management methods |
| `src/znote_mcp/server/mcp_server.py` | Add 4 project tools, update `zk_create_note` |
| `src/znote_mcp/storage/note_repository.py` | Update `_mirror_to_obsidian`, `_delete_from_obsidian` |
| `src/znote_mcp/config.py` | Add organization config options |
| `.env.example` | Document new options |
| `tests/` | New tests for all features |

## Design Decisions (Resolved)

1. **`plan_id` handling:** Auto-generate as `plan-{timestamp}` when purpose=planning. User can override.

2. **Project deletion:** Prevent deletion if notes exist. Must delete/move notes first.

3. **Context detection:** Use `zk_configure_projects` for one-time repo survey instead of per-call detection. Stores rich project registry with summaries in `.znote/projects.json`.

## Success Criteria

1. LLM cannot create notes without valid project (enforcement works)
2. Notes organized in Obsidian by project/purpose/plan hierarchy
3. Sub-projects work via hierarchical naming
4. Human can browse vault and understand organization
5. Links still work for agent navigation
