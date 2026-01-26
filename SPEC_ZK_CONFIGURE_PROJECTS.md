# Specification: `zk_configure_projects` Tool

## Overview

A one-time intelligent repository survey tool that analyzes a codebase and proposes a project structure for the znote-mcp registry. This tool bridges the gap between "no projects registered" and "productive note-taking" by automating the discovery and registration of projects.

## Motivation

### The Problem

When a user starts using znote-mcp in a new repository:
1. No projects exist in the registry
2. `zk_create_note` warns about unregistered projects
3. User must manually create each project with `zk_create_project`
4. User may not know what projects to create for their repo structure

### The Solution

`zk_configure_projects` performs intelligent analysis of the repository and:
1. Detects project structure from code/config files
2. Proposes a project hierarchy with descriptions
3. Allows user to confirm/adjust
4. Registers all projects automatically
5. Exports to `.znote/projects.json` for version control

## API Design

### Tool Signature

```python
@self.mcp.tool(name="zk_configure_projects")
def zk_configure_projects(
    root_path: Optional[str] = None,
    mode: str = "interactive",
    include_suggestions: bool = True
) -> str:
    """Analyze repository and configure project registry.

    This tool surveys your repository structure and proposes a project
    hierarchy for organizing notes. Run this once when setting up
    znote-mcp for a new repository.

    Args:
        root_path: Root directory to analyze (defaults to current working directory)
        mode: Operation mode:
            - "interactive": Propose projects and wait for confirmation (default)
            - "detect": Only detect and report, don't register
            - "auto": Auto-register detected projects without confirmation
        include_suggestions: Include AI-generated descriptions for projects

    Returns:
        In "interactive" mode: Proposed project structure for user review
        In "detect" mode: Detected project structure (no changes made)
        In "auto" mode: Confirmation of registered projects
    """
```

### Output Format (Interactive Mode)

```
## Detected Repository Structure

Root: /home/user/my-monorepo

### Proposed Projects

1. **my-monorepo** (root)
   - Description: Monorepo root for shared tooling and cross-cutting concerns
   - Path: /
   - Detected from: package.json (workspaces), README.md

2. **my-monorepo/frontend**
   - Description: React TypeScript web application with Vite bundler
   - Path: /packages/frontend
   - Parent: my-monorepo
   - Detected from: package.json (react, vite), tsconfig.json

3. **my-monorepo/backend**
   - Description: FastAPI Python service with PostgreSQL database
   - Path: /packages/backend
   - Parent: my-monorepo
   - Detected from: pyproject.toml (fastapi), alembic.ini

4. **my-monorepo/shared**
   - Description: Shared TypeScript utilities and types
   - Path: /packages/shared
   - Parent: my-monorepo
   - Detected from: package.json, tsconfig.json

---

To register these projects, call:
  zk_configure_projects(mode="auto")

To modify, use zk_create_project/zk_delete_project after registration.
```

## Implementation Details

### Detection Strategies

The tool should detect projects using multiple heuristics:

#### 1. Package Manager Detection

```python
PACKAGE_INDICATORS = {
    # Node.js
    "package.json": {
        "name_field": "name",
        "workspace_fields": ["workspaces", "packages"],
        "type_hints": {
            "react": "React application",
            "vue": "Vue.js application",
            "next": "Next.js application",
            "express": "Express.js server",
            "fastify": "Fastify server",
        }
    },
    # Python
    "pyproject.toml": {
        "name_field": "project.name",
        "type_hints": {
            "fastapi": "FastAPI service",
            "django": "Django application",
            "flask": "Flask application",
        }
    },
    "setup.py": {"name_field": "name"},
    # Rust
    "Cargo.toml": {
        "name_field": "package.name",
        "workspace_field": "workspace.members"
    },
    # Go
    "go.mod": {"name_field": "module"},
}
```

#### 2. Directory Structure Detection

```python
STRUCTURE_PATTERNS = {
    # Monorepo patterns
    "packages/*": "monorepo_packages",
    "apps/*": "monorepo_apps",
    "services/*": "microservices",
    "libs/*": "shared_libraries",

    # Standard directories (lower confidence)
    "src/": "source_root",
    "frontend/": "frontend_app",
    "backend/": "backend_service",
    "api/": "api_service",
    "web/": "web_application",
    "mobile/": "mobile_app",
}
```

#### 3. Configuration File Detection

```python
CONFIG_INDICATORS = {
    # Build tools
    "webpack.config.js": "bundled_frontend",
    "vite.config.ts": "vite_frontend",
    "rollup.config.js": "library_package",

    # Frameworks
    "next.config.js": "nextjs_app",
    "nuxt.config.ts": "nuxt_app",
    "angular.json": "angular_app",

    # Infrastructure
    "Dockerfile": "containerized_service",
    "docker-compose.yml": "multi_container_app",
    "serverless.yml": "serverless_functions",

    # Database
    "alembic.ini": "python_with_migrations",
    "prisma/schema.prisma": "prisma_database",
}
```

### Detection Algorithm

```python
def detect_projects(root_path: Path) -> List[DetectedProject]:
    """Detect projects in repository."""
    projects = []

    # 1. Check root for project indicators
    root_project = detect_root_project(root_path)
    if root_project:
        projects.append(root_project)

    # 2. Check for workspace/monorepo patterns
    workspaces = detect_workspaces(root_path)
    for ws_path in workspaces:
        ws_project = detect_project_at_path(ws_path)
        if ws_project:
            ws_project.parent_id = root_project.id if root_project else None
            projects.append(ws_project)

    # 3. Check known directory patterns
    for pattern, project_type in STRUCTURE_PATTERNS.items():
        for match_path in root_path.glob(pattern):
            if match_path.is_dir() and not is_already_detected(match_path, projects):
                project = detect_project_at_path(match_path)
                if project:
                    projects.append(project)

    # 4. Generate descriptions using context
    for project in projects:
        if not project.description:
            project.description = generate_description(project)

    return projects

@dataclass
class DetectedProject:
    id: str                      # e.g., "my-monorepo/frontend"
    name: str                    # e.g., "Frontend App"
    path: str                    # e.g., "/packages/frontend"
    parent_id: Optional[str]     # e.g., "my-monorepo"
    description: Optional[str]   # Generated or from package.json
    confidence: float            # 0.0-1.0 detection confidence
    detected_from: List[str]     # Files that triggered detection
    tech_stack: List[str]        # Detected technologies
```

### Description Generation

Descriptions should be concise and useful for LLM routing:

```python
def generate_description(project: DetectedProject) -> str:
    """Generate a description from detected context."""
    parts = []

    # Technology stack
    if project.tech_stack:
        stack_str = ", ".join(project.tech_stack[:3])
        parts.append(f"{stack_str} project")

    # Purpose hints from name/path
    purpose_hints = {
        "frontend": "user interface",
        "backend": "server-side logic",
        "api": "API endpoints",
        "shared": "shared utilities",
        "common": "common code",
        "mobile": "mobile application",
        "web": "web application",
        "services": "microservice",
        "infra": "infrastructure",
    }

    for hint, desc in purpose_hints.items():
        if hint in project.path.lower() or hint in project.name.lower():
            parts.append(f"handling {desc}")
            break

    return " ".join(parts) if parts else f"Project at {project.path}"
```

### Storage Integration

After confirmation, projects are:

1. **Registered in database** via `ProjectRepository.create()`
2. **Exported to `.znote/projects.json`** for version control

```python
def register_detected_projects(projects: List[DetectedProject]) -> int:
    """Register all detected projects."""
    registered = 0

    # Sort by depth (parents first)
    projects.sort(key=lambda p: p.id.count("/"))

    for detected in projects:
        project = Project(
            id=detected.id,
            name=detected.name,
            description=detected.description,
            parent_id=detected.parent_id,
            path=detected.path,
            metadata={
                "detected_from": detected.detected_from,
                "tech_stack": detected.tech_stack,
                "confidence": detected.confidence,
            }
        )

        if not self.project_repository.exists(project.id):
            self.project_repository.create(project)
            registered += 1

    # Export to .znote/projects.json
    self.project_repository.export_to_json()

    return registered
```

## Edge Cases

### 1. Empty Repository
```
No project indicators found in /home/user/empty-repo

Creating default project "empty-repo" based on directory name.

To add more projects, use zk_create_project.
```

### 2. Already Configured
```
Found existing projects in registry:
  - my-project (5 notes)
  - my-project/frontend (12 notes)

Detected 1 new project not in registry:
  - my-project/backend

Register new project? Use mode="auto" to confirm.
```

### 3. Conflicting Detection
```
Warning: Conflicting project indicators at /packages/api

Detected as:
  1. Express.js server (from package.json)
  2. Python FastAPI (from requirements.txt)

Using highest-confidence detection: Express.js server

To override, manually create project with zk_create_project.
```

### 4. Deep Nesting
```
Warning: Deep project nesting detected (4+ levels)

  my-monorepo/packages/apps/web/components

Consider flattening hierarchy. Recommended structure:
  my-monorepo/web (instead of full path)

Proceeding with simplified hierarchy.
```

## Testing Requirements

### Unit Tests

```python
class TestConfigureProjects:
    def test_detect_simple_node_project(self, tmp_path):
        """Detect single Node.js project."""
        (tmp_path / "package.json").write_text('{"name": "my-app", "dependencies": {"react": "^18"}}')

        projects = detect_projects(tmp_path)

        assert len(projects) == 1
        assert projects[0].name == "my-app"
        assert "react" in projects[0].tech_stack

    def test_detect_monorepo_workspaces(self, tmp_path):
        """Detect monorepo with workspaces."""
        (tmp_path / "package.json").write_text('{"name": "monorepo", "workspaces": ["packages/*"]}')
        (tmp_path / "packages/frontend/package.json").write_text('{"name": "frontend"}')
        (tmp_path / "packages/backend/package.json").write_text('{"name": "backend"}')

        projects = detect_projects(tmp_path)

        assert len(projects) == 3
        assert any(p.id == "monorepo" for p in projects)
        assert any(p.parent_id == "monorepo" for p in projects)

    def test_detect_python_project(self, tmp_path):
        """Detect Python project with pyproject.toml."""
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "my-api"\ndependencies = ["fastapi"]')

        projects = detect_projects(tmp_path)

        assert len(projects) == 1
        assert "fastapi" in projects[0].tech_stack.lower()

    def test_skip_node_modules(self, tmp_path):
        """Ignore node_modules directories."""
        (tmp_path / "package.json").write_text('{"name": "app"}')
        nm = tmp_path / "node_modules/some-dep"
        nm.mkdir(parents=True)
        (nm / "package.json").write_text('{"name": "some-dep"}')

        projects = detect_projects(tmp_path)

        assert len(projects) == 1
        assert projects[0].name == "app"

    def test_interactive_mode_returns_proposal(self):
        """Interactive mode returns proposal without registering."""
        result = zk_configure_projects(mode="interactive")

        assert "Proposed Projects" in result
        assert self.project_repository.get_all() == []  # Nothing registered

    def test_auto_mode_registers_projects(self):
        """Auto mode registers all detected projects."""
        result = zk_configure_projects(mode="auto")

        assert "registered" in result.lower()
        assert len(self.project_repository.get_all()) > 0
```

### Integration Tests

```python
class TestConfigureProjectsE2E:
    def test_full_workflow(self, isolated_env, sample_monorepo):
        """Test complete configure workflow."""
        # 1. Initially no projects
        assert self.project_repository.get_all() == []

        # 2. Detect projects
        result = zk_configure_projects(root_path=sample_monorepo, mode="detect")
        assert "frontend" in result
        assert "backend" in result

        # 3. Register projects
        result = zk_configure_projects(root_path=sample_monorepo, mode="auto")
        assert "registered" in result

        # 4. Verify registry
        projects = self.project_repository.get_all()
        assert len(projects) >= 2

        # 5. Verify .znote/projects.json created
        json_path = sample_monorepo / ".znote/projects.json"
        assert json_path.exists()

        # 6. Create note with detected project
        note = self.zettel_service.create_note(
            title="Frontend feature plan",
            content="Planning new feature",
            project="monorepo/frontend"  # Should work without warning
        )
        assert note.project == "monorepo/frontend"
```

## Files to Modify

| File | Changes |
|------|---------|
| `src/znote_mcp/server/mcp_server.py` | Add `zk_configure_projects` tool |
| `src/znote_mcp/services/project_detection.py` | New file - detection logic |
| `tests/test_project_detection.py` | New file - detection tests |
| `tests/fixtures/` | Add sample repo structures |

## Future Enhancements

1. **Git integration**: Detect projects from git submodules
2. **IDE integration**: Read project structure from `.vscode/`, `.idea/`
3. **Custom patterns**: Allow user-defined detection patterns in config
4. **Re-detection**: Option to re-scan and update existing registry
5. **Conflict resolution UI**: Interactive prompt for ambiguous detections
