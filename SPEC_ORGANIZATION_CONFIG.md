# Specification: Organization Configuration Options

## Overview

Configuration options to customize how notes are organized in the Obsidian vault. These settings allow users to choose different folder structures based on their workflow preferences.

## Motivation

### The Problem

Different users have different organizational preferences:
- **Project-centric**: Developers working on multiple projects want `project/purpose/` structure
- **Temporal**: Researchers might prefer `YYYY/MM/` date-based organization
- **Type-based**: Some users want notes grouped by Zettelkasten type (fleeting, permanent, etc.)
- **Flat**: Power users may prefer flat structure with good search

### The Solution

Configurable organization strategies via environment variables and config file, with sensible defaults.

## Configuration Options

### Environment Variables

```bash
# ============================================================
# OBSIDIAN ORGANIZATION SETTINGS
# ============================================================

# Organization strategy for Obsidian vault
# Options: "purpose" (default), "flat", "type", "date", "custom"
ZETTELKASTEN_OBSIDIAN_ORGANIZATION=purpose

# Date format for date-based organization (strftime format)
# Only used when ZETTELKASTEN_OBSIDIAN_ORGANIZATION=date
# Default: %Y/%m (creates YYYY/MM/ folders)
ZETTELKASTEN_OBSIDIAN_DATE_FORMAT=%Y/%m

# Custom organization template (advanced)
# Only used when ZETTELKASTEN_OBSIDIAN_ORGANIZATION=custom
# Available variables: {project}, {purpose}, {type}, {year}, {month}, {day}
# Default: {project}/{purpose}
ZETTELKASTEN_OBSIDIAN_CUSTOM_TEMPLATE={project}/{type}/{purpose}

# Whether to enforce project registry when creating notes
# Options: "strict" (reject unknown projects), "warn" (default), "none"
ZETTELKASTEN_PROJECT_ENFORCEMENT=warn

# Whether to include note type in folder structure
# Only applies to "purpose" and "custom" strategies
# Default: false
ZETTELKASTEN_OBSIDIAN_INCLUDE_TYPE=false

# ============================================================
# FILENAME SETTINGS
# ============================================================

# Filename format for Obsidian files
# Options: "title_id" (default), "id_title", "id_only", "title_only"
# Default: title_id creates "Title (id_suffix).md"
ZETTELKASTEN_OBSIDIAN_FILENAME_FORMAT=title_id

# Length of ID suffix in filenames
# Default: 8
ZETTELKASTEN_OBSIDIAN_ID_SUFFIX_LENGTH=8

# ============================================================
# CLEANUP SETTINGS
# ============================================================

# Automatically remove empty directories after note moves/deletes
# Default: true
ZETTELKASTEN_OBSIDIAN_CLEANUP_EMPTY_DIRS=true

# Create backup before reorganization operations
# Default: true
ZETTELKASTEN_OBSIDIAN_BACKUP_ON_REORG=true
```

### Organization Strategies

#### 1. Purpose-Based (Default)

```
Strategy: purpose
Template: {project}/{purpose}/

Structure:
vault/
├── my-project/
│   ├── research/
│   │   └── Study findings (abc12345).md
│   ├── planning/
│   │   └── Architecture plan (def67890).md
│   ├── bugfixing/
│   │   └── Memory leak investigation (ghi11111).md
│   └── general/
│       └── Meeting notes (jkl22222).md
└── another-project/
    └── research/
        └── API analysis (mno33333).md
```

**Best for**: Developers working on multiple projects who want to see workflow context.

#### 2. Flat Structure

```
Strategy: flat
Template: {project}/

Structure:
vault/
├── my-project/
│   ├── Study findings (abc12345).md
│   ├── Architecture plan (def67890).md
│   └── Meeting notes (jkl22222).md
└── another-project/
    └── API analysis (mno33333).md
```

**Best for**: Users who prefer simplicity and rely on search/links for navigation.

#### 3. Type-Based

```
Strategy: type
Template: {type}/

Structure:
vault/
├── fleeting/
│   └── Quick thought (abc12345).md
├── literature/
│   └── Book summary (def67890).md
├── permanent/
│   ├── Core concept (ghi11111).md
│   └── Key insight (jkl22222).md
├── structure/
│   └── Topic index (mno33333).md
└── hub/
    └── Project overview (pqr44444).md
```

**Best for**: Traditional Zettelkasten practitioners who organize by note maturity.

#### 4. Date-Based

```
Strategy: date
Template: {year}/{month}/

Structure:
vault/
├── 2024/
│   ├── 01/
│   │   ├── Study findings (abc12345).md
│   │   └── Architecture plan (def67890).md
│   └── 02/
│       └── Meeting notes (jkl22222).md
└── 2023/
    └── 12/
        └── Year review (mno33333).md
```

**Best for**: Researchers who want chronological organization, journaling use cases.

#### 5. Custom Template

```
Strategy: custom
Template: {project}/{type}/{purpose}  # or any combination

Structure:
vault/
├── my-project/
│   ├── permanent/
│   │   ├── research/
│   │   │   └── Core finding (abc12345).md
│   │   └── planning/
│   │       └── Architecture decision (def67890).md
│   └── fleeting/
│       └── general/
│           └── Quick note (ghi11111).md
```

**Best for**: Power users with specific organizational needs.

## Implementation Details

### Config Class Updates

```python
# src/znote_mcp/config.py

from enum import Enum
from typing import Optional

class OrganizationStrategy(str, Enum):
    PURPOSE = "purpose"      # project/purpose/
    FLAT = "flat"           # project/
    TYPE = "type"           # type/
    DATE = "date"           # year/month/
    CUSTOM = "custom"       # user-defined template

class ProjectEnforcement(str, Enum):
    STRICT = "strict"       # Reject unknown projects
    WARN = "warn"           # Warn but allow (default)
    NONE = "none"           # No enforcement

class FilenameFormat(str, Enum):
    TITLE_ID = "title_id"   # "Title (id_suffix).md"
    ID_TITLE = "id_title"   # "(id_suffix) Title.md"
    ID_ONLY = "id_only"     # "id_suffix.md"
    TITLE_ONLY = "title_only"  # "Title.md" (collision risk!)

class ZettelkastenConfig(BaseSettings):
    # ... existing fields ...

    # Organization settings
    obsidian_organization: OrganizationStrategy = Field(
        default=OrganizationStrategy.PURPOSE,
        env="ZETTELKASTEN_OBSIDIAN_ORGANIZATION"
    )

    obsidian_date_format: str = Field(
        default="%Y/%m",
        env="ZETTELKASTEN_OBSIDIAN_DATE_FORMAT"
    )

    obsidian_custom_template: str = Field(
        default="{project}/{purpose}",
        env="ZETTELKASTEN_OBSIDIAN_CUSTOM_TEMPLATE"
    )

    project_enforcement: ProjectEnforcement = Field(
        default=ProjectEnforcement.WARN,
        env="ZETTELKASTEN_PROJECT_ENFORCEMENT"
    )

    obsidian_include_type: bool = Field(
        default=False,
        env="ZETTELKASTEN_OBSIDIAN_INCLUDE_TYPE"
    )

    # Filename settings
    obsidian_filename_format: FilenameFormat = Field(
        default=FilenameFormat.TITLE_ID,
        env="ZETTELKASTEN_OBSIDIAN_FILENAME_FORMAT"
    )

    obsidian_id_suffix_length: int = Field(
        default=8,
        env="ZETTELKASTEN_OBSIDIAN_ID_SUFFIX_LENGTH",
        ge=4,
        le=32
    )

    # Cleanup settings
    obsidian_cleanup_empty_dirs: bool = Field(
        default=True,
        env="ZETTELKASTEN_OBSIDIAN_CLEANUP_EMPTY_DIRS"
    )

    obsidian_backup_on_reorg: bool = Field(
        default=True,
        env="ZETTELKASTEN_OBSIDIAN_BACKUP_ON_REORG"
    )

    def get_obsidian_path_template(self) -> str:
        """Get the path template for the current strategy."""
        templates = {
            OrganizationStrategy.PURPOSE: "{project}/{purpose}",
            OrganizationStrategy.FLAT: "{project}",
            OrganizationStrategy.TYPE: "{type}",
            OrganizationStrategy.DATE: "{year}/{month}",
            OrganizationStrategy.CUSTOM: self.obsidian_custom_template,
        }
        return templates[self.obsidian_organization]
```

### Path Calculation

```python
# src/znote_mcp/storage/note_repository.py

def calculate_obsidian_path(self, note: Note) -> Path:
    """Calculate Obsidian path based on current organization strategy."""
    template = config.get_obsidian_path_template()

    # Build substitution variables
    variables = {
        "project": sanitize_for_path(note.project),
        "purpose": note.note_purpose.value if note.note_purpose else "general",
        "type": note.note_type.value,
        "year": note.created_at.strftime("%Y"),
        "month": note.created_at.strftime("%m"),
        "day": note.created_at.strftime("%d"),
    }

    # Apply template
    try:
        relative_dir = template.format(**variables)
    except KeyError as e:
        logger.warning(f"Invalid template variable {e}, using default")
        relative_dir = f"{variables['project']}/{variables['purpose']}"

    # Build filename
    filename = self._build_obsidian_filename(note)

    return Path(relative_dir) / filename

def _build_obsidian_filename(self, note: Note) -> str:
    """Build filename based on configuration."""
    safe_title = sanitize_for_path(note.title)
    suffix_length = config.obsidian_id_suffix_length
    id_suffix = note.id[-suffix_length:] if len(note.id) >= suffix_length else note.id

    format_map = {
        FilenameFormat.TITLE_ID: f"{safe_title} ({id_suffix}).md",
        FilenameFormat.ID_TITLE: f"({id_suffix}) {safe_title}.md",
        FilenameFormat.ID_ONLY: f"{id_suffix}.md",
        FilenameFormat.TITLE_ONLY: f"{safe_title}.md",
    }

    return format_map[config.obsidian_filename_format]
```

### Project Enforcement

```python
# src/znote_mcp/server/mcp_server.py

def validate_project(self, project: str) -> Tuple[bool, Optional[str]]:
    """Validate project based on enforcement setting.

    Returns:
        (is_valid, warning_message)
    """
    enforcement = config.project_enforcement

    if enforcement == ProjectEnforcement.NONE:
        return True, None

    project_exists = self.project_repository.exists(project)

    if enforcement == ProjectEnforcement.STRICT:
        if not project_exists:
            raise ValidationError(
                f"Project '{project}' not found in registry. "
                f"Create it with zk_create_project or use zk_list_projects to see available projects.",
                code=ErrorCode.VALIDATION_FAILED
            )
        return True, None

    elif enforcement == ProjectEnforcement.WARN:
        if not project_exists:
            return True, f"Warning: Project '{project}' not in registry. Consider creating it with zk_create_project."
        return True, None

    return True, None
```

## MCP Tool: `zk_config`

Add a tool to view and modify configuration at runtime:

```python
@self.mcp.tool(name="zk_config")
def zk_config(
    action: str = "show",
    key: Optional[str] = None,
    value: Optional[str] = None
) -> str:
    """View or modify znote-mcp configuration.

    Args:
        action: Operation to perform:
            - "show": Display current configuration (default)
            - "get": Get specific config value
            - "strategies": List available organization strategies
        key: Config key for "get" action
        value: (Reserved for future "set" action)

    Note: Configuration changes require environment variables or config file.
    This tool is read-only for safety.
    """
    if action == "show":
        return format_config_display()
    elif action == "get":
        return get_config_value(key)
    elif action == "strategies":
        return format_strategies_help()
    else:
        return f"Unknown action: {action}. Use: show, get, strategies"
```

### Output Example

```
## znote-mcp Configuration

### Obsidian Organization
- Strategy: purpose
- Template: {project}/{purpose}
- Filename format: title_id (Title (id_suffix).md)
- ID suffix length: 8

### Project Enforcement
- Mode: warn
- Registered projects: 5

### Paths
- Notes directory: /home/user/.zettelkasten/notes
- Database: /home/user/.zettelkasten/db/zettelkasten.db
- Obsidian vault: /home/user/obsidian-vault

### Cleanup
- Auto-cleanup empty dirs: yes
- Backup on reorganize: yes

---
To change settings, set environment variables or edit config file.
See: ZETTELKASTEN_OBSIDIAN_ORGANIZATION, ZETTELKASTEN_PROJECT_ENFORCEMENT
```

## .env.example Updates

```bash
# ============================================================
# ZNOTE-MCP CONFIGURATION
# ============================================================

# Core paths
ZETTELKASTEN_NOTES_DIR=~/.zettelkasten/notes
ZETTELKASTEN_DATABASE_PATH=~/.zettelkasten/db/zettelkasten.db
ZETTELKASTEN_LOG_LEVEL=INFO

# Obsidian integration
ZETTELKASTEN_OBSIDIAN_VAULT=/path/to/obsidian/vault

# ------------------------------------------------------------
# ORGANIZATION SETTINGS
# ------------------------------------------------------------

# Organization strategy for Obsidian vault
# Options: purpose (default), flat, type, date, custom
# - purpose: project/purpose/filename.md (recommended for developers)
# - flat: project/filename.md (simple, relies on search)
# - type: type/filename.md (traditional Zettelkasten)
# - date: YYYY/MM/filename.md (chronological)
# - custom: use ZETTELKASTEN_OBSIDIAN_CUSTOM_TEMPLATE
ZETTELKASTEN_OBSIDIAN_ORGANIZATION=purpose

# Custom template (only when organization=custom)
# Variables: {project}, {purpose}, {type}, {year}, {month}, {day}
# ZETTELKASTEN_OBSIDIAN_CUSTOM_TEMPLATE={project}/{type}/{purpose}

# Date format for date-based organization (strftime)
# ZETTELKASTEN_OBSIDIAN_DATE_FORMAT=%Y/%m

# Project enforcement mode
# - strict: reject notes with unregistered projects
# - warn: allow but show warning (default)
# - none: no enforcement
ZETTELKASTEN_PROJECT_ENFORCEMENT=warn

# ------------------------------------------------------------
# FILENAME SETTINGS
# ------------------------------------------------------------

# Filename format in Obsidian
# - title_id: "Title (abc12345).md" (default, recommended)
# - id_title: "(abc12345) Title.md"
# - id_only: "abc12345.md"
# - title_only: "Title.md" (warning: collision risk!)
ZETTELKASTEN_OBSIDIAN_FILENAME_FORMAT=title_id

# Length of ID suffix in filenames (4-32)
ZETTELKASTEN_OBSIDIAN_ID_SUFFIX_LENGTH=8

# ------------------------------------------------------------
# CLEANUP SETTINGS
# ------------------------------------------------------------

# Auto-remove empty directories after moves/deletes
ZETTELKASTEN_OBSIDIAN_CLEANUP_EMPTY_DIRS=true

# Create backup before reorganization operations
ZETTELKASTEN_OBSIDIAN_BACKUP_ON_REORG=true
```

## Testing Requirements

### Unit Tests

```python
class TestOrganizationConfig:
    def test_default_strategy_is_purpose(self):
        """Default organization strategy is purpose-based."""
        assert config.obsidian_organization == OrganizationStrategy.PURPOSE

    def test_purpose_template(self):
        """Purpose strategy uses project/purpose template."""
        config.obsidian_organization = OrganizationStrategy.PURPOSE
        assert config.get_obsidian_path_template() == "{project}/{purpose}"

    def test_flat_template(self):
        """Flat strategy uses project-only template."""
        config.obsidian_organization = OrganizationStrategy.FLAT
        assert config.get_obsidian_path_template() == "{project}"

    def test_date_template(self):
        """Date strategy uses year/month template."""
        config.obsidian_organization = OrganizationStrategy.DATE
        assert config.get_obsidian_path_template() == "{year}/{month}"

    def test_custom_template(self):
        """Custom strategy uses user-defined template."""
        config.obsidian_organization = OrganizationStrategy.CUSTOM
        config.obsidian_custom_template = "{type}/{project}"
        assert config.get_obsidian_path_template() == "{type}/{project}"

    def test_env_var_parsing(self, monkeypatch):
        """Environment variables are parsed correctly."""
        monkeypatch.setenv("ZETTELKASTEN_OBSIDIAN_ORGANIZATION", "date")
        monkeypatch.setenv("ZETTELKASTEN_PROJECT_ENFORCEMENT", "strict")

        new_config = ZettelkastenConfig()

        assert new_config.obsidian_organization == OrganizationStrategy.DATE
        assert new_config.project_enforcement == ProjectEnforcement.STRICT


class TestPathCalculation:
    def test_purpose_path(self, note):
        """Purpose strategy creates project/purpose path."""
        config.obsidian_organization = OrganizationStrategy.PURPOSE
        note.project = "my-project"
        note.note_purpose = NotePurpose.RESEARCH

        path = calculate_obsidian_path(note)

        assert path.parts[:2] == ("my-project", "research")

    def test_date_path(self, note):
        """Date strategy creates year/month path."""
        config.obsidian_organization = OrganizationStrategy.DATE
        note.created_at = datetime(2024, 3, 15)

        path = calculate_obsidian_path(note)

        assert path.parts[:2] == ("2024", "03")

    def test_filename_formats(self, note):
        """Different filename formats are applied correctly."""
        note.title = "My Note"
        note.id = "20240115_143022_abcd1234"

        config.obsidian_filename_format = FilenameFormat.TITLE_ID
        assert _build_obsidian_filename(note) == "My Note (abcd1234).md"

        config.obsidian_filename_format = FilenameFormat.ID_TITLE
        assert _build_obsidian_filename(note) == "(abcd1234) My Note.md"

        config.obsidian_filename_format = FilenameFormat.ID_ONLY
        assert _build_obsidian_filename(note) == "abcd1234.md"


class TestProjectEnforcement:
    def test_strict_rejects_unknown(self, project_repository):
        """Strict mode rejects unknown projects."""
        config.project_enforcement = ProjectEnforcement.STRICT

        with pytest.raises(ValidationError):
            validate_project("unknown-project")

    def test_warn_allows_unknown(self, project_repository):
        """Warn mode allows unknown projects with warning."""
        config.project_enforcement = ProjectEnforcement.WARN

        is_valid, warning = validate_project("unknown-project")

        assert is_valid
        assert "not in registry" in warning

    def test_none_no_validation(self, project_repository):
        """None mode performs no validation."""
        config.project_enforcement = ProjectEnforcement.NONE

        is_valid, warning = validate_project("anything-goes")

        assert is_valid
        assert warning is None
```

### Integration Tests

```python
class TestOrganizationStrategiesE2E:
    @pytest.mark.parametrize("strategy,expected_parts", [
        (OrganizationStrategy.PURPOSE, ("project", "research")),
        (OrganizationStrategy.FLAT, ("project",)),
        (OrganizationStrategy.TYPE, ("permanent",)),
        (OrganizationStrategy.DATE, ("2024", "01")),
    ])
    def test_strategy_creates_correct_structure(
        self, isolated_env, obsidian_vault, strategy, expected_parts
    ):
        """Each strategy creates the expected folder structure."""
        config.obsidian_organization = strategy

        note = create_note(
            title="Test Note",
            project="project",
            note_type=NoteType.PERMANENT,
            note_purpose=NotePurpose.RESEARCH
        )

        # Trigger sync
        sync_to_obsidian()

        # Find the file
        files = list(obsidian_vault.rglob("*.md"))
        assert len(files) == 1

        # Check path matches expected structure
        rel_path = files[0].relative_to(obsidian_vault)
        assert rel_path.parts[:len(expected_parts)] == expected_parts
```

## Migration Considerations

When changing organization strategies:

1. **Existing notes remain in place** - Obsidian files aren't automatically moved
2. **New notes use new strategy** - Only affects notes created after change
3. **Use `zk_reorganize_obsidian`** - To migrate existing files to new structure

```python
# Example migration workflow:

# 1. Change strategy
export ZETTELKASTEN_OBSIDIAN_ORGANIZATION=date

# 2. Preview reorganization
zk_reorganize_obsidian(mode="preview")

# 3. Execute migration
zk_reorganize_obsidian(mode="execute", backup=True)
```

## Files to Modify

| File | Changes |
|------|---------|
| `src/znote_mcp/config.py` | Add organization config fields and enums |
| `src/znote_mcp/storage/note_repository.py` | Update `calculate_obsidian_path()` |
| `src/znote_mcp/server/mcp_server.py` | Add `zk_config` tool, update enforcement |
| `.env.example` | Document new environment variables |
| `tests/test_config.py` | New file - configuration tests |

## Future Enhancements

1. **Runtime config changes**: Allow changing settings via `zk_config(action="set")`
2. **Per-project strategies**: Different projects use different organization
3. **Tag-based organization**: Organize by primary tag
4. **Hybrid strategies**: Combine multiple strategies (e.g., date + type)
5. **Obsidian plugin**: Sync settings with Obsidian configuration
