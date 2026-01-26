# Specification: `zk_reorganize_obsidian` Tool

## Overview

A tool to reorganize existing notes in the Obsidian vault according to the current organization strategy. This handles migration from old folder structures to new ones, and allows users to change organization strategies without losing notes.

## Motivation

### The Problem

1. **Schema evolution**: When `note_purpose` was added, existing notes in Obsidian are still in flat `project/` folders instead of `project/purpose/`
2. **Strategy changes**: User might want to switch from `purpose` organization to `date` or `type` organization
3. **Project renames**: When projects are renamed/restructured, Obsidian mirrors become stale
4. **Orphaned files**: Notes deleted from znote-mcp may leave orphans in Obsidian vault

### The Solution

`zk_reorganize_obsidian` performs a controlled reorganization:
1. Scans current Obsidian vault structure
2. Compares against source of truth (markdown files + database)
3. Moves files to correct locations based on current strategy
4. Cleans up orphaned files
5. Reports changes made

## API Design

### Tool Signature

```python
@self.mcp.tool(name="zk_reorganize_obsidian")
def zk_reorganize_obsidian(
    mode: str = "preview",
    clean_orphans: bool = False,
    backup: bool = True
) -> str:
    """Reorganize Obsidian vault to match current organization strategy.

    Use this after:
    - Changing organization strategy (e.g., flat -> purpose-based)
    - Bulk updating note projects or purposes
    - Renaming or restructuring projects
    - Recovering from sync issues

    Args:
        mode: Operation mode:
            - "preview": Show what would be changed (default, safe)
            - "execute": Actually perform the reorganization
        clean_orphans: Remove files in Obsidian that don't exist in znote-mcp
        backup: Create backup before reorganization (default: true)

    Returns:
        Report of changes (preview) or changes made (execute)
    """
```

### Output Format (Preview Mode)

```
## Obsidian Vault Reorganization Preview

Vault: /home/user/obsidian-vault
Strategy: project/purpose (current)
Notes in znote-mcp: 47

### Files to Move (23)

| Current Location | New Location | Reason |
|-----------------|--------------|--------|
| general/Note A (abc123).md | my-project/research/Note A (abc123).md | Project changed |
| my-project/Note B (def456).md | my-project/planning/Note B (def456).md | Purpose added |
| old-project/Note C (ghi789).md | new-project/general/Note C (ghi789).md | Project renamed |

### Orphaned Files (3)

Files in Obsidian vault not found in znote-mcp:
- old-project/Deleted Note (xyz999).md
- temp/scratch (aaa111).md
- .trash/Old Note (bbb222).md

### Files Already Correct (21)

21 files are already in the correct location.

---

To execute this reorganization:
  zk_reorganize_obsidian(mode="execute")

To also remove orphans:
  zk_reorganize_obsidian(mode="execute", clean_orphans=True)
```

### Output Format (Execute Mode)

```
## Obsidian Vault Reorganization Complete

### Summary
- Files moved: 23
- Orphans removed: 3
- Errors: 0
- Backup created: /home/user/.zettelkasten/backups/obsidian_20240126_143022.tar.gz

### Moved Files

✓ general/Note A (abc123).md → my-project/research/Note A (abc123).md
✓ my-project/Note B (def456).md → my-project/planning/Note B (def456).md
✓ old-project/Note C (ghi789).md → new-project/general/Note C (ghi789).md
... (20 more)

### Removed Orphans

✓ old-project/Deleted Note (xyz999).md
✓ temp/scratch (aaa111).md
✓ .trash/Old Note (bbb222).md

### Empty Directories Cleaned

✓ old-project/ (removed - empty after move)
✓ temp/ (removed - empty after orphan removal)
```

## Implementation Details

### Core Algorithm

```python
def reorganize_obsidian(
    mode: str,
    clean_orphans: bool,
    backup: bool
) -> ReorganizationReport:
    """Reorganize Obsidian vault."""

    if not self.obsidian_vault_path:
        raise ValueError("Obsidian vault not configured")

    report = ReorganizationReport()

    # 1. Create backup if requested
    if backup and mode == "execute":
        backup_path = create_obsidian_backup()
        report.backup_path = backup_path

    # 2. Build map of current Obsidian files
    current_files = scan_obsidian_vault()

    # 3. Build map of expected locations from source of truth
    expected_locations = build_expected_locations()

    # 4. Compare and categorize
    for obsidian_file in current_files:
        note_id = extract_note_id(obsidian_file)

        if note_id in expected_locations:
            expected_path = expected_locations[note_id]
            if obsidian_file.relative_path != expected_path:
                report.files_to_move.append(MoveOperation(
                    current=obsidian_file,
                    target=expected_path,
                    reason=determine_reason(obsidian_file, expected_path)
                ))
            else:
                report.files_correct.append(obsidian_file)
        else:
            report.orphaned_files.append(obsidian_file)

    # 5. Execute if requested
    if mode == "execute":
        execute_moves(report.files_to_move)
        if clean_orphans:
            remove_orphans(report.orphaned_files)
        cleanup_empty_directories()

    return report
```

### File Scanning

```python
def scan_obsidian_vault() -> List[ObsidianFile]:
    """Scan Obsidian vault for all markdown files."""
    files = []

    for md_path in self.obsidian_vault_path.rglob("*.md"):
        # Skip hidden directories (like .obsidian, .trash)
        if any(part.startswith('.') for part in md_path.parts):
            continue

        # Extract note ID from filename pattern: "Title (id_suffix).md"
        note_id = extract_note_id_from_filename(md_path.name)

        files.append(ObsidianFile(
            absolute_path=md_path,
            relative_path=md_path.relative_to(self.obsidian_vault_path),
            note_id=note_id,
            filename=md_path.name
        ))

    return files

def extract_note_id_from_filename(filename: str) -> Optional[str]:
    """Extract note ID from Obsidian filename.

    Filename format: "Title (id_suffix).md"
    We need to find the full note ID from the suffix.
    """
    import re

    # Pattern: anything followed by (8-char-suffix).md
    match = re.search(r'\(([a-zA-Z0-9_-]{8,})\)\.md$', filename)
    if match:
        id_suffix = match.group(1)
        # Look up full ID from database using suffix
        return lookup_note_id_by_suffix(id_suffix)

    # Fallback: filename might be just the ID
    stem = filename.replace('.md', '')
    if self.note_repository.get(stem):
        return stem

    return None
```

### Expected Location Calculation

```python
def build_expected_locations() -> Dict[str, Path]:
    """Build map of note_id -> expected Obsidian path."""
    expected = {}

    for note in self.note_repository.get_all():
        expected[note.id] = calculate_obsidian_path(note)

    return expected

def calculate_obsidian_path(note: Note) -> Path:
    """Calculate expected Obsidian path based on current strategy.

    Current strategy: project/purpose/filename.md
    """
    # Sanitize project for directory name
    safe_project = sanitize_for_path(note.project)

    # Get purpose (default to "general")
    purpose = note.note_purpose.value if note.note_purpose else "general"

    # Build filename with ID suffix
    safe_title = sanitize_for_path(note.title)
    id_suffix = note.id[-8:] if len(note.id) >= 8 else note.id
    filename = f"{safe_title} ({id_suffix}).md"

    return Path(safe_project) / purpose / filename
```

### Move Execution

```python
def execute_moves(moves: List[MoveOperation]) -> None:
    """Execute file moves in Obsidian vault."""
    for move in moves:
        source = self.obsidian_vault_path / move.current.relative_path
        target = self.obsidian_vault_path / move.target

        # Create target directory
        target.parent.mkdir(parents=True, exist_ok=True)

        # Move file
        try:
            shutil.move(str(source), str(target))
            move.status = "success"
        except Exception as e:
            move.status = "failed"
            move.error = str(e)
            logger.error(f"Failed to move {source} to {target}: {e}")

def cleanup_empty_directories() -> List[Path]:
    """Remove empty directories after moves."""
    removed = []

    # Walk bottom-up to find empty directories
    for dirpath in sorted(
        self.obsidian_vault_path.rglob("*"),
        key=lambda p: len(p.parts),
        reverse=True
    ):
        if dirpath.is_dir() and not any(dirpath.iterdir()):
            # Don't remove vault root or hidden directories
            if dirpath != self.obsidian_vault_path and not dirpath.name.startswith('.'):
                dirpath.rmdir()
                removed.append(dirpath)

    return removed
```

### Backup Creation

```python
def create_obsidian_backup() -> Path:
    """Create backup of Obsidian vault before reorganization."""
    import tarfile
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = config.get_absolute_path(config.database_path).parent / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)

    backup_path = backup_dir / f"obsidian_{timestamp}.tar.gz"

    with tarfile.open(backup_path, "w:gz") as tar:
        tar.add(
            self.obsidian_vault_path,
            arcname=self.obsidian_vault_path.name
        )

    logger.info(f"Created Obsidian backup: {backup_path}")
    return backup_path
```

## Edge Cases

### 1. No Obsidian Vault Configured

```
Error: Obsidian vault not configured.

Set ZETTELKASTEN_OBSIDIAN_VAULT environment variable or configure in settings.
```

### 2. Vault is Empty

```
## Obsidian Vault Reorganization Preview

Vault: /home/user/obsidian-vault
Status: Empty vault

No files found in Obsidian vault. Nothing to reorganize.

To sync notes to Obsidian:
  zk_system(action="sync")
```

### 3. All Files Already Correct

```
## Obsidian Vault Reorganization Preview

Vault: /home/user/obsidian-vault
Strategy: project/purpose

All 47 files are already in the correct location.

No reorganization needed.
```

### 4. Note ID Cannot Be Determined

```
## Obsidian Vault Reorganization Preview

### Warning: Unidentified Files (2)

These files don't match any notes in znote-mcp:
- random-notes/My Ideas.md (no ID suffix)
- scratch/temp.md (no ID suffix)

These may be:
- Notes created directly in Obsidian (not via znote-mcp)
- Notes with corrupted filenames
- Non-note markdown files

Use clean_orphans=True to remove, or manually move/delete.
```

### 5. Move Would Overwrite

```
## Obsidian Vault Reorganization Preview

### Warning: Potential Overwrites (1)

Moving these files would overwrite existing files:
- my-project/planning/Plan A (abc123).md
  Target already exists: my-project/planning/Plan A (abc123).md

This may indicate:
- Duplicate notes with same ID
- Previous failed reorganization

Review manually before proceeding.
```

### 6. Permission Errors

```
## Obsidian Vault Reorganization Complete

### Errors (2)

✗ old-project/Note A (abc123).md → new-project/Note A (abc123).md
  Error: Permission denied

✗ restricted/Note B (def456).md → public/Note B (def456).md
  Error: Read-only file system

23 files moved successfully. 2 files failed.
Run again after fixing permissions.
```

## Integration with Other Tools

### With `zk_bulk_update_project`

After bulk moving notes to new projects:
```python
# 1. Move notes in znote-mcp
zk_bulk_update_project(note_ids="abc123,def456,ghi789", project="new-project")

# 2. Reorganize Obsidian to match
zk_reorganize_obsidian(mode="execute")
```

### With `zk_configure_projects`

After setting up projects:
```python
# 1. Detect and register projects
zk_configure_projects(mode="auto")

# 2. Update existing notes with detected projects
zk_bulk_update_project(note_ids="...", project="detected-project")

# 3. Reorganize Obsidian
zk_reorganize_obsidian(mode="execute")
```

### With `zk_system(action="sync")`

The sync action already mirrors notes correctly. `zk_reorganize_obsidian` is for when:
- Notes exist in wrong locations (legacy structure)
- Organization strategy changed
- Need to clean orphans

```python
# Sync creates/updates files in correct locations
zk_system(action="sync")

# Reorganize moves misplaced files and cleans up
zk_reorganize_obsidian(mode="execute", clean_orphans=True)
```

## Testing Requirements

### Unit Tests

```python
class TestReorganizeObsidian:
    def test_preview_mode_no_changes(self, obsidian_vault, notes):
        """Preview mode doesn't modify files."""
        original_files = list(obsidian_vault.rglob("*.md"))

        result = zk_reorganize_obsidian(mode="preview")

        assert "Preview" in result
        assert list(obsidian_vault.rglob("*.md")) == original_files

    def test_detect_files_needing_move(self, obsidian_vault):
        """Detect files in wrong locations."""
        # Create file in old flat structure
        old_path = obsidian_vault / "project" / "Note (abc12345).md"
        old_path.parent.mkdir(parents=True, exist_ok=True)
        old_path.write_text("content")

        result = zk_reorganize_obsidian(mode="preview")

        assert "Files to Move" in result
        assert "project/general/Note" in result  # Expected new location

    def test_execute_moves_files(self, obsidian_vault, note_in_db):
        """Execute mode moves files to correct locations."""
        # File in wrong location
        wrong_path = obsidian_vault / "wrong" / f"Note ({note_in_db.id[-8:]}).md"
        wrong_path.parent.mkdir(parents=True, exist_ok=True)
        wrong_path.write_text("content")

        zk_reorganize_obsidian(mode="execute")

        # File should be moved
        assert not wrong_path.exists()
        correct_path = obsidian_vault / note_in_db.project / note_in_db.note_purpose.value
        assert (correct_path / f"Note ({note_in_db.id[-8:]}).md").exists()

    def test_detect_orphans(self, obsidian_vault):
        """Detect files without corresponding notes."""
        orphan = obsidian_vault / "project" / "general" / "Orphan (zzz99999).md"
        orphan.parent.mkdir(parents=True, exist_ok=True)
        orphan.write_text("orphaned content")

        result = zk_reorganize_obsidian(mode="preview")

        assert "Orphaned Files" in result
        assert "Orphan" in result

    def test_clean_orphans(self, obsidian_vault):
        """Remove orphaned files when requested."""
        orphan = obsidian_vault / "orphan" / "Old Note (zzz99999).md"
        orphan.parent.mkdir(parents=True, exist_ok=True)
        orphan.write_text("orphaned content")

        zk_reorganize_obsidian(mode="execute", clean_orphans=True)

        assert not orphan.exists()

    def test_creates_backup(self, obsidian_vault, backup_dir):
        """Backup created before execute."""
        zk_reorganize_obsidian(mode="execute", backup=True)

        backups = list(backup_dir.glob("obsidian_*.tar.gz"))
        assert len(backups) == 1

    def test_skip_backup_when_disabled(self, obsidian_vault, backup_dir):
        """No backup when disabled."""
        zk_reorganize_obsidian(mode="execute", backup=False)

        backups = list(backup_dir.glob("obsidian_*.tar.gz"))
        assert len(backups) == 0

    def test_cleanup_empty_directories(self, obsidian_vault, note_in_db):
        """Empty directories removed after moves."""
        # Create nested empty structure
        old_dir = obsidian_vault / "old-project" / "old-purpose"
        old_dir.mkdir(parents=True, exist_ok=True)
        old_file = old_dir / f"Note ({note_in_db.id[-8:]}).md"
        old_file.write_text("content")

        zk_reorganize_obsidian(mode="execute")

        assert not old_dir.exists()
        assert not old_dir.parent.exists()
```

### Integration Tests

```python
class TestReorganizeObsidianE2E:
    def test_full_reorganization_workflow(self, isolated_env):
        """Test complete reorganization workflow."""
        # 1. Create notes with old structure
        for i in range(5):
            note = create_note(project="project-a", note_purpose=NotePurpose.GENERAL)

        # 2. Manually place files in flat structure (simulating old behavior)
        for note in get_all_notes():
            old_path = obsidian_vault / note.project / f"{note.title} ({note.id[-8:]}).md"
            old_path.parent.mkdir(parents=True, exist_ok=True)
            old_path.write_text(note.to_markdown())

        # 3. Preview reorganization
        preview = zk_reorganize_obsidian(mode="preview")
        assert "Files to Move (5)" in preview

        # 4. Execute reorganization
        result = zk_reorganize_obsidian(mode="execute")
        assert "Files moved: 5" in result

        # 5. Verify new structure
        for note in get_all_notes():
            expected = obsidian_vault / note.project / "general" / f"{note.title} ({note.id[-8:]}).md"
            assert expected.exists()

    def test_incremental_reorganization(self, isolated_env):
        """Test reorganizing after adding new notes."""
        # 1. Initial sync
        zk_system(action="sync")

        # 2. Add notes with different purpose
        new_note = create_note(project="project-a", note_purpose=NotePurpose.RESEARCH)

        # 3. Sync adds new note correctly
        zk_system(action="sync")

        # 4. Verify no reorganization needed
        preview = zk_reorganize_obsidian(mode="preview")
        assert "No reorganization needed" in preview or "Files to Move (0)" in preview
```

## Files to Modify

| File | Changes |
|------|---------|
| `src/znote_mcp/server/mcp_server.py` | Add `zk_reorganize_obsidian` tool |
| `src/znote_mcp/storage/note_repository.py` | Add `reorganize_obsidian()` method |
| `src/znote_mcp/backup.py` | Add `create_obsidian_backup()` function |
| `tests/test_obsidian_reorganize.py` | New file - reorganization tests |

## Future Enhancements

1. **Dry-run with diff**: Show actual file content diffs in preview
2. **Selective reorganization**: Only reorganize specific projects
3. **Undo capability**: Track changes and allow reverting
4. **Progress reporting**: For large vaults, show progress percentage
5. **Conflict resolution**: Interactive prompts for overwrite decisions
6. **Obsidian link updates**: Update `[[wikilinks]]` if note titles change
