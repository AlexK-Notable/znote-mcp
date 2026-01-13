# Phase 9: Implementation Plan

Based on validated findings from Phase 8 agent review. Addresses critical data integrity, concurrency, and performance issues.

## Plan Date: 2026-01-11

---

# Phase 9A: Data Integrity (Critical)

## 9A-1: Obsidian Title Collision Fix (LD-3)

### Problem
Notes with titles "Hello: World" and "Hello; World" both sanitize to "Hello_ World.md", causing silent overwrites in Obsidian vault.

### Current Code (`note_repository.py:397-399`)
```python
safe_title = "".join(
    c if c.isalnum() or c in " -_" else "_"
    for c in note.title
).strip()
```

### Solution: Append Note ID Suffix for Uniqueness
Instead of relying solely on sanitized title, append note ID to guarantee uniqueness.

### Implementation

**Step 1: Update `_mirror_to_obsidian()` signature**
```python
def _mirror_to_obsidian(self, note: Note, markdown: str) -> None:
    """Mirror a note to the Obsidian vault if configured."""
    if not self.obsidian_vault_path:
        return

    # Sanitize project name
    safe_project = "".join(
        c if c.isalnum() or c in " -_" else "_"
        for c in note.project
    ).strip() or "general"

    # Sanitize title and append ID for uniqueness
    safe_title = "".join(
        c if c.isalnum() or c in " -_" else "_"
        for c in note.title
    ).strip()

    # Append short ID suffix to prevent collisions
    # Use last 8 chars of ID for brevity while maintaining uniqueness
    id_suffix = note.id[-8:] if len(note.id) >= 8 else note.id
    safe_filename = f"{safe_title} ({id_suffix})" if safe_title else note.id

    # Create project subdirectory
    project_dir = self.obsidian_vault_path / safe_project
    project_dir.mkdir(parents=True, exist_ok=True)

    obsidian_file_path = project_dir / f"{safe_filename}.md"
    # ... rest of method
```

**Step 2: Update `_delete_from_obsidian()` to match new naming**
The method currently searches by title only. Update to search for files containing the ID suffix.

```python
def _delete_from_obsidian(
    self, note_id: str, note_title: Optional[str], note_project: Optional[str]
) -> None:
    """Delete a note's mirror from Obsidian vault."""
    if not self.obsidian_vault_path:
        return

    id_suffix = note_id[-8:] if len(note_id) >= 8 else note_id

    # If we have project, search only in that directory
    if note_project:
        safe_project = "".join(
            c if c.isalnum() or c in " -_" else "_"
            for c in note_project
        ).strip() or "general"
        search_dirs = [self.obsidian_vault_path / safe_project]
    else:
        # Search all subdirectories
        search_dirs = [d for d in self.obsidian_vault_path.iterdir() if d.is_dir()]

    # Search for file with matching ID suffix
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for file_path in search_dir.glob(f"*({id_suffix}).md"):
            try:
                file_path.unlink()
                logger.debug(f"Deleted Obsidian mirror: {file_path}")
                return  # Found and deleted
            except OSError as e:
                logger.warning(f"Failed to delete Obsidian mirror {file_path}: {e}")
```

### Testing Requirements
- [ ] Test: Two notes with colliding sanitized titles create separate files
- [ ] Test: Deleting note removes correct Obsidian mirror
- [ ] Test: Updating note title updates Obsidian filename correctly

### Risk Assessment
- **Breaking Change**: Existing Obsidian mirrors will have old names. Need migration or full re-sync.
- **Mitigation**: Add `sync_to_obsidian()` call after upgrade, or document manual re-sync.

---

## 9A-2: Add Project Column to Database (AR-1)

### Problem
`DBNote` has no `project` column. Project is only in markdown frontmatter. Cannot filter notes by project in SQL queries.

### Current Schema (`db_models.py:23-52`)
```python
class DBNote(Base):
    __tablename__ = "notes"
    id = Column(String(255), primary_key=True, index=True)
    title = Column(String(255), nullable=False, index=True)
    content = Column(Text, nullable=False)
    note_type = Column(String(50), default=NoteType.PERMANENT.value, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.datetime.now, nullable=False)
    updated_at = Column(DateTime, default=datetime.datetime.now, nullable=False)
    # NO project column
```

### Solution: Add Project Column with Index

### Implementation

**Step 1: Add column to `DBNote` model**
```python
class DBNote(Base):
    __tablename__ = "notes"
    id = Column(String(255), primary_key=True, index=True)
    title = Column(String(255), nullable=False, index=True)
    content = Column(Text, nullable=False)
    note_type = Column(String(50), default=NoteType.PERMANENT.value, nullable=False, index=True)
    project = Column(String(255), default="default", nullable=False, index=True)  # NEW
    created_at = Column(DateTime, default=datetime.datetime.now, nullable=False)
    updated_at = Column(DateTime, default=datetime.datetime.now, nullable=False)
```

**Step 2: Add migration helper to NoteRepository**
Since no Alembic setup exists, add SQLite ALTER TABLE in `_init_database()`:

```python
def _init_database(self) -> None:
    """Initialize database tables and run migrations."""
    Base.metadata.create_all(self.engine)

    # Migration: Add project column if missing (Phase 9A-2)
    with self.engine.connect() as conn:
        # Check if project column exists
        result = conn.execute(text("PRAGMA table_info(notes)"))
        columns = [row[1] for row in result.fetchall()]

        if "project" not in columns:
            logger.info("Migrating database: adding 'project' column to notes table")
            conn.execute(text(
                "ALTER TABLE notes ADD COLUMN project VARCHAR(255) DEFAULT 'default' NOT NULL"
            ))
            # Create index
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_notes_project ON notes(project)"))
            conn.commit()

            # Trigger rebuild to populate from markdown files
            logger.info("Rebuilding index to populate project column from markdown")
            self.rebuild_index()
```

**Step 3: Update `_index_note()` to include project**
```python
def _index_note(self, note: Note) -> None:
    """Index a note in the database."""
    with self.session_factory() as session:
        db_note = DBNote(
            id=note.id,
            title=note.title,
            content=note.content,
            note_type=note.note_type.value,
            project=note.project,  # NEW
            created_at=note.created_at,
            updated_at=note.updated_at
        )
        # ... rest of method
```

**Step 4: Update all DB queries that create/update notes**
- `update()` - Add `db_note.project = note.project`
- `bulk_update_project()` - Update `DBNote.project` directly instead of re-indexing

**Step 5: Enable SQL-level project filtering**
Add helper method:
```python
def get_by_project(self, project: str) -> List[Note]:
    """Get all notes in a project using SQL filter."""
    with self.session_factory() as session:
        db_notes = session.scalars(
            select(DBNote).where(DBNote.project == project)
        ).all()
        return [self.get(n.id) for n in db_notes if self.get(n.id)]
```

### Testing Requirements
- [ ] Test: New database gets project column
- [ ] Test: Existing database migrates and adds project column
- [ ] Test: Rebuild index populates project from markdown
- [ ] Test: `get_by_project()` returns correct notes

### Risk Assessment
- **Breaking Change**: None - additive schema change with default value
- **Data**: Existing notes get `project='default'` until rebuild_index runs

---

# Phase 9B: Concurrency (High Priority)

## 9B-1: Tag Deduplication Race Fix (LD-2)

### Problem
Two concurrent `bulk_add_tags` operations adding the same new tag: both check "tag doesn't exist", both try INSERT, one hits UNIQUE constraint and rolls back entire transaction.

### Current Code (`note_repository.py:1296-1305`)
```python
db_tag = session.scalar(
    select(DBTag).where(DBTag.name == tag_name)
)
if not db_tag:
    db_tag = DBTag(name=tag_name)
    session.add(db_tag)
    session.flush()  # <-- UNIQUE violation here
```

### Solution: Use INSERT OR IGNORE / ON CONFLICT

### Implementation

**Step 1: Add upsert helper for tags**
```python
def _get_or_create_tag(self, session: Session, tag_name: str) -> DBTag:
    """Get existing tag or create new one, handling race conditions."""
    # Try to get existing tag first
    db_tag = session.scalar(select(DBTag).where(DBTag.name == tag_name))
    if db_tag:
        return db_tag

    # Try to insert, handling race condition
    try:
        db_tag = DBTag(name=tag_name)
        session.add(db_tag)
        session.flush()
        return db_tag
    except IntegrityError:
        # Another transaction created this tag - roll back and fetch
        session.rollback()
        db_tag = session.scalar(select(DBTag).where(DBTag.name == tag_name))
        if db_tag:
            return db_tag
        # If still not found, something is very wrong
        raise StorageError(
            f"Failed to create or retrieve tag '{tag_name}'",
            operation="get_or_create_tag",
            code=ErrorCode.STORAGE_WRITE_FAILED
        )
```

**Alternative: SQLite-specific INSERT OR IGNORE**
```python
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

def _ensure_tag_exists(self, session: Session, tag_name: str) -> DBTag:
    """Ensure tag exists using INSERT OR IGNORE."""
    stmt = sqlite_insert(DBTag).values(name=tag_name).on_conflict_do_nothing()
    session.execute(stmt)
    session.flush()
    return session.scalar(select(DBTag).where(DBTag.name == tag_name))
```

**Step 2: Update all tag creation sites**
- `bulk_add_tags()` line 1296
- `update()` line 616
- `_index_note()` (if tags created there)

### Testing Requirements
- [ ] Test: Concurrent tag creation doesn't fail
- [ ] Test: Multiple bulk_add_tags with same new tag both succeed
- [ ] Test: Tag deduplication still works

### Risk Assessment
- **Performance**: Slight overhead from retry logic
- **Compatibility**: SQLite INSERT OR IGNORE is dialect-specific; generic version uses try/except

---

## 9B-2: Update-Delete Race Fix (LD-12)

### Problem
Thread A reads note, Thread B deletes note, Thread A writes (recreates file). Deleted note resurrects as zombie.

### Current Pattern
```python
# update() - gap between read and write
existing_note = self.get(note.id)  # Read
if not existing_note:
    raise ValueError(...)
# ... processing ...
with self.file_lock:  # Lock acquired AFTER read
    with open(file_path, "w") as f:
        f.write(markdown)
```

### Solution: Expand Lock Scope to Cover Check-Modify-Write

### Implementation

**Step 1: Create note-level locking mechanism**
```python
from threading import RLock
from weakref import WeakValueDictionary

class NoteRepository:
    def __init__(self, ...):
        # ... existing init ...
        self._note_locks: WeakValueDictionary[str, RLock] = WeakValueDictionary()
        self._note_locks_lock = RLock()  # Lock for accessing _note_locks dict

    def _get_note_lock(self, note_id: str) -> RLock:
        """Get or create a lock for a specific note."""
        with self._note_locks_lock:
            if note_id not in self._note_locks:
                self._note_locks[note_id] = RLock()
            return self._note_locks[note_id]
```

**Step 2: Update `update()` method**
```python
def update(self, note: Note) -> Note:
    """Update a note with per-note locking."""
    note_lock = self._get_note_lock(note.id)

    with note_lock:
        # Check if note exists
        existing_note = self.get(note.id)
        if not existing_note:
            raise ValueError(f"Note with ID {note.id} does not exist")

        # ... rest of update logic (already protected by note_lock)
```

**Step 3: Update `delete()` method**
```python
def delete(self, id: str) -> None:
    """Delete a note with per-note locking."""
    validate_safe_path_component(id, "Note ID")
    note_lock = self._get_note_lock(id)

    with note_lock:
        file_path = self.notes_dir / f"{id}.md"
        if not file_path.exists():
            raise ValueError(f"Note with ID {id} does not exist")

        # ... rest of delete logic (already protected by note_lock)
```

### Testing Requirements
- [ ] Test: Concurrent update and delete on same note doesn't create zombie
- [ ] Test: Concurrent updates on different notes don't block each other
- [ ] Test: Lock cleanup (WeakValueDictionary) works correctly

### Risk Assessment
- **Complexity**: Per-note locking adds complexity
- **Deadlock**: RLock prevents self-deadlock; no cross-note dependencies
- **Multi-process**: Still doesn't work across processes (requires file-based locking for that)

---

# Phase 9C: Performance (High Priority)

## 9C-1: Memory Bomb Fix in get_all() (PA-1)

### Problem
Triple `joinedload()` creates Cartesian product: note × tags × outgoing_links × incoming_links. 10 tags + 20 links = 200 rows per note deduplicated in memory.

### Current Code (`note_repository.py:531-536`)
```python
query = select(DBNote).options(
    joinedload(DBNote.tags),
    joinedload(DBNote.outgoing_links),
    joinedload(DBNote.incoming_links)
)
```

### Solution: Use `selectinload()` Instead

`selectinload()` executes separate queries (1 for notes, 1 for tags, 1 for links) instead of Cartesian joins.

### Implementation

**Step 1: Replace joinedload with selectinload**
```python
from sqlalchemy.orm import selectinload

def get_all(self) -> List[Note]:
    """Get all notes with efficient loading."""
    with self.session_factory() as session:
        # Use selectinload to avoid Cartesian product
        query = select(DBNote).options(
            selectinload(DBNote.tags),
            selectinload(DBNote.outgoing_links),
            selectinload(DBNote.incoming_links)
        )
        result = session.execute(query)
        db_notes = result.scalars().all()  # No unique() needed with selectinload

        # ... rest of method
```

**Step 2: Apply same fix to search() and other bulk loading methods**
- `search()`
- `get_by_tag()` if it exists
- Any method using joinedload on multiple relationships

### Testing Requirements
- [ ] Test: get_all() returns same results with selectinload
- [ ] Test: Memory usage decreased for large note sets
- [ ] Test: Performance benchmark (selectinload may be slower for small sets but scales better)

### Risk Assessment
- **N+1**: selectinload adds 2 extra queries (for tags and links), but avoids memory explosion
- **Trade-off**: Slightly more queries but much better memory profile

---

## 9C-2: N+1 Query Fix in search() (PA-2)

### Problem
Database query returns N results, then N individual `self.get(note_id)` calls read files from disk.

### Current Code
```python
for db_note in db_notes:
    note = self.get(db_note.id)  # File read per result
    if note:
        notes.append(note)
```

### Solution: Batch Process Results

### Implementation

**Step 1: Create batch file reader**
```python
def _get_notes_batch(self, note_ids: List[str]) -> Dict[str, Note]:
    """Load multiple notes from files in batch."""
    results = {}
    for note_id in note_ids:
        try:
            note = self.get(note_id)
            if note:
                results[note_id] = note
        except (IOError, OSError, ValueError, yaml.YAMLError) as e:
            logger.warning(f"Failed to load note {note_id}: {e}")
    return results
```

**Step 2: Update search() to use batch loading**
Since files are the source of truth and we need frontmatter parsing, the file reads are necessary. The optimization is to:
1. Use DB data directly when possible (title, content already in DB)
2. Only read files for notes that will be returned (already true)
3. Consider caching parsed notes

**Alternative: Return DB data directly, lazy-load metadata**
```python
def search(self, ...) -> List[Note]:
    """Search notes with minimal file I/O."""
    # ... existing DB query ...

    notes = []
    for db_note in db_notes[:limit]:
        # Use DB data directly for most fields
        note = Note(
            id=db_note.id,
            title=db_note.title,
            content=db_note.content,
            note_type=NoteType(db_note.note_type),
            project=db_note.project,  # Requires 9A-2
            created_at=db_note.created_at,
            updated_at=db_note.updated_at,
            tags=[Tag(name=t.name) for t in db_note.tags],
            links=[...],  # From db_note relationships
            metadata={}  # Metadata lost without file read
        )
        notes.append(note)
    return notes
```

### Trade-off Analysis
| Approach | Pros | Cons |
|----------|------|------|
| Keep N file reads | Full accuracy, metadata preserved | Slow for large result sets |
| DB-only results | Fast, no file I/O | Metadata lost, requires project column |
| Hybrid | Metadata available on demand | Complexity |

### Recommendation
Add `include_metadata: bool = False` parameter. When False, return DB-only results. When True, read files.

### Testing Requirements
- [ ] Test: Search results identical with and without metadata
- [ ] Test: Performance improvement measurable
- [ ] Test: Metadata available when requested

---

## 9C-3: O(N²) Similarity Fix (PA-5)

### Problem
`find_similar_notes()` loads ALL notes via `get_all()`, then iterates through each to compute similarity. 10,000 notes = 100,000,000 comparisons.

### Current Code (`zettel_service.py:320-321`)
```python
all_notes = self.repository.get_all()  # O(N) with memory issues
# ...
for other_note in all_notes:  # O(N) iteration
```

### Solution: Limit and Paginate + Use DB for Pre-filtering

### Implementation

**Step 1: Add limit parameter**
```python
def find_similar_notes(
    self,
    note_id: str,
    threshold: float = 0.5,
    limit: int = 50  # NEW: limit results
) -> List[Tuple[Note, float]]:
```

**Step 2: Pre-filter using DB queries**
```python
def find_similar_notes(self, note_id: str, threshold: float = 0.5, limit: int = 50):
    """Find similar notes using optimized queries."""
    note = self.repository.get(note_id)
    if not note:
        raise NoteNotFoundError(note_id)

    note_tags = {tag.name for tag in note.tags}
    note_link_targets = {link.target_id for link in note.links}

    # Step 1: Get candidate notes from DB (those sharing tags or links)
    with self.repository.session_factory() as session:
        # Notes sharing at least one tag
        tag_sharing_ids = set()
        if note_tags:
            result = session.execute(text("""
                SELECT DISTINCT nt.note_id
                FROM note_tags nt
                JOIN tags t ON nt.tag_id = t.id
                WHERE t.name IN :tags AND nt.note_id != :note_id
            """), {"tags": tuple(note_tags), "note_id": note_id})
            tag_sharing_ids = {row[0] for row in result}

        # Notes with link relationships
        link_related_ids = set()
        result = session.execute(text("""
            SELECT DISTINCT source_id FROM links WHERE target_id = :note_id
            UNION
            SELECT DISTINCT target_id FROM links WHERE source_id = :note_id
        """), {"note_id": note_id})
        link_related_ids = {row[0] for row in result}

        # Combine candidates
        candidate_ids = tag_sharing_ids | link_related_ids

    # Step 2: Load only candidate notes
    candidates = [self.repository.get(nid) for nid in candidate_ids if nid]
    candidates = [c for c in candidates if c is not None]

    # Step 3: Calculate similarity for candidates only
    results = []
    for other_note in candidates:
        similarity = self._calculate_similarity(note, other_note)
        if similarity >= threshold:
            results.append((other_note, similarity))

    # Sort and limit
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:limit]
```

**Step 3: Extract similarity calculation to helper**
```python
def _calculate_similarity(self, note: Note, other_note: Note) -> float:
    """Calculate similarity score between two notes."""
    note_tags = {tag.name for tag in note.tags}
    other_tags = {tag.name for tag in other_note.tags}
    note_links = {link.target_id for link in note.links}
    other_links = {link.target_id for link in other_note.links}

    # ... existing similarity calculation ...
```

### Performance Analysis
| Notes | Before (O(N²)) | After (Candidate-based) |
|-------|----------------|-------------------------|
| 100   | 10,000 ops    | ~50 ops (50 candidates) |
| 1,000 | 1,000,000 ops | ~200 ops |
| 10,000| 100,000,000 ops| ~500 ops |

### Testing Requirements
- [ ] Test: Similar notes still found correctly
- [ ] Test: Performance scales sub-linearly with note count
- [ ] Test: Limit parameter works

---

# Phase 9D: Polish (Medium Priority)

## Deferred Items

These items are validated but lower priority:

| ID | Issue | Recommendation |
|----|-------|----------------|
| LD-1 | File-DB consistency phantom | Add content hash comparison option |
| LD-4 | FTS5 syntax injection | Escape OR/AND/NOT or document as feature |
| LD-5 | Link timestamp lie | Store link created_at in DB, preserve on parse |
| LD-7 | LIKE injection | Escape % and _ in search terms |
| LD-8 | ID collision window | Add process ID to timestamp format |
| PA-3 | Sequential Obsidian sync | Add parallel file writes with ThreadPoolExecutor |
| PA-6 | Double rebuild processing | Combine index + FTS in single pass |
| PA-7 | Serialized bulk ops | Pipeline DB and file operations |
| AR-2 | FTS5 coupled to repository | Create SearchBackend abstraction |
| AR-3 | Embedded serialization | Create NoteSerializer interface |
| AR-4 | Scattered business rules | Create ValidationService |
| AR-7 | FTS5 syntax in API | Abstract to "simple" and "advanced" search modes |

---

# Implementation Order

```
Week 1: Phase 9A (Data Integrity)
├─ Day 1-2: 9A-1 Obsidian title collision
├─ Day 3-4: 9A-2 Project column migration
└─ Day 5: Testing and documentation

Week 2: Phase 9B (Concurrency)
├─ Day 1-2: 9B-1 Tag race condition
├─ Day 3-4: 9B-2 Update-delete race
└─ Day 5: Testing and integration

Week 3: Phase 9C (Performance)
├─ Day 1: 9C-1 Memory bomb fix
├─ Day 2-3: 9C-2 N+1 query fix
├─ Day 4: 9C-3 O(N²) similarity fix
└─ Day 5: Benchmarking and validation

Week 4: Stabilization
├─ Day 1-3: Integration testing
├─ Day 4: Documentation updates
└─ Day 5: Release preparation
```

---

# Testing Strategy

## Unit Tests Required
- [ ] `test_obsidian_title_collision.py` - Verify unique filenames
- [ ] `test_project_column_migration.py` - DB migration works
- [ ] `test_tag_race_condition.py` - Concurrent tag creation
- [ ] `test_update_delete_race.py` - No zombie notes
- [ ] `test_selectinload_memory.py` - Memory usage validation
- [ ] `test_similarity_performance.py` - Benchmark similarity function

## Integration Tests Required
- [ ] Full workflow with concurrent operations
- [ ] Large dataset performance tests (1000+ notes)
- [ ] Migration from existing database

---

# Risk Matrix

| Fix | Risk Level | Mitigation |
|-----|------------|------------|
| 9A-1 Obsidian naming | Medium | Document migration, provide re-sync |
| 9A-2 Project column | Low | Additive change with default |
| 9B-1 Tag race | Low | Standard pattern, well-tested |
| 9B-2 Update-delete race | Medium | New locking mechanism, needs testing |
| 9C-1 selectinload | Low | Drop-in replacement |
| 9C-2 N+1 fix | Medium | Trade-off in API (metadata flag) |
| 9C-3 O(N²) fix | Medium | Query changes, verify correctness |

---

# Success Criteria

Phase 9 is complete when:
1. All unit tests pass
2. No data loss scenarios in testing
3. Memory usage stays bounded for 10,000+ notes
4. Concurrent operations don't cause failures
5. Performance benchmarks meet targets:
   - `get_all()` < 2GB RAM for 10,000 notes
   - `find_similar_notes()` < 5 seconds for 10,000 notes
   - `search()` < 1 second for 500 results

