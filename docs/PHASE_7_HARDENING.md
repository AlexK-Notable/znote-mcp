# Phase 7: Error Handling Hardening

Issues identified by review agents after Phase 5 & 6 fixes. Focuses on eliminating silent failures and strengthening type invariants.

## Review Date: 2026-01-11
## Completion Date: 2026-01-11

---

## Critical Issues

### 1. Broad Exception Catch in `rebuild_index()`

**Status:** ✅ Fixed

**Location:** `note_repository.py` lines 100-106

**Problem:** Catches bare `Exception` when processing files, logs error, and silently continues. Masks serious issues like `PermissionError`, `MemoryError`, `KeyboardInterrupt`.

**Impact:** Users have notes that silently fail to index. Search doesn't find certain notes with no indication why.

**Resolution:**
- Catches specific exceptions: `IOError`, `OSError`, `ValueError`, `yaml.YAMLError`
- Tracks failed files in `failed_files` list with warning log
- System errors (`MemoryError`, `KeyboardInterrupt`) now propagate

---

### 2. Broad Exception Catch in Link Parsing

**Status:** ✅ Fixed

**Location:** `note_repository.py` `_parse_note_from_markdown()`

**Problem:** Link parsing catches bare `Exception` and silently continues, hiding bugs in parsing logic.

**Impact:** Broken links silently ignored. Note appears to load but relationships are lost.

**Resolution:**
- Catches only `ValueError`, `IndexError` for malformed input
- Added `logger.warning()` for unknown link types
- `AttributeError`, `TypeError` now propagate (indicates bugs)

---

### 3. Broad Exception Catch in `get_all()`

**Status:** ✅ Fixed

**Location:** `note_repository.py` `get_all()`

**Problem:** Individual note loading failures logged and skipped silently.

**Impact:** Incomplete results from `zk_list_all_notes` without user awareness.

**Resolution:**
- Tracks failed IDs in `failed_ids` list
- Catches specific exceptions: `IOError`, `OSError`, `ValueError`, `yaml.YAMLError`
- Logs warning with failed count and IDs

---

### 4. Broad Exception Catch in `sync_to_obsidian()`

**Status:** ✅ Fixed

**Location:** `note_repository.py` `sync_to_obsidian()`

**Problem:** Catches bare `Exception` during sync, potentially leaving vault inconsistent.

**Impact:** Users see "Successfully synced X notes" but Y notes silently failed.

**Resolution:**
- Catches specific exceptions: `IOError`, `OSError`, `PermissionError`, `ValueError`, `yaml.YAMLError`
- Tracks failed files in `failed_files` list
- Logs warning with failed file list

---

### 5. Silent `except Exception: pass` in `delete()`

**Status:** ✅ Fixed

**Location:** `note_repository.py` `delete()`

**Problem:** When getting note info before deletion, all exceptions are swallowed with `pass`.

**Impact:** Could mask critical system errors, data corruption.

**Resolution:**
- Catches specific exceptions: `IOError`, `OSError`, `ValueError`, `yaml.YAMLError`
- Logs warning when note can't be read for Obsidian cleanup
- System errors propagate

---

## High Priority Issues

### 6. File/DB Operation Ordering in Bulk Tag Methods

**Status:** ✅ Fixed

**Location:** `note_repository.py` `bulk_add_tags()` and `bulk_remove_tags()`

**Problem:** File writes happen inside DB session loop. If file write fails mid-loop, DB changes still commit at end.

**Impact:** Potential DB/file desync on partial file failures.

**Resolution:**
- Restructured to match `bulk_update_project` pattern
- Added `notes_to_update: List[Tuple[str, Note]]` to collect notes during DB loop
- DB commits first, then all file operations happen with single `file_lock` acquisition

---

### 7. BulkOperationError Weak Invariant Enforcement

**Status:** ✅ Fixed

**Location:** `exceptions.py` `BulkOperationError` class

**Problem:** No constructor validation. Can create invalid states like `success_count > total_count`.

**Resolution:**
- Added constructor validation for count invariants:
  - `total_count` must be non-negative
  - `success_count` must be non-negative
  - `success_count` cannot exceed `total_count`
- Added defensive copy of `failed_ids` list
- Added `@property failed_count` computed from `total - success`

---

## Medium Priority Issues

### 8. Silent Default for Invalid NoteType

**Status:** ✅ Fixed

**Location:** `note_repository.py` `_parse_note_from_markdown()`

**Problem:** Invalid note type silently defaults to `PERMANENT` without logging.

**Resolution:** Added `logger.warning()` when defaulting.

---

### 9. Silent Default for Invalid LinkType

**Status:** ✅ Fixed

**Location:** `note_repository.py` `_parse_note_from_markdown()`

**Problem:** Invalid link type silently defaults to `REFERENCE` without logging.

**Resolution:** Added `logger.warning()` when defaulting.

---

### 10. `_fallback_text_search()` Missing Error Handling

**Status:** ✅ Fixed

**Location:** `note_repository.py` `_fallback_text_search()`

**Problem:** Docstring claims method raises `SearchError` but no try-except exists.

**Resolution:** Wrapped DB operation in try-except, raises `SearchError` as documented.

---

## Implementation Order

1. ✅ **BulkOperationError invariants** (Issue 7) - Foundation for other fixes
2. ✅ **Critical exception handling** (Issues 1-5) - Most impactful
3. ✅ **Bulk tag file ordering** (Issue 6) - Consistency improvement
4. ✅ **Silent defaults logging** (Issues 8-9) - User feedback
5. ✅ **Fallback search error handling** (Issue 10) - API consistency

## Testing Requirements

- [x] All existing tests pass (88 tests)
- [ ] Test rebuild_index with corrupted file
- [ ] Test get_all with unreadable note
- [ ] Test sync_to_obsidian with permission error
- [ ] Test BulkOperationError invariant validation

## Summary

All 10 issues identified during Phase 7 review have been resolved:
- **5 Critical Issues**: Fixed
- **2 High Priority Issues**: Fixed
- **3 Medium Priority Issues**: Fixed

Key improvements:
1. All broad `Exception` catches replaced with specific exception types
2. Silent failures now tracked and logged for user awareness
3. BulkOperationError has proper invariant enforcement
4. Bulk tag operations follow atomic DB-then-file pattern
5. Invalid defaults logged for debugging
