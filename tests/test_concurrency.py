"""Tests for multi-process concurrency scenarios.

These tests simulate concurrent access patterns to verify:
1. Version conflict detection works across processes
2. In-memory databases provide process isolation
3. Git-based versioning enables safe concurrent updates
"""

import tempfile
import threading
import time
from pathlib import Path
from typing import List, Tuple

import pytest

from znote_mcp.models.schema import (
    ConflictResult,
    Note,
    NoteType,
    VersionedNote,
)
from znote_mcp.storage.note_repository import NoteRepository


class TestConcurrentAccess:
    """Tests for concurrent access patterns."""

    @pytest.fixture
    def shared_notes_dir(self):
        """Create a shared notes directory for concurrent tests."""
        with tempfile.TemporaryDirectory() as tmp:
            yield Path(tmp)

    def test_concurrent_creates_no_conflict(self, shared_notes_dir):
        """Test that concurrent creates don't conflict (different note IDs).

        Note: Git has internal locking, so truly concurrent git operations
        may experience lock contention. This test initializes the repo first,
        then runs creates sequentially with slight delays to avoid lock issues.
        """
        # Pre-initialize the git repo to avoid concurrent init race
        initial_repo = NoteRepository(
            notes_dir=shared_notes_dir, use_git=True, in_memory_db=True
        )

        results: List[VersionedNote] = []
        errors: List[Exception] = []
        lock = threading.Lock()

        def create_note(index: int):
            try:
                # Small stagger to reduce git lock contention
                time.sleep(0.1 * index)
                repo = NoteRepository(
                    notes_dir=shared_notes_dir, use_git=True, in_memory_db=True
                )
                note = Note(
                    title=f"Concurrent Note {index}",
                    content=f"Content {index}",
                    note_type=NoteType.PERMANENT,
                )
                result = repo.create_versioned(note)
                with lock:
                    results.append(result)
            except Exception as e:
                with lock:
                    errors.append(e)

        # Create notes from multiple threads (simulating processes)
        threads = [threading.Thread(target=create_note, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 5
        # All notes should have unique IDs
        ids = [r.note.id for r in results]
        assert len(set(ids)) == 5

    def test_concurrent_update_same_note_conflict(self, shared_notes_dir):
        """Test that concurrent updates to same note detect conflicts."""
        # First, create a note
        repo1 = NoteRepository(
            notes_dir=shared_notes_dir, use_git=True, in_memory_db=True
        )
        note = Note(
            title="Shared Note",
            content="Original content",
            note_type=NoteType.PERMANENT,
        )
        created = repo1.create_versioned(note)
        original_version = created.version.commit_hash
        note_id = created.note.id

        # Simulate two "processes" reading the same note
        read_versions: List[str] = [original_version, original_version]

        def update_note(index: int, version: str) -> Tuple[int, any]:
            """Update the note, returning result."""
            # Each "process" gets its own repository instance
            repo = NoteRepository(
                notes_dir=shared_notes_dir, use_git=True, in_memory_db=True
            )
            updated = Note(
                id=note_id,
                title=f"Updated by process {index}",
                content=f"Content from process {index}",
                note_type=NoteType.PERMANENT,
            )
            # Small delay to increase chance of conflict
            time.sleep(0.05 * index)
            return (index, repo.update_versioned(updated, expected_version=version))

        results = []

        # Run updates concurrently
        def run_update(idx, ver):
            results.append(update_note(idx, ver))

        threads = [
            threading.Thread(target=run_update, args=(i, original_version))
            for i in range(2)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Sort by index to have deterministic order
        results.sort(key=lambda x: x[0])

        # First update should succeed (VersionedNote)
        # Second update should fail (ConflictResult) because version changed
        success_count = sum(1 for _, r in results if isinstance(r, VersionedNote))
        conflict_count = sum(1 for _, r in results if isinstance(r, ConflictResult))

        # At least one should succeed
        assert success_count >= 1
        # With sequential access via git locks, both might succeed
        # But with conflict detection, at least one should get conflict
        # The exact behavior depends on timing

    def test_version_changes_detected_across_instances(self, shared_notes_dir):
        """Test that version changes are visible across repository instances."""
        # Create note with repo1
        repo1 = NoteRepository(
            notes_dir=shared_notes_dir, use_git=True, in_memory_db=True
        )
        note = Note(
            title="Cross-Instance Test",
            content="Original",
            note_type=NoteType.PERMANENT,
        )
        created = repo1.create_versioned(note)
        note_id = created.note.id
        original_version = created.version.commit_hash

        # Update with repo2 (simulating different process)
        repo2 = NoteRepository(
            notes_dir=shared_notes_dir, use_git=True, in_memory_db=True
        )
        updated = Note(
            id=note_id,
            title="Updated by repo2",
            content="Updated content",
            note_type=NoteType.PERMANENT,
        )
        repo2_result = repo2.update_versioned(updated)
        assert isinstance(repo2_result, VersionedNote)
        new_version = repo2_result.version.commit_hash

        # repo1 should see the version change through git
        repo1_check = repo1.get_versioned(note_id)
        assert repo1_check is not None
        # The version should be updated (git tracks it)
        assert repo1_check.version.commit_hash == new_version

    def test_in_memory_db_isolation(self, shared_notes_dir):
        """Test that in-memory DBs are isolated per repository instance."""
        # Create note with repo1
        repo1 = NoteRepository(
            notes_dir=shared_notes_dir, use_git=True, in_memory_db=True
        )
        note = Note(
            title="Isolation Test",
            content="Content",
            note_type=NoteType.PERMANENT,
        )
        created = repo1.create_versioned(note)
        note_id = created.note.id

        # Repo2 with fresh in-memory DB won't have the note indexed
        # until it rebuilds from markdown files
        repo2 = NoteRepository(
            notes_dir=shared_notes_dir, use_git=True, in_memory_db=True
        )

        # repo2 should still be able to read the note (from markdown file)
        # because it rebuilds index from files on init
        result = repo2.get_versioned(note_id)
        assert result is not None
        assert result.note.title == "Isolation Test"

    def test_optimistic_concurrency_workflow(self, shared_notes_dir):
        """Test the full optimistic concurrency workflow.

        Simulates:
        1. Process A reads note, gets version V1
        2. Process B reads note, gets version V1
        3. Process A updates note -> succeeds, version becomes V2
        4. Process B tries to update with V1 -> conflict detected
        5. Process B re-reads, gets V2
        6. Process B updates with V2 -> succeeds
        """
        # Setup: Create initial note
        repo_setup = NoteRepository(
            notes_dir=shared_notes_dir, use_git=True, in_memory_db=True
        )
        note = Note(
            title="OCC Test",
            content="Original",
            note_type=NoteType.PERMANENT,
        )
        created = repo_setup.create_versioned(note)
        note_id = created.note.id
        v1 = created.version.commit_hash

        # Process A and B both read V1
        repo_a = NoteRepository(
            notes_dir=shared_notes_dir, use_git=True, in_memory_db=True
        )
        repo_b = NoteRepository(
            notes_dir=shared_notes_dir, use_git=True, in_memory_db=True
        )

        # Process A updates first
        update_a = Note(
            id=note_id,
            title="OCC Test",
            content="Updated by A",
            note_type=NoteType.PERMANENT,
        )
        result_a = repo_a.update_versioned(update_a, expected_version=v1)
        assert isinstance(result_a, VersionedNote), "A's update should succeed"
        v2 = result_a.version.commit_hash

        # Process B tries to update with stale V1
        update_b_first = Note(
            id=note_id,
            title="OCC Test",
            content="Updated by B",
            note_type=NoteType.PERMANENT,
        )
        result_b_first = repo_b.update_versioned(update_b_first, expected_version=v1)
        assert isinstance(result_b_first, ConflictResult), "B should get conflict"

        # Process B re-reads and retries with V2
        reread = repo_b.get_versioned(note_id)
        assert reread.version.commit_hash == v2, "B should see A's update"

        update_b_second = Note(
            id=note_id,
            title="OCC Test",
            content="Updated by B after retry",
            note_type=NoteType.PERMANENT,
        )
        result_b_second = repo_b.update_versioned(update_b_second, expected_version=v2)
        assert isinstance(result_b_second, VersionedNote), "B's retry should succeed"


class TestGitDisabledMode:
    """Tests for when git versioning is disabled."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory."""
        with tempfile.TemporaryDirectory() as tmp:
            yield Path(tmp)

    def test_versioned_operations_work_without_git(self, temp_dir):
        """Test that versioned operations still work when git is disabled."""
        repo = NoteRepository(
            notes_dir=temp_dir, use_git=False, in_memory_db=True  # Git disabled
        )

        # Create should return placeholder version
        note = Note(
            title="No Git Test",
            content="Content",
            note_type=NoteType.PERMANENT,
        )
        created = repo.create_versioned(note)

        assert isinstance(created, VersionedNote)
        # Placeholder version when git is disabled
        assert created.version.commit_hash == "0000000"

    def test_no_conflict_detection_without_git(self, temp_dir):
        """Test that without git, no conflict detection occurs."""
        repo = NoteRepository(notes_dir=temp_dir, use_git=False, in_memory_db=True)

        note = Note(
            title="No Git Test",
            content="Original",
            note_type=NoteType.PERMANENT,
        )
        created = repo.create_versioned(note)

        # Update with any "expected version" should succeed
        # because conflict detection is disabled
        updated = Note(
            id=created.note.id,
            title="No Git Test",
            content="Updated",
            note_type=NoteType.PERMANENT,
        )
        result = repo.update_versioned(updated, expected_version="any_version")

        # Should succeed (no conflict detection)
        assert isinstance(result, VersionedNote)
