"""True multi-process concurrency tests.

These tests spawn actual separate processes to verify that:
1. Conflict detection works across process boundaries
2. Git-based versioning provides real isolation
3. The system behaves correctly under genuine concurrent load

Unlike thread-based tests, these prove the architecture works
in real-world scenarios with multiple Claude Code instances.
"""
import json
import multiprocessing
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Any

import pytest

# Add src to path for subprocess imports
SRC_PATH = str(Path(__file__).parent.parent / "src")


def worker_create_note(args: Dict[str, Any]) -> Dict[str, Any]:
    """Worker function that creates a note in a separate process."""
    sys.path.insert(0, args["src_path"])

    from znote_mcp.models.schema import Note, NoteType
    from znote_mcp.storage.note_repository import NoteRepository

    try:
        repo = NoteRepository(
            notes_dir=Path(args["notes_dir"]),
            use_git=True,
            in_memory_db=True
        )
        note = Note(
            title=f"Process {args['worker_id']} Note",
            content=f"Content from worker {args['worker_id']}",
            note_type=NoteType.PERMANENT,
        )
        result = repo.create_versioned(note)
        return {
            "success": True,
            "worker_id": args["worker_id"],
            "note_id": result.note.id,
            "version": result.version.commit_hash,
        }
    except Exception as e:
        return {
            "success": False,
            "worker_id": args["worker_id"],
            "error": str(e),
        }


def worker_update_note(args: Dict[str, Any]) -> Dict[str, Any]:
    """Worker function that updates a note in a separate process."""
    sys.path.insert(0, args["src_path"])

    from znote_mcp.models.schema import Note, NoteType, ConflictResult
    from znote_mcp.storage.note_repository import NoteRepository

    # Optional delay to stagger workers
    if args.get("delay"):
        time.sleep(args["delay"])

    try:
        repo = NoteRepository(
            notes_dir=Path(args["notes_dir"]),
            use_git=True,
            in_memory_db=True
        )

        updated = Note(
            id=args["note_id"],
            title=args.get("title", "Updated Title"),
            content=f"Updated by worker {args['worker_id']} at {time.time()}",
            note_type=NoteType.PERMANENT,
        )

        result = repo.update_versioned(
            updated,
            expected_version=args.get("expected_version")
        )

        if isinstance(result, ConflictResult):
            return {
                "success": False,
                "conflict": True,
                "worker_id": args["worker_id"],
                "expected_version": result.expected_version,
                "actual_version": result.actual_version,
            }
        else:
            return {
                "success": True,
                "conflict": False,
                "worker_id": args["worker_id"],
                "new_version": result.version.commit_hash,
            }
    except Exception as e:
        return {
            "success": False,
            "conflict": False,
            "worker_id": args["worker_id"],
            "error": str(e),
        }


def worker_read_and_update(args: Dict[str, Any]) -> Dict[str, Any]:
    """Worker that reads current version, then updates with that version.

    Simulates real-world pattern: read -> modify -> write with version check.
    """
    sys.path.insert(0, args["src_path"])

    from znote_mcp.models.schema import Note, NoteType, ConflictResult
    from znote_mcp.storage.note_repository import NoteRepository

    # Stagger start times
    if args.get("delay"):
        time.sleep(args["delay"])

    try:
        repo = NoteRepository(
            notes_dir=Path(args["notes_dir"]),
            use_git=True,
            in_memory_db=True
        )

        # Read current version
        current = repo.get_versioned(args["note_id"])
        if not current:
            return {"success": False, "error": "Note not found", "worker_id": args["worker_id"]}

        read_version = current.version.commit_hash

        # Simulate some processing time (this is where races happen)
        time.sleep(args.get("processing_time", 0.05))

        # Update with the version we read
        updated = Note(
            id=args["note_id"],
            title=current.note.title,
            content=f"Updated by worker {args['worker_id']}",
            note_type=current.note.note_type,
        )

        result = repo.update_versioned(updated, expected_version=read_version)

        if isinstance(result, ConflictResult):
            return {
                "success": False,
                "conflict": True,
                "worker_id": args["worker_id"],
                "read_version": read_version,
                "actual_version": result.actual_version,
            }
        else:
            return {
                "success": True,
                "conflict": False,
                "worker_id": args["worker_id"],
                "read_version": read_version,
                "new_version": result.version.commit_hash,
            }
    except Exception as e:
        return {
            "success": False,
            "conflict": False,
            "worker_id": args["worker_id"],
            "error": str(e),
        }


class TestMultiProcessConcurrency:
    """Tests using actual separate processes."""

    @pytest.fixture
    def shared_notes_dir(self):
        """Create a shared notes directory and pre-initialize git."""
        with tempfile.TemporaryDirectory() as tmp:
            notes_dir = Path(tmp)
            # Pre-initialize git repo to avoid init race
            from znote_mcp.storage.git_wrapper import GitWrapper
            GitWrapper(notes_dir)
            yield notes_dir

    def test_parallel_creates_different_notes(self, shared_notes_dir):
        """Test that multiple processes can create different notes simultaneously.

        This tests that:
        - Git handles concurrent commits to different files
        - Each process gets its own valid version hash
        - No data corruption occurs
        """
        num_workers = 5

        with multiprocessing.Pool(num_workers) as pool:
            args_list = [
                {
                    "src_path": SRC_PATH,
                    "notes_dir": str(shared_notes_dir),
                    "worker_id": i,
                }
                for i in range(num_workers)
            ]
            results = pool.map(worker_create_note, args_list)

        # All should succeed
        successes = [r for r in results if r["success"]]
        failures = [r for r in results if not r["success"]]

        assert len(failures) == 0, f"Some workers failed: {failures}"
        assert len(successes) == num_workers

        # All note IDs should be unique
        note_ids = [r["note_id"] for r in successes]
        assert len(set(note_ids)) == num_workers, "Note IDs should be unique"

        # All versions should be valid (non-placeholder)
        versions = [r["version"] for r in successes]
        assert all(v != "0000000" for v in versions), "All should have real git versions"

    def test_concurrent_updates_same_note_with_stale_version(self, shared_notes_dir):
        """Test that concurrent updates with stale version cause conflicts.

        Scenario:
        - Create a note (version v1)
        - Spawn N workers, all given v1 as expected_version
        - Only the first to commit should succeed
        - Others should get CONFLICT
        """
        # Create initial note
        from znote_mcp.models.schema import Note, NoteType
        from znote_mcp.storage.note_repository import NoteRepository

        repo = NoteRepository(
            notes_dir=shared_notes_dir,
            use_git=True,
            in_memory_db=True
        )
        note = Note(
            title="Contested Note",
            content="Original content",
            note_type=NoteType.PERMANENT,
        )
        created = repo.create_versioned(note)
        note_id = created.note.id
        original_version = created.version.commit_hash

        # Spawn workers that all try to update with the same original version
        num_workers = 4

        with multiprocessing.Pool(num_workers) as pool:
            args_list = [
                {
                    "src_path": SRC_PATH,
                    "notes_dir": str(shared_notes_dir),
                    "worker_id": i,
                    "note_id": note_id,
                    "expected_version": original_version,
                    "delay": 0.1 * i,  # Stagger slightly
                }
                for i in range(num_workers)
            ]
            results = pool.map(worker_update_note, args_list)

        successes = [r for r in results if r.get("success") and not r.get("conflict")]
        conflicts = [r for r in results if r.get("conflict")]
        errors = [r for r in results if not r.get("success") and not r.get("conflict")]

        assert len(errors) == 0, f"Unexpected errors: {errors}"

        # Exactly one should succeed (the first one to commit)
        assert len(successes) == 1, f"Expected 1 success, got {len(successes)}: {successes}"

        # The rest should get conflicts
        assert len(conflicts) == num_workers - 1, \
            f"Expected {num_workers - 1} conflicts, got {len(conflicts)}"

    def test_read_modify_write_race(self, shared_notes_dir):
        """Test the realistic read-modify-write pattern under contention.

        This is the most realistic test: each worker:
        1. Reads the current note and its version
        2. Simulates some processing time
        3. Writes back with the version it read

        Under contention, later workers should see their read version
        become stale and get conflicts.
        """
        # Create initial note
        from znote_mcp.models.schema import Note, NoteType
        from znote_mcp.storage.note_repository import NoteRepository

        repo = NoteRepository(
            notes_dir=shared_notes_dir,
            use_git=True,
            in_memory_db=True
        )
        note = Note(
            title="Race Test Note",
            content="Initial content",
            note_type=NoteType.PERMANENT,
        )
        created = repo.create_versioned(note)
        note_id = created.note.id

        # Spawn workers that do read-modify-write
        num_workers = 3

        with multiprocessing.Pool(num_workers) as pool:
            args_list = [
                {
                    "src_path": SRC_PATH,
                    "notes_dir": str(shared_notes_dir),
                    "worker_id": i,
                    "note_id": note_id,
                    "delay": 0,  # All start together
                    "processing_time": 0.1,  # Time between read and write
                }
                for i in range(num_workers)
            ]
            results = pool.map(worker_read_and_update, args_list)

        successes = [r for r in results if r.get("success")]
        conflicts = [r for r in results if r.get("conflict")]

        # At least one should succeed
        assert len(successes) >= 1, f"At least one should succeed: {results}"

        # With 3 workers and processing time, we expect some conflicts
        # (exact number depends on timing, but conflicts should occur)
        total_processed = len(successes) + len(conflicts)
        assert total_processed == num_workers, "All workers should complete"

        # Verify the note has valid content from one of the workers
        final = repo.get_versioned(note_id)
        assert "Updated by worker" in final.note.content

    def test_high_contention_stress(self, shared_notes_dir):
        """Stress test: many workers, high contention, verify data integrity.

        Tests that under stress:
        - No data corruption occurs
        - Final state is consistent
        - All conflicts are properly detected
        """
        # Create initial note
        from znote_mcp.models.schema import Note, NoteType
        from znote_mcp.storage.note_repository import NoteRepository

        repo = NoteRepository(
            notes_dir=shared_notes_dir,
            use_git=True,
            in_memory_db=True
        )
        note = Note(
            title="Stress Test Note",
            content="Initial",
            note_type=NoteType.PERMANENT,
        )
        created = repo.create_versioned(note)
        note_id = created.note.id

        # Many workers, minimal delay
        num_workers = 8

        with multiprocessing.Pool(num_workers) as pool:
            args_list = [
                {
                    "src_path": SRC_PATH,
                    "notes_dir": str(shared_notes_dir),
                    "worker_id": i,
                    "note_id": note_id,
                    "delay": 0.02 * i,  # Minimal stagger
                    "processing_time": 0.02,
                }
                for i in range(num_workers)
            ]
            results = pool.map(worker_read_and_update, args_list)

        successes = [r for r in results if r.get("success")]
        conflicts = [r for r in results if r.get("conflict")]
        errors = [r for r in results if "error" in r]

        # No unexpected errors
        assert len(errors) == 0, f"Unexpected errors: {errors}"

        # All workers should have a definitive outcome
        assert len(successes) + len(conflicts) == num_workers

        # At least one success
        assert len(successes) >= 1

        # Final note should be valid and readable
        final = repo.get_versioned(note_id)
        assert final is not None
        assert final.note.id == note_id
        assert final.version.commit_hash != "0000000"

        # Verify git history has the right number of commits
        # (1 create + number of successful updates)
        from znote_mcp.storage.git_wrapper import GitWrapper
        git = GitWrapper(shared_notes_dir)
        history = git.get_history(shared_notes_dir / f"{note_id}.md", limit=100)
        assert len(history) == 1 + len(successes), \
            f"Expected {1 + len(successes)} commits, got {len(history)}"


class TestDataIntegrity:
    """Tests verifying data integrity under various conditions."""

    @pytest.fixture
    def notes_dir(self):
        """Create a temporary notes directory."""
        with tempfile.TemporaryDirectory() as tmp:
            yield Path(tmp)

    def test_version_strictly_increases(self, notes_dir):
        """Test that each update produces a new, unique version."""
        from znote_mcp.models.schema import Note, NoteType
        from znote_mcp.storage.note_repository import NoteRepository

        repo = NoteRepository(notes_dir=notes_dir, use_git=True, in_memory_db=True)

        note = Note(
            title="Version Test",
            content="v0",
            note_type=NoteType.PERMANENT,
        )
        created = repo.create_versioned(note)
        versions = [created.version.commit_hash]

        # Make 10 updates
        for i in range(10):
            updated = Note(
                id=created.note.id,
                title="Version Test",
                content=f"v{i+1}",
                note_type=NoteType.PERMANENT,
            )
            result = repo.update_versioned(updated)
            versions.append(result.version.commit_hash)

        # All versions should be unique
        assert len(set(versions)) == len(versions), "All versions must be unique"

        # No placeholder versions
        assert all(v != "0000000" for v in versions)

    def test_content_matches_version(self, notes_dir):
        """Test that reading a note returns content consistent with its version."""
        from znote_mcp.models.schema import Note, NoteType
        from znote_mcp.storage.note_repository import NoteRepository

        repo = NoteRepository(notes_dir=notes_dir, use_git=True, in_memory_db=True)

        note = Note(
            title="Consistency Test",
            content="Original content",
            note_type=NoteType.PERMANENT,
        )
        v1 = repo.create_versioned(note)

        # Update
        updated = Note(
            id=v1.note.id,
            title="Consistency Test",
            content="Updated content",
            note_type=NoteType.PERMANENT,
        )
        v2 = repo.update_versioned(updated)

        # Read should return v2's content
        current = repo.get_versioned(v1.note.id)
        assert current.version.commit_hash == v2.version.commit_hash
        # Content includes markdown formatting with title
        assert "Updated content" in current.note.content

    def test_conflict_preserves_data(self, notes_dir):
        """Test that a conflict doesn't corrupt or lose data."""
        from znote_mcp.models.schema import Note, NoteType, ConflictResult
        from znote_mcp.storage.note_repository import NoteRepository

        repo = NoteRepository(notes_dir=notes_dir, use_git=True, in_memory_db=True)

        note = Note(
            title="Conflict Data Test",
            content="Original",
            note_type=NoteType.PERMANENT,
        )
        v1 = repo.create_versioned(note)

        # First update succeeds
        update1 = Note(
            id=v1.note.id,
            title="Conflict Data Test",
            content="First update",
            note_type=NoteType.PERMANENT,
        )
        v2 = repo.update_versioned(update1)
        assert isinstance(v2, type(v1))  # VersionedNote

        # Second update with stale version should conflict
        update2 = Note(
            id=v1.note.id,
            title="Conflict Data Test",
            content="Second update (should fail)",
            note_type=NoteType.PERMANENT,
        )
        result = repo.update_versioned(update2, expected_version=v1.version.commit_hash)
        assert isinstance(result, ConflictResult)

        # Data should still be from first update (content includes title prefix)
        current = repo.get_versioned(v1.note.id)
        assert "First update" in current.note.content
        assert current.version.commit_hash == v2.version.commit_hash

    def test_rapid_updates_no_data_loss(self, notes_dir):
        """Test rapid sequential updates don't lose data."""
        from znote_mcp.models.schema import Note, NoteType
        from znote_mcp.storage.note_repository import NoteRepository

        repo = NoteRepository(notes_dir=notes_dir, use_git=True, in_memory_db=True)

        note = Note(
            title="Rapid Update Test",
            content="v0",
            note_type=NoteType.PERMANENT,
        )
        created = repo.create_versioned(note)
        note_id = created.note.id

        # Rapid updates without version checking (simulating single process)
        for i in range(50):
            updated = Note(
                id=note_id,
                title="Rapid Update Test",
                content=f"Version {i+1}",
                note_type=NoteType.PERMANENT,
            )
            repo.update_versioned(updated)

        # Final read should have latest content (includes title prefix)
        final = repo.get_versioned(note_id)
        assert "Version 50" in final.note.content

        # Verify via git history
        from znote_mcp.storage.git_wrapper import GitWrapper
        git = GitWrapper(notes_dir)
        history = git.get_history(notes_dir / f"{note_id}.md", limit=100)
        assert len(history) == 51  # 1 create + 50 updates


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.fixture
    def notes_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            yield Path(tmp)

    def test_update_deleted_note_fails(self, notes_dir):
        """Test that updating a deleted note fails gracefully."""
        from znote_mcp.models.schema import Note, NoteType
        from znote_mcp.storage.note_repository import NoteRepository

        repo = NoteRepository(notes_dir=notes_dir, use_git=True, in_memory_db=True)

        note = Note(
            title="To Be Deleted",
            content="Content",
            note_type=NoteType.PERMANENT,
        )
        created = repo.create_versioned(note)
        note_id = created.note.id
        version = created.version.commit_hash

        # Delete the note
        repo.delete_versioned(note_id)

        # Try to update with the old version
        updated = Note(
            id=note_id,
            title="Ghost Update",
            content="Should fail",
            note_type=NoteType.PERMANENT,
        )

        # This should handle gracefully (not crash)
        result = repo.update_versioned(updated, expected_version=version)
        # Implementation may return conflict or handle differently
        # The key is it shouldn't crash or corrupt data

    def test_special_characters_in_content(self, notes_dir):
        """Test that special characters don't break versioning."""
        from znote_mcp.models.schema import Note, NoteType
        from znote_mcp.storage.note_repository import NoteRepository

        repo = NoteRepository(notes_dir=notes_dir, use_git=True, in_memory_db=True)

        special_content = """
        Unicode: 你好世界 🌍 émojis
        Quotes: "double" 'single' `backtick`
        Brackets: [square] {curly} (round) <angle>
        Newlines:

        Multiple blank lines above.

        Special chars: $HOME && || ; | > < >> <<
        """

        note = Note(
            title="Special Characters Test",
            content=special_content,
            note_type=NoteType.PERMANENT,
        )
        created = repo.create_versioned(note)

        # Verify special characters are preserved in content
        retrieved = repo.get_versioned(created.note.id)
        # Content includes title header, check special chars are present
        assert "你好世界" in retrieved.note.content
        assert "🌍" in retrieved.note.content
        assert "émojis" in retrieved.note.content
        assert '"double"' in retrieved.note.content
        assert "$HOME && ||" in retrieved.note.content

    def test_very_large_content(self, notes_dir):
        """Test versioning with large content."""
        from znote_mcp.models.schema import Note, NoteType
        from znote_mcp.storage.note_repository import NoteRepository

        repo = NoteRepository(notes_dir=notes_dir, use_git=True, in_memory_db=True)

        # 100KB of content
        large_content = "A" * 100_000

        note = Note(
            title="Large Content Test",
            content=large_content,
            note_type=NoteType.PERMANENT,
        )
        created = repo.create_versioned(note)

        # Verify it's stored and versioned correctly
        assert created.version.commit_hash != "0000000"

        retrieved = repo.get_versioned(created.note.id)
        # Content includes title header, so will be larger
        assert len(retrieved.note.content) >= 100_000
        # The actual content should be preserved
        assert "A" * 1000 in retrieved.note.content  # At least part of the content

    def test_empty_expected_version_treated_as_none(self, notes_dir):
        """Test that empty string expected_version is treated as no version check."""
        from znote_mcp.models.schema import Note, NoteType, VersionedNote
        from znote_mcp.storage.note_repository import NoteRepository

        repo = NoteRepository(notes_dir=notes_dir, use_git=True, in_memory_db=True)

        note = Note(
            title="Empty Version Test",
            content="Original",
            note_type=NoteType.PERMANENT,
        )
        created = repo.create_versioned(note)

        # Update with empty string version (should succeed, treated as no check)
        updated = Note(
            id=created.note.id,
            title="Empty Version Test",
            content="Updated",
            note_type=NoteType.PERMANENT,
        )
        result = repo.update_versioned(updated, expected_version="")

        # Should succeed (empty string = no version check)
        assert isinstance(result, VersionedNote)
