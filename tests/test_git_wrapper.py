"""Tests for the GitWrapper module."""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from znote_mcp.storage.git_wrapper import (
    GitConflictError,
    GitError,
    GitVersion,
    GitWrapper,
)


class TestGitWrapper:
    """Tests for GitWrapper class."""

    @pytest.fixture
    def git_dir(self):
        """Create a temporary directory for git tests."""
        with tempfile.TemporaryDirectory() as tmp:
            yield Path(tmp)

    @pytest.fixture
    def git_wrapper(self, git_dir):
        """Create a GitWrapper instance."""
        return GitWrapper(git_dir)

    def test_init_creates_git_repo(self, git_dir):
        """Test that GitWrapper initializes a git repository."""
        wrapper = GitWrapper(git_dir)
        assert (git_dir / ".git").exists()
        assert (git_dir / ".git").is_dir()

    def test_init_idempotent(self, git_dir):
        """Test that initializing twice doesn't fail."""
        wrapper1 = GitWrapper(git_dir)
        wrapper2 = GitWrapper(git_dir)
        assert (git_dir / ".git").exists()

    def test_commit_file_creates_commit(self, git_wrapper, git_dir):
        """Test committing a file creates a git commit."""
        test_file = git_dir / "test.md"
        test_file.write_text("# Test\n\nContent here.")

        version = git_wrapper.commit_file(test_file, "Create test file")

        assert version is not None
        assert len(version.commit_hash) >= 7
        assert isinstance(version.timestamp, datetime)

    def test_commit_file_returns_different_hashes(self, git_wrapper, git_dir):
        """Test that different commits have different hashes."""
        test_file = git_dir / "test.md"

        test_file.write_text("Version 1")
        version1 = git_wrapper.commit_file(test_file, "Version 1")

        test_file.write_text("Version 2")
        version2 = git_wrapper.commit_file(test_file, "Version 2")

        assert version1.commit_hash != version2.commit_hash

    def test_get_file_version(self, git_wrapper, git_dir):
        """Test retrieving version info for a file."""
        test_file = git_dir / "test.md"
        test_file.write_text("Content")
        expected_version = git_wrapper.commit_file(test_file, "Initial commit")

        actual_version = git_wrapper.get_file_version(test_file)

        assert actual_version is not None
        assert actual_version.commit_hash == expected_version.commit_hash

    def test_get_file_version_untracked(self, git_wrapper, git_dir):
        """Test getting version of untracked file returns None."""
        test_file = git_dir / "untracked.md"
        test_file.write_text("Not committed")

        version = git_wrapper.get_file_version(test_file)
        assert version is None

    def test_check_version_match_success(self, git_wrapper, git_dir):
        """Test version check succeeds when versions match."""
        test_file = git_dir / "test.md"
        test_file.write_text("Content")
        version = git_wrapper.commit_file(test_file, "Create file")

        matches, actual = git_wrapper.check_version_match(
            test_file, version.commit_hash
        )

        assert matches is True
        assert actual == version.commit_hash

    def test_check_version_match_failure(self, git_wrapper, git_dir):
        """Test version check fails when versions differ."""
        test_file = git_dir / "test.md"
        test_file.write_text("Version 1")
        version1 = git_wrapper.commit_file(test_file, "Version 1")

        test_file.write_text("Version 2")
        version2 = git_wrapper.commit_file(test_file, "Version 2")

        matches, actual = git_wrapper.check_version_match(
            test_file, version1.commit_hash
        )

        assert matches is False
        assert actual == version2.commit_hash

    def test_delete_file_commits_deletion(self, git_wrapper, git_dir):
        """Test deleting a file creates a commit."""
        test_file = git_dir / "test.md"
        test_file.write_text("To be deleted")
        git_wrapper.commit_file(test_file, "Create file")

        version = git_wrapper.delete_file(test_file, "Delete file")

        assert version is not None
        assert not test_file.exists()

    def test_delete_file_with_version_check(self, git_wrapper, git_dir):
        """Test delete with version check succeeds when version matches."""
        test_file = git_dir / "test.md"
        test_file.write_text("To be deleted")
        version = git_wrapper.commit_file(test_file, "Create file")

        delete_version = git_wrapper.delete_file(
            test_file, "Delete file", expected_version=version.commit_hash
        )

        assert delete_version is not None
        assert not test_file.exists()

    def test_delete_file_version_conflict(self, git_wrapper, git_dir):
        """Test delete with wrong version raises GitConflictError."""
        test_file = git_dir / "test.md"
        test_file.write_text("Version 1")
        version1 = git_wrapper.commit_file(test_file, "Version 1")

        test_file.write_text("Version 2")
        git_wrapper.commit_file(test_file, "Version 2")

        with pytest.raises(GitConflictError) as exc:
            git_wrapper.delete_file(
                test_file, "Delete file", expected_version=version1.commit_hash
            )

        assert exc.value.expected_version == version1.commit_hash

    def test_get_history(self, git_wrapper, git_dir):
        """Test getting commit history for a file."""
        test_file = git_dir / "test.md"

        test_file.write_text("V1")
        git_wrapper.commit_file(test_file, "Version 1")

        test_file.write_text("V2")
        git_wrapper.commit_file(test_file, "Version 2")

        test_file.write_text("V3")
        git_wrapper.commit_file(test_file, "Version 3")

        history = git_wrapper.get_history(test_file, limit=10)

        assert len(history) == 3
        assert all(isinstance(v, GitVersion) for v in history)

    def test_get_history_limit(self, git_wrapper, git_dir):
        """Test history limit is respected."""
        test_file = git_dir / "test.md"

        for i in range(5):
            test_file.write_text(f"Version {i}")
            git_wrapper.commit_file(test_file, f"Version {i}")

        history = git_wrapper.get_history(test_file, limit=2)

        assert len(history) == 2


class TestGitVersion:
    """Tests for GitVersion dataclass."""

    def test_git_version_creation(self):
        """Test creating a GitVersion."""
        ts = datetime.now()
        version = GitVersion(commit_hash="abc1234", timestamp=ts)
        assert version.commit_hash == "abc1234"
        assert version.timestamp == ts

    def test_git_version_short_hash(self):
        """Test short_hash property."""
        version = GitVersion(commit_hash="abc123456789", timestamp=datetime.now())
        assert version.short_hash == "abc1234"


class TestGitConflictError:
    """Tests for GitConflictError exception."""

    def test_conflict_error_message(self):
        """Test conflict error contains version info."""
        error = GitConflictError(
            "Version mismatch", expected_version="abc1234", actual_version="def5678"
        )
        assert "abc1234" in str(error)
        assert error.expected_version == "abc1234"
        assert error.actual_version == "def5678"
