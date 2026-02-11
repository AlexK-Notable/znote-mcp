"""Git wrapper for version control operations.

Provides subprocess-based git operations for portability and
version checking with conflict detection for optimistic concurrency control.
"""

import logging
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


class GitError(Exception):
    """Base exception for git operations.

    Attributes:
        message: Human-readable error message
        command: The git command that failed (if applicable)
        returncode: Exit code from git (if applicable)
        stderr: Error output from git (if applicable)
    """

    def __init__(
        self,
        message: str,
        command: Optional[List[str]] = None,
        returncode: Optional[int] = None,
        stderr: Optional[str] = None,
    ):
        self.message = message
        self.command = command
        self.returncode = returncode
        self.stderr = stderr
        super().__init__(message)

    def __str__(self) -> str:
        parts = [self.message]
        if self.command:
            parts.append(f"command: {' '.join(self.command)}")
        if self.returncode is not None:
            parts.append(f"returncode: {self.returncode}")
        if self.stderr:
            parts.append(f"stderr: {self.stderr[:200]}")
        return " | ".join(parts)


class GitConflictError(GitError):
    """Raised when version conflict is detected.

    This indicates optimistic concurrency control failure - another
    process modified the file since it was last read.

    Attributes:
        note_id: The ID of the note with the conflict
        expected_version: The version that was expected
        actual_version: The actual current version
    """

    def __init__(
        self, note_id: str, expected_version: str, actual_version: Optional[str]
    ):
        self.note_id = note_id
        self.expected_version = expected_version
        self.actual_version = actual_version

        message = (
            f"Version conflict for note '{note_id}': "
            f"expected {expected_version[:7]}, "
            f"got {actual_version[:7] if actual_version else 'None'}"
        )
        super().__init__(message)


@dataclass
class GitVersion:
    """Represents a git version (commit).

    Attributes:
        commit_hash: Full SHA-1 hash of the commit
        timestamp: UTC datetime of the commit
    """

    commit_hash: str
    timestamp: datetime

    @property
    def short_hash(self) -> str:
        """Return the first 7 characters of the commit hash."""
        return self.commit_hash[:7]

    def __str__(self) -> str:
        return f"{self.short_hash} ({self.timestamp.isoformat()})"


class GitWrapper:
    """Wrapper for git operations via subprocess.

    Provides methods for version control operations on files,
    including commit, delete, and version checking.

    All operations use subprocess to call git for maximum portability.
    The repo_path is passed to git via -C flag for all commands.
    """

    def __init__(self, repo_path: Path):
        """Initialize the GitWrapper.

        Args:
            repo_path: Path to the git repository root.
                      Will be initialized if .git doesn't exist.
        """
        self.repo_path = repo_path.resolve()
        self._ensure_git_repo()

    def _run_git(
        self,
        args: List[str],
        check: bool = True,
        capture_output: bool = True,
        retries: int = 3,
        retry_delay: float = 0.1,
    ) -> subprocess.CompletedProcess:
        """Run a git command via subprocess with retry for lock contention.

        Args:
            args: Git command arguments (without 'git' prefix)
            check: If True, raise GitError on non-zero exit
            capture_output: If True, capture stdout and stderr
            retries: Number of retries for index.lock contention (default: 3)
            retry_delay: Seconds to wait between retries (default: 0.1)

        Returns:
            CompletedProcess with command results

        Raises:
            GitError: If check=True and command fails after all retries
        """
        import time

        cmd = ["git", "-C", str(self.repo_path)] + args
        last_error = None

        for attempt in range(retries + 1):
            try:
                result = subprocess.run(
                    cmd,
                    check=False,
                    capture_output=capture_output,
                    text=True,
                    timeout=30,  # Prevent hanging
                )

                # Check for index.lock contention (can retry)
                if result.returncode != 0 and result.stderr:
                    if "index.lock" in result.stderr and attempt < retries:
                        logger.debug(
                            f"Git index.lock contention, retry {attempt + 1}/{retries}: {args}"
                        )
                        time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                        continue

                if check and result.returncode != 0:
                    raise GitError(
                        message=f"Git command failed: {' '.join(args)}",
                        command=cmd,
                        returncode=result.returncode,
                        stderr=result.stderr.strip() if result.stderr else None,
                    )

                return result

            except subprocess.TimeoutExpired as e:
                last_error = GitError(
                    message=f"Git command timed out: {' '.join(args)}", command=cmd
                )
                if attempt < retries:
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                raise last_error from e
            except FileNotFoundError:
                raise GitError(
                    message="Git is not installed or not in PATH", command=cmd
                )

        # Should not reach here, but just in case
        if last_error:
            raise last_error
        raise GitError(f"Git command failed after {retries} retries: {args}")

    def _ensure_git_repo(self) -> None:
        """Initialize git repo if .git doesn't exist.

        Also sets user.email and user.name for commits.
        """
        git_dir = self.repo_path / ".git"

        if not git_dir.exists():
            logger.info(f"Initializing git repository at {self.repo_path}")
            self.repo_path.mkdir(parents=True, exist_ok=True)
            self._run_git(["init"])

            # Set local git config for commits
            self._run_git(["config", "user.email", "znote-mcp@localhost"])
            self._run_git(["config", "user.name", "znote-mcp"])

            logger.info("Git repository initialized")
        else:
            logger.debug(f"Git repository already exists at {self.repo_path}")

    def get_file_version(self, file_path: Path) -> Optional[GitVersion]:
        """Get the last commit version for a specific file.

        Args:
            file_path: Path to the file (absolute or relative to repo)

        Returns:
            GitVersion for the last commit affecting this file,
            or None if file has no commits
        """
        # Make path relative to repo
        try:
            rel_path = file_path.resolve().relative_to(self.repo_path)
        except ValueError:
            # Path is not under repo_path
            logger.warning(f"File {file_path} is not under repo {self.repo_path}")
            return None

        # Get last commit for this file: format is "full_hash unix_timestamp"
        result = self._run_git(
            ["log", "-1", "--format=%H %ct", "--", str(rel_path)], check=False
        )

        if result.returncode != 0 or not result.stdout.strip():
            return None

        return self._parse_log_line(result.stdout.strip())

    def get_head_version(self) -> Optional[GitVersion]:
        """Get the HEAD commit version.

        Returns:
            GitVersion for HEAD, or None if no commits exist
        """
        result = self._run_git(["log", "-1", "--format=%H %ct"], check=False)

        if result.returncode != 0 or not result.stdout.strip():
            return None

        return self._parse_log_line(result.stdout.strip())

    def _parse_log_line(self, line: str) -> GitVersion:
        """Parse a git log line in format '%H %ct'.

        Args:
            line: Log line with "full_hash unix_timestamp"

        Returns:
            GitVersion object
        """
        parts = line.strip().split()
        commit_hash = parts[0]
        unix_timestamp = int(parts[1])
        timestamp = datetime.fromtimestamp(unix_timestamp, tz=timezone.utc)

        return GitVersion(commit_hash=commit_hash, timestamp=timestamp)

    def check_version_match(
        self, file_path: Path, expected_version: str
    ) -> Tuple[bool, Optional[str]]:
        """Compare expected version with current file version.

        Args:
            file_path: Path to the file to check
            expected_version: Expected commit hash (short or full)

        Returns:
            Tuple of (matches, actual_version) where:
            - matches: True if versions match
            - actual_version: Current version hash or None
        """
        current = self.get_file_version(file_path)
        actual_version = current.commit_hash if current else None

        if actual_version is None:
            # File has no commits - only matches if expected is empty/None
            return (not expected_version), None

        # Compare using the length of expected_version
        # This allows matching short hashes (7 chars) against full hashes
        expected_len = len(expected_version)
        matches = actual_version[:expected_len] == expected_version[:expected_len]

        return matches, actual_version

    def commit_file(
        self, file_path: Path, message: str, expected_version: Optional[str] = None
    ) -> GitVersion:
        """Stage and commit a file with optional version checking.

        Args:
            file_path: Path to the file to commit
            message: Commit message
            expected_version: If provided, check that current version matches
                            before committing (optimistic concurrency)

        Returns:
            GitVersion for the new commit

        Raises:
            GitConflictError: If expected_version provided and doesn't match
            GitError: If commit fails
        """
        # Make path relative to repo
        try:
            rel_path = file_path.resolve().relative_to(self.repo_path)
        except ValueError:
            raise GitError(f"File {file_path} is not under repo {self.repo_path}")

        # Check version if expected_version provided
        if expected_version:
            matches, actual = self.check_version_match(file_path, expected_version)
            if not matches:
                # Extract note_id from filename (without extension)
                note_id = file_path.stem
                raise GitConflictError(
                    note_id=note_id,
                    expected_version=expected_version,
                    actual_version=actual,
                )

        # Stage the file
        self._run_git(["add", str(rel_path)])

        # Check if there are changes to commit
        status_result = self._run_git(["status", "--porcelain", str(rel_path)])
        if not status_result.stdout.strip():
            # No changes to commit - check if version changed (conflict)
            current = self.get_file_version(file_path)
            if current:
                if expected_version:
                    # Version changed since our check - this is a conflict
                    if (
                        current.commit_hash[: len(expected_version)]
                        != expected_version[: len(expected_version)]
                    ):
                        note_id = file_path.stem
                        raise GitConflictError(
                            note_id=note_id,
                            expected_version=expected_version,
                            actual_version=current.commit_hash,
                        )
                return current
            # File exists but has no commits - create initial commit
            # This shouldn't happen after add, but handle gracefully

        # Try to commit - may fail if another process commits first
        commit_result = self._run_git(
            ["commit", "-m", message, "--", str(rel_path)],
            check=False,  # Don't raise on error, we'll handle it
        )

        if commit_result.returncode != 0:
            # Commit failed - check if it's because another process committed first
            # This manifests as "nothing to commit" or similar
            current = self.get_file_version(file_path)
            if current:
                if expected_version:
                    # Check if version changed (conflict from another process)
                    if (
                        current.commit_hash[: len(expected_version)]
                        != expected_version[: len(expected_version)]
                    ):
                        note_id = file_path.stem
                        raise GitConflictError(
                            note_id=note_id,
                            expected_version=expected_version,
                            actual_version=current.commit_hash,
                        )
                # Version didn't change or no expected_version - return current
                return current

            # Commit failed for some other reason
            raise GitError(
                message=f"Git commit failed: {commit_result.stderr or 'unknown error'}",
                command=["commit", "-m", message, "--", str(rel_path)],
                returncode=commit_result.returncode,
                stderr=commit_result.stderr,
            )

        # Get and return the new version
        new_version = self.get_file_version(file_path)
        if not new_version:
            raise GitError(f"Failed to get version after commit for {file_path}")

        logger.debug(f"Committed {rel_path}: {new_version.short_hash}")
        return new_version

    def delete_file(
        self, file_path: Path, message: str, expected_version: Optional[str] = None
    ) -> GitVersion:
        """Remove a file and commit the deletion.

        Args:
            file_path: Path to the file to delete
            message: Commit message
            expected_version: If provided, check that current version matches

        Returns:
            GitVersion for the delete commit

        Raises:
            GitConflictError: If expected_version provided and doesn't match
            GitError: If delete fails
        """
        # Make path relative to repo
        try:
            rel_path = file_path.resolve().relative_to(self.repo_path)
        except ValueError:
            raise GitError(f"File {file_path} is not under repo {self.repo_path}")

        # Check version if expected_version provided
        if expected_version:
            matches, actual = self.check_version_match(file_path, expected_version)
            if not matches:
                note_id = file_path.stem
                raise GitConflictError(
                    note_id=note_id,
                    expected_version=expected_version,
                    actual_version=actual,
                )

        # Remove file from git (also deletes from working tree)
        self._run_git(["rm", str(rel_path)])

        # Commit the deletion
        self._run_git(["commit", "-m", message])

        # Get HEAD version (since file no longer exists)
        head_version = self.get_head_version()
        if not head_version:
            raise GitError("Failed to get HEAD version after delete")

        logger.debug(f"Deleted {rel_path}: {head_version.short_hash}")
        return head_version

    def get_history(self, file_path: Path, limit: int = 10) -> List[GitVersion]:
        """Get commit history for a file.

        Args:
            file_path: Path to the file
            limit: Maximum number of commits to return

        Returns:
            List of GitVersion objects, most recent first
        """
        # Make path relative to repo
        try:
            rel_path = file_path.resolve().relative_to(self.repo_path)
        except ValueError:
            logger.warning(f"File {file_path} is not under repo {self.repo_path}")
            return []

        result = self._run_git(
            ["log", f"-{limit}", "--format=%H %ct", "--", str(rel_path)], check=False
        )

        if result.returncode != 0 or not result.stdout.strip():
            return []

        versions = []
        for line in result.stdout.strip().split("\n"):
            if line.strip():
                versions.append(self._parse_log_line(line))

        return versions
