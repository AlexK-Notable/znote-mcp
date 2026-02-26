"""Git sync service for remote note synchronization.

Manages sparse checkout of shared repository, push to personal branch,
and import from main. Uses subprocess git (same pattern as GitWrapper)
but manages a DIFFERENT repo (the shared remote in ~/.zettelkasten/.remote/).
"""

import logging
import os
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from znote_mcp.exceptions import ErrorCode, SyncError

logger = logging.getLogger(__name__)


class GitSyncService:
    """Manages remote sync via sparse checkout of a shared repository."""

    _MAX_PUSH_DELAY = 1800  # 30 minutes

    def __init__(
        self,
        user_id: str,
        repo_url: str,
        branch: str,
        remote_dir: Path,
        notes_dir: Path,
        imports_dir: Path,
        push_delay: int = 120,
        push_extend: int = 60,
        import_users: Optional[List[str]] = None,
    ) -> None:
        self._user_id = user_id
        self._repo_url = repo_url
        self._branch = branch
        self._remote_dir = remote_dir
        self._notes_dir = notes_dir
        self._imports_dir = imports_dir
        self._push_delay = push_delay
        self._push_extend = push_extend
        self._import_users = import_users or []
        self._push_lock = threading.Lock()
        self._push_timer: Optional[threading.Timer] = None
        self._pending_writes = 0
        self._first_write_time: Optional[datetime] = None
        self._last_push_time: Optional[datetime] = None
        self._last_import_time: Optional[datetime] = None

    def _run_git(
        self,
        args: List[str],
        cwd: Optional[Path] = None,
        check: bool = True,
        timeout: int = 300,
        retries: int = 3,
        retry_delay: float = 0.1,
    ) -> subprocess.CompletedProcess:
        """Run git command (same pattern as GitWrapper._run_git).

        Args:
            args: Git subcommand and arguments (e.g. ["clone", "--sparse", ...])
            cwd: Working directory for the command. Defaults to remote_dir.
            check: If True, raise SyncError on non-zero exit.
            timeout: Command timeout in seconds (default 300s for network ops).
            retries: Number of retries on transient failures (index.lock).
            retry_delay: Base delay between retries (multiplied by attempt number).

        Returns:
            The completed subprocess result.

        Raises:
            SyncError: On command failure, timeout, or missing git binary.
        """
        work_dir = str(cwd or self._remote_dir)
        cmd = ["git", "-C", work_dir] + args
        env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
        last_error: Optional[SyncError] = None

        for attempt in range(retries + 1):
            try:
                result = subprocess.run(
                    cmd,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    env=env,
                )
                if result.returncode != 0 and result.stderr:
                    if "index.lock" in result.stderr and attempt < retries:
                        time.sleep(retry_delay * (attempt + 1))
                        continue
                if check and result.returncode != 0:
                    raise SyncError(
                        message=f"Git command failed: {' '.join(args)}",
                        operation=" ".join(args[:2]),
                        code=ErrorCode.SYNC_REMOTE_FAILED,
                    )
                return result
            except subprocess.TimeoutExpired:
                last_error = SyncError(
                    message=(
                        f"Git command timed out ({timeout}s): " f"{' '.join(args[:2])}"
                    ),
                    operation=" ".join(args[:2]),
                    code=ErrorCode.SYNC_REMOTE_FAILED,
                )
                if attempt < retries:
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                raise last_error
            except FileNotFoundError:
                raise SyncError(
                    "Git is not installed or not in PATH",
                    operation="git",
                    code=ErrorCode.SYNC_NOT_CONFIGURED,
                )
        if last_error:
            raise last_error
        raise SyncError(
            f"Git failed after {retries} retries",
            operation=" ".join(args[:2]),
        )

    def check_prerequisites(self) -> Dict[str, Any]:
        """Check prerequisites for sync setup.

        Returns:
            Dict with check results: git_installed, repo_reachable, already_setup.
        """
        result: Dict[str, Any] = {
            "git_installed": False,
            "repo_reachable": False,
            "already_setup": self.is_setup,
        }

        # Check git is installed
        try:
            proc = self._run_git(
                ["--version"], cwd=Path("/tmp"), check=False, retries=0
            )
            result["git_installed"] = proc.returncode == 0
            if proc.returncode == 0:
                result["git_version"] = proc.stdout.strip()
        except SyncError:
            result["git_installed"] = False

        if not result["git_installed"]:
            return result

        # Check repo URL is reachable
        try:
            proc = self._run_git(
                ["ls-remote", "--exit-code", self._repo_url, "HEAD"],
                cwd=Path("/tmp"),
                check=False,
                timeout=30,
                retries=0,
            )
            result["repo_reachable"] = proc.returncode == 0
            if proc.returncode != 0:
                result["repo_error"] = proc.stderr.strip()
        except SyncError as e:
            result["repo_reachable"] = False
            result["repo_error"] = str(e)

        return result

    def setup(self) -> None:
        """One-time sparse checkout setup. Idempotent.

        Clones the shared repository with partial clone and sparse checkout,
        configures sparse-checkout patterns for the user's directory and
        any import users, then checks out the personal branch.

        Raises:
            SyncError: If clone or checkout fails.
        """
        if self.is_setup:
            logger.debug("Sparse checkout already set up at %s", self._remote_dir)
            return

        self._remote_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Partial clone with sparse checkout
            self._run_git(
                [
                    "clone",
                    "--filter=blob:none",
                    "--sparse",
                    self._repo_url,
                    str(self._remote_dir),
                ],
                cwd=self._remote_dir.parent,
            )

            # Set sparse checkout patterns
            patterns = [f"notes/{self._user_id}/"]
            for user in self._import_users:
                patterns.append(f"notes/{user}/")
            self._run_git(["sparse-checkout", "set"] + patterns)

            # Set git config for the remote repo
            self._run_git(["config", "user.email", "znote-mcp@localhost"])
            self._run_git(["config", "user.name", "znote-mcp"])
            self._run_git(["config", "core.symlinks", "false"])

            # Checkout or create personal branch
            result = self._run_git(["checkout", self._branch], check=False)
            if result.returncode != 0:
                self._run_git(["checkout", "-b", self._branch])

            logger.info("Sparse checkout set up at %s", self._remote_dir)
        except SyncError:
            raise
        except Exception as e:
            raise SyncError(
                f"Sparse checkout setup failed: {e}",
                operation="setup",
                code=ErrorCode.SYNC_SPARSE_CHECKOUT_FAILED,
                original_error=e,
            )

    @property
    def is_setup(self) -> bool:
        """Check if sparse checkout is initialized."""
        return (self._remote_dir / ".git").exists()

    @property
    def sync_remote_dir(self) -> Path:
        """The sparse checkout directory path."""
        return self._remote_dir

    def get_status(self) -> Dict[str, Any]:
        """Get sync status information."""
        return {
            "enabled": True,
            "user_id": self._user_id,
            "branch": self._branch,
            "is_setup": self.is_setup,
            "last_push_time": (
                self._last_push_time.isoformat() if self._last_push_time else None
            ),
            "last_import_time": (
                self._last_import_time.isoformat() if self._last_import_time else None
            ),
            "pending_writes": self._pending_writes,
            "import_users": self._import_users,
        }

    # =========================================================================
    # Push with Debounce
    # =========================================================================

    def signal_write(self) -> None:
        """Signal that a write occurred. Starts/resets debounce timer.

        The first write starts a timer with ``push_delay``. Subsequent writes
        within the window reset the timer to ``push_extend`` (shorter).  A
        hard cap of 30 minutes from the first write ensures eventual push
        even under continuous writes.
        """
        with self._push_lock:
            self._pending_writes += 1
            now = datetime.now(timezone.utc)
            if self._first_write_time is None:
                self._first_write_time = now

            if self._push_timer is not None:
                self._push_timer.cancel()

            # Check max delay cap
            elapsed = (now - self._first_write_time).total_seconds()
            if elapsed >= self._MAX_PUSH_DELAY:
                # Cap reached, push immediately in background
                self._push_timer = threading.Timer(0.1, self._debounce_push)
            else:
                delay = (
                    self._push_extend if self._pending_writes > 1 else self._push_delay
                )
                remaining = self._MAX_PUSH_DELAY - elapsed
                delay = min(delay, remaining)
                self._push_timer = threading.Timer(delay, self._debounce_push)

            self._push_timer.daemon = True
            self._push_timer.start()

    def flush_push(self) -> bool:
        """Cancel timer, push immediately. Returns True on success."""
        with self._push_lock:
            if self._push_timer is not None:
                self._push_timer.cancel()
                self._push_timer = None
        return self.stage_and_push()

    def stage_and_push(self, force: bool = False) -> bool:
        """Copy notes to staging area, commit, push.

        Returns True on success, never raises.
        """
        if not self.is_setup:
            return False
        try:
            user_staging = self._remote_dir / "notes" / self._user_id
            user_staging.mkdir(parents=True, exist_ok=True)
            self._sync_files(self._notes_dir, user_staging)
            self._run_git(["add", f"notes/{self._user_id}/"])
            # Check for changes
            status = self._run_git(["status", "--porcelain"], check=False)
            if not status.stdout.strip() and not force:
                return True  # Nothing to push
            self._run_git(
                ["commit", "-m", f"Sync notes from {self._user_id}"], check=False
            )
            self._run_git(["push", "origin", self._branch])
            self._last_push_time = datetime.now(timezone.utc)
            with self._push_lock:
                self._pending_writes = 0
                self._first_write_time = None
            logger.info("Pushed to %s", self._branch)
            return True
        except Exception as e:
            logger.error("Push failed: %s", e)
            return False

    def _debounce_push(self) -> None:
        """Called by timer. Performs the actual push."""
        self.stage_and_push()

    def _sync_files(self, src: Path, dst: Path) -> None:
        """Mirror copy src to dst: delete removed, copy new/changed."""
        # Remove files in dst that don't exist in src
        if dst.exists():
            for dst_file in dst.rglob("*.md"):
                rel = dst_file.relative_to(dst)
                if not (src / rel).exists():
                    dst_file.unlink()
        # Copy new/changed files
        for src_file in src.rglob("*.md"):
            rel = src_file.relative_to(src)
            dst_file = dst / rel
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            if (
                not dst_file.exists()
                or src_file.stat().st_mtime > dst_file.stat().st_mtime
            ):
                shutil.copy2(src_file, dst_file)

    # =========================================================================
    # Import from Main
    # =========================================================================

    def pull_imports(self) -> Dict[str, int]:
        """Fetch main, create symlinks for import users' notes directories.

        Returns:
            Dict mapping source_user to count of .md files found.

        Raises:
            SyncError: If sync not set up or git operations fail.
        """
        if not self.is_setup:
            raise SyncError(
                "Sync not set up",
                operation="pull_imports",
                code=ErrorCode.SYNC_NOT_CONFIGURED,
            )
        try:
            self._run_git(["fetch", "origin", "main"])
            results: Dict[str, int] = {}
            for user in self._import_users:
                # Checkout user's notes from main into the sparse checkout
                self._run_git(
                    ["checkout", "origin/main", "--", f"notes/{user}/"],
                    check=False,
                )
                src = self._remote_dir / "notes" / user
                dst = self._imports_dir / user
                if src.exists():
                    # Create or update symlink
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    if dst.is_symlink() or dst.exists():
                        if dst.is_symlink():
                            dst.unlink()
                        else:
                            shutil.rmtree(dst)
                    dst.symlink_to(src)
                    count = len(list(src.rglob("*.md")))
                    results[user] = count
                else:
                    results[user] = 0
            self._last_import_time = datetime.now(timezone.utc)
            return results
        except SyncError:
            raise
        except Exception as e:
            raise SyncError(
                f"Pull imports failed: {e}",
                operation="pull_imports",
                code=ErrorCode.SYNC_PULL_FAILED,
                original_error=e,
            )

    def remove_user(self, user: str) -> Dict[str, Any]:
        """Remove an imported user's notes and cleanup symlinks.

        Args:
            user: Username to remove from imports.

        Returns:
            Dict with cleanup stats (symlink_removed, sparse_updated).
        """
        stats: Dict[str, Any] = {"symlink_removed": False, "sparse_updated": False}

        # Remove symlink (or directory if it's not a symlink)
        import_link = self._imports_dir / user
        if import_link.is_symlink():
            import_link.unlink()
            stats["symlink_removed"] = True
        elif import_link.is_dir():
            shutil.rmtree(import_link)
            stats["symlink_removed"] = True

        # Update sparse checkout patterns to exclude the user
        if user in self._import_users:
            self._import_users.remove(user)
            if self.is_setup:
                patterns = [f"notes/{self._user_id}/"]
                for u in self._import_users:
                    patterns.append(f"notes/{u}/")
                self._run_git(["sparse-checkout", "set"] + patterns, check=False)
                stats["sparse_updated"] = True

        return stats

    def shutdown(self) -> None:
        """Shutdown sync service. Cancels timer, flushes pending writes."""
        with self._push_lock:
            if self._push_timer is not None:
                self._push_timer.cancel()
                self._push_timer = None
            pending = self._pending_writes
        if pending > 0:
            logger.info("Flushing %d pending writes on shutdown", pending)
            self.stage_and_push()
        logger.info("GitSyncService shut down")
