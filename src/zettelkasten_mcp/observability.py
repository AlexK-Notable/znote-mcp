"""Observability utilities for the Zettelkasten MCP server.

Provides structured logging, timing metrics, operation tracking,
and persistent disk logging with rotation.
"""
import functools
import json
import logging
import os
import time
import uuid
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

logger = logging.getLogger(__name__)

# Default log directory (can be overridden via configure_logging)
DEFAULT_LOG_DIR = Path.home() / ".zettelkasten" / "logs"
DEFAULT_METRICS_FILE = Path.home() / ".zettelkasten" / "metrics.json"

# Logging format with ISO 8601 timestamps
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S%z"

# Type variable for decorators
F = TypeVar('F', bound=Callable[..., Any])

# Global flag to track if logging has been configured
_logging_configured = False


def configure_logging(
    log_dir: Optional[Union[str, Path]] = None,
    level: int = logging.INFO,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB per file
    backup_count: int = 5,
    console: bool = True,
) -> Path:
    """Configure persistent file logging with rotation.

    Sets up rotating file handler for the zettelkasten logger hierarchy.
    Log files are rotated when they reach max_bytes, keeping backup_count old files.

    Args:
        log_dir: Directory for log files. Defaults to ~/.zettelkasten/logs/
        level: Logging level (default: INFO)
        max_bytes: Maximum size per log file before rotation (default: 10 MB)
        backup_count: Number of rotated files to keep (default: 5)
        console: Also log to console (default: True)

    Returns:
        Path to the log directory

    Example:
        configure_logging(level=logging.DEBUG, backup_count=10)
    """
    global _logging_configured

    log_path = Path(log_dir) if log_dir else DEFAULT_LOG_DIR
    log_path.mkdir(parents=True, exist_ok=True)

    # Get root zettelkasten logger
    root_logger = logging.getLogger("zettelkasten")
    root_logger.setLevel(level)

    # Also configure the module logger
    module_logger = logging.getLogger(__name__)
    module_logger.setLevel(level)

    # Create formatter
    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    # Add rotating file handler
    log_file = log_path / "zettelkasten.log"
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Optionally add console handler
    if console and not any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler)
        for h in root_logger.handlers
    ):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    _logging_configured = True
    root_logger.info(f"Logging configured: {log_file} (max {max_bytes} bytes, {backup_count} backups)")

    return log_path


def is_logging_configured() -> bool:
    """Check if file logging has been configured."""
    return _logging_configured


@dataclass
class OperationMetrics:
    """Metrics for a single operation type."""
    count: int = 0
    success_count: int = 0
    error_count: int = 0
    total_duration_ms: float = 0.0
    min_duration_ms: float = float('inf')
    max_duration_ms: float = 0.0
    last_error: Optional[str] = None
    last_error_time: Optional[datetime] = None


class MetricsCollector:
    """Thread-safe metrics collection for server operations.

    Collects timing, success/failure rates, and error information
    for each operation type (create_note, search_notes, etc.).

    Supports persistence to disk with auto-save on significant events.
    """

    def __init__(
        self,
        metrics_file: Optional[Union[str, Path]] = None,
        auto_save_interval: int = 100,
    ):
        """Initialize the metrics collector.

        Args:
            metrics_file: Path to persist metrics. Defaults to ~/.zettelkasten/metrics.json
            auto_save_interval: Save to disk every N operations (0 to disable)
        """
        self._metrics: Dict[str, OperationMetrics] = defaultdict(OperationMetrics)
        self._lock = Lock()
        self._start_time = datetime.now(timezone.utc)
        self._metrics_file = Path(metrics_file) if metrics_file else DEFAULT_METRICS_FILE
        self._auto_save_interval = auto_save_interval
        self._operation_count_since_save = 0

        # Try to load existing metrics on startup
        self._load_metrics()

    def record_operation(
        self,
        operation: str,
        duration_ms: float,
        success: bool,
        error: Optional[str] = None
    ) -> None:
        """Record metrics for an operation.

        Args:
            operation: The operation name (e.g., 'create_note', 'search_notes')
            duration_ms: Duration in milliseconds
            success: Whether the operation succeeded
            error: Error message if the operation failed
        """
        with self._lock:
            m = self._metrics[operation]
            m.count += 1
            m.total_duration_ms += duration_ms
            m.min_duration_ms = min(m.min_duration_ms, duration_ms)
            m.max_duration_ms = max(m.max_duration_ms, duration_ms)

            if success:
                m.success_count += 1
            else:
                m.error_count += 1
                m.last_error = error
                m.last_error_time = datetime.now(timezone.utc)

            # Auto-save periodically
            self._operation_count_since_save += 1
            if (
                self._auto_save_interval > 0
                and self._operation_count_since_save >= self._auto_save_interval
            ):
                self._save_metrics_unlocked()

    def get_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get a snapshot of all metrics.

        Returns:
            Dictionary mapping operation names to their metrics.
        """
        with self._lock:
            result = {}
            for op, m in self._metrics.items():
                avg_duration = m.total_duration_ms / m.count if m.count > 0 else 0
                # Handle case where no operations have completed
                min_dur = m.min_duration_ms if m.min_duration_ms != float('inf') else 0
                result[op] = {
                    'count': m.count,
                    'success_count': m.success_count,
                    'error_count': m.error_count,
                    'success_rate': m.success_count / m.count if m.count > 0 else 0,
                    'avg_duration_ms': round(avg_duration, 2),
                    'min_duration_ms': round(min_dur, 2),
                    'max_duration_ms': round(m.max_duration_ms, 2),
                    'last_error': m.last_error,
                    'last_error_time': m.last_error_time.isoformat() if m.last_error_time else None
                }
            return result

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of overall server health.

        Returns:
            Dictionary with aggregate statistics.
        """
        with self._lock:
            total_ops = sum(m.count for m in self._metrics.values())
            total_success = sum(m.success_count for m in self._metrics.values())
            total_errors = sum(m.error_count for m in self._metrics.values())

            return {
                'uptime_seconds': (datetime.now(timezone.utc) - self._start_time).total_seconds(),
                'total_operations': total_ops,
                'total_success': total_success,
                'total_errors': total_errors,
                'overall_success_rate': total_success / total_ops if total_ops > 0 else 1.0,
                'operations_tracked': list(self._metrics.keys())
            }

    def reset(self) -> None:
        """Reset all metrics (useful for testing)."""
        with self._lock:
            self._metrics.clear()
            self._start_time = datetime.now(timezone.utc)
            self._operation_count_since_save = 0

    def _load_metrics(self) -> bool:
        """Load metrics from disk (called internally during init).

        Returns:
            True if metrics were loaded successfully, False otherwise.
        """
        try:
            if not self._metrics_file.exists():
                return False

            with open(self._metrics_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Restore start time
            if "start_time" in data:
                self._start_time = datetime.fromisoformat(data["start_time"])

            # Restore operation metrics
            for op_name, op_data in data.get("operations", {}).items():
                m = self._metrics[op_name]
                m.count = op_data.get("count", 0)
                m.success_count = op_data.get("success_count", 0)
                m.error_count = op_data.get("error_count", 0)
                m.total_duration_ms = op_data.get("total_duration_ms", 0.0)
                m.min_duration_ms = op_data.get("min_duration_ms", float("inf"))
                m.max_duration_ms = op_data.get("max_duration_ms", 0.0)
                m.last_error = op_data.get("last_error")
                if op_data.get("last_error_time"):
                    m.last_error_time = datetime.fromisoformat(op_data["last_error_time"])

            logger.debug(f"Loaded metrics from {self._metrics_file}")
            return True

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to load metrics from {self._metrics_file}: {e}")
            return False

    def _save_metrics_unlocked(self) -> bool:
        """Save metrics to disk (must be called with lock held).

        Returns:
            True if saved successfully, False otherwise.
        """
        try:
            # Ensure parent directory exists
            self._metrics_file.parent.mkdir(parents=True, exist_ok=True)

            # Build serializable data
            data = {
                "start_time": self._start_time.isoformat(),
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "operations": {},
            }

            for op_name, m in self._metrics.items():
                data["operations"][op_name] = {
                    "count": m.count,
                    "success_count": m.success_count,
                    "error_count": m.error_count,
                    "total_duration_ms": m.total_duration_ms,
                    "min_duration_ms": m.min_duration_ms if m.min_duration_ms != float("inf") else None,
                    "max_duration_ms": m.max_duration_ms,
                    "last_error": m.last_error,
                    "last_error_time": m.last_error_time.isoformat() if m.last_error_time else None,
                }

            # Atomic write via temp file
            temp_file = self._metrics_file.with_suffix(".tmp")
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            temp_file.rename(self._metrics_file)
            self._operation_count_since_save = 0
            return True

        except (OSError, TypeError) as e:
            logger.error(f"Failed to save metrics to {self._metrics_file}: {e}")
            return False

    def save_metrics(self) -> bool:
        """Explicitly save metrics to disk.

        Returns:
            True if saved successfully, False otherwise.

        Example:
            # Save before shutdown
            metrics.save_metrics()
        """
        with self._lock:
            return self._save_metrics_unlocked()

    def get_metrics_file(self) -> Path:
        """Get the path to the metrics file."""
        return self._metrics_file


# Global metrics collector instance
metrics = MetricsCollector()


@contextmanager
def timed_operation(operation: str, **context):
    """Context manager for timing and logging operations.

    Args:
        operation: Name of the operation being performed
        **context: Additional context to include in log messages

    Yields:
        A dictionary where you can store result info (e.g., result_count)

    Example:
        with timed_operation('search_notes', query='test') as op:
            results = do_search()
            op['result_count'] = len(results)
    """
    correlation_id = str(uuid.uuid4())[:8]
    start_time = time.perf_counter()
    result_info: Dict[str, Any] = {'correlation_id': correlation_id}

    # Log start
    context_str = ', '.join(f'{k}={v}' for k, v in context.items())
    logger.debug(f"[{correlation_id}] START {operation} ({context_str})")

    error_msg = None
    success = True

    try:
        yield result_info
    except Exception as e:
        success = False
        error_msg = str(e)
        raise
    finally:
        duration_ms = (time.perf_counter() - start_time) * 1000
        metrics.record_operation(operation, duration_ms, success, error_msg)

        # Log completion
        result_str = ', '.join(f'{k}={v}' for k, v in result_info.items() if k != 'correlation_id')
        status = 'OK' if success else f'ERROR: {error_msg}'
        logger.debug(
            f"[{correlation_id}] END {operation} "
            f"({duration_ms:.2f}ms) [{status}] {result_str}"
        )


def traced(operation_name: Optional[str] = None) -> Callable[[F], F]:
    """Decorator for automatic operation tracing.

    Automatically times the function, records metrics, and logs
    start/end with correlation ID.

    Args:
        operation_name: Name to use for the operation. If None, uses function name.

    Example:
        @traced('create_note')
        def create_note(self, title: str, content: str) -> Note:
            ...
    """
    def decorator(func: F) -> F:
        op_name = operation_name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Extract useful context from args/kwargs for logging
            context = {}
            if 'note_id' in kwargs:
                context['note_id'] = kwargs['note_id']
            elif 'title' in kwargs:
                context['title'] = kwargs['title'][:50] if kwargs['title'] else None

            with timed_operation(op_name, **context) as op:
                result = func(*args, **kwargs)
                # Try to extract useful result info
                if hasattr(result, '__len__'):
                    op['result_count'] = len(result)
                elif result is not None:
                    op['has_result'] = True
                return result

        return wrapper  # type: ignore
    return decorator


def log_context(**context) -> Dict[str, Any]:
    """Create a structured logging context.

    Args:
        **context: Key-value pairs to include in the context

    Returns:
        Dictionary suitable for structured logging

    Example:
        logger.info("Note created", extra=log_context(note_id=note.id, title=note.title))
    """
    return {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        **context
    }


class StructuredLogger:
    """Logger adapter that adds structured context to all messages.

    Wraps the standard logging module to automatically include
    component name and optional persistent context.
    """

    def __init__(self, component: str):
        """Initialize with a component name.

        Args:
            component: Name of the component (e.g., 'mcp_server', 'repository')
        """
        self._logger = logging.getLogger(f"zettelkasten.{component}")
        self._component = component
        self._context: Dict[str, Any] = {}

    def set_context(self, **context) -> None:
        """Set persistent context that will be included in all log messages."""
        self._context.update(context)

    def clear_context(self) -> None:
        """Clear the persistent context."""
        self._context.clear()

    def _format_message(self, msg: str, **extra) -> str:
        """Format message with context."""
        all_context = {**self._context, **extra}
        if all_context:
            ctx_str = ' '.join(f'{k}={v}' for k, v in all_context.items())
            return f"[{self._component}] {msg} | {ctx_str}"
        return f"[{self._component}] {msg}"

    def debug(self, msg: str, **extra) -> None:
        self._logger.debug(self._format_message(msg, **extra))

    def info(self, msg: str, **extra) -> None:
        self._logger.info(self._format_message(msg, **extra))

    def warning(self, msg: str, **extra) -> None:
        self._logger.warning(self._format_message(msg, **extra))

    def error(self, msg: str, exc_info: bool = False, **extra) -> None:
        self._logger.error(self._format_message(msg, **extra), exc_info=exc_info)


def get_logger(component: str) -> StructuredLogger:
    """Get a structured logger for a component.

    Args:
        component: Component name (e.g., 'mcp_server', 'repository')

    Returns:
        A StructuredLogger instance
    """
    return StructuredLogger(component)
