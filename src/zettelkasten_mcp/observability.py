"""Observability utilities for the Zettelkasten MCP server.

Provides structured logging, timing metrics, and operation tracking.
"""
import functools
import logging
import time
import uuid
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, TypeVar

logger = logging.getLogger(__name__)

# Type variable for decorators
F = TypeVar('F', bound=Callable[..., Any])


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
    """

    def __init__(self):
        self._metrics: Dict[str, OperationMetrics] = defaultdict(OperationMetrics)
        self._lock = Lock()
        self._start_time = datetime.now()

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
                m.last_error_time = datetime.now()

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
                'uptime_seconds': (datetime.now() - self._start_time).total_seconds(),
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
            self._start_time = datetime.now()


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
        'timestamp': datetime.now().isoformat(),
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
