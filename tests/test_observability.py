"""Tests for the observability module.

Tests for metrics collection, logging configuration, and error sanitization.
"""
import json
import logging
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from znote_mcp.observability import (
    MetricsCollector,
    _sanitize_error_message,
    configure_logging,
    timed_operation,
)


class TestErrorMessageSanitization:
    """Tests for error message sanitization."""

    def test_sanitize_none_returns_none(self):
        """Sanitizing None should return None."""
        assert _sanitize_error_message(None) is None

    def test_sanitize_simple_message(self):
        """Simple messages should pass through unchanged."""
        assert _sanitize_error_message("Simple error") == "Simple error"

    def test_sanitize_removes_home_directory(self):
        """Home directory paths should be replaced with ~."""
        home = str(Path.home())
        message = f"{home}/secret/file.txt: Permission denied"
        result = _sanitize_error_message(message)
        assert home not in result
        assert "~" in result
        assert "secret/file.txt" in result

    def test_sanitize_removes_newlines(self):
        """Newlines should be replaced with spaces."""
        message = "Line 1\nLine 2\rLine 3"
        result = _sanitize_error_message(message)
        assert "\n" not in result
        assert "\r" not in result
        assert "Line 1 Line 2 Line 3" == result

    def test_sanitize_truncates_long_messages(self):
        """Long messages should be truncated with ellipsis."""
        long_message = "a" * 300
        result = _sanitize_error_message(long_message)
        assert len(result) == 200  # default max length
        assert result.endswith("...")

    def test_sanitize_custom_max_length(self):
        """Custom max length should be respected."""
        message = "a" * 100
        result = _sanitize_error_message(message, max_length=50)
        assert len(result) == 50
        assert result.endswith("...")

    def test_sanitize_collapses_multiple_spaces(self):
        """Multiple consecutive spaces should be collapsed."""
        message = "word1    word2     word3"
        result = _sanitize_error_message(message)
        assert "  " not in result
        assert "word1 word2 word3" == result

    def test_sanitize_strips_whitespace(self):
        """Leading and trailing whitespace should be stripped."""
        message = "  padded message  "
        result = _sanitize_error_message(message)
        assert result == "padded message"


class TestMetricsCollector:
    """Tests for MetricsCollector class."""

    @pytest.fixture
    def temp_metrics_file(self):
        """Create a temporary file for metrics storage."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        yield path
        if path.exists():
            path.unlink()

    @pytest.fixture
    def metrics_collector(self, temp_metrics_file):
        """Create a MetricsCollector with temp file."""
        return MetricsCollector(
            metrics_file=temp_metrics_file,
            auto_save_interval=0  # Disable auto-save
        )

    def test_record_successful_operation(self, metrics_collector):
        """Test recording a successful operation."""
        metrics_collector.record_operation("test_op", 100.0, True)

        metrics = metrics_collector.get_metrics()
        assert "test_op" in metrics
        assert metrics["test_op"]["count"] == 1
        assert metrics["test_op"]["success_count"] == 1
        assert metrics["test_op"]["error_count"] == 0
        assert metrics["test_op"]["avg_duration_ms"] == 100.0

    def test_record_failed_operation(self, metrics_collector):
        """Test recording a failed operation with error."""
        metrics_collector.record_operation("test_op", 50.0, False, "Test error")

        metrics = metrics_collector.get_metrics()
        assert metrics["test_op"]["count"] == 1
        assert metrics["test_op"]["success_count"] == 0
        assert metrics["test_op"]["error_count"] == 1
        assert metrics["test_op"]["last_error"] == "Test error"
        assert metrics["test_op"]["last_error_time"] is not None

    def test_error_message_is_sanitized(self, metrics_collector):
        """Test that error messages are sanitized before storage."""
        home = str(Path.home())
        sensitive_error = f"{home}/private/data.txt: Access denied"

        metrics_collector.record_operation("test_op", 100.0, False, sensitive_error)

        metrics = metrics_collector.get_metrics()
        stored_error = metrics["test_op"]["last_error"]
        assert home not in stored_error
        assert "~" in stored_error

    def test_multiple_operations_aggregated(self, metrics_collector):
        """Test that multiple operations are aggregated correctly."""
        metrics_collector.record_operation("test_op", 100.0, True)
        metrics_collector.record_operation("test_op", 200.0, True)
        metrics_collector.record_operation("test_op", 300.0, False, "Error")

        metrics = metrics_collector.get_metrics()
        assert metrics["test_op"]["count"] == 3
        assert metrics["test_op"]["success_count"] == 2
        assert metrics["test_op"]["error_count"] == 1
        assert metrics["test_op"]["avg_duration_ms"] == 200.0  # (100+200+300)/3
        assert metrics["test_op"]["min_duration_ms"] == 100.0
        assert metrics["test_op"]["max_duration_ms"] == 300.0

    def test_save_and_load_metrics(self, temp_metrics_file):
        """Test saving and loading metrics."""
        # Create and save
        collector1 = MetricsCollector(
            metrics_file=temp_metrics_file,
            auto_save_interval=0
        )
        collector1.record_operation("op1", 100.0, True)
        collector1.record_operation("op2", 200.0, False, "Error")
        collector1.save_metrics()

        # Verify file exists and has content
        assert temp_metrics_file.exists()
        with open(temp_metrics_file) as f:
            data = json.load(f)
        assert "operations" in data
        assert "op1" in data["operations"]
        assert "op2" in data["operations"]

        # Create new collector and verify it loads the data
        collector2 = MetricsCollector(
            metrics_file=temp_metrics_file,
            auto_save_interval=0
        )
        metrics = collector2.get_metrics()
        assert "op1" in metrics
        assert metrics["op1"]["count"] == 1
        assert "op2" in metrics
        assert metrics["op2"]["error_count"] == 1

    def test_get_summary(self, metrics_collector):
        """Test getting metrics summary."""
        metrics_collector.record_operation("op1", 100.0, True)
        metrics_collector.record_operation("op2", 200.0, False, "Error")

        summary = metrics_collector.get_summary()
        assert summary["total_operations"] == 2
        assert summary["total_success"] == 1
        assert summary["total_errors"] == 1
        assert "op1" in summary["operations_tracked"]
        assert "op2" in summary["operations_tracked"]

    def test_reset_metrics(self, metrics_collector):
        """Test resetting all metrics."""
        metrics_collector.record_operation("test_op", 100.0, True)
        assert len(metrics_collector.get_metrics()) == 1

        metrics_collector.reset()
        assert len(metrics_collector.get_metrics()) == 0


class TestTimedOperation:
    """Tests for timed_operation context manager."""

    def test_timed_operation_records_success(self):
        """Test that successful operations are timed and recorded."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            metrics_file = Path(f.name)

        try:
            collector = MetricsCollector(metrics_file=metrics_file, auto_save_interval=0)

            with patch('znote_mcp.observability.metrics', collector):
                with timed_operation("test_op") as op:
                    time.sleep(0.01)  # 10ms
                    op["custom_data"] = "value"

            metrics = collector.get_metrics()
            assert "test_op" in metrics
            assert metrics["test_op"]["success_count"] == 1
            assert metrics["test_op"]["avg_duration_ms"] >= 10  # At least 10ms
        finally:
            metrics_file.unlink()

    def test_timed_operation_records_failure(self):
        """Test that failed operations are recorded with error."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            metrics_file = Path(f.name)

        try:
            collector = MetricsCollector(metrics_file=metrics_file, auto_save_interval=0)

            with patch('znote_mcp.observability.metrics', collector):
                try:
                    with timed_operation("test_op"):
                        raise ValueError("Test error")
                except ValueError:
                    pass

            metrics = collector.get_metrics()
            assert "test_op" in metrics
            assert metrics["test_op"]["error_count"] == 1
            assert "Test error" in metrics["test_op"]["last_error"]
        finally:
            metrics_file.unlink()


class TestConfigureLogging:
    """Tests for configure_logging function."""

    def test_configure_logging_creates_directory(self):
        """Test that configure_logging creates log directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "logs"

            configure_logging(log_dir=log_dir)

            assert log_dir.exists()
            assert log_dir.is_dir()

    def test_configure_logging_returns_path(self):
        """Test that configure_logging returns the log directory path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "logs"

            result = configure_logging(log_dir=log_dir)

            assert result == log_dir

    def test_configure_logging_sets_level(self):
        """Test that configure_logging sets the correct log level."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "logs"

            configure_logging(log_dir=log_dir, level=logging.DEBUG)

            logger = logging.getLogger("zettelkasten")
            assert logger.level == logging.DEBUG
