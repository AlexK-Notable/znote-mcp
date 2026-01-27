"""Utility functions for the Zettelkasten MCP server."""
import logging
import sys
from typing import Optional


def escape_like_pattern(value: str) -> str:
    """Escape SQL LIKE wildcards to treat them as literals.

    Prevents SQL LIKE pattern injection where user input containing
    '%' or '_' could match unintended patterns.

    Args:
        value: User input string that may contain LIKE wildcards

    Returns:
        String with '%', '_', and '\\' escaped for safe use in LIKE clauses

    Example:
        >>> escape_like_pattern("100% complete")
        '100\\% complete'
        >>> escape_like_pattern("file_name")
        'file\\_name'
    """
    # Use str.translate() for single-pass efficiency
    escape_table = str.maketrans({
        '\\': '\\\\',  # Escape backslash first
        '%': '\\%',
        '_': '\\_',
    })
    return value.translate(escape_table)


def setup_logging(level: str = "INFO", log_file: Optional[str] = None):
    """Set up logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to a log file
    """
    # Convert string level to logging level
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO

    # Base configuration
    log_config = {
        "level": numeric_level,
        "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        "datefmt": "%Y-%m-%d %H:%M:%S",
    }

    # Add file handler if log file is specified
    if log_file:
        log_config["filename"] = log_file
        log_config["filemode"] = "a"
    else:
        # Otherwise, log to stderr
        log_config["stream"] = sys.stderr

    # Apply configuration
    logging.basicConfig(**log_config)
