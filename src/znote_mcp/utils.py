"""Utility functions for the Zettelkasten MCP server."""
import logging
import sys
from typing import Optional


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
