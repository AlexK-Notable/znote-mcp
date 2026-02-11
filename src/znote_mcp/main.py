#!/usr/bin/env python
"""Main entry point for the Zettelkasten MCP server."""
import argparse
import atexit
import logging
import os
import sys
from pathlib import Path

from znote_mcp import __version__
from znote_mcp.config import config
from znote_mcp.models.db_models import init_db
from znote_mcp.observability import configure_logging, metrics
from znote_mcp.server.mcp_server import ZettelkastenMcpServer
from znote_mcp.setup_manager import ensure_semantic_deps, warmup_models_background


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Zettelkasten MCP Server")
    parser.add_argument(
        "--notes-dir",
        help="Directory for storing note files",
        type=str,
        default=os.environ.get("ZETTELKASTEN_NOTES_DIR")
    )
    parser.add_argument(
        "--database-path",
        help="SQLite database file path",
        type=str,
        default=os.environ.get("ZETTELKASTEN_DATABASE_PATH")
    )
    parser.add_argument(
        "--log-level",
        help="Logging level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=os.environ.get("ZETTELKASTEN_LOG_LEVEL", "INFO")
    )
    return parser.parse_args()


def update_config(args):
    """Update the global config with command line arguments."""
    if args.notes_dir:
        config.notes_dir = Path(args.notes_dir)
    if args.database_path:
        config.database_path = Path(args.database_path)


def _save_metrics_on_exit():
    """Save metrics to disk on server shutdown."""
    try:
        if metrics.save_metrics():
            logging.getLogger(__name__).info("Metrics saved to disk on shutdown")
    except Exception as e:
        logging.getLogger(__name__).warning(f"Failed to save metrics on shutdown: {e}")


def main():
    """Run the Zettelkasten MCP server."""
    # Parse arguments and update config
    args = parse_args()
    update_config(args)

    # Configure logging (console + persistent file logging with rotation)
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    try:
        log_dir = configure_logging(level=log_level, console=True)
    except Exception as e:
        # Fall back to basic console logging if file logging fails
        logging.basicConfig(level=log_level)
        logging.getLogger(__name__).warning(f"Failed to configure file logging: {e}")
        log_dir = None

    logger = logging.getLogger(__name__)
    if log_dir:
        logger.info(f"Persistent logging enabled: {log_dir}")

    # Register metrics save on shutdown
    atexit.register(_save_metrics_on_exit)

    # Auto-install semantic search dependencies if needed
    project_root = Path(__file__).resolve().parent.parent.parent
    deps_ok = ensure_semantic_deps(project_root, __version__)
    logger.info("Semantic deps available: %s", deps_ok)

    # Auto-enable embeddings when deps are available and user hasn't explicitly set the env var
    if deps_ok and not os.getenv("ZETTELKASTEN_EMBEDDINGS_ENABLED"):
        config.embeddings_enabled = True

    # Pre-download embedding models in background so first search is fast
    if deps_ok and config.embeddings_enabled:
        warmup_models_background(
            venv_dir=project_root / ".venv",
            version=__version__,
            embedding_model=config.embedding_model,
            reranker_model=config.reranker_model,
            cache_dir=config.embedding_model_cache_dir,
        )

    # Ensure directories exist
    notes_dir = config.get_absolute_path(config.notes_dir)
    notes_dir.mkdir(parents=True, exist_ok=True)
    db_dir = config.get_absolute_path(config.database_path).parent
    db_dir.mkdir(parents=True, exist_ok=True)

    # Initialize database schema — single engine shared by all repositories
    try:
        logger.info(f"Using SQLite database: {config.get_db_url()}")
        engine = init_db()
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        sys.exit(1)

    # Create and run the MCP server with shared engine
    try:
        logger.info("Starting Zettelkasten MCP server")
        server = ZettelkastenMcpServer(engine=engine)
        server.run()
    except Exception as e:
        logger.error(f"Error running server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
