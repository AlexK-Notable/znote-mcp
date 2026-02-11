"""FTS5 full-text search index for Zettelkasten notes.

Encapsulates FTS5 querying, graceful degradation, and recovery logic.
Extracted from NoteRepository for cohesion.
"""
import logging
import re
import sqlite3
from typing import Any, Callable, Dict, List, Optional

from sqlalchemy import text
from sqlalchemy.exc import DatabaseError as SQLAlchemyDatabaseError
from sqlalchemy.exc import OperationalError as SQLAlchemyOperationalError

from znote_mcp.exceptions import ErrorCode, SearchError
from znote_mcp.models.db_models import rebuild_fts_index
from znote_mcp.utils import escape_like_pattern

logger = logging.getLogger(__name__)


class FtsIndex:
    """FTS5 full-text search index with graceful degradation.

    Args:
        engine: SQLAlchemy engine used for database access.
        session_factory: Callable returning a context-manager session.
    """

    def __init__(
        self,
        engine: Any,
        session_factory: Callable,
    ) -> None:
        self.engine = engine
        self._session_factory = session_factory
        self.available: bool = True

    # ------------------------------------------------------------------
    # Public query API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        limit: int = 50,
        highlight: bool = False,
        literal: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """Full-text search using FTS5 with graceful fallback.

        Args:
            query: Search query (supports FTS5 syntax).
            limit: Maximum results.
            highlight: Include highlighted snippets.
            literal: None = auto-detect, True = escape, False = preserve syntax.

        Returns:
            List of result dicts (id, title, rank, optionally snippet, search_mode).
        """
        if not self.available:
            logger.debug("FTS5 unavailable, using fallback search")
            return self._fallback_text_search(query, limit)

        if literal is None:
            literal = self._should_escape(query)

        if literal:
            safe_query = self._escape_query(query)
        else:
            safe_query = query.replace('"', '""')

        if highlight:
            sql = text("""
                SELECT
                    id, title,
                    bm25(notes_fts) as rank,
                    snippet(notes_fts, 2, '<mark>', '</mark>', '...', 32) as snippet
                FROM notes_fts
                WHERE notes_fts MATCH :query
                ORDER BY rank
                LIMIT :limit
            """)
        else:
            sql = text("""
                SELECT id, title, bm25(notes_fts) as rank
                FROM notes_fts
                WHERE notes_fts MATCH :query
                ORDER BY rank
                LIMIT :limit
            """)

        results: List[Dict[str, Any]] = []
        with self._session_factory() as session:
            try:
                result = session.execute(sql, {"query": safe_query, "limit": limit})
                for row in result.fetchall():
                    entry: Dict[str, Any] = {
                        "id": row[0],
                        "title": row[1],
                        "rank": row[2],
                        "search_mode": "fts5",
                    }
                    if highlight and len(row) > 3:
                        entry["snippet"] = row[3]
                    results.append(entry)

            except (sqlite3.OperationalError, SQLAlchemyOperationalError) as e:
                logger.warning(
                    f"FTS5 query failed for '{query}': {e}. Using fallback search."
                )
                return self._fallback_text_search(query, limit)

            except (sqlite3.DatabaseError, SQLAlchemyDatabaseError) as e:
                error_msg = str(e).lower()
                if "malformed" in error_msg or "corrupt" in error_msg:
                    logger.error(
                        f"FTS5 corruption detected: {e}. Attempting auto-rebuild..."
                    )
                    if self._attempt_recovery():
                        logger.info("FTS5 rebuilt successfully, retrying search")
                        return self.search(query, limit, highlight)
                    else:
                        logger.error(
                            "FTS5 recovery failed. Disabling FTS5 for this session."
                        )
                        self.available = False
                        return self._fallback_text_search(query, limit)
                else:
                    logger.error(f"FTS5 database error: {e}. Using fallback search.")
                    return self._fallback_text_search(query, limit)

        return results

    def rebuild(self) -> int:
        """Rebuild the FTS5 index from the notes table."""
        return rebuild_fts_index(self.engine)

    def reset_availability(self) -> bool:
        """Re-enable FTS5 after manual repair."""
        try:
            with self._session_factory() as session:
                session.execute(
                    text("INSERT INTO notes_fts(notes_fts) VALUES('integrity-check')")
                )
            self.available = True
            logger.info("FTS5 availability reset — FTS5 is now enabled")
            return True
        except Exception as e:
            logger.error(f"FTS5 still unavailable: {e}")
            self.available = False
            return False

    # ------------------------------------------------------------------
    # Query escaping helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _should_escape(query: str) -> bool:
        """Auto-detect whether a query needs FTS5 escaping."""
        FTS5_KEYWORDS = {"AND", "OR", "NOT", "NEAR"}
        words = query.upper().split()
        if any(kw in words for kw in FTS5_KEYWORDS):
            return False
        if query.count('"') >= 2:
            return False
        if re.search(r"\b\w+\*", query):
            return False
        if re.search(r"\b\w+:", query):
            return False
        return True

    @staticmethod
    def _escape_query(query: str) -> str:
        """Escape query for FTS5 literal matching (quoted phrase)."""
        result = query.replace('"', '""')
        result = re.sub(r"[*^]", "", result)
        return f'"{result}"'

    # ------------------------------------------------------------------
    # Fallback & recovery
    # ------------------------------------------------------------------

    def _fallback_text_search(
        self, query: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """LIKE-based fallback when FTS5 is unavailable."""
        escaped_query = escape_like_pattern(query)
        search_term = f"%{escaped_query}%"
        results: List[Dict[str, Any]] = []

        try:
            with self._session_factory() as session:
                sql = text("""
                    SELECT id, title, content
                    FROM notes
                    WHERE title LIKE :term ESCAPE '\\' OR content LIKE :term ESCAPE '\\'
                    LIMIT :limit
                """)
                result = session.execute(sql, {"term": search_term, "limit": limit})
                for row in result.fetchall():
                    title_match = query.lower() in row[1].lower() if row[1] else False
                    rank = -2.0 if title_match else -1.0
                    results.append({
                        "id": row[0],
                        "title": row[1],
                        "rank": rank,
                        "search_mode": "fallback",
                    })
        except Exception as e:
            raise SearchError(
                f"Fallback text search failed: {e}",
                query=query,
                code=ErrorCode.SEARCH_FAILED,
            ) from e

        logger.debug(
            f"Fallback search returned {len(results)} results for query '{query}'"
        )
        return results

    def _attempt_recovery(self) -> bool:
        """Attempt to recover FTS5 by rebuilding the index."""
        try:
            count = self.rebuild()
            logger.info(f"FTS5 index rebuilt with {count} notes")
            return True
        except Exception as e:
            logger.error(f"FTS5 rebuild failed: {e}")
            return False
