"""Service for searching and discovering notes in the Zettelkasten."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

from znote_mcp.config import config
from znote_mcp.models.schema import LinkType, Note, NoteType, Tag
from znote_mcp.services.zettel_service import ZettelService

if TYPE_CHECKING:
    from znote_mcp.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A search result with a note and its relevance score."""

    note: Note
    score: float
    matched_terms: Set[str]
    matched_context: str


@dataclass
class SemanticSearchResult:
    """A semantic search result with vector distance and optional rerank score.

    Attributes:
        note: The matched note.
        distance: L2 distance from sqlite-vec (lower = more similar).
            For L2-normalised vectors: cosine_sim = 1 - (distance² / 2).
        score: Final relevance score (rerank score if reranked, else
            1 / (1 + distance) as a simple monotonic transform).
        reranked: Whether the result was refined by the cross-encoder reranker.
    """

    note: Note
    distance: float
    score: float
    reranked: bool = False


class SearchService:
    """Service for searching notes in the Zettelkasten."""

    def __init__(
        self,
        zettel_service: Optional[ZettelService] = None,
        embedding_service: Optional[EmbeddingService] = None,
    ):
        """Initialize the search service.

        Args:
            zettel_service: Zettel CRUD service.
            embedding_service: Optional embedding service for semantic search.
                When provided and config.embeddings_enabled is True, enables
                semantic_search() and find_related().
        """
        self.zettel_service = zettel_service or ZettelService()
        self._embedding_service = embedding_service

    @property
    def has_semantic_search(self) -> bool:
        """Whether semantic search is available (embedding service configured and enabled)."""
        return self._embedding_service is not None and config.embeddings_enabled

    def find_orphaned_notes(self) -> List[Note]:
        """Find notes with no incoming or outgoing links."""
        orphaned_ids = self.zettel_service.find_orphaned_note_ids()
        return self.zettel_service.get_notes_by_ids(orphaned_ids)

    def find_central_notes(self, limit: int = 10) -> List[Tuple[Note, int]]:
        """Find notes with the most connections (incoming + outgoing links)."""
        # Get note IDs with connection counts from repository
        id_counts = self.zettel_service.find_central_note_ids_with_counts(limit)

        if not id_counts:
            return []

        # Build connection count map and collect IDs for batch retrieval
        note_ids = [note_id for note_id, _ in id_counts]
        connection_counts = {note_id: count for note_id, count in id_counts}

        # Batch retrieve all notes at once
        notes = self.zettel_service.get_notes_by_ids(note_ids)

        # Build result list preserving SQL ordering
        note_map = {note.id: note for note in notes}
        return [
            (note_map[note_id], connection_counts[note_id])
            for note_id in note_ids
            if note_id in note_map
        ]

    def find_notes_by_date_range(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        use_updated: bool = False,
        limit: Optional[int] = None,
    ) -> List[Note]:
        """Find notes created or updated within a date range.

        Delegates to the SQL-level repository.search() to avoid loading
        all notes into memory.
        """
        kwargs: Dict[str, Any] = {}
        if start_date:
            key = "updated_after" if use_updated else "created_after"
            kwargs[key] = start_date
        if end_date:
            key = "updated_before" if use_updated else "created_before"
            kwargs[key] = end_date
        return self.zettel_service.repository.search(limit=limit, **kwargs)

    def find_similar_notes(self, note_id: str) -> List[Tuple[Note, float]]:
        """Find notes similar to the given note based on shared tags and links."""
        return self.zettel_service.find_similar_notes(note_id)

    # =========================================================================
    # Semantic Search (embedding-powered)
    # =========================================================================

    def semantic_search(
        self,
        query: str,
        limit: int = 10,
        use_reranker: bool = True,
        exclude_ids: Optional[List[str]] = None,
    ) -> List[SemanticSearchResult]:
        """Search notes by semantic similarity using embeddings.

        Embeds the query text, runs a KNN search via sqlite-vec, retrieves
        full Note objects, and optionally refines ranking with the cross-encoder
        reranker.

        Gracefully returns an empty list when:
        - No embedding service is configured
        - embeddings_enabled is False in config
        - sqlite-vec is unavailable in the repository

        Args:
            query: Natural language search query.
            limit: Maximum number of results to return.
            use_reranker: Whether to refine results with the reranker.
                Only applies if a reranker provider is configured.
            exclude_ids: Note IDs to exclude from results.

        Returns:
            List of SemanticSearchResult, ordered by relevance (best first).
        """
        if self._embedding_service is None or not config.embeddings_enabled:
            return []

        if not query.strip():
            return []

        try:
            # 1. Embed the query
            query_vector = self._embedding_service.embed(query)

            # 2. KNN search — fetch extra candidates for reranking
            fetch_limit = limit * 3 if use_reranker else limit
            raw_results = self.zettel_service.vec_similarity_search(
                query_vector,
                limit=fetch_limit,
                exclude_ids=exclude_ids,
            )

            if not raw_results:
                return []

            # 3. Retrieve full Note objects
            result_ids = [nid for nid, _ in raw_results]
            notes = self.zettel_service.get_notes_by_ids(result_ids)
            note_map = {n.id: n for n in notes}
            dist_map = {nid: dist for nid, dist in raw_results}

            # 4. Optionally rerank with cross-encoder
            if use_reranker and self._embedding_service.has_reranker and len(notes) > 1:
                return self._rerank_results(
                    query, result_ids, note_map, dist_map, limit
                )

            # 5. Without reranker: score = 1 / (1 + distance)
            return self._build_distance_results(result_ids, note_map, dist_map, limit)

        except Exception as e:
            logger.warning(f"Semantic search failed: {e}")
            return []

    def find_related(
        self,
        note_id: str,
        limit: int = 10,
        use_reranker: bool = False,
    ) -> List[SemanticSearchResult]:
        """Find notes semantically related to the given note.

        Retrieves the note's stored embedding and searches for nearest
        neighbours.  This is the "more like this" feature.

        Falls back to an empty list if the note has no embedding or
        the embedding service is unavailable.

        Args:
            note_id: ID of the seed note.
            limit: Maximum number of related notes to return.
            use_reranker: Whether to refine results with the reranker.

        Returns:
            List of SemanticSearchResult (excluding the seed note).
        """
        if self._embedding_service is None or not config.embeddings_enabled:
            return []

        try:
            # Get the stored embedding for this note
            embedding = self.zettel_service.get_embedding(note_id)
            if embedding is None:
                return []

            # KNN search, excluding the seed note
            fetch_limit = limit * 3 if use_reranker else limit
            raw_results = self.zettel_service.vec_similarity_search(
                embedding,
                limit=fetch_limit,
                exclude_ids=[note_id],
            )

            if not raw_results:
                return []

            # Retrieve full Note objects
            result_ids = [nid for nid, _ in raw_results]
            notes = self.zettel_service.get_notes_by_ids(result_ids)
            note_map = {n.id: n for n in notes}
            dist_map = {nid: dist for nid, dist in raw_results}

            # Optionally rerank
            if use_reranker and self._embedding_service.has_reranker and len(notes) > 1:
                # Use the seed note's content as the query for reranking
                seed_note = self.zettel_service.get_note(note_id)
                if seed_note is not None:
                    query_text = f"{seed_note.title}\n{seed_note.content}"
                    return self._rerank_results(
                        query_text, result_ids, note_map, dist_map, limit
                    )

            # Without reranker
            return self._build_distance_results(result_ids, note_map, dist_map, limit)

        except Exception as e:
            logger.warning(f"find_related failed for note {note_id}: {e}")
            return []

    @staticmethod
    def _build_distance_results(
        result_ids: List[str],
        note_map: Dict[str, Note],
        dist_map: Dict[str, float],
        limit: int,
    ) -> List[SemanticSearchResult]:
        """Convert raw KNN results into scored SemanticSearchResult objects.

        Score formula: 1 / (1 + distance).  Lower distance = higher score.
        """
        results = []
        for nid in result_ids:
            note = note_map.get(nid)
            if note is None:
                continue
            dist = dist_map[nid]
            results.append(
                SemanticSearchResult(
                    note=note,
                    distance=dist,
                    score=1.0 / (1.0 + dist),
                    reranked=False,
                )
            )
        return results[:limit]

    def _rerank_results(
        self,
        query: str,
        result_ids: List[str],
        note_map: Dict[str, Note],
        dist_map: Dict[str, float],
        limit: int,
    ) -> List[SemanticSearchResult]:
        """Rerank KNN candidates using the cross-encoder reranker.

        Args:
            query: The search query or seed text.
            result_ids: Ordered list of note IDs from KNN search.
            note_map: Mapping of note ID → Note object.
            dist_map: Mapping of note ID → L2 distance.
            limit: Max results to return after reranking.

        Returns:
            Reranked list of SemanticSearchResult.
        """
        # Build document texts for the reranker
        doc_ids: List[str] = []
        doc_texts: List[str] = []
        for nid in result_ids:
            note = note_map.get(nid)
            if note is None:
                continue
            doc_ids.append(nid)
            doc_texts.append(f"{note.title}\n{note.content}")

        if not doc_texts:
            return []

        # Rerank returns (original_index, score) sorted by score desc
        ranked = self._embedding_service.rerank(query, doc_texts, top_k=limit)

        results = []
        for idx, rerank_score in ranked:
            nid = doc_ids[idx]
            note = note_map[nid]
            results.append(
                SemanticSearchResult(
                    note=note,
                    distance=dist_map[nid],
                    score=rerank_score,
                    reranked=True,
                )
            )
        return results

    def search_combined(
        self,
        text: Optional[str] = None,
        tags: Optional[List[str]] = None,
        note_type: Optional[NoteType] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 200,
    ) -> List[SearchResult]:
        """Perform a combined search with multiple criteria.

        When a text query is provided and FTS5 is available, uses FTS5 for
        fast BM25-ranked candidate retrieval, then applies post-filters.
        Falls back to O(N) Python scanning when FTS5 is unavailable.
        """
        # Fast path: FTS5 text search with post-filtering
        if text and text.strip():
            fts_results = self._fts_combined_search(
                text, tags, note_type, start_date, end_date, limit
            )
            if fts_results is not None:
                return fts_results

        # Filter-only path: delegate to SQL when there are filters but no text
        has_filters = any([tags, note_type, start_date, end_date])
        if has_filters and not (text and text.strip()):
            kwargs: Dict[str, Any] = {}
            if tags:
                kwargs["tags"] = tags
            if note_type:
                kwargs["note_type"] = note_type
            if start_date:
                kwargs["created_after"] = start_date
            if end_date:
                kwargs["created_before"] = end_date
            filtered_notes = self.zettel_service.repository.search(
                limit=limit, **kwargs
            )
            return [
                SearchResult(
                    note=note, score=1.0, matched_terms=set(), matched_context=""
                )
                for note in filtered_notes
            ]

        # Fallback: O(N) scan (text query with FTS5 unavailable)
        all_notes = self.zettel_service.get_all_notes()
        filtered_notes = self._apply_filters(
            all_notes, tags, note_type, start_date, end_date
        )

        results: List[SearchResult] = []
        if text and text.strip():
            results = self._score_notes_by_text(filtered_notes, text)
        else:
            results = [
                SearchResult(
                    note=note, score=1.0, matched_terms=set(), matched_context=""
                )
                for note in filtered_notes
            ]

        results.sort(key=lambda x: x.score, reverse=True)
        return results

    def _fts_combined_search(
        self,
        text: str,
        tags: Optional[List[str]],
        note_type: Optional[NoteType],
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        limit: int,
    ) -> Optional[List[SearchResult]]:
        """Attempt FTS5-accelerated search. Returns None if FTS5 unavailable."""
        if not self.zettel_service.has_fts():
            return None

        try:
            # Fetch extra candidates to account for post-filter attrition
            fetch_limit = limit * 3
            fts_results = self.zettel_service.fts_search(
                text.strip(), limit=fetch_limit
            )

            if not fts_results:
                return (
                    None  # Fall through to O(N) fallback for non-FTS-tokenizable text
                )

            # Retrieve full Note objects for FTS hits
            fts_ids = [r["id"] for r in fts_results]
            notes = self.zettel_service.get_notes_by_ids(fts_ids)
            note_map = {n.id: n for n in notes}
            rank_map = {r["id"]: abs(r["rank"]) for r in fts_results}

            # Apply post-filters and build SearchResult objects
            results: List[SearchResult] = []
            filtered_notes = self._apply_filters(
                [note_map[fid] for fid in fts_ids if fid in note_map],
                tags,
                note_type,
                start_date,
                end_date,
            )
            filtered_ids = {n.id for n in filtered_notes}

            for fts_id in fts_ids:
                if fts_id not in filtered_ids:
                    continue
                note = note_map[fts_id]

                bm25_rank = rank_map.get(fts_id, 0.0)
                results.append(
                    SearchResult(
                        note=note,
                        score=bm25_rank,
                        matched_terms=set(text.lower().split()),
                        matched_context=f"FTS5 match (BM25: {bm25_rank:.2f})",
                    )
                )

                if len(results) >= limit:
                    break

            # FTS5 results are already ranked by BM25 (higher = better)
            results.sort(key=lambda x: x.score, reverse=True)
            return results

        except Exception as e:
            logger.warning(f"FTS5 combined search failed, falling back: {e}")
            return None

    @staticmethod
    def _apply_filters(
        notes: List[Note],
        tags: Optional[List[str]],
        note_type: Optional[NoteType],
        start_date: Optional[datetime],
        end_date: Optional[datetime],
    ) -> List[Note]:
        """Apply tag/type/date filters to a list of notes."""
        filtered = []
        for note in notes:
            if note_type and note.note_type != note_type:
                continue
            if start_date and note.created_at < start_date:
                continue
            if end_date and note.created_at > end_date:
                continue
            if tags:
                note_tag_names = {tag.name for tag in note.tags}
                if not any(tag in note_tag_names for tag in tags):
                    continue
            filtered.append(note)
        return filtered

    @staticmethod
    def _score_notes_by_text(notes: List[Note], text: str) -> List[SearchResult]:
        """Score notes by text match in title and content (O(N) fallback)."""
        text_lower = text.lower()
        query_terms = set(text_lower.split())
        results: List[SearchResult] = []

        for note in notes:
            score = 0.0
            matched_terms: Set[str] = set()
            matched_context = ""

            title_lower = note.title.lower()
            if text_lower in title_lower:
                score += 2.0
                matched_context = f"Title: {note.title}"
            for term in query_terms:
                if term in title_lower:
                    score += 0.5
                    matched_terms.add(term)

            content_lower = note.content.lower()
            if text_lower in content_lower:
                score += 1.0
                index = content_lower.find(text_lower)
                start = max(0, index - 40)
                end = min(len(content_lower), index + len(text_lower) + 40)
                snippet = note.content[start:end]
                matched_context = f"Content: ...{snippet}..."
            for term in query_terms:
                if term in content_lower:
                    score += 0.2
                    matched_terms.add(term)

            if score > 0:
                results.append(
                    SearchResult(
                        note=note,
                        score=score,
                        matched_terms=matched_terms,
                        matched_context=matched_context,
                    )
                )

        return results
