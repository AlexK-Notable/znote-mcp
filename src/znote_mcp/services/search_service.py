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

    def find_orphaned_notes(self, limit: Optional[int] = None) -> List[Note]:
        """Find notes with no incoming or outgoing links."""
        orphaned_ids = self.zettel_service.find_orphaned_note_ids(limit=limit)
        return self.zettel_service.get_notes_by_ids(orphaned_ids)

    def count_orphaned_notes(self) -> int:
        """Count notes with no incoming or outgoing links."""
        return self.zettel_service.count_orphaned_notes()

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
        tags: Optional[List[str]] = None,
        note_type: Optional[NoteType] = None,
        project: Optional[str] = None,
    ) -> List[SemanticSearchResult]:
        """Search notes by semantic similarity using embeddings.

        Embeds the query text, runs a KNN search via sqlite-vec, retrieves
        full Note objects, and optionally refines ranking with the cross-encoder
        reranker.

        When tag, note_type, or project filters are provided, uses selectivity-
        based routing:
        - Small candidate set (<=threshold): brute-force distance computation
        - Large candidate set (>threshold): KNN with adaptive over-fetch + post-filter

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
            tags: Filter results to notes with any of these tags.
            note_type: Filter results to this note type.
            project: Filter results to this project.

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

            # 2. Determine filter strategy
            strategy, candidate_ids = self._resolve_filter_strategy(
                tags, note_type, project
            )

            if strategy == "brute_force":
                if not candidate_ids:
                    return []
                return self._brute_force_semantic_search(
                    query_vector,
                    candidate_ids,
                    limit,
                    use_reranker,
                    query,
                    exclude_ids=exclude_ids,
                )

            if strategy == "knn_postfilter":
                return self._knn_postfilter_search(
                    query_vector,
                    candidate_ids,
                    limit,
                    use_reranker,
                    query,
                    exclude_ids=exclude_ids,
                )

            # 3. Unfiltered: existing KNN path
            fetch_limit = limit * 3 if use_reranker else limit
            raw_results = self.zettel_service.vec_similarity_search(
                query_vector,
                limit=fetch_limit,
                exclude_ids=exclude_ids,
            )

            if not raw_results:
                return []

            return self._finalize_semantic_results(
                raw_results, query, limit, use_reranker
            )

        except Exception as e:
            logger.warning(f"Semantic search failed: {e}")
            return []

    def _resolve_filter_strategy(
        self,
        tags: Optional[List[str]],
        note_type: Optional[NoteType],
        project: Optional[str],
    ) -> Tuple[str, Set[str]]:
        """Determine search strategy based on filter selectivity.

        Returns:
            ("unfiltered", set()) - no filters, use standard KNN
            ("brute_force", candidate_ids) - small candidate set, compute distances directly
            ("knn_postfilter", candidate_ids) - large candidate set, KNN + intersection
        """
        if not tags and note_type is None and not project:
            return ("unfiltered", set())

        # Build filter kwargs for search_note_ids
        kwargs: Dict[str, Any] = {}
        if tags:
            kwargs["tags"] = tags
        if note_type is not None:
            kwargs["note_type"] = note_type
        if project:
            kwargs["project"] = project

        candidate_ids = set(self.zettel_service.search_note_ids(**kwargs))

        if not candidate_ids:
            return ("brute_force", set())

        threshold = config.semantic_filter_brute_force_threshold
        if len(candidate_ids) <= threshold:
            return ("brute_force", candidate_ids)

        return ("knn_postfilter", candidate_ids)

    def _brute_force_semantic_search(
        self,
        query_embedding: Any,
        candidate_ids: Set[str],
        limit: int,
        use_reranker: bool,
        query: str,
        exclude_ids: Optional[List[str]] = None,
    ) -> List[SemanticSearchResult]:
        """Compute semantic similarity via brute-force for small candidate sets.

        Loads embeddings for the candidate notes, computes L2 distance against
        the query embedding, and returns the top results.

        Args:
            query_embedding: Query vector (np.ndarray).
            candidate_ids: Set of note IDs to search within.
            limit: Maximum results to return.
            use_reranker: Whether to refine with cross-encoder reranker.
            query: Original query text (for reranking).
            exclude_ids: Note IDs to exclude from results.
        """
        import numpy as np

        # Apply exclusions
        if exclude_ids:
            candidate_ids = candidate_ids - set(exclude_ids)
        if not candidate_ids:
            return []

        # Load embeddings for candidates
        embeddings = self.zettel_service.get_embeddings_for_note_ids(
            list(candidate_ids)
        )
        if not embeddings:
            return []

        # Vectorized L2 distance computation
        ids = list(embeddings.keys())
        matrix = np.stack(list(embeddings.values()))  # shape: (n, 768)
        dists = np.linalg.norm(matrix - query_embedding, axis=1)  # shape: (n,)

        # Sort by distance ascending (closest first)
        order = np.argsort(dists)
        fetch_limit = limit * 3 if use_reranker else limit
        order = order[:fetch_limit]
        scored = [(ids[i], float(dists[i])) for i in order]

        return self._finalize_semantic_results(scored, query, limit, use_reranker)

    def _knn_postfilter_search(
        self,
        query_embedding: Any,
        candidate_ids: Set[str],
        limit: int,
        use_reranker: bool,
        query: str,
        exclude_ids: Optional[List[str]] = None,
    ) -> List[SemanticSearchResult]:
        """KNN search with adaptive over-fetch and post-filter intersection.

        For large candidate sets where brute-force would be expensive,
        uses the KNN index with over-fetching proportional to selectivity,
        then intersects results with the pre-filtered candidate set.

        Args:
            query_embedding: Query vector (np.ndarray).
            candidate_ids: Set of note IDs that pass filters.
            limit: Maximum results to return.
            use_reranker: Whether to refine with cross-encoder reranker.
            query: Original query text (for reranking).
            exclude_ids: Note IDs to exclude from results.
        """
        # Compute selectivity-based over-fetch multiplier
        total_embedded = self.zettel_service.repository.count_embedded_notes()
        selectivity = len(candidate_ids) / max(total_embedded, 1)
        over_fetch = min(max(3, int(1.0 / max(selectivity, 0.01))), 20)
        fetch_limit = limit * over_fetch

        # KNN search with over-fetch
        raw_results = self.zettel_service.vec_similarity_search(
            query_embedding,
            limit=fetch_limit,
            exclude_ids=exclude_ids,
        )

        if not raw_results:
            return []

        # Intersect with candidate IDs
        raw_results = [(nid, dist) for nid, dist in raw_results if nid in candidate_ids]

        if not raw_results:
            return []

        # Take top candidates for reranking or final output
        rerank_limit = limit * 3 if use_reranker else limit
        raw_results = raw_results[:rerank_limit]

        return self._finalize_semantic_results(
            raw_results, query, limit, use_reranker
        )

    @staticmethod
    def _reciprocal_rank_fusion(
        ranked_lists: List[List[str]],
        k: int = 60,
    ) -> List[Tuple[str, float]]:
        """Fuse multiple ranked result lists using Reciprocal Rank Fusion.

        Args:
            ranked_lists: Each list contains note_ids ordered by relevance
                (best first).
            k: RRF smoothing constant (default 60, standard value).

        Returns:
            List of (note_id, rrf_score) tuples sorted by descending RRF score.
        """
        scores: Dict[str, float] = {}
        for ranked_list in ranked_lists:
            for rank_idx, note_id in enumerate(ranked_list):
                rank = rank_idx + 1  # 1-indexed
                scores[note_id] = scores.get(note_id, 0.0) + 1.0 / (k + rank)

        fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return fused

    def hybrid_search(
        self,
        query: str,
        limit: int = 10,
        tags: Optional[List[str]] = None,
        note_type: Optional[NoteType] = None,
        project: Optional[str] = None,
        use_reranker: bool = True,
    ) -> List[SemanticSearchResult]:
        """Run FTS5 and semantic search, merge results via Reciprocal Rank Fusion.

        Combines BM25 text relevance with embedding-based semantic similarity
        for higher-quality results than either path alone.

        Graceful degradation:
        - No semantic search available: FTS-only results
        - No FTS available: semantic-only results
        - Neither available: empty list

        Args:
            query: Natural language search query.
            limit: Maximum number of results to return.
            tags: Filter results to notes with any of these tags.
            note_type: Filter results to this note type.
            project: Filter results to this project.
            use_reranker: Whether to refine fused results with the reranker.

        Returns:
            List of SemanticSearchResult, ordered by relevance (best first).
        """
        if not query or not query.strip():
            return []

        has_fts = self.zettel_service.has_fts()
        has_semantic = self.has_semantic_search

        if not has_fts and not has_semantic:
            logger.info("hybrid_search: neither FTS nor semantic available")
            return []

        fetch_limit = limit * 3

        # --- FTS path ---
        fts_results: List[SearchResult] = []
        if has_fts:
            try:
                fts_results = self.search_combined(
                    text=query, tags=tags, note_type=note_type, limit=fetch_limit
                )
                # Post-filter by project (search_combined doesn't support it)
                if project and fts_results:
                    fts_results = [
                        r for r in fts_results if r.note.project == project
                    ]
            except Exception as e:
                logger.warning(f"hybrid_search: FTS path failed: {e}")

        # --- Semantic path ---
        sem_results: List[SemanticSearchResult] = []
        if has_semantic:
            try:
                sem_results = self.semantic_search(
                    query=query,
                    limit=fetch_limit,
                    tags=tags,
                    note_type=note_type,
                    project=project,
                    use_reranker=False,  # RRF handles fusion
                )
            except Exception as e:
                logger.warning(f"hybrid_search: semantic path failed: {e}")

        # --- Single-path fallback ---
        if not fts_results and not sem_results:
            return []

        if not sem_results:
            # FTS-only: convert SearchResult -> SemanticSearchResult
            logger.info("hybrid_search: FTS-only (no semantic results)")
            return [
                SemanticSearchResult(
                    note=r.note, distance=0.0, score=r.score, reranked=False
                )
                for r in fts_results[:limit]
            ]

        if not fts_results:
            logger.info("hybrid_search: semantic-only (no FTS results)")
            return sem_results[:limit]

        # --- RRF fusion ---
        fts_ids = [r.note.id for r in fts_results]
        sem_ids = [r.note.id for r in sem_results]

        fused = self._reciprocal_rank_fusion([fts_ids, sem_ids])
        fused_top = fused[: fetch_limit]

        # Collect Note objects from both result sets (avoid re-fetching)
        note_map: Dict[str, Note] = {}
        for r in fts_results:
            note_map[r.note.id] = r.note
        for r in sem_results:
            note_map[r.note.id] = r.note

        # --- Optional reranking ---
        if (
            use_reranker
            and has_semantic
            and self._embedding_service is not None
            and self._embedding_service.has_reranker
            and len(fused_top) > 1
        ):
            fused_ids = [nid for nid, _ in fused_top if nid in note_map]
            try:
                return self._rerank_results(
                    query, fused_ids, note_map, limit=limit
                )
            except Exception as e:
                logger.warning(f"hybrid_search: reranking failed: {e}")

        # --- Build RRF-scored results ---
        results = []
        for note_id, rrf_score in fused_top:
            note = note_map.get(note_id)
            if note is None:
                continue
            results.append(
                SemanticSearchResult(
                    note=note, distance=0.0, score=rrf_score, reranked=False
                )
            )
        return results[:limit]

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

    def _finalize_semantic_results(
        self,
        scored: List[Tuple[str, float]],
        query: str,
        limit: int,
        use_reranker: bool,
    ) -> List[SemanticSearchResult]:
        """Retrieve notes for scored pairs and optionally rerank.

        Shared tail for unfiltered KNN, brute-force, and KNN+postfilter paths.

        Args:
            scored: List of (note_id, distance) tuples, sorted by distance.
            query: Original query text (for reranking).
            limit: Maximum results to return.
            use_reranker: Whether to refine with cross-encoder reranker.
        """
        if not scored:
            return []

        result_ids = [nid for nid, _ in scored]
        notes = self.zettel_service.get_notes_by_ids(result_ids)
        note_map = {n.id: n for n in notes}
        dist_map = dict(scored)

        if use_reranker and self._embedding_service.has_reranker and len(notes) > 1:
            return self._rerank_results(query, result_ids, note_map, dist_map, limit)

        return self._build_distance_results(result_ids, note_map, dist_map, limit)

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
        dist_map: Optional[Dict[str, float]] = None,
        limit: int = 10,
    ) -> List[SemanticSearchResult]:
        """Rerank candidates using the cross-encoder reranker.

        Args:
            query: The search query or seed text.
            result_ids: Ordered list of note IDs.
            note_map: Mapping of note ID → Note object.
            dist_map: Mapping of note ID → L2 distance. When None,
                distance defaults to 0.0 (e.g. for hybrid/RRF results).
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
                    distance=dist_map[nid] if dist_map else 0.0,
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
        return results[:limit]

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
