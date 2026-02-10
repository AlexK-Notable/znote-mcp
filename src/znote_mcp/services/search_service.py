"""Service for searching and discovering notes in the Zettelkasten."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Union

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
    
    def initialize(self) -> None:
        """Initialize the service and dependencies."""
        # Initialize the zettel service if it hasn't been initialized
        self.zettel_service.initialize()
    
    def search_by_text(
        self, query: str, include_content: bool = True, include_title: bool = True
    ) -> List[SearchResult]:
        """Search for notes by text content."""
        if not query:
            return []
        
        # Normalize query
        query = query.lower()
        query_terms = set(query.split())
        
        # Get all notes
        all_notes = self.zettel_service.get_all_notes()
        results = []
        
        for note in all_notes:
            score = 0.0
            matched_terms: Set[str] = set()
            matched_context = ""
            
            # Check title
            if include_title and note.title:
                title_lower = note.title.lower()
                # Exact match in title is highest score
                if query in title_lower:
                    score += 2.0
                    matched_context = f"Title: {note.title}"
                # Check for term matches in title
                for term in query_terms:
                    if term in title_lower:
                        score += 0.5
                        matched_terms.add(term)
            
            # Check content
            if include_content and note.content:
                content_lower = note.content.lower()
                # Exact match in content
                if query in content_lower:
                    score += 1.0
                    # Extract a snippet around the match
                    index = content_lower.find(query)
                    start = max(0, index - 40)
                    end = min(len(content_lower), index + len(query) + 40)
                    snippet = note.content[start:end]
                    matched_context = f"Content: ...{snippet}..."
                # Check for term matches in content
                for term in query_terms:
                    if term in content_lower:
                        score += 0.2
                        matched_terms.add(term)
            
            # Add to results if score is positive
            if score > 0:
                results.append(
                    SearchResult(
                        note=note,
                        score=score,
                        matched_terms=matched_terms,
                        matched_context=matched_context
                    )
                )
        
        # Sort by score (descending)
        results.sort(key=lambda x: x.score, reverse=True)
        return results
    
    def search_by_tag(self, tags: Union[str, List[str]]) -> List[Note]:
        """Search for notes by tags."""
        if isinstance(tags, str):
            return self.zettel_service.get_notes_by_tag(tags)
        else:
            # If we have multiple tags, find notes with any of the tags
            all_matching_notes = []
            for tag in tags:
                notes = self.zettel_service.get_notes_by_tag(tag)
                all_matching_notes.extend(notes)
            # Remove duplicates by converting to a dictionary by ID
            unique_notes = {note.id: note for note in all_matching_notes}
            return list(unique_notes.values())
    
    def search_by_link(self, note_id: str, direction: str = "both") -> List[Note]:
        """Search for notes linked to/from a note."""
        return self.zettel_service.get_linked_notes(note_id, direction)
    
    def find_orphaned_notes(self) -> List[Note]:
        """Find notes with no incoming or outgoing links."""
        orphaned_ids = self.zettel_service.repository.find_orphaned_note_ids()
        return self.zettel_service.repository.get_by_ids(orphaned_ids)

    def find_central_notes(self, limit: int = 10) -> List[Tuple[Note, int]]:
        """Find notes with the most connections (incoming + outgoing links)."""
        # Get note IDs with connection counts from repository
        id_counts = self.zettel_service.repository.find_central_note_ids_with_counts(limit)

        if not id_counts:
            return []

        # Build connection count map and collect IDs for batch retrieval
        note_ids = [note_id for note_id, _ in id_counts]
        connection_counts = {note_id: count for note_id, count in id_counts}

        # Batch retrieve all notes at once
        notes = self.zettel_service.repository.get_by_ids(note_ids)

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
        use_updated: bool = False
    ) -> List[Note]:
        """Find notes created or updated within a date range."""
        all_notes = self.zettel_service.get_all_notes()
        matching_notes = []
        
        for note in all_notes:
            # Get the relevant date
            date = note.updated_at if use_updated else note.created_at
            
            # Check if in range
            if start_date and date < start_date:
                continue
            if end_date and date >= end_date + timedelta(seconds=1):
                continue
            
            matching_notes.append(note)
        
        # Sort by date (descending)
        matching_notes.sort(
            key=lambda x: x.updated_at if use_updated else x.created_at,
            reverse=True
        )
        
        return matching_notes
    
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
            repo = self.zettel_service.repository
            fetch_limit = limit * 3 if use_reranker else limit
            raw_results = repo.vec_similarity_search(
                query_vector, limit=fetch_limit, exclude_ids=exclude_ids,
            )

            if not raw_results:
                return []

            # 3. Retrieve full Note objects
            result_ids = [nid for nid, _ in raw_results]
            notes = repo.get_by_ids(result_ids)
            note_map = {n.id: n for n in notes}
            dist_map = {nid: dist for nid, dist in raw_results}

            # 4. Optionally rerank with cross-encoder
            if (
                use_reranker
                and self._embedding_service.has_reranker
                and len(notes) > 1
            ):
                return self._rerank_results(
                    query, result_ids, note_map, dist_map, limit
                )

            # 5. Without reranker: score = 1 / (1 + distance)
            results = []
            for nid in result_ids:
                note = note_map.get(nid)
                if note is None:
                    continue
                dist = dist_map[nid]
                results.append(SemanticSearchResult(
                    note=note,
                    distance=dist,
                    score=1.0 / (1.0 + dist),
                    reranked=False,
                ))
            return results[:limit]

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

        repo = self.zettel_service.repository

        try:
            # Get the stored embedding for this note
            embedding = repo.get_embedding(note_id)
            if embedding is None:
                return []

            # KNN search, excluding the seed note
            fetch_limit = limit * 3 if use_reranker else limit
            raw_results = repo.vec_similarity_search(
                embedding, limit=fetch_limit, exclude_ids=[note_id],
            )

            if not raw_results:
                return []

            # Retrieve full Note objects
            result_ids = [nid for nid, _ in raw_results]
            notes = repo.get_by_ids(result_ids)
            note_map = {n.id: n for n in notes}
            dist_map = {nid: dist for nid, dist in raw_results}

            # Optionally rerank
            if (
                use_reranker
                and self._embedding_service.has_reranker
                and len(notes) > 1
            ):
                # Use the seed note's content as the query for reranking
                seed_note = self.zettel_service.get_note(note_id)
                if seed_note is not None:
                    query_text = f"{seed_note.title}\n{seed_note.content}"
                    return self._rerank_results(
                        query_text, result_ids, note_map, dist_map, limit
                    )

            # Without reranker
            results = []
            for nid in result_ids:
                note = note_map.get(nid)
                if note is None:
                    continue
                dist = dist_map[nid]
                results.append(SemanticSearchResult(
                    note=note,
                    distance=dist,
                    score=1.0 / (1.0 + dist),
                    reranked=False,
                ))
            return results[:limit]

        except Exception as e:
            logger.warning(f"find_related failed for note {note_id}: {e}")
            return []

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
            results.append(SemanticSearchResult(
                note=note,
                distance=dist_map[nid],
                score=rerank_score,
                reranked=True,
            ))
        return results

    def search_combined(
        self,
        text: Optional[str] = None,
        tags: Optional[List[str]] = None,
        note_type: Optional[NoteType] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[SearchResult]:
        """Perform a combined search with multiple criteria."""
        # Start with all notes
        all_notes = self.zettel_service.get_all_notes()
        
        # Filter by criteria
        filtered_notes = []
        for note in all_notes:
            # Check note type
            if note_type and note.note_type != note_type:
                continue
            
            # Check date range
            if start_date and note.created_at < start_date:
                continue
            if end_date and note.created_at > end_date:
                continue
            
            # Check tags
            if tags:
                note_tag_names = {tag.name for tag in note.tags}
                if not any(tag in note_tag_names for tag in tags):
                    continue
            
            # Made it through all filters
            filtered_notes.append(note)
        
        # If we have a text query, score the notes
        results = []
        if text:
            text = text.lower()
            query_terms = set(text.split())
            
            for note in filtered_notes:
                score = 0.0
                matched_terms: Set[str] = set()
                matched_context = ""
                
                # Check title
                title_lower = note.title.lower()
                if text in title_lower:
                    score += 2.0
                    matched_context = f"Title: {note.title}"
                
                for term in query_terms:
                    if term in title_lower:
                        score += 0.5
                        matched_terms.add(term)
                
                # Check content
                content_lower = note.content.lower()
                if text in content_lower:
                    score += 1.0
                    index = content_lower.find(text)
                    start = max(0, index - 40)
                    end = min(len(content_lower), index + len(text) + 40)
                    snippet = note.content[start:end]
                    matched_context = f"Content: ...{snippet}..."
                
                for term in query_terms:
                    if term in content_lower:
                        score += 0.2
                        matched_terms.add(term)
                
                # Add to results if score is positive
                if score > 0:
                    results.append(
                        SearchResult(
                            note=note,
                            score=score,
                            matched_terms=matched_terms,
                            matched_context=matched_context
                        )
                    )
        else:
            # If no text query, just add all filtered notes with a default score
            results = [
                SearchResult(note=note, score=1.0, matched_terms=set(), matched_context="")
                for note in filtered_notes
            ]
        
        # Sort by score (descending)
        results.sort(key=lambda x: x.score, reverse=True)
        return results
