# tests/test_hybrid_search.py
"""Unit tests for Reciprocal Rank Fusion (RRF) in SearchService.

Tests the _reciprocal_rank_fusion() static method directly with known
inputs and expected outputs. No database or embedding infrastructure
required -- pure math tests.
"""

import pytest

from znote_mcp.services.search_service import SearchService


class TestReciprocalRankFusion:
    """Tests for SearchService._reciprocal_rank_fusion()."""

    def test_rrf_score_formula(self):
        """Two lists with known ranks produce scores matching 1/(k+rank)."""
        # Single list, k=60: rank 1 -> 1/61, rank 2 -> 1/62
        ranked_lists = [["note_a", "note_b"]]
        result = SearchService._reciprocal_rank_fusion(ranked_lists, k=60)

        scores = {nid: score for nid, score in result}
        assert scores["note_a"] == pytest.approx(1.0 / 61)
        assert scores["note_b"] == pytest.approx(1.0 / 62)

    def test_rrf_merge_two_lists(self):
        """Two ranked lists with partial overlap produce correct fused order."""
        # List 1: A(rank1), B(rank2), C(rank3)
        # List 2: C(rank1), B(rank2), D(rank3)
        # Expected scores (k=60):
        #   A: 1/61 = 0.01639
        #   B: 1/62 + 1/62 = 2/62 = 0.03226
        #   C: 1/63 + 1/61 = (61+63)/(63*61) = 0.03226 (different calculation)
        #   D: 1/63 = 0.01587
        list1 = ["A", "B", "C"]
        list2 = ["C", "B", "D"]

        result = SearchService._reciprocal_rank_fusion([list1, list2], k=60)
        scores = {nid: score for nid, score in result}

        # B and C appear in both lists, so they should score higher
        assert scores["B"] > scores["A"]
        assert scores["C"] > scores["A"]
        assert scores["B"] > scores["D"]
        assert scores["C"] > scores["D"]

        # Verify fused order: B and C first (both in 2 lists), then A, then D
        result_ids = [nid for nid, _ in result]
        assert result_ids.index("A") > result_ids.index("B")
        assert result_ids.index("D") > result_ids.index("B")

    def test_rrf_single_source_note(self):
        """Note appearing in only one list gets a single-source score."""
        list1 = ["A", "B"]
        list2 = ["C", "D"]

        result = SearchService._reciprocal_rank_fusion([list1, list2], k=60)
        scores = {nid: score for nid, score in result}

        # Each note has exactly one contribution
        assert scores["A"] == pytest.approx(1.0 / 61)
        assert scores["C"] == pytest.approx(1.0 / 61)
        assert scores["B"] == pytest.approx(1.0 / 62)
        assert scores["D"] == pytest.approx(1.0 / 62)

    def test_rrf_deduplication(self):
        """Same note in both lists produces one result with summed score."""
        list1 = ["X"]
        list2 = ["X"]

        result = SearchService._reciprocal_rank_fusion([list1, list2], k=60)

        # X should appear exactly once
        assert len(result) == 1
        note_id, score = result[0]
        assert note_id == "X"
        # Score = 1/61 + 1/61 = 2/61
        assert score == pytest.approx(2.0 / 61)

    def test_rrf_empty_list(self):
        """One or both empty lists handled gracefully."""
        # Both empty
        result = SearchService._reciprocal_rank_fusion([[], []], k=60)
        assert result == []

        # One empty, one populated
        result = SearchService._reciprocal_rank_fusion([["A", "B"], []], k=60)
        assert len(result) == 2
        scores = {nid: score for nid, score in result}
        assert scores["A"] == pytest.approx(1.0 / 61)

        # Single empty list
        result = SearchService._reciprocal_rank_fusion([[]], k=60)
        assert result == []

    def test_rrf_k_constant_effect(self):
        """Different k values affect relative scores predictably."""
        list1 = ["A", "B"]
        list2 = ["B", "A"]

        # With k=60: A gets 1/61 + 1/62, B gets 1/62 + 1/61 -> same scores
        result_k60 = SearchService._reciprocal_rank_fusion([list1, list2], k=60)
        scores_k60 = {nid: score for nid, score in result_k60}
        assert scores_k60["A"] == pytest.approx(scores_k60["B"])

        # With k=1: A gets 1/2 + 1/3, B gets 1/3 + 1/2 -> same scores
        # but the absolute values differ from k=60
        result_k1 = SearchService._reciprocal_rank_fusion([list1, list2], k=1)
        scores_k1 = {nid: score for nid, score in result_k1}
        assert scores_k1["A"] == pytest.approx(scores_k1["B"])

        # With smaller k, absolute scores are larger
        assert scores_k1["A"] > scores_k60["A"]

        # With k=1, the difference between rank 1 and rank 2 is more pronounced
        # Test with non-symmetric lists:
        list3 = ["X", "Y"]  # X rank 1, Y rank 2
        result_k1_asym = SearchService._reciprocal_rank_fusion([list3], k=1)
        result_k60_asym = SearchService._reciprocal_rank_fusion([list3], k=60)
        scores_k1_a = {nid: score for nid, score in result_k1_asym}
        scores_k60_a = {nid: score for nid, score in result_k60_asym}

        # Ratio X/Y with k=1: (1/2)/(1/3) = 1.5
        # Ratio X/Y with k=60: (1/61)/(1/62) ~ 1.016
        ratio_k1 = scores_k1_a["X"] / scores_k1_a["Y"]
        ratio_k60 = scores_k60_a["X"] / scores_k60_a["Y"]
        assert ratio_k1 > ratio_k60  # Smaller k amplifies rank differences
