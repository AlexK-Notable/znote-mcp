"""Tests for resilience coordinator (cross-component linking + events)."""

import pytest


class TestResilienceCoordinatorInit:

    def test_creates_two_aimd_controllers(self):
        from znote_mcp.services.resilience_coordinator import ResilienceCoordinator

        coord = ResilienceCoordinator(max_budget_gb=6.0)
        assert coord.embedder_aimd is not None
        assert coord.reranker_aimd is not None

    def test_creates_two_circuit_breakers(self):
        from znote_mcp.services.resilience_coordinator import ResilienceCoordinator

        coord = ResilienceCoordinator(max_budget_gb=6.0)
        assert coord.embedder_breaker is not None
        assert coord.reranker_breaker is not None


class TestCrossComponentLinking:

    def test_embedder_floor_sends_caution_to_reranker(self):
        from znote_mcp.services.resilience_coordinator import ResilienceCoordinator

        coord = ResilienceCoordinator(max_budget_gb=6.0, min_budget_gb=1.0)
        # Drive embedder to floor
        coord.embedder_aimd._cwnd = 1.5
        coord.on_embedder_failure()
        # Reranker should have received caution
        assert coord.reranker_aimd.phase == "congestion_avoidance"

    def test_reranker_floor_sends_caution_to_embedder(self):
        from znote_mcp.services.resilience_coordinator import ResilienceCoordinator

        coord = ResilienceCoordinator(max_budget_gb=6.0, min_budget_gb=1.0)
        coord.reranker_aimd._cwnd = 1.5
        coord.on_reranker_failure()
        assert coord.embedder_aimd.phase == "congestion_avoidance"


class TestEventEmission:

    def test_failure_emits_event(self):
        from znote_mcp.services.resilience_coordinator import ResilienceCoordinator

        events = []
        coord = ResilienceCoordinator(max_budget_gb=6.0, on_event=events.append)
        coord.on_embedder_failure()
        assert len(events) == 1
        assert "budget_reduced" in events[0]["type"]

    def test_circuit_breaker_trip_emits_event(self):
        from znote_mcp.services.resilience_coordinator import ResilienceCoordinator

        events = []
        coord = ResilienceCoordinator(
            max_budget_gb=6.0,
            on_event=events.append,
            trip_threshold=1,
        )
        coord.on_embedder_failure()
        # Should have both: budget_reduced + circuit_breaker_tripped
        types = [e["type"] for e in events]
        assert "circuit_breaker_tripped" in types

    def test_recovery_emits_event(self):
        from znote_mcp.services.resilience_coordinator import ResilienceCoordinator

        events = []
        coord = ResilienceCoordinator(max_budget_gb=6.0, on_event=events.append)
        coord.on_embedder_failure()
        events.clear()
        # Grow back to max
        coord.embedder_aimd._cwnd = coord.embedder_aimd.max_cwnd
        coord.on_embedder_success(utilization=0.8)
        recovery_events = [e for e in events if "recovered" in e["type"]]
        assert len(recovery_events) == 1


class TestPendingNotices:

    def test_drain_returns_and_clears(self):
        from znote_mcp.services.resilience_coordinator import ResilienceCoordinator

        events = []
        coord = ResilienceCoordinator(max_budget_gb=6.0, on_event=events.append)
        coord.on_embedder_failure()
        notices = coord.drain_pending_notices()
        assert len(notices) >= 1
        assert coord.drain_pending_notices() == []  # cleared


class TestCoordinatorReset:

    def test_reset_all_clears_both_components(self):
        from znote_mcp.services.resilience_coordinator import ResilienceCoordinator

        coord = ResilienceCoordinator(max_budget_gb=6.0)
        coord.on_embedder_failure()
        coord.on_reranker_failure()
        coord.reset_all()
        assert coord.embedder_aimd.phase == "slow_start"
        assert coord.reranker_aimd.phase == "slow_start"
        assert coord.embedder_breaker.state == "closed"


class TestCoordinatorSnapshot:

    def test_snapshot_has_both_components(self):
        from znote_mcp.services.resilience_coordinator import ResilienceCoordinator

        coord = ResilienceCoordinator(max_budget_gb=6.0)
        snap = coord.snapshot()
        assert "embedder" in snap
        assert "reranker" in snap
        assert "aimd" in snap["embedder"]
        assert "circuit_breaker" in snap["embedder"]
