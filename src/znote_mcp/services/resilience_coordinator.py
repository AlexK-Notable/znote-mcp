"""Resilience coordinator: orchestrates AIMD + circuit breaker per component.

Owns one AIMDController and one CircuitBreaker for each component (embedder,
reranker). Handles cross-component linking (stress on one sends caution to the
other), structured event emission for agent signaling, and provides the public
API consumed by EmbeddingService.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable

from znote_mcp.services.aimd import AIMDController
from znote_mcp.services.circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)

EventCallback = Callable[[dict[str, Any]], None]


class ResilienceCoordinator:
    """Orchestrates AIMD controllers and circuit breakers for embedder/reranker.

    Args:
        max_budget_gb: Hardware tier ceiling passed to both AIMD controllers.
        min_budget_gb: Hard floor for AIMD (below this, breaker trips).
        on_event: Optional callback fired on each structured event.
        trip_threshold: Failure count to trip circuit breakers.
    """

    def __init__(
        self,
        max_budget_gb: float,
        min_budget_gb: float = 1.0,
        on_event: EventCallback | None = None,
        trip_threshold: int = 3,
    ) -> None:
        self._on_event = on_event
        self._notices_lock = threading.Lock()
        self._pending_notices: list[dict[str, Any]] = []

        # One AIMD + breaker per component
        self.embedder_aimd = AIMDController(
            max_cwnd=max_budget_gb, min_cwnd=min_budget_gb
        )
        self.reranker_aimd = AIMDController(
            max_cwnd=max_budget_gb, min_cwnd=min_budget_gb
        )
        self.embedder_breaker = CircuitBreaker(trip_threshold=trip_threshold)
        self.reranker_breaker = CircuitBreaker(trip_threshold=trip_threshold)

        # Track whether each component has degraded (for recovery events)
        self._degraded: dict[str, bool] = {"embedder": False, "reranker": False}

    # --- Failure handlers ---

    def on_embedder_failure(self) -> None:
        """Record embedder failure: AIMD decrease, breaker record, cross-link."""
        self._handle_failure(
            component="embedder",
            aimd=self.embedder_aimd,
            breaker=self.embedder_breaker,
            peer_aimd=self.reranker_aimd,
        )

    def on_reranker_failure(self) -> None:
        """Record reranker failure: AIMD decrease, breaker record, cross-link."""
        self._handle_failure(
            component="reranker",
            aimd=self.reranker_aimd,
            breaker=self.reranker_breaker,
            peer_aimd=self.embedder_aimd,
        )

    # --- Success handlers ---

    def on_embedder_success(self, utilization: float) -> None:
        """Record embedder success."""
        self._handle_success(
            component="embedder",
            aimd=self.embedder_aimd,
            breaker=self.embedder_breaker,
            utilization=utilization,
        )

    def on_reranker_success(self, utilization: float) -> None:
        """Record reranker success."""
        self._handle_success(
            component="reranker",
            aimd=self.reranker_aimd,
            breaker=self.reranker_breaker,
            utilization=utilization,
        )

    # --- Pending notices ---

    def drain_pending_notices(self) -> list[dict[str, Any]]:
        """Return and clear accumulated event notices (thread-safe)."""
        with self._notices_lock:
            notices = list(self._pending_notices)
            self._pending_notices.clear()
            return notices

    # --- Control methods ---

    def reset_all(self) -> None:
        """Reset both AIMD controllers and both circuit breakers."""
        self.embedder_aimd.reset()
        self.reranker_aimd.reset()
        self.embedder_breaker.manual_reset()
        self.reranker_breaker.manual_reset()
        self._degraded = {"embedder": False, "reranker": False}

    def force_cpu_all(self) -> None:
        """Force CPU on both breakers."""
        self.embedder_breaker.force_cpu()
        self.reranker_breaker.force_cpu()

    def disable_all(self) -> None:
        """Disable both components via circuit breakers."""
        self.embedder_breaker.on_cpu_failure()
        self.reranker_breaker.on_cpu_failure()

    def enable_all(self) -> None:
        """Re-enable both components by resetting breakers to closed."""
        self.embedder_breaker.reset_to_closed()
        self.reranker_breaker.reset_to_closed()

    # --- Snapshot ---

    def snapshot(self) -> dict[str, Any]:
        """Return dict with embedder and reranker sub-dicts."""
        return {
            "embedder": {
                "aimd": self.embedder_aimd.snapshot(),
                "circuit_breaker": self.embedder_breaker.snapshot(),
            },
            "reranker": {
                "aimd": self.reranker_aimd.snapshot(),
                "circuit_breaker": self.reranker_breaker.snapshot(),
            },
        }

    # --- Internal ---

    def _handle_failure(
        self,
        component: str,
        aimd: AIMDController,
        breaker: CircuitBreaker,
        peer_aimd: AIMDController,
    ) -> None:
        """Common failure handling for a component."""
        result = aimd.on_failure()
        breaker.record_failure()

        self._degraded[component] = True

        self._emit(
            {
                "type": "budget_reduced",
                "component": component,
                "new_cwnd": result.new_cwnd,
                "at_floor": result.at_floor,
            }
        )

        # If AIMD hit floor, trip breaker immediately and send caution to peer
        if result.at_floor:
            breaker.on_budget_floor_breach()
            peer_aimd.receive_caution()

        # Check if breaker tripped (could be from record_failure or floor breach)
        if breaker.state == "open":
            aimd.enter_cooldown()
            self._emit(
                {
                    "type": "circuit_breaker_tripped",
                    "component": component,
                    "cooldown_seconds": breaker.cooldown_seconds,
                }
            )

    def _handle_success(
        self,
        component: str,
        aimd: AIMDController,
        breaker: CircuitBreaker,
        utilization: float,
    ) -> None:
        """Common success handling for a component."""
        aimd.on_success(utilization=utilization)
        breaker.record_gpu_success()

        if self._degraded[component] and aimd.cwnd >= aimd.max_cwnd:
            self._degraded[component] = False
            self._emit(
                {
                    "type": "budget_recovered",
                    "component": component,
                    "cwnd": aimd.cwnd,
                }
            )

    def _emit(self, event: dict[str, Any]) -> None:
        """Emit an event to callback and pending notices."""
        with self._notices_lock:
            self._pending_notices.append(event)
        if self._on_event is not None:
            self._on_event(event)
