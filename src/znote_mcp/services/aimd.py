"""AIMD adaptive memory budget controller.

Implements Additive Increase / Multiplicative Decrease for dynamically
adjusting the memory budget envelope used by the adaptive batcher.
Each component (embedder, reranker) gets its own controller instance.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)

Phase = Literal["slow_start", "congestion_avoidance", "cooldown"]

UTILIZATION_THRESHOLD = 0.75  # minimum utilization to count as "at capacity"


@dataclass
class FailureResult:
    """Result of an on_failure() call."""

    new_cwnd: float
    at_floor: bool


class AIMDController:
    """AIMD controller governing a memory budget envelope (in GB).

    Args:
        max_cwnd: Hardware tier ceiling (GB). AIMD can grow up to this.
        min_cwnd: Hard floor (GB). Below this, circuit breaker should trip.
        initial_cwnd: Starting budget. Defaults to max_cwnd / 2.
    """

    def __init__(
        self,
        max_cwnd: float,
        min_cwnd: float = 1.0,
        initial_cwnd: float | None = None,
    ) -> None:
        self._max_cwnd = max_cwnd
        self._min_cwnd = min_cwnd
        self._initial_cwnd = initial_cwnd if initial_cwnd is not None else max_cwnd / 2
        self._cwnd = self._initial_cwnd
        self._ssthresh = max_cwnd
        self._phase: Phase = "slow_start"
        self._lock = threading.Lock()
        # Additive increase increment: 5% of max, gives ~20 successes to full recovery
        self._increment = max_cwnd * 0.05

    # --- Properties (read outside lock is safe for floats/strings) ---

    @property
    def cwnd(self) -> float:
        return self._cwnd

    @property
    def ssthresh(self) -> float:
        return self._ssthresh

    @property
    def phase(self) -> Phase:
        return self._phase

    @property
    def max_cwnd(self) -> float:
        return self._max_cwnd

    @property
    def min_cwnd(self) -> float:
        return self._min_cwnd

    @property
    def at_floor(self) -> bool:
        return self._cwnd <= self._min_cwnd

    # --- State transitions ---

    def on_success(self, utilization: float) -> None:
        """Record a successful batch. Only grows if utilization >= threshold."""
        if utilization < UTILIZATION_THRESHOLD:
            return

        with self._lock:
            if self._phase == "cooldown":
                return

            if self._phase == "slow_start":
                new = min(self._cwnd * 2, self._ssthresh)
                if new >= self._ssthresh:
                    self._phase = "congestion_avoidance"
                self._cwnd = min(new, self._max_cwnd)
            else:
                self._cwnd = min(self._cwnd + self._increment, self._max_cwnd)

    def on_failure(self) -> FailureResult:
        """Record a failure (OOM). Multiplicative decrease."""
        with self._lock:
            new_cwnd = max(self._cwnd / 2, self._min_cwnd)
            self._ssthresh = new_cwnd
            self._cwnd = new_cwnd
            self._phase = "congestion_avoidance"

            hit_floor = self._cwnd <= self._min_cwnd
            logger.warning(
                "AIMD failure: cwnd %.1f -> %.1f GB (ssthresh=%.1f, floor=%s)",
                self._cwnd * 2,
                self._cwnd,
                self._ssthresh,
                hit_floor,
            )
            return FailureResult(new_cwnd=self._cwnd, at_floor=hit_floor)

    def enter_cooldown(self) -> None:
        """Enter cooldown phase (circuit breaker tripped)."""
        with self._lock:
            self._phase = "cooldown"

    def exit_cooldown(self) -> None:
        """Exit cooldown, resume congestion avoidance at current cwnd."""
        with self._lock:
            self._phase = "congestion_avoidance"

    def receive_caution(self) -> None:
        """Receive cross-component caution signal.

        Sets ssthresh = min(ssthresh, cwnd/2) and exits slow start.
        Does NOT reduce cwnd — just makes future growth more cautious.
        """
        with self._lock:
            self._ssthresh = min(self._ssthresh, self._cwnd / 2)
            if self._phase == "slow_start":
                self._phase = "congestion_avoidance"

    def reset(self) -> None:
        """Reset to initial state (agent-triggered via zk_system)."""
        with self._lock:
            self._cwnd = self._initial_cwnd
            self._ssthresh = self._max_cwnd
            self._phase = "slow_start"

    def snapshot(self) -> dict:
        """Return current state for zk_status reporting."""
        return {
            "cwnd": round(self._cwnd, 2),
            "ssthresh": round(self._ssthresh, 2),
            "phase": self._phase,
            "max_cwnd": self._max_cwnd,
            "min_cwnd": self._min_cwnd,
        }
