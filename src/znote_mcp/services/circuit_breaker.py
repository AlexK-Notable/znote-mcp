"""Circuit breaker for GPU/CPU provider switching.

Watches error rates via a sliding window. When failures exceed a
threshold within a time window, trips to CPU fallback with a cooldown
timer. Escalates cooldown on repeated trips without GPU success.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Literal

logger = logging.getLogger(__name__)

State = Literal["closed", "open", "disabled", "forced_cpu"]


class CircuitBreaker:
    """Sliding window circuit breaker for GPU/CPU switching.

    Args:
        trip_threshold: Number of failures within window to trip.
        window_seconds: Sliding window duration for failure counting.
        cooldown_seconds: Initial cooldown duration after trip.
        max_cooldown_seconds: Cap for escalated cooldown.
    """

    def __init__(
        self,
        trip_threshold: int = 3,
        window_seconds: float = 60.0,
        cooldown_seconds: float = 120.0,
        max_cooldown_seconds: float = 600.0,
    ) -> None:
        self._trip_threshold = trip_threshold
        self._window_seconds = window_seconds
        self._initial_cooldown = cooldown_seconds
        self._cooldown_seconds = cooldown_seconds
        self._max_cooldown = max_cooldown_seconds

        self._state: State = "closed"
        self._provider: Literal["gpu", "cpu"] = "gpu"
        self._failures: deque[float] = deque()
        self._trip_count = 0
        self._last_trip_time: float | None = None
        self._had_gpu_success_since_trip = False
        self._lock = threading.Lock()

    # --- Properties ---

    @property
    def state(self) -> State:
        return self._state

    @property
    def provider(self) -> Literal["gpu", "cpu"]:
        return self._provider

    @property
    def cooldown_seconds(self) -> float:
        return self._cooldown_seconds

    @property
    def cooldown_remaining(self) -> float:
        if self._last_trip_time is None or self._state != "open":
            return 0.0
        elapsed = time.monotonic() - self._last_trip_time
        return max(0.0, self._cooldown_seconds - elapsed)

    @property
    def is_cooldown_expired(self) -> bool:
        return self._state == "open" and self.cooldown_remaining <= 0

    @property
    def is_enabled(self) -> bool:
        return self._state != "disabled"

    # --- Actions ---

    def record_failure(self) -> None:
        """Record an OOM failure. May trip the breaker."""
        with self._lock:
            now = time.monotonic()
            self._failures.append(now)
            self._prune_window(now)

            if (
                self._state == "closed"
                and len(self._failures) >= self._trip_threshold
            ):
                self._trip(now)

    def on_budget_floor_breach(self) -> None:
        """Called when AIMD budget hits the floor. Trips immediately."""
        with self._lock:
            if self._state == "closed":
                self._trip(time.monotonic())

    def on_cpu_failure(self) -> None:
        """Called when CPU provider also fails. Disables component."""
        with self._lock:
            self._state = "disabled"
            logger.error("Circuit breaker: CPU also failed, component disabled")

    def record_gpu_success(self) -> None:
        """Record a successful GPU operation. Resets escalation."""
        with self._lock:
            self._had_gpu_success_since_trip = True
            self._cooldown_seconds = self._initial_cooldown

    def reset_to_closed(self) -> None:
        """Return to closed state (e.g., after cooldown for GPU retry)."""
        with self._lock:
            self._state = "closed"
            self._provider = "gpu"
            self._failures.clear()

    def manual_reset(self) -> None:
        """Full reset triggered by agent via zk_system."""
        with self._lock:
            self._state = "closed"
            self._provider = "gpu"
            self._failures.clear()
            self._trip_count = 0
            self._last_trip_time = None
            self._cooldown_seconds = self._initial_cooldown
            self._had_gpu_success_since_trip = False

    def force_cpu(self) -> None:
        """Agent-triggered forced CPU mode."""
        with self._lock:
            self._state = "forced_cpu"
            self._provider = "cpu"

    # --- Internal ---

    def _trip(self, now: float) -> None:
        """Trip the breaker: switch to CPU, start cooldown."""
        # Escalate if no GPU success since last trip
        if self._trip_count > 0 and not self._had_gpu_success_since_trip:
            self._cooldown_seconds = min(
                self._cooldown_seconds * 2, self._max_cooldown
            )

        self._state = "open"
        self._provider = "cpu"
        self._last_trip_time = now
        self._trip_count += 1
        self._had_gpu_success_since_trip = False
        self._failures.clear()

        logger.warning(
            "Circuit breaker tripped (count=%d, cooldown=%.0fs)",
            self._trip_count,
            self._cooldown_seconds,
        )

    def _prune_window(self, now: float) -> None:
        """Remove failures older than the window."""
        cutoff = now - self._window_seconds
        while self._failures and self._failures[0] < cutoff:
            self._failures.popleft()

    def snapshot(self) -> dict:
        """Return current state for zk_status reporting."""
        return {
            "state": self._state,
            "provider": self._provider,
            "trip_count": self._trip_count,
            "cooldown_remaining": round(self.cooldown_remaining, 1),
            "cooldown_seconds": self._cooldown_seconds,
            "failures_in_window": len(self._failures),
        }
