# AIMD Adaptive Resilience Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the fixed 5-level ONNX resilience staircase with AIMD-based adaptive memory budget control, a circuit breaker for GPU/CPU switching, and inline agent signaling on meaningful state transitions.

**Architecture:** Three layers — AIMD controller (memory budget envelope), circuit breaker (GPU↔CPU switching + cooldowns), and a resilience coordinator that owns both per-component and handles cross-component linking. The adaptive batcher operates within the AIMD-governed budget. Inline notices prepended to tool responses on state transitions only.

**Tech Stack:** Python 3.10+, threading for concurrency, time.monotonic for timestamps, existing EmbeddingService/OnnxProvider patterns.

**Design doc:** `docs/plans/2026-03-07-aimd-resilience-design.md`
**Znote decision rationale:** `j4FniOi9JF_lOf8n9T49d`
**Znote plan:** `nX7FhpYJP0U_UGdtmO391`

---

### Task 1: AIMD Controller — Failing Tests

**Files:**
- Create: `tests/test_aimd.py`

**Step 1: Write failing tests for AIMDController**

```python
"""Tests for AIMD adaptive memory budget controller."""

import pytest


class TestAIMDControllerInit:
    """Test initial state."""

    def test_initial_cwnd_is_half_max(self):
        from znote_mcp.services.aimd import AIMDController

        ctrl = AIMDController(max_cwnd=6.0, min_cwnd=1.0)
        assert ctrl.cwnd == 3.0

    def test_starts_in_slow_start(self):
        from znote_mcp.services.aimd import AIMDController

        ctrl = AIMDController(max_cwnd=6.0, min_cwnd=1.0)
        assert ctrl.phase == "slow_start"

    def test_ssthresh_starts_at_max(self):
        from znote_mcp.services.aimd import AIMDController

        ctrl = AIMDController(max_cwnd=6.0, min_cwnd=1.0)
        assert ctrl.ssthresh == 6.0

    def test_custom_initial_cwnd(self):
        from znote_mcp.services.aimd import AIMDController

        ctrl = AIMDController(max_cwnd=10.0, min_cwnd=1.0, initial_cwnd=2.0)
        assert ctrl.cwnd == 2.0


class TestAIMDSlowStart:
    """Test slow start phase (exponential growth)."""

    def test_success_at_capacity_doubles_cwnd(self):
        from znote_mcp.services.aimd import AIMDController

        ctrl = AIMDController(max_cwnd=10.0, min_cwnd=1.0)
        initial = ctrl.cwnd  # 5.0
        ctrl.on_success(utilization=0.85)
        assert ctrl.cwnd == min(initial * 2, ctrl.ssthresh)

    def test_success_below_threshold_no_change(self):
        from znote_mcp.services.aimd import AIMDController

        ctrl = AIMDController(max_cwnd=6.0, min_cwnd=1.0)
        initial = ctrl.cwnd
        ctrl.on_success(utilization=0.5)  # Below 0.75 threshold
        assert ctrl.cwnd == initial

    def test_slow_start_stops_at_ssthresh(self):
        from znote_mcp.services.aimd import AIMDController

        ctrl = AIMDController(max_cwnd=10.0, min_cwnd=1.0, initial_cwnd=1.0)
        ctrl._ssthresh = 3.0
        ctrl.on_success(utilization=0.8)  # 1.0 -> 2.0
        ctrl.on_success(utilization=0.8)  # 2.0 -> 3.0 (capped at ssthresh)
        assert ctrl.cwnd == 3.0
        assert ctrl.phase == "congestion_avoidance"


class TestAIMDCongestionAvoidance:
    """Test congestion avoidance phase (linear growth)."""

    def test_additive_increase_after_ssthresh(self):
        from znote_mcp.services.aimd import AIMDController

        ctrl = AIMDController(max_cwnd=10.0, min_cwnd=1.0)
        ctrl._phase = "congestion_avoidance"
        ctrl._cwnd = 5.0
        ctrl.on_success(utilization=0.8)
        assert ctrl.cwnd == 5.5  # increment = max_cwnd * 0.05

    def test_cwnd_capped_at_max(self):
        from znote_mcp.services.aimd import AIMDController

        ctrl = AIMDController(max_cwnd=6.0, min_cwnd=1.0)
        ctrl._phase = "congestion_avoidance"
        ctrl._cwnd = 5.8
        ctrl.on_success(utilization=0.8)
        assert ctrl.cwnd == 6.0


class TestAIMDFailure:
    """Test multiplicative decrease on failure."""

    def test_failure_halves_cwnd(self):
        from znote_mcp.services.aimd import AIMDController

        ctrl = AIMDController(max_cwnd=10.0, min_cwnd=1.0)
        ctrl._cwnd = 8.0
        ctrl.on_failure()
        assert ctrl.cwnd == 4.0

    def test_failure_sets_ssthresh(self):
        from znote_mcp.services.aimd import AIMDController

        ctrl = AIMDController(max_cwnd=10.0, min_cwnd=1.0)
        ctrl._cwnd = 8.0
        ctrl.on_failure()
        assert ctrl.ssthresh == 4.0

    def test_failure_exits_slow_start(self):
        from znote_mcp.services.aimd import AIMDController

        ctrl = AIMDController(max_cwnd=10.0, min_cwnd=1.0)
        assert ctrl.phase == "slow_start"
        ctrl.on_failure()
        assert ctrl.phase == "congestion_avoidance"

    def test_failure_respects_min_cwnd(self):
        from znote_mcp.services.aimd import AIMDController

        ctrl = AIMDController(max_cwnd=6.0, min_cwnd=1.0)
        ctrl._cwnd = 1.5
        ctrl.on_failure()
        assert ctrl.cwnd == 1.0  # min_cwnd floor

    def test_failure_at_floor_signals_breach(self):
        from znote_mcp.services.aimd import AIMDController

        ctrl = AIMDController(max_cwnd=6.0, min_cwnd=1.0)
        ctrl._cwnd = 1.0
        result = ctrl.on_failure()
        assert result.at_floor is True

    def test_multiple_failures_keep_halving(self):
        from znote_mcp.services.aimd import AIMDController

        ctrl = AIMDController(max_cwnd=10.0, min_cwnd=1.0, initial_cwnd=8.0)
        ctrl.on_failure()  # 8.0 -> 4.0
        assert ctrl.cwnd == 4.0
        ctrl.on_failure()  # 4.0 -> 2.0
        assert ctrl.cwnd == 2.0
        ctrl.on_failure()  # 2.0 -> 1.0
        assert ctrl.cwnd == 1.0


class TestAIMDCooldown:
    """Test cooldown phase transitions."""

    def test_enter_cooldown(self):
        from znote_mcp.services.aimd import AIMDController

        ctrl = AIMDController(max_cwnd=6.0, min_cwnd=1.0)
        ctrl.enter_cooldown()
        assert ctrl.phase == "cooldown"

    def test_exit_cooldown_resumes_congestion_avoidance(self):
        from znote_mcp.services.aimd import AIMDController

        ctrl = AIMDController(max_cwnd=6.0, min_cwnd=1.0)
        ctrl._cwnd = 2.0
        ctrl.enter_cooldown()
        ctrl.exit_cooldown()
        assert ctrl.phase == "congestion_avoidance"
        assert ctrl.cwnd == 2.0  # preserved

    def test_no_change_during_cooldown(self):
        from znote_mcp.services.aimd import AIMDController

        ctrl = AIMDController(max_cwnd=6.0, min_cwnd=1.0)
        ctrl._cwnd = 3.0
        ctrl.enter_cooldown()
        ctrl.on_success(utilization=0.9)
        assert ctrl.cwnd == 3.0  # no change during cooldown


class TestAIMDReset:
    """Test manual reset."""

    def test_reset_restores_initial_state(self):
        from znote_mcp.services.aimd import AIMDController

        ctrl = AIMDController(max_cwnd=6.0, min_cwnd=1.0)
        ctrl.on_failure()
        ctrl.on_failure()
        ctrl.reset()
        assert ctrl.cwnd == 3.0  # half of max
        assert ctrl.ssthresh == 6.0
        assert ctrl.phase == "slow_start"


class TestAIMDCautionSignal:
    """Test cross-component caution signal."""

    def test_receive_caution_caps_ssthresh(self):
        from znote_mcp.services.aimd import AIMDController

        ctrl = AIMDController(max_cwnd=10.0, min_cwnd=1.0)
        ctrl._cwnd = 8.0
        ctrl.receive_caution()
        assert ctrl.ssthresh == 4.0  # min(10.0, 8.0 / 2)
        assert ctrl.phase == "congestion_avoidance"

    def test_receive_caution_doesnt_reduce_cwnd(self):
        from znote_mcp.services.aimd import AIMDController

        ctrl = AIMDController(max_cwnd=10.0, min_cwnd=1.0)
        ctrl._cwnd = 8.0
        ctrl.receive_caution()
        assert ctrl.cwnd == 8.0  # cwnd preserved


class TestAIMDStateSnapshot:
    """Test state reporting for zk_status."""

    def test_snapshot_contains_all_fields(self):
        from znote_mcp.services.aimd import AIMDController

        ctrl = AIMDController(max_cwnd=6.0, min_cwnd=1.0)
        snap = ctrl.snapshot()
        assert "cwnd" in snap
        assert "ssthresh" in snap
        assert "phase" in snap
        assert "max_cwnd" in snap
        assert "min_cwnd" in snap
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_aimd.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'znote_mcp.services.aimd'`

**Step 3: Commit test file**

```bash
git add tests/test_aimd.py
git commit -m "test: add failing tests for AIMD controller core"
```

---

### Task 2: AIMD Controller — Implementation

**Files:**
- Create: `src/znote_mcp/services/aimd.py`

**Step 1: Implement AIMDController**

```python
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
        """Record a successful batch. Only grows if utilization >= threshold.

        Args:
            utilization: fraction of budget used (0.0 to 1.0).
        """
        if utilization < UTILIZATION_THRESHOLD:
            return

        with self._lock:
            if self._phase == "cooldown":
                return  # no changes during cooldown

            if self._phase == "slow_start":
                new = min(self._cwnd * 2, self._ssthresh)
                if new >= self._ssthresh:
                    self._phase = "congestion_avoidance"
                self._cwnd = min(new, self._max_cwnd)
            else:
                self._cwnd = min(self._cwnd + self._increment, self._max_cwnd)

    def on_failure(self) -> FailureResult:
        """Record a failure (OOM). Multiplicative decrease.

        Returns:
            FailureResult with new cwnd and whether we hit the floor.
        """
        with self._lock:
            new_cwnd = max(self._cwnd / 2, self._min_cwnd)
            self._ssthresh = new_cwnd
            self._cwnd = new_cwnd
            self._phase = "congestion_avoidance"

            hit_floor = self._cwnd <= self._min_cwnd
            logger.warning(
                "AIMD failure: cwnd %.1f -> %.1f GB (ssthresh=%.1f, floor=%s)",
                self._cwnd * 2,  # original
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
```

**Step 2: Run tests to verify they pass**

Run: `uv run pytest tests/test_aimd.py -v`
Expected: All PASS

**Step 3: Commit**

```bash
git add src/znote_mcp/services/aimd.py
git commit -m "feat: add AIMD adaptive memory budget controller"
```

---

### Task 3: Circuit Breaker — Failing Tests

**Files:**
- Create: `tests/test_circuit_breaker.py`

**Step 1: Write failing tests for CircuitBreaker**

```python
"""Tests for circuit breaker with sliding window error rate tracking."""

import time

import pytest


class TestCircuitBreakerInit:

    def test_starts_closed(self):
        from znote_mcp.services.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker()
        assert cb.state == "closed"

    def test_starts_on_gpu(self):
        from znote_mcp.services.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker()
        assert cb.provider == "gpu"


class TestCircuitBreakerTrip:

    def test_trips_after_threshold_failures_in_window(self):
        from znote_mcp.services.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(trip_threshold=3, window_seconds=60)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "closed"
        cb.record_failure()
        assert cb.state == "open"

    def test_failures_outside_window_dont_trip(self):
        from znote_mcp.services.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(trip_threshold=3, window_seconds=0.1)
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.15)
        cb.record_failure()
        assert cb.state == "closed"  # first two expired

    def test_trip_switches_to_cpu(self):
        from znote_mcp.services.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(trip_threshold=2, window_seconds=60)
        cb.record_failure()
        cb.record_failure()
        assert cb.provider == "cpu"

    def test_budget_floor_breach_trips(self):
        from znote_mcp.services.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker()
        cb.on_budget_floor_breach()
        assert cb.state == "open"
        assert cb.provider == "cpu"


class TestCircuitBreakerCooldown:

    def test_cooldown_duration_default(self):
        from znote_mcp.services.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(cooldown_seconds=120)
        assert cb.cooldown_seconds == 120

    def test_cooldown_remaining_decreases(self):
        from znote_mcp.services.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(trip_threshold=1, cooldown_seconds=60)
        cb.record_failure()
        remaining = cb.cooldown_remaining
        assert 0 < remaining <= 60

    def test_cooldown_expired_allows_gpu_retry(self):
        from znote_mcp.services.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(trip_threshold=1, cooldown_seconds=0.1)
        cb.record_failure()
        assert cb.state == "open"
        time.sleep(0.15)
        assert cb.is_cooldown_expired is True

    def test_escalation_doubles_cooldown(self):
        from znote_mcp.services.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(
            trip_threshold=1, cooldown_seconds=10, max_cooldown_seconds=600
        )
        cb.record_failure()  # Trip 1
        assert cb.cooldown_seconds == 10
        cb.reset_to_closed()  # Simulate GPU retry
        cb.record_failure()  # Trip 2 without GPU success
        assert cb.cooldown_seconds == 20

    def test_escalation_capped_at_max(self):
        from znote_mcp.services.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(
            trip_threshold=1, cooldown_seconds=300, max_cooldown_seconds=600
        )
        cb.record_failure()
        cb.reset_to_closed()
        cb.record_failure()
        assert cb.cooldown_seconds == 600  # capped

    def test_gpu_success_resets_escalation(self):
        from znote_mcp.services.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(trip_threshold=1, cooldown_seconds=10)
        cb.record_failure()  # Trip 1, cooldown=10
        cb.reset_to_closed()
        cb.record_gpu_success()  # GPU works!
        cb.record_failure()  # Trip again
        assert cb.cooldown_seconds == 10  # NOT escalated


class TestCircuitBreakerDisable:

    def test_cpu_trip_disables_component(self):
        from znote_mcp.services.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(trip_threshold=1)
        cb.record_failure()  # GPU trip -> CPU
        assert cb.provider == "cpu"
        cb.on_cpu_failure()
        assert cb.state == "disabled"

    def test_disabled_means_not_enabled(self):
        from znote_mcp.services.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(trip_threshold=1)
        cb.record_failure()
        cb.on_cpu_failure()
        assert cb.is_enabled is False


class TestCircuitBreakerReset:

    def test_manual_reset_clears_everything(self):
        from znote_mcp.services.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(trip_threshold=1, cooldown_seconds=10)
        cb.record_failure()
        cb.on_cpu_failure()
        assert cb.state == "disabled"
        cb.manual_reset()
        assert cb.state == "closed"
        assert cb.provider == "gpu"

    def test_force_cpu_sets_provider(self):
        from znote_mcp.services.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker()
        cb.force_cpu()
        assert cb.provider == "cpu"
        assert cb.state == "forced_cpu"


class TestCircuitBreakerSnapshot:

    def test_snapshot_contains_all_fields(self):
        from znote_mcp.services.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker()
        snap = cb.snapshot()
        assert "state" in snap
        assert "provider" in snap
        assert "trip_count" in snap
        assert "cooldown_remaining" in snap
        assert "failures_in_window" in snap
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_circuit_breaker.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Commit**

```bash
git add tests/test_circuit_breaker.py
git commit -m "test: add failing tests for circuit breaker"
```

---

### Task 4: Circuit Breaker — Implementation

**Files:**
- Create: `src/znote_mcp/services/circuit_breaker.py`

**Step 1: Implement CircuitBreaker**

```python
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
```

**Step 2: Run tests**

Run: `uv run pytest tests/test_circuit_breaker.py -v`
Expected: All PASS

**Step 3: Commit**

```bash
git add src/znote_mcp/services/circuit_breaker.py
git commit -m "feat: add circuit breaker for GPU/CPU provider switching"
```

---

### Task 5: Resilience Coordinator — Failing Tests

**Files:**
- Create: `tests/test_resilience_coordinator.py`

**Step 1: Write failing tests**

```python
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
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_resilience_coordinator.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Commit**

```bash
git add tests/test_resilience_coordinator.py
git commit -m "test: add failing tests for resilience coordinator"
```

---

### Task 6: Resilience Coordinator — Implementation

**Files:**
- Create: `src/znote_mcp/services/resilience_coordinator.py`

Implement `ResilienceCoordinator` that:
- Owns one `AIMDController` + `CircuitBreaker` per component
- Handles cross-component caution signals
- Emits structured events on state transitions
- Accumulates pending notices that tool responses can drain
- Provides `snapshot()` for `zk_status`
- Provides `reset_all()`, `force_cpu()`, `disable()`, `enable()` for agent controls

Follow the patterns established by the AIMD and circuit breaker classes. Thread-safe via the underlying component locks.

**Step 1: Implement**

*(Full implementation follows the test contract above — coordinator delegates to AIMD + breaker, adds cross-linking and event emission.)*

**Step 2: Run tests**

Run: `uv run pytest tests/test_resilience_coordinator.py -v`
Expected: All PASS

**Step 3: Commit**

```bash
git add src/znote_mcp/services/resilience_coordinator.py
git commit -m "feat: add resilience coordinator with cross-component linking"
```

---

### Task 7: Integration — Wire into EmbeddingService

**Files:**
- Modify: `src/znote_mcp/services/embedding_service.py`
- Modify: `src/znote_mcp/hardware.py`

**Step 1: Update hardware.py**

Add `aimd_max_budget_gb` field to `TuningResult` (same value as `embedding_memory_budget_gb`). This makes the AIMD ceiling explicit.

**Step 2: Update EmbeddingService.__init__**

Replace `OnnxResilienceManager` with `ResilienceCoordinator`:
- Change import from `resilience` to `resilience_coordinator`
- Pass `max_budget_gb` from hardware tuning (config.embedding_memory_budget_gb)
- Wire `on_event` callback for MCP notifications

**Step 3: Update embed/embed_batch/embed_batch_adaptive/rerank methods**

Replace:
- `self.resilience.is_embedder_enabled` → `self.coordinator.is_embedder_enabled`
- `self.resilience.advance_embedder()` → `self.coordinator.on_embedder_failure()`
- `self.resilience.embedder_batch_size` → budget from `self.coordinator.embedder_aimd.cwnd`
- `embed_batch_adaptive` memory_budget_gb parameter → `self.coordinator.embedder_aimd.cwnd`
- On success: call `self.coordinator.on_embedder_success(utilization=...)`
- Same pattern for reranker methods

**Step 4: Update CPU switching**

Replace `_switch_embedder_to_cpu` / `_switch_reranker_to_cpu` to be triggered by circuit breaker state instead of degradation level.

**Step 5: Update AIMD state preservation across idle**

In `_idle_unload_embedder` / `_idle_unload_reranker`: do NOT reset AIMD state. State persists even when model is unloaded.

**Step 6: Run existing tests**

Run: `uv run pytest tests/test_resilience.py tests/test_embedding_phase1.py tests/test_embedding_phase2.py -v`
Expected: Some failures from old resilience API — update tests in next task.

**Step 7: Commit**

```bash
git add src/znote_mcp/services/embedding_service.py src/znote_mcp/hardware.py
git commit -m "feat: wire AIMD resilience coordinator into embedding service"
```

---

### Task 8: Update Existing Resilience Tests

**Files:**
- Modify: `tests/test_resilience.py`
- Modify: `tests/test_mcp_resilience_protocol.py`

**Step 1: Update test_resilience.py**

- `TestResilienceManager` class: replace with tests against new `ResilienceCoordinator`
- `TestEmbeddingServiceResilience`: update to use coordinator API
- `TestCpuFallback`: update to use circuit breaker state instead of `DegradationLevel`
- `TestAdaptiveBatchResilience`: update to check AIMD budget instead of batch_size
- Keep `FakeFailingProvider` and `FakeFailingReranker` — still useful
- `TestEndToEndResilience`: rewrite to test AIMD → circuit breaker → CPU → recovery flow

**Step 2: Update test_mcp_resilience_protocol.py**

Update protocol-level tests to reflect new status output format and new agent control actions.

**Step 3: Run all tests**

Run: `uv run pytest tests/ -v --ignore=tests/test_e2e.py`
Expected: All PASS

**Step 4: Commit**

```bash
git add tests/test_resilience.py tests/test_mcp_resilience_protocol.py
git commit -m "test: update resilience tests for AIMD coordinator"
```

---

### Task 9: Agent Signaling — Inline Notices

**Files:**
- Modify: `src/znote_mcp/server/mcp_server.py`

**Step 1: Add notice formatting helper**

Create a method `_format_resilience_notice(event: dict) -> str` that produces the structured notice block from a state transition event.

**Step 2: Add notice prepending to tool responses**

In each tool handler that touches embeddings (at minimum: `zk_search_notes`, `zk_create_note`, `zk_update_note`, `zk_system` reindex), drain pending notices from the coordinator and prepend to the response string.

Pattern:
```python
notices = self.coordinator.drain_pending_notices()
notice_text = "".join(self._format_resilience_notice(e) for e in notices)
return notice_text + normal_response
```

**Step 3: Write tests for notice formatting**

Test that notices are prepended on state transitions and absent on steady-state calls.

**Step 4: Run tests**

Run: `uv run pytest tests/test_mcp_resilience_protocol.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/znote_mcp/server/mcp_server.py
git commit -m "feat: prepend inline resilience notices on state transitions"
```

---

### Task 10: Agent Controls — zk_system Actions

**Files:**
- Modify: `src/znote_mcp/server/mcp_server.py`

**Step 1: Add new zk_system actions**

In the `zk_system` handler's action dispatch, add:
- `embedding_reset` → `coordinator.reset_all()`, return status
- `embedding_force_cpu` → `coordinator.force_cpu_all()`, return status
- `embedding_disable` → `coordinator.disable_all()`, return status
- `embedding_enable` → `coordinator.enable_all()`, return status

**Step 2: Enhanced zk_status embeddings section**

In the `zk_status` handler's embeddings section, replace the old resilience output with coordinator snapshot data: AIMD state, circuit breaker state, provider, last event, session stats.

**Step 3: Write tests**

Test each action through the MCP protocol and verify status output.

**Step 4: Run tests**

Run: `uv run pytest tests/test_mcp_resilience_protocol.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/znote_mcp/server/mcp_server.py
git commit -m "feat: add embedding control actions and enhanced status reporting"
```

---

### Task 11: Remove Old Resilience Manager

**Files:**
- Delete: `src/znote_mcp/services/resilience.py`
- Modify: any remaining imports

**Step 1: Remove resilience.py**

```bash
git rm src/znote_mcp/services/resilience.py
```

**Step 2: Search for stale imports**

```bash
grep -r "from znote_mcp.services.resilience import" src/ tests/
grep -r "OnnxResilienceManager\|DegradationLevel" src/ tests/
```

Fix any remaining references.

**Step 3: Run full test suite**

Run: `uv run pytest tests/ -v --ignore=tests/test_e2e.py`
Expected: All PASS

**Step 4: Commit**

```bash
git add -A
git commit -m "refactor: remove old OnnxResilienceManager"
```

---

### Task 12: E2E and Protocol Integration Tests

**Files:**
- Modify: `tests/test_mcp_resilience_protocol.py`
- Optionally create: `tests/test_aimd_e2e.py`

**Step 1: Protocol-level AIMD tests**

Test through the full MCP JSON-RPC pipeline:
- Simulate OOM → verify AIMD halve → verify notice in search response
- Simulate recovery → verify notice on full capacity
- Test `zk_system(action="embedding_reset")` clears state
- Test `zk_system(action="embedding_force_cpu")` switches provider
- Test `zk_status(sections="embeddings")` shows AIMD + breaker state

**Step 2: Circuit breaker integration test**

- Rapid-fire OOM simulation → verify breaker trips → verify CPU fallback → verify cooldown → verify GPU retry

**Step 3: Cross-component test**

- Embedder pressure → verify reranker receives caution signal → verify reranker exits slow start

**Step 4: Run full suite**

Run: `uv run pytest tests/ -v --ignore=tests/test_e2e.py`
Expected: All PASS

**Step 5: Commit**

```bash
git add tests/
git commit -m "test: AIMD resilience protocol and integration tests"
```

---

### Task 13: Documentation Update

**Files:**
- Modify: `CLAUDE.md`
- Modify: `README.md` (if resilience section exists)

**Step 1: Update CLAUDE.md architecture section**

Replace references to 5-level staircase with AIMD + circuit breaker architecture. Update the layer diagram, key files table, and environment variables section.

**Step 2: Update README if needed**

If README mentions resilience or degradation levels, update to reflect new system.

**Step 3: Commit**

```bash
git add CLAUDE.md README.md
git commit -m "docs: update architecture docs for AIMD resilience"
```
