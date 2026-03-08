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
