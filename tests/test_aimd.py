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
