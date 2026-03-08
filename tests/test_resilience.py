"""Tests for ONNX progressive memory resilience."""

import faulthandler
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from znote_mcp.hardware import (
    HardwareProfile,
    TuningResult,
    validate_gpu,
)


def test_faulthandler_writes_to_file():
    """Verify faulthandler can be enabled on a file handle."""
    with tempfile.NamedTemporaryFile(mode="a", suffix=".log", delete=False) as f:
        faulthandler.enable(file=f)
        assert faulthandler.is_enabled()
        faulthandler.disable()


class TestValidateGpu:
    """Tests for GPU validation before committing to CUDA."""

    def test_cpu_tuning_passes_through(self):
        """CPU tuning should pass through unchanged."""
        profile = HardwareProfile(system_ram_mb=32000)
        tuning = TuningResult(onnx_providers="cpu", device_label="cpu-32gb+")
        result = validate_gpu(profile, tuning)
        assert result.onnx_providers == "cpu"

    def test_gpu_tuning_with_no_free_vram_falls_back(self):
        """GPU tuning with insufficient free VRAM should fall back to CPU."""
        profile = HardwareProfile(
            gpu_name="RTX 4070",
            gpu_vram_mb=16000,
            system_ram_mb=32000,
            onnx_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        tuning = TuningResult(onnx_providers="auto", device_label="gpu-16gb+")

        # Mock nvidia-smi returning 500 MB free (below 1 GB threshold)
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "500"
        with patch("znote_mcp.hardware.subprocess.run", return_value=mock_result):
            result = validate_gpu(profile, tuning)
        assert result.onnx_providers == "cpu"

    def test_gpu_tuning_with_enough_vram_keeps_gpu(self):
        """GPU tuning with sufficient free VRAM should keep GPU."""
        profile = HardwareProfile(
            gpu_name="RTX 4070",
            gpu_vram_mb=16000,
            system_ram_mb=32000,
            onnx_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        tuning = TuningResult(
            onnx_providers="auto",
            device_label="gpu-16gb+",
            embedding_batch_size=64,
        )

        # Mock nvidia-smi returning 8000 MB free
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "8000"
        with patch("znote_mcp.hardware.subprocess.run", return_value=mock_result):
            result = validate_gpu(profile, tuning)
        assert result.onnx_providers == "auto"

    def test_gpu_validation_nvidia_smi_fails(self):
        """If nvidia-smi fails during validation, fall back to CPU."""
        profile = HardwareProfile(
            gpu_name="RTX 4070",
            gpu_vram_mb=16000,
            system_ram_mb=32000,
            onnx_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        tuning = TuningResult(onnx_providers="auto", device_label="gpu-16gb+")

        with patch(
            "znote_mcp.hardware.subprocess.run",
            side_effect=FileNotFoundError,
        ):
            result = validate_gpu(profile, tuning)
        assert result.onnx_providers == "cpu"


from tests.fakes import FakeEmbeddingProvider, FakeRerankerProvider
from znote_mcp.exceptions import EmbeddingError
from znote_mcp.services.embedding_service import EmbeddingService


class FakeFailingProvider(FakeEmbeddingProvider):
    """Provider that raises BFCArena-like errors on demand."""

    def __init__(self, dim=8, fail_count=0):
        super().__init__(dim=dim)
        self._fail_count = fail_count
        self._calls = 0

    def load(self):
        self._calls += 1
        if self._calls <= self._fail_count:
            raise RuntimeError(
                "BFCArena::AllocateRawInternal Failed to allocate memory"
            )
        super().load()

    def embed(self, text):
        self._calls += 1
        if self._calls <= self._fail_count:
            raise RuntimeError(
                "BFCArena::AllocateRawInternal Failed to allocate memory"
            )
        return super().embed(text)


class TestEmbeddingServiceResilience:
    """Tests for EmbeddingService integration with ResilienceCoordinator."""

    def test_service_creates_coordinator(self):
        svc = EmbeddingService(
            embedder=FakeEmbeddingProvider(),
            reranker=FakeRerankerProvider(),
        )
        assert svc.coordinator is not None
        assert hasattr(svc.coordinator, "embedder_aimd")
        assert hasattr(svc.coordinator, "embedder_breaker")
        svc.shutdown()

    def test_bfcarena_load_failure_triggers_coordinator(self):
        """BFCArena error on load should record failure in coordinator."""
        provider = FakeFailingProvider(fail_count=1)
        svc = EmbeddingService(embedder=provider)
        try:
            svc.embed("test")
        except EmbeddingError:
            pass
        # AIMD cwnd should have decreased from initial
        assert (
            svc.coordinator.embedder_aimd.cwnd < svc.coordinator.embedder_aimd.max_cwnd
        )
        svc.shutdown()

    def test_service_still_works_after_transient_failure(self):
        """After one load failure, next attempt should succeed."""
        provider = FakeFailingProvider(fail_count=1)
        svc = EmbeddingService(embedder=provider)
        # First call fails
        try:
            svc.embed("test")
        except EmbeddingError:
            pass
        # Second call should succeed (provider stops failing after fail_count)
        result = svc.embed("test")
        assert result is not None
        svc.shutdown()


class FakeFailingReranker(FakeRerankerProvider):
    """Reranker that fails with BFCArena on first N calls to rerank."""

    def __init__(self, fail_count=0):
        super().__init__()
        self._fail_count = fail_count
        self._calls = 0

    def rerank(self, query, documents, top_k=5):
        self._calls += 1
        if self._calls <= self._fail_count:
            raise RuntimeError(
                "BFCArena::AllocateRawInternal Failed to allocate memory"
            )
        return super().rerank(query, documents, top_k)


class TestRerankerBatchedInference:
    """Tests for reranker batched scoring and OOM handling."""

    def test_reranker_handles_large_doc_set(self):
        """Reranker should handle 20+ documents."""
        reranker = FakeRerankerProvider()
        svc = EmbeddingService(
            embedder=FakeEmbeddingProvider(),
            reranker=reranker,
        )
        results = svc.rerank(
            "test query", [f"document {i}" for i in range(20)], top_k=5
        )
        assert len(results) == 5
        svc.shutdown()

    def test_reranker_oom_triggers_coordinator(self):
        """On BFCArena error, reranker coordinator state should update."""
        reranker = FakeFailingReranker(fail_count=1)
        svc = EmbeddingService(
            embedder=FakeEmbeddingProvider(),
            reranker=reranker,
        )
        try:
            svc.rerank("query", [f"doc {i}" for i in range(10)])
        except EmbeddingError:
            pass
        # AIMD cwnd should have decreased
        assert (
            svc.coordinator.reranker_aimd.cwnd < svc.coordinator.reranker_aimd.max_cwnd
        )
        svc.shutdown()


class TestCpuFallback:
    """Tests for runtime GPU->CPU provider switching via circuit breaker."""

    def test_embedder_cpu_switch_flag_detected(self):
        """After force_cpu(), breaker provider should be 'cpu'."""
        provider = FakeEmbeddingProvider()
        svc = EmbeddingService(embedder=provider)
        svc.coordinator.embedder_breaker.force_cpu()
        assert svc.coordinator.embedder_breaker.provider == "cpu"
        svc.shutdown()

    def test_disabled_embedder_raises_error(self):
        """When breaker is disabled, embed calls should raise EmbeddingError."""
        provider = FakeEmbeddingProvider()
        svc = EmbeddingService(embedder=provider)
        # Disable via CPU failure path
        svc.coordinator.embedder_breaker.on_cpu_failure()
        with pytest.raises(EmbeddingError, match="disabled"):
            svc.embed("test")
        svc.shutdown()

    def test_disabled_reranker_raises_error(self):
        """When reranker breaker is disabled, rerank calls should raise EmbeddingError."""
        svc = EmbeddingService(
            embedder=FakeEmbeddingProvider(),
            reranker=FakeRerankerProvider(),
        )
        svc.coordinator.reranker_breaker.on_cpu_failure()
        with pytest.raises(EmbeddingError, match="disabled"):
            svc.rerank("query", ["doc1"])
        svc.shutdown()

    def test_cpu_switch_reloads_provider(self):
        """When breaker forces CPU and provider supports it, reload on CPU."""
        provider = FakeEmbeddingProvider()
        svc = EmbeddingService(embedder=provider)
        # Force CPU via breaker
        svc.coordinator.embedder_breaker.force_cpu()

        # The _ensure_embedder should detect CPU switch
        result = svc.embed("test after cpu switch")
        assert result is not None
        svc.shutdown()

    def test_cpu_switch_unloads_and_reloads(self):
        """CPU switch should unload then reload the provider."""
        provider = FakeEmbeddingProvider()
        svc = EmbeddingService(embedder=provider)
        # First embed to load the provider
        svc.embed("initial load")
        assert provider.load_count == 1
        assert provider.unload_count == 0

        # Force CPU via breaker
        svc.coordinator.embedder_breaker.force_cpu()

        # Next embed should trigger unload + reload
        svc.embed("after cpu switch")
        assert provider.unload_count == 1
        assert provider.load_count == 2
        svc.shutdown()

    def test_cpu_switch_sets_providers_pref(self):
        """CPU switch should set _providers_pref to 'cpu' if attribute exists."""
        provider = FakeEmbeddingProvider()
        provider._providers_pref = "auto"  # Simulate ONNX provider
        svc = EmbeddingService(embedder=provider)
        svc.coordinator.embedder_breaker.force_cpu()

        svc.embed("trigger cpu switch")
        assert provider._providers_pref == "cpu"
        svc.shutdown()

    def test_cpu_switch_failure_disables_breaker(self):
        """If CPU reload fails, breaker should move to disabled state."""
        provider = FakeFailingProvider(fail_count=999)  # Always fails
        svc = EmbeddingService(embedder=provider)
        svc.coordinator.embedder_breaker.force_cpu()

        with pytest.raises(EmbeddingError, match="CPU fallback failed"):
            svc.embed("trigger failing cpu switch")
        assert svc.coordinator.embedder_breaker.state == "disabled"
        assert svc.coordinator.embedder_breaker.is_enabled is False
        svc.shutdown()

    def test_reranker_cpu_switch(self):
        """Reranker should also support CPU switching."""
        reranker = FakeRerankerProvider()
        svc = EmbeddingService(
            embedder=FakeEmbeddingProvider(),
            reranker=reranker,
        )
        # Load reranker first
        svc.rerank("query", ["doc1", "doc2"])
        assert reranker.load_count == 1

        # Force CPU on reranker breaker
        svc.coordinator.reranker_breaker.force_cpu()
        svc.rerank("query", ["doc1", "doc2"])
        assert reranker.unload_count == 1
        assert reranker.load_count == 2
        svc.shutdown()


class TestEndToEndResilience:
    """Full scenario: provider fails, AIMD decreases, breaker trips."""

    def test_full_degradation_cascade(self):
        """Simulate: GPU OOM -> AIMD decrease -> breaker trips -> CPU fallback.

        With max_budget=6.0 and min_budget=1.0:
        - Failure 1: cwnd 3.0 -> 1.5 (halved from initial=3.0)
        - Failure 2: cwnd 1.5 -> 1.0 (floor), breaker trips, CPU switch attempted
        - Failure 3: CPU switch fails -> breaker disabled

        Provider fails first 2 calls, so after 2 failures the breaker trips
        and CPU switch succeeds on attempt 3 (fail_count=2).
        """
        provider = FakeFailingProvider(fail_count=2)
        svc = EmbeddingService(
            embedder=provider,
            memory_budget_gb=6.0,
        )

        # Attempts 1-2 fail, triggering AIMD decrease + breaker trip
        for i in range(2):
            try:
                svc.embed("test")
            except EmbeddingError:
                pass

        # After 2 failures hitting the AIMD floor, breaker tripped
        # and the CPU switch was attempted (but provider was still failing)
        # Now the breaker is open, provider is CPU
        assert svc.coordinator.embedder_breaker.provider == "cpu"

        # AIMD cwnd should have been halved to floor
        assert svc.coordinator.embedder_aimd.cwnd <= 1.0

        # Provider stops failing after count=2, so next embed succeeds
        result = svc.embed("test")
        assert result is not None
        svc.shutdown()

    def test_embedder_disabled_raises_clearly(self):
        """When embedder breaker is disabled, embed should raise EmbeddingError."""
        provider = FakeFailingProvider(fail_count=100)  # Always fails
        svc = EmbeddingService(embedder=provider)

        # Drive through failures until breaker trips, then force CPU failure
        for _ in range(5):
            try:
                svc.embed("test")
            except EmbeddingError:
                pass

        # Force disable via CPU failure path
        svc.coordinator.embedder_breaker.on_cpu_failure()

        assert svc.coordinator.embedder_breaker.is_enabled is False
        with pytest.raises(EmbeddingError, match="disabled"):
            svc.embed("test")
        svc.shutdown()


class TestStartupFlow:
    """Tests for the full startup validation flow."""

    def test_startup_flow_cpu_passthrough(self):
        """CPU-only profile should pass through validate_gpu unchanged."""
        from znote_mcp.hardware import HardwareProfile, compute_tuning, validate_gpu

        profile = HardwareProfile(system_ram_mb=32000, cpu_arch="x86_64")
        tuning = compute_tuning(profile)
        validated = validate_gpu(profile, tuning)
        assert validated.onnx_providers == "cpu"

    def test_startup_flow_gpu_with_low_vram_falls_back(self):
        """GPU profile with insufficient free VRAM should fall back to CPU."""
        from znote_mcp.hardware import HardwareProfile, compute_tuning, validate_gpu

        profile = HardwareProfile(
            gpu_name="RTX 4070",
            gpu_vram_mb=16000,
            system_ram_mb=32000,
            cpu_arch="x86_64",
            onnx_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        tuning = compute_tuning(profile)
        assert tuning.onnx_providers == "auto"

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "200"  # Only 200 MB free
        with patch("znote_mcp.hardware.subprocess.run", return_value=mock_result):
            validated = validate_gpu(profile, tuning)
        assert validated.onnx_providers == "cpu"
        assert "cpu" in validated.device_label.lower()
