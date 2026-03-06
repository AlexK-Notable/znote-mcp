"""Tests for ONNX progressive memory resilience."""

import faulthandler
import tempfile
from unittest.mock import MagicMock, patch

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


from znote_mcp.services.resilience import OnnxResilienceManager, DegradationLevel


class TestResilienceManager:
    """Tests for the OnnxResilienceManager state machine."""

    def test_initial_level_is_normal(self):
        mgr = OnnxResilienceManager(
            initial_batch_size=64,
            initial_max_tokens=8192,
        )
        assert mgr.embedder_level == DegradationLevel.NORMAL
        assert mgr.reranker_level == DegradationLevel.NORMAL

    def test_advance_embedder_reduces_batch(self):
        mgr = OnnxResilienceManager(
            initial_batch_size=64,
            initial_max_tokens=8192,
        )
        mgr.advance_embedder()
        assert mgr.embedder_level == DegradationLevel.REDUCED_BATCH
        assert mgr.embedder_batch_size == 32

    def test_advance_embedder_twice_reduces_tokens(self):
        mgr = OnnxResilienceManager(
            initial_batch_size=64,
            initial_max_tokens=8192,
        )
        mgr.advance_embedder()
        mgr.advance_embedder()
        assert mgr.embedder_level == DegradationLevel.REDUCED_TOKENS
        assert mgr.embedder_batch_size == 32
        assert mgr.embedder_max_tokens == 4096

    def test_advance_to_cpu_fallback(self):
        mgr = OnnxResilienceManager(
            initial_batch_size=64,
            initial_max_tokens=8192,
        )
        mgr.advance_embedder()
        mgr.advance_embedder()
        mgr.advance_embedder()
        assert mgr.embedder_level == DegradationLevel.CPU_FALLBACK

    def test_advance_to_disabled(self):
        mgr = OnnxResilienceManager(
            initial_batch_size=64,
            initial_max_tokens=8192,
        )
        for _ in range(4):
            mgr.advance_embedder()
        assert mgr.embedder_level == DegradationLevel.DISABLED

    def test_advance_past_disabled_stays_disabled(self):
        mgr = OnnxResilienceManager(
            initial_batch_size=64,
            initial_max_tokens=8192,
        )
        for _ in range(10):
            mgr.advance_embedder()
        assert mgr.embedder_level == DegradationLevel.DISABLED

    def test_reranker_independent_of_embedder(self):
        mgr = OnnxResilienceManager(
            initial_batch_size=64,
            initial_max_tokens=8192,
        )
        mgr.advance_reranker()
        assert mgr.reranker_level == DegradationLevel.REDUCED_BATCH
        assert mgr.embedder_level == DegradationLevel.NORMAL

    def test_is_component_enabled(self):
        mgr = OnnxResilienceManager(
            initial_batch_size=64,
            initial_max_tokens=8192,
        )
        assert mgr.is_embedder_enabled is True
        for _ in range(4):
            mgr.advance_embedder()
        assert mgr.is_embedder_enabled is False

    def test_needs_cpu_fallback(self):
        mgr = OnnxResilienceManager(
            initial_batch_size=64,
            initial_max_tokens=8192,
        )
        assert mgr.embedder_needs_cpu_switch is False
        for _ in range(3):
            mgr.advance_embedder()
        assert mgr.embedder_needs_cpu_switch is True

    def test_notification_callback_called_on_advance(self):
        notifications = []
        mgr = OnnxResilienceManager(
            initial_batch_size=64,
            initial_max_tokens=8192,
            on_notify=lambda level, msg: notifications.append((level, msg)),
        )
        mgr.advance_embedder()
        assert len(notifications) == 1
        assert "batch size" in notifications[0][1].lower()


from tests.fakes import FakeEmbeddingProvider, FakeRerankerProvider
from znote_mcp.services.embedding_service import EmbeddingService
from znote_mcp.exceptions import EmbeddingError


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
    """Tests for EmbeddingService integration with resilience manager."""

    def test_service_creates_resilience_manager(self):
        svc = EmbeddingService(
            embedder=FakeEmbeddingProvider(),
            reranker=FakeRerankerProvider(),
        )
        assert svc.resilience is not None
        svc.shutdown()

    def test_bfcarena_load_failure_advances_level(self):
        """BFCArena error on load should advance degradation level."""
        provider = FakeFailingProvider(fail_count=1)
        svc = EmbeddingService(embedder=provider)
        try:
            svc.embed("test")
        except EmbeddingError:
            pass
        assert svc.resilience.embedder_level.value >= 1
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
        results = svc.rerank("test query", [f"document {i}" for i in range(20)], top_k=5)
        assert len(results) == 5
        svc.shutdown()

    def test_reranker_oom_advances_resilience_level(self):
        """On BFCArena error, reranker resilience level should advance."""
        reranker = FakeFailingReranker(fail_count=1)
        svc = EmbeddingService(
            embedder=FakeEmbeddingProvider(),
            reranker=reranker,
        )
        try:
            svc.rerank("query", [f"doc {i}" for i in range(10)])
        except EmbeddingError:
            pass
        assert svc.resilience.reranker_level.value >= 1
        svc.shutdown()


class TestCpuFallback:
    """Tests for runtime GPU→CPU provider switching."""

    def test_embedder_cpu_switch_flag_detected(self):
        """After advancing to CPU_FALLBACK, needs_cpu_switch should be True."""
        provider = FakeEmbeddingProvider()
        svc = EmbeddingService(embedder=provider)
        # Force to level 3 (CPU_FALLBACK)
        svc.resilience._embedder_level = DegradationLevel.CPU_FALLBACK
        assert svc.resilience.embedder_needs_cpu_switch is True
        svc.shutdown()

    def test_disabled_embedder_raises_error(self):
        """At DISABLED level, embed calls should raise EmbeddingError."""
        provider = FakeEmbeddingProvider()
        svc = EmbeddingService(embedder=provider)
        # Force to disabled
        svc.resilience._embedder_level = DegradationLevel.DISABLED
        import pytest
        with pytest.raises(EmbeddingError, match="disabled"):
            svc.embed("test")
        svc.shutdown()

    def test_disabled_reranker_raises_error(self):
        """At DISABLED level, rerank calls should raise EmbeddingError."""
        svc = EmbeddingService(
            embedder=FakeEmbeddingProvider(),
            reranker=FakeRerankerProvider(),
        )
        svc.resilience._reranker_level = DegradationLevel.DISABLED
        import pytest
        with pytest.raises(EmbeddingError, match="disabled"):
            svc.rerank("query", ["doc1"])
        svc.shutdown()

    def test_cpu_switch_reloads_provider(self):
        """When needs_cpu_switch is True and provider supports it, reload on CPU."""
        provider = FakeEmbeddingProvider()
        svc = EmbeddingService(embedder=provider)
        # Simulate: provider loaded on GPU, then resilience hits CPU_FALLBACK
        svc.resilience._embedder_level = DegradationLevel.CPU_FALLBACK

        # The _ensure_embedder should detect needs_cpu_switch
        # For fake provider (no _providers_pref), it should still work
        # by unloading and reloading
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

        # Force CPU_FALLBACK
        svc.resilience._embedder_level = DegradationLevel.CPU_FALLBACK

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
        svc.resilience._embedder_level = DegradationLevel.CPU_FALLBACK

        svc.embed("trigger cpu switch")
        assert provider._providers_pref == "cpu"
        svc.shutdown()

    def test_cpu_switch_failure_advances_to_disabled(self):
        """If CPU reload fails, should advance to DISABLED."""
        provider = FakeFailingProvider(fail_count=999)  # Always fails
        svc = EmbeddingService(embedder=provider)
        svc.resilience._embedder_level = DegradationLevel.CPU_FALLBACK

        import pytest
        with pytest.raises(EmbeddingError, match="CPU fallback failed"):
            svc.embed("trigger failing cpu switch")
        assert svc.resilience.embedder_level == DegradationLevel.DISABLED
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

        # Force CPU_FALLBACK on reranker
        svc.resilience._reranker_level = DegradationLevel.CPU_FALLBACK
        svc.rerank("query", ["doc1", "doc2"])
        assert reranker.unload_count == 1
        assert reranker.load_count == 2
        svc.shutdown()


class TestMcpNotifications:
    """Tests for MCP notification wiring."""

    def test_notification_callback_receives_messages(self):
        """Verify notifications flow from resilience manager through EmbeddingService."""
        messages = []

        def on_notify(level, msg):
            messages.append((level, msg))

        svc = EmbeddingService(
            embedder=FakeEmbeddingProvider(),
            on_notify=on_notify,
        )
        svc.resilience.advance_embedder()
        assert len(messages) == 1
        assert messages[0][0] == "warning"
        svc.shutdown()

    def test_notification_on_multiple_advances(self):
        """Each advance should produce a notification."""
        messages = []
        svc = EmbeddingService(
            embedder=FakeEmbeddingProvider(),
            on_notify=lambda level, msg: messages.append((level, msg)),
        )
        svc.resilience.advance_embedder()
        svc.resilience.advance_embedder()
        svc.resilience.advance_embedder()
        assert len(messages) == 3
        svc.shutdown()
