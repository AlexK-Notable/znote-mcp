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
