"""Tests for hardware detection and auto-tuning."""

import os
from unittest.mock import patch

import pytest

from znote_mcp.hardware import (
    HardwareProfile,
    TuningResult,
    apply_tuning,
    compute_tuning,
    detect_hardware,
)

# Env vars that apply_tuning checks — must be cleared in tests that
# expect auto-tuned values to be written (dotenv loads them at import time).
_TUNING_ENV_VARS = [
    "ZETTELKASTEN_ONNX_PROVIDERS",
    "ZETTELKASTEN_EMBEDDING_BATCH_SIZE",
    "ZETTELKASTEN_EMBEDDING_MAX_TOKENS",
    "ZETTELKASTEN_RERANKER_MAX_TOKENS",
    "ZETTELKASTEN_EMBEDDING_MEMORY_BUDGET_GB",
]


def _env_without_tuning_vars() -> dict[str, str]:
    """Return os.environ copy with all tuning env vars removed."""
    return {k: v for k, v in os.environ.items() if k not in _TUNING_ENV_VARS}


# --- compute_tuning tier tests (pure function, no mocking needed) ---


class TestComputeTuningGpuTiers:
    def test_gpu_16gb_tier(self):
        profile = HardwareProfile(
            gpu_name="NVIDIA RTX 4080",
            gpu_vram_mb=16384,
            system_ram_mb=32768,
            cpu_arch="x86_64",
        )
        result = compute_tuning(profile)
        assert result.embedding_batch_size == 64
        assert result.embedding_max_tokens == 8192
        assert result.reranker_max_tokens == 8192
        assert result.embedding_memory_budget_gb == 10.0
        assert result.gpu_mem_limit_gb == 12.0
        assert result.onnx_providers == "auto"
        assert "gpu-16gb+" in result.device_label
        assert "RTX 4080" in result.device_label

    def test_gpu_8gb_tier(self):
        profile = HardwareProfile(
            gpu_name="NVIDIA RTX 3070",
            gpu_vram_mb=8192,
            system_ram_mb=16384,
            cpu_arch="x86_64",
        )
        result = compute_tuning(profile)
        assert result.embedding_batch_size == 32
        assert result.embedding_max_tokens == 4096
        assert result.reranker_max_tokens == 4096
        assert result.embedding_memory_budget_gb == 6.0
        assert result.gpu_mem_limit_gb == 6.0
        assert result.onnx_providers == "auto"
        assert "gpu-8gb+" in result.device_label

    def test_gpu_small_tier(self):
        profile = HardwareProfile(
            gpu_name="NVIDIA GTX 1650",
            gpu_vram_mb=4096,
            system_ram_mb=16384,
            cpu_arch="x86_64",
        )
        result = compute_tuning(profile)
        assert result.embedding_batch_size == 16
        assert result.embedding_max_tokens == 2048
        assert result.reranker_max_tokens == 2048
        assert result.embedding_memory_budget_gb == 3.0
        assert result.gpu_mem_limit_gb == 3.0
        assert result.onnx_providers == "auto"
        assert "gpu-small" in result.device_label


class TestComputeTuningCpuTiers:
    def test_cpu_32gb_tier(self):
        profile = HardwareProfile(
            gpu_vram_mb=0,
            system_ram_mb=32768,
            cpu_arch="x86_64",
        )
        result = compute_tuning(profile)
        assert result.embedding_batch_size == 16
        assert result.embedding_max_tokens == 8192
        assert result.reranker_max_tokens == 4096
        assert result.embedding_memory_budget_gb == 8.0
        assert result.gpu_mem_limit_gb == 0.0
        assert result.onnx_providers == "cpu"
        assert "cpu-32gb+" in result.device_label

    def test_cpu_16gb_tier(self):
        profile = HardwareProfile(
            gpu_vram_mb=0,
            system_ram_mb=16384,
            cpu_arch="aarch64",
        )
        result = compute_tuning(profile)
        assert result.embedding_batch_size == 8
        assert result.embedding_max_tokens == 4096
        assert result.reranker_max_tokens == 2048
        assert result.embedding_memory_budget_gb == 4.0
        assert result.gpu_mem_limit_gb == 0.0
        assert result.onnx_providers == "cpu"
        assert "cpu-16gb+" in result.device_label
        assert "aarch64" in result.device_label

    def test_cpu_8gb_tier(self):
        profile = HardwareProfile(
            gpu_vram_mb=0,
            system_ram_mb=8192,
            cpu_arch="x86_64",
        )
        result = compute_tuning(profile)
        assert result.embedding_batch_size == 4
        assert result.embedding_max_tokens == 2048
        assert result.reranker_max_tokens == 1024
        assert result.embedding_memory_budget_gb == 2.0
        assert result.onnx_providers == "cpu"
        assert "cpu-8gb+" in result.device_label

    def test_cpu_small_tier(self):
        profile = HardwareProfile(
            gpu_vram_mb=0,
            system_ram_mb=4096,
            cpu_arch="x86_64",
        )
        result = compute_tuning(profile)
        assert result.embedding_batch_size == 2
        assert result.embedding_max_tokens == 512
        assert result.reranker_max_tokens == 512
        assert result.embedding_memory_budget_gb == 1.0
        assert result.onnx_providers == "cpu"
        assert "cpu-small" in result.device_label

    def test_cpu_zero_ram_falls_to_small(self):
        """If RAM detection fails (0), falls to the smallest tier."""
        profile = HardwareProfile(gpu_vram_mb=0, system_ram_mb=0, cpu_arch="x86_64")
        result = compute_tuning(profile)
        assert result.embedding_batch_size == 2
        assert result.embedding_max_tokens == 512


# --- Tier boundary tests ---


class TestComputeTuningBoundaries:
    def test_vram_exactly_14000(self):
        profile = HardwareProfile(gpu_vram_mb=14000, system_ram_mb=0, cpu_arch="x86_64")
        result = compute_tuning(profile)
        assert "gpu-16gb+" in result.device_label

    def test_vram_just_below_14000(self):
        profile = HardwareProfile(gpu_vram_mb=13999, system_ram_mb=0, cpu_arch="x86_64")
        result = compute_tuning(profile)
        assert "gpu-8gb+" in result.device_label

    def test_vram_exactly_7000(self):
        profile = HardwareProfile(gpu_vram_mb=7000, system_ram_mb=0, cpu_arch="x86_64")
        result = compute_tuning(profile)
        assert "gpu-8gb+" in result.device_label

    def test_ram_exactly_28000(self):
        profile = HardwareProfile(gpu_vram_mb=0, system_ram_mb=28000, cpu_arch="x86_64")
        result = compute_tuning(profile)
        assert "cpu-32gb+" in result.device_label

    def test_ram_exactly_14000(self):
        profile = HardwareProfile(gpu_vram_mb=0, system_ram_mb=14000, cpu_arch="x86_64")
        result = compute_tuning(profile)
        assert "cpu-16gb+" in result.device_label

    def test_ram_exactly_7000(self):
        profile = HardwareProfile(gpu_vram_mb=0, system_ram_mb=7000, cpu_arch="x86_64")
        result = compute_tuning(profile)
        assert "cpu-8gb+" in result.device_label


# --- apply_tuning tests ---


class TestApplyTuning:
    def test_applies_all_fields_when_no_env_vars(self):
        """When no env vars are set, all tuned fields get applied."""

        class FakeConfig:
            onnx_providers = "auto"
            embedding_batch_size = 8
            embedding_max_tokens = 2048
            reranker_max_tokens = 2048
            embedding_memory_budget_gb = 6.0

        cfg = FakeConfig()
        tuning = TuningResult(
            onnx_providers="cpu",
            embedding_batch_size=64,
            embedding_max_tokens=8192,
            reranker_max_tokens=4096,
            embedding_memory_budget_gb=10.0,
        )

        with patch.dict(os.environ, _env_without_tuning_vars(), clear=True):
            apply_tuning(cfg, tuning)

        assert cfg.onnx_providers == "cpu"
        assert cfg.embedding_batch_size == 64
        assert cfg.embedding_max_tokens == 8192
        assert cfg.reranker_max_tokens == 4096
        assert cfg.embedding_memory_budget_gb == 10.0

    def test_env_var_override_preserves_user_value(self):
        """When an env var is set, apply_tuning skips that field."""

        class FakeConfig:
            onnx_providers = "auto"
            embedding_batch_size = 1
            embedding_max_tokens = 2048
            reranker_max_tokens = 2048
            embedding_memory_budget_gb = 6.0

        cfg = FakeConfig()
        tuning = TuningResult(
            onnx_providers="cpu",
            embedding_batch_size=64,
            embedding_max_tokens=8192,
            reranker_max_tokens=4096,
            embedding_memory_budget_gb=10.0,
        )

        # Start clean, then add only the one override
        clean_env = _env_without_tuning_vars()
        clean_env["ZETTELKASTEN_EMBEDDING_BATCH_SIZE"] = "1"
        with patch.dict(os.environ, clean_env, clear=True):
            apply_tuning(cfg, tuning)

        # batch_size should NOT be overwritten — env var was set
        assert cfg.embedding_batch_size == 1
        # Other fields should be updated
        assert cfg.embedding_max_tokens == 8192
        assert cfg.reranker_max_tokens == 4096

    def test_multiple_env_var_overrides(self):
        """Multiple env vars set → multiple fields preserved."""

        class FakeConfig:
            onnx_providers = "CUDAExecutionProvider"
            embedding_batch_size = 4
            embedding_max_tokens = 512
            reranker_max_tokens = 512
            embedding_memory_budget_gb = 2.0

        cfg = FakeConfig()
        tuning = TuningResult(
            onnx_providers="cpu",
            embedding_batch_size=64,
            embedding_max_tokens=8192,
            reranker_max_tokens=8192,
            embedding_memory_budget_gb=10.0,
        )

        clean_env = _env_without_tuning_vars()
        clean_env["ZETTELKASTEN_ONNX_PROVIDERS"] = "CUDAExecutionProvider"
        clean_env["ZETTELKASTEN_EMBEDDING_BATCH_SIZE"] = "4"
        with patch.dict(os.environ, clean_env, clear=True):
            apply_tuning(cfg, tuning)

        assert cfg.onnx_providers == "CUDAExecutionProvider"
        assert cfg.embedding_batch_size == 4
        # These should be overwritten
        assert cfg.embedding_max_tokens == 8192
        assert cfg.reranker_max_tokens == 8192
        assert cfg.embedding_memory_budget_gb == 10.0


# --- detect_hardware tests ---


class TestDetectHardware:
    def test_detect_hardware_no_nvidia(self):
        """When nvidia-smi is not available, GPU fields are zero/None."""
        with patch("znote_mcp.hardware.subprocess.run", side_effect=FileNotFoundError):
            profile = detect_hardware()
        assert profile.gpu_vram_mb == 0
        assert profile.gpu_name is None

    def test_detect_hardware_nvidia_timeout(self):
        """When nvidia-smi times out, GPU fields are zero/None."""
        import subprocess as sp

        with patch(
            "znote_mcp.hardware.subprocess.run", side_effect=sp.TimeoutExpired("cmd", 5)
        ):
            profile = detect_hardware()
        assert profile.gpu_vram_mb == 0
        assert profile.gpu_name is None

    def test_detect_hardware_nvidia_parse_success(self):
        """Successful nvidia-smi output is parsed correctly."""
        mock_result = type("Result", (), {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 4090, 24564\n",
        })()
        with patch("znote_mcp.hardware.subprocess.run", return_value=mock_result):
            profile = detect_hardware()
        assert profile.gpu_name == "NVIDIA GeForce RTX 4090"
        assert profile.gpu_vram_mb == 24564

    def test_detect_hardware_cpu_arch(self):
        """CPU architecture is always detected."""
        with patch("znote_mcp.hardware.subprocess.run", side_effect=FileNotFoundError):
            profile = detect_hardware()
        assert profile.cpu_arch != ""

    def test_detect_hardware_ram_detected(self):
        """System RAM should be detected on Linux."""
        with patch("znote_mcp.hardware.subprocess.run", side_effect=FileNotFoundError):
            profile = detect_hardware()
        # On the test machine, RAM should be > 0
        assert profile.system_ram_mb > 0
