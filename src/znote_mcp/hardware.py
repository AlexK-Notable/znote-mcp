"""Hardware detection and auto-tuning for embedding/reranker configuration."""

from __future__ import annotations

import logging
import os
import platform
import subprocess
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class HardwareProfile:
    """Detected hardware capabilities."""

    gpu_name: str | None = None
    gpu_vram_mb: int = 0
    system_ram_mb: int = 0
    cpu_arch: str = ""
    onnx_providers: list[str] = field(default_factory=list)


@dataclass
class TuningResult:
    """Auto-tuned configuration values."""

    onnx_providers: str = "cpu"
    embedding_batch_size: int = 8
    embedding_max_tokens: int = 2048
    reranker_max_tokens: int = 2048
    embedding_memory_budget_gb: float = 6.0
    gpu_mem_limit_gb: float = 0.0
    device_label: str = "cpu-unknown"


def detect_hardware() -> HardwareProfile:
    """Detect available hardware: GPU, RAM, CPU arch, ONNX providers."""
    profile = HardwareProfile()

    # CPU architecture
    profile.cpu_arch = platform.machine()

    # System RAM
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
        if page_size > 0 and page_count > 0:
            profile.system_ram_mb = (page_size * page_count) // (1024 * 1024)
    except (ValueError, OSError, AttributeError):
        # Fallback: try /proc/meminfo
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        profile.system_ram_mb = kb // 1024
                        break
        except (OSError, ValueError):
            pass

    # GPU detection via nvidia-smi
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            line = result.stdout.strip().split("\n")[0]
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                profile.gpu_name = parts[0]
                profile.gpu_vram_mb = int(float(parts[1]))
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, OSError):
        pass

    # ONNX Runtime available providers
    try:
        import onnxruntime

        profile.onnx_providers = onnxruntime.get_available_providers()
    except ImportError:
        pass

    return profile


def compute_tuning(profile: HardwareProfile) -> TuningResult:
    """Compute optimal tuning parameters from a hardware profile.

    Pure function — no side effects, no config mutation.
    """
    vram = profile.gpu_vram_mb
    ram = profile.system_ram_mb

    # GPU tiers (check VRAM first)
    if vram >= 14000:
        return TuningResult(
            onnx_providers="auto",
            embedding_batch_size=64,
            embedding_max_tokens=8192,
            reranker_max_tokens=8192,
            embedding_memory_budget_gb=10.0,
            gpu_mem_limit_gb=12.0,
            device_label=f"gpu-16gb+ ({profile.gpu_name or 'unknown'})",
        )
    if vram >= 7000:
        return TuningResult(
            onnx_providers="auto",
            embedding_batch_size=32,
            embedding_max_tokens=4096,
            reranker_max_tokens=4096,
            embedding_memory_budget_gb=6.0,
            gpu_mem_limit_gb=6.0,
            device_label=f"gpu-8gb+ ({profile.gpu_name or 'unknown'})",
        )
    if vram > 0:
        return TuningResult(
            onnx_providers="auto",
            embedding_batch_size=16,
            embedding_max_tokens=2048,
            reranker_max_tokens=2048,
            embedding_memory_budget_gb=3.0,
            gpu_mem_limit_gb=3.0,
            device_label=f"gpu-small ({profile.gpu_name or 'unknown'})",
        )

    # CPU tiers (no GPU detected)
    if ram >= 28000:
        return TuningResult(
            onnx_providers="cpu",
            embedding_batch_size=16,
            embedding_max_tokens=8192,
            reranker_max_tokens=4096,
            embedding_memory_budget_gb=8.0,
            gpu_mem_limit_gb=0.0,
            device_label=f"cpu-32gb+ ({profile.cpu_arch})",
        )
    if ram >= 14000:
        return TuningResult(
            onnx_providers="cpu",
            embedding_batch_size=8,
            embedding_max_tokens=4096,
            reranker_max_tokens=2048,
            embedding_memory_budget_gb=4.0,
            gpu_mem_limit_gb=0.0,
            device_label=f"cpu-16gb+ ({profile.cpu_arch})",
        )
    if ram >= 7000:
        return TuningResult(
            onnx_providers="cpu",
            embedding_batch_size=4,
            embedding_max_tokens=2048,
            reranker_max_tokens=1024,
            embedding_memory_budget_gb=2.0,
            gpu_mem_limit_gb=0.0,
            device_label=f"cpu-8gb+ ({profile.cpu_arch})",
        )

    # Fallback: minimal resources
    return TuningResult(
        onnx_providers="cpu",
        embedding_batch_size=2,
        embedding_max_tokens=512,
        reranker_max_tokens=512,
        embedding_memory_budget_gb=1.0,
        gpu_mem_limit_gb=0.0,
        device_label=f"cpu-small ({profile.cpu_arch})",
    )


def apply_tuning(config: object, result: TuningResult) -> None:
    """Apply hardware-detected tuning to config. Env var overrides always win.

    Only mutates fields where the user hasn't set an explicit env var.
    """
    field_env_map = {
        "onnx_providers": "ZETTELKASTEN_ONNX_PROVIDERS",
        "embedding_batch_size": "ZETTELKASTEN_EMBEDDING_BATCH_SIZE",
        "embedding_max_tokens": "ZETTELKASTEN_EMBEDDING_MAX_TOKENS",
        "reranker_max_tokens": "ZETTELKASTEN_RERANKER_MAX_TOKENS",
        "embedding_memory_budget_gb": "ZETTELKASTEN_EMBEDDING_MEMORY_BUDGET_GB",
    }
    for field_name, env_var in field_env_map.items():
        if env_var not in os.environ:
            setattr(config, field_name, getattr(result, field_name))
