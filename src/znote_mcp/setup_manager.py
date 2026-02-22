"""Self-setup manager for semantic search dependencies.

Ensures the [semantic] optional dependencies (onnxruntime, tokenizers,
huggingface-hub, numpy, sqlite-vec) are installed into the active venv,
using a version-keyed marker file to skip the check on subsequent starts.

Also pre-downloads embedding/reranker ONNX models in a background thread
so the first semantic search doesn't block on a large download.
"""

import logging
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_SEMANTIC_PACKAGES_BASE = [
    "tokenizers>=0.15.0",
    "huggingface-hub>=0.20.0",
    "numpy>=1.24.0",
    "sqlite-vec>=0.1.7a2",
]

_ONNXRUNTIME_CPU = "onnxruntime>=1.17.1"
_ONNXRUNTIME_GPU = "onnxruntime-gpu>=1.17.1"

SEMANTIC_IMPORT_NAMES = [
    "onnxruntime",
    "tokenizers",
    "huggingface_hub",
    "numpy",
    "sqlite_vec",
]

_MARKER_PREFIX = ".semantic_setup_"
_MODEL_MARKER_PREFIX = ".semantic_models_"
_INSTALL_TIMEOUT = 300  # 5 minutes

# Files needed from each HuggingFace model repo
_MODEL_FILES = ["onnx/model.onnx", "tokenizer.json", "tokenizer_config.json"]


def _has_nvidia_gpu() -> bool:
    """Check for NVIDIA GPU via nvidia-smi.

    Same detection approach as hardware.py but lightweight — only checks
    whether nvidia-smi exists and reports a GPU.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


def _get_semantic_packages() -> list[str]:
    """Return the semantic package list with GPU-appropriate onnxruntime variant."""
    if _has_nvidia_gpu():
        logger.info("NVIDIA GPU detected — using onnxruntime-gpu")
        ort_pkg = _ONNXRUNTIME_GPU
    else:
        logger.info("No NVIDIA GPU detected — using onnxruntime (CPU)")
        ort_pkg = _ONNXRUNTIME_CPU
    return [ort_pkg] + _SEMANTIC_PACKAGES_BASE


def _marker_path(venv_dir: Path, version: str) -> Path:
    gpu_suffix = "_gpu" if _has_nvidia_gpu() else "_cpu"
    return venv_dir / f"{_MARKER_PREFIX}{version}{gpu_suffix}"


def _cleanup_old_markers(
    venv_dir: Path, current_marker_name: str, prefix: str = _MARKER_PREFIX
) -> None:
    """Remove marker files that don't match the current marker name."""
    for p in venv_dir.glob(f"{prefix}*"):
        if p.name != current_marker_name:
            p.unlink(missing_ok=True)
            logger.debug("Removed old marker: %s", p.name)


def _check_imports() -> bool:
    """Return True if all semantic packages are importable."""
    import importlib

    for name in SEMANTIC_IMPORT_NAMES:
        try:
            importlib.import_module(name)
        except ImportError:
            return False
    return True


def _get_cuda_version() -> tuple[int, ...] | None:
    """Return the CUDA driver version as a tuple (e.g. (13, 1)) or None.

    Parses the CUDA Version field from nvidia-smi output.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None
        # Parse "CUDA Version: 13.1" from nvidia-smi output
        for line in result.stdout.split("\n"):
            if "CUDA Version" in line:
                # Extract version string after "CUDA Version:"
                part = line.split("CUDA Version:")[-1].strip().rstrip("|").strip()
                parts = part.split(".")
                return tuple(int(p) for p in parts if p.isdigit())
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError, ValueError):
        pass
    return None


def _has_cuda_provider() -> bool:
    """Return True if onnxruntime can actually USE CUDAExecutionProvider.

    Unlike just checking get_available_providers() (which can list CUDA even
    when the runtime libraries are missing/incompatible), this creates a
    minimal ONNX session to verify CUDA actually initializes.
    """
    try:
        import onnxruntime

        if "CUDAExecutionProvider" not in onnxruntime.get_available_providers():
            return False

        # Create a minimal session to test if CUDA actually works.
        # Use a tiny synthetic ONNX model to avoid needing real model files.
        import numpy as np

        try:
            import onnx
            from onnx import TensorProto, helper

            # Build a trivial Identity graph
            X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1])
            Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1])
            node = helper.make_node("Identity", ["X"], ["Y"])
            graph = helper.make_graph([node], "test", [X], [Y])
            model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
            model_bytes = model.SerializeToString()

            sess = onnxruntime.InferenceSession(
                model_bytes,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            active = sess.get_providers()
            return "CUDAExecutionProvider" in active
        except ImportError:
            # onnx package not installed — fall back to availability check only
            # (better than nothing, but less reliable)
            logger.debug(
                "onnx package not available for CUDA validation — "
                "falling back to provider list check"
            )
            return True
    except ImportError:
        return False


def _run_install() -> bool:
    """Install semantic packages via uv pip install. Returns True on success.

    For GPU systems with CUDA 13+, installs from the onnxruntime nightly
    feed since stable releases only support up to CUDA 12.x.
    """
    uv = shutil.which("uv")
    if uv is None:
        logger.warning("uv not found on PATH — cannot auto-install semantic deps")
        return False

    packages = _get_semantic_packages()
    cmd = [uv, "pip", "install"] + packages

    # CUDA 13+ needs the nightly build — stable onnxruntime-gpu only supports CUDA 12
    if _has_nvidia_gpu():
        cuda_ver = _get_cuda_version()
        if cuda_ver and cuda_ver[0] >= 13:
            logger.info(
                "CUDA %s detected — stable onnxruntime-gpu only supports "
                "CUDA 12.x; installing nightly with CUDA 13 support",
                ".".join(str(v) for v in cuda_ver),
            )
            cmd.extend([
                "--prerelease=allow",
                "--extra-index-url",
                "https://aiinfra.pkgs.visualstudio.com/PublicPackages/"
                "_packaging/onnxruntime-cuda-13/pypi/simple/",
            ])

    logger.info("Installing semantic dependencies: %s", " ".join(packages))
    try:
        subprocess.run(
            cmd,
            check=True,
            timeout=_INSTALL_TIMEOUT,
            capture_output=True,
            text=True,
        )
        logger.info("Semantic dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as exc:
        logger.warning(
            "Failed to install semantic deps (exit %d): %s", exc.returncode, exc.stderr
        )
        return False
    except subprocess.TimeoutExpired:
        logger.warning("Semantic deps install timed out after %ds", _INSTALL_TIMEOUT)
        return False


def ensure_semantic_deps(project_root: Path, version: str) -> bool:
    """Ensure semantic search dependencies are available.

    Fast path: if a version-keyed marker file exists, returns True immediately.
    Otherwise checks imports, optionally installs missing packages, and writes
    the marker on success.

    Args:
        project_root: Project root directory (contains .venv/).
        version: Current package version — used in the marker filename.

    Returns:
        True if semantic deps are available, False otherwise.
    """
    venv_dir = project_root / ".venv"
    if not venv_dir.is_dir():
        logger.debug("No .venv directory at %s — skipping semantic setup", venv_dir)
        return False

    marker = _marker_path(venv_dir, version)

    # Fast path: marker exists → deps are already set up for this version
    if marker.exists():
        return True

    # Clean up markers from older versions or different hardware configs
    _cleanup_old_markers(venv_dir, marker.name)

    # Check if deps are already importable
    if _check_imports():
        # GPU available but CUDA doesn't actually work → need upgrade
        if _has_nvidia_gpu() and not _has_cuda_provider():
            logger.info(
                "NVIDIA GPU detected but CUDAExecutionProvider is not functional "
                "— upgrading to onnxruntime-gpu (may need nightly for CUDA 13+)"
            )
            if _run_install():
                # Re-validate after install
                if _has_cuda_provider():
                    logger.info("CUDA validation passed after install")
                else:
                    logger.warning(
                        "onnxruntime-gpu installed but CUDA still not functional — "
                        "will fall back to CPU inference"
                    )
                marker.touch()
                return True
            # Fall through — CPU onnxruntime still works, just slower
            logger.warning(
                "onnxruntime-gpu install failed — falling back to CPU inference"
            )
        marker.touch()
        logger.info("Semantic deps already present — wrote marker %s", marker.name)
        return True

    # Attempt install
    if _run_install():
        marker.touch()
        return True

    return False


# ---------------------------------------------------------------------------
# Model warmup (background download)
# ---------------------------------------------------------------------------


def _warmup_models(
    venv_dir: Path,
    version: str,
    embedding_model: str,
    reranker_model: str,
    cache_dir: Optional[Path] = None,
) -> bool:
    """Download embedding and reranker ONNX models into the HuggingFace cache.

    Uses a version-keyed marker file for a fast-path skip on subsequent starts.

    Returns True if models are cached (or already were), False on failure.
    """
    marker = venv_dir / f"{_MODEL_MARKER_PREFIX}{version}"

    if marker.exists():
        return True

    _cleanup_old_markers(venv_dir, marker.name, prefix=_MODEL_MARKER_PREFIX)

    try:
        from huggingface_hub import snapshot_download

        cache_str = str(cache_dir) if cache_dir else None

        for model_id in (embedding_model, reranker_model):
            logger.info("Pre-downloading model: %s", model_id)
            snapshot_download(
                repo_id=model_id,
                allow_patterns=_MODEL_FILES,
                cache_dir=cache_str,
            )

        marker.touch()
        logger.info("Model warmup complete — wrote marker %s", marker.name)
        return True
    except Exception as exc:
        logger.warning("Model warmup failed: %s", exc)
        return False


def warmup_models_background(
    venv_dir: Path,
    version: str,
    embedding_model: str,
    reranker_model: str,
    cache_dir: Optional[Path] = None,
) -> threading.Thread:
    """Launch model pre-download in a background daemon thread.

    Returns the thread so callers can join() in tests if needed.
    """
    thread = threading.Thread(
        target=_warmup_models,
        args=(venv_dir, version, embedding_model, reranker_model, cache_dir),
        daemon=True,
        name="model-warmup",
    )
    thread.start()
    logger.info("Model warmup started in background")
    return thread
