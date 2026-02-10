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

SEMANTIC_PACKAGES = [
    "onnxruntime>=1.17.0",
    "tokenizers>=0.15.0",
    "huggingface-hub>=0.20.0",
    "numpy>=1.24.0",
    "sqlite-vec>=0.1.6",
]

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


def _marker_path(venv_dir: Path, version: str) -> Path:
    return venv_dir / f"{_MARKER_PREFIX}{version}"


def _cleanup_old_markers(
    venv_dir: Path, current_version: str, prefix: str = _MARKER_PREFIX
) -> None:
    """Remove marker files from previous versions."""
    for p in venv_dir.glob(f"{prefix}*"):
        if p.name != f"{prefix}{current_version}":
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


def _run_install() -> bool:
    """Install semantic packages via uv pip install. Returns True on success."""
    uv = shutil.which("uv")
    if uv is None:
        logger.warning("uv not found on PATH — cannot auto-install semantic deps")
        return False

    cmd = [uv, "pip", "install"] + SEMANTIC_PACKAGES
    logger.info("Installing semantic dependencies: %s", " ".join(SEMANTIC_PACKAGES))
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
        logger.warning("Failed to install semantic deps (exit %d): %s", exc.returncode, exc.stderr)
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

    # Clean up markers from older versions
    _cleanup_old_markers(venv_dir, version)

    # Check if deps are already importable
    if _check_imports():
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

    _cleanup_old_markers(venv_dir, version, prefix=_MODEL_MARKER_PREFIX)

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
