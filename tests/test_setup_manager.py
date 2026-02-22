"""Tests for the self-setup manager (semantic dependency auto-install)."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from znote_mcp.setup_manager import (
    _MARKER_PREFIX,
    _MODEL_FILES,
    _MODEL_MARKER_PREFIX,
    _check_imports,
    _cleanup_old_markers,
    _get_cuda_version,
    _has_nvidia_gpu,
    _run_install,
    _warmup_models,
    ensure_semantic_deps,
    warmup_models_background,
)


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    """Create a fake project root with a .venv directory."""
    venv = tmp_path / ".venv"
    venv.mkdir()
    return tmp_path


def _marker_name(version: str, gpu: bool = False) -> str:
    """Build expected marker filename."""
    suffix = "_gpu" if gpu else "_cpu"
    return f"{_MARKER_PREFIX}{version}{suffix}"


class TestMarkerFastPath:
    """Marker file exists and version matches → skip everything."""

    def test_marker_exists_returns_true(self, project_root: Path) -> None:
        marker = project_root / ".venv" / _marker_name("1.4.0")
        marker.touch()
        with patch("znote_mcp.setup_manager._has_nvidia_gpu", return_value=False):
            assert ensure_semantic_deps(project_root, "1.4.0") is True

    def test_marker_exists_no_imports_no_subprocess(self, project_root: Path) -> None:
        marker = project_root / ".venv" / _marker_name("1.4.0")
        marker.touch()
        with (
            patch("znote_mcp.setup_manager._has_nvidia_gpu", return_value=False),
            patch("znote_mcp.setup_manager._check_imports") as mock_ci,
            patch("znote_mcp.setup_manager._run_install") as mock_ri,
        ):
            ensure_semantic_deps(project_root, "1.4.0")
            mock_ci.assert_not_called()
            mock_ri.assert_not_called()

    def test_gpu_marker_exists_returns_true(self, project_root: Path) -> None:
        marker = project_root / ".venv" / _marker_name("1.4.0", gpu=True)
        marker.touch()
        with patch("znote_mcp.setup_manager._has_nvidia_gpu", return_value=True):
            assert ensure_semantic_deps(project_root, "1.4.0") is True


class TestVersionMismatch:
    """Marker exists for old version → re-check deps."""

    def test_old_marker_triggers_recheck(self, project_root: Path) -> None:
        old_marker = project_root / ".venv" / _marker_name("1.3.0")
        old_marker.touch()
        with (
            patch("znote_mcp.setup_manager._has_nvidia_gpu", return_value=False),
            patch("znote_mcp.setup_manager._has_cuda_provider", return_value=False),
            patch("znote_mcp.setup_manager._check_imports", return_value=True),
        ):
            result = ensure_semantic_deps(project_root, "1.4.0")
        assert result is True
        # New marker written
        assert (project_root / ".venv" / _marker_name("1.4.0")).exists()
        # Old marker cleaned up
        assert not old_marker.exists()


class TestDepsAlreadyInstalled:
    """All packages importable → write marker, no subprocess."""

    def test_importable_writes_marker(self, project_root: Path) -> None:
        with (
            patch("znote_mcp.setup_manager._has_nvidia_gpu", return_value=False),
            patch("znote_mcp.setup_manager._has_cuda_provider", return_value=False),
            patch("znote_mcp.setup_manager._check_imports", return_value=True),
        ):
            result = ensure_semantic_deps(project_root, "2.0.0")
        assert result is True
        assert (project_root / ".venv" / _marker_name("2.0.0")).exists()

    def test_importable_no_install(self, project_root: Path) -> None:
        with (
            patch("znote_mcp.setup_manager._has_nvidia_gpu", return_value=False),
            patch("znote_mcp.setup_manager._has_cuda_provider", return_value=False),
            patch("znote_mcp.setup_manager._check_imports", return_value=True),
            patch("znote_mcp.setup_manager._run_install") as mock_ri,
        ):
            ensure_semantic_deps(project_root, "2.0.0")
            mock_ri.assert_not_called()


class TestGpuUpgrade:
    """GPU available but onnxruntime lacks CUDA → upgrade to onnxruntime-gpu."""

    def test_gpu_detected_triggers_upgrade(self, project_root: Path) -> None:
        with (
            patch("znote_mcp.setup_manager._has_nvidia_gpu", return_value=True),
            patch("znote_mcp.setup_manager._has_cuda_provider", return_value=False),
            patch("znote_mcp.setup_manager._check_imports", return_value=True),
            patch("znote_mcp.setup_manager._run_install", return_value=True) as mock_ri,
        ):
            result = ensure_semantic_deps(project_root, "1.5.0")
        assert result is True
        mock_ri.assert_called_once()
        assert (project_root / ".venv" / _marker_name("1.5.0", gpu=True)).exists()

    def test_gpu_upgrade_fails_falls_back(self, project_root: Path) -> None:
        with (
            patch("znote_mcp.setup_manager._has_nvidia_gpu", return_value=True),
            patch("znote_mcp.setup_manager._has_cuda_provider", return_value=False),
            patch("znote_mcp.setup_manager._check_imports", return_value=True),
            patch("znote_mcp.setup_manager._run_install", return_value=False),
        ):
            result = ensure_semantic_deps(project_root, "1.5.0")
        # Still True — CPU onnxruntime works
        assert result is True
        # Marker still written (CPU fallback is functional)
        assert (project_root / ".venv" / _marker_name("1.5.0", gpu=True)).exists()

    def test_gpu_with_cuda_no_upgrade(self, project_root: Path) -> None:
        """GPU present and CUDA provider available → no upgrade needed."""
        with (
            patch("znote_mcp.setup_manager._has_nvidia_gpu", return_value=True),
            patch("znote_mcp.setup_manager._has_cuda_provider", return_value=True),
            patch("znote_mcp.setup_manager._check_imports", return_value=True),
            patch("znote_mcp.setup_manager._run_install") as mock_ri,
        ):
            result = ensure_semantic_deps(project_root, "1.5.0")
        assert result is True
        mock_ri.assert_not_called()


class TestInstallSucceeds:
    """Deps missing, uv install succeeds → marker written."""

    def test_install_success_writes_marker(self, project_root: Path) -> None:
        with (
            patch("znote_mcp.setup_manager._has_nvidia_gpu", return_value=False),
            patch("znote_mcp.setup_manager._check_imports", return_value=False),
            patch("znote_mcp.setup_manager._run_install", return_value=True),
        ):
            result = ensure_semantic_deps(project_root, "1.4.0")
        assert result is True
        assert (project_root / ".venv" / _marker_name("1.4.0")).exists()


class TestInstallFails:
    """Deps missing, install fails → return False, no marker."""

    def test_install_failure_returns_false(self, project_root: Path) -> None:
        with (
            patch("znote_mcp.setup_manager._has_nvidia_gpu", return_value=False),
            patch("znote_mcp.setup_manager._check_imports", return_value=False),
            patch("znote_mcp.setup_manager._run_install", return_value=False),
        ):
            result = ensure_semantic_deps(project_root, "1.4.0")
        assert result is False
        assert not (project_root / ".venv" / _marker_name("1.4.0")).exists()


class TestRunInstall:
    """Tests for the _run_install helper."""

    def test_uv_not_found_returns_false(self) -> None:
        with patch("shutil.which", return_value=None):
            assert _run_install() is False

    def test_subprocess_called_with_cpu_packages(self) -> None:
        with (
            patch("shutil.which", return_value="/usr/bin/uv"),
            patch("znote_mcp.setup_manager._has_nvidia_gpu", return_value=False),
            patch("subprocess.run") as mock_run,
        ):
            _run_install()
            mock_run.assert_called_once()
            cmd = mock_run.call_args[0][0]
            assert cmd[0] == "/usr/bin/uv"
            assert cmd[1:3] == ["pip", "install"]
            assert "onnxruntime>=1.17.1" in cmd
            assert "onnxruntime-gpu>=1.17.1" not in cmd

    def test_subprocess_called_with_gpu_packages(self) -> None:
        with (
            patch("shutil.which", return_value="/usr/bin/uv"),
            patch("znote_mcp.setup_manager._has_nvidia_gpu", return_value=True),
            patch("znote_mcp.setup_manager._get_cuda_version", return_value=(12, 6)),
            patch("subprocess.run") as mock_run,
        ):
            _run_install()
            # Find the uv pip install call (not any nvidia-smi calls)
            install_calls = [
                c for c in mock_run.call_args_list if c[0][0][0] == "/usr/bin/uv"
            ]
            assert len(install_calls) == 1
            cmd = install_calls[0][0][0]
            assert "onnxruntime-gpu>=1.17.1" in cmd
            assert "onnxruntime>=1.17.1" not in cmd
            # CUDA 12 → no nightly flags
            assert "--prerelease=allow" not in cmd

    def test_subprocess_called_with_nightly_for_cuda13(self) -> None:
        """CUDA 13+ systems get the nightly onnxruntime-gpu from the CUDA 13 feed."""
        with (
            patch("shutil.which", return_value="/usr/bin/uv"),
            patch("znote_mcp.setup_manager._has_nvidia_gpu", return_value=True),
            patch("znote_mcp.setup_manager._get_cuda_version", return_value=(13, 1)),
            patch("subprocess.run") as mock_run,
        ):
            _run_install()
            install_calls = [
                c for c in mock_run.call_args_list if c[0][0][0] == "/usr/bin/uv"
            ]
            assert len(install_calls) == 1
            cmd = install_calls[0][0][0]
            assert "--prerelease=allow" in cmd
            assert any("onnxruntime-cuda-13" in arg for arg in cmd)

    def test_subprocess_failure_returns_false(self) -> None:
        with (
            patch("shutil.which", return_value="/usr/bin/uv"),
            patch("znote_mcp.setup_manager._has_nvidia_gpu", return_value=False),
            patch(
                "subprocess.run",
                side_effect=subprocess.CalledProcessError(1, "uv", stderr="error"),
            ),
        ):
            assert _run_install() is False

    def test_subprocess_timeout_returns_false(self) -> None:
        with (
            patch("shutil.which", return_value="/usr/bin/uv"),
            patch("znote_mcp.setup_manager._has_nvidia_gpu", return_value=False),
            patch("subprocess.run", side_effect=subprocess.TimeoutExpired("uv", 300)),
        ):
            assert _run_install() is False


class TestHasNvidiaGpu:
    """Tests for the _has_nvidia_gpu helper."""

    def test_nvidia_smi_found(self) -> None:
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "NVIDIA GeForce RTX 4070 Ti SUPER\n"
        with patch("subprocess.run", return_value=mock_result):
            assert _has_nvidia_gpu() is True

    def test_nvidia_smi_not_found(self) -> None:
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert _has_nvidia_gpu() is False

    def test_nvidia_smi_no_gpu(self) -> None:
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        with patch("subprocess.run", return_value=mock_result):
            assert _has_nvidia_gpu() is False

    def test_nvidia_smi_fails(self) -> None:
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        with patch("subprocess.run", return_value=mock_result):
            assert _has_nvidia_gpu() is False


class TestGetCudaVersion:
    """Tests for the _get_cuda_version helper."""

    def test_cuda_13_1(self) -> None:
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = (
            "+-------------------------+\n"
            "| NVIDIA-SMI 590.48.01    Driver Version: 590.48.01    CUDA Version: 13.1 |\n"
            "+-------------------------+\n"
        )
        with patch("subprocess.run", return_value=mock_result):
            assert _get_cuda_version() == (13, 1)

    def test_cuda_12_6(self) -> None:
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = (
            "| NVIDIA-SMI 560.35.03    Driver Version: 560.35.03    CUDA Version: 12.6 |\n"
        )
        with patch("subprocess.run", return_value=mock_result):
            assert _get_cuda_version() == (12, 6)

    def test_nvidia_smi_not_found(self) -> None:
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert _get_cuda_version() is None

    def test_nvidia_smi_fails(self) -> None:
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        with patch("subprocess.run", return_value=mock_result):
            assert _get_cuda_version() is None


class TestCheckImports:
    """Tests for the _check_imports helper."""

    def test_all_importable_returns_true(self) -> None:
        with patch("importlib.import_module") as mock_import:
            mock_import.return_value = MagicMock()
            assert _check_imports() is True

    def test_one_missing_returns_false(self) -> None:
        def side_effect(name: str) -> MagicMock:
            if name == "sqlite_vec":
                raise ImportError("No module named 'sqlite_vec'")
            return MagicMock()

        with patch("importlib.import_module", side_effect=side_effect):
            assert _check_imports() is False


class TestOldMarkerCleanup:
    """Old version markers are removed."""

    def test_old_markers_removed(self, project_root: Path) -> None:
        venv = project_root / ".venv"
        (venv / f"{_MARKER_PREFIX}1.0.0_cpu").touch()
        (venv / f"{_MARKER_PREFIX}1.2.0_cpu").touch()
        (venv / f"{_MARKER_PREFIX}1.4.0_cpu").touch()

        _cleanup_old_markers(venv, f"{_MARKER_PREFIX}1.4.0_cpu")

        assert not (venv / f"{_MARKER_PREFIX}1.0.0_cpu").exists()
        assert not (venv / f"{_MARKER_PREFIX}1.2.0_cpu").exists()
        assert (venv / f"{_MARKER_PREFIX}1.4.0_cpu").exists()

    def test_hardware_change_cleans_old_marker(self, project_root: Path) -> None:
        """CPU→GPU transition cleans CPU marker."""
        venv = project_root / ".venv"
        (venv / f"{_MARKER_PREFIX}1.4.0_cpu").touch()

        _cleanup_old_markers(venv, f"{_MARKER_PREFIX}1.4.0_gpu")

        assert not (venv / f"{_MARKER_PREFIX}1.4.0_cpu").exists()


class TestNoVenv:
    """No .venv directory → return False gracefully."""

    def test_no_venv_returns_false(self, tmp_path: Path) -> None:
        with patch("znote_mcp.setup_manager._has_nvidia_gpu", return_value=False):
            assert ensure_semantic_deps(tmp_path, "1.4.0") is False


# ---------------------------------------------------------------------------
# Model warmup tests
# ---------------------------------------------------------------------------

EMBED_MODEL = "Alibaba-NLP/gte-modernbert-base"
RERANKER_MODEL = "Alibaba-NLP/gte-reranker-modernbert-base"


class TestWarmupModelMarkerFastPath:
    """Model marker exists → skip download entirely."""

    def test_marker_exists_returns_true(self, project_root: Path) -> None:
        marker = project_root / ".venv" / f"{_MODEL_MARKER_PREFIX}1.4.0"
        marker.touch()
        assert (
            _warmup_models(project_root / ".venv", "1.4.0", EMBED_MODEL, RERANKER_MODEL)
            is True
        )

    def test_marker_exists_no_download(self, project_root: Path) -> None:
        marker = project_root / ".venv" / f"{_MODEL_MARKER_PREFIX}1.4.0"
        marker.touch()
        with patch("znote_mcp.setup_manager.snapshot_download", create=True) as mock_dl:
            _warmup_models(project_root / ".venv", "1.4.0", EMBED_MODEL, RERANKER_MODEL)
            mock_dl.assert_not_called()


class TestWarmupDownloadsBothModels:
    """No marker → snapshot_download called for both models."""

    def test_downloads_both_models(self, project_root: Path) -> None:
        mock_dl = MagicMock()
        with patch.dict(
            "sys.modules",
            {"huggingface_hub": MagicMock(snapshot_download=mock_dl)},
        ):
            result = _warmup_models(
                project_root / ".venv", "1.4.0", EMBED_MODEL, RERANKER_MODEL
            )
        assert result is True
        assert mock_dl.call_count == 2
        mock_dl.assert_any_call(
            repo_id=EMBED_MODEL, allow_patterns=_MODEL_FILES, cache_dir=None
        )
        mock_dl.assert_any_call(
            repo_id=RERANKER_MODEL, allow_patterns=_MODEL_FILES, cache_dir=None
        )

    def test_writes_marker_on_success(self, project_root: Path) -> None:
        with patch.dict(
            "sys.modules",
            {"huggingface_hub": MagicMock(snapshot_download=MagicMock())},
        ):
            _warmup_models(project_root / ".venv", "1.4.0", EMBED_MODEL, RERANKER_MODEL)
        assert (project_root / ".venv" / f"{_MODEL_MARKER_PREFIX}1.4.0").exists()

    def test_passes_cache_dir(self, project_root: Path) -> None:
        mock_dl = MagicMock()
        custom_cache = Path("/tmp/my-cache")
        with patch.dict(
            "sys.modules",
            {"huggingface_hub": MagicMock(snapshot_download=mock_dl)},
        ):
            _warmup_models(
                project_root / ".venv",
                "1.4.0",
                EMBED_MODEL,
                RERANKER_MODEL,
                cache_dir=custom_cache,
            )
        for c in mock_dl.call_args_list:
            assert c.kwargs["cache_dir"] == str(custom_cache)


class TestWarmupFailure:
    """Download fails → return False, no marker."""

    def test_download_exception_returns_false(self, project_root: Path) -> None:
        mock_dl = MagicMock(side_effect=OSError("network error"))
        with patch.dict(
            "sys.modules",
            {"huggingface_hub": MagicMock(snapshot_download=mock_dl)},
        ):
            result = _warmup_models(
                project_root / ".venv", "1.4.0", EMBED_MODEL, RERANKER_MODEL
            )
        assert result is False
        assert not (project_root / ".venv" / f"{_MODEL_MARKER_PREFIX}1.4.0").exists()


class TestWarmupVersionMismatch:
    """Old model marker → re-download and clean up."""

    def test_old_model_marker_cleaned(self, project_root: Path) -> None:
        venv = project_root / ".venv"
        old = venv / f"{_MODEL_MARKER_PREFIX}1.3.0"
        old.touch()

        with patch.dict(
            "sys.modules",
            {"huggingface_hub": MagicMock(snapshot_download=MagicMock())},
        ):
            _warmup_models(venv, "1.4.0", EMBED_MODEL, RERANKER_MODEL)

        assert not old.exists()
        assert (venv / f"{_MODEL_MARKER_PREFIX}1.4.0").exists()


class TestWarmupBackground:
    """Background thread is a daemon and actually runs."""

    def test_thread_is_daemon(self, project_root: Path) -> None:
        marker = project_root / ".venv" / f"{_MODEL_MARKER_PREFIX}1.4.0"
        marker.touch()  # fast path so thread finishes instantly
        thread = warmup_models_background(
            project_root / ".venv", "1.4.0", EMBED_MODEL, RERANKER_MODEL
        )
        assert thread.daemon is True
        thread.join(timeout=2)

    def test_thread_runs_warmup(self, project_root: Path) -> None:
        mock_dl = MagicMock()
        with patch.dict(
            "sys.modules",
            {"huggingface_hub": MagicMock(snapshot_download=mock_dl)},
        ):
            thread = warmup_models_background(
                project_root / ".venv", "1.4.0", EMBED_MODEL, RERANKER_MODEL
            )
            thread.join(timeout=5)
        assert mock_dl.call_count == 2
