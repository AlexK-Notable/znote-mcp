"""Tests for ONNX progressive memory resilience."""

import faulthandler
import tempfile


def test_faulthandler_writes_to_file():
    """Verify faulthandler can be enabled on a file handle."""
    with tempfile.NamedTemporaryFile(mode="a", suffix=".log", delete=False) as f:
        faulthandler.enable(file=f)
        assert faulthandler.is_enabled()
        faulthandler.disable()
