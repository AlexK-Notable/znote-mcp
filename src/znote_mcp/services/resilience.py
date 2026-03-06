"""ONNX progressive memory resilience manager.

Tracks degradation level per component (embedder/reranker) and provides
adjusted parameters at each level. Supports a notification callback for
MCP log messages.

Levels:
    0 — Normal: full config from startup
    1 — Reduced batch: halve batch_size
    2 — Reduced tokens: also halve max_tokens
    3 — CPU fallback: signal provider switch to CPU
    4 — Disabled: component off for session
"""

from __future__ import annotations

import enum
import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class DegradationLevel(enum.IntEnum):
    NORMAL = 0
    REDUCED_BATCH = 1
    REDUCED_TOKENS = 2
    CPU_FALLBACK = 3
    DISABLED = 4


_NOTIFICATIONS = {
    DegradationLevel.REDUCED_BATCH: "Semantic search: GPU memory pressure detected, reducing batch size",
    DegradationLevel.REDUCED_TOKENS: "Semantic search: continued memory pressure, reducing sequence length",
    DegradationLevel.CPU_FALLBACK: "Semantic search: falling back to CPU (slower but stable)",
    DegradationLevel.DISABLED: "Semantic search: disabled for this session (FTS search still available)",
}


class OnnxResilienceManager:
    """Tracks and manages ONNX degradation state for embedder and reranker.

    Args:
        initial_batch_size: Starting batch size from hardware tuning.
        initial_max_tokens: Starting max_tokens from hardware tuning.
        on_notify: Optional callback(level: str, message: str) for MCP notifications.
    """

    def __init__(
        self,
        initial_batch_size: int,
        initial_max_tokens: int,
        on_notify: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        self._initial_batch_size = initial_batch_size
        self._initial_max_tokens = initial_max_tokens
        self._on_notify = on_notify

        # Embedder state
        self._embedder_level = DegradationLevel.NORMAL
        self._embedder_batch_size = initial_batch_size
        self._embedder_max_tokens = initial_max_tokens

        # Reranker state (same knobs)
        self._reranker_level = DegradationLevel.NORMAL
        self._reranker_batch_size = initial_batch_size
        self._reranker_max_tokens = initial_max_tokens

    # --- Properties ---

    @property
    def embedder_level(self) -> DegradationLevel:
        return self._embedder_level

    @property
    def reranker_level(self) -> DegradationLevel:
        return self._reranker_level

    @property
    def embedder_batch_size(self) -> int:
        return self._embedder_batch_size

    @property
    def embedder_max_tokens(self) -> int:
        return self._embedder_max_tokens

    @property
    def reranker_batch_size(self) -> int:
        return self._reranker_batch_size

    @property
    def reranker_max_tokens(self) -> int:
        return self._reranker_max_tokens

    @property
    def is_embedder_enabled(self) -> bool:
        return self._embedder_level < DegradationLevel.DISABLED

    @property
    def is_reranker_enabled(self) -> bool:
        return self._reranker_level < DegradationLevel.DISABLED

    @property
    def embedder_needs_cpu_switch(self) -> bool:
        return self._embedder_level == DegradationLevel.CPU_FALLBACK

    @property
    def reranker_needs_cpu_switch(self) -> bool:
        return self._reranker_level == DegradationLevel.CPU_FALLBACK

    # --- State transitions ---

    def advance_embedder(self) -> DegradationLevel:
        """Advance embedder to next degradation level. Returns new level."""
        return self._advance("embedder")

    def advance_reranker(self) -> DegradationLevel:
        """Advance reranker to next degradation level. Returns new level."""
        return self._advance("reranker")

    def _advance(self, component: str) -> DegradationLevel:
        level_attr = f"_{component}_level"
        batch_attr = f"_{component}_batch_size"
        tokens_attr = f"_{component}_max_tokens"

        current = getattr(self, level_attr)
        if current >= DegradationLevel.DISABLED:
            return current

        new_level = DegradationLevel(current + 1)
        setattr(self, level_attr, new_level)

        if new_level == DegradationLevel.REDUCED_BATCH:
            setattr(self, batch_attr, max(1, getattr(self, batch_attr) // 2))
        elif new_level == DegradationLevel.REDUCED_TOKENS:
            setattr(self, tokens_attr, max(128, getattr(self, tokens_attr) // 2))

        msg = _NOTIFICATIONS.get(new_level, "")
        logger.warning(
            "ONNX %s degraded to level %d (%s): %s",
            component,
            new_level,
            new_level.name,
            msg,
        )

        if self._on_notify and msg:
            self._on_notify("warning", msg)

        return new_level
