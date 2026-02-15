"""Token-aware text chunking for embedding long notes.

Splits text into overlapping chunks that respect sentence boundaries
where possible. Each chunk stays within the configured token limit
so the embedding model sees complete, focused content rather than
a single truncated input.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Protocol


class Tokenizer(Protocol):
    """Minimal tokenizer interface (matches tokenizers.Tokenizer)."""

    def encode(self, text: str, add_special_tokens: bool = ...) -> object:
        """Encode text. Result must have an .ids attribute."""
        ...


@dataclass(frozen=True)
class TextChunk:
    """A chunk of text with positional metadata.

    Attributes:
        text: The chunk content.
        index: 0-based chunk index within the source document.
        start_char: Character offset of chunk start in the original text.
        end_char: Character offset of chunk end in the original text.
    """

    text: str
    index: int
    start_char: int
    end_char: int


# Sentence-ending pattern: period/question/exclamation followed by whitespace.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


class TextChunker:
    """Split text into overlapping, token-aware chunks.

    Args:
        chunk_size: Maximum tokens per chunk (default 4096).
        chunk_overlap: Token overlap between consecutive chunks (default 256).
        tokenizer: Optional tokenizer instance. If None, uses a character-level
            approximation (~4 chars per token) which is sufficient for chunking
            decisions. The actual embedding model tokenizes independently.
    """

    def __init__(
        self,
        chunk_size: int = 4096,
        chunk_overlap: int = 256,
        tokenizer: Optional[object] = None,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be less than "
                f"chunk_size ({chunk_size})"
            )
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._tokenizer = tokenizer

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text, using tokenizer if available."""
        if self._tokenizer is not None:
            try:
                encoding = self._tokenizer.encode(text, add_special_tokens=False)
                return len(encoding.ids)
            except Exception:
                pass
        # Fallback: ~4 characters per token (conservative estimate)
        return len(text) // 4

    def _split_segments(self, text: str) -> List[str]:
        """Split text into sentence-level segments, with word-level fallback.

        First splits on sentence boundaries. Any segment that still exceeds
        chunk_size is further split on whitespace to produce word groups
        that fit within chunk_size.
        """
        raw_segments = _SENTENCE_END.split(text)
        segments: List[str] = []

        for seg in raw_segments:
            if self._count_tokens(seg) <= self._chunk_size:
                segments.append(seg)
            else:
                # Oversized segment: split on whitespace
                words = seg.split()
                current_words: List[str] = []
                current_tok = 0
                for word in words:
                    wtok = self._count_tokens(word)
                    if current_tok + wtok > self._chunk_size and current_words:
                        segments.append(" ".join(current_words))
                        current_words = []
                        current_tok = 0
                    current_words.append(word)
                    current_tok += wtok
                if current_words:
                    segments.append(" ".join(current_words))

        return segments

    def chunk(self, text: str) -> List[TextChunk]:
        """Split text into overlapping chunks.

        Short texts (under chunk_size tokens) are returned as a single chunk.
        Longer texts are split at sentence boundaries where possible,
        falling back to word boundaries for very long sentences.

        Args:
            text: The text to chunk.

        Returns:
            List of TextChunk instances, each within chunk_size tokens.
        """
        if not text or not text.strip():
            return [TextChunk(text=text, index=0, start_char=0, end_char=len(text))]

        total_tokens = self._count_tokens(text)
        if total_tokens <= self._chunk_size:
            return [TextChunk(text=text, index=0, start_char=0, end_char=len(text))]

        segments = self._split_segments(text)
        if len(segments) <= 1:
            # Cannot split further
            return [TextChunk(text=text, index=0, start_char=0, end_char=len(text))]

        chunks: List[TextChunk] = []
        current_segments: List[str] = []
        current_tokens = 0
        # Track approximate character position in the original text
        char_pos = 0
        chunk_start_char = 0

        for seg in segments:
            seg_tokens = self._count_tokens(seg)

            if current_tokens + seg_tokens > self._chunk_size and current_segments:
                # Emit current chunk
                chunk_text = " ".join(current_segments)
                chunk_end_char = char_pos
                chunks.append(
                    TextChunk(
                        text=chunk_text,
                        index=len(chunks),
                        start_char=chunk_start_char,
                        end_char=chunk_end_char,
                    )
                )

                # Compute overlap: keep trailing segments until overlap budget
                overlap_segs: List[str] = []
                overlap_tokens = 0
                for s in reversed(current_segments):
                    s_tok = self._count_tokens(s)
                    if overlap_tokens + s_tok > self._chunk_overlap:
                        break
                    overlap_segs.insert(0, s)
                    overlap_tokens += s_tok

                current_segments = overlap_segs
                current_tokens = overlap_tokens
                if overlap_segs:
                    overlap_text = " ".join(overlap_segs)
                    chunk_start_char = chunk_end_char - len(overlap_text)
                else:
                    chunk_start_char = char_pos

            current_segments.append(seg)
            current_tokens += seg_tokens
            char_pos += len(seg) + 1  # +1 for whitespace separator

        # Emit final chunk
        if current_segments:
            chunk_text = " ".join(current_segments)
            chunks.append(
                TextChunk(
                    text=chunk_text,
                    index=len(chunks),
                    start_char=chunk_start_char,
                    end_char=len(text),
                )
            )

        return chunks

    @staticmethod
    def make_chunk_id(note_id: str, chunk_index: int) -> str:
        """Build a chunk_id from note_id and chunk index.

        Format: ``{note_id}::chunk_{index}``
        """
        return f"{note_id}::chunk_{chunk_index}"

    @staticmethod
    def parse_chunk_id(chunk_id: str) -> tuple[str, int]:
        """Extract (note_id, chunk_index) from a chunk_id.

        Raises ValueError if the format is invalid.
        """
        sep = "::chunk_"
        idx = chunk_id.rfind(sep)
        if idx == -1:
            raise ValueError(f"Invalid chunk_id format: {chunk_id!r}")
        note_id = chunk_id[:idx]
        try:
            chunk_index = int(chunk_id[idx + len(sep) :])
        except ValueError:
            raise ValueError(f"Invalid chunk index in chunk_id: {chunk_id!r}")
        return note_id, chunk_index
