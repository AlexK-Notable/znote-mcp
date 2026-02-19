"""Direct ONNX Runtime providers for embedding and reranking.

Implements EmbeddingProvider and RerankerProvider using onnxruntime
directly, without fastembed or sentence-transformers. This gives us
minimal dependencies (~50MB) and full model flexibility.

Models:
- Embedding: Alibaba-NLP/gte-modernbert-base (149M params, 768-dim, Apache-2.0)
- Reranker: Alibaba-NLP/gte-reranker-modernbert-base (149M params, Apache-2.0)
"""

from __future__ import annotations

import logging
import os
import resource
import time as _time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _get_rss_mb() -> float:
    """Return current RSS (Resident Set Size) in MB via /proc on Linux,
    falling back to resource.getrusage elsewhere."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024  # kB → MB
    except (OSError, ValueError):
        pass
    # Fallback: ru_maxrss is in KB on Linux
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _get_peak_rss_mb() -> float:
    """Return peak RSS (VmHWM) in MB."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1024  # kB → MB
    except (OSError, ValueError):
        pass
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _get_cpu_percent(interval_cpu_time: float, interval_wall_time: float) -> float:
    """Compute CPU utilization % over an interval.

    Args:
        interval_cpu_time: CPU time consumed during interval (seconds).
        interval_wall_time: Wall clock time during interval (seconds).

    Returns:
        CPU usage percentage (0-100 * num_cores).
    """
    if interval_wall_time <= 0:
        return 0.0
    return (interval_cpu_time / interval_wall_time) * 100


def _cpu_time() -> float:
    """Return total process CPU time (user + system) in seconds."""
    t = os.times()
    return t.user + t.system

# Lazy imports — these are optional dependencies
# These are populated by _ensure_imports()
_ort = None
_tokenizers = None
_hf_hub = None


def _ensure_imports() -> None:
    """Import optional dependencies, raising a clear error if missing."""
    global _ort, _tokenizers, _hf_hub
    if _ort is None:
        try:
            import onnxruntime as ort

            _ort = ort
        except ImportError:
            raise ImportError(
                "onnxruntime is required for embeddings. "
                "Install with: pip install znote-mcp[semantic]"
            )
    if _tokenizers is None:
        try:
            import tokenizers as tok

            _tokenizers = tok
        except ImportError:
            raise ImportError(
                "tokenizers is required for embeddings. "
                "Install with: pip install znote-mcp[semantic]"
            )
    if _hf_hub is None:
        try:
            import huggingface_hub as hfh

            _hf_hub = hfh
        except ImportError:
            raise ImportError(
                "huggingface-hub is required for embeddings. "
                "Install with: pip install znote-mcp[semantic]"
            )


def _resolve_providers(preference: str = "auto") -> List[str]:
    """Resolve ONNX execution providers based on preference string.

    Args:
        preference: One of:
            - "auto": detect available providers (CUDA > CPU)
            - "cpu": force CPUExecutionProvider only
            - comma-separated list: use as-is (e.g. "CUDAExecutionProvider,CPUExecutionProvider")

    Returns:
        Ordered list of provider names for ort.InferenceSession.
    """
    _ensure_imports()

    pref = preference.strip().lower()

    if pref == "cpu":
        return ["CPUExecutionProvider"]

    if pref == "auto":
        available = _ort.get_available_providers()
        providers = []
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        # Always include CPU as fallback
        providers.append("CPUExecutionProvider")
        return providers

    # Explicit comma-separated list
    return [p.strip() for p in preference.split(",") if p.strip()]


def _cuda_session_options() -> dict:
    """Return provider_options dict for CUDA with arena strategy and VRAM cap.

    Returns:
        Dict mapping provider name to options dict, suitable for passing
        as provider_options to ort.InferenceSession.
    """
    return {
        "CUDAExecutionProvider": {
            "arena_extend_strategy": "kSameAsRequested",
            "gpu_mem_limit": str(4 * 1024 * 1024 * 1024),  # 4GB VRAM cap
        }
    }


def _download_model_files(
    model_id: str,
    filenames: List[str],
    cache_dir: Optional[Path] = None,
) -> Path:
    """Download model files from HuggingFace Hub and return the cache directory.

    Args:
        model_id: HuggingFace model ID (e.g., "Alibaba-NLP/gte-modernbert-base").
        filenames: List of files to download from the repo.
        cache_dir: Optional custom cache directory.

    Returns:
        Path to the snapshot directory containing the downloaded files.
    """
    _ensure_imports()
    snapshot_dir = _hf_hub.snapshot_download(
        repo_id=model_id,
        allow_patterns=filenames,
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    return Path(snapshot_dir)


class OnnxEmbeddingProvider:
    """Embedding provider using direct ONNX Runtime inference.

    Loads Alibaba-NLP/gte-modernbert-base (or compatible BERT-family model)
    from the HuggingFace Hub's official /onnx/ folder. Uses CLS pooling
    and L2 normalization to produce unit-length vectors.

    Args:
        model_id: HuggingFace model ID.
        onnx_filename: Path to the ONNX model file within the repo.
        max_length: Maximum token length for truncation.
        cache_dir: Optional custom cache directory for model files.
        providers: Provider preference string ("auto", "cpu", or comma-separated).
    """

    def __init__(
        self,
        model_id: str = "Alibaba-NLP/gte-modernbert-base",
        onnx_filename: str = "onnx/model.onnx",
        max_length: int = 8192,
        cache_dir: Optional[Path] = None,
        providers: str = "auto",
    ) -> None:
        self._model_id = model_id
        self._onnx_filename = onnx_filename
        self._max_length = max_length
        self._cache_dir = cache_dir
        self._providers_pref = providers
        self._session: Optional[object] = None  # ort.InferenceSession
        self._tokenizer: Optional[object] = None  # tokenizers.Tokenizer
        self._dim: int = 768  # Default for gte-modernbert-base

    @property
    def dimension(self) -> int:
        return self._dim

    def load(self) -> None:
        """Download and load the ONNX model and tokenizer."""
        if self._session is not None:
            return  # Already loaded

        _ensure_imports()
        rss_before = _get_rss_mb()
        logger.info(
            f"Loading embedding model: {self._model_id} "
            f"[{self._onnx_filename}] "
            f"(RSS before load: {rss_before:.0f}MB)"
        )

        # Download model files
        model_dir = _download_model_files(
            self._model_id,
            [self._onnx_filename, "tokenizer.json", "tokenizer_config.json"],
            self._cache_dir,
        )

        # Load tokenizer
        tokenizer_path = model_dir / "tokenizer.json"
        self._tokenizer = _tokenizers.Tokenizer.from_file(str(tokenizer_path))
        self._tokenizer.enable_truncation(max_length=self._max_length)
        self._tokenizer.enable_padding(length=None)  # Dynamic padding per batch

        # Load ONNX model (fall back to FP32 if quantized not found)
        onnx_path = model_dir / self._onnx_filename
        if not onnx_path.exists():
            fallback = "onnx/model.onnx"
            fallback_path = model_dir / fallback
            if self._onnx_filename != fallback and fallback_path.exists():
                logger.warning(
                    f"Quantized model not found at {onnx_path}, "
                    f"falling back to {fallback}"
                )
                onnx_path = fallback_path
                self._onnx_filename = fallback
            else:
                raise FileNotFoundError(
                    f"ONNX model not found at {onnx_path}. "
                    f"Check that {self._model_id} has an ONNX model at {self._onnx_filename}"
                )

        sess_options = _ort.SessionOptions()
        sess_options.graph_optimization_level = (
            _ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        providers = _resolve_providers(self._providers_pref)

        # Disable CPU memory arena — ONNX Runtime's arena allocator never
        # returns memory between batches (onnxruntime#23339), causing RSS
        # to grow monotonically during reindex.
        if providers == ["CPUExecutionProvider"]:
            sess_options.enable_cpu_mem_arena = False
            logger.info("CPU memory arena disabled (prevents memory hoarding)")

        # Build provider_options list matching providers order
        cuda_opts = _cuda_session_options()
        provider_options = [cuda_opts.get(p, {}) for p in providers]

        self._session = _ort.InferenceSession(
            str(onnx_path),
            sess_options=sess_options,
            providers=providers,
            provider_options=provider_options,
        )

        # Detect actual dimension from model output shape
        outputs = self._session.get_outputs()
        if outputs and len(outputs[0].shape) >= 2:
            self._dim = outputs[0].shape[-1]

        active = self._session.get_providers()
        rss_after = _get_rss_mb()
        logger.info(
            f"Embedding model loaded: dim={self._dim}, "
            f"max_tokens={self._max_length}, providers={active}, "
            f"RSS: {rss_after:.0f}MB (+{rss_after - rss_before:.0f}MB)"
        )

    def unload(self) -> None:
        """Release model from memory."""
        rss_before = _get_rss_mb()
        self._session = None
        self._tokenizer = None
        rss_after = _get_rss_mb()
        logger.info(
            f"Embedding model unloaded: {self._model_id}, "
            f"RSS: {rss_after:.0f}MB (freed ~{rss_before - rss_after:.0f}MB)"
        )

    @property
    def is_loaded(self) -> bool:
        return self._session is not None

    def _tokenize(self, texts: Sequence[str]) -> dict:
        """Tokenize texts and return numpy arrays for ONNX input."""
        encodings = self._tokenizer.encode_batch(list(texts))

        # Find max length in this batch for padding
        max_len = max(len(e.ids) for e in encodings)

        input_ids = np.zeros((len(texts), max_len), dtype=np.int64)
        attention_mask = np.zeros((len(texts), max_len), dtype=np.int64)

        for i, encoding in enumerate(encodings):
            length = len(encoding.ids)
            input_ids[i, :length] = encoding.ids
            attention_mask[i, :length] = encoding.attention_mask

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def _forward(self, inputs: dict) -> np.ndarray:
        """Run ONNX inference and apply CLS pooling + L2 normalization."""
        # Check which inputs the model expects
        input_names = {inp.name for inp in self._session.get_inputs()}
        feed = {}
        if "input_ids" in input_names:
            feed["input_ids"] = inputs["input_ids"]
        if "attention_mask" in input_names:
            feed["attention_mask"] = inputs["attention_mask"]
        if "token_type_ids" in input_names:
            # Some models expect token_type_ids (all zeros for single-sequence)
            feed["token_type_ids"] = np.zeros_like(inputs["input_ids"])

        outputs = self._session.run(None, feed)
        # outputs[0] shape: (batch_size, seq_len, hidden_dim)
        hidden_states = outputs[0]

        # CLS pooling: take the first token's representation
        cls_embeddings = hidden_states[:, 0, :]

        # L2 normalization
        norms = np.linalg.norm(cls_embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)  # Avoid division by zero
        normalized = cls_embeddings / norms

        return normalized

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text into a normalized dense vector."""
        if not self.is_loaded:
            self.load()

        inputs = self._tokenize([text])
        embeddings = self._forward(inputs)
        return embeddings[0]

    def embed_batch(
        self, texts: Sequence[str], batch_size: int = 32
    ) -> List[np.ndarray]:
        """Embed multiple texts, processing in fixed-size batches."""
        if not self.is_loaded:
            self.load()

        total_batches = (len(texts) + batch_size - 1) // batch_size
        rss_start = _get_rss_mb()
        cpu_start = _cpu_time()
        logger.info(
            f"embed_batch: {len(texts)} texts in {total_batches} batches "
            f"(batch_size={batch_size}, max_tokens={self._max_length}), "
            f"RSS: {rss_start:.0f}MB"
        )

        results: List[np.ndarray] = []
        t0_total = _time.perf_counter()

        for i in range(0, len(texts), batch_size):
            batch_num = i // batch_size + 1
            batch = texts[i : i + batch_size]

            t0 = _time.perf_counter()
            cpu_t0 = _cpu_time()
            inputs = self._tokenize(batch)
            max_seq_len = inputs["input_ids"].shape[1]
            embeddings = self._forward(inputs)
            elapsed = _time.perf_counter() - t0
            cpu_elapsed = _cpu_time() - cpu_t0

            results.extend(embeddings[j] for j in range(len(batch)))
            rss_now = _get_rss_mb()
            cpu_pct = _get_cpu_percent(cpu_elapsed, elapsed)
            logger.info(
                f"  batch {batch_num}/{total_batches}: "
                f"{len(batch)} texts, seq_len={max_seq_len}, "
                f"{elapsed:.2f}s, "
                f"CPU: {cpu_pct:.0f}%, RSS: {rss_now:.0f}MB"
            )

        total_elapsed = _time.perf_counter() - t0_total
        total_cpu = _cpu_time() - cpu_start
        rss_end = _get_rss_mb()
        peak_rss = _get_peak_rss_mb()
        logger.info(
            f"embed_batch complete: {len(texts)} texts in {total_elapsed:.1f}s "
            f"({len(texts) / max(total_elapsed, 0.001):.1f} texts/sec), "
            f"CPU: {_get_cpu_percent(total_cpu, total_elapsed):.0f}% avg, "
            f"RSS: {rss_end:.0f}MB (delta {rss_end - rss_start:+.0f}MB), "
            f"peak RSS: {peak_rss:.0f}MB"
        )

        return results

    def embed_batch_adaptive(
        self,
        texts: Sequence[str],
        memory_budget_gb: float = 6.0,
    ) -> List[np.ndarray]:
        """Embed texts with optimal batching based on actual token lengths.

        Instead of forcing texts into fixed-size buckets (which wastes
        memory on padding), this method:

        1. Pre-tokenizes all texts to get exact token counts
        2. Sorts by length
        3. Greedily forms maximally-large batches where
           batch_size × per_item_cost(max_len_in_batch) ≤ budget
        4. Pads only to the actual longest text in each batch

        Because attention cost is O(seq²), even small reductions in
        padding yield significant batch size improvements.  A group of
        texts at [500, 510, 520, 530] tokens gets batched at max_len=530
        instead of being padded to the next bucket boundary (e.g. 768).

        Args:
            texts: Texts to embed.
            memory_budget_gb: Max attention memory in GB (default 6.0).

        Returns:
            List of embedding vectors in the same order as input texts.
        """
        if not self.is_loaded:
            self.load()

        if not texts:
            return []

        _ensure_imports()
        t0_total = _time.perf_counter()
        cpu_start = _cpu_time()
        rss_start = _get_rss_mb()

        # Model constants for gte-modernbert-base memory estimation.
        # ONNX Runtime keeps ~3 transformer layers of activations resident
        # simultaneously during inference. Per-item memory per layer:
        #   - Attention scores: heads × seq² × 4 bytes
        #   - Q/K/V projections: 3 × seq × hidden × 4 bytes
        #   - FFN intermediate: seq × ff_dim × 4 bytes
        #   - Output buffer: seq × hidden × 4 bytes
        _NUM_HEADS = 12
        _HIDDEN = 768
        _FF_DIM = 1152
        _EFF_LAYERS = 3  # conservative; observed ~2.5 from benchmarks
        _MAX_BATCH = 64  # cap per batch to avoid diminishing returns
        _MAX_SEQ = 8192  # pre-tokenization truncation cap
        budget_bytes = memory_budget_gb * 1024**3

        def _per_item_cost(seq_len: int) -> float:
            """Memory cost (bytes) for one item at the given sequence length."""
            attn = _NUM_HEADS * seq_len * seq_len * 4
            linear = seq_len * (3 * _HIDDEN + _FF_DIM + _HIDDEN) * 4
            return (attn + linear) * _EFF_LAYERS

        # Pre-tokenize to get actual token counts (fast, no ONNX inference).
        self._tokenizer.no_padding()
        self._tokenizer.enable_truncation(max_length=_MAX_SEQ)
        try:
            encodings = self._tokenizer.encode_batch(list(texts))
            lengths = [len(e.ids) for e in encodings]
        finally:
            self._tokenizer.enable_padding(length=None)  # Restore dynamic padding
            self._tokenizer.enable_truncation(max_length=self._max_length)

        # Sort by token count, keeping original indices for reordering
        indexed = sorted(enumerate(lengths), key=lambda x: x[1])
        n = len(indexed)
        results: List[Optional[np.ndarray]] = [None] * n

        # Greedy optimal batching with length-ratio constraint.
        #
        # Two constraints per batch (texts are sorted by token count):
        #   1. Memory: batch_size × per_item_cost(max_len) ≤ budget
        #   2. Padding: max_len ≤ min_len × _PAD_RATIO  (limits wasted compute)
        #
        # Constraint #2 is critical because attention is O(seq²). Without it,
        # a batch of [100, 200, ..., 2000] pads the 100-token text to 2000,
        # costing 2000²=4M ops instead of 100²=10K — a 400x waste.
        # With ratio=2, the batch splits at length boundaries so the shortest
        # item is never padded beyond 2× its actual length (≤4× compute waste).
        _PAD_RATIO = 1.1

        i = 0
        num_batches = 0
        total_processed = 0
        batch_sizes: list = []

        while i < n:
            min_len = indexed[i][1]

            # Upper bound on j from length-ratio constraint:
            # find largest j where indexed[j][1] ≤ min_len × _PAD_RATIO
            max_allowed = max(int(min_len * _PAD_RATIO), min_len + 1)
            ratio_limit = i
            while (
                ratio_limit + 1 < n
                and indexed[ratio_limit + 1][1] <= max_allowed
            ):
                ratio_limit += 1

            # Binary search within [i, min(ratio_limit, i+MAX_BATCH-1)]
            # for largest j where memory constraint holds
            hi_bound = min(ratio_limit, i + _MAX_BATCH - 1, n - 1)
            lo, hi = i, hi_bound
            while lo < hi:
                mid = lo + (hi - lo + 1) // 2  # upper-mid bias
                candidate_size = mid - i + 1
                max_len = indexed[mid][1]
                if candidate_size * _per_item_cost(max_len) <= budget_bytes:
                    lo = mid  # feasible → try larger
                else:
                    hi = mid - 1  # too large → shrink
            j = lo  # largest feasible end (at minimum j == i → batch of 1)

            batch_size = j - i + 1
            max_len = indexed[j][1]
            num_batches += 1

            # Set truncation to actual max of this batch — no excess padding
            self._tokenizer.enable_truncation(max_length=max_len)

            batch_indices = [indexed[k][0] for k in range(i, j + 1)]
            batch_texts = [texts[idx] for idx in batch_indices]

            logger.debug(
                f"  batch {num_batches}: {batch_size} texts, "
                f"seq_len={indexed[i][1]}-{max_len}, "
                f"mem≈{batch_size * _per_item_cost(max_len) / 1024**3:.2f}GB"
            )

            try:
                inputs = self._tokenize(batch_texts)
                embeddings = self._forward(inputs)
                for k, idx in enumerate(batch_indices):
                    results[idx] = embeddings[k]
                total_processed += batch_size
                batch_sizes.append(batch_size)
            finally:
                self._tokenizer.enable_truncation(max_length=self._max_length)

            i = j + 1

        total_elapsed = _time.perf_counter() - t0_total
        total_cpu = _cpu_time() - cpu_start
        rss_end = _get_rss_mb()
        peak_rss = _get_peak_rss_mb()

        if batch_sizes:
            logger.info(
                f"adaptive batching: {total_processed} texts in "
                f"{num_batches} batches "
                f"({min(batch_sizes)}-{max(batch_sizes)} items/batch, "
                f"avg {sum(batch_sizes) / len(batch_sizes):.1f}), "
                f"seq_len {indexed[0][1]}-{indexed[-1][1]}, "
                f"{total_elapsed:.1f}s "
                f"({total_processed / max(total_elapsed, 0.001):.1f} texts/sec), "
                f"CPU: {_get_cpu_percent(total_cpu, total_elapsed):.0f}% avg, "
                f"RSS: {rss_end:.0f}MB (delta {rss_end - rss_start:+.0f}MB), "
                f"peak RSS: {peak_rss:.0f}MB"
            )

        return results  # type: ignore[return-value]


class OnnxRerankerProvider:
    """Reranker provider using direct ONNX Runtime inference.

    Loads Alibaba-NLP/gte-reranker-modernbert-base as a text-classification
    model. Takes query-document pairs and produces relevance scores.

    The reranker ONNX model must be exported beforehand via:
        optimum-cli export onnx -m Alibaba-NLP/gte-reranker-modernbert-base \\
            --task text-classification --optimize O4 ./gte-reranker-onnx/

    Args:
        model_id: HuggingFace model ID or local path to exported ONNX.
        onnx_filename: Path to the ONNX model file.
        max_length: Maximum token length for query+document pairs.
        cache_dir: Optional custom cache directory.
        providers: Provider preference string ("auto", "cpu", or comma-separated).
    """

    def __init__(
        self,
        model_id: str = "Alibaba-NLP/gte-reranker-modernbert-base",
        onnx_filename: str = "onnx/model.onnx",
        max_length: int = 8192,
        cache_dir: Optional[Path] = None,
        providers: str = "auto",
    ) -> None:
        self._model_id = model_id
        self._onnx_filename = onnx_filename
        self._max_length = max_length
        self._cache_dir = cache_dir
        self._providers_pref = providers
        self._session: Optional[object] = None
        self._tokenizer: Optional[object] = None

    def load(self) -> None:
        """Download and load the ONNX reranker model and tokenizer."""
        if self._session is not None:
            return

        _ensure_imports()
        rss_before = _get_rss_mb()
        logger.info(
            f"Loading reranker model: {self._model_id} "
            f"(RSS before load: {rss_before:.0f}MB)"
        )

        model_dir = _download_model_files(
            self._model_id,
            [self._onnx_filename, "tokenizer.json", "tokenizer_config.json"],
            self._cache_dir,
        )

        # Load tokenizer
        tokenizer_path = model_dir / "tokenizer.json"
        self._tokenizer = _tokenizers.Tokenizer.from_file(str(tokenizer_path))
        self._tokenizer.enable_truncation(max_length=self._max_length)
        self._tokenizer.enable_padding(length=None)

        # Load ONNX model
        onnx_path = model_dir / self._onnx_filename
        if not onnx_path.exists():
            raise FileNotFoundError(
                f"Reranker ONNX model not found at {onnx_path}. "
                f"Export it first with: optimum-cli export onnx "
                f"-m {self._model_id} --task text-classification ./onnx/"
            )

        sess_options = _ort.SessionOptions()
        sess_options.graph_optimization_level = (
            _ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        providers = _resolve_providers(self._providers_pref)

        # Disable CPU memory arena (same rationale as embedder)
        if providers == ["CPUExecutionProvider"]:
            sess_options.enable_cpu_mem_arena = False

        cuda_opts = _cuda_session_options()
        provider_options = [cuda_opts.get(p, {}) for p in providers]

        self._session = _ort.InferenceSession(
            str(onnx_path),
            sess_options=sess_options,
            providers=providers,
            provider_options=provider_options,
        )

        active = self._session.get_providers()
        rss_after = _get_rss_mb()
        logger.info(
            f"Reranker model loaded: {self._model_id}, providers={active}, "
            f"RSS: {rss_after:.0f}MB (+{rss_after - rss_before:.0f}MB)"
        )

    def unload(self) -> None:
        """Release reranker model from memory."""
        rss_before = _get_rss_mb()
        self._session = None
        self._tokenizer = None
        rss_after = _get_rss_mb()
        logger.info(
            f"Reranker model unloaded: {self._model_id}, "
            f"RSS: {rss_after:.0f}MB (freed ~{rss_before - rss_after:.0f}MB)"
        )

    @property
    def is_loaded(self) -> bool:
        return self._session is not None

    def _score_pairs(self, query: str, documents: Sequence[str]) -> List[float]:
        """Score query-document pairs via the cross-encoder model."""
        # Encode each (query, document) pair
        # tokenizers library handles [CLS] query [SEP] document [SEP] automatically
        encodings = []
        for doc in documents:
            encoded = self._tokenizer.encode(query, doc)
            encodings.append(encoded)

        # Pad to max length in batch
        max_len = max(len(e.ids) for e in encodings)
        input_ids = np.zeros((len(encodings), max_len), dtype=np.int64)
        attention_mask = np.zeros((len(encodings), max_len), dtype=np.int64)
        token_type_ids = np.zeros((len(encodings), max_len), dtype=np.int64)

        for i, encoding in enumerate(encodings):
            length = len(encoding.ids)
            input_ids[i, :length] = encoding.ids
            attention_mask[i, :length] = encoding.attention_mask
            if hasattr(encoding, "type_ids") and encoding.type_ids:
                token_type_ids[i, :length] = encoding.type_ids

        # Run inference
        input_names = {inp.name for inp in self._session.get_inputs()}
        feed = {}
        if "input_ids" in input_names:
            feed["input_ids"] = input_ids
        if "attention_mask" in input_names:
            feed["attention_mask"] = attention_mask
        if "token_type_ids" in input_names:
            feed["token_type_ids"] = token_type_ids

        outputs = self._session.run(None, feed)
        logits = outputs[0]  # shape: (batch_size, num_labels)

        # For binary relevance models, take the positive class score
        if logits.shape[-1] == 1:
            scores = logits[:, 0].tolist()
        elif logits.shape[-1] >= 2:
            # Softmax and take positive class probability
            exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
            scores = probs[:, -1].tolist()  # Last class = relevant
        else:
            scores = logits.flatten().tolist()

        return scores

    def rerank(
        self, query: str, documents: Sequence[str], top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """Rerank documents by relevance to query."""
        if not self.is_loaded:
            self.load()

        if not documents:
            return []

        scores = self._score_pairs(query, documents)
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        return indexed_scores[:top_k]
