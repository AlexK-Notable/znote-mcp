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
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Lazy imports — these are optional dependencies
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
    """

    def __init__(
        self,
        model_id: str = "Alibaba-NLP/gte-modernbert-base",
        onnx_filename: str = "onnx/model.onnx",
        max_length: int = 8192,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self._model_id = model_id
        self._onnx_filename = onnx_filename
        self._max_length = max_length
        self._cache_dir = cache_dir
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
        logger.info(f"Loading embedding model: {self._model_id}")

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

        # Load ONNX model
        onnx_path = model_dir / self._onnx_filename
        if not onnx_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found at {onnx_path}. "
                f"Check that {self._model_id} has an ONNX model at {self._onnx_filename}"
            )

        sess_options = _ort.SessionOptions()
        sess_options.graph_optimization_level = (
            _ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        # Use CPU provider (most portable, avoids GPU dependency)
        self._session = _ort.InferenceSession(
            str(onnx_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

        # Detect actual dimension from model output shape
        outputs = self._session.get_outputs()
        if outputs and len(outputs[0].shape) >= 2:
            self._dim = outputs[0].shape[-1]

        logger.info(
            f"Embedding model loaded: dim={self._dim}, max_tokens={self._max_length}"
        )

    def unload(self) -> None:
        """Release model from memory."""
        self._session = None
        self._tokenizer = None
        logger.info(f"Embedding model unloaded: {self._model_id}")

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
        """Embed multiple texts, processing in batches."""
        if not self.is_loaded:
            self.load()

        results: List[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self._tokenize(batch)
            embeddings = self._forward(inputs)
            results.extend(embeddings[j] for j in range(len(batch)))

        return results


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
    """

    def __init__(
        self,
        model_id: str = "Alibaba-NLP/gte-reranker-modernbert-base",
        onnx_filename: str = "onnx/model.onnx",
        max_length: int = 8192,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self._model_id = model_id
        self._onnx_filename = onnx_filename
        self._max_length = max_length
        self._cache_dir = cache_dir
        self._session: Optional[object] = None
        self._tokenizer: Optional[object] = None

    def load(self) -> None:
        """Download and load the ONNX reranker model and tokenizer."""
        if self._session is not None:
            return

        _ensure_imports()
        logger.info(f"Loading reranker model: {self._model_id}")

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
        self._session = _ort.InferenceSession(
            str(onnx_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

        logger.info(f"Reranker model loaded: {self._model_id}")

    def unload(self) -> None:
        """Release reranker model from memory."""
        self._session = None
        self._tokenizer = None
        logger.info(f"Reranker model unloaded: {self._model_id}")

    @property
    def is_loaded(self) -> bool:
        return self._session is not None

    def _score_pairs(
        self, query: str, documents: Sequence[str]
    ) -> List[float]:
        """Score query-document pairs via the cross-encoder model."""
        # Encode each (query, document) pair
        # tokenizers library handles [CLS] query [SEP] document [SEP] automatically
        encodings = []
        for doc in documents:
            encoded = self._tokenizer.encode(query, doc)
            encodings.append(encoded)

        # Pad to max length in batch
        max_len = max(len(e.ids) for e in encodings)
        input_ids = np.zeros((len(pairs), max_len), dtype=np.int64)
        attention_mask = np.zeros((len(pairs), max_len), dtype=np.int64)
        token_type_ids = np.zeros((len(pairs), max_len), dtype=np.int64)

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
