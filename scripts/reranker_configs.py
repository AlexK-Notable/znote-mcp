"""Model registry for the reranker benchmark test matrix.

Each configuration specifies everything needed to instantiate an
OnnxRerankerProvider for benchmarking: model repo, ONNX filename,
max token length, and any extra files needed for external weights.

7 reranker configurations spanning lightweight cross-encoders (MiniLM)
through large rerankers (bge-reranker-large, bge-reranker-v2-m3).
"""

from __future__ import annotations

from typing import Any, Dict, List

# Type alias for a single reranker configuration
RerankerConfig = Dict[str, Any]


# ---------------------------------------------------------------------------
# Reranker configurations (7 models)
# ---------------------------------------------------------------------------

RERANKERS: Dict[str, RerankerConfig] = {
    # -------------------------------------------------------------------
    # cross-encoder/ms-marco-MiniLM-L-6-v2  (~22M params, 512 tokens)
    # Lightweight baseline cross-encoder trained on MS MARCO.
    # Note: repo may only have INT8 ONNX; FP32 may need optimum export.
    # -------------------------------------------------------------------
    "minilm-l6-rerank": {
        "model_id": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "onnx_filename": "onnx/model.onnx",
        "max_tokens": 512,
    },

    # -------------------------------------------------------------------
    # cross-encoder/ms-marco-MiniLM-L-12-v2  (~33M params, 512 tokens)
    # Deeper MiniLM cross-encoder, slightly better quality.
    # Note: repo may only have INT8 ONNX; FP32 may need optimum export.
    # -------------------------------------------------------------------
    "minilm-l12-rerank": {
        "model_id": "cross-encoder/ms-marco-MiniLM-L-12-v2",
        "onnx_filename": "onnx/model.onnx",
        "max_tokens": 512,
    },

    # -------------------------------------------------------------------
    # BAAI/bge-reranker-base  (~300M params, 512 tokens)
    # BGE cross-encoder, solid mid-range reranker.
    # -------------------------------------------------------------------
    "bge-reranker-base": {
        "model_id": "BAAI/bge-reranker-base",
        "onnx_filename": "onnx/model.onnx",
        "max_tokens": 512,
    },

    # -------------------------------------------------------------------
    # BAAI/bge-reranker-large  (~560M params, 512 tokens)
    # Large BGE cross-encoder with external weight file.
    # -------------------------------------------------------------------
    "bge-reranker-large": {
        "model_id": "BAAI/bge-reranker-large",
        "onnx_filename": "onnx/model.onnx",
        "max_tokens": 512,
        "extra_files": ["onnx/model.onnx_data"],
    },

    # -------------------------------------------------------------------
    # Alibaba-NLP/gte-reranker-modernbert-base  (~149M params, 8192 tokens)
    # Current production reranker for znote-mcp. Long-context capable.
    # -------------------------------------------------------------------
    "gte-reranker": {
        "model_id": "Alibaba-NLP/gte-reranker-modernbert-base",
        "onnx_filename": "onnx/model.onnx",
        "max_tokens": 8192,
    },

    # -------------------------------------------------------------------
    # hooman650/bge-reranker-v2-m3-onnx-o4  (~568M params, 8192 tokens)
    # Community O4 quantized export of BAAI/bge-reranker-v2-m3.
    # Official BAAI repo has no ONNX; this is the best available.
    # ONNX file at root (not onnx/ subdirectory).
    # -------------------------------------------------------------------
    "bge-reranker-v2-m3": {
        "model_id": "hooman650/bge-reranker-v2-m3-onnx-o4",
        "onnx_filename": "model.onnx",
        "max_tokens": 8192,
        "extra_files": ["model.onnx.data"],
    },

    # -------------------------------------------------------------------
    # jinaai/jina-reranker-v2-base-multilingual  (~278M params, 1024 tokens)
    # Multilingual reranker. Needs trust_remote_code for PyTorch but
    # ONNX inference should work with standard tokenizer. May fail.
    # -------------------------------------------------------------------
    "jina-reranker-v2": {
        "model_id": "jinaai/jina-reranker-v2-base-multilingual",
        "onnx_filename": "onnx/model.onnx",
        "max_tokens": 1024,
    },
}


def get_config(key: str) -> RerankerConfig:
    """Return a reranker config by key, raising KeyError if not found."""
    return RERANKERS[key]


def list_configs() -> List[str]:
    """List all reranker config keys."""
    return list(RERANKERS.keys())
