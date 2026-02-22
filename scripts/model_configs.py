"""Model registry for the embedding benchmark test matrix.

Each configuration specifies everything needed to instantiate an
OnnxEmbeddingProvider and a TextChunker for benchmarking: model repo,
ONNX filename, embedding dimension, tokenizer max tokens, chunk size
for splitting long notes, output mode (CLS pooling vs direct), and
architecture profile for adaptive batch memory estimation.

12 FP32 configurations: 9 models with chunk_size variants for models
supporting >2048 tokens.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# Type alias for a single model configuration
ModelConfig = Dict[str, Any]

# ---------------------------------------------------------------------------
# Architecture profiles for adaptive batch memory estimation
# ---------------------------------------------------------------------------

_PROFILE_MINILM = {
    "num_heads": 12,
    "hidden_dim": 384,
    "ff_dim": 1536,
    "eff_layers": 3,
}

_PROFILE_BGE_SMALL = {
    "num_heads": 12,
    "hidden_dim": 384,
    "ff_dim": 1536,
    "eff_layers": 3,
}

_PROFILE_BGE_BASE = {
    "num_heads": 12,
    "hidden_dim": 768,
    "ff_dim": 3072,
    "eff_layers": 3,
}

_PROFILE_GTE_MODERNBERT = {
    "num_heads": 12,
    "hidden_dim": 768,
    "ff_dim": 1152,
    "eff_layers": 3,
}

_PROFILE_NOMIC = {
    "num_heads": 12,
    "hidden_dim": 768,
    "ff_dim": 3072,
    "eff_layers": 3,
}

_PROFILE_ARCTIC_M = {
    "num_heads": 12,
    "hidden_dim": 768,
    "ff_dim": 3072,
    "eff_layers": 3,
}

_PROFILE_ARCTIC_L = {
    "num_heads": 16,
    "hidden_dim": 1024,
    "ff_dim": 4096,
    "eff_layers": 3,
}

_PROFILE_MXBAI_LARGE = {
    "num_heads": 16,
    "hidden_dim": 1024,
    "ff_dim": 4096,
    "eff_layers": 3,
}

_PROFILE_EMBEDDINGGEMMA = {
    "num_heads": 4,
    "hidden_dim": 768,
    "ff_dim": 6144,
    "eff_layers": 3,
}


# ---------------------------------------------------------------------------
# Model configurations (12 FP32 configs)
# ---------------------------------------------------------------------------

MODELS: Dict[str, ModelConfig] = {
    # -----------------------------------------------------------------------
    # sentence-transformers/all-MiniLM-L6-v2  (22M params, 384-dim, 512 tokens)
    # Lightweight baseline — the most widely deployed embedding model.
    # -----------------------------------------------------------------------
    "minilm-fp32": {
        "model_id": "sentence-transformers/all-MiniLM-L6-v2",
        "onnx_filename": "onnx/model.onnx",
        "dim": 384,
        "max_tokens": 512,
        "chunk_size": 512,
        "output_mode": "cls",
        "profile": _PROFILE_MINILM,
    },

    # -----------------------------------------------------------------------
    # BAAI/bge-small-en-v1.5  (33M params, 384-dim, 512 tokens)
    # -----------------------------------------------------------------------
    "bge-small-fp32": {
        "model_id": "BAAI/bge-small-en-v1.5",
        "onnx_filename": "onnx/model.onnx",
        "dim": 384,
        "max_tokens": 512,
        "chunk_size": 512,
        "output_mode": "cls",
        "profile": _PROFILE_BGE_SMALL,
    },

    # -----------------------------------------------------------------------
    # BAAI/bge-base-en-v1.5  (109M params, 768-dim, 512 tokens)
    # -----------------------------------------------------------------------
    "bge-base-fp32": {
        "model_id": "BAAI/bge-base-en-v1.5",
        "onnx_filename": "onnx/model.onnx",
        "dim": 768,
        "max_tokens": 512,
        "chunk_size": 512,
        "output_mode": "cls",
        "profile": _PROFILE_BGE_BASE,
    },

    # -----------------------------------------------------------------------
    # Alibaba-NLP/gte-modernbert-base  (149M params, 768-dim, 8192 tokens)
    # Current production model for znote-mcp.
    # -----------------------------------------------------------------------
    "gte-modernbert-c2048-fp32": {
        "model_id": "Alibaba-NLP/gte-modernbert-base",
        "onnx_filename": "onnx/model.onnx",
        "dim": 768,
        "max_tokens": 8192,
        "chunk_size": 2048,
        "output_mode": "cls",
        "profile": _PROFILE_GTE_MODERNBERT,
    },
    "gte-modernbert-c8192-fp32": {
        "model_id": "Alibaba-NLP/gte-modernbert-base",
        "onnx_filename": "onnx/model.onnx",
        "dim": 768,
        "max_tokens": 8192,
        "chunk_size": 8192,
        "output_mode": "cls",
        "profile": _PROFILE_GTE_MODERNBERT,
    },

    # -----------------------------------------------------------------------
    # nomic-ai/nomic-embed-text-v1.5  (137M params, 768-dim, 2048 tokens)
    # NomicBERT with rotary embeddings and Matryoshka dimension support.
    # -----------------------------------------------------------------------
    "nomic-v1.5-fp32": {
        "model_id": "nomic-ai/nomic-embed-text-v1.5",
        "onnx_filename": "onnx/model.onnx",
        "dim": 768,
        "max_tokens": 2048,
        "chunk_size": 2048,
        "output_mode": "cls",
        "profile": _PROFILE_NOMIC,
    },

    # -----------------------------------------------------------------------
    # Snowflake/snowflake-arctic-embed-m-v2.0  (305M params, 768-dim, 8192 tokens)
    # -----------------------------------------------------------------------
    "arctic-m-c2048-fp32": {
        "model_id": "Snowflake/snowflake-arctic-embed-m-v2.0",
        "onnx_filename": "onnx/model.onnx",
        "dim": 768,
        "max_tokens": 8192,
        "chunk_size": 2048,
        "output_mode": "cls",
        "profile": _PROFILE_ARCTIC_M,
    },
    "arctic-m-c8192-fp32": {
        "model_id": "Snowflake/snowflake-arctic-embed-m-v2.0",
        "onnx_filename": "onnx/model.onnx",
        "dim": 768,
        "max_tokens": 8192,
        "chunk_size": 8192,
        "output_mode": "cls",
        "profile": _PROFILE_ARCTIC_M,
    },

    # -----------------------------------------------------------------------
    # Snowflake/snowflake-arctic-embed-l-v2.0  (~400M params, 1024-dim, 8192 tokens)
    # -----------------------------------------------------------------------
    "arctic-l-c2048-fp32": {
        "model_id": "Snowflake/snowflake-arctic-embed-l-v2.0",
        "onnx_filename": "onnx/model.onnx",
        "extra_files": ["onnx/model.onnx_data"],
        "dim": 1024,
        "max_tokens": 8192,
        "chunk_size": 2048,
        "output_mode": "cls",
        "profile": _PROFILE_ARCTIC_L,
    },
    "arctic-l-c8192-fp32": {
        "model_id": "Snowflake/snowflake-arctic-embed-l-v2.0",
        "onnx_filename": "onnx/model.onnx",
        "extra_files": ["onnx/model.onnx_data"],
        "dim": 1024,
        "max_tokens": 8192,
        "chunk_size": 8192,
        "output_mode": "cls",
        "profile": _PROFILE_ARCTIC_L,
    },

    # -----------------------------------------------------------------------
    # mixedbread-ai/mxbai-embed-large-v1  (~335M params, 1024-dim, 512 tokens)
    # High-MTEB BERT model, second 1024-dim option alongside arctic-l.
    # -----------------------------------------------------------------------
    "mxbai-large-fp32": {
        "model_id": "mixedbread-ai/mxbai-embed-large-v1",
        "onnx_filename": "onnx/model.onnx",
        "dim": 1024,
        "max_tokens": 512,
        "chunk_size": 512,
        "output_mode": "cls",
        "profile": _PROFILE_MXBAI_LARGE,
    },

    # -----------------------------------------------------------------------
    # onnx-community/embeddinggemma-300m-ONNX  (303M params, 768-dim, 2048 tokens)
    # Decoder model — produces pre-pooled embeddings, no CLS pooling needed.
    # -----------------------------------------------------------------------
    "embeddinggemma-fp32": {
        "model_id": "onnx-community/embeddinggemma-300m-ONNX",
        "onnx_filename": "onnx/model.onnx",
        "extra_files": ["onnx/model.onnx_data"],
        "dim": 768,
        "max_tokens": 2048,
        "chunk_size": 2048,
        "output_mode": "direct",
        "profile": _PROFILE_EMBEDDINGGEMMA,
    },
}


def get_config(key: str) -> ModelConfig:
    """Return a model config by key, raising KeyError if not found."""
    return MODELS[key]


def list_configs() -> List[str]:
    """List all config keys."""
    return list(MODELS.keys())
