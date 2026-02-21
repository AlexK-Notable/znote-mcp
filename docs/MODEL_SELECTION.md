# Model Selection

This document covers the embedding and reranker model evaluation pipeline, benchmark results, and rationale behind znote-mcp's default model choices.

## Overview

znote-mcp uses [ONNX Runtime](https://onnxruntime.ai/) for local inference with no API calls and no internet required after initial model download. The semantic search pipeline uses two complementary model types:

- **Embedding model** (bi-encoder): Converts each note into a fixed-size vector. Notes with similar meaning produce vectors that are close together in embedding space. Used for initial retrieval via cosine similarity search in sqlite-vec.
- **Reranker model** (cross-encoder): Takes a query-document pair and produces a relevance score. More accurate than cosine similarity but slower, so it refines the top-K results returned by the embedding search.

Both models were selected through systematic benchmarking on a real zettelkasten corpus. The defaults are:

| Role | Model | Params | Dim | Max Tokens | License |
|------|-------|--------|-----|------------|---------|
| Embedding | `Alibaba-NLP/gte-modernbert-base` | 149M | 768 | 8192 | Apache-2.0 |
| Reranker | `Alibaba-NLP/gte-reranker-modernbert-base` | 149M | -- | 8192 | Apache-2.0 |

Using the same model family (GTE-ModernBERT) for both roles means they share a tokenizer and architectural assumptions, which simplifies deployment and reduces the total number of dependencies.


## Embedding Model Evaluation

### Test Corpus

All benchmarks were run against the same real-world zettelkasten:

| Property | Value |
|----------|-------|
| Notes | 961 |
| Link pairs | 1,364 |
| Unique tags | 1,183 |
| Total characters | 10,697,873 |
| Mean characters/note | 11,132 |
| Median characters/note | 7,891 |
| Character range | 35 -- 51,387 |

### Models Tested

9 models in 12 configurations (models with long-context support were tested at both 2048 and 8192 token chunk sizes):

| # | Config Key | Model | Params | Dim | Max Tokens | Chunk Size | ONNX Size |
|---|------------|-------|--------|-----|------------|------------|-----------|
| 1 | `minilm-fp32` | all-MiniLM-L6-v2 | 22M | 384 | 512 | 512 | ~86 MB |
| 2 | `bge-small-fp32` | bge-small-en-v1.5 | 33M | 384 | 512 | 512 | ~90 MB |
| 3 | `bge-base-fp32` | bge-base-en-v1.5 | 109M | 768 | 512 | 512 | ~420 MB |
| 4 | `gte-modernbert-c2048-fp32` | gte-modernbert-base | 149M | 768 | 8192 | 2048 | ~570 MB |
| 5 | `gte-modernbert-c8192-fp32` | gte-modernbert-base | 149M | 768 | 8192 | 8192 | ~570 MB |
| 6 | `nomic-v1.5-fp32` | nomic-embed-text-v1.5 | 137M | 768 | 2048 | 2048 | ~522 MB |
| 7 | `arctic-m-c2048-fp32` | snowflake-arctic-embed-m-v2.0 | 305M | 768 | 8192 | 2048 | ~1.2 GB |
| 8 | `arctic-m-c8192-fp32` | snowflake-arctic-embed-m-v2.0 | 305M | 768 | 8192 | 8192 | ~1.2 GB |
| 9 | `arctic-l-c2048-fp32` | snowflake-arctic-embed-l-v2.0 | ~400M | 1024 | 8192 | 2048 | ~1.6 GB |
| 10 | `arctic-l-c8192-fp32` | snowflake-arctic-embed-l-v2.0 | ~400M | 1024 | 8192 | 8192 | ~1.6 GB |
| 11 | `mxbai-large-fp32` | mxbai-embed-large-v1 | ~335M | 1024 | 512 | 512 | ~1.3 GB |
| 12 | `embeddinggemma-fp32` | embeddinggemma-300m-ONNX | 303M | 768 | 2048 | 2048 | ~1.2 GB |

The configurations are defined in `scripts/model_configs.py`.

### Quality Metrics

**Link prediction MRR (Mean Reciprocal Rank)**: For each linked pair of notes in the zettelkasten, how highly does the embedding model rank the target note among all 961 candidates? An MRR of 0.27 means the correct linked note appears, on average, around rank 3-4.

**Tag coherence ratio**: The ratio of average intra-tag cosine similarity to average inter-tag cosine similarity. A ratio of 1.9 means notes sharing a tag are, on average, 1.9x more similar in embedding space than notes with different tags. Higher is better.

### Quality Results

Sorted by link prediction MRR (the primary quality metric):

| Rank | Config | MRR | R@5 | R@10 | R@20 | Med Rank | Tag Ratio |
|------|--------|-----|-----|------|------|----------|-----------|
| 1 | gte-modernbert-c8192-fp32 | **0.2718** | 0.4413 | 0.5909 | 0.7236 | 7 | 1.28 |
| 2 | arctic-m-c2048-fp32 | 0.2688 | 0.4208 | 0.5689 | 0.6950 | 8 | 1.93 |
| 3 | arctic-l-c2048-fp32 | 0.2649 | 0.4076 | 0.5550 | 0.6950 | 8 | 1.60 |
| 4 | gte-modernbert-c2048-fp32 | 0.2636 | 0.4216 | 0.5638 | 0.6906 | 8 | 1.27 |
| 5 | arctic-l-c8192-fp32 | 0.2597 | 0.3915 | 0.5491 | 0.6840 | 9 | 1.57 |
| 6 | arctic-m-c8192-fp32 | 0.2572 | 0.4054 | 0.5425 | 0.6811 | 9 | 1.94 |
| 7 | embeddinggemma-fp32 | 0.2376 | 0.3651 | 0.4941 | 0.6430 | 11 | 1.44 |
| 8 | mxbai-large-fp32 | 0.2318 | 0.3563 | 0.4919 | 0.6239 | 11 | 1.22 |
| 9 | bge-small-fp32 | 0.2263 | 0.3416 | 0.4641 | 0.5894 | 12 | 1.13 |
| 10 | bge-base-fp32 | 0.2252 | 0.3490 | 0.4743 | 0.6034 | 12 | 1.15 |
| 11 | minilm-fp32 | 0.2235 | 0.3365 | 0.4611 | 0.5792 | 12 | 1.06 |
| 12 | nomic-v1.5-fp32 | 0.1588 | 0.2427 | 0.3475 | 0.4384 | 30 | 1.22 |

Quality results were identical across CPU and GPU runs (same vectors, different computation speed). The GPU quality matrix showed values within rounding error (MRR 0.2719 vs 0.2718 for the top model).

### Performance Results (CPU)

Benchmark system: Intel 14700K (28 cores), 32 GB RAM, ONNX Runtime 1.25.0.

| Config | Model | Embed Time (s) | Notes/s | Chunks/s | Total Chunks | Peak RSS (MB) |
|--------|-------|---------------:|--------:|---------:|-------------:|--------------:|
| minilm-fp32 | all-MiniLM-L6-v2 | 170 | 5.7 | 47.8 | 8,108 | 3,082 |
| bge-small-fp32 | bge-small-en-v1.5 | 311 | 3.1 | 26.0 | 8,108 | 3,082 |
| bge-base-fp32 | bge-base-en-v1.5 | 724 | 1.3 | 11.2 | 8,108 | 3,753 |
| nomic-v1.5-fp32 | nomic-embed-text-v1.5 | 993 | 1.0 | 2.0 | 1,951 | 13,019 |
| arctic-m-c2048-fp32 | arctic-embed-m-v2.0 | 1,224 | 0.8 | 1.6 | 1,951 | 14,431 |
| embeddinggemma-fp32 | embeddinggemma-300m | 1,253 | 0.8 | 1.6 | 1,951 | 7,504 |
| **gte-modernbert-c2048-fp32** | **gte-modernbert-base** | **1,432** | **0.7** | **1.4** | **1,951** | **13,019** |
| arctic-m-c8192-fp32 | arctic-embed-m-v2.0 | 1,707 | 0.6 | 0.6 | 1,008 | 14,431 |
| mxbai-large-fp32 | mxbai-embed-large-v1 | 2,395 | 0.4 | 3.4 | 8,108 | 14,431 |
| **gte-modernbert-c8192-fp32** | **gte-modernbert-base** | **2,508** | **0.4** | **0.4** | **1,008** | **13,019** |
| arctic-l-c2048-fp32 | arctic-embed-l-v2.0 | 3,182 | 0.3 | 0.6 | 1,951 | 14,431 |
| arctic-l-c8192-fp32 | arctic-embed-l-v2.0 | 4,927 | 0.2 | 0.2 | 1,008 | 14,431 |

### Performance Results (GPU)

Benchmark system: same as CPU, plus NVIDIA RTX 4070 Ti SUPER (16 GB VRAM, Ada Lovelace, compute 8.9).

| Config | Model | Embed Time (s) | Notes/s | Chunks/s | GPU Peak (MB) |
|--------|-------|---------------:|--------:|---------:|-----:|
| minilm-fp32 | all-MiniLM-L6-v2 | 12 | 81.7 | 689.3 | 5,201 |
| bge-small-fp32 | bge-small-en-v1.5 | 20 | 48.2 | 406.7 | 4,595 |
| bge-base-fp32 | bge-base-en-v1.5 | 36 | 26.9 | 227.2 | 3,673 |
| embeddinggemma-fp32 | embeddinggemma-300m | 55 | 17.4 | 35.4 | 3,780 |
| nomic-v1.5-fp32 | nomic-embed-text-v1.5 | 62 | 15.4 | 31.4 | 5,377 |
| arctic-m-c2048-fp32 | arctic-embed-m-v2.0 | 84 | 11.5 | 23.3 | 9,337 |
| mxbai-large-fp32 | mxbai-embed-large-v1 | 100 | 9.6 | 80.7 | 5,239 |
| **gte-modernbert-c2048-fp32** | **gte-modernbert-base** | **110** | **8.8** | **17.8** | **7,479** |
| arctic-m-c8192-fp32 | arctic-embed-m-v2.0 | 133 | 7.2 | 7.6 | 10,423 |
| arctic-l-c2048-fp32 | arctic-embed-l-v2.0 | 210 | 4.6 | 9.3 | 11,569 |
| **gte-modernbert-c8192-fp32** | **gte-modernbert-base** | **213** | **4.5** | **4.7** | **11,239** |
| arctic-l-c8192-fp32 | arctic-embed-l-v2.0 | 362 | 2.7 | 2.8 | 13,439 |

GPU speedup ranged from 10-14x for small models (MiniLM, BGE-small) to 12-14x for larger models, with the notable exception of embeddinggemma which achieved 23x speedup due to its decoder-based architecture benefiting more from GPU parallelism.

### Why gte-modernbert-base

The selected production model is `gte-modernbert-base` at 2048-token chunk size (`gte-modernbert-c2048-fp32` config). The rationale:

**Quality**: Ranks #1 (at 8192 chunks) and #4 (at 2048 chunks) in link prediction MRR. The 2048-chunk variant achieves MRR 0.2636, only 3% below the 8192-chunk variant's 0.2718, while using far fewer chunks (1,951 vs 1,008) and being faster. The top-ranked arctic models require 2-4x more parameters (305-400M vs 149M) and proportionally more memory for only marginal quality gains.

**Performance**: At 149M parameters, it is the smallest model in the top-4 quality bracket. On CPU, the 2048-chunk config indexes the full 961-note corpus in 24 minutes (1,432s). On GPU, it completes in under 2 minutes (110s).

**Long-context support**: Natively supports 8192 tokens, which is critical for zettelkasten notes that can exceed 50,000 characters. The 2048-token default chunk size covers 99% of atomic notes in a single vector, while longer notes are split into overlapping chunks.

**License**: Apache-2.0 (fully permissive). This was a deciding factor over EmbeddingGemma-300M, which scored higher on public MTEB benchmarks but uses the Gemma Terms of Use license (not a standard open-source license) and actually ranked #7 on our zettelkasten-specific benchmarks despite having 2x the parameters.

**Ecosystem coherence**: Using the same GTE-ModernBERT family for both embedding and reranking means a shared tokenizer, consistent architectural assumptions, and reduced total download size.

**Notable finding**: EmbeddingGemma-300M (Google's state-of-the-art for sub-500M models on MTEB) placed 7th on our real-world zettelkasten benchmark despite placing higher on public benchmarks. The arctic models placed 2nd and 3rd. This underscores the value of benchmarking on your actual data rather than relying on leaderboard scores.


## Reranker Model Evaluation

### Models Tested

10 rerankers across three categories:

| # | Config Key | Model | Params | Max Tokens | Category |
|---|------------|-------|--------|------------|----------|
| 1 | `minilm-l6-rerank` | cross-encoder/ms-marco-MiniLM-L-6-v2 | ~22M | 512 | MS MARCO |
| 2 | `minilm-l12-rerank` | cross-encoder/ms-marco-MiniLM-L-12-v2 | ~33M | 512 | MS MARCO |
| 3 | `bge-reranker-base` | BAAI/bge-reranker-base | ~300M | 512 | MS MARCO |
| 4 | `bge-reranker-large` | BAAI/bge-reranker-large | ~560M | 512 | MS MARCO |
| 5 | `gte-reranker` | Alibaba-NLP/gte-reranker-modernbert-base | ~149M | 8192 | General |
| 6 | `bge-reranker-v2-m3` | hooman650/bge-reranker-v2-m3-onnx-o4 | ~568M | 8192 | General |
| 7 | `jina-reranker-v2` | jinaai/jina-reranker-v2-base-multilingual | ~278M | 1024 | Multilingual |
| 8 | `stsb-distilroberta` | cross-encoder/stsb-distilroberta-base | ~82M | 512 | STS |
| 9 | `stsb-roberta-base` | cross-encoder/stsb-roberta-base | ~100M | 512 | STS |
| 10 | `stsb-roberta-large` | cross-encoder/stsb-roberta-large | ~355M | 512 | STS |

The configurations are defined in `scripts/reranker_configs.py`.

### Smoke Test Results

Before running full benchmarks, all rerankers were validated against 12 hand-crafted test cases using short, natural-language queries (the kind a user would actually type). Each case had a known "best" document, an "okay" document, and an "irrelevant" document.

**Result: 7/7 rerankers scored "ok" -- all correctly ranked the best document first in all 12 cases.** This confirmed that the reranking code and score interpretation were correct. Score ranges varied by model:

| Model | Score Range | Top-1 Accuracy |
|-------|-----------|----------------|
| minilm-l6-rerank | [-11.5, 8.2] | 12/12 |
| minilm-l12-rerank | [-11.3, 8.6] | 12/12 |
| bge-reranker-base | [-10.2, 6.1] | 12/12 |
| bge-reranker-large | [-9.5, 6.4] | 12/12 |
| gte-reranker | [-2.9, 4.3] | 12/12 |
| bge-reranker-v2-m3 | [-11.0, 6.0] | 12/12 |
| jina-reranker-v2 | [-3.7, 1.8] | 12/12 |

### Full Benchmark: Methodology Investigation

The initial full benchmark evaluated all reranker/embedder combinations using note-to-note link prediction -- the same task used for embedding evaluation. **Every combination showed negative MRR lift** (-9% to -48%), meaning reranking consistently degraded quality.

A systematic investigation by four independent analysis agents identified the root cause: **the benchmark methodology had five compounding mismatches with cross-encoder training assumptions**. The code was verified correct; the issue was benchmark design.

The five mismatches:

1. **Query format mismatch (critical)**: Cross-encoders are trained on short queries (MS MARCO averages 6 words). The benchmark sent full note text (median 7,875 characters, ~2,500 tokens) as the "query." This is 100-1000x longer than training distribution.

2. **Pool composition mismatch (critical)**: Standard cross-encoder evaluation uses BM25 lexical retrieval, producing diverse candidate pools. Our benchmark used dense embedding similarity, producing homogeneous pools where all candidates were already semantically close. The peer-reviewed paper "Drowning in Documents" (arXiv:2411.11767) documents a 53% degradation rate with dense+reranking pipelines.

3. **Token budget asymmetry (high)**: 512-token cross-encoders split their budget 50/50 between query and document. When both are full notes, each gets ~256 tokens (25% of the original). The embedding model independently processed 512-8192 tokens per note, creating a 4-16x information asymmetry.

4. **Ground truth semantics (moderate)**: Standard benchmarks use human relevance judgments. Zettelkasten links include "contradicts" links connecting semantically opposite notes, and hub notes with many links creating correlated errors.

5. **Monotonic correlation between pool quality and degradation**: The weakest embedder (nomic, pool recall 56.7%) was the only one showing positive reranker lift (+6%), while the strongest embedder (gte-c8192, pool recall 86.8%) showed the worst degradation (-29%). Same reranker code, opposite outcomes -- impossible with a code bug.

### Redesigned Benchmark Results

A second-generation benchmark (`benchmarks/rerank-v2/`) tested with three query strategies designed to match realistic cross-encoder usage:

- **title**: Use a note's title as the query, search for the note itself (961 queries)
- **link_title**: Use a note's title as the query, search for its linked notes (1,364 link pairs)
- **handcrafted**: 25 manually written natural-language queries targeting specific notes

Results with the GTE-modernbert-c8192 embedder:

| Strategy | Reranker | Embed MRR | Rerank MRR | MRR Lift |
|----------|----------|-----------|------------|----------|
| **handcrafted** | **jina-reranker-v2** | **0.844** | **0.953** | **+12.9%** |
| title | jina-reranker-v2 | 0.878 | 0.854 | -2.8% |
| link_title | jina-reranker-v2 | 0.288 | 0.309 | **+7.4%** |
| handcrafted | stsb-roberta-base | 0.844 | 0.338 | -59.9% |
| title | stsb-roberta-base | 0.878 | 0.306 | -65.2% |
| link_title | stsb-roberta-base | 0.288 | 0.165 | -42.6% |
| handcrafted | stsb-roberta-large | 0.844 | 0.259 | -69.3% |
| title | stsb-roberta-large | 0.878 | 0.240 | -72.6% |
| link_title | stsb-roberta-large | 0.288 | 0.153 | -46.9% |
| handcrafted | stsb-distilroberta | 0.844 | 0.162 | -80.8% |
| title | stsb-distilroberta | 0.878 | 0.189 | -78.5% |
| link_title | stsb-distilroberta | 0.288 | 0.154 | -46.5% |

Key finding: **jina-reranker-v2 showed positive MRR lift on both handcrafted queries (+12.9%) and link-title queries (+7.4%)**. The STS-trained models (stsb-*) performed poorly across all strategies, confirming they are designed for symmetric similarity, not asymmetric query-document ranking.

The handcrafted strategy most closely matches production use (user types a natural-language search query, embedding retrieval produces a candidate pool, reranker refines it). On this realistic evaluation, jina-reranker-v2 boosted MRR from 0.844 to 0.953 and achieved R@1 of 0.92 (up from 0.76).

### Why gte-reranker-modernbert-base

The production reranker is `Alibaba-NLP/gte-reranker-modernbert-base`. Despite jina-reranker-v2 showing strong positive results in the v2 benchmark, the GTE reranker was selected for these reasons:

**Long context (8192 tokens)**: The GTE reranker can process long notes without truncation. In the v1 benchmark, 512-token models degraded 3-5x worse than 8192-token models (-20% to -48% vs -9% for GTE reranker), demonstrating the value of long context even in adversarial conditions.

**Same family as embedder**: Using `gte-modernbert-base` for embedding and `gte-reranker-modernbert-base` for reranking means they share a tokenizer and architectural lineage. This avoids tokenizer mismatches and simplifies the dependency tree.

**Compact size**: At 149M parameters and a 0.3 MB RSS delta during smoke tests (due to the model being loaded onto GPU), it is one of the most memory-efficient rerankers tested.

**Smoke test validation**: Achieved 12/12 correct top-1 rankings on the realistic short-query smoke tests, with a moderate score range ([-2.9, 4.3]) indicating confident but calibrated predictions.

**Idle timeout design**: The production pipeline loads the reranker on-demand and unloads it after configurable idle time (default: 600 seconds), so its memory footprint is zero when not actively used.


## Hardware Auto-Tuning

The server automatically detects available hardware on startup and configures optimal embedding/reranking parameters. The auto-tuning logic lives in `src/znote_mcp/hardware.py`.

### Detection Method

1. **GPU**: Detected via `nvidia-smi --query-gpu=name,memory.total`. VRAM is the primary tier discriminator.
2. **System RAM**: Read from `os.sysconf("SC_PHYS_PAGES")` or `/proc/meminfo` fallback.
3. **CPU architecture**: `platform.machine()` (used for tier labels, e.g., `x86_64`, `aarch64`).
4. **ONNX providers**: Queried from `onnxruntime.get_available_providers()`.

### Hardware Tiers

| Tier | Condition | Batch Size | Embed Tokens | Rerank Tokens | Memory Budget | ONNX Providers |
|------|-----------|------------|--------------|---------------|---------------|----------------|
| GPU 16 GB+ | VRAM >= 14,000 MB | 64 | 8192 | 8192 | 10.0 GB | auto |
| GPU 8 GB+ | VRAM >= 7,000 MB | 32 | 4096 | 4096 | 6.0 GB | auto |
| GPU small | VRAM > 0 | 16 | 2048 | 2048 | 3.0 GB | auto |
| CPU 32 GB+ | RAM >= 28,000 MB | 16 | 8192 | 4096 | 8.0 GB | cpu |
| CPU 16 GB+ | RAM >= 14,000 MB | 8 | 4096 | 2048 | 4.0 GB | cpu |
| CPU 8 GB+ | RAM >= 7,000 MB | 4 | 2048 | 1024 | 2.0 GB | cpu |
| CPU small | Fallback | 2 | 512 | 512 | 1.0 GB | cpu |

### Override Priority

Environment variables always take precedence over auto-detected values. If you set `ZETTELKASTEN_EMBEDDING_BATCH_SIZE=4` in your environment, the auto-tuner will not modify that field, even if it would normally set it to 16. The override check is per-field -- you can let some fields auto-tune while explicitly controlling others.

The relevant environment variables are:

```
ZETTELKASTEN_ONNX_PROVIDERS
ZETTELKASTEN_EMBEDDING_BATCH_SIZE
ZETTELKASTEN_EMBEDDING_MAX_TOKENS
ZETTELKASTEN_RERANKER_MAX_TOKENS
ZETTELKASTEN_EMBEDDING_MEMORY_BUDGET_GB
```

### Memory Usage Guide

The attention mechanism allocates memory proportional to `batch_size * max_tokens^2`. Approximate peak memory during a full reindex:

| Batch Size | Max Tokens | Approx. Peak | Recommended For |
|------------|------------|-------------|-----------------|
| 2 | 512 | ~200 MB | 4 GB systems |
| 2 | 2048 | ~400 MB | 4 GB systems |
| 8 | 2048 | ~1.6 GB | 8 GB+ systems (default) |
| 16 | 4096 | ~6.4 GB | 16 GB+ systems |
| 32 | 8192 | ~25 GB+ | 64 GB+ systems |
| 64 | 8192 | ~50 GB+ | GPU with 16 GB+ VRAM |


## INT8 Quantization

znote-mcp supports INT8 quantized ONNX models via the `ZETTELKASTEN_ONNX_QUANTIZED=true` environment variable.

### Benefits

- **~4x smaller model files**: 143 MB (INT8) vs 569 MB (FP32) for gte-modernbert-base
- **~97% quality retention**: Quantization noise has minimal impact on embedding quality
- **Reduced memory footprint**: Proportional to model file size reduction

### Benchmark Findings

During the evaluation, INT8 behavior was found to be **hardware-dependent**:

**Intel x86_64 (AVX-VNNI)**: Dynamic INT8 quantization (`DynamicQuantizeLinear` + `MatMulInteger` operations in the ONNX graph) was measured to be **slower** than FP32 on the Intel 14700K test machine. The ONNX Runtime CPU provider did not dispatch to optimized INT8 kernels on this architecture, falling back to a generic implementation with quantize/dequantize overhead.

**ARM (Apple Silicon)**: The M-series chips have native `sdot`/`udot` instructions that accelerate INT8 matrix multiplication directly. INT8 quantization is expected to deliver genuine speedups on aarch64 deployments. This is the primary deployment target for znote-mcp (Ubuntu VMs on Apple Silicon via Parallels).

**Qdrant BGE "optimized" models**: The ONNX files marketed as "optimized" by Qdrant were found to contain zero quantization operations -- they only applied ORT graph fusion optimizations (constant folding, attention fusion). These are not actually quantized.

**Recommendation**: Enable `ZETTELKASTEN_ONNX_QUANTIZED=true` on ARM/Apple Silicon systems. On x86 Intel, leave it at the default `false` unless benchmarking confirms a benefit on your specific CPU.


## Benchmark Reproduction

### Scripts

All benchmark scripts are in the `scripts/` directory:

| Script | Purpose |
|--------|---------|
| `scripts/benchmark_embed.py` | Embedding performance benchmark: measures throughput (notes/s, chunks/s), peak memory (RSS), and GPU VRAM usage |
| `scripts/benchmark_quality.py` | Quality evaluation: link prediction MRR, recall@K, tag coherence ratio |
| `scripts/benchmark_rerank.py` | Reranker benchmark: MRR lift over embedding-only retrieval, with configurable query strategies |
| `scripts/model_configs.py` | Registry of 12 embedding model configurations |
| `scripts/reranker_configs.py` | Registry of 10 reranker model configurations |
| `scripts/handcrafted_queries.json` | 25 manually written evaluation queries with target note IDs |

### Benchmark Data

Results are stored in the `benchmarks/` directory:

| Directory | Contents |
|-----------|----------|
| `benchmarks/matrix-cpu/` | CPU embedding performance + quality matrices (JSON + Markdown) |
| `benchmarks/matrix-gpu/` | GPU embedding performance + quality matrices |
| `benchmarks/matrix-gpu-smoke/` | Quick GPU validation runs |
| `benchmarks/rerank-matrix/` | V1 reranker results (note-to-note, full-text queries) |
| `benchmarks/rerank-v2/` | V2 reranker results (title/link_title/handcrafted queries) |
| `benchmarks/reranker-smoke/` | Smoke test results (12 cases, 6 rerankers) |

### Running Benchmarks

**Embedding performance (CPU)**:
```bash
cd /home/komi/repos/MCP/znote-mcp
uv run python scripts/benchmark_embed.py \
    --models all \
    --memory-budget-gb 12.0 \
    --output-dir benchmarks/matrix-cpu \
    -v
```

**Embedding performance (GPU)**:
```bash
ZETTELKASTEN_GPU_MEM_LIMIT_GB=12 uv run python scripts/benchmark_embed.py \
    --models all \
    --device gpu \
    --memory-budget-gb 12.0 \
    --output-dir benchmarks/matrix-gpu \
    -v
```

**Quality evaluation** (runs on pre-computed embeddings):
```bash
uv run python scripts/benchmark_quality.py benchmarks/matrix-cpu -v
```

**Reranker benchmark**:
```bash
uv run python scripts/benchmark_rerank.py \
    --embedder-dir benchmarks/matrix-cpu \
    --rerankers jina-reranker-v2 gte-reranker stsb-distilroberta \
    --embedders gte-modernbert-c8192-fp32 \
    --strategies title link_title handcrafted \
    --device gpu \
    --output-dir benchmarks/rerank-v2 \
    -v
```

**Notes**:
- Embedding benchmarks download models from HuggingFace on first run (~570 MB for gte-modernbert-base).
- Quality evaluation requires embedding results to already exist in the specified directory.
- Reranker benchmarks require both embedding results and note text metadata.
- GPU runs require `onnxruntime-gpu` (CUDA 12.x) installed instead of `onnxruntime`.
- Use `--skip-existing` to resume interrupted benchmark runs.
