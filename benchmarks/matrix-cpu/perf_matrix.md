# Performance Matrix — CPU

Generated: 2026-02-21T00:09:55Z

## System

| Property | Value |
|----------|-------|
| Platform | Linux-6.19.3-2-cachyos-x86_64-with-glibc2.43 |
| Python | 3.11.14 |
| ONNX Runtime | 1.25.0.dev20260219001 |
| CPU cores | 28 |
| RAM | 31.1 GB |
| Memory budget | 12.0 GB |

## Corpus

| Property | Value |
|----------|-------|
| Total notes | 961 |
| Link pairs | 1364 |
| Unique tags | 1183 |
| Total chars | 10,697,873 |
| Mean chars/note | 11,132 |
| Median chars/note | 7,891 |
| Range | 35 – 51,387 |
| P25 / P75 / P95 | 3,056 / 15,702 / 33,210 |

## Performance (sorted by speed)

| Config | Model | Dim | Chunk | Load (s) | Embed (s) | n/s | c/s | Chunks | Peak RSS (MB) | Model RSS (MB) |
|--------|-------|-----|-------|----------|-----------|-----|-----|--------|---------------|----------------|
| minilm-fp32 | all-MiniLM-L6-v2 | 384 | 512 | 0.3 | 169.5 | 5.7 | 47.8 | 8108 | 3082 | 140 |
| bge-small-fp32 | bge-small-en-v1.5 | 384 | 512 | 0.4 | 311.3 | 3.1 | 26.0 | 8108 | 3082 | 48 |
| bge-base-fp32 | bge-base-en-v1.5 | 768 | 512 | 0.6 | 724.2 | 1.3 | 11.2 | 8108 | 3753 | 425 |
| nomic-v1.5-fp32 | nomic-embed-text-v1.5 | 768 | 2048 | 0.7 | 993.3 | 1.0 | 2.0 | 1951 | 13019 | 199 |
| arctic-m-c2048-fp32 | snowflake-arctic-embed-m-v2.0 | 768 | 2048 | 1.7 | 1223.6 | 0.8 | 1.6 | 1951 | 14431 | 1224 |
| embeddinggemma-fp32 | embeddinggemma-300m-ONNX | 768 | 2048 | 1.2 | 1253.3 | 0.8 | 1.6 | 1951 | 7504 | 534 |
| gte-modernbert-c2048-fp32 | gte-modernbert-base | 768 | 2048 | 0.7 | 1432.1 | 0.7 | 1.4 | 1951 | 13019 | 670 |
| arctic-m-c8192-fp32 | snowflake-arctic-embed-m-v2.0 | 768 | 8192 | 1.6 | 1707.3 | 0.6 | 0.6 | 1008 | 14431 | 1016 |
| mxbai-large-fp32 | mxbai-embed-large-v1 | 1024 | 512 | 1.3 | 2395.1 | 0.4 | 3.4 | 8108 | 14431 | 192 |
| gte-modernbert-c8192-fp32 | gte-modernbert-base | 768 | 8192 | 0.7 | 2508.0 | 0.4 | 0.4 | 1008 | 13019 | 673 |
| arctic-l-c2048-fp32 | snowflake-arctic-embed-l-v2.0 | 1024 | 2048 | 1.3 | 3182.4 | 0.3 | 0.6 | 1951 | 14431 | 635 |
| arctic-l-c8192-fp32 | snowflake-arctic-embed-l-v2.0 | 1024 | 8192 | 1.1 | 4926.8 | 0.2 | 0.2 | 1008 | 14431 | 6 |

## Chunking Distribution

| Config | Chunk Size | Total Chunks | Min | Max | Avg |
|--------|-----------|--------------|-----|-----|-----|
| arctic-l-c2048-fp32 | 2048 | 1951 | 1 | 8 | 2.03 |
| arctic-l-c8192-fp32 | 8192 | 1008 | 1 | 2 | 1.05 |
| arctic-m-c2048-fp32 | 2048 | 1951 | 1 | 8 | 2.03 |
| arctic-m-c8192-fp32 | 8192 | 1008 | 1 | 2 | 1.05 |
| bge-base-fp32 | 512 | 8108 | 1 | 42 | 8.44 |
| bge-small-fp32 | 512 | 8108 | 1 | 42 | 8.44 |
| embeddinggemma-fp32 | 2048 | 1951 | 1 | 8 | 2.03 |
| gte-modernbert-c2048-fp32 | 2048 | 1951 | 1 | 8 | 2.03 |
| gte-modernbert-c8192-fp32 | 8192 | 1008 | 1 | 2 | 1.05 |
| minilm-fp32 | 512 | 8108 | 1 | 42 | 8.44 |
| mxbai-large-fp32 | 512 | 8108 | 1 | 42 | 8.44 |
| nomic-v1.5-fp32 | 2048 | 1951 | 1 | 8 | 2.03 |
