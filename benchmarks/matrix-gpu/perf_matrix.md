# Performance Matrix — GPU

Generated: 2026-02-21T00:09:55Z

## System

| Property | Value |
|----------|-------|
| Platform | Linux-6.19.3-2-cachyos-x86_64-with-glibc2.43 |
| Python | 3.11.14 |
| ONNX Runtime | 1.25.0.dev20260219001 |
| CPU cores | 28 |
| RAM | 31.1 GB |
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER |
| VRAM | 16376 MB |
| Driver | 590.48.01 |
| Compute capability | 8.9 |
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

| Config | Model | Dim | Chunk | Load (s) | Embed (s) | n/s | c/s | Chunks | Peak RSS (MB) | GPU Peak (MB) | GPU Model (MB) |
|--------|-------|-----|-------|----------|-----------|-----|-----|--------|---------------|---------------|----------------|
| minilm-fp32 | all-MiniLM-L6-v2 | 384 | 512 | 0.6 | 11.8 | 81.7 | 689.3 | 8108 | 1648 | 5201.0 | 120.0 |
| bge-small-fp32 | bge-small-en-v1.5 | 384 | 512 | 0.6 | 19.9 | 48.2 | 406.7 | 8108 | 1622 | 4595.0 | 184.0 |
| bge-base-fp32 | bge-base-en-v1.5 | 768 | 512 | 0.8 | 35.7 | 26.9 | 227.2 | 8108 | 1672 | 3673.0 | 534.0 |
| embeddinggemma-fp32 | embeddinggemma-300m-ONNX | 768 | 2048 | 1.1 | 55.2 | 17.4 | 35.4 | 1951 | 1537 | 3780.0 | 1328.0 |
| nomic-v1.5-fp32 | nomic-embed-text-v1.5 | 768 | 2048 | 1.0 | 62.2 | 15.4 | 31.4 | 1951 | 1608 | 5377.0 | 604.0 |
| arctic-m-c2048-fp32 | snowflake-arctic-embed-m-v2.0 | 768 | 2048 | 1.9 | 83.9 | 11.5 | 23.3 | 1951 | 2674 | 9337.0 | 1228.0 |
| mxbai-large-fp32 | mxbai-embed-large-v1 | 1024 | 512 | 1.9 | 100.4 | 9.6 | 80.7 | 8108 | 2398 | 5239.0 | 1284.0 |
| gte-modernbert-c2048-fp32 | gte-modernbert-base | 768 | 2048 | 0.9 | 109.6 | 8.8 | 17.8 | 1951 | 1607 | 7479.0 | 686.0 |
| arctic-m-c8192-fp32 | snowflake-arctic-embed-m-v2.0 | 768 | 8192 | 1.9 | 132.8 | 7.2 | 7.6 | 1008 | 2674 | 10423.0 | 1228.0 |
| arctic-l-c2048-fp32 | snowflake-arctic-embed-l-v2.0 | 1024 | 2048 | 1.1 | 210.1 | 4.6 | 9.3 | 1951 | 1946 | 11569.0 | 2174.0 |
| gte-modernbert-c8192-fp32 | gte-modernbert-base | 768 | 8192 | 0.8 | 212.6 | 4.5 | 4.7 | 1008 | 1707 | 11239.0 | 686.0 |
| arctic-l-c8192-fp32 | snowflake-arctic-embed-l-v2.0 | 1024 | 8192 | 1.1 | 361.7 | 2.7 | 2.8 | 1008 | 1995 | 13439.0 | 2174.0 |

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
