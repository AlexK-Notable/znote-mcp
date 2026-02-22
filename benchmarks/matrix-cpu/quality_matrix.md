# Quality Matrix — CPU

Generated: 2026-02-21T00:09:28Z

## Corpus

| Property | Value |
|----------|-------|
| Notes | 961 |
| Link pairs | 1365 |
| Unique tags | 1183 |
| Configs evaluated | 12 |

## Link Prediction (sorted by MRR)

How well does cosine similarity between note embeddings predict actual note links?

| Rank | Config | MRR | R@5 | R@10 | R@20 | Med Rank | Mean Rank | P95 Rank | Pairs |
|------|--------|-----|-----|------|------|----------|-----------|----------|-------|
| 1 | gte-modernbert-c8192-fp32 | 0.2718 | 0.4413 | 0.5909 | 0.7236 | 7 | 35.9 | 192 | 1364 |
| 2 | arctic-m-c2048-fp32 | 0.2688 | 0.4208 | 0.5689 | 0.6950 | 8 | 39.2 | 205 | 1364 |
| 3 | arctic-l-c2048-fp32 | 0.2649 | 0.4076 | 0.5550 | 0.6950 | 8 | 40.7 | 232 | 1364 |
| 4 | gte-modernbert-c2048-fp32 | 0.2636 | 0.4216 | 0.5638 | 0.6906 | 8 | 43.4 | 248 | 1364 |
| 5 | arctic-l-c8192-fp32 | 0.2597 | 0.3915 | 0.5491 | 0.6840 | 9 | 43.9 | 250 | 1364 |
| 6 | arctic-m-c8192-fp32 | 0.2572 | 0.4054 | 0.5425 | 0.6811 | 9 | 41.6 | 241 | 1364 |
| 7 | embeddinggemma-fp32 | 0.2376 | 0.3651 | 0.4941 | 0.6430 | 11 | 51.5 | 271 | 1364 |
| 8 | mxbai-large-fp32 | 0.2318 | 0.3563 | 0.4919 | 0.6239 | 11 | 58.9 | 346 | 1364 |
| 9 | bge-small-fp32 | 0.2263 | 0.3416 | 0.4641 | 0.5894 | 12 | 71.0 | 411 | 1364 |
| 10 | bge-base-fp32 | 0.2252 | 0.3490 | 0.4743 | 0.6034 | 12 | 70.3 | 401 | 1364 |
| 11 | minilm-fp32 | 0.2235 | 0.3365 | 0.4611 | 0.5792 | 12 | 82.6 | 468 | 1364 |
| 12 | nomic-v1.5-fp32 | 0.1588 | 0.2427 | 0.3475 | 0.4384 | 30 | 129.1 | 582 | 1364 |

## Tag Coherence (sorted by ratio)

Ratio of intra-tag similarity to inter-tag similarity. Higher = notes with shared tags cluster more tightly.

| Rank | Config | Tag Ratio | Tags Evaluated |
|------|--------|-----------|----------------|
| 1 | arctic-m-c8192-fp32 | 1.9436 | 330 |
| 2 | arctic-m-c2048-fp32 | 1.9293 | 330 |
| 3 | arctic-l-c2048-fp32 | 1.6028 | 330 |
| 4 | arctic-l-c8192-fp32 | 1.5664 | 330 |
| 5 | embeddinggemma-fp32 | 1.4416 | 330 |
| 6 | gte-modernbert-c8192-fp32 | 1.2841 | 330 |
| 7 | gte-modernbert-c2048-fp32 | 1.2677 | 330 |
| 8 | mxbai-large-fp32 | 1.2194 | 330 |
| 9 | nomic-v1.5-fp32 | 1.2189 | 330 |
| 10 | bge-base-fp32 | 1.1524 | 330 |
| 11 | bge-small-fp32 | 1.1345 | 330 |
| 12 | minilm-fp32 | 1.0581 | 330 |

## Tag Detail Sample (gte-modernbert-c8192-fp32)

Top 15 and bottom 5 tags by coherence ratio:

| Tag | Notes | Intra Sim | Inter Sim | Ratio |
|-----|-------|-----------|-----------|-------|
| microswiss | 3 | 0.9340 | 0.4910 | 1.9021 |
| flowtech | 3 | 0.9340 | 0.4910 | 1.9021 |
| 0.8mm | 3 | 0.9340 | 0.4910 | 1.9021 |
| nozzle | 4 | 0.9304 | 0.4977 | 1.8695 |
| slicer-settings | 4 | 0.9304 | 0.4977 | 1.8695 |
| orphan-test | 4 | 0.8820 | 0.4814 | 1.8323 |
| central-test | 6 | 0.8023 | 0.4640 | 1.7292 |
| test | 4 | 0.8309 | 0.4874 | 1.7050 |
| wiring | 3 | 0.7601 | 0.4526 | 1.6794 |
| wallhaven-table | 3 | 0.9514 | 0.5676 | 1.6760 |
| ble | 3 | 0.9388 | 0.5630 | 1.6674 |
| tmc5160 | 4 | 0.8211 | 0.4942 | 1.6614 |
| hyprtasking-debug | 13 | 0.8641 | 0.5209 | 1.6587 |
| faillock | 3 | 0.9231 | 0.5578 | 1.6549 |
| sudo | 3 | 0.9098 | 0.5513 | 1.6503 |
| ... | | | | |
| threading | 3 | 0.5752 | 0.5887 | 0.9772 |
| bug-fix | 6 | 0.5504 | 0.5636 | 0.9766 |
| best-practices | 3 | 0.5708 | 0.5875 | 0.9715 |
| workflow | 3 | 0.5025 | 0.5293 | 0.9494 |
| index | 4 | 0.5191 | 0.5582 | 0.9300 |

## Combined Ranking

| Config | MRR | R@10 | Med Rank | Tag Ratio |
|--------|-----|------|----------|-----------|
| gte-modernbert-c8192-fp32 | 0.2718 | 0.5909 | 7 | 1.2841 |
| arctic-m-c2048-fp32 | 0.2688 | 0.5689 | 8 | 1.9293 |
| arctic-l-c2048-fp32 | 0.2649 | 0.5550 | 8 | 1.6028 |
| gte-modernbert-c2048-fp32 | 0.2636 | 0.5638 | 8 | 1.2677 |
| arctic-l-c8192-fp32 | 0.2597 | 0.5491 | 9 | 1.5664 |
| arctic-m-c8192-fp32 | 0.2572 | 0.5425 | 9 | 1.9436 |
| embeddinggemma-fp32 | 0.2376 | 0.4941 | 11 | 1.4416 |
| mxbai-large-fp32 | 0.2318 | 0.4919 | 11 | 1.2194 |
| bge-small-fp32 | 0.2263 | 0.4641 | 12 | 1.1345 |
| bge-base-fp32 | 0.2252 | 0.4743 | 12 | 1.1524 |
| minilm-fp32 | 0.2235 | 0.4611 | 12 | 1.0581 |
| nomic-v1.5-fp32 | 0.1588 | 0.3475 | 30 | 1.2189 |
