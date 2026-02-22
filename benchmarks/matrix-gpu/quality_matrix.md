# Quality Matrix — GPU

Generated: 2026-02-21T00:09:36Z

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
| 1 | gte-modernbert-c8192-fp32 | 0.2719 | 0.4413 | 0.5909 | 0.7229 | 7 | 35.9 | 192 | 1364 |
| 2 | arctic-m-c2048-fp32 | 0.2687 | 0.4208 | 0.5689 | 0.6950 | 8 | 39.2 | 205 | 1364 |
| 3 | arctic-l-c2048-fp32 | 0.2649 | 0.4076 | 0.5550 | 0.6950 | 8 | 40.7 | 232 | 1364 |
| 4 | gte-modernbert-c2048-fp32 | 0.2639 | 0.4216 | 0.5638 | 0.6906 | 8 | 43.4 | 248 | 1364 |
| 5 | arctic-l-c8192-fp32 | 0.2594 | 0.3915 | 0.5477 | 0.6840 | 9 | 44.0 | 251 | 1364 |
| 6 | arctic-m-c8192-fp32 | 0.2572 | 0.4054 | 0.5425 | 0.6811 | 9 | 41.6 | 241 | 1364 |
| 7 | embeddinggemma-fp32 | 0.2376 | 0.3651 | 0.4941 | 0.6430 | 11 | 51.5 | 271 | 1364 |
| 8 | mxbai-large-fp32 | 0.2318 | 0.3563 | 0.4919 | 0.6239 | 11 | 58.9 | 346 | 1364 |
| 9 | bge-small-fp32 | 0.2263 | 0.3416 | 0.4641 | 0.5894 | 12 | 71.0 | 411 | 1364 |
| 10 | bge-base-fp32 | 0.2252 | 0.3497 | 0.4743 | 0.6034 | 12 | 70.3 | 401 | 1364 |
| 11 | minilm-fp32 | 0.2235 | 0.3365 | 0.4611 | 0.5792 | 12 | 82.6 | 468 | 1364 |
| 12 | nomic-v1.5-fp32 | 0.1587 | 0.2427 | 0.3475 | 0.4391 | 30 | 129.1 | 583 | 1364 |

## Tag Coherence (sorted by ratio)

Ratio of intra-tag similarity to inter-tag similarity. Higher = notes with shared tags cluster more tightly.

| Rank | Config | Tag Ratio | Tags Evaluated |
|------|--------|-----------|----------------|
| 1 | arctic-m-c8192-fp32 | 1.9436 | 330 |
| 2 | arctic-m-c2048-fp32 | 1.9293 | 330 |
| 3 | arctic-l-c2048-fp32 | 1.6028 | 330 |
| 4 | arctic-l-c8192-fp32 | 1.5663 | 330 |
| 5 | embeddinggemma-fp32 | 1.4416 | 330 |
| 6 | gte-modernbert-c8192-fp32 | 1.2841 | 330 |
| 7 | gte-modernbert-c2048-fp32 | 1.2677 | 330 |
| 8 | mxbai-large-fp32 | 1.2194 | 330 |
| 9 | nomic-v1.5-fp32 | 1.2190 | 330 |
| 10 | bge-base-fp32 | 1.1524 | 330 |
| 11 | bge-small-fp32 | 1.1345 | 330 |
| 12 | minilm-fp32 | 1.0581 | 330 |

## Tag Detail Sample (gte-modernbert-c8192-fp32)

Top 15 and bottom 5 tags by coherence ratio:

| Tag | Notes | Intra Sim | Inter Sim | Ratio |
|-----|-------|-----------|-----------|-------|
| microswiss | 3 | 0.9340 | 0.4911 | 1.9020 |
| flowtech | 3 | 0.9340 | 0.4911 | 1.9020 |
| 0.8mm | 3 | 0.9340 | 0.4911 | 1.9020 |
| nozzle | 4 | 0.9304 | 0.4977 | 1.8694 |
| slicer-settings | 4 | 0.9304 | 0.4977 | 1.8694 |
| orphan-test | 4 | 0.8821 | 0.4814 | 1.8323 |
| central-test | 6 | 0.8024 | 0.4640 | 1.7292 |
| test | 4 | 0.8310 | 0.4874 | 1.7050 |
| wiring | 3 | 0.7601 | 0.4526 | 1.6794 |
| wallhaven-table | 3 | 0.9514 | 0.5677 | 1.6759 |
| ble | 3 | 0.9388 | 0.5630 | 1.6674 |
| tmc5160 | 4 | 0.8212 | 0.4942 | 1.6614 |
| hyprtasking-debug | 13 | 0.8641 | 0.5210 | 1.6587 |
| faillock | 3 | 0.9231 | 0.5578 | 1.6548 |
| sudo | 3 | 0.9098 | 0.5513 | 1.6502 |
| ... | | | | |
| threading | 3 | 0.5752 | 0.5886 | 0.9771 |
| bug-fix | 6 | 0.5504 | 0.5636 | 0.9767 |
| best-practices | 3 | 0.5707 | 0.5875 | 0.9715 |
| workflow | 3 | 0.5026 | 0.5293 | 0.9495 |
| index | 4 | 0.5192 | 0.5583 | 0.9300 |

## Combined Ranking

| Config | MRR | R@10 | Med Rank | Tag Ratio |
|--------|-----|------|----------|-----------|
| gte-modernbert-c8192-fp32 | 0.2719 | 0.5909 | 7 | 1.2841 |
| arctic-m-c2048-fp32 | 0.2687 | 0.5689 | 8 | 1.9293 |
| arctic-l-c2048-fp32 | 0.2649 | 0.5550 | 8 | 1.6028 |
| gte-modernbert-c2048-fp32 | 0.2639 | 0.5638 | 8 | 1.2677 |
| arctic-l-c8192-fp32 | 0.2594 | 0.5477 | 9 | 1.5663 |
| arctic-m-c8192-fp32 | 0.2572 | 0.5425 | 9 | 1.9436 |
| embeddinggemma-fp32 | 0.2376 | 0.4941 | 11 | 1.4416 |
| mxbai-large-fp32 | 0.2318 | 0.4919 | 11 | 1.2194 |
| bge-small-fp32 | 0.2263 | 0.4641 | 12 | 1.1345 |
| bge-base-fp32 | 0.2252 | 0.4743 | 12 | 1.1524 |
| minilm-fp32 | 0.2235 | 0.4611 | 12 | 1.0581 |
| nomic-v1.5-fp32 | 0.1587 | 0.3475 | 30 | 1.2190 |
