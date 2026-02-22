#!/usr/bin/env python
"""Quality evaluator for embedding benchmark results.

Loads saved embeddings and note metadata, computes:
1. Link prediction (MRR, Recall@K) — cross-model comparable
2. Tag coherence (intra/inter-tag similarity ratio) — cross-model comparable
3. Quantization fidelity (INT8 vs own FP32) — cosine, Spearman, rank agreement

No model inference needed — pure numpy computation.

Usage:
    cd /home/komi/repos/MCP/znote-mcp
    uv run python scripts/benchmark_quality.py
    uv run python scripts/benchmark_quality.py --matrix-dir benchmarks/matrix
    uv run python scripts/benchmark_quality.py --configs gte-modernbert-c2048-fp32,bge-small-fp32
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

# Ensure project root is on sys.path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from model_configs import MODELS

logger = logging.getLogger("benchmark_quality")


# ---------------------------------------------------------------------------
# Spearman rank correlation (no scipy dependency)
# ---------------------------------------------------------------------------

def _rankdata(arr: np.ndarray) -> np.ndarray:
    """Assign ranks to data, handling ties by averaging."""
    sorter = np.argsort(arr)
    ranks = np.empty_like(sorter, dtype=float)
    ranks[sorter] = np.arange(1, len(arr) + 1, dtype=float)

    # Average ranks for tied values
    unique_vals, inverse, counts = np.unique(arr, return_inverse=True, return_counts=True)
    for i, count in enumerate(counts):
        if count > 1:
            mask = inverse == i
            ranks[mask] = ranks[mask].mean()
    return ranks


def spearman_rho(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Spearman rank correlation coefficient."""
    if len(a) < 2:
        return 0.0
    ra = _rankdata(a)
    rb = _rankdata(b)
    # Pearson on ranks
    ra_centered = ra - ra.mean()
    rb_centered = rb - rb.mean()
    num = (ra_centered * rb_centered).sum()
    denom = np.sqrt((ra_centered**2).sum() * (rb_centered**2).sum())
    if denom < 1e-12:
        return 0.0
    return float(num / denom)


# ---------------------------------------------------------------------------
# Metric computations
# ---------------------------------------------------------------------------

def compute_link_prediction(
    embeddings: np.ndarray,
    note_ids: List[str],
    link_pairs: List[Tuple[str, str]],
) -> Dict[str, float]:
    """Compute link prediction metrics: MRR and Recall@K.

    For each linked pair (source, target), compute source's cosine
    similarity to ALL notes, find target's rank.
    """
    if not link_pairs:
        return {"mrr": 0.0, "recall_at_5": 0.0, "recall_at_10": 0.0, "recall_at_20": 0.0, "num_pairs": 0}

    id_to_idx = {nid: i for i, nid in enumerate(note_ids)}

    # Filter to pairs where both exist
    valid_pairs = [
        (s, t) for s, t in link_pairs
        if s in id_to_idx and t in id_to_idx
    ]
    if not valid_pairs:
        return {"mrr": 0.0, "recall_at_5": 0.0, "recall_at_10": 0.0, "recall_at_20": 0.0, "num_pairs": 0}

    # Pre-compute all pairwise similarities (embeddings are L2-normalized)
    sim_matrix = embeddings @ embeddings.T  # (N, N) cosine similarities

    reciprocal_ranks = []
    recall_5 = 0
    recall_10 = 0
    recall_20 = 0
    ranks: List[int] = []

    for source_id, target_id in valid_pairs:
        src_idx = id_to_idx[source_id]
        tgt_idx = id_to_idx[target_id]

        # Get similarities from source to all notes
        sims = sim_matrix[src_idx].copy()
        sims[src_idx] = -np.inf  # Exclude self-similarity

        # Rank: how high is the target?
        # Sort descending, find target's position
        sorted_indices = np.argsort(sims)[::-1]
        rank = int(np.where(sorted_indices == tgt_idx)[0][0]) + 1  # 1-indexed

        reciprocal_ranks.append(1.0 / rank)
        ranks.append(rank)
        if rank <= 5:
            recall_5 += 1
        if rank <= 10:
            recall_10 += 1
        if rank <= 20:
            recall_20 += 1

    n = len(valid_pairs)
    ranks_arr = np.array(ranks)
    return {
        "mrr": round(float(np.mean(reciprocal_ranks)), 4),
        "recall_at_5": round(recall_5 / n, 4),
        "recall_at_10": round(recall_10 / n, 4),
        "recall_at_20": round(recall_20 / n, 4),
        "num_pairs": n,
        "median_rank": int(np.median(ranks_arr)),
        "mean_rank": round(float(np.mean(ranks_arr)), 1),
        "p95_rank": int(np.percentile(ranks_arr, 95)),
    }


def compute_tag_coherence(
    embeddings: np.ndarray,
    note_ids: List[str],
    note_tags: Dict[str, List[str]],
    min_tag_notes: int = 3,
) -> Dict[str, Any]:
    """Compute tag coherence: intra-tag vs inter-tag similarity ratio.

    For each tag with >=min_tag_notes notes, compute:
    - Mean cosine similarity between notes sharing that tag
    - Mean cosine similarity between notes NOT sharing that tag

    Higher ratio = better tag-aligned embeddings.
    """
    id_to_idx = {nid: i for i, nid in enumerate(note_ids)}

    # Build tag → note indices
    tag_groups: Dict[str, List[int]] = defaultdict(list)
    for note_id, tags in note_tags.items():
        if note_id in id_to_idx:
            idx = id_to_idx[note_id]
            for tag in tags:
                tag_groups[tag].append(idx)

    # Filter to tags with enough notes
    qualifying_tags = {
        tag: indices for tag, indices in tag_groups.items()
        if len(indices) >= min_tag_notes
    }

    if not qualifying_tags:
        return {
            "mean_ratio": 0.0,
            "num_tags_evaluated": 0,
            "tag_details": {},
        }

    # Pre-compute similarity matrix
    sim_matrix = embeddings @ embeddings.T
    np.fill_diagonal(sim_matrix, 0.0)  # Exclude self-similarity
    N = len(note_ids)

    tag_details = {}
    ratios = []

    for tag, indices in qualifying_tags.items():
        indices_set = set(indices)
        other_indices = [i for i in range(N) if i not in indices_set]

        if len(indices) < 2 or not other_indices:
            continue

        # Intra-tag similarity: mean sim between all pairs within this tag
        intra_sims = []
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                intra_sims.append(sim_matrix[indices[i], indices[j]])
        mean_intra = float(np.mean(intra_sims))

        # Inter-tag similarity: mean sim between tag notes and non-tag notes
        inter_sims = []
        for idx in indices:
            for other_idx in other_indices:
                inter_sims.append(sim_matrix[idx, other_idx])
        mean_inter = float(np.mean(inter_sims))

        ratio = mean_intra / max(mean_inter, 1e-8)
        ratios.append(ratio)

        tag_details[tag] = {
            "num_notes": len(indices),
            "intra_sim": round(mean_intra, 4),
            "inter_sim": round(mean_inter, 4),
            "ratio": round(ratio, 4),
        }

    return {
        "mean_ratio": round(float(np.mean(ratios)), 4) if ratios else 0.0,
        "num_tags_evaluated": len(tag_details),
        "tag_details": tag_details,
    }


def compute_quantization_fidelity(
    fp32_embeddings: np.ndarray,
    int8_embeddings: np.ndarray,
    top_k: int = 5,
) -> Dict[str, float]:
    """Compare INT8 embeddings against their FP32 counterparts.

    Metrics:
    - Mean cosine similarity of corresponding vectors
    - Spearman rank correlation of pairwise similarity matrices
    - Top-K rank agreement at K=5 and K=10
    """
    if fp32_embeddings.shape != int8_embeddings.shape:
        return {"error": "shape_mismatch", "fp32": list(fp32_embeddings.shape), "int8": list(int8_embeddings.shape)}

    N = len(fp32_embeddings)
    if N == 0:
        return {"mean_cosine": 0.0, "spearman_rho": 0.0, "topk_agreement_5": 0.0, "topk_agreement_10": 0.0}

    # 1. Mean cosine similarity of corresponding vectors
    # Both are L2-normalized, so dot product = cosine
    per_vector_cosine = np.sum(fp32_embeddings * int8_embeddings, axis=1)
    mean_cosine = float(np.mean(per_vector_cosine))

    # 2. Spearman rank correlation of flattened pairwise similarity matrices
    # For large N, sample to keep computation tractable
    if N > 500:
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(N, size=500, replace=False)
        fp32_sub = fp32_embeddings[sample_idx]
        int8_sub = int8_embeddings[sample_idx]
    else:
        fp32_sub = fp32_embeddings
        int8_sub = int8_embeddings

    fp32_sims = fp32_sub @ fp32_sub.T
    int8_sims = int8_sub @ int8_sub.T

    # Flatten upper triangle (exclude diagonal)
    triu_idx = np.triu_indices(len(fp32_sub), k=1)
    fp32_flat = fp32_sims[triu_idx]
    int8_flat = int8_sims[triu_idx]

    rho = spearman_rho(fp32_flat, int8_flat)

    # 3. Top-K rank agreement
    def _topk_agreement(sim_a: np.ndarray, sim_b: np.ndarray, k: int) -> float:
        """Fraction of top-K neighbors that agree between two similarity matrices."""
        agreements = 0
        n = len(sim_a)
        for i in range(n):
            a_neighbors = set(np.argsort(sim_a[i])[::-1][1:k+1])  # exclude self
            b_neighbors = set(np.argsort(sim_b[i])[::-1][1:k+1])
            agreements += len(a_neighbors & b_neighbors) / k
        return agreements / n

    # Use subsampled matrices for top-K agreement too
    topk_5 = _topk_agreement(fp32_sims, int8_sims, 5)
    topk_10 = _topk_agreement(fp32_sims, int8_sims, 10)

    return {
        "mean_cosine": round(mean_cosine, 6),
        "spearman_rho": round(rho, 6),
        "topk_agreement_5": round(topk_5, 4),
        "topk_agreement_10": round(topk_10, 4),
        "num_vectors": N,
        "min_cosine": round(float(np.min(per_vector_cosine)), 6),
        "max_cosine": round(float(np.max(per_vector_cosine)), 6),
    }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_embeddings(config_dir: Path) -> Optional[np.ndarray]:
    """Load note-level embeddings from an npz file."""
    path = config_dir / "embeddings.npz"
    if not path.exists():
        return None
    data = np.load(path)
    return data["embeddings"]


def load_note_ids(config_dir: Path) -> Optional[List[str]]:
    """Load note ID ordering."""
    path = config_dir / "note_ids.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return data["note_ids"]


def load_notes_metadata(matrix_dir: Path) -> Optional[Dict[str, Any]]:
    """Load notes metadata saved by benchmark_embed."""
    path = matrix_dir / "notes_metadata.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _save_json(path: Path, data: Any) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute quality metrics for embedding benchmark results",
    )
    parser.add_argument(
        "--matrix-dir",
        default="benchmarks/matrix",
        help="Directory with benchmark results (default: benchmarks/matrix)",
    )
    parser.add_argument(
        "--configs",
        default=None,
        help="Comma-separated config keys to evaluate (default: all found)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    matrix_dir = Path(args.matrix_dir)
    if not matrix_dir.exists():
        logger.error(f"Matrix directory not found: {matrix_dir}")
        logger.info("Run benchmark_embed.py first to generate embeddings.")
        sys.exit(1)

    # Load notes metadata
    metadata = load_notes_metadata(matrix_dir)
    if not metadata:
        logger.error("notes_metadata.json not found in matrix directory")
        sys.exit(1)

    # Build lookup structures
    notes_data = metadata["notes"]
    note_tags: Dict[str, List[str]] = {n["id"]: n["tags"] for n in notes_data}

    # Extract all link pairs (directed)
    all_link_pairs: List[Tuple[str, str]] = []
    for n in notes_data:
        for link in n["links"]:
            all_link_pairs.append((link["source_id"], link["target_id"]))

    logger.info(
        f"Loaded metadata: {len(notes_data)} notes, "
        f"{len(all_link_pairs)} link pairs, "
        f"{len({t for n in notes_data for t in n['tags']})} unique tags"
    )

    # Discover configs to evaluate
    if args.configs:
        config_keys = [k.strip() for k in args.configs.split(",")]
    else:
        config_keys = sorted([
            d.name for d in matrix_dir.iterdir()
            if d.is_dir() and (d / "embeddings.npz").exists()
        ])

    if not config_keys:
        logger.error("No embedding results found to evaluate")
        sys.exit(1)

    logger.info(f"Evaluating {len(config_keys)} configs: {', '.join(config_keys)}")

    # Evaluate each config
    all_quality: Dict[str, Dict[str, Any]] = {}

    for key in config_keys:
        logger.info(f"\n{'='*50}")
        logger.info(f"Evaluating: {key}")
        config_dir = matrix_dir / key

        embeddings = load_embeddings(config_dir)
        note_ids = load_note_ids(config_dir)

        if embeddings is None or note_ids is None:
            logger.warning(f"  Missing embeddings or note_ids for {key}, skipping")
            all_quality[key] = {"status": "missing_data"}
            continue

        quality: Dict[str, Any] = {"status": "ok", "config_key": key}

        # 1. Link prediction
        logger.info(f"  Computing link prediction ({len(all_link_pairs)} pairs)...")
        link_metrics = compute_link_prediction(embeddings, note_ids, all_link_pairs)
        quality["link_prediction"] = link_metrics
        logger.info(
            f"  Link prediction: MRR={link_metrics['mrr']:.4f}, "
            f"R@5={link_metrics['recall_at_5']:.4f}, "
            f"R@10={link_metrics['recall_at_10']:.4f}, "
            f"R@20={link_metrics['recall_at_20']:.4f}, "
            f"medRank={link_metrics.get('median_rank', '?')} "
            f"({link_metrics['num_pairs']} pairs)"
        )

        # 2. Tag coherence
        logger.info(f"  Computing tag coherence...")
        tag_metrics = compute_tag_coherence(embeddings, note_ids, note_tags)
        quality["tag_coherence"] = {
            k: v for k, v in tag_metrics.items() if k != "tag_details"
        }
        quality["tag_coherence_details"] = tag_metrics.get("tag_details", {})
        logger.info(
            f"  Tag coherence: ratio={tag_metrics['mean_ratio']:.4f} "
            f"({tag_metrics['num_tags_evaluated']} tags)"
        )

        # 3. Quantization fidelity (for INT8 vs FP32 comparisons)
        # Skipped — current matrix is FP32-only.  The function
        # compute_quantization_fidelity() is retained for future use.

        # Save per-config results
        _save_json(config_dir / "quality.json", quality)
        all_quality[key] = quality

    # Save combined quality matrix
    _save_json(matrix_dir / "quality_matrix.json", {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "matrix_dir": str(matrix_dir),
        "num_notes": len(notes_data),
        "num_link_pairs": len(all_link_pairs),
        "num_unique_tags": len({t for n in notes_data for t in n["tags"]}),
        "num_configs_evaluated": len(config_keys),
        "results": all_quality,
    })

    # Print summary table
    logger.info("\n" + "=" * 70)
    logger.info("QUALITY SUMMARY")
    logger.info("=" * 70)
    header = (
        f"{'Config':<28} {'MRR':<8} {'R@5':<8} {'R@10':<8} {'R@20':<8} "
        f"{'MedRk':<7} {'TagRat':<8} {'#Tags':<6}"
    )
    logger.info(header)
    logger.info("-" * 90)

    for key in config_keys:
        q = all_quality.get(key, {})
        if q.get("status") != "ok":
            logger.info(f"{key:<28} {'SKIP'}")
            continue

        lp = q.get("link_prediction", {})
        tc = q.get("tag_coherence", {})

        mrr = f"{lp.get('mrr', 0):.4f}"
        r5 = f"{lp.get('recall_at_5', 0):.4f}"
        r10 = f"{lp.get('recall_at_10', 0):.4f}"
        r20 = f"{lp.get('recall_at_20', 0):.4f}"
        med_rank = f"{lp.get('median_rank', '?')}"
        ratio = f"{tc.get('mean_ratio', 0):.4f}"
        n_tags = f"{tc.get('num_tags_evaluated', 0)}"

        logger.info(
            f"{key:<28} {mrr:<8} {r5:<8} {r10:<8} {r20:<8} "
            f"{med_rank:<7} {ratio:<8} {n_tags:<6}"
        )

    logger.info(f"\nResults saved to {matrix_dir}/quality_matrix.json")


if __name__ == "__main__":
    main()
