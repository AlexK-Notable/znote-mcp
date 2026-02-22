#!/usr/bin/env python
"""Reranker v2 benchmark: agent-query evaluation.

Tests the production retrieval pipeline: short query → embed → KNN top-K →
rerank(short_query, doc_texts) → ranked results.

Four query strategies simulate how agents actually search:
  - title:          Note title as query, target is the note itself
  - first_sentence: First prose sentence as query, target is the note itself
  - link_title:     Source note's title as query, target is linked note
  - handcrafted:    25 hand-written agent queries with known target notes

Two-phase execution:
  Phase 1 (embed-queries): Pre-compute query embeddings per (strategy, embedder)
  Phase 2 (evaluate):      Reranker matrix over pre-computed queries

Usage:
    cd /home/komi/repos/MCP/znote-mcp

    # Phase 1: Embed queries (once per embedder set)
    uv run python scripts/benchmark_rerank.py --phase embed-queries \\
        --strategies title,first_sentence,link_title,handcrafted \\
        --embedders gte-modernbert-c8192-fp32

    # Phase 2: Evaluate rerankers (default, re-runnable cheaply)
    uv run python scripts/benchmark_rerank.py \\
        --strategies title,link_title,handcrafted \\
        --embedders gte-modernbert-c8192-fp32 \\
        --rerankers gte-reranker

    # Full matrix
    uv run python scripts/benchmark_rerank.py \\
        --strategies title,first_sentence,link_title,handcrafted \\
        --embedders all --rerankers all
"""

from __future__ import annotations

import argparse
import datetime
import gc
import json
import logging
import os
import platform
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Ensure project root and scripts dir are on sys.path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from model_configs import MODELS, get_config as get_model_config, list_configs as list_model_configs
from reranker_configs import RERANKERS, get_config as get_reranker_config, list_configs as list_reranker_configs
from znote_mcp.services.onnx_providers import OnnxEmbeddingProvider, OnnxRerankerProvider, _get_rss_mb
from znote_mcp.storage.markdown_parser import MarkdownParser

logger = logging.getLogger("benchmark_rerank")

ALL_STRATEGIES = ["title", "first_sentence", "link_title", "handcrafted"]


# ---------------------------------------------------------------------------
# System info (reused pattern from benchmark_embed.py)
# ---------------------------------------------------------------------------

def _collect_system_info(device: str) -> Dict[str, Any]:
    """Collect system metadata for reproducibility."""
    import onnxruntime as ort

    info: Dict[str, Any] = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu": platform.processor() or "unknown",
        "onnxruntime_version": ort.__version__,
        "ort_available_providers": ort.get_available_providers(),
        "device": device,
    }

    try:
        info["cpu_count"] = os.cpu_count()
    except Exception:
        pass

    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    info["ram_total_gb"] = round(kb / 1024 / 1024, 1)
                    break
    except Exception:
        pass

    return info


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


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


def load_note_texts(notes_dir: Path) -> Dict[str, str]:
    """Load note texts from markdown files, returning {id: "title\\ncontent"}."""
    parser = MarkdownParser()
    texts: Dict[str, str] = {}
    for md_file in sorted(notes_dir.glob("*.md")):
        try:
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()
            note = parser.parse_note(content)
            texts[note.id] = f"{note.title}\n{note.content}"
        except Exception as e:
            logger.warning(f"Skipping {md_file.name}: {e}")
    return texts


def load_handcrafted_queries() -> List[Dict[str, str]]:
    """Load handcrafted queries from scripts/handcrafted_queries.json."""
    path = _SCRIPT_DIR / "handcrafted_queries.json"
    if not path.exists():
        logger.error(f"Handcrafted queries not found: {path}")
        return []
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Query generation
# ---------------------------------------------------------------------------

def extract_first_sentence(content: str) -> Optional[str]:
    """Extract the first prose sentence (>=5 words, ends with .?!) from content.

    Skips headings, metadata lines (key: value), code fences, list items,
    blank lines, and other non-prose content.
    """
    in_code_fence = False
    for line in content.split("\n"):
        stripped = line.strip()

        # Toggle code fence state
        if stripped.startswith("```"):
            in_code_fence = not in_code_fence
            continue
        if in_code_fence:
            continue

        # Skip blanks, headings, list items, metadata-style lines
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        if stripped.startswith(("-", "*", "+", ">")) and len(stripped) > 1 and stripped[1] == " ":
            continue
        if re.match(r"^[A-Za-z_]+\s*:", stripped):
            continue
        # Skip lines that are just links or very short
        if stripped.startswith("[") and stripped.endswith(")"):
            continue

        # Try to extract a sentence ending with . ? !
        # Take up to the first sentence-ending punctuation
        match = re.match(r"^(.+?[.?!])\s", stripped + " ")
        if match:
            sentence = match.group(1).strip()
            if len(sentence.split()) >= 5:
                return sentence

        # If the whole line is prose-like and long enough, use it
        if len(stripped.split()) >= 5 and not stripped.startswith("|"):
            return stripped

    return None


def generate_queries(
    strategy: str,
    notes_metadata: List[Dict[str, Any]],
    note_ids_set: set,
    note_texts: Optional[Dict[str, str]] = None,
) -> List[Tuple[str, str, str]]:
    """Generate (query_id, query_text, target_id) tuples for a strategy.

    Args:
        strategy: One of "title", "first_sentence", "link_title", "handcrafted"
        notes_metadata: List of note dicts from notes_metadata.json
        note_ids_set: Set of note IDs that have embeddings (for filtering)
        note_texts: {id: "title\\ncontent"} needed for first_sentence strategy

    Returns:
        List of (query_id, query_text, target_id) tuples
    """
    queries: List[Tuple[str, str, str]] = []

    if strategy == "title":
        for note in notes_metadata:
            nid = note["id"]
            title = note["title"]
            if nid in note_ids_set and title and title.strip():
                queries.append((nid, title.strip(), nid))

    elif strategy == "first_sentence":
        if note_texts is None:
            logger.error("first_sentence strategy requires note_texts")
            return []
        for note in notes_metadata:
            nid = note["id"]
            if nid not in note_ids_set or nid not in note_texts:
                continue
            text = note_texts[nid]
            # Skip the title line (first line) to extract from content
            content_lines = text.split("\n", 1)
            content = content_lines[1] if len(content_lines) > 1 else ""
            sentence = extract_first_sentence(content)
            if sentence:
                queries.append((nid, sentence, nid))

    elif strategy == "link_title":
        for note in notes_metadata:
            src_id = note["id"]
            src_title = note["title"]
            if src_id not in note_ids_set or not src_title or not src_title.strip():
                continue
            for link in note.get("links", []):
                tgt_id = link["target_id"]
                if tgt_id in note_ids_set:
                    query_id = f"{src_id}->{tgt_id}"
                    queries.append((query_id, src_title.strip(), tgt_id))

    elif strategy == "handcrafted":
        hc_queries = load_handcrafted_queries()
        for hc in hc_queries:
            tgt_id = hc["target_id"]
            if tgt_id in note_ids_set:
                queries.append((tgt_id, hc["query"], tgt_id))
            else:
                logger.warning(f"Handcrafted target {tgt_id} not in note embeddings")

    else:
        logger.error(f"Unknown strategy: {strategy}")

    return queries


# ---------------------------------------------------------------------------
# Phase 1: Embed queries
# ---------------------------------------------------------------------------

def phase_embed_queries(
    strategies: List[str],
    embedder_keys: List[str],
    embedder_dir: Path,
    notes_metadata: List[Dict[str, Any]],
    note_texts: Optional[Dict[str, str]],
    device: str,
) -> None:
    """Pre-compute query embeddings for each (strategy, embedder) pair."""
    if device == "gpu":
        ort_providers = "CUDAExecutionProvider,CPUExecutionProvider"
    else:
        ort_providers = "cpu"

    for ekey in embedder_keys:
        config_dir = embedder_dir / ekey
        note_ids = load_note_ids(config_dir)
        if note_ids is None:
            logger.warning(f"No note_ids for {ekey}, skipping")
            continue
        note_ids_set = set(note_ids)

        # Load embedder model once per embedder
        config = get_model_config(ekey)
        logger.info(f"\nLoading embedder: {ekey}")
        provider = OnnxEmbeddingProvider(
            model_id=config["model_id"],
            onnx_filename=config["onnx_filename"],
            max_length=config["max_tokens"],
            providers=ort_providers,
            output_mode=config["output_mode"],
            model_profile=config.get("profile"),
            extra_files=config.get("extra_files"),
        )

        try:
            t0 = time.perf_counter()
            provider.load()
            logger.info(f"  Loaded in {time.perf_counter() - t0:.1f}s")
        except Exception as e:
            logger.error(f"  Failed to load {ekey}: {e}")
            continue

        for strategy in strategies:
            logger.info(f"  Strategy: {strategy}")

            queries = generate_queries(strategy, notes_metadata, note_ids_set, note_texts)
            if not queries:
                logger.warning(f"    No queries generated for {strategy}")
                continue

            query_texts = [q[1] for q in queries]
            query_ids = [q[0] for q in queries]
            target_ids = [q[2] for q in queries]

            # Compute average query length
            avg_len = sum(len(t) for t in query_texts) / len(query_texts)
            logger.info(f"    {len(queries)} queries, avg {avg_len:.0f} chars")

            # Embed all queries
            t0 = time.perf_counter()
            embeddings_list = provider.embed_batch_adaptive(query_texts)
            embed_time = time.perf_counter() - t0

            # Stack and normalize
            query_embeddings = np.vstack(embeddings_list).astype(np.float32)
            norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            query_embeddings = query_embeddings / norms

            logger.info(f"    Embedded in {embed_time:.1f}s, shape {query_embeddings.shape}")

            # Save embeddings
            npz_path = config_dir / f"queries_{strategy}.npz"
            np.savez_compressed(npz_path, embeddings=query_embeddings)

            # Save metadata
            samples = [(qid, qt[:80], tid) for qid, qt, tid in queries[:5]]
            meta_path = config_dir / f"queries_{strategy}.json"
            _save_json(meta_path, {
                "strategy": strategy,
                "embedder": ekey,
                "count": len(queries),
                "avg_query_len_chars": round(avg_len, 1),
                "embed_time_s": round(embed_time, 2),
                "shape": list(query_embeddings.shape),
                "query_ids": query_ids,
                "target_ids": target_ids,
                "samples": [{"query_id": s[0], "query_text": s[1], "target_id": s[2]} for s in samples],
            })

            logger.info(f"    Saved to {npz_path}")

        # Unload embedder
        provider.unload()
        del provider
        gc.collect()


# ---------------------------------------------------------------------------
# Core evaluation helpers
# ---------------------------------------------------------------------------

def _truncate_for_reranker(text: str, max_chars: int) -> str:
    """Truncate text to approximate character limit for reranker input."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def _score_pairs_batched(
    reranker: OnnxRerankerProvider,
    query: str,
    documents: List[str],
    batch_size: int = 1,
    max_chars: int = 0,
) -> List[float]:
    """Score query-document pairs in sub-batches to avoid OOM.

    OOM recovery strategy (3 levels):
      1. Retry the same batch with 1000-char truncation
      2. Fall back to scoring one document at a time (batch_size=1)
      3. If single-doc also OOMs, assign score 0.0 for that document

    Args:
        max_chars: If >0, truncate documents to this many chars.
    """
    if max_chars > 0:
        documents = [_truncate_for_reranker(d, max_chars) for d in documents]

    all_scores: List[float] = []
    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        try:
            scores = reranker._score_pairs(query, batch)
        except Exception as e:
            if "Available memory" in str(e) or "OOM" in str(e).upper():
                logger.warning(
                    "GPU OOM on batch %d (size=%d), retrying with 1000-char truncation",
                    i, len(batch),
                )
                short_batch = [_truncate_for_reranker(d, 1000) for d in batch]
                try:
                    scores = reranker._score_pairs(query, short_batch)
                except Exception:
                    # Batch still too large — fall back to one-at-a-time
                    logger.warning(
                        "OOM retry failed, falling back to single-doc scoring"
                    )
                    scores = []
                    for j, doc in enumerate(short_batch):
                        try:
                            s = reranker._score_pairs(query, [doc])
                            scores.extend(s)
                        except Exception:
                            logger.error(
                                "OOM on single doc %d (len=%d), assigning 0.0",
                                i + j, len(doc),
                            )
                            scores.append(0.0)
            else:
                raise
        all_scores.extend(scores)
    return all_scores


def recall_at_k(ranks: np.ndarray, k: int) -> float:
    """Fraction of items ranked at or above position k."""
    return float(np.sum(ranks <= k) / len(ranks)) if len(ranks) > 0 else 0.0


# ---------------------------------------------------------------------------
# Phase 2: Evaluate
# ---------------------------------------------------------------------------

def evaluate_strategy(
    strategy: str,
    reranker: OnnxRerankerProvider,
    note_embeddings: np.ndarray,
    note_ids: List[str],
    query_embeddings: np.ndarray,
    query_ids: List[str],
    query_texts: List[str],
    target_ids: List[str],
    note_texts: Dict[str, str],
    pool_size: int,
    rerank_batch_size: int = 5,
    max_text_chars: int = 8000,
) -> Dict[str, Any]:
    """Evaluate one (strategy, reranker, embedder) triple.

    Args:
        strategy: Query strategy name
        reranker: Loaded reranker model
        note_embeddings: (N_notes, dim) pre-computed note embeddings
        note_ids: Note IDs corresponding to note_embeddings rows
        query_embeddings: (N_queries, dim) pre-computed query embeddings
        query_ids: Query IDs (note_id for self-retrieval, "src->tgt" for link_title)
        query_texts: Original query text strings for reranking
        target_ids: Target note ID for each query
        note_texts: {note_id: "title\\ncontent"} for document text
        pool_size: Number of KNN candidates
        rerank_batch_size: Sub-batch size for reranker (5 is safe default)
        max_text_chars: Truncation limit for doc text
    """
    id_to_idx = {nid: i for i, nid in enumerate(note_ids)}
    n_queries = len(query_ids)

    # Asymmetric similarity: query_embeddings @ note_embeddings.T
    sim_matrix = query_embeddings @ note_embeddings.T  # (N_queries, N_notes)

    embed_ranks: List[int] = []
    rerank_ranks: List[int] = []
    pool_hits = 0
    pool_misses = 0
    t0 = time.perf_counter()

    if strategy == "link_title":
        # Group by source note to rerank each source's pool only once
        source_groups: Dict[str, List[int]] = defaultdict(list)
        for qi, qid in enumerate(query_ids):
            src_id = qid.split("->")[0]
            source_groups[src_id].append(qi)

        unique_sources = sorted(source_groups.keys())
        logger.info(f"    {len(unique_sources)} unique sources for {n_queries} pairs")

        for src_num, src_id in enumerate(unique_sources):
            qi_list = source_groups[src_id]
            if src_id not in id_to_idx:
                pool_misses += len(qi_list)
                continue

            src_idx = id_to_idx[src_id]
            # All queries in this group share the same query text and embedding
            sims = sim_matrix[qi_list[0]].copy()
            sims[src_idx] = -np.inf  # Exclude source from its own pool

            pool_indices = np.argsort(sims)[::-1][:pool_size]
            pool_set = set(pool_indices.tolist())

            # Find which targets are in the pool
            qi_in_pool = []
            for qi in qi_list:
                tgt_id = target_ids[qi]
                if tgt_id not in id_to_idx:
                    pool_misses += 1
                    continue
                tgt_idx = id_to_idx[tgt_id]
                if tgt_idx in pool_set:
                    qi_in_pool.append((qi, tgt_idx))
                    pool_hits += 1
                else:
                    pool_misses += 1

            if not qi_in_pool:
                continue

            # Rerank this source's pool ONCE using the short query text
            q_text = query_texts[qi_list[0]]  # Same for all in group

            pool_doc_texts = []
            for pidx in pool_indices:
                nid = note_ids[pidx]
                pool_doc_texts.append(note_texts.get(nid, ""))

            scores = _score_pairs_batched(
                reranker, q_text, pool_doc_texts, rerank_batch_size, max_text_chars
            )
            rerank_order = np.argsort(scores)[::-1]

            # Embedding-order for pool
            pool_sims = sims[pool_indices]
            embed_order = np.argsort(pool_sims)[::-1]

            for qi, tgt_idx in qi_in_pool:
                tgt_pos_in_pool = int(np.where(pool_indices == tgt_idx)[0][0])
                e_rank = int(np.where(embed_order == tgt_pos_in_pool)[0][0]) + 1
                r_rank = int(np.where(rerank_order == tgt_pos_in_pool)[0][0]) + 1
                embed_ranks.append(e_rank)
                rerank_ranks.append(r_rank)

            if (src_num + 1) % 50 == 0 or src_num == 0:
                elapsed = time.perf_counter() - t0
                rate = (src_num + 1) / elapsed
                eta = (len(unique_sources) - src_num - 1) / rate
                logger.info(f"    [{src_num + 1}/{len(unique_sources)} sources] "
                            f"{rate:.1f} src/s, {len(embed_ranks)} pairs scored, "
                            f"ETA {eta:.0f}s")

    else:
        # title, first_sentence, handcrafted: each query is independent
        for qi in range(n_queries):
            tgt_id = target_ids[qi]
            if tgt_id not in id_to_idx:
                pool_misses += 1
                continue

            tgt_idx = id_to_idx[tgt_id]
            sims = sim_matrix[qi].copy()

            # Self-retrieval: query embedding differs from note embedding
            # (different text was embedded), so target CAN appear in pool.
            # No exclusion needed.

            pool_indices = np.argsort(sims)[::-1][:pool_size]

            if tgt_idx not in set(pool_indices.tolist()):
                pool_misses += 1
                continue
            pool_hits += 1

            # Rerank using the short query text
            q_text = query_texts[qi]

            pool_doc_texts = []
            for pidx in pool_indices:
                nid = note_ids[pidx]
                pool_doc_texts.append(note_texts.get(nid, ""))

            scores = _score_pairs_batched(
                reranker, q_text, pool_doc_texts, rerank_batch_size, max_text_chars
            )
            rerank_order = np.argsort(scores)[::-1]

            # Embedding rank
            pool_sims = sims[pool_indices]
            embed_order = np.argsort(pool_sims)[::-1]
            tgt_pos_in_pool = int(np.where(pool_indices == tgt_idx)[0][0])
            e_rank = int(np.where(embed_order == tgt_pos_in_pool)[0][0]) + 1
            r_rank = int(np.where(rerank_order == tgt_pos_in_pool)[0][0]) + 1

            embed_ranks.append(e_rank)
            rerank_ranks.append(r_rank)

            if (qi + 1) % 50 == 0 or qi == 0:
                elapsed = time.perf_counter() - t0
                rate = (qi + 1) / elapsed
                eta = (n_queries - qi - 1) / rate
                logger.info(f"    [{qi + 1}/{n_queries}] "
                            f"{rate:.1f} q/s, ETA {eta:.0f}s")

    eval_time = time.perf_counter() - t0
    total_queries = pool_hits + pool_misses

    if not embed_ranks:
        return {
            "pool_size": pool_size,
            "total_queries": total_queries,
            "queries_evaluated": 0,
            "pool_recall": 0.0,
            "pool_hits": pool_hits,
            "pool_misses": pool_misses,
            "embed_mrr": 0.0,
            "rerank_mrr": 0.0,
            "mrr_lift": 0.0,
            "embed_recall_at_1": 0.0, "rerank_recall_at_1": 0.0,
            "embed_recall_at_5": 0.0, "rerank_recall_at_5": 0.0,
            "embed_recall_at_10": 0.0, "rerank_recall_at_10": 0.0,
            "eval_time_s": round(eval_time, 2),
        }

    embed_ranks_arr = np.array(embed_ranks)
    rerank_ranks_arr = np.array(rerank_ranks)
    n = len(embed_ranks)

    embed_mrr = float(np.mean(1.0 / embed_ranks_arr))
    rerank_mrr = float(np.mean(1.0 / rerank_ranks_arr))
    mrr_lift = (rerank_mrr / embed_mrr - 1) if embed_mrr > 0 else 0.0

    avg_query_len = sum(len(t) for t in query_texts) / len(query_texts)

    return {
        "strategy": strategy,
        "pool_size": pool_size,
        "total_queries": total_queries,
        "queries_evaluated": n,
        "pool_recall": round(pool_hits / total_queries, 4) if total_queries > 0 else 0.0,
        "pool_hits": pool_hits,
        "pool_misses": pool_misses,
        "embed_mrr": round(embed_mrr, 4),
        "rerank_mrr": round(rerank_mrr, 4),
        "mrr_lift": round(mrr_lift, 4),
        "embed_recall_at_1": round(recall_at_k(embed_ranks_arr, 1), 4),
        "rerank_recall_at_1": round(recall_at_k(rerank_ranks_arr, 1), 4),
        "embed_recall_at_5": round(recall_at_k(embed_ranks_arr, 5), 4),
        "rerank_recall_at_5": round(recall_at_k(rerank_ranks_arr, 5), 4),
        "embed_recall_at_10": round(recall_at_k(embed_ranks_arr, 10), 4),
        "rerank_recall_at_10": round(recall_at_k(rerank_ranks_arr, 10), 4),
        "embed_mean_rank": round(float(np.mean(embed_ranks_arr)), 2),
        "rerank_mean_rank": round(float(np.mean(rerank_ranks_arr)), 2),
        "avg_query_len_chars": round(avg_query_len, 1),
        "eval_time_s": round(eval_time, 2),
    }


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def _print_summary(
    strategies: List[str],
    reranker_keys: List[str],
    embedder_keys: List[str],
    all_results: Dict[str, Dict[str, Dict[str, Any]]],
    pool_size: int,
    num_notes: int,
    total_wall_s: float,
    output_dir: Path,
) -> None:
    """Print formatted summary tables, one per strategy."""
    logger.info("")
    logger.info("=" * 90)
    logger.info(f"RERANKER v2 MATRIX  [pool={pool_size}, notes={num_notes}]")
    logger.info("=" * 90)

    for strategy in strategies:
        strat_results = all_results.get(strategy, {})
        if not strat_results:
            continue

        # Get query count from first available result
        sample_result = None
        for rkey in reranker_keys:
            for ekey in embedder_keys:
                r = strat_results.get(rkey, {}).get(ekey)
                if r and isinstance(r, dict) and r.get("queries_evaluated", 0) > 0:
                    sample_result = r
                    break
            if sample_result:
                break

        if not sample_result:
            continue

        total_q = sample_result.get("total_queries", 0)
        avg_len = sample_result.get("avg_query_len_chars", 0)

        if strategy == "link_title":
            # Count unique sources from query_ids (not directly available here)
            logger.info(f"\nStrategy: {strategy} ({total_q} pairs, avg {avg_len:.0f} chars)")
        else:
            logger.info(f"\nStrategy: {strategy} ({total_q} queries, avg {avg_len:.0f} chars)")

        # Column headers (embedders)
        col_width = 24
        header = f"{'Reranker':<24}"
        for ekey in embedder_keys:
            short = ekey[:20]
            header += f" {short:>{col_width}}"
        logger.info(header)

        sub = f"{'':<24}"
        for _ in embedder_keys:
            sub += f" {'MRR    Lift   R@5   R@10':>{col_width}}"
        logger.info(sub)
        logger.info("-" * (24 + (col_width + 1) * len(embedder_keys)))

        # Reranker rows
        for rkey in reranker_keys:
            row = f"{rkey:<24}"
            for ekey in embedder_keys:
                result = strat_results.get(rkey, {}).get(ekey, {})
                if not result or not isinstance(result, dict) or result.get("queries_evaluated", 0) == 0:
                    row += f" {'---':>{col_width}}"
                else:
                    mrr = result.get("rerank_mrr", 0)
                    lift = result.get("mrr_lift", 0)
                    r5 = result.get("rerank_recall_at_5", 0)
                    r10 = result.get("rerank_recall_at_10", 0)
                    cell = f"{mrr:.3f} {lift:+.0%} {r5:.3f} {r10:.3f}"
                    row += f" {cell:>{col_width}}"
            logger.info(row)

        # Embed-only baseline row
        row = f"{'(embed-only)':<24}"
        for ekey in embedder_keys:
            # Get baseline from any reranker's result for this embedder
            baseline = None
            for rkey in reranker_keys:
                r = strat_results.get(rkey, {}).get(ekey, {})
                if r and isinstance(r, dict) and r.get("queries_evaluated", 0) > 0:
                    baseline = r
                    break
            if baseline:
                mrr = baseline.get("embed_mrr", 0)
                r5 = baseline.get("embed_recall_at_5", 0)
                r10 = baseline.get("embed_recall_at_10", 0)
                cell = f"{mrr:.3f}  ---  {r5:.3f} {r10:.3f}"
                row += f" {cell:>{col_width}}"
            else:
                row += f" {'---':>{col_width}}"
        logger.info(row)

    logger.info(f"\nTotal wall time: {total_wall_s:.0f}s ({total_wall_s / 60:.1f}min)")
    logger.info(f"Results saved to {output_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reranker v2 benchmark: agent-query evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--phase",
        choices=["embed-queries", "evaluate"],
        default="evaluate",
        help="Phase to run: embed-queries (pre-compute) or evaluate (default)",
    )
    parser.add_argument(
        "--strategies",
        default="title,link_title,handcrafted",
        help=f"Comma-separated strategies: {','.join(ALL_STRATEGIES)} (default: title,link_title,handcrafted)",
    )
    parser.add_argument(
        "--embedder-dir",
        default="benchmarks/matrix-cpu",
        help="Directory with pre-computed note embeddings (default: benchmarks/matrix-cpu)",
    )
    parser.add_argument(
        "--embedders",
        default="gte-modernbert-c8192-fp32",
        help='Comma-separated embedder config keys, or "all" (default: gte-modernbert-c8192-fp32)',
    )
    parser.add_argument(
        "--rerankers",
        default="all",
        help='Comma-separated reranker config keys, or "all" (default: all)',
    )
    parser.add_argument(
        "--output-dir",
        default="benchmarks/rerank-v2",
        help="Output directory for results (default: benchmarks/rerank-v2)",
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=30,
        help="Number of KNN candidates for reranking (default: 30, matches production limit*3)",
    )
    parser.add_argument(
        "--rerank-batch-size",
        type=int,
        default=5,
        help="Sub-batch size for reranker scoring (default: 5). "
        "Lower values use less memory. Higher values are faster. "
        "Set to 1 for minimal memory with large-context (8192) rerankers.",
    )
    parser.add_argument(
        "--max-text-chars",
        type=int,
        default=8000,
        help="Truncate doc text to N chars before reranking (default: 8000). Set 0 to disable.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "gpu"],
        default="cpu",
        help="Inference device (default: cpu)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip completed (strategy, reranker, embedder) triples",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    # Logging setup — force line-buffered output even when redirected to file
    level = logging.DEBUG if args.verbose else logging.INFO
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    ))
    handler.terminator = "\n"
    logging.basicConfig(level=level, handlers=[handler])
    # Force unbuffered stderr so logs appear immediately in redirected output
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)

    # Parse strategies
    strategies = [s.strip() for s in args.strategies.split(",")]
    for s in strategies:
        if s not in ALL_STRATEGIES:
            logger.error(f"Unknown strategy: {s}")
            logger.info(f"Available: {', '.join(ALL_STRATEGIES)}")
            sys.exit(1)

    embedder_dir = Path(args.embedder_dir)
    output_dir = Path(args.output_dir)

    # Resolve embedder configs
    if args.embedders == "all":
        embedder_keys = sorted([
            d.name for d in embedder_dir.iterdir()
            if d.is_dir() and (d / "embeddings.npz").exists()
        ])
    else:
        embedder_keys = [k.strip() for k in args.embedders.split(",")]
        for k in embedder_keys:
            if not (embedder_dir / k / "embeddings.npz").exists():
                logger.error(f"No embeddings found for {k} at {embedder_dir / k}")
                sys.exit(1)

    if not embedder_keys:
        logger.error(f"No embedding results found in {embedder_dir}")
        sys.exit(1)

    # Load shared data
    metadata = load_notes_metadata(embedder_dir)
    if not metadata:
        logger.error(f"notes_metadata.json not found in {embedder_dir}")
        sys.exit(1)

    notes_data = metadata["notes"]
    notes_dir = Path(metadata["notes_dir"])

    logger.info(f"Strategies: {', '.join(strategies)}")
    logger.info(f"Embedders ({len(embedder_keys)}): {', '.join(embedder_keys)}")

    # Load note texts if needed (for first_sentence extraction or reranking)
    note_texts: Optional[Dict[str, str]] = None
    needs_texts = args.phase == "evaluate" or "first_sentence" in strategies
    if needs_texts:
        if not notes_dir.exists():
            logger.error(f"Notes directory not found: {notes_dir}")
            sys.exit(1)
        logger.info(f"Loading note texts from {notes_dir}...")
        note_texts = load_note_texts(notes_dir)
        logger.info(f"Loaded {len(note_texts)} note texts")

    # -----------------------------------------------------------------------
    # Phase 1: Embed queries
    # -----------------------------------------------------------------------
    if args.phase == "embed-queries":
        # Validate embedder configs exist in model_configs
        for k in embedder_keys:
            if k not in MODELS:
                logger.error(f"Unknown embedder config: {k}")
                logger.info(f"Available: {', '.join(sorted(MODELS.keys()))}")
                sys.exit(1)

        phase_embed_queries(
            strategies=strategies,
            embedder_keys=embedder_keys,
            embedder_dir=embedder_dir,
            notes_metadata=notes_data,
            note_texts=note_texts,
            device=args.device,
        )
        logger.info("\nPhase 1 complete. Query embeddings saved.")
        logger.info("Run without --phase to evaluate rerankers.")
        return

    # -----------------------------------------------------------------------
    # Phase 2: Evaluate
    # -----------------------------------------------------------------------
    # Resolve reranker configs
    if args.rerankers == "all":
        reranker_keys = list_reranker_configs()
    else:
        reranker_keys = [k.strip() for k in args.rerankers.split(",")]
        for k in reranker_keys:
            if k not in RERANKERS:
                logger.error(f"Unknown reranker config: {k}")
                logger.info(f"Available: {', '.join(sorted(RERANKERS.keys()))}")
                sys.exit(1)

    logger.info(f"Rerankers ({len(reranker_keys)}): {', '.join(reranker_keys)}")
    logger.info(f"Pool size: {args.pool_size}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Pre-load all note embeddings
    logger.info("Pre-loading note embeddings...")
    embeddings_cache: Dict[str, Tuple[np.ndarray, List[str]]] = {}
    for ekey in embedder_keys:
        config_dir = embedder_dir / ekey
        embs = load_embeddings(config_dir)
        nids = load_note_ids(config_dir)
        if embs is None or nids is None:
            logger.warning(f"  Missing data for {ekey}, skipping")
            continue
        embeddings_cache[ekey] = (embs, nids)
        logger.debug(f"  {ekey}: {embs.shape}")

    if not embeddings_cache:
        logger.error("No valid embeddings loaded")
        sys.exit(1)

    # Pre-load query embeddings and texts for each (strategy, embedder)
    logger.info("Pre-loading query embeddings...")
    query_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}  # strategy -> embedder -> data
    for strategy in strategies:
        query_cache[strategy] = {}
        for ekey in embedder_keys:
            config_dir = embedder_dir / ekey
            npz_path = config_dir / f"queries_{strategy}.npz"
            meta_path = config_dir / f"queries_{strategy}.json"

            if not npz_path.exists() or not meta_path.exists():
                logger.warning(f"  No query embeddings for {strategy}/{ekey}")
                logger.warning(f"  Run: --phase embed-queries --strategies {strategy} --embedders {ekey}")
                continue

            q_embs = np.load(npz_path)["embeddings"]
            with open(meta_path) as f:
                q_meta = json.load(f)

            # We need the original query texts for reranking.
            # Reconstruct from the strategy and metadata.
            note_ids_set = set(load_note_ids(config_dir) or [])
            query_tuples = generate_queries(strategy, notes_data, note_ids_set, note_texts)

            # Verify alignment: the query count should match
            if len(query_tuples) != q_embs.shape[0]:
                logger.warning(
                    f"  Query count mismatch for {strategy}/{ekey}: "
                    f"generated {len(query_tuples)} vs embedded {q_embs.shape[0]}. "
                    f"Re-run --phase embed-queries."
                )
                continue

            query_cache[strategy][ekey] = {
                "embeddings": q_embs,
                "query_ids": [q[0] for q in query_tuples],
                "query_texts": [q[1] for q in query_tuples],
                "target_ids": [q[2] for q in query_tuples],
                "count": len(query_tuples),
                "avg_query_len_chars": q_meta.get("avg_query_len_chars", 0),
            }
            logger.debug(f"  {strategy}/{ekey}: {q_embs.shape}")

    # Save run metadata
    sys_info = _collect_system_info(args.device)
    sys_info["embedder_dir"] = str(embedder_dir)
    sys_info["output_dir"] = str(output_dir)
    sys_info["pool_size"] = args.pool_size
    sys_info["strategies"] = strategies
    sys_info["reranker_keys"] = reranker_keys
    sys_info["embedder_keys"] = embedder_keys
    _save_json(output_dir / "run_metadata.json", sys_info)

    # Resolve ORT providers
    if args.device == "gpu":
        ort_providers = "CUDAExecutionProvider,CPUExecutionProvider"
    else:
        ort_providers = "cpu"

    # Run: iterate rerankers (outer) → strategies → embedders (inner)
    # This minimizes model load/unload cycles
    # all_results: strategy -> reranker -> embedder -> result
    all_results: Dict[str, Dict[str, Dict[str, Any]]] = {s: {} for s in strategies}
    total_triples = 0
    skipped = 0
    t_run_start = time.perf_counter()

    for ri, rkey in enumerate(reranker_keys, 1):
        logger.info(f"\n{'=' * 70}")
        logger.info(f"[{ri}/{len(reranker_keys)}] Reranker: {rkey}")
        logger.info(f"{'=' * 70}")

        rconfig = get_reranker_config(rkey)

        # Load reranker
        reranker = OnnxRerankerProvider(
            model_id=rconfig["model_id"],
            onnx_filename=rconfig["onnx_filename"],
            max_length=rconfig["max_tokens"],
            providers=ort_providers,
            extra_files=rconfig.get("extra_files"),
        )

        try:
            t0_load = time.perf_counter()
            reranker.load()
            load_time = time.perf_counter() - t0_load
            logger.info(f"  Loaded in {load_time:.1f}s")
        except Exception as e:
            logger.error(f"  Failed to load {rkey}: {e}")
            for strategy in strategies:
                all_results[strategy][rkey] = {"status": "failed", "error": str(e)}
            continue

        for strategy in strategies:
            logger.info(f"\n  Strategy: {strategy}")

            if rkey not in all_results[strategy]:
                all_results[strategy][rkey] = {}

            for ei, ekey in enumerate(embedder_keys, 1):
                if ekey not in embeddings_cache:
                    logger.warning(f"    [{ei}/{len(embedder_keys)}] {ekey}: no note embeddings")
                    continue

                if strategy not in query_cache or ekey not in query_cache.get(strategy, {}):
                    logger.warning(f"    [{ei}/{len(embedder_keys)}] {ekey}: no query embeddings")
                    continue

                # Check skip-existing
                result_dir = output_dir / strategy / rkey
                result_file = result_dir / f"{ekey}.json"
                if args.skip_existing and result_file.exists():
                    try:
                        with open(result_file) as f:
                            existing = json.load(f)
                        if existing.get("queries_evaluated", 0) > 0:
                            logger.info(f"    [{ei}/{len(embedder_keys)}] {ekey}: skipping (exists)")
                            all_results[strategy][rkey][ekey] = existing
                            skipped += 1
                            continue
                    except (json.JSONDecodeError, KeyError):
                        pass

                logger.info(f"    [{ei}/{len(embedder_keys)}] {ekey}...")

                note_embs, note_ids = embeddings_cache[ekey]
                qdata = query_cache[strategy][ekey]

                t0_eval = time.perf_counter()
                result = evaluate_strategy(
                    strategy=strategy,
                    reranker=reranker,
                    note_embeddings=note_embs,
                    note_ids=note_ids,
                    query_embeddings=qdata["embeddings"],
                    query_ids=qdata["query_ids"],
                    query_texts=qdata["query_texts"],
                    target_ids=qdata["target_ids"],
                    note_texts=note_texts or {},
                    pool_size=args.pool_size,
                    rerank_batch_size=args.rerank_batch_size,
                    max_text_chars=args.max_text_chars,
                )
                eval_time = time.perf_counter() - t0_eval

                result["reranker"] = rkey
                result["embedder"] = ekey
                result["strategy"] = strategy
                result["eval_time_s"] = round(eval_time, 2)

                # Save per-triple result
                result_dir.mkdir(parents=True, exist_ok=True)
                _save_json(result_file, result)
                all_results[strategy][rkey][ekey] = result
                total_triples += 1

                logger.info(
                    f"      pool_recall={result['pool_recall']:.3f}, "
                    f"embed_mrr={result['embed_mrr']:.4f}, "
                    f"rerank_mrr={result['rerank_mrr']:.4f}, "
                    f"lift={result['mrr_lift']:+.1%}, "
                    f"R@5: {result['embed_recall_at_5']:.3f}->{result['rerank_recall_at_5']:.3f}, "
                    f"R@10: {result['embed_recall_at_10']:.3f}->{result['rerank_recall_at_10']:.3f}, "
                    f"{eval_time:.1f}s"
                )

        # Unload reranker
        reranker.unload()
        del reranker
        gc.collect()

    total_wall_s = time.perf_counter() - t_run_start

    # Save combined matrix
    _save_json(output_dir / "rerank_matrix.json", {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "pool_size": args.pool_size,
        "num_notes": len(notes_data),
        "strategies": strategies,
        "num_rerankers": len(reranker_keys),
        "num_embedders": len(embedder_keys),
        "total_wall_time_s": round(total_wall_s, 1),
        "triples_evaluated": total_triples,
        "triples_skipped": skipped,
        "results": all_results,
    })

    # Print summary table
    _print_summary(
        strategies=strategies,
        reranker_keys=reranker_keys,
        embedder_keys=embedder_keys,
        all_results=all_results,
        pool_size=args.pool_size,
        num_notes=len(notes_data),
        total_wall_s=total_wall_s,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
