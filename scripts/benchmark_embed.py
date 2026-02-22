#!/usr/bin/env python
"""Embedding benchmark: embed notes with each model config sequentially.

Loads notes from markdown files, embeds them with each model configuration
one at a time, captures performance metrics, and saves raw vectors for
downstream quality evaluation.

Usage:
    cd /home/komi/repos/MCP/znote-mcp

    # CPU benchmark (all models, 12GB budget)
    uv run python scripts/benchmark_embed.py --models all --memory-budget-gb 12.0 -v

    # GPU benchmark (requires onnxruntime-gpu with CUDA)
    uv run python scripts/benchmark_embed.py --device gpu --output-dir benchmarks/matrix-gpu -v

    # Resume after crash (skip completed configs)
    uv run python scripts/benchmark_embed.py --skip-existing -v

    # Specific models only
    uv run python scripts/benchmark_embed.py --models bge-small-fp32,nomic-v1.5-fp32
"""

from __future__ import annotations

import argparse
import datetime
import gc
import json
import logging
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Ensure project root is on sys.path for script imports
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from model_configs import MODELS, get_config, list_configs
from znote_mcp.services.onnx_providers import (
    OnnxEmbeddingProvider,
    _get_peak_rss_mb,
    _get_rss_mb,
)
from znote_mcp.services.text_chunker import TextChunker
from znote_mcp.storage.markdown_parser import MarkdownParser

logger = logging.getLogger("benchmark_embed")


# ---------------------------------------------------------------------------
# System info
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

    # CPU count
    try:
        info["cpu_count"] = os.cpu_count()
    except Exception:
        pass

    # Total RAM
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    info["ram_total_gb"] = round(kb / 1024 / 1024, 1)
                    break
    except Exception:
        pass

    # GPU info
    if device == "gpu":
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(", ")
                if len(parts) >= 3:
                    info["gpu_name"] = parts[0]
                    info["gpu_vram_mb"] = int(parts[1].replace(" MiB", ""))
                    info["gpu_driver"] = parts[2]
        except Exception:
            pass
        # CUDA version
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=compute_cap",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                info["gpu_compute_cap"] = result.stdout.strip()
        except Exception:
            pass

    return info


def _setup_config_log(config_dir: Path, config_key: str) -> logging.FileHandler:
    """Add a file handler to capture all log output for this config."""
    log_path = config_dir / "embed.log"
    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s",
                          datefmt="%H:%M:%S")
    )
    # Attach to root logger so we capture OnnxEmbeddingProvider logs too
    logging.getLogger().addHandler(handler)
    return handler


def _teardown_config_log(handler: logging.FileHandler) -> None:
    """Remove the per-config file handler."""
    logging.getLogger().removeHandler(handler)
    handler.close()


# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------

def _get_gpu_mem_mb() -> Optional[float]:
    """Return current GPU memory used in MB, or None if unavailable."""
    try:
        import subprocess

        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return float(result.stdout.strip().split("\n")[0])
    except Exception:
        pass
    return None


def _get_gpu_total_mb() -> Optional[float]:
    """Return total GPU memory in MB, or None if unavailable."""
    try:
        import subprocess

        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return float(result.stdout.strip().split("\n")[0])
    except Exception:
        pass
    return None


def _check_gpu_available() -> bool:
    """Check if CUDAExecutionProvider actually works (not just listed)."""
    try:
        import onnxruntime as ort

        if "CUDAExecutionProvider" not in ort.get_available_providers():
            return False
        # Create a minimal session to verify CUDA actually loads
        # (ORT lists CUDA as available even when libs are missing)
        import tempfile

        import onnx
        from onnx import TensorProto, helper

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1])
        node = helper.make_node("Identity", ["X"], ["Y"])
        graph = helper.make_graph([node], "test", [X], [Y])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            tmp_path = f.name
        try:
            sess = ort.InferenceSession(
                tmp_path, providers=["CUDAExecutionProvider"]
            )
            active = sess.get_providers()
            return "CUDAExecutionProvider" in active
        finally:
            os.unlink(tmp_path)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Note loading
# ---------------------------------------------------------------------------


def load_notes(notes_dir: Path) -> List[Dict[str, Any]]:
    """Load notes from markdown files, returning dicts with id/title/content/tags/links."""
    parser = MarkdownParser()
    notes = []
    for md_file in sorted(notes_dir.glob("*.md")):
        try:
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()
            note = parser.parse_note(content)
            notes.append(
                {
                    "id": note.id,
                    "title": note.title,
                    "content": note.content,
                    "tags": [t.name for t in note.tags],
                    "links": [
                        {"source_id": link.source_id, "target_id": link.target_id}
                        for link in note.links
                    ],
                }
            )
        except Exception as e:
            logger.warning(f"Skipping {md_file.name}: {e}")
    return notes


def count_link_pairs(notes: List[Dict[str, Any]]) -> int:
    """Count linked pairs where both source and target exist in the note set."""
    note_ids = {n["id"] for n in notes}
    count = 0
    for note in notes:
        for link in note["links"]:
            if link["target_id"] in note_ids:
                count += 1
    return count


def sample_notes(
    notes: List[Dict[str, Any]], sample_size: int, min_link_pairs: int = 10
) -> List[Dict[str, Any]]:
    """Sample notes, ensuring enough link pairs for quality metrics."""
    if sample_size >= len(notes):
        return notes

    rng = np.random.default_rng(42)
    indices = rng.choice(len(notes), size=sample_size, replace=False)
    sampled = [notes[i] for i in sorted(indices)]
    pairs = count_link_pairs(sampled)
    if pairs >= min_link_pairs:
        logger.info(
            f"Sampled {len(sampled)} notes with {pairs} link pairs (>={min_link_pairs})"
        )
        return sampled

    logger.warning(
        f"Sample of {sample_size} has only {pairs} link pairs (<{min_link_pairs}). "
        f"Using full corpus ({len(notes)} notes) for meaningful quality metrics."
    )
    return notes


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


def embed_config(
    config_key: str,
    config: Dict[str, Any],
    notes: List[Dict[str, Any]],
    output_dir: Path,
    memory_budget_gb: float,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Embed all notes with a single model config. Returns performance results."""
    logger.info(f"{'=' * 60}")
    logger.info(f"Config: {config_key}  [device={device}]")
    logger.info(f"  model_id: {config['model_id']}")
    logger.info(f"  onnx: {config['onnx_filename']}")
    logger.info(
        f"  chunk_size: {config['chunk_size']}, max_tokens: {config['max_tokens']}"
    )
    logger.info(f"  output_mode: {config['output_mode']}")
    logger.info(f"{'=' * 60}")

    config_dir = output_dir / config_key
    config_dir.mkdir(parents=True, exist_ok=True)

    # Per-config log capture
    log_handler = _setup_config_log(config_dir, config_key)

    result: Dict[str, Any] = {
        "config_key": config_key,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_id": config["model_id"],
        "onnx_filename": config["onnx_filename"],
        "dim": config["dim"],
        "max_tokens": config["max_tokens"],
        "chunk_size": config["chunk_size"],
        "output_mode": config["output_mode"],
        "device": device,
        "status": "pending",
    }

    rss_before_load = _get_rss_mb()
    gpu_before_load = _get_gpu_mem_mb() if device == "gpu" else None

    # Select ORT providers based on device
    if device == "gpu":
        ort_providers = "CUDAExecutionProvider,CPUExecutionProvider"
        # Set ORT arena to 90% of total VRAM.  The BFC arena will fragment
        # across varying batch sizes, but the OOM retry in embed_batch_adaptive
        # handles this by splitting oversized batches.  Keeping the arena large
        # ensures big models (22+ layers) have enough room for activations.
        gpu_total = _get_gpu_total_mb()
        if gpu_total:
            arena_gb = round(gpu_total * 0.9 / 1024, 1)
            os.environ["ZETTELKASTEN_GPU_MEM_LIMIT_GB"] = str(arena_gb)
    else:
        ort_providers = "cpu"

    # Create provider
    provider = OnnxEmbeddingProvider(
        model_id=config["model_id"],
        onnx_filename=config["onnx_filename"],
        max_length=config["max_tokens"],
        providers=ort_providers,
        output_mode=config["output_mode"],
        model_profile=config.get("profile"),
        extra_files=config.get("extra_files"),
    )

    # Load model
    try:
        t0_load = time.perf_counter()
        provider.load()
        load_time = time.perf_counter() - t0_load
    except Exception as e:
        logger.error(f"Failed to load {config_key}: {e}")
        result["status"] = "failed"
        result["error"] = str(e)
        _save_json(config_dir / "perf.json", result)
        _teardown_config_log(log_handler)
        return result

    rss_after_load = _get_rss_mb()
    gpu_after_load = _get_gpu_mem_mb() if device == "gpu" else None

    result["load_time_s"] = round(load_time, 2)
    result["rss_before_load_mb"] = round(rss_before_load, 1)
    result["rss_after_load_mb"] = round(rss_after_load, 1)
    result["rss_model_delta_mb"] = round(rss_after_load - rss_before_load, 1)
    result["actual_dim"] = provider.dimension

    if gpu_before_load is not None and gpu_after_load is not None:
        result["gpu_before_load_mb"] = round(gpu_before_load, 1)
        result["gpu_after_load_mb"] = round(gpu_after_load, 1)
        result["gpu_model_delta_mb"] = round(gpu_after_load - gpu_before_load, 1)

    load_msg = (
        f"  Loaded in {load_time:.1f}s, dim={provider.dimension}, "
        f"RSS: {rss_after_load:.0f}MB (+{rss_after_load - rss_before_load:.0f}MB)"
    )
    if gpu_after_load is not None:
        load_msg += f", VRAM: {gpu_after_load:.0f}MB"
    logger.info(load_msg)

    # Prepare texts and chunk
    chunker = TextChunker(
        chunk_size=config["chunk_size"],
        chunk_overlap=256,
    )

    note_ids: List[str] = []
    note_texts: List[str] = []
    for note in notes:
        text = f"{note['title']}\n{note['content']}"
        note_ids.append(note["id"])
        note_texts.append(text)

    # Chunk all texts
    all_chunks: List[str] = []
    chunk_to_note: List[int] = []
    chunks_per_note: List[int] = []

    for note_idx, text in enumerate(note_texts):
        text_chunks = chunker.chunk(text)
        n_chunks = len(text_chunks)
        chunks_per_note.append(n_chunks)
        for tc in text_chunks:
            all_chunks.append(tc.text)
            chunk_to_note.append(note_idx)

    total_chunks = len(all_chunks)
    logger.info(
        f"  {len(notes)} notes → {total_chunks} chunks "
        f"(avg {total_chunks / max(len(notes), 1):.1f} chunks/note, "
        f"max {max(chunks_per_note) if chunks_per_note else 0})"
    )

    # For GPU: use a conservative activation budget.  The batcher's
    # _per_item_cost formula underestimates GPU memory by ~5-10x because
    # it uses eff_layers=3 while GPU keeps more layers resident.
    # We compensate by using a small budget (1GB), which creates smaller
    # batches.  The OOM retry in embed_batch_adaptive catches any
    # remaining overestimates.  GPU throughput is mostly batch-insensitive
    # since CUDA kernels parallelize within each item.
    effective_budget_gb = memory_budget_gb
    if device == "gpu":
        effective_budget_gb = 1.0
        logger.info(f"  GPU activation budget: {effective_budget_gb:.1f}GB")

    # Embed chunks
    try:
        t0_embed = time.perf_counter()
        chunk_embeddings = provider.embed_batch_adaptive(
            all_chunks, memory_budget_gb=effective_budget_gb
        )
        embed_time = time.perf_counter() - t0_embed
    except Exception as e:
        logger.error(f"Failed to embed with {config_key}: {e}")
        result["status"] = "failed"
        result["error"] = str(e)
        provider.unload()
        gc.collect()
        _save_json(config_dir / "perf.json", result)
        _teardown_config_log(log_handler)
        return result

    peak_rss = _get_peak_rss_mb()
    gpu_peak = _get_gpu_mem_mb() if device == "gpu" else None

    result["total_notes"] = len(notes)
    result["total_chunks"] = total_chunks
    result["embed_time_s"] = round(embed_time, 2)
    result["notes_per_sec"] = round(len(notes) / max(embed_time, 0.001), 1)
    result["chunks_per_sec"] = round(total_chunks / max(embed_time, 0.001), 1)
    result["peak_rss_mb"] = round(peak_rss, 1)
    if gpu_peak is not None:
        result["gpu_peak_mb"] = round(gpu_peak, 1)
    result["chunks_per_note"] = {
        "min": min(chunks_per_note) if chunks_per_note else 0,
        "max": max(chunks_per_note) if chunks_per_note else 0,
        "avg": round(sum(chunks_per_note) / max(len(chunks_per_note), 1), 2),
    }
    result["status"] = "ok"

    perf_msg = (
        f"  Embedded in {embed_time:.1f}s "
        f"({result['notes_per_sec']} notes/s, {result['chunks_per_sec']} chunks/s), "
        f"peak RSS: {peak_rss:.0f}MB"
    )
    if gpu_peak is not None:
        perf_msg += f", VRAM: {gpu_peak:.0f}MB"
    logger.info(perf_msg)

    # Compute note-level embeddings by mean-pooling chunk embeddings
    note_embeddings_list: List[np.ndarray] = []
    emb_matrix = np.stack(chunk_embeddings)  # (total_chunks, dim)

    for note_idx in range(len(notes)):
        chunk_indices = [
            ci for ci, ni in enumerate(chunk_to_note) if ni == note_idx
        ]
        if chunk_indices:
            pooled = emb_matrix[chunk_indices].mean(axis=0)
            norm = np.linalg.norm(pooled)
            if norm > 1e-12:
                pooled = pooled / norm
            note_embeddings_list.append(pooled)
        else:
            note_embeddings_list.append(
                np.zeros(provider.dimension, dtype=np.float32)
            )

    note_embeddings = np.stack(note_embeddings_list)  # (N, dim)

    # Save results
    np.savez_compressed(
        config_dir / "chunk_embeddings.npz",
        embeddings=emb_matrix,
    )
    np.savez_compressed(
        config_dir / "embeddings.npz",
        embeddings=note_embeddings,
    )
    _save_json(
        config_dir / "note_ids.json",
        {
            "note_ids": note_ids,
            "chunk_to_note": chunk_to_note,
            "chunks_per_note": chunks_per_note,
        },
    )
    _save_json(config_dir / "perf.json", result)

    # Unload model and free memory
    provider.unload()
    del chunk_embeddings, emb_matrix, note_embeddings, note_embeddings_list
    gc.collect()

    rss_after_unload = _get_rss_mb()
    logger.info(f"  Unloaded. RSS: {rss_after_unload:.0f}MB")

    _teardown_config_log(log_handler)
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save_json(path: Path, data: Any) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def _resolve_notes_dir(args_notes_dir: Optional[str]) -> Path:
    """Resolve notes directory from args or environment."""
    if args_notes_dir:
        return Path(args_notes_dir)

    env_dir = os.environ.get("ZETTELKASTEN_NOTES_DIR")
    if env_dir:
        return Path(env_dir)

    default = Path.home() / ".zettelkasten" / "notes"
    if default.exists():
        return default

    raise ValueError(
        "No notes directory found. Specify --notes-dir or set ZETTELKASTEN_NOTES_DIR"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embedding model benchmark matrix",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--models",
        default="all",
        help='Comma-separated config keys, or "all" (default: all)',
    )
    parser.add_argument("--notes-dir", help="Path to notes directory")
    parser.add_argument(
        "--output-dir",
        default="benchmarks/matrix",
        help="Output directory (default: benchmarks/matrix)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Sample N notes (0 = use all, default: 0)",
    )
    parser.add_argument(
        "--memory-budget-gb",
        type=float,
        default=12.0,
        help="Memory budget for adaptive batching in GB (default: 12.0)",
    )
    parser.add_argument(
        "--min-link-pairs",
        type=int,
        default=10,
        help="Minimum link pairs for quality metrics (default: 10). "
        "If sample has fewer, auto-escalates to full corpus.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "gpu"],
        default="cpu",
        help="Inference device: cpu or gpu (default: cpu)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip configs that already have perf.json in the output directory",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    # Logging setup
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # GPU validation
    if args.device == "gpu":
        logger.info("Checking GPU availability...")
        if _check_gpu_available():
            gpu_total = _get_gpu_total_mb()
            gpu_used = _get_gpu_mem_mb()
            logger.info(
                f"  CUDA GPU available. "
                f"VRAM: {gpu_used:.0f}/{gpu_total:.0f} MB used"
            )
        else:
            logger.error(
                "CUDAExecutionProvider not available or failed to load. "
                "Install onnxruntime-gpu with matching CUDA version. "
                "For CUDA 13: pip install --pre --index-url "
                "https://aiinfra.pkgs.visualstudio.com/PublicPackages/"
                "_packaging/ort-cuda-13-nightly/pypi/simple/ "
                "onnxruntime-gpu --no-deps"
            )
            sys.exit(1)

    # Resolve model configs
    if args.models == "all":
        config_keys = list_configs()
    else:
        config_keys = [k.strip() for k in args.models.split(",")]
        for k in config_keys:
            if k not in MODELS:
                logger.error(f"Unknown config key: {k}")
                logger.info(f"Available: {', '.join(sorted(MODELS.keys()))}")
                sys.exit(1)

    # Load notes
    notes_dir = _resolve_notes_dir(args.notes_dir)
    logger.info(f"Loading notes from {notes_dir}")
    all_notes = load_notes(notes_dir)
    logger.info(f"Loaded {len(all_notes)} notes")

    if not all_notes:
        logger.error("No notes found!")
        sys.exit(1)

    # Sample if requested
    if args.sample > 0:
        notes = sample_notes(all_notes, args.sample, args.min_link_pairs)
    else:
        notes = all_notes

    link_pairs = count_link_pairs(notes)
    total_tags = len({t for n in notes for t in n["tags"]})
    logger.info(
        f"Using {len(notes)} notes, {link_pairs} link pairs, {total_tags} unique tags"
    )

    # Corpus text statistics
    note_lengths = [len(f"{n['title']}\n{n['content']}") for n in notes]
    note_lengths_sorted = sorted(note_lengths)
    corpus_stats = {
        "total_chars": sum(note_lengths),
        "mean_chars": round(sum(note_lengths) / max(len(note_lengths), 1), 1),
        "median_chars": note_lengths_sorted[len(note_lengths_sorted) // 2] if note_lengths_sorted else 0,
        "min_chars": min(note_lengths) if note_lengths else 0,
        "max_chars": max(note_lengths) if note_lengths else 0,
        "p25_chars": note_lengths_sorted[len(note_lengths_sorted) // 4] if note_lengths_sorted else 0,
        "p75_chars": note_lengths_sorted[3 * len(note_lengths_sorted) // 4] if note_lengths_sorted else 0,
        "p95_chars": note_lengths_sorted[int(0.95 * len(note_lengths_sorted))] if note_lengths_sorted else 0,
    }
    logger.info(
        f"Corpus stats: {corpus_stats['total_chars']:,} chars total, "
        f"mean={corpus_stats['mean_chars']:.0f}, median={corpus_stats['median_chars']}, "
        f"p95={corpus_stats['p95_chars']}, max={corpus_stats['max_chars']}"
    )

    # Save note metadata for quality evaluator
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect and save system info
    sys_info = _collect_system_info(args.device)
    sys_info["notes_dir"] = str(notes_dir)
    sys_info["output_dir"] = str(output_dir)
    sys_info["memory_budget_gb"] = args.memory_budget_gb
    sys_info["num_configs"] = len(config_keys)
    sys_info["config_keys"] = config_keys
    _save_json(output_dir / "run_metadata.json", sys_info)
    logger.info(f"System info saved to {output_dir}/run_metadata.json")

    _save_json(
        output_dir / "notes_metadata.json",
        {
            "notes_dir": str(notes_dir),
            "total_notes": len(notes),
            "link_pairs": link_pairs,
            "unique_tags": total_tags,
            "corpus_stats": corpus_stats,
            "notes": [
                {
                    "id": n["id"],
                    "title": n["title"],
                    "tags": n["tags"],
                    "links": n["links"],
                }
                for n in notes
            ],
        },
    )

    # Run benchmarks sequentially.
    # For GPU: run each config in a subprocess to get a clean CUDA context.
    # Without this, ORT's BFC arena + CUDA caching allocator hoard VRAM
    # across configs, causing cascading OOM failures after the first config.
    all_results: Dict[str, Any] = {}
    skipped = 0
    t_run_start = time.perf_counter()
    for i, key in enumerate(config_keys, 1):
        # Skip existing if requested
        if args.skip_existing:
            perf_file = output_dir / key / "perf.json"
            if perf_file.exists():
                try:
                    with open(perf_file) as f:
                        existing = json.load(f)
                    if existing.get("status") == "ok":
                        logger.info(
                            f"[{i}/{len(config_keys)}] Skipping {key} (already completed)"
                        )
                        all_results[key] = existing
                        skipped += 1
                        continue
                except (json.JSONDecodeError, KeyError):
                    pass  # Re-run if perf.json is corrupt

        logger.info(f"\n[{i}/{len(config_keys)}] Running {key}")

        if args.device == "gpu" and len(config_keys) > 1:
            # Fork a subprocess for each GPU config so CUDA context is fresh.
            import subprocess as _sp

            sub_args = [
                sys.executable, __file__,
                "--models", key,
                "--notes-dir", str(notes_dir),
                "--output-dir", str(output_dir),
                "--memory-budget-gb", str(args.memory_budget_gb),
                "--device", "gpu",
            ]
            if args.sample > 0:
                sub_args += ["--sample", str(args.sample)]
            if args.min_link_pairs != 10:
                sub_args += ["--min-link-pairs", str(args.min_link_pairs)]
            if logger.isEnabledFor(logging.DEBUG):
                sub_args.append("-v")
            proc = _sp.run(sub_args, capture_output=False, text=True)
            # Read result from saved perf.json
            perf_file = output_dir / key / "perf.json"
            if perf_file.exists():
                with open(perf_file) as f:
                    result = json.load(f)
            else:
                result = {"config_key": key, "status": "failed",
                          "error": f"subprocess exited {proc.returncode}"}
        else:
            config = get_config(key)
            result = embed_config(
                key, config, notes, output_dir, args.memory_budget_gb, args.device
            )

        all_results[key] = result
        logger.info(f"  Status: {result['status']}")

    total_wall_s = time.perf_counter() - t_run_start

    # Write combined performance matrix with run totals
    _save_json(output_dir / "perf_matrix.json", {
        "run_metadata": {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "device": args.device,
            "total_notes": len(notes),
            "total_configs": len(config_keys),
            "configs_ok": sum(1 for r in all_results.values() if r.get("status") == "ok"),
            "configs_failed": sum(1 for r in all_results.values() if r.get("status") == "failed"),
            "configs_skipped": skipped,
            "total_wall_time_s": round(total_wall_s, 1),
            "memory_budget_gb": args.memory_budget_gb,
        },
        "results": all_results,
    })

    # Print summary
    logger.info("\n" + "=" * 90)
    logger.info(f"PERFORMANCE SUMMARY  [device={args.device}, notes={len(notes)}, wall={total_wall_s:.0f}s]")
    if skipped:
        logger.info(f"({skipped} configs skipped via --skip-existing)")
    logger.info("=" * 90)

    if args.device == "gpu":
        header = (
            f"{'Config':<28} {'St':<4} {'Load':<6} {'Embed':<8} "
            f"{'n/s':<7} {'c/s':<8} {'Chunks':<7} {'RSS':<7} {'VRAM':<7} {'Dim':<5}"
        )
    else:
        header = (
            f"{'Config':<28} {'St':<4} {'Load':<6} {'Embed':<8} "
            f"{'n/s':<7} {'c/s':<8} {'Chunks':<7} {'RSS':<7} {'Dim':<5}"
        )
    logger.info(header)
    logger.info("-" * 90)

    for key in config_keys:
        r = all_results.get(key)
        if not r:
            continue
        if r["status"] == "ok":
            line = (
                f"{key:<28} {'ok':<4} "
                f"{r['load_time_s']:<6.1f} "
                f"{r['embed_time_s']:<8.1f} "
                f"{r['notes_per_sec']:<7.1f} "
                f"{r['chunks_per_sec']:<8.1f} "
                f"{r['total_chunks']:<7} "
                f"{r['peak_rss_mb']:<7.0f} "
            )
            if args.device == "gpu":
                line += f"{r.get('gpu_peak_mb', 0):<7.0f} "
            line += f"{r.get('actual_dim', '?'):<5}"
            logger.info(line)
        else:
            logger.info(
                f"{key:<28} {'FAIL':<4} {r.get('error', 'unknown')[:50]}"
            )

    logger.info(f"\nTotal wall time: {total_wall_s:.0f}s ({total_wall_s/60:.1f}min)")
    logger.info(f"Results saved to {output_dir}/")
    logger.info(f"Per-config logs in {output_dir}/<config>/embed.log")


if __name__ == "__main__":
    main()
