#!/usr/bin/env python
"""Reindex benchmark with structured result capture.

Runs a full embedding reindex, captures per-bucket and per-phase timing,
and saves results to benchmarks/results/<timestamp>.json plus appends
a summary row to benchmarks/summary.csv.

Usage:
    cd /home/komi/repos/MCP/znote-mcp
    uv run python scripts/reindex_benchmark.py
    uv run python scripts/reindex_benchmark.py --label "baseline INT8"
    uv run python scripts/reindex_benchmark.py --label "6GB + fine buckets"
"""

import argparse
import csv
import json
import logging
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
env_file = Path.home() / ".zettelkasten" / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

# Project imports (after env loaded)
from znote_mcp.config import config
from znote_mcp.models.db_models import init_db
from znote_mcp.observability import configure_logging
from znote_mcp.services.embedding_service import EmbeddingService
from znote_mcp.services.onnx_providers import (
    OnnxEmbeddingProvider,
    OnnxRerankerProvider,
    _get_peak_rss_mb,
    _get_rss_mb,
)
from znote_mcp.services.zettel_service import ZettelService

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
BENCHMARKS_DIR = PROJECT_ROOT / "benchmarks" / "results"
SUMMARY_CSV = PROJECT_ROOT / "benchmarks" / "summary.csv"

# ---------------------------------------------------------------------------
# Bucket-stats log collector
# ---------------------------------------------------------------------------
_BUCKET_RE = re.compile(
    r"adaptive bucket ([≤>]\d+): (\d+) texts, "
    r"batch_size=(\d+), mem≈([\d.]+)GB/batch"
)
_PHASE2_RE = re.compile(
    r"Phase 2 complete: (\d+) notes, (\d+) chunks in (\d+)s"
)


class BucketStatsHandler(logging.Handler):
    """Captures per-bucket stats from adaptive batching log lines."""

    def __init__(self):
        super().__init__()
        self.buckets = []
        self.phase2_summary = None

    def emit(self, record):
        msg = record.getMessage()
        m = _BUCKET_RE.search(msg)
        if m:
            self.buckets.append({
                "bucket": m.group(1),
                "count": int(m.group(2)),
                "batch_size": int(m.group(3)),
                "mem_gb_per_batch": float(m.group(4)),
            })
        m2 = _PHASE2_RE.search(msg)
        if m2:
            self.phase2_summary = {
                "notes": int(m2.group(1)),
                "chunks": int(m2.group(2)),
                "seconds": int(m2.group(3)),
            }


# ---------------------------------------------------------------------------
# System info
# ---------------------------------------------------------------------------
def get_system_info() -> dict:
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor() or "unknown",
        "cpu_count": os.cpu_count(),
    }
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
        info["ram_gb"] = round(page_size * page_count / 1024**3, 1)
    except (ValueError, OSError):
        info["ram_gb"] = None
    return info


def get_git_info() -> dict:
    info = {}
    try:
        info["commit"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT, text=True, timeout=5,
        ).strip()
        info["branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=PROJECT_ROOT, text=True, timeout=5,
        ).strip()
        info["dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT, text=True, timeout=5,
        ).strip())
    except Exception:
        pass
    return info


def get_config_snapshot() -> dict:
    return {
        "embedding_model": config.embedding_model,
        "onnx_quantized": config.onnx_quantized,
        "embedding_max_tokens": config.embedding_max_tokens,
        "embedding_batch_size": config.embedding_batch_size,
        "embedding_memory_budget_gb": config.embedding_memory_budget_gb,
        "embedding_chunk_size": config.embedding_chunk_size,
        "embedding_chunk_overlap": config.embedding_chunk_overlap,
        "onnx_providers": config.onnx_providers,
        "notes_dir": str(config.notes_dir),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Reindex embedding benchmark")
    parser.add_argument(
        "--label", "-l", default="",
        help="Human-readable label for this run (e.g. 'baseline INT8')",
    )
    parser.add_argument(
        "--notes-dir", default=None,
        help="Override notes directory (e.g. for sample testing)",
    )
    args = parser.parse_args()

    config.embeddings_enabled = True
    if args.notes_dir:
        config.notes_dir = Path(args.notes_dir)
        logger_init = logging.getLogger(__name__)
        logger_init.info(f"Notes dir override: {config.notes_dir}")

    # Logging — file + console
    log_dir = configure_logging(level=logging.DEBUG, console=True)
    logger = logging.getLogger("znote_mcp.reindex_benchmark")
    logger.setLevel(logging.DEBUG)

    # Attach bucket stats collector to the onnx_providers logger
    bucket_handler = BucketStatsHandler()
    bucket_handler.setLevel(logging.INFO)
    logging.getLogger("znote_mcp.services.onnx_providers").addHandler(bucket_handler)
    logging.getLogger("znote_mcp.services.zettel_service").addHandler(bucket_handler)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    logger.info("=" * 60)
    logger.info(f"REINDEX BENCHMARK — {timestamp}")
    if args.label:
        logger.info(f"Label: {args.label}")
    logger.info("=" * 60)

    conf = get_config_snapshot()
    for k, v in conf.items():
        logger.info(f"Config: {k}={v}")

    # Init DB (in-memory)
    engine = init_db(in_memory=True)

    # Create providers
    onnx_file = (
        "onnx/model_quantized.onnx" if config.onnx_quantized else "onnx/model.onnx"
    )
    logger.info(f"ONNX file: {onnx_file}")

    embedder = OnnxEmbeddingProvider(
        model_id=config.embedding_model,
        onnx_filename=onnx_file,
        max_length=config.embedding_max_tokens,
        cache_dir=config.embedding_model_cache_dir,
        providers=config.onnx_providers,
    )
    reranker = OnnxRerankerProvider(
        model_id=config.reranker_model,
        max_length=config.embedding_max_tokens,
        cache_dir=config.embedding_model_cache_dir,
        providers=config.onnx_providers,
    )
    embedding_service = EmbeddingService(
        embedder=embedder,
        reranker=reranker,
        reranker_idle_timeout=config.reranker_idle_timeout,
    )
    service = ZettelService(embedding_service=embedding_service, engine=engine)

    # Phase 0: Rebuild index from markdown
    rss_before = _get_rss_mb()
    logger.info("Rebuilding index from markdown files...")
    t0_rebuild = time.perf_counter()
    service.rebuild_index()
    rebuild_elapsed = time.perf_counter() - t0_rebuild

    all_notes = service.repository.get_all()
    logger.info(
        f"Index rebuilt: {len(all_notes)} notes in {rebuild_elapsed:.1f}s"
    )

    # Phase 1+2: Reindex embeddings
    logger.info("Starting embedding reindex...")
    t0 = time.perf_counter()

    try:
        stats = service.reindex_embeddings()
        elapsed = time.perf_counter() - t0
        peak_rss = _get_peak_rss_mb()
        rss_after = _get_rss_mb()

        logger.info("=" * 60)
        logger.info("REINDEX COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total time: {elapsed:.1f}s")
        logger.info(
            f"Notes: {stats['total']} total, {stats['embedded']} embedded, "
            f"{stats['skipped']} skipped, {stats['failed']} failed"
        )
        logger.info(f"Chunks: {stats['chunks']}")
        if stats["embedded"] > 0:
            logger.info(
                f"Throughput: {stats['embedded'] / elapsed:.1f} notes/s, "
                f"{stats['chunks'] / elapsed:.1f} chunks/s"
            )
        logger.info(f"Peak RSS: {peak_rss:.0f}MB")

        # ----- Build result dict -----
        result = {
            "timestamp": timestamp,
            "label": args.label or None,
            "git": get_git_info(),
            "system": get_system_info(),
            "config": conf,
            "timing": {
                "rebuild_index_s": round(rebuild_elapsed, 2),
                "reindex_total_s": round(elapsed, 2),
            },
            "stats": {
                "total_notes": stats["total"],
                "embedded": stats["embedded"],
                "skipped": stats["skipped"],
                "failed": stats["failed"],
                "chunks": stats["chunks"],
            },
            "memory": {
                "rss_before_mb": round(rss_before, 0),
                "rss_after_mb": round(rss_after, 0),
                "peak_rss_mb": round(peak_rss, 0),
            },
            "throughput": {
                "notes_per_sec": round(stats["embedded"] / elapsed, 2)
                if elapsed > 0
                else 0,
                "chunks_per_sec": round(stats["chunks"] / elapsed, 2)
                if elapsed > 0
                else 0,
            },
            "buckets": bucket_handler.buckets,
            "phase2": bucket_handler.phase2_summary,
            "log_dir": str(log_dir) if log_dir else None,
        }

        # ----- Save JSON -----
        BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
        json_path = BENCHMARKS_DIR / f"{timestamp}.json"
        json_path.write_text(json.dumps(result, indent=2) + "\n")
        logger.info(f"Results saved to {json_path}")

        # ----- Append CSV summary -----
        csv_exists = SUMMARY_CSV.exists()
        fieldnames = [
            "timestamp",
            "label",
            "git_commit",
            "quantized",
            "memory_budget_gb",
            "total_notes",
            "chunks",
            "failed",
            "reindex_s",
            "notes_per_sec",
            "chunks_per_sec",
            "peak_rss_mb",
        ]
        with open(SUMMARY_CSV, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not csv_exists:
                writer.writeheader()
            writer.writerow({
                "timestamp": timestamp,
                "label": args.label or "",
                "git_commit": result["git"].get("commit", ""),
                "quantized": conf["onnx_quantized"],
                "memory_budget_gb": conf["embedding_memory_budget_gb"],
                "total_notes": stats["total"],
                "chunks": stats["chunks"],
                "failed": stats["failed"],
                "reindex_s": round(elapsed, 1),
                "notes_per_sec": round(stats["embedded"] / elapsed, 2)
                if elapsed > 0
                else 0,
                "chunks_per_sec": round(stats["chunks"] / elapsed, 2)
                if elapsed > 0
                else 0,
                "peak_rss_mb": round(peak_rss, 0),
            })
        logger.info(f"Summary appended to {SUMMARY_CSV}")

        # ----- Console summary -----
        print(f"\n{'='*60}")
        print(f"RESULTS: {stats}")
        print(f"Time: {elapsed:.1f}s  |  Peak RSS: {peak_rss:.0f}MB")
        print(f"Throughput: {stats['embedded']/elapsed:.1f} notes/s, "
              f"{stats['chunks']/elapsed:.1f} chunks/s")
        if bucket_handler.buckets:
            print(f"\nPer-bucket breakdown:")
            for b in bucket_handler.buckets:
                print(f"  {b['bucket']:>6}: {b['count']:>4} texts, "
                      f"batch={b['batch_size']:>2}, "
                      f"mem≈{b['mem_gb_per_batch']:.2f}GB")
        print(f"\nJSON: {json_path}")
        print(f"CSV:  {SUMMARY_CSV}")
        print(f"{'='*60}")

    except Exception as e:
        logger.error(f"Reindex failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        embedding_service.shutdown()


if __name__ == "__main__":
    main()
