#!/bin/bash
# Full CPU+GPU embedding benchmark suite
# Runs all 12 FP32 configs on the full note corpus, then quality eval.
#
# Usage:
#   ./scripts/run_full_benchmark.sh
#   NOTES_DIR=/path/to/notes ./scripts/run_full_benchmark.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

NOTES_DIR="${NOTES_DIR:-$HOME/.zettelkasten/notes}"
CPU_OUT="benchmarks/matrix-cpu"
GPU_OUT="benchmarks/matrix-gpu"
TIMESTAMP=$(date -u +"%Y%m%dT%H%M%SZ")
LOGFILE="benchmarks/benchmark_${TIMESTAMP}.log"

mkdir -p benchmarks

echo "=== Full Embedding Benchmark Suite ===" | tee "$LOGFILE"
echo "Started: $(date -u)" | tee -a "$LOGFILE"
echo "Notes dir: $NOTES_DIR" | tee -a "$LOGFILE"
echo "Note count: $(ls "$NOTES_DIR"/*.md 2>/dev/null | wc -l)" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"

# ---- CPU benchmark ----
echo "=== Phase 1: CPU Benchmark (12 configs) ===" | tee -a "$LOGFILE"
echo "Output: $CPU_OUT" | tee -a "$LOGFILE"
echo "Started: $(date -u)" | tee -a "$LOGFILE"

uv run python scripts/benchmark_embed.py \
    --models all \
    --notes-dir "$NOTES_DIR" \
    --output-dir "$CPU_OUT" \
    --memory-budget-gb 12.0 \
    -v \
    2>&1 | tee -a "$LOGFILE"

echo "" | tee -a "$LOGFILE"
echo "CPU benchmark finished: $(date -u)" | tee -a "$LOGFILE"

# ---- GPU benchmark ----
echo "" | tee -a "$LOGFILE"
echo "=== Phase 2: GPU Benchmark (12 configs) ===" | tee -a "$LOGFILE"
echo "Output: $GPU_OUT" | tee -a "$LOGFILE"
echo "Started: $(date -u)" | tee -a "$LOGFILE"

uv run python scripts/benchmark_embed.py \
    --models all \
    --notes-dir "$NOTES_DIR" \
    --output-dir "$GPU_OUT" \
    --memory-budget-gb 12.0 \
    --device gpu \
    -v \
    2>&1 | tee -a "$LOGFILE"

echo "" | tee -a "$LOGFILE"
echo "GPU benchmark finished: $(date -u)" | tee -a "$LOGFILE"

# ---- Quality evaluation ----
echo "" | tee -a "$LOGFILE"
echo "=== Phase 3: Quality Evaluation ===" | tee -a "$LOGFILE"

echo "--- CPU quality ---" | tee -a "$LOGFILE"
uv run python scripts/benchmark_quality.py \
    --matrix-dir "$CPU_OUT" \
    -v \
    2>&1 | tee -a "$LOGFILE"

echo "" | tee -a "$LOGFILE"
echo "--- GPU quality ---" | tee -a "$LOGFILE"
uv run python scripts/benchmark_quality.py \
    --matrix-dir "$GPU_OUT" \
    -v \
    2>&1 | tee -a "$LOGFILE"

echo "" | tee -a "$LOGFILE"
echo "=== All done ===" | tee -a "$LOGFILE"
echo "Finished: $(date -u)" | tee -a "$LOGFILE"
echo "Master log: $LOGFILE" | tee -a "$LOGFILE"
echo "CPU results: $CPU_OUT/perf_matrix.json, $CPU_OUT/quality_matrix.json" | tee -a "$LOGFILE"
echo "GPU results: $GPU_OUT/perf_matrix.json, $GPU_OUT/quality_matrix.json" | tee -a "$LOGFILE"
