#!/usr/bin/env python
"""Create a representative ~200 note sample for fast benchmarking.

Stratified sampling: measures each note's file size, bins into quartiles,
and samples proportionally so the sample matches the full corpus distribution.

Usage:
    cd /home/komi/repos/MCP/znote-mcp
    uv run python scripts/create_sample.py
    uv run python scripts/create_sample.py --count 200 --output /tmp/sample_notes
"""

import argparse
import random
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Create stratified note sample")
    parser.add_argument(
        "--source", default=str(Path.home() / ".zettelkasten" / "notes"),
        help="Source notes directory",
    )
    parser.add_argument(
        "--output", default=str(Path.home() / ".zettelkasten" / "sample_notes"),
        help="Output directory for sampled notes",
    )
    parser.add_argument("--count", type=int, default=200, help="Target sample size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    source = Path(args.source)
    output = Path(args.output)

    # Collect all markdown files with sizes
    notes = []
    for f in source.glob("*.md"):
        notes.append((f, f.stat().st_size))

    if not notes:
        print(f"No .md files found in {source}")
        return

    notes.sort(key=lambda x: x[1])
    total = len(notes)
    target = min(args.count, total)

    print(f"Source: {source} ({total} notes)")
    print(f"Target sample: {target} notes")

    # Stratified sampling by size quartile
    n_bins = 5
    bin_size = total // n_bins
    sampled = []

    for i in range(n_bins):
        start = i * bin_size
        end = start + bin_size if i < n_bins - 1 else total
        bin_notes = notes[start:end]
        # Proportional allocation
        bin_target = round(target * len(bin_notes) / total)
        bin_target = min(bin_target, len(bin_notes))
        chosen = random.sample(bin_notes, bin_target)
        sampled.extend(chosen)

        sizes = [s for _, s in bin_notes]
        chosen_sizes = [s for _, s in chosen]
        print(
            f"  Bin {i+1}/{n_bins}: {len(bin_notes)} notes "
            f"({min(sizes)}-{max(sizes)} bytes), "
            f"sampled {len(chosen)}"
        )

    # Adjust if we're slightly off target
    remaining_pool = [n for n in notes if n not in sampled]
    while len(sampled) < target and remaining_pool:
        extra = random.choice(remaining_pool)
        sampled.append(extra)
        remaining_pool.remove(extra)

    # Copy to output
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    for f, _ in sampled:
        shutil.copy2(f, output / f.name)

    # Stats
    sizes = sorted(s for _, s in sampled)
    full_sizes = sorted(s for _, s in notes)
    print(f"\nSample created: {len(sampled)} notes in {output}")
    print(f"  Full corpus:  min={min(full_sizes)}, "
          f"median={full_sizes[len(full_sizes)//2]}, "
          f"max={max(full_sizes)}, "
          f"total={sum(full_sizes)/1024:.0f}KB")
    print(f"  Sample:       min={min(sizes)}, "
          f"median={sizes[len(sizes)//2]}, "
          f"max={max(sizes)}, "
          f"total={sum(sizes)/1024:.0f}KB")


if __name__ == "__main__":
    main()
