#!/usr/bin/env python
"""Reranker smoke tests: download, load, score, validate each model.

Validates that each reranker model in the registry can be downloaded,
loaded, and produces sensible relevance scores. Runs sequentially —
one model at a time, load → test → unload.

Usage:
    cd /home/komi/repos/MCP/znote-mcp

    # Test all rerankers (CPU)
    uv run python scripts/smoke_test_rerankers.py

    # Test specific rerankers
    uv run python scripts/smoke_test_rerankers.py --models gte-reranker,bge-reranker-base

    # Verbose logging
    uv run python scripts/smoke_test_rerankers.py -v
"""

from __future__ import annotations

import argparse
import datetime
import gc
import json
import logging
import math
import os
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is on sys.path for script imports
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from reranker_configs import RERANKERS, list_configs
from znote_mcp.services.onnx_providers import (
    OnnxRerankerProvider,
    _get_rss_mb,
)

logger = logging.getLogger("smoke_test_rerankers")


# ---------------------------------------------------------------------------
# Smoke test cases — diverse topics with clear relevance ordering
# ---------------------------------------------------------------------------

SMOKE_CASES = [
    # -----------------------------------------------------------------
    # 1. BASIC RELEVANCE — obvious topic match vs total mismatch
    # -----------------------------------------------------------------
    {
        "tag": "basic-relevance",
        "query": "How to configure GPU passthrough in a virtual machine",
        "docs": [
            "GPU passthrough allows a VM to directly access a physical GPU "
            "using IOMMU technology. This enables near-native GPU performance "
            "for workloads like gaming, machine learning, and video rendering "
            "inside virtual machines.",
            "The best chocolate cake recipe uses Dutch process cocoa powder, "
            "buttermilk, and a touch of espresso to deepen the chocolate "
            "flavor. Bake at 350F for 30 minutes.",
            "VFIO and IOMMU groups are essential for PCIe passthrough on "
            "Linux. You need to enable IOMMU in BIOS, add kernel parameters "
            "(intel_iommu=on or amd_iommu=on), and bind the GPU to the "
            "vfio-pci driver before starting the VM.",
        ],
        # Both doc 0 and 2 are relevant; models may rank either on top.
        "expected_top": None,
        "expected_bottom": 1,  # Cake recipe
    },

    # -----------------------------------------------------------------
    # 2. TWO RELEVANT + ONE IRRELEVANT — discriminate topic boundaries
    # -----------------------------------------------------------------
    {
        "tag": "topic-boundary",
        "query": "Python async programming with asyncio event loop",
        "docs": [
            "Pandas DataFrames provide powerful tabular data structures for "
            "data analysis. Use df.groupby() for aggregation, df.merge() for "
            "joins, and df.pivot_table() for reshaping data.",
            "The asyncio module in Python provides infrastructure for writing "
            "single-threaded concurrent code using coroutines, multiplexing "
            "I/O access over sockets, and running network clients and servers. "
            "Use async/await syntax with asyncio.run() as the entry point.",
            "SQLAlchemy's async extension allows database operations within "
            "an asyncio event loop. Use create_async_engine() and "
            "AsyncSession for non-blocking database access in async Python "
            "applications.",
        ],
        "expected_top": None,  # Both doc 1 and 2 are relevant
        "expected_bottom": 0,  # Pandas doc
    },

    # -----------------------------------------------------------------
    # 3. CLEAR WINNER — one doc is a direct answer
    # -----------------------------------------------------------------
    {
        "tag": "clear-winner",
        "query": "Kubernetes pod networking and service discovery",
        "docs": [
            "Each Kubernetes pod gets its own IP address. Pods communicate "
            "with each other using these IPs directly, while Services provide "
            "stable endpoints via ClusterIP, NodePort, or LoadBalancer types. "
            "CoreDNS handles service discovery within the cluster.",
            "React hooks like useState and useEffect allow functional "
            "components to manage state and side effects. Custom hooks "
            "extract reusable stateful logic across components.",
            "Container networking uses network namespaces and virtual "
            "ethernet pairs (veth) to provide isolated network stacks. CNI "
            "plugins like Calico and Flannel implement the networking model "
            "for container orchestrators.",
        ],
        "expected_top": 0,  # K8s doc is the direct answer
        "expected_bottom": 1,  # React doc
    },

    # -----------------------------------------------------------------
    # 4. LEXICAL OVERLAP TRAP — high keyword overlap but wrong topic
    # The distractor doc shares many words with the query (memory, leak,
    # application, process) but is about plumbing, not software.
    # -----------------------------------------------------------------
    {
        "tag": "lexical-trap",
        "query": "How to debug memory leaks in a Node.js application",
        "docs": [
            "Use Chrome DevTools to take heap snapshots and compare memory "
            "allocations over time. The --inspect flag enables remote "
            "debugging. Look for detached DOM nodes and growing closures "
            "as common causes of memory leaks in Node.js.",
            "Water leaks in residential plumbing can cause significant "
            "memory loss in building materials. The application of leak "
            "detection processes involves pressurizing the system and "
            "monitoring for drops. Memory foam insulation around pipes "
            "prevents condensation leaks.",
            "Garbage collection in V8 uses a generational approach with "
            "young and old spaces. Objects that survive multiple GC cycles "
            "are promoted to old space, which is collected less frequently.",
        ],
        "expected_top": 0,  # DevTools/heap snapshots — direct answer
        "expected_bottom": None,  # Plumbing (1) vs GC internals (2) — both partially relevant
    },

    # -----------------------------------------------------------------
    # 5. PARAPHRASE DETECTION — relevant doc uses different vocabulary
    # No shared keywords between query and the best doc.
    # -----------------------------------------------------------------
    {
        "tag": "paraphrase",
        "query": "techniques for reducing neural network model size for deployment",
        "docs": [
            "Data augmentation strategies such as random cropping, flipping, "
            "and color jittering expand the effective training dataset size "
            "without collecting new data. Mixup and CutMix further improve "
            "generalization by blending training examples.",
            "Quantization converts floating-point weights to lower-precision "
            "integers (INT8 or INT4), reducing file size by 2-4x. Knowledge "
            "distillation trains a compact student network to mimic a larger "
            "teacher's outputs. Structured pruning removes entire channels "
            "or attention heads deemed least important.",
            "The ONNX format enables interoperability between ML frameworks. "
            "Models trained in PyTorch can be exported to ONNX and served "
            "with ONNX Runtime for optimized inference on CPU and GPU.",
        ],
        # All three docs are ML-adjacent; some models score all negatively
        # and struggle to separate them.  Quality signal, not smoke gate.
        "expected_top": None,  # Quantization (1) should win but scores are close
        "expected_bottom": None,  # Data augmentation (0) vs ONNX (2) — both weak matches
    },

    # -----------------------------------------------------------------
    # 6. NEGATION SENSITIVITY — doc discusses topic but negates it
    # A naive keyword matcher would rank the negation doc highly.
    # -----------------------------------------------------------------
    {
        "tag": "negation",
        "query": "benefits of using microservices architecture",
        "docs": [
            "Microservices architecture allows teams to deploy services "
            "independently, scale individual components based on demand, "
            "and use different technology stacks per service. This enables "
            "faster release cycles and better fault isolation.",
            "Microservices architecture is often not beneficial for small "
            "teams. The operational overhead of managing dozens of services, "
            "distributed tracing, service mesh configuration, and eventual "
            "consistency problems frequently outweigh any gains. Most "
            "startups would be better served by a well-structured monolith.",
            "GraphQL provides a flexible query language for APIs that lets "
            "clients request exactly the data they need. It reduces "
            "over-fetching compared to REST endpoints.",
        ],
        # Negation is the hardest semantic challenge; most rerankers treat
        # both microservices docs as relevant.  Only assert bottom (GraphQL).
        "expected_top": None,  # Positive (0) vs critique (1) — hard for rerankers
        "expected_bottom": 2,  # GraphQL is off-topic
    },

    # -----------------------------------------------------------------
    # 7. SPECIFICITY — general overview vs precise answer
    # Query asks about a specific technique; one doc is broad, one is exact.
    # -----------------------------------------------------------------
    {
        "tag": "specificity",
        "query": "how does WAL mode work in SQLite",
        "docs": [
            "SQLite is a lightweight, serverless database engine used in "
            "mobile apps, embedded systems, and as an application file format. "
            "It supports ACID transactions and is the most widely deployed "
            "database in the world.",
            "In WAL (Write-Ahead Logging) mode, SQLite writes changes to a "
            "separate WAL file instead of directly modifying the database. "
            "Readers see a consistent snapshot while writers append to the "
            "WAL. Checkpointing merges WAL changes back into the main "
            "database file. This allows concurrent reads during writes.",
            "PostgreSQL uses MVCC (Multi-Version Concurrency Control) and "
            "WAL for crash recovery. The WAL in PostgreSQL streams to "
            "replicas for high availability setups using pg_basebackup.",
        ],
        "expected_top": 1,  # Specific SQLite WAL explanation
        "expected_bottom": None,  # Don't assert — doc 0 (general SQLite) and doc 2
                                  # (PostgreSQL WAL) both have partial relevance
    },

    # -----------------------------------------------------------------
    # 8. CROSS-DOMAIN HOMONYM — same word, different meaning
    # "Python" as programming language vs the snake.
    # -----------------------------------------------------------------
    {
        "tag": "homonym",
        "query": "Python type hints and static type checking with mypy",
        "docs": [
            "Python type hints (PEP 484) annotate function signatures with "
            "expected types. Use mypy or pyright for static analysis. Common "
            "patterns include Optional[T], Union[A, B], and generic "
            "containers like List[int] and Dict[str, Any].",
            "The Burmese python (Python bivittatus) is one of the largest "
            "snake species in the world, reaching lengths of 23 feet. "
            "These constrictors are found in Southeast Asian tropical "
            "forests and grasslands. Their population status is assessed "
            "as vulnerable by the IUCN.",
            "Static analysis tools examine source code without executing "
            "it. Linters like ESLint for JavaScript and Pylint for Python "
            "catch common errors. Type checkers go further by verifying "
            "type correctness across function boundaries.",
        ],
        "expected_top": 0,  # PEP 484 / mypy doc
        "expected_bottom": 1,  # Snake doc
    },

    # -----------------------------------------------------------------
    # 9. TEMPORAL SENSITIVITY — outdated vs current information
    # Query implies current best practice; one doc is clearly dated.
    # -----------------------------------------------------------------
    {
        "tag": "temporal",
        "query": "current best practices for React state management",
        "docs": [
            "React state management in 2024 favors built-in hooks: "
            "useState for local state, useReducer for complex logic, and "
            "useContext + useSyncExternalStore for shared state. Server "
            "Components reduce client-side state needs. Libraries like "
            "Zustand and Jotai offer lightweight alternatives to Redux.",
            "Managing state in React class components requires calling "
            "this.setState() with an updater object. Redux with "
            "mapStateToProps and connect() HOC is the standard pattern "
            "for global state. Use componentDidMount for data fetching "
            "and componentWillUnmount for cleanup.",
            "Vue.js uses a reactive data model where changes to component "
            "data automatically trigger DOM updates. The Composition API "
            "provides ref() and reactive() for fine-grained reactivity.",
        ],
        # Both React docs (0 and 1) are relevant; models may not
        # distinguish modern vs legacy.  Only assert Vue (2) is bottom.
        "expected_top": None,  # Modern (0) vs class-based (1) — temporal nuance
        "expected_bottom": 2,  # Vue.js is wrong framework entirely
    },

    # -----------------------------------------------------------------
    # 10. CONCEPTUAL SIMILARITY — related but distinct concepts
    # Query is about one concept; distractor is a closely related but
    # different concept from the same domain.
    # -----------------------------------------------------------------
    {
        "tag": "conceptual-near-miss",
        "query": "difference between authentication and authorization",
        "docs": [
            "Authentication verifies who a user is (identity), while "
            "authorization determines what they can access (permissions). "
            "Authentication happens first — you prove your identity with "
            "credentials. Authorization follows — the system checks your "
            "roles and permissions against the requested resource.",
            "OAuth 2.0 is an authorization framework that enables "
            "third-party applications to obtain limited access to HTTP "
            "services. It uses access tokens, refresh tokens, and scopes "
            "to delegate permissions without sharing credentials.",
            "Symmetric encryption (AES) uses the same key for encryption "
            "and decryption, while asymmetric encryption (RSA) uses a "
            "public-private key pair. TLS handshakes combine both: "
            "asymmetric for key exchange, symmetric for data transfer.",
        ],
        "expected_top": 0,  # Directly explains the difference
        "expected_bottom": 2,  # Encryption is a different security topic
    },

    # -----------------------------------------------------------------
    # 11. SHORT QUERY — single-word or very terse query
    # Tests model behavior with minimal query context.
    # -----------------------------------------------------------------
    {
        "tag": "short-query",
        "query": "mutex",
        "docs": [
            "A mutex (mutual exclusion) is a synchronization primitive that "
            "prevents multiple threads from simultaneously accessing a shared "
            "resource. Only the thread that locks the mutex can unlock it. "
            "Unlike semaphores, mutexes enforce ownership semantics.",
            "The Rust programming language enforces memory safety through "
            "its ownership system: each value has exactly one owner, "
            "references are checked at compile time, and dangling pointers "
            "are impossible. The borrow checker prevents data races.",
            "In music production, mixing involves adjusting levels, panning, "
            "EQ, and effects across multiple audio tracks to create a "
            "balanced stereo output. Mastering is the final step before "
            "distribution.",
        ],
        "expected_top": 0,  # Mutex definition
        "expected_bottom": None,  # Rust ownership (1) vs music (2) — both far from "mutex"
    },

    # -----------------------------------------------------------------
    # 12. LONG QUERY — verbose, multi-sentence query
    # Tests whether models handle context-rich queries well.
    # -----------------------------------------------------------------
    {
        "tag": "long-query",
        "query": (
            "I'm building a web application that needs to handle file "
            "uploads from users. The files can be large (up to 2GB) and "
            "I need to show upload progress to the user. I'm using React "
            "on the frontend and Express.js on the backend. What's the "
            "best approach for implementing resumable uploads with "
            "progress tracking?"
        ),
        "docs": [
            "The tus protocol is an open standard for resumable file "
            "uploads. It breaks large files into chunks and tracks upload "
            "progress server-side. If a connection drops, the client "
            "queries the server for the last received offset and resumes "
            "from there. Libraries like tus-js-client (React) and "
            "tus-node-server (Express) implement the protocol.",
            "React's useCallback hook memoizes function references to "
            "prevent unnecessary re-renders in child components. Use it "
            "when passing callbacks to optimized children wrapped in "
            "React.memo(). Don't overuse it — the memoization itself "
            "has a cost.",
            "Express.js middleware functions execute during the "
            "request-response cycle. Use app.use() for global middleware "
            "and router.use() for route-specific middleware. Common "
            "middleware includes body-parser, cors, and helmet for "
            "security headers.",
        ],
        "expected_top": 0,  # tus protocol — directly answers the question
        "expected_bottom": 1,  # useCallback — wrong topic despite React mention
    },
]


# ---------------------------------------------------------------------------
# Fallback ONNX filenames for ms-marco models
# ---------------------------------------------------------------------------

_MS_MARCO_FALLBACKS = [
    "onnx/model_qint8_avx512_vnni.onnx",
    "onnx/model_qint8.onnx",
]


# ---------------------------------------------------------------------------
# System info
# ---------------------------------------------------------------------------

def _collect_system_info() -> Dict[str, Any]:
    """Collect system metadata for reproducibility."""
    import onnxruntime as ort

    info: Dict[str, Any] = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu": platform.processor() or "unknown",
        "onnxruntime_version": ort.__version__,
        "ort_available_providers": ort.get_available_providers(),
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
# Smoke test runner
# ---------------------------------------------------------------------------

def _validate_scores(
    scores: List[float],
    expected_top: Optional[int],
    expected_bottom: Optional[int],
) -> Dict[str, Any]:
    """Validate reranker scores for a single test case.

    Returns dict with:
      - 'finite': all scores are finite numbers
      - 'varying': scores are not all identical
      - 'order_top': top-scored doc matches expected (True if not asserted)
      - 'order_bottom': bottom-scored doc matches expected (True if not asserted)
      - 'actual_top': index of highest-scored doc
      - 'actual_bottom': index of lowest-scored doc
    """
    finite = all(math.isfinite(s) for s in scores)
    varying = len(set(f"{s:.6f}" for s in scores)) > 1
    top_idx = max(range(len(scores)), key=lambda i: scores[i])
    bottom_idx = min(range(len(scores)), key=lambda i: scores[i])
    order_top = (expected_top is None) or (top_idx == expected_top)
    order_bottom = (expected_bottom is None) or (bottom_idx == expected_bottom)

    return {
        "finite": finite,
        "varying": varying,
        "order_top": order_top,
        "order_bottom": order_bottom,
        "actual_top": top_idx,
        "actual_bottom": bottom_idx,
    }


def _run_smoke_test(
    config_key: str,
    config: Dict[str, Any],
    providers: str = "cpu",
) -> Dict[str, Any]:
    """Run smoke test for a single reranker config.

    Returns result dict with status, metrics, and per-case details.
    """
    result: Dict[str, Any] = {
        "config_key": config_key,
        "model_id": config["model_id"],
        "status": "fail",
        "error": None,
        "load_time_s": None,
        "rss_delta_mb": None,
        "cases": [],
        "scores_ok": False,
        "order_ok": False,
    }

    onnx_filename = config["onnx_filename"]
    extra_files = config.get("extra_files", [])
    max_length = config["max_tokens"]

    # Try loading with configured filename, then fallbacks for ms-marco
    filenames_to_try = [onnx_filename]
    if "ms-marco" in config["model_id"]:
        filenames_to_try.extend(_MS_MARCO_FALLBACKS)

    provider = None
    loaded = False

    for fname in filenames_to_try:
        try:
            provider = OnnxRerankerProvider(
                model_id=config["model_id"],
                onnx_filename=fname,
                max_length=max_length,
                providers=providers,
                extra_files=extra_files if fname == onnx_filename else [],
            )

            rss_before = _get_rss_mb()
            t0 = time.perf_counter()
            provider.load()
            load_time = time.perf_counter() - t0
            rss_after = _get_rss_mb()

            result["load_time_s"] = round(load_time, 2)
            result["rss_delta_mb"] = round(rss_after - rss_before, 1)
            if fname != onnx_filename:
                result["onnx_filename_used"] = fname
                logger.info(
                    f"  [{config_key}] Fell back to {fname}"
                )
            loaded = True
            break

        except FileNotFoundError as e:
            logger.debug(f"  [{config_key}] {fname}: not found, trying next")
            result["error"] = str(e)
            continue
        except Exception as e:
            result["error"] = f"{type(e).__name__}: {e}"
            logger.error(f"  [{config_key}] Load failed: {result['error']}")
            return result

    if not loaded:
        result["error"] = (
            f"No ONNX file found. Tried: {filenames_to_try}. "
            f"May need export via: optimum-cli export onnx "
            f"--model {config['model_id']} --task text-classification ./onnx/"
        )
        logger.warning(f"  [{config_key}] {result['error']}")
        return result

    # Score test cases
    try:
        all_scores_ok = True
        all_order_ok = True

        for i, case in enumerate(SMOKE_CASES):
            ranked = provider.rerank(
                case["query"], case["docs"], top_k=len(case["docs"])
            )
            # Extract scores in original document order
            scores = [0.0] * len(case["docs"])
            for idx, score in ranked:
                scores[idx] = score

            validation = _validate_scores(
                scores, case["expected_top"], case["expected_bottom"]
            )

            tag = case.get("tag", f"case-{i+1}")
            case_result = {
                "tag": tag,
                "query": case["query"][:60] + ("..." if len(case["query"]) > 60 else ""),
                "scores": [round(s, 6) for s in scores],
                "ranked_order": [idx for idx, _ in ranked],
                **validation,
            }
            result["cases"].append(case_result)

            if not (validation["finite"] and validation["varying"]):
                all_scores_ok = False
            if not validation["order_top"]:
                all_order_ok = False

            logger.info(
                f"  [{config_key}] {tag}: "
                f"scores={[f'{s:.4f}' for s in scores]}, "
                f"top={'ok' if validation['order_top'] else 'WRONG'}, "
                f"bottom={'ok' if validation['order_bottom'] else 'WRONG'}"
            )

        result["scores_ok"] = all_scores_ok
        result["order_ok"] = all_order_ok
        result["status"] = "ok" if (all_scores_ok and all_order_ok) else "degraded"

    except Exception as e:
        result["error"] = f"Scoring failed: {type(e).__name__}: {e}"
        logger.error(f"  [{config_key}] {result['error']}")
        logger.debug(traceback.format_exc())

    finally:
        if provider is not None and provider.is_loaded:
            provider.unload()
        gc.collect()

    return result


# ---------------------------------------------------------------------------
# Summary formatting
# ---------------------------------------------------------------------------

def _print_summary(results: List[Dict[str, Any]]) -> None:
    """Print aligned summary table."""
    print()
    print("=" * 80)
    print("RERANKER SMOKE TEST SUMMARY")
    print("=" * 80)

    header = (
        f"{'Config':<26s} {'Status':<10s} {'Load(s)':>8s} "
        f"{'RSS(MB)':>9s} {'Scores':>8s} {'Order':>8s}"
    )
    print(header)
    print("-" * 80)

    for r in results:
        status = r["status"].upper() if r["status"] == "fail" else r["status"]
        load_s = f"{r['load_time_s']:.1f}" if r["load_time_s"] is not None else "-"
        rss = f"{r['rss_delta_mb']:.0f}" if r["rss_delta_mb"] is not None else "-"
        scores = "yes" if r["scores_ok"] else ("no" if r["cases"] else "-")
        order = "yes" if r["order_ok"] else ("no" if r["cases"] else "-")

        line = (
            f"{r['config_key']:<26s} {status:<10s} {load_s:>8s} "
            f"{rss:>9s} {scores:>8s} {order:>8s}"
        )
        print(line)

        if r["error"] and r["status"] == "fail":
            # Wrap long error messages
            err = r["error"]
            if len(err) > 70:
                err = err[:70] + "..."
            print(f"  {'':26s} -> {err}")

    print("=" * 80)

    ok = sum(1 for r in results if r["status"] == "ok")
    degraded = sum(1 for r in results if r["status"] == "degraded")
    failed = sum(1 for r in results if r["status"] == "fail")
    print(f"\n{ok} ok, {degraded} degraded, {failed} failed out of {len(results)} models")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke test reranker models: download, load, score, validate."
    )
    parser.add_argument(
        "--models",
        default="all",
        help="Comma-separated config keys, or 'all' (default: all)",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "gpu"],
        default="cpu",
        help="Execution provider (default: cpu)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/reranker-smoke"),
        help="Directory for results (default: benchmarks/reranker-smoke)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )

    args = parser.parse_args()

    # Logging setup
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    # Quiet down noisy loggers
    logging.getLogger("onnxruntime").setLevel(logging.WARNING)

    # Resolve model list
    if args.models == "all":
        config_keys = list_configs()
    else:
        config_keys = [k.strip() for k in args.models.split(",")]
        for k in config_keys:
            if k not in RERANKERS:
                parser.error(f"Unknown reranker config: {k!r}. "
                             f"Available: {list_configs()}")

    providers = "cpu" if args.device == "cpu" else "auto"

    logger.info(
        f"Smoke testing {len(config_keys)} reranker models: "
        f"{', '.join(config_keys)}"
    )
    logger.info(f"Device: {args.device}, providers: {providers}")

    # Collect system info
    sys_info = _collect_system_info()

    # Run smoke tests sequentially
    results: List[Dict[str, Any]] = []
    for config_key in config_keys:
        config = RERANKERS[config_key]
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing: {config_key} ({config['model_id']})")
        logger.info(f"{'='*60}")

        result = _run_smoke_test(config_key, config, providers=providers)
        results.append(result)

        logger.info(f"  -> {result['status'].upper()}")

    # Print summary
    _print_summary(results)

    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "smoke_results.json"
    output_data = {
        "system_info": sys_info,
        "results": results,
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
