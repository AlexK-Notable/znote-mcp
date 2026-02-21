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
# Smoke test cases — derived from actual zettelkasten content
#
# These cases reflect real note content and query patterns from a ~962-note
# engineering-focused zettelkasten spanning: anamnesis (code intelligence),
# znote-mcp (this project), in-memoria (Rust/Node FFI), komi-zone (plugin
# system), hyprtasking (Hyprland WM), VCV Rack, 3D printing (K1C/Klipper),
# and keyboards.  Documents are condensed from actual notes; queries mimic
# real search patterns.
# ---------------------------------------------------------------------------

SMOKE_CASES = [
    # -----------------------------------------------------------------
    # 1. CROSS-PROJECT ARCHITECTURE — embedding system internals
    # Query targets znote-mcp embedding architecture; distractors are
    # from a different project (anamnesis) and a hobby topic.
    # -----------------------------------------------------------------
    {
        "tag": "embedding-architecture",
        "query": "how does the embedding service handle idle timeout and model unloading",
        "docs": [
            "EmbeddingService orchestrates thread-safe embedding and reranking "
            "with lazy loading. The reranker has a configurable idle timeout "
            "(default 600s) — a threading.Timer fires after inactivity and "
            "calls provider.unload() to release memory. The embedder stays "
            "warm-loaded since it's used more frequently. shutdown() cancels "
            "the timer and chains to both providers' unload methods.",
            "The anamnesis LSP subsystem solidlsp (~7,400 LOC) has performance "
            "issues: duplicate filesystem operations from recursive directory "
            "walks, unbounded caches growing monotonically, and O(n) symbol "
            "lookups that should be O(1) with proper indexing.",
            "VCV Rack v2 plugins must be migrated from the v1 API. Key changes "
            "include replacing Widget::step() with process(), using "
            "createModel<>() instead of Model::create(), and updating the "
            "plugin.json manifest format. Install .vcvplugin files via "
            "Library > Install from file.",
        ],
        "expected_top": 0,  # Embedding idle timeout doc
        "expected_bottom": 2,  # VCV Rack is completely unrelated
    },

    # -----------------------------------------------------------------
    # 2. CODE REVIEW vs REMEDIATION — distinguishing similar note types
    # Both doc 0 and 1 are about the same codebase (znote-mcp) but one
    # is a review finding and the other is the fix plan.
    # -----------------------------------------------------------------
    {
        "tag": "review-vs-remediation",
        "query": "SQL injection vulnerability in LIKE pattern handling",
        "docs": [
            "Security review found residual vulnerability: LIKE pattern "
            "escaping is not consistently applied across all search code "
            "paths (CWE-89). The tag search uses parameterized queries "
            "but the content FTS5 search interpolates user input into "
            "MATCH expressions without escaping special characters.",
            "Remediation plan Phase 3 addresses the LIKE injection finding: "
            "add _escape_like() helper to SearchService, apply to all "
            "4 search methods, add regression tests with payloads like "
            "'%; DROP TABLE notes; --' and 'test%_wildcard'. Estimated "
            "effort: 2 hours. Depends on Phase 2 (test infrastructure).",
            "The dual storage system uses markdown files as source of truth "
            "with SQLite as an indexing layer. WAL mode enables concurrent "
            "reads during writes. The database can be fully rebuilt from "
            "markdown files via zk_system(action='rebuild').",
        ],
        # Both doc 0 (finding) and doc 1 (fix plan) are directly relevant
        # to the SQL injection query. Doc 2 (architecture overview) is not.
        "expected_top": None,  # Finding (0) vs fix plan (1) — both relevant
        "expected_bottom": 2,  # General architecture is not about SQL injection
    },

    # -----------------------------------------------------------------
    # 3. CROSS-DOMAIN TERMINOLOGY — "hub" means different things
    # In zettelkasten, "hub" is a note type. In networking, it's hardware.
    # -----------------------------------------------------------------
    {
        "tag": "hub-homonym",
        "query": "hub notes that synthesize cross-domain implementation findings",
        "docs": [
            "Hub notes serve as synthesis points in the zettelkasten, "
            "aggregating findings from multiple specialist reviews into a "
            "coherent whole. A typical hub links to 4-8 phase records, "
            "gate verifications, and code review notes, providing a single "
            "entry point for understanding a complete implementation cycle.",
            "In-Memoria General Codebase Synthesis identifies compound "
            "problems invisible to single-domain analysis. Key insight: "
            "a Safety Net Failure Chain where Rust panic crashes Node.js, "
            "225 'any' types bypass TypeScript checks, missing Zod "
            "validation allows malformed data, and no panic handling in "
            "FFI means unrecoverable crashes.",
            "A network hub is a basic Layer 1 device that broadcasts all "
            "incoming traffic to every connected port. Unlike switches, "
            "hubs cannot learn MAC addresses or create dedicated circuits. "
            "They have been largely replaced by switches in modern LANs.",
        ],
        "expected_top": None,  # Both doc 0 (hub concept) and doc 1 (actual hub) relevant
        "expected_bottom": 2,  # Networking hardware hub
    },

    # -----------------------------------------------------------------
    # 4. SAME PROJECT, DIFFERENT PHASE — distinguishing phases
    # All docs are about anamnesis but at different lifecycle stages.
    # -----------------------------------------------------------------
    {
        "tag": "phase-discrimination",
        "query": "anamnesis test coverage gaps and missing test functions",
        "docs": [
            "Anamnesis test coverage remediation plan: ~85 new test "
            "functions across 4 new files and 5 modified files. Execution "
            "order: Phase 1 (unit tests for helpers), Phase 2 (integration "
            "tests for matchers), Phase 3 (dispatch routing tests), Phase 4 "
            "(cache invalidation edge cases). Each phase has a gate check.",
            "Anamnesis Feb 6 Remediation Phase Sequencing: 33 items across "
            "6 phases. Architecture cleanup (remove phantom deps), code "
            "quality (fix type annotations), performance (cache tuning), "
            "security (input validation), documentation (inline comments), "
            "and testing (coverage gaps). Priority-ordered by risk.",
            "Post-remediation assessment: 183+ lines of synergy edge-case "
            "tests added, -2,400 LOC dead code removed, test health improved "
            "from 'Needs Work' to 'Adequate with Known Gaps'. Remaining "
            "gaps: no property-based tests, limited concurrency testing, "
            "mock theater in 3 integration test files.",
        ],
        # All three are about anamnesis testing/remediation. Doc 0 is the
        # specific test coverage plan; doc 2 is the post-remediation review.
        "expected_top": 0,  # Specific test coverage plan
        "expected_bottom": None,  # Docs 1 and 2 both relate to test gaps
    },

    # -----------------------------------------------------------------
    # 5. RUST FFI vs PYTHON FFI — related concepts, different ecosystems
    # Query is about Rust/Node FFI; distractor is Python C extension.
    # -----------------------------------------------------------------
    {
        "tag": "ffi-ecosystem",
        "query": "Rust panic crashes Node.js process through FFI boundary",
        "docs": [
            "In-Memoria has two critical crash vectors: Rust panic! in "
            "native modules propagates through the napi-rs FFI boundary and "
            "terminates the Node.js process with SIGABRT. The second vector "
            "is a forced process.exit() in src/index.ts error handlers. "
            "Both bypass graceful shutdown. Priority: CRITICAL.",
            "Python C extensions using ctypes or cffi must handle segfaults "
            "carefully. A SIGSEGV in native code kills the Python interpreter. "
            "Use faulthandler module to get tracebacks from crashes. "
            "Consider using Cython for safer memory management at the "
            "Python/C boundary.",
            "The TypeScript FFI type definitions have a subtle bug: "
            "lowercase 'symbol' (JavaScript primitive type) is used where "
            "'Symbol' (the interface) is needed. This causes type errors "
            "in strict mode. Wave 1, P1 High, Low effort fix.",
        ],
        "expected_top": 0,  # Rust/Node panic crash — direct match
        "expected_bottom": 1,  # Python C extensions — different ecosystem
    },

    # -----------------------------------------------------------------
    # 6. HARDWARE vs SOFTWARE — 3D printer config vs software config
    # "Configuration" appears in both but means very different things.
    # -----------------------------------------------------------------
    {
        "tag": "config-homonym",
        "query": "Klipper configuration for probe calibration and mesh leveling",
        "docs": [
            "Klipper config for Beacon probe integration: split across "
            "beacon.cfg (probe settings, mesh parameters, z-offset), "
            "beacon_macro.cfg (calibration macros, PROBE_CALIBRATE), and "
            "printer.cfg (include order, stepper definitions). The Beacon "
            "probe uses eddy current sensing for non-contact mesh leveling.",
            "ZettelkastenConfig uses Pydantic BaseSettings with env var "
            "support. Key settings: ZETTELKASTEN_NOTES_DIR, "
            "ZETTELKASTEN_DATABASE_PATH, ZETTELKASTEN_EMBEDDINGS_ENABLED. "
            "The config.py module validates paths, sets defaults, and "
            "supports .env file loading via python-dotenv.",
            "Manta M5P board wiring: TMC5160T Plus drivers on X/Y steppers "
            "(48V, 2.0A RMS), TMC2209 on Z and extruder (24V). Heater SSR "
            "outputs on HE0/HE1 pins. Thermistor inputs on TH0 (hotend) "
            "and TH1 (bed). Fan PWM on FAN0-FAN2.",
        ],
        "expected_top": 0,  # Klipper/Beacon probe config
        "expected_bottom": 1,  # Software config system — wrong domain
    },

    # -----------------------------------------------------------------
    # 7. GATE vs PLAN vs REVIEW — zettelkasten workflow stages
    # Tests whether reranker can distinguish note purposes within the
    # same project workflow.
    # -----------------------------------------------------------------
    {
        "tag": "workflow-stage",
        "query": "verification criteria for passing a phase gate",
        "docs": [
            "Gate: Phase 6 Test Coverage Verification Criteria. 8 checks: "
            "(1) pytest tests/test_dispatch.py passes, (2) cache tests "
            "cover invalidation edge cases, (3) AST matcher handles all "
            "node types, (4) regex matcher fallback works when tree-sitter "
            "unavailable, (5) helper unit tests achieve 90% branch coverage. "
            "Each check has an exact command and expected result.",
            "Implementation Plan: Review Finding Remediation — 5 Phases, "
            "16 items sequenced by dependency. Phase 1: crash prevention "
            "(2 items), Phase 2: type safety (4 items), Phase 3: security "
            "(3 items), Phase 4: performance (4 items), Phase 5: cleanup "
            "(3 items). Each phase produces a deliverable and gate check.",
            "Code Quality Review: Persistent Backend Wiring in ProjectContext. "
            "Findings: (1) ProjectContext.__init__ creates new service "
            "instances instead of sharing the singleton, (2) shutdown() "
            "doesn't propagate to child services, (3) no connection pooling "
            "for SQLite backends across project switches.",
        ],
        "expected_top": 0,  # Gate criteria — exactly what's asked for
        "expected_bottom": 2,  # Code review finding — not about gates
    },

    # -----------------------------------------------------------------
    # 8. LEXICAL TRAP — "test" in different contexts
    # "Test" appears in testing strategy AND in the test note content,
    # but only one is about testing methodology.
    # -----------------------------------------------------------------
    {
        "tag": "test-lexical-trap",
        "query": "test strategy for embedding integration with fake providers",
        "docs": [
            "The 5-phase embedding test suite (~1,923 lines) uses "
            "FakeEmbeddingProvider and FakeRerankerProvider throughout. "
            "Critical finding: OnnxRerankerProvider._score_pairs() has a "
            "NameError bug that no test catches because all phases use "
            "fakes. Real ONNX providers are never exercised in the test "
            "suite, creating a false confidence problem.",
            "Test Note 3. This is a test note created for verifying the "
            "zettelkasten CRUD operations. Tags: test, sample. Content is "
            "minimal and exists only to validate create/read/update/delete "
            "workflows in the MCP server integration tests.",
            "The protocol test suite (test_mcp_protocol.py) exercises all "
            "22 MCP tools through the full JSON-RPC pipeline using "
            "mcp.shared.memory transport. No mocking — real service layer "
            "with FakeEmbeddingProvider for deterministic vector search. "
            "30 tests covering CRUD, search, links, batch, and semantic.",
        ],
        # Doc 0 and 2 are both about test strategy with fake providers.
        # Doc 1 is a literal "test note" — lexical match but wrong meaning.
        "expected_top": None,  # Both doc 0 and 2 are relevant
        "expected_bottom": 1,  # Literal test note, not about testing strategy
    },

    # -----------------------------------------------------------------
    # 9. BRANCH MERGE ANALYSIS — specific git workflow
    # Narrow query about branch divergence; distractors are general
    # code review and architecture.
    # -----------------------------------------------------------------
    {
        "tag": "branch-merge",
        "query": "branch divergence analysis between main and local-work-backup",
        "docs": [
            "Refactoring analysis of znote-mcp branch merge strategy. "
            "Main branch (5 commits) adds project management and Obsidian "
            "mirroring. local-work-backup (13 commits) adds git-based "
            "concurrency control and versioned operations. 4 missing "
            "cross-branch components, 2 schema conflicts in models, "
            "6 silent error handlers that swallow exceptions.",
            "Meta-analysis of Round 1 code review findings for znote-mcp: "
            "validated 12 critical findings, discovered 6 previously missed "
            "issues, recalibrated 2 overstated concerns. Final assessment: "
            "functional but needs work. Key risk: insufficient error "
            "propagation in storage layer.",
            "The dual storage architecture uses markdown files as source "
            "of truth with SQLite as an indexing layer. Notes are stored "
            "as individual .md files with YAML frontmatter. The SQLite "
            "database provides FTS5 full-text search and can be rebuilt "
            "from scratch via zk_system(action='rebuild').",
        ],
        "expected_top": 0,  # Branch merge analysis — exact match
        "expected_bottom": 2,  # Architecture overview — unrelated to branches
    },

    # -----------------------------------------------------------------
    # 10. SHORT QUERY — terse search term from daily workflow
    # -----------------------------------------------------------------
    {
        "tag": "short-query",
        "query": "obsidian mirroring",
        "docs": [
            "Obsidian vault mirroring writes shadow copies of notes to a "
            "configured vault path. _build_obsidian_filename() adds a date "
            "prefix (YYYY-MM-DD_) to all filenames. Link rewriting converts "
            "zettelkasten IDs to wikilinks with the date prefix. Controlled "
            "by ZETTELKASTEN_OBSIDIAN_VAULT env var.",
            "OrcaSlicer 0.8mm nozzle configuration: layer height 0.4mm, "
            "line width 0.88mm, wall loops 3, sparse infill 15%, support "
            "threshold angle 40 degrees. Print speed 150mm/s for inner "
            "walls, 200mm/s for infill. Retraction 0.5mm at 40mm/s for "
            "direct drive extruder.",
            "The zettelkasten search service supports multiple modes: "
            "text (FTS5), tag filtering, link traversal, and semantic "
            "vector search. The default mode 'auto' routes based on "
            "available capabilities and filter parameters.",
        ],
        "expected_top": 0,  # Obsidian mirroring doc
        "expected_bottom": 1,  # 3D printer settings
    },

    # -----------------------------------------------------------------
    # 11. LONG QUERY — verbose multi-sentence search
    # Mimics a detailed investigation query about a specific bug.
    # -----------------------------------------------------------------
    {
        "tag": "long-query",
        "query": (
            "I'm investigating a bug where the session-manager agent enters "
            "an infinite search loop, making excessive zettelkasten queries "
            "during project initialization. It seems to be searching for "
            "prior session notes repeatedly without finding them, causing "
            "the session startup to take several minutes instead of seconds."
        ),
        "docs": [
            "Bug: Session-Manager Agent Search Loop. The session-manager "
            "agent's prior-work discovery phase calls zk_search_notes in a "
            "loop, searching for session tracking notes with increasingly "
            "broad queries when initial searches return empty. Root cause: "
            "the search tag filter uses OR semantics but the agent expects "
            "AND. Fix: switch to explicit tag intersection in the search "
            "service, add a max-retry limit to the discovery loop.",
            "Zettelkasten Steward Report - 2026-02-16. Routine maintenance "
            "completed: tag cleanup removed 14 orphan tags, database "
            "integrity check passed, 3 notes had stale project references "
            "updated, embedding index is current (962 notes, 1,847 chunks). "
            "No action items.",
            "The anamnesis session tracking system uses start_session and "
            "end_session to bracket work periods. Sessions record: project "
            "context, tools used, decisions made, and duration. The "
            "get_sessions endpoint retrieves historical session data for "
            "continuity across conversations.",
        ],
        "expected_top": 0,  # Session-manager search loop bug — exact match
        "expected_bottom": 1,  # Steward report — routine maintenance, not a bug
    },

    # -----------------------------------------------------------------
    # 12. SAFETY NET PATTERN — cross-cutting architectural concern
    # Tests whether reranker can match an abstract concept (cascading
    # failure) to a concrete description.
    # -----------------------------------------------------------------
    {
        "tag": "cascading-failure",
        "query": "safety net failure chain where multiple protective layers have aligned holes",
        "docs": [
            "Cross-domain synthesis identifying compound problems: Safety "
            "Net Failure Chain — Rust panic! crashes Node.js (no FFI panic "
            "handler), 225 'any' types bypass TypeScript checking (no strict "
            "mode), missing Zod validation allows malformed data through "
            "API boundary (no runtime checks). Each layer's gap aligns with "
            "the next, creating an unprotected path from input to crash.",
            "Remove phantom dependencies from pyproject.toml: pydantic, "
            "tenacity, and anyio are listed as direct dependencies but are "
            "only pulled in transitively through other packages. Removing "
            "them reduces install size and prevents version conflicts. "
            "Verification: run import checks after removal.",
            "CircuitBreaker absence risk assessment: the service layer has "
            "no circuit breaker pattern for external calls (embedding model "
            "inference, HuggingFace Hub downloads). A slow or failing "
            "external service causes cascading timeouts across all MCP "
            "tool handlers. Recommended: add circuit breaker with "
            "exponential backoff.",
        ],
        # Doc 0 is the exact safety-net-failure-chain note. Doc 2 is about
        # cascading failure but a different pattern (circuit breaker).
        "expected_top": 0,  # Safety net failure chain — exact concept match
        "expected_bottom": 1,  # Dependency cleanup — unrelated
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
