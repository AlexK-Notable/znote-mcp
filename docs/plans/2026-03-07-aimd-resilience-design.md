# AIMD Adaptive Resilience & Agent Signaling Design

**Date:** 2026-03-07
**Status:** Approved
**Scope:** Replace fixed 5-level ONNX resilience staircase with AIMD-based adaptive resilience, add circuit breaker for GPU/CPU switching, and enhance agent-facing signaling for embedding system state changes.

## Problem Statement

The current resilience manager (`OnnxResilienceManager`) uses a fixed 5-level one-way degradation staircase (NORMAL → REDUCED_BATCH → REDUCED_TOKENS → CPU_FALLBACK → DISABLED). It never recovers — once degraded, only a server restart restores full capacity. Agents consuming the MCP tools get opaque errors with no context about what's happening or when things will improve.

## Design Principles

1. **Graceful degradation with context** — tool calls succeed during pressure; agents get informed about state changes
2. **Automatic recovery** — AIMD probes back toward full capacity when conditions improve
3. **Embedding failures never take down the server** — CRUD, FTS, links, tags always available
4. **Agents can introspect and adjust** — query AIMD state, manually reset or force CPU

## Architecture Overview

Three layers with distinct responsibilities:

| Layer | Governs | Triggers |
|-------|---------|----------|
| **AIMD Controller** | `memory_budget_gb` envelope | success/failure of individual batches |
| **Adaptive Batcher** | actual batch composition within budget | per-batch token/chunk analysis |
| **Circuit Breaker** | GPU↔CPU switching, cooldowns | budget < 1GB floor, or error rate threshold |

Token budget and chunk_size are **not** pressure relief valves. The chunker already splits long notes into small overlapping chunks. Reducing chunk_size under pressure creates more work (more chunks to embed), not less. These stay fixed.

## 1. AIMD Controller Core

Each component (embedder, reranker) gets an `AIMDController`:

**State:**
- `cwnd` — current memory budget in GB (starts at half the hardware tier ceiling)
- `ssthresh` — slow-start threshold (starts at max, set to `cwnd/2` on first failure)
- `phase` — `slow_start` | `congestion_avoidance` | `cooldown`
- `min_cwnd` — 1.0 GB (hard floor, triggers circuit breaker)
- `max_cwnd` — hardware tier ceiling (from auto-tuning)

**Transitions:**
- `on_success(budget_utilization)` — only counts if `budget_utilization >= 0.75` (batch exercised near the budget ceiling). In slow start: double cwnd up to ssthresh. In congestion avoidance: cwnd += increment.
- `on_failure()` — `ssthresh = cwnd / 2`, `cwnd = ssthresh`, exit slow start. Increments error rate tracker.
- `on_cooldown_exit()` — resume at current cwnd in congestion avoidance (no slow start).

**Cross-component linking:**
When component A hits the budget floor or enters cooldown, component B preemptively sets `ssthresh = min(ssthresh, cwnd / 2)` and exits slow start into congestion avoidance. This is a caution signal — B doesn't halve immediately, but stops being aggressive.

## 2. Circuit Breaker (Error Rate Safety Valve)

Sliding window error rate tracker, separate from AIMD:

- **Window:** ring buffer of last N failure timestamps (e.g., size 10)
- **Trip condition:** >= 3 failures within 60 seconds
- **Trip response:** switch to CPU provider, start cooldown timer
- **Cooldown:** component stays operational on CPU. After `cooldown_seconds` (e.g., 120s), attempt GPU again
- **Escalation:** second trip without intervening GPU success doubles cooldown (capped at 600s)
- **Max trips:** stay on CPU indefinitely; agent can manually reset via `zk_system`

**Budget floor integration:**
When AIMD halves the budget below 1GB, the circuit breaker also trips — interpreting this as "GPU can't sustain minimal workloads."

Two independent paths into the circuit breaker:
- **Error rate** — sudden storm (3 OOMs in 60s)
- **Budget floor** — slow squeeze (AIMD halved budget below 1GB)

Both lead to the same CPU fallback response.

On CPU, the AIMD controller resets its budget to a CPU-appropriate starting value (half the CPU tier ceiling) and continues operating normally.

**Hard disable:** only if CPU also trips the circuit breaker. FTS always available.

## 3. Adaptive Batcher Integration

The adaptive batcher operates within the AIMD-governed memory budget:

```
AIMD controller → sets memory_budget_gb (e.g., 3.0GB)
    ↓
Adaptive batcher receives batch of note chunks
    → estimates per-item cost from token counts
    → fits items into the AIMD-governed budget
    → runs inference
    ↓
Success? → AIMD.on_success(budget_utilization)
OOM?     → AIMD.on_failure() → budget halved
           next batch naturally gets smaller sub-batches
```

Chunking is upstream — long notes are already split into overlapping token-aware chunks before reaching the batcher. The batcher treats each chunk as an independent item.

## 4. Hardware Tuning Changes

Current hardware tiers become `max_cwnd` — the ceiling AIMD can grow to, not the starting point:

| Tier | max_cwnd (budget GB) | AIMD start (cwnd) |
|------|---------------------|--------------------|
| gpu-16gb+ | 10.0 | 5.0 |
| gpu-8gb+ | 6.0 | 3.0 |
| gpu-small | 3.0 | 1.5 |
| cpu-32gb+ | 8.0 | 4.0 |
| cpu-16gb+ | 4.0 | 2.0 |
| cpu-8gb+ | 2.0 | 1.0 |
| cpu-small | 1.0 | 0.5 |

AIMD slow-starts from half the ceiling and discovers the actual safe operating point for the current environment.

## 5. Agent Signaling — Inline State Transition Notices

**Prepended** to tool responses on meaningful state transitions only:

**Triggers:**
- AIMD halve (budget reduced)
- AIMD recovery to full capacity
- Circuit breaker trip (switching to CPU)
- Circuit breaker cooldown expired (attempting GPU)
- Component disabled
- Component re-enabled (via agent action)

**Format:**
```
---
⚠ Embedding system state change
  Event: memory pressure — budget reduced (3.0GB→1.5GB)
  Provider: GPU (CUDA)
  Semantic search: operational, may be slower
---

[normal tool response follows]
```

Recovery notice:
```
---
✅ Embedding system recovered
  Event: GPU resumed, budget restored to 6.0GB
  Semantic search: fully operational
---
```

## 6. Agent Controls

**New `zk_system` actions:**

| Action | Effect |
|--------|--------|
| `embedding_reset` | Reset AIMD controllers, clear circuit breaker, attempt GPU from initial state |
| `embedding_force_cpu` | Switch to CPU, bypass AIMD/circuit breaker. Stays on CPU until `embedding_reset` |
| `embedding_disable` | Disable semantic search for session. FTS only |
| `embedding_enable` | Re-enable after manual disable. Enters slow start on GPU |

**Enhanced `zk_status` embeddings section adds:**
- AIMD state per component (phase, cwnd, ssthresh, max_cwnd)
- Circuit breaker state (closed/open, failure count, cooldown remaining)
- Active provider (GPU/CPU)
- Last event and time since
- Session stats (total adjustments, trip count)

## 7. Edge Cases

- **Reindex operations:** respect current AIMD budget. Slower under pressure, but won't crash.
- **Idle timeout/reload:** AIMD state preserved across model unload/reload cycles. The state reflects the environment, not the model lifecycle.
- **Concurrent requests:** AIMD state updates are thread-safe. In-flight batches aren't affected by a halve triggered by a parallel batch; the change applies to the next attempt.
- **Agent reset during cooldown:** `embedding_reset` clears cooldown immediately and attempts GPU from initial state.

## 8. What Doesn't Change

- FTS search — always available, no embedding dependency
- All CRUD tools — note creation, updates, links, tags, bulk ops
- Idle timeouts — embedder/reranker still unload after inactivity
- `zk_search_notes` mode="auto" routing — falls back to FTS when semantic unavailable
- MCP server process — never crashes due to embedding issues
- Chunk size and token budget — fixed, not pressure relief valves
