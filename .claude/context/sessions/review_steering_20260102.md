# Multi-Agent Code Review: Steering Capability

**Session ID:** review_steering_20260102
**Date:** 2026-01-02
**Type:** comprehensive review
**Scope:** Multi-prompt steering (single and batch modes)

## Executive Summary

The steering capability is well-architected with solid Celery-based async execution, proper process isolation, and comprehensive WebSocket progress tracking. The code shows good patterns for GPU memory management and task recovery. Several areas need attention around edge cases in batch mode, error recovery, and testing.

---

## Product Engineer Review

### Requirements Alignment ✅ GOOD

**User Stories Covered:**
- [x] Single prompt steering with multiple features (up to 4)
- [x] Multiple strengths per feature for comparison
- [x] Batch mode for testing multiple prompts
- [x] Real-time progress feedback via WebSocket
- [x] Cancel/abort running tasks
- [x] Save experiments for later reference
- [x] Recent comparisons persistence across page refresh
- [x] Task recovery after page refresh

**Business Logic Correctness:**
- Feature selection with unique instance_id allows same feature at different strengths
- comparison_id injection links features to specific runs (good for traceability)
- Rate limiting (5 req/min) protects from abuse

**Context Gaps:**
1. **No comparison history visualization** - Users cannot view past batch results in aggregate
2. **No progress estimation** - No ETA for multi-feature generations
3. **Batch failures don't show recovery options** - Failed prompts in batch lack retry UX

### Recommendations
| Priority | Finding | Recommendation |
|----------|---------|----------------|
| P2 | Batch results lack unified view | Add batch results summary component |
| P3 | No generation ETA | Add time estimation based on feature count |

---

## QA Engineer Review

### Code Quality ✅ EXCELLENT

**Standards Compliance:**
- [x] TypeScript strict typing throughout frontend
- [x] Python type hints in backend
- [x] Consistent naming (camelCase frontend, snake_case backend)
- [x] Comprehensive docstrings in API endpoints
- [x] Clean separation of concerns

**Error Handling Assessment:**

| Area | Quality | Notes |
|------|---------|-------|
| API validation | ✅ Excellent | SAE status, feature bounds, model lookup |
| Celery timeouts | ✅ Excellent | Soft (SIGTERM) + hard (SIGKILL) limits |
| WebSocket failures | ⚠️ Moderate | Silent failures logged but not surfaced |
| Batch error recovery | ⚠️ Moderate | Continues on error, but no retry mechanism |

**Critical Code Quality Issues:**

1. **Module-level mutable state** ([steeringStore.ts:43-46](frontend/src/stores/steeringStore.ts#L43-L46))
   ```typescript
   let pendingBatchResolver: { resolve, reject } | null = null;
   ```
   - Risk: Memory leak if promise never resolves, race condition if multiple batches started
   - Impact: Medium - could cause hung UI state
   - Fix: Move into store state or use cleanup pattern

2. **Race condition in batch subscribe** ([steeringStore.ts:804-805](frontend/src/stores/steeringStore.ts#L804-L805))
   ```typescript
   await new Promise(resolve => setTimeout(resolve, 100));
   ```
   - Risk: 100ms may not be enough for WebSocket subscription
   - Impact: Low - works in practice but fragile
   - Fix: Subscribe before submitting task, not after

3. **Sweep uses polling instead of WebSocket** ([steeringStore.ts:956-972](frontend/src/stores/steeringStore.ts#L956-L972))
   - Risk: Inconsistent pattern with comparison
   - Impact: Low - works but adds unnecessary load
   - Fix: Use same WebSocket pattern as comparison

### Testing Coverage ❌ INSUFFICIENT

**Current State:**
- No unit tests found for steeringStore
- No integration tests for steering workflow
- No WebSocket event handling tests

**Testing Gaps:**
1. Feature selection edge cases (max 4, duplicates, strength bounds)
2. Batch abort mid-processing
3. WebSocket reconnection during generation
4. Task recovery after various failure modes

### Recommendations
| Priority | Finding | Recommendation |
|----------|---------|----------------|
| P0 | pendingBatchResolver leak risk | Add cleanup in abortBatch, add timeout |
| P1 | No unit tests | Add tests for core store logic |
| P1 | Sweep polling inconsistency | Migrate sweep to WebSocket pattern |
| P2 | Race condition in batch subscribe | Subscribe before task submission |

---

## Architect Review

### Design Pattern Consistency ✅ EXCELLENT

**Strengths:**
1. **Celery task isolation** - GPU operations run in separate process with SIGKILL guarantee
2. **Worker recycling** - `--max-tasks-per-child=1` prevents memory accumulation
3. **WebSocket progress pattern** - Consistent emission through HTTP callback to avoid direct WS coupling
4. **State persistence** - localStorage persistence with selective partialize

**Architecture Diagram:**
```
┌─────────────────────────────────────────────────────────────────────┐
│                         FRONTEND                                     │
├─────────────────────────────────────────────────────────────────────┤
│  SteeringPanel.tsx ──► steeringStore.ts ──► steering.ts (API)       │
│        │                    ▲                      │                 │
│        │                    │                      ▼                 │
│        └───► useSteeringWebSocket.ts ◄── WebSocketContext           │
├─────────────────────────────────────────────────────────────────────┤
│                         BACKEND                                      │
├─────────────────────────────────────────────────────────────────────┤
│  steering.py (API) ──► steering_tasks.py (Celery) ──► Redis Queue   │
│        │                      │                                      │
│        │                      ▼                                      │
│        │              steering_service.py (GPU ops)                  │
│        │                      │                                      │
│        ▼                      ▼                                      │
│  ws_manager ◄───────  websocket_emitter.py ──► HTTP /ws/emit        │
└─────────────────────────────────────────────────────────────────────┘
```

### Integration Quality

| Integration | Quality | Notes |
|-------------|---------|-------|
| Store ↔ API | ✅ Excellent | Clean async/await, proper error propagation |
| API ↔ Celery | ✅ Excellent | Type-safe request serialization |
| Celery ↔ WebSocket | ✅ Good | HTTP callback decouples worker from WS |
| Store ↔ WebSocket | ⚠️ Moderate | Event filtering by task_id works but fragile |

### Technical Debt

1. **Batch mode uses module-level resolver** - Breaks encapsulation, should be in store
2. **Sweep uses polling** - Inconsistent with comparison pattern
3. **100ms sleep for subscription timing** - Should be event-driven

### Scalability Assessment

| Aspect | Current State | Concern |
|--------|---------------|---------|
| Concurrent users | Single worker | One steering task at a time globally |
| GPU memory | Per-task worker | Good isolation, cleaned on SIGKILL |
| Result storage | Redis with TTL | Results expire, experiment save required |
| Rate limiting | In-memory per-client | Won't work with multiple backend instances |

### Recommendations
| Priority | Finding | Recommendation |
|----------|---------|----------------|
| P1 | Single steering worker | Consider queue priority for short vs long tasks |
| P2 | Rate limiter in-memory | Move to Redis for multi-instance support |
| P2 | pendingBatchResolver | Refactor into store state machine |
| P3 | Sweep polling | Align with WebSocket pattern |

---

## Test Engineer Review

### Risk Assessment

**High-Risk Areas:**

1. **Zombie process prevention** (P0)
   - Context: This was the main reason for Celery migration
   - Current protection: Worker recycling, SIGKILL timeout
   - Gap: No automated test verifying GPU cleanup after timeout

2. **Batch WebSocket coordination** (P1)
   - Context: Module-level promise resolver could deadlock
   - Current protection: 5-minute timeout
   - Gap: No test for WebSocket disconnect mid-batch

3. **Task recovery after page refresh** (P1)
   - Context: localStorage persists task state
   - Current protection: recoverActiveTask polls backend
   - Gap: No test for recovery of different task states

4. **Double submission prevention** (P0)
   - Context: Recent fix added isGenerating guard
   - Current protection: Guard at start of generateComparison
   - Gap: No unit test verifying guard behavior

### Test Strategy Recommendations

**Unit Tests Needed:**
```typescript
// steeringStore.test.ts
describe('generateComparison', () => {
  it('should reject duplicate calls when isGenerating is true')
  it('should set isGenerating before API call')
  it('should clear isGenerating on error')
})

describe('generateBatchComparison', () => {
  it('should process prompts sequentially')
  it('should continue on individual prompt failure')
  it('should stop when aborted')
  it('should handle WebSocket timeout per prompt')
})

describe('handleAsyncCompleted', () => {
  it('should resolve batch promise when in batch mode')
  it('should update state when in single mode')
  it('should add to recentComparisons')
})
```

**Integration Tests Needed:**
```python
# test_steering_workflow.py
def test_steering_comparison_happy_path()
def test_steering_timeout_cleanup()
def test_steering_abort_running_task()
def test_steering_rate_limiting()
def test_steering_invalid_feature_index()
```

### Debugging Capability

| Capability | Status |
|------------|--------|
| Console logging | ✅ Good - [Steering WS] prefixed logs |
| Progress visibility | ✅ Good - Real-time WebSocket updates |
| Error traceability | ⚠️ Moderate - Backend errors truncated to 100 chars |
| Task state inspection | ✅ Good - /async/result/{task_id} endpoint |

### Recommendations
| Priority | Finding | Recommendation |
|----------|---------|----------------|
| P0 | No test for double-submission guard | Add unit test |
| P0 | No test for batch abort | Add integration test |
| P1 | No test for task recovery | Add test for each recovery state |
| P1 | Error truncation | Log full error, truncate only for display |

---

## Summary of Findings

### Critical (P0)
1. **pendingBatchResolver memory leak risk** - Module-level mutable state could hang if promise never resolves
2. **No unit tests for critical guards** - Double-submission prevention untested

### High (P1)
1. **Sweep uses polling** - Inconsistent with WebSocket pattern used by comparison
2. **No test coverage** - Batch mode, abort, recovery paths untested
3. **100ms subscription delay** - Race condition with WebSocket subscription

### Medium (P2)
1. **Rate limiter in-memory** - Won't scale to multiple backend instances
2. **No batch results visualization** - User can't see aggregate batch results
3. **Error messages truncated** - Full errors not preserved for debugging

### Low (P3)
1. **No generation ETA** - Could estimate based on feature count
2. **Sweep pattern inconsistency** - Minor technical debt

---

## Recommended Actions

### Immediate (This Session)
1. Add cleanup of pendingBatchResolver in abortBatch to prevent leak
2. Add timeout cleanup pattern for abandoned promises

### Short Term (This Week)
1. Add unit tests for steeringStore core functions
2. Migrate sweep to WebSocket pattern for consistency

### Medium Term (This Month)
1. Add integration tests for steering workflow
2. Move rate limiter to Redis for horizontal scaling
3. Add batch results summary component

---

## Files Reviewed

- [frontend/src/stores/steeringStore.ts](frontend/src/stores/steeringStore.ts) (1292 lines)
- [frontend/src/api/steering.ts](frontend/src/api/steering.ts) (195 lines)
- [frontend/src/types/steering.ts](frontend/src/types/steering.ts) (353 lines)
- [frontend/src/hooks/useSteeringWebSocket.ts](frontend/src/hooks/useSteeringWebSocket.ts) (180 lines)
- [backend/src/api/v1/endpoints/steering.py](backend/src/api/v1/endpoints/steering.py) (670 lines)
- [backend/src/workers/steering_tasks.py](backend/src/workers/steering_tasks.py) (267 lines)
- [backend/src/workers/websocket_emitter.py](backend/src/workers/websocket_emitter.py) (1340 lines)

---

*Review completed by multi-agent collaboration: Product Engineer, QA Engineer, Architect, Test Engineer*
