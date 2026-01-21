# Documentation Update Tasks - January 2025

**Document ID:** DOCS_TASKS|Documentation_Update_Jan2025
**Created:** 2026-01-21
**Status:** In Progress
**Priority:** P1

---

## 1. Overview

This task list documents all changes made to miStudio since the last documentation update (December 2025) and plans the updates needed to bring all 0xcc documentation in sync with the current implementation state.

### 1.1 CI/CD Pipeline Context

```
┌─────────────────────┐     Release      ┌─────────────────────┐
│      GitHub:        │    Codebase      │      GitHub:        │
│  Onegaishimas/      │───────────────►  │   hitsainet/        │
│    miStudio         │   (sync.yml)     │    miStudio         │
│  (Private Dev)      │                  │  (Public Release)   │
└─────────────────────┘                  └─────────┬───────────┘
         ▲                                         │
         │                                         │ GitHub Actions
    Private                                        │ (docker-images.yml)
    Codebase                                       ▼
         │                               ┌─────────────────────┐
┌────────┴────────────┐                  │    Docker Hub:      │
│   Claude Code       │                  │  hitsai/mistudio-   │
│   w/local git       │                  │    backend:latest   │
└─────────────────────┘                  │  hitsai/mistudio-   │
                                         │    frontend:latest  │
                                         └─────────┬───────────┘
                                                   │
                              ┌─────────────────────┴─────────────────────┐
                              │                                           │
                              ▼                                           ▼
                    ┌─────────────────┐                         ┌─────────────────┐
                    │   Kubernetes    │                         │  Docker-Compose │
                    │   (MicroK8s)    │                         │   (Local Dev)   │
                    └─────────────────┘                         └─────────────────┘
```

**Key Point:** Claude Code pushes to Onegaishimas/miStudio only. The sync workflow then pushes to hitsainet/miStudio, which triggers GitHub Actions to build and push Docker images to hitsai Docker Hub organization.

---

## 2. Changes Since Last Documentation Update

### 2.1 Major Features Implemented

| Feature | Commits | PRD Impact | TDD Impact |
|---------|---------|------------|------------|
| Multi-Dataset Training | `db593ca` | 003_FPRD | 003_FTDD |
| Multi-Hook/Multi-Layer SAE Training | `b4b7285`, `16cda1b` | 003_FPRD, 005_FPRD | 003_FTDD |
| Batch SAE Extraction | `b4b7285`, `810cecd` | 004_FPRD | 004_FTDD |
| External SAE Import (Multi-SAE) | `d01607b`, `ae170c5` | 005_FPRD | 005_FTDD |
| Save as Template (Extraction Modal) | `2bef750` | 004_FPRD | 004_FTDD |
| Live Extraction Metrics | `c57ebc2`, `92879a2` | 004_FPRD | 004_FTDD |
| Sequential Batch + NLP Pipeline | `810cecd`, `d87039c` | 004_FPRD | 004_FTDD |
| Deployment Portability (Docker/K8s) | `b4b7285`, `1b182c9` | ADR | ADR |

### 2.2 Infrastructure Improvements

| Improvement | Commits | Impact |
|-------------|---------|--------|
| GitHub Actions CI/CD | `aa4d691`, `bf1c901` | ADR (deployment) |
| K8s Simplified Deployment | `1b182c9` | ADR |
| Docker Hub Scout Security | `9c89589` | ADR |
| Idempotent Migrations | `749e4db`, `13814cb` | ADR |
| Auto-Migrations on Startup | `8ecc962` | ADR |

### 2.3 Bug Fixes & Improvements

| Fix | Commits | Related Feature |
|-----|---------|-----------------|
| WebSocket Status Field | `b2ed6f6` | 004 (Extraction) |
| VRAM Detection for Batch Size | `a8487f8`, `639dfb6` | 004 (Extraction) |
| Model Name Display | `0a8a1ca`, `4751179` | 005 (SAE Management) |
| Training_id Nullable | `13814cb` | 004, 005 |
| spaCy Download Fix | `61d1bf5` | 004 (NLP Analysis) |
| Steering Worker Paths | `4fff38e`, `cf1cf6f` | 006 (Steering) |

---

## 3. Detailed Todo List: Graph Metrics Fix (IMMEDIATE)

### 3.1 Backend Changes

#### Task 3.1.1: Add get_stats() to IncrementalTopKHeap
- [ ] **File:** `backend/src/services/extraction_vectorized.py`
- [ ] **Location:** After `get_heaps()` method (around line 155)
- [ ] **Code:**
```python
def get_stats(self) -> Dict[str, int]:
    """
    Get current heap statistics for progress reporting.

    Returns:
        Dictionary with:
        - features_in_heap: Number of features with at least one example
        - heap_examples_count: Total examples across all heaps
    """
    features_in_heap = len(self.heaps)
    heap_examples_count = sum(len(heap) for heap in self.heaps.values())
    return {
        "features_in_heap": features_in_heap,
        "heap_examples_count": heap_examples_count
    }
```

#### Task 3.1.2: Update emit_progress in extraction_service.py
- [ ] **File:** `backend/src/services/extraction_service.py`
- [ ] **Location:** Progress emission block (around line 1369)
- [ ] **Changes:**
  - Call `heap_stats = incremental_heap.get_stats()` before emit
  - Add `"features_in_heap": heap_stats["features_in_heap"]` to data dict
  - Add `"heap_examples_count": heap_stats["heap_examples_count"]` to data dict

### 3.2 Frontend Verification

#### Task 3.2.1: Verify Type Definition
- [ ] **File:** `frontend/src/types/features.ts`
- [ ] **Check:** `ExtractionProgressEvent` includes `features_in_heap?: number` and `heap_examples_count?: number`
- [ ] **Update if missing**

#### Task 3.2.2: Verify Hook Handling
- [ ] **File:** `frontend/src/hooks/useExtractionWebSocket.ts`
- [ ] **Check:** Hook properly passes through `features_in_heap` and `heap_examples_count`

### 3.3 Testing & Deployment

#### Task 3.3.1: Local Testing
- [ ] Run extraction job locally
- [ ] Verify console shows `features_in_heap` and `heap_examples_count` in WebSocket messages
- [ ] Verify UI graphs update in real-time

#### Task 3.3.2: Commit & Push
- [ ] Stage changes: `git add .`
- [ ] Commit with message: `feat(extraction): add graph metrics (features_in_heap, heap_examples_count)`
- [ ] Push to Onegaishimas/miStudio

#### Task 3.3.3: Verify CI/CD
- [ ] Wait for sync workflow to push to hitsainet/miStudio
- [ ] Wait for docker-images.yml to build new images
- [ ] Verify images on Docker Hub: `hitsai/mistudio-backend:latest`

#### Task 3.3.4: K8s Deployment
- [ ] Rollout restart: `kubectl rollout restart deployment/mistudio-backend -n mistudio`
- [ ] Verify pods healthy: `kubectl get pods -n mistudio`
- [ ] Run test extraction and verify graphs

---

## 4. Documentation Update Plan

### 4.1 PRD Updates

#### 004_FPRD|Feature_Discovery.md

**Section 2.1 Feature Extraction - ADD:**
```markdown
| FR-1.7 | Live progress metrics (samples/second, ETA) | Implemented |
| FR-1.8 | Time-based progress emission (every 2 seconds) | Implemented |
| FR-1.9 | Features in heap count for progress graphs | Pending |
| FR-1.10 | Heap examples count for collection rate graphs | Pending |
```

**Section 2.6 Extraction Templates - ADD:**
```markdown
| FR-6.5 | "Save as Template" button in extraction modal | Implemented |
| FR-6.6 | Auto-generated template names from config | Implemented |
```

**ADD New Section 2.10 Batch Extraction:**
```markdown
### 2.10 Batch Extraction (Sub-feature - Added Jan 2026)
| Requirement | Description | Status |
|-------------|-------------|--------|
| FR-10.1 | Select multiple SAEs for batch extraction | Implemented |
| FR-10.2 | Sequential processing (one at a time) | Implemented |
| FR-10.3 | Auto-continue after NLP completion | Implemented |
| FR-10.4 | Batch progress tracking (position/total) | Implemented |
| FR-10.5 | Continue batch even if one job fails NLP | Implemented |
```

#### 003_FPRD|SAE_Training.md

**ADD Section for Multi-Dataset Training:**
```markdown
### 2.X Multi-Dataset Training (Added Jan 2026)
| Requirement | Description | Status |
|-------------|-------------|--------|
| FR-X.1 | Select multiple datasets for training | Implemented |
| FR-X.2 | Dataset concatenation at training time | Implemented |
| FR-X.3 | dataset_ids JSONB array storage | Implemented |
| FR-X.4 | Backward compat with single dataset_id | Implemented |
```

**ADD Section for Multi-Hook Training:**
```markdown
### 2.Y Multi-Hook Training (Added Jan 2026)
| Requirement | Description | Status |
|-------------|-------------|--------|
| FR-Y.1 | Select multiple hook types per layer | Implemented |
| FR-Y.2 | Train separate SAE for each layer/hook combo | Implemented |
| FR-Y.3 | Linked SAEs share training_id | Implemented |
| FR-Y.4 | Display hook types on training cards | Implemented |
```

#### 005_FPRD|SAE_Management.md

**ADD Section for External SAE Import Enhancements:**
```markdown
### 2.X External SAE Import Enhancements (Added Jan 2026)
| Requirement | Description | Status |
|-------------|-------------|--------|
| FR-X.1 | Import all SAEs from multi-hook trainings | Implemented |
| FR-X.2 | Auto-detect hook type from cfg.json | Implemented |
| FR-X.3 | Filter already-imported SAEs from list | Implemented |
| FR-X.4 | Show already-imported SAEs (disabled) | Implemented |
| FR-X.5 | Support external SAEs without training_id | Implemented |
```

### 4.2 ADR Updates (000_PADR|miStudio.md)

**ADD Section: CI/CD Pipeline**
```markdown
## X. CI/CD Pipeline

### X.1 Repository Structure
- **Development:** Onegaishimas/miStudio (private)
- **Release:** hitsainet/miStudio (public)
- **Sync:** GitHub Actions workflow syncs main branch

### X.2 Docker Image Build
- **Trigger:** Push to hitsainet/miStudio main branch
- **Builder:** GitHub Actions (docker-images.yml)
- **Registry:** Docker Hub (hitsai organization)
- **Images:**
  - hitsai/mistudio-backend:latest
  - hitsai/mistudio-frontend:latest

### X.3 Deployment Modes
| Mode | Migration Handling | Image Source |
|------|-------------------|--------------|
| Docker Compose | Entrypoint auto-migrate | Docker Hub |
| Kubernetes | Entrypoint auto-migrate | Docker Hub |
| Native | Manual alembic | Local build |
```

**UPDATE Section: Database Migrations**
```markdown
### Y.Z Idempotent Migrations
All migrations use existence checks to be safely re-runnable:
- `column_exists()` - Check before ADD COLUMN
- `constraint_exists()` - Check before DROP CONSTRAINT
- `column_is_nullable()` - Check before ALTER COLUMN

This supports:
- Running same migration on multiple deployment targets
- Recovery from partial migration failures
- Safe re-runs during development
```

### 4.3 TDD Updates

#### 004_FTDD|Feature_Discovery.md

**ADD: Live Metrics Architecture**
```markdown
## X. Live Extraction Metrics

### X.1 Time-Based Progress Emission
- Emit progress every 2 seconds OR at 5% intervals (whichever comes first)
- Prevents UI stall during slow extractions
- Controlled by `last_emit_time` tracking

### X.2 Metrics Included in Progress Events
| Metric | Type | Description |
|--------|------|-------------|
| progress | float | 0.0-1.0 completion percentage |
| current_batch | int | Current batch number |
| total_batches | int | Total batches to process |
| samples_processed | int | Samples completed |
| total_samples | int | Total samples in dataset |
| samples_per_second | float | Processing rate |
| eta_seconds | int | Estimated time remaining |
| features_in_heap | int | Features with examples |
| heap_examples_count | int | Total examples collected |

### X.3 Batch Extraction Flow
```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Job 1: Extract  │────►│  Job 1: NLP      │────►│  Job 2: Extract  │──►...
│  (EXTRACTING)    │     │  (NLP_ANALYSIS)  │     │  (EXTRACTING)    │
└──────────────────┘     └──────────────────┘     └──────────────────┘
                                  │
                                  ▼
                         _start_next_batch_job()
                         Triggers next in sequence
```

### X.4 WebSocket Channel Pattern
- Channel: `extraction/{extraction_id}`
- Events: `extraction:progress`, `extraction:completed`, `extraction:failed`
- Status field always included in progress events
```

### 4.4 TID Updates

#### 004_FTID|Feature_Discovery.md

**ADD: Implementation Details**
```markdown
## X. Live Metrics Implementation

### X.1 Backend (extraction_service.py)

#### Time tracking variables (before batch loop):
```python
extraction_start_time = time.time()
last_emit_time = extraction_start_time
```

#### Time-based emission condition:
```python
current_time = time.time()
should_emit = (
    (current_time - last_emit_time >= 2.0) or
    (int(progress * 20) > int((batch_start / len(dataset)) * 20))
)
if should_emit:
    last_emit_time = current_time
    # ... emit progress
```

#### Metrics calculation:
```python
elapsed_time = time.time() - extraction_start_time
samples_per_second = batch_end / elapsed_time if elapsed_time > 0 else 0
eta_seconds = remaining_samples / samples_per_second if samples_per_second > 0 else 0
```

### X.2 Frontend (useExtractionWebSocket.ts)

#### Defensive state updates:
```typescript
const update: Partial<ExtractionProgressEvent> = {};
if (data.status !== undefined) update.status = data.status;
if (data.progress !== undefined) update.progress = data.progress;
// Only include defined fields to prevent overwriting
```

### X.3 Batch Continuation (nlp_analysis_tasks.py)

#### Helper function after NLP success:
```python
def _start_next_batch_job(db: Session, current_job) -> None:
    next_job = db.query(ExtractionJob).filter(
        ExtractionJob.batch_id == current_job.batch_id,
        ExtractionJob.batch_position == current_job.batch_position + 1,
        ExtractionJob.status == ExtractionStatus.QUEUED.value
    ).first()

    if next_job:
        task_result = extract_features_from_sae_task.apply_async(...)
        next_job.celery_task_id = task_result.id
```
```

### 4.5 TASKS Updates

#### 004_FTASKS|Feature_Discovery.md

**Mark as Completed:**
```markdown
### X. Live Extraction Metrics (Jan 2026)
- [x] Add time-based progress emission (every 2 seconds)
- [x] Include status field in all progress events
- [x] Add samples_per_second metric
- [x] Add eta_seconds metric
- [x] Add current_batch/total_batches metrics
- [ ] Add features_in_heap metric (PENDING)
- [ ] Add heap_examples_count metric (PENDING)

### Y. Batch Extraction Workflow (Jan 2026)
- [x] Support multiple SAE selection in extraction modal
- [x] Queue all jobs with batch_id, batch_position, batch_total
- [x] Only start first job immediately
- [x] Start next job after NLP completion
- [x] Continue batch even if NLP fails

### Z. Save as Template (Jan 2026)
- [x] Add "Save as Template" button to extraction modal
- [x] Auto-generate template name from config
- [x] Store SAE-specific settings in extra_metadata
```

---

## 5. Execution Order

### Phase 1: Immediate (Graph Metrics Fix)
1. [ ] Complete Task 3.1.1 (get_stats method)
2. [ ] Complete Task 3.1.2 (emit_progress update)
3. [ ] Complete Task 3.2.x (frontend verification)
4. [ ] Complete Task 3.3.x (testing & deployment)

### Phase 2: Documentation Updates
5. [ ] Update 004_FPRD|Feature_Discovery.md
6. [ ] Update 003_FPRD|SAE_Training.md
7. [ ] Update 005_FPRD|SAE_Management.md
8. [ ] Update 000_PADR|miStudio.md
9. [ ] Update 004_FTDD|Feature_Discovery.md
10. [ ] Update 004_FTID|Feature_Discovery.md
11. [ ] Update 004_FTASKS|Feature_Discovery.md

### Phase 3: Verification
12. [ ] Review all updates for consistency
13. [ ] Update CLAUDE.md session history
14. [ ] Commit documentation updates

---

## 6. File Inventory

### Files to Modify
| File | Type | Status |
|------|------|--------|
| `backend/src/services/extraction_vectorized.py` | Code | Pending |
| `backend/src/services/extraction_service.py` | Code | Pending |
| `frontend/src/types/features.ts` | Code | Verify |
| `0xcc/prds/004_FPRD\|Feature_Discovery.md` | PRD | Update |
| `0xcc/prds/003_FPRD\|SAE_Training.md` | PRD | Update |
| `0xcc/prds/005_FPRD\|SAE_Management.md` | PRD | Update |
| `0xcc/adrs/000_PADR\|miStudio.md` | ADR | Update |
| `0xcc/tdds/004_FTDD\|Feature_Discovery.md` | TDD | Update |
| `0xcc/tids/004_FTID\|Feature_Discovery.md` | TID | Update |
| `0xcc/tasks/004_FTASKS\|Feature_Discovery.md` | Tasks | Update |
| `CLAUDE.md` | Meta | Update |

---

## 7. Commit Message Templates

### Graph Metrics Fix
```
feat(extraction): add graph metrics (features_in_heap, heap_examples_count)

- Add get_stats() method to IncrementalTopKHeap class
- Include heap statistics in progress emission
- Enables real-time "Examples Collected" and "Collection Rate" graphs

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
```

### Documentation Update
```
docs: comprehensive documentation update for Jan 2026 features

- Update PRDs with batch extraction, multi-dataset training, live metrics
- Update ADR with CI/CD pipeline documentation
- Update TDD/TID with implementation details
- Mark completed tasks in FTASKS

Features documented:
- Batch SAE extraction with sequential processing
- Multi-dataset training support
- Live extraction metrics (samples/sec, ETA)
- Save as Template functionality
- External SAE import enhancements

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
```
