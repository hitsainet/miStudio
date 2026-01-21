# miStudio Changelog - January 2026

**Document ID:** CHANGELOG_Jan2026
**Period:** 2026-01-01 to 2026-01-21
**Status:** Current

---

## Overview

This document summarizes all features, fixes, and infrastructure improvements made to miStudio during January 2026. Changes are organized by category with links to relevant commits.

---

## CI/CD Pipeline Architecture

### Repository & Deployment Flow

```
┌─────────────────────┐     sync-to-clean    ┌─────────────────────┐
│      GitHub:        │      workflow        │      GitHub:        │
│  Onegaishimas/      │─────────────────────►│   hitsainet/        │
│    miStudio         │                      │    miStudio         │
│  (Private Dev)      │                      │  (Public Release)   │
└─────────────────────┘                      └─────────┬───────────┘
         ▲                                             │
         │                                             │ docker-images.yml
    Claude Code                                        │ GitHub Actions
    Development                                        ▼
                                             ┌─────────────────────┐
                                             │    Docker Hub:      │
                                             │  hitsai/mistudio-   │
                                             │    backend:latest   │
                                             │  hitsai/mistudio-   │
                                             │    frontend:latest  │
                                             └─────────┬───────────┘
                                                       │
                              ┌────────────────────────┴───────────────────────┐
                              │                                                │
                              ▼                                                ▼
                    ┌─────────────────┐                              ┌─────────────────┐
                    │   Kubernetes    │                              │  Docker-Compose │
                    │   (MicroK8s)    │                              │   (Local Dev)   │
                    └─────────────────┘                              └─────────────────┘
```

### Key Points
- **Development:** All code changes go through Onegaishimas/miStudio (private)
- **Release:** sync-to-clean.yml workflow pushes to hitsainet/miStudio (public)
- **Build:** docker-images.yml in hitsainet/miStudio builds and pushes to Docker Hub
- **Images:** hitsai/mistudio-backend:latest, hitsai/mistudio-frontend:latest

---

## 1. Feature Extraction Improvements

### 1.1 Live Extraction Metrics
**Commits:** `c57ebc2`, `92879a2`

Added real-time metrics during feature extraction to power the UI progress panel.

**Backend Changes (extraction_service.py):**
- Time-based progress emission (every 2 seconds OR at 5% intervals)
- Added timing variables: `extraction_start_time`, `last_emit_time`
- New metrics in progress events:
  - `samples_per_second`: Processing rate
  - `eta_seconds`: Estimated time remaining
  - `current_batch` / `total_batches`: Batch progress
  - `samples_processed` / `total_samples`: Sample progress

**Code Pattern:**
```python
current_time = time.time()
should_emit = (current_time - last_emit_time >= 2.0) or \
              (int(progress * 20) > int((batch_start / len(dataset)) * 20))
if should_emit:
    last_emit_time = current_time
    # ... emit progress
```

### 1.2 Graph Metrics (features_in_heap, heap_examples_count)
**Commits:** This update

Added heap statistics to enable "Examples Collected" and "Collection Rate" graphs.

**Backend Changes:**
1. **extraction_vectorized.py** - Added `get_stats()` method to `IncrementalTopKHeap`:
```python
def get_stats(self) -> Dict[str, int]:
    features_in_heap = len(self.heaps)
    heap_examples_count = sum(len(heap) for heap in self.heaps.values())
    return {
        "features_in_heap": features_in_heap,
        "heap_examples_count": heap_examples_count
    }
```

2. **extraction_service.py** - Include heap stats in progress emission:
```python
heap_stats = incremental_heap.get_stats()
emit_progress(
    data={
        # ... existing fields ...
        "features_in_heap": heap_stats["features_in_heap"],
        "heap_examples_count": heap_stats["heap_examples_count"],
    }
)
```

**Frontend Support:**
- Type already defined in `useExtractionWebSocket.ts`
- Handler passes through values defensively (only if defined)

### 1.3 Status Field Fix
**Commit:** `b2ed6f6`

Fixed "Unknown" status display in extraction progress UI.

**Root Cause:** Progress events were missing the `status` field.

**Fix:** Always include `status: ExtractionStatus.EXTRACTING.value` in progress emissions.

---

## 2. Batch Extraction Workflow

### 2.1 Sequential Batch Processing
**Commits:** `810cecd`, `d87039c`

Implemented sequential batch extraction where each job waits for the previous job's NLP analysis to complete.

**Flow:**
```
Job 1: Extract → Job 1: NLP → Job 2: Extract → Job 2: NLP → ...
```

**Backend Changes (extraction_service.py):**
- Only queue first job immediately
- Later jobs stay in QUEUED status
- Added `batch_id`, `batch_position`, `batch_total` to track batch membership

**Backend Changes (nlp_analysis_tasks.py):**
- Added `_start_next_batch_job()` helper function
- Called after NLP success to start next job in batch
- Also called after NLP failure (batch continues despite individual failures)

**Helper Function:**
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

### 2.2 Import Path Fix
**Commit:** `d87039c`

Fixed `ModuleNotFoundError: No module named 'src.models.extraction'`.

**Root Cause:** Wrong import path in nlp_analysis_tasks.py

**Fix:** Changed from `src.models.extraction` to `src.models.extraction_job`

---

## 3. Database & Schema Fixes

### 3.1 Nullable training_id
**Commit:** `13814cb`

Made `training_id` nullable in `extraction_jobs` and `features` tables to support SAE extractions without associated trainings (e.g., externally imported SAEs).

**Migration:** `j6k7l8m9n0o1_make_extraction_training_id_nullable.py`

**Changes:**
- Drop foreign key constraints (`extraction_jobs_training_id_fkey`, `features_training_id_fkey`)
- Alter column to `nullable=True`
- Idempotent design with existence checks

### 3.2 VRAM Detection Fix
**Commits:** `a8487f8`, `639dfb6`

Fixed batch size auto-calculation based on available GPU VRAM.

**Issue:** Batch size was being calculated incorrectly, causing OOM errors.

**Fix:** Corrected VRAM detection and memory-per-sample calculation.

---

## 4. Frontend Fixes

### 4.1 Model Name Display
**Commits:** `0a8a1ca`, `4751179`, `9923edc`

Fixed "Unknown Model" appearing for SAEs with missing `model_name`.

**Fix (ExtractionJobCard.tsx, SAECard.tsx):**
```typescript
const getModelDisplayName = (): string => {
  if (extraction.model_name) return extraction.model_name;
  if (extraction.model_id) {
    const model = models.find((m) => m.id === extraction.model_id);
    if (model) return model.name;
  }
  return 'Unknown Model';
};
```

### 4.2 Save as Template
**Commit:** `2bef750`

Added "Save as Template" button to extraction modal for quick template creation.

**Features:**
- Auto-generates template name from config (e.g., "10K samples, 100 examples, standard filtering")
- Stores SAE-specific settings in `extra_metadata`
- Success/error notification with auto-dismiss

---

## 5. Kubernetes Deployment

### 5.1 Simplified Migration Handling
**Commit:** `1b182c9`

Removed init container in favor of entrypoint-based migrations.

**Changes:**
- Removed 42-line db-migrate init container
- Added `SERVICE_TYPE=api` to backend container
- Migrations now run via docker-entrypoint.sh
- Updated image references to `hitsai/mistudio-*`

**Benefits:**
- Eliminates race conditions
- Consistent behavior across deployment modes
- Simpler deployment manifest

---

## 6. CI/CD Improvements

### 6.1 GitHub Actions Workflows
**Commits:** `aa4d691`, `bf1c901`, various fixes

**sync-to-clean.yml:**
- Syncs Onegaishimas/miStudio → hitsainet/miStudio
- Excludes sensitive files
- Force push to handle divergent history

**docker-images.yml:**
- Builds both backend and frontend images
- Pushes to Docker Hub (hitsai organization)
- Only runs on hitsainet/miStudio repository

### 6.2 Simplified Build Trigger
**Commit:** `bf1c901`

Changed from change-detection to always-build on push to main.

**Rationale:** Simpler, more reliable, ensures both images stay in sync.

---

## 7. Files Modified Summary

| Category | File | Changes |
|----------|------|---------|
| Backend | `extraction_vectorized.py` | Added `get_stats()` method |
| Backend | `extraction_service.py` | Graph metrics, time-based emission |
| Backend | `nlp_analysis_tasks.py` | Batch continuation, import fix |
| Backend | Alembic migration | Nullable training_id |
| Frontend | `useExtractionWebSocket.ts` | Defensive updates, type defs |
| Frontend | `ExtractionJobCard.tsx` | Model name fallback |
| Frontend | `SAECard.tsx` | Model name fallback |
| Frontend | `StartExtractionModal.tsx` | Save as Template |
| K8s | `mistudio-deployment.yaml` | Simplified migrations |
| CI/CD | `docker-images.yml` | Always build both images |
| CI/CD | `sync-to-clean.yml` | Repository sync |

---

## 8. Testing Verification

### 8.1 Graph Metrics
1. Start extraction job
2. Monitor WebSocket messages in browser console
3. Verify `features_in_heap` and `heap_examples_count` appear in progress events
4. Verify UI graphs update in real-time

### 8.2 Batch Extraction
1. Select multiple SAEs in extraction modal
2. Start batch extraction
3. Verify jobs process sequentially (Job 1 → NLP → Job 2 → ...)
4. Verify batch continues if one job's NLP fails

### 8.3 Deployment
1. Push changes to Onegaishimas/miStudio
2. Verify sync workflow pushes to hitsainet/miStudio
3. Verify docker-images.yml builds new images
4. Rollout restart K8s deployment
5. Verify pods healthy and graphs working

---

## 9. Commit History (Chronological)

```
d87039c fix: correct import path and start next batch job on NLP failure
92879a2 fix: emit progress updates every 2 seconds for responsive live metrics
c57ebc2 feat: add live metrics to extraction progress events
5fe8b50 fix: add missing Session import in nlp_analysis_tasks
810cecd feat: batch extractions wait for NLP completion before starting next job
b2ed6f6 fix: add missing status field to progress events
639dfb6 Fix batch size calculation for feature extraction
a8487f8 Fix VRAM detection for batch size auto-calculation
9923edc Remove unused useModelsStore import and models variable
4751179 Fix TypeScript error: remove model_id reference from ExtractionJobCard
1b182c9 fix(k8s): simplify deployment with entrypoint-based migrations
0a8a1ca fix(frontend): improve model name display for SAEs
7b8c74c Fix UnboundLocalError in extraction_service.py finally block
13814cb fix(db): make training_id nullable in extraction_jobs and features tables
bf1c901 ci: always build both Docker images on push to main
2bef750 feat(extraction): add 'Save as Template' button to extraction modal
```

---

**Document Version:** 1.0
**Last Updated:** 2026-01-21
