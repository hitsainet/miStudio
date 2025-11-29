# Bug Fix Plan: Extraction Architecture Refactor Issues

**Created:** 2025-11-28
**Scope:** P0-P3 bugs from multi-agent code review
**Estimated Time:** 3-4 hours total

---

## Executive Summary

This plan addresses 11 issues identified in the extraction architecture refactor code review. Issues range from critical bugs (P0) that will cause runtime failures to technical debt (P3) that should be addressed for maintainability.

---

## Phase 1: P0 Critical Bug (15 mins)

### Task 1.1: Validate Dataset Exists Before SAE Extraction

**File:** `backend/src/services/extraction_service.py`
**Lines:** 263-265
**Issue:** `start_extraction_for_sae()` only checks if `dataset_id` is provided, but doesn't verify the dataset exists or is ready.

**Implementation:**
```python
# After line 265, add dataset validation:
from src.models.dataset import Dataset

# Validate dataset exists and is ready
dataset_result = await self.db.execute(
    select(Dataset).where(Dataset.id == dataset_id)
)
dataset = dataset_result.scalar_one_or_none()
if not dataset:
    raise ValueError(f"Dataset {dataset_id} not found")
if dataset.status != 'ready':
    raise ValueError(f"Dataset {dataset_id} is not ready (status: {dataset.status})")
```

**Testing:** Manual test - try to start extraction with non-existent dataset ID

---

## Phase 2: P1 Bugs (45 mins)

### Task 2.1: Populate Dataset Name in Extraction Response

**File:** `backend/src/api/v1/endpoints/saes.py`
**Lines:** 576-577
**Issue:** `dataset_name=None` is hardcoded, should lookup from dataset_id.

**Implementation:**
```python
# In start_sae_extraction(), before building ExtractionStatusResponse:
from ....models.dataset import Dataset

# Lookup dataset name
dataset_name = None
if dataset_id:
    dataset_result = await db.execute(
        select(Dataset).where(Dataset.id == dataset_id)
    )
    dataset = dataset_result.scalar_one_or_none()
    if dataset:
        dataset_name = dataset.name

# Then use dataset_name in the response
```

---

### Task 2.2: Consolidate Duplicate Type Definitions

**Files:**
- `frontend/src/types/features.ts` (lines 45-78) - Keep this one (canonical)
- `frontend/src/api/saes.ts` (lines 178-196) - Remove and import from features.ts

**Implementation:**
1. In `saes.ts`, remove `SAEExtractionStatusResponse` interface
2. Import `ExtractionStatusResponse` from `../types/features`
3. Update function signatures to use `ExtractionStatusResponse`
4. Update any type mismatches (the types are nearly identical)

---

### Task 2.3: Revoke Celery Task on SAE Extraction Cancel

**File:** `backend/src/api/v1/endpoints/saes.py`
**Lines:** 661-666
**Issue:** Cancel only updates DB status, doesn't revoke the actual Celery task.

**Implementation:**
```python
# Before updating status to FAILED, add task revocation:
if extraction_job.celery_task_id:
    try:
        from ....core.celery_app import celery_app
        celery_app.control.revoke(
            extraction_job.celery_task_id,
            terminate=True,
            signal='SIGTERM'
        )
        logger.info(f"Revoked Celery task {extraction_job.celery_task_id}")
    except Exception as e:
        logger.error(f"Failed to revoke Celery task: {e}")
        # Continue anyway - will update database status
```

**Reference:** See pattern in `backend/src/api/v1/endpoints/models.py:863-871`

---

### Task 2.4: Pre-validate SAE Format Before Queuing Task

**File:** `backend/src/workers/extraction_tasks.py`
**Function:** `extract_features_from_sae_task()`
**Lines:** 157-161
**Issue:** Task doesn't verify SAE can be loaded before starting work.

**Implementation:**
```python
# After getting external_sae record, add format validation:
from src.ml.community_format import load_sae_auto_detect
from pathlib import Path

# Verify SAE can be loaded
try:
    sae_path = Path(external_sae.local_path)
    if not sae_path.exists():
        raise ValueError(f"SAE local path does not exist: {external_sae.local_path}")

    # Try loading to validate format
    # Note: This is a lightweight check, actual loading happens in service
    logger.info(f"SAE path validated: {sae_path}")
except Exception as e:
    logger.error(f"SAE validation failed: {e}")
    # Update status to failed
    extraction_job.status = ExtractionStatus.FAILED.value
    extraction_job.error_message = f"SAE validation failed: {str(e)}"
    db.commit()
    raise
```

---

## Phase 3: P2 Bugs (60 mins)

### Task 3.1: Display Error When SAE Fetch Fails

**File:** `frontend/src/components/extraction/StartExtractionModal.tsx`
**Lines:** 96-103
**Issue:** Silent catch with console.error, user sees empty dropdown.

**Implementation:**
```tsx
// Add state for SAE loading error
const [saeLoadError, setSaeLoadError] = useState<string | null>(null);

// Update useEffect:
getReadySAEs()
  .then((response) => {
    setSaes(response.data);
    setSaeLoadError(null);
  })
  .catch((err) => {
    console.error('Failed to fetch SAEs:', err);
    setSaes([]);
    setSaeLoadError('Failed to load SAEs. Please try again.');
  });

// Add error display in JSX (near SAE dropdown):
{saeLoadError && (
  <p className="text-xs text-red-400 mt-1">{saeLoadError}</p>
)}
```

---

### Task 3.2: Persist Error Message for SAE Extraction Failures

**File:** `backend/src/api/v1/endpoints/saes.py`
**Lines:** 589-593
**Issue:** Generic error, doesn't persist actual failure reason.

**Implementation:**
```python
except ValueError as e:
    raise HTTPException(400, str(e))
except Exception as e:
    error_message = f"Error starting extraction: {str(e)}"
    logger.error(error_message, exc_info=True)

    # Try to update the extraction job with error (if it was created)
    try:
        # Find recent extraction job for this SAE
        result = await db.execute(
            select(ExtractionJob)
            .where(ExtractionJob.external_sae_id == sae_id)
            .order_by(desc(ExtractionJob.created_at))
            .limit(1)
        )
        extraction_job = result.scalar_one_or_none()
        if extraction_job and extraction_job.status in [ExtractionStatus.QUEUED.value, ExtractionStatus.EXTRACTING.value]:
            extraction_job.status = ExtractionStatus.FAILED.value
            extraction_job.error_message = str(e)
            await db.commit()
    except Exception:
        pass  # Best effort

    raise HTTPException(500, error_message)
```

---

### Task 3.3: Add WebSocket Progress for SAE Extractions

**Files:**
- `backend/src/workers/websocket_emitter.py` - Add new function
- `backend/src/workers/extraction_tasks.py` - Use new function
- `frontend/src/hooks/useExtractionWebSocket.ts` - Subscribe to new channel

**Implementation:**

**Step 1:** Add `emit_sae_extraction_progress()` to websocket_emitter.py:
```python
def emit_sae_extraction_progress(
    sae_id: str,
    extraction_id: str,
    progress: float,
    status: str,
    message: str,
    features_extracted: Optional[int] = None,
    total_features: Optional[int] = None,
) -> bool:
    """
    Emit progress update for SAE feature extraction.

    Channel Convention:
        sae/{sae_id}/extraction
    """
    channel = f"sae/{sae_id}/extraction"
    data = {
        "sae_id": sae_id,
        "extraction_id": extraction_id,
        "progress": progress,
        "status": status,
        "message": message,
        "features_extracted": features_extracted,
        "total_features": total_features,
    }
    return emit_progress(channel, "sae:extraction", data)
```

**Step 2:** Call from `extract_features_from_sae_task()` at key points:
- Task start (progress=0, status="starting")
- During extraction (progress=X, status="extracting")
- On completion (progress=100, status="completed")
- On failure (status="failed")

**Step 3:** Frontend subscription (similar to existing extraction WebSocket hook)

---

### Task 3.4: Improve Training Dropdown Display

**File:** `frontend/src/components/extraction/StartExtractionModal.tsx`
**Lines:** 349-352
**Issue:** Shows only truncated ID, should show more context.

**Implementation:**
```tsx
// Update the option text to include model and dataset info:
{completedTrainings.map((training) => {
  const datasetName = getDatasetName(training.dataset_id);
  const modelId = training.model_id?.slice(0, 8) || 'Unknown';
  return (
    <option key={training.id} value={training.id}>
      {`${training.id.slice(0, 8)} - ${modelId} / ${datasetName}`}
    </option>
  );
})}
```

---

## Phase 4: P3 Technical Debt (45 mins)

### Task 4.1: Extract Shared Active Extraction Check Logic

**File:** `backend/src/services/extraction_service.py`
**Issue:** Duplicated active extraction check in `start_extraction()` and `start_extraction_for_sae()`

**Implementation:**
```python
async def _check_active_extraction(
    self,
    training_id: Optional[str] = None,
    sae_id: Optional[str] = None
) -> None:
    """
    Check if there's an active extraction for the given source.

    Raises:
        ValueError: If active extraction exists
    """
    query = select(ExtractionJob).where(
        ExtractionJob.status.in_([
            ExtractionStatus.QUEUED,
            ExtractionStatus.EXTRACTING
        ])
    )

    if training_id:
        query = query.where(ExtractionJob.training_id == training_id)
    elif sae_id:
        query = query.where(ExtractionJob.external_sae_id == sae_id)
    else:
        raise ValueError("Must specify either training_id or sae_id")

    query = query.order_by(desc(ExtractionJob.created_at)).limit(1)
    result = await self.db.execute(query)
    active_extraction = result.scalar_one_or_none()

    if active_extraction:
        # Existing Celery task verification logic here...
        pass
```

Then refactor both methods to call this shared function.

---

### Task 4.2: Add Unit Tests for SAE Extraction

**File:** `backend/tests/unit/test_sae_extraction.py` (new file)

**Test Cases:**
1. `test_start_extraction_for_sae_validates_dataset()`
2. `test_start_extraction_for_sae_validates_sae_ready()`
3. `test_start_extraction_for_sae_rejects_active_extraction()`
4. `test_get_extraction_status_for_sae_returns_none_if_not_found()`
5. `test_get_extraction_status_for_sae_returns_correct_status()`

**File:** `backend/tests/integration/test_sae_extraction_api.py` (new file)

**Test Cases:**
1. `test_start_sae_extraction_success()`
2. `test_start_sae_extraction_missing_dataset()`
3. `test_start_sae_extraction_sae_not_ready()`
4. `test_get_sae_extraction_status()`
5. `test_cancel_sae_extraction()`

---

## Implementation Order

1. **Phase 1** (P0) - Do first, critical bug
2. **Phase 2** (P1) - Do second, important functionality bugs
3. **Phase 3** (P2) - Do third, UX and architecture improvements
4. **Phase 4** (P3) - Do last, technical debt cleanup

---

## Verification Checklist

After implementation:

- [ ] `npx tsc --noEmit` passes
- [ ] `npm run build` succeeds
- [ ] Backend starts without errors
- [ ] Can list SAEs on SAEs page
- [ ] Can start extraction from Training source
- [ ] Can start extraction from SAE source with dataset selection
- [ ] Cancel extraction revokes Celery task
- [ ] Error messages display correctly
- [ ] Extraction progress updates via WebSocket (for SAE extractions)
- [ ] All new unit tests pass
- [ ] All new integration tests pass

---

## Files Modified Summary

| File | Changes |
|------|---------|
| `backend/src/services/extraction_service.py` | Dataset validation, extract shared logic |
| `backend/src/api/v1/endpoints/saes.py` | Dataset lookup, Celery revoke, error persistence |
| `backend/src/workers/extraction_tasks.py` | SAE format validation |
| `backend/src/workers/websocket_emitter.py` | New SAE extraction progress function |
| `frontend/src/components/extraction/StartExtractionModal.tsx` | Error display, training dropdown |
| `frontend/src/api/saes.ts` | Remove duplicate types |
| `frontend/src/types/features.ts` | (keep as canonical type source) |
| `backend/tests/unit/test_sae_extraction.py` | New test file |
| `backend/tests/integration/test_sae_extraction_api.py` | New test file |
