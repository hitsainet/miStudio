# Model Management Enhancement 01 - Architectural Refactoring Tasks

**Feature**: Architectural consistency and code quality improvements across Dataset and Model Management services
**Priority**: P1 (High - fixes critical issues and improves maintainability)
**Status**: Planning
**Created**: 2025-10-12
**Related Documents**:
- Architectural Review Report (completed 2025-10-12)
- 001_FTASKS|Dataset_Management.md (Phase 6 completed)

---

## Overview

This enhancement implements all recommendations from the comprehensive architectural review to improve:
- **Consistency**: Standardize patterns between datasets and models
- **Maintainability**: Eliminate code duplication
- **Reliability**: Fix critical bugs (file cleanup, hardcoded URLs)
- **Completeness**: Add missing features (dataset cancellation)
- **Quality**: Improve test coverage and error handling

**Scope**: 3 weeks of focused refactoring work
**Impact**: ~5,000 lines of code reviewed, ~800 lines to be modified/added

---

## Decision Points Required Before Implementation

### ⚠️ DECISIONS NEEDED - Please answer these questions:

#### **Decision 1: Database Session Strategy for Celery Workers**

**Background**: Dataset workers currently use async sessions (incorrect for Celery synchronous context), while model workers use sync sessions (correct).

**Options**:
- **A**: Migrate dataset workers to sync sessions (RECOMMENDED)
  - ✅ Correct approach for Celery
  - ✅ Matches model workers pattern
  - ⚠️ Requires refactoring all async/await code in dataset_tasks.py
  - ⚠️ Need to test thoroughly to avoid breaking existing functionality

- **B**: Keep async sessions for datasets, add compatibility layer
  - ✅ Less code change
  - ❌ Maintains architectural inconsistency
  - ❌ Potential deadlock issues remain

**Your Choice**: ✅ **A - Migrate dataset workers to sync sessions**

---

#### **Decision 2: UUID Migration for Model Primary Keys**

**Background**: Datasets use UUID primary keys (16 bytes, type-safe), models use String "m_{uuid}" (variable length, less efficient).

**Options**:
- **A**: Migrate models to UUID immediately (RECOMMENDED for long-term)
  - ✅ Consistent with datasets
  - ✅ Better database performance
  - ✅ Stronger type safety
  - ⚠️ Requires database migration
  - ⚠️ Breaking change for API (need versioning or migration period)
  - ⚠️ Frontend needs UUID handling updates

- **B**: Keep String IDs for now, add to backlog for v2.0
  - ✅ No breaking changes
  - ✅ Can focus on other fixes first
  - ❌ Maintains inconsistency
  - ❌ Technical debt accumulates

- **C**: Create abstraction layer to support both (hybrid approach)
  - ✅ Gradual migration possible
  - ❌ Most complex option
  - ❌ Temporary duplication of logic

**Your Choice**: ✅ **A - Migrate models to UUID immediately**

---

#### **Decision 3: File Deletion Strategy**

**Background**: Datasets do synchronous file deletion (blocking), models should use background tasks.

**Options**:
- **A**: Standardize on background tasks for all deletions (RECOMMENDED)
  - ✅ Non-blocking API responses
  - ✅ Better error handling and retry logic
  - ✅ Consistent pattern
  - ⚠️ Requires refactoring dataset deletion

- **B**: Keep inline deletion for datasets, background for models
  - ✅ Less change required
  - ❌ Inconsistent patterns
  - ❌ Datasets block on large file deletions

**Your Choice**: ✅ **A - Standardize on background tasks for all deletions**

---

#### **Decision 4: Progress Monitoring Implementation**

**Background**: Models have sophisticated file-based progress tracking, datasets use hardcoded percentages.

**Options**:
- **A**: Implement shared ProgressMonitor for both (RECOMMENDED)
  - ✅ Accurate progress for both services
  - ✅ Consistent UX
  - ✅ Reusable for future features
  - ⚠️ More code to write and test

- **B**: Keep current dataset progress, only use monitor for models
  - ✅ Less work
  - ❌ Inconsistent UX (datasets show inaccurate progress)
  - ❌ Users confused by different progress behaviors

**Your Choice**: ✅ **A - Implement shared ProgressMonitor for both**

---

#### **Decision 5: Test Coverage Priority**

**Background**: Current coverage ~60%, target is >80%. We have 3 weeks of work planned.

**Options**:
- **A**: Write tests as we go (Test-Driven Development approach)
  - ✅ Ensures nothing breaks during refactoring
  - ✅ Better long-term quality
  - ⚠️ Takes longer (adds ~30% time)
  - Total time: ~4 weeks instead of 3

- **B**: Refactor first, add tests at end
  - ✅ Faster refactoring
  - ⚠️ Higher risk of breaking things
  - ⚠️ Tests might not catch all issues

- **C**: Focus on critical path tests only
  - ✅ Balanced approach
  - ✅ Tests for new code and critical fixes
  - ⚠️ Coverage might be ~70% instead of 80%

**Your Choice**: ✅ **A - Write tests as we go (TDD approach)**

---

#### **Decision 6: Breaking Changes and Versioning**

**Background**: Some fixes (like UUID migration) are breaking changes for the API.

**Options**:
- **A**: Implement breaking changes in new API version (v2)
  - ✅ Clean separation
  - ✅ Both versions can coexist
  - ⚠️ Need to maintain two API versions temporarily
  - ⚠️ More complex deployment

- **B**: Apply breaking changes directly (acceptable for pre-release)
  - ✅ Simpler implementation
  - ✅ Single codebase
  - ⚠️ Current clients/UI need immediate updates
  - ⚠️ Can't rollback easily

- **C**: Avoid breaking changes entirely (keep String IDs, add new endpoints)
  - ✅ No breaking changes
  - ❌ Technical debt remains
  - ❌ Inconsistent API design

**Your Choice**: ✅ **A - Implement breaking changes in new API version (v2)**

---

## DECISIONS FINALIZED ✅

**All decisions have been made - proceeding with FULL REFACTORING path:**
1. ✅ Migrate dataset workers to sync sessions
2. ✅ Migrate models to UUID immediately
3. ✅ Standardize on background tasks for all deletions
4. ✅ Implement shared ProgressMonitor for both
5. ✅ Write tests as we go (TDD approach)
6. ✅ Implement breaking changes in new API version (v2)

**Timeline**: 17-18 days (including TDD)
**Risk Level**: High initially, but with comprehensive testing and rollback plans
**Benefits**: Best long-term architecture, maximum consistency, >80% test coverage

---

## Phase 1: Critical Fixes (Week 1 - Priority 1)

**Goal**: Fix bugs that break functionality or cause data loss
**Duration**: 5 days
**Risk Level**: Medium (touching core functionality)

### Task 1.1: Create Shared WebSocket Emitter Utility

**Priority**: P0 (Critical)
**Files Modified**:
- [ ] **NEW**: `backend/src/workers/websocket_emitter.py` (create shared utility)
- [ ] **MODIFY**: `backend/src/workers/dataset_tasks.py` (replace emit_progress)
- [ ] **MODIFY**: `backend/src/workers/model_tasks.py` (replace send_progress_update)

**Subtasks**:
- [ ] 1.1.1 Create `backend/src/workers/websocket_emitter.py`
  - [ ] Implement `emit_progress()` function with standardized signature
  - [ ] Use `httpx` (consistent with datasets)
  - [ ] Use `settings.websocket_emit_url` from config (no hardcoding)
  - [ ] Add error handling with logging
  - [ ] Add docstrings with usage examples

- [ ] 1.1.2 Update `dataset_tasks.py` to use shared emitter
  - [ ] Import `emit_progress` from `websocket_emitter`
  - [ ] Replace `DatasetTask.emit_progress()` method calls
  - [ ] Update all call sites (lines 47-80, 178, 214, 279, 324, 405, 472, 538)
  - [ ] Remove old `emit_progress` method from `DatasetTask` class
  - [ ] Test dataset download progress updates

- [ ] 1.1.3 Update `model_tasks.py` to use shared emitter
  - [ ] Import `emit_progress` from `websocket_emitter`
  - [ ] Replace `send_progress_update()` function calls
  - [ ] Update all call sites (lines 146-176, 268, 309, 352, 706)
  - [ ] Remove old `send_progress_update` function
  - [ ] Fix hardcoded URL at line 161
  - [ ] Test model download progress updates

- [ ] 1.1.4 Write unit tests
  - [ ] **NEW**: `backend/tests/unit/test_websocket_emitter.py`
  - [ ] Test successful emission
  - [ ] Test network failure handling
  - [ ] Test timeout handling
  - [ ] Test different resource types (datasets vs models)

**Success Criteria**:
- ✅ No hardcoded URLs in worker code
- ✅ Both services use identical WebSocket emission pattern
- ✅ All progress updates working in UI
- ✅ Unit tests passing

**Estimated Time**: 1 day

---

### Task 1.2: Standardize Database Sessions in Workers

**Priority**: P0 (Critical)
**Files Modified**:
- [ ] **NEW**: `backend/src/workers/base_task.py` (create base task class)
- [ ] **MODIFY**: `backend/src/workers/dataset_tasks.py` (migrate to sync sessions)
- [ ] **MODIFY**: `backend/src/workers/model_tasks.py` (use base class)

**Subtasks**:
- [ ] 1.2.1 Create base task class
  - [ ] **NEW**: `backend/src/workers/base_task.py`
  - [ ] Define `DatabaseTask` base class
  - [ ] Implement `get_db_session()` method (returns sync session)
  - [ ] Implement `with_db_session()` decorator pattern
  - [ ] Add session cleanup logic
  - [ ] Add docstrings and usage examples

- [ ] 1.2.2 Migrate dataset tasks to sync sessions (⚠️ DEPENDS ON DECISION 1)
  - [ ] Update `DatasetTask` to inherit from `DatabaseTask`
  - [ ] Replace all `async def` with `def` in task functions
  - [ ] Replace `await db.execute()` with synchronous `db.execute()`
  - [ ] Replace `await db.commit()` with `db.commit()`
  - [ ] Replace `await db.refresh()` with `db.refresh()`
  - [ ] Remove `asyncio.new_event_loop()` usage
  - [ ] Update `download_dataset_task` (lines 116-324)
  - [ ] Update `tokenize_dataset_task` (lines 327-664)
  - [ ] Test all dataset operations thoroughly

- [ ] 1.2.3 Update model tasks to use base class
  - [ ] Update `download_and_load_model` to use `DatabaseTask`
  - [ ] Update `extract_activations` to use `DatabaseTask`
  - [ ] Update `cancel_download` to use `DatabaseTask`
  - [ ] Update `delete_model_files` to use `DatabaseTask`
  - [ ] Remove duplicate session creation code

- [ ] 1.2.4 Write integration tests
  - [ ] **NEW**: `backend/tests/integration/test_worker_sessions.py`
  - [ ] Test session creation and cleanup
  - [ ] Test concurrent task execution
  - [ ] Test transaction rollback on errors
  - [ ] Test session isolation between tasks

**Success Criteria**:
- ✅ All Celery tasks use synchronous database sessions
- ✅ No async/await in Celery tasks
- ✅ Session cleanup happens properly
- ✅ No deadlocks or connection leaks
- ✅ All existing functionality working

**Estimated Time**: 2 days

**⚠️ RISK**: This is a significant refactoring. Recommend thorough testing after completion.

---

### Task 1.3: Fix Model File Cleanup

**Priority**: P0 (Critical - prevents disk space leaks)
**Files Modified**:
- [ ] **MODIFY**: `backend/src/services/model_service.py` (update delete_model)
- [ ] **MODIFY**: `backend/src/api/v1/endpoints/models.py` (trigger cleanup task)
- [ ] **MODIFY**: `backend/src/workers/model_tasks.py` (ensure task exists)

**Subtasks**:
- [ ] 1.3.1 Update `ModelService.delete_model()` (lines 328-348)
  - [ ] Change return type to `dict` instead of `bool`
  - [ ] Capture `file_path` and `quantized_path` before deletion
  - [ ] Return dict with deletion status and file paths
  - [ ] Add `cleanup_files` parameter (default True)
  - [ ] Update docstring

- [ ] 1.3.2 Update models API endpoint (lines 239-260)
  - [ ] Import `delete_model_files` task
  - [ ] Call service method and get file paths
  - [ ] Queue `delete_model_files.delay()` after successful deletion
  - [ ] Pass model_id, file_path, quantized_path to task
  - [ ] Add logging for queued cleanup
  - [ ] Handle case where deletion succeeds but cleanup queueing fails

- [ ] 1.3.3 Verify `delete_model_files` task (lines 392-438)
  - [ ] Check task is properly registered with Celery
  - [ ] Verify it handles missing paths gracefully
  - [ ] Ensure proper error logging
  - [ ] Test with valid and invalid paths

- [ ] 1.3.4 Write integration tests
  - [ ] **NEW**: `backend/tests/integration/test_model_cleanup.py`
  - [ ] Test model deletion triggers file cleanup
  - [ ] Test cleanup handles missing files
  - [ ] Test cleanup with only file_path (no quantized_path)
  - [ ] Test cleanup with both paths
  - [ ] Verify files actually deleted from disk

**Success Criteria**:
- ✅ Model deletion triggers background file cleanup
- ✅ Files removed from disk after deletion
- ✅ No disk space leaks
- ✅ Cleanup works even if some paths are None
- ✅ Tests verify cleanup behavior

**Estimated Time**: 1 day

---

### Task 1.4: Standardize HTTP Client Library

**Priority**: P1 (Important for consistency)
**Files Modified**:
- [ ] **MODIFY**: `backend/src/workers/model_tasks.py` (replace requests with httpx)
- [ ] **MODIFY**: `backend/pyproject.toml` (update dependencies)
- [ ] **MODIFY**: `backend/requirements.txt` (update dependencies)

**Subtasks**:
- [ ] 1.4.1 Replace `requests` with `httpx` in model_tasks.py
  - [ ] Remove `import requests` (line 157)
  - [ ] Add `import httpx` at top
  - [ ] Replace `requests.post()` with `httpx.Client().post()`
  - [ ] Update error handling to match httpx exceptions
  - [ ] Verify all WebSocket emission calls work

- [ ] 1.4.2 Update dependencies (if requests not used elsewhere)
  - [ ] Check if `requests` used in other files: `grep -r "import requests" backend/src/`
  - [ ] If not used elsewhere, remove from pyproject.toml
  - [ ] Ensure `httpx` in dependencies
  - [ ] Run `poetry lock` to update lockfile
  - [ ] Update requirements.txt

- [ ] 1.4.3 Test all HTTP calls
  - [ ] Test WebSocket emission from model tasks
  - [ ] Test error handling
  - [ ] Test timeout behavior
  - [ ] Verify no regression in functionality

**Success Criteria**:
- ✅ All workers use `httpx` consistently
- ✅ No `requests` library in worker code
- ✅ Dependencies updated
- ✅ All HTTP calls working

**Estimated Time**: 0.5 days

---

### Task 1.5: Integration Testing and Validation

**Priority**: P0 (Critical - verify nothing broke)
**Files Modified**:
- [ ] **NEW**: `backend/tests/integration/test_critical_fixes.py`

**Subtasks**:
- [ ] 1.5.1 Test complete dataset workflow
  - [ ] Download dataset from HuggingFace
  - [ ] Verify progress updates via WebSocket
  - [ ] Verify database session handling
  - [ ] Tokenize dataset
  - [ ] Delete dataset and verify file cleanup

- [ ] 1.5.2 Test complete model workflow
  - [ ] Download model from HuggingFace
  - [ ] Verify progress updates via WebSocket
  - [ ] Verify database session handling
  - [ ] Delete model and verify file cleanup triggered
  - [ ] Verify files actually deleted

- [ ] 1.5.3 Test concurrent operations
  - [ ] Download dataset and model simultaneously
  - [ ] Verify no session conflicts
  - [ ] Verify both complete successfully

- [ ] 1.5.4 Manual UI testing
  - [ ] Test dataset download in UI
  - [ ] Test model download in UI
  - [ ] Verify progress bars working
  - [ ] Test deletion from UI
  - [ ] Check browser console for errors

**Success Criteria**:
- ✅ All integration tests passing
- ✅ Manual UI testing successful
- ✅ No regressions from Phase 1 changes
- ✅ Performance acceptable

**Estimated Time**: 1 day

---

## Phase 2: Consistency Improvements (Week 2 - Priority 2)

**Goal**: Add missing features and improve consistency
**Duration**: 5 days
**Risk Level**: Low-Medium

### Task 2.1: Add Dataset Cancellation Support

**Priority**: P1 (Important for UX parity)
**Files Modified**:
- [ ] **MODIFY**: `backend/src/workers/dataset_tasks.py` (add cancel task)
- [ ] **MODIFY**: `backend/src/api/v1/endpoints/datasets.py` (add cancel endpoint)
- [ ] **MODIFY**: `frontend/src/stores/datasetsStore.ts` (add cancel action)
- [ ] **MODIFY**: `frontend/src/api/datasets.ts` (add cancel API call)
- [ ] **MODIFY**: `frontend/src/components/datasets/DatasetCard.tsx` (add cancel button)

**Subtasks**:
- [ ] 2.1.1 Create cancel task in dataset_tasks.py
  - [ ] Add `cancel_dataset_download()` task (follow model pattern)
  - [ ] Revoke Celery task if task_id provided
  - [ ] Update dataset status to ERROR with "Cancelled by user"
  - [ ] Clean up partial download files
  - [ ] Emit WebSocket notification
  - [ ] Add comprehensive error handling
  - [ ] Add logging

- [ ] 2.1.2 Add cancel API endpoint (datasets.py)
  - [ ] Add `@router.delete("/{dataset_id}/cancel")` endpoint
  - [ ] Verify dataset exists
  - [ ] Check status is cancellable (DOWNLOADING or PROCESSING)
  - [ ] Call cancel task synchronously
  - [ ] Return cancellation status
  - [ ] Add error handling for non-cancellable states

- [ ] 2.1.3 Update frontend API client
  - [ ] Add `cancelDatasetDownload(datasetId: string)` function
  - [ ] Use DELETE method
  - [ ] Handle errors appropriately

- [ ] 2.1.4 Update datasetsStore
  - [ ] Add `cancelDatasetDownload` action
  - [ ] Call API endpoint
  - [ ] Update local state on success
  - [ ] Handle errors with user feedback

- [ ] 2.1.5 Update UI component
  - [ ] Check if DatasetCard component exists (might be DatasetList)
  - [ ] Add "Cancel" button (only show for DOWNLOADING/PROCESSING status)
  - [ ] Wire up to store action
  - [ ] Disable button during cancellation
  - [ ] Show loading state

- [ ] 2.1.6 Write tests
  - [ ] **NEW**: `backend/tests/unit/test_dataset_cancellation.py`
  - [ ] **NEW**: `backend/tests/integration/test_dataset_cancel_flow.py`
  - [ ] **NEW**: `frontend/src/stores/datasetsStore.test.ts` (add cancel test)
  - [ ] Test successful cancellation
  - [ ] Test cancellation of non-cancellable dataset
  - [ ] Test UI button states

**Success Criteria**:
- ✅ Users can cancel dataset downloads from UI
- ✅ Partial files cleaned up after cancellation
- ✅ Status updated to ERROR with clear message
- ✅ Feature parity with model cancellation
- ✅ Tests passing

**Estimated Time**: 2 days

---

### Task 2.2: Extract Frontend Shared Utilities

**Priority**: P1 (Reduces duplication)
**Files Modified**:
- [ ] **NEW**: `frontend/src/api/client.ts` (shared API client)
- [ ] **NEW**: `frontend/src/hooks/usePolling.ts` (shared polling hook)
- [ ] **MODIFY**: `frontend/src/api/datasets.ts` (use shared client)
- [ ] **MODIFY**: `frontend/src/api/models.ts` (use shared client)
- [ ] **MODIFY**: `frontend/src/stores/datasetsStore.ts` (use polling hook)
- [ ] **MODIFY**: `frontend/src/stores/modelsStore.ts` (use polling hook)
- [ ] **OBSOLETE**: Remove duplicate `fetchAPI` from datasets.ts and models.ts

**Subtasks**:
- [ ] 2.2.1 Create shared API client (frontend/src/api/client.ts)
  - [ ] Define `API_V1_BASE` constant
  - [ ] Create `APIError` class with status and detail
  - [ ] Implement `fetchAPI<T>()` function
  - [ ] Add authentication header injection
  - [ ] Add error handling with structured errors
  - [ ] Handle 204 No Content responses
  - [ ] Create `buildQueryString()` helper
  - [ ] Add JSDoc comments with examples

- [ ] 2.2.2 Update datasets.ts to use shared client
  - [ ] Import `fetchAPI` and `buildQueryString` from `./client`
  - [ ] Remove duplicate `fetchAPI` function (lines 19-49)
  - [ ] Update `getDatasets()` to use `buildQueryString`
  - [ ] Update all API functions to use shared `fetchAPI`
  - [ ] Test all dataset API calls

- [ ] 2.2.3 Update models.ts to use shared client
  - [ ] Import `fetchAPI` and `buildQueryString` from `./client`
  - [ ] Remove duplicate `fetchAPI` function (lines 24-57)
  - [ ] Update `getModels()` to use `buildQueryString`
  - [ ] Update all API functions to use shared `fetchAPI`
  - [ ] Test all model API calls

- [ ] 2.2.4 Create shared polling hook (frontend/src/hooks/usePolling.ts)
  - [ ] Define `PollingConfig` interface
  - [ ] Implement `usePolling()` hook
  - [ ] Add interval management with useRef
  - [ ] Add terminal state detection
  - [ ] Add max polls timeout
  - [ ] Add cleanup on unmount
  - [ ] Add comprehensive logging
  - [ ] Add JSDoc with usage examples

- [ ] 2.2.5 Update datasetsStore to use polling hook
  - [ ] Remove inline polling logic (lines 116-157)
  - [ ] Import and use `usePolling` hook
  - [ ] Configure for datasets (terminalStates: ['ready', 'error'])
  - [ ] Set maxPolls: 50 (25 seconds)
  - [ ] Test dataset download with polling

- [ ] 2.2.6 Update modelsStore to use polling hook
  - [ ] Remove inline polling logic (lines 123-164)
  - [ ] Import and use `usePolling` hook
  - [ ] Configure for models (terminalStates: ['ready', 'error'])
  - [ ] Set maxPolls: 100 (50 seconds)
  - [ ] Test model download with polling

- [ ] 2.2.7 Write tests
  - [ ] **NEW**: `frontend/src/api/client.test.ts`
  - [ ] **NEW**: `frontend/src/hooks/usePolling.test.ts`
  - [ ] Test fetchAPI success and error cases
  - [ ] Test buildQueryString with various inputs
  - [ ] Test polling hook start/stop behavior
  - [ ] Test polling termination conditions

**Success Criteria**:
- ✅ No duplicate `fetchAPI` functions
- ✅ Both stores use shared polling hook
- ✅ ~150 lines of duplicate code removed
- ✅ All API calls and polling working
- ✅ Tests passing

**Estimated Time**: 2 days

---

### Task 2.3: Implement File Deletion Strategy (⚠️ DEPENDS ON DECISION 3)

**Priority**: P1 (Consistency)
**Files Modified**:
- [ ] **MODIFY**: `backend/src/services/dataset_service.py` (update delete_dataset)
- [ ] **MODIFY**: `backend/src/workers/dataset_tasks.py` (add delete task if choosing A)
- [ ] **MODIFY**: `backend/src/api/v1/endpoints/datasets.py` (update endpoint)

**Subtasks** (IF DECISION 3 = A: Background tasks for all):
- [ ] 2.3.1 Create dataset file cleanup task
  - [ ] Add `delete_dataset_files()` task to dataset_tasks.py
  - [ ] Follow same pattern as `delete_model_files()`
  - [ ] Handle raw_path and tokenized_path
  - [ ] Add logging and error handling

- [ ] 2.3.2 Update DatasetService.delete_dataset()
  - [ ] Change return type to dict
  - [ ] Capture file paths before deletion
  - [ ] Return paths for cleanup
  - [ ] Remove inline file deletion logic

- [ ] 2.3.3 Update datasets API endpoint
  - [ ] Queue `delete_dataset_files.delay()` after deletion
  - [ ] Pass file paths to task
  - [ ] Add logging

- [ ] 2.3.4 Write tests
  - [ ] Test dataset deletion triggers file cleanup
  - [ ] Test cleanup task handles missing files
  - [ ] Verify files actually deleted

**Subtasks** (IF DECISION 3 = B: Keep current):
- [ ] 2.3.1 Verify current dataset deletion works correctly
- [ ] 2.3.2 Add tests for current behavior
- [ ] 2.3.3 Document the inconsistency for future refactoring

**Success Criteria** (for A):
- ✅ Both services use background task deletion
- ✅ API endpoints return immediately
- ✅ Files cleaned up asynchronously
- ✅ Tests passing

**Estimated Time**: 1 day (if A), 0.5 days (if B)

---

### Task 2.4: Integration Testing Phase 2

**Priority**: P0 (Validate changes)
**Files Modified**:
- [ ] **NEW**: `backend/tests/integration/test_phase2_features.py`
- [ ] **NEW**: `frontend/src/__tests__/integration/phase2.test.ts`

**Subtasks**:
- [ ] 2.4.1 Test dataset cancellation end-to-end
  - [ ] Start dataset download
  - [ ] Cancel from API
  - [ ] Verify status updated
  - [ ] Verify files cleaned up
  - [ ] Test UI cancellation button

- [ ] 2.4.2 Test frontend utilities
  - [ ] Test shared API client with multiple endpoints
  - [ ] Test polling hook behavior
  - [ ] Test error handling

- [ ] 2.4.3 Test file deletion (based on Decision 3)
  - [ ] Test dataset deletion
  - [ ] Test model deletion
  - [ ] Verify files removed

- [ ] 2.4.4 Manual testing
  - [ ] Cancel dataset download from UI
  - [ ] Cancel model download from UI
  - [ ] Delete datasets and models
  - [ ] Verify no UI regressions

**Success Criteria**:
- ✅ All Phase 2 features working
- ✅ Integration tests passing
- ✅ Manual testing successful
- ✅ No regressions

**Estimated Time**: 0.5 days

---

## Phase 3: Nice-to-Have Improvements (Week 3+ - Priority 3)

**Goal**: Long-term quality improvements
**Duration**: 5 days
**Risk Level**: Low

### Task 3.1: Create Shared Progress Monitor (⚠️ DEPENDS ON DECISION 4)

**Priority**: P2 (Nice-to-have)
**Files Modified** (IF DECISION 4 = A):
- [ ] **NEW**: `backend/src/workers/progress_monitor.py` (shared monitor)
- [ ] **MODIFY**: `backend/src/workers/model_tasks.py` (use shared monitor)
- [ ] **MODIFY**: `backend/src/workers/dataset_tasks.py` (use shared monitor)
- [ ] **OBSOLETE**: Remove `DownloadProgressMonitor` class from model_tasks.py

**Subtasks** (IF DECISION 4 = A):
- [ ] 3.1.1 Create shared progress monitor
  - [ ] **NEW**: `backend/src/workers/progress_monitor.py`
  - [ ] Implement `get_directory_size()` helper
  - [ ] Implement `ProgressMonitor` class
  - [ ] Add configurable check interval
  - [ ] Add configurable max progress
  - [ ] Add callback pattern for updates
  - [ ] Add comprehensive logging
  - [ ] Add docstrings

- [ ] 3.1.2 Update model tasks to use shared monitor
  - [ ] Remove `DownloadProgressMonitor` class (lines 57-144)
  - [ ] Import `ProgressMonitor` from shared module
  - [ ] Update `download_and_load_model` to use new monitor
  - [ ] Test model download progress

- [ ] 3.1.3 Update dataset tasks to use shared monitor
  - [ ] Remove hardcoded progress updates
  - [ ] Import `ProgressMonitor` from shared module
  - [ ] Estimate dataset size (use HF Hub API if available)
  - [ ] Integrate monitor into `download_dataset_task`
  - [ ] Test dataset download progress

- [ ] 3.1.4 Write tests
  - [ ] **NEW**: `backend/tests/unit/test_progress_monitor.py`
  - [ ] Test size calculation
  - [ ] Test progress calculation
  - [ ] Test callback invocation
  - [ ] Test start/stop behavior
  - [ ] Test max progress cap

**Subtasks** (IF DECISION 4 = B):
- [ ] 3.1.1 Document current progress monitoring difference
- [ ] 3.1.2 Add to technical debt backlog

**Success Criteria** (for A):
- ✅ Both services show accurate file-based progress
- ✅ Consistent progress UX
- ✅ ~100 lines of duplicate code removed
- ✅ Tests passing

**Estimated Time**: 2 days (if A), 0.5 days (if B)

---

### Task 3.2: Improve Search Consistency

**Priority**: P2 (Small improvement)
**Files Modified**:
- [ ] **MODIFY**: `backend/src/services/model_service.py` (update list_models)

**Subtasks**:
- [ ] 3.2.1 Add repo_id search to models (lines 148-150)
  - [ ] Change search filter to use `or_()` like datasets
  - [ ] Add `Model.repo_id.ilike(f"%{search}%")` to search
  - [ ] Test search by name
  - [ ] Test search by repo_id
  - [ ] Test search with no results

- [ ] 3.2.2 Write tests
  - [ ] **UPDATE**: `backend/tests/unit/test_model_service.py`
  - [ ] Test search by name
  - [ ] Test search by repo_id
  - [ ] Test search with partial matches

**Success Criteria**:
- ✅ Users can search models by repo_id
- ✅ Consistent with dataset search behavior
- ✅ Tests passing

**Estimated Time**: 0.5 days

---

### Task 3.3: Fix Type Hint Inconsistencies

**Priority**: P3 (Low - cosmetic)
**Files Modified**:
- [ ] **MODIFY**: `backend/src/services/dataset_service.py`
- [ ] **MODIFY**: `backend/src/services/model_service.py`

**Subtasks**:
- [ ] 3.3.1 Standardize on lowercase tuple (Python 3.10+ style)
  - [ ] Replace `Tuple` with `tuple` in dataset_service.py
  - [ ] Replace `Tuple` with `tuple` in model_service.py
  - [ ] Remove `from typing import Tuple` imports if unused
  - [ ] Run mypy to verify type checking still works

- [ ] 3.3.2 Check for other typing inconsistencies
  - [ ] Scan for `List` vs `list`
  - [ ] Scan for `Dict` vs `dict`
  - [ ] Scan for `Optional` vs `| None`
  - [ ] Update to modern style consistently

**Success Criteria**:
- ✅ Consistent type hint style
- ✅ Mypy checks passing
- ✅ No runtime changes

**Estimated Time**: 0.5 days

---

### Task 3.4: UUID Migration Plan (⚠️ DEPENDS ON DECISION 2)

**Priority**: P2 (If Decision 2 = A), P3 (If Decision 2 = C)
**Files Modified** (IF DECISION 2 = A):
- [ ] **NEW**: `backend/alembic/versions/xxxx_standardize_model_ids.py` (migration)
- [ ] **MODIFY**: `backend/src/models/model.py`
- [ ] **MODIFY**: `backend/src/services/model_service.py`
- [ ] **MODIFY**: `backend/src/api/v1/endpoints/models.py`
- [ ] **MODIFY**: `frontend/src/types/model.ts`
- [ ] **MODIFY**: `frontend/src/api/models.ts`
- [ ] **MODIFY**: All frontend components using model IDs

**Subtasks** (IF DECISION 2 = A - Immediate migration):
- [ ] 3.4.1 Create Alembic migration
  - [ ] Create migration file with upgrade/downgrade
  - [ ] Add UUID column
  - [ ] Generate UUIDs for existing records
  - [ ] Drop old primary key
  - [ ] Create new UUID primary key
  - [ ] Update foreign keys if any exist
  - [ ] Test migration on development database

- [ ] 3.4.2 Update backend model
  - [ ] Change `id` column to UUID type
  - [ ] Update `generate_model_id()` to return UUID
  - [ ] Update all type hints (str -> UUID)
  - [ ] Update docstrings

- [ ] 3.4.3 Update backend service
  - [ ] Update `get_model()` parameter type
  - [ ] Update all methods to use UUID
  - [ ] Update return type hints

- [ ] 3.4.4 Update backend API
  - [ ] Update path parameters to UUID type
  - [ ] Update request/response schemas
  - [ ] Test all endpoints

- [ ] 3.4.5 Update frontend types
  - [ ] Change model.id to string (UUID string representation)
  - [ ] Update all interfaces

- [ ] 3.4.6 Update frontend components
  - [ ] Update all components using model IDs
  - [ ] Update stores
  - [ ] Update API client
  - [ ] Test UI thoroughly

- [ ] 3.4.7 Write comprehensive tests
  - [ ] Test migration upgrade
  - [ ] Test migration downgrade
  - [ ] Test all API endpoints
  - [ ] Test frontend UUID handling

**Subtasks** (IF DECISION 2 = B - Defer to v2.0):
- [ ] 3.4.1 Document migration plan
- [ ] 3.4.2 Add to v2.0 backlog
- [ ] 3.4.3 Create technical debt ticket

**Subtasks** (IF DECISION 2 = C - Hybrid approach):
- [ ] 3.4.1 Create abstraction layer
- [ ] 3.4.2 Support both ID formats
- [ ] 3.4.3 Plan gradual migration
- [ ] 3.4.4 Add deprecation warnings

**Success Criteria** (for A):
- ✅ All models use UUID primary keys
- ✅ Consistent with datasets
- ✅ Migration reversible
- ✅ No data loss
- ✅ All tests passing
- ✅ Frontend working with UUIDs

**Estimated Time**: 2 days (if A), 0.5 days (if B or C)

---

### Task 3.5: Comprehensive Test Coverage (⚠️ DEPENDS ON DECISION 5)

**Priority**: Varies based on Decision 5
**Files Modified**: Many test files

**Subtasks** (IF DECISION 5 = A - TDD approach):
- Already covered in individual tasks above
- Tests written before or during implementation
- Continuous test execution during refactoring

**Subtasks** (IF DECISION 5 = B - Tests at end):
- [ ] 3.5.1 Write unit tests for all modified code
  - [ ] Backend service tests
  - [ ] Backend worker tests
  - [ ] Frontend store tests
  - [ ] Frontend hook tests

- [ ] 3.5.2 Write integration tests
  - [ ] End-to-end workflow tests
  - [ ] Cross-service tests

- [ ] 3.5.3 Measure coverage
  - [ ] Run coverage report
  - [ ] Identify gaps
  - [ ] Add tests for uncovered code

**Subtasks** (IF DECISION 5 = C - Critical path only):
- [ ] 3.5.1 Write tests for new features
  - [ ] Dataset cancellation
  - [ ] Shared utilities
  - [ ] Progress monitor

- [ ] 3.5.2 Write tests for bug fixes
  - [ ] File cleanup
  - [ ] WebSocket emission
  - [ ] Session handling

- [ ] 3.5.3 Add integration tests
  - [ ] Happy path tests
  - [ ] Error handling tests

**Success Criteria**:
- A: >80% coverage
- B: >80% coverage (but risky approach)
- C: ~70% coverage (focused on critical code)

**Estimated Time**:
- A: Distributed across tasks (already included)
- B: 2 days at end
- C: 1 day distributed

---

### Task 3.6: Documentation Updates

**Priority**: P2
**Files Modified**:
- [ ] **UPDATE**: `0xcc/tasks/002_FTASKS|Model_Management-ENH_01.md` (this file)
- [ ] **NEW**: `0xcc/docs/Architecture_Patterns.md` (new patterns guide)
- [ ] **UPDATE**: `backend/README.md` (if exists)
- [ ] **UPDATE**: `frontend/README.md` (if exists)

**Subtasks**:
- [ ] 3.6.1 Document new patterns
  - [ ] Create architecture patterns guide
  - [ ] Document WebSocket emission pattern
  - [ ] Document database session pattern
  - [ ] Document progress monitoring pattern
  - [ ] Document file cleanup pattern

- [ ] 3.6.2 Update API documentation
  - [ ] Update endpoint docs with new cancel endpoints
  - [ ] Update schema docs if UUIDs changed
  - [ ] Update example requests/responses

- [ ] 3.6.3 Update developer guide
  - [ ] Document how to add new resource types
  - [ ] Document testing requirements
  - [ ] Document shared utilities

- [ ] 3.6.4 Update this task file with completion status
  - [ ] Mark all tasks complete
  - [ ] Add lessons learned section
  - [ ] Add metrics (lines changed, tests added, etc.)

**Success Criteria**:
- ✅ All patterns documented
- ✅ Future developers can follow patterns
- ✅ Task file marked complete

**Estimated Time**: 1 day

---

## Phase 4: API Versioning (Optional - depends on Decision 6)

**Goal**: Manage breaking changes if UUID migration happens
**Duration**: 2-3 days (if needed)
**Risk Level**: Medium

### Task 4.1: Create API v2 (IF DECISION 6 = A)

**Priority**: P1 (if doing UUID migration)
**Files Modified**:
- [ ] **NEW**: `backend/src/api/v2/` (new API version)
- [ ] **MODIFY**: `backend/src/main.py` (register v2 routes)

**Subtasks**:
- [ ] 4.1.1 Create v2 directory structure
  - [ ] Copy v1 structure to v2
  - [ ] Update imports

- [ ] 4.1.2 Implement UUID changes in v2
  - [ ] Update models endpoints for UUID
  - [ ] Update schemas
  - [ ] Update documentation

- [ ] 4.1.3 Keep v1 for compatibility
  - [ ] Add deprecation warnings to v1
  - [ ] Set sunset date for v1
  - [ ] Document migration path

- [ ] 4.1.4 Update frontend to use v2
  - [ ] Create config flag for API version
  - [ ] Update base URL
  - [ ] Test both versions work

**Success Criteria**:
- ✅ v2 API working with UUIDs
- ✅ v1 API still working (deprecated)
- ✅ Clear migration path documented

**Estimated Time**: 2 days (if needed)

---

## Summary of Files

### Files to be Created (NEW):
1. `backend/src/workers/websocket_emitter.py` - Shared WebSocket utility
2. `backend/src/workers/base_task.py` - Base Celery task class
3. `backend/src/workers/progress_monitor.py` - Shared progress monitor (if Decision 4 = A)
4. `frontend/src/api/client.ts` - Shared API client
5. `frontend/src/hooks/usePolling.ts` - Shared polling hook
6. `backend/tests/unit/test_websocket_emitter.py` - WebSocket emitter tests
7. `backend/tests/unit/test_progress_monitor.py` - Progress monitor tests (if Decision 4 = A)
8. `backend/tests/integration/test_worker_sessions.py` - Session handling tests
9. `backend/tests/integration/test_model_cleanup.py` - Model cleanup tests
10. `backend/tests/integration/test_critical_fixes.py` - Phase 1 integration tests
11. `backend/tests/unit/test_dataset_cancellation.py` - Dataset cancel tests
12. `backend/tests/integration/test_dataset_cancel_flow.py` - Dataset cancel integration
13. `backend/tests/integration/test_phase2_features.py` - Phase 2 integration tests
14. `frontend/src/api/client.test.ts` - API client tests
15. `frontend/src/hooks/usePolling.test.ts` - Polling hook tests
16. `frontend/src/__tests__/integration/phase2.test.ts` - Frontend integration tests
17. `backend/alembic/versions/xxxx_standardize_model_ids.py` - UUID migration (if Decision 2 = A)
18. `0xcc/docs/Architecture_Patterns.md` - Patterns documentation

### Files to be Modified (MODIFY):
1. `backend/src/workers/dataset_tasks.py` - Use shared utilities, add cancellation
2. `backend/src/workers/model_tasks.py` - Use shared utilities, fix sessions
3. `backend/src/services/model_service.py` - Fix deletion, improve search
4. `backend/src/services/dataset_service.py` - Update deletion pattern
5. `backend/src/api/v1/endpoints/models.py` - Fix file cleanup trigger
6. `backend/src/api/v1/endpoints/datasets.py` - Add cancel endpoint, update deletion
7. `frontend/src/stores/datasetsStore.ts` - Use shared utilities, add cancellation
8. `frontend/src/stores/modelsStore.ts` - Use shared utilities
9. `frontend/src/api/datasets.ts` - Use shared client, add cancel
10. `frontend/src/api/models.ts` - Use shared client
11. `frontend/src/components/datasets/DatasetCard.tsx` - Add cancel button (if exists)
12. `backend/pyproject.toml` - Update dependencies
13. `backend/requirements.txt` - Update dependencies
14. `backend/src/models/model.py` - UUID migration (if Decision 2 = A)

### Files to be Removed/Obsoleted (OBSOLETE):
1. Duplicate `fetchAPI` function in `datasets.ts` (lines 19-49)
2. Duplicate `fetchAPI` function in `models.ts` (lines 24-57)
3. `DownloadProgressMonitor` class in `model_tasks.py` (lines 57-144) - if Decision 4 = A
4. Inline polling logic in `datasetsStore.ts` (lines 116-157)
5. Inline polling logic in `modelsStore.ts` (lines 123-164)
6. `send_progress_update()` function in `model_tasks.py` (lines 146-176)
7. `emit_progress()` method in `dataset_tasks.py` (lines 47-80)
8. `requests` dependency (if not used elsewhere)

### Test Files to be Updated (TEST):
1. `backend/tests/unit/test_model_service.py` - Add search tests
2. `frontend/src/stores/datasetsStore.test.ts` - Add cancel tests
3. `frontend/src/stores/modelsStore.test.ts` - Update for shared utilities

---

## Risk Mitigation Strategies

### High Risk Areas:
1. **Database Session Migration** (Task 1.2)
   - Risk: Breaking existing dataset operations
   - Mitigation: Comprehensive integration tests, staging environment testing
   - Rollback: Keep async code commented out temporarily

2. **UUID Migration** (Task 3.4 - if chosen)
   - Risk: Data loss, broken API contracts
   - Mitigation: Test migration on copy of production data, have rollback script
   - Rollback: Alembic downgrade available

3. **Breaking Changes** (if Decision 6 = B)
   - Risk: Breaking frontend or external clients
   - Mitigation: Coordinate UI updates, version API if needed
   - Rollback: Keep old code paths temporarily

### Medium Risk Areas:
1. **File Cleanup Changes**
   - Risk: Orphaned files or premature deletion
   - Mitigation: Thorough testing, logging, soft deletes
   - Rollback: Disable background tasks, use inline deletion

2. **Frontend Refactoring**
   - Risk: Breaking UI functionality
   - Mitigation: Component tests, manual testing checklist
   - Rollback: Git revert frontend changes independently

### Testing Strategy by Risk Level:
- **High Risk**: Integration tests + manual testing + staging deployment
- **Medium Risk**: Unit tests + integration tests + manual spot checks
- **Low Risk**: Unit tests + code review

---

## Success Metrics

### Code Quality Metrics:
- [ ] Code duplication reduced by >60% (~200 lines → <80 lines)
- [ ] Test coverage >75% (backend) and >70% (frontend)
- [ ] Zero critical bugs in production
- [ ] All linting and type checking passing

### Performance Metrics:
- [ ] No regression in API response times
- [ ] Progress updates still < 1 second latency
- [ ] Background tasks complete within expected timeframes
- [ ] No increase in database connection usage

### UX Metrics:
- [ ] Dataset cancellation works same as model cancellation
- [ ] Progress bars show accurate percentage (if Decision 4 = A)
- [ ] File cleanup happens without user intervention
- [ ] No increase in error messages or failed operations

### Developer Experience Metrics:
- [ ] New resource types can be added following documented patterns
- [ ] Code review time reduced (less complexity)
- [ ] Onboarding documentation complete
- [ ] Zero instances of "why do we do it differently here?"

---

## Estimated Total Effort

### By Phase:
- **Phase 1** (Critical Fixes): 5 days
- **Phase 2** (Consistency): 5 days
- **Phase 3** (Nice-to-Have): 5 days
- **Phase 4** (API Versioning): 2-3 days (if needed)

### By Developer Experience Level:
- **Senior Developer**: 10-12 days
- **Mid-Level Developer**: 15-18 days
- **Junior Developer**: 20-25 days (with supervision)

### By Decision Path:
- **Minimal Changes** (All B/C decisions): ~8-10 days
- **Balanced Approach** (Mix of A/B): ~12-15 days
- **Full Refactoring** (All A decisions): ~15-18 days

---

## Dependencies and Prerequisites

### Before Starting:
- [ ] All pending PRs merged
- [ ] Clean main branch
- [ ] All tests passing on main
- [ ] Development environment verified
- [ ] Staging environment available for testing

### External Dependencies:
- [ ] PostgreSQL 14+ available
- [ ] Redis 7+ available
- [ ] Celery worker running
- [ ] Frontend dev server working
- [ ] All team members available for code review

### Technical Prerequisites:
- [ ] Understanding of async vs sync Python
- [ ] Understanding of Celery task patterns
- [ ] Understanding of Zustand state management
- [ ] Understanding of database migrations with Alembic
- [ ] Git branching strategy agreed upon

---

## Rollback Plan

### Phase 1 Rollback:
```bash
# If critical issues found in Phase 1
git checkout main
git branch -D feature/refactoring-phase1
# Restart services
./stop-mistudio.sh && ./start-mistudio.sh
```

### Phase 2 Rollback:
```bash
# Can rollback Phase 2 independently of Phase 1
git revert <phase2-commit-range>
# Or cherry-pick Phase 1 commits to new branch
```

### Database Migration Rollback:
```bash
# If UUID migration causes issues
alembic downgrade -1
# Restart backend
```

### Feature Flag Approach (Alternative):
- Add feature flags for new features
- Can disable in production without code rollback
- Recommended for Phase 2 and 3

---

## Communication Plan

### Stakeholder Updates:
- **Daily**: Brief status update in team channel
- **Weekly**: Progress report with metrics
- **Blockers**: Immediate notification
- **Decisions**: Document and get approval within 24 hours

### Code Review Process:
- **Phase 1**: Senior developer review required
- **Phase 2**: Peer review sufficient
- **Phase 3**: Peer review sufficient
- **UUID Migration**: Team review + architect approval

### Deployment Strategy:
- **Phase 1**: Deploy to staging for 2 days testing before production
- **Phase 2**: Can deploy independently after Phase 1 stable
- **Phase 3**: Low-risk, can deploy incrementally

---

## Questions and Blockers

### Open Questions:
1. Should we implement feature flags for gradual rollout?
2. Do we need API documentation generator updates?
3. Should we notify existing API consumers of changes?
4. What's the maintenance window for database migrations?

### Known Blockers:
- None currently identified
- Will update as discovered

---

## Appendix: Decision Summary Table

| Decision | Options | Your Choice | Impact on Timeline | Impact on Risk |
|----------|---------|-------------|-------------------|----------------|
| 1. Database Sessions | A: Sync sessions<br>B: Keep async | ✅ **A** | +1 day | +Medium risk (mitigated by TDD) |
| 2. UUID Migration | A: Immediate<br>B: Defer<br>C: Hybrid | ✅ **A** | +2 days | +High risk (mitigated by v2 API) |
| 3. File Deletion | A: Background<br>B: Keep inline | ✅ **A** | +1 day | +Low risk |
| 4. Progress Monitor | A: Shared<br>B: Keep separate | ✅ **A** | +2 days | +Low risk |
| 5. Test Coverage | A: TDD<br>B: Tests at end<br>C: Critical only | ✅ **A** | +0 days (distributed) | +Low risk |
| 6. Breaking Changes | A: API v2<br>B: Direct changes<br>C: Avoid breaking | ✅ **A** | +2 days | +Low risk |

**Total Timeline**: **17-18 days** (full refactoring with all best practices)
**Total Added Time**: +8 days for quality (vs minimal changes)
**Risk Mitigation**: TDD + API versioning + comprehensive testing + rollback plans

---

## Next Steps

1. **Review this document thoroughly**
2. **Make decisions on the 6 decision points above**
3. **Create feature branch: `feature/architectural-refactoring`**
4. **Start with Phase 1, Task 1.1**
5. **Update this document as you progress**

---

**Document Status**: 🚀 **IN PROGRESS - Phase 1 Starting**
**Created By**: Claude (Architectural Review Agent)
**Created Date**: 2025-10-12
**Last Updated**: 2025-10-12 (Decisions finalized)
**Version**: 1.1
**Timeline**: 17-18 days
**Current Task**: Creating feature branch and starting Task 1.1
