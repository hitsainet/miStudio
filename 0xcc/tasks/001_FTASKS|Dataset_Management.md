# Feature Tasks: Dataset Management

**Document ID:** 001_FTASKS|Dataset_Management
**Version:** 1.1
**Last Updated:** 2025-12-16
**Status:** Implemented
**Related PRD:** [001_FPRD|Dataset_Management](../prds/001_FPRD|Dataset_Management.md)

---

## Task Summary

| Phase | Tasks | Status |
|-------|-------|--------|
| Phase 1: Database | 4 tasks | ✅ Complete |
| Phase 2: Backend Services | 5 tasks | ✅ Complete |
| Phase 3: API Endpoints | 4 tasks | ✅ Complete |
| Phase 4: Frontend Store | 3 tasks | ✅ Complete |
| Phase 5: UI Components | 5 tasks | ✅ Complete |
| Phase 6: Integration | 3 tasks | ✅ Complete |

**Total: 24 tasks**

---

## Phase 1: Database Setup

### Task 1.1: Create Dataset Migration
- [x] Create Alembic migration file
- [x] Define datasets table schema
- [x] Add indexes for common queries
- [x] Test migration up/down

**Files:**
- `backend/alembic/versions/xxx_create_datasets_table.py`

### Task 1.2: Create SQLAlchemy Model
- [x] Define Dataset model class
- [x] Add column definitions with types
- [x] Configure relationships (if any)
- [x] Add model to __init__.py

**Files:**
- `backend/src/models/dataset.py`
- `backend/src/models/__init__.py`

### Task 1.3: Create Pydantic Schemas
- [x] DatasetCreate schema
- [x] DatasetResponse schema
- [x] DatasetDownloadRequest schema
- [x] DatasetUpdate schema (optional)

**Files:**
- `backend/src/schemas/dataset.py`

### Task 1.4: Run Migration
- [x] Apply migration to dev database
- [x] Verify table structure
- [x] Test basic CRUD operations

---

## Phase 2: Backend Services

### Task 2.1: Create Dataset Service
- [x] Implement create() method
- [x] Implement get_by_id() method
- [x] Implement list_all() method
- [x] Implement update_status() method
- [x] Implement delete() method

**Files:**
- `backend/src/services/dataset_service.py`

### Task 2.2: Create HuggingFace Service
- [x] Implement download_dataset() method
- [x] Add streaming for large datasets
- [x] Implement progress tracking
- [x] Handle subset/split selection
- [x] Save in efficient format (Arrow)

**Files:**
- `backend/src/services/huggingface_dataset_service.py`

### Task 2.3: Create Celery Task
- [x] Define download_dataset_task
- [x] Configure queue routing
- [x] Add error handling
- [x] Emit WebSocket progress

**Files:**
- `backend/src/workers/dataset_tasks.py`

### Task 2.4: Add WebSocket Emission
- [x] Add emit_dataset_progress function
- [x] Define channel naming convention
- [x] Add completed/failed events

**Files:**
- `backend/src/workers/websocket_emitter.py`

### Task 2.5: Unit Tests
- [x] Test dataset service CRUD
- [x] Test HuggingFace service (mocked)
- [x] Test Celery task execution

**Files:**
- `backend/tests/test_dataset_service.py`

---

## Phase 3: API Endpoints

### Task 3.1: Create Router
- [x] Define router with prefix
- [x] Add to main API router

**Files:**
- `backend/src/api/v1/endpoints/datasets.py`
- `backend/src/api/v1/router.py`

### Task 3.2: Implement Endpoints
- [x] GET /datasets - List all
- [x] POST /datasets/download - Start download
- [x] GET /datasets/{id} - Get by ID
- [x] DELETE /datasets/{id} - Delete

### Task 3.3: Add Request Validation
- [x] Validate repo_id format
- [x] Validate split options
- [x] Handle duplicate detection

### Task 3.4: API Documentation
- [x] Add OpenAPI descriptions
- [x] Document response models
- [x] Add example values

---

## Phase 4: Frontend Store

### Task 4.1: Create Types
- [x] Define Dataset interface
- [x] Define DatasetDownloadRequest interface
- [x] Define status union type

**Files:**
- `frontend/src/types/dataset.ts`

### Task 4.2: Create API Client
- [x] list() function
- [x] download() function
- [x] get() function
- [x] delete() function

**Files:**
- `frontend/src/api/datasets.ts`

### Task 4.3: Create Zustand Store
- [x] Define state shape
- [x] Implement fetchDatasets action
- [x] Implement downloadDataset action
- [x] Implement deleteDataset action
- [x] Implement updateDatasetProgress action

**Files:**
- `frontend/src/stores/datasetsStore.ts`

---

## Phase 5: UI Components

### Task 5.1: Create DatasetCard Component
- [x] Display dataset info
- [x] Show download progress
- [x] Add action buttons
- [x] Style matching design system

**Files:**
- `frontend/src/components/datasets/DatasetCard.tsx`

### Task 5.2: Create DownloadForm Component
- [x] Repository ID input
- [x] Subset/split selectors
- [x] Max samples input
- [x] Submit handling
- [x] Validation

**Files:**
- `frontend/src/components/datasets/DownloadForm.tsx`

### Task 5.3: Create DatasetsPanel
- [x] Grid layout for cards
- [x] Download button/modal
- [x] Empty state
- [x] Loading state

**Files:**
- `frontend/src/components/panels/DatasetsPanel.tsx`

### Task 5.4: Create WebSocket Hook
- [x] Subscribe to dataset channels
- [x] Handle progress events
- [x] Update store on events
- [x] Clean up on unmount

**Files:**
- `frontend/src/hooks/useDatasetWebSocket.ts`

### Task 5.5: Component Tests
- [x] Test DatasetCard rendering
- [x] Test DownloadForm submission
- [x] Test progress updates

---

## Phase 6: Integration

### Task 6.1: Navigation Integration
- [x] Add Datasets to sidebar
- [x] Configure routing
- [x] Add icon

### Task 6.2: End-to-End Testing
- [x] Test full download flow
- [x] Verify WebSocket updates
- [x] Test delete functionality

### Task 6.3: Documentation
- [x] Update API docs
- [x] Add user guide section
- [x] Document supported formats

---

## Relevant Files Summary

### Backend
| File | Purpose |
|------|---------|
| `backend/src/models/dataset.py` | SQLAlchemy model |
| `backend/src/schemas/dataset.py` | Pydantic schemas |
| `backend/src/services/dataset_service.py` | Business logic |
| `backend/src/services/huggingface_dataset_service.py` | HF integration |
| `backend/src/workers/dataset_tasks.py` | Celery tasks |
| `backend/src/api/v1/endpoints/datasets.py` | API routes |

### Frontend
| File | Purpose |
|------|---------|
| `frontend/src/types/dataset.ts` | TypeScript types |
| `frontend/src/api/datasets.ts` | API client |
| `frontend/src/stores/datasetsStore.ts` | Zustand store |
| `frontend/src/components/datasets/DatasetCard.tsx` | Card component |
| `frontend/src/components/datasets/DownloadForm.tsx` | Form component |
| `frontend/src/components/panels/DatasetsPanel.tsx` | Panel component |

---

*Related: [PRD](../prds/001_FPRD|Dataset_Management.md) | [TDD](../tdds/001_FTDD|Dataset_Management.md) | [TID](../tids/001_FTID|Dataset_Management.md)*
