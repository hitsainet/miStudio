# Feature Tasks: Neuronpedia Export

**Document ID:** 007_FTASKS|Neuronpedia_Export
**Version:** 1.0
**Last Updated:** 2025-12-05
**Status:** Implemented
**Related PRD:** [007_FPRD|Neuronpedia_Export](../prds/007_FPRD|Neuronpedia_Export.md)

---

## Task Summary

| Phase | Tasks | Status |
|-------|-------|--------|
| Phase 1: Database | 3 tasks | ✅ Complete |
| Phase 2: Data Computation | 4 tasks | ✅ Complete |
| Phase 3: Export Service | 4 tasks | ✅ Complete |
| Phase 4: Celery Task | 3 tasks | ✅ Complete |
| Phase 5: API Endpoints | 4 tasks | ✅ Complete |
| Phase 6: Frontend | 4 tasks | ✅ Complete |

**Total: 22 tasks**

---

## Phase 1: Database Setup

### Task 1.1: Create Export Job Migration
- [x] Create neuronpedia_export_jobs table
- [x] Add config JSONB column
- [x] Add progress tracking columns

**Files:**
- `backend/alembic/versions/xxx_create_neuronpedia_exports.py`

### Task 1.2: Create Feature Dashboard Data Migration
- [x] Create feature_dashboard_data table
- [x] Add logit_lens_data JSONB
- [x] Add histogram_data JSONB

### Task 1.3: Create SQLAlchemy Models
- [x] Define NeuronpediaExportJob model
- [x] Define FeatureDashboardData model

**Files:**
- `backend/src/models/neuronpedia_export.py`
- `backend/src/models/feature_dashboard.py`

---

## Phase 2: Data Computation Services

### Task 2.1: Create Logit Lens Service
- [x] Implement compute_logit_lens()
- [x] Get decoder weight
- [x] Project through unembedding
- [x] Get top positive/negative tokens

**Files:**
- `backend/src/services/logit_lens_service.py`

### Task 2.2: Create Histogram Service
- [x] Implement compute_histogram()
- [x] Configurable bin count
- [x] Include statistics

**Files:**
- `backend/src/services/histogram_service.py`

### Task 2.3: Create Token Aggregator Service
- [x] Implement aggregate_top_tokens()
- [x] Group by token
- [x] Compute per-token statistics

**Files:**
- `backend/src/services/token_aggregator_service.py`

### Task 2.4: TransformerLens Mapping
- [x] Map model names to hook names
- [x] Handle different architectures

**Files:**
- `backend/src/utils/transformerlens_mapping.py`

---

## Phase 3: Export Service

### Task 3.1: Create Export Service
- [x] Implement create_job()
- [x] Implement get_features_to_export()
- [x] Handle feature selection modes

**Files:**
- `backend/src/services/neuronpedia_export_service.py`

### Task 3.2: Generate Feature JSON
- [x] Implement generate_feature_json()
- [x] Include all optional data
- [x] Match Neuronpedia format

### Task 3.3: Generate Metadata
- [x] SAE configuration
- [x] Export date
- [x] Included data flags

### Task 3.4: Create Archive
- [x] Implement create_export_archive()
- [x] Add metadata.json
- [x] Add README.md
- [x] Add feature JSONs
- [x] Add SAELens weights

---

## Phase 4: Celery Task

### Task 4.1: Create Export Task
- [x] Define neuronpedia_export_task
- [x] Configure export queue
- [x] Handle task binding

**Files:**
- `backend/src/workers/neuronpedia_tasks.py`

### Task 4.2: Progress Emission
- [x] Emit via WebSocket
- [x] Stage indicators
- [x] Feature progress

### Task 4.3: Error Handling
- [x] Catch exceptions
- [x] Update job status
- [x] Store error message

---

## Phase 5: API Endpoints

### Task 5.1: Create Router
- [x] Define router
- [x] Add to main router

**Files:**
- `backend/src/api/v1/endpoints/neuronpedia.py`

### Task 5.2: Export Job Endpoints
- [x] POST /neuronpedia/export - Start export
- [x] GET /neuronpedia/export/{id} - Get status
- [x] POST /neuronpedia/export/{id}/cancel - Cancel

### Task 5.3: Download Endpoint
- [x] GET /neuronpedia/export/{id}/download
- [x] Stream file response
- [x] Set content disposition

### Task 5.4: History Endpoint
- [x] GET /neuronpedia/exports - List all
- [x] Filter by status
- [x] Pagination

---

## Phase 6: Frontend

### Task 6.1: Create Types
- [x] Define NeuronpediaExportJob interface
- [x] Define ExportConfig interface

**Files:**
- `frontend/src/types/neuronpedia.ts`

### Task 6.2: Create API Client
- [x] startExport() function
- [x] getExportStatus() function
- [x] cancelExport() function
- [x] downloadExport() function

**Files:**
- `frontend/src/api/neuronpedia.ts`

### Task 6.3: Create Export Store
- [x] Export jobs state
- [x] Current job tracking
- [x] Progress updates

**Files:**
- `frontend/src/stores/neuronpediaExportStore.ts`

### Task 6.4: Create ExportToNeuronpedia Modal
- [x] Feature selection options
- [x] Include data checkboxes
- [x] Progress display
- [x] Download button

**Files:**
- `frontend/src/components/saes/ExportToNeuronpedia.tsx`

---

## Relevant Files Summary

### Backend
| File | Purpose |
|------|---------|
| `backend/src/models/neuronpedia_export.py` | Export job model |
| `backend/src/models/feature_dashboard.py` | Dashboard data model |
| `backend/src/schemas/neuronpedia.py` | Pydantic schemas |
| `backend/src/services/neuronpedia_export_service.py` | Export orchestration |
| `backend/src/services/logit_lens_service.py` | Logit lens computation |
| `backend/src/services/histogram_service.py` | Histogram generation |
| `backend/src/workers/neuronpedia_tasks.py` | Celery task |
| `backend/src/api/v1/endpoints/neuronpedia.py` | API routes |

### Frontend
| File | Purpose |
|------|---------|
| `frontend/src/types/neuronpedia.ts` | TypeScript types |
| `frontend/src/api/neuronpedia.ts` | API client |
| `frontend/src/stores/neuronpediaExportStore.ts` | Zustand store |
| `frontend/src/components/saes/ExportToNeuronpedia.tsx` | Export modal |

---

*Related: [PRD](../prds/007_FPRD|Neuronpedia_Export.md) | [TDD](../tdds/007_FTDD|Neuronpedia_Export.md) | [TID](../tids/007_FTID|Neuronpedia_Export.md)*
