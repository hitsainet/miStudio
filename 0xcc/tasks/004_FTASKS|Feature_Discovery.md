# Feature Tasks: Feature Discovery

**Document ID:** 004_FTASKS|Feature_Discovery
**Version:** 1.1
**Last Updated:** 2025-12-16
**Status:** Implemented
**Related PRD:** [004_FPRD|Feature_Discovery](../prds/004_FPRD|Feature_Discovery.md)

---

## Task Summary

| Phase | Tasks | Status |
|-------|-------|--------|
| Phase 1: Database | 4 tasks | ✅ Complete |
| Phase 2: Extraction Service | 5 tasks | ✅ Complete |
| Phase 3: Token Analysis | 3 tasks | ✅ Complete |
| Phase 4: Labeling System | 6 tasks | ✅ Complete |
| Phase 5: API Endpoints | 4 tasks | ✅ Complete |
| Phase 6: Frontend Browser | 6 tasks | ✅ Complete |
| Phase 7: Labeling UI | 4 tasks | ✅ Complete |

**Total: 32 tasks**

---

## Phase 1: Database Setup

### Task 1.1: Create Feature Migration
- [x] Create features table
- [x] Create feature_activations table (partitioned)
- [x] Create extraction_jobs table
- [x] Add indexes for queries

**Files:**
- `backend/alembic/versions/xxx_create_feature_tables.py`

### Task 1.2: Create SQLAlchemy Models
- [x] Define Feature model
- [x] Define FeatureActivation model
- [x] Define ExtractionJob model
- [x] Configure relationships

**Files:**
- `backend/src/models/feature.py`
- `backend/src/models/extraction_job.py`

### Task 1.3: Create Pydantic Schemas
- [x] FeatureResponse schema
- [x] FeatureActivationResponse schema
- [x] ExtractionJobCreate schema
- [x] ExtractionConfig schema

**Files:**
- `backend/src/schemas/feature.py`
- `backend/src/schemas/extraction.py`

### Task 1.4: Run Migrations
- [x] Apply migrations
- [x] Verify partitioning works
- [x] Test cascade deletes

---

## Phase 2: Extraction Service

### Task 2.1: Create Extraction Service
- [x] Implement create_job() method
- [x] Implement extract_features() method
- [x] Implement save_features() method
- [x] Handle batch processing

**Files:**
- `backend/src/services/extraction_service.py`

### Task 2.2: Implement Feature Statistics
- [x] Calculate activation_frequency
- [x] Calculate max_activation
- [x] Calculate mean_activation
- [x] Calculate std_activation

### Task 2.3: Implement Top Token Aggregation
- [x] Get top activating tokens per feature
- [x] Aggregate by token
- [x] Store in denormalized JSONB

**Files:**
- `backend/src/services/token_aggregator_service.py`

### Task 2.4: Create Extraction Task
- [x] Define extract_features_task
- [x] Configure queue routing
- [x] Emit progress via WebSocket

**Files:**
- `backend/src/workers/extraction_tasks.py`

### Task 2.5: Vectorized Extraction
- [x] Implement batch processing
- [x] Optimize memory usage
- [x] Add chunked processing

**Files:**
- `backend/src/services/extraction_vectorized.py`

---

## Phase 3: Token Analysis

### Task 3.1: Token Statistics
- [x] Count token occurrences
- [x] Calculate mean activation per token
- [x] Calculate max activation per token
- [x] Sort by frequency

### Task 3.2: Context Extraction
- [x] Extract context before token
- [x] Extract context after token
- [x] Store with activation

### Task 3.3: Stop Words Filter
- [x] Define stop words list
- [x] Add filter option
- [x] Enable by default

---

## Phase 4: Labeling System

### Task 4.1: Create Labeling Models
- [x] Create labeling_jobs table
- [x] Create labeling_prompt_templates table
- [x] Define default template

**Files:**
- `backend/src/models/labeling_job.py`
- `backend/src/models/labeling_prompt_template.py`

### Task 4.2: Create OpenAI Labeling Service
- [x] Implement label_feature() method
- [x] Implement batch labeling
- [x] Add rate limiting
- [x] Handle JSON response format

**Files:**
- `backend/src/services/openai_labeling_service.py`

### Task 4.3: Create Context Formatter
- [x] Format top tokens for prompt
- [x] Format example contexts
- [x] Format statistics

**Files:**
- `backend/src/services/labeling_context_formatter.py`

### Task 4.4: Create Labeling Task
- [x] Define label_features_task
- [x] Process in batches
- [x] Update features with labels
- [x] Emit progress

**Files:**
- `backend/src/workers/labeling_tasks.py`

### Task 4.5: Labeling Template Management
- [x] CRUD for templates
- [x] Default template handling
- [x] Template variables

### Task 4.6: Dual Label System
- [x] Support semantic_label
- [x] Support category_label
- [x] Allow manual override

---

## Phase 5: API Endpoints

### Task 5.1: Feature Endpoints
- [x] GET /features - List with filters
- [x] GET /features/{id} - Get details
- [x] GET /features/{id}/activations - Get activations
- [x] PUT /features/{id}/label - Update label

**Files:**
- `backend/src/api/v1/endpoints/features.py`

### Task 5.2: Extraction Endpoints
- [x] POST /extractions - Start extraction
- [x] GET /extractions/{id} - Get status
- [x] POST /extractions/{id}/cancel - Cancel

### Task 5.3: Labeling Endpoints
- [x] POST /labeling - Start labeling job
- [x] GET /labeling/{id} - Get status
- [x] GET /labeling/templates - List templates
- [x] POST /labeling/templates - Create template

### Task 5.4: Search/Filter API
- [x] Search by label text
- [x] Filter by category
- [x] Sort options
- [x] Pagination

---

## Phase 6: Frontend Browser

### Task 6.1: Create Types
- [x] Define Feature interface
- [x] Define FeatureActivation interface
- [x] Define ExtractionJob interface

**Files:**
- `frontend/src/types/features.ts`

### Task 6.2: Create API Client
- [x] listFeatures() function
- [x] getFeature() function
- [x] getActivations() function
- [x] updateLabel() function

**Files:**
- `frontend/src/api/features.ts`

### Task 6.3: Create Features Store
- [x] Feature list state
- [x] Selected feature state
- [x] Fetch and filter actions

**Files:**
- `frontend/src/stores/featuresStore.ts`

### Task 6.4: Create FeatureBrowser Component
- [x] Search input
- [x] Sort selector
- [x] Grid layout
- [x] Pagination

**Files:**
- `frontend/src/components/features/FeatureBrowser.tsx`

### Task 6.5: Create FeatureCard Component
- [x] Display feature index
- [x] Display label
- [x] Display top tokens
- [x] Display statistics

**Files:**
- `frontend/src/components/features/FeatureCard.tsx`

### Task 6.6: Create FeatureDetailModal
- [x] Statistics section
- [x] Top tokens section
- [x] Example activations
- [x] Token highlighting

**Files:**
- `frontend/src/components/features/FeatureDetailModal.tsx`

---

## Phase 7: Labeling UI

### Task 7.1: Create TokenHighlight Component
- [x] Parse context string
- [x] Highlight target token
- [x] Color by activation strength

**Files:**
- `frontend/src/components/features/TokenHighlight.tsx`

### Task 7.2: Create ExtractionJobCard
- [x] Show job progress
- [x] Show feature count
- [x] Cancel button

**Files:**
- `frontend/src/components/features/ExtractionJobCard.tsx`

### Task 7.3: Create StartLabelingButton
- [x] Template selector
- [x] Feature selection
- [x] Start labeling action

**Files:**
- `frontend/src/components/labeling/StartLabelingButton.tsx`

### Task 7.4: Create TemplatesPanel
- [x] List templates
- [x] Create/edit modal
- [x] Set default template

**Files:**
- `frontend/src/components/panels/LabelingPromptTemplatesPanel.tsx`

---

## Relevant Files Summary

### Backend
| File | Purpose |
|------|---------|
| `backend/src/models/feature.py` | Feature models |
| `backend/src/models/labeling_job.py` | Labeling job model |
| `backend/src/schemas/feature.py` | Feature schemas |
| `backend/src/schemas/extraction.py` | Extraction schemas |
| `backend/src/services/extraction_service.py` | Extraction logic |
| `backend/src/services/openai_labeling_service.py` | AI labeling |
| `backend/src/workers/extraction_tasks.py` | Celery tasks |
| `backend/src/api/v1/endpoints/features.py` | API routes |

### Frontend
| File | Purpose |
|------|---------|
| `frontend/src/types/features.ts` | TypeScript types |
| `frontend/src/api/features.ts` | API client |
| `frontend/src/stores/featuresStore.ts` | Zustand store |
| `frontend/src/components/features/FeatureBrowser.tsx` | Browser |
| `frontend/src/components/features/FeatureCard.tsx` | Card |
| `frontend/src/components/features/FeatureDetailModal.tsx` | Modal |
| `frontend/src/components/features/TokenHighlight.tsx` | Highlight |

---

*Related: [PRD](../prds/004_FPRD|Feature_Discovery.md) | [TDD](../tdds/004_FTDD|Feature_Discovery.md) | [TID](../tids/004_FTID|Feature_Discovery.md)*
