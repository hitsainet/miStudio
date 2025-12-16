# Feature Tasks: SAE Training

**Document ID:** 003_FTASKS|SAE_Training
**Version:** 1.1
**Last Updated:** 2025-12-16
**Status:** Implemented
**Related PRD:** [003_FPRD|SAE_Training](../prds/003_FPRD|SAE_Training.md)

---

## Task Summary

| Phase | Tasks | Status |
|-------|-------|--------|
| Phase 1: SAE Architectures | 5 tasks | ✅ Complete |
| Phase 2: Database | 4 tasks | ✅ Complete |
| Phase 3: Activation Extraction | 3 tasks | ✅ Complete |
| Phase 4: Training Service | 5 tasks | ✅ Complete |
| Phase 5: Celery Tasks | 4 tasks | ✅ Complete |
| Phase 6: API Endpoints | 4 tasks | ✅ Complete |
| Phase 7: Frontend | 8 tasks | ✅ Complete |
| Phase 8: Templates | 4 tasks | ✅ Complete |

**Total: 37 tasks**

---

## Phase 1: SAE Architectures

### Task 1.1: Create Base SAE Class
- [x] Define abstract BaseSAE class
- [x] Define encode() abstract method
- [x] Define decode() abstract method
- [x] Define loss() abstract method
- [x] Implement forward() method

**Files:**
- `backend/src/ml/sparse_autoencoder.py`

### Task 1.2: Implement Standard SAE
- [x] Define StandardSAE class
- [x] Implement L1 sparsity loss
- [x] Initialize with proper weight scaling
- [x] Normalize decoder weights

### Task 1.3: Implement JumpReLU SAE
- [x] Define JumpReLUSAE class
- [x] Implement learnable thresholds
- [x] Implement L0 penalty loss
- [x] Use log-space for threshold stability

### Task 1.4: Implement Skip SAE
- [x] Define SkipSAE class
- [x] Add skip connection coefficient
- [x] Modify forward for residual

### Task 1.5: Implement Transcoder SAE
- [x] Define TranscoderSAE class
- [x] Support different input/output dimensions
- [x] Modify loss for layer-to-layer mapping

---

## Phase 2: Database Setup

### Task 2.1: Create Training Migration
- [x] Create trainings table
- [x] Create training_metrics table
- [x] Create checkpoints table
- [x] Add proper indexes

**Files:**
- `backend/alembic/versions/xxx_create_training_tables.py`

### Task 2.2: Create SQLAlchemy Models
- [x] Define Training model
- [x] Define TrainingMetric model
- [x] Define Checkpoint model
- [x] Configure relationships

**Files:**
- `backend/src/models/training.py`

### Task 2.3: Create Pydantic Schemas
- [x] TrainingCreate schema
- [x] TrainingResponse schema
- [x] TrainingConfig schema
- [x] MetricResponse schema

**Files:**
- `backend/src/schemas/training.py`

### Task 2.4: Run Migrations
- [x] Apply migrations
- [x] Verify relationships
- [x] Test cascade deletes

---

## Phase 3: Activation Extraction

### Task 3.1: Create Activation Extractor
- [x] Implement hook registration
- [x] Support multiple hook points
- [x] Handle different output formats
- [x] Clean up hooks properly

**Files:**
- `backend/src/ml/activation_extraction.py`

### Task 3.2: Implement Hook Points
- [x] Support resid_post (after layer)
- [x] Support mlp_out (MLP output)
- [x] Support attn_out (attention output)

### Task 3.3: Activation Storage
- [x] Save activations to disk
- [x] Efficient tensor format
- [x] Memory-mapped loading

---

## Phase 4: Training Service

### Task 4.1: Create Training Service
- [x] Implement create() method
- [x] Implement get_by_id() method
- [x] Implement list_all() method
- [x] Implement update_status() method

**Files:**
- `backend/src/services/training_service.py`

### Task 4.2: Implement Metrics Logging
- [x] Implement add_metric() method
- [x] Implement get_metrics() method
- [x] Support various metric types

### Task 4.3: Implement Checkpoint Management
- [x] Implement save_checkpoint() method
- [x] Track best checkpoints
- [x] Implement load_checkpoint()

### Task 4.4: Create SAE Factory
- [x] Implement create_sae() function
- [x] Map architecture string to class
- [x] Handle architecture-specific params

### Task 4.5: Training Loop Implementation
- [x] Implement main training loop
- [x] Add gradient accumulation support
- [x] Add mixed precision support
- [x] Add learning rate scheduling

---

## Phase 5: Celery Tasks

### Task 5.1: Create Training Task
- [x] Define train_sae_task
- [x] Configure SAE queue
- [x] Handle task binding

**Files:**
- `backend/src/workers/training_tasks.py`

### Task 5.2: Implement Progress Emission
- [x] Emit at regular intervals
- [x] Include loss, L0, FVU metrics
- [x] Handle completed/failed states

### Task 5.3: Implement Checkpointing
- [x] Save at configurable intervals
- [x] Track best checkpoint
- [x] Save final model in community format

### Task 5.4: Error Handling
- [x] Catch and log exceptions
- [x] Update database on failure
- [x] Clean up resources

---

## Phase 6: API Endpoints

### Task 6.1: Create Router
- [x] Define router with prefix
- [x] Add to main router

**Files:**
- `backend/src/api/v1/endpoints/trainings.py`

### Task 6.2: Implement CRUD Endpoints
- [x] POST /trainings - Create and start
- [x] GET /trainings - List all
- [x] GET /trainings/{id} - Get details
- [x] DELETE /trainings/{id} - Delete

### Task 6.3: Implement Control Endpoints
- [x] POST /trainings/{id}/pause - Pause
- [x] POST /trainings/{id}/resume - Resume
- [x] POST /trainings/{id}/stop - Stop
- [x] POST /trainings/{id}/retry - Retry

### Task 6.4: Implement Metrics Endpoint
- [x] GET /trainings/{id}/metrics - Get metrics
- [x] Support pagination
- [x] Support time range filter

---

## Phase 7: Frontend

### Task 7.1: Create Types
- [x] Define Training interface
- [x] Define TrainingConfig interface
- [x] Define TrainingMetric interface

**Files:**
- `frontend/src/types/training.ts`

### Task 7.2: Create API Client
- [x] create() function
- [x] list() function
- [x] getMetrics() function
- [x] Control functions (pause/resume/stop)

**Files:**
- `frontend/src/api/trainings.ts`

### Task 7.3: Create Zustand Store
- [x] Define state shape
- [x] Implement actions
- [x] Handle WebSocket updates

**Files:**
- `frontend/src/stores/trainingsStore.ts`

### Task 7.4: Create TrainingCard Component
- [x] Display training info
- [x] Show progress bar
- [x] Show live metrics
- [x] Add control buttons

**Files:**
- `frontend/src/components/training/TrainingCard.tsx`

### Task 7.5: Create TrainingForm Component
- [x] Model/Dataset selectors
- [x] Architecture selector
- [x] Hyperparameter inputs
- [x] Validation

**Files:**
- `frontend/src/components/training/TrainingForm.tsx`

### Task 7.6: Create TrainingPanel
- [x] Form modal trigger
- [x] Training list
- [x] Bulk actions

**Files:**
- `frontend/src/components/panels/TrainingPanel.tsx`

### Task 7.7: Create Loss Chart Component
- [x] Real-time loss curve
- [x] L0 and FVU metrics
- [x] Zoom and pan

**Files:**
- `frontend/src/components/training/LossChart.tsx`

### Task 7.8: Create WebSocket Hook
- [x] Subscribe to training channel
- [x] Handle progress events
- [x] Handle completed/failed

**Files:**
- `frontend/src/hooks/useTrainingWebSocket.ts`

---

## Phase 8: Training Templates

### Task 8.1: Create Template Migration
- [x] Create training_templates table
- [x] Add all hyperparameter columns

**Files:**
- `backend/alembic/versions/xxx_create_training_templates.py`

### Task 8.2: Backend Template Support
- [x] Template model
- [x] Template service
- [x] Template API endpoints

### Task 8.3: Frontend Template Components
- [x] TemplateForm component
- [x] TemplateCard component
- [x] TemplateList component
- [x] TemplatesPanel

### Task 8.4: Template Features
- [x] Save current config as template
- [x] Load template into form
- [x] Import/export templates
- [x] Favorites management

---

## Relevant Files Summary

### Backend
| File | Purpose |
|------|---------|
| `backend/src/ml/sparse_autoencoder.py` | SAE architectures |
| `backend/src/ml/activation_extraction.py` | Activation hooks |
| `backend/src/models/training.py` | SQLAlchemy models |
| `backend/src/schemas/training.py` | Pydantic schemas |
| `backend/src/services/training_service.py` | Business logic |
| `backend/src/workers/training_tasks.py` | Celery tasks |
| `backend/src/api/v1/endpoints/trainings.py` | API routes |

### Frontend
| File | Purpose |
|------|---------|
| `frontend/src/types/training.ts` | TypeScript types |
| `frontend/src/api/trainings.ts` | API client |
| `frontend/src/stores/trainingsStore.ts` | Zustand store |
| `frontend/src/components/training/TrainingCard.tsx` | Card |
| `frontend/src/components/training/TrainingForm.tsx` | Form |
| `frontend/src/components/training/LossChart.tsx` | Chart |
| `frontend/src/components/panels/TrainingPanel.tsx` | Panel |

---

*Related: [PRD](../prds/003_FPRD|SAE_Training.md) | [TDD](../tdds/003_FTDD|SAE_Training.md) | [TID](../tids/003_FTID|SAE_Training.md)*
