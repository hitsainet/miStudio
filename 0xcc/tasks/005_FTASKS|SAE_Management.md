# Feature Tasks: SAE Management

**Document ID:** 005_FTASKS|SAE_Management
**Version:** 1.0
**Last Updated:** 2025-12-05
**Status:** Implemented
**Related PRD:** [005_FPRD|SAE_Management](../prds/005_FPRD|SAE_Management.md)

---

## Task Summary

| Phase | Tasks | Status |
|-------|-------|--------|
| Phase 1: Database | 3 tasks | ✅ Complete |
| Phase 2: Format Support | 4 tasks | ✅ Complete |
| Phase 3: HuggingFace Integration | 4 tasks | ✅ Complete |
| Phase 4: SAE Manager Service | 4 tasks | ✅ Complete |
| Phase 5: API Endpoints | 3 tasks | ✅ Complete |
| Phase 6: Frontend | 5 tasks | ✅ Complete |

**Total: 23 tasks**

---

## Phase 1: Database Setup

### Task 1.1: Create External SAE Migration
- [x] Create external_saes table
- [x] Add architecture column
- [x] Add source tracking columns
- [x] Add format column

**Files:**
- `backend/alembic/versions/xxx_create_external_saes_table.py`

### Task 1.2: Create SQLAlchemy Model
- [x] Define ExternalSAE model
- [x] Add all columns
- [x] Add metadata JSONB

**Files:**
- `backend/src/models/external_sae.py`

### Task 1.3: Create Pydantic Schemas
- [x] ExternalSAECreate schema
- [x] ExternalSAEResponse schema
- [x] SAEDownloadRequest schema

**Files:**
- `backend/src/schemas/sae.py`

---

## Phase 2: Format Support

### Task 2.1: Community Format Loader
- [x] Implement load() method
- [x] Implement save() method
- [x] Implement validate() method

**Files:**
- `backend/src/ml/community_format.py`

### Task 2.2: miStudio Format Loader
- [x] Implement load() method
- [x] Implement to_community_format()
- [x] Handle key mapping

### Task 2.3: Format Detection
- [x] Detect by file presence
- [x] Detect by config keys
- [x] Return format type

### Task 2.4: SAE Loader Utility
- [x] Unified load_sae() function
- [x] Auto-detect format
- [x] Return correct SAE class

**Files:**
- `backend/src/ml/sae_loader.py`

---

## Phase 3: HuggingFace Integration

### Task 3.1: HuggingFace SAE Service
- [x] Implement get_repo_info()
- [x] Implement download_sae()
- [x] Implement load_sae_config()

**Files:**
- `backend/src/services/huggingface_sae_service.py`

### Task 3.2: Gemma Scope Support
- [x] Parse layer/width/l0 structure
- [x] List available variants
- [x] Download specific variant

### Task 3.3: Download Task
- [x] Define download_sae_task
- [x] Emit progress via WebSocket
- [x] Update database on completion

**Files:**
- `backend/src/workers/sae_download_tasks.py`

### Task 3.4: Format Detection Service
- [x] Detect format after download
- [x] Convert if needed
- [x] Validate weights

---

## Phase 4: SAE Manager Service

### Task 4.1: List All SAEs
- [x] Query trained SAEs
- [x] Query external SAEs
- [x] Combine and sort
- [x] Support filters

**Files:**
- `backend/src/services/sae_manager_service.py`

### Task 4.2: Get SAE by ID
- [x] Check trained SAEs first
- [x] Check external SAEs
- [x] Return unified format

### Task 4.3: Create External SAE
- [x] Validate input
- [x] Create record
- [x] Return response

### Task 4.4: Delete SAE
- [x] Delete files
- [x] Delete database record
- [x] Handle cascades

---

## Phase 5: API Endpoints

### Task 5.1: Create Router
- [x] Define router
- [x] Add to main router

**Files:**
- `backend/src/api/v1/endpoints/saes.py`

### Task 5.2: Core Endpoints
- [x] GET /saes - List all
- [x] GET /saes/{id} - Get details
- [x] DELETE /saes/{id} - Delete
- [x] GET /saes/{id}/config - Get config

### Task 5.3: Download Endpoints
- [x] POST /saes/download-hf - Download from HuggingFace
- [x] GET /saes/gemma-scope/variants - List Gemma Scope variants

---

## Phase 6: Frontend

### Task 6.1: Create Types
- [x] Define SAE interface
- [x] Define source union type
- [x] Define GemmaScopeVariant interface

**Files:**
- `frontend/src/types/sae.ts`

### Task 6.2: Create API Client
- [x] list() function
- [x] download() function
- [x] getGemmaScopeVariants() function
- [x] delete() function

**Files:**
- `frontend/src/api/saes.ts`

### Task 6.3: Create SAEs Store
- [x] Combined SAE list state
- [x] Source and model filters
- [x] Download action

**Files:**
- `frontend/src/stores/saesStore.ts`

### Task 6.4: Create SAECard Component
- [x] Display SAE info
- [x] Source badge
- [x] Feature count (if trained)
- [x] Action buttons

**Files:**
- `frontend/src/components/saes/SAECard.tsx`

### Task 6.5: Create DownloadFromHF Modal
- [x] Custom repo mode
- [x] Gemma Scope mode
- [x] Variant selector
- [x] Progress display

**Files:**
- `frontend/src/components/saes/DownloadFromHF.tsx`

---

## Relevant Files Summary

### Backend
| File | Purpose |
|------|---------|
| `backend/src/models/external_sae.py` | SQLAlchemy model |
| `backend/src/schemas/sae.py` | Pydantic schemas |
| `backend/src/services/sae_manager_service.py` | Unified management |
| `backend/src/services/huggingface_sae_service.py` | HF downloads |
| `backend/src/ml/community_format.py` | Format handling |
| `backend/src/ml/sae_loader.py` | SAE loading |
| `backend/src/api/v1/endpoints/saes.py` | API routes |

### Frontend
| File | Purpose |
|------|---------|
| `frontend/src/types/sae.ts` | TypeScript types |
| `frontend/src/api/saes.ts` | API client |
| `frontend/src/stores/saesStore.ts` | Zustand store |
| `frontend/src/components/saes/SAECard.tsx` | Card component |
| `frontend/src/components/saes/DownloadFromHF.tsx` | Download modal |
| `frontend/src/components/panels/SAEsPanel.tsx` | Panel |

---

*Related: [PRD](../prds/005_FPRD|SAE_Management.md) | [TDD](../tdds/005_FTDD|SAE_Management.md) | [TID](../tids/005_FTID|SAE_Management.md)*
