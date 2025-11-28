# Feature Extraction Architecture Refactor

## Overview

This document details the migration plan to:
1. Move feature extraction from the **Training tab** to the **Extraction tab**
2. Support extraction from both **miStudio-trained SAEs** and **external SAEs** (downloaded from HuggingFace)
3. Keep SAE training on the Training tab, all extraction on the Extraction tab

---

## Current Architecture Analysis

### Database Schema Issues

The current schema tightly couples extraction to training:

```
extraction_jobs
├── id: str (PK) - Format: ext_{training_id}_{timestamp}
├── training_id: str (FK) - REQUIRED, references trainings.id
├── celery_task_id: str
├── config: JSONB
├── status: enum
├── ...

features
├── id: str (PK) - Format: feat_{training_id}_{neuron_index}
├── training_id: str (FK) - REQUIRED, references trainings.id
├── extraction_job_id: str (FK) - REQUIRED
├── neuron_index: int
├── ...
```

**Problem**: `training_id` is required in both tables, making it impossible to extract features from external SAEs that have no associated training.

### External SAEs Table

```
external_saes
├── id: str (PK) - Format: sae_{uuid}
├── training_id: str (FK) - OPTIONAL (for trained SAEs exported to SAEs tab)
├── source: enum (huggingface, local, trained)
├── model_name: str - e.g., "google/gemma-2-2b"
├── layer: int
├── n_features: int
├── d_model: int
├── architecture: str - e.g., "jumprelu", "standard"
├── local_path: str - Path to downloaded SAE weights
├── ...
```

**Opportunity**: External SAEs already have all the metadata needed for extraction (model_name, layer, local_path).

### Current UI Flow

```
Training Tab
└── TrainingCard (completed training)
    └── "Start Extraction" button
        └── StartExtractionModal
            └── API: POST /api/v1/trainings/{training_id}/extract-features

Extraction Tab
└── ExtractionsPanel
    └── ExtractionJobCard (view only, no start capability)
        └── Features browser
```

**Problem**: Extraction can only be started from Training tab, tied to a training.

### Desired UI Flow

```
Training Tab
└── TrainingCard (completed training)
    └── "Import to SAEs" button (existing)
    [NO extraction capability - training only]

SAEs Tab
└── SAECard (view/manage SAEs)
    [NO extraction capability - management only]

Extraction Tab (SINGLE PLACE FOR ALL EXTRACTIONS)
└── "Start New Extraction" button
    └── SAE Selector (shows ALL SAEs - trained + external)
        └── StartExtractionModal
└── ExtractionsPanel
    └── ExtractionJobCard
        └── Features browser
```

---

## Proposed Architecture

### Database Schema Changes

**Option A: Add external_sae_id to existing tables (Recommended)**

```sql
-- Migration: Add external_sae_id to extraction_jobs
ALTER TABLE extraction_jobs
    ADD COLUMN external_sae_id VARCHAR(255) REFERENCES external_saes(id) ON DELETE CASCADE,
    ALTER COLUMN training_id DROP NOT NULL;

-- Add constraint: exactly one source must be set
ALTER TABLE extraction_jobs ADD CONSTRAINT check_single_source
    CHECK ((training_id IS NOT NULL AND external_sae_id IS NULL) OR
           (training_id IS NULL AND external_sae_id IS NOT NULL));

-- Same for features table
ALTER TABLE features
    ADD COLUMN external_sae_id VARCHAR(255) REFERENCES external_saes(id) ON DELETE CASCADE,
    ALTER COLUMN training_id DROP NOT NULL;

ALTER TABLE features ADD CONSTRAINT check_single_source
    CHECK ((training_id IS NOT NULL AND external_sae_id IS NULL) OR
           (training_id IS NULL AND external_sae_id IS NOT NULL));
```

**New ID Formats:**
- Extraction from training: `ext_{training_id}_{timestamp}`
- Extraction from SAE: `ext_sae_{sae_id}_{timestamp}`
- Feature from training: `feat_{training_id}_{neuron_index}`
- Feature from SAE: `feat_sae_{sae_id}_{neuron_index}`

### New API Endpoints

```
POST /api/v1/saes/{sae_id}/extract-features
  - Start extraction from external SAE
  - Request body: ExtractionConfigRequest (same as training extraction)

GET /api/v1/saes/{sae_id}/extraction-status
  - Get extraction status for an SAE

POST /api/v1/saes/{sae_id}/cancel-extraction
  - Cancel active extraction for an SAE

GET /api/v1/saes/{sae_id}/features
  - List features for an SAE (post-extraction)
```

### Updated UI Flow

```
Training Tab
└── TrainingCard (completed training)
    └── "Import to SAEs" button (existing)
        └── Creates entry in external_saes with source=trained
    [REMOVED: "Start Extraction" button]

SAEs Tab
└── SAECard
    └── View/manage SAEs only
    [NO extraction capability - just SAE management]

Extraction Tab (SINGLE PLACE FOR ALL EXTRACTIONS)
└── "Start New Extraction" button (NEW)
    └── SAE Selector modal (shows ALL SAEs: trained + HuggingFace)
        └── StartExtractionModal (moved from features/)
            └── API: POST /api/v1/saes/{sae_id}/extract-features
└── ExtractionsPanel
    └── ExtractionJobCard
        └── Source indicator (miStudio Trained | HuggingFace)
        └── Features browser (unchanged)
```

---

## Implementation Tasks

### Phase 1: Database Schema Migration [Backend]

- [ ] **1.1** Create Alembic migration to add `external_sae_id` to `extraction_jobs`
  - Add nullable FK column
  - Make `training_id` nullable
  - Add CHECK constraint for exactly one source

- [ ] **1.2** Create Alembic migration to add `external_sae_id` to `features`
  - Add nullable FK column
  - Make `training_id` nullable
  - Add CHECK constraint

- [ ] **1.3** Update SQLAlchemy models
  - [backend/src/models/extraction_job.py](backend/src/models/extraction_job.py)
    - Add `external_sae_id` column
    - Add relationship to ExternalSAE
    - Make `training_id` nullable
  - [backend/src/models/feature.py](backend/src/models/feature.py)
    - Add `external_sae_id` column
    - Add relationship to ExternalSAE
    - Make `training_id` nullable

- [ ] **1.4** Update Pydantic schemas
  - [backend/src/schemas/extraction.py](backend/src/schemas/extraction.py)
    - Add `external_sae_id` field
    - Add `source_type` computed field (training|sae)
    - Update validators

### Phase 2: Backend Service Layer [Backend]

- [ ] **2.1** Refactor ExtractionService
  - [backend/src/services/extraction_service.py](backend/src/services/extraction_service.py)
  - Create new method: `extract_features_for_sae(sae_id, config)`
  - Refactor `extract_features_for_training` to share logic
  - Extract common SAE loading logic into helper method
  - Support loading SAE from either:
    - Training checkpoint path
    - External SAE local_path (using `load_sae_auto_detect`)

- [ ] **2.2** Update FeatureService
  - [backend/src/services/feature_service.py](backend/src/services/feature_service.py)
  - Support features by `external_sae_id`
  - Update ID generation logic

- [ ] **2.3** Update Celery tasks
  - [backend/src/workers/extraction_tasks.py](backend/src/workers/extraction_tasks.py)
  - Create new task: `extract_features_from_sae_task(sae_id, config)`
  - Share core logic with training extraction task

### Phase 3: API Endpoints [Backend]

- [ ] **3.1** Create new SAE extraction endpoints
  - [backend/src/api/v1/endpoints/saes.py](backend/src/api/v1/endpoints/saes.py)
  - `POST /api/v1/saes/{sae_id}/extract-features`
  - `GET /api/v1/saes/{sae_id}/extraction-status`
  - `POST /api/v1/saes/{sae_id}/cancel-extraction`
  - `GET /api/v1/saes/{sae_id}/features`

- [ ] **3.2** Update extractions list endpoint
  - [backend/src/api/v1/endpoints/extractions.py](backend/src/api/v1/endpoints/extractions.py)
  - Return `source_type` (training|sae) for each extraction
  - Include SAE metadata when source is SAE

- [ ] **3.3** (Optional) Keep training extraction endpoints working
  - Existing `/api/v1/trainings/{id}/extract-features` can stay for backward compat
  - Consider deprecation warning

### Phase 4: Frontend - Shared Components [Frontend]

- [ ] **4.1** Move StartExtractionModal to shared location
  - FROM: [frontend/src/components/features/StartExtractionModal.tsx](frontend/src/components/features/StartExtractionModal.tsx)
  - TO: `frontend/src/components/extraction/StartExtractionModal.tsx`
  - Make it accept either `training` OR `sae` prop
  - Update API call based on source type

- [ ] **4.2** Create SAE selector component (for Extraction tab)
  - `frontend/src/components/extraction/SAESelector.tsx`
  - Dropdown/modal to select an SAE for extraction
  - Shows both trained SAEs (with training_id) and external SAEs

- [ ] **4.3** Update ExtractionJobCard to show source
  - [frontend/src/components/features/ExtractionJobCard.tsx](frontend/src/components/features/ExtractionJobCard.tsx)
  - Add source indicator (Training vs External SAE)
  - Show appropriate metadata based on source

### Phase 5: Frontend - Training Tab Changes [Frontend]

- [ ] **5.1** Remove "Start Extraction" from TrainingCard
  - [frontend/src/components/training/TrainingCard.tsx](frontend/src/components/training/TrainingCard.tsx)
  - Remove lines 563-576 (Start Extraction button)
  - Remove `showExtractionModal` state
  - Remove `StartExtractionModal` import and usage
  - Keep "Import to SAEs" button

### Phase 6: Frontend - Extraction Tab Changes [Frontend]

- [ ] **6.1** Add "Start New Extraction" capability to ExtractionsPanel
  - [frontend/src/components/panels/ExtractionsPanel.tsx](frontend/src/components/panels/ExtractionsPanel.tsx)
  - Add "Start Extraction" button in header
  - Create SAE Selector modal component
  - After SAE selection, open StartExtractionModal

- [ ] **6.2** Create SAE Selector component
  - `frontend/src/components/extraction/SAESelector.tsx`
  - List ALL available SAEs (both trained and HuggingFace)
  - Show SAE metadata: name, source, model, layer, features count
  - Filter by source type (trained/huggingface)
  - Search by name

- [ ] **6.3** Update features store API calls
  - [frontend/src/stores/featuresStore.ts](frontend/src/stores/featuresStore.ts)
  - Add `startExtractionFromSAE(saeId, config)` action
  - Support features by SAE ID

### Phase 7: Testing & Documentation [Testing]

- [ ] **7.1** Backend unit tests
  - Test extraction from external SAE
  - Test feature ID generation
  - Test constraint validation

- [ ] **7.2** Integration tests
  - End-to-end extraction from downloaded SAE
  - Verify features are browsable in Extraction tab

- [ ] **7.3** Update documentation
  - Update user guide for new workflow
  - Add API documentation for new endpoints

---

## Data Model Diagram (After Migration)

```
┌──────────────────┐       ┌───────────────────┐       ┌─────────────────┐
│    trainings     │       │  external_saes    │       │     models      │
├──────────────────┤       ├───────────────────┤       ├─────────────────┤
│ id (PK)          │──┐    │ id (PK)           │──┐    │ id (PK)         │
│ model_id (FK)    │  │    │ name              │  │    │ name            │
│ dataset_id (FK)  │  │    │ source            │  │    │ ...             │
│ checkpoint_path  │  │    │ training_id (FK)──│──│────│                 │
│ ...              │  │    │ model_id (FK) ────│──│──┘ └─────────────────┘
└──────────────────┘  │    │ model_name        │  │
                      │    │ layer             │  │
                      │    │ local_path        │  │
                      │    │ architecture      │  │
                      │    │ ...               │  │
                      │    └───────────────────┘  │
                      │                           │
                      ▼                           ▼
              ┌───────────────────────────────────────────┐
              │            extraction_jobs                 │
              ├───────────────────────────────────────────┤
              │ id (PK)                                    │
              │ training_id (FK) ─────────────────────────┤──(NULL or set)
              │ external_sae_id (FK) ─────────────────────┤──(NULL or set)
              │ config: JSONB                             │
              │ status                                     │
              │ progress                                   │
              │ ...                                        │
              └───────────────────────────────────────────┘
                      │
                      ▼
              ┌───────────────────────────────────────────┐
              │              features                      │
              ├───────────────────────────────────────────┤
              │ id (PK)                                    │
              │ training_id (FK) ─────────────────────────┤──(NULL or set)
              │ external_sae_id (FK) ─────────────────────┤──(NULL or set)
              │ extraction_job_id (FK)                     │
              │ neuron_index                               │
              │ name                                       │
              │ activation_frequency                       │
              │ ...                                        │
              └───────────────────────────────────────────┘
                      │
                      ▼
              ┌───────────────────────────────────────────┐
              │          feature_activations               │
              ├───────────────────────────────────────────┤
              │ id (PK)                                    │
              │ feature_id (FK)                            │
              │ activation_value                           │
              │ context_tokens                             │
              │ ...                                        │
              └───────────────────────────────────────────┘
```

---

## API Changes Summary

### New Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/saes/{sae_id}/extract-features` | Start extraction from external SAE |
| GET | `/api/v1/saes/{sae_id}/extraction-status` | Get extraction status for SAE |
| POST | `/api/v1/saes/{sae_id}/cancel-extraction` | Cancel active extraction |
| GET | `/api/v1/saes/{sae_id}/features` | List features for SAE |

### Modified Endpoints

| Method | Endpoint | Change |
|--------|----------|--------|
| GET | `/api/v1/extractions` | Returns `source_type` field (training\|sae) |
| GET | `/api/v1/extractions/{id}` | Returns `source_type` and related SAE/training info |

### Deprecated (but still functional)

| Method | Endpoint | Note |
|--------|----------|------|
| POST | `/api/v1/trainings/{id}/extract-features` | Still works, but "Start Extraction" button removed from UI |

---

## Estimated Effort

| Phase | Complexity | Estimated Time |
|-------|------------|----------------|
| Phase 1: Database Migration | Medium | 2-3 hours |
| Phase 2: Backend Services | High | 4-6 hours |
| Phase 3: API Endpoints | Medium | 2-3 hours |
| Phase 4: Shared Components | Medium | 2-3 hours |
| Phase 5: Training Tab | Low | 1 hour |
| Phase 6: Extraction Tab | Medium | 3-4 hours |
| Phase 7: Testing | Medium | 3-4 hours |
| **Total** | | **17-24 hours** |

---

## Risks & Mitigations

1. **Data Migration Risk**: Existing extractions have non-null `training_id`
   - Mitigation: Migration only makes column nullable, doesn't modify existing data

2. **API Breaking Change**: Frontend using old endpoints
   - Mitigation: Keep old training endpoints working, add new SAE endpoints

3. **SAE Format Compatibility**: External SAEs may have different formats
   - Mitigation: Use `load_sae_auto_detect()` which already supports community, gemma_scope, mistudio formats

4. **Model Compatibility**: External SAE may need specific model loaded
   - Mitigation: Use `model_id` FK from external_saes to load correct model
   - Fallback: Infer model from `model_name` field

---

## Success Criteria

1. User can download SAE from HuggingFace (existing functionality)
2. User can start extraction from SAEs tab on downloaded SAE
3. Extraction appears in Extractions tab with proper source indicator
4. Features are browsable in ExtractionJobCard
5. Labeled features can be used for steering (existing functionality)
6. Training tab no longer shows "Start Extraction" button
7. All existing trained SAE extractions continue to work

---

## Files to Modify

### Backend (12 files)

```
backend/
├── alembic/versions/
│   └── xxx_add_external_sae_to_extraction.py    [NEW - migration]
├── src/
│   ├── models/
│   │   ├── extraction_job.py                     [MODIFY]
│   │   └── feature.py                            [MODIFY]
│   ├── schemas/
│   │   └── extraction.py                         [MODIFY]
│   ├── services/
│   │   ├── extraction_service.py                 [MODIFY]
│   │   └── feature_service.py                    [MODIFY]
│   ├── api/v1/endpoints/
│   │   ├── saes.py                               [MODIFY - add endpoints]
│   │   └── extractions.py                        [MODIFY]
│   └── workers/
│       └── extraction_tasks.py                   [MODIFY]
```

### Frontend (9 files)

```
frontend/src/
├── components/
│   ├── extraction/
│   │   ├── StartExtractionModal.tsx              [MOVE from features/]
│   │   └── SAESelector.tsx                       [NEW]
│   ├── features/
│   │   └── ExtractionJobCard.tsx                 [MODIFY]
│   ├── training/
│   │   └── TrainingCard.tsx                      [MODIFY - remove extraction]
│   └── panels/
│       └── ExtractionsPanel.tsx                  [MODIFY - add Start Extraction]
├── stores/
│   ├── featuresStore.ts                          [MODIFY]
│   └── saesStore.ts                              [MODIFY - add fetchSAEs for selector]
└── types/
    └── features.ts                               [MODIFY]
```
