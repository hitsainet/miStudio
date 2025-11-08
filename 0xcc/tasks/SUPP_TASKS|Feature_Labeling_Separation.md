# Supplemental Task: Separate Feature Extraction and Semantic Labeling

## Overview
Decouple feature extraction from semantic labeling to allow independent execution and re-labeling of already-extracted features without re-running the entire extraction pipeline.

**Priority:** P1 - High Value
**Estimated Effort:** 16-24 hours
**Dependencies:** None
**Related Files:**
- `backend/src/services/extraction_service.py`
- `backend/src/services/openai_labeling_service.py`
- `backend/src/workers/extraction_tasks.py`
- `backend/src/api/v1/endpoints/features.py`
- `backend/src/models/extraction_job.py`
- `frontend/src/components/panels/FeatureDiscoveryPanel.tsx`

## Business Value
- **Flexibility:** Re-label features with different LLM models or prompts without re-extracting
- **Cost Optimization:** Extraction is expensive (GPU compute), labeling is cheap (API calls)
- **Experimentation:** Try different labeling strategies on the same feature set
- **Recovery:** Fix failed labeling without losing extraction work
- **Development:** Iterate on labeling prompts and quality

## Current State Analysis

### Problems with Current Implementation
1. **Tight Coupling:** Labeling is embedded in `ExtractionService.extract_features_for_training()`
2. **No Re-labeling:** Once extraction completes, labels cannot be changed
3. **All-or-Nothing:** Extraction must run to completion including labeling
4. **Failure Impact:** Labeling failure after 1.5 hours of extraction wastes all work
5. **No Experimentation:** Cannot test different labeling approaches on same features

### Current Workflow
```
Start Extraction → Extract Features → Compute Activations → Label with LLM → Complete
                    (50 mins)         (30 mins)            (20 mins)      DONE

If labeling fails at minute 100 → All 100 minutes of work lost
```

### Desired Workflow
```
Start Extraction → Extract Features → Compute Activations → Complete (unlabeled)
                    (50 mins)         (30 mins)              DONE ✓

Later (anytime):
Start Labeling → Label Features → Complete
                 (20 mins)        DONE ✓

Can re-run labeling multiple times without re-extracting features
```

## Architecture Design

### Database Schema Changes

#### Option A: Add Labeling Job Table (Recommended)
```sql
CREATE TABLE labeling_jobs (
    id VARCHAR PRIMARY KEY,
    extraction_job_id VARCHAR NOT NULL REFERENCES extraction_jobs(id),
    status VARCHAR NOT NULL,  -- 'queued', 'labeling', 'completed', 'failed'
    progress FLOAT DEFAULT 0.0,
    features_labeled INTEGER DEFAULT 0,
    total_features INTEGER,

    -- Configuration
    labeling_method VARCHAR NOT NULL,  -- 'openai', 'local', 'none'
    openai_model VARCHAR,
    openai_api_key VARCHAR,
    local_model VARCHAR,

    -- Metadata
    celery_task_id VARCHAR,
    error_message TEXT,
    statistics JSONB,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_labeling_jobs_extraction ON labeling_jobs(extraction_job_id);
CREATE INDEX idx_labeling_jobs_status ON labeling_jobs(status);
```

#### Option B: Add Fields to Features Table
```sql
ALTER TABLE features ADD COLUMN labeling_job_id VARCHAR;
ALTER TABLE features ADD COLUMN labeled_at TIMESTAMP WITH TIME ZONE;
```

**Recommendation:** Use Option A (separate table) for better tracking and history.

### Service Layer Design

#### New: `LabelingService`
```python
class LabelingService:
    """
    Service for semantic labeling of already-extracted features.
    Completely independent of extraction process.
    """

    async def start_labeling(
        self,
        extraction_id: str,
        config: Dict[str, Any]
    ) -> LabelingJob:
        """Queue a labeling job for an extraction."""
        pass

    async def label_features(
        self,
        labeling_job_id: str
    ) -> Dict[str, Any]:
        """Execute labeling for all features in an extraction."""
        pass

    async def get_labeling_status(
        self,
        labeling_job_id: str
    ) -> Dict[str, Any]:
        """Get current labeling progress."""
        pass

    async def cancel_labeling(
        self,
        labeling_job_id: str
    ) -> None:
        """Cancel an active labeling job."""
        pass
```

#### Modified: `ExtractionService`
```python
class ExtractionService:
    """
    Service for feature extraction (NO LABELING).
    Extraction now completes without labels.
    """

    async def extract_features_for_training(
        self,
        training_id: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract features WITHOUT labeling.
        Features are created with label_source='none' and name='feature_XXXXX'.
        """
        # 1. Load SAE model
        # 2. Extract feature activations
        # 3. Compute interpretability scores
        # 4. Collect max-activating examples
        # 5. Save features (unlabeled)
        # 6. Complete extraction
        # NO LABELING STEP
        pass
```

### API Endpoints

#### New: Labeling Endpoints
```python
# Start labeling for an extraction
POST /api/v1/extractions/{extraction_id}/label-features
Body: {
    "labeling_method": "openai",  # or "local", "none"
    "openai_model": "gpt4-mini",
    "openai_api_key": "sk-...",  # optional, uses default if not provided
    "local_model": "phi3"  # only if labeling_method = "local"
}
Response: LabelingJobResponse

# Get labeling status
GET /api/v1/labeling-jobs/{labeling_job_id}
Response: LabelingStatusResponse

# Cancel labeling
POST /api/v1/labeling-jobs/{labeling_job_id}/cancel
Response: 200 OK

# List all labeling jobs
GET /api/v1/labeling-jobs?extraction_id={extraction_id}
Response: LabelingJobListResponse

# Re-label (creates new labeling job)
POST /api/v1/extractions/{extraction_id}/relabel
Body: {
    "labeling_method": "openai",
    "openai_model": "gpt-4o-mini"
}
Response: LabelingJobResponse
```

### Celery Tasks

#### New: `label_features_task`
```python
@celery_app.task(
    bind=True,
    base=DatabaseTask,
    name="label_features",
    max_retries=0
)
def label_features_task(
    self,
    labeling_job_id: str
) -> Dict[str, Any]:
    """
    Celery task for labeling features.
    Completely independent of extraction.
    """
    pass
```

### Frontend Components

#### New: Labeling UI
- **LabelingButton** - Trigger labeling on unlabeled extraction
- **RelabelButton** - Re-label already labeled features
- **LabelingProgressModal** - Show labeling progress
- **LabelingConfigModal** - Configure labeling method and model

## Task Breakdown

### Phase 1: Backend - Database & Models (4-6 hours)

#### Task 1.1: Create Labeling Job Model
- [ ] Create database migration for `labeling_jobs` table
- [ ] Create `LabelingJob` SQLAlchemy model
- [ ] Create `LabelingStatus` enum
- [ ] Add relationship to `ExtractionJob`
- [ ] Create Alembic migration
- [ ] Run migration on dev database

**Files:**
- `backend/alembic/versions/XXX_add_labeling_jobs.py` (new)
- `backend/src/models/labeling_job.py` (new)
- `backend/src/models/extraction_job.py` (modify - add relationship)

#### Task 1.2: Create Labeling Schemas
- [ ] Create `LabelingConfigRequest` schema
- [ ] Create `LabelingStatusResponse` schema
- [ ] Create `LabelingJobResponse` schema
- [ ] Create `LabelingJobListResponse` schema
- [ ] Add validation for labeling methods

**Files:**
- `backend/src/schemas/labeling.py` (new)

### Phase 2: Backend - Service Layer (6-8 hours)

#### Task 2.1: Create LabelingService
- [ ] Create `LabelingService` class
- [ ] Implement `start_labeling()` - create labeling job
- [ ] Implement `label_features()` - core labeling logic
- [ ] Implement `get_labeling_status()` - status queries
- [ ] Implement `cancel_labeling()` - cancel job
- [ ] Implement `list_labeling_jobs()` - list jobs for extraction
- [ ] Add WebSocket progress emissions
- [ ] Add error handling and status updates
- [ ] Extract labeling logic from ExtractionService

**Files:**
- `backend/src/services/labeling_service.py` (new)

#### Task 2.2: Refactor ExtractionService
- [ ] Remove labeling code from `extract_features_for_training()`
- [ ] Features now created with `label_source='none'`
- [ ] Features created with fallback names `feature_XXXXX`
- [ ] Remove OpenAI labeling service initialization
- [ ] Update extraction completion logic
- [ ] Update tests to reflect no labeling

**Files:**
- `backend/src/services/extraction_service.py` (modify)
- `backend/tests/unit/test_extraction_service.py` (modify)

#### Task 2.3: Refactor OpenAILabelingService
- [ ] Make service completely standalone
- [ ] Remove dependencies on extraction context
- [ ] Accept list of features to label
- [ ] Return updated feature records
- [ ] Add batch processing with progress callbacks
- [ ] Add retry logic for API failures

**Files:**
- `backend/src/services/openai_labeling_service.py` (modify)

### Phase 3: Backend - Celery Tasks (2-3 hours)

#### Task 3.1: Create Labeling Task
- [ ] Create `label_features_task` Celery task
- [ ] Implement main labeling workflow
- [ ] Add progress tracking with WebSocket emissions
- [ ] Add error handling and status updates
- [ ] Add task routing to `labeling` queue
- [ ] Update Celery configuration

**Files:**
- `backend/src/workers/labeling_tasks.py` (new)
- `backend/src/core/celery_app.py` (modify - add routing)

#### Task 3.2: Update WebSocket Emitters
- [ ] Add `emit_labeling_progress()` function
- [ ] Add `emit_labeling_completed()` function
- [ ] Add `emit_labeling_failed()` function
- [ ] Define channel naming: `labeling/{labeling_job_id}/progress`

**Files:**
- `backend/src/workers/websocket_emitter.py` (modify)

### Phase 4: Backend - API Endpoints (3-4 hours)

#### Task 4.1: Create Labeling Endpoints
- [ ] Create `POST /api/v1/extractions/{extraction_id}/label-features`
- [ ] Create `GET /api/v1/labeling-jobs/{labeling_job_id}`
- [ ] Create `POST /api/v1/labeling-jobs/{labeling_job_id}/cancel`
- [ ] Create `GET /api/v1/labeling-jobs`
- [ ] Create `POST /api/v1/extractions/{extraction_id}/relabel`
- [ ] Add request validation
- [ ] Add error handling (404, 409, 422)
- [ ] Add OpenAPI documentation

**Files:**
- `backend/src/api/v1/endpoints/labeling.py` (new)
- `backend/src/api/v1/__init__.py` (modify - add router)

#### Task 4.2: Update Extraction Endpoints
- [ ] Update extraction config to make labeling optional
- [ ] Add `skip_labeling` flag to extraction config
- [ ] Update extraction response to show labeling status
- [ ] Update extraction list to include labeling info

**Files:**
- `backend/src/api/v1/endpoints/features.py` (modify)
- `backend/src/schemas/extraction.py` (modify)

### Phase 5: Frontend - API Client (1-2 hours)

#### Task 5.1: Create Labeling API Client
- [ ] Create `labelingApi.ts` with API functions
- [ ] Implement `startLabeling()`
- [ ] Implement `getLabelingStatus()`
- [ ] Implement `cancelLabeling()`
- [ ] Implement `listLabelingJobs()`
- [ ] Implement `relabelFeatures()`
- [ ] Add TypeScript types for requests/responses

**Files:**
- `frontend/src/api/labeling.ts` (new)
- `frontend/src/types/labeling.ts` (new)

### Phase 6: Frontend - State Management (1-2 hours)

#### Task 6.1: Create Labeling Store
- [ ] Create `labelingStore.ts` Zustand store
- [ ] Add state for active labeling jobs
- [ ] Add state for labeling progress
- [ ] Implement `startLabeling()` action
- [ ] Implement `cancelLabeling()` action
- [ ] Implement `updateProgress()` action
- [ ] Add WebSocket connection management

**Files:**
- `frontend/src/stores/labelingStore.ts` (new)

#### Task 6.2: Create Labeling WebSocket Hook
- [ ] Create `useLabelingWebSocket.ts` hook
- [ ] Subscribe to `labeling/{labeling_job_id}/progress` channel
- [ ] Handle progress events
- [ ] Handle completed events
- [ ] Handle failed events
- [ ] Update labeling store on events

**Files:**
- `frontend/src/hooks/useLabelingWebSocket.ts` (new)

### Phase 7: Frontend - UI Components (4-6 hours)

#### Task 7.1: Create Labeling Configuration Modal
- [ ] Create `LabelingConfigModal.tsx`
- [ ] Add labeling method selector (OpenAI, Local, None)
- [ ] Add OpenAI model dropdown
- [ ] Add OpenAI API key input (optional)
- [ ] Add local model dropdown
- [ ] Add validation
- [ ] Add start button

**Files:**
- `frontend/src/components/labeling/LabelingConfigModal.tsx` (new)

#### Task 7.2: Create Labeling Progress Modal
- [ ] Create `LabelingProgressModal.tsx`
- [ ] Show progress bar (0-100%)
- [ ] Show features labeled count
- [ ] Show elapsed time
- [ ] Show estimated time remaining
- [ ] Add cancel button
- [ ] Add WebSocket connection status

**Files:**
- `frontend/src/components/labeling/LabelingProgressModal.tsx` (new)

#### Task 7.3: Update Feature Discovery Panel
- [ ] Add "Label Features" button for unlabeled extractions
- [ ] Add "Re-label Features" button for labeled extractions
- [ ] Show labeling status badge (unlabeled, labeling, labeled)
- [ ] Add labeling progress indicator
- [ ] Update extraction card to show labeling info
- [ ] Add labeling job history view

**Files:**
- `frontend/src/components/panels/FeatureDiscoveryPanel.tsx` (modify)
- `frontend/src/components/extraction/ExtractionCard.tsx` (modify)

#### Task 7.4: Create Labeling History View
- [ ] Create `LabelingHistoryModal.tsx`
- [ ] Show list of all labeling jobs for an extraction
- [ ] Show labeling method, model, status
- [ ] Show created/completed timestamps
- [ ] Allow comparison of different labeling attempts
- [ ] Show which labels are currently active

**Files:**
- `frontend/src/components/labeling/LabelingHistoryModal.tsx` (new)

### Phase 8: Testing & Documentation (3-4 hours)

#### Task 8.1: Backend Tests
- [ ] Unit tests for `LabelingService`
- [ ] Unit tests for `label_features_task`
- [ ] Integration tests for labeling endpoints
- [ ] Test extraction without labeling
- [ ] Test labeling separately
- [ ] Test re-labeling
- [ ] Test concurrent labeling jobs
- [ ] Test labeling cancellation

**Files:**
- `backend/tests/unit/test_labeling_service.py` (new)
- `backend/tests/unit/test_labeling_tasks.py` (new)
- `backend/tests/integration/test_labeling_endpoints.py` (new)

#### Task 8.2: Frontend Tests
- [ ] Component tests for labeling modals
- [ ] Store tests for labeling actions
- [ ] Hook tests for WebSocket integration
- [ ] Integration tests for labeling workflow

**Files:**
- `frontend/src/components/labeling/__tests__/` (new)
- `frontend/src/stores/__tests__/labelingStore.test.ts` (new)

#### Task 8.3: Documentation
- [ ] Update API documentation with labeling endpoints
- [ ] Add labeling workflow diagram
- [ ] Document labeling configuration options
- [ ] Add examples of re-labeling scenarios
- [ ] Update CLAUDE.md with new patterns

**Files:**
- `backend/README.md` (modify)
- `0xcc/docs/Feature_Labeling_Architecture.md` (new)

### Phase 9: Migration & Deployment (2-3 hours)

#### Task 9.1: Data Migration
- [ ] Write migration script for existing extractions
- [ ] Create initial labeling jobs for already-labeled extractions
- [ ] Backfill `labeling_job_id` in features table
- [ ] Validate migration results

**Files:**
- `backend/scripts/migrate_existing_labels.py` (new)

#### Task 9.2: Deployment
- [ ] Update Celery worker to include `labeling` queue
- [ ] Update Celery Beat if needed
- [ ] Run database migrations
- [ ] Run data migration script
- [ ] Deploy backend changes
- [ ] Deploy frontend changes
- [ ] Verify WebSocket connectivity

## Success Criteria

### Functional Requirements
- ✅ Can extract features without labeling them
- ✅ Can label features after extraction completes
- ✅ Can re-label features with different methods/models
- ✅ Labeling progress shows in real-time via WebSocket
- ✅ Multiple labeling jobs can exist for same extraction
- ✅ Can compare labels from different labeling attempts
- ✅ Failed labeling doesn't affect extraction data
- ✅ Can cancel in-progress labeling job

### Non-Functional Requirements
- ✅ Extraction performance unchanged (no regression)
- ✅ Labeling can process 44k features in < 30 minutes
- ✅ WebSocket updates every 2 seconds during labeling
- ✅ API response time < 200ms for status queries
- ✅ Database migration completes in < 5 minutes
- ✅ No data loss during migration

### User Experience Requirements
- ✅ Clear visual indication of labeling status
- ✅ Easy to trigger labeling/re-labeling
- ✅ Progress visible and intuitive
- ✅ Errors clearly communicated
- ✅ Can experiment with different labeling configs

## Rollout Plan

### Stage 1: Backend Core (Days 1-2)
- Database migration
- LabelingService implementation
- Celery task creation
- Basic API endpoints

### Stage 2: Frontend Core (Day 3)
- API client
- State management
- Basic UI components

### Stage 3: Integration (Day 4)
- WebSocket integration
- End-to-end testing
- Bug fixes

### Stage 4: Polish (Day 5)
- Advanced UI features
- Documentation
- Performance optimization

### Stage 5: Migration & Deployment (Day 6)
- Data migration
- Production deployment
- Monitoring

## Risks & Mitigations

### Risk 1: Breaking Existing Extractions
**Mitigation:**
- Backward compatibility in extraction service
- Gradual rollout with feature flag
- Comprehensive testing before deployment

### Risk 2: Performance Impact on Large Feature Sets
**Mitigation:**
- Batch processing with progress tracking
- Rate limiting for API calls
- Efficient database queries with indexes

### Risk 3: WebSocket Connection Stability
**Mitigation:**
- Automatic fallback to HTTP polling
- Reconnection logic
- Status persistence in database

### Risk 4: Data Migration Complexity
**Mitigation:**
- Test migration on copy of production database
- Rollback plan if migration fails
- Validation scripts to verify data integrity

## Future Enhancements

1. **Labeling Quality Metrics**
   - Track labeling quality scores
   - Compare quality across different models
   - Suggest best labeling method per dataset

2. **Batch Re-labeling**
   - Re-label multiple extractions at once
   - Queue labeling jobs efficiently
   - Prioritize based on importance

3. **Custom Labeling Prompts**
   - Allow users to customize prompts
   - Save prompt templates
   - Share prompts across team

4. **Labeling Cost Tracking**
   - Track API costs per labeling job
   - Show cost estimates before labeling
   - Budget controls and alerts

5. **A/B Testing for Labels**
   - Compare labels from different models side-by-side
   - Vote on best labels
   - Export label comparison reports

## Notes

- This refactor significantly improves development velocity for labeling features
- OpenAI API cost for 44k features: ~$6-8 (vs hours of GPU compute for extraction)
- Allows experimentation with new labeling techniques without re-extraction
- Better separation of concerns in codebase
- Easier to test and maintain labeling logic independently
