# Fix: Prevent Extraction Job Auto-Restart

**Issue ID:** FIX-EXT-001
**Priority:** CRITICAL
**Created:** 2025-10-28
**Status:** In Progress

## Problem Summary

Feature extraction jobs automatically restart after completion due to lack of idempotency checks and missing Celery task ID tracking. This causes:
- Duplicate processing (80 minutes wasted)
- Database status corruption (completed → extracting)
- GPU memory not released
- User confusion

**Root Cause:** Task implementation is not idempotent - it can be executed multiple times on the same extraction job without proper status validation.

---

## Tasks

### Phase 1: Immediate Protection (Critical)

#### Task 1.1: Add Idempotency Check in Service Layer
- **File:** `backend/src/services/extraction_service.py`
- **Location:** Line 580 (after `if not extraction_job:`)
- **Priority:** CRITICAL
- **Estimated Time:** 10 minutes

**Implementation:**
```python
if not extraction_job:
    raise ValueError(f"No extraction job found for training {training_id}")

# ADD: Idempotency check
if extraction_job.status == ExtractionStatus.COMPLETED.value:
    logger.warning(
        f"Extraction {extraction_job.id} already completed at {extraction_job.completed_at}. "
        f"Returning existing statistics."
    )
    return extraction_job.statistics or {}

if extraction_job.status == ExtractionStatus.FAILED.value:
    logger.warning(
        f"Extraction {extraction_job.id} previously failed. "
        f"Please create a new extraction job to retry."
    )
    return {}
```

**Testing:**
- [ ] Start extraction → complete successfully
- [ ] Try to call `extract_features_for_training()` again on completed extraction
- [ ] Verify early return with warning log
- [ ] Verify statistics returned from first run

**Status:** ⏳ Pending

---

#### Task 1.2: Add Task Retry Control
- **File:** `backend/src/workers/extraction_tasks.py`
- **Location:** Line 18
- **Priority:** CRITICAL
- **Estimated Time:** 5 minutes

**Implementation:**
```python
@celery_app.task(
    bind=True,
    base=DatabaseTask,
    name="extract_features",
    max_retries=0,  # ADD: No automatic retries
    autoretry_for=None,  # ADD: Explicit no auto-retry
)
def extract_features_task(
    self,
    training_id: str,
    config: Dict[str, Any]
) -> Dict[str, Any]:
```

**Testing:**
- [ ] Simulate task failure
- [ ] Verify task does NOT automatically retry
- [ ] Check Celery logs for retry attempts (should be none)

**Status:** ⏳ Pending

---

#### Task 1.3: Add Status Validation in Task Layer
- **File:** `backend/src/workers/extraction_tasks.py`
- **Location:** Line 44 (inside `with self.get_db()` block)
- **Priority:** HIGH
- **Estimated Time:** 15 minutes

**Implementation:**
```python
with self.get_db() as db:
    try:
        extraction_service = ExtractionService(db)

        # ADD: Pre-flight check for completed status
        from src.models.extraction_job import ExtractionJob, ExtractionStatus
        from sqlalchemy import desc

        extraction_job = db.query(ExtractionJob).filter(
            ExtractionJob.training_id == training_id
        ).order_by(desc(ExtractionJob.created_at)).first()

        if extraction_job and extraction_job.status == ExtractionStatus.COMPLETED.value:
            logger.info(
                f"Extraction {extraction_job.id} already completed at "
                f"{extraction_job.completed_at}, skipping re-execution"
            )
            return extraction_job.statistics or {}

        if extraction_job and extraction_job.status == ExtractionStatus.EXTRACTING.value:
            # Check if it's been running for too long (> 3 hours = likely stuck)
            from datetime import datetime, timezone, timedelta
            if extraction_job.updated_at:
                time_since_update = datetime.now(timezone.utc) - extraction_job.updated_at
                if time_since_update > timedelta(hours=3):
                    logger.warning(
                        f"Extraction {extraction_job.id} appears stuck "
                        f"(no update for {time_since_update}), allowing restart"
                    )
                else:
                    logger.info(
                        f"Extraction {extraction_job.id} is already in progress "
                        f"(last update: {time_since_update} ago), skipping"
                    )
                    return {}

        # Core extraction logic is delegated to service
        statistics = extraction_service.extract_features_for_training(training_id, config)
```

**Testing:**
- [ ] Start extraction, let it complete
- [ ] Queue another task for same training_id
- [ ] Verify task exits early with log message
- [ ] Start extraction, stop it mid-way, wait 3+ hours
- [ ] Verify stuck detection allows restart

**Status:** ⏳ Pending

---

### Phase 2: Task Tracking & Prevention (High Priority)

#### Task 2.1: Store Celery Task ID in Database
- **File:** `backend/src/services/extraction_service.py`
- **Location:** Lines 137-140
- **Priority:** HIGH
- **Estimated Time:** 20 minutes

**Implementation:**
```python
# Enqueue Celery task for async extraction
from src.workers.extraction_tasks import extract_features_task
task_result = extract_features_task.delay(training_id, config)  # CHANGE: capture result

# Store task ID in database
extraction_job.celery_task_id = task_result.id  # ADD
await self.db.commit()  # ADD
await self.db.refresh(extraction_job)  # ADD

logger.info(f"Queued extraction task {task_result.id} for job {extraction_job.id}")  # ADD

return extraction_job
```

**Testing:**
- [ ] Start new extraction
- [ ] Verify `celery_task_id` is populated in database
- [ ] Check logs for task ID
- [ ] Query Celery for task status using ID

**Status:** ⏳ Pending

---

#### Task 2.2: Enhanced Active Task Check with Celery Verification
- **File:** `backend/src/services/extraction_service.py`
- **Location:** Lines 100-116
- **Priority:** HIGH
- **Estimated Time:** 25 minutes

**Implementation:**
```python
# Check for active extraction on this training
result = await self.db.execute(
    select(ExtractionJob).where(
        ExtractionJob.training_id == training_id,
        ExtractionJob.status.in_([
            ExtractionStatus.QUEUED,
            ExtractionStatus.EXTRACTING
        ])
    )
)
active_extraction = result.scalar_one_or_none()

if active_extraction:
    # ADD: Verify Celery task is actually running
    from src.core.celery_app import get_task_status
    from datetime import datetime, timezone, timedelta

    if active_extraction.celery_task_id:
        task_status = get_task_status(active_extraction.celery_task_id)

        # Check if task is genuinely active
        if task_status['state'] in ['PENDING', 'STARTED', 'RETRY']:
            raise ValueError(
                f"Training {training_id} already has an active extraction job: "
                f"{active_extraction.id} (Celery task: {active_extraction.celery_task_id}, "
                f"state: {task_status['state']})"
            )
        elif task_status['state'] in ['SUCCESS', 'FAILURE', 'REVOKED']:
            # Task finished but DB not updated - check staleness
            time_since_update = datetime.now(timezone.utc) - active_extraction.updated_at
            if time_since_update < timedelta(minutes=5):
                # Recent activity, task may still be committing results
                raise ValueError(
                    f"Training {training_id} has a recently completed extraction task "
                    f"that may still be finalizing: {active_extraction.id}"
                )
            else:
                # Stale - allow new extraction but log warning
                logger.warning(
                    f"Found stale extraction {active_extraction.id} with finished "
                    f"Celery task (state: {task_status['state']}), allowing new extraction"
                )
        else:
            # Unknown state
            logger.warning(
                f"Extraction {active_extraction.id} has Celery task in unknown state: "
                f"{task_status['state']}, allowing new extraction"
            )
    else:
        # Legacy job without task_id - check staleness by timestamp
        time_since_update = datetime.now(timezone.utc) - active_extraction.updated_at
        if time_since_update < timedelta(hours=3):
            raise ValueError(
                f"Training {training_id} already has an active extraction job: "
                f"{active_extraction.id} (last updated {time_since_update} ago)"
            )
        else:
            logger.warning(
                f"Found stale extraction {active_extraction.id} (no task_id, "
                f"last update {time_since_update} ago), allowing new extraction"
            )
```

**Testing:**
- [ ] Start extraction with task_id tracking
- [ ] Try starting second extraction immediately
- [ ] Verify rejection with task state info
- [ ] Complete first extraction
- [ ] Start second extraction (should succeed)
- [ ] Test legacy extraction without task_id
- [ ] Test stale extraction detection (>3 hours old)

**Status:** ⏳ Pending

---

### Phase 3: Additional Safety Measures (Medium Priority)

#### Task 3.1: Add Extraction Job Unique Constraint
- **File:** Create new migration `backend/alembic/versions/xxx_add_extraction_unique_constraint.py`
- **Priority:** MEDIUM
- **Estimated Time:** 30 minutes

**Implementation:**
Create migration to add partial unique constraint:
```python
"""Add unique constraint to prevent duplicate active extractions

Revision ID: xxx
Create Date: 2025-10-28
"""
from alembic import op
import sqlalchemy as sa

def upgrade():
    # Add partial unique index: only one QUEUED or EXTRACTING job per training
    op.create_index(
        'idx_extraction_active_training_unique',
        'extraction_jobs',
        ['training_id'],
        unique=True,
        postgresql_where=sa.text("status IN ('queued', 'extracting')")
    )

def downgrade():
    op.drop_index('idx_extraction_active_training_unique', table_name='extraction_jobs')
```

**Testing:**
- [ ] Run migration on dev database
- [ ] Start extraction (status=EXTRACTING)
- [ ] Try inserting another extraction with status=EXTRACTING for same training
- [ ] Verify database rejects with unique constraint violation
- [ ] Complete first extraction (status=COMPLETED)
- [ ] Start new extraction (should succeed)

**Status:** ⏳ Pending

---

#### Task 3.2: Add Celery Task Deduplication
- **File:** `backend/src/workers/extraction_tasks.py`
- **Priority:** MEDIUM
- **Estimated Time:** 20 minutes

**Implementation:**
Add task deduplication using Redis:
```python
from src.core.celery_app import celery_app
from celery import Task
from redis import Redis
import hashlib

class DeduplicatedTask(DatabaseTask):
    """Base task with deduplication support."""

    def before_start(self, task_id, args, kwargs):
        """Check for duplicate task before starting."""
        # Create dedup key from task name + args
        dedup_key = f"task_dedup:{self.name}:{hashlib.md5(str(args).encode()).hexdigest()}"
        redis_client = Redis.from_url(celery_app.conf.broker_url)

        # Try to acquire lock
        if not redis_client.set(dedup_key, task_id, nx=True, ex=3600):  # 1 hour TTL
            existing_task_id = redis_client.get(dedup_key)
            self.update_state(
                state='DUPLICATE',
                meta={'message': f'Duplicate of task {existing_task_id}'}
            )
            raise ValueError(f"Duplicate task detected, existing task: {existing_task_id}")

    def after_return(self, status, retval, task_id, args, kwargs, einfo):
        """Release dedup lock after task completes."""
        dedup_key = f"task_dedup:{self.name}:{hashlib.md5(str(args).encode()).hexdigest()}"
        redis_client = Redis.from_url(celery_app.conf.broker_url)
        redis_client.delete(dedup_key)


@celery_app.task(
    bind=True,
    base=DeduplicatedTask,  # CHANGE: use deduplicated base
    name="extract_features",
    max_retries=0,
    autoretry_for=None,
)
def extract_features_task(...):
    ...
```

**Testing:**
- [ ] Start extraction task
- [ ] Immediately queue duplicate task with same args
- [ ] Verify second task is rejected
- [ ] Wait for first task to complete
- [ ] Queue new task with same args
- [ ] Verify new task succeeds (lock released)

**Status:** ⏳ Pending

---

#### Task 3.3: Add Monitoring/Alerting for Stuck Extractions
- **File:** Create new `backend/src/workers/monitoring_tasks.py`
- **Priority:** LOW
- **Estimated Time:** 40 minutes

**Implementation:**
```python
from celery import shared_task
from datetime import datetime, timezone, timedelta
from src.models.extraction_job import ExtractionJob, ExtractionStatus

@shared_task(name="monitor_stuck_extractions")
def monitor_stuck_extractions():
    """Check for extractions stuck in EXTRACTING state."""
    from src.core.database import get_db

    threshold = timedelta(hours=3)
    cutoff_time = datetime.now(timezone.utc) - threshold

    with next(get_db()) as db:
        stuck_extractions = db.query(ExtractionJob).filter(
            ExtractionJob.status == ExtractionStatus.EXTRACTING.value,
            ExtractionJob.updated_at < cutoff_time
        ).all()

        for extraction in stuck_extractions:
            logger.error(
                f"ALERT: Extraction {extraction.id} appears stuck! "
                f"Status: {extraction.status}, "
                f"Last update: {extraction.updated_at}, "
                f"Training: {extraction.training_id}"
            )

            # Optional: Auto-mark as failed
            # extraction.status = ExtractionStatus.FAILED.value
            # extraction.error_message = "Extraction timeout (no updates for 3+ hours)"
            # db.commit()

    return f"Checked for stuck extractions, found {len(stuck_extractions)}"
```

Add to `backend/src/core/celery_app.py` beat schedule:
```python
beat_schedule={
    # ... existing schedules ...
    "monitor-stuck-extractions": {
        "task": "monitor_stuck_extractions",
        "schedule": 1800.0,  # Every 30 minutes
        "options": {"queue": "low_priority"},
    },
}
```

**Testing:**
- [ ] Create test extraction with old `updated_at` timestamp
- [ ] Run monitoring task manually
- [ ] Verify alert logged
- [ ] Test auto-cleanup (if enabled)

**Status:** ⏳ Pending

---

### Phase 4: Testing & Validation

#### Task 4.1: Integration Tests
- **File:** Create `backend/tests/integration/test_extraction_idempotency.py`
- **Priority:** HIGH
- **Estimated Time:** 60 minutes

**Test Cases:**
```python
def test_extraction_completes_once():
    """Verify extraction completes and doesn't restart."""
    pass

def test_duplicate_extraction_rejected():
    """Verify duplicate extraction requests are rejected."""
    pass

def test_completed_extraction_not_restarted():
    """Verify completed extraction cannot be restarted."""
    pass

def test_task_id_tracked():
    """Verify Celery task ID is stored in database."""
    pass

def test_stale_extraction_cleanup():
    """Verify stuck extractions are detected and handled."""
    pass
```

**Status:** ⏳ Pending

---

#### Task 4.2: Manual Validation
- **Priority:** CRITICAL
- **Estimated Time:** 30 minutes

**Steps:**
- [ ] Deploy fixes to dev environment
- [ ] Start extraction job
- [ ] Monitor completion
- [ ] Verify status stays COMPLETED
- [ ] Check Celery logs for any restart attempts
- [ ] Try starting duplicate extraction
- [ ] Verify rejection
- [ ] Restart Celery worker
- [ ] Verify extraction still COMPLETED
- [ ] Check GPU memory released

**Status:** ⏳ Pending

---

## Success Criteria

- [ ] Extraction jobs complete exactly once
- [ ] Duplicate extraction requests are rejected
- [ ] Completed extractions cannot be restarted
- [ ] Celery task IDs are tracked in database
- [ ] Stuck extractions are detected and logged
- [ ] All tests pass
- [ ] Manual validation successful
- [ ] Documentation updated

---

## Rollback Plan

If issues arise after deployment:

1. **Immediate:** Revert changes to `extraction_tasks.py` (task retry settings)
2. **Quick:** Revert service layer idempotency checks
3. **Database:** Drop unique constraint index if causing issues
4. **Monitor:** Check Celery logs for task execution patterns

---

## Documentation Updates

- [ ] Update `backend/README.md` with extraction idempotency guarantees
- [ ] Add troubleshooting section for stuck extractions
- [ ] Document Celery task deduplication behavior
- [ ] Update API docs for extraction endpoints

---

## Timeline

- **Phase 1 (Critical):** 30 minutes - Deploy ASAP
- **Phase 2 (High):** 1 hour - Deploy within 24 hours
- **Phase 3 (Medium):** 1.5 hours - Deploy within 1 week
- **Phase 4 (Testing):** 1.5 hours - Complete before Phase 3

**Total Estimated Time:** 4.5 hours

---

## Notes

- Original extraction preserved with 8,155/8,192 features
- Statistics intact from successful completion
- Restoration script available at `backend/restore_extraction.py`
- Issue discovered during production use on 2025-10-28
