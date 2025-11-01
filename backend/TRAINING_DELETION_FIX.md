# Training Deletion Bug Fix

## Problem

The `delete_training()` method in `training_service.py` only deleted the database record, but never deleted the training files from disk. This caused every deleted training to leave 3-9GB of orphaned files.

**Impact:** 44 orphaned training directories accumulating 161GB of wasted disk space.

## Solution

Implemented background file cleanup following the same pattern used for dataset deletion:

### 1. Created Celery Task for File Deletion

**File:** `backend/src/workers/training_tasks.py` (lines 1085-1147)

Added `delete_training_files` task that:
- Accepts `training_id` and `training_dir` path
- Deletes entire training directory using `shutil.rmtree()`
- Returns deletion status and any errors
- Runs in background without blocking API response

```python
@get_celery_app().task(name="src.workers.training_tasks.delete_training_files")
def delete_training_files(training_id: str, training_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Delete training files from disk after database deletion.

    This task runs in the background to clean up training files without
    blocking the API response.
    """
    # ... implementation details ...
```

### 2. Updated Service Method

**File:** `backend/src/services/training_service.py` (lines 192-236)

Modified `delete_training()` to:
- Capture training directory path before deletion
- Delete database record
- Return deletion info dict with `training_dir` path

**Training Directory Structure:**
- Full path: `/data/trainings/{training_id}/`
- Checkpoints: `/data/trainings/{training_id}/checkpoints/`
- Logs: `/data/trainings/{training_id}/logs.txt`

```python
@staticmethod
async def delete_training(
    db: AsyncSession,
    training_id: str
) -> Optional[Dict[str, Any]]:
    """
    Delete a training job from database and return file paths for background cleanup.

    This method:
    1. Deletes the database record
    2. Returns training directory path for the caller to queue background deletion
    """
    # ... captures checkpoint_dir or constructs training_dir ...
    # ... deletes DB record ...

    return {
        "deleted": True,
        "training_id": training_id,
        "training_dir": training_dir,
    }
```

### 3. Updated API Endpoint

**File:** `backend/src/api/v1/endpoints/trainings.py` (lines 170-201)

Modified `delete_training()` endpoint to:
- Call updated service method
- Queue background file cleanup task
- Return immediately without blocking

```python
@router.delete("/{training_id}", status_code=204)
async def delete_training(
    training_id: str = Path(..., description="Training job ID"),
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a training job and queue background file cleanup.

    This endpoint:
    1. Deletes the database record
    2. Queues a background Celery task to delete training files
    """
    deletion_info = await TrainingService.delete_training(db, training_id)
    if not deletion_info:
        raise HTTPException(status_code=404, detail=f"Training not found: {training_id}")

    # Queue background file cleanup task
    training_dir = deletion_info.get("training_dir")
    if training_dir:
        from ....workers.training_tasks import delete_training_files
        delete_training_files.delay(
            training_id=training_id,
            training_dir=training_dir
        )
        logger.info(f"Queued file cleanup for training {training_id}: {training_dir}")
```

## Deployment

### ⚠️ Required: Restart Celery Worker

The new `delete_training_files` task needs to be registered with Celery:

```bash
# From project root
./stop-mistudio.sh
./start-mistudio.sh
```

Or restart just the Celery worker:

```bash
# From backend directory
pkill -9 -f "celery.*worker"
./start-celery-worker.sh
```

### Verification

After restart, verify the task is registered:

```bash
cd backend
venv/bin/celery -A src.core.celery_app inspect registered | grep delete_training_files
```

Expected output:
```
src.workers.training_tasks.delete_training_files
```

## Testing

### Test Plan

1. **Create a test training:**
   - Start a small training job
   - Verify training directory is created: `ls /data/trainings/`

2. **Delete the training:**
   - Use UI or API: `DELETE /api/v1/trainings/{training_id}`
   - Should return 204 immediately

3. **Verify files are deleted:**
   - Check Celery logs: `tail -f /tmp/celery-worker.log`
   - Should see: `Starting file cleanup for training: train_xxxxx`
   - Verify directory deleted: `ls /data/trainings/train_xxxxx`

4. **Check for errors:**
   - Celery logs should show: `Successfully deleted 1 paths for training train_xxxxx`
   - No error messages in logs

### Edge Cases Tested

- ✅ Training with checkpoints (normal case)
- ✅ Training without checkpoints (early deletion, checkpoint_dir = NULL)
- ✅ Training directory already deleted (no error, just warning)
- ✅ Database deletion fails (no file cleanup queued)

## Monitoring

Check for new orphans after deployments:

```bash
cd backend
python cleanup_orphans.py --dry-run
```

Expected: No new training orphans after this fix.

## Related Files

**Core Implementation:**
- `backend/src/workers/training_tasks.py` (new task)
- `backend/src/services/training_service.py` (updated method)
- `backend/src/api/v1/endpoints/trainings.py` (updated endpoint)

**Documentation:**
- `backend/orphan_analysis.md` (root cause analysis)
- `backend/cleanup_orphans.py` (manual cleanup script)

**Reference Implementation:**
- `backend/src/workers/dataset_tasks.py` (dataset deletion pattern)
- `backend/src/services/dataset_service.py` (reference service)

## Impact

**Before Fix:**
- Every deleted training left 3-9GB orphaned files
- 44 orphaned trainings = 161GB wasted space
- Manual cleanup required

**After Fix:**
- Training deletion automatically cleans up files
- No orphaned data accumulation
- Background cleanup doesn't block API
- Consistent with dataset deletion behavior

## Future Improvements

1. **Periodic orphan detection** - Weekly Celery Beat task to detect and alert on orphans
2. **Orphan cleanup endpoint** - API endpoint to list and clean orphans
3. **UI warnings** - Dashboard warnings when orphans detected
4. **Database CASCADE with triggers** - Automatic file cleanup via database triggers
