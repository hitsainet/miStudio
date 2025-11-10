# Disk Space Orphan Analysis

## Current Disk Usage
- **Total Disk:** 466GB
- **Used:** 420GB (95%)
- **Available:** 27GB (5%)

## Data Directory Breakdown
- **Total data/:** 371GB
  - `trainings/`: 174GB
  - `datasets/`: 124GB
  - `activations/`: 40GB
  - `models/`: 35GB

## Orphaned Data Summary

| Resource | On Disk | In DB | Orphaned | Estimated Size | Recovery Potential |
|----------|---------|-------|----------|----------------|-------------------|
| **Trainings** | 45 | 1 | **44** | **~174GB** | **High - Safe to delete** |
| **Datasets** | 43 | 4 | **39** | **~124GB** | **High - Safe to delete** |
| **Activations** | 1 | 1 | 0 | 40GB | N/A |
| **Models** | 2 | 2 | 0 | 35GB | N/A |

**Total Orphaned: ~298GB (80% of data directory)**

## Root Causes

### Bug #1: Training Deletion Doesn't Clean Up Files
**File:** `backend/src/services/training_service.py`

**Current Code:**
```python
async def delete_training(db: AsyncSession, training_id: str) -> bool:
    """Delete a training job."""
    db_training = await TrainingService.get_training(db, training_id)
    if not db_training:
        return False

    await db.delete(db_training)  # Only deletes DB record!
    await db.commit()
    return True
```

**Problem:** Only deletes the database record. Never deletes:
- Checkpoint directory (`data/trainings/{training_id}/`)
- Checkpoint files (`.safetensors`)
- Training logs
- Metrics files

**Impact:** Every deleted training leaves orphaned files. With 44 deleted trainings, this accounts for ~174GB.

**Fix Needed:**
1. Get checkpoint_dir path before deletion
2. Delete database record
3. Queue Celery background task to delete files (like datasets do)

### Bug #2: Dataset Orphans from Failed Cleanup Tasks
**File:** `backend/src/api/v1/endpoints/datasets.py`

**Current Code (Correct Pattern):**
```python
@router.delete("/{dataset_id}", status_code=204)
async def delete_dataset(dataset_id: UUID, db: AsyncSession = Depends(get_db)):
    deletion_info = await DatasetService.delete_dataset(db, dataset_id)

    # Queue background file cleanup
    if raw_path or tokenized_path:
        delete_dataset_files.delay(
            dataset_id=str(dataset_id),
            raw_path=raw_path,
            tokenized_path=tokenized_path
        )
```

**Problem:** Code is correct NOW, but 39 orphaned datasets exist from:
1. Past deletions before this code was implemented
2. Failed Celery tasks that weren't retried
3. Manual database deletions (e.g., during debugging/testing)

**Impact:** 39 orphaned dataset directories account for ~124GB.

## Orphaned Training IDs (Sample - First 10 of 44)
```
train_00d10dde
train_132ef5bd
train_16091885
train_1b4d5fff
train_21de9472
train_238da321
train_33e3ed8d
train_361c7233
train_38265547
train_38f89250
```

## Recommendations

### Immediate Action: Clean Up Orphans
1. **Run cleanup script** to delete orphaned directories
2. **Reclaim ~298GB** of disk space
3. **Monitor** for new orphans after deletions

### Short-term Fix: Fix Training Deletion ✅ COMPLETED
1. **✅ Updated `training_service.py`** to return training directory path for cleanup
2. **✅ Created `delete_training_files` Celery task** in training_tasks.py (mirrors dataset pattern)
3. **✅ Updated API endpoint** in trainings.py to queue background file cleanup
4. **⚠️ Requires Celery worker restart** to load new task: `./stop-mistudio.sh && ./start-mistudio.sh`
5. **Test** with a dummy training (pending)

### Long-term Prevention: Orphan Detection
1. **Add periodic cleanup job** (weekly Celery Beat task)
2. **Add orphan detection endpoint** for monitoring
3. **Add orphan warnings** to UI dashboard
4. **Consider database CASCADE deletes** with triggers for file cleanup

## Cleanup Script Usage

```bash
# Dry run (see what will be deleted)
python cleanup_orphans.py --dry-run

# Delete orphaned trainings only
python cleanup_orphans.py --trainings-only

# Delete orphaned datasets only
python cleanup_orphans.py --datasets-only

# Delete ALL orphans (trainings + datasets)
python cleanup_orphans.py --delete-all

# Interactive mode (ask before each deletion)
python cleanup_orphans.py --interactive
```

## Safety Checks
- Script compares database vs. filesystem
- Excludes any ID found in database (fail-safe)
- Creates backup log of deleted files
- Provides rollback instructions (if files backed up)

## Expected Recovery
After cleanup: **Free up ~298GB → Disk usage: 95% → 31%**
