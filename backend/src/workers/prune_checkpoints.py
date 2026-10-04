"""
Periodic task to prune redundant training checkpoints.

Training writes a checkpoint every N steps and nothing ever removes them, so
``/data/trainings`` grows without bound (18 trainings / ~24 steps each / 78 GB
when this was written).

SAFETY POSTURE — this deletes data, so:
  * disabled by default (``checkpoint_prune_enabled``)
  * dry-run by default when enabled (``checkpoint_prune_dry_run``)
  * never touches a training that is pending/initializing/running/paused
  * never deletes the newest step, the ``keep_last`` newest, or an is_best step
  * never deletes checkpoints younger than ``checkpoint_prune_min_age_hours``

The dry-run path logs exactly what it WOULD remove, so an operator can review
real numbers before enabling deletion.
"""

import logging
from collections import defaultdict
from typing import Dict

from src.core.celery_app import celery_app
from src.models.checkpoint import Checkpoint
from src.services.checkpoint_retention import (
    build_plan,
    iter_prunable_trainings,
    load_policy,
)
from src.services.checkpoint_service import CheckpointService
from src.workers.base_task import DatabaseTask

logger = logging.getLogger(__name__)


def _execute_plan(db, plan, policy) -> tuple[int, int, int]:
    """Delete the checkpoints in ``plan``.

    Returns:
        (rows_deleted, bytes_freed, files_failed)

    ORDERING: the file is unlinked FIRST and the row is only deleted+committed
    once that succeeds. Committing the row over a failed unlink strands the file
    permanently, because planning is row-driven: with no row, no future prune can
    ever plan it again, and the run reports "0.00 GB freed" as though nothing
    needed doing. The residual risk (a crash between unlink and commit) leaves a
    row whose file is gone, which the next pass can detect and clean up.
    """
    deleted = 0
    freed = 0
    failed = 0

    # Selection is per-STEP, so execution must be too. Re-checking is_best per
    # ROW and skipping just that row would delete a step's other layers and leave
    # an unloadable partial checkpoint — the very outcome step-granularity exists
    # to prevent. Drop the whole step if any of its rows became best.
    rows = [
        r for r in (
            db.query(Checkpoint).filter_by(id=cid).first()
            for cid in plan.checkpoint_ids
        )
        if r is not None
    ]
    if policy.keep_best:
        promoted_steps = {r.step for r in rows if r.is_best}
        if promoted_steps:
            logger.warning(
                "Skipping step(s) %s entirely: a checkpoint became is_best after "
                "planning; deleting the step's other layers would leave it unloadable",
                sorted(promoted_steps),
            )
            rows = [r for r in rows if r.step not in promoted_steps]

    # Execute per STEP, not per row. Committing row-by-row meant a mid-step
    # failure (EACCES/ENOSPC on layer 2 of 3) left layer 1's row committed as
    # deleted while its siblings survived — and expected_sae_keys derives the
    # expected layer set FROM THE SURVIVING ROWS, so a later finalize would then
    # declare that torn step complete and export 2 of 3 SAEs as a whole run.
    by_step: Dict[int, list] = defaultdict(list)
    for checkpoint in rows:
        by_step[checkpoint.step].append(checkpoint)

    for step, step_rows in sorted(by_step.items()):
        step_freed = 0
        step_failed = False
        for checkpoint in step_rows:
            try:
                step_freed += CheckpointService.delete_checkpoint_files(
                    checkpoint.storage_path
                )
            except OSError as e:
                step_failed = True
                logger.error(
                    "Step %s: keeping ALL its rows — %s could not be deleted (%s)",
                    step, checkpoint.id, e,
                )
                break

        if step_failed:
            # Some of the step's files may already be gone, but keeping every row
            # means the next pass retries the whole step and the metadata still
            # describes the full layer set.
            failed += 1
            continue

        for checkpoint in step_rows:
            db.delete(checkpoint)
        db.commit()
        deleted += len(step_rows)
        freed += step_freed

    return deleted, freed, failed


@celery_app.task(
    bind=True,
    base=DatabaseTask,
    # Fully-qualified so the "src.workers.prune_checkpoints.*" route glob matches.
    name="src.workers.prune_checkpoints.prune_checkpoints",
)
def prune_checkpoints_task(self, force: bool = False):
    """Apply the checkpoint retention policy across all eligible trainings.

    ``force`` is for an explicit operator request — "Run cleanup now" — which
    should work whether or not the DAILY sweep is enabled, exactly as the
    per-training route already does. It bypasses ONLY the enabled flag. Every
    safety guard still applies, and ``checkpoint_prune_dry_run`` is still
    honoured, so a report-only installation still reports.
    """
    with self.get_db() as db:
        try:
            policy = load_policy(db)

            if not policy.enabled and not force:
                logger.debug(
                    "Checkpoint pruning disabled (checkpoint_prune_enabled=false)"
                )
                return {"enabled": False, "trainings_scanned": 0}

            trainings = list(iter_prunable_trainings(db))
            total_deleted = 0
            total_freed = 0
            total_candidates = 0
            total_failed_files = 0
            failed_trainings: list[str] = []

            for training in trainings:
                plan = build_plan(db, training, policy)
                if not plan.is_actionable:
                    continue

                total_candidates += len(plan.checkpoint_ids)

                if policy.dry_run:
                    logger.info(
                        "[dry-run] would prune %d checkpoint(s) (~%.2f GB) from "
                        "training %s: steps %s (keeping %s)",
                        len(plan.checkpoint_ids),
                        plan.estimated_bytes / 1e9,
                        training.id,
                        plan.prunable_steps,
                        plan.kept_steps,
                    )
                    continue

                try:
                    deleted, freed, failed = _execute_plan(db, plan, policy)
                except Exception as e:  # noqa: BLE001 - one bad training must not
                    # abort the sweep and discard the record of what was already
                    # deleted. Earlier trainings are already committed.
                    db.rollback()
                    failed_trainings.append(training.id)
                    logger.error(
                        "Prune failed for training %s: %s", training.id, e, exc_info=True
                    )
                    continue

                total_deleted += deleted
                total_freed += freed
                total_failed_files += failed
                logger.info(
                    "Pruned %d checkpoint(s) (%.2f GB) from training %s: steps %s"
                    "%s",
                    deleted, freed / 1e9, training.id, plan.prunable_steps,
                    f" ({failed} file(s) could not be deleted)" if failed else "",
                )

            result = {
                "enabled": True,
                "dry_run": policy.dry_run,
                "trainings_scanned": len(trainings),
                "candidates": total_candidates,
                "deleted": total_deleted,
                "bytes_freed": total_freed,
                # Surfaced so a silent "0.00 GB freed" cannot hide stranded files.
                "files_failed": total_failed_files,
                "failed_trainings": failed_trainings,
            }
            if total_candidates:
                logger.info("Checkpoint prune summary: %s", result)
            return result

        except Exception as e:
            db.rollback()
            logger.error(f"Error in checkpoint prune: {e}", exc_info=True)
            raise


@celery_app.task(
    bind=True,
    base=DatabaseTask,
    name="src.workers.prune_checkpoints.prune_single_training_checkpoints",
)
def prune_single_training_task(self, training_id: str):
    """Prune ONE training's checkpoints on explicit operator request.

    Unlike the scheduled sweep this ignores ``checkpoint_prune_enabled`` — the
    operator asked for this specific training here and now. Every safety guard
    still applies, and ``checkpoint_prune_dry_run`` is still honoured so a
    dry-run installation reports instead of deleting.
    """
    from src.models.training import Training

    with self.get_db() as db:
        try:
            training = db.query(Training).filter_by(id=training_id).first()
            if training is None:
                logger.warning("prune: training not found: %s", training_id)
                return {"training_id": training_id, "error": "not_found"}

            policy = load_policy(db)
            plan = build_plan(db, training, policy)

            if plan.skipped_reason:
                logger.info(
                    "prune: skipping training %s (%s)", training_id, plan.skipped_reason
                )
                return {
                    "training_id": training_id,
                    "skipped_reason": plan.skipped_reason,
                    "deleted": 0,
                }

            if policy.dry_run:
                logger.info(
                    "[dry-run] would prune %d checkpoint(s) (~%.2f GB) from training "
                    "%s: steps %s (keeping %s)",
                    len(plan.checkpoint_ids), plan.estimated_bytes / 1e9,
                    training_id, plan.prunable_steps, plan.kept_steps,
                )
                return {
                    "training_id": training_id,
                    "dry_run": True,
                    "candidates": len(plan.checkpoint_ids),
                    "estimated_bytes": plan.estimated_bytes,
                    "deleted": 0,
                }

            deleted, freed, failed = _execute_plan(db, plan, policy)
            logger.info(
                "Pruned %d checkpoint(s) (%.2f GB) from training %s: steps %s%s",
                deleted, freed / 1e9, training_id, plan.prunable_steps,
                f" ({failed} file(s) could not be deleted)" if failed else "",
            )
            return {
                "training_id": training_id,
                "dry_run": False,
                "deleted": deleted,
                "bytes_freed": freed,
                "files_failed": failed,
            }

        except Exception as e:
            db.rollback()
            logger.error(
                f"Error pruning checkpoints for {training_id}: {e}", exc_info=True
            )
            raise
