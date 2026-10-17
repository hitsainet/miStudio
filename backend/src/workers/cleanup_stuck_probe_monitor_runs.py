"""Reclaim probe monitor runs whose worker died (032).

Every long-running status on this estate needs a janitor, and a probe run has TWO
statuses that can hang — the run itself and a judge run. Without one, a run whose
worker vanished holds `running` forever: the panel shows a progress bar frozen at
whatever percentage was written last, with no error, no retry affordance, and nothing
that will ever change it. Worse for this feature than for most: a run holds a GPU lease
and a multi-gigabyte token memmap, so a phantom run keeps a card AND the disk.

A worker can vanish in ways the task cannot catch — a rolling deploy SIGTERMing a
single-slot worker mid-stage, an OOM kill, a pod eviction, a node reboot. None give the
task a chance to write a terminal status, so the sweep has to come from outside.

⚠ THREE CONDITIONS, ALL REQUIRED, AND THE THIRD IS THE ONE THAT MATTERS. Stale by the
clock, no live Celery task, AND the progress counter has not moved. `progress_stalled_
seconds` returns None when it cannot answer, and None means "do not reap on this basis"
— never zero. This estate reaped a LIVE 5.8-hour packing job as "worker lost" because
the phase reported only over the WebSocket, and the run's own stage heartbeat is what
keeps that from happening here.
"""

import logging
from datetime import datetime, timedelta, timezone

from src.core.celery_app import celery_app
from src.models.probe_monitor import ProbeMonitorJudgeRun, ProbeMonitorRun
from src.workers.base_task import DatabaseTask
from src.workers.job_progress import progress_stalled_seconds
from src.workers.websocket_emitter import emit_probe_monitor_failed

logger = logging.getLogger(__name__)

#: A probe run writes a stage heartbeat at every boundary, and the longest single stage
#: (token capture over a large set) is measured in tens of minutes rather than hours. 90
#: minutes of total silence therefore means the process is gone rather than slow —
#: generous on purpose, because reaping a live GPU job is worse than a late reap.
STUCK_THRESHOLD_MINUTES = 90

#: The same logic applies to a judge run, but it is network-bound and much faster; an
#: hour of silence on an HTTP loop is already far beyond a slow endpoint.
JUDGE_STUCK_THRESHOLD_MINUTES = 60

LIVE_STATUSES = ("pending", "running")


def _looks_alive(row, kind: str) -> bool:
    """Is a worker still on this row? Conservative — any evidence of life wins."""
    if row.celery_task_id:
        from src.workers.task_heartbeat import task_looks_alive

        # A dead worker and a queued task are BOTH Celery-PENDING, so a bare state
        # check can never be false here (MIS-E2E-092); `task_looks_alive` is what
        # distinguishes them.
        if task_looks_alive(
            row.celery_task_id, row, started=str(row.status or "").lower() == "running"
        ):
            return True
    stalled = progress_stalled_seconds(kind, row.id, row.progress)
    if stalled is None:
        # NO EVIDENCE IS NOT EVIDENCE OF DEATH. Redis unavailable, or the first
        # sighting of this row — either way the clock alone must not reap a job
        # holding a GPU.
        return True
    return False


@celery_app.task(
    bind=True,
    base=DatabaseTask,
    name="cleanup_stuck_probe_monitor_runs",
)
def cleanup_stuck_probe_monitor_runs_task(self):
    """Fail probe runs and judge runs whose worker is gone."""
    logger.info("Running stuck probe monitor cleanup task")
    reaped = {"runs": 0, "judge_runs": 0}

    with self.get_db() as db:
        now = datetime.now(timezone.utc)
        run_threshold = now - timedelta(minutes=STUCK_THRESHOLD_MINUTES)
        judge_threshold = now - timedelta(minutes=JUDGE_STUCK_THRESHOLD_MINUTES)

        for row in (
            db.query(ProbeMonitorRun)
            .filter(
                ProbeMonitorRun.status.in_(LIVE_STATUSES),
                ProbeMonitorRun.updated_at < run_threshold,
            )
            .all()
        ):
            if _looks_alive(row, "probe_monitor_run"):
                logger.info("Probe run %s still looks alive; leaving it", row.id)
                continue
            minutes = (now - row.updated_at.replace(tzinfo=timezone.utc)).total_seconds() / 60
            row.status = "failed"
            row.error_message = (
                f"The worker running this probe stopped reporting {minutes:.0f} minutes "
                f"ago at stage '{row.stage or 'unknown'}' and no live task was found. "
                f"Its artifacts are kept: re-submit the run, or inspect "
                f"{row.artifact_dir or 'its artifact directory'}."
            )
            db.commit()
            emit_probe_monitor_failed(row.id, "worker lost")
            reaped["runs"] += 1

        for judge in (
            db.query(ProbeMonitorJudgeRun)
            .filter(
                ProbeMonitorJudgeRun.status.in_(LIVE_STATUSES),
                ProbeMonitorJudgeRun.created_at < judge_threshold,
            )
            .all()
        ):
            if _looks_alive(judge, "probe_monitor_judge"):
                continue
            judge.status = "failed"
            judge.error_message = (
                "The worker running this judge baseline stopped reporting and no live "
                "task was found. No partial baseline is recorded: a judge run over some "
                "of the rows is a different sample, not a smaller one."
            )
            db.commit()
            reaped["judge_runs"] += 1

    if reaped["runs"] or reaped["judge_runs"]:
        logger.warning("Reaped stuck probe monitor work: %s", reaped)
    return reaped
