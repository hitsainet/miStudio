"""Celery tasks for probe monitors (032 FR-5–FR-12).

Four tasks: `run_probe_monitor` (GPU), `evaluate_probe_monitor` (GPU),
`score_probe_monitor` (GPU, short) and `run_probe_monitor_judge` (CPU, HTTP).

⚠ EVERY TASK NAME IS FULLY QUALIFIED, AND THAT IS NOT STYLE. `task_routes` globs
match the TASK NAME, not the module path — so a short name never matches
`src.workers.probe_monitor_tasks.*` and lands silently on the DEFAULT queue. This
estate has already put two GPU trainings on the wrong queue that way, and found it
only with a test that resolves every registered task's route.

⚠ THE ROW IS COMMITTED BY THE API BEFORE DISPATCH, AND THIS TASK REFUSES WITHOUT ONE.
That order is what makes `missing_row="cancelled"` correct on the cancel scope: a
missing row can only mean somebody deleted the run. When a fix once passed an id
without creating the row first, every UI extraction was silently refused for a week —
so the precondition lives beside the guard rather than in a comment elsewhere.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from pathlib import Path

from ..core.celery_app import celery_app
from ..core.clock import utc_now
from .base_task import DatabaseTask
from .gpu_job import gpu_job
from .websocket_emitter import (
    emit_probe_monitor_completed,
    emit_probe_monitor_failed,
    emit_probe_monitor_progress,
)

logger = logging.getLogger(__name__)


def _failure_detail(exc: BaseException) -> str:
    """`TypeName: message (file:line)` — actionable without the pod log.

    `str(IndexError(...))` is exactly "tuple index out of range": no type, no file,
    no line, and that string is all the UI would ever show.
    """
    import traceback

    where = ""
    tb = exc.__traceback__
    if tb is not None:
        frame = traceback.extract_tb(tb)[-1]
        where = f" ({frame.filename.rsplit('/', 1)[-1]}:{frame.lineno})"
    return f"{type(exc).__name__}: {exc}{where}"


def _require_row(db, model_cls, row_id: str, what: str):
    """Load a row or refuse. See the module docstring on why absence is an error."""
    row = db.query(model_cls).filter(model_cls.id == row_id).first()
    if row is None:
        raise ValueError(
            f"{what} {row_id} has no row. The API commits it before dispatch, so this "
            f"means it was deleted — the task stops rather than doing work nothing can read"
        )
    return row


@celery_app.task(
    bind=True,
    base=DatabaseTask,
    name="src.workers.probe_monitor_tasks.run_probe_monitor",
    max_retries=0,
)
@gpu_job("probe_monitor_run")
def run_probe_monitor(self, run_id: str) -> Dict[str, Any]:
    """Render → pooled capture → layer selection → token capture → train → calibrate.

    Stages are written to the row as they begin, so a run that dies is diagnosable
    from the row alone: "failed at token_capture" and "failed at training" have
    different causes and different fixes.
    """
    from ..core.cancellation import OperatorCancelled, record_progress
    from ..models.probe_monitor import ProbeMonitorRun
    from ..services.probe_monitor_run import execute_probe_run

    with self.get_db() as db:
        try:
            row = _require_row(db, ProbeMonitorRun, run_id, "probe monitor run")
            row.status = "running"
            db.commit()

            def on_stage(stage: str, progress: float, message: Optional[str] = None) -> None:
                # `record_progress` refuses to move a TERMINAL row, which is what stops
                # an in-flight write from overwriting the endpoint's CANCELLED seconds
                # after the operator asked to stop.
                record_progress(
                    "probe_monitor_run", run_id, progress=progress, stage=stage
                )
                emit_probe_monitor_progress(run_id, progress, stage=stage, message=message)

            result = execute_probe_run(db, run_id, on_stage=on_stage)
            emit_probe_monitor_completed(run_id, summary=result)
            return result
        except OperatorCancelled as cancelled:
            # `OperatorCancelled` derives from BaseException so an `except Exception`
            # cannot swallow it. Partial artifacts are KEPT and the row marked, because
            # a cancelled capture's memmap is often worth resuming from.
            logger.info("Probe monitor run %s cancelled by the operator", run_id)
            emit_probe_monitor_progress(run_id, 0.0, stage="cancelled", status="cancelled")
            return {
                "status": "cancelled",
                "scope": cancelled.scope,
                "target_id": cancelled.target_id,
            }
        except Exception as exc:
            logger.exception("Probe monitor run %s failed", run_id)
            detail = _failure_detail(exc)
            row = db.query(ProbeMonitorRun).filter(ProbeMonitorRun.id == run_id).first()
            if row is not None:
                row.status = "failed"
                row.error_message = detail[:2000]
                db.commit()
            emit_probe_monitor_failed(run_id, detail[:500])
            raise


@celery_app.task(
    bind=True,
    base=DatabaseTask,
    name="src.workers.probe_monitor_tasks.evaluate_probe_monitor",
    max_retries=0,
)
@gpu_job("probe_monitor_run")
def evaluate_probe_monitor(self, probe_id: str, dataset_ids: list) -> Dict[str, Any]:
    """Evaluate an existing probe on ADDITIONAL sets, then recompute its rung.

    Separate from the run because rung 2 and 3 are reached by adding evaluation sets
    to a probe that already exists — a probe should not have to be retrained to be
    measured on new data.
    """
    from ..models.probe_monitor import ProbeMonitor
    from ..services.probe_monitor_run import evaluate_probe

    with self.get_db() as db:
        try:
            _require_row(db, ProbeMonitor, probe_id, "probe monitor")
            result = evaluate_probe(db, probe_id, list(dataset_ids))
            return result
        except Exception as exc:
            logger.exception("Probe monitor evaluation %s failed", probe_id)
            raise


@celery_app.task(
    bind=True,
    base=DatabaseTask,
    name="src.workers.probe_monitor_tasks.score_probe_monitor",
    max_retries=0,
)
@gpu_job("probe_monitor_run")
def score_probe_monitor(
    self,
    probe_id: str,
    text: Optional[str] = None,
    messages: Optional[list] = None,
) -> Dict[str, Any]:
    """Score one input offline and return the per-token trace (FR-12).

    A GPU task rather than a synchronous route: loading a model in the API process
    would block the event loop for minutes and hold VRAM outside the lease system.
    """
    from ..models.probe_monitor import ProbeMonitor
    from ..services.probe_monitor_run import score_one

    with self.get_db() as db:
        _require_row(db, ProbeMonitor, probe_id, "probe monitor")
        return score_one(db, probe_id, text=text, messages=messages)


@celery_app.task(
    bind=True,
    base=DatabaseTask,
    name="src.workers.probe_monitor_tasks.run_probe_monitor_judge",
    max_retries=0,
)
def run_probe_monitor_judge(self, judge_run_id: str) -> Dict[str, Any]:
    """The LLM-monitor baseline over the same sets (FR-11). CPU: it is HTTP calls.

    NOT a GPU job — the model is served by miLLM, so taking a lease here would idle a
    card for the duration of a network-bound loop.
    """
    from ..models.probe_monitor import ProbeMonitorJudgeRun
    from ..services.probe_monitor_judge import execute_judge_run

    with self.get_db() as db:
        try:
            _require_row(db, ProbeMonitorJudgeRun, judge_run_id, "judge run")
            return execute_judge_run(db, judge_run_id)
        except Exception as exc:
            logger.exception("Probe monitor judge run %s failed", judge_run_id)
            detail = _failure_detail(exc)
            row = (
                db.query(ProbeMonitorJudgeRun)
                .filter(ProbeMonitorJudgeRun.id == judge_run_id)
                .first()
            )
            if row is not None:
                row.status = "failed"
                row.error_message = detail[:2000]
                db.commit()
            raise


@celery_app.task(
    bind=True,
    base=DatabaseTask,
    name="src.workers.probe_monitor_tasks.build_probe_definition",
    max_retries=0,
)
@gpu_job("probe_monitor_definition")
def build_probe_definition(
    self,
    probe_id: str,
    acknowledge_below_rung2: Optional[Dict[str, Any]] = None,
    actor: str = "operator",
    vector_count: int = 16,
    seed: int = 1337,
) -> Dict[str, Any]:
    """Build and cache a probe's `mistudio.probe-definition/v1` (033 FR-7, task 2.6).

    ⚠ A GPU JOB, BECAUSE THE TEST VECTORS ARE REAL FORWARD PASSES. The vectors go through
    `forward_scores` — the same code that produced this probe's metrics — so building one loads the
    model and runs it. Doing that on the CPU would produce vectors a consumer could not reproduce:
    fp16 against fp32 already moves a score, and a quantized model moves it further.
    """
    import json

    from ..core.cancellation import record_progress
    from ..core.config import settings
    from ..models.probe_monitor import ProbeMonitor
    from ..services.probe_definition_builder import (
        ProbeExportRefused,
        build,
        write_definition,
    )

    with self.get_db() as db:
        probe = _require_row(db, ProbeMonitor, probe_id, "probe monitor")
        try:
            definition, record = build(
                db,
                probe_id,
                acknowledge_below_rung2=acknowledge_below_rung2,
                actor=actor,
                vector_count=vector_count,
                seed=seed,
            )
        except ProbeExportRefused as refused:
            # A refusal is a RESULT, not a crash: the caller asked for something the evidence or
            # the configuration does not permit, and the reason is the useful part.
            logger.info("Probe definition for %s refused: %s", probe_id, refused)
            return {"status": "refused", "reason": str(refused), "http_status": refused.status}

        path = Path(settings.data_dir) / "probe_definitions" / f"{probe_id}.probe.json"
        # ⚠ THE WRITE, THE SIZE AND THE DIGEST COME FROM ONE CALL. They used to be three
        # serialisations — the file indented, the sha and the byte count compact — so the row's
        # `definition_sha256` did not match the file it named, and the cap was checked against a
        # form 1.6x smaller than the one on disk. `write_definition` returns the measurements of
        # the bytes it wrote, and is not given the chance to serialise twice.
        written_bytes, written_sha = write_definition(definition, path)

        probe.definition_path = str(path)
        probe.definition_built_at = utc_now()
        probe.definition_sha256 = written_sha
        record["resolved"]["bytes"] = written_bytes
        record["resolved"]["sha256"] = written_sha
        probe.definition_build = record
        db.commit()
        record_progress("probe_monitor_run", probe.run_id, progress=None, db=db)
        return {
            "status": "completed",
            "probe_id": probe_id,
            "path": str(path),
            "sha256": written_sha,
            "bytes": written_bytes,
            "vectors": record["resolved"]["vectors"],
        }


@celery_app.task(
    bind=True,
    base=DatabaseTask,
    name="src.workers.probe_monitor_tasks.publish_probe_definition",
    max_retries=0,
)
def publish_probe_definition(
    self,
    probe_id: str,
    repo_id: str,
    private: bool = True,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """Upload a built definition to HuggingFace (033 FR-8, task 4.2).

    ⚠ NOT A GPU JOB, AND ON THE `processing` QUEUE. It is an HTTP upload: on the GPU queue it
    would take a card's lease and hold it idle for the whole network-bound transfer, which is the
    same reasoning that put the judge run there.

    ⚠ THE TOKEN IS NEVER PERSISTED. It arrives as an argument, is used, and is not written to the
    row, the task result or a log line. A token in `task_queue.args` would be readable by anyone
    who can list tasks.
    """
    from ..models.probe_monitor import ProbeMonitor
    from ..services.probe_definition_publisher import publish

    with self.get_db() as db:
        _require_row(db, ProbeMonitor, probe_id, "probe monitor")
        result = publish(db, probe_id, repo_id=repo_id, private=private, token=token)
        return result
