"""REST surface for probe monitors (032 FR-1, FR-5, FR-9, FR-11–FR-14).

⚠ VALIDATION THAT NEEDS DATA LIVES IN THE SERVICE, NOT HERE. The schemas reject what a
request alone can settle (an unknown rule, a stride below 1); the 20-per-class floor, a
layer beyond the model's depth and a missing SAE all need the database or the model, and
they return 422 naming the offending value. A route that guessed at those would produce a
202 for a run that can only ever fail when the worker places it.

⚠ `GET /probes/{id}` ASSEMBLES THE REPORT SERVER-SIDE, including the rung's wording. A
detector's language is the thing most likely to drift above its evidence, and miLLM
mirrors these strings verbatim — so the client is never handed a rung number and left to
phrase it.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ....core.config import settings
from ....core.deps import get_db
from ....models.dataset import Dataset
from ....models.probe_monitor import (
    ProbeMonitor,
    ProbeMonitorDataset,
    ProbeMonitorEvaluation,
    ProbeMonitorJudgeRun,
    ProbeMonitorRun,
)
from ....schemas.probe_monitor import (
    JudgeRunCreate,
    ProbeDatasetCreate,
    ProbeDatasetResponse,
    ProbeEvaluationResponse,
    ProbeMonitorSummary,
    ProbeReport,
    ProbeRunCreate,
    ProbeRunResponse,
    ScoreRequest,
)
from ....services.gpu_dispatch import gpu_delay
from ..gpu_request import resolve_gpu_request

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/probe-monitors", tags=["probe-monitors"])


# ── probe datasets ────────────────────────────────────────────────────────────


@router.post("/datasets", response_model=ProbeDatasetResponse, status_code=201)
async def create_probe_dataset(
    request: ProbeDatasetCreate, db: AsyncSession = Depends(get_db)
) -> Any:
    """Create a label-mapped view over a downloaded dataset.

    The counts are computed NOW and stored. A view that cannot be scored is refused
    here rather than at run submission, because a view nobody can score is not worth
    keeping — and the refusal names both class counts so it can be acted on.
    """
    import uuid as _uuid

    from ....services.probe_monitor_service import (
        ProbeDatasetRefused,
        ResolvedColumns,
        build_view,
        describe_counts,
        load_columns,
        resolve_dataset_path,
    )

    try:
        dataset_uuid = _uuid.UUID(str(request.dataset_id))
    except (ValueError, AttributeError, TypeError):
        raise HTTPException(status_code=422, detail=f"{request.dataset_id} is not a dataset id")

    dataset = (
        await db.execute(select(Dataset).where(Dataset.id == dataset_uuid))
    ).scalar_one_or_none()
    if dataset is None:
        raise HTTPException(status_code=404, detail=f"Dataset {request.dataset_id} not found")

    try:
        path = resolve_dataset_path(dataset.raw_path)
        columns = ResolvedColumns(
            input_column=request.input_column,
            label_column=request.label_column,
            pair_column=request.pair_column,
        )
        inputs, labels, pairs, _total = load_columns(path, columns, split=request.split)
        built = build_view(
            inputs,
            labels,
            dict(request.label_mapping),
            keyword_filter=request.keyword_filter.model_dump()
            if request.keyword_filter
            else None,
            pair_values=pairs,
            role=request.role,
        )
    except ProbeDatasetRefused as refused:
        raise HTTPException(status_code=422, detail=str(refused))
    except FileNotFoundError as missing:
        raise HTTPException(status_code=422, detail=str(missing))
    except ValueError as bad:
        raise HTTPException(status_code=422, detail=str(bad))

    view = ProbeMonitorDataset(
        name=request.name,
        dataset_id=dataset_uuid,
        config=request.config,
        split=request.split,
        input_column=request.input_column,
        label_column=request.label_column,
        label_mapping=dict(request.label_mapping),
        keyword_filter=request.keyword_filter.model_dump() if request.keyword_filter else None,
        pair_column=request.pair_column,
        role=request.role,
        distribution=request.distribution,
        counts=describe_counts(built),
    )
    db.add(view)
    await db.commit()
    await db.refresh(view)
    return view


@router.get("/datasets", response_model=List[ProbeDatasetResponse])
async def list_probe_datasets(
    role: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
) -> Any:
    query = select(ProbeMonitorDataset).order_by(ProbeMonitorDataset.created_at.desc())
    if role:
        query = query.where(ProbeMonitorDataset.role == role)
    return list((await db.execute(query)).scalars().all())


@router.delete("/datasets/{dataset_id}", status_code=204)
async def delete_probe_dataset(dataset_id: str, db: AsyncSession = Depends(get_db)) -> None:
    view = (
        await db.execute(
            select(ProbeMonitorDataset).where(ProbeMonitorDataset.id == dataset_id)
        )
    ).scalar_one_or_none()
    if view is None:
        raise HTTPException(status_code=404, detail=f"Probe dataset {dataset_id} not found")
    # A view a run TRAINED on cannot be deleted: the run's FK is CASCADE, so deleting
    # it would take the run and every probe with it. Refusing names what to do instead.
    users = (
        await db.execute(
            select(ProbeMonitorRun.id).where(ProbeMonitorRun.train_dataset_id == dataset_id)
        )
    ).scalars().all()
    if users:
        raise HTTPException(
            status_code=409,
            detail=(
                f"{len(users)} run(s) trained on this view ({', '.join(users[:3])}), and "
                f"deleting it would cascade to them and their probes. Delete those runs "
                f"first if that is what you want."
            ),
        )
    await db.delete(view)
    await db.commit()


# ── runs ──────────────────────────────────────────────────────────────────────


@router.post("/runs", status_code=202)
async def submit_probe_run(
    request: ProbeRunCreate, db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Queue a probe run. 202 with the row's id and the Celery task id."""
    from ....models.model import Model
    from ....workers.probe_monitor_tasks import run_probe_monitor

    # BEFORE ANYTHING ELSE: an unknown card is a 400 naming the cards that exist, and
    # `can_split=False` because this job places on ONE card — a 202 for an "all" request
    # would queue work only the worker could refuse.
    gpu_request = resolve_gpu_request(request.gpu, can_split=False)

    model = (
        await db.execute(select(Model).where(Model.id == request.model_id))
    ).scalar_one_or_none()
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model {request.model_id} not found")

    train_view = (
        await db.execute(
            select(ProbeMonitorDataset).where(
                ProbeMonitorDataset.id == request.train_dataset_id
            )
        )
    ).scalar_one_or_none()
    if train_view is None:
        raise HTTPException(
            status_code=404, detail=f"Probe dataset {request.train_dataset_id} not found"
        )
    if train_view.role != "train":
        raise HTTPException(
            status_code=422,
            detail=(
                f"probe dataset {train_view.id} has role '{train_view.role}'; a run must "
                f"train on a 'train' view. The roles are not interchangeable — a "
                f"calibration view has no positives at all."
            ),
        )
    for dataset_id in request.eval_dataset_ids:
        view = (
            await db.execute(
                select(ProbeMonitorDataset).where(ProbeMonitorDataset.id == dataset_id)
            )
        ).scalar_one_or_none()
        if view is None:
            raise HTTPException(status_code=404, detail=f"Probe dataset {dataset_id} not found")
        if view.role == "calibration":
            raise HTTPException(
                status_code=422,
                detail=(
                    f"probe dataset {dataset_id} is a calibration view and cannot be an "
                    f"evaluation set: it is not labelled for the concept, so an AUROC "
                    f"over it would measure nothing"
                ),
            )

    # 409: one active run per (model, training view). Two concurrent runs would write
    # to different artifact directories but contend for the same card and produce two
    # probes nobody asked to compare.
    active = (
        await db.execute(
            select(ProbeMonitorRun.id).where(
                ProbeMonitorRun.model_id == request.model_id,
                ProbeMonitorRun.train_dataset_id == request.train_dataset_id,
                ProbeMonitorRun.status.in_(("pending", "running")),
            )
        )
    ).scalars().first()
    if active:
        raise HTTPException(
            status_code=409,
            detail=(
                f"run {active} is already {'' }active for this model and training view; "
                f"cancel it before starting another"
            ),
        )

    row = ProbeMonitorRun(
        model_id=request.model_id,
        train_dataset_id=request.train_dataset_id,
        eval_dataset_ids=list(request.eval_dataset_ids),
        calibration_dataset_id=request.calibration_dataset_id,
        config=request.config.model_dump(),
        status="pending",
        gpu_request=gpu_request,
        environment={},
    )
    db.add(row)
    # ⚠ COMMITTED BEFORE DISPATCH, AND THE TASK REFUSES WITHOUT A ROW. That order is
    # what makes `missing_row="cancelled"` correct on the cancel scope. When a fix once
    # passed an id without creating the row, every UI extraction was silently refused
    # for a week.
    await db.commit()
    await db.refresh(row)

    try:
        task = gpu_delay(run_probe_monitor, gpu_request)(run_id=row.id)
    except Exception as exc:  # noqa: BLE001 - a broker outage must not leave a ghost row
        logger.exception("Could not dispatch probe run %s", row.id)
        row.status = "failed"
        row.error_message = f"Could not queue the run: {type(exc).__name__}: {exc}"[:2000]
        await db.commit()
        raise HTTPException(
            status_code=503,
            detail="Could not queue the run — the task broker is unavailable",
        )

    row.celery_task_id = task.id
    await db.commit()
    return {"id": row.id, "task_id": task.id, "status": "queued"}


@router.get("/runs", response_model=List[ProbeRunResponse])
async def list_probe_runs(
    status: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
) -> Any:
    query = select(ProbeMonitorRun).order_by(ProbeMonitorRun.created_at.desc()).limit(limit)
    if status:
        query = query.where(ProbeMonitorRun.status == status)
    return list((await db.execute(query)).scalars().all())


@router.get("/runs/{run_id}", response_model=ProbeRunResponse)
async def get_probe_run(run_id: str, db: AsyncSession = Depends(get_db)) -> Any:
    row = (
        await db.execute(select(ProbeMonitorRun).where(ProbeMonitorRun.id == run_id))
    ).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Probe monitor run {run_id} not found")
    return row


@router.post("/runs/{run_id}/cancel", status_code=202)
async def cancel_probe_run(run_id: str, db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """Request a COOPERATIVE stop. The worker notices at its next stage boundary.

    Not a `revoke(terminate=True)`: every worker here is `--pool=solo`, which has no
    pool child to signal and never reads the control queue while busy — so a
    terminating revoke returns cleanly and does nothing at all.
    """
    from ....core.cancellation import request_cancel

    row = (
        await db.execute(select(ProbeMonitorRun).where(ProbeMonitorRun.id == run_id))
    ).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Probe monitor run {run_id} not found")
    if row.status not in ("pending", "running"):
        raise HTTPException(
            status_code=409,
            detail=f"run {run_id} is already {row.status}; there is nothing to cancel",
        )
    request_cancel("probe_monitor_run", run_id, celery_task_id=row.celery_task_id)
    return {"id": run_id, "status": "cancelling"}


@router.delete("/runs/{run_id}", status_code=204)
async def delete_probe_run(run_id: str, db: AsyncSession = Depends(get_db)) -> None:
    """Delete a run, its probes, and its artifacts.

    A RUNNING run is STOPPED first. Deleting the row while the worker continues leaves
    it computing over an unlinked artifact directory — which happened here on 2026-09-12
    and held a card for twenty minutes with a model download queued behind it.
    """
    import shutil

    from ....core.cancellation import request_cancel
    from ....core.config import settings

    row = (
        await db.execute(select(ProbeMonitorRun).where(ProbeMonitorRun.id == run_id))
    ).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Probe monitor run {run_id} not found")
    if row.status in ("pending", "running"):
        request_cancel("probe_monitor_run", run_id, celery_task_id=row.celery_task_id)

    artifact_dir = row.artifact_dir
    await db.delete(row)
    await db.commit()

    if artifact_dir:
        try:
            target = settings.resolve_deletable_path(artifact_dir)
            if target.exists():
                shutil.rmtree(target)
        except ValueError as refused:
            # A stored path is not a trusted one; refusing to delete outside the
            # deletable roots must not fail the request that already removed the row.
            logger.error("Refusing to delete %s: %s", artifact_dir, refused)
        except OSError as exc:
            logger.warning("Could not remove %s: %s", artifact_dir, exc)


# ── probes ────────────────────────────────────────────────────────────────────


def summary_without_curve(probe: Any) -> ProbeMonitorSummary:
    """A probe summary with `val_metrics["history"]` removed.

    ⚠ THE CURVE BELONGS IN THE REPORT, NOT IN THE LIST. `val_metrics["history"]` carries
    one entry per epoch — a few hundred for a normal run — and the list endpoint is what
    the panel polls while a run is going. Returning every probe's full curve on every poll
    is hundreds of kilobytes for a payload whose job is one line per probe.
    `GET /probes/{id}` returns it in full, which is where a reader who wants the curve is
    already looking.

    A separate function rather than a comprehension inside the route, so the decision can
    be tested without an HTTP client and the call can be asserted by walking the AST.
    """
    summary = ProbeMonitorSummary.model_validate(probe)
    metrics = summary.val_metrics or {}
    if "history" not in metrics:
        return summary
    return summary.model_copy(
        update={"val_metrics": {k: v for k, v in metrics.items() if k != "history"}}
    )


@router.get("/probes", response_model=List[ProbeMonitorSummary])
async def list_probes(
    run_id: Optional[str] = Query(None),
    selected_only: bool = Query(False),
    db: AsyncSession = Depends(get_db),
) -> Any:
    query = select(ProbeMonitor).order_by(ProbeMonitor.created_at.desc())
    if run_id:
        query = query.where(ProbeMonitor.run_id == run_id)
    if selected_only:
        query = query.where(ProbeMonitor.selected.is_(True))
    probes = list((await db.execute(query)).scalars().all())
    # ⚠ THE TRAINING CURVE IS STRIPPED FROM THE LIST, AND KEPT BY THE REPORT.
    #
    # `val_metrics["history"]` carries one entry per epoch — a few hundred for a normal
    # run — and this endpoint is what the panel polls while a run is going. Returning
    # every probe's full curve on every poll is hundreds of kilobytes for a payload whose
    # job is a table of one line per probe. `GET /probes/{id}` returns it in full, which
    # is where a reader who wants the curve is already looking.
    return [summary_without_curve(probe) for probe in probes]


@router.get("/probes/{probe_id}", response_model=ProbeReport)
async def get_probe_report(probe_id: str, db: AsyncSession = Depends(get_db)) -> Any:
    """The assembled report: evaluations, the rung WITH ITS WORDING, and the SAE pair."""
    from ....schemas.evidence_ladder import probe_rung_language, probe_rung_next_step

    probe = (
        await db.execute(select(ProbeMonitor).where(ProbeMonitor.id == probe_id))
    ).scalar_one_or_none()
    if probe is None:
        raise HTTPException(status_code=404, detail=f"Probe {probe_id} not found")

    evaluations = list(
        (
            await db.execute(
                select(ProbeMonitorEvaluation)
                .where(ProbeMonitorEvaluation.probe_id == probe_id)
                .order_by(ProbeMonitorEvaluation.created_at.asc())
            )
        ).scalars().all()
    )
    # The dense↔SAE pair: the same run, layer and rule, the other variant. Assembled
    # here so a reader can see what k-sparsity cost without doing the join themselves.
    paired = (
        await db.execute(
            select(ProbeMonitor.id).where(
                ProbeMonitor.run_id == probe.run_id,
                ProbeMonitor.layer == probe.layer,
                ProbeMonitor.rule == probe.rule,
                ProbeMonitor.variant != probe.variant,
            )
        )
    ).scalars().first()
    judge_runs = list(
        (
            await db.execute(
                select(ProbeMonitorJudgeRun).where(ProbeMonitorJudgeRun.probe_id == probe_id)
            )
        ).scalars().all()
    )

    # 033: the dictionary's HuggingFace home, so the export section can say WHY an SAE probe
    # cannot be exported rather than only disabling its button.
    sae_hf_repo = None
    if probe.variant == "sae" and probe.sae_id:
        from ....models.external_sae import ExternalSAE

        sae_row = (
            await db.execute(select(ExternalSAE).where(ExternalSAE.id == probe.sae_id))
        ).scalar_one_or_none()
        if sae_row is not None and sae_row.hf_repo_id and sae_row.hf_filepath:
            sae_hf_repo = sae_row.hf_repo_id

    return ProbeReport(
        probe=ProbeMonitorSummary.model_validate(probe),
        sae_hf_repo=sae_hf_repo,
        rung_language=probe_rung_language(probe.rung),
        rung_next_step=probe_rung_next_step(probe.rung),
        evaluations=[ProbeEvaluationResponse.model_validate(e) for e in evaluations],
        paired_probe_id=paired,
        judge_runs=[
            {
                "id": run.id,
                "model": run.model,
                "status": run.status,
                "prompt_version": run.prompt_version,
                "parse_failures": run.parse_failures,
                "metrics": run.metrics,
            }
            for run in judge_runs
        ],
    )


@router.post("/probes/{probe_id}/evaluate", status_code=202)
async def evaluate_probe_endpoint(
    probe_id: str,
    dataset_ids: List[str],
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """Evaluate an existing probe on additional sets (this is how rung 2 is reached)."""
    from ....workers.probe_monitor_tasks import evaluate_probe_monitor

    probe = (
        await db.execute(select(ProbeMonitor).where(ProbeMonitor.id == probe_id))
    ).scalar_one_or_none()
    if probe is None:
        raise HTTPException(status_code=404, detail=f"Probe {probe_id} not found")
    if not dataset_ids:
        raise HTTPException(status_code=422, detail="name at least one evaluation set")
    task = gpu_delay(evaluate_probe_monitor, "auto")(
        probe_id=probe_id, dataset_ids=list(dataset_ids)
    )
    return {"probe_id": probe_id, "task_id": task.id, "status": "queued"}


@router.post("/probes/{probe_id}/score", status_code=202)
async def score_probe(
    probe_id: str, request: ScoreRequest, db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Score one input offline (FR-12). 202 — there is no synchronous model load here."""
    from ....workers.probe_monitor_tasks import score_probe_monitor

    probe = (
        await db.execute(select(ProbeMonitor).where(ProbeMonitor.id == probe_id))
    ).scalar_one_or_none()
    if probe is None:
        raise HTTPException(status_code=404, detail=f"Probe {probe_id} not found")
    task = gpu_delay(score_probe_monitor, "auto")(
        probe_id=probe_id, text=request.text, messages=request.messages
    )
    return {"probe_id": probe_id, "task_id": task.id, "status": "queued"}


@router.get("/probes/{probe_id}/score/{task_id}")
async def get_score_result(
    probe_id: str, task_id: str, db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Collect an offline score (FR-12). The second half of `POST …/score`'s contract.

    ⚠ WITHOUT THIS, SCORING WAS WRITE-ONLY. The POST returned a 202 and a task id and
    there was nothing to read it back with, so the whole feature could be exercised and
    never observed — the shape this repo keeps finding: implemented, tested by importing
    the piece directly, and unreachable for any caller.

    A FAILED task reports its REASON rather than an empty trace. An empty trace reads as
    "the probe scored nothing"; a failure reads as "the probe did not run", and those are
    different facts.

    PENDING is reported as-is and never as an error. Celery returns PENDING both for a
    task it has not started and for a task id it has never heard of, so this endpoint
    cannot tell "queued" from "unknown" — and claiming either would be a guess.
    """
    import asyncio

    from ....core.celery_app import celery_app

    probe = (
        await db.execute(select(ProbeMonitor).where(ProbeMonitor.id == probe_id))
    ).scalar_one_or_none()
    if probe is None:
        raise HTTPException(status_code=404, detail=f"Probe {probe_id} not found")

    def _read():
        result = celery_app.AsyncResult(task_id)
        return result.state, result.info

    # `AsyncResult` touches the broker, which blocks — off the event loop.
    state, info = await asyncio.to_thread(_read)

    if state == "SUCCESS":
        return {"task_id": task_id, "probe_id": probe_id, "status": state, "result": info}
    if state == "FAILURE":
        return {
            "task_id": task_id,
            "probe_id": probe_id,
            "status": state,
            "error": str(info),
            "result": None,
        }
    return {
        "task_id": task_id,
        "probe_id": probe_id,
        "status": state,
        "result": None,
    }


# ── judge runs ────────────────────────────────────────────────────────────────


@router.post("/judge-runs", status_code=202)
async def submit_judge_run(
    request: JudgeRunCreate, db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Queue the LLM-monitor baseline. CPU — it is HTTP calls to a served model."""
    from ....services.probe_monitor_judge import JUDGE_PROMPT_VERSION
    from ....utils.url_validation import validate_llm_endpoint_url
    from ....workers.probe_monitor_tasks import run_probe_monitor_judge

    try:
        endpoint = validate_llm_endpoint_url(request.endpoint)
    except ValueError as bad:
        raise HTTPException(status_code=422, detail=str(bad))

    if request.probe_id:
        probe = (
            await db.execute(select(ProbeMonitor).where(ProbeMonitor.id == request.probe_id))
        ).scalar_one_or_none()
        if probe is None:
            raise HTTPException(status_code=404, detail=f"Probe {request.probe_id} not found")

    row = ProbeMonitorJudgeRun(
        endpoint=endpoint,
        model=request.model,
        prompt_version=JUDGE_PROMPT_VERSION,
        dataset_ids=list(request.dataset_ids),
        probe_id=request.probe_id,
        status="pending",
        metrics={},
        # ⚠ PERSISTED, BECAUSE THEY WERE BEING DROPPED HERE. Both are declared and validated on
        # `JudgeRunCreate` and neither reached the row, the task or the service: a run submitted
        # with `max_rows_per_set: 200` judged all 5,737 rows of its five sets, 5.7x the work, and a
        # tightened parse-failure limit was silently the module default.
        max_rows_per_set=request.max_rows_per_set,
        parse_failure_limit=request.parse_failure_limit,
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)

    task = run_probe_monitor_judge.delay(judge_run_id=row.id)
    row.celery_task_id = task.id
    await db.commit()
    return {"id": row.id, "task_id": task.id, "status": "queued"}


@router.post("/judge-runs/{judge_run_id}/cancel", status_code=202)
async def cancel_judge_run(
    judge_run_id: str, db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Stop a judge baseline.

    ⚠ THIS ROUTE WAS MISSING AND THE SCOPE WAS ALREADY REGISTERED — a lifecycle that
    could be started and never stopped, which for the judge means thousands of paid
    calls against a served model for a result nobody is waiting for.
    `test_cancel_registry_completeness` names exactly that: every registered scope needs
    an operator-facing route, or a recorded reason why it has none.
    """
    from ....core.cancellation import request_cancel

    row = (
        await db.execute(
            select(ProbeMonitorJudgeRun).where(ProbeMonitorJudgeRun.id == judge_run_id)
        )
    ).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Judge run {judge_run_id} not found")
    if row.status not in ("pending", "running"):
        raise HTTPException(
            status_code=409,
            detail=f"judge run {judge_run_id} is already {row.status}",
        )
    request_cancel("probe_monitor_judge", judge_run_id, celery_task_id=row.celery_task_id)
    return {"id": judge_run_id, "status": "cancelling"}


@router.get("/judge-runs", response_model=List[Dict[str, Any]])
async def list_judge_runs(
    probe_id: Optional[str] = Query(None), db: AsyncSession = Depends(get_db)
) -> Any:
    query = select(ProbeMonitorJudgeRun).order_by(ProbeMonitorJudgeRun.created_at.desc())
    if probe_id:
        query = query.where(ProbeMonitorJudgeRun.probe_id == probe_id)
    rows = list((await db.execute(query)).scalars().all())
    return [
        {
            "id": row.id,
            "endpoint": row.endpoint,
            "model": row.model,
            "prompt_version": row.prompt_version,
            "dataset_ids": row.dataset_ids,
            "probe_id": row.probe_id,
            "status": row.status,
            "progress": row.progress,
            "parse_failures": row.parse_failures,
            "metrics": row.metrics,
            "error_message": row.error_message,
        }
        for row in rows
    ]


# ── 033: the definition, its download, and publication ────────────────────────


class BuildDefinitionRequest(BaseModel):
    """What a build takes. `acknowledge_below_rung2` is the FR-5 gate's key."""

    model_config = ConfigDict(extra="forbid")

    acknowledge_below_rung2: Optional[Dict[str, Any]] = None
    vector_count: int = Field(16, ge=8, le=32)
    seed: int = 1337


class PublishDefinitionRequest(BaseModel):
    """⚠ THE TOKEN IS A REQUEST FIELD AND IS NEVER PERSISTED. It reaches the task as an argument
    and is written to no row, no task result and no log line. A token in `task_queue.args` would be
    readable by anyone who can list tasks."""

    model_config = ConfigDict(extra="forbid")

    repo_id: str = Field(min_length=1, max_length=200)
    private: bool = True
    token: Optional[str] = None


@router.post("/probes/{probe_id}/definition", status_code=202)
async def build_definition(
    probe_id: str,
    request: BuildDefinitionRequest,
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """Build and cache the probe's portable definition (033 FR-7). 202 — it is a GPU job.

    The evidence gate lives in the BUILDER, not here, and deliberately: a build dispatched from the
    MCP tool or re-run by a task must meet the same refusals, and a gate in the endpoint would be
    one the other callers bypass.
    """
    from ....workers.probe_monitor_tasks import build_probe_definition

    probe = (
        await db.execute(select(ProbeMonitor).where(ProbeMonitor.id == probe_id))
    ).scalar_one_or_none()
    if probe is None:
        raise HTTPException(status_code=404, detail=f"Probe {probe_id} not found")
    task = gpu_delay(build_probe_definition, "auto")(
        probe_id=probe_id,
        acknowledge_below_rung2=request.acknowledge_below_rung2,
        vector_count=request.vector_count,
        seed=request.seed,
    )
    return {"probe_id": probe_id, "task_id": task.id, "status": "queued"}


@router.get("/probes/{probe_id}/definition")
async def get_definition(probe_id: str, db: AsyncSession = Depends(get_db)) -> Any:
    """The cached definition as JSON, or 409 with what to do about it."""
    probe = (
        await db.execute(select(ProbeMonitor).where(ProbeMonitor.id == probe_id))
    ).scalar_one_or_none()
    if probe is None:
        raise HTTPException(status_code=404, detail=f"Probe {probe_id} not found")
    if not probe.definition_path:
        invalidated = (probe.definition_build or {}).get("invalidated")
        detail = (
            f"probe {probe_id} has no built definition"
            + (
                f"; the previous one was invalidated ({invalidated.get('reason')}) because what "
                f"it stated changed — build again"
                if invalidated
                else "; POST to this path to build one"
            )
        )
        raise HTTPException(status_code=409, detail=detail)
    import json
    from pathlib import Path

    path = settings.resolve_data_path(probe.definition_path)
    if not path.exists():
        raise HTTPException(
            status_code=409,
            detail=(
                f"probe {probe_id} records a definition at {probe.definition_path} and the file is "
                f"gone; build again"
            ),
        )
    return json.loads(path.read_text())


@router.get("/probes/{probe_id}/export")
async def export_definition(probe_id: str, db: AsyncSession = Depends(get_db)) -> Any:
    """Download it as `<slug>.probe.json` (033 FR-7)."""
    from fastapi.responses import FileResponse

    probe = (
        await db.execute(select(ProbeMonitor).where(ProbeMonitor.id == probe_id))
    ).scalar_one_or_none()
    if probe is None:
        raise HTTPException(status_code=404, detail=f"Probe {probe_id} not found")
    if not probe.definition_path:
        raise HTTPException(
            status_code=409,
            detail=f"probe {probe_id} has no built definition; POST /probes/{probe_id}/definition",
        )
    path = settings.resolve_data_path(probe.definition_path)
    if not path.exists():
        raise HTTPException(status_code=409, detail="the definition file is gone; build again")
    return FileResponse(
        str(path),
        media_type="application/json",
        filename=f"{probe_id}.probe.json",
    )



def _huggingface_rejects(token: str) -> Optional[str]:
    """`None` when HuggingFace accepts the token; a reason to refuse with when it rejects it.

    ⚠ "A TOKEN IS PRESENT" AND "A TOKEN WORKS" ARE DIFFERENT CLAIMS, and this endpoint used to make
    the first while promising the second. It checked `resolve_hf_token` for a truthy value and
    queued the upload, so an expired token produced a 202, a task that created nothing, and a
    failure at the last step — the exact shape the docstring above says it exists to prevent.

    ⚠ FOUND THE HARD WAY DURING 033 ACCEPTANCE: the token stored in this installation's Settings is
    37 characters, starts with `hf_`, and HuggingFace answers `Invalid user token`. Nothing had ever
    said so, because a public dataset download needs no credential and the only paths that would
    have noticed — gated downloads and publishes — had never run. Two datasets recorded as "gated,
    not downloaded" in the project notes were very likely this and not their gates.

    An authentication failure is a refusal. Anything else — a timeout, DNS, a 5xx — is NOT: it does
    not prove the token is bad, and refusing on it would make the publish path fail closed against
    an outage it has no business having an opinion about.
    """
    try:
        from huggingface_hub import HfApi

        HfApi(token=token).whoami()
        return None
    except Exception as exc:  # noqa: BLE001 - classified below, never swallowed silently
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status in (401, 403) or "Invalid user token" in str(exc):
            logger.warning("HuggingFace rejected the token for a publish request: %s", exc)
            return (
                "HuggingFace rejected this token (it is present but not valid — expired, revoked, "
                "or without write access). Replace it in Settings → API Keys or pass a working one "
                "in the request. Refused here rather than queueing an upload that would fail at "
                "its last step"
            )
        logger.warning(
            "could not verify the HuggingFace token (%s: %s) — proceeding, because this does not "
            "prove the token is bad",
            type(exc).__name__, exc,
        )
        return None

@router.post("/probes/{probe_id}/publish", status_code=202)
async def publish_definition(
    probe_id: str,
    request: PublishDefinitionRequest,
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """Publish to HuggingFace (033 FR-8). 202.

    ⚠ THE TOKEN IS CHECKED BEFORE ANY WORK IS QUEUED — a 401 here rather than a task that starts,
    uploads nothing and fails at the end. `resolve_hf_token` is the resolver every HuggingFace path
    here shares, so a token stored in Settings works without being repeated in the request.

    ⚠ THAT SENTENCE USED TO READ "the same resolver the SAE upload uses" AND IT WAS FALSE.
    `HuggingFaceSAEService.upload_sae` took a REQUIRED `access_token` straight to `HfApi` and
    resolved nothing, so a stored write token worked for a probe definition and not for an SAE.
    Found while publishing an SAE for 033 acceptance 8.1. It is true now because the upload path
    was changed to match this one — a claim about a neighbour is worth nothing until the neighbour
    is checked.
    """
    from ....services.huggingface_sae_service import resolve_hf_token

    probe = (
        await db.execute(select(ProbeMonitor).where(ProbeMonitor.id == probe_id))
    ).scalar_one_or_none()
    if probe is None:
        raise HTTPException(status_code=404, detail=f"Probe {probe_id} not found")
    if not probe.definition_path:
        raise HTTPException(
            status_code=409,
            detail=f"probe {probe_id} has no built definition; build it before publishing",
        )
    token = resolve_hf_token(request.token)
    if not token:
        raise HTTPException(
            status_code=401,
            detail=(
                "no HuggingFace token: pass one in the request or store it in Settings → API "
                "Keys. Checked before the upload is queued, so this is a refusal rather than a "
                "task that starts and fails at its last step"
            ),
        )
    rejected = _huggingface_rejects(token)
    if rejected:
        raise HTTPException(status_code=401, detail=rejected)
    from ....workers.probe_monitor_tasks import publish_probe_definition

    task = publish_probe_definition.delay(
        probe_id=probe_id,
        repo_id=request.repo_id,
        private=request.private,
        token=token,
    )
    return {"probe_id": probe_id, "task_id": task.id, "status": "queued"}
