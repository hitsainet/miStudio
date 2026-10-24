"""REST surface of the behaviour-detector reports (Feature 040 S2; FTDD §5.2; PADR IDL-61).

Every route is CPU-only and reads stored arrays, snapshots and rows — none loads a model or takes a GPU
lease (FTDD §5.5). Every refusal is `{code, message, …details, next}` (`services/detector_next.py`);
every write records `X-MiStudio-Agent` as its actor.

* `PATCH /datasets/{id}/columns`              — declare grouping columns on a view (additive; FR-14)
* `GET  /probes/{id}/evaluations/{eid}/breakdown` — recall at the deployed decision by subgroup (FR-15–18)
* `POST /behaviours/{key}/success-reports`    — build (or return the stored) success report (FR-22–25)
* `GET  /behaviours/{key}/success-reports`    — list them, newest first, paged
* `GET  /success-reports/{id}`                — one, with `stale` computed on read
* `GET  /behaviours/{key}/model-comparison`   — one report per model, side by side (FR-26)
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from ....core.config import settings
from ....core.deps import get_db
from ....models.detector_report import DetectorSuccessReport
from ....models.probe_monitor import (
    ProbeMonitor,
    ProbeMonitorDataset,
    ProbeMonitorEvaluation,
    ProbeMonitorRun,
)
from ....services import behaviour_catalogue as catalogue
from ....services import detector_success_report as reports
from ....services import probe_breakdown, view_columns
from ....services.detector_next import DetectorRefused
from ....services.probe_dataset_lifecycle import actor_of
from ....services.probe_evaluation_scores import EvaluationArtifactError

from ._probe_ids import refuse_blend_report_ids  # noqa: E402

# Feature 041 FR-18: the breakdown route takes `{probe_id}`; a blend report id is refused here too.
router = APIRouter(prefix="/probe-monitors", tags=["behaviour-detectors"], dependencies=[Depends(refuse_blend_report_ids)])


class ViewColumnsUpdate(BaseModel):
    """S2 accepts `group_columns` only (S3 adds `text_column`, `rule_columns`)."""

    model_config = ConfigDict(extra="forbid")

    group_columns: List[str] = Field(min_length=1)


class SuccessReportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    version: Optional[int] = Field(default=None, ge=1)
    model_id: Optional[str] = Field(default=None, max_length=255)
    #: A miDataworks dataset-version id (every view registered from it), or explicit `view_ids`.
    dataset_version: Optional[str] = Field(default=None, max_length=255)
    view_ids: Optional[List[str]] = Field(default=None, min_length=1, max_length=200)
    probe_id: Optional[str] = Field(default=None, max_length=36)
    #: Feature 041 FR-20: judge `blend_alone` criteria on this blend report's candidate (both or neither);
    #: without them, on the current suggested blend of the model when there is one.
    blend_report_id: Optional[str] = Field(default=None, max_length=36)
    candidate: Optional[str] = Field(default=None, max_length=64)


def _http(refusal: Exception) -> HTTPException:
    if isinstance(refusal, DetectorRefused):
        return refusal.to_http()
    if isinstance(refusal, catalogue.BehaviourRefused):
        return HTTPException(status_code=refusal.status, detail=refusal.as_detail())
    raise refusal


def _broken(exc: Exception, what: str) -> HTTPException:
    """An integrity failure is a 500 that names the check, never an empty result (FTID §13)."""
    check = getattr(exc, "check", "unknown")
    return HTTPException(status_code=500, detail={
        "code": "artifacts_broken", "check": check,
        "message": f"{what} failed the {check!r} check and cannot be served; see the miStudio log", "next": []})


@router.patch("/datasets/{dataset_id}/columns")
async def update_probe_dataset_columns(
    dataset_id: str, body: ViewColumnsUpdate, request: Request, db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Declare grouping columns on a registered view (FR-14). ADDITIVE: send every declared column plus
    the new ones; a removal or rename is refused `column_not_additive`. New columns are read from the
    downloaded split and snapshotted by split position."""
    view = (await db.execute(select(ProbeMonitorDataset).where(ProbeMonitorDataset.id == dataset_id))).scalar_one_or_none()
    if view is None:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": f"probe dataset {dataset_id} not found", "next": []})
    try:
        return await view_columns.add(db, view, body.group_columns, actor_of(request),
                                       view_columns.ColumnLimits.from_settings())
    except DetectorRefused as refusal:
        raise refusal.to_http() from refusal


async def _floor_override(db: AsyncSession, probe: ProbeMonitor) -> Optional[int]:
    """The behaviour's own calibration floor (FR-19): the version the probe cites, else the current one."""
    if not probe.behaviour:
        return None
    from ....models.probe_behaviour import ProbeBehaviour, ProbeBehaviourVersion

    head = (await db.execute(select(ProbeBehaviour).where(ProbeBehaviour.key == probe.behaviour))).scalar_one_or_none()
    if head is None:
        return None
    number = getattr(probe, "behaviour_version", None) or head.current_version
    version = (await db.execute(select(ProbeBehaviourVersion).where(
        ProbeBehaviourVersion.key == probe.behaviour, ProbeBehaviourVersion.version == number))).scalar_one_or_none()
    return version.calibration_floor if version is not None else None


@router.get("/probes/{probe_id}/evaluations/{evaluation_id}/breakdown")
async def get_probe_breakdown(
    probe_id: str,
    evaluation_id: str,
    by_column: str = Query(..., min_length=1, max_length=255, description="A column declared on the evaluated view"),
    filter: List[str] = Query(default=[], description="Repeatable `column=value` on declared columns"),  # noqa: A002
    mode: Literal["recorded", "target_fpr"] = Query("recorded"),
    target_fpr: Optional[float] = Query(None, gt=0.0, lt=1.0),
    min_rows: Optional[int] = Query(None, ge=1),
    limit: Optional[int] = Query(None, ge=1),
    cursor: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """Recall and false-positive rate per value of one declared column, at the decision a deployment
    applies (or at a target FPR cut on the calibration negatives), from stored arrays. No GPU."""
    evaluation = (await db.execute(select(ProbeMonitorEvaluation).where(
        ProbeMonitorEvaluation.id == evaluation_id))).scalar_one_or_none()
    if evaluation is None or evaluation.probe_id != probe_id:
        raise HTTPException(status_code=404, detail={
            "code": "not_found", "message": f"evaluation {evaluation_id} of probe {probe_id} not found", "next": []})
    probe = (await db.execute(select(ProbeMonitor).where(ProbeMonitor.id == probe_id))).scalar_one()
    run = (await db.execute(select(ProbeMonitorRun).where(ProbeMonitorRun.id == probe.run_id))).scalar_one_or_none()
    view = (await db.execute(select(ProbeMonitorDataset).where(
        ProbeMonitorDataset.id == evaluation.dataset_id))).scalar_one_or_none()
    if view is None:
        raise HTTPException(status_code=404, detail={
            "code": "view_not_found", "message": f"the view evaluation {evaluation_id} read is gone", "next": []})
    try:
        request = probe_breakdown.BreakdownRequest(
            by_column=by_column, filters=probe_breakdown.parse_filters(filter), mode=mode,
            target_fpr=target_fpr, min_rows=min_rows, limit=limit, cursor=cursor)
        floor = await _floor_override(db, probe)
        return await run_in_threadpool(
            probe_breakdown.compute, probe=probe, run=run, evaluation=evaluation, view=view, request=request,
            settings=probe_breakdown.BreakdownSettings.from_settings(),
            root=Path(settings.data_dir) / "probe_monitors", floor_override=floor)
    except DetectorRefused as refusal:
        raise refusal.to_http() from refusal
    except (EvaluationArtifactError, view_columns.SnapshotBroken) as exc:
        raise _broken(exc, f"evaluation {evaluation_id}") from exc


@router.post("/behaviours/{key}/success-reports")
async def create_success_report(
    key: str, body: SuccessReportRequest, request: Request, db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Judge every success criterion of a behaviour version for one model and dataset identity. Stored
    immutably; the same inputs return the same report id."""
    try:
        return await reports.build(db, key, body.model_dump(), actor_of(request),
                                   root=Path(settings.data_dir) / "probe_monitors")
    except (DetectorRefused, catalogue.BehaviourRefused) as refusal:
        raise _http(refusal) from refusal
    except (EvaluationArtifactError, view_columns.SnapshotBroken) as exc:
        raise _broken(exc, "a cited evaluation") from exc


def _offset(cursor: Optional[str]) -> int:
    if not cursor:
        return 0
    try:
        data = json.loads(base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)))
        return int(data["o"])
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail={"code": "invalid_field", "message": "malformed cursor",
                                                     "field": "cursor", "rule": "malformed", "next": []}) from exc


def _cursor(offset: int) -> str:
    return base64.urlsafe_b64encode(json.dumps({"o": offset}).encode()).decode().rstrip("=")


@router.get("/behaviours/{key}/success-reports")
async def list_success_reports(
    key: str, limit: Optional[int] = Query(None, ge=1), cursor: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    from sqlalchemy import func

    from ....core.runtime_settings import effective

    _settings = effective()
    size = limit if limit is not None else min(_settings.report_list_page_default, _settings.report_list_page_max)
    if size > _settings.report_list_page_max:
        raise HTTPException(status_code=422, detail={
            "code": "over_cap",
            "message": f"a page of {size} is more than report_list_page_max ({_settings.report_list_page_max})",
            "setting": "report_list_page_max", "value": _settings.report_list_page_max, "next": []})
    offset = _offset(cursor)
    rows = (await db.execute(select(DetectorSuccessReport).where(DetectorSuccessReport.behaviour_key == key)
                             .order_by(DetectorSuccessReport.created_at.desc(), DetectorSuccessReport.id.desc())
                             .offset(offset).limit(size + 1))).scalars().all()
    total = (await db.execute(select(func.count()).select_from(DetectorSuccessReport)
                              .where(DetectorSuccessReport.behaviour_key == key))).scalar_one()
    page = rows[:size]
    return {
        "key": key,
        "reports": [{"id": r.id, "behaviour_version": r.behaviour_version, "model_id": r.model_id,
                     "overall_verdict": r.overall_verdict, "report_algorithm": r.report_algorithm,
                     "dataset_identity": r.dataset_identity, "created_by": r.created_by,
                     "created_at": r.created_at.isoformat() if r.created_at else None} for r in page],
        "total": int(total),
        "next_cursor": _cursor(offset + size) if len(rows) > size else None,
    }


@router.get("/success-reports/{report_id}")
async def read_success_report(report_id: str, db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    row = (await db.execute(select(DetectorSuccessReport).where(DetectorSuccessReport.id == report_id))).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": f"success report {report_id} not found", "next": []})
    return await reports.envelope(db, row)


@router.get("/behaviours/{key}/model-comparison")
async def compare_models(
    key: str,
    request: Request,
    version: Optional[int] = Query(None, ge=1),
    dataset_version: Optional[str] = Query(None, max_length=255),
    view_ids: List[str] = Query(default=[]),
    limit: Optional[int] = Query(None, ge=1),
    cursor: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """One success report per model, in adjacent rows (FR-26)."""
    from ....services import model_comparison

    try:
        return await model_comparison.compare(
            db, key, version=version, dataset_version=dataset_version, view_ids=view_ids or None,
            limit=limit, offset=_offset(cursor), actor=actor_of(request),
            root=Path(settings.data_dir) / "probe_monitors", cursor_of=_cursor)
    except (DetectorRefused, catalogue.BehaviourRefused) as refusal:
        raise _http(refusal) from refusal
    except (EvaluationArtifactError, view_columns.SnapshotBroken) as exc:
        raise _broken(exc, "a cited evaluation") from exc
