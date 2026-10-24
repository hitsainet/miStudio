"""REST surface of the behaviour catalogue (Feature 037, FR-13; ENH-005 D20–D27, D48).

Every rule lives in `services/behaviour_catalogue.py`; these routes translate its `BehaviourRefused`
into an HTTP status with ``{code, message}`` (Part A's `PromptRefused` convention).

WHO ASKED is read from the ``X-MiStudio-Agent`` header, never from the body (TI4): the MCP client sends it
on every request, so history rows read ``agent:mcp`` for an agent and ``operator`` for the UI.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from sqlalchemy.ext.asyncio import AsyncSession

from ....core.deps import get_db
from ....schemas.behaviour import (
    BehaviourArchive,
    BehaviourCreate,
    BehaviourRoleUpdate,
    BehaviourVersionCreate,
    BehaviourViewLink,
    PositiveLabelsConfirm,
)
from ....services import behaviour_catalogue as catalogue
from ....services.detector_next import DetectorRefused
from ....services.probe_dataset_lifecycle import actor_of
from ....schemas.probe_blend import FamilyPublish, FamilyScoreBatch, SuggestedBlendMark
from ._probe_ids import refuse_blend_report_ids

# Feature 041 FR-18: `/behaviours/{key}/probes/{probe_id}/role` (and any later `{probe_id}` route here) refuses a
# blend report id. Removing the dependency fails test_not_a_probe.py.
router = APIRouter(prefix="/probe-monitors", tags=["probe-behaviours"],
                   dependencies=[Depends(refuse_blend_report_ids)])

BASE = "/behaviours"


def _refused(refusal: catalogue.BehaviourRefused) -> HTTPException:
    return HTTPException(status_code=refusal.status, detail=refusal.as_detail())


#: How many pages of `limit` heads a `status`-filtered list scans before it answers with what it found
#: (`total_exact: false`): status is computed on read (D49), so a filter on it cannot be pushed into SQL.
STATUS_SCAN_PAGES = 10


@router.get(BASE)
async def list_behaviours(
    include_uncatalogued: bool = Query(True, description="Also list keys in use that have no catalogue row"),
    include_archived: bool = Query(False),
    limit: Optional[int] = Query(None, ge=1, description="Behaviours per page (default: the runtime setting behaviour_list_page_default)"),
    cursor: Optional[str] = Query(None, description="`next_cursor` of the previous page, with the same filters"),
    q: Optional[str] = Query(None, max_length=200, description="Only keys or names containing this text"),
    status: Optional[Literal["draft", "validated", "monitoring", "unknown", "archived"]] = Query(None),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """The catalogue, one page at a time in key order (Feature 040 FR-6): catalogued behaviours with their
    computed status (D49) and the keys in use that are not catalogued (FR-4).

    ⚠ A PAGE COSTS THE SAME AT ANY CATALOGUE SIZE: heads by keyset, one batched version read, counts and
    the checklist for the page's keys only (`checklist_many`), and miLLM read at most ONCE per request.
    With fewer behaviours than a page and no new parameters, `behaviours` and `uncatalogued` are what 037
    returned."""
    from starlette.concurrency import run_in_threadpool

    from ....core.runtime_settings import effective
    from ....services import behaviour_checklist, millm_read

    settings = effective()
    page_max = settings.behaviour_list_page_max
    size = limit if limit is not None else min(settings.behaviour_list_page_default, page_max)
    if size > page_max:
        raise HTTPException(status_code=422, detail={
            "code": "over_cap",
            "message": f"a page of {size} is more than the runtime setting behaviour_list_page_max ({page_max})",
            "setting": "behaviour_list_page_max", "value": page_max, "field": "limit", "got": size, "next": [],
        })
    filters = {"include_uncatalogued": include_uncatalogued, "include_archived": include_archived,
               "q": q, "status": status}
    try:
        position = catalogue.decode_cursor(cursor, filters) if cursor else catalogue.START
    except DetectorRefused as refusal:
        raise refusal.to_http() from refusal

    armed_cache: list = []

    def armed_once():
        if not armed_cache:
            armed_cache.append(millm_read.armed_source_probe_ids())
        return armed_cache[0]

    async def page_rows(heads):
        rows = await catalogue.rows_for(db, heads)
        checklists = await behaviour_checklist.checklist_many(db, heads, armed_once)
        for row in rows:
            row["status"] = checklists[row["key"]]["status"]
        return rows

    after = position["k"]
    rows: list = []
    done_k = position["done_k"]
    total: Optional[int] = None
    total_exact = False
    if not done_k and status is None:
        heads = await catalogue.heads_page(db, include_archived=include_archived, q=q, after=after,
                                           limit=size + 1)
        done_k = len(heads) <= size
        rows = await page_rows(heads[:size])
        after = rows[-1]["key"] if rows else after
        total = await catalogue.count_heads(db, include_archived=include_archived, q=q)
        total_exact = True
    elif not done_k:
        # Status is computed on read (D49): scan forward in pages until this page fills, the catalogue
        # ends, or STATUS_SCAN_PAGES pages have been read (then `next_cursor` continues the scan).
        for _ in range(STATUS_SCAN_PAGES):
            heads = await catalogue.heads_page(db, include_archived=include_archived, q=q, after=after,
                                               limit=size)
            for row in await page_rows(heads):
                after = row["key"]
                if row["status"] == status:
                    rows.append(row)
                    if len(rows) == size:
                        break
            if len(rows) == size:
                break
            if len(heads) < size:
                done_k = True
                break
        if done_k and cursor is None:
            total, total_exact = len(rows), True
    elif status is None:
        total = await catalogue.count_heads(db, include_archived=include_archived, q=q)
        total_exact = True

    uncatalogued: list = []
    after_u = position["u"]
    done_u = position["done_u"] or not include_uncatalogued or status is not None
    uncatalogued_total = 0
    if not done_u:
        found = await catalogue.uncatalogued_page(db, q=q, after=after_u, limit=size + 1)
        done_u = len(found) <= size
        uncatalogued = found[:size]
        after_u = uncatalogued[-1]["key"] if uncatalogued else after_u
    if include_uncatalogued and status is None:
        uncatalogued_total = await catalogue.count_uncatalogued(db, q=q)

    next_cursor = None
    if not (done_k and done_u):
        next_cursor = catalogue.encode_cursor(filters, {"k": after, "u": after_u, "done_k": done_k,
                                                        "done_u": done_u})
    return {
        "behaviours": rows,
        "uncatalogued": uncatalogued,
        "next_cursor": next_cursor,
        "total": total,
        "total_exact": total_exact,
        "uncatalogued_total": uncatalogued_total,
        "limit": size,
    }


@router.post(BASE, status_code=201)
async def create_behaviour(
    body: BehaviourCreate, request: Request, db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Catalogue a key with version 1 of its meaning; records already using the key are adopted."""
    try:
        return await catalogue.create(db, body.model_dump(mode="json"), actor_of(request))
    except catalogue.BehaviourRefused as refusal:
        raise _refused(refusal) from refusal
    except DetectorRefused as refusal:
        raise refusal.to_http() from refusal


@router.get(f"{BASE}/{{key}}")
async def get_behaviour(key: str, db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """The behaviour, its current meaning, and every prompt, run, probe and view that uses its key."""
    try:
        return await catalogue.detail(db, key)
    except catalogue.BehaviourRefused as refusal:
        raise _refused(refusal) from refusal


@router.delete(f"{BASE}/{{key}}", status_code=204)
async def delete_behaviour(key: str, request: Request, db: AsyncSession = Depends(get_db)) -> Response:
    """Only a behaviour no record uses (409 otherwise — archive it)."""
    try:
        await catalogue.delete_if_unused(db, key, actor_of(request))
    except catalogue.BehaviourRefused as refusal:
        raise _refused(refusal) from refusal
    return Response(status_code=204)


@router.get(f"{BASE}/{{key}}/versions")
async def list_behaviour_versions(key: str, db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    try:
        return {"key": key, "versions": await catalogue.versions(db, key)}
    except catalogue.BehaviourRefused as refusal:
        raise _refused(refusal) from refusal


@router.post(f"{BASE}/{{key}}/versions", status_code=201)
async def new_behaviour_version(
    key: str, body: BehaviourVersionCreate, request: Request, db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """A new version of the meaning. Exported probes keep the version they were exported with.

    Feature 040 FR-3: with `base_version`, omitted fields carry over from that version; the response
    adds `diff` and `warnings`."""
    try:
        return await catalogue.new_version(
            db, key, body.model_dump(mode="json"), actor_of(request), fields_set=body.model_fields_set
        )
    except catalogue.BehaviourRefused as refusal:
        raise _refused(refusal) from refusal
    except DetectorRefused as refusal:
        raise refusal.to_http() from refusal


@router.put(f"{BASE}/{{key}}/probes/{{probe_id}}/role")
async def set_behaviour_probe_role(
    key: str,
    probe_id: str,
    body: BehaviourRoleUpdate,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    try:
        return await catalogue.set_role(db, key, probe_id, body.role, actor_of(request), body.reason)
    except catalogue.BehaviourRefused as refusal:
        raise _refused(refusal) from refusal


@router.post(f"{BASE}/{{key}}/views/{{dataset_id}}")
async def link_behaviour_view(
    key: str,
    dataset_id: str,
    request: Request,
    body: Optional[BehaviourViewLink] = None,
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    body = body or BehaviourViewLink()
    try:
        return await catalogue.link_view(db, key, dataset_id, body.use, actor_of(request), body.reason)
    except catalogue.BehaviourRefused as refusal:
        raise _refused(refusal) from refusal


@router.delete(f"{BASE}/{{key}}/views/{{dataset_id}}")
async def unlink_behaviour_view(
    key: str, dataset_id: str, request: Request, db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    try:
        return await catalogue.unlink_view(db, key, dataset_id, actor_of(request))
    except catalogue.BehaviourRefused as refusal:
        raise _refused(refusal) from refusal


@router.post(f"{BASE}/{{key}}/archive")
async def archive_behaviour(
    key: str, request: Request, body: Optional[BehaviourArchive] = None, db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    try:
        return await catalogue.set_archived(
            db, key, True, actor_of(request), (body or BehaviourArchive()).reason
        )
    except catalogue.BehaviourRefused as refusal:
        raise _refused(refusal) from refusal


@router.post(f"{BASE}/{{key}}/unarchive")
async def unarchive_behaviour(
    key: str, request: Request, db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    try:
        return await catalogue.set_archived(db, key, False, actor_of(request))
    except catalogue.BehaviourRefused as refusal:
        raise _refused(refusal) from refusal


@router.get(f"{BASE}/{{key}}/checklist")
async def behaviour_checklist_route(key: str, db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """FR-12: the eight steps from definition to armed, each read from real records, plus the
    computed status. "unknown" (miLLM not connected or cannot say) is never reported as "not done"."""
    from ....services import behaviour_checklist

    try:
        return await behaviour_checklist.checklist(db, key)
    except catalogue.BehaviourRefused as refusal:
        raise _refused(refusal) from refusal


@router.get(f"{BASE}/{{key}}/family")
async def get_probe_family(
    key: str,
    model: Optional[str] = Query(None, description="The model whose members to list; required when the behaviour has probes on more than one"),
    member_ids: Optional[List[str]] = Query(None, description="Explicit members instead of the default family (any role, one model)"),
    include_retired: bool = Query(False),
    offset: int = Query(0, ge=0),
    limit: Optional[int] = Query(None, ge=1, description="Members per page (default and most: the runtime setting blend_family_page_size)"),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """Feature 041 FR-2 – FR-4: the behaviour's probe family on one model — members with their facts and
    flags (against the majority of all members), sub-question coverage with the training `next`, and the
    default blend set. Computed on request; nothing is stored."""
    from ....services import probe_family

    try:
        return await probe_family.family(db, key, model=model, member_ids=member_ids,
                                         include_retired=include_retired, offset=offset, limit=limit)
    except catalogue.BehaviourRefused as refusal:
        raise _refused(refusal) from refusal
    except DetectorRefused as refusal:
        raise refusal.to_http() from refusal


@router.get(f"{BASE}/{{key}}/history")
async def behaviour_history(
    key: str,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    return await catalogue.history(db, key, limit=limit, offset=offset)


@router.post("/datasets/{dataset_id}/confirm-positive")
async def confirm_positive_labels(
    dataset_id: str, body: PositiveLabelsConfirm, request: Request, db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """D48: the OPERATOR confirms that this view's positive labels are behaviour `key` (TI5).

    Refused for agents (403): a probe can be made primary only on a confirmed training view, so an agent
    confirming its own data would remove the one check a person makes.
    """
    if actor_of(request) != "operator":
        raise HTTPException(
            status_code=403,
            detail={
                "code": "confirm_operator_only",
                "message": (
                    "Confirming what a view's positive labels mean is the operator's, on miStudio's "
                    "Probes screen. An agent can read the catalogue and propose roles."
                ),
            },
        )
    try:
        return await catalogue.confirm_positive(db, dataset_id, body.key, "operator")
    except catalogue.BehaviourRefused as refusal:
        raise _refused(refusal) from refusal


# ── Feature 041 S2: the suggested blend (FR-21) ───────────────────────────────────────────────


async def _suggested_envelope(db: AsyncSession, key: str, model: str) -> Dict[str, Any]:
    import json

    from ....services.probe_blend import service as blends
    from ....services import detector_next as nx

    current, reasons = await blends.suggested(db, key, model)
    return {"key": key, "model": model, "report_id": current.report_id, "candidate": current.candidate,
            "record": json.loads(current.record), "record_sha256": current.record_sha256,
            "history_id": current.history_id,
            "updated_at": current.updated_at.isoformat() if current.updated_at else None,
            "stale": bool(reasons), "stale_reasons": reasons,
            "next": [nx.next_rerun_report(current.report_id)] if reasons else []}


@router.get(f"{BASE}/{{key}}/suggested-blend")
async def get_suggested_blend(key: str, model: str = Query(...), db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """The current suggested blend: the record (advisory; binds nothing), its sha256, and `stale` computed
    on read from its report."""
    try:
        return await _suggested_envelope(db, key, model)
    except DetectorRefused as refusal:
        raise refusal.to_http() from refusal


@router.put(f"{BASE}/{{key}}/suggested-blend")
async def set_suggested_blend(key: str, body: SuggestedBlendMark, request: Request,
                              db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """Mark a candidate of a done, current report as the suggested blend: the record is SLICED from the
    stored report; who, when and why go to the append-only history."""
    from ....services.probe_blend import service as blends

    try:
        await blends.mark_suggested(db, key, body.model, body.report_id, body.candidate, actor_of(request), body.reason)
        return await _suggested_envelope(db, key, body.model)
    except catalogue.BehaviourRefused as refusal:
        raise _refused(refusal) from refusal
    except DetectorRefused as refusal:
        raise refusal.to_http() from refusal


@router.get(f"{BASE}/{{key}}/suggested-blend/history")
async def suggested_blend_history(
    key: str, model: str = Query(...), offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    from ....services.probe_blend import service as blends

    return await blends.suggested_history(db, key, model, offset, limit)


# ── Feature 041 S3: family scoring, export and publish (FR-22 – FR-24) ────────────────────────


async def _family_members(db: AsyncSession, key: str, model: Optional[str], member_ids: Optional[List[str]]):
    """The members a family action takes: the named ones, else the default blend set; with their runs."""
    from ....models.probe_monitor import ProbeMonitor, ProbeMonitorRun
    from ....services import probe_family

    resolved = await probe_family.resolve_members(db, key, model=model, member_ids=member_ids)
    ids = [m["probe_id"] for m in resolved["members"]
           if member_ids is not None or m["probe_id"] in resolved["default_blend_set"]]
    from sqlalchemy import select

    probes = {p.id: p for p in (await db.execute(select(ProbeMonitor).where(ProbeMonitor.id.in_(ids or [""])))).scalars()}
    members = [probes[i] for i in ids if i in probes]
    runs = {r.id: r for r in (await db.execute(select(ProbeMonitorRun).where(
        ProbeMonitorRun.id.in_([p.run_id for p in members] or [""])))).scalars()}
    return resolved["model"], members, runs


@router.post(f"{BASE}/{{key}}/family/score-batch", status_code=202)
async def score_probe_family(key: str, body: FamilyScoreBatch, db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """FR-22: one queued task PER INPUT, each scoring every member with one model load. ⚠ THE CAP COUNTS
    MEMBERS × INPUTS (each is a forward pass), against the one setting the single batch reads."""
    import json

    from ....core.runtime_settings import effective
    from ....services.gpu_dispatch import gpu_delay
    from ....services.probe_blend import service as blends
    from ....workers.probe_monitor_tasks import score_probe_family as score_family_task

    try:
        member_ids = body.member_ids
        report = None
        if body.blend_report_id:
            row = await blends.get_row(db, body.blend_report_id)
            if row.behaviour_key != key:
                raise DetectorRefused(422, "report_not_of_behaviour", f"report {row.id} is not of {key!r}",
                                      {"report_behaviour": row.behaviour_key})
            report = json.loads(blends.report_bytes(row))
            for name in body.candidates or []:
                blends.candidate_of(report, name)
            member_ids = member_ids or [m["probe_id"] for m in report["members"]]
        model, members, _runs = await _family_members(db, key, body.model, member_ids)
    except catalogue.BehaviourRefused as refusal:
        raise _refused(refusal) from refusal
    except DetectorRefused as refusal:
        raise refusal.to_http() from refusal
    cap = int(effective().probe_score_batch_cap)
    product = len(members) * len(body.inputs)
    if product > cap:
        raise HTTPException(status_code=422, detail={
            "code": "cap_exceeded", "message": f"{len(members)} members x {len(body.inputs)} inputs = {product} scores; "
                                               f"probe_score_batch_cap allows {cap}",
            "members": len(members), "inputs": len(body.inputs), "product": product,
            "setting": "probe_score_batch_cap", "value": cap, "next": []})
    tasks = []
    for index, item in enumerate(body.inputs):
        queued = gpu_delay(score_family_task, "auto")(
            behaviour_key=key, member_ids=[m.id for m in members], text=item.text, messages=item.messages,
            blend_report_id=body.blend_report_id, candidates=body.candidates, include_traces=body.include_traces)
        tasks.append({"index": index, "task_id": queued.id})
    return {"key": key, "model": model, "members": [m.id for m in members], "tasks": tasks, "submitted": len(tasks)}


@router.get(f"{BASE}/{{key}}/family/score/{{task_id}}")
async def get_probe_family_score_result(key: str, task_id: str) -> Dict[str, Any]:
    """Collect a family score. 404 when the finished task's result belongs to another behaviour (PT-8)."""
    import asyncio

    from ....core.celery_app import celery_app

    def _read():
        result = celery_app.AsyncResult(task_id)
        return result.state, result.info

    state, info = await asyncio.to_thread(_read)
    if state == "SUCCESS":
        if not isinstance(info, dict) or info.get("behaviour_key") != key:
            raise HTTPException(status_code=404, detail={"code": "not_this_behaviours_task", "next": [],
                                                         "message": f"task {task_id} is not a family score of {key!r}"})
        return {"task_id": task_id, "key": key, "status": state, "result": info}
    if state == "FAILURE":
        return {"task_id": task_id, "key": key, "status": state, "error": str(info), "result": None}
    return {"task_id": task_id, "key": key, "status": state, "result": None}


async def _export_entries(db: AsyncSession, key: str, model: Optional[str], member_ids: Optional[List[str]]):
    from starlette.concurrency import run_in_threadpool

    from ....services import probe_family_export as export

    model, members, runs = await _family_members(db, key, model, member_ids)
    export.refuse_if_blocked(members, runs)
    return model, await run_in_threadpool(export.files, members)


@router.get(f"{BASE}/{{key}}/family/export-index")
async def export_probe_family(
    key: str, model: Optional[str] = Query(None), member_ids: Optional[List[str]] = Query(None),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """FR-23: every member's built definition — single-export route, sha256 and size — and the archive's route;
    ALL OR NOTHING: one blocker refuses the whole export, every blocker listed with its `next`."""
    from urllib.parse import urlencode

    from ....services import probe_family_export as export

    try:
        model, entries = await _export_entries(db, key, model, member_ids)
    except catalogue.BehaviourRefused as refusal:
        raise _refused(refusal) from refusal
    except DetectorRefused as refusal:
        raise refusal.to_http() from refusal
    query = urlencode([("model", model)] + [("member_ids", i) for i in (member_ids or [])])
    out = export.index(entries, f"/api/v1/probe-monitors/behaviours/{key}/family/export?{query}")
    return {"key": key, "model": model, **out}


@router.get(f"{BASE}/{{key}}/family/export")
async def export_probe_family_archive(
    key: str, model: Optional[str] = Query(None), member_ids: Optional[List[str]] = Query(None),
    db: AsyncSession = Depends(get_db),
) -> Response:
    """T13: a ZIP of every member's UNTOUCHED built bytes (stored, sorted, fixed timestamps — deterministic)."""
    from ....services import probe_family_export as export

    try:
        model, entries = await _export_entries(db, key, model, member_ids)
    except catalogue.BehaviourRefused as refusal:
        raise _refused(refusal) from refusal
    except DetectorRefused as refusal:
        raise refusal.to_http() from refusal
    return Response(content=export.archive(entries), media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{key}-family.zip"'})


@router.post(f"{BASE}/{{key}}/family/publish", status_code=202)
async def publish_probe_family(key: str, body: FamilyPublish, db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """FR-24: one Hugging Face commit with every member. All or nothing (the export's blockers, including two
    members whose definition names collide), and the token is checked BEFORE anything is queued."""
    from ....services import probe_family_export as export
    from ....services.huggingface_sae_service import resolve_hf_token
    from ....workers.probe_monitor_tasks import publish_probe_family as publish_family_task
    from .probe_monitors import _huggingface_rejects

    try:
        model, members, runs = await _family_members(db, key, body.model, body.member_ids)
        export.refuse_if_blocked(members, runs)
    except catalogue.BehaviourRefused as refusal:
        raise _refused(refusal) from refusal
    except DetectorRefused as refusal:
        raise refusal.to_http() from refusal
    token = resolve_hf_token(body.token)
    if not token:
        raise HTTPException(status_code=401, detail={"code": "no_token", "next": [], "message": (
            "no Hugging Face token: pass one or store it in Settings → API Keys (checked before anything is queued)")})
    rejected = _huggingface_rejects(token)
    if rejected:
        raise HTTPException(status_code=401, detail={"code": "token_rejected", "message": rejected, "next": []})
    queued = publish_family_task.delay(behaviour_key=key, member_ids=[m.id for m in members], repo_id=body.repo_id,
                        private=body.private, token=token)
    return {"key": key, "model": model, "members": [m.id for m in members], "task_id": queued.id, "status": "queued"}
