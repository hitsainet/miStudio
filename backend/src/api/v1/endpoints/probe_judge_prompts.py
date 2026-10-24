"""REST surface for the judge-prompt library (operator decisions D1/D2, 2026-10-21; PADR IDL-57).

Every rule lives in `services/judge_prompt_library.py`; these routes translate its `PromptRefused`
into an HTTP status with ``{code, message}``, so the UI and the MCP tools read the same refusal.

Also here, because it belongs to the rung-3 match and not to training: setting what a PROBE
detects (`PATCH /probe-monitors/probes/{id}/behaviour`), which recomputes its rung at once.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from ....core.deps import get_db
from ....models.probe_monitor import ProbeMonitor
from ....schemas.judge_prompt import (
    JudgePromptChanges,
    JudgePromptClone,
    JudgePromptCreate,
    JudgePromptImport,
    JudgePromptPreview,
    JudgePromptTry,
    RubricImport,
)
from ....schemas.probe_monitor import ProbeBehaviourUpdate
from ....services import behaviour_catalogue
from ....services import judge_prompt_library as library
from ....services.probe_dataset_lifecycle import actor_of
from ....services.judge_prompt_forms import TEMPLATE_PLACEHOLDERS, reply_shape_warnings

logger = logging.getLogger(__name__)

from ._probe_ids import refuse_blend_report_ids  # noqa: E402

# Feature 041 FR-18: `PATCH /probes/{probe_id}/behaviour` lives here, so this router refuses a blend
# report id too. Removing the dependency fails test_not_a_probe.py.
router = APIRouter(prefix="/probe-monitors", tags=["probe-judge-prompts"],
                   dependencies=[Depends(refuse_blend_report_ids)])

BASE = "/judge-prompts"


def _refused(refusal: library.PromptRefused) -> HTTPException:
    return HTTPException(status_code=refusal.status, detail=refusal.as_detail())


async def _out(db: AsyncSession, prompt_id: str) -> Dict[str, Any]:
    return await library.detail(db, prompt_id)


@router.get(BASE)
async def list_judge_prompts(
    behaviour: Optional[str] = Query(None, description="Only prompts that judge this behaviour"),
    name: Optional[str] = Query(None, description="Only the versions of this prompt name"),
    include_archived: bool = Query(False),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """The library, newest version first within each name, with "used by N runs" on each."""
    return {
        "prompts": await library.list_prompts(
            db, behaviour=behaviour, name=name, include_archived=include_archived
        ),
        # What a template may name, so the editor and an agent never guess.
        "placeholders": TEMPLATE_PLACEHOLDERS,
    }


@router.post(BASE, status_code=201)
async def create_judge_prompt(
    request: JudgePromptCreate, db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    try:
        row = await library.create_prompt(db, request.model_dump(), request.actor)
    except library.PromptRefused as refusal:
        raise _refused(refusal) from refusal
    return await _out(db, row.id)


@router.post(f"{BASE}/preview")
async def preview_judge_prompt(request: JudgePromptPreview, db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """The judge's exact messages for a few rows of a view. Nothing is sent to any model."""
    try:
        spec = await _spec_for(db, request.prompt_id, request.draft)
        rows = await run_in_threadpool(_rows, request.dataset_id, request.rows)
    except library.PromptRefused as refusal:
        raise _refused(refusal) from refusal
    interactions, labels, view_name = rows
    return {
        "prompt_key": spec.key,
        "content_hash": spec.content_hash,
        "dataset_name": view_name,
        "system_prompt": spec.system_prompt,
        "warnings": reply_shape_warnings(spec.system_prompt, spec.user_template, spec.answer_form),
        "rows": [
            {"label": "positive" if label == 1 else "negative", "user_message": spec.render_user(m)}
            for m, label in zip(interactions, labels)
        ],
    }


@router.post(f"{BASE}/try")
async def try_judge_prompt(request: JudgePromptTry, db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """A few rows through the judge, with each reply and how it was read.

    ⚠ NEVER LOADS A MODEL. A judge run asks miLLM to load its model when it is not loaded; a quick
    check must not evict whatever miLLM is serving, so this calls the endpoint as it is and reports
    the error if the model is not there. It records nothing and does not lock the prompt.
    """
    from ....utils.url_validation import validate_llm_endpoint_url

    try:
        endpoint = validate_llm_endpoint_url(request.endpoint)
    except ValueError as bad:
        raise HTTPException(status_code=422, detail={"code": "invalid_endpoint", "message": str(bad)})
    try:
        spec = await _spec_for(db, request.prompt_id, request.draft)
        rows = await run_in_threadpool(_rows, request.dataset_id, request.rows)
    except library.PromptRefused as refusal:
        raise _refused(refusal) from refusal
    interactions, labels, view_name = rows
    result = await run_in_threadpool(_try_rows, endpoint, request.model, spec, interactions, labels)
    result.update({"prompt_key": spec.key, "dataset_name": view_name, "model": request.model})
    return result


@router.get(f"{BASE}/midataworks/rubrics")
async def list_midataworks_rubrics() -> Dict[str, Any]:
    """miDataworks' rubrics, when miDataworks is connected; a 503 saying so when it is not."""
    try:
        return {"rubrics": await library.list_midataworks_rubrics()}
    except library.PromptRefused as refusal:
        raise _refused(refusal) from refusal


@router.post(f"{BASE}/import-rubric", status_code=201)
async def import_rubric(request: RubricImport, db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """A miDataworks rubric as a new library prompt (version 1 of `name`)."""
    try:
        source_ref: Dict[str, Any]
        if request.rubric_id is not None:
            document, record = await library.fetch_midataworks_rubric(request.rubric_id)
            source_ref = {
                "rubric_id": record.get("id"),
                "ref": record.get("ref"),
                "content_hash": record.get("content_hash"),
                "fetched_from": library.midataworks_url(),
            }
        else:
            document = dict(request.document or {})
            source_ref = {"file": True}
        source_ref.update(
            {"format": document.get("format"), "name": document.get("name"),
             "version": document.get("version"), "style": (document.get("body") or {}).get("style")}
        )
        payload = library.rubric_to_prompt(
            document,
            name=request.name,
            title=request.title,
            behaviour=request.behaviour,
            field_map=request.field_map,
            literals=request.literals,
            positive_verdicts=request.positive_verdicts,
            positive_from=request.positive_from,
        )
        payload["behaviour_label"] = request.behaviour_label
        row = await library.create_prompt(db, payload, request.actor)
        row.source = "midataworks"
        row.source_ref = source_ref
        await db.commit()
    except library.PromptRefused as refusal:
        raise _refused(refusal) from refusal
    return await _out(db, row.id)


@router.post(f"{BASE}/import", status_code=201)
async def import_judge_prompt(request: JudgePromptImport, db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """A `mistudio.judge-prompt/v1` document, stored under its own name and version."""
    try:
        row, created = await library.import_document(db, request.document, request.actor)
    except library.PromptRefused as refusal:
        raise _refused(refusal) from refusal
    out = await _out(db, row.id)
    out["created"] = created
    return out


@router.get(f"{BASE}/{{prompt_id}}")
async def get_judge_prompt(prompt_id: str, db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """One version, with every version of its name and how many runs used each."""
    try:
        return await _out(db, prompt_id)
    except library.PromptRefused as refusal:
        raise _refused(refusal) from refusal


@router.patch(f"{BASE}/{{prompt_id}}")
async def update_judge_prompt(
    prompt_id: str, request: JudgePromptChanges, db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Edit in place: title/description always; text, answer form and behaviour only while no
    run has used it (409 `prompt_locked` otherwise — make a new version)."""
    try:
        row = await library.update_prompt(
            db, prompt_id, request.model_dump(exclude={"actor"}), request.actor
        )
    except library.PromptRefused as refusal:
        raise _refused(refusal) from refusal
    return await _out(db, row.id)


@router.post(f"{BASE}/{{prompt_id}}/versions", status_code=201)
async def new_judge_prompt_version(
    prompt_id: str, request: JudgePromptChanges, db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    try:
        row = await library.new_version(
            db, prompt_id, request.model_dump(exclude={"actor"}), request.actor
        )
    except library.PromptRefused as refusal:
        raise _refused(refusal) from refusal
    return await _out(db, row.id)


@router.post(f"{BASE}/{{prompt_id}}/clone", status_code=201)
async def clone_judge_prompt(
    prompt_id: str, request: JudgePromptClone, db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    try:
        row = await library.clone_prompt(
            db, prompt_id, request.model_dump(exclude={"actor"}), request.actor
        )
    except library.PromptRefused as refusal:
        raise _refused(refusal) from refusal
    return await _out(db, row.id)


@router.post(f"{BASE}/{{prompt_id}}/archive")
async def archive_judge_prompt(prompt_id: str, db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    try:
        await library.set_archived(db, prompt_id, True)
    except library.PromptRefused as refusal:
        raise _refused(refusal) from refusal
    return await _out(db, prompt_id)


@router.post(f"{BASE}/{{prompt_id}}/unarchive")
async def unarchive_judge_prompt(prompt_id: str, db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    try:
        await library.set_archived(db, prompt_id, False)
    except library.PromptRefused as refusal:
        raise _refused(refusal) from refusal
    return await _out(db, prompt_id)


@router.delete(f"{BASE}/{{prompt_id}}", status_code=204)
async def delete_judge_prompt(prompt_id: str, db: AsyncSession = Depends(get_db)) -> None:
    """Only a version no run has used (409 otherwise — archive it instead)."""
    try:
        await library.delete_prompt(db, prompt_id)
    except library.PromptRefused as refusal:
        raise _refused(refusal) from refusal


@router.get(f"{BASE}/{{prompt_id}}/export")
async def export_judge_prompt(prompt_id: str, db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    try:
        return library.export_document(await library.get_prompt(db, prompt_id))
    except library.PromptRefused as refusal:
        raise _refused(refusal) from refusal


@router.patch("/probes/{probe_id}/behaviour")
async def set_probe_behaviour(
    probe_id: str,
    request: ProbeBehaviourUpdate,
    http_request: Request,
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """Say what a probe detects (or clear it), then recompute its rung — the judge runs that
    count toward rung 3 are exactly those whose prompt judges this behaviour."""
    from ....services.probe_monitor_run import recompute_rung

    probe = (
        await db.execute(select(ProbeMonitor).where(ProbeMonitor.id == probe_id))
    ).scalar_one_or_none()
    if probe is None:
        raise HTTPException(status_code=404, detail=f"Probe {probe_id} not found")
    old_behaviour = probe.behaviour
    probe.behaviour = request.behaviour
    probe.behaviour_label = request.behaviour_label if request.behaviour else None
    # Feature 037 (FR-8): the catalogue's append-only history records the move, in this transaction.
    # Removing this line fails test_behaviour_catalogue.py::test_probe_behaviour_change_is_recorded.
    await behaviour_catalogue.record_probe_key_change(
        db, probe_id, old_behaviour, request.behaviour, actor_of(http_request)
    )
    await db.commit()

    def _recompute() -> Dict[str, Any]:
        from ....core.database import SyncSessionLocal

        session = SyncSessionLocal()
        try:
            return recompute_rung(session, probe_id)
        finally:
            session.close()

    rung = await run_in_threadpool(_recompute)
    return {
        "probe_id": probe_id,
        "behaviour": request.behaviour,
        "behaviour_label": probe.behaviour_label,
        "rung": rung,
    }


# ── helpers ──────────────────────────────────────────────────────────────────────────────────


async def _spec_for(db: AsyncSession, prompt_id: Optional[str], draft: Any):
    from ....services.judge_prompt_forms import content_hash
    from ....services.probe_monitor_judge import JudgePromptSpec, spec_from_row

    if prompt_id is not None:
        return spec_from_row(await library.get_prompt(db, prompt_id))
    content = library.validated_content(
        name=draft.name,
        behaviour=draft.behaviour,
        system_prompt=draft.system_prompt,
        user_template=draft.user_template,
        answer_form=draft.answer_form,
        max_interaction_chars=draft.max_interaction_chars,
    )
    return JudgePromptSpec(
        key=f"{draft.name} (unsaved draft)",
        content_hash=content_hash(name=draft.name, version=1, **content),
        behaviour=draft.behaviour,
        **content,
    )


def _rows(dataset_id: str, n: int):
    """A balanced, seeded sample of `n` rows of a view — the reader the judge run itself uses."""
    from ....core.database import SyncSessionLocal
    from ....models.probe_monitor import ProbeMonitorDataset
    from ....services.probe_monitor_judge import _interactions_for

    session = SyncSessionLocal()
    try:
        view = session.query(ProbeMonitorDataset).filter(ProbeMonitorDataset.id == dataset_id).first()
        if view is None:
            raise library.PromptRefused(404, "dataset_not_found", f"probe dataset {dataset_id} not found")
        if getattr(view, "archived_at", None) is not None:
            # Archived views are out of use for NEW work (runs, evaluations, judge baselines), and
            # a preview or try is the first step of new work. Same refusal and code as those.
            raise library.PromptRefused(
                409, "probe_dataset_archived",
                f"'{view.name}' ({view.id}) is archived. Unarchive it on the Probes screen "
                f"(Datasets, Show archived) before using it again.",
            )
        try:
            interactions, labels = _interactions_for(session, view, limit=n)
        except (ValueError, FileNotFoundError) as exc:
            raise library.PromptRefused(
                422, "rows_unreadable", f"the rows of {view.name} could not be read: {exc}"
            ) from exc
        return interactions[:n], labels[:n], view.name
    finally:
        session.close()


def _try_rows(endpoint: str, model: str, spec: Any, interactions: List[Any], labels: List[int]) -> Dict[str, Any]:
    from ....core.database import SyncSessionLocal
    from ....services.labeling_judge_health import is_job_level_failure
    from ....services.probe_monitor_judge import _client_for, ask_judge, resolve_api_key

    session = SyncSessionLocal()
    try:
        client = _client_for(endpoint, resolve_api_key(session))
    finally:
        session.close()
    rows: List[Dict[str, Any]] = []
    stopped: Optional[str] = None
    for messages, label in zip(interactions, labels):
        entry: Dict[str, Any] = {
            "label": "positive" if label == 1 else "negative",
            "user_message": spec.render_user(messages),
        }
        try:
            reply = ask_judge(client, model, spec, messages)
        except Exception as exc:  # noqa: BLE001 - reported per row
            detail = f"{type(exc).__name__}: {exc}"[:300]
            entry.update({"reply": None, "answer": None, "error": detail})
            rows.append(entry)
            if is_job_level_failure(detail):
                stopped = f"the judge endpoint is failing ({detail}); stopped after {len(rows)} row(s)"
                break
            continue
        answer = spec.parse(reply)
        entry.update(
            {
                "reply": (reply or "")[:2000],
                "answer": (
                    {"value": answer.value, "positive": answer.positive, "score": answer.score}
                    if answer is not None
                    else None
                ),
                "error": None if answer is not None else "the reply could not be read under this answer form",
                "agrees_with_label": (answer.positive == (label == 1)) if answer is not None else None,
            }
        )
        rows.append(entry)
    return {
        "rows": rows,
        "parsed": sum(1 for r in rows if r.get("answer")),
        "unparseable": sum(1 for r in rows if r.get("answer") is None),
        "stopped": stopped,
    }
