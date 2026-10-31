"""Archive, unarchive and purge probe dataset VIEWS (operator decision 2026-10-21).

The suite's pattern (miForge and miDataworks, shipped 2026-10-20), rebuilt in miStudio's idiom:

* **ARCHIVE hides; nothing is deleted.** Reversible. Anyone may archive — the operator from the
  UI, an agent through MCP (`archived_by` says which).
* **PURGE is permanent and the operator's alone.** Only an ARCHIVED view can be purged, and a
  purge is refused with EVERY blocker named when anything still uses the view. Each purge writes
  an append-only tombstone (`ProbeMonitorDatasetPurge`) with who, why and the view as it was.
  Agents never purge: every purge route refuses the MCP server's `X-MiStudio-Agent` header, and
  no MCP tool reaches a purge route.

WHAT A VIEW PURGE REMOVES: one `probe_monitor_datasets` row. A view owns no files (IDL-51: it is a
label mapping over a download, not a copy).

WHAT IT DOES NOT REMOVE — THE DOWNLOADED DATASET, deliberately. The download is a separate asset
with its own screen (Datasets) and its own delete: it can carry tokenizations, feed extractions
and SAE trainings, and serve other views, none of which a probe-view purge can see whole. A view
purge that reached into acquisition would turn "remove a label mapping" into "delete gigabytes",
which is not what the operator asked for when they ticked a view. The plan reports the download
under `kept`, with how many other views still read it, so the operator can delete it from the
Datasets screen if that is what they want.

BLOCKERS (each is its own code, all are reported together):
  run_in_progress       a pending/running probe run names the view (train, eval or calibration)
  run_uses_it           a probe run that still exists names it — its FK would CASCADE the run
                        and its probes away (train) or silently null its provenance (calibration)
  probe_calibrated_on_it  a probe's threshold was cut on it (outside a run already listed)
  evaluation_on_it      a probe was evaluated on it (outside a run already listed)
  judge_run_in_progress / judge_run_uses_it   an LLM-judge baseline names it
  approval_names_it     a pending or approved, unconsumed agent approval carries its id
  behaviour_link        a catalogued behaviour links the view explicitly (Feature 037) — its FK
                        would CASCADE the link away; unlink it on the behaviour first
  unchecked_reference   a foreign key into probe_monitor_datasets that this module does not
                        know how to check — fail closed until it is taught (registry-derived)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from fastapi import HTTPException, Request
from sqlalchemy import String, cast, delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.clock import utc_now
from ..core.database import Base
from ..models.agent_approval import AgentApprovalRequest
from ..models.dataset import Dataset
from ..models.probe_monitor import (
    ProbeMonitor,
    ProbeMonitorDataset,
    ProbeMonitorDatasetPurge,
    ProbeMonitorEvaluation,
    ProbeMonitorJudgeRun,
    ProbeMonitorRun,
)

#: The header the miStudio MCP server sends on every request (`mcp_server/client.py`).
AGENT_HEADER = "X-MiStudio-Agent"
_AGENT_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,47}$")
OPERATOR = "operator"

MAX_REASON = 2000
MAX_BATCH = 200

ACTIVE_STATUSES = ("pending", "running")

#: Every foreign key INTO probe_monitor_datasets that `blockers()` checks, as (table, column).
#: `test_probe_dataset_lifecycle.py` compares this with the live ORM registry, and a key the
#: registry has and this set lacks is ALSO reported at runtime as `unchecked_reference`.
CHECKED_FOREIGN_KEYS: Set[Tuple[str, str]] = {
    ("probe_monitor_runs", "train_dataset_id"),
    ("probe_monitor_runs", "calibration_dataset_id"),
    ("probe_monitors", "calibration_dataset_id"),
    ("probe_monitor_evaluations", "dataset_id"),
    # Feature 037: an explicit behaviour → view link (`behaviour_link`).
    ("probe_behaviour_view_links", "dataset_id"),
}

#: References with no foreign key (ids inside JSON), also checked by `blockers()`.
CHECKED_JSON_REFERENCES: Set[Tuple[str, str]] = {
    ("probe_monitor_runs", "eval_dataset_ids"),
    ("probe_monitor_judge_runs", "dataset_ids"),
    ("agent_approval_requests", "payload"),
}

ROLE_WORDS = {"train": "training set", "eval": "evaluation set", "calibration": "calibration set",
              "blend_fit": "blend-fitting set", "everyday_text": "everyday-text set",
              "nonsense_text": "nonsense-text set"}


def registry_foreign_keys() -> Set[Tuple[str, str]]:
    """(table, column) of every FK the ORM declares into probe_monitor_datasets."""
    found: Set[Tuple[str, str]] = set()
    for table in Base.metadata.tables.values():
        for fk in table.foreign_keys:
            if fk.column.table.name == ProbeMonitorDataset.__tablename__:
                found.add((table.name, fk.parent.name))
    return found


# ── who is asking ─────────────────────────────────────────────────────────────


def actor_of(request: Request) -> str:
    """"operator" for the UI, "agent:<name>" when the MCP server asked. A malformed header is a
    400 rather than being read as the operator — guessing the operator is the one wrong answer."""
    raw = request.headers.get(AGENT_HEADER)
    if raw is None:
        return OPERATOR
    name = raw.strip().lower()
    if not _AGENT_NAME.match(name):
        raise HTTPException(
            status_code=400,
            detail={
                "code": "agent_header_invalid",
                "message": f"{AGENT_HEADER} must be a short lowercase name, e.g. 'mcp'",
            },
        )
    return f"agent:{name}"


def refuse_agents(request: Request) -> None:
    """FastAPI dependency on every purge route: permanent delete is the operator's alone."""
    if request.headers.get(AGENT_HEADER) is not None:
        raise HTTPException(
            status_code=403,
            detail={
                "code": "purge_operator_only",
                "message": (
                    "Permanent delete is the operator's alone, from miStudio's Probes screen. "
                    "Agents can archive and unarchive a probe dataset view; they cannot purge "
                    "one, preview a purge or read the purge log."
                ),
            },
        )


def clean_reason(reason: Optional[str]) -> Optional[str]:
    if reason is None:
        return None
    text = reason.strip()
    if not text:
        return None
    if len(text) > MAX_REASON:
        raise HTTPException(
            status_code=422, detail=f"a reason is at most {MAX_REASON} characters"
        )
    return text


# ── archive ───────────────────────────────────────────────────────────────────


async def _view(db: AsyncSession, view_id: str, *, lock: bool = False) -> ProbeMonitorDataset:
    query = select(ProbeMonitorDataset).where(ProbeMonitorDataset.id == view_id)
    if lock:
        query = query.with_for_update()
    view = (await db.execute(query)).scalar_one_or_none()
    if view is None:
        raise HTTPException(status_code=404, detail=f"Probe dataset {view_id} not found")
    return view


async def archive(
    db: AsyncSession, view_id: str, actor: str, reason: Optional[str]
) -> Tuple[ProbeMonitorDataset, bool]:
    """Hide a view. Idempotent: archiving an archived view changes nothing (its first archive
    record stands). Returns (view, changed)."""
    view = await _view(db, view_id, lock=True)
    if view.archived_at is not None:
        await db.rollback()
        return await _view(db, view_id), False
    view.archived_at = utc_now()
    view.archived_by = actor
    view.archive_reason = clean_reason(reason)
    await db.commit()
    await db.refresh(view)
    return view, True


async def unarchive(db: AsyncSession, view_id: str) -> Tuple[ProbeMonitorDataset, bool]:
    view = await _view(db, view_id, lock=True)
    if view.archived_at is None:
        await db.rollback()
        return await _view(db, view_id), False
    view.archived_at = None
    view.archived_by = None
    view.archive_reason = None
    await db.commit()
    await db.refresh(view)
    return view, True


async def refuse_archived_inputs(db: AsyncSession, view_ids: Sequence[Optional[str]]) -> None:
    """A NEW run, evaluation or judge baseline may not take an archived view as input (409).

    Archive is meant to take a view out of use; letting new work pick it up would also make an
    archived view unpurgeable behind the operator's back. Rows are read FOR SHARE so a purge
    (FOR UPDATE) of the same view serialises against the submit."""
    wanted = [v for v in dict.fromkeys(view_ids) if v]
    if not wanted:
        return
    rows = (
        await db.execute(
            select(ProbeMonitorDataset.id, ProbeMonitorDataset.name)
            .where(ProbeMonitorDataset.id.in_(wanted))
            .where(ProbeMonitorDataset.archived_at.is_not(None))
            .with_for_update(read=True)
        )
    ).all()
    if rows:
        named = ", ".join(f"'{name}' ({vid})" for vid, name in rows)
        raise HTTPException(
            status_code=409,
            detail={
                "code": "probe_dataset_archived",
                "message": (
                    f"{named} {'is' if len(rows) == 1 else 'are'} archived. Unarchive it on the "
                    f"Probes screen (Datasets, Show archived) before using it again."
                ),
                "archived": [vid for vid, _ in rows],
            },
        )


# ── blockers and the plan ─────────────────────────────────────────────────────


@dataclass
class Blocker:
    code: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {"code": self.code, "message": self.message, **self.details}


async def blockers(db: AsyncSession, view_id: str) -> List[Blocker]:
    """Everything that still uses this view, each named. Empty means nothing does."""
    found: List[Blocker] = []

    # Runs: training (FK, CASCADE), calibration (FK, SET NULL) and evaluation (JSON list).
    runs = (
        await db.execute(
            select(ProbeMonitorRun).where(
                (ProbeMonitorRun.train_dataset_id == view_id)
                | (ProbeMonitorRun.calibration_dataset_id == view_id)
                | ProbeMonitorRun.eval_dataset_ids.contains([view_id])
            ).order_by(ProbeMonitorRun.created_at)
        )
    ).scalars().all()
    blocking_runs: Set[str] = set()
    for run in runs:
        roles = []
        if run.train_dataset_id == view_id:
            roles.append("train")
        if view_id in (run.eval_dataset_ids or []):
            roles.append("eval")
        if run.calibration_dataset_id == view_id:
            roles.append("calibration")
        as_what = " and ".join(ROLE_WORDS[r] for r in roles)
        blocking_runs.add(run.id)
        if run.status in ACTIVE_STATUSES:
            found.append(Blocker(
                "run_in_progress",
                f"Probe run {run.id} is {run.status} and uses this view as its {as_what}. "
                f"Wait for it to finish or cancel it.",
                {"run_id": run.id, "roles": roles, "status": run.status},
            ))
        else:
            found.append(Blocker(
                "run_uses_it",
                f"Probe run {run.id} ({run.status}) uses this view as its {as_what}. Delete "
                f"that run on the Runs tab first if it should go too.",
                {"run_id": run.id, "roles": roles, "status": run.status},
            ))

    probes = (
        await db.execute(
            select(ProbeMonitor.id, ProbeMonitor.run_id).where(
                ProbeMonitor.calibration_dataset_id == view_id
            )
        )
    ).all()
    for probe_id, run_id in probes:
        if run_id in blocking_runs:
            continue
        found.append(Blocker(
            "probe_calibrated_on_it",
            f"Probe {probe_id}'s threshold was calibrated on this view.",
            {"probe_id": probe_id, "run_id": run_id},
        ))

    evaluations = (
        await db.execute(
            select(ProbeMonitorEvaluation.probe_id, func.count(), ProbeMonitor.run_id)
            .join(ProbeMonitor, ProbeMonitor.id == ProbeMonitorEvaluation.probe_id)
            .where(ProbeMonitorEvaluation.dataset_id == view_id)
            .group_by(ProbeMonitorEvaluation.probe_id, ProbeMonitor.run_id)
        )
    ).all()
    for probe_id, n, run_id in evaluations:
        if run_id in blocking_runs:
            continue
        found.append(Blocker(
            "evaluation_on_it",
            f"Probe {probe_id} was evaluated on this view ({n} evaluation"
            f"{'' if n == 1 else 's'}); purging the view would delete that evidence.",
            {"probe_id": probe_id, "evaluations": int(n)},
        ))

    judges = (
        await db.execute(
            select(ProbeMonitorJudgeRun.id, ProbeMonitorJudgeRun.status).where(
                ProbeMonitorJudgeRun.dataset_ids.contains([view_id])
            )
        )
    ).all()
    for judge_id, status in judges:
        active = status in ACTIVE_STATUSES
        found.append(Blocker(
            "judge_run_in_progress" if active else "judge_run_uses_it",
            f"Judge baseline {judge_id} ({status}) was measured on this view."
            + (" Wait for it to finish or cancel it." if active else ""),
            {"judge_run_id": judge_id, "status": status},
        ))

    approvals = (
        await db.execute(
            select(AgentApprovalRequest.id, AgentApprovalRequest.tool_name).where(
                AgentApprovalRequest.status.in_(("pending", "approved")),
                AgentApprovalRequest.consumed_at.is_(None),
                cast(AgentApprovalRequest.payload, String).contains(view_id),
            )
        )
    ).all()
    for approval_id, tool in approvals:
        found.append(Blocker(
            "approval_names_it",
            f"Agent request {approval_id} ({tool}) is waiting or approved and names this view. "
            f"Deny it or let it run first.",
            {"approval_id": approval_id, "tool": tool},
        ))

    from ..models.probe_behaviour import ProbeBehaviourViewLink

    links = (
        await db.execute(
            select(ProbeBehaviourViewLink.key).where(ProbeBehaviourViewLink.dataset_id == view_id)
        )
    ).scalars().all()
    for key in links:
        found.append(Blocker(
            "behaviour_link",
            f"The behaviour {key!r} links this view. Unlink it on that behaviour first "
            f"(the catalogue's history keeps the link).",
            {"behaviour": key},
        ))

    for table, column in sorted(registry_foreign_keys() - CHECKED_FOREIGN_KEYS):
        found.append(Blocker(
            "unchecked_reference",
            f"{table}.{column} can reference a probe dataset view and the purge does not know "
            f"how to check it yet, so it refuses rather than guess.",
            {"table": table, "column": column},
        ))
    return found


def not_archived_message(view: ProbeMonitorDataset) -> str:
    return (
        f"'{view.name}' is not archived. Archive it first; only an archived probe dataset view "
        f"can be deleted permanently."
    )


def snapshot(view: ProbeMonitorDataset) -> Dict[str, Any]:
    """The view as the tombstone records it."""
    def iso(value: Any) -> Optional[str]:
        return value.isoformat() if value is not None else None

    return {
        "id": view.id,
        "name": view.name,
        "dataset_id": str(view.dataset_id),
        "config": view.config,
        "split": view.split,
        "input_column": view.input_column,
        "label_column": view.label_column,
        "pair_column": view.pair_column,
        "label_mapping": view.label_mapping,
        "keyword_filter": view.keyword_filter,
        "role": view.role,
        "distribution": view.distribution,
        "counts": view.counts,
        "dataset_version": view.dataset_version,
        "behaviour": view.behaviour,
        "positive_meaning": view.positive_meaning,
        "positive_labels_confirmed": view.positive_labels_confirmed,
        "created_at": iso(view.created_at),
        "archived_at": iso(view.archived_at),
        "archived_by": view.archived_by,
        "archive_reason": view.archive_reason,
    }


async def kept(db: AsyncSession, view: ProbeMonitorDataset) -> List[Dict[str, Any]]:
    """What a purge leaves alone: the downloaded dataset the view reads."""
    dataset = (
        await db.execute(select(Dataset.id, Dataset.name).where(Dataset.id == view.dataset_id))
    ).first()
    others = (
        await db.execute(
            select(func.count()).select_from(ProbeMonitorDataset).where(
                ProbeMonitorDataset.dataset_id == view.dataset_id,
                ProbeMonitorDataset.id != view.id,
            )
        )
    ).scalar_one()
    return [{
        "what": "downloaded dataset",
        "dataset_id": str(view.dataset_id),
        "name": dataset[1] if dataset else None,
        "other_views": int(others),
        "why": (
            "a view is a label mapping over a download, not a copy; the download has its own "
            "delete on the Datasets screen"
            + (f", and {others} other view{'s' if others != 1 else ''} still read it"
               if others else "")
        ),
    }]


async def plan_one(db: AsyncSession, view_id: str) -> Dict[str, Any]:
    view = (
        await db.execute(select(ProbeMonitorDataset).where(ProbeMonitorDataset.id == view_id))
    ).scalar_one_or_none()
    if view is None:
        return {
            "id": view_id,
            "found": False,
            "name": None,
            "purgeable": False,
            "not_archived": None,
            "blockers": [{"code": "not_found", "message": f"Probe dataset {view_id} not found"}],
            "removes": {"rows": {}},
            "kept": [],
        }
    found = [b.as_dict() for b in await blockers(db, view_id)]
    archived = view.archived_at is not None
    return {
        "id": view.id,
        "found": True,
        "name": view.name,
        "role": view.role,
        "archived": archived,
        "archived_at": view.archived_at.isoformat() if archived else None,
        "purgeable": archived and not found,
        "not_archived": None if archived else {
            "code": "not_archived", "message": not_archived_message(view),
        },
        "blockers": found,
        "removes": {"rows": {ProbeMonitorDataset.__tablename__: 1}},
        "kept": await kept(db, view),
    }


async def plan(db: AsyncSession, view_ids: Sequence[str]) -> List[Dict[str, Any]]:
    return [await plan_one(db, view_id) for view_id in dict.fromkeys(view_ids)]


# ── purge ─────────────────────────────────────────────────────────────────────


async def purge_one(
    db: AsyncSession, view_id: str, actor: str, reason: Optional[str]
) -> Dict[str, Any]:
    """Purge one archived view in its own transaction, or raise 404/409 having changed nothing.

    The view row is locked FOR UPDATE before the blockers are read, so a run submitted
    concurrently (which reads the view FOR SHARE, and whose FK takes KEY SHARE) waits for this
    transaction and then fails, instead of slipping in between the check and the delete."""
    reason = clean_reason(reason)
    try:
        view = await _view(db, view_id, lock=True)
        if view.archived_at is None:
            raise HTTPException(
                status_code=409,
                detail={"code": "not_archived", "message": not_archived_message(view)},
            )
        found = await blockers(db, view_id)
        if found:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "purge_blocked",
                    "message": f"'{view.name}' cannot be deleted: "
                    + "; ".join(b.message for b in found),
                    "blockers": [b.as_dict() for b in found],
                },
            )
        record = snapshot(view)
        keep = await kept(db, view)
        from .view_columns import snapshot_files

        owned = snapshot_files(view.id)
        # Feature 040 S3 (FTASKS 20.3): the view's cached rule verdicts go too (recomputable, no FK).
        from .rules_evaluation import view_cache_dir

        if view_cache_dir(view.id).exists():
            owned["rule_verdicts"] = str(view_cache_dir(view.id))
        tombstone = ProbeMonitorDatasetPurge(
            view_id=view.id,
            name=view.name,
            purged_by=actor,
            reason=reason,
            # Feature 040 S2: a view can own files (its column snapshots); recorded here.
            rows_removed={ProbeMonitorDataset.__tablename__: 1, **owned},
            removed=record,
            kept=keep,
        )
        db.add(tombstone)
        await db.flush()
        result = await db.execute(
            delete(ProbeMonitorDataset).where(
                ProbeMonitorDataset.id == view_id,
                ProbeMonitorDataset.archived_at.is_not(None),
            ).execution_options(synchronize_session=False)
        )
        if result.rowcount != 1:
            raise HTTPException(
                status_code=409,
                detail={"code": "purge_raced", "message": f"'{view.name}' changed during the purge; nothing was deleted"},
            )
        from ..models.rules_evaluation import RuleVerdictCache

        await db.execute(delete(RuleVerdictCache).where(RuleVerdictCache.view_id == view_id)
                         .execution_options(synchronize_session=False))
        await db.commit()
        # Feature 040 S2 (FR-14): the view's column snapshots go AFTER the row's delete commits (a failed
        # purge leaves both); the tombstone already names the directory.
        if owned:
            from .view_columns import purge_snapshots

            purge_snapshots(view_id)
        # Feature 040 S3 (20.3): and its cached rule verdicts.
        from .rules_evaluation import purge_view

        purge_view(view_id)
    except BaseException:
        await db.rollback()
        raise
    return {
        "outcome": "purged",
        "id": view_id,
        "name": record["name"],
        "tombstone_id": tombstone.id,
        "purged_by": actor,
        "reason": reason,
        "rows_removed": tombstone.rows_removed,
        "kept": keep,
        "purged_at": tombstone.purged_at.isoformat(),
    }


async def purge_many(
    db: AsyncSession, view_ids: Sequence[str], actor: str, reason: Optional[str]
) -> Dict[str, Any]:
    """Each view purged on its own; a blocked one never stops the others."""
    results: List[Dict[str, Any]] = []
    for view_id in dict.fromkeys(view_ids):
        try:
            results.append(await purge_one(db, view_id, actor, reason))
        except HTTPException as refused:
            detail = refused.detail if isinstance(refused.detail, dict) else {
                "code": "not_found" if refused.status_code == 404 else "refused",
                "message": str(refused.detail),
            }
            results.append({
                "outcome": "blocked",
                "id": view_id,
                "code": detail.get("code"),
                "reason": detail.get("message"),
                "blockers": detail.get("blockers", []),
            })
    purged = sum(1 for r in results if r["outcome"] == "purged")
    return {"results": results, "purged": purged, "blocked": len(results) - purged}


async def purge_log(db: AsyncSession, limit: int = 100) -> List[Dict[str, Any]]:
    rows = (
        await db.execute(
            select(ProbeMonitorDatasetPurge)
            .order_by(ProbeMonitorDatasetPurge.purged_at.desc())
            .limit(limit)
        )
    ).scalars().all()
    return [
        {
            "id": r.id,
            "view_id": r.view_id,
            "name": r.name,
            "purged_by": r.purged_by,
            "reason": r.reason,
            "rows_removed": r.rows_removed,
            "removed": r.removed,
            "kept": r.kept,
            "purged_at": r.purged_at.isoformat() if r.purged_at else None,
        }
        for r in rows
    ]
