"""A behaviour's probe family on one model (Feature 041 S1; FR-2 – FR-4; FTDD §6.1, FTID §3.1).

A **family** is computed on request — there is no table: the probes whose `behaviour` is the key, joined
to their run (for the model) and to their catalogue role (037; a probe with no role row is an implicit
`candidate`). The default family is `primary` + `candidate` on ONE model; `retired` members only on
request; or an explicit `member_ids` list (any role, one model).

What it answers, for any behaviour (no key, sub-question or rule id is written here):

* **members** with their facts (FR-3) — precision, render form, scope, layer, rule, sub-question, rung,
  threshold, the 039 identity (decision revision and digests), the training view's dataset version and
  whether a definition is built;
* **flags** (FR-3), each against the MAJORITY OF ALL MEMBERS — never against the primary: a candidate that
  differs from the family is the fact, whichever member happens to be primary. Flags never drop a member;
* **coverage** (FR-4): every sub-question the behaviour's CURRENT version declares, with its members or
  `missing` and the `next` call that trains one (or registers its training view first). There is no
  orchestration tool (L2): each GPU start is approved on its own.

`flags()` and `coverage()` are pure, so the rules are tested without a database; `family()` is the shell
that reads rows and calls them.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from ..models.probe_behaviour import ProbeBehaviourProbeRole, ProbeBehaviourVersion
from ..models.probe_monitor import ProbeMonitor, ProbeMonitorDataset, ProbeMonitorRun
from . import behaviour_catalogue as catalogue
from .detector_next import (
    DetectorRefused,
    next_get_family,
    next_register_specialist_view,
    next_train_specialist,
)

#: Stable member order: role, then sub-question, then creation, then id (FTID §3.1).
ROLE_RANK = {"primary": 0, "candidate": 1, "retired": 2}

#: The facts a `differs_*` flag compares, and the flag each one raises.
COMPARED_FIELDS = (
    ("precision", "differs_precision"),
    ("render_form", "differs_render"),
    ("scope", "differs_scope"),
    ("training_dataset_version", "differs_dataset_version"),
)


class FamilyRefused(DetectorRefused):
    """A family request refused; the body is data (`code`, the reasons as fields, `next`)."""


# ── reading rows ─────────────────────────────────────────────────────────────────────────────


async def member_rows(
    db: AsyncSession, keys: Sequence[str], *, member_ids: Optional[Sequence[str]] = None
) -> List[Tuple[ProbeMonitor, ProbeMonitorRun, str, bool]]:
    """`(probe, run, role, role_recorded)` for every probe carrying one of `keys` — ONE statement for
    any number of keys (the checklist pages call it for a whole page).

    A role row counts only when it belongs to the probe's own key (D24: a role is the behaviour's)."""
    if not keys:
        return []
    query = (
        select(ProbeMonitor, ProbeMonitorRun, ProbeBehaviourProbeRole.role)
        .join(ProbeMonitorRun, ProbeMonitorRun.id == ProbeMonitor.run_id)
        .outerjoin(
            ProbeBehaviourProbeRole,
            and_(ProbeBehaviourProbeRole.probe_id == ProbeMonitor.id,
                 ProbeBehaviourProbeRole.key == ProbeMonitor.behaviour),
        )
        .where(ProbeMonitor.behaviour.in_(list(keys)))
    )
    if member_ids is not None:
        query = query.where(ProbeMonitor.id.in_(list(member_ids)))
    rows = (await db.execute(query)).all()
    out = [(probe, run, role or "candidate", role is not None) for probe, run, role in rows]
    out.sort(key=lambda r: _order_key(r[0], r[2]))
    return out


def _order_key(probe: ProbeMonitor, role: str) -> Tuple[Any, ...]:
    return (ROLE_RANK.get(role, 9), probe.sub_question or "", probe.created_at.isoformat()
            if probe.created_at is not None else "", probe.id)


def default_members(rows: Iterable[Tuple[ProbeMonitor, ProbeMonitorRun, str, bool]], *,
                    include_retired: bool = False) -> List[Tuple[ProbeMonitor, ProbeMonitorRun, str, bool]]:
    """FR-2: `primary` + `candidate` (implicit candidates included); `retired` only when asked."""
    return [r for r in rows if include_retired or r[2] != "retired"]


def declared_sub_questions(version: Optional[ProbeBehaviourVersion]) -> List[str]:
    """The sub-question keys a version declares, in its order — read from the catalogue, never a constant."""
    if version is None:
        return []
    return [entry.get("key") for entry in (version.sub_questions or []) if entry.get("key")]


# ── facts ────────────────────────────────────────────────────────────────────────────────────


def _precisions(run_ids: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """`probe_precision` for every run, in ONE thread hop with ONE sync session (`probe_precision` is
    sync and may stat the model snapshot)."""
    from ..core.database import SyncSessionLocal
    from .probe_monitor_run import probe_precision

    session = SyncSessionLocal()
    try:
        runs = {r.id: r for r in session.query(ProbeMonitorRun).filter(ProbeMonitorRun.id.in_(list(run_ids)))}
        return {run_id: probe_precision(session, run) for run_id, run in runs.items()}
    finally:
        session.close()


def member_facts(probe: ProbeMonitor, run: ProbeMonitorRun, role: str, role_recorded: bool,
                 views: Dict[str, ProbeMonitorDataset], precision: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """FR-3: one member's facts, read from its rows (no file is read)."""
    from ..schemas.evidence_ladder import probe_rung_language
    from ..schemas.probe_identity import current_revision
    from .probe_breakdown import decision_inputs
    from .probe_decision import identity_fields
    from .probe_monitor_render import render_form_status

    identity = identity_fields(probe)
    view = views.get(run.train_dataset_id)
    dataset_version = None
    if view is not None and isinstance(view.dataset_version, dict):
        dataset_version = view.dataset_version.get("version_id")
    return {
        "probe_id": probe.id,
        "run_id": run.id,
        "model_id": run.model_id,
        "role": role,
        "role_recorded": role_recorded,
        "precision": (precision or {}).get("recorded"),
        "precision_refusal": (precision or {}).get("refusal"),
        "render_form": render_form_status(probe.render_form),
        "scope": str((run.config or {}).get("scope", "all")),
        "layer": probe.layer,
        "rule": probe.rule,
        "variant": probe.variant,
        "sub_question": probe.sub_question,
        "behaviour_version": probe.behaviour_version,
        "rung": int(probe.rung or 0),
        # The server's wording of the rung (never composed by a client; `RungChip` shows it).
        "rung_language": probe_rung_language(int(probe.rung or 0)),
        "threshold": probe.threshold,
        "decision_revision": identity["decision_revision"],
        "decision_digest": probe.decision_digest,
        # The revision of the decision IN FORCE (039 `current_revision`), which a blend records and a stale
        # check compares — `decision_revision` above is the BUILT definition's.
        "current_decision_revision": current_revision(probe.calibration_history),
        "detector_digest": identity["detector_digest"],
        "identity_current": identity["identity_current"],
        "identity_stale_reason": identity["identity_stale_reason"],
        "training_view": run.train_dataset_id,
        "training_dataset_version": dataset_version,
        "definition_built": probe.definition_path is not None,
        "created_at": probe.created_at.isoformat() if probe.created_at is not None else None,
        # The decision in force, as 040's `decide_many` takes it (a blend records it at start).
        "decision": decision_inputs(probe, str((run.config or {}).get("scope", "all"))),
    }


def _same(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def flags(members: Sequence[Dict[str, Any]], *, declared: Sequence[str],
          version: Optional[int]) -> Dict[str, List[Dict[str, Any]]]:
    """FR-3: `{probe_id: [flag…]}` — PURE. Each `differs_*` flag compares a member with the MAJORITY OF
    ALL MEMBERS (ties: the value of the member first in stable order, marked `majority_tie`), naming the
    member's value and the majority's. `definition_stale`: no built definition, or 039's
    `identity_current` is false. `sub_question_not_declared`: the member's sub-question is not declared by
    the current version (named)."""
    out: Dict[str, List[Dict[str, Any]]] = {m["probe_id"]: [] for m in members}
    for field, name in COMPARED_FIELDS:
        tally: Dict[str, int] = {}
        first: Dict[str, Any] = {}
        for member in members:
            token = _same(member[field])
            tally[token] = tally.get(token, 0) + 1
            first.setdefault(token, member[field])
        if len(tally) <= 1:
            continue
        top = max(tally.values())
        leaders = [t for t in tally if tally[t] == top]   # dict order = stable member order
        majority = leaders[0]
        tie = len(leaders) > 1
        for member in members:
            if _same(member[field]) != majority:
                flag = {"flag": name, "field": field, "value": member[field], "majority": first[majority]}
                if tie:
                    flag["majority_tie"] = True
                out[member["probe_id"]].append(flag)
    for member in members:
        if not member["definition_built"]:
            out[member["probe_id"]].append({"flag": "definition_stale", "reason": "not_built"})
        elif member["identity_current"] is False:
            out[member["probe_id"]].append({"flag": "definition_stale",
                                            "reason": member["identity_stale_reason"] or "decision_changed"})
        if member["sub_question"] is not None and member["sub_question"] not in declared:
            out[member["probe_id"]].append({"flag": "sub_question_not_declared",
                                            "sub_question": member["sub_question"], "version": version})
    return out


def coverage(declared: Sequence[str], members: Sequence[Dict[str, Any]], *, key: str, model: Optional[str],
             version: Optional[int], train_views: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    """FR-4: every declared sub-question (declared order) with its members, `missing` ones FIRST, then
    the `main_question` group (members with no sub-question) — PURE.

    A missing sub-question's `next` is one `start_probe_training_run` on its registered training view
    (`train_views[sub_question]`, newest first) on this model; without a view, the
    `register_probe_dataset` call that adds one. No `next` can be built without a model."""
    by_sub: Dict[Optional[str], List[str]] = {}
    for member in members:
        by_sub.setdefault(member["sub_question"], []).append(member["probe_id"])
    missing: List[Dict[str, Any]] = []
    covered: List[Dict[str, Any]] = []
    for sub in declared:
        ids = by_sub.get(sub, [])
        if ids:
            covered.append({"sub_question": sub, "state": "covered", "members": ids})
            continue
        entry: Dict[str, Any] = {"sub_question": sub, "state": "missing", "members": []}
        views = train_views.get(sub) or []
        if views and model:
            entry["next"] = [next_train_specialist(model, views[0], key)]
        else:
            entry["next"] = [next_register_specialist_view(key, sub, version)]
        missing.append(entry)
    main = {"sub_question": None, "group": "main_question", "state": "covered" if by_sub.get(None) else "empty",
            "members": by_sub.get(None, [])}
    return missing + covered + [main]


def default_blend_set(members: Sequence[Dict[str, Any]], member_flags: Dict[str, List[Dict[str, Any]]]) -> List[str]:
    """L4: primary + candidates, minus members whose sub-question the current version does not declare."""
    return [m["probe_id"] for m in members
            if m["role"] in ("primary", "candidate")
            and not any(f["flag"] == "sub_question_not_declared" for f in member_flags.get(m["probe_id"], []))]


async def train_views_by_sub_question(db: AsyncSession, keys: Sequence[str]) -> Dict[str, Dict[str, List[str]]]:
    """`{key: {sub_question: [train view ids, newest first]}}` — unarchived `train` views carrying a
    sub-question, for many keys in one statement."""
    if not keys:
        return {}
    rows = (await db.execute(
        select(ProbeMonitorDataset.behaviour, ProbeMonitorDataset.sub_question, ProbeMonitorDataset.id)
        .where(ProbeMonitorDataset.behaviour.in_(list(keys)), ProbeMonitorDataset.role == "train",
               ProbeMonitorDataset.sub_question.isnot(None), ProbeMonitorDataset.archived_at.is_(None))
        .order_by(ProbeMonitorDataset.created_at.desc(), ProbeMonitorDataset.id.desc())
    )).all()
    out: Dict[str, Dict[str, List[str]]] = {}
    for key, sub, view_id in rows:
        out.setdefault(key, {}).setdefault(sub, []).append(view_id)
    return out


# ── the family ───────────────────────────────────────────────────────────────────────────────


async def resolve_members(db: AsyncSession, key: str, *, model: Optional[str] = None,
                          member_ids: Optional[Sequence[str]] = None,
                          include_retired: bool = False) -> Dict[str, Any]:
    """The family of `key` on one model, UNPAGED — what `family()` pages and what a blend evaluation takes,
    so the two can never resolve members two ways. Raises `BehaviourRefused` (404) for an uncatalogued key
    and `FamilyRefused` for `model_required` and `member_not_in_family`."""
    head = await catalogue.get_head(db, key)
    version = (await db.execute(select(ProbeBehaviourVersion).where(
        ProbeBehaviourVersion.key == key, ProbeBehaviourVersion.version == head.current_version,
    ))).scalar_one_or_none()
    declared = declared_sub_questions(version)

    rows = await member_rows(db, [key], member_ids=member_ids)
    if member_ids is not None:
        found = {probe.id for probe, _r, _role, _rec in rows}
        absent = sorted(set(member_ids) - found)
        if absent:
            raise FamilyRefused(
                422, "member_not_in_family",
                f"{len(absent)} named probe(s) do not exist or do not carry the key {key!r}: {absent}",
                {"member_ids": absent, "key": key},
            )
    else:
        rows = default_members(rows, include_retired=include_retired)

    counts: Dict[str, int] = {}
    for _probe, run, _role, _rec in rows:
        counts[run.model_id] = counts.get(run.model_id, 0) + 1
    models = [{"model_id": m, "members": counts[m]} for m in sorted(counts)]
    if model is None:
        if len(counts) > 1:
            raise FamilyRefused(
                422, "model_required",
                f"the family of {key!r} has members on {len(counts)} models; name one",
                {"models": models}, next=[next_get_family(key, m) for m in sorted(counts)],
            )
        model = next(iter(counts), None)
    elif member_ids is not None and any(run.model_id != model for _p, run, _r, _c in rows):
        raise FamilyRefused(
            422, "model_required",
            f"the named members are not all on model {model!r}",
            {"models": models}, next=[next_get_family(key, m) for m in sorted(counts)],
        )
    rows = [r for r in rows if r[1].model_id == model]

    view_ids = {run.train_dataset_id for _p, run, _r, _c in rows}
    views = {v.id: v for v in (await db.execute(
        select(ProbeMonitorDataset).where(ProbeMonitorDataset.id.in_(list(view_ids)))
    )).scalars()} if view_ids else {}
    run_ids = sorted({run.id for _p, run, _r, _c in rows})
    precisions = await run_in_threadpool(_precisions, run_ids) if run_ids else {}

    members = [member_facts(probe, run, role, recorded, views, precisions.get(run.id))
               for probe, run, role, recorded in rows]
    member_flags = flags(members, declared=declared, version=head.current_version)
    for member in members:
        member["flags"] = member_flags[member["probe_id"]]
    blend_set = default_blend_set(members, member_flags)
    for member in members:
        member["default_blend"] = member["probe_id"] in blend_set
    return {"head": head, "version": version, "declared": declared, "model": model, "models": models,
            "members": members, "default_blend_set": blend_set}


async def family(db: AsyncSession, key: str, *, model: Optional[str] = None,
                 member_ids: Optional[Sequence[str]] = None, include_retired: bool = False,
                 offset: int = 0, limit: Optional[int] = None) -> Dict[str, Any]:
    """FR-2 – FR-4: the family of `key` on one model, paged, with flags and coverage.

    Raises `BehaviourRefused` (404) for an uncatalogued key and `FamilyRefused` for `model_required`,
    `member_not_in_family` and `cap_exceeded`."""
    from ..core.runtime_settings import effective

    page_cap = int(effective().blend_family_page_size)
    size = page_cap if limit is None else int(limit)
    if size > page_cap:
        raise FamilyRefused(
            422, "cap_exceeded",
            f"a page of {size} members is more than the runtime setting blend_family_page_size ({page_cap})",
            {"setting": "blend_family_page_size", "value": page_cap, "field": "limit", "got": size},
        )
    resolved = await resolve_members(db, key, model=model, member_ids=member_ids, include_retired=include_retired)
    head, declared, model, members = resolved["head"], resolved["declared"], resolved["model"], resolved["members"]
    trains = (await train_views_by_sub_question(db, [key])).get(key, {})
    page = members[offset: offset + size]
    return {
        "key": key,
        "version": head.current_version,
        "model": model,
        "models": resolved["models"],
        "declared_sub_questions": declared,
        "coverage": coverage(declared, members, key=key, model=model, version=head.current_version,
                             train_views=trains),
        "default_blend_set": resolved["default_blend_set"],
        "flags_basis": "majority_of_all_members",
        "members": page,
        "total": len(members),
        "offset": offset,
        "limit": size,
        "next_offset": offset + size if offset + size < len(members) else None,
    }
