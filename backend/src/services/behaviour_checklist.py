"""The guided checklist and the computed status of a behaviour (Feature 037 FR-12; D26, D49).

Nine steps (eight from 037, one from 040), each read from REAL records and never ticked by hand (D26):

  1 defined          the catalogue row exists
  2 data             a probe dataset view carries the key, or is linked to it
  3 trained          a completed probe run carries the key
  4 rung 2           a probe with the key has rung >= 2 (evaluated on unseen data, catching at least half
                     the positives there at its bar, and few false alarms on everyday and nonsense
                     text — ENH-006 D82, D84)
  5 prompt chosen    an unarchived judge prompt judges the key
  6 rung 3           a probe with the key has rung 3 (judge-compared)
  7 exported         the PRIMARY probe has a built definition (`definition_sha256`)
  8 armed            miLLM has the primary probe armed (read through `millm_read`)
  9 definition_complete  (Feature 040 FR-41) the current version declares at least one success
                     criterion and every criterion's held-out test is a declared column; else not_done
                     with the `next` that adds them

Each step is independent: one fact, one query. `state` is "done", "not_done", "unknown" or (step 7
only, Feature 039) "stale" — and
⚠ "unknown" IS NOT "not_done" (D-T5): when miLLM is not connected or cannot say which miStudio probe a
served probe came from, the armed step is unknown, with the reason, never false.

Status (D49): archived (hand-set) > monitoring (armed) > unknown (exported, miLLM cannot say) >
validated (primary at rung >= 2) > draft.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Union

from sqlalchemy import func, select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from ..models.probe_behaviour import (
    ProbeBehaviour,
    ProbeBehaviourProbeRole,
    ProbeBehaviourVersion,
    ProbeBehaviourViewLink,
)
from ..models.probe_judge_prompt import ProbeJudgePrompt
from ..models.probe_monitor import ProbeMonitor, ProbeMonitorDataset, ProbeMonitorRun
from . import behaviour_catalogue as catalogue
from .millm_read import Unknown, armed_source_probe_ids

ArmedReader = Callable[[], Union[Set[str], Unknown]]

#: (id, label, where the operator does it) — the order the UI shows them in.
STEPS = (
    ("defined", "Behaviour defined", {"tab": "behaviours"}),
    ("data", "Data registered", {"tab": "datasets"}),
    ("trained", "Probe trained", {"tab": "runs"}),
    ("rung2", "Evaluated on unseen data, everyday text and nonsense text (rung 2)", {"tab": "probes"}),
    ("prompt", "Judge prompt chosen", {"tab": "judge-prompts"}),
    ("rung3", "Compared with a judge (rung 3)", {"tab": "judge"}),
    ("exported", "Primary probe exported", {"tab": "probes"}),
    ("armed", "Armed in miLLM", {"app": "millm"}),
    # Feature 040 FR-41 (S1): the definition says what the detector must achieve.
    ("definition_complete", "Success criteria defined", {"tab": "behaviours"}),
    # Feature 040 FR-41 (S2): from views to a stored verdict.
    ("views_registered", "Test, calibration and sub-question views registered", {"tab": "datasets"}),
    ("evaluated", "Evaluated on every criterion's test view", {"tab": "probes"}),
    ("breakdown_available", "Breakdown by a declared column available", {"tab": "behaviours"}),
    ("success_report", "Success report built", {"tab": "behaviours"}),
    # Feature 041 FR-5 (S1): the family, after 040's steps.
    ("family_trained", "Every sub-question has a family member", {"tab": "behaviours"}),
    ("blend_report", "Blend report built", {"tab": "behaviours"}),
    ("family_exported", "Every family member exported", {"tab": "behaviours"}),
)


def _step(step_id: str, state: str, reason: str, evidence: Optional[List[str]] = None) -> Dict[str, Any]:
    label, link = next((lbl, lnk) for sid, lbl, lnk in STEPS if sid == step_id)
    return {"id": step_id, "label": label, "state": state, "reason": reason,
            "evidence": evidence or [], "link": link}


async def _ids_many(db: AsyncSession, key_column, id_column, keys: Sequence[str], *where) -> Dict[str, List[str]]:
    """`{key: [up to 20 ids, ascending]}` for every key in ONE query (a row_number window per key), so a
    page of N behaviours costs one statement per step, not N (FR-6, FTID D-7 / TI-14)."""
    if not keys:
        return {}
    rank = func.row_number().over(partition_by=key_column, order_by=id_column).label("rank")
    inner = select(key_column.label("k"), id_column.label("i"), rank).where(key_column.in_(list(keys)), *where).subquery()
    rows = (await db.execute(select(inner.c.k, inner.c.i).where(inner.c.rank <= 20).order_by(inner.c.k, inner.c.i))).all()
    out: Dict[str, List[str]] = {}
    for key, ident in rows:
        out.setdefault(key, []).append(str(ident))
    return out


async def _views_many(db: AsyncSession, keys: Sequence[str]) -> Dict[str, List[str]]:
    """Views that carry a key or are linked to it (step 2), for many keys in two statements."""
    by_key = await _ids_many(db, ProbeMonitorDataset.behaviour, ProbeMonitorDataset.id, keys)
    links = (await db.execute(
        select(ProbeBehaviourViewLink.key, ProbeBehaviourViewLink.dataset_id)
        .where(ProbeBehaviourViewLink.key.in_(list(keys)))
        .order_by(ProbeBehaviourViewLink.key, ProbeBehaviourViewLink.dataset_id)
    )).all() if keys else []
    for key, dataset_id in links:
        ids = by_key.setdefault(key, [])
        if str(dataset_id) not in ids and len(ids) < 20:
            ids.append(str(dataset_id))
    return {key: sorted(ids) for key, ids in by_key.items()}


async def _primaries_many(db: AsyncSession, keys: Sequence[str]) -> Dict[str, ProbeMonitor]:
    """The primary probe of each key whose own key is still that behaviour (D24), in one statement."""
    if not keys:
        return {}
    rows = (await db.execute(
        select(ProbeBehaviourProbeRole.key, ProbeMonitor)
        .join(ProbeMonitor, ProbeMonitor.id == ProbeBehaviourProbeRole.probe_id)
        .where(ProbeBehaviourProbeRole.key.in_(list(keys)), ProbeBehaviourProbeRole.role == "primary",
               ProbeMonitor.behaviour == ProbeBehaviourProbeRole.key)
    )).all()
    return {key: probe for key, probe in rows}


async def _versions_many(db: AsyncSession, heads: Sequence[ProbeBehaviour]) -> Dict[str, ProbeBehaviourVersion]:
    if not heads:
        return {}
    pairs = [(h.key, h.current_version) for h in heads]
    rows = (await db.execute(
        select(ProbeBehaviourVersion).where(
            tuple_(ProbeBehaviourVersion.key, ProbeBehaviourVersion.version).in_(pairs)
        )
    )).scalars().all()
    return {row.key: row for row in rows}


def definition_step(key: str, version: Optional[ProbeBehaviourVersion]) -> Dict[str, Any]:
    """Feature 040 FR-41: at least one success criterion, each held-out test a declared column."""
    from .detector_next import next_add_success_criteria

    criteria = (getattr(version, "success_criteria", None) or []) if version is not None else []
    declared = set(getattr(version, "held_out_columns", None) or []) if version is not None else set()
    number = getattr(version, "version", None)
    if not criteria:
        step = _step("definition_complete", "not_done",
                     "the current version declares no success criterion: nothing says what the detector "
                     "must achieve", [f"{key}@{number}"] if number else [])
        if number:
            step["next"] = [next_add_success_criteria(key, number)]
        return step
    undeclared = sorted({c.get("held_out_test") for c in criteria if c.get("held_out_test")} - declared)
    if undeclared:
        step = _step("definition_complete", "not_done",
                     f"criteria name held-out columns the version does not declare: {undeclared}",
                     [f"{key}@{number}"])
        step["next"] = [next_add_success_criteria(key, number)]
        return step
    return _step("definition_complete", "done",
                 f"version {number} declares {len(criteria)} success criterion(s)", [f"{key}@{number}"])


async def checklist_many(
    db: AsyncSession, heads: Sequence[ProbeBehaviour], armed_reader: Optional[ArmedReader] = None
) -> Dict[str, Dict[str, Any]]:
    """The checklist of many behaviours at a cost independent of how many there are (TI-14): one
    statement per step for the whole set, and miLLM read at most ONCE (only when some behaviour has a
    primary probe). `checklist()` is this for one key, so the two can never disagree."""
    keys = [head.key for head in heads]
    views = await _views_many(db, keys)
    runs = await _ids_many(db, ProbeMonitorRun.behaviour, ProbeMonitorRun.id, keys,
                           ProbeMonitorRun.status == "completed")
    rung2 = await _ids_many(db, ProbeMonitor.behaviour, ProbeMonitor.id, keys, ProbeMonitor.rung >= 2)
    prompts = await _ids_many(db, ProbeJudgePrompt.behaviour, ProbeJudgePrompt.id, keys,
                              ProbeJudgePrompt.archived_at.is_(None))
    rung3 = await _ids_many(db, ProbeMonitor.behaviour, ProbeMonitor.id, keys, ProbeMonitor.rung == 3)
    primaries = await _primaries_many(db, keys)
    versions = await _versions_many(db, heads)
    facts = await _detector_facts(db, keys, primaries)
    family = await _family_facts(db, keys, versions)
    armed: Any = None
    # miLLM is read at most ONCE, and only when some behaviour has a family member to ask about.
    if primaries or any(family["members"].get(k) for k in keys):
        reader = armed_reader or armed_source_probe_ids
        armed = await run_in_threadpool(reader)

    out: Dict[str, Dict[str, Any]] = {}
    for head in heads:
        key = head.key
        steps: List[Dict[str, Any]] = [_step("defined", "done", f"{key!r} is catalogued", [key])]
        v = views.get(key, [])
        steps.append(_step("data", "done" if v else "not_done",
                           f"{len(v)} view(s) carry or link the key" if v
                           else "no probe dataset view carries this key or is linked to it", v))
        r = runs.get(key, [])
        steps.append(_step("trained", "done" if r else "not_done",
                           "a completed probe run carries the key" if r
                           else "no completed probe run carries this key", r))
        r2 = rung2.get(key, [])
        steps.append(_step("rung2", "done" if r2 else "not_done",
                           "a probe with this key reached rung 2" if r2
                           else "no probe with this key has reached rung 2 (evaluated on unseen data, catching at "
                           "least half the positives there at its bar, with few false alarms on everyday and "
                           "nonsense text)", r2))
        p = prompts.get(key, [])
        steps.append(_step("prompt", "done" if p else "not_done",
                           "a judge prompt judges this behaviour" if p
                           else "no unarchived judge prompt judges this behaviour", p))
        r3 = rung3.get(key, [])
        steps.append(_step("rung3", "done" if r3 else "not_done",
                           "a probe with this key was compared with a judge of it (rung 3)" if r3
                           else "no probe with this key has reached rung 3", r3))
        primary = primaries.get(key)
        members = family["members"].get(key) or []
        models = family["models"].get(key) or []
        # Feature 041 FR-5: exported and armed report EVERY member of the default family on each model.
        steps.append(family_exported_step(primary, members, models))
        steps.append(family_armed_step(members, models, armed))
        steps.append(definition_step(key, versions.get(key)))
        steps.extend(detector_steps(key, versions.get(key), facts))
        steps.extend(family_steps(key, versions.get(key), members, models, family["train_views"].get(key) or {},
                                  family["blend_reports"].get(key) or []))
        # ⚠ STATUS KEEPS 037'S MEANING (OI-2 answered, ENH-006 D36): "monitoring" is the PRIMARY armed;
        # the family's other members show their own state in the steps. Read from the primary's own
        # facts, never from the (family) `armed` / `exported` steps (FTID D-13).
        out[key] = {"key": key, "steps": steps,
                    "status": status(head, primary, primary_facts(primary, armed))}
    return out


async def checklist(db: AsyncSession, key: str, armed_reader: Optional[ArmedReader] = None) -> Dict[str, Any]:
    head = await catalogue.get_head(db, key)
    return (await checklist_many(db, [head], armed_reader))[key]


def exported_step(primary: ProbeMonitor) -> Dict[str, Any]:
    """Step 7 (Feature 039 FR-19, D-T7): done, STALE, or not done.

    * **done** when the primary has a built definition (`definition_sha256`) — a definition built
      before digests stays done; the detail carries its identity and where it came from;
    * **stale** when a definition was built and then invalidated (any cause; the reason says which):
      a file exists and states an old decision or rung, which is not "never exported" — with the
      `next` call that rebuilds it;
    * **not_done** otherwise.
    """
    from .next_calls import build_probe_definition
    from .probe_decision import identity_fields

    identity = identity_fields(primary)
    if primary.definition_sha256:
        step = _step("exported", "done", f"primary probe {primary.id} has a built definition",
                     [primary.id])
        step["identity"] = {key: identity[key] for key in (
            "detector_digest", "decision_digest", "decision_revision", "identity_current")}
        step["identity_source"] = identity["identity_source"]
        return step
    invalidated = (primary.definition_build or {}).get("invalidated")
    if invalidated:
        step = _step(
            "exported", "stale",
            f"primary probe {primary.id}'s definition was invalidated "
            f"({invalidated.get('reason')}; cause: {invalidated.get('cause') or 'not recorded'}) — "
            f"the file states what is no longer true; build it again",
            [primary.id],
        )
        step["next"] = [build_probe_definition(
            primary.id, "Rebuild the primary probe's definition so it states the current decision.",
            rung=primary.rung,
        )]
        step["identity_source"] = identity["identity_source"]
        return step
    return _step("exported", "not_done", f"primary probe {primary.id} has not been exported",
                 [primary.id])


def primary_facts(primary: Optional[ProbeMonitor], armed: Any) -> Dict[str, Optional[str]]:
    """The PRIMARY's own exported and armed states (`done` / `stale` / `not_done` / `unknown`), which
    `status()` reads — separate from the family steps (Feature 041 FR-5; OI-2 / D36; FTID D-13)."""
    if primary is None:
        return {"armed": None, "exported": None}
    if isinstance(armed, Unknown):
        armed_state = "unknown"
    else:
        armed_state = "done" if armed is not None and primary.id in armed else "not_done"
    return {"armed": armed_state, "exported": exported_step(primary)["state"]}


def status(head: ProbeBehaviour, primary: Optional[ProbeMonitor], facts: Dict[str, Optional[str]]) -> str:
    """D49: computed, never stored; only "archived" is set by hand. Reads the PRIMARY's facts (OI-2)."""
    if head.archived_at is not None:
        return "archived"
    if facts.get("armed") == "done":
        return "monitoring"
    # ⚠ `stale` IS NOT DONE (039 FR-19): a stale export with miLLM unknown is not "unknown" status.
    if facts.get("armed") == "unknown" and facts.get("exported") == "done":
        return "unknown"
    if primary is not None and (primary.rung or 0) >= 2:
        return "validated"
    return "draft"


# ── Feature 041 S1 (FR-5): the family in the checklist, page-batched ──────────────────────────


async def _family_facts(db: AsyncSession, keys: Sequence[str],
                        versions: Dict[str, ProbeBehaviourVersion]) -> Dict[str, Any]:
    """The default family of every key (members on every model), the models each behaviour is judged on
    (its version's `models`, resolved to miStudio ids, else every model with a member) and the training
    views per sub-question — a FIXED number of statements for any number of keys (TI-14)."""
    from ..models.model import Model
    from sqlalchemy import or_

    from . import probe_family

    members: Dict[str, List[Any]] = {}
    for row in probe_family.default_members(await probe_family.member_rows(db, keys)):
        members.setdefault(row[0].behaviour, []).append(row)
    refs = sorted({m.get("model_id") for v in versions.values() for m in (v.models or []) if m.get("model_id")})
    resolved: Dict[str, str] = {}
    if refs:
        for model_id, repo_id in (await db.execute(
            select(Model.id, Model.repo_id).where(or_(Model.id.in_(refs), Model.repo_id.in_(refs)))
        )).all():
            resolved.setdefault(model_id, model_id)
            if repo_id:
                resolved.setdefault(repo_id, model_id)
    models: Dict[str, List[str]] = {}
    for key in keys:
        version = versions.get(key)
        declared = [m.get("model_id") for m in ((version.models if version is not None else None) or [])
                    if m.get("model_id")]
        if declared:
            models[key] = list(dict.fromkeys(resolved.get(ref, ref) for ref in declared))
        else:
            models[key] = sorted({run.model_id for _p, run, _r, _c in members.get(key, [])})
    # Feature 041 S2: every DONE blend report of these keys, newest first, with the rows its staleness
    # reads — a fixed number of statements (`stale_facts`) for any page.
    from ..models.probe_blend import ProbeBlendReport
    from .probe_blend import service as blends

    reports = list((await db.execute(
        select(ProbeBlendReport).where(ProbeBlendReport.behaviour_key.in_(list(keys)), ProbeBlendReport.status == "done")
        .order_by(ProbeBlendReport.created_at.desc(), ProbeBlendReport.id.desc())
    )).scalars().all()) if keys else []
    stale_facts = await blends.stale_facts(db, reports) if reports else {}
    by_key: Dict[str, List[Any]] = {}
    for row in reports:
        by_key.setdefault(row.behaviour_key, []).append((row, blends.stale_from(row, stale_facts)))
    return {"members": members, "models": models,
            "train_views": await probe_family.train_views_by_sub_question(db, keys),
            "blend_reports": by_key}


def _per_model(members: Sequence[Any], models: Sequence[str]) -> Dict[str, List[Any]]:
    out: Dict[str, List[Any]] = {model: [] for model in models}
    for row in members:
        if row[1].model_id in out:
            out[row[1].model_id].append(row)
    return out


def _rollup(states: Dict[str, str], empty_models: Sequence[str]) -> str:
    """`done` when every member is and no judged model is empty; `partly_done` when some are; else
    `not_done`."""
    done = [p for p, s in states.items() if s == "done"]
    if done and len(done) == len(states) and not empty_models:
        return "done"
    return "partly_done" if done else "not_done"


def family_exported_step(primary: Optional[ProbeMonitor], members: Sequence[Any], models: Sequence[str]) -> Dict[str, Any]:
    """Step 7 per member (FR-5): `done` / `stale` / `not_done` for each member of the default family on
    each model (039's `exported_step` per member); the step is `done` when every member is, `partly_done`
    naming the rest, `stale` when none is done and one was invalidated. The primary's 039 identity rides
    along when there is a primary."""
    from .next_calls import build_probe_definition

    per_model = _per_model(members, models)
    empty = [m for m, rows in per_model.items() if not rows]
    if not members:
        return _step("exported", "not_done", "no probe of this behaviour exists yet")
    per_member = {row[0].id: exported_step(row[0]) for rows in per_model.values() for row in rows}
    states = {pid: step["state"] for pid, step in per_member.items()}
    state = _rollup(states, empty)
    rest = [pid for pid, s in states.items() if s != "done"]
    if state == "not_done" and any(s == "stale" for s in states.values()):
        state = "stale"
    if state == "done":
        reason = f"every member ({len(states)}) has a built definition"
    elif len(per_member) == 1 and not empty:
        reason = next(iter(per_member.values()))["reason"]
    else:
        reason = (f"{len(states) - len(rest)} of {len(states)} member(s) exported; not exported: {rest}"
                  + (f"; no member on model(s) {empty}" if empty else ""))
    step = _step("exported", state, reason, sorted(states))
    step["members"] = [{"probe_id": row[0].id, "model_id": row[1].model_id, "role": row[2],
                        "state": states[row[0].id]} for rows in per_model.values() for row in rows]
    step["models"] = [{"model_id": m, "members": len(rows)} for m, rows in per_model.items()]
    if rest:
        step["next"] = [build_probe_definition(
            pid, "Build this family member's definition so it states the current decision.",
            rung=next(row[0].rung for row in members if row[0].id == pid),
        ) for pid in rest]
    if primary is not None and primary.id in per_member:
        primary_step = per_member[primary.id]
        for field in ("identity", "identity_source"):
            if field in primary_step:
                step[field] = primary_step[field]
    return step


def family_armed_step(members: Sequence[Any], models: Sequence[str], armed: Any) -> Dict[str, Any]:
    """Step 8 per member (FR-5): which members miLLM has armed. ⚠ `unknown` stays `unknown` (037 D-T5):
    when miLLM cannot say, no member is reported unarmed."""
    per_model = _per_model(members, models)
    empty = [m for m, rows in per_model.items() if not rows]
    ids = [row[0].id for rows in per_model.values() for row in rows]
    if not members:
        return _step("armed", "not_done", "no probe of this behaviour exists yet")
    if isinstance(armed, Unknown):
        step = _step("armed", "unknown", armed.reason, ids)
        step["members"] = [{"probe_id": row[0].id, "model_id": row[1].model_id, "role": row[2],
                            "state": "unknown"} for rows in per_model.values() for row in rows]
        return step
    armed_ids = armed if armed is not None else set()
    states = {pid: ("done" if pid in armed_ids else "not_done") for pid in ids}
    state = _rollup(states, empty)
    unarmed = [pid for pid, s in states.items() if s != "done"]
    if state == "done":
        reason = f"miLLM has every member ({len(ids)}) armed"
    else:
        reason = (f"miLLM has {len(ids) - len(unarmed)} of {len(ids)} member(s) armed; not armed: {unarmed}"
                  + (f"; no member on model(s) {empty}" if empty else ""))
    step = _step("armed", state, reason, ids)
    step["members"] = [{"probe_id": row[0].id, "model_id": row[1].model_id, "role": row[2],
                        "state": states[row[0].id]} for rows in per_model.values() for row in rows]
    step["models"] = [{"model_id": m, "members": len(rows)} for m, rows in per_model.items()]
    return step


def family_steps(key: str, version: Optional[ProbeBehaviourVersion], members: Sequence[Any],
                 models: Sequence[str], train_views: Dict[str, List[str]],
                 reports: Sequence[Any] = ()) -> List[Dict[str, Any]]:
    """`family_trained`, `blend_report`, `family_exported` (FR-5), each from real rows with its `next`."""
    from . import probe_family
    from .detector_next import next_build_member, next_find_views, next_register_specialist_view, next_train_specialist
    from .probe_decision import identity_fields

    declared = probe_family.declared_sub_questions(version)
    number = getattr(version, "version", None)
    per_model = _per_model(members, models)
    steps: List[Dict[str, Any]] = []

    # family_trained — every declared sub-question has a member on every judged model.
    if not members:
        step = _step("family_trained", "not_done", "no probe of this behaviour has been trained")
        step["next"] = [next_find_views()]
    elif not declared:
        step = _step("family_trained", "done",
                     f"the current version declares no sub-question; the main question has "
                     f"{len(members)} member(s)", sorted(row[0].id for row in members))
    else:
        gaps: List[Dict[str, Any]] = []
        nexts: List[Any] = []
        for model, rows in per_model.items():
            have = {row[0].sub_question for row in rows}
            for sub in declared:
                if sub in have:
                    continue
                gaps.append({"model_id": model, "sub_question": sub})
                views = train_views.get(sub) or []
                nexts.append(next_train_specialist(model, views[0], key) if views
                             else next_register_specialist_view(key, sub, number))
        if gaps:
            step = _step("family_trained", "not_done",
                         f"{len(gaps)} declared sub-question(s) have no member: "
                         + ", ".join(f"{g['sub_question']} on {g['model_id']}" for g in gaps))
            step["missing"] = gaps
            step["next"] = nexts
        else:
            step = _step("family_trained", "done",
                         f"every declared sub-question ({len(declared)}) has a member on every model",
                         sorted(row[0].id for row in members))
    steps.append(step)

    # blend_report — a done, NON-STALE report exists on every judged model (stale computed on read, T30).
    from .detector_next import next_evaluate_blends, next_rerun_report

    current = {}
    stale_on: Dict[str, Any] = {}
    for row, reasons in reports:
        if reasons:
            stale_on.setdefault(row.model_id, row)
        elif row.model_id not in current:
            current[row.model_id] = row
    wanted = list(models) or sorted({row.model_id for row, _r in reports})
    lacking = [m for m in wanted if m not in current]
    if wanted and not lacking:
        step = _step("blend_report", "done", f"a current blend report exists on every model ({len(wanted)})",
                     sorted(r.id for r in current.values()))
    else:
        step = _step("blend_report", "partly_done" if current else "not_done",
                     (f"no current blend report on model(s) {lacking}" if wanted else
                      "no blend report has been built for this behaviour")
                     + ("; the latest is stale" if any(m in stale_on for m in lacking) else ""),
                     sorted(r.id for r in current.values()))
        step["next"] = [next_rerun_report(stale_on[m].id) if m in stale_on else next_evaluate_blends(key, m, None)
                        for m in lacking] if wanted else []
    steps.append(step)

    # family_exported — every member has a CURRENT built definition (039 identity not behind the bar).
    if not members:
        step = _step("family_exported", "not_done", "no probe of this behaviour exists yet")
    else:
        current = {row[0].id: (row[0].definition_path is not None
                               and identity_fields(row[0])["identity_current"] is not False)
                   for rows in per_model.values() for row in rows}
        states = {pid: ("done" if ok else "not_done") for pid, ok in current.items()}
        empty = [m for m, rows in per_model.items() if not rows]
        state = _rollup(states, empty)
        missing = [pid for pid, ok in current.items() if not ok]
        step = _step("family_exported", state,
                     f"every member ({len(current)}) has a current built definition" if state == "done"
                     else f"{len(missing)} member(s) without a current built definition: {missing}"
                     + (f"; no member on model(s) {empty}" if empty else ""), sorted(current))
        if missing:
            step["next"] = [next_build_member(pid, next(row[0].rung for row in members if row[0].id == pid))
                            for pid in missing]
    steps.append(step)
    return steps


# ── Feature 040 S2 (FR-41): the four detector steps after `definition_complete`, page-batched ────────


async def _detector_facts(db: AsyncSession, keys: Sequence[str], primaries: Dict[str, ProbeMonitor]) -> Dict[str, Any]:
    """Every fact the four S2 steps read, for all `keys`, in a fixed number of statements (TI-14)."""
    from ..models.detector_report import DetectorSuccessReport
    from ..models.probe_monitor import ProbeMonitorEvaluation

    if not keys:
        return {"views": {}, "calibration": {}, "probes": {}, "evaluations": {}, "reports": {}}
    views: Dict[str, Dict[str, ProbeMonitorDataset]] = {}
    for view in (await db.execute(select(ProbeMonitorDataset).where(ProbeMonitorDataset.behaviour.in_(list(keys))))).scalars():
        views.setdefault(view.behaviour, {})[view.id] = view
    for key, use, view in (await db.execute(
        select(ProbeBehaviourViewLink.key, ProbeBehaviourViewLink.use, ProbeMonitorDataset)
        .join(ProbeMonitorDataset, ProbeMonitorDataset.id == ProbeBehaviourViewLink.dataset_id)
        .where(ProbeBehaviourViewLink.key.in_(list(keys)))
    )).all():
        views.setdefault(key, {})[view.id] = view
    calibration: Dict[str, List[str]] = {}
    for key, cal in (await db.execute(
        select(ProbeMonitorRun.behaviour, ProbeMonitorRun.calibration_dataset_id)
        .where(ProbeMonitorRun.behaviour.in_(list(keys)), ProbeMonitorRun.calibration_dataset_id.isnot(None))
    )).all():
        calibration.setdefault(key, [])
        if cal not in calibration[key]:
            calibration[key].append(cal)
    probes: Dict[str, ProbeMonitor] = dict(primaries)
    for probe in (await db.execute(
        select(ProbeMonitor).where(ProbeMonitor.behaviour.in_(list(keys)), ProbeMonitor.selected.is_(True))
        .order_by(ProbeMonitor.created_at.desc(), ProbeMonitor.id.desc())
    )).scalars():
        probes.setdefault(probe.behaviour, probe)
    evaluations: Dict[str, Dict[str, str]] = {}
    ids = [p.id for p in probes.values()]
    if ids:
        for probe_id, view_id, eval_id in (await db.execute(
            select(ProbeMonitorEvaluation.probe_id, ProbeMonitorEvaluation.dataset_id, ProbeMonitorEvaluation.id)
            .where(ProbeMonitorEvaluation.probe_id.in_(ids), ProbeMonitorEvaluation.positions_path.isnot(None))
            .order_by(ProbeMonitorEvaluation.created_at)
        )).all():
            evaluations.setdefault(probe_id, {})[view_id] = eval_id
    reports: Dict[str, Any] = {}
    for row in (await db.execute(
        select(DetectorSuccessReport).where(DetectorSuccessReport.behaviour_key.in_(list(keys)))
        .order_by(DetectorSuccessReport.created_at.desc(), DetectorSuccessReport.id.desc())
    )).scalars():
        reports.setdefault(row.behaviour_key, row)
    return {"views": views, "calibration": calibration, "probes": probes, "evaluations": evaluations,
            "reports": reports}


def detector_steps(key: str, version: Optional[ProbeBehaviourVersion], facts: Dict[str, Any]) -> List[Dict[str, Any]]:
    """`views_registered`, `evaluated`, `breakdown_available`, `success_report` for one key."""
    from .detector_next import next_breakdown, next_evaluate, next_find_views, next_success_report

    views = list((facts["views"].get(key) or {}).values())
    tests = [v for v in views if v.role == "eval"]
    calibration = facts["calibration"].get(key) or [v.id for v in views if v.role == "calibration"]
    subs = [s.get("key") for s in (getattr(version, "sub_questions", None) or [])]
    covered = {v.sub_question for v in views if v.sub_question}
    missing_subs = [s for s in subs if s not in covered]
    steps: List[Dict[str, Any]] = []
    gaps = ([] if tests else ["a test (evaluation) view"]) + ([] if calibration else ["a calibration view"]) + \
        [f"a view for sub-question {s!r}" for s in missing_subs]
    step = _step("views_registered", "not_done" if gaps else "done",
                 ("missing: " + ", ".join(gaps)) if gaps else
                 f"{len(tests)} test view(s), {len(calibration)} calibration view(s)"
                 + (f", every sub-question covered" if subs else ""),
                 sorted({v.id for v in tests} | set(calibration)))
    if gaps:
        step["next"] = [next_find_views()]
    steps.append(step)

    probe = facts["probes"].get(key)
    stored = facts["evaluations"].get(probe.id, {}) if probe is not None else {}
    criteria = getattr(version, "success_criteria", None) or []
    wanted: List[str] = []
    for criterion in criteria:
        held = criterion.get("held_out_test")
        for v in tests:
            if (v.distribution == "out_of_distribution") == bool(held) and v.id not in wanted:
                wanted.append(v.id)
    if probe is None:
        steps.append(_step("evaluated", "not_done", "no primary or selected probe yet"))
    elif not wanted:
        steps.append(_step("evaluated", "not_done",
                           "no test view matches a criterion yet (held-out criteria need an "
                           "out-of-distribution view)", [probe.id]))
    else:
        missing = [v for v in wanted if v not in stored]
        step = _step("evaluated", "not_done" if missing else "done",
                     f"probe {probe.id} has no stored evaluation on {missing}" if missing
                     else f"probe {probe.id} has stored arrays on every criterion's test view",
                     [probe.id, *[stored[v] for v in wanted if v in stored]])
        if missing:
            step["next"] = [next_evaluate(probe.id, v) for v in missing]
        steps.append(step)

    grouped = [(view_id, eval_id) for view_id, eval_id in stored.items()
               if any(v.id == view_id and (v.group_columns or []) for v in views)]
    if probe is not None and grouped:
        view_id, eval_id = grouped[0]
        column = next(v for v in views if v.id == view_id).group_columns[0]
        step = _step("breakdown_available", "done", f"evaluation {eval_id} can be broken down by {column!r}", [eval_id])
        step["next"] = [next_breakdown(probe.id, eval_id, column)]
    else:
        step = _step("breakdown_available", "not_done",
                     "no stored evaluation of the probe is on a view with declared grouping columns",
                     [probe.id] if probe is not None else [])
        step["next"] = [next_find_views()]
    steps.append(step)

    report = facts["reports"].get(key)
    if report is None:
        step = _step("success_report", "not_done", "no success report has been built")
        step["next"] = [next_success_report(key)]
    else:
        step = _step("success_report", "done", f"latest report {report.id}: {report.overall_verdict}", [report.id])
        step["verdict"] = report.overall_verdict
        step["behaviour_version"] = report.behaviour_version
    steps.append(step)
    return steps
