"""The behaviour catalogue's rules (Feature 037; ENH-005 D20–D25, D48; PADR IDL-58).

Every rule is here; the router only translates `BehaviourRefused` into ``{code, message}`` (Part A's
`PromptRefused` pattern), so the UI and the MCP tools read the same refusal.

⚠ ADOPTION IS A JOIN AND NOTHING HERE WRITES PART A'S COLUMNS (D25). `adopted` and
`uncatalogued_keys` only SELECT from ``probe_judge_prompts``, ``probe_monitor_runs``, ``probe_monitors``
and ``probe_monitor_datasets``. `test_behaviour_catalogue.py::test_adoption_writes_nothing` compares every
adopted row before and after a create.

⚠ EVERY MUTATION WRITES ITS HISTORY ROW IN THE SAME TRANSACTION (FR-8): one ``db.commit()`` per call,
after both the change and its history row are added, so a history row exists exactly when the change does.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from sqlalchemy import func, or_, select, tuple_
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.clock import utc_now
from ..models.probe_behaviour import (
    ProbeBehaviour,
    ProbeBehaviourHistory,
    ProbeBehaviourProbeRole,
    ProbeBehaviourVersion,
    ProbeBehaviourViewLink,
)
from ..models.probe_judge_prompt import ProbeJudgePrompt
from ..models.probe_monitor import ProbeMonitor, ProbeMonitorDataset, ProbeMonitorRun
from . import behaviour_detector_definition as definition_rules
from . import rule_set_store
from .detector_next import (
    DetectorRefused,
    next_create_with_inline_rule_set,
    next_list_versions,
    next_version_with_inline_rule_set,
)
from .judge_prompt_forms import SLUG_PATTERN, is_slug, prompt_key
from .probe_decision import identity_fields

#: The 037 definition text (D22).
TEXT_FIELDS = ("name", "description", "positive_meaning", "negative_meaning", "target_fpr")
#: The definition fields a version carries (D22; Feature 040 FR-1 adds the six detector fields, stored
#: with the rule set as its digest). Changing any of them is a new version.
VERSION_FIELDS = TEXT_FIELDS + definition_rules.STORED_FIELDS


class BehaviourRefused(Exception):
    """A request the catalogue will not carry out, with an HTTP status and a plain reason.

    Feature 040 (TI-5): `next` is added ADDITIVELY — every 037 refusal keeps its body and gains
    ``"next": []``; a refusal with a remedy lists the calls that apply it (FR-40)."""

    def __init__(self, status: int, code: str, message: str, details: Optional[dict] = None,
                 next: Optional[list] = None):  # noqa: A002 - the wire name
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.details = details or {}
        self.next = list(next or [])

    def as_detail(self) -> Dict[str, Any]:
        detail: Dict[str, Any] = {"code": self.code, "message": self.message}
        if self.details:
            detail["details"] = self.details
        detail["next"] = self.next
        return detail


def _iso(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() if value is not None else None


def _check_key(key: str) -> None:
    if not is_slug(key):
        raise BehaviourRefused(
            422,
            "invalid_key",
            f"{key!r} is not a behaviour key. A key is 2 to 64 lower-case letters, digits and "
            f"hyphens, starting and ending with a letter or digit (the judge-prompt library's rule, "
            f"{SLUG_PATTERN}).",
        )


def _history(
    db: AsyncSession,
    *,
    key: str,
    subject_kind: str,
    subject_id: Optional[str],
    field: str,
    old: Any,
    new: Any,
    actor: str,
    reason: Optional[str] = None,
    approval_id: Optional[str] = None,
) -> None:
    db.add(
        ProbeBehaviourHistory(
            key=key,
            subject_kind=subject_kind,
            subject_id=subject_id,
            field=field,
            old_value=old,
            new_value=new,
            actor=actor,
            approval_id=approval_id,
            reason=reason,
        )
    )


# ── reading ───────────────────────────────────────────────────────────────────────────────────


async def get_head(db: AsyncSession, key: str) -> ProbeBehaviour:
    head = (
        await db.execute(select(ProbeBehaviour).where(ProbeBehaviour.key == key))
    ).scalar_one_or_none()
    if head is None:
        raise BehaviourRefused(
            404, "not_found", f"no behaviour {key!r} is catalogued; create it to catalogue the key"
        )
    return head


async def current_version(db: AsyncSession, head: ProbeBehaviour) -> ProbeBehaviourVersion:
    return (
        await db.execute(
            select(ProbeBehaviourVersion).where(
                ProbeBehaviourVersion.key == head.key,
                ProbeBehaviourVersion.version == head.current_version,
            )
        )
    ).scalar_one()


def version_out(version: ProbeBehaviourVersion) -> Dict[str, Any]:
    out = {
        "version": version.version,
        "name": version.name,
        "description": version.description,
        "positive_meaning": version.positive_meaning,
        "negative_meaning": version.negative_meaning,
        "target_fpr": version.target_fpr,
        "created_by": version.created_by,
        "created_at": _iso(version.created_at),
    }
    # Feature 040 FR-1: each detector field the version DECLARES. ⚠ NULL IS "NOT DECLARED" AND READS
    # AS AN ABSENT KEY, so a version written before 040 serialises exactly as it did (FR-1, FR-6's
    # byte-for-byte compatibility), and `[]` (declared empty) is kept distinct from absent.
    for field in definition_rules.STORED_FIELDS:
        value = getattr(version, field, None)
        if value is None:
            continue
        if field == "rule_set_digest":
            out["rule_set"] = rule_set_store.describe(None, value)
        else:
            out[field] = value
    return out


def head_out(head: ProbeBehaviour, version: ProbeBehaviourVersion) -> Dict[str, Any]:
    """ONE serialiser for a catalogue row (memory: one-list-needs-one-serialiser)."""
    return {
        "key": head.key,
        "catalogued": True,
        "current_version": head.current_version,
        "archived": head.archived_at is not None,
        "archived_at": _iso(head.archived_at),
        "archived_by": head.archived_by,
        "created_by": head.created_by,
        "created_at": _iso(head.created_at),
        "updated_at": _iso(head.updated_at),
        "definition": version_out(version),
    }


async def _counts_by_key(db: AsyncSession, column, keys: Optional[Iterable[str]] = None) -> Dict[str, int]:
    query = select(column, func.count()).where(column.isnot(None)).group_by(column)
    if keys is not None:
        query = query.where(column.in_(list(keys)))
    return {key: int(n) for key, n in (await db.execute(query)).all()}


#: The four places a behaviour key is written: Part A's three columns and 037's view column (TI2).
def _key_columns():
    return {
        "prompts": ProbeJudgePrompt.behaviour,
        "runs": ProbeMonitorRun.behaviour,
        "probes": ProbeMonitor.behaviour,
        "views": ProbeMonitorDataset.behaviour,
    }


async def usage_counts(db: AsyncSession, keys: Optional[Iterable[str]] = None) -> Dict[str, Dict[str, int]]:
    """``{key: {prompts, runs, probes, views}}`` — one GROUP BY per kind (FTDD §9)."""
    out: Dict[str, Dict[str, int]] = {}
    for kind, column in _key_columns().items():
        for key, n in (await _counts_by_key(db, column, keys)).items():
            out.setdefault(key, {"prompts": 0, "runs": 0, "probes": 0, "views": 0})[kind] = n
    return out


async def uncatalogued_keys(db: AsyncSession) -> List[Dict[str, Any]]:
    """Every key in use with no catalogue row, with per-kind counts (FR-4). Valid, not an error (D25)."""
    catalogued = set((await db.execute(select(ProbeBehaviour.key))).scalars().all())
    counts = await usage_counts(db)
    return [
        {"key": key, "catalogued": False, **counts[key]}
        for key in sorted(counts)
        if key not in catalogued
    ]


# ── Feature 040 FR-6: the catalogue in pages, at a cost independent of its size ──────────────────
#
# ⚠ ONE BATCHED READ PER PAGE, NEVER ONE PER ROW. 037 called `current_version` once per head, so a
# list cost grew with the catalogue; a page is now: the heads (keyset on the primary key), ONE
# `(key, version) IN (...)` read of their current versions, `usage_counts` limited to the page's keys,
# and the same for the uncatalogued keys. `test_behaviour_list_route_queries.py` counts the statements
# at the ROUTE for catalogues of 100 and 1,000 behaviours and requires them equal.
#
# The cursor is opaque: base64url of JCS {"k": last catalogued key, "u": last uncatalogued key,
# "f": digest of the filters}. A cursor sent with other filters is refused (`filter_changed`).


def _filter_digest(filters: Dict[str, Any]) -> str:
    from .detector_identity import jcs_bytes, sha256_hex

    return sha256_hex(jcs_bytes(filters))[:16]


#: A first page: nothing read yet on either side.
START: Dict[str, Any] = {"k": None, "u": None, "done_k": False, "done_u": False}


def encode_cursor(filters: Dict[str, Any], position: Dict[str, Any]) -> str:
    """`position` = {k: last catalogued key read, u: last uncatalogued key read, done_k, done_u}."""
    import base64

    from .detector_identity import jcs_bytes

    raw = jcs_bytes({"f": _filter_digest(filters), **{name: position[name] for name in START}})
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_cursor(cursor: str, filters: Dict[str, Any]) -> Dict[str, Any]:
    import base64
    import json

    from .detector_next import next_restart_listing

    try:
        data = json.loads(base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)))
        if not isinstance(data, dict) or set(data) != {"f", *START}:
            raise ValueError("shape")
        if not all(isinstance(data[n], bool) for n in ("done_k", "done_u")):
            raise ValueError("shape")
        if not all(data[n] is None or isinstance(data[n], str) for n in ("k", "u")):
            raise ValueError("shape")
    except Exception as exc:  # noqa: BLE001 - any undecodable cursor is the same refusal
        raise DetectorRefused(422, "invalid_field", "this cursor was not issued by this list",
                              {"field": "cursor", "rule": "malformed"},
                              next=[next_restart_listing(filters.get("q"), filters.get("status"))]) from exc
    if data["f"] != _filter_digest(filters):
        raise DetectorRefused(
            422, "invalid_field",
            "this cursor was issued for other filters; list again from the first page with these filters",
            {"field": "cursor", "rule": "filter_changed"},
            next=[next_restart_listing(filters.get("q"), filters.get("status"))],
        )
    return {name: data[name] for name in START}


def _like(q: str) -> str:
    return "%" + q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"


def _heads_query(include_archived: bool, q: Optional[str]):
    query = select(ProbeBehaviour)
    if not include_archived:
        query = query.where(ProbeBehaviour.archived_at.is_(None))
    if q:
        names = select(ProbeBehaviourVersion.key).where(
            ProbeBehaviourVersion.key == ProbeBehaviour.key,
            ProbeBehaviourVersion.version == ProbeBehaviour.current_version,
            ProbeBehaviourVersion.name.ilike(_like(q), escape="\\"),
        ).exists()
        query = query.where(or_(ProbeBehaviour.key.ilike(_like(q), escape="\\"), names))
    return query


async def heads_page(
    db: AsyncSession, *, include_archived: bool, q: Optional[str], after: Optional[str], limit: int
) -> List[ProbeBehaviour]:
    """Up to `limit` heads after `after`, in key order (keyset on the primary key)."""
    query = _heads_query(include_archived, q)
    if after is not None:
        query = query.where(ProbeBehaviour.key > after)
    return list((await db.execute(query.order_by(ProbeBehaviour.key.asc()).limit(limit))).scalars().all())


async def count_heads(db: AsyncSession, *, include_archived: bool, q: Optional[str]) -> int:
    inner = _heads_query(include_archived, q).with_only_columns(ProbeBehaviour.key).subquery()
    return int((await db.execute(select(func.count()).select_from(inner))).scalar_one())


async def rows_for(db: AsyncSession, heads: List[ProbeBehaviour]) -> List[Dict[str, Any]]:
    """`head_out` rows for a page: ONE batched version read and ONE `usage_counts` for its keys.

    ⚠ EVERY ROW GOES THROUGH `head_out` (037's one-serialiser fix; memory one-list-needs-one-serialiser)."""
    if not heads:
        return []
    pairs = [(h.key, h.current_version) for h in heads]
    versions = {
        v.key: v for v in (await db.execute(
            select(ProbeBehaviourVersion).where(
                tuple_(ProbeBehaviourVersion.key, ProbeBehaviourVersion.version).in_(pairs)
            )
        )).scalars().all()
    }
    counts = await usage_counts(db, [h.key for h in heads])
    rows = []
    for head in heads:
        out = head_out(head, versions[head.key])
        out["counts"] = counts.get(head.key, {"prompts": 0, "runs": 0, "probes": 0, "views": 0})
        rows.append(out)
    return rows


def _used_keys():
    """Every key written on a prompt, run, probe or view, as one UNION (TI2's four columns)."""
    from sqlalchemy import union

    parts = [select(column.label("key")).where(column.isnot(None)) for column in _key_columns().values()]
    return union(*parts).subquery()


def _uncatalogued_query(q: Optional[str]):
    used = _used_keys()
    query = select(used.c.key).where(~used.c.key.in_(select(ProbeBehaviour.key)))
    if q:
        query = query.where(used.c.key.ilike(_like(q), escape="\\"))
    return query


async def uncatalogued_page(
    db: AsyncSession, *, q: Optional[str], after: Optional[str], limit: int
) -> List[Dict[str, Any]]:
    query = _uncatalogued_query(q)
    if after is not None:
        query = query.where(query.selected_columns[0] > after)
    keys = list((await db.execute(query.order_by(query.selected_columns[0]).limit(limit))).scalars().all())
    counts = await usage_counts(db, keys) if keys else {}
    return [{"key": key, "catalogued": False, **counts[key]} for key in keys]


async def count_uncatalogued(db: AsyncSession, *, q: Optional[str]) -> int:
    inner = _uncatalogued_query(q).subquery()
    return int((await db.execute(select(func.count()).select_from(inner))).scalar_one())


async def list_behaviours(
    db: AsyncSession, *, include_uncatalogued: bool = True, include_archived: bool = False
) -> Dict[str, Any]:
    """Every catalogued behaviour and every uncatalogued key, unpaged (the 037 read, kept for callers
    that need the whole catalogue). Built on the same batched page reads, so it cannot drift from them."""
    heads = await heads_page(db, include_archived=include_archived, q=None, after=None, limit=1 << 30)
    return {
        "behaviours": await rows_for(db, heads),
        "uncatalogued": await uncatalogued_keys(db) if include_uncatalogued else [],
    }


async def roles_of(db: AsyncSession, key: str) -> Dict[str, ProbeBehaviourProbeRole]:
    rows = (
        await db.execute(select(ProbeBehaviourProbeRole).where(ProbeBehaviourProbeRole.key == key))
    ).scalars().all()
    return {row.probe_id: row for row in rows}


async def primary_probe(db: AsyncSession, key: str) -> Optional[ProbeMonitor]:
    """The probe whose role is primary AND whose key is still this behaviour (D24)."""
    return (
        await db.execute(
            select(ProbeMonitor)
            .join(ProbeBehaviourProbeRole, ProbeBehaviourProbeRole.probe_id == ProbeMonitor.id)
            .where(
                ProbeBehaviourProbeRole.key == key,
                ProbeBehaviourProbeRole.role == "primary",
                ProbeMonitor.behaviour == key,
            )
        )
    ).scalar_one_or_none()


def _confirmed_for(view: Optional[ProbeMonitorDataset], key: str) -> bool:
    record = getattr(view, "positive_labels_confirmed", None) or {}
    return isinstance(record, dict) and record.get("key") == key


def view_out(view: ProbeMonitorDataset, *, key: str, linked_by: str, use: Optional[str] = None) -> Dict[str, Any]:
    facts = view.dataset_version or {}
    return {
        "id": view.id,
        "name": view.name,
        "role": view.role,
        "use": use or view.role,
        "linked_by": linked_by,
        "behaviour": view.behaviour,
        "positive_meaning": view.positive_meaning,
        "positive_labels_confirmed": view.positive_labels_confirmed,
        "confirmed_for_this_behaviour": _confirmed_for(view, key),
        "archived": view.archived_at is not None,
        # miDataworks facts when the view came with a dataset-version manifest (FR-7); hidden in the UI
        # when absent (BR-011: miDataworks may not exist at all).
        "dataset_version": (
            {
                "version_id": facts.get("version_id"),
                "version_number": facts.get("version_number"),
                "dataset_name": facts.get("dataset_name"),
                "split": facts.get("split"),
            }
            if facts
            else None
        ),
    }


async def adopted(db: AsyncSession, key: str) -> Dict[str, List[Dict[str, Any]]]:
    """Every record that belongs to `key`, by the join (FR-3). SELECT only."""
    prompts = (
        await db.execute(
            select(ProbeJudgePrompt)
            .where(ProbeJudgePrompt.behaviour == key)
            .order_by(ProbeJudgePrompt.name, ProbeJudgePrompt.version.desc())
        )
    ).scalars().all()
    runs = (
        await db.execute(
            select(ProbeMonitorRun)
            .where(ProbeMonitorRun.behaviour == key)
            .order_by(ProbeMonitorRun.created_at.desc())
        )
    ).scalars().all()
    probes = (
        await db.execute(
            select(ProbeMonitor)
            .where(ProbeMonitor.behaviour == key)
            .order_by(ProbeMonitor.created_at.desc())
        )
    ).scalars().all()
    keyed_views = (
        await db.execute(select(ProbeMonitorDataset).where(ProbeMonitorDataset.behaviour == key))
    ).scalars().all()
    links = (
        await db.execute(select(ProbeBehaviourViewLink).where(ProbeBehaviourViewLink.key == key))
    ).scalars().all()
    linked_ids = {link.dataset_id: link for link in links}
    linked_views = (
        (
            await db.execute(
                select(ProbeMonitorDataset).where(ProbeMonitorDataset.id.in_(list(linked_ids)))
            )
        ).scalars().all()
        if linked_ids
        else []
    )

    roles = await roles_of(db, key)
    run_ids = {probe.run_id for probe in probes}
    train_views: Dict[str, Optional[ProbeMonitorDataset]] = {}
    if run_ids:
        pairs = (
            await db.execute(
                select(ProbeMonitorRun.id, ProbeMonitorDataset)
                .join(ProbeMonitorDataset, ProbeMonitorDataset.id == ProbeMonitorRun.train_dataset_id)
                .where(ProbeMonitorRun.id.in_(list(run_ids)))
            )
        ).all()
        train_views = {run_id: view for run_id, view in pairs}

    views: Dict[str, Dict[str, Any]] = {}
    for view in keyed_views:
        link = linked_ids.get(view.id)
        views[view.id] = view_out(view, key=key, linked_by="key", use=link.use if link else None)
    for view in linked_views:
        if view.id not in views:
            views[view.id] = view_out(view, key=key, linked_by="link", use=linked_ids[view.id].use)

    return {
        "prompts": [
            {
                "id": p.id,
                "key": prompt_key(p.name, p.version),
                "title": p.title,
                "locked": p.locked_at is not None,
                "archived": p.archived_at is not None,
            }
            for p in prompts
        ],
        "runs": [
            {
                "id": r.id,
                "status": r.status,
                "model_id": r.model_id,
                "train_dataset_id": r.train_dataset_id,
                "created_at": _iso(r.created_at),
            }
            for r in runs
        ],
        "probes": [
            {
                "id": p.id,
                "run_id": p.run_id,
                "layer": p.layer,
                "rule": p.rule,
                "variant": p.variant,
                "rung": p.rung,
                "behaviour_label": p.behaviour_label,
                "definition_sha256": p.definition_sha256,
                # Feature 039 (FR-16): the identity of the built definition — the one read.
                **identity_fields(p),
                "role": roles[p.id].role if p.id in roles else "candidate",
                "training_view_confirmed": _confirmed_for(train_views.get(p.run_id), key),
                "created_at": _iso(p.created_at),
            }
            for p in probes
        ],
        "views": list(views.values()),
    }


async def detail(db: AsyncSession, key: str) -> Dict[str, Any]:
    head = await get_head(db, key)
    version = await current_version(db, head)
    out = head_out(head, version)
    if version.rule_set_digest is not None:  # provenance of the set, read once for one behaviour
        row = await rule_set_store.get_by_digest(db, version.rule_set_digest)
        out["definition"]["rule_set"] = rule_set_store.describe(row, version.rule_set_digest)
    out["adopted"] = await adopted(db, key)
    out["counts"] = {kind: len(rows) for kind, rows in out["adopted"].items()}
    return out


async def versions(db: AsyncSession, key: str) -> List[Dict[str, Any]]:
    await get_head(db, key)
    rows = (
        await db.execute(
            select(ProbeBehaviourVersion)
            .where(ProbeBehaviourVersion.key == key)
            .order_by(ProbeBehaviourVersion.version.desc())
        )
    ).scalars().all()
    return [version_out(v) for v in rows]


async def history(db: AsyncSession, key: str, *, limit: int = 100, offset: int = 0) -> Dict[str, Any]:
    rows = (
        await db.execute(
            select(ProbeBehaviourHistory)
            .where(ProbeBehaviourHistory.key == key)
            .order_by(ProbeBehaviourHistory.created_at.desc(), ProbeBehaviourHistory.id.desc())
            .limit(limit)
            .offset(offset)
        )
    ).scalars().all()
    total = (
        await db.execute(
            select(func.count()).select_from(ProbeBehaviourHistory).where(ProbeBehaviourHistory.key == key)
        )
    ).scalar_one()
    return {
        "key": key,
        "total": int(total),
        "entries": [
            {
                "id": row.id,
                "subject_kind": row.subject_kind,
                "subject_id": row.subject_id,
                "field": row.field,
                "old_value": row.old_value,
                "new_value": row.new_value,
                "actor": row.actor,
                "approval_id": row.approval_id,
                "reason": row.reason,
                "created_at": _iso(row.created_at),
            }
            for row in rows
        ],
    }


# ── writing ───────────────────────────────────────────────────────────────────────────────────


def _definition(body: Dict[str, Any]) -> Dict[str, Any]:
    return {field: body.get(field) for field in VERSION_FIELDS}


def _arguments(body: Dict[str, Any], **extra: Any) -> Dict[str, Any]:
    """The request as an agent would repeat it: every field it set, JSON values (for a `next`)."""
    values = definition_rules.detector_values(body)
    out: Dict[str, Any] = dict(extra)
    for field in TEXT_FIELDS:
        if body.get(field) is not None:
            out[field] = body[field]
    out.update({field: value for field, value in values.items() if value is not None})
    return out


async def _version_row(db: AsyncSession, key: str, number: int) -> Optional[ProbeBehaviourVersion]:
    return (
        await db.execute(
            select(ProbeBehaviourVersion).where(
                ProbeBehaviourVersion.key == key, ProbeBehaviourVersion.version == number
            )
        )
    ).scalar_one_or_none()


def _stored(version: ProbeBehaviourVersion) -> Dict[str, Any]:
    return {field: getattr(version, field) for field in VERSION_FIELDS}


async def _refuse_or_validate(db: AsyncSession, definition: Dict[str, Any],
                              limits: "definition_rules.Limits") -> List[Dict[str, Any]]:
    """FR-2 on the WHOLE new definition (carried fields included); returns the warnings."""
    definition_rules.validate(definition, definition.get("target_fpr"), limits)
    return await definition_rules.warnings(db, definition)


async def create(db: AsyncSession, body: Dict[str, Any], actor: str) -> Dict[str, Any]:
    """Catalogue a key: the head row and version 1. Adopts every record already using it (D25).

    Feature 040: the version may carry the six detector fields (FR-1), checked by
    `behaviour_detector_definition.validate` (FR-2); the response adds `warnings`."""
    key = body["key"]
    _check_key(key)
    exists = (
        await db.execute(select(ProbeBehaviour.key).where(ProbeBehaviour.key == key))
    ).first()
    if exists is not None:
        raise BehaviourRefused(
            409,
            "behaviour_exists",
            f"{key!r} is already catalogued; edit its meaning to make a new version",
        )
    limits = definition_rules.Limits.from_settings()
    definition = _definition(body)
    definition.update(definition_rules.detector_values(body))
    warnings = await _refuse_or_validate(db, definition, limits)
    definition["rule_set_digest"] = await rule_set_store.resolve(
        db, body.get("rule_set"), key=key, max_bytes=limits.rule_set_max_bytes, actor=actor,
        retry=next_create_with_inline_rule_set(_arguments(body, key=key)),
    )
    db.add(ProbeBehaviour(key=key, current_version=1, created_by=actor))
    await db.flush()
    db.add(ProbeBehaviourVersion(key=key, version=1, created_by=actor, **definition))
    _history(
        db, key=key, subject_kind="behaviour", subject_id=key, field="created",
        old=None, new={"version": 1, **definition}, actor=actor,
    )
    try:
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()
        raise BehaviourRefused(409, "behaviour_exists", f"{key!r} is already catalogued") from exc
    out = await detail(db, key)
    out["warnings"] = warnings
    return out


async def new_version(
    db: AsyncSession, key: str, body: Dict[str, Any], actor: str,
    fields_set: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Version n+1 of the definition (D22). Earlier versions stay readable and unchanged.

    Feature 040 FR-3: with `base_version`, every field the request did not SET (`fields_set`) carries
    over from THAT version's row and a field set to null is removed (FR-4); the response adds the
    field-level `diff` against the base and `warnings`. A base that is not the current version is
    refused `stale_base_version` (D31 / OQ-040-6 (a)). Without `base_version` the body is a full
    replacement, as in 037."""
    head = await get_head(db, key)
    if head.archived_at is not None:
        raise BehaviourRefused(
            409, "behaviour_archived", f"{key!r} is archived; unarchive it before changing its meaning"
        )
    current = await current_version(db, head)
    limits = definition_rules.Limits.from_settings()
    base_number = body.get("base_version")
    sent = set(fields_set if fields_set is not None else (k for k, v in body.items() if v is not None))
    change = _definition(body)
    change.update(definition_rules.detector_values(body))
    arguments = _arguments(body, key=key, **({"base_version": base_number} if base_number else {}))
    if base_number is not None:
        if base_number != head.current_version:
            raise DetectorRefused(
                409, "stale_base_version",
                f"version {base_number} of {key!r} is not the current version ({head.current_version}); "
                f"the definition changed since you read it. Read the current version and send your "
                f"change against it.",
                {"current_version": head.current_version, "base_version": base_number},
                next=[next_list_versions(key)],
            )
        base = await _version_row(db, key, base_number)
        if "rule_set" in sent:
            sent.add("rule_set_digest")
        definition = definition_rules.carry_over(base, change, sent, VERSION_FIELDS)
    else:
        missing = [field for field in ("name", "positive_meaning") if not body.get(field)]
        if missing:
            raise DetectorRefused(
                422, "invalid_field",
                f"{missing[0]} is required: without base_version a new version replaces every field",
                {"field": missing[0], "rule": "required_without_base_version"},
            )
        base = current
        definition = change
    warnings = await _refuse_or_validate(db, definition, limits)
    if base_number is None or "rule_set" in sent:
        definition["rule_set_digest"] = await rule_set_store.resolve(
            db, body.get("rule_set"), key=key, max_bytes=limits.rule_set_max_bytes, actor=actor,
            retry=next_version_with_inline_rule_set(arguments),
        )
    before = _stored(base)
    if not definition_rules.field_diff(_stored(current), definition, VERSION_FIELDS):
        raise BehaviourRefused(
            422, "no_change", f"this is the same text as version {current.version}; nothing to version"
        )
    number = head.current_version + 1
    db.add(ProbeBehaviourVersion(key=key, version=number, created_by=actor, **definition))
    head.current_version = number
    _history(
        db, key=key, subject_kind="version", subject_id=str(number), field="version",
        old=version_out(current), new={"version": number, **definition}, actor=actor,
    )
    await db.commit()
    out = await detail(db, key)
    out["warnings"] = warnings
    out["base_version"] = base.version
    out["diff"] = definition_rules.field_diff(before, definition, VERSION_FIELDS)
    return out


async def edit_version(db: AsyncSession, key: str, version: int, body: Dict[str, Any], actor: str) -> None:  # noqa: ARG001
    """Always refused (FR-2): a version is never changed. Here so the refusal has one wording."""
    raise BehaviourRefused(
        409,
        "version_immutable",
        f"version {version} of {key!r} cannot be changed; make a new version instead, so probes "
        f"exported with version {version} keep the meaning they were exported with",
    )


async def _probe(db: AsyncSession, probe_id: str) -> ProbeMonitor:
    probe = (
        await db.execute(select(ProbeMonitor).where(ProbeMonitor.id == probe_id))
    ).scalar_one_or_none()
    if probe is None:
        raise BehaviourRefused(404, "probe_not_found", f"probe {probe_id} not found")
    return probe


async def training_view_of(db: AsyncSession, probe: ProbeMonitor) -> Optional[ProbeMonitorDataset]:
    return (
        await db.execute(
            select(ProbeMonitorDataset)
            .join(ProbeMonitorRun, ProbeMonitorRun.train_dataset_id == ProbeMonitorDataset.id)
            .where(ProbeMonitorRun.id == probe.run_id)
        )
    ).scalar_one_or_none()


async def set_role(
    db: AsyncSession,
    key: str,
    probe_id: str,
    role: str,
    actor: str,
    reason: Optional[str] = None,
) -> Dict[str, Any]:
    """Primary, candidate or retired (FR-6), checked per FR-9."""
    head = await get_head(db, key)
    probe = await _probe(db, probe_id)
    if probe.behaviour != key:
        raise BehaviourRefused(
            422,
            "key_mismatch",
            f"probe {probe_id} detects {probe.behaviour!r}, not {key!r}; only a probe whose own "
            f"behaviour key is {key!r} can have a role in it. Set the probe's behaviour first.",
            {"probe_behaviour": probe.behaviour, "behaviour": key},
        )
    if head.archived_at is not None:
        raise BehaviourRefused(
            409, "behaviour_archived", f"{key!r} is archived; its probes' roles are frozen"
        )
    existing = (
        await db.execute(
            select(ProbeBehaviourProbeRole).where(ProbeBehaviourProbeRole.probe_id == probe_id)
        )
    ).scalar_one_or_none()
    old_role = existing.role if existing is not None and existing.key == key else "candidate"
    if role == "primary":
        view = await training_view_of(db, probe)
        if not _confirmed_for(view, key):
            raise BehaviourRefused(
                422,
                "training_view_unconfirmed",
                f"probe {probe_id} was trained on "
                f"{(view.name + ' (' + view.id + ')') if view is not None else 'a view that no longer exists'}, "
                f"whose positive labels nobody has confirmed are {key!r}. Confirm the view first "
                f"(Datasets, Confirm positive labels); a send from miDataworks with this behaviour "
                f"counts as confirmed.",
            )
        current = await primary_probe(db, key)
        if current is not None and current.id != probe_id:
            raise BehaviourRefused(
                409,
                "primary_exists",
                f"{current.id} is already the primary probe of {key!r}; make it a candidate or "
                f"retire it first",
                {"primary_probe_id": current.id},
            )
    if existing is None:
        db.add(ProbeBehaviourProbeRole(probe_id=probe_id, key=key, role=role, updated_by=actor))
    else:
        existing.key = key
        existing.role = role
        existing.updated_by = actor
    _history(
        db, key=key, subject_kind="probe", subject_id=probe_id, field="role",
        old=old_role, new=role, actor=actor, reason=reason,
    )
    try:
        await db.commit()
    except IntegrityError as exc:  # the partial unique index, under a race
        await db.rollback()
        raise BehaviourRefused(
            409, "primary_exists", f"{key!r} already has a primary probe"
        ) from exc
    return await detail(db, key)


async def _view(db: AsyncSession, dataset_id: str) -> ProbeMonitorDataset:
    view = (
        await db.execute(select(ProbeMonitorDataset).where(ProbeMonitorDataset.id == dataset_id))
    ).scalar_one_or_none()
    if view is None:
        raise BehaviourRefused(404, "dataset_not_found", f"probe dataset {dataset_id} not found")
    return view


async def link_view(
    db: AsyncSession, key: str, dataset_id: str, use: Optional[str], actor: str, reason: Optional[str] = None
) -> Dict[str, Any]:
    await get_head(db, key)
    view = await _view(db, dataset_id)
    existing = (
        await db.execute(
            select(ProbeBehaviourViewLink).where(
                ProbeBehaviourViewLink.key == key, ProbeBehaviourViewLink.dataset_id == dataset_id
            )
        )
    ).scalar_one_or_none()
    if existing is not None:
        raise BehaviourRefused(409, "already_linked", f"{view.name} is already linked to {key!r}")
    chosen = use or view.role
    db.add(ProbeBehaviourViewLink(key=key, dataset_id=dataset_id, use=chosen, created_by=actor))
    _history(
        db, key=key, subject_kind="view", subject_id=dataset_id, field="link",
        old=None, new={"use": chosen}, actor=actor, reason=reason,
    )
    await db.commit()
    return await detail(db, key)


async def unlink_view(
    db: AsyncSession, key: str, dataset_id: str, actor: str, reason: Optional[str] = None
) -> Dict[str, Any]:
    await get_head(db, key)
    link = (
        await db.execute(
            select(ProbeBehaviourViewLink).where(
                ProbeBehaviourViewLink.key == key, ProbeBehaviourViewLink.dataset_id == dataset_id
            )
        )
    ).scalar_one_or_none()
    if link is None:
        raise BehaviourRefused(
            404, "not_linked",
            f"{dataset_id} has no explicit link to {key!r} (a view whose own behaviour is {key!r} "
            f"is listed by its key and cannot be unlinked)",
        )
    _history(
        db, key=key, subject_kind="view", subject_id=dataset_id, field="unlink",
        old={"use": link.use}, new=None, actor=actor, reason=reason,
    )
    await db.delete(link)
    await db.commit()
    return await detail(db, key)


async def confirm_positive(db: AsyncSession, dataset_id: str, key: str, actor: str, *, origin: str = "operator") -> Dict[str, Any]:
    """D48: confirm, once per view, that its positive labels are behaviour `key`."""
    _check_key(key)
    view = await _view(db, dataset_id)
    if view.role == "calibration":
        raise BehaviourRefused(
            422, "calibration_view",
            f"{view.name} is a calibration set: it supplies negatives for the false-positive rate "
            f"and has no positive labels to confirm",
        )
    head = (await db.execute(select(ProbeBehaviour).where(ProbeBehaviour.key == key))).scalar_one_or_none()
    record = confirmation_record(key, head.current_version if head is not None else None, actor, origin)
    old = view.positive_labels_confirmed
    view.positive_labels_confirmed = record
    _history(
        db, key=key, subject_kind="dataset", subject_id=dataset_id, field="confirmed",
        old=old, new=record, actor=actor,
    )
    await db.commit()
    return view_out(view, key=key, linked_by="key" if view.behaviour == key else "confirmation")


def confirmation_record(key: str, version: Optional[int], actor: str, origin: str) -> Dict[str, Any]:
    return {"key": key, "version": version, "by": actor, "at": utc_now().isoformat(), "origin": origin}


async def set_archived(db: AsyncSession, key: str, archived: bool, actor: str, reason: Optional[str] = None) -> Dict[str, Any]:
    head = await get_head(db, key)
    if (head.archived_at is not None) == archived:
        return await detail(db, key)
    old = _iso(head.archived_at)
    head.archived_at = utc_now() if archived else None
    head.archived_by = actor if archived else None
    _history(
        db, key=key, subject_kind="behaviour", subject_id=key,
        field="archived" if archived else "unarchived",
        old=old, new=_iso(head.archived_at), actor=actor, reason=reason,
    )
    await db.commit()
    return await detail(db, key)


async def delete_if_unused(db: AsyncSession, key: str, actor: str) -> None:
    """FR-5: only a behaviour no record uses. Its history stays (no FK)."""
    head = await get_head(db, key)
    counts = (await usage_counts(db, [key])).get(key, {})
    links = (
        await db.execute(
            select(func.count()).select_from(ProbeBehaviourViewLink).where(ProbeBehaviourViewLink.key == key)
        )
    ).scalar_one()
    in_use = {kind: n for kind, n in counts.items() if n}
    if links:
        in_use["linked_views"] = int(links)
    # Feature 041 (PT-9): a behaviour with blend evidence is archived, not deleted — a report keys on the
    # behaviour by value (no FK, T6), so deleting the head would orphan it silently.
    from ..models.probe_blend import ProbeBlendReport

    reports = (
        await db.execute(select(func.count()).select_from(ProbeBlendReport).where(ProbeBlendReport.behaviour_key == key))
    ).scalar_one()
    if reports:
        in_use["blend_reports"] = int(reports)
    if in_use:
        described = ", ".join(f"{n} {kind.replace('_', ' ')}" for kind, n in in_use.items())
        raise BehaviourRefused(
            409, "behaviour_in_use",
            f"{key!r} is used by {described}; archive it instead (every record keeps its key)",
            {"in_use": in_use},
        )
    _history(
        db, key=key, subject_kind="behaviour", subject_id=key, field="deleted",
        old={"current_version": head.current_version}, new=None, actor=actor,
    )
    await db.delete(head)
    await db.commit()


async def record_probe_key_change(
    db: AsyncSession, probe_id: str, old: Optional[str], new: Optional[str], actor: str
) -> None:
    """Part A's `PATCH /probes/{id}/behaviour` moved a probe between behaviours (FR-8, R-4).

    One history row on the old key and one on the new key; a role held under the old key is dropped,
    because a role belongs to a behaviour and a probe has one behaviour (D24). Does NOT commit: the
    caller's commit covers the key change and these rows together.
    """
    if old == new:
        return
    role = (
        await db.execute(
            select(ProbeBehaviourProbeRole).where(ProbeBehaviourProbeRole.probe_id == probe_id)
        )
    ).scalar_one_or_none()
    dropped = None
    if role is not None and role.key != new:
        dropped = role.role
        await db.delete(role)
    for key in (old, new):
        if key:
            _history(
                db, key=key, subject_kind="probe", subject_id=probe_id, field="behaviour",
                old={"behaviour": old, "role": dropped} if key == old else {"behaviour": old},
                new={"behaviour": new}, actor=actor,
            )
