"""The judge-prompt library: create, version, clone, archive, import — and rung-3 eligibility.

Operator decisions D1/D2 (2026-10-21); PADR IDL-57. The REST routes in
`api/v1/endpoints/probe_judge_prompts.py` and the MCP tools in `mcp_server/tools/judge_prompts.py`
are thin: every rule lives here, so the two surfaces cannot disagree about what is allowed.

THE RULES, each enforced below and tested in `test_judge_prompt_library.py`:

1. A version is identified by ``name/vN`` and its content hash. ``(name, version)`` is unique.
2. A version NOBODY HAS USED may be edited in place (its hash is recomputed) or deleted.
3. The first judge run that names a version LOCKS it. A locked version's content, behaviour and
   provenance are frozen (here, and by a database trigger); only its title, description and
   archive flag change. Editing a locked version means making a new version.
4. An archived version is refused for NEW runs and kept for every existing one.
5. A new version keeps its prompt's behaviour; a different behaviour is a different prompt (clone).
6. RUNG 3 COUNTS A JUDGE RUN ONLY WHEN THE PROMPT'S BEHAVIOUR IS THE PROBE'S (`rung_eligibility`).
   A mismatched run is allowed, stored and shown — it just never raises a rung.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.clock import utc_now
from ..models.probe_judge_prompt import ProbeJudgePrompt
from ..models.probe_monitor import ProbeMonitorJudgeRun
from .judge_prompt_forms import (
    DEFAULT_MAX_INTERACTION_CHARS,
    MAX_INTERACTION_CHARS,
    MIN_INTERACTION_CHARS,
    TEMPLATE_PLACEHOLDERS,
    TemplateInvalid,
    content_hash,
    describe_answer_form,
    is_slug,
    normalise_answer_form,
    prompt_key,
    reply_shape_warnings,
    validate_user_template,
)

logger = logging.getLogger(__name__)

#: The export document's format string (`export_document` / `import_document`).
EXPORT_FORMAT = "mistudio.judge-prompt/v1"
#: miDataworks' rubric export (its `schemas/labeling.RUBRIC_EXPORT_FORMAT`).
RUBRIC_FORMAT = "midataworks.rubric/v1"

#: The content fields. Changing any of them on a locked version is refused.
CONTENT_FIELDS = ("system_prompt", "user_template", "answer_form", "max_interaction_chars")
#: Fields that may change at any time.
DISPLAY_FIELDS = ("title", "description")


class PromptRefused(Exception):
    """A request the library will not carry out, with an HTTP status and a plain reason."""

    def __init__(self, status: int, code: str, message: str, details: Optional[dict] = None):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.details = details or {}

    def as_detail(self) -> Dict[str, Any]:
        detail = {"code": self.code, "message": self.message}
        if self.details:
            detail["details"] = self.details
        return detail


# ── validation ────────────────────────────────────────────────────────────────────────────────


def _validation_message(exc: ValidationError) -> str:
    parts = []
    for error in exc.errors():
        where = ".".join(str(p) for p in error.get("loc", ()) if p not in ("rating", "binary", "labels"))
        message = str(error.get("msg", "")).removeprefix("Value error, ")
        parts.append(f"{where}: {message}" if where else message)
    return "; ".join(parts)


def validated_content(
    *,
    name: str,
    behaviour: str,
    system_prompt: str,
    user_template: str,
    answer_form: Any,
    max_interaction_chars: int,
) -> Dict[str, Any]:
    """The content fields, checked and normalised — or `PromptRefused` (422) saying why."""
    if not is_slug(name):
        raise PromptRefused(
            422, "invalid_name",
            f"the name {name!r} must be 2-64 lower-case letters, digits and hyphens, starting "
            f"and ending with a letter or digit (for example '<prompt-name>')",
        )
    if not is_slug(behaviour):
        raise PromptRefused(
            422, "invalid_behaviour",
            f"the behaviour {behaviour!r} must be 2-64 lower-case letters, digits and hyphens "
            f"(for example '<behaviour-key>')",
        )
    if not system_prompt or not system_prompt.strip():
        raise PromptRefused(422, "empty_system_prompt", "the instructions to the judge are empty")
    try:
        validate_user_template(user_template)
    except TemplateInvalid as bad:
        raise PromptRefused(422, "invalid_template", str(bad)) from bad
    try:
        form = normalise_answer_form(answer_form)
    except ValidationError as bad:
        raise PromptRefused(
            422, "invalid_answer_form", f"the answer form is not valid: {_validation_message(bad)}"
        ) from bad
    chars = int(max_interaction_chars)
    if not MIN_INTERACTION_CHARS <= chars <= MAX_INTERACTION_CHARS:
        raise PromptRefused(
            422, "invalid_max_chars",
            f"the judge must see between {MIN_INTERACTION_CHARS} and {MAX_INTERACTION_CHARS} "
            f"characters of each row; {chars} was asked for",
        )
    return {
        "system_prompt": system_prompt,
        "user_template": user_template,
        "answer_form": form,
        "max_interaction_chars": chars,
    }


def _hash_of(name: str, version: int, content: Dict[str, Any]) -> str:
    return content_hash(name=name, version=version, **content)


# ── reading ───────────────────────────────────────────────────────────────────────────────────


async def usage_counts(db: AsyncSession, prompt_ids: Iterable[str]) -> Dict[str, int]:
    """How many judge runs name each prompt — the "used by N runs" badge."""
    ids = list(prompt_ids)
    if not ids:
        return {}
    rows = await db.execute(
        select(ProbeMonitorJudgeRun.prompt_id, func.count(ProbeMonitorJudgeRun.id))
        .where(ProbeMonitorJudgeRun.prompt_id.in_(ids))
        .group_by(ProbeMonitorJudgeRun.prompt_id)
    )
    return {prompt_id: int(count) for prompt_id, count in rows.all()}


def is_default_prompt(name: str, version: int) -> bool:
    """The version `resolve_for_run` uses when a run names no prompt."""
    from .judge_prompt_seeds import DEFAULT_PROMPT_NAME, DEFAULT_PROMPT_VERSION

    return name == DEFAULT_PROMPT_NAME and version == DEFAULT_PROMPT_VERSION


def summary(prompt: ProbeJudgePrompt, used_by: int = 0) -> Dict[str, Any]:
    """ONE serialiser for a library row, used by every route and every list.

    (memory: one-list-needs-one-serialiser — two shapes for one list drift apart.)
    """
    return {
        "id": prompt.id,
        "name": prompt.name,
        "version": prompt.version,
        "key": prompt_key(prompt.name, prompt.version),
        "title": prompt.title,
        "description": prompt.description,
        "behaviour": prompt.behaviour,
        "behaviour_label": prompt.behaviour_label,
        "system_prompt": prompt.system_prompt,
        "user_template": prompt.user_template,
        "answer_form": prompt.answer_form,
        "answer_form_summary": describe_answer_form(prompt.answer_form),
        "max_interaction_chars": prompt.max_interaction_chars,
        "content_hash": prompt.content_hash,
        "source": prompt.source,
        "source_ref": prompt.source_ref,
        "parent_id": prompt.parent_id,
        "created_by": prompt.created_by,
        "created_at": prompt.created_at,
        "updated_at": prompt.updated_at,
        "locked": prompt.locked_at is not None,
        "locked_at": prompt.locked_at,
        "archived": prompt.archived_at is not None,
        "archived_at": prompt.archived_at,
        "used_by_runs": int(used_by),
        # Feature 040 FR-9: the version a judge run uses when no prompt is named, so a screen can
        # label its "default" choice from this row instead of spelling a behaviour in its source.
        "is_default": is_default_prompt(prompt.name, prompt.version),
        "warnings": reply_shape_warnings(
            prompt.system_prompt, prompt.user_template, prompt.answer_form
        ),
    }


async def list_prompts(
    db: AsyncSession,
    *,
    behaviour: Optional[str] = None,
    name: Optional[str] = None,
    include_archived: bool = False,
) -> List[Dict[str, Any]]:
    query = select(ProbeJudgePrompt).order_by(
        ProbeJudgePrompt.name.asc(), ProbeJudgePrompt.version.desc()
    )
    if behaviour:
        query = query.where(ProbeJudgePrompt.behaviour == behaviour)
    if name:
        query = query.where(ProbeJudgePrompt.name == name)
    if not include_archived:
        query = query.where(ProbeJudgePrompt.archived_at.is_(None))
    rows = list((await db.execute(query)).scalars().all())
    counts = await usage_counts(db, [row.id for row in rows])
    return [summary(row, counts.get(row.id, 0)) for row in rows]


async def get_prompt(db: AsyncSession, prompt_id: str) -> ProbeJudgePrompt:
    row = (
        await db.execute(select(ProbeJudgePrompt).where(ProbeJudgePrompt.id == prompt_id))
    ).scalar_one_or_none()
    if row is None:
        raise PromptRefused(404, "not_found", f"judge prompt {prompt_id} not found")
    return row


async def detail(db: AsyncSession, prompt_id: str) -> Dict[str, Any]:
    """One version with its whole history (every version of the name, newest first)."""
    row = await get_prompt(db, prompt_id)
    versions = list(
        (
            await db.execute(
                select(ProbeJudgePrompt)
                .where(ProbeJudgePrompt.name == row.name)
                .order_by(ProbeJudgePrompt.version.desc())
            )
        ).scalars().all()
    )
    counts = await usage_counts(db, [v.id for v in versions])
    out = summary(row, counts.get(row.id, 0))
    out["versions"] = [
        {
            "id": v.id,
            "version": v.version,
            "key": prompt_key(v.name, v.version),
            "content_hash": v.content_hash,
            "locked": v.locked_at is not None,
            "archived": v.archived_at is not None,
            "used_by_runs": counts.get(v.id, 0),
            "created_at": v.created_at,
            "created_by": v.created_by,
        }
        for v in versions
    ]
    return out


async def _name_exists(db: AsyncSession, name: str) -> bool:
    return (
        await db.execute(select(ProbeJudgePrompt.id).where(ProbeJudgePrompt.name == name).limit(1))
    ).first() is not None


async def _next_version(db: AsyncSession, name: str) -> int:
    current = (
        await db.execute(
            select(func.max(ProbeJudgePrompt.version)).where(ProbeJudgePrompt.name == name)
        )
    ).scalar()
    return int(current or 0) + 1


def _source_for(actor: str) -> str:
    return "agent" if actor.startswith("agent:") else "operator"


# ── writing ───────────────────────────────────────────────────────────────────────────────────


async def _insert(
    db: AsyncSession,
    *,
    name: str,
    version: int,
    title: str,
    description: Optional[str],
    behaviour: str,
    content: Dict[str, Any],
    actor: str,
    source: str,
    source_ref: Optional[dict] = None,
    parent_id: Optional[str] = None,
    behaviour_label: Optional[str] = None,
) -> ProbeJudgePrompt:
    if not title or not title.strip():
        raise PromptRefused(422, "empty_title", "give the prompt a title people will recognise")
    row = ProbeJudgePrompt(
        name=name,
        version=version,
        title=title.strip(),
        description=(description or None),
        behaviour=behaviour,
        behaviour_label=(behaviour_label or None),
        content_hash=_hash_of(name, version, content),
        source=source,
        source_ref=source_ref,
        parent_id=parent_id,
        created_by=actor,
        **content,
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return row


async def create_prompt(db: AsyncSession, payload: Dict[str, Any], actor: str) -> ProbeJudgePrompt:
    """Version 1 of a NEW prompt. A name already in the library is refused — make a new version."""
    name = payload["name"]
    content = validated_content(
        name=name,
        behaviour=payload["behaviour"],
        system_prompt=payload["system_prompt"],
        user_template=payload.get("user_template") or "{interaction}",
        answer_form=payload["answer_form"],
        max_interaction_chars=payload.get("max_interaction_chars") or DEFAULT_MAX_INTERACTION_CHARS,
    )
    if await _name_exists(db, name):
        raise PromptRefused(
            409, "name_exists",
            f"a judge prompt called {name!r} already exists; make a new version of it, or "
            f"choose another name",
        )
    return await _insert(
        db,
        name=name,
        version=1,
        title=payload["title"],
        description=payload.get("description"),
        behaviour=payload["behaviour"],
        behaviour_label=payload.get("behaviour_label"),
        content=content,
        actor=actor,
        source=_source_for(actor),
    )


def _merged_content(base: ProbeJudgePrompt, changes: Dict[str, Any]) -> Dict[str, Any]:
    return {
        field: changes[field] if changes.get(field) is not None else getattr(base, field)
        for field in CONTENT_FIELDS
    }


async def update_prompt(
    db: AsyncSession, prompt_id: str, changes: Dict[str, Any], actor: str  # noqa: ARG001
) -> ProbeJudgePrompt:
    """Edit in place. Title and description always; content and behaviour only while UNUSED."""
    row = await get_prompt(db, prompt_id)
    content_change = any(changes.get(field) is not None for field in CONTENT_FIELDS)
    behaviour_change = changes.get("behaviour") is not None and changes["behaviour"] != row.behaviour
    if (content_change or behaviour_change) and row.locked_at is not None:
        used = (await usage_counts(db, [row.id])).get(row.id, 0)
        raise PromptRefused(
            409, "prompt_locked",
            f"{prompt_key(row.name, row.version)} has been used by {used} judge run(s), so its "
            f"text, answer form and behaviour are fixed. Make a new version instead; the runs "
            f"that used this one keep pointing at it.",
            {"used_by_runs": used},
        )
    if content_change or behaviour_change:
        behaviour = changes.get("behaviour") or row.behaviour
        merged = _merged_content(row, changes)
        content = validated_content(name=row.name, behaviour=behaviour, **merged)
        for field, value in content.items():
            setattr(row, field, value)
        row.behaviour = behaviour
        row.content_hash = _hash_of(row.name, row.version, content)
    if changes.get("title") is not None:
        if not changes["title"].strip():
            raise PromptRefused(422, "empty_title", "the title cannot be blank")
        row.title = changes["title"].strip()
    if changes.get("description") is not None:
        row.description = changes["description"] or None
    if changes.get("behaviour_label") is not None:
        row.behaviour_label = changes["behaviour_label"] or None
    await db.commit()
    await db.refresh(row)
    return row


async def new_version(
    db: AsyncSession, prompt_id: str, changes: Dict[str, Any], actor: str
) -> ProbeJudgePrompt:
    """The next version of a prompt's name, from `prompt_id` plus `changes`."""
    base = await get_prompt(db, prompt_id)
    if changes.get("behaviour") and changes["behaviour"] != base.behaviour:
        raise PromptRefused(
            422, "behaviour_changes_identity",
            f"a new version judges the same behaviour as {prompt_key(base.name, base.version)} "
            f"({base.behaviour!r}). To judge {changes['behaviour']!r}, clone it under a new name.",
        )
    content = validated_content(
        name=base.name, behaviour=base.behaviour, **_merged_content(base, changes)
    )
    unchanged = all(
        normalise_answer_form(content["answer_form"]) == normalise_answer_form(base.answer_form)
        if field == "answer_form"
        else content[field] == getattr(base, field)
        for field in CONTENT_FIELDS
    )
    if unchanged:
        raise PromptRefused(
            422, "nothing_changed",
            "the new version would send exactly what the old one sends; change the text, the "
            "template, the answer form or the length first",
        )
    return await _insert(
        db,
        name=base.name,
        version=await _next_version(db, base.name),
        title=changes.get("title") or base.title,
        description=changes.get("description") if changes.get("description") is not None else base.description,
        behaviour=base.behaviour,
        behaviour_label=changes.get("behaviour_label") or base.behaviour_label,
        content=content,
        actor=actor,
        source=_source_for(actor),
        parent_id=base.id,
    )


async def clone_prompt(
    db: AsyncSession, prompt_id: str, changes: Dict[str, Any], actor: str
) -> ProbeJudgePrompt:
    """Version 1 of a NEW name, copied from `prompt_id` — how a new behaviour starts."""
    base = await get_prompt(db, prompt_id)
    name = changes["name"]
    behaviour = changes.get("behaviour") or base.behaviour
    content = validated_content(name=name, behaviour=behaviour, **_merged_content(base, changes))
    if await _name_exists(db, name):
        raise PromptRefused(409, "name_exists", f"a judge prompt called {name!r} already exists")
    return await _insert(
        db,
        name=name,
        version=1,
        title=changes.get("title") or f"{base.title} (copy)",
        description=changes.get("description") if changes.get("description") is not None else base.description,
        behaviour=behaviour,
        behaviour_label=changes.get("behaviour_label")
        or (base.behaviour_label if behaviour == base.behaviour else None),
        content=content,
        actor=actor,
        source=_source_for(actor),
        parent_id=base.id,
    )


async def set_archived(db: AsyncSession, prompt_id: str, archived: bool) -> ProbeJudgePrompt:
    row = await get_prompt(db, prompt_id)
    row.archived_at = (row.archived_at or utc_now()) if archived else None
    await db.commit()
    await db.refresh(row)
    return row


async def delete_prompt(db: AsyncSession, prompt_id: str) -> None:
    """Only a version no run has used. A used one is the instrument of a recorded measurement."""
    row = await get_prompt(db, prompt_id)
    used = (await usage_counts(db, [row.id])).get(row.id, 0)
    if row.locked_at is not None or used:
        raise PromptRefused(
            409, "prompt_locked",
            f"{prompt_key(row.name, row.version)} has been used by {used} judge run(s) and "
            f"cannot be deleted. Archive it to hide it from new runs.",
            {"used_by_runs": used},
        )
    await db.delete(row)
    await db.commit()


async def resolve_for_run(
    db: AsyncSession, prompt_id: Optional[str]
) -> ProbeJudgePrompt:
    """The version a new judge run will use. Omitted → ``stakes-rating/v1`` (the pre-library
    behaviour, so existing callers keep their meaning). Archived → refused."""
    if prompt_id:
        row = await get_prompt(db, prompt_id)
    else:
        from .judge_prompt_seeds import DEFAULT_PROMPT_NAME, DEFAULT_PROMPT_VERSION, ensure_seeds

        def _default():
            return select(ProbeJudgePrompt).where(
                ProbeJudgePrompt.name == DEFAULT_PROMPT_NAME,
                ProbeJudgePrompt.version == DEFAULT_PROMPT_VERSION,
            )

        row = (await db.execute(_default())).scalar_one_or_none()
        if row is None:
            # Startup loads the seed files; this covers a schema built after startup (the unit
            # suite, a database restored under a running API).
            await ensure_seeds(db)
            row = (await db.execute(_default())).scalar_one_or_none()
        if row is None:
            raise PromptRefused(
                409, "default_prompt_missing",
                f"no prompt was named and the default, "
                f"{prompt_key(DEFAULT_PROMPT_NAME, DEFAULT_PROMPT_VERSION)}, is not in the library "
                f"(its seed file did not load; the API log says why). Choose a prompt.",
            )
    if row.archived_at is not None:
        raise PromptRefused(
            409, "prompt_archived",
            f"{prompt_key(row.name, row.version)} is archived and cannot start new runs; "
            f"unarchive it or choose another version",
        )
    return row


def lock(row: ProbeJudgePrompt) -> None:
    """Set `locked_at` once. Called in the SAME transaction that inserts the judge run, so a
    run can never exist that names an unlocked version."""
    if row.locked_at is None:
        row.locked_at = utc_now()


# ── export / import ───────────────────────────────────────────────────────────────────────────


def export_document(row: ProbeJudgePrompt) -> Dict[str, Any]:
    """A portable copy that reproduces this version's hash on any install."""
    return {
        "format": EXPORT_FORMAT,
        "name": row.name,
        "version": row.version,
        "title": row.title,
        "description": row.description,
        "behaviour": row.behaviour,
        "behaviour_label": row.behaviour_label,
        "system_prompt": row.system_prompt,
        "user_template": row.user_template,
        "answer_form": row.answer_form,
        "max_interaction_chars": row.max_interaction_chars,
        "content_hash": row.content_hash,
    }


@dataclass(frozen=True)
class ParsedDocument:
    """A `mistudio.judge-prompt/v1` document, checked: what `import_document` and the seed loader
    (`judge_prompt_seeds.ensure_seeds`) store. One reader, so a seed file and an imported file can
    never be held to different rules."""

    name: str
    version: int
    title: str
    description: Optional[str]
    behaviour: str
    behaviour_label: Optional[str]
    content: Dict[str, Any]
    content_hash: str

    @property
    def key(self) -> str:
        return prompt_key(self.name, self.version)


def parse_document(document: Any) -> ParsedDocument:
    """Check an exported document and derive its hash — or `PromptRefused` (422) saying why.

    A ``content_hash`` in the document is optional; when present it must equal the hash of the
    text, so a document edited after export is refused rather than stored under the wrong hash.
    An omitted ``user_template`` or ``max_interaction_chars`` takes the same default as a prompt
    made with New prompt (`create_prompt`), so a hand-written seed file need not spell them out.
    """
    if not isinstance(document, dict):
        raise PromptRefused(
            422, "wrong_format",
            f"expected a {EXPORT_FORMAT} document (a JSON object); got {type(document).__name__}",
        )
    if document.get("format") != EXPORT_FORMAT:
        raise PromptRefused(
            422, "wrong_format",
            f"expected a {EXPORT_FORMAT} document; got format {document.get('format')!r}",
        )
    name = str(document.get("name", ""))
    version = document.get("version")
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise PromptRefused(422, "invalid_version", "the document's version must be a whole number from 1")
    try:
        chars = int(document.get("max_interaction_chars") or DEFAULT_MAX_INTERACTION_CHARS)
    except (TypeError, ValueError) as bad:
        raise PromptRefused(
            422, "invalid_max_chars", "max_interaction_chars must be a whole number"
        ) from bad
    content = validated_content(
        name=name,
        behaviour=str(document.get("behaviour", "")),
        system_prompt=str(document.get("system_prompt", "")),
        user_template=str(document.get("user_template") or "{interaction}"),
        answer_form=document.get("answer_form"),
        max_interaction_chars=chars,
    )
    derived = _hash_of(name, version, content)
    if document.get("content_hash") and document["content_hash"] != derived:
        raise PromptRefused(
            422, "hash_mismatch",
            f"the document says its hash is {str(document['content_hash'])[:12]}… but its text "
            f"hashes to {derived[:12]}…; it was edited after export, so it is not the version "
            f"it names",
        )
    title = str(document.get("title") or name)
    if not title.strip():
        raise PromptRefused(422, "empty_title", "give the prompt a title people will recognise")
    return ParsedDocument(
        name=name,
        version=version,
        title=title,
        description=document.get("description"),
        behaviour=str(document["behaviour"]),
        behaviour_label=document.get("behaviour_label"),
        content=content,
        content_hash=derived,
    )


async def import_document(db: AsyncSession, document: Dict[str, Any], actor: str) -> Tuple[ProbeJudgePrompt, bool]:
    """Store an exported version under ITS OWN name and version. Returns (row, created).

    The version number is kept because ``name/vN`` is what runs record: re-numbering on import
    would make two installs disagree about which instrument ``pii-presence/v2`` is. The same
    document twice returns the stored row; a different document under a stored key is refused.
    """
    parsed = parse_document(document)
    name, version, content, derived = parsed.name, parsed.version, parsed.content, parsed.content_hash
    existing = (
        await db.execute(
            select(ProbeJudgePrompt).where(
                ProbeJudgePrompt.name == name, ProbeJudgePrompt.version == version
            )
        )
    ).scalar_one_or_none()
    if existing is not None:
        if existing.content_hash == derived:
            return existing, False
        raise PromptRefused(
            409, "version_exists",
            f"{prompt_key(name, version)} already exists here with different content; import it "
            f"under another name, or as a new version",
        )
    row = await _insert(
        db,
        name=name,
        version=version,
        title=parsed.title,
        description=parsed.description,
        behaviour=parsed.behaviour,
        behaviour_label=parsed.behaviour_label,
        content=content,
        actor=actor,
        source="import",
        source_ref={"format": EXPORT_FORMAT, "content_hash": derived},
    )
    return row, True


# ── miDataworks rubrics (optional; standalone first) ─────────────────────────────────────────


def _rubric_template(content: str, field_map: Dict[str, str], literals: Dict[str, str], fields: Sequence[str]) -> str:
    """A rubric message (a Python format string over the rubric's fields) as a library template.

    Every rubric field must be mapped to a placeholder or given a fixed value. Literal text that
    would look like a placeholder here is refused, because it would be filled where the rubric
    meant it literally.
    """
    import re
    import string

    out: List[str] = []
    for literal, field_name, format_spec, conversion in string.Formatter().parse(content):
        if re.search(r"\{[A-Za-z_][A-Za-z0-9_]*\}", literal):
            raise PromptRefused(
                422, "rubric_unmappable",
                "the rubric contains literal text in braces that this library would read as a "
                "placeholder; it cannot be imported unchanged",
            )
        out.append(literal)
        if field_name is None:
            continue
        if format_spec or conversion:
            raise PromptRefused(
                422, "rubric_unmappable",
                f"the rubric formats {{{field_name}}} with a conversion or format spec, which "
                f"the library's template does not support",
            )
        if field_name in literals:
            out.append(literals[field_name])
        elif field_name in field_map:
            out.append("{" + field_map[field_name] + "}")
        elif field_name in TEMPLATE_PLACEHOLDERS:
            out.append("{" + field_name + "}")
        else:
            raise PromptRefused(
                422, "rubric_field_unmapped",
                f"the rubric uses the field {{{field_name}}}. Map it to one of "
                f"{', '.join(sorted(TEMPLATE_PLACEHOLDERS))} or give it a fixed value",
                {"rubric_fields": list(fields)},
            )
    return "".join(out)


def rubric_to_prompt(
    document: Dict[str, Any],
    *,
    name: str,
    title: str,
    behaviour: str,
    field_map: Optional[Dict[str, str]] = None,
    literals: Optional[Dict[str, str]] = None,
    positive_verdicts: Optional[List[str]] = None,
    positive_from: Optional[int] = None,
) -> Dict[str, Any]:
    """A miDataworks rubric export as a library prompt payload — or `PromptRefused` saying what
    has no mapping. Pure: nothing is fetched or stored here.

    Supported: ``binary`` and ``pointwise`` rubrics with one optional system message and one user
    message. A pointwise rubric whose verdicts are consecutive whole numbers becomes a RATING
    (``positive_from`` required); any other verdict set becomes a LABEL SET (``positive_verdicts``
    required). Matching stays case-sensitive, as miDataworks' own parsers are, and the reply is
    read the way the rubric says: ``verdict_line_v1`` → last line ``VERDICT: x``; ``json_v1`` →
    the ``verdict`` key.
    """
    field_map = dict(field_map or {})
    literals = dict(literals or {})
    if document.get("format") != RUBRIC_FORMAT:
        raise PromptRefused(
            422, "wrong_format",
            f"expected a {RUBRIC_FORMAT} document; got format {document.get('format')!r}",
        )
    body = document.get("body") or {}
    style = body.get("style")
    if style not in ("binary", "pointwise"):
        raise PromptRefused(
            422, "rubric_style_unsupported",
            f"a {style!r} rubric has no mapping: a judge prompt reads ONE row and gives one "
            f"answer, so only binary and pointwise rubrics can be imported",
        )
    bad_targets = sorted(v for v in field_map.values() if v not in TEMPLATE_PLACEHOLDERS)
    if bad_targets:
        raise PromptRefused(
            422, "invalid_field_map",
            f"{bad_targets} are not placeholders; use {', '.join(sorted(TEMPLATE_PLACEHOLDERS))}",
        )
    messages = body.get("messages") or []
    systems = [m for m in messages if m.get("role") == "system"]
    users = [m for m in messages if m.get("role") == "user"]
    others = [m for m in messages if m.get("role") not in ("system", "user")]
    if others or len(users) != 1 or len(systems) > 1 or (systems and messages[0].get("role") != "system"):
        raise PromptRefused(
            422, "rubric_messages_unsupported",
            "only a rubric with one optional system message followed by one user message can "
            "be imported; this one has "
            + ", ".join(m.get("role", "?") for m in messages),
        )
    fields = list(body.get("input_fields") or [])
    system_text = (
        _rubric_template(systems[0]["content"], field_map, literals, fields) if systems else ""
    )
    if any(
        name_ in TEMPLATE_PLACEHOLDERS or name_ in field_map
        for name_ in _fields_used(systems[0]["content"] if systems else "")
        if name_ not in literals
    ):
        raise PromptRefused(
            422, "rubric_unmappable",
            "the rubric's system message uses a row field; the judge's instructions must be the "
            "same for every row. Give that field a fixed value, or move it to the user message",
        )
    user_template = _rubric_template(users[0]["content"], field_map, literals, fields)
    if not system_text.strip():
        raise PromptRefused(
            422, "rubric_no_system",
            "the rubric has no system message; library prompts carry their instructions there. "
            "Clone the rubric in miDataworks with one, or create the prompt here directly",
        )

    parser = body.get("parser")
    reply = {"verdict_line_v1": "verdict_line", "json_v1": "json"}.get(parser)
    if reply is None:
        raise PromptRefused(422, "rubric_parser_unsupported", f"parser {parser!r} has no mapping")
    verdicts = [str(v) for v in (body.get("allowed_verdicts") or [])]
    numeric = all(v.isdigit() for v in verdicts) and verdicts
    consecutive = (
        numeric
        and sorted(int(v) for v in verdicts) == list(range(min(int(v) for v in verdicts), max(int(v) for v in verdicts) + 1))
        and len(verdicts) >= 2
    )
    if style == "pointwise" and consecutive:
        if positive_from is None:
            raise PromptRefused(
                422, "positive_from_required",
                f"this rubric rates {min(map(int, verdicts))} to {max(map(int, verdicts))}; say "
                f"which rating and above means the behaviour is present (positive_from)",
            )
        answer_form = {
            "kind": "rating",
            "min": min(int(v) for v in verdicts),
            "max": max(int(v) for v in verdicts),
            "positive_from": int(positive_from),
            "json_key": "verdict",
            "reply": reply,
        }
    else:
        if not positive_verdicts:
            raise PromptRefused(
                422, "positive_verdicts_required",
                f"say which of the verdicts {verdicts} mean the behaviour is present "
                f"(positive_verdicts)",
            )
        answer_form = {
            "kind": "labels",
            "labels": verdicts,
            "positive_labels": list(positive_verdicts),
            "json_key": "verdict",
            "reply": reply,
            "case_sensitive": True,
        }
    return {
        "name": name,
        "title": title,
        "behaviour": behaviour,
        "behaviour_label": None,
        "description": (
            f"Imported from miDataworks rubric {document.get('name')} v{document.get('version')} "
            f"({style})."
        ),
        "system_prompt": system_text,
        "user_template": user_template,
        "answer_form": answer_form,
        "max_interaction_chars": DEFAULT_MAX_INTERACTION_CHARS,
    }


def _fields_used(content: str) -> List[str]:
    import string

    return [f for _, f, _, _ in string.Formatter().parse(content) if f]


def midataworks_url() -> str:
    """miDataworks' base URL, or `PromptRefused` (503) saying it is not connected.

    STANDALONE FIRST (memory: midataworks-standalone-first): absent is a normal state, and every
    other library feature works without it — including importing a rubric FILE.
    """
    from ..core.config import settings

    url = (getattr(settings, "midataworks_api_url", None) or "").strip().rstrip("/")
    if not url:
        raise PromptRefused(
            503, "midataworks_not_connected",
            "miDataworks is not connected to this miStudio (MIDATAWORKS_API_URL is not set). "
            "Export the rubric from miDataworks and import the file instead.",
        )
    return url


async def _midataworks_get(path: str) -> Any:
    import httpx

    base = midataworks_url()
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(15.0, connect=5.0)) as http:
            response = await http.get(f"{base}{path}")
    except httpx.HTTPError as exc:
        raise PromptRefused(
            502, "midataworks_unreachable",
            f"miDataworks did not answer at {base} ({type(exc).__name__}); try again, or import "
            f"an exported rubric file",
        ) from exc
    if response.status_code == 404:
        raise PromptRefused(404, "rubric_not_found", f"miDataworks has no {path}")
    if response.status_code >= 400:
        raise PromptRefused(
            502, "midataworks_error",
            f"miDataworks answered {response.status_code} for {path}: {response.text[:200]}",
        )
    return response.json()


async def list_midataworks_rubrics() -> List[Dict[str, Any]]:
    """miDataworks' rubrics, with whether each style can be imported here."""
    rows = await _midataworks_get("/api/v1/rubrics")
    out = []
    for row in rows if isinstance(rows, list) else []:
        style = row.get("style")
        out.append(
            {
                "id": row.get("id"),
                "name": row.get("name"),
                "version": row.get("version"),
                "ref": row.get("ref"),
                "style": style,
                "content_hash": row.get("content_hash"),
                "importable": style in ("binary", "pointwise"),
                "input_fields": (row.get("body") or {}).get("input_fields"),
                "allowed_verdicts": (row.get("body") or {}).get("allowed_verdicts"),
            }
        )
    return out


async def fetch_midataworks_rubric(rubric_id: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """(export document, the rubric's own record) for one rubric."""
    record = await _midataworks_get(f"/api/v1/rubrics/{rubric_id}")
    document = await _midataworks_get(f"/api/v1/rubrics/{rubric_id}/export")
    return document, record


# ── rung 3 ────────────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RungEligibility:
    counts: bool
    note: str


def rung_eligibility(
    prompt_behaviour: Optional[str], prompt_label: str, probe_behaviour: Optional[str]
) -> RungEligibility:
    """Whether a judge run can count toward a probe's rung 3, and the sentence that says why.

    ⚠ EQUALITY OF TWO RECORDED KEYS, NOTHING INFERRED. A rater of one behaviour compared with a probe of
    another produces a perfectly computable AUROC that measures nothing about the probe's concept, and
    rung 3 would then claim the probe was compared with a judge of its own behaviour. Unknown on
    either side is "does not count", stated, never a match by default.
    """
    if not prompt_behaviour:
        return RungEligibility(
            False,
            f"the judge prompt ({prompt_label}) records no behaviour, so it cannot count toward "
            f"rung 3",
        )
    if not probe_behaviour:
        return RungEligibility(
            False,
            f"the probe has no behaviour recorded, so the {prompt_label} judge "
            f"({prompt_behaviour}) cannot count toward rung 3. Set the probe's behaviour on its "
            f"report if it detects {prompt_behaviour!r}",
        )
    if prompt_behaviour != probe_behaviour:
        return RungEligibility(
            False,
            f"the {prompt_label} judge rates {prompt_behaviour!r} but this probe detects "
            f"{probe_behaviour!r}; the comparison is shown and never raises the rung",
        )
    return RungEligibility(
        True, f"the {prompt_label} judge rates {prompt_behaviour!r}, the probe's own behaviour"
    )


def probe_behaviour_sync(db: Any, probe: Any) -> Optional[str]:  # noqa: ARG001
    """A probe's behaviour key — its OWN column, set from its run and editable per probe."""
    return getattr(probe, "behaviour", None)


async def probe_behaviour(db: AsyncSession, probe_id: str) -> Optional[str]:
    from ..models.probe_monitor import ProbeMonitor

    row = (
        await db.execute(select(ProbeMonitor.behaviour).where(ProbeMonitor.id == probe_id))
    ).first()
    return row[0] if row else None


async def run_rung_view(
    db: AsyncSession, runs: Sequence[ProbeMonitorJudgeRun]
) -> Dict[str, Dict[str, Any]]:
    """For each run: the prompt it used (title, key, behaviour) and whether it counts for its
    probe's rung 3. One query per kind, not per run."""
    prompt_ids = {run.prompt_id for run in runs if run.prompt_id}
    prompts: Dict[str, ProbeJudgePrompt] = {}
    if prompt_ids:
        prompts = {
            p.id: p
            for p in (
                await db.execute(select(ProbeJudgePrompt).where(ProbeJudgePrompt.id.in_(prompt_ids)))
            ).scalars().all()
        }
    behaviours: Dict[str, Optional[str]] = {}
    for probe_id in {run.probe_id for run in runs if run.probe_id}:
        behaviours[probe_id] = await probe_behaviour(db, probe_id)
    out: Dict[str, Dict[str, Any]] = {}
    for run in runs:
        prompt = prompts.get(run.prompt_id) if run.prompt_id else None
        entry: Dict[str, Any] = {
            "prompt_id": run.prompt_id,
            "prompt_hash": run.prompt_hash,
            "prompt_title": prompt.title if prompt else None,
            "prompt_behaviour": prompt.behaviour if prompt else None,
            "counts_toward_rung": None,
            "rung_note": None,
        }
        if run.probe_id:
            verdict = rung_eligibility(
                prompt.behaviour if prompt else None,
                run.prompt_version,
                behaviours.get(run.probe_id),
            )
            entry["counts_toward_rung"] = verdict.counts
            entry["rung_note"] = verdict.note
        out[run.id] = entry
    return out
