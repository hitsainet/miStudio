"""Rule sets a behaviour version names: resolve, hash, store (Feature 040 FR-1, FR-2, FR-45; FTDD §4.3).

S1 half. A rule set arrives one of two ways and ends as ONE row of the append-only `rule_sets`, keyed by
the producer's identity (RFC 8785 JCS SHA-256 of the document as received; miDataworks 016, spec D38-4):

* **inline** (`{document}`) — hashed here; works with no miDataworks at all (FR-45, standalone first);
* **referenced** (`{name, version, digest}`) — fetched ONCE, at write time, from miDataworks'
  ``GET /api/v1/rule-sets/{name}?version=`` (016 FTDD §6.1: an envelope carrying ``document`` and
  ``sha256``), through the `_midataworks_get` pattern (15 s, connect 5 s). The identity is recomputed here
  over ``document`` and must equal both the envelope's ``sha256`` and the reference's digest, or the write
  is refused ``rule_set_digest_mismatch`` naming them. A set already stored under the digest is not
  fetched again (content-addressed). miDataworks unset or unreachable → ``503 rule_set_unresolvable`` with
  `next` = the same call, the set to be sent inline.

  Verified 2026-10-24 (040 S3, FTASKS 16.1): miDataworks serves this route (`endpoints/rules.py`,
  `get_rule_set`), answering the envelope with `document` and `sha256`.

A set whose own ``behaviour_key`` is not the version's key is refused ``rule_set_behaviour_mismatch``
(040-P1).

S3 half (FR-29, FTASKS 19.1): `validate_for_use` parses, schema-validates against the vendored
`midataworks-rule-set-v1.json`, compiles (RE2 under the memory cap) and runs the set's OWN cases with
miStudio's engine (`src/rule_signals`). It runs at ATTACH — every `resolve`, so a behaviour write cannot
store a set that fails — and at every USE (`evaluate_rule_signals`, a rules evaluation). A failing case is
refused ``rule_set_cases_failed`` naming each case (index, id, and the column that disagreed — never the
text). A set stored before S3 is no longer reported ``unvalidated``: `describe` validates it on read.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.clock import utc_now
from ..models.rule_set import DIGEST_SCHEME, RuleSet
from .detector_identity import CanonicalInputRefused, jcs_bytes, sha256_hex
from .detector_next import DetectorRefused, NextCall

RULE_SET_FORMAT = "midataworks.rule-set/v1"

#: Tests set this to an `httpx.MockTransport` factory (the `millm_read` pattern); production leaves it
#: None and talks to `MIDATAWORKS_API_URL`.
_transport_factory: Optional[Callable[[], Any]] = None

#: The S1 answer for a stored set before the engine existed (FR-2). Kept as data for the S1 tests' history;
#: since S3 `describe` reports a validation instead.
UNVALIDATED = {"status": "unvalidated", "reason": "engine_not_available"}

def engine_limits(settings: Any = None) -> Any:
    """The engine's `Limits`, read from the runtime settings (never inside the engine)."""
    from ..rule_signals import Limits

    if settings is None:
        from ..core.runtime_settings import effective

        settings = effective()
    return Limits.from_settings(settings)


def validate_for_use(document: Any, *, field: str = "rule_set", limits: Any = None) -> Dict[str, Any]:
    """FR-29: load (schema, references, caps, compile) and run the set's own cases. Returns the validation
    record; raises `DetectorRefused` (422) when the set cannot be used. Pure CPU, no I/O."""
    from .. import rule_signals

    limits = limits if limits is not None else engine_limits()
    try:
        compiled = rule_signals.load(document, limits)
    except rule_signals.RuleSetRefused as exc:
        raise DetectorRefused(422, exc.code, f"the rule set cannot be used: {exc.message}",
                              {"field": field, **exc.details}) from exc
    report = rule_signals.run_cases(compiled)
    if not report.ok:
        failing = [f.report() for f in report.failures]
        raise DetectorRefused(
            422, "rule_set_cases_failed",
            f"{len(failing)} of the rule set's own cases fail on this engine; a set is used only when every "
            f"case it carries passes",
            {"field": field, "failing_cases": failing, "passed": report.passed},
        )
    return validation_record(compiled, report, limits)


def validation_record(compiled: Any, report: Any, limits: Any) -> Dict[str, Any]:
    from .. import rule_signals
    from ..rule_signals.document import provenance

    return {
        "status": "valid",
        "engine_version": rule_signals.ENGINE_VERSION,
        "engine_digest": rule_signals.engine_digest(limits),
        "cases": {"passed": report.passed, "unverified": report.unverified},
        "match_expressions": list(compiled.match_expressions),
        "has_vetoes": compiled.has_vetoes,
        "pending_expressions": sorted(compiled.pending_expressions),
        "provenance": provenance(compiled.document),
    }


def _check_document(document: Any, field: str, key: str, max_bytes: int) -> bytes:
    if not isinstance(document, dict):
        raise DetectorRefused(422, "invalid_field", "a rule set is a JSON object",
                              {"field": field, "rule": "not_an_object"})
    if document.get("format") != RULE_SET_FORMAT:
        raise DetectorRefused(
            422, "invalid_field",
            f"a rule set's format is {RULE_SET_FORMAT!r}, not {document.get('format')!r}",
            {"field": f"{field}.format", "rule": "unknown_format"},
        )
    owner = document.get("behaviour_key")
    if not isinstance(owner, str) or not owner:
        raise DetectorRefused(422, "invalid_field", "a rule set names the behaviour it is for (behaviour_key)",
                              {"field": f"{field}.behaviour_key", "rule": "required"})
    if owner != key:
        raise DetectorRefused(
            422, "rule_set_behaviour_mismatch",
            f"this rule set is for behaviour {owner!r}, not {key!r}; a version can carry only a set "
            f"written for its own key",
            {"field": f"{field}.behaviour_key", "rule_set_behaviour_key": owner, "behaviour_key": key},
        )
    try:
        canonical = jcs_bytes(document)
    except CanonicalInputRefused as exc:
        raise DetectorRefused(422, "invalid_field", f"the rule set cannot be canonicalised: {exc}",
                              {"field": field, "rule": "not_canonicalisable"}) from exc
    if len(canonical) > max_bytes:
        raise DetectorRefused(
            422, "over_cap",
            f"the rule set is {len(canonical)} bytes in canonical form; the runtime setting "
            f"rule_set_max_bytes allows {max_bytes}",
            {"setting": "rule_set_max_bytes", "value": max_bytes, "field": field, "got": len(canonical)},
        )
    return canonical


async def _store(db: AsyncSession, *, digest: str, canonical: bytes, key: str, source: str,
                 source_ref: Optional[Dict[str, Any]], actor: str, fetched: bool) -> None:
    """INSERT … ON CONFLICT DO NOTHING: one row per identity, never rewritten (append-only)."""
    await db.execute(
        insert(RuleSet).values(
            digest=digest, bytes=canonical, schema_version=RULE_SET_FORMAT, behaviour_key=key,
            source=source, source_ref=source_ref, digest_scheme=DIGEST_SCHEME,
            fetched_at=utc_now() if fetched else None, created_by=actor, created_at=utc_now(),
        ).on_conflict_do_nothing(index_elements=["digest"])
    )


async def get_by_digest(db: AsyncSession, digest: str) -> Optional[RuleSet]:
    return (await db.execute(select(RuleSet).where(RuleSet.digest == digest))).scalar_one_or_none()


async def _fetch(name: str, version: int) -> Mapping[str, Any]:
    """016's envelope `{document, sha256}`; any failure is `rule_set_unresolvable` (raised by caller)."""
    import httpx

    from ..core.config import settings

    base = (getattr(settings, "midataworks_api_url", None) or "").strip().rstrip("/")
    if not base:
        raise LookupError("miDataworks is not connected to this miStudio (MIDATAWORKS_API_URL is not set)")
    try:
        transport = _transport_factory() if _transport_factory is not None else None
        async with httpx.AsyncClient(timeout=httpx.Timeout(15.0, connect=5.0), transport=transport) as http:
            response = await http.get(f"{base}/api/v1/rule-sets/{name}", params={"version": version})
    except httpx.HTTPError as exc:
        raise LookupError(f"miDataworks did not answer at {base} ({type(exc).__name__})") from exc
    if response.status_code >= 400:
        raise LookupError(f"miDataworks answered {response.status_code} for rule set {name} v{version}")
    body = response.json()
    if not isinstance(body, dict) or not isinstance(body.get("document"), dict):
        raise LookupError("miDataworks' answer carries no rule-set document")
    return body


async def resolve(
    db: AsyncSession,
    rule_set: Any,
    *,
    key: str,
    max_bytes: int,
    actor: str,
    retry: NextCall,
    field: str = "rule_set",
) -> Optional[str]:
    """The digest to store on the version (the row written if new), or None for no rule set.

    `rule_set` is the request's `RuleSetRef`/`RuleSetInline` (or its dict); `retry` is the `next` entry
    that repeats the caller's request with the set inline. Does not commit."""
    if rule_set is None:
        return None
    data = rule_set.model_dump(mode="json") if hasattr(rule_set, "model_dump") else dict(rule_set)
    if "document" in data:
        canonical = _check_document(data["document"], f"{field}.document", key, max_bytes)
        validate_for_use(data["document"], field=f"{field}.document")  # FR-29: at attach
        digest = sha256_hex(canonical)
        await _store(db, digest=digest, canonical=canonical, key=key, source="inline",
                     source_ref=None, actor=actor, fetched=False)
        return digest

    ref = {"name": data["name"], "version": data["version"], "digest": data["digest"]}
    existing = await get_by_digest(db, ref["digest"])
    if existing is not None:
        if existing.behaviour_key != key:
            raise DetectorRefused(
                422, "rule_set_behaviour_mismatch",
                f"rule set {ref['digest'][:12]}… is for behaviour {existing.behaviour_key!r}, not {key!r}",
                {"field": f"{field}.digest", "rule_set_behaviour_key": existing.behaviour_key,
                 "behaviour_key": key},
            )
        validate_for_use(document_of(existing)[0], field=field)  # FR-29: at attach
        return existing.digest
    try:
        envelope = await _fetch(ref["name"], ref["version"])
    except LookupError as exc:
        raise DetectorRefused(
            503, "rule_set_unresolvable",
            f"rule set {ref['name']} v{ref['version']} could not be read from miDataworks: {exc}. "
            f"Send the set inline instead.",
            {"field": field, "reference": ref}, next=[retry],
        ) from exc
    canonical = _check_document(envelope["document"], f"{field} (fetched)", key, max_bytes)
    computed = sha256_hex(canonical)
    served = envelope.get("sha256")
    if computed != ref["digest"] or (served is not None and served != computed):
        raise DetectorRefused(
            422, "rule_set_digest_mismatch",
            f"rule set {ref['name']} v{ref['version']} hashes to {computed}, which is not the digest "
            f"the request named ({ref['digest']})"
            + (f" or the one miDataworks served ({served})" if served not in (None, computed) else ""),
            {"field": f"{field}.digest", "requested": ref["digest"], "computed": computed, "served": served},
        )
    validate_for_use(envelope["document"], field=field)  # FR-29: at attach
    await _store(db, digest=computed, canonical=canonical, key=key, source="referenced",
                 source_ref=ref, actor=actor, fetched=True)
    return computed


def describe(row: Optional[RuleSet], digest: Optional[str]) -> Optional[Dict[str, Any]]:
    """How a version's rule set reads: identity, provenance and — with the stored row — its validation by
    THIS engine, computed on read (never stored: a re-vendored engine can change it, FR-38)."""
    if digest is None:
        return None
    out: Dict[str, Any] = {"digest": digest, "digest_scheme": DIGEST_SCHEME}
    if row is not None:
        out.update({"source": row.source, "source_ref": row.source_ref, "behaviour_key": row.behaviour_key})
        try:
            out["validation"] = validate_for_use(document_of(row)[0])
        except DetectorRefused as refusal:
            out["validation"] = {"status": "invalid", "code": refusal.code,
                                 **{k: v for k, v in refusal.details.items() if k != "field"}}
    return out


def document_of(row: RuleSet) -> Tuple[Dict[str, Any], bytes]:
    import json

    return json.loads(row.bytes.decode("utf-8")), bytes(row.bytes)
