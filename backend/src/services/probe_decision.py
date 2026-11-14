"""The ONE assembly of a probe's contract `decision` block, and its stored snapshot (039 FR-8).

⚠ TWO ASSEMBLIES OF ONE BLOCK DRIFT, AND HERE THE DRIFT IS A WRONG JOIN KEY. Before 039 the builder
formed `decision` inline from the row while a re-cut described its bar in its own words. With
digests, the re-cut's preview, the next build, every read and the MCP push must state the SAME
`decision_digest`; a second assembly that, say, omitted `calibration` would make the preview promise a
digest the build never carries — and an agent joining on it would find nothing. So:

* `decision_from_values` is the one function that turns bar values into a contract `Decision`
  (it absorbs the builder's `_window_decisions`, `_length_bands` and `bands_or_none`, so the
  "drop entries without a threshold" and "whole table or none" rules have one home);
* every bar writer stores the formed block and its digest beside the bar, in the same transaction
  (`write_decision_snapshot`): the run's calibrating stage and `probe_recalibration.apply` (the GPU
  window re-cut commits through it);
* the build COPIES the stored block, and refuses `decision_snapshot_stale` if the row's bar no longer
  forms it — a writer that forgot the snapshot is a defect, and the build fails loudly rather than
  publishing either value.

⚠ THE CALIBRATION REFERENCE IS FROZEN AT CUT TIME (X4, TI5). `decision.calibration` names the
calibration dataset only when `threshold_source == "calibration_set"`; a re-cut keeps the snapshot's
reference while the source is unchanged, and re-derives it from the run only when the source changes
(a NULL source becomes `calibration_set` on re-cut, `propose`'s fallback). Re-reading the dataset row
on every write would let an edited row move the digest of a bar nobody re-cut.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

from pydantic import ValidationError

from ..core.clock import utc_now
from ..schemas.probe_definition import DatasetRef, Decision, LengthBand, WindowDecision
from ..schemas.probe_identity import IdentityInputRefused, current_revision, decision_digest_of

logger = logging.getLogger(__name__)

#: The `stale_reason` a snapshot carries when its bar cannot be written as a contract decision.
NOT_FORMABLE = "decision_not_formable"


class NotFormable:
    """A bar the contract cannot express — the snapshot records the reason, the digest is null."""

    def __init__(self, detail: str, *, status: int = 422) -> None:
        self.detail = detail
        self.status = status

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"NotFormable({self.detail!r})"


Calibration = Union[DatasetRef, None, NotFormable]


# ── the one assembly ──────────────────────────────────────────────────────────


def bands_or_none(stored: Any) -> Optional[List[LengthBand]]:
    """A stored band table as contract `LengthBand`s, or `None` — whole or not at all.

    ⚠ ENTRIES WITHOUT A THRESHOLD ARE NOT DROPPED ONE BY ONE: dropping one would tear a hole in a
    table the contract requires to be contiguous, so if ANY band lacks a threshold the whole table is
    omitted. A partial table is worse than none — a consumer would find the gap and refuse the
    document at import. Shared by the global table and every window's own (2026-10-04).
    """
    if not isinstance(stored, list) or not stored:
        return None
    if any(entry.get("threshold") is None for entry in stored if isinstance(entry, dict)):
        return None
    return [LengthBand(**entry) for entry in stored]


def window_decisions_or_none(stored: Any) -> Optional[Dict[str, WindowDecision]]:
    """`decision.windows` from a stored `window_decisions` map, or `None`.

    ⚠ ENTRIES WITHOUT A THRESHOLD ARE DROPPED, not exported as nulls: an entry claiming a window was
    calibrated while placing no bar is indistinguishable from "not calibrated" while looking like
    more evidence, and the contract refuses it. Copied field by field, so internal bookkeeping
    (`scores_path`, `lengths_path`) never reaches a published document — or a digest.
    """
    if not isinstance(stored, dict) or not stored:
        return None
    out: Dict[str, WindowDecision] = {}
    for window, entry in stored.items():
        if not isinstance(entry, dict) or entry.get("threshold") is None:
            continue
        out[str(window)] = WindowDecision(
            threshold=entry.get("threshold"),
            target_fpr=entry.get("target_fpr"),
            realised_fpr=entry.get("realised_fpr"),
            n_negatives=entry.get("n_negatives"),
            scope=entry.get("scope"),
            length_bands=bands_or_none(entry.get("length_bands")),
        )
    return out or None


def decision_from_values(
    *,
    threshold: Optional[float],
    target_fpr: Optional[float],
    realised_fpr: Optional[float],
    threshold_source: Optional[str],
    calibration: Optional[DatasetRef],
    window_decisions: Any,
    length_bands: Any,
) -> Decision:
    """THE contract `decision` block for a bar. Pure: no session, no I/O."""
    return Decision(
        threshold=threshold,
        target_fpr=target_fpr,
        realised_fpr=realised_fpr,
        threshold_source=threshold_source,
        calibration=calibration,
        windows=window_decisions_or_none(window_decisions),
        length_bands=bands_or_none(length_bands),
    )


def decision_for_row(probe: Any, calibration: Optional[DatasetRef]) -> Decision:
    """`decision_from_values` over the bar a probe row holds now."""
    return decision_from_values(
        threshold=probe.threshold,
        target_fpr=probe.target_fpr,
        realised_fpr=probe.realised_fpr,
        threshold_source=probe.threshold_source,
        calibration=calibration,
        window_decisions=probe.window_decisions,
        length_bands=probe.length_bands,
    )


# ── the calibration reference ─────────────────────────────────────────────────


def calibration_ref_for(
    db: Any, calibration_dataset_id: Optional[str], threshold_source: Optional[str]
) -> Optional[DatasetRef]:
    """The `decision.calibration` reference, or `None`.

    ⚠ NAMED ONLY WHEN THE THRESHOLD ACTUALLY CAME FROM IT. Until 2026-09-28 the builder emitted this
    whenever the RUN carried a calibration view while the threshold came from validation negatives,
    so a document named a calibration corpus beside a `threshold_source` contradicting it. Gated on
    `threshold_source` so the two cannot disagree. Raises `ProbeExportRefused` for a dataset with no
    HuggingFace identifier, exactly as the builder always has (`dataset_ref_from_view`).
    """
    if not calibration_dataset_id or threshold_source != "calibration_set":
        return None
    from ..models.probe_monitor import ProbeMonitorDataset
    from .probe_definition_builder import dataset_ref_from_view

    view = (
        db.query(ProbeMonitorDataset)
        .filter(ProbeMonitorDataset.id == calibration_dataset_id)
        .first()
    )
    return dataset_ref_from_view(db, view)


def calibration_or_not_formable(
    db: Any, calibration_dataset_id: Optional[str], threshold_source: Optional[str]
) -> Calibration:
    """`calibration_ref_for`, with its refusal turned into `NotFormable` (the snapshot's reason)."""
    from .probe_definition_builder import ProbeExportRefused

    try:
        return calibration_ref_for(db, calibration_dataset_id, threshold_source)
    except ProbeExportRefused as refused:
        return NotFormable(str(refused), status=refused.status)


def calibration_from_run(db: Any, probe: Any, threshold_source: Optional[str]) -> Calibration:
    """The reference from the probe's RUN (the builder's source), or `NotFormable` with the reason."""
    from ..models.probe_monitor import ProbeMonitorRun

    if threshold_source != "calibration_set":
        return None
    run = db.query(ProbeMonitorRun).filter(ProbeMonitorRun.id == probe.run_id).first()
    return calibration_or_not_formable(
        db, getattr(run, "calibration_dataset_id", None) if run is not None else None,
        threshold_source,
    )


def snapshot_threshold_source(snapshot: Optional[Dict[str, Any]]) -> Any:
    """The `threshold_source` a stored snapshot was formed under (a sentinel when none was formed)."""
    if not snapshot:
        return _NO_SNAPSHOT
    if "threshold_source" in snapshot:
        return snapshot["threshold_source"]
    return (snapshot.get("decision") or {}).get("threshold_source", _NO_SNAPSHOT)


_NO_SNAPSHOT = object()


def snapshot_calibration(probe: Any) -> Calibration:
    """The calibration the stored snapshot was formed with: a `DatasetRef`, `None`, or the stored
    `NotFormable`. Only meaningful while `threshold_source` is unchanged — see `calibration_for_bar`."""
    snapshot = getattr(probe, "decision_snapshot", None) or {}
    if snapshot.get("decision") is None:
        return NotFormable(snapshot.get("detail") or NOT_FORMABLE, status=snapshot.get("status") or 422)
    stored = snapshot["decision"].get("calibration")
    return DatasetRef(**stored) if stored else None


def calibration_for_bar(db: Any, probe: Any, threshold_source: Optional[str]) -> Calibration:
    """The calibration reference for a bar about to be written under `threshold_source` (TI5).

    ⚠ THE SNAPSHOT'S FROZEN REFERENCE ONLY WHILE THE SOURCE IS UNCHANGED (X4). `propose` re-cuts with
    `threshold_source or "calibration_set"`, so a probe whose source was NULL GAINS `calibration_set`
    on its first re-cut — which changes whether `decision.calibration` exists at all. Copying the
    snapshot's `null` across that change would publish a calibration-set bar that names no corpus.
    """
    snapshot = getattr(probe, "decision_snapshot", None)
    if snapshot and snapshot_threshold_source(snapshot) == threshold_source:
        return snapshot_calibration(probe)
    return calibration_from_run(db, probe, threshold_source)


# ── the snapshot ──────────────────────────────────────────────────────────────


def form_snapshot(probe: Any, calibration: Calibration) -> Dict[str, Any]:
    """The snapshot JSON for the bar `probe` holds now. Pure."""
    base = {
        "formed_at": utc_now().isoformat(),
        "threshold_source": probe.threshold_source,
    }
    if isinstance(calibration, NotFormable):
        return {**base, "decision": None, "digest": None, "stale_reason": NOT_FORMABLE,
                "detail": calibration.detail, "status": calibration.status}
    try:
        decision = decision_for_row(probe, calibration)
        digest = decision_digest_of(decision)
    except (ValidationError, IdentityInputRefused, ValueError) as exc:
        # A bar the contract refuses (a table that does not tile, a non-finite number) is recorded
        # as not formable WITH its reason, rather than raising inside a bar writer's commit.
        return {**base, "decision": None, "digest": None, "stale_reason": NOT_FORMABLE,
                "detail": f"the bar is not a valid contract decision: {exc}", "status": 422}
    return {**base, "decision": decision.model_dump(mode="json"), "digest": digest,
            "stale_reason": None, "detail": None}


def write_decision_snapshot(probe: Any, calibration: Calibration) -> Dict[str, Any]:
    """Set `decision_snapshot` and `decision_digest` on the row object. NO COMMIT: the caller's
    transaction commits it with the bar, so the two can never be seen apart."""
    snapshot = form_snapshot(probe, calibration)
    probe.decision_snapshot = snapshot
    probe.decision_digest = snapshot["digest"]
    return snapshot


def ensure_decision_snapshot(db: Any, probe: Any) -> None:
    """Form the snapshot of a probe that has none (a probe cut before 039), once.

    A compare-and-set (`… WHERE decision_snapshot IS NULL`), committed at once in its own short
    transaction: two concurrent first reads write equal values once, and no row lock outlives this
    call — the build calls it before its GPU work, and a lock held across that would stall a re-cut.
    """
    if getattr(probe, "decision_snapshot", None) is not None:
        return
    from sqlalchemy import update

    from ..models.probe_monitor import ProbeMonitor

    snapshot = form_snapshot(probe, calibration_from_run(db, probe, probe.threshold_source))
    db.execute(
        update(ProbeMonitor)
        .where(ProbeMonitor.id == probe.id, ProbeMonitor.decision_snapshot.is_(None))
        .values(decision_snapshot=snapshot, decision_digest=snapshot["digest"])
    )
    db.commit()
    db.refresh(probe)


def stale_fields(stored: Optional[Dict[str, Any]], formed: Optional[Dict[str, Any]]) -> List[str]:
    """The top-level `decision` keys on which two contract-form blocks disagree."""
    stored = stored or {}
    formed = formed or {}
    return sorted(key for key in set(stored) | set(formed) if stored.get(key) != formed.get(key))


def decision_for_build(db: Any, probe: Any) -> Decision:
    """The `decision` a build publishes: the STORED snapshot, after proving it is the row's bar.

    Refuses (`ProbeExportRefused`):
    * the snapshot's own not-formable reason (the refusal the builder always gave for a local-only
      calibration dataset), at its recorded status;
    * `decision_snapshot_stale` (409) when the row's bar, re-formed with the snapshot's frozen
      calibration (TI5), is not the stored block — a bar writer forgot the snapshot (D-T4).
    """
    from .probe_definition_builder import ProbeExportRefused

    if getattr(probe, "decision_snapshot", None) is None:
        ensure_decision_snapshot(db, probe)
    snapshot = probe.decision_snapshot
    # ⚠ A NOT-FORMABLE SNAPSHOT IS RE-TRIED, NOT TRUSTED. It carries no digest, so nobody can hold
    # a value it would move; and the usual cause (a calibration dataset with no HuggingFace id) is
    # fixed on the DATASET row, after which the build must succeed as it always did. Re-formed from
    # the run under the same source and stored when it now forms.
    #
    # ⚠ A COMPARE-AND-SET, NOT A PLAIN WRITE (review R1). `probe` was read before this retry; a
    # re-cut committed since then wrote a NEW bar and its snapshot. Assigning the attributes flushed
    # only these two columns, so the retried block — formed from the OLD bar in memory — replaced the
    # re-cut's snapshot beside the re-cut's bar, and every later build refused
    # `decision_snapshot_stale` until someone re-cut again. Keyed on the not-formable snapshot this
    # build read (`formed_at`); when another writer got there first, the row is re-read and the retry
    # runs once more against what it holds now.
    attempts = 0
    while snapshot.get("decision") is None:
        attempts += 1
        if attempts > 2:
            raise ProbeExportRefused(
                "decision_snapshot_stale: the probe's bar kept changing while this build re-formed "
                "its decision; retry the build", status=409,
            )
        retried = form_snapshot(probe, calibration_from_run(db, probe, probe.threshold_source))
        if retried.get("decision") is None:
            raise ProbeExportRefused(
                retried.get("detail") or NOT_FORMABLE, status=int(retried.get("status") or 422)
            )
        from sqlalchemy import update

        from ..models.probe_monitor import ProbeMonitor

        db.execute(
            update(ProbeMonitor)
            .where(
                ProbeMonitor.id == probe.id,
                ProbeMonitor.decision_digest.is_(None),
                ProbeMonitor.decision_snapshot["formed_at"].astext == snapshot.get("formed_at"),
            )
            .values(decision_snapshot=retried, decision_digest=retried["digest"])
        )
        db.commit()
        db.refresh(probe)
        snapshot = probe.decision_snapshot or {}
    calibration = calibration_for_bar(db, probe, probe.threshold_source)
    reformed = form_snapshot(probe, calibration)
    fields = stale_fields(snapshot.get("decision"), reformed.get("decision"))
    if fields or reformed.get("digest") != snapshot.get("digest"):
        logger.info(
            "decision_snapshot_stale probe=%s stored_digest=%s row_digest=%s fields=%s",
            probe.id, snapshot.get("digest"), reformed.get("digest"), fields,
        )
        raise ProbeExportRefused(
            f"decision_snapshot_stale: fields {', '.join(fields) or 'digest'} — the probe's bar no "
            f"longer forms the decision stored when it was set, so a bar writer skipped the "
            f"snapshot. Re-cut the probe (recalibrate_probe_monitor) to re-form it; nothing is "
            f"published from a stale block",
            status=409,
        )
    return Decision.model_validate(snapshot["decision"])


# ── Feature 039 Slice 2: the one identity read (FR-16, FR-17; FTDD §7.2) ──────────────────────

#: The six summary keys `identity_fields` returns — exactly these, on every read surface.
IDENTITY_KEYS = (
    "detector_digest", "decision_digest", "decision_revision",
    "identity_current", "identity_stale_reason", "identity_source",
)

#: `invalidated.cause` → the `identity_stale_reason` a reader is told.
_STALE_REASON_FOR_CAUSE = {
    "decision": "decision_recalibrated",
    "build_race": "decision_changed_during_build",
}


def _blank(source: Optional[str] = None, reason: Optional[str] = None) -> Dict[str, Any]:
    out: Dict[str, Any] = dict.fromkeys(IDENTITY_KEYS)
    out["identity_source"] = source
    out["identity_stale_reason"] = reason
    return out


def identity_fields(probe: Any, *, file_missing: bool = False) -> Dict[str, Any]:
    """The identity of the definition a probe's row records, and whether it is the CURRENT decision.

    FTDD §7.2, rules 1 – 6. Pure over the row: no file is read and no value is guessed.

    1. never built → every key null;
    2. a built definition → its recorded identity; `identity_current` compares its decision digest
       with the row's (both non-null), else `false` with `decision_differs_from_current_bar`;
       a RECOMPUTED file states a revision only when its digest is the current one (FR-17);
    3. an invalidated definition → the identity it carried (`was_identity`); after a rung-only
       invalidation `identity_current` stays true (E3) — the stale FILE is reported elsewhere;
    4. invalidated before 039 (no `was_identity`) → `not_recorded`, the old file is NOT read (E5);
    5. a pre-039 file that is gone → `not_recorded`, `file_missing` (`file_missing` is the transient
       flag `backfill_identity` returns for this request, TI10 — no stat on every read);
    6. ⚠ `decision_revision` is NEVER defaulted (to 1, or to the history's maximum without the
       equality check): a revision is the second half of a join key, and a guessed one joins a
       consumer's events to a bar that never judged them.
    """
    build = getattr(probe, "definition_build", None) or {}
    resolved = build.get("resolved") or {}
    invalidated = build.get("invalidated") or None
    path = getattr(probe, "definition_path", None)

    if path:
        if not resolved.get("detector_digest"):
            return _blank("not_recorded", "file_missing" if file_missing else None)
        recorded, cause = resolved, None
    elif invalidated is not None:
        recorded = invalidated.get("was_identity") or None
        if not recorded or not recorded.get("detector_digest"):
            return _blank("not_recorded")
        cause = invalidated.get("cause")
    else:
        return _blank()

    source = recorded.get("identity_source") or "built"
    decision_digest = recorded.get("decision_digest")
    current_digest = getattr(probe, "decision_digest", None)
    snapshot = getattr(probe, "decision_snapshot", None)
    reason: Optional[str] = None
    if decision_digest is not None and current_digest is not None:
        current: Optional[bool] = decision_digest == current_digest
        if not current:
            reason = _STALE_REASON_FOR_CAUSE.get(cause or "", "decision_differs_from_current_bar")
    elif snapshot is not None and current_digest is None and snapshot.get("decision") is None:
        current, reason = False, NOT_FORMABLE
    else:
        current = None
    if source == "recomputed":
        revision = (
            current_revision(getattr(probe, "calibration_history", None)) if current else None
        )
    else:
        revision = recorded.get("decision_revision")
    return {
        "detector_digest": recorded.get("detector_digest"),
        "decision_digest": decision_digest,
        "decision_revision": revision,
        "identity_current": current,
        "identity_stale_reason": reason,
        "identity_source": source,
    }


def needs_backfill(probe: Any) -> bool:
    """A probe whose cached definition was built before 039: a path, and no recorded identity."""
    resolved = ((getattr(probe, "definition_build", None) or {}).get("resolved") or {})
    return bool(getattr(probe, "definition_path", None)) and not resolved.get("detector_digest")


def backfill_identity(probe_ids: List[str]) -> Dict[str, str]:
    """Recompute and record the identity of definitions built before 039, once each (FTDD §7.3).

    For each probe: read the cached file, validate it, compute `document_identity`, form the decision
    snapshot if it has none, and write `resolved.{detector_digest, decision_digest,
    decision_revision, identity_source: "recomputed"}` and `built_detector_digest` with a
    COMPARE-AND-SET keyed on the file's sha — a concurrent rebuild changes the sha and wins, and two
    concurrent first reads write the same values once. After one read nothing is pending, so the
    steady-state list does no file I/O.

    Returns `{probe_id: "file_missing"}` for files that are gone (TI10): the caller passes that to
    `identity_fields` for this one response; nothing is persisted for a missing file.

    Runs in a threadpool with its own short `SyncSessionLocal` session.
    """
    import json

    from sqlalchemy import update

    from ..core.config import settings
    from ..core.database import SyncSessionLocal
    from ..models.probe_monitor import ProbeMonitor
    from ..schemas.probe_definition import ProbeDefinitionV1
    from ..schemas.probe_identity import current_revision, document_identity

    flags: Dict[str, str] = {}
    if not probe_ids:
        return flags
    session = SyncSessionLocal()
    try:
        probes = session.query(ProbeMonitor).filter(ProbeMonitor.id.in_(list(probe_ids))).all()
        for probe in probes:
            if not needs_backfill(probe):
                continue
            # ⚠ THE COMPARE-AND-SET KEY IS THE SHA OF THE FILE READ HERE, captured NOW.
            # `ensure_decision_snapshot` below commits and REFRESHES the row, so reading
            # `probe.definition_sha256` after it would compare against a re-cut's invalidation
            # (sha NULL → `IS NULL` matches) and write this stale record over the `invalidated`
            # block (found by control C5.3b's regression test).
            read_sha = probe.definition_sha256
            file = settings.resolve_data_path(probe.definition_path)
            if not file.exists():
                flags[probe.id] = "file_missing"
                continue
            try:
                document = json.loads(file.read_text())
                ProbeDefinitionV1.model_validate(document)
                detector, decision = document_identity(document)
            except Exception as exc:  # noqa: BLE001 - an unreadable file is "not recorded", logged
                logger.warning("probe %s: the cached definition's identity could not be "
                               "recomputed (%s)", probe.id, exc)
                continue
            ensure_decision_snapshot(session, probe)
            revision = (
                current_revision(probe.calibration_history)
                if decision == probe.decision_digest else None
            )
            merged = dict(probe.definition_build or {})
            merged["resolved"] = {
                **(merged.get("resolved") or {}),
                "detector_digest": detector,
                "decision_digest": decision,
                "decision_revision": revision,
                "identity_source": "recomputed",
            }
            session.execute(
                update(ProbeMonitor)
                .where(
                    ProbeMonitor.id == probe.id,
                    ProbeMonitor.definition_sha256 == read_sha,
                    ProbeMonitor.built_detector_digest.is_(None),
                )
                .values(definition_build=merged, built_detector_digest=detector)
            )
            session.commit()
    finally:
        session.close()
    return flags


def digest_match(probe: Any, *, detector: Optional[str] = None,
                 decision: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Why a probe matched a digest filter (FR-18): `{by, source, revisions}`.

    `source` is the first of `current` (the decision in force), `history` (a recorded revision) and
    `last_built` (the identity the last built definition stated, D-T8) that holds the digest;
    `revisions` lists EVERY revision at which it occurs — a re-cut A → B → A gives A at two.
    """
    from ..schemas.probe_identity import current_revision

    build = getattr(probe, "definition_build", None) or {}
    built = [build.get("resolved") or {},
             ((build.get("invalidated") or {}).get("was_identity") or {})]
    if decision:
        revisions = sorted({
            int(entry["revision"]) for entry in (getattr(probe, "calibration_history", None) or [])
            if isinstance(entry, dict) and entry.get("decision_digest") == decision
            and isinstance(entry.get("revision"), int)
        })
        in_history = bool(revisions)
        if getattr(probe, "decision_digest", None) == decision:
            revisions = sorted(set(revisions) | {current_revision(probe.calibration_history)})
            source = "current"
        elif in_history:
            source = "history"
        else:
            source = "last_built"
            revisions = sorted({int(r["decision_revision"]) for r in built
                                if r.get("decision_digest") == decision
                                and isinstance(r.get("decision_revision"), int)})
        return {"by": "decision", "source": source, "revisions": revisions}
    if detector:
        return {"by": "detector", "source": "last_built", "revisions": []}
    return None


def decision_response(db: Any, probe_id: str) -> Optional[Dict[str, Any]]:
    """`GET /probes/{id}/decision`: the decision in force, forming a pre-039 probe's snapshot once
    (compare-and-set). `None` for an unknown probe."""
    from ..models.probe_monitor import ProbeMonitor
    from ..schemas.probe_identity import current_revision

    probe = db.query(ProbeMonitor).filter(ProbeMonitor.id == probe_id).first()
    if probe is None:
        return None
    ensure_decision_snapshot(db, probe)
    snapshot = probe.decision_snapshot or {}
    identity = identity_fields(probe)
    return {
        "probe_id": probe.id,
        "decision": snapshot.get("decision"),
        "decision_digest": probe.decision_digest,
        "decision_revision": current_revision(probe.calibration_history),
        "detector_digest": probe.built_detector_digest,
        "identity_current": identity["identity_current"],
        "identity_source": identity["identity_source"],
        "stale_reason": snapshot.get("stale_reason"),
        "detail": snapshot.get("detail"),
    }
