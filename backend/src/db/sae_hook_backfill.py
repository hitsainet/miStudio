"""Fill in the hook and layer of SAE rows that recorded neither (review R3-B, R3B-11).

Every consumer that reads an SAE at its layer's output takes a NULL ``hook_type`` as
residual, and seven of them turned a NULL ``layer`` into **layer 0 without a word**
(``extraction_service.py``, the Neuronpedia export and push, the feature browser, the
steering loader). R3B-6 stopped new rows arriving with either field NULL; rows written
before it keep them, and refusing a NULL layer at the consumers before those rows are
repaired would fail them all at once. So: backfill first, then refuse.

**What is resolved, and by whom.** The SAME two functions the writers call --
``resolve_sae_hook`` and ``resolve_sae_layer`` in ``services/sae_manager_service.py`` --
so a backfilled row is byte-identical to a fresh download of the same SAE. There is no
second resolver here, and nothing is guessed: a row whose files and names say nothing
keeps its NULLs and is reported as unresolved.

**What is excluded.** ``source='trained'`` and any row with a ``training_id``. A training
import already recorded its own layer and hook from the export directory's suffix, and a
pre-A5 training export carries a ``cfg.json`` naming ``resid_post`` for EVERY hook -- so
re-resolving such a row from its config would write a WRONG hook over a right one. This
exclusion is the reason the backfill cannot simply run over the whole table.

**What is written.** Only fields that are NULL. A recorded hook or layer is never
overwritten, and a recorded layer that DISAGREES with the resolved one is reported, not
changed -- the row is a claim someone made, and silently correcting it would hide the
disagreement that matters. Each changed row records ``sae_metadata.hook_backfill``
(when, by what, and the previous values) so the change is auditable and reversible.

**Dry run is the default.** ``--apply`` writes, in one transaction, and only when no row
it would touch is referenced by an IN-FLIGHT job: a running extraction or push reads the
row's layer while it works, and moving it underneath would change where the job reads
half way through.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..core.clock import utc_now
from ..models.circuit_runs import CircuitCaptureRun
from ..models.cluster_profile import ClusterProfile
from ..models.external_sae import ExternalSAE
from ..models.extraction_job import ExtractionJob
from ..models.feature import Feature
from ..models.neuronpedia_export import NeuronpediaExportJob
from ..models.neuronpedia_push import NeuronpediaPushJob
from ..models.steering_experiment import SteeringExperiment

logger = logging.getLogger(__name__)

#: Written into ``sae_metadata.hook_backfill.by`` so a row's provenance names this script.
BACKFILL_BY = "backfill_sae_hooks"

#: The only sources this backfill may resolve. ``trained`` is excluded by design (above).
BACKFILLABLE_SOURCES = ("huggingface", "local")

#: Job statuses that mean the job is still queued or running, per job table. A row
#: referenced by any of these is not repaired: the job reads its layer as it works.
LIVE_EXTRACTION_STATUSES = ("queued", "loading", "extracting", "saving")
LIVE_PUSH_STATUSES = ("queued", "preparing", "pushing")
LIVE_EXPORT_STATUSES = ("pending", "computing", "packaging")
LIVE_CAPTURE_STATUSES = ("pending", "estimating", "running")


class BackfillBlocked(RuntimeError):
    """``--apply`` was asked for while a job is reading one of the rows it would change."""


def _status_text(value: Any) -> str:
    """The status as a plain string, whether the column gave an Enum or a str."""
    return str(getattr(value, "value", value) or "").lower()


@dataclass
class RowPlan:
    """What the backfill would do to one row, and what depends on it."""

    sae_id: str
    name: Optional[str]
    source: str
    status: str
    old_hook: Optional[str]
    new_hook: Optional[str]
    hook_source: Optional[str]
    old_layer: Optional[int]
    new_layer: Optional[int]
    #: The resolved layer when the row already records a DIFFERENT one (reported, never written).
    layer_disagreement: Optional[int] = None
    #: Whether the SAE's files were found on disk; a name-only resolution still works.
    files_present: bool = False
    #: Why the hook the row would end up with is refused by the residual-only consumers,
    #: or None. A backfill that records ``mlp`` makes an existing artifact unusable, and
    #: the operator has to see that before applying.
    refusal_after: Optional[str] = None
    #: {kind: count} of rows that reference this SAE. Nothing is deleted.
    dependents: Dict[str, int] = field(default_factory=dict)
    #: Human-readable descriptions of jobs still queued or running against this SAE.
    in_flight: List[str] = field(default_factory=list)

    @property
    def writes_hook(self) -> bool:
        return self.old_hook is None and self.new_hook is not None

    @property
    def writes_layer(self) -> bool:
        return self.old_layer is None and self.new_layer is not None

    @property
    def changes(self) -> bool:
        return self.writes_hook or self.writes_layer

    @property
    def unresolved(self) -> List[str]:
        """The fields that are NULL and stay NULL: nothing on disk or in the names said."""
        missing = []
        if self.old_hook is None and self.new_hook is None:
            missing.append("hook_type")
        if self.old_layer is None and self.new_layer is None:
            missing.append("layer")
        return missing


def candidate_rows(session) -> List[ExternalSAE]:
    """Every row the backfill may consider: a downloaded or locally imported SAE
    missing its hook, its layer, or both.

    Deliberately NOT filtered by status -- a ``deleted`` row can be restored, and a row
    in ``error`` can be retried, so both would be read with a fabricated layer 0 later.
    """
    from sqlalchemy import or_, select

    stmt = (
        select(ExternalSAE)
        .where(ExternalSAE.source.in_(BACKFILLABLE_SOURCES))
        .where(ExternalSAE.training_id.is_(None))
        .where(or_(ExternalSAE.hook_type.is_(None), ExternalSAE.layer.is_(None)))
        .order_by(ExternalSAE.created_at)
    )
    return list(session.execute(stmt).scalars().all())


def row_origins(sae: ExternalSAE) -> Tuple[Optional[str], ...]:
    """The names an SAE is known by, for the parts of the resolver that read a name.

    A HuggingFace row is known by its repository AND its path within it (Gemma Scope 2
    needs both together); a local import by the path it was imported from.
    """
    if sae.source == "huggingface":
        return (sae.hf_repo_id, sae.hf_filepath)
    original = (sae.sae_metadata or {}).get("original_path")
    return (original,) if original else ()


def row_location(sae: ExternalSAE) -> Optional[Path]:
    """Where the SAE's files are, or None when the row records no path."""
    from ..core.config import settings

    if not sae.local_path:
        return None
    try:
        return settings.resolve_data_path(sae.local_path)
    except Exception as exc:  # pragma: no cover - a malformed stored path
        logger.warning("SAE %s: cannot resolve %s (%s)", sae.id, sae.local_path, exc)
        return None


def resolve_row(sae: ExternalSAE) -> Tuple[Optional[str], Optional[str], Optional[int], bool]:
    """``(hook, hook_source, layer, files_present)`` for one row, through the WRITERS' resolver.

    Importing here, not at module scope: ``sae_manager_service`` imports the HuggingFace
    service, and the backfill must not drag that in merely to be imported.
    """
    from ..services.sae_manager_service import resolve_sae_hook, resolve_sae_layer

    location = row_location(sae)
    files_present = bool(location and location.exists())
    origins = row_origins(sae)
    recorded = resolve_sae_hook(location if files_present else None, *origins)
    hook = recorded.hook_type
    layer = resolve_sae_layer(
        location if files_present else None,
        hook or sae.hook_type,
        *origins,
    )
    return hook, recorded.source, layer, files_present


def dependents_of(session, sae_id: str) -> Dict[str, int]:
    """How many rows of each kind reference this SAE. Counted, never deleted.

    Their features and attributions were computed at a point the row is about to name
    differently; the report lists them so an operator can decide what to recompute.
    """
    from sqlalchemy import func, select

    def _count(model, column) -> int:
        return int(session.execute(
            select(func.count()).select_from(model).where(column == sae_id)
        ).scalar() or 0)

    counts = {
        "features": _count(Feature, Feature.external_sae_id),
        "extraction_jobs": _count(ExtractionJob, ExtractionJob.external_sae_id),
        "cluster_profiles": _count(ClusterProfile, ClusterProfile.sae_id),
        "neuronpedia_push_jobs": _count(NeuronpediaPushJob, NeuronpediaPushJob.sae_id),
        "neuronpedia_export_jobs": _count(NeuronpediaExportJob, NeuronpediaExportJob.sae_id),
        "steering_experiments": _count(SteeringExperiment, SteeringExperiment.sae_id),
        "circuit_capture_runs": len(_capture_runs_for(session, sae_id)),
    }
    return {kind: n for kind, n in counts.items() if n}


def _capture_runs_for(session, sae_id: str) -> List[CircuitCaptureRun]:
    """Capture runs whose manifest names this SAE (the manifest is JSONB, so scan it,
    exactly as ``CircuitCaptureService.mark_stale_for_sae`` does)."""
    from sqlalchemy import select

    runs = session.execute(select(CircuitCaptureRun)).scalars().all()
    return [
        run for run in runs
        if any(entry.get("sae_id") == sae_id
               for entry in (run.manifest or {}).get("layers", []))
    ]


def in_flight_for(session, sae_id: str) -> List[str]:
    """Jobs still queued or running against this SAE.

    A running extraction re-reads ``external_sae.layer`` while it works, so changing the
    row underneath one would move where it hooks half way through the corpus.
    """
    from sqlalchemy import select

    blocking: List[str] = []
    for job in session.execute(
        select(ExtractionJob).where(ExtractionJob.external_sae_id == sae_id)
    ).scalars().all():
        if _status_text(job.status) in LIVE_EXTRACTION_STATUSES:
            blocking.append(f"extraction job {job.id} ({_status_text(job.status)})")
    for push in session.execute(
        select(NeuronpediaPushJob).where(NeuronpediaPushJob.sae_id == sae_id)
    ).scalars().all():
        if _status_text(push.status) in LIVE_PUSH_STATUSES:
            blocking.append(f"Neuronpedia push {push.id} ({_status_text(push.status)})")
    for export in session.execute(
        select(NeuronpediaExportJob).where(NeuronpediaExportJob.sae_id == sae_id)
    ).scalars().all():
        if _status_text(export.status) in LIVE_EXPORT_STATUSES:
            blocking.append(f"Neuronpedia export {export.id} ({_status_text(export.status)})")
    for run in _capture_runs_for(session, sae_id):
        if _status_text(run.status) in LIVE_CAPTURE_STATUSES:
            blocking.append(f"circuit capture {run.id} ({_status_text(run.status)})")
    return blocking


def plan_backfill(session) -> List[RowPlan]:
    """What the backfill would do, without doing any of it."""
    from ..services.sae_hook_support import non_residual_hook_reason

    plans: List[RowPlan] = []
    for sae in candidate_rows(session):
        hook, hook_source, layer, files_present = resolve_row(sae)
        new_hook = sae.hook_type if sae.hook_type is not None else hook
        disagreement = None
        if sae.layer is not None and layer is not None and layer != sae.layer:
            disagreement = layer
        new_layer = sae.layer if sae.layer is not None else layer
        plans.append(RowPlan(
            sae_id=sae.id,
            name=sae.name,
            source=str(sae.source),
            status=_status_text(sae.status),
            old_hook=sae.hook_type,
            new_hook=new_hook,
            hook_source=hook_source if sae.hook_type is None else None,
            old_layer=sae.layer,
            new_layer=new_layer,
            layer_disagreement=disagreement,
            files_present=files_present,
            refusal_after=non_residual_hook_reason(new_hook, "Feature extraction"),
            dependents=dependents_of(session, sae.id),
            in_flight=in_flight_for(session, sae.id),
        ))
    return plans


def apply_backfill(session, plans: Sequence[RowPlan]) -> int:
    """Write the planned NULL fields. Returns the number of rows changed.

    Refuses outright -- writing nothing -- when ANY row it would change is referenced by
    an in-flight job: a partial repair with a job reading the other half is worse than
    no repair.
    """
    changing = [plan for plan in plans if plan.changes]
    blocked = [(plan.sae_id, plan.in_flight) for plan in changing if plan.in_flight]
    if blocked:
        detail = "; ".join(
            f"{sae_id}: " + ", ".join(jobs) for sae_id, jobs in blocked
        )
        raise BackfillBlocked(
            "Refusing to backfill while jobs are reading these SAEs -- "
            "wait for them to finish or cancel them, then run again. " + detail
        )

    changed = 0
    stamp = utc_now().isoformat()
    for plan in changing:
        sae = session.get(ExternalSAE, plan.sae_id)
        if sae is None:  # pragma: no cover - removed between the plan and the apply
            logger.warning("SAE %s vanished between the plan and the apply", plan.sae_id)
            continue
        previous = {"hook_type": sae.hook_type, "layer": sae.layer}
        # ONLY NULL FIELDS. Re-checked against the live row, not the plan: the plan may
        # have been built against a row someone has since filled in.
        if sae.hook_type is None and plan.new_hook is not None:
            sae.hook_type = plan.new_hook
        if sae.layer is None and plan.new_layer is not None:
            sae.layer = plan.new_layer
        if sae.hook_type == previous["hook_type"] and sae.layer == previous["layer"]:
            continue
        metadata = dict(sae.sae_metadata or {})
        if plan.hook_source is not None and previous["hook_type"] is None:
            metadata["hook_source"] = plan.hook_source
        metadata["hook_backfill"] = {
            "at": stamp,
            "by": BACKFILL_BY,
            "previous": previous,
        }
        sae.sae_metadata = metadata
        changed += 1
        logger.info(
            "SAE %s: hook %r -> %r, layer %r -> %r",
            sae.id, previous["hook_type"], sae.hook_type, previous["layer"], sae.layer,
        )
    session.commit()
    return changed


def format_report(plans: Sequence[RowPlan], applied: bool = False) -> str:
    """The table an operator reads before deciding to apply."""
    verb = "Changed" if applied else "Would change"
    lines = [
        f"{len(plans)} candidate row(s): source in {BACKFILLABLE_SOURCES}, no training_id, "
        "hook or layer NULL.",
        "",
    ]
    if not plans:
        lines.append("Nothing to do.")
        return "\n".join(lines)

    for plan in plans:
        lines.append(f"── {plan.sae_id}  [{plan.source}/{plan.status}]  {plan.name or ''}")
        lines.append(
            f"   files on disk: {'yes' if plan.files_present else 'NO (resolved by name only)'}"
        )
        lines.append(
            f"   hook : {plan.old_hook!r} -> {plan.new_hook!r}"
            + (f"   (source: {plan.hook_source})" if plan.hook_source else "")
            + ("" if plan.writes_hook else "   [unchanged]")
        )
        lines.append(
            f"   layer: {plan.old_layer!r} -> {plan.new_layer!r}"
            + ("" if plan.writes_layer else "   [unchanged]")
        )
        if plan.layer_disagreement is not None:
            lines.append(
                f"   ⚠ the SAE's files/names say layer {plan.layer_disagreement}, the row says "
                f"{plan.old_layer}. REPORTED ONLY -- a recorded layer is never overwritten."
            )
        if plan.unresolved:
            lines.append(
                "   ⚠ still unrecorded after this run: " + ", ".join(plan.unresolved)
                + " -- nothing on disk or in its names records them."
            )
        if plan.refusal_after:
            lines.append("   ⚠ consumers will now REFUSE this SAE: " + plan.refusal_after)
        if plan.dependents:
            lines.append("   dependents: " + ", ".join(
                f"{kind}={n}" for kind, n in sorted(plan.dependents.items())
            ) + "  (nothing is deleted; recompute what the new hook/layer invalidates)")
        if plan.in_flight:
            lines.append("   ⛔ in flight: " + ", ".join(plan.in_flight))
        lines.append("")

    changing = [plan for plan in plans if plan.changes]
    lines.append(f"{verb}: {len(changing)} row(s).")
    if not applied:
        lines.append("Dry run -- nothing was written. Re-run with --apply to write.")
    return "\n".join(lines)
