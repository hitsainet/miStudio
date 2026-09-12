"""
Cluster profile API endpoints (Feature 014, IDL-30).

CRUD for durable cluster profiles plus the portable-definition surface:
export (single definition / bundle) and import with the compatibility matrix.
Exports always serialize the STORED profile (save-then-export); import caps
are enforced here (1 MB body, ≤50 definitions, ≤20 members each — the member
cap rides on the schema).
"""

import json
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.deps import get_db
from src.models.cluster_profile import ClusterProfile
from src.models.external_sae import ExternalSAE
from src.schemas.cluster_profile import (
    BUNDLE_KIND,
    MAX_BUNDLE,
    ClusterBundleV1,
    ClusterProfileCreate,
    ClusterProfileListResponse,
    ClusterProfileOut,
    ClusterProfileUpdate,
    ExportBundleRequest,
    ImportItemResult,
    ImportRequest,
    ImportResponse,
)
from src.services.cluster_profile_service import (
    ClusterProfileService,
    decide_compatibility,
    parse_import_payload,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/cluster-profiles", tags=["cluster-profiles"])

MAX_IMPORT_BYTES = 1_048_576  # 1 MB — a definition is a few KB; anything near this is hostile


async def _get_profile_or_404(db: AsyncSession, profile_id: str) -> ClusterProfile:
    profile = await ClusterProfileService.get(db, profile_id)
    if not profile:
        raise HTTPException(status_code=404, detail=f"Cluster profile {profile_id} not found")
    return profile


async def _local_sae_summaries(db: AsyncSession) -> list:
    rows = (
        await db.execute(
            select(ExternalSAE.id, ExternalSAE.n_features, ExternalSAE.layer, ExternalSAE.model_name)
        )
    ).all()
    return [
        {"id": r.id, "n_features": r.n_features, "layer": r.layer, "model_name": r.model_name}
        for r in rows
    ]


# ── CRUD ────────────────────────────────────────────────────────────────────

@router.get("", response_model=ClusterProfileListResponse)
async def list_profiles(
    sae_id: Optional[str] = Query(None, description="Filter by bound SAE"),
    search: Optional[str] = Query(None, description="Name/display-token substring"),
    db: AsyncSession = Depends(get_db),
):
    profiles, total = await ClusterProfileService.list(db, sae_id=sae_id, search=search)
    return ClusterProfileListResponse(
        data=[ClusterProfileOut.model_validate(p) for p in profiles], total=total
    )


@router.post("", response_model=ClusterProfileOut, status_code=201)
async def create_profile(data: ClusterProfileCreate, db: AsyncSession = Depends(get_db)):
    try:
        profile = await ClusterProfileService.create(db, data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return ClusterProfileOut.model_validate(profile)


@router.get("/{profile_id}", response_model=ClusterProfileOut)
async def get_profile(profile_id: str, db: AsyncSession = Depends(get_db)):
    return ClusterProfileOut.model_validate(await _get_profile_or_404(db, profile_id))


@router.patch("/{profile_id}", response_model=ClusterProfileOut)
async def update_profile(
    profile_id: str, data: ClusterProfileUpdate, db: AsyncSession = Depends(get_db)
):
    profile = await _get_profile_or_404(db, profile_id)
    try:
        profile = await ClusterProfileService.update(db, profile, data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return ClusterProfileOut.model_validate(profile)


@router.delete("/{profile_id}")
async def delete_profile(profile_id: str, db: AsyncSession = Depends(get_db)):
    profile = await _get_profile_or_404(db, profile_id)
    await ClusterProfileService.delete(db, profile)
    return {"message": f"Cluster profile {profile_id} deleted"}


# ── Export ──────────────────────────────────────────────────────────────────

def _slug(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in name.lower())[:60] or "cluster"


@router.get("/{profile_id}/export")
async def export_profile(profile_id: str, db: AsyncSession = Depends(get_db)):
    """Export ONE profile as a portable `mistudio.cluster-definition/v1` JSON file."""
    profile = await _get_profile_or_404(db, profile_id)
    definition = await ClusterProfileService.to_definition(db, profile)
    return JSONResponse(
        content=json.loads(definition.model_dump_json(exclude_none=True)),
        headers={
            "Content-Disposition": f'attachment; filename="{_slug(profile.name)}.cluster.json"'
        },
    )


@router.post("/export-bundle")
async def export_bundle(request: ExportBundleRequest, db: AsyncSession = Depends(get_db)):
    """Export several profiles as one `mistudio.cluster-bundle/v1` file."""
    definitions = []
    for pid in request.ids:
        profile = await _get_profile_or_404(db, pid)
        definitions.append(await ClusterProfileService.to_definition(db, profile))
    bundle = ClusterBundleV1(definitions=definitions)
    return JSONResponse(
        content=json.loads(bundle.model_dump_json(exclude_none=True)),
        headers={"Content-Disposition": 'attachment; filename="clusters.bundle.json"'},
    )


# ── Import ──────────────────────────────────────────────────────────────────

@router.post("/import", response_model=ImportResponse)
async def import_profiles(request: ImportRequest, db: AsyncSession = Depends(get_db)):
    """
    Import a definition or bundle. Per-item results: each definition binds,
    warn-binds, imports unbound, blocks (n_features mismatch), or errors —
    one bad item never poisons the rest of a bundle.
    """
    # This is NOT the transport cap — `core.body_limit` bounds what arrives on
    # the wire for every route (MIS-E2E-036). This bounds what the payload
    # becomes once parsed, which is a different quantity: a compact document
    # can re-serialize far larger. Keep both.
    approx_size = len(json.dumps(request.payload, separators=(",", ":")))
    if approx_size > MAX_IMPORT_BYTES:
        raise HTTPException(status_code=413, detail="Import payload exceeds 1 MB")

    if request.payload.get("kind") == BUNDLE_KIND:
        raw_defs = request.payload.get("definitions")
        if isinstance(raw_defs, list) and len(raw_defs) > MAX_BUNDLE:
            raise HTTPException(status_code=400, detail=f"Bundle exceeds {MAX_BUNDLE} definitions")

    try:
        definitions = parse_import_payload(request.payload)
    except (ValueError, ValidationError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid import payload: {e}")

    local_saes = await _local_sae_summaries(db)
    results: list[ImportItemResult] = []
    imported = blocked = errors = 0

    for definition in definitions:
        try:
            decision = decide_compatibility(definition, local_saes, request.bind_sae_id)
            if decision.action == "block":
                blocked += 1
                results.append(
                    ImportItemResult(
                        name=definition.name, status="blocked", warnings=decision.warnings
                    )
                )
                continue
            bind_id = decision.sae_id if decision.action in ("bind", "warn_bind") else None
            # Member indices must fit the SAE actually being bound — metadata
            # in the definition can lie or be absent (review finding 014-B).
            if bind_id:
                cand = next((s for s in local_saes if s["id"] == bind_id), None)
                if cand and cand.get("n_features"):
                    bad = [
                        m.feature_idx for m in definition.members
                        if m.feature_idx >= cand["n_features"]
                    ]
                    if bad:
                        blocked += 1
                        results.append(
                            ImportItemResult(
                                name=definition.name,
                                status="blocked",
                                warnings=decision.warnings
                                + [f"Member indices out of bounds for {bind_id} "
                                   f"({cand['n_features']} features): {bad}"],
                            )
                        )
                        continue
            profile = ClusterProfileService.from_definition(definition, bind_id)
            db.add(profile)
            await db.commit()
            await db.refresh(profile)
            imported += 1
            results.append(
                ImportItemResult(
                    name=definition.name,
                    status="imported" if bind_id else "imported_unbound",
                    profile_id=profile.id,
                    warnings=decision.warnings,
                )
            )
        except Exception as e:  # per-item isolation
            await db.rollback()
            logger.exception("Import failed for definition %r", definition.name)
            errors += 1
            # Validation messages help the user fix their file; anything else
            # stays generic (no internals in responses — house security rule).
            detail = str(e) if isinstance(e, (ValueError, ValidationError)) else "internal error"
            results.append(
                ImportItemResult(name=definition.name, status="error", error=detail)
            )

    return ImportResponse(results=results, imported=imported, blocked=blocked, errors=errors)
