"""
Cluster profile service (Feature 014, IDL-30).

CRUD for durable cluster profiles, (de)serialization to/from the
`mistudio.cluster-definition/v1` interchange format, and the import
compatibility matrix. Profiles are snapshots decoupled from the recomputable
grouping index; export always serializes the STORED profile (save-then-export),
never live UI state.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.cluster_profile import ClusterProfile
from ..models.external_sae import ExternalSAE
from ..models.feature import Feature
from ..schemas.cluster_profile import (
    MemberMeta,
    BUNDLE_KIND,
    DEFINITION_KIND,
    SCHEMA_VERSION,
    ClusterBundleV1,
    ClusterDefinitionV1,
    ClusterProfileCreate,
    ClusterProfileUpdate,
    DefinitionModelRef,
    DefinitionProvenance,
    DefinitionSAERef,
    ProfileBudget,
    ProfileMember,
)
from ..core.clock import utc_now

logger = logging.getLogger(__name__)

APP_VERSION = "0.5.0"


class CompatibilityDecision:
    """Outcome of matching an incoming definition against local SAEs (FTDD §3)."""

    def __init__(self, action: str, sae_id: Optional[str] = None, warnings: Optional[List[str]] = None):
        self.action = action  # bind | warn_bind | block | unbound
        self.sae_id = sae_id
        self.warnings = warnings or []


def decide_compatibility(
    definition: ClusterDefinitionV1,
    local_saes: List[Dict[str, Any]],
    explicit_sae_id: Optional[str] = None,
) -> CompatibilityDecision:
    """
    Pure compatibility matrix (unit-testable without a DB).

    Args:
        definition: The incoming portable definition.
        local_saes: [{id, n_features, layer, model_name}] of locally available SAEs.
        explicit_sae_id: User-chosen binding target (overrides auto-binding).

    Returns:
        CompatibilityDecision with action ∈ {bind, warn_bind, block, unbound}.
        `block` means the CHOSEN candidate is incompatible (n_features mismatch);
        no local SAE at all ⇒ `unbound` (import as profile, steerable later).
    """
    ref = definition.sae
    by_id = {s["id"]: s for s in local_saes}

    def check(candidate: Dict[str, Any], warnings: List[str]) -> CompatibilityDecision:
        if (
            ref.n_features is not None
            and candidate.get("n_features") is not None
            and ref.n_features != candidate["n_features"]
        ):
            return CompatibilityDecision(
                "block",
                warnings=[
                    f"SAE n_features mismatch: definition={ref.n_features}, "
                    f"local {candidate['id']}={candidate['n_features']} — member indices are meaningless"
                ],
            )
        if (
            definition.model.hf_id
            and candidate.get("model_name")
            and definition.model.hf_id != candidate["model_name"]
        ):
            warnings.append(
                f"Model mismatch: definition={definition.model.hf_id}, "
                f"local SAE model={candidate['model_name']}"
            )
        if ref.layer is not None and candidate.get("layer") is not None and ref.layer != candidate["layer"]:
            warnings.append(f"Layer mismatch: definition L{ref.layer}, local SAE L{candidate['layer']}")
        action = "warn_bind" if warnings else "bind"
        return CompatibilityDecision(action, sae_id=candidate["id"], warnings=warnings)

    # Explicit user choice wins (still bounds-checked).
    if explicit_sae_id:
        cand = by_id.get(explicit_sae_id)
        if not cand:
            return CompatibilityDecision("unbound", warnings=[f"Chosen SAE {explicit_sae_id} not found"])
        return check(cand, [])

    if not local_saes:
        return CompatibilityDecision("unbound", warnings=["No local SAEs — imported unbound"])

    # Same-id binding is silent when compatible.
    if ref.mistudio_sae_id and ref.mistudio_sae_id in by_id:
        return check(by_id[ref.mistudio_sae_id], [])

    # Otherwise prefer candidates matching n_features (+ layer when known).
    candidates = [
        s for s in local_saes
        if ref.n_features is None or s.get("n_features") == ref.n_features
    ]
    if not candidates:
        return CompatibilityDecision(
            "block",
            warnings=[
                f"No local SAE has n_features={ref.n_features}; import as unbound profile instead"
            ],
        )
    layered = [s for s in candidates if ref.layer is None or s.get("layer") == ref.layer]
    pick = (layered or candidates)[0]
    return check(pick, [f"Bound to {pick['id']} (id differs from definition)"])


class ClusterProfileService:
    """DB-facing CRUD + serialization for cluster profiles."""

    @staticmethod
    async def _enrich_members(
        db: AsyncSession,
        members: List[ProfileMember],
        sae_id: Optional[str],
        extraction_id: Optional[str],
    ) -> List[ProfileMember]:
        """Fill member.label + member.meta from the Feature records
        (contract rev 2026-07-17): the group index carries no labels, so
        saves/exports were shipping label=null while features had rich
        names/descriptions. Enrichment is best-effort and NEVER overwrites
        values already present (imported definitions keep their authors'
        words). Display-only — steering math never reads meta."""
        need = [m for m in members if m.label is None or m.meta is None]
        if not need or (not sae_id and not extraction_id):
            return members
        q = select(Feature).where(
            Feature.neuron_index.in_([m.feature_idx for m in need])
        )
        if extraction_id:
            q = q.where(Feature.extraction_job_id == extraction_id)
        else:
            q = q.where(Feature.external_sae_id == sae_id)
        rows = (await db.execute(q)).scalars().all()
        by_idx: Dict[int, Any] = {}
        for row in rows:  # newest extraction wins; None timestamps lose
            existing = by_idx.get(row.neuron_index)
            if existing is None:
                by_idx[row.neuron_index] = row
                continue
            row_ts = row.created_at
            existing_ts = existing.created_at
            if existing_ts is None or (row_ts is not None
                                       and row_ts > existing_ts):
                by_idx[row.neuron_index] = row
        enriched: List[ProfileMember] = []
        for member in members:
            feature = by_idx.get(member.feature_idx)
            if feature is None:
                enriched.append(member)
                continue
            update: Dict[str, Any] = {}
            if member.label is None and feature.name:
                update["label"] = feature.name
            if member.meta is None:
              try:
                nlp = feature.nlp_analysis or {}
                top_tokens = None
                dist = ((nlp.get("prime_token_analysis") or {})
                        .get("word_lowercase_distribution") or {})
                if dist:
                    top_tokens = [w for w, _ in sorted(
                        dist.items(), key=lambda kv: -kv[1])[:5]]
                signature = None
                patterns = nlp.get("context_patterns") or {}
                prefixes = patterns.get("prefix_trigrams") or []
                suffixes = patterns.get("suffix_bigrams") or []
                if prefixes and suffixes:
                    signature = (" ".join(prefixes[0].get("tokens", []))
                                 + " ___ "
                                 + " ".join(suffixes[0].get("tokens", [])))[:200]
                label_source = feature.label_source
                if hasattr(label_source, "value"):
                    label_source = label_source.value
                score = feature.interpretability_score
                if score is not None:
                    score = max(0.0, min(1.0, float(score)))
                # Truncate to the contract's field caps: DB columns are wider
                # (category String(255), description unbounded Text) and a
                # best-effort enrichment must NEVER 500 a save/export
                # (contract-rev review #1)
                update["meta"] = MemberMeta(
                    description=(feature.description or None)
                    and feature.description[:1000],
                    category=(feature.category or None)
                    and feature.category[:50],
                    label_source=(label_source or None)
                    and str(label_source)[:50],
                    interpretability=score,
                    mean_activation=feature.mean_activation,
                    top_tokens=top_tokens,
                    signature=signature,
                )
              except Exception:
                logger.warning(
                    "member_meta_enrichment_failed",
                    extra={"feature_idx": member.feature_idx},
                )
            enriched.append(member.model_copy(update=update) if update
                            else member)
        return enriched

    @staticmethod
    async def create(db: AsyncSession, data: ClusterProfileCreate) -> ClusterProfile:
        """Create a profile; validates member bounds against the bound SAE."""
        await ClusterProfileService._validate_bounds(db, data.sae_id, data.members)
        data.members = await ClusterProfileService._enrich_members(
            db, data.members, data.sae_id, data.extraction_id
        )
        profile = ClusterProfile(
            sae_id=data.sae_id,
            model_id=data.model_id,
            extraction_id=data.extraction_id,
            source_group_id=data.source_group_id,
            name=data.name,
            narrative=data.narrative,
            display_token=data.display_token,
            members=[m.model_dump() for m in data.members],
            budget=data.budget.model_dump() if data.budget else None,
            schema_version=SCHEMA_VERSION,
        )
        db.add(profile)
        await db.commit()
        await db.refresh(profile)
        return profile

    @staticmethod
    async def list(
        db: AsyncSession, sae_id: Optional[str] = None, search: Optional[str] = None
    ) -> Tuple[List[ClusterProfile], int]:
        """List profiles, newest first, optionally filtered by SAE and name/token search."""
        q = select(ClusterProfile)
        if sae_id:
            q = q.where(ClusterProfile.sae_id == sae_id)
        if search:
            like = f"%{search}%"
            q = q.where(
                ClusterProfile.name.ilike(like) | ClusterProfile.display_token.ilike(like)
            )
        q = q.order_by(ClusterProfile.updated_at.desc())
        rows = (await db.execute(q)).scalars().all()
        return list(rows), len(rows)

    @staticmethod
    async def get(db: AsyncSession, profile_id: str) -> Optional[ClusterProfile]:
        return (
            await db.execute(select(ClusterProfile).where(ClusterProfile.id == profile_id))
        ).scalar_one_or_none()

    @staticmethod
    async def update(
        db: AsyncSession, profile: ClusterProfile, data: ClusterProfileUpdate
    ) -> ClusterProfile:
        if data.name is not None:
            profile.name = data.name
        # fields_set distinguishes "omitted" from an explicit null — narrative
        # must be CLEARABLE via PATCH {"narrative": null} (review finding 014-F).
        if "narrative" in data.model_fields_set:
            profile.narrative = data.narrative
        if data.members is not None:
            await ClusterProfileService._validate_bounds(db, profile.sae_id, data.members)
            enriched = await ClusterProfileService._enrich_members(
                db, data.members, profile.sae_id, profile.extraction_id
            )
            profile.members = [m.model_dump() for m in enriched]
        if data.budget is not None:
            profile.budget = data.budget.model_dump()
        profile.updated_at = utc_now()
        await db.commit()
        await db.refresh(profile)
        return profile

    @staticmethod
    async def delete(db: AsyncSession, profile: ClusterProfile) -> None:
        await db.delete(profile)
        await db.commit()

    @staticmethod
    async def count_for_sae(db: AsyncSession, sae_id: str) -> int:
        """Used by the SAE-delete guard (structured 409 with profile count)."""
        return (
            await db.execute(
                select(func.count()).select_from(ClusterProfile).where(ClusterProfile.sae_id == sae_id)
            )
        ).scalar_one()

    @staticmethod
    async def unbind_for_sae(db: AsyncSession, sae_id: str) -> int:
        """
        Detach all profiles from an SAE about to be deleted (force path of the
        delete guard). Profiles survive unbound — user-authored narratives and
        tuned strengths are never destroyed by an SAE deletion.
        """
        profiles = (
            (await db.execute(select(ClusterProfile).where(ClusterProfile.sae_id == sae_id)))
            .scalars()
            .all()
        )
        for p in profiles:
            p.sae_id = None
        await db.commit()
        return len(profiles)

    # ── Serialization ───────────────────────────────────────────────────────

    @staticmethod
    async def to_definition(db: AsyncSession, profile: ClusterProfile) -> ClusterDefinitionV1:
        """Serialize a STORED profile to the portable interchange format."""
        sae_ref = DefinitionSAERef()
        model_ref = DefinitionModelRef(mistudio_model_id=profile.model_id)
        if profile.sae_id:
            sae = (
                await db.execute(select(ExternalSAE).where(ExternalSAE.id == profile.sae_id))
            ).scalar_one_or_none()
            if sae:
                sae_ref = DefinitionSAERef(
                    mistudio_sae_id=sae.id,
                    layer=sae.layer,
                    hook_type=sae.hook_type,
                    n_features=sae.n_features,
                    d_model=sae.d_model,
                    source_hint=f"hf:{sae.hf_repo_id}/{sae.hf_filepath}" if sae.hf_repo_id else None,
                )
                model_ref = DefinitionModelRef(
                    hf_id=sae.model_name, mistudio_model_id=profile.model_id or sae.model_id
                )
        members = await ClusterProfileService._enrich_members(
            db,
            [ProfileMember(**m) for m in profile.members],
            profile.sae_id,
            profile.extraction_id,
        )
        return ClusterDefinitionV1(
            name=profile.name,
            narrative=profile.narrative,
            display_token=profile.display_token,
            model=model_ref,
            sae=sae_ref,
            members=members,
            budget=ProfileBudget(**profile.budget) if profile.budget else None,
            provenance=DefinitionProvenance(
                created_at=profile.created_at,
                exported_at=utc_now(),
                mistudio_version=APP_VERSION,
            ),
        )

    @staticmethod
    def from_definition(
        definition: ClusterDefinitionV1, bind_sae_id: Optional[str]
    ) -> ClusterProfile:
        """Materialize an imported definition as a profile (binding decided upstream)."""
        return ClusterProfile(
            sae_id=bind_sae_id,
            model_id=definition.model.mistudio_model_id,
            name=definition.name,
            narrative=definition.narrative,
            display_token=definition.display_token,
            members=[m.model_dump() for m in definition.members],
            budget=definition.budget.model_dump() if definition.budget else None,
            schema_version=SCHEMA_VERSION,
            imported_from={
                "kind": DEFINITION_KIND,
                "schema_version": definition.schema_version,
                "exported_at": definition.provenance.exported_at.isoformat()
                if definition.provenance.exported_at
                else None,
                "source_sae_id": definition.sae.mistudio_sae_id,
            },
        )

    @staticmethod
    async def _validate_bounds(
        db: AsyncSession, sae_id: Optional[str], members: List[ProfileMember]
    ) -> None:
        """Member indices must fit the bound SAE (unbound profiles skip this)."""
        if not sae_id:
            return
        sae = (
            await db.execute(select(ExternalSAE).where(ExternalSAE.id == sae_id))
        ).scalar_one_or_none()
        if sae is None:
            raise ValueError(f"SAE not found: {sae_id}")
        if sae.n_features:
            bad = [m.feature_idx for m in members if m.feature_idx >= sae.n_features]
            if bad:
                raise ValueError(
                    f"Member indices out of bounds for SAE {sae_id} ({sae.n_features} features): {bad}"
                )


def parse_import_payload(payload: Dict[str, Any]) -> List[ClusterDefinitionV1]:
    """
    Parse an import payload as a definition or bundle (strict, versioned).

    Raises:
        ValueError: unknown kind or unsupported schema major version.
    """
    kind = payload.get("kind")
    if kind == DEFINITION_KIND:
        return [ClusterDefinitionV1.model_validate(payload)]
    if kind == BUNDLE_KIND:
        return list(ClusterBundleV1.model_validate(payload).definitions)
    raise ValueError(
        f"Unknown kind {kind!r} — expected {DEFINITION_KIND!r} or {BUNDLE_KIND!r}"
    )
