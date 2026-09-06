"""
mistudio.circuit-definition/v1 — the portable circuit contract (IDL-33).

A NEW kind in the interchange family; mistudio.cluster-definition/v1 is
untouched (additive-only rule — old consumers reject unknown kinds cleanly).
Carries per-layer SAE refs, members keyed to layers (feature or cluster
refs), edges with their full evidence (rung, type, statistics, attribution,
validation-manifest refs), per-layer budgets + a global intensity, optional
faithfulness scores, and discovery provenance. Position/attention fields are
present-but-nullable from day one (Tier-2.5 lands with no migration —
CIRCUITS-002 BR-023).

Amendments land INSIDE v1 pre-freeze (CIRCUITS-002). Vendored JSON schema:
docs/schemas/circuit-definition-v1.json (sync-tested).
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from .cluster_profile import (
    MAX_MEMBERS,
    MAX_NAME,
    MAX_NARRATIVE,
    ClusterDefinitionV1,
    DefinitionModelRef,
    DefinitionProvenance,
    DefinitionSAERef,
    ProfileBudget,
    ProfileMember,
)
from .evidence_ladder import EvidenceRung, circuit_rung

CIRCUIT_KIND = "mistudio.circuit-definition"
CIRCUIT_SCHEMA_VERSION = "1"
MAX_LAYERS = 16
MAX_EDGES = 200


class CircuitNodeRef(BaseModel):
    """An edge endpoint: a feature (or cluster supernode) at a layer."""

    layer: int = Field(..., ge=0)
    kind: Literal["feature", "cluster"] = "feature"
    feature_idx: Optional[int] = Field(None, ge=0)
    cluster_profile_id: Optional[str] = None

    @model_validator(mode="after")
    def _endpoint_shape(self) -> "CircuitNodeRef":
        if self.kind == "feature" and self.feature_idx is None:
            raise ValueError("feature endpoint requires feature_idx")
        if self.kind == "cluster" and not self.cluster_profile_id:
            raise ValueError("cluster endpoint requires cluster_profile_id")
        return self

    def key(self) -> tuple:
        return (self.layer, self.kind,
                self.feature_idx if self.kind == "feature" else self.cluster_profile_id)


class EdgeCoactivation(BaseModel):
    """Tier-1 statistics snapshot (CIRCUITS-002 A.3)."""

    pmi: Optional[float] = None
    lift: Optional[float] = None
    support: Optional[int] = Field(None, ge=0)
    spearman: Optional[float] = None
    null_percentile: Optional[float] = None
    replicated_heldout: Optional[bool] = None
    corpus_ref: Optional[str] = None


class EdgeAttribution(BaseModel):
    """Tier-2 attribution snapshot (CIRCUITS-002 A.6)."""

    score: Optional[float] = None
    sign_consistency: Optional[float] = Field(None, ge=0.0, le=1.0)
    method: Optional[Literal["raw", "ig_lite"]] = None


class EdgePosition(BaseModel):
    """Tier-2.5 fields — nullable by design until the fast-follow lands (BR-023)."""

    source_role: Optional[str] = None
    target_role: Optional[str] = None
    mediating_heads: Optional[List[int]] = None


class CircuitEdge(BaseModel):
    up: CircuitNodeRef
    down: CircuitNodeRef
    type: Literal["computed", "persistence", "attention_mediated"] = "computed"
    # Classifier disclosure (BR-021): numeric signals PLUS method strings,
    # threshold maps and vote maps — deliberately open-typed so the
    # classifier's full disclosure always fits (review R1 finding #1).
    type_signals: Optional[Dict[str, Any]] = None
    rung: EvidenceRung = EvidenceRung.MINED
    tested_and_failed: List[EvidenceRung] = Field(default_factory=list)
    coactivation: Optional[EdgeCoactivation] = None
    weight_prior: Optional[float] = None
    attribution: Optional[EdgeAttribution] = None
    validation_manifest_ref: Optional[str] = None
    effect_size: Optional[float] = None  # measured ES when rung >= 2 (hazard-v2 consumes)
    position: Optional[EdgePosition] = None

    @model_validator(mode="after")
    def _direction(self) -> "CircuitEdge":
        if self.up.layer >= self.down.layer:
            raise ValueError("edge must go from a lower layer to a higher layer")
        return self


class CircuitMember(BaseModel):
    """A circuit member at a layer — a feature ref or a cluster (supernode) ref.

    Cluster members carry an expansion snapshot so exports stay portable even
    if the referenced profile later changes (live records resolve dynamically;
    the definition freezes the membership at export time).
    """

    layer: int = Field(..., ge=0)
    member_kind: Literal["feature_ref", "cluster_ref"] = "feature_ref"
    # feature_ref payload (ProfileMember shape, feature_idx required)
    feature: Optional[ProfileMember] = None
    # cluster_ref payload
    cluster_profile_id: Optional[str] = None
    cluster_name: Optional[str] = None
    expanded_members: Optional[List[ProfileMember]] = Field(None, max_length=MAX_MEMBERS)

    @model_validator(mode="after")
    def _member_shape(self) -> "CircuitMember":
        if self.member_kind == "feature_ref" and self.feature is None:
            raise ValueError("feature_ref member requires feature")
        if self.member_kind == "cluster_ref":
            if not self.cluster_profile_id:
                raise ValueError("cluster_ref member requires cluster_profile_id")
            if not self.expanded_members:
                raise ValueError(
                    "cluster_ref member requires expanded_members (portability snapshot)"
                )
        return self

    def node_keys(self) -> List[tuple]:
        """Endpoint keys this member can satisfy."""
        keys: List[tuple] = []
        if self.member_kind == "feature_ref" and self.feature is not None:
            keys.append((self.layer, "feature", self.feature.feature_idx))
        if self.member_kind == "cluster_ref":
            keys.append((self.layer, "cluster", self.cluster_profile_id))
            for m in self.expanded_members or []:
                keys.append((self.layer, "feature", m.feature_idx))
        return keys


class CircuitBudget(BaseModel):
    """Per-layer budgets composed under one global intensity (IDL-31)."""

    formula_id: Optional[str] = "freq-budget/sim-alloc/per-layer@1"
    layers: Dict[str, ProfileBudget] = Field(default_factory=dict)  # key: str(layer)
    intensity: float = Field(1.0, ge=0.0, le=2.0)
    intensity_range: List[float] = Field(default_factory=lambda: [0.0, 2.0])

    @model_validator(mode="after")
    def _dial_within_range(self) -> "CircuitBudget":
        # The calibration guarantee ("a served dial cannot exceed the cliff")
        # rests on intensity_range being the true envelope AND intensity sitting
        # inside it. We enforce the SHAPE strictly (2 ordered elements — a
        # malformed range is a real error) but CLAMP intensity into the range
        # rather than reject it. Rejecting would break backward compatibility:
        # a legacy/hand-authored budget could carry a narrowed range (e.g.
        # [0, 0.5]) while intensity kept its default 1.0 — those documents must
        # still round-trip. Clamping normalizes them to a servable state instead
        # of failing every subsequent update/import/write-back (Feature 20 R2).
        r = self.intensity_range
        if len(r) != 2:
            raise ValueError(f"intensity_range must be [lo, hi]; got {r!r}")
        lo, hi = r
        if lo > hi:
            raise ValueError(
                f"intensity_range must be ordered [lo, hi]; got [{lo}, {hi}]")
        if self.intensity < lo:
            self.intensity = lo
        elif self.intensity > hi:
            self.intensity = hi
        return self


class CircuitFaithfulness(BaseModel):
    """Circuit-level faithfulness snapshot (CIRCUITS-002 A.7); badge, not gate."""

    necessity: Optional[float] = None
    sufficiency: Optional[float] = None
    sufficiency_k: Optional[int] = None
    metric_id: Optional[str] = None
    manifest_ref: Optional[str] = None


class CalibrationProbe(BaseModel):
    """One falsifiable correctness probe used to find the cliff (IDL-37).

    Deliberately a NEUTRAL factual question — a topic the circuit should NOT
    influence — so that degradation shows up as the circuit's tint corrupting an
    unrelated fact (detectable), rather than as a change to output on the
    circuit's own concept (whose "right answer" is not falsifiable). The probe
    travels in the contract so a serve-time re-verify is one cheap pass.
    """

    prompt: str = Field(..., min_length=1, max_length=2000)
    expected: str = Field(..., min_length=1, max_length=2000)


class CircuitCalibration(BaseModel):
    """Calibrated usable-strength band for the circuit's serving dial (IDL-37).

    Badge, not gate. When present, its `[onset, cliff]` is what
    `CircuitBudget.intensity_range` is clamped to (a served dial cannot reach
    the nonsense zone) and `sweet_spot` is the default `intensity`.

    `provisional` is True whenever the probes were GENERATED (not authored) or
    the band was measured on the discovery-plane model rather than re-verified
    against the served model — an honest confidence marker, never a blocker.
    """

    onset: float = Field(..., ge=0.0, description="min dial with influence above baseline noise")
    sweet_spot: float = Field(..., ge=0.0, description="usable dial: strongest still-correct point, with margin")
    cliff: float = Field(..., ge=0.0, description="max dial still correct; above this the judge flips to broken")
    provisional: bool = True
    probe_set: List[CalibrationProbe] = Field(default_factory=list)
    judge_metric_id: Optional[str] = None
    step_budget: Optional[int] = Field(None, ge=1)
    non_monotone: bool = False
    manifest_ref: Optional[str] = None

    @model_validator(mode="after")
    def _ordered(self) -> "CircuitCalibration":
        # onset ≤ sweet_spot ≤ cliff, or the band is meaningless and the clamp
        # would produce an inverted/empty intensity_range.
        if not (self.onset <= self.sweet_spot <= self.cliff):
            raise ValueError(
                f"calibration band must satisfy onset ≤ sweet_spot ≤ cliff; "
                f"got onset={self.onset}, sweet_spot={self.sweet_spot}, "
                f"cliff={self.cliff}"
            )
        return self


class CircuitDiscoveryProvenance(BaseModel):
    mode: Optional[Literal["seeded", "open"]] = None
    granularity: Optional[Literal["feature", "cluster"]] = None
    corpus_ref: Optional[str] = None
    split: Optional[str] = None
    thresholds: Optional[Dict[str, float]] = None
    discovered_at: Optional[str] = None


class CircuitDefinitionV1(BaseModel):
    """One portable circuit definition (IDL-33)."""

    kind: Literal["mistudio.circuit-definition"] = CIRCUIT_KIND
    schema_version: Literal["1"] = CIRCUIT_SCHEMA_VERSION
    name: str = Field(..., min_length=1, max_length=MAX_NAME)
    narrative: Optional[str] = Field(None, max_length=MAX_NARRATIVE)
    model: DefinitionModelRef = Field(default_factory=DefinitionModelRef)
    saes: List[DefinitionSAERef] = Field(..., min_length=1, max_length=MAX_LAYERS)
    members: List[CircuitMember] = Field(..., min_length=1)
    edges: List[CircuitEdge] = Field(default_factory=list, max_length=MAX_EDGES)
    budget: Optional[CircuitBudget] = None
    faithfulness: Optional[CircuitFaithfulness] = None
    calibration: Optional[CircuitCalibration] = None
    provenance: DefinitionProvenance = Field(default_factory=DefinitionProvenance)
    discovery: Optional[CircuitDiscoveryProvenance] = None

    @model_validator(mode="after")
    def _structure(self) -> "CircuitDefinitionV1":
        # Per-layer SAE refs must carry layers and be unique per layer.
        sae_layers: List[int] = []
        for ref in self.saes:
            if ref.layer is None:
                raise ValueError("every circuit SAE ref must carry its layer")
            sae_layers.append(ref.layer)
        if len(set(sae_layers)) != len(sae_layers):
            raise ValueError("duplicate SAE ref for a layer")

        # Members: per-layer cap (BR-025 — caps apply PER LAYER, never total),
        # and every member's layer must have an SAE ref.
        by_layer: Dict[int, int] = {}
        node_keys = set()
        seen_features: set = set()
        for m in self.members:
            if m.layer not in sae_layers:
                raise ValueError(f"member at layer {m.layer} has no SAE ref")
            weight = len(m.expanded_members) if m.member_kind == "cluster_ref" else 1
            by_layer[m.layer] = by_layer.get(m.layer, 0) + weight
            # A feature may appear once per layer — as a feature_ref OR inside
            # one cluster expansion. Duplicates would double-steer the feature
            # in every slice/profile consumer (review R1 finding #7).
            for key in m.node_keys():
                if key[1] != "feature":
                    continue
                if key in seen_features:
                    raise ValueError(
                        f"feature {key[2]} at layer {key[0]} appears more than once "
                        "(as a member or inside a cluster expansion)"
                    )
                seen_features.add(key)
            node_keys.update(m.node_keys())
        for layer, count in by_layer.items():
            if count > MAX_MEMBERS:
                raise ValueError(
                    f"layer {layer} exceeds the per-layer member cap "
                    f"({count} > {MAX_MEMBERS})"
                )

        # Edges must connect declared members (by feature or cluster key).
        for e in self.edges:
            for endpoint in (e.up, e.down):
                if endpoint.key() not in node_keys:
                    raise ValueError(
                        f"edge endpoint {endpoint.key()} does not reference a circuit member"
                    )
        return self

    def displayed_rung(self) -> EvidenceRung:
        return circuit_rung([e.rung for e in self.edges])


# ── Projection (BR-014) ─────────────────────────────────────────────────────

def to_layer_slice(defn: CircuitDefinitionV1, layer: int) -> ClusterDefinitionV1:
    """Project one layer of a circuit as a VALID cluster-definition/v1 slice.

    The slice is an ordinary v1 definition to every consumer (miLLM imports it
    unchanged); the partial-rendering markers travel in display-only fields
    (name suffix + provenance.source_note) because v1 has no top-level meta —
    slices must never add non-v1 fields.
    """
    sae = next((s for s in defn.saes if s.layer == layer), None)
    if sae is None:
        raise ValueError(f"circuit has no SAE ref for layer {layer}")

    members: List[ProfileMember] = []
    for m in defn.members:
        if m.layer != layer:
            continue
        if m.member_kind == "feature_ref" and m.feature is not None:
            members.append(m.feature)
        elif m.member_kind == "cluster_ref":
            members.extend(m.expanded_members or [])
    if not members:
        raise ValueError(f"circuit has no members at layer {layer}")
    if len(members) > MAX_MEMBERS:
        raise ValueError(f"layer {layer} expands beyond the v1 member cap")

    budget = None
    if defn.budget is not None:
        layer_budget = defn.budget.layers.get(str(layer))
        if layer_budget is not None:
            budget = layer_budget.model_copy(
                update={"intensity": defn.budget.intensity,
                        "intensity_range": list(defn.budget.intensity_range)}
            )

    parent_rung = int(defn.displayed_rung())
    marker = (
        f"projection_of={defn.name!r} circuit; parent_rung={parent_rung}; "
        f"partial_rendering=true — a slice is NOT the circuit"
    )
    suffix = f" [L{layer} slice]"
    base = defn.name[: MAX_NAME - len(suffix)]
    return ClusterDefinitionV1(
        name=f"{base}{suffix}",
        narrative=defn.narrative,
        display_token=None,
        model=defn.model,
        sae=sae,
        members=members,
        budget=budget,
        provenance=DefinitionProvenance(
            created_at=defn.provenance.created_at,
            exported_at=defn.provenance.exported_at,
            mistudio_version=defn.provenance.mistudio_version,
            source_note=marker[:500],
        ),
    )
