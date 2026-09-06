"""
Cluster profile schemas + the portable cluster-definition contract
(Feature 014, IDL-30).

Two families:
- Profile CRUD (`ClusterProfileCreate/Update/Out`) — the durable in-app entity.
- Interchange (`ClusterDefinitionV1` / `ClusterBundleV1`) — the versioned,
  consumer-neutral JSON that travels (export/import; future MILLM / unified-MCP
  / Open WebUI consumers). Strict validators; no secrets; no local paths.
"""

from datetime import datetime
import json
from typing import ClassVar, Any, Dict, List, Literal, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = "1"
DEFINITION_KIND = "mistudio.cluster-definition"
BUNDLE_KIND = "mistudio.cluster-bundle"

MAX_MEMBERS = 20
MAX_BUNDLE = 50
MAX_NAME = 120
MAX_NARRATIVE = 10_000


# ── Shared member shape ─────────────────────────────────────────────────────

class MemberExample(BaseModel):
    """One representative activating snippet (span = the prime token)."""

    model_config = ConfigDict(extra="allow")

    text: Optional[str] = Field(None, max_length=500)
    span: Optional[str] = Field(None, max_length=100)


class MemberMeta(BaseModel):
    """Optional display/interpretability metadata for a member — additive
    contract revision 2026-07-17 (standardized so importing apps can render
    feature tiles without a miStudio round-trip).

    EVERY field is optional and unknown keys are preserved (extra="allow"):
    consumers render what they understand and pass the rest through. Nothing
    here may be load-bearing for steering math — it is display/reference
    data only."""

    model_config = ConfigDict(extra="allow")

    description: Optional[str] = Field(None, max_length=1000)
    category: Optional[str] = Field(None, max_length=50)
    label_source: Optional[str] = Field(None, max_length=50)
    interpretability: Optional[float] = Field(None, ge=0.0, le=1.0)
    mean_activation: Optional[float] = None
    top_tokens: Optional[List[str]] = Field(None, max_length=10)
    signature: Optional[str] = Field(None, max_length=200)
    example: Optional[MemberExample] = None
    neuronpedia: Optional[str] = Field(None, max_length=500)

    #: Ceiling on everything NOT declared above, serialized. MIS-E2E-042.
    #:
    #: Every declared field here is bounded — `max_length` on the strings, a
    #: range on the float, 10 entries on `top_tokens`. The extras were not
    #: bounded at all, and `extra="allow"` means they are persisted AND
    #: re-exported: a circuit definition is the artifact this product exists to
    #: exchange, including through a HuggingFace marketplace. Anything imported
    #: travels onward under miStudio's name.
    #:
    #: 8 KB is far more than display metadata needs (the declared fields cap out
    #: near 2 KB) and far less than a useful smuggling channel.
    MAX_EXTRA_BYTES: ClassVar[int] = 8192

    @model_validator(mode="after")
    def _bound_the_extras(self) -> "MemberMeta":
        extras = self.__pydantic_extra__ or {}
        if not extras:
            return self
        size = len(json.dumps(extras, default=str, separators=(",", ":")))
        if size > self.MAX_EXTRA_BYTES:
            raise ValueError(
                f"member meta carries {size} bytes of undeclared keys "
                f"({sorted(extras)[:5]}…), over the {self.MAX_EXTRA_BYTES}-byte "
                f"limit. `meta` is deliberately extensible so consumers can pass "
                f"through what they do not understand — it is not a payload "
                f"channel."
            )
        return self

    @field_validator("neuronpedia")
    @classmethod
    def http_only(cls, v: Optional[str]) -> Optional[str]:
        """Portable references only — never filesystem paths."""
        if v and not (v.startswith("http://") or v.startswith("https://")):
            raise ValueError("neuronpedia must be an http(s) URL")
        return v


class ProfileMember(BaseModel):
    """A cluster member snapshot with its tuned strength."""

    feature_idx: int = Field(..., ge=0)
    label: Optional[str] = None
    similarity: Optional[float] = Field(None, ge=0.0, le=1.0)
    activation_frequency: Optional[float] = None
    max_activation: Optional[float] = None
    strength: float = Field(..., ge=-300.0, le=300.0)
    sign: Literal[1, -1] = 1
    pinned: bool = False
    meta: Optional[MemberMeta] = None


class ProfileBudget(BaseModel):
    """Allocation snapshot from Feature 013 (self-describing: formula + constants travel)."""

    B: Optional[float] = None
    B_dir: Optional[float] = None
    G: Optional[float] = None
    f_eff: Optional[float] = None
    formula_id: Optional[str] = None
    constants: Optional[Dict[str, float]] = None
    intensity: float = Field(1.0, ge=0.0, le=2.0)
    intensity_range: List[float] = Field(default_factory=lambda: [0.0, 2.0])


# ── Profile CRUD ────────────────────────────────────────────────────────────

class ClusterProfileCreate(BaseModel):
    sae_id: Optional[str] = None
    model_id: Optional[str] = None
    extraction_id: Optional[str] = None
    source_group_id: Optional[str] = None
    name: str = Field(..., min_length=1, max_length=MAX_NAME)
    narrative: Optional[str] = Field(None, max_length=MAX_NARRATIVE)
    display_token: Optional[str] = None
    members: List[ProfileMember] = Field(..., min_length=1, max_length=MAX_MEMBERS)
    budget: Optional[ProfileBudget] = None


class ClusterProfileUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=MAX_NAME)
    narrative: Optional[str] = Field(None, max_length=MAX_NARRATIVE)
    members: Optional[List[ProfileMember]] = Field(None, min_length=1, max_length=MAX_MEMBERS)
    budget: Optional[ProfileBudget] = None


class ClusterProfileOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    sae_id: Optional[str]
    model_id: Optional[str]
    extraction_id: Optional[str]
    source_group_id: Optional[str]
    name: str
    narrative: Optional[str]
    display_token: Optional[str]
    members: List[ProfileMember]
    budget: Optional[ProfileBudget]
    schema_version: str
    imported_from: Optional[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime


class ClusterProfileListResponse(BaseModel):
    data: List[ClusterProfileOut]
    total: int


# ── Interchange format (versioned, consumer-neutral) ───────────────────────

class DefinitionModelRef(BaseModel):
    hf_id: Optional[str] = None
    mistudio_model_id: Optional[str] = None


class DefinitionSAERef(BaseModel):
    """One SAE the definition references.

    `mistudio_sae_id` is the wire name, and `sae_id` is accepted as an input
    alias because that is what every other tool in this codebase calls the same
    value (`start_circuit_capture` takes `[{layer, sae_id}]`). Pydantic's
    default `extra="ignore"` used to turn a caller's `sae_id` into a SILENTLY
    NULL `mistudio_sae_id`: the definition validated, persisted and exported
    clean, then failed much later at miLLM as an unbound SAE, with nothing
    pointing back here.

    An entry that names no SAE id at all is rejected by `CircuitService`, not
    by this model — see the implementation comment below for why the check
    cannot live in the schema without publishing a contract that lies.
    """

    # THE POINT: `sae_id` must not become a silent null.
    #
    # `validation_alias` (NOT `alias`) — a plain `alias` also renames the field
    # on SERIALISATION, which republished the schema with `sae_id` and NO
    # `mistudio_sae_id`. With extra="forbid" that made every previously
    # exported document INVALID. The schema-sync guard caught it before it
    # shipped; without that guard this "fix" would have broken every consumer.
    #
    # `extra="ignore"` is KEPT deliberately, not restored by accident. The
    # published JSON Schema cannot express "accepts either name" — pydantic
    # omits validation aliases in BOTH schema modes — so extra="forbid" would
    # publish a contract that rejects the very alias the model accepts. A
    # schema that lies about what it takes is worse than a permissive one.
    #
    # Typo protection therefore lives in the SERVICE layer instead, where it
    # can see the whole request: see CircuitService._validate. That is the
    # right place anyway — it can name the offending key AND the layer.
    model_config = ConfigDict(populate_by_name=True)

    mistudio_sae_id: Optional[str] = Field(
        None, validation_alias=AliasChoices("mistudio_sae_id", "sae_id")
    )
    layer: Optional[int] = None
    hook_type: Optional[str] = None
    n_features: Optional[int] = None
    d_model: Optional[int] = None
    source_hint: Optional[str] = Field(
        None, description="e.g. 'hf:repo/path' — NEVER an absolute local path"
    )

    @field_validator("source_hint")
    @classmethod
    def no_local_paths(cls, v: Optional[str]) -> Optional[str]:
        """Reject absolute/relative filesystem paths — the format must stay portable."""
        if v and (v.startswith("/") or v.startswith("~") or v.startswith("..") or ":\\" in v):
            raise ValueError("source_hint must not be a filesystem path")
        return v


class DefinitionProvenance(BaseModel):
    created_at: Optional[datetime] = None
    exported_at: Optional[datetime] = None
    mistudio_version: Optional[str] = None
    source_note: Optional[str] = Field(None, max_length=500)


class ClusterDefinitionV1(BaseModel):
    """One portable cluster definition (the mobile artifact — IDL-30)."""

    kind: Literal["mistudio.cluster-definition"] = DEFINITION_KIND
    schema_version: Literal["1"] = SCHEMA_VERSION
    name: str = Field(..., min_length=1, max_length=MAX_NAME)
    narrative: Optional[str] = Field(None, max_length=MAX_NARRATIVE)
    display_token: Optional[str] = None
    model: DefinitionModelRef = Field(default_factory=DefinitionModelRef)
    sae: DefinitionSAERef = Field(default_factory=DefinitionSAERef)
    members: List[ProfileMember] = Field(..., min_length=1, max_length=MAX_MEMBERS)
    budget: Optional[ProfileBudget] = None
    provenance: DefinitionProvenance = Field(default_factory=DefinitionProvenance)


class ClusterBundleV1(BaseModel):
    """A multi-cluster export: an array of definitions in one file."""

    kind: Literal["mistudio.cluster-bundle"] = BUNDLE_KIND
    schema_version: Literal["1"] = SCHEMA_VERSION
    definitions: List[ClusterDefinitionV1] = Field(..., min_length=1, max_length=MAX_BUNDLE)


# ── Import ──────────────────────────────────────────────────────────────────

class ImportRequest(BaseModel):
    """Import a definition or bundle (frontend reads the file client-side)."""

    payload: Dict[str, Any] = Field(..., description="Parsed JSON of a definition or bundle")
    bind_sae_id: Optional[str] = Field(
        None, description="Explicit SAE to bind to (overrides auto-binding)"
    )


class ImportItemResult(BaseModel):
    name: str
    status: Literal["imported", "imported_unbound", "blocked", "error"]
    profile_id: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)
    error: Optional[str] = None


class ImportResponse(BaseModel):
    results: List[ImportItemResult]
    imported: int
    blocked: int
    errors: int


class ExportBundleRequest(BaseModel):
    ids: List[str] = Field(..., min_length=1, max_length=MAX_BUNDLE)
