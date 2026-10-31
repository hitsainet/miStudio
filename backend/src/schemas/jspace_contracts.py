"""
J-space interchange kinds (BR-021, BR-022, BR-023).

ADDITIVE ONLY. New kinds are added; EXISTING kinds do not change shape, because
miLLM consumes them today and a schema change is a silent import failure at the
far end rather than a loud one.

Two mechanisms have already broken this in this codebase, and both are why the
rules below are rules rather than intentions:

  * A pydantic `alias` RENAMES ON OUTPUT as well as input. It once republished a
    schema without its wire field and invalidated every exported document. So no
    field here carries an alias, and a test asserts that of the existing kinds
    too.
  * miLLM holds a HAND-WRITTEN mirror of the contract. Re-vendoring without
    updating it silently drops the new field on import and re-export.

VERSION LIVES IN THE KIND IDENTIFIER — `mistudio.jlens-artifact/v1`, not a
`version` field beside it. A consumer that does not understand v2 then REJECTS
it, instead of reading it as v1 with some fields missing.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# Kind identifiers. Version in the string, deliberately — see the module
# docstring. Adding a field to any of these is prohibited; the next version is a
# new identifier.
KIND_JLENS_ARTIFACT = "mistudio.jlens-artifact/v1"
KIND_WORKSPACE_ANNOTATION = "mistudio.workspace-annotation/v1"
KIND_READOUT_RECORD = "mistudio.jlens-readout/v1"
KIND_WATCHLIST = "mistudio.jspace-watchlist/v1"

#: Every kind this feature ADDS. Kinds that already shipped are deliberately
#: absent: this list is what may change, and they may not.
JSPACE_KINDS = (
    KIND_JLENS_ARTIFACT,
    KIND_WORKSPACE_ANNOTATION,
    KIND_READOUT_RECORD,
    KIND_WATCHLIST,
)


class RecipeProvenance(BaseModel):
    """How an artifact was built (BR-007), carried by every J-space kind.

    A document without its recipe cannot be rebuilt or compared, which makes it
    an assertion rather than a result.
    """

    model_config = ConfigDict(from_attributes=True)

    model_id: str
    target_layer: str = "penultimate"
    attention_gradients: str = "full"
    aggregation: str = "mean"
    corpus: str
    n_prompts: int = Field(..., ge=0)
    seq_len: int = Field(..., gt=0)
    dtype: str = "fp16"
    library_versions: Dict[str, str] = Field(default_factory=dict)

    #: Per layer, because a recipe choice can be INAPPLICABLE to a layer rather
    #: than merely unset — frozen-Q/K is undefined wherever a layer does not
    #: attend. An artifact must not be described as "frozen_qk" wholesale when
    #: the treatment reached a subset.
    per_layer_applicability: List[Dict[str, object]] = Field(default_factory=list)


class JLensArtifactDocument(BaseModel):
    """Track A: the artifact a consumer MOUNTS (BR-022, PADR IDL-46).

    Describes a directory rather than carrying weights. There is no ingestion
    API upstream — J-lens is compute-on-demand from a mounted artifact — so a
    document that embedded the tensors would be describing a transfer nobody
    performs.
    """

    model_config = ConfigDict(from_attributes=True)

    kind: Literal["mistudio.jlens-artifact/v1"] = KIND_JLENS_ARTIFACT
    slug: str
    d_model: int = Field(..., gt=0)
    n_layers: int = Field(..., gt=0)
    n_vocab: int = Field(..., gt=0)
    layers: List[int]
    size_bytes: int = Field(..., gt=0)
    recipe: RecipeProvenance

    #: Which validation classes passed. Absent means NOT RUN, which is not a
    #: pass — the same fail-closed rule the suite itself uses.
    validation: Dict[str, str] = Field(default_factory=dict)


class WorkspaceAnnotationDocument(BaseModel):
    """Track B: per-feature annotation, exported via the EXISTING upload path."""

    model_config = ConfigDict(from_attributes=True)

    kind: Literal["mistudio.workspace-annotation/v1"] = KIND_WORKSPACE_ANNOTATION
    sae_id: str
    recipe: RecipeProvenance

    #: Both BR-012 fields travel, and both may be absent. A consumer that sees
    #: only the geometric field would reconstruct alignment from kurtosis alone
    #: — which labels every motor feature workspace.
    features: List[Dict[str, object]] = Field(default_factory=list)


class ReadoutRecordDocument(BaseModel):
    """A position x layer readout, in the upstream wire format (PADR IDL-45)."""

    model_config = ConfigDict(from_attributes=True)

    kind: Literal["mistudio.jlens-readout/v1"] = KIND_READOUT_RECORD
    model: str
    prompt: str
    meta: Dict[str, object]
    tokens: List[Dict[str, object]]

    #: Rung 0. Carried in the document so a consumer cannot receive a readout
    #: stripped of what it is.
    evidence_rung: int = 0


class TemplateLensSpec(BaseModel):
    """Multi-token concepts (BR-023). FIELDS DAY-ONE, compute optional.

    The compute path may be a fast-follow; the fields are not. Adding a field to
    a shipped kind later is precisely the change BR-021 forbids, so they exist
    now and are ABSENT when the path has not run — which is different from zero
    and different from unsupported.
    """

    model_config = ConfigDict(from_attributes=True)

    phrase: str
    n_contexts: Optional[int] = None
    baseline_set: Optional[str] = None
    whitening: Optional[str] = None
    #: None means NOT COMPUTED. A consumer must not read it as "no direction".
    direction_available: Optional[bool] = None


class WatchlistDocument(BaseModel):
    """A named concept set for per-token evaluation at inference (BR-025)."""

    model_config = ConfigDict(from_attributes=True)

    kind: Literal["mistudio.jspace-watchlist/v1"] = KIND_WATCHLIST
    name: str
    recipe: RecipeProvenance

    #: The scoring definition travels WITH the list. A threshold without the
    #: definition it was measured under is not portable: the consumer would
    #: apply it to a differently-computed score and get a different detector.
    scoring_definition: str
    concepts: List[Dict[str, object]] = Field(default_factory=list)
    templates: List[TemplateLensSpec] = Field(default_factory=list)
