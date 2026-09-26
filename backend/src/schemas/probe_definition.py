"""`mistudio.probe-definition/v1` — the portable probe monitor (033 FR-1, FR-2, BR-015).

WHAT THIS IS FOR. A probe trained here is a few kilobytes of weights plus the statements that
make them interpretable: which layer, which normalisation, which combining rule, what threshold,
and — the part that costs the most to earn — what evidence there is that it detects anything. This
contract carries all of it so miLLM can serve the probe without re-deriving any of it, and can
REFUSE to serve one whose evidence does not support the claim being made.

⚠ THE VERSION LIVES IN THE `kind` STRING, NOT IN A FIELD (IDL-45/46, the J-space convention).
A consumer dispatches on `kind` before it parses anything else, so the major version must be
readable without a successful parse. A `version: 1` field would be unreadable exactly when it
matters — on a document from a future major that this code cannot validate.

⚠ NO ALIASES, ANYWHERE. `alias` renames a field on OUTPUT as well as input, and doing that once
republished this estate's cluster schema without its wire field and invalidated every exported
document (memory: `pydantic-alias-renames-on-serialisation`). If an input spelling ever has to be
accepted, `validation_alias` is the only safe tool, and `extra="forbid"` means a typo is refused
rather than silently dropped.

⚠ ADDITIVE-ONLY ONCE ANYTHING IS PUBLISHED. Until the first HuggingFace upload, fix freely. After
that, a removed or retyped field breaks every consumer holding a published document, and nothing
recalls those. New fields are optional with a default.

⚠ DATASET REFERENCES ARE NEVER LOCAL PATHS. A definition that says
`/data/datasets/foo` is unusable off this box and leaks the filesystem layout; `DatasetRef` refuses
one outright. Same rule as the cluster contract's `source_hint`.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..ml.probe_monitor_model import RULES, STREAMABLE

#: The kind string consumers dispatch on. The `/v1` is the major version.
PROBE_DEFINITION_KIND = "mistudio.probe-definition/v1"

# ── caps (FR-2) ────────────────────────────────────────────────────────────────
# Every one of these is a REFUSAL, not a truncation: a definition that silently dropped
# half its evaluations would still validate and would overstate its evidence.

#: Residual width, or k for an SAE basis. 16,384 covers every model on this estate with room;
#: the point of the cap is that a corrupt document cannot ask a consumer for a gigabyte.
MAX_D = 16_384
MIN_TEST_VECTORS = 8
MAX_TEST_VECTORS = 32
#: Tokens per test vector. Long rows make the document large without making the parity check
#: stronger — 1,024 tokens is already far more than any combining rule needs to disagree.
MAX_VECTOR_TOKENS = 1_024
#: Serialised size. Checked by the builder against `model_dump_json()`, because that is what a
#: consumer downloads; a cap on field counts would not bound the bytes.
MAX_DEFINITION_BYTES = 2 * 1024 * 1024
MAX_EVALUATIONS = 20
MAX_NAME = 200
MAX_TEXT = 2_000
MAX_REASON = 1_000

#: The combining rules, imported from 032 rather than restated. A second list here would drift,
#: and a definition naming a rule the trainer cannot produce is unserveable.
ProbeRuleName = Literal["mean", "max", "last", "softmax", "attention", "rolling_mean_max"]

ProbeScope = Literal["all", "prompt", "response"]
ProbeBasis = Literal["residual", "sae_features"]
HookPoint = Literal["resid_post"]


def _refuse_local_path(value: Optional[str], field: str) -> Optional[str]:
    """Shared with the cluster contract's rule, and deliberately strict.

    A path is refused whether absolute (`/data/...`), home-relative (`~/...`), parent-relative
    (`../...`) or a Windows drive (`C:\\...`). An `hf:owner/repo` hint is fine.
    """
    if value and (
        value.startswith("/")
        or value.startswith("~")
        or value.startswith("..")
        or ":\\" in value
        or value.startswith("file://")
    ):
        raise ValueError(
            f"{field} must not be a filesystem path ({value!r}); a definition names a "
            f"HuggingFace dataset or repository so it is usable off the machine that built it"
        )
    return value


class DatasetRef(BaseModel):
    """Where a dataset came from, portably. `{hf_id, config, split, revision}`."""

    model_config = ConfigDict(extra="forbid")

    hf_id: str = Field(min_length=1, max_length=MAX_NAME)
    config: Optional[str] = Field(None, max_length=MAX_NAME)
    split: Optional[str] = Field(None, max_length=MAX_NAME)
    revision: Optional[str] = Field(None, max_length=MAX_NAME)

    @field_validator("hf_id")
    @classmethod
    def _hf_id_is_not_a_path(cls, value: str) -> str:
        return _refuse_local_path(value, "hf_id")


class ModelIdentity(BaseModel):
    """Which model's activations this probe reads, pinned hard enough to serve against.

    ⚠ `revision` IS A RESOLVED COMMIT SHA, NOT A BRANCH. "main" moves, and a probe read from a
    different revision of the same repo is a probe over a different distribution — the estate has
    already paid for that once at the tokenizer level, where two models sharing a family name
    agreed on 0.00% of token ids.

    ⚠ `chat_template_sha256` IS PART OF THE IDENTITY. The role mask and every rendered row depend
    on the template. A model whose template changed renders different text from the same messages,
    so a consumer must be able to notice.
    """

    model_config = ConfigDict(extra="forbid")

    hf_id: str = Field(min_length=1, max_length=MAX_NAME)
    revision: str = Field(min_length=1, max_length=MAX_NAME)
    d_model: int = Field(gt=0, le=MAX_D)
    n_layers: int = Field(gt=0, le=1_024)
    architecture: str = Field(min_length=1, max_length=MAX_NAME)
    chat_template_sha256: Optional[str] = Field(None, min_length=64, max_length=64)
    mistudio_model_id: Optional[str] = Field(None, max_length=MAX_NAME)

    @field_validator("hf_id")
    @classmethod
    def _hf_id_is_not_a_path(cls, value: str) -> str:
        return _refuse_local_path(value, "model.hf_id")


class ReadPoint(BaseModel):
    """⚠ `hook_point` IS A Literal WITH ONE MEMBER, AND THAT IS THE POINT.

    This estate captured at a post-attention norm for months while every document said
    "residual", and nothing could tell: the numbers were plausible and the SAEs trained fine on a
    signal miLLM never reads. A single-member Literal means a document claiming any other capture
    point is refused rather than interpreted.
    """

    model_config = ConfigDict(extra="forbid")

    layer: int = Field(ge=0, description="0-based decoder block index")
    hook_point: HookPoint = "resid_post"


class ProbeHead(BaseModel):
    """The readout: `w`, `b`, and the standardisation it was FITTED with.

    ⚠ THE NORMALISATION TRAVELS WITH THE WEIGHTS BECAUSE IT IS PART OF THE FUNCTION. A head
    applied to unstandardised activations is not a worse probe, it is a different one — the same
    wrong-basis class as encoding with an SAE's normalisation dropped (MIS-E2E-083).
    """

    model_config = ConfigDict(extra="forbid")

    weights: List[float] = Field(min_length=1)
    bias: float = 0.0
    norm_mean: List[float] = Field(min_length=1)
    norm_std: List[float] = Field(min_length=1)
    #: Required by `attention` and unused by every other rule. Carried inline because a probe
    #: whose weighting cannot be reconstructed is unserveable.
    attention_query: Optional[List[float]] = None

    @field_validator("weights", "norm_mean", "norm_std", "attention_query")
    @classmethod
    def _within_the_width_cap(cls, value: Optional[List[float]]) -> Optional[List[float]]:
        if value is not None and len(value) > MAX_D:
            raise ValueError(f"vector length {len(value)} exceeds the {MAX_D} cap")
        return value

    @model_validator(mode="after")
    def _all_vectors_share_one_width(self) -> "ProbeHead":
        widths = {
            "weights": len(self.weights),
            "norm_mean": len(self.norm_mean),
            "norm_std": len(self.norm_std),
        }
        if self.attention_query is not None:
            widths["attention_query"] = len(self.attention_query)
        if len(set(widths.values())) != 1:
            raise ValueError(
                f"the head's vectors disagree on width: {widths}. A probe's normalisation must "
                f"match its weights, or it standardises with the wrong statistics"
            )
        return self

    @field_validator("norm_std")
    @classmethod
    def _no_zero_or_negative_std(cls, value: List[float]) -> List[float]:
        """⚠ A ZERO std IS NOT A ROUNDING PROBLEM, IT IS AN AMPLIFIER.

        Dividing by a 1e-6 floor turned a 0.001 serving-time drift into 1000.0 and made the probe
        fire on a channel carrying no signal at all. 032 writes 1.0 for a degenerate channel,
        which contributes nothing; a document carrying 0 or a negative std did not come from that
        code and must not be served.
        """
        bad = [index for index, std in enumerate(value) if std <= 0.0]
        if bad:
            raise ValueError(
                f"norm_std has non-positive entries at {bad[:8]}"
                f"{'…' if len(bad) > 8 else ''}; a degenerate channel is recorded as 1.0, never 0"
            )
        return value


class SaeReference(BaseModel):
    """Where the SAE lives and exactly how it normalises — required for `sae_features`.

    ⚠ `normalization` IS NOT DECORATION. `encode_with_training_normalization` exists because
    encoding with a bare `encode()` reads the right weights in the wrong basis, produces
    plausible features and is invisible in the numbers. A consumer that cannot reproduce the
    normalisation cannot reproduce the probe.
    """

    model_config = ConfigDict(extra="forbid")

    hf_repo: str = Field(min_length=1, max_length=MAX_NAME)
    path: str = Field(min_length=1, max_length=MAX_NAME)
    revision: str = Field(min_length=1, max_length=MAX_NAME)
    weights_sha256: str = Field(min_length=64, max_length=64)
    architecture: str = Field(min_length=1, max_length=MAX_NAME)
    d_model: int = Field(gt=0, le=MAX_D)
    n_features: int = Field(gt=0)
    #: The training-normalization mode and its constants, exactly as applied.
    normalization: Dict[str, Any] = Field(default_factory=dict)
    feature_indices: List[int] = Field(min_length=1)
    feature_labels: Optional[List[str]] = None

    @field_validator("hf_repo", "path")
    @classmethod
    def _not_a_local_path(cls, value: str) -> str:
        return _refuse_local_path(value, "sae location")

    @field_validator("feature_indices")
    @classmethod
    def _sorted_distinct_and_non_negative(cls, value: List[int]) -> List[int]:
        if any(index < 0 for index in value):
            raise ValueError("feature_indices must be non-negative")
        if len(set(value)) != len(value):
            raise ValueError("feature_indices contains a duplicate")
        if value != sorted(value):
            raise ValueError(
                "feature_indices must be sorted; the head's weights are ordered to match them, "
                "so a different order silently pairs each weight with another feature"
            )
        if len(value) > MAX_D:
            raise ValueError(f"{len(value)} features exceeds the {MAX_D} cap")
        return value

    @model_validator(mode="after")
    def _indices_are_inside_the_dictionary(self) -> "SaeReference":
        if self.feature_indices and self.feature_indices[-1] >= self.n_features:
            raise ValueError(
                f"feature index {self.feature_indices[-1]} is outside a dictionary of "
                f"{self.n_features} features"
            )
        if self.feature_labels is not None and len(self.feature_labels) != len(self.feature_indices):
            raise ValueError(
                f"{len(self.feature_labels)} labels for {len(self.feature_indices)} features"
            )
        return self


class Aggregation(BaseModel):
    """The per-token combining rule. `streamable` is DERIVED, and checked, not trusted."""

    model_config = ConfigDict(extra="forbid")

    rule: ProbeRuleName
    params: Dict[str, Any] = Field(default_factory=dict)
    streamable: bool

    @model_validator(mode="after")
    def _streamable_agrees_with_the_rule(self) -> "Aggregation":
        expected = self.rule in STREAMABLE
        if self.streamable != expected:
            raise ValueError(
                f"streamable={self.streamable} for rule {self.rule!r}, but only "
                f"{sorted(STREAMABLE)} can be computed incrementally — `last` needs the final "
                f"token, so a consumer told it may stream would report a partial score as final"
            )
        return self


class Decision(BaseModel):
    """The operating point. `threshold=None` means FIRE ON NOTHING, which is a real point."""

    model_config = ConfigDict(extra="forbid")

    threshold: Optional[float] = None
    target_fpr: Optional[float] = Field(None, gt=0.0, lt=1.0)
    realised_fpr: Optional[float] = Field(None, ge=0.0, le=1.0)
    threshold_source: Optional[str] = Field(None, max_length=MAX_NAME)
    calibration: Optional[DatasetRef] = None


class EvaluationEntry(BaseModel):
    """One evaluation set's result, as measured."""

    model_config = ConfigDict(extra="forbid")

    dataset: DatasetRef
    distribution: Literal["in_distribution", "out_of_distribution"]
    n_positive: int = Field(ge=0)
    n_negative: int = Field(ge=0)
    auroc: float = Field(ge=0.0, le=1.0)
    auroc_ci: Optional[List[float]] = None
    recall_at_target_fpr: Optional[float] = Field(None, ge=0.0, le=1.0)

    @field_validator("auroc_ci")
    @classmethod
    def _ci_is_an_ordered_pair(cls, value: Optional[List[float]]) -> Optional[List[float]]:
        if value is None:
            return value
        if len(value) != 2:
            raise ValueError("auroc_ci must be [low, high]")
        low, high = value
        if not 0.0 <= low <= high <= 1.0:
            raise ValueError(f"auroc_ci {value} is not an ordered pair inside [0, 1]")
        return value


class Acknowledgement(BaseModel):
    """Who accepted a below-rung-2 export, when, and why (FR-5).

    The reason is required and has a floor, because "ok" records nothing. It exists so a probe
    served on thin evidence carries the name of the person who decided that was acceptable.
    """

    model_config = ConfigDict(extra="forbid")

    by: str = Field(min_length=1, max_length=MAX_NAME)
    at: datetime
    reason: str = Field(min_length=10, max_length=MAX_REASON)


class JudgeEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1, max_length=MAX_NAME)
    prompt_version: str = Field(min_length=1, max_length=MAX_NAME)
    per_set_auroc: Dict[str, float] = Field(default_factory=dict)


class Evidence(BaseModel):
    """What is actually known about this probe — the expensive part of the document.

    ⚠ `rung_language` IS CARRIED, NOT RECOMPUTED BY THE CONSUMER. The wording that describes a
    rung is the product's claim about it, and 033's whole purpose is that miLLM can refuse to
    serve a probe whose claim is weaker than the use. A consumer inventing its own phrasing is a
    second source of truth for a causal statement.
    """

    model_config = ConfigDict(extra="forbid")

    rung: int = Field(ge=0, le=3)
    rung_language: str = Field(min_length=1, max_length=MAX_TEXT)
    acknowledgement: Optional[Acknowledgement] = None
    evaluations: List[EvaluationEntry] = Field(default_factory=list)
    judge: Optional[JudgeEvidence] = None

    @field_validator("evaluations")
    @classmethod
    def _within_the_evaluation_cap(cls, value: List[EvaluationEntry]) -> List[EvaluationEntry]:
        if len(value) > MAX_EVALUATIONS:
            raise ValueError(f"{len(value)} evaluations exceeds the {MAX_EVALUATIONS} cap")
        return value

    @model_validator(mode="after")
    def _a_low_rung_needs_an_acknowledgement(self) -> "Evidence":
        """FR-5 at the CONTRACT level, not only at the endpoint.

        The endpoint gate can be bypassed — by a hand-edited file, by a future caller, by an
        import path nobody has written yet. A document below rung 2 with no acknowledgement is
        refused here so the refusal travels with the format.
        """
        if self.rung < 2 and self.acknowledgement is None:
            raise ValueError(
                f"rung {self.rung} is below 2, so this definition must carry an "
                f"`evidence.acknowledgement` recording who accepted that and why"
            )
        return self


class Provenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    train_dataset: Optional[DatasetRef] = None
    label_mapping: Dict[str, str] = Field(default_factory=dict)
    keyword_filter: Optional[Dict[str, Any]] = None
    split: Optional[str] = Field(None, max_length=MAX_NAME)
    run_id: Optional[str] = Field(None, max_length=MAX_NAME)
    probe_id: Optional[str] = Field(None, max_length=MAX_NAME)
    created_at: Optional[datetime] = None
    exported_at: Optional[datetime] = None
    mistudio_version: Optional[str] = Field(None, max_length=MAX_NAME)


class TestVector(BaseModel):
    """One row scored by the exporting build, for a consumer to reproduce.

    ⚠ `token_scores` HAS ONE ENTRY PER SCORED TOKEN, NOT PER TOKEN. The scope decides which
    tokens are scored; a consumer comparing a full-length score array against its own masked one
    would disagree for a reason that has nothing to do with the weights.
    """

    model_config = ConfigDict(extra="forbid")

    messages: List[Dict[str, Any]] = Field(min_length=1)
    token_ids: List[int] = Field(min_length=1)
    token_scores: List[float] = Field(min_length=1)
    score: float
    verdict: Optional[bool] = None

    @field_validator("token_ids")
    @classmethod
    def _within_the_token_cap(cls, value: List[int]) -> List[int]:
        if len(value) > MAX_VECTOR_TOKENS:
            raise ValueError(f"{len(value)} tokens exceeds the {MAX_VECTOR_TOKENS} cap")
        return value

    @model_validator(mode="after")
    def _scores_are_no_longer_than_the_tokens(self) -> "TestVector":
        if len(self.token_scores) > len(self.token_ids):
            raise ValueError(
                f"{len(self.token_scores)} token scores for {len(self.token_ids)} tokens; "
                f"scores are per SCORED token, so there can never be more of them"
            )
        return self


class TestVectors(BaseModel):
    """The parity set, with the tolerance the producer suggests comparing at.

    ⚠ SCORE `token_ids`. NOT `messages`. THIS IS NOT A PREFERENCE.

    Measured in 033 acceptance 8.1 on a real definition, 16 vectors:

        scored from the recorded `token_ids`   max |Δ| = 0.000e+00   (exact, all 16)
        re-rendered from `messages`            max |Δ| = 1.153e+00   (23x the tolerance)

    `messages` is a **reconstruction**. The training corpus here is plain prose, not conversations,
    so the exporter wraps each row in a single user turn to have something a consumer can send —
    and re-rendering that through the model's chat template adds six tokens of scaffolding the
    scored row never had, with the first difference at index 2. Two of the sixteen rows were
    truncated to the token cap, keeping the TAIL, so their re-render differs from index 0.

    A consumer that took `messages`, scored them and compared against `score` within `tolerance`
    would fail on **every vector** and conclude its implementation was wrong. A parity check exists
    to tell a correct implementation from an incorrect one; one that reports "incorrect" against a
    correct implementation is worse than no check at all, because it is believed once.

    `messages_reproduce_token_ids` is measured at build time — the exporter re-renders its own
    `messages` and compares — so a consumer is told whether it may start from them rather than
    finding out from a failed comparison.
    """

    model_config = ConfigDict(extra="forbid")

    #: An ABSOLUTE score tolerance. Measured on this estate in 033 acceptance 8.1: batch
    #: composition moves a score by at most 5.78e-03 (16 vectors, one batch of 16 against 16
    #: batches of one), and fp16 versus bf16 at resid_post differs by about 1.5% relative. 0.05 is
    #: about 8.7x the measured batching floor — enough margin that a correct implementation never
    #: trips it, small enough against a score range of -12.9..7.5 that a wrong one does.
    tolerance: float = Field(gt=0.0)
    #: Which field a consumer must score. One value today, and it is stated rather than implied
    #: because the alternative silently fails — see the class docstring.
    authoritative_input: Literal["token_ids"] = "token_ids"
    #: True when re-rendering every vector's `messages` reproduces its `token_ids` exactly, so a
    #: consumer may start from the conversation. None when the exporter could not check.
    messages_reproduce_token_ids: Optional[bool] = None
    vectors: List[TestVector] = Field(min_length=MIN_TEST_VECTORS, max_length=MAX_TEST_VECTORS)


class ProbeDefinitionV1(BaseModel):
    """`mistudio.probe-definition/v1`."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["mistudio.probe-definition/v1"] = PROBE_DEFINITION_KIND
    name: str = Field(min_length=1, max_length=MAX_NAME)
    description: Optional[str] = Field(None, max_length=MAX_TEXT)
    #: What "positive" means, taken from the label mapping rather than from a human summary — a
    #: probe whose positive class nobody can state is a probe nobody can act on.
    concept: Optional[str] = Field(None, max_length=MAX_TEXT)

    model: ModelIdentity
    read: ReadPoint
    scope: ProbeScope
    basis: ProbeBasis
    head: ProbeHead
    sae: Optional[SaeReference] = None
    aggregation: Aggregation
    decision: Decision
    evidence: Evidence
    provenance: Provenance = Field(default_factory=Provenance)
    test_vectors: TestVectors

    @model_validator(mode="after")
    def _the_sae_block_is_present_exactly_when_the_basis_needs_it(self) -> "ProbeDefinitionV1":
        if self.basis == "sae_features" and self.sae is None:
            raise ValueError(
                "basis is 'sae_features' but there is no `sae` block; a consumer cannot encode "
                "without the dictionary's location, revision and normalisation"
            )
        if self.basis == "residual" and self.sae is not None:
            raise ValueError(
                "basis is 'residual' but an `sae` block is present; one of the two is wrong and "
                "guessing which would serve a probe in the wrong basis"
            )
        return self

    @model_validator(mode="after")
    def _the_head_width_matches_the_basis(self) -> "ProbeDefinitionV1":
        width = len(self.head.weights)
        if self.basis == "residual":
            if width != self.model.d_model:
                raise ValueError(
                    f"the head is {width}-dimensional but the model's d_model is "
                    f"{self.model.d_model}; a residual probe reads the full width"
                )
        else:
            assert self.sae is not None  # guaranteed by the validator above
            if width != len(self.sae.feature_indices):
                raise ValueError(
                    f"the head is {width}-dimensional but {len(self.sae.feature_indices)} SAE "
                    f"features are selected; each weight pairs with one selected feature"
                )
        return self

    @model_validator(mode="after")
    def _the_attention_query_is_present_exactly_for_attention(self) -> "ProbeDefinitionV1":
        needs = self.aggregation.rule == "attention"
        has = self.head.attention_query is not None
        if needs and not has:
            raise ValueError(
                "the `attention` rule needs `head.attention_query`; without it a consumer would "
                "fall back to the scores as weights, which is `softmax` at tau=1 — a different "
                "detector under the same name"
            )
        if has and not needs:
            raise ValueError(
                f"`head.attention_query` is set but the rule is {self.aggregation.rule!r}, which "
                f"does not use it; the document describes two different readouts"
            )
        return self

    @model_validator(mode="after")
    def _the_layer_is_inside_the_model(self) -> "ProbeDefinitionV1":
        if self.read.layer >= self.model.n_layers:
            raise ValueError(
                f"layer {self.read.layer} is outside a {self.model.n_layers}-layer model"
            )
        return self

    @model_validator(mode="after")
    def _the_sae_width_matches_the_model(self) -> "ProbeDefinitionV1":
        if self.sae is not None and self.sae.d_model != self.model.d_model:
            raise ValueError(
                f"the SAE was trained at d_model={self.sae.d_model} but the model is "
                f"{self.model.d_model}; it cannot encode this model's activations"
            )
        return self


#: Asserted by a test rather than trusted: the Literal above must equal 032's rule set, or a
#: trainable rule would be unexportable (or an unexportable one would validate).
def rule_names_match_the_trainer() -> bool:
    import typing

    return set(typing.get_args(ProbeRuleName)) == set(RULES)
