"""Request and response models for probe monitors (032 FR-1, FR-15).

THE DEFAULTS LIVE HERE AND NOWHERE ELSE (FTID §9). There are no new environment
variables: a default that can be set per deployment is a default nobody can
reproduce from a run record, and FR-15 requires a run to be reproducible from what
it stored. Every value below is echoed into `ProbeMonitorRun.config`, so a run
carries the numbers it actually used rather than the numbers today's constants say.

VALIDATION BELONGS IN THE SERVICE, NOT ONLY HERE (FTID §5). These models reject what
is decidable from the request alone — a stride below 1, a rule that does not exist,
an FPR outside (0, 1). What needs the database or the model — fewer than 20 rows per
class, a layer beyond the model's depth, an SAE variant with no ready SAE at that
layer — is refused by the service with a 422 that names the offending value, because
those checks need data this schema cannot see.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..ml.probe_monitor_model import RULES

# ── defaults (FTID §9) ────────────────────────────────────────────────────────

#: Sweep every 5th layer. A full sweep costs one pooled forward pass either way,
#: but the layer-selection grid it produces is what a reader has to interpret.
DEFAULT_STRIDE: int = 5
#: Train token-level rules at the single best layer. The pooled sweep is cheap; the
#: token capture is what costs disk (about 5 GB per layer on the reference data).
DEFAULT_TOP_N_LAYERS: int = 1
DEFAULT_VAL_FRACTION: float = 0.15
DEFAULT_SEED: int = 1337
DEFAULT_MAX_LENGTH: int = 4096
DEFAULT_TARGET_FPR: float = 0.01
#: k for the k-sparse SAE variant. A LIST because the point of the variant is to
#: ask what sparsity costs, and one k answers nothing.
DEFAULT_SAE_K: List[int] = [128]
#: Above this share of unparseable judge replies the judge run REFUSES rather than
#: reporting the remainder — a silently smaller sample is a different measurement.
DEFAULT_JUDGE_PARSE_FAILURE_LIMIT: float = 0.05

#: Which tokens a probe is allowed to score. `all` includes the prompt; `assistant`
#: scores only what the model generated. They are different detectors, not a
#: preference — an "assistant" probe cannot fire on a user's text at all.
ProbeScope = Literal["all", "assistant", "user", "last_assistant"]
LabelTarget = Literal["positive", "negative", "excluded"]
DatasetRole = Literal["train", "eval", "calibration"]
Distribution = Literal["in_distribution", "out_of_distribution"]


class KeywordFilterSpec(BaseModel):
    """⚠ A FILTER NARROWS A SET; IT NEVER LABELS ONE (BR-003).

    There is deliberately no field here that could assign a label. A spec that could
    say "rows containing 'urgent' are positive" would be keyword labelling wearing a
    filter's name, and the resulting AUROC would measure the keyword.
    """

    model_config = ConfigDict(extra="forbid")

    terms: List[str] = Field(min_length=1)
    mode: Literal["any", "all"] = "any"
    case_sensitive: bool = False

    @field_validator("terms")
    @classmethod
    def _no_blank_terms(cls, value: List[str]) -> List[str]:
        cleaned = [t for t in value if t.strip()]
        if not cleaned:
            raise ValueError("a keyword filter needs at least one non-blank term")
        return cleaned


class ProbeDatasetCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=255)
    dataset_id: str
    config: Optional[str] = None
    split: Optional[str] = None
    input_column: str = Field(min_length=1)
    label_column: str = Field(min_length=1)

    #: {raw label value (as a string): "positive"|"negative"|"excluded"}.
    #: Every distinct value must be mapped — see the validator.
    label_mapping: Dict[str, LabelTarget]
    keyword_filter: Optional[KeywordFilterSpec] = None
    pair_column: Optional[str] = None
    role: DatasetRole = "eval"
    distribution: Optional[Distribution] = None

    @model_validator(mode="after")
    def _mapping_must_produce_both_classes_unless_calibration(self) -> "ProbeDatasetCreate":
        """A train or eval set needs both classes; a calibration set needs neither.

        Calibration supplies NEGATIVES for the FPR threshold and is not labelled for
        the concept at all (FR-2 / BR-003) — ultrachat is ordinary conversation, not
        "low stakes" ground truth. Demanding positives of it would force a caller to
        invent a label, which is the failure this separation exists to prevent.
        """
        targets = set(self.label_mapping.values())
        if self.role == "calibration":
            if "positive" in targets:
                raise ValueError(
                    "a calibration set must not map any value to 'positive': it "
                    "supplies negatives for the FPR threshold and is not labelled "
                    "for the concept"
                )
            return self
        missing = {"positive", "negative"} - targets
        if missing:
            raise ValueError(
                f"label_mapping produces no {' or '.join(sorted(missing))} rows, so a "
                f"{self.role} set cannot be scored; map at least one value to each"
            )
        return self

    @model_validator(mode="after")
    def _distribution_is_for_eval_sets(self) -> "ProbeDatasetCreate":
        if self.distribution is not None and self.role != "eval":
            raise ValueError(
                f"distribution is meaningless on a {self.role} set — it says whether an "
                f"EVALUATION set is out-of-distribution, which is what rung 2 turns on"
            )
        return self


class ProbeDatasetResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    dataset_id: Any
    config: Optional[str]
    split: Optional[str]
    input_column: str
    label_column: str
    label_mapping: Dict[str, str]
    keyword_filter: Optional[Dict[str, Any]]
    pair_column: Optional[str]
    role: str
    distribution: Optional[str]
    counts: Dict[str, Any]
    created_at: datetime


class ProbeRunConfig(BaseModel):
    """What a run was asked to do. Echoed verbatim onto the row (FR-15)."""

    model_config = ConfigDict(extra="forbid")

    #: Explicit layers, or a stride over the model's depth. Exactly one.
    layers: Optional[List[int]] = None
    stride: Optional[int] = Field(default=DEFAULT_STRIDE, ge=1)

    rules: List[str] = Field(default_factory=lambda: ["mean", "max", "last", "attention"])
    scope: ProbeScope = "all"
    max_length: int = Field(default=DEFAULT_MAX_LENGTH, ge=16, le=131072)
    val_fraction: float = Field(default=DEFAULT_VAL_FRACTION, gt=0.0, lt=0.5)
    seed: int = DEFAULT_SEED
    top_n_layers: int = Field(default=DEFAULT_TOP_N_LAYERS, ge=1, le=8)

    sae_variant: bool = False
    sae_k: List[int] = Field(default_factory=lambda: list(DEFAULT_SAE_K))

    target_fpr: float = Field(default=DEFAULT_TARGET_FPR, gt=0.0, lt=1.0)
    batch_size: Optional[int] = Field(default=None, ge=1)
    dtype: Literal["float16", "bfloat16", "float32"] = "float16"

    @field_validator("rules")
    @classmethod
    def _rules_must_exist(cls, value: List[str]) -> List[str]:
        unknown = sorted(set(value) - set(RULES))
        if unknown:
            raise ValueError(
                f"unknown rules {unknown}; available: {sorted(RULES)}"
            )
        if not value:
            raise ValueError("a run must train at least one rule")
        # De-duplicate while keeping the caller's order: two identical rules would
        # train two identical probes and double the report.
        seen, ordered = set(), []
        for rule in value:
            if rule not in seen:
                seen.add(rule)
                ordered.append(rule)
        return ordered

    @field_validator("sae_k")
    @classmethod
    def _k_must_be_positive(cls, value: List[int]) -> List[int]:
        if any(k < 1 for k in value):
            raise ValueError("every sae_k must be at least 1")
        return sorted(set(value))

    @model_validator(mode="after")
    def _layers_or_stride_not_both(self) -> "ProbeRunConfig":
        """Both set is ambiguous, and silently preferring one is how a run sweeps
        layers the caller did not ask for."""
        if self.layers is not None:
            if any(layer < 0 for layer in self.layers):
                raise ValueError("layer indices must be non-negative")
            if not self.layers:
                raise ValueError("layers must not be empty; omit it to use stride")
            # An explicit list wins and the stride is cleared, so the stored config
            # cannot claim a stride that governed nothing.
            self.stride = None
        elif self.stride is None:
            raise ValueError("give either layers or stride")
        return self

    @model_validator(mode="after")
    def _sae_variant_needs_k(self) -> "ProbeRunConfig":
        if self.sae_variant and not self.sae_k:
            raise ValueError("sae_variant needs at least one sae_k")
        return self


class ProbeRunCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str
    train_dataset_id: str
    eval_dataset_ids: List[str] = Field(default_factory=list)
    calibration_dataset_id: Optional[str] = None
    config: ProbeRunConfig = Field(default_factory=ProbeRunConfig)
    #: "auto", "all", a CUDA index or a GPU UUID — resolved by `resolve_gpu_request`.
    gpu: str = "auto"

    @model_validator(mode="after")
    def _train_set_is_not_also_an_eval_set(self) -> "ProbeRunCreate":
        """Evaluating on the training set reports memorisation as detection, and it
        is the single easiest way to produce a probe that looks excellent."""
        if self.train_dataset_id in self.eval_dataset_ids:
            raise ValueError(
                "the training set cannot also be an evaluation set: the resulting "
                "AUROC would measure memorisation, not detection"
            )
        if self.calibration_dataset_id == self.train_dataset_id:
            raise ValueError(
                "the training set cannot also be the calibration set: a threshold "
                "calibrated on training negatives does not hold on new data"
            )
        if len(set(self.eval_dataset_ids)) != len(self.eval_dataset_ids):
            raise ValueError("eval_dataset_ids contains a duplicate")
        return self


class ProbeRunResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    model_id: str
    train_dataset_id: str
    eval_dataset_ids: List[str]
    calibration_dataset_id: Optional[str]
    config: Dict[str, Any]
    stage: Optional[str]
    status: str
    progress: Optional[float]
    error_message: Optional[str]
    celery_task_id: Optional[str]
    gpu_request: Optional[str]
    gpu_uuid: Optional[str]
    artifact_dir: Optional[str]
    environment: Dict[str, Any]
    layer_selection: Optional[Any]
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime]


class ProbeMonitorSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    run_id: str
    layer: int
    rule: str
    rule_params: Dict[str, Any]
    variant: str
    sae_id: Optional[str]
    sae_feature_indices: Optional[List[int]]
    val_metrics: Dict[str, Any]
    selected: bool
    threshold: Optional[float]
    target_fpr: Optional[float]
    realised_fpr: Optional[float]
    threshold_source: Optional[str]
    streamable: bool
    rung: int
    rung_reasons: List[str]
    # ── 033: the exported definition's state ──────────────────────────────────
    #: Set once a definition has been built; cleared when 032 changes something the document
    #: states, so a UI can tell "never built" from "built and now stale".
    definition_built_at: Optional[datetime] = None
    definition_sha256: Optional[str] = None
    #: What the build asked for and resolved, plus an `invalidated` block when it was cleared.
    definition_build: Optional[Dict[str, Any]] = None
    #: Every HuggingFace publication, appended never replaced.
    published: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: datetime


class ProbeEvaluationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    probe_id: str
    dataset_id: str
    status: str
    n_positive: Optional[int]
    n_negative: Optional[int]
    metrics: Dict[str, Any]
    created_at: datetime


class ProbeReport(BaseModel):
    """Assembled SERVER-SIDE so the frontend has no scoring logic (FTID §5).

    `rung_language` and `rung_next_step` come from `schemas/evidence_ladder.py` and
    are never composed in the client: a detector's wording is the thing most likely
    to drift above its evidence, and miLLM mirrors these strings verbatim.
    """

    probe: ProbeMonitorSummary
    rung_language: str
    rung_next_step: str
    evaluations: List[ProbeEvaluationResponse]
    #: The dense probe this SAE probe pairs with (or the reverse), so a reader can
    #: see what k-sparsity cost without assembling the pair themselves.
    paired_probe_id: Optional[str] = None
    #: 033: the SAE's HuggingFace repo, when this is a k-sparse probe and its dictionary has one.
    #: `None` on a dense probe AND on an SAE probe with no published home — the UI distinguishes
    #: them by `probe.variant`, and an SAE probe with None here cannot be exported at all.
    sae_hf_repo: Optional[str] = None
    judge_runs: List[Dict[str, Any]] = Field(default_factory=list)


class JudgeRunCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    endpoint: str
    model: str
    dataset_ids: List[str] = Field(min_length=1)
    probe_id: Optional[str] = None
    max_rows_per_set: int = Field(default=2000, ge=20, le=50000)
    parse_failure_limit: float = Field(
        default=DEFAULT_JUDGE_PARSE_FAILURE_LIMIT, gt=0.0, lt=1.0
    )


class ScoreRequest(BaseModel):
    """Offline scoring of one input (FR-12).

    Text OR messages, never both: two inputs would need a precedence rule, and a
    silently ignored field is how a caller ends up scoring something other than what
    they sent.
    """

    model_config = ConfigDict(extra="forbid")

    #: 16k characters, matching FTDD §8. A cap, not a truncation: over it the
    #: request is refused, because a silently truncated input produces a score for
    #: text the caller did not submit.
    text: Optional[str] = Field(default=None, max_length=16384)
    messages: Optional[List[Dict[str, str]]] = None

    @model_validator(mode="after")
    def _exactly_one_input(self) -> "ScoreRequest":
        if (self.text is None) == (self.messages is None):
            raise ValueError("give exactly one of text or messages")
        if self.messages is not None:
            if not self.messages:
                raise ValueError("messages must not be empty")
            for i, message in enumerate(self.messages):
                if "role" not in message or "content" not in message:
                    raise ValueError(f"messages[{i}] needs both role and content")
            total = sum(len(m.get("content") or "") for m in self.messages)
            if total > 16384:
                raise ValueError(
                    f"messages carry {total} characters, over the 16384 cap; a "
                    f"truncated input would be scored as if it were the whole thing"
                )
        return self
