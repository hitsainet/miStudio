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
import math

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator

from ..ml.probe_monitor_model import DEFAULT_WINDOW, RULES
from .probe_definition import HEX64

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

#: How many rows of a calibration view to score. A calibration corpus is a corpus — the
#: OpenHermes view on this estate holds 1,001,551 conversations — and the threshold needs a
#: representative sample of the serving distribution, not all of it.
#:
#: ⚠ The FLOOR is not cosmetic: `calibrate` allows `int(target_fpr * n)` false positives, so at a
#: 1% target fewer than 100 negatives place no threshold at all and the probe ranks without
#: deciding. `_cap_can_afford_the_target` refuses that combination rather than producing a probe
#: that silently never fires.
#: Feature 040 S2 (040-T34): 2,000 → 5,000, so a 1% target allows 50 false positives rather than 20;
#: the effective value is stamped into `run.config`, and a run whose config predates the key re-cuts on the
#: 2,000 rows it was calibrated on (`probe_monitor_run.LEGACY_CALIBRATION_MAX_ROWS`).
DEFAULT_CALIBRATION_MAX_ROWS: int = 5000
#: k for the k-sparse SAE variant. A LIST because the point of the variant is to
#: ask what sparsity costs, and one k answers nothing.
DEFAULT_SAE_K: List[int] = [128]
#: Above this share of unparseable judge replies the judge run REFUSES rather than
#: reporting the remainder — a silently smaller sample is a different measurement.
DEFAULT_JUDGE_PARSE_FAILURE_LIMIT: float = 0.05

#: Which tokens a probe is allowed to score. `all` includes the prompt; `assistant`
#: scores only what the model generated. They are different detectors, not a
#: preference — an "assistant" probe cannot fire on a user's text at all.
ProbeScope = Literal["all", "assistant", "user", "last_assistant", "content"]
LabelTarget = Literal["positive", "negative", "excluded"]
#: Feature 040 S2 (FR-13): `blend_fit` — labelled data a blend of probes (041) is fitted on. Never a run's
#: training view (the run refuses any role but `train`); it may be an evaluation set.
DatasetRole = Literal["train", "eval", "calibration", "blend_fit"]
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

    #: Feature 034 (FR-26). An optional miDataworks `midataworks.dataset-version` manifest for
    #: the dataset being registered. When present it is validated against the vendored schema,
    #: `split` becomes required, and the manifest's row count for that split must equal the
    #: downloaded split's. ⚠ THE FIELD NAME IS A CROSS-REPO CONTRACT (XR-1): miDataworks detects
    #: support by reading this property from `/api/openapi.json`, so a rename silently turns its
    #: manifest send off. Pinned by `test_probe_dataset_manifest.py`.
    dataset_version_manifest: Optional[Dict[str, Any]] = None

    #: Feature 037 (FR-15; D7, D37): the behaviour this view's POSITIVE class is, and what positive
    #: means in the sender's words. ⚠ BOTH NAMES ARE A CROSS-REPO CONTRACT: miDataworks detects support
    #: by reading them from `/api/openapi.json` and sends them only when they are there, so a rename
    #: silently turns its send off. Pinned by
    #: `test_dataworks_provenance_names.py::test_dataworks_behaviour_names`. The pattern is Part A's
    #: slug rule (`BEHAVIOUR_PATTERN` below, pinned to the library's).
    behaviour: Optional[str] = Field(
        default=None, pattern=r"^[a-z0-9][a-z0-9-]{0,62}[a-z0-9]$", max_length=64
    )
    positive_meaning: Optional[str] = Field(default=None, max_length=1000)

    #: Feature 040 S2 (FR-12, FR-13). ⚠ CROSS-REPO NAMES: miDataworks 017 sends them only when they are in
    #: `/api/openapi.json` (pinned by `test_dataworks_provenance_names.py`).
    #: Grouping columns of the split, read and snapshotted at registration.
    group_columns: Optional[List[str]] = Field(default=None, max_length=256)
    #: A sub-question key of the behaviour version this view cites (`behaviour_version`, else current).
    sub_question: Optional[str] = Field(default=None, pattern=r"^[a-z0-9][a-z0-9-]{0,62}[a-z0-9]$", max_length=64)
    #: The behaviour version `sub_question` is checked against; default the current one.
    behaviour_version: Optional[int] = Field(default=None, ge=1)

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
    #: Feature 034: what a manifest verified at registration; null without one.
    dataset_version: Optional[Dict[str, Any]] = None
    created_at: datetime
    #: Archive (2026-10-21): visibility only. All three null means "not archived".
    archived_at: Optional[datetime] = None
    archived_by: Optional[str] = None
    archive_reason: Optional[str] = None
    #: Feature 037 (FR-15, D48): the registration's behaviour, its meaning, and the confirmation.
    behaviour: Optional[str] = None
    positive_meaning: Optional[str] = None
    positive_labels_confirmed: Optional[Dict[str, Any]] = None
    #: Feature 040 S2.
    group_columns: Optional[List[str]] = None
    column_snapshots: Optional[Dict[str, Any]] = None
    sub_question: Optional[str] = None
    behaviour_version: Optional[int] = None


class ProbeDatasetArchiveRequest(BaseModel):
    """Optional body of `POST /datasets/{id}/archive`. A reason is kept on the view."""

    model_config = ConfigDict(extra="forbid")

    reason: Optional[str] = Field(default=None, max_length=2000)


class ProbeDatasetPurgePlanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ids: List[str] = Field(min_length=1, max_length=200)


class ProbeDatasetPurgeRequest(BaseModel):
    """`POST /datasets/purge`: exactly the views to delete, and why (kept in the tombstone)."""

    model_config = ConfigDict(extra="forbid")

    ids: List[str] = Field(min_length=1, max_length=200)
    reason: Optional[str] = Field(default=None, max_length=2000)


class ProbeRunConfig(BaseModel):
    """What a run was asked to do. Echoed verbatim onto the row (FR-15)."""

    model_config = ConfigDict(extra="forbid")

    #: Explicit layers, or a stride over the model's depth. Exactly one.
    layers: Optional[List[int]] = None
    stride: Optional[int] = Field(default=DEFAULT_STRIDE, ge=1)

    #: ⚠ `rolling_mean_max` WAS ADDED TO THIS DEFAULT ON 2026-10-01 AND REMOVED THE SAME DAY,
    #: because the argument for it was wrong and a run measured it. The claim was that a
    #: windowed max "has no 1/n term, which is precisely why the mean probe's score decayed
    #: and this one should not". Removing the division does not produce length invariance —
    #: **a maximum over N windows grows with N**.
    #:
    #: Measured on `pmr_54563518010f`, 24 probes x 5 out-of-distribution sets, as score
    #: movement across length quartiles in units of positive/negative separation (0 would be
    #: length-invariant):
    #:
    #:     rule                 L11     L16     L21
    #:     mean                 0.231   0.262   0.464   <- least sensitive at every layer
    #:     rolling_mean_max 64  0.661   0.454   0.685
    #:     max                  0.518   0.794   1.836
    #:     softmax              0.487   0.746   1.862
    #:
    #: Every rolling variant drifts UPWARD by half to a full separation unit. On
    #: `mental_health_balanced`, w=64 moves from recall 0.071 on the shortest length quartile
    #: to 0.800 on the longest — blind to short documents, firing on nearly every long one.
    #:
    #: The real fix is not a different rule: it is normalising the score by length. See
    #: `0xcc/reviews/probe_quality_recipe_2026-10-01.md` §4.2. `rolling_mean_max` remains
    #: TRAINABLE by naming it explicitly; it is simply not a default anyone should inherit.
    rules: List[str] = Field(default_factory=lambda: ["mean", "max", "attention"])
    scope: ProbeScope = "all"
    max_length: int = Field(default=DEFAULT_MAX_LENGTH, ge=16, le=131072)
    val_fraction: float = Field(default=DEFAULT_VAL_FRACTION, gt=0.0, lt=0.5)
    seed: int = DEFAULT_SEED
    top_n_layers: int = Field(default=DEFAULT_TOP_N_LAYERS, ge=1, le=8)

    #: One probe per window, exactly as `sae_k` means one probe per k. Only `rolling_mean_max`
    #: reads it; listing several windows with that rule absent trains nothing extra.
    #:
    #: ⚠ BEFORE THIS FIELD THE WINDOW WAS UNREACHABLE. `DEFAULT_WINDOW = 16` bound through
    #: `train_rule`'s own default because the orchestrator passed no `window=`, and this model
    #: is `extra="forbid"`, so no caller could set it. Every rolling probe this system could
    #: produce had a window nobody chose.
    rolling_windows: List[int] = Field(default_factory=lambda: [DEFAULT_WINDOW])

    sae_variant: bool = False
    sae_k: List[int] = Field(default_factory=lambda: list(DEFAULT_SAE_K))

    #: Feature 040 S2 (FR-21, 040-T28): how the run picks its `selected` probe. Stamped into `run.config`
    #: at submit; a run without the stamp (submitted before S2) keeps the AUROC ranking, so none re-ranks.
    selection_rule: Literal["auroc", "recall_at_decision"] = "recall_at_decision"

    target_fpr: float = Field(default=DEFAULT_TARGET_FPR, gt=0.0, lt=1.0)
    #: Rows sampled from a calibration view, seeded-random. Ignored without one.
    calibration_max_rows: int = Field(
        default=DEFAULT_CALIBRATION_MAX_ROWS, ge=100, le=200000
    )
    batch_size: Optional[int] = Field(default=None, ge=1)
    #: ⚠ NOT A CHOICE, AND NEVER WAS. This defaulted to "float16", was persisted on every run,
    #: and was read by nothing — the loader hardcoded float16 regardless. A run now loads at the
    #: precision the shared rule gives its model row (`ml/native_dtype.py`) and records it in
    #: `environment.model_dtype`. Leave this None; a value that DIFFERS from the resolved one is
    #: refused at submit rather than silently ignored, because a request field that does nothing
    #: is the defect that hid the float16/bfloat16 mismatch for a month.
    dtype: Optional[Literal["float16", "bfloat16", "float32"]] = None

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

    @field_validator("rolling_windows")
    @classmethod
    def _windows_must_be_positive(cls, value: List[int]) -> List[int]:
        """Sorted and de-duplicated, like `sae_k`.

        A window of 0 or less is not a degenerate window, it is a crash inside the rule;
        and two identical windows would train two identical probes and double the report.
        """
        if any(w < 1 for w in value):
            raise ValueError("rolling_windows must all be >= 1")
        if not value:
            raise ValueError("rolling_windows must name at least one window")
        return sorted(set(value))

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
    def _cap_can_afford_the_target(self) -> "ProbeRunConfig":
        """A cap too small for the target FPR places NO threshold, silently.

        `calibrate` allows `int(target_fpr * n)` false positives and returns
        `threshold=None` — "fire on nothing" — when that rounds to zero. That is a legitimate
        operating point when it is what the negatives say, and a misconfiguration when it is
        just arithmetic: at target 0.001 a 2,000-row cap can never place a threshold. Refused
        here, where the caller can still change it.
        """
        if int(self.target_fpr * self.calibration_max_rows) < 1:
            needed = math.ceil(1.0 / self.target_fpr)
            raise ValueError(
                f"calibration_max_rows={self.calibration_max_rows} cannot afford "
                f"target_fpr={self.target_fpr}: it allows "
                f"int({self.target_fpr} x {self.calibration_max_rows}) = 0 false positives, so no "
                f"threshold would be placed and the probe would never fire. Use at least "
                f"{needed} rows, or raise target_fpr."
            )
        return self

    @model_validator(mode="after")
    def _sae_variant_needs_k(self) -> "ProbeRunConfig":
        if self.sae_variant and not self.sae_k:
            raise ValueError("sae_variant needs at least one sae_k")
        return self


#: A behaviour key: lower-case words joined by hyphens (`judge_prompt_forms.SLUG_PATTERN`).
BEHAVIOUR_PATTERN = r"^[a-z0-9][a-z0-9-]{0,62}[a-z0-9]$"


class ProbeRunCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str
    train_dataset_id: str
    eval_dataset_ids: List[str] = Field(default_factory=list)
    calibration_dataset_id: Optional[str] = None
    config: ProbeRunConfig = Field(default_factory=ProbeRunConfig)
    #: "auto", "all", a CUDA index or a GPU UUID — resolved by `resolve_gpu_request`.
    gpu: str = "auto"
    #: What the probes detect (PADR IDL-57): a short key (`<behaviour-key>`) and its words
    #: (`<behaviour label>`). Optional — but without a key no judge run can raise these probes to rung 3.
    behaviour: Optional[str] = Field(default=None, pattern=BEHAVIOUR_PATTERN)
    behaviour_label: Optional[str] = Field(default=None, max_length=255)

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
    behaviour: Optional[str] = None
    behaviour_label: Optional[str] = None
    stage: Optional[str]
    status: str
    progress: Optional[float]
    #: The live reading behind `progress`: stage, units done/total, the item inside the stage, rate
    #: and ETA (`services/live_progress.py`). None until the run reports.
    progress_detail: Optional[Dict[str, Any]] = None
    error_message: Optional[str]
    celery_task_id: Optional[str]
    gpu_request: Optional[str]
    gpu_uuid: Optional[str]
    artifact_dir: Optional[str]
    environment: Dict[str, Any]
    layer_selection: Optional[Any]
    #: Feature 040 S2 (FR-21): both rankings of the run's probes and which rule chose (from
    #: `environment.selection`; None for a run completed before S2).
    selection: Optional[Dict[str, Any]] = None
    sub_question: Optional[str] = None
    behaviour_version: Optional[int] = None
    #: Feature 040 S2 (FR-19): calibration size and sentence of the run's selected probe.
    calibration_size: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def model_dtype_label(self) -> Dict[str, Optional[str]]:
        """The precision this run trained at: `environment.model_dtype` when recorded, else the
        inferred float16 of every run before 2026-10-03 (`services/artifact_dtype.py`)."""
        from ..services.artifact_dtype import describe_dtype

        values = self.__dict__  # tolerant of a `model_construct` partial, as on TrainingResponse
        return describe_dtype(
            (values.get("environment") or {}).get("model_dtype"), "probe_run", values.get("created_at")
        )


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
    #: What this probe detects (PADR IDL-57); rung 3 matches the judge prompt's key against it.
    behaviour: Optional[str] = None
    behaviour_label: Optional[str] = None
    rung: int
    rung_reasons: List[str]
    #: The rung's phrase and its next step, RENDERED BY THE SERVER — the same strings the report
    #: and the exported document carry, from `schemas/evidence_ladder.py`.
    #:
    #: ⚠ THEY ARE ON THE LIST, NOT ONLY THE REPORT, SO A TILE NEVER HAS TO MAP A NUMBER TO A
    #: PHRASE. `RungChip` refuses to hold such a map on purpose: miLLM mirrors these strings
    #: verbatim, so a second copy in the frontend is a second vocabulary that drifts, and the
    #: drifted one is what a person reads. Giving the list the wording is what lets the rule hold
    #: for the probe tiles too.
    rung_language: str = ""
    rung_next_step: str = ""
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
    # ── Feature 039: the identity of the definition the row records (FR-16, FR-17) ──
    #: Filled by `endpoints.probe_monitors.probe_summary` from `probe_decision.identity_fields` —
    #: and by nothing else. `None` = not computed / not recorded, never a guessed value. Note
    #: `decision_digest` is the BUILT definition's (the row's own column is the decision in force);
    #: `identity_current` says whether the two are the same decision.
    detector_digest: Optional[str] = None
    decision_digest: Optional[str] = None
    decision_revision: Optional[int] = None
    identity_current: Optional[bool] = None
    identity_stale_reason: Optional[str] = None
    identity_source: Optional[str] = None
    #: On a digest-filtered list (FR-18): `{by, source, revisions}` — why this probe matched.
    matched: Optional[Dict[str, Any]] = None
    # ── WHICH TOKENS: what it was trained on, and the bar it places over each window ──
    #: ⚠ ON THE LIST, BECAUSE A TILE THAT CANNOT SAY THIS IS MISLEADING (operator, 2026-10-04).
    #: A run's nine probes are three LAYERS x three RULES; every one was trained on the SAME scope
    #: and carries a bar for EACH contract window. Without these fields a tile showed neither, and a
    #: reader took the nine for "one probe per window per layer".
    #:
    #: The internal scope the probe was TRAINED and calibrated under — the run's (`all`, `user`,
    #: `input`, ...). `None` only when the run row is gone.
    scope: Optional[str] = None
    #: `{contract window: threshold}` for each window miStudio cut its OWN bar for
    #: (`window_decisions`). `{}` means no per-window bar was placed (no calibration set): every
    #: window is then judged against `threshold`.
    #:
    #: ⚠ THE DERIVED FIELDS DEFAULT TO `None` = NOT COMPUTED, NEVER TO A VALUE (2026-10-08). They
    #: are filled by `endpoints.probe_monitors.probe_summary` and by nothing else. They used to
    #: default to `{}` and `0`, and the report route skipped that function — so every report said
    #: "no per-window bar" and "0 length bands" for probes that had both, while the list said 4.
    #: A count nobody computed must not read as zero: `None` says the field was not filled, which
    #: a reader (and a test) can tell from "filled, and there are none".
    window_thresholds: Optional[Dict[str, Optional[float]]] = None
    #: How many length bands refine the bar over the probe's OWN scope. 0 = none; None = not
    #: computed (see above).
    length_band_count: Optional[int] = None
    #: THE RENDER FORM IT WAS TRAINED UNDER (2026-10-08), from the probe row: e.g.
    #: `{"generation_prompt": true, "add_special_tokens": false, "bos_handling": {...}}`. `None`
    #: means NOT RECORDED — rendered WITHOUT the generation prompt, every probe before that date.
    render_form: Optional[Dict[str, Any]] = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def render_served(self) -> bool:
        """Whether `render_form` is the form miLLM serves on its face — decided HERE, by the same
        function the gates use (`is_served_render_form`), so a tile never re-derives the rule."""
        from ..services.probe_monitor_render import is_served_render_form

        return is_served_render_form(self.__dict__.get("render_form"))

    @computed_field  # type: ignore[prop-decorator]
    @property
    def render_bos_recorded(self) -> bool:
        """Whether the form records what the start-of-text (BOS) rule did (`bos_handling`, F1).
        False on a served form tokenized before that record existed: the gates then check its
        model's tokenizer before any reuse, and a tile says it was not recorded."""
        from ..services.probe_monitor_render import RENDER_SERVED, render_form_status

        return render_form_status(self.__dict__.get("render_form")) == RENDER_SERVED


class ProbeDecisionResponse(BaseModel):
    """`GET /probes/{id}/decision` (Feature 039 FR-15, E1): the decision IN FORCE, in contract form.

    What `millm_recalibrate_probe` compares an agent's values with before it pushes anything. The
    block is the stored snapshot — the same object a build publishes — never re-assembled here.
    """

    probe_id: str
    #: The contract `decision` block; `None` when the bar cannot be written as one (`stale_reason`).
    decision: Optional[Dict[str, Any]] = None
    decision_digest: Optional[str] = None
    decision_revision: int
    #: The last built detector digest (a re-cut never moves it); `None` = never built.
    detector_digest: Optional[str] = None
    identity_current: Optional[bool] = None
    identity_source: Optional[str] = None
    #: Why `decision` is null (`decision_not_formable`), with the stored detail.
    stale_reason: Optional[str] = None
    detail: Optional[str] = None


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
    #: Feature 041 T22: which question the view's labels answer for THIS probe (`own` | `main` |
    #: `other`; null when the view is gone) and whether the evaluation counts towards the rung — both
    #: from `probe_monitor_rung.rung_eligibility_of`, the rule `recompute_rung` applies.
    label_question: Optional[str] = None
    counts_towards_rung: Optional[bool] = None


class ProbeScoreRow(BaseModel):
    """One evaluated row (035 FR-13, D28). `position` is the row's position in the DOWNLOADED split
    (the probe dataset's source dataset at `revision`) — the stable id a consumer joins on (D14)."""

    position: int
    score: float
    label: int
    n_scored: int


class ProbeEvaluationScores(BaseModel):
    """One page of one evaluation's per-row scores (035 FR-13; D26, D27, D30).

    miDataworks probes for this route by path in miStudio's served OpenAPI (`per_row_scores`):
    a GET under `/api/v1/probe-monitors/probes/` ending in `/scores`. Pinned by
    `test_dataworks_provenance_names.py`.
    """

    evaluation_id: str
    probe_id: str
    #: The PROBE dataset (view) the evaluation ran on.
    dataset_id: str
    #: The downloaded dataset that view reads, and the commit it holds (null = not recorded).
    source_dataset_id: str
    revision: Optional[str]
    status: str
    #: `metrics["auroc"]` as the evaluation reported it; null for a refused evaluation (D30).
    reported_auroc: Optional[float]
    total: int
    offset: int
    next_offset: Optional[int]
    rows: List[ProbeScoreRow]
    #: Feature 041 T22 (additive): as on `ProbeEvaluationResponse`.
    label_question: Optional[str] = None
    counts_towards_rung: Optional[bool] = None


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
    #: What `probe.val_auroc` actually is: the maximum over the epochs AND layers that the same
    #: validation split selected, rather than a held-out estimate. `None` when the probe has no
    #: validation AUROC.
    #:
    #: ⚠ It sits beside the out-of-distribution numbers and invites the wrong reading. Stage 1
    #: recorded 0.9982 in-distribution against 0.8841 OOD, which looks like a probe that
    #: collapses off its training distribution; it is one upper bound beside one measurement.
    validation_caveat: Optional[Dict[str, Any]] = None
    #: How the shipped threshold behaves on each evaluation set, beside that set's own 1% point.
    #:
    #: ⚠ A probe's absolute score does not transfer between distributions — it ranks well within
    #: one while the scale shifts between them. The first shipped probe's five sets spanned 24
    #: points in their own 1% thresholds, and the single shipped threshold was above every score
    #: one of them produces. Reported so a single threshold is never read as universal.
    threshold_transfer: Dict[str, Any] = Field(default_factory=dict)
    #: Feature 040 S2 (FR-19): how many negatives the bar rests on, the floor (and whose it is), whether
    #: it is `few_negatives`, and the sentence "1% of n = k rows".
    calibration_size: Optional[Dict[str, Any]] = None


class JudgeRunCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    endpoint: str
    model: str
    dataset_ids: List[str] = Field(min_length=1)
    probe_id: Optional[str] = None
    #: The library version to judge with (`jp_…`). Omitted → ``stakes-rating/v1``, which is what
    #: every run before the library used, so existing callers keep their meaning.
    prompt_id: Optional[str] = None
    max_rows_per_set: int = Field(default=2000, ge=20, le=50000)
    parse_failure_limit: float = Field(
        default=DEFAULT_JUDGE_PARSE_FAILURE_LIMIT, gt=0.0, lt=1.0
    )


class RecalibrateRequest(BaseModel):
    """Move a probe's operating point to another `target_fpr`, from the negatives on disk.

    ⚠ `preview` DEFAULTS TO TRUE, AND THAT IS DELIBERATE. The preview costs microseconds and
    returns the per-set consequence of the candidate bar on every evaluation set, so there is no
    reason to move a threshold before looking at what it does. A dial whose default is "commit"
    is a dial that gets turned without reading the dial.
    """

    model_config = ConfigDict(extra="forbid")

    #: The false-positive budget to re-cut at. Bounds are the open interval `calibrate` accepts;
    #: whether a given sample can AFFORD the target is a separate refusal that names `1/n`,
    #: because "0.0001 is a valid rate" and "2,000 negatives can express it" are different facts.
    target_fpr: float = Field(gt=0.0, lt=1.0)
    preview: bool = True
    #: A target too fine for the sample puts the bar above every negative — the probe fires on
    #: NOTHING. Refused unless this says otherwise, because going silently dark is the failure
    #: mode a monitor exists to prevent.
    allow_fire_on_nothing: bool = False
    #: Recorded verbatim in `calibration_history`. The operator's question on seeing a changed
    #: number is "who moved it, and why", and the row is the only place that can answer.
    reason: str = Field(default="", max_length=500)
    #: Feature 039 C3: the `decision_digest` the caller READ (from a preview or the probe's
    #: decision). The commit is refused `decision_changed` (409) when the decision in force is not
    #: that one — so an agent cannot commit over a decision it never saw. Honoured on both the
    #: synchronous path and the GPU window re-cut (`recut_windows=true`, X6).
    expect_decision_digest: Optional[str] = Field(default=None, pattern=HEX64)


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


class ProbeBehaviourUpdate(BaseModel):
    """Set or clear what a probe detects. The rung is recomputed at once (PADR IDL-57)."""

    model_config = ConfigDict(extra="forbid")

    behaviour: Optional[str] = Field(default=None, pattern=BEHAVIOUR_PATTERN)
    behaviour_label: Optional[str] = Field(default=None, max_length=255)
