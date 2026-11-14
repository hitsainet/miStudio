"""Request bodies of the behaviour catalogue (Feature 037, FR-13).

No body carries an ``actor``: who asked is read from the ``X-MiStudio-Agent`` header the MCP client sends
on every request (`probe_dataset_lifecycle.actor_of`), so an agent cannot record itself as the operator.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr

#: A behaviour key: Part A's slug rule. ⚠ A literal copy of `judge_prompt_forms.SLUG_PATTERN` (a schema
#: module imports no service); pinned equal by `test_behaviour_catalogue.py::test_slug_rule_is_the_librarys`.
#: Used only where a malformed path or body value should be a 422 from the framework; the service checks
#: every key with `judge_prompt_forms.is_slug` itself.
KEY_PATTERN = r"^[a-z0-9][a-z0-9-]{0,62}[a-z0-9]$"

MAX_MEANING = 4000


# ── Feature 040 FR-1 / FR-5: the detector data a version carries ─────────────────────────────────
#
# ⚠ THE VOCABULARY IS FIXED AND NAMES NO BEHAVIOUR (FR-5). A value outside a literal is refused by
# the framework, naming the allowed values. Everything that depends on more than one field or on a
# runtime setting — unique keys and ids, `held_out_test` declared, FPR and recall bounds, target
# inheritance, `named_probe` ⇔ `probe_id`, list caps — is checked by
# `services/behaviour_detector_definition.validate`, which refuses `invalid_field` / `over_cap` with
# the field path (FR-2): a framework 422 would not name the rule.

#: Detector forms a success criterion may judge (FR-5).
DETECTOR_FORMS = ("rules_or_blend", "blend_alone", "best_single", "rules_alone", "named_probe")
#: Named subsets; a `{column, value}` filter on a declared column is the fourth kind.
NAMED_SUBSETS = ("all", "rule_invisible_positives", "rule_visible_positives")
#: A miDataworks rule-set name (016's pattern) and a SHA-256.
RULE_SET_NAME_PATTERN = r"^[a-z0-9][a-z0-9-]{0,63}$"
HEX64_PATTERN = r"^[0-9a-f]{64}$"
MAX_COLUMN = 255


class SubQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str = Field(pattern=KEY_PATTERN, max_length=64)
    question: str = Field(min_length=1, max_length=MAX_MEANING)
    positive_meaning: str = Field(min_length=1, max_length=MAX_MEANING)


class ColumnFilter(BaseModel):
    """A subset of positives by a declared column's value."""

    model_config = ConfigDict(extra="forbid")

    column: str = Field(min_length=1, max_length=MAX_COLUMN)
    value: Union[StrictStr, StrictBool, StrictInt]


class SuccessCriterion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=KEY_PATTERN, max_length=64)
    detector: Literal["rules_or_blend", "blend_alone", "best_single", "rules_alone", "named_probe"]
    subset: Union[Literal["all", "rule_invisible_positives", "rule_visible_positives"], ColumnFilter]
    #: A column in `held_out_columns`; absent → judged in distribution and labelled so (FR-5, 040-D19).
    held_out_test: Optional[str] = Field(default=None, min_length=1, max_length=MAX_COLUMN)
    #: Absent → the version's `target_fpr` (FR-2 inheritance).
    target_fpr: Optional[float] = None
    min_recall: float
    #: Only with `detector == "named_probe"`, and required by it.
    probe_id: Optional[str] = Field(default=None, min_length=1, max_length=36)


class ModelRef(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    #: A miStudio model id, or a Hugging Face repo id not yet downloaded (a warning, not a refusal).
    model_id: str = Field(min_length=1, max_length=500)


class RuleSetRef(BaseModel):
    """A miDataworks rule set by name, version and identity (016's identifiers), fetched once at write."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=RULE_SET_NAME_PATTERN)
    version: int = Field(ge=1)
    digest: str = Field(pattern=HEX64_PATTERN)


class RuleSetInline(BaseModel):
    """A rule set sent in the request: the `midataworks.rule-set/v1` document as JSON (FR-45)."""

    model_config = ConfigDict(extra="forbid")

    document: Dict[str, Any]


class DetectorFields(BaseModel):
    """The six optional FR-1 fields. None has a default that names anything: absent is "not declared"."""

    model_config = ConfigDict(extra="forbid")

    sub_questions: Optional[List[SubQuestion]] = None
    success_criteria: Optional[List[SuccessCriterion]] = None
    held_out_columns: Optional[List[str]] = None
    models: Optional[List[ModelRef]] = None
    calibration_floor: Optional[int] = None
    rule_set: Optional[Union[RuleSetRef, RuleSetInline]] = None


DETECTOR_FIELDS = tuple(DetectorFields.model_fields)


class BehaviourDefinition(DetectorFields):
    """The versioned definition text (D22) and, from 040, its detector data. Every field of a version,
    all at once (full replacement)."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=255)
    description: Optional[str] = Field(default=None, max_length=MAX_MEANING)
    positive_meaning: str = Field(min_length=1, max_length=MAX_MEANING)
    negative_meaning: Optional[str] = Field(default=None, max_length=MAX_MEANING)
    #: The false-positive rate probes of this behaviour are calibrated to (the operating point).
    target_fpr: Optional[float] = Field(default=None, gt=0.0, lt=1.0)


class BehaviourVersionCreate(DetectorFields):
    """A new version (FR-3). Without `base_version` it is a full replacement, as in 037: `name` and
    `positive_meaning` are required. With `base_version`, every field the body OMITS carries over from
    that version, and a field sent as null is removed (FR-4: a removal is a new version)."""

    model_config = ConfigDict(extra="forbid")

    base_version: Optional[int] = Field(default=None, ge=1)
    name: Optional[str] = Field(default=None, min_length=1, max_length=255)
    description: Optional[str] = Field(default=None, max_length=MAX_MEANING)
    positive_meaning: Optional[str] = Field(default=None, min_length=1, max_length=MAX_MEANING)
    negative_meaning: Optional[str] = Field(default=None, max_length=MAX_MEANING)
    target_fpr: Optional[float] = Field(default=None, gt=0.0, lt=1.0)


class BehaviourCreate(BehaviourDefinition):
    #: Not pattern-checked here on purpose: the service refuses a bad key with a plain sentence
    #: (`invalid_key`) that names the rule, which a framework 422 does not.
    key: str = Field(min_length=1, max_length=64)


class BehaviourRoleUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["primary", "candidate", "retired"]
    reason: Optional[str] = Field(default=None, max_length=2000)


class BehaviourViewLink(BaseModel):
    model_config = ConfigDict(extra="forbid")

    #: What the view is used for here; defaults to the view's own role.
    use: Optional[Literal["train", "eval", "calibration", "blend_fit"]] = None
    reason: Optional[str] = Field(default=None, max_length=2000)


class BehaviourArchive(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: Optional[str] = Field(default=None, max_length=2000)


class PositiveLabelsConfirm(BaseModel):
    """D48: the operator confirms, once per view, that its positive labels are this behaviour."""

    model_config = ConfigDict(extra="forbid")

    key: str = Field(min_length=1, max_length=64)
