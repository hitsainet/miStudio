"""Request bodies for the judge-prompt library (PADR IDL-57).

What a request alone can settle is checked here (shapes, lengths, the slug pattern); what needs
the library — a name already taken, a locked version, a template naming an unknown placeholder,
an answer form whose positive mapping is impossible — is refused by `judge_prompt_library` with a
422 or 409 that says what to do, so the REST routes and the MCP tools get the same answer.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..services.judge_prompt_forms import SLUG_PATTERN

_FORBID = ConfigDict(extra="forbid")

#: Who made the change: "operator" from the UI, "agent:<name>" from the MCP server.
ACTOR_PATTERN = r"^(operator|agent:[A-Za-z0-9._-]{1,60})$"


class JudgePromptCreate(BaseModel):
    model_config = _FORBID

    name: str = Field(pattern=SLUG_PATTERN)
    title: str = Field(min_length=1, max_length=255)
    description: Optional[str] = Field(default=None, max_length=4000)
    behaviour: str = Field(pattern=SLUG_PATTERN)
    behaviour_label: Optional[str] = Field(default=None, max_length=255)
    system_prompt: str = Field(min_length=1, max_length=40000)
    user_template: str = Field(default="{interaction}", min_length=1, max_length=20000)
    answer_form: Dict[str, Any]
    max_interaction_chars: int = Field(default=8000, ge=200, le=100000)
    actor: str = Field(default="operator", pattern=ACTOR_PATTERN)


class JudgePromptChanges(BaseModel):
    """Fields to change. Omitted (null) fields keep their value."""

    model_config = _FORBID

    title: Optional[str] = Field(default=None, max_length=255)
    description: Optional[str] = Field(default=None, max_length=4000)
    behaviour: Optional[str] = Field(default=None, pattern=SLUG_PATTERN)
    behaviour_label: Optional[str] = Field(default=None, max_length=255)
    system_prompt: Optional[str] = Field(default=None, min_length=1, max_length=40000)
    user_template: Optional[str] = Field(default=None, min_length=1, max_length=20000)
    answer_form: Optional[Dict[str, Any]] = None
    max_interaction_chars: Optional[int] = Field(default=None, ge=200, le=100000)
    actor: str = Field(default="operator", pattern=ACTOR_PATTERN)


class JudgePromptClone(JudgePromptChanges):
    """A NEW prompt (version 1) from an existing one; `name` is required."""

    name: str = Field(pattern=SLUG_PATTERN)


class JudgePromptDraft(BaseModel):
    """An unsaved prompt, for preview and try, so the editor can be checked before saving."""

    model_config = _FORBID

    name: str = Field(default="draft", pattern=SLUG_PATTERN)
    behaviour: str = Field(default="draft", pattern=SLUG_PATTERN)
    system_prompt: str = Field(min_length=1, max_length=40000)
    user_template: str = Field(default="{interaction}", min_length=1, max_length=20000)
    answer_form: Dict[str, Any]
    max_interaction_chars: int = Field(default=8000, ge=200, le=100000)


class _PromptOrDraft(BaseModel):
    model_config = _FORBID

    prompt_id: Optional[str] = None
    draft: Optional[JudgePromptDraft] = None
    #: The probe dataset view whose rows are rendered.
    dataset_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def _exactly_one(self) -> "_PromptOrDraft":
        if (self.prompt_id is None) == (self.draft is None):
            raise ValueError("give exactly one of prompt_id (a saved version) or draft")
        return self


class JudgePromptPreview(_PromptOrDraft):
    """Render the judge's messages for a few rows. No model is called."""

    rows: int = Field(default=4, ge=1, le=20)


class JudgePromptTry(_PromptOrDraft):
    """Send a few rows through the judge and show what came back. Never loads a model."""

    endpoint: str = Field(min_length=1)
    model: str = Field(min_length=1)
    rows: int = Field(default=4, ge=1, le=10)


class JudgePromptImport(BaseModel):
    """A `mistudio.judge-prompt/v1` document, as `GET …/export` produced it."""

    model_config = _FORBID

    document: Dict[str, Any]
    actor: str = Field(default="operator", pattern=ACTOR_PATTERN)


class RubricImport(BaseModel):
    """A miDataworks rubric as a library prompt.

    Exactly one source: `document` (a `midataworks.rubric/v1` export — works with miDataworks
    absent) or `rubric_id` (fetched from the connected miDataworks).

    ⚠ `midataworks.rubric/v1` HAS NO PUBLISHED SCHEMA FILE (miDataworks publishes only
    `midataworks-dataset-version-v1.json`). Nothing is vendored: the document is read defensively
    against the shape of miDataworks' own `schemas/labeling.RubricExport`/`RubricBody`, and every
    field this mapping needs is checked by `judge_prompt_library.rubric_to_prompt`.
    """

    model_config = _FORBID

    document: Optional[Dict[str, Any]] = None
    rubric_id: Optional[str] = Field(default=None, min_length=1, max_length=64)
    name: str = Field(pattern=SLUG_PATTERN)
    title: str = Field(min_length=1, max_length=255)
    behaviour: str = Field(pattern=SLUG_PATTERN)
    behaviour_label: Optional[str] = Field(default=None, max_length=255)
    #: rubric field → library placeholder (`interaction`, `text`, `last_user_message`, …).
    field_map: Dict[str, str] = Field(default_factory=dict)
    #: rubric field → fixed text (e.g. the rubric's `question`).
    literals: Dict[str, str] = Field(default_factory=dict)
    #: For a label-set rubric: which verdicts mean the behaviour is present.
    positive_verdicts: Optional[List[str]] = None
    #: For a rating rubric: the lowest rating that means the behaviour is present.
    positive_from: Optional[int] = None
    actor: str = Field(default="operator", pattern=ACTOR_PATTERN)

    @model_validator(mode="after")
    def _one_source(self) -> "RubricImport":
        if (self.document is None) == (self.rubric_id is None):
            raise ValueError("give exactly one of document (an exported rubric) or rubric_id")
        return self
