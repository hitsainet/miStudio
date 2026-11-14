"""Judge-prompt answer forms, the user-message template, and the content hash (PADR IDL-57).

A library judge prompt is three things the judge is held to, and this module owns all three
without touching a database, so each can be tested against the bytes a model actually returns:

* **The template.** The judge's user message is the prompt's ``user_template`` with a FIXED set of
  named placeholders (`TEMPLATE_PLACEHOLDERS`) filled from the row. Nothing else is substituted,
  and an unknown ``{name}`` is refused when the prompt is saved — never rendered as literal text
  the judge then reads as an instruction.
* **The answer form (D2).** Each prompt declares how its reply is read: a 1-N rating, a yes/no
  answer, or one label from a set. Each form has ONE fixed parser and ONE mapping to
  positive/negative, so a judge's scores land on the same AUROC/recall arithmetic as a probe's.
* **The content hash.** Everything that changes what is sent or how the reply is read is hashed.

⚠ A PARSER NEVER GUESSES. An unparseable reply is ``None`` and counts toward the run's
parse-failure limit; it never becomes a default value. Out-of-range ratings are refused, not
clamped (a model answering 0 or 99 has not understood the scale). Prose containing a number or a
label is not read as an answer ("there are 3 risks here" is not a 3; "Yes, but…" is not a yes).
These are the stances `parse_rating` took for the single pinned prompt (FTID 032 §14a); they now
hold for every form, because the stakes-rating prompt is parsed by this module too.

⚠ THE STAKES HASH IS UNCHANGED, AND THAT IS WHY `LEGACY_LAYOUT` EXISTS. Runs before the library
recorded ``sha256("stakes-rating/v1\\n" + system prompt)``. A prompt whose template, answer form and
truncation equal that original layout hashes the same way, so the seeded ``stakes-rating/v1`` row's
hash equals every old run's ``prompt_hash`` and those runs stay linked. Any other layout appends
its canonical JSON to the hashed text, so two prompts that differ only in how the reply is parsed
can never share a hash. ``(name, version)`` is unique, so the two hash shapes cannot collide.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Annotated, Any, Dict, List, Literal, Mapping, Optional, Sequence, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

# ── names ─────────────────────────────────────────────────────────────────────────────────────

#: A prompt name and a behaviour key: lower-case words joined by hyphens, 2-64 characters.
SLUG_PATTERN = r"^[a-z0-9][a-z0-9-]{0,62}[a-z0-9]$"
_SLUG = re.compile(SLUG_PATTERN)
#: A JSON key the reply is read from.
JSON_KEY_PATTERN = r"^[A-Za-z_][A-Za-z0-9_]{0,63}$"


def is_slug(value: Optional[str]) -> bool:
    return bool(value) and bool(_SLUG.match(str(value)))


def prompt_key(name: str, version: int) -> str:
    """``name/vN`` — the string every run records as ``prompt_version``."""
    return f"{name}/v{int(version)}"


# ── the template ──────────────────────────────────────────────────────────────────────────────

#: The ONLY placeholders a user template may name, with the words the editor shows for each.
TEMPLATE_PLACEHOLDERS: Dict[str, str] = {
    "interaction": (
        "the whole conversation, one turn per line as 'role: text', shortened from the "
        "front when it is too long"
    ),
    "text": (
        "the text of every turn without role names, separated by blank lines, shortened from "
        "the front when it is too long"
    ),
    "last_user_message": "the last message the user wrote (empty when there is none)",
    "last_assistant_message": "the last message the assistant wrote (empty when there is none)",
}

_PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")

DEFAULT_USER_TEMPLATE = "{interaction}"
DEFAULT_MAX_INTERACTION_CHARS = 8000
MIN_INTERACTION_CHARS = 200
MAX_INTERACTION_CHARS = 100_000


class TemplateInvalid(ValueError):
    """A user template names something this renderer would not fill."""


def template_placeholders(template: str) -> List[str]:
    """Every ``{name}`` in `template`, in order, duplicates kept."""
    return _PLACEHOLDER.findall(template or "")


def validate_user_template(template: str) -> List[str]:
    """The placeholders a template uses — or `TemplateInvalid` naming what is wrong.

    Refused: no placeholder at all (the judge would see no row), and any ``{name}`` outside
    `TEMPLATE_PLACEHOLDERS` (it would reach the judge as literal text). JSON in a template is
    fine: ``{"answer": "yes"}`` does not look like a placeholder.
    """
    if not template or not template.strip():
        raise TemplateInvalid("the user message template is empty")
    used = template_placeholders(template)
    unknown = sorted({name for name in used if name not in TEMPLATE_PLACEHOLDERS})
    if unknown:
        raise TemplateInvalid(
            f"the template names {', '.join('{' + n + '}' for n in unknown)}, which this "
            f"judge cannot fill. Use one of: "
            f"{', '.join('{' + n + '}' for n in TEMPLATE_PLACEHOLDERS)}"
        )
    if not used:
        raise TemplateInvalid(
            "the template contains no placeholder, so every row would send the judge the same "
            f"text. Add one of: {', '.join('{' + n + '}' for n in TEMPLATE_PLACEHOLDERS)}"
        )
    return used


def _tail(text: str, max_chars: int) -> str:
    """The END of `text`, marked with an ellipsis when it was shortened.

    The end of a conversation is where the consequential turn is, so head-truncation would
    remove exactly what the rating depends on.
    """
    if len(text) > max_chars:
        return "…" + text[-max_chars:]
    return text


def render_interaction(messages: Sequence[Mapping[str, Any]], *, max_chars: int = 8000) -> str:
    """The interaction as the judge sees it — role-labelled, and TRUNCATED FROM THE FRONT."""
    lines = [f"{m.get('role', '?')}: {m.get('content', '')}" for m in messages]
    return _tail("\n".join(lines), max_chars)


def _render_text(messages: Sequence[Mapping[str, Any]], *, max_chars: int) -> str:
    return _tail("\n\n".join(str(m.get("content", "")) for m in messages), max_chars)


def _last(messages: Sequence[Mapping[str, Any]], role: str, *, max_chars: int) -> str:
    for message in reversed(list(messages)):
        if message.get("role") == role:
            return _tail(str(message.get("content", "")), max_chars)
    return ""


def placeholder_values(
    messages: Sequence[Mapping[str, Any]], *, max_chars: int
) -> Dict[str, str]:
    """What each placeholder stands for on this row."""
    return {
        "interaction": render_interaction(messages, max_chars=max_chars),
        "text": _render_text(messages, max_chars=max_chars),
        "last_user_message": _last(messages, "user", max_chars=max_chars),
        "last_assistant_message": _last(messages, "assistant", max_chars=max_chars),
    }


def render_user_message(
    template: str, messages: Sequence[Mapping[str, Any]], *, max_chars: int
) -> str:
    """Fill `template` from one row. Only known placeholders are replaced, in one pass, so a
    row's own text containing ``{interaction}`` is never expanded a second time."""
    values = placeholder_values(messages, max_chars=max_chars)
    return _PLACEHOLDER.sub(
        lambda match: values[match.group(1)] if match.group(1) in values else match.group(0),
        template,
    )


# ── answer forms (D2) ─────────────────────────────────────────────────────────────────────────

_FORBID = ConfigDict(extra="forbid")


class RatingForm(BaseModel):
    """A whole number from `min` to `max`; `positive_from` and above counts as positive."""

    model_config = _FORBID

    kind: Literal["rating"] = "rating"
    min: int = Field(ge=0, le=99)
    max: int = Field(ge=1, le=100)
    positive_from: int
    json_key: str = Field(default="rating", pattern=JSON_KEY_PATTERN)
    #: "json": a JSON object with `json_key` (a bare number alone also reads).
    #: "verdict_line": the reply's last line is ``VERDICT: <value>`` (miDataworks' parser).
    reply: Literal["json", "verdict_line"] = "json"

    @model_validator(mode="after")
    def _ordered(self) -> "RatingForm":
        if self.min >= self.max:
            raise ValueError(f"the lowest rating ({self.min}) must be below the highest ({self.max})")
        if not self.min < self.positive_from <= self.max:
            raise ValueError(
                f"'positive from' ({self.positive_from}) must be above the lowest rating "
                f"({self.min}) and no higher than the highest ({self.max}), so both answers exist"
            )
        return self


class BinaryForm(BaseModel):
    """Yes or no. `positive_answer` says which of the two means the behaviour is present."""

    model_config = _FORBID

    kind: Literal["binary"] = "binary"
    yes_token: str = Field(default="yes", min_length=1, max_length=40)
    no_token: str = Field(default="no", min_length=1, max_length=40)
    positive_answer: Literal["yes", "no"] = "yes"
    json_key: str = Field(default="answer", pattern=JSON_KEY_PATTERN)
    reply: Literal["json", "verdict_line"] = "json"
    case_sensitive: bool = False

    @model_validator(mode="after")
    def _distinct(self) -> "BinaryForm":
        fold = (lambda s: s.strip()) if self.case_sensitive else (lambda s: s.strip().casefold())
        if not self.yes_token.strip() or not self.no_token.strip():
            raise ValueError("the yes and no words must not be blank")
        if fold(self.yes_token) == fold(self.no_token):
            raise ValueError("the yes and no words must differ")
        return self


class LabelsForm(BaseModel):
    """One label from a fixed set; the labels in `positive_labels` count as positive."""

    model_config = _FORBID

    kind: Literal["labels"] = "labels"
    labels: List[str] = Field(min_length=2, max_length=20)
    positive_labels: List[str] = Field(min_length=1)
    json_key: str = Field(default="label", pattern=JSON_KEY_PATTERN)
    reply: Literal["json", "verdict_line"] = "json"
    case_sensitive: bool = False

    @model_validator(mode="after")
    def _consistent(self) -> "LabelsForm":
        fold = (lambda s: s.strip()) if self.case_sensitive else (lambda s: s.strip().casefold())
        if any(not label.strip() or len(label) > 80 for label in self.labels):
            raise ValueError("every label must be 1-80 characters and not blank")
        folded = [fold(label) for label in self.labels]
        if len(set(folded)) != len(folded):
            raise ValueError("labels must be distinct" + ("" if self.case_sensitive else " (ignoring case)"))
        missing = [p for p in self.positive_labels if fold(p) not in folded]
        if missing:
            raise ValueError(f"positive labels {missing} are not in the label set")
        if len({fold(p) for p in self.positive_labels}) >= len(folded):
            raise ValueError(
                "every label is positive, so the judge could never say the behaviour is absent"
            )
        return self


AnswerForm = Annotated[Union[RatingForm, BinaryForm, LabelsForm], Field(discriminator="kind")]
_ANSWER_FORM = TypeAdapter(AnswerForm)


def load_answer_form(value: Any) -> Union[RatingForm, BinaryForm, LabelsForm]:
    """A stored or submitted answer form, validated. Raises `pydantic.ValidationError`."""
    if isinstance(value, (RatingForm, BinaryForm, LabelsForm)):
        return value
    return _ANSWER_FORM.validate_python(value)


def normalise_answer_form(value: Any) -> Dict[str, Any]:
    """The form with every default written out — what is stored and what is hashed."""
    return load_answer_form(value).model_dump(mode="json")


def describe_answer_form(value: Any) -> str:
    """The form in one plain sentence, for the library list and the run's report."""
    form = load_answer_form(value)
    where = (
        f'a JSON object with "{form.json_key}"'
        if form.reply == "json"
        else 'a last line "VERDICT: …"'
    )
    if isinstance(form, RatingForm):
        return (
            f"A rating from {form.min} to {form.max}, read from {where}; "
            f"{form.positive_from} or higher counts as positive."
        )
    if isinstance(form, BinaryForm):
        positive = form.yes_token if form.positive_answer == "yes" else form.no_token
        return (
            f'"{form.yes_token}" or "{form.no_token}", read from {where}; '
            f'"{positive}" counts as positive.'
        )
    return (
        f"One of {', '.join(repr(label) for label in form.labels)}, read from {where}; "
        f"positive: {', '.join(repr(label) for label in form.positive_labels)}."
    )


@dataclass(frozen=True)
class ParsedAnswer:
    """One readable reply. `score` is what the AUROC ranks; `positive` is the judge's own call."""

    value: Union[int, str]
    score: float
    positive: bool


_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)
_VERDICT_LINE = re.compile(r"^VERDICT:\s*(?P<value>.+?)\s*$")
_MISSING = object()


def _raw_value(reply: Optional[str], json_key: str, reply_format: str, *, numeric: bool) -> Any:
    """The answer's raw value from a reply, or `_MISSING`. No interpretation happens here."""
    if reply is None or not str(reply).strip():
        return _MISSING
    if reply_format == "verdict_line":
        lines = [line.strip() for line in str(reply).splitlines() if line.strip()]
        match = _VERDICT_LINE.match(lines[-1]) if lines else None
        return match.group("value") if match else _MISSING

    text = _FENCE.sub("", str(reply).strip())
    try:
        payload = json.loads(text)
    except (ValueError, TypeError):
        payload = _MISSING
    if payload is not _MISSING:
        if isinstance(payload, dict):
            return payload.get(json_key, _MISSING)
        if numeric:
            # `json.loads("6")` succeeds and returns an int, so a bare number lands HERE.
            # A bool is excluded: `json.loads("true")` is True, which would read as a 1.
            if isinstance(payload, (int, float)) and not isinstance(payload, bool):
                return payload
            return _MISSING
        if isinstance(payload, (str, bool)):
            return payload
        if isinstance(payload, int):
            # A label set may be numbers ("1", "2", "3"); the label is still matched as text.
            return str(payload)
        return _MISSING

    # Not JSON. A JSON object embedded in chatter still names its answer.
    key = re.escape(json_key)
    if numeric:
        match = re.search(rf'"{key}"\s*:\s*(\d+)', text)
        if match:
            return match.group(1)
        # A bare number ALONE. Deliberately `fullmatch`: picking the first integer out of
        # prose would read "there are 3 risks here" as a rating of 3.
        bare = re.fullmatch(r"\s*(\d{1,3})\s*", text)
        return bare.group(1) if bare else _MISSING
    match = re.search(rf'"{key}"\s*:\s*(?:"([^"]*)"|(true|false))', text)
    if match:
        if match.group(1) is not None:
            return match.group(1)
        return match.group(2) == "true"
    # The whole reply as the answer, so "yes" and "Yes." read but "Yes, because…" does not.
    return text


def _rating(form: RatingForm, raw: Any) -> Optional[ParsedAnswer]:
    if raw is _MISSING or raw is None or isinstance(raw, bool):
        return None
    try:
        value = int(str(raw).strip())
    except (ValueError, TypeError):
        return None
    if not form.min <= value <= form.max:
        # Out of range is a failure, not something to clamp.
        return None
    return ParsedAnswer(value=value, score=float(value), positive=value >= form.positive_from)


def _match_label(candidates: Sequence[str], raw: Any, case_sensitive: bool) -> Optional[str]:
    if isinstance(raw, int) and not isinstance(raw, bool):
        raw = str(raw)
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if text.endswith(".") and len(text) > 1:
        text = text[:-1].rstrip()
    for label in candidates:
        if case_sensitive:
            if text == label.strip():
                return label
        elif text.casefold() == label.strip().casefold():
            return label
    return None


def _binary(form: BinaryForm, raw: Any) -> Optional[ParsedAnswer]:
    if raw is _MISSING or raw is None:
        return None
    if isinstance(raw, bool):
        answer = "yes" if raw else "no"
    else:
        token = _match_label([form.yes_token, form.no_token], raw, form.case_sensitive)
        if token is None:
            return None
        answer = "yes" if token == form.yes_token else "no"
    positive = answer == form.positive_answer
    return ParsedAnswer(
        value=form.yes_token if answer == "yes" else form.no_token,
        score=1.0 if positive else 0.0,
        positive=positive,
    )


def _labels(form: LabelsForm, raw: Any) -> Optional[ParsedAnswer]:
    if raw is _MISSING or raw is None or isinstance(raw, bool):
        return None
    label = _match_label(form.labels, raw, form.case_sensitive)
    if label is None:
        return None
    fold = (lambda s: s.strip()) if form.case_sensitive else (lambda s: s.strip().casefold())
    positive = fold(label) in {fold(p) for p in form.positive_labels}
    return ParsedAnswer(value=label, score=1.0 if positive else 0.0, positive=positive)


def parse_answer(form_value: Any, reply: Optional[str]) -> Optional[ParsedAnswer]:
    """The judge's answer under `form_value`, or None. None IS A PARSE FAILURE AND IS COUNTED."""
    form = load_answer_form(form_value)
    raw = _raw_value(reply, form.json_key, form.reply, numeric=isinstance(form, RatingForm))
    if isinstance(form, RatingForm):
        return _rating(form, raw)
    if isinstance(form, BinaryForm):
        return _binary(form, raw)
    return _labels(form, raw)


def reply_shape_warnings(system_prompt: str, user_template: str, form_value: Any) -> List[str]:
    """Plain-language warnings when the prompt does not ask for the shape the parser reads.

    A prompt and a parser that disagree produce 100% parse failures and a refusal that blames the
    judge. Warnings, not refusals: the words can be in a language this check does not read.
    """
    form = load_answer_form(form_value)
    text = f"{system_prompt}\n{user_template}"
    warnings: List[str] = []
    if form.reply == "json" and form.json_key not in text:
        warnings.append(
            f'The prompt never mentions "{form.json_key}", the JSON key the answer is read from. '
            "Ask for it explicitly, or every reply may fail to parse."
        )
    if form.reply == "verdict_line" and "VERDICT" not in text:
        warnings.append(
            'The prompt never asks for a last line "VERDICT: …", which is how the answer is read.'
        )
    if isinstance(form, BinaryForm):
        for token in (form.yes_token, form.no_token):
            if token.casefold() not in text.casefold():
                warnings.append(f'The prompt never mentions the answer word "{token}".')
    if isinstance(form, LabelsForm):
        absent = [label for label in form.labels if label.casefold() not in text.casefold()]
        if absent:
            warnings.append(
                f"The prompt never mentions the labels {', '.join(repr(a) for a in absent)}."
            )
    if isinstance(form, RatingForm) and (str(form.min) not in text or str(form.max) not in text):
        warnings.append(
            f"The prompt does not state the scale ({form.min} to {form.max}) the rating is read on."
        )
    return warnings


# ── the content hash ──────────────────────────────────────────────────────────────────────────

#: The layout every run before the library used. A prompt with exactly this layout hashes as
#: ``sha256("{key}\n{system}")`` — the formula `probe_monitor_judge.prompt_hash()` has always used.
#: `positive_from` 6 is the stakes prompt's own decision point (ratings 6-10 count as positive, the
#: upper half of its 1-10 scale); it affects only the judge's own recall/false-positive figures,
#: never its AUROC, and was chosen on 2026-10-21 when the library gave the prompt an answer form.
LEGACY_LAYOUT: Dict[str, Any] = {
    "user_template": DEFAULT_USER_TEMPLATE,
    "max_interaction_chars": DEFAULT_MAX_INTERACTION_CHARS,
    "answer_form": {
        "kind": "rating",
        "min": 1,
        "max": 10,
        "positive_from": 6,
        "json_key": "rating",
        "reply": "json",
    },
}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def content_hash(
    *,
    name: str,
    version: int,
    system_prompt: str,
    user_template: str,
    answer_form: Any,
    max_interaction_chars: int,
) -> str:
    """sha256 of everything that changes what the judge is sent or how its reply is read."""
    layout = {
        "user_template": user_template,
        "max_interaction_chars": int(max_interaction_chars),
        "answer_form": normalise_answer_form(answer_form),
    }
    text = f"{prompt_key(name, version)}\n{system_prompt}"
    if _canonical(layout) != _canonical(LEGACY_LAYOUT):
        text += "\n" + _canonical(layout)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
