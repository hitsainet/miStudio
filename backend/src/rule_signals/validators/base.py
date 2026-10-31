"""The validator interface (miStudio 040 FR-27; the producer's FR-016.1 – FR-016.4).

A validator is a registered, self-describing entry: a declaration (:class:`ValidatorDecl`), a
``check`` of one value with its neighbouring text, and a ``scan`` that returns EVERY candidate in a
text, in text order, with its span in Unicode code points and its verdict.

miStudio implements every registered validator itself (D11: each app implements the checks; miDataworks
publishes the conformance vectors every app vendors byte-identically, ``docs/schemas/rule-signal-cases-v1.json``,
run by ``tests/unit/test_rule_signals_conformance.py``). A port of miDataworks' validators, as miLLM 039
ported them (TD39-24); no app imports another's code.

⚠ NAMED BY WHAT THEY COMPUTE. Internal names describe the computation (a weighted sum, a country
length table, an issuer range); a registered validator's ID is the producer's, and is the only place
a format's name appears (as data).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable


@dataclass(frozen=True)
class Verdict:
    #: The value is well formed: length, prefix or range, check digit(s).
    valid: bool
    #: ``ok``, else the first rule that failed validity, else why the tier stays below medium.
    reason: str
    #: ``none``, ``low`` or ``medium``.
    tier: str
    #: ``popular``, ``irregular`` or ``none``.
    layout: str
    #: The normalised value ("" when there is none).
    normalised: str
    #: A brand, a country, a format, or None.
    detail: Optional[str] = None


@dataclass(frozen=True)
class MaskShape:
    """How a masked value of a format may look."""

    lengths: tuple[int, ...]
    layouts: tuple[tuple[int, ...], ...]
    delimiters: str
    keep_leading: int
    keep_trailing: int


@dataclass(frozen=True, eq=False)
class ValidatorDecl:
    id: str
    version: int
    description: str
    params_schema: dict[str, Any]
    reasons: tuple[str, ...]
    outputs: tuple[tuple[str, str], ...]
    algorithms: tuple[str, ...]
    value_bearing: bool = True
    mask_shape: Optional[MaskShape] = field(default=None)

    @property
    def ref(self) -> str:
        return f"{self.id}@{self.version}"


@dataclass(frozen=True)
class Candidate:
    """One candidate of a scan: ``[start, end)`` in code points of the scanned text."""

    start: int
    end: int
    verdict: Verdict


@runtime_checkable
class Validator(Protocol):
    decl: ValidatorDecl

    def check(self, value: str, before: str, after: str, params: Mapping[str, Any]) -> Verdict:
        ...

    def scan(self, text: str, params: Mapping[str, Any]) -> list[Candidate]:
        ...


class ValidatorUnknown(LookupError):
    def __init__(self, validator_id: str, version: int) -> None:
        super().__init__(f"validator {validator_id}@{version} is not registered")
        self.validator_id = validator_id
        self.version = version
