"""Rule-set documents for tests, with RANDOM keys and ids (Feature 040 S3; FTDD §10.1).

Every behaviour key, signal id, expression name and output column is a random slug drawn from a seeded
`random.Random`, so no test can pass because a name happens to match something in `src/`. The documents are
`midataworks.rule-set/v1` and validate against the vendored schema.

`masking_set` is the general shape FR-33 / FR-47 (b) need: a value-shaped pattern (MATCH, through a `fired`
output), a validator on its spans that reads "masked" (the VETO `when`), and a veto that removes the value's
span when the masked reading holds. A masked value then vetoes and an unmasked one matches.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from tests.support.random_behaviour import rng_for, slug


def ident(rng: random.Random, prefix: str = "s") -> str:
    """A schema-valid id (`^[a-z][a-z0-9_]{0,47}$`)."""
    return (prefix + "_" + slug(rng).replace("-", "_"))[:40]


def key(rng: random.Random) -> str:
    return slug(rng, words=2)


def pattern(sid: str, regex: str, *, ignore_case: bool = False, fold: bool = False) -> Dict[str, Any]:
    return {"id": sid, "kind": "pattern", "pattern": regex, "ignore_case": ignore_case,
            "fold_fullwidth_ascii": fold, "reads": "text"}


def document(behaviour_key: str, rng: random.Random, *, signals: List[Dict[str, Any]],
             expressions: Dict[str, Any], outputs: List[Dict[str, Any]], vetoes: Optional[List[Dict[str, Any]]] = None,
             cases: Optional[List[Dict[str, Any]]] = None, **extra: Any) -> Dict[str, Any]:
    doc: Dict[str, Any] = {
        "format": "midataworks.rule-set/v1",
        "name": slug(rng, words=2),
        "behaviour_key": behaviour_key,
        "description": "A rule set written by a test.",
        "signals": signals,
        "expressions": expressions,
        "outputs": outputs,
        "cases": list(cases or []),
    }
    if vetoes is not None:
        doc["vetoes"] = vetoes
    doc.update(extra)
    return doc


def fired(column: str, expression: str) -> Dict[str, Any]:
    return {"column": column, "type": "bool", "value": {"fired": expression}}


def case(case_id: str, text: str, outputs: Dict[str, Any], vetoed: bool) -> Dict[str, Any]:
    return {"id": case_id, "input": {"text": text}, "expect": {"outputs": outputs, "vetoed": vetoed},
            "source": {"kind": "constructed"}, "note": "test case"}


#: A value shape the masking validator understands (format `ssn_us` of `masked@1`): three, two, four,
#: digits or mask characters, hyphenated. Constructed values only (D28).
SHAPE = "[0-9X*]{3}-[0-9X*]{2}-[0-9]{4}"
UNMASKED = "923-45-6789"
MASKED = "XXX-XX-6789"


def masking_set(seed: str, behaviour_key: Optional[str] = None, *, with_cases: bool = True,
                vetoes: bool = True) -> Dict[str, Any]:
    """MATCH: a value-shaped span; VETO: the same span reads masked. Random names throughout."""
    rng = rng_for(seed)
    bkey = behaviour_key or key(rng)
    value, masked, hit, is_masked, column = ident(rng, "v"), ident(rng, "m"), ident(rng, "e"), ident(rng, "w"), ident(rng, "c")
    signals = [pattern(value, SHAPE),
               {"id": masked, "kind": "validator", "validator": {"id": "masked", "version": 1},
                "params": {"format": "ssn_us"}, "candidates": {"span_of": value}}]
    cases = [case(ident(rng, "k"), f"id {UNMASKED} end", {column: True}, False),
             case(ident(rng, "k"), f"id {MASKED} end", {column: False}, True)] if with_cases else []
    if not vetoes:
        cases = [case(ident(rng, "k"), f"id {UNMASKED} end", {column: True}, False)] if with_cases else []
    return document(
        bkey, rng, signals=signals, expressions={hit: {"signal": value}, is_masked: {"signal": masked}},
        outputs=[fired(column, hit)],
        vetoes=[{"id": ident(rng, "x"), "when": is_masked, "removes": [value]}] if vetoes else None,
        cases=cases,
    )


def code_rule_signal(sid: str) -> Dict[str, Any]:
    return {"id": sid, "kind": "code_rule", "code_sha256": "a" * 64, "runner_contract": "rule-runner/v1",
            "function": "check", "params": {}, "outputs": [{"name": "fires", "type": "bool"}]}


#: A context word for `cued_set` (the SSN pack's "cued" reading, ENH-006 D124), constructed text only.
CUE = "taxid"


def cued_set(seed: str, behaviour_key: Optional[str] = None, *, match: Optional[List[str]] = None,
             fired_outputs: str = "both") -> "tuple[Dict[str, Any], Dict[str, str]]":
    """The `dlp-ssn-us` shape (D53 / G19-4), random names: a value-shaped pattern (``valid``), a proximity cue
    anchored on it (``cued``), a masking validator and a veto removing both when the value reads masked.

    ``fired_outputs``: ``both`` (a `fired` column for ``valid`` and for ``cued``, as the pack writes, so the
    pre-D53 inference reads both) or ``valid`` (one column, for ``valid`` only). ``match``: the names, by ROLE
    (``"valid"``, ``"cued"``, ``"masked"``), the document's explicit ``match`` states; None leaves it absent.
    Returns the document with ``doc["_names"]`` popped into a side dict via `cued_names`."""
    rng = rng_for(seed)
    bkey = behaviour_key or key(rng)
    value, cue, masked = ident(rng, "v"), ident(rng, "q"), ident(rng, "m")
    names = {"valid": ident(rng, "e"), "cued": ident(rng, "u"), "masked": ident(rng, "w")}
    signals = [pattern(value, SHAPE),
               {"id": cue, "kind": "proximity", "anchor": value, "phrases": [CUE], "word_boundary": True,
                "max_distance": 10, "unit": "code_points", "side": "either", "ignore_case": False,
                "fold_fullwidth_ascii": False},
               {"id": masked, "kind": "validator", "validator": {"id": "masked", "version": 1},
                "params": {"format": "ssn_us"}, "candidates": {"span_of": value}}]
    outputs = [fired(ident(rng, "c"), names["valid"])]
    if fired_outputs == "both":
        outputs.append(fired(ident(rng, "d"), names["cued"]))
    extra: Dict[str, Any] = {}
    if match is not None:
        extra["match"] = {"version": 1, "any_of": [names[m] for m in match]}
    doc = document(bkey, rng, signals=signals,
                   expressions={names["valid"]: {"signal": value}, names["cued"]: {"signal": cue},
                                names["masked"]: {"signal": masked}},
                   outputs=outputs,
                   vetoes=[{"id": ident(rng, "x"), "when": names["masked"], "removes": [value, cue]}],
                   **extra)
    return doc, names
