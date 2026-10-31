"""A rule-set document: parse, the producer's checks, the vendored schema, references (Feature 040 FR-27, FR-29).

The document is miDataworks' (`midataworks-rule-set-v1.json`, vendored byte-identically into
``src/schemas/vendored/`` and pinned by ``tests/unit/test_vendored_rule_set_schema.py``). It is validated
with ``jsonschema`` against that file and NEVER mirrored as a model: the producer owns every field, and a
hand-written mirror would silently drop the next one it adds (suite memory
`circuit-contract-mirror-drift`). The file is READ by path, not imported: this package imports nothing
from the app (`tests/unit/test_rule_signals_is_self_contained.py`).

Checks, in order, each a `RuleSetRefused` with the producer's code and a JSON pointer or a name — never
the document's text:

1. size before parsing (`rule_set_max_bytes`, passed in);
2. strict JSON: no NaN/Infinity, no repeated key;
3. every string NFC (`not_nfc`), every integer within ±(2^53 − 1) (`unsafe_integer`), `provenance`
   absent or an object, never null;
4. the vendored JSON Schema (Draft 2020-12) → `rule_set_invalid`;
5. references (the producer's `_references`): unique ids; ``anchor`` / ``span_of`` name an EARLIER
   signal; expressions name known signals within depth 8; outputs name known expressions with the
   matching type; vetoes, renderings and decoys name known names; every case states every output column;
6. caps (`rule_set_max_signals`, `rule_set_max_cases`) → `over_cap` naming the setting.

⚠ D41 / D44: a ``code_rule`` signal is ACCEPTED here (the shared vector `refuse.document.code_rule`
states the producer's refusal at save, which miStudio does not apply; the conformance test lists that
one case as D41-accepted, the way miLLM's does). A POSITIONAL read of one (a ``within_n`` operand, an
``anchor``, a ``span_of``, a veto ``when`` / ``removes``, a ``field`` output) is refused at compile (ENH-006 D65 /
D74; `compose.CompiledRuleSet._refuse_positional_code`).
"""
from __future__ import annotations

import functools
import json
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from jsonschema import Draft202012Validator

FORMAT = "midataworks.rule-set/v1"
SAFE_INTEGER = 2 ** 53 - 1
MAX_EXPRESSION_DEPTH = 8
VENDORED_DIR = Path(__file__).resolve().parents[1] / "schemas" / "vendored"
SCHEMA_PATH = VENDORED_DIR / "midataworks-rule-set-v1.json"
CASES_PATH = VENDORED_DIR / "rule-signal-cases-v1.json"
REDUCTION_CASES_PATH = VENDORED_DIR / "rule-text-reduction-cases.json"


class RuleSetRefused(ValueError):
    """The document is refused. `code` is stable; `details` names where, never the text."""

    def __init__(self, code: str, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = dict(details or {})


@functools.lru_cache(maxsize=1)
def schema() -> Dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


@functools.lru_cache(maxsize=1)
def _validator() -> Draft202012Validator:
    return Draft202012Validator(schema())


def _no_constants(token: str) -> Any:
    raise RuleSetRefused("rule_set_invalid", f"{token} is not JSON", {"token": token})


def _unique_pairs(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise RuleSetRefused("rule_set_invalid", "a key is repeated in one object", {"key": key})
        out[key] = value
    return out


def parse_bytes(raw: bytes, max_bytes: int) -> Any:
    if len(raw) > max_bytes:
        raise RuleSetRefused("over_cap", f"the document is {len(raw)} bytes; rule_set_max_bytes allows {max_bytes}",
                             {"setting": "rule_set_max_bytes", "value": max_bytes, "got": len(raw)})
    try:
        return json.loads(raw.decode("utf-8"), parse_constant=_no_constants, object_pairs_hook=_unique_pairs)
    except RuleSetRefused:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuleSetRefused("rule_set_invalid", "the document is not UTF-8 JSON",
                             {"position": getattr(exc, "pos", None)}) from None


def _escape(key: Any) -> str:
    return str(key).replace("~", "~0").replace("/", "~1")


def _walk_strings(value: Any, pointer: str = "") -> Iterator[Tuple[str, str]]:
    if isinstance(value, str):
        yield pointer, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield pointer + "/" + _escape(key), str(key)
            yield from _walk_strings(item, pointer + "/" + _escape(key))
    elif isinstance(value, list):
        for i, item in enumerate(value):
            yield from _walk_strings(item, f"{pointer}/{i}")


def _walk_integers(value: Any, pointer: str = "") -> Iterator[Tuple[str, int]]:
    if isinstance(value, bool):
        return
    if isinstance(value, int):
        yield pointer, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _walk_integers(item, pointer + "/" + _escape(key))
    elif isinstance(value, list):
        for i, item in enumerate(value):
            yield from _walk_integers(item, f"{pointer}/{i}")


# ── expressions ───────────────────────────────────────────────────────────────────────────────────


def expression_form(expr: Dict[str, Any]) -> str:
    return next(iter(expr))


def expression_depth(expr: Dict[str, Any]) -> int:
    form = expression_form(expr)
    if form == "signal":
        return 1
    if form in ("all", "any"):
        return 1 + max(expression_depth(e) for e in expr[form])
    if form == "not":
        return 1 + expression_depth(expr["not"])
    body = expr["within_n"]
    return 1 + max(expression_depth(body["a"]), expression_depth(body["b"]))


def signals_named(expr: Dict[str, Any]) -> List[str]:
    form = expression_form(expr)
    if form == "signal":
        return [expr["signal"]]
    if form in ("all", "any"):
        return [s for e in expr[form] for s in signals_named(e)]
    if form == "not":
        return signals_named(expr["not"])
    body = expr["within_n"]
    return signals_named(body["a"]) + signals_named(body["b"])


def output_value_key(value: Dict[str, Any]) -> str:
    return next(iter(value))


_VALUE_TYPE = {"fired": "bool", "count": "int", "first": "string", "spans": "string"}


def signal_reference(signal: Dict[str, Any]) -> Optional[str]:
    if signal["kind"] == "proximity":
        return signal["anchor"]
    if signal["kind"] == "validator":
        return (signal.get("candidates") or {}).get("span_of")
    return None


def _unique(items: List[str], what: str) -> None:
    seen: set = set()
    for item in items:
        if item in seen:
            raise RuleSetRefused("rule_set_invalid", f"{what} {item!r} is repeated", {"what": what, "name": item})
        seen.add(item)


def _invalid(message: str, **details: Any) -> RuleSetRefused:
    return RuleSetRefused("rule_set_invalid", message, details)


def _references(doc: Dict[str, Any]) -> None:  # noqa: C901 - the producer's checks, in its order
    signals = doc["signals"]
    _unique([s["id"] for s in signals], "signal id")
    earlier: set = set()
    for signal in signals:
        ref = signal_reference(signal)
        if ref is not None and ref not in earlier:
            raise _invalid(f"signal {signal['id']!r} refers to {ref!r}, which is not an earlier signal",
                           signal=signal["id"], refers_to=ref)
        if signal["kind"] == "lexical_weights":
            _unique([t["term"] for t in signal["terms"]], f"term of {signal['id']!r}")
        earlier.add(signal["id"])
    expressions = doc["expressions"]
    for name, expr in expressions.items():
        if expression_depth(expr) > MAX_EXPRESSION_DEPTH:
            raise _invalid(f"expression {name!r} nests deeper than {MAX_EXPRESSION_DEPTH}", expression=name)
        for s in signals_named(expr):
            if s not in earlier:
                raise _invalid(f"expression {name!r} names an unknown signal {s!r}", expression=name, signal=s)
    columns = [o["column"] for o in doc["outputs"]]
    _unique(columns, "output column")
    for out in doc["outputs"]:
        key = output_value_key(out["value"])
        if key == "field":
            sig = out["value"]["field"][0]
            if sig not in earlier:
                raise _invalid(f"output {out['column']!r} reads an unknown signal {sig!r}", column=out["column"],
                               signal=sig)
        else:
            if out["value"][key] not in expressions:
                raise _invalid(f"output {out['column']!r} names an unknown expression", column=out["column"])
            if _VALUE_TYPE[key] != out["type"]:
                raise _invalid(f"output {out['column']!r}: {key} is a {_VALUE_TYPE[key]}, not {out['type']}",
                               column=out["column"])
    vetoes = doc.get("vetoes") or []
    _unique([v["id"] for v in vetoes], "veto id")
    for veto in vetoes:
        if veto["when"] not in expressions:
            raise _invalid(f"veto {veto['id']!r} names an unknown expression", veto=veto["id"])
        for s in veto["removes"]:
            if s not in earlier:
                raise _invalid(f"veto {veto['id']!r} removes an unknown signal {s!r}", veto=veto["id"], signal=s)
    renderings = doc.get("renderings") or []
    _unique([r["rendering"] for r in renderings], "rendering")
    for rendering in renderings:
        for s in rendering.get("invisible_to") or []:
            if s not in earlier:
                raise _invalid(f"rendering {rendering['rendering']!r}: unknown signal {s!r}",
                               rendering=rendering["rendering"], signal=s)
    decoys = doc.get("decoys") or []
    _unique([d["id"] for d in decoys], "decoy id")
    for decoy in decoys:
        for c in decoy.get("must_not_fire") or []:
            if c not in columns:
                raise _invalid(f"decoy {decoy['id']!r}: unknown output column {c!r}", decoy=decoy["id"], column=c)
    _unique([s["id"] for s in doc.get("substitutions") or []], "substitution id")
    cases = doc["cases"]
    _unique([c["id"] for c in cases], "case id")
    for index, case in enumerate(cases):
        if set(case["expect"]["outputs"]) != set(columns):
            raise _invalid(f"case {case['id']!r} must state every output column", case=case["id"], index=index)
        for name in case["expect"].get("spans") or {}:
            if name not in expressions:
                raise _invalid(f"case {case['id']!r} states spans of an unknown name", case=case["id"], index=index)


def check(document: Any, *, max_signals: Optional[int] = None, max_cases: Optional[int] = None) -> Dict[str, Any]:
    """Checks 3 – 6 on a parsed document; returns it unchanged. Raises `RuleSetRefused`."""
    if isinstance(document, dict):
        for pointer, text in _walk_strings(document):
            if not unicodedata.is_normalized("NFC", text):
                raise RuleSetRefused("not_nfc", "a string is not Unicode NFC; normalise it first", {"pointer": pointer})
        for pointer, number in _walk_integers(document):
            if abs(number) > SAFE_INTEGER:
                raise RuleSetRefused("unsafe_integer", "an integer is outside ±(2^53 − 1); RFC 8785 cannot write it",
                                     {"pointer": pointer})
        if "provenance" in document and document["provenance"] is None:
            raise _invalid("provenance is absent or an object, never null", pointer="/provenance")
    errors = sorted(_validator().iter_errors(document), key=lambda e: list(e.absolute_path))
    if errors:
        first = errors[0]
        pointer = "".join(f"/{_escape(p)}" for p in first.absolute_path)
        raise _invalid("the document does not match the rule-set schema", pointer=pointer or "/", rule=first.validator)
    _references(document)
    for setting, cap, got in (("rule_set_max_signals", max_signals, len(document["signals"])),
                              ("rule_set_max_cases", max_cases, len(document["cases"]))):
        if cap is not None and got > cap:
            raise RuleSetRefused("over_cap", f"{got} items; the runtime setting {setting} allows {cap}",
                                 {"setting": setting, "value": cap, "got": got})
    return document


def provenance(document: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """FR-27: a rule derived from a labelled dataset reports its recorded provenance (source version,
    derivation method) — `None` for an authored set. The evaluator never reads it."""
    prov = document.get("provenance")
    if not isinstance(prov, dict):
        return None
    keys = ("derivation_id", "method", "method_version", "source_version_id", "label_column", "seed", "params_sha256")
    return {"origin": "derived", **{k: prov.get(k) for k in keys}}
