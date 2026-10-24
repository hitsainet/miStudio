"""`docs/schemas/probe-identity-cases.json` — the shared digest cases, checked by TWO implementations (039 FR-6).

The case file freezes `mistudio.probe-identity/1`: the framing, both projections, the defaults an
absent key is hashed as, ~100 number forms, and complete tiny documents with their expected canonical
bytes and digests. Every implementation — miStudio's module, the standard-library one below, the
Vitest one (`frontend/src/test/probeIdentityCases.test.ts`) and miLLM's (038 FR-38.7) — must agree
with it byte for byte.

⚠ THE CHECKER BELOW IMPORTS NOTHING FROM `src`. It reimplements RFC 8785 from the RFC — ECMAScript
`Number::toString` from the shortest decimal digits, UTF-16 key order, the RFC's string escaping —
and builds the projections from the FILE's `projection` section, not from the module's literals. Two
implementations sharing one author's mistake would agree with each other and with a wrong file; this
one shares no code with the module (`test_the_checker_imports_only_the_standard_library` pins that).
Only `test_module_matches_file` imports the module, inside the test.

Regenerate the file (then READ THE DIFF, the schema-sync pattern):

    cd backend && venv/bin/python -m tests.unit.test_probe_identity_cases --write

The tests read `$PROBE_IDENTITY_CASES` when set, so a control can point them at an edited COPY; the
committed file is never edited by a test.
"""
from __future__ import annotations

import copy
import decimal
import hashlib
import json
import math
import os
import struct
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import pytest

ROOT = Path(__file__).resolve().parents[3]
COMMITTED = ROOT / "docs" / "schemas" / "probe-identity-cases.json"


def cases_path() -> Path:
    return Path(os.environ.get("PROBE_IDENTITY_CASES") or COMMITTED)


# ══ the standard-library implementation (no `src` import) ══════════════════════

SAFE_INTEGER = 2**53 - 1


def es_number(value: float) -> str:
    """ECMAScript `Number::toString(x)` (ECMA-262 §6.1.6.1.20), as RFC 8785 §3.2.2.3 requires.

    Built from the SHORTEST round-trip decimal digits (`repr`, which Python guarantees), then laid
    out by the ES rules on `k` (digit count) and `n` (decimal point position).
    """
    if math.isnan(value) or math.isinf(value):
        raise ValueError(f"{value!r} has no JSON form")
    if value == 0:
        return "0"                                  # -0 as well: ES writes "0"
    if value < 0:
        return "-" + es_number(-value)
    _sign, digit_tuple, exponent = decimal.Decimal(repr(value)).as_tuple()
    digits = "".join(str(d) for d in digit_tuple)
    stripped = digits.rstrip("0")
    exponent += len(digits) - len(stripped)
    s, k = stripped, len(stripped)
    n = exponent + k                                # value = 0.s × 10^n
    if k <= n <= 21:
        return s + "0" * (n - k)
    if 0 < n <= 21:
        return s[:n] + "." + s[n:]
    if -6 < n <= 0:
        return "0." + "0" * (-n) + s
    e = n - 1
    mantissa = s if k == 1 else s[0] + "." + s[1:]
    return f"{mantissa}e{'+' if e > 0 else '-'}{abs(e)}"


_SHORT_ESCAPES = {'"': '\\"', "\\": "\\\\", "\b": "\\b", "\f": "\\f", "\n": "\\n", "\r": "\\r",
                  "\t": "\\t"}


def jcs_string(text: str) -> str:
    """RFC 8785 §3.2.2.2: escape only `"`, `\\` and control characters; everything else verbatim."""
    out = ['"']
    for char in text:
        if char in _SHORT_ESCAPES:
            out.append(_SHORT_ESCAPES[char])
        elif ord(char) < 0x20:
            out.append("\\u%04x" % ord(char))
        else:
            out.append(char)
    out.append('"')
    return "".join(out)


def jcs(value: Any) -> str:
    """RFC 8785 canonical JSON text of a parsed JSON value."""
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int):
        if abs(value) > SAFE_INTEGER:
            raise ValueError(f"{value} is outside JSON's exact integer range")
        return str(value)
    if isinstance(value, float):
        return es_number(value)
    if isinstance(value, str):
        return jcs_string(value)
    if isinstance(value, list):
        return "[" + ",".join(jcs(item) for item in value) + "]"
    if isinstance(value, dict):
        keys = sorted(value, key=lambda key: key.encode("utf-16-be"))
        return "{" + ",".join(jcs_string(key) + ":" + jcs(value[key]) for key in keys) + "}"
    raise TypeError(f"not a JSON value: {type(value).__name__}")


def _join(path: str, key: str) -> str:
    return key if path == "" else f"{path}.{key}"


def project(value: Mapping[str, Any], spec: Mapping[str, Any], defaults: Mapping[str, Any],
            prefix: str, path: str = "") -> Dict[str, Any]:
    """A projection built from the case FILE's `projection` section: every key present, an absent
    key replaced by its published default (or `null`), nested objects projected the same way."""
    include = spec["include"]
    added = spec.get("added_after_v1", {})
    out: Dict[str, Any] = {}

    def nested(key: str, item: Any) -> Any:
        child = _join(path, key)
        if item is None:
            return None
        if child in include:
            return project(item, spec, defaults, prefix, child)
        if child + "[]" in include:
            return [project(entry, spec, defaults, prefix, child + "[]") for entry in item]
        if child + "{}" in include:
            return {name: project(entry, spec, defaults, prefix, child + "{}")
                    for name, entry in item.items()}
        return copy.deepcopy(item)

    for key in include.get(path, []):
        full = _join(prefix, _join(path, key))
        item = value[key] if key in value else copy.deepcopy(defaults.get(full))
        out[key] = nested(key, item)
    for key in added.get(path, []):
        full = _join(prefix, _join(path, key))
        if value.get(key) is None or value.get(key) == defaults.get(full):
            continue
        out[key] = nested(key, value[key])
    return out


def framed(algorithm: str, kind: str, payload: Dict[str, Any]) -> str:
    return jcs({"algorithm": algorithm, "digest": kind, "payload": payload})


def stdlib_identity(document: Mapping[str, Any], cases: Mapping[str, Any]) -> Dict[str, str]:
    """`{detector_canonical, decision_canonical, detector_digest, decision_digest}` of a parsed
    document, from the case file's rules alone."""
    projection = cases["projection"]
    defaults = projection["defaults"]
    algorithm = cases["algorithm"]
    detector = framed(algorithm, "detector",
                      project(document, projection["detector"], defaults, "detector"))
    decision = framed(algorithm, "decision",
                      project(document.get("decision") or {}, projection["decision"], defaults,
                              "decision"))
    return {
        "detector_canonical": detector,
        "decision_canonical": decision,
        "detector_digest": hashlib.sha256(detector.encode("utf-8")).hexdigest(),
        "decision_digest": hashlib.sha256(decision.encode("utf-8")).hexdigest(),
    }


def float_from_bits(bits: str) -> float:
    return struct.unpack(">d", bytes.fromhex(bits))[0]


# ══ the tests ═════════════════════════════════════════════════════════════════

def load_cases() -> Dict[str, Any]:
    return json.loads(cases_path().read_text(encoding="utf-8"))


EXPECTED_KEYS = ("detector_canonical", "decision_canonical", "detector_digest", "decision_digest")


def test_the_file_has_its_sections():
    cases = load_cases()
    assert list(cases) == ["version", "about", "algorithm", "framing", "projection", "numbers", "cases"]
    assert cases["version"] == 1 and cases["algorithm"] == "mistudio.probe-identity/1"
    assert "integrity" in cases["about"] and "not authenticity" in cases["about"]
    assert len(cases["numbers"]) >= 90 and len(cases["cases"]) >= 40


def test_stdlib_implementation_matches_file():
    cases = load_cases()
    wrong_numbers = [
        (entry["bits"], entry["jcs"], es_number(float_from_bits(entry["bits"])))
        for entry in cases["numbers"]
        if es_number(float_from_bits(entry["bits"])) != entry["jcs"]
    ]
    assert wrong_numbers == []
    wrong_cases = []
    for case in cases["cases"]:
        got = stdlib_identity(case["document"], cases)
        for key in EXPECTED_KEYS:
            if got[key] != case[key]:
                wrong_cases.append((case["name"], key))
    assert wrong_cases == []


def test_the_relations_hold():
    """Each case states whether it shares its comparison case's detector and decision; the stdlib
    implementation's digests must agree with the statement (an included field that did not move
    the digest, or an excluded one that did, fails here)."""
    cases = load_cases()
    by_name = {case["name"]: case for case in cases["cases"]}
    assert len(by_name) == len(cases["cases"]), "case names must be unique"
    wrong = []
    for case in cases["cases"]:
        relation = case.get("relation")
        if relation is None:
            continue
        other = stdlib_identity(by_name[relation["to"]]["document"], cases)
        mine = stdlib_identity(case["document"], cases)
        for kind in ("detector", "decision"):
            same = mine[f"{kind}_digest"] == other[f"{kind}_digest"]
            if same != (relation[kind] == "equal"):
                wrong.append((case["name"], kind, relation[kind]))
    assert wrong == []


def test_module_matches_file():
    """miStudio's module against the file: projection literals, defaults, numbers, every case."""
    from src.schemas import probe_identity as pi
    from src.schemas.probe_definition import ProbeDefinitionV1

    cases = load_cases()
    projection = cases["projection"]
    assert projection["detector"]["include"] == {k: list(v) for k, v in pi.DETECTOR_V1.items()}
    assert projection["detector"]["exclude"] == {k: list(v) for k, v in pi.EXCLUDED_V1.items()}
    assert projection["detector"]["added_after_v1"] == {
        k: list(v) for k, v in pi.DETECTOR_ADDED_AFTER_V1.items()
    }
    assert projection["decision"]["include"] == {k: list(v) for k, v in pi.DECISION_V1.items()}
    assert projection["defaults"] == pi.defaults_table()
    assert cases["algorithm"] == pi.IDENTITY_ALGORITHM
    for entry in cases["numbers"]:
        got = pi.canonical_bytes("detector", {"n": float_from_bits(entry["bits"])}).decode()
        assert got.endswith('"payload":{"n":%s}}' % entry["jcs"]), entry
    wrong = []
    for case in cases["cases"]:
        document = case["document"]
        detector = pi.canonical_bytes("detector", pi.detector_payload(document)).decode()
        decision = pi.canonical_bytes("decision", pi.decision_payload(document["decision"])).decode()
        got = dict(zip(EXPECTED_KEYS, (detector, decision, *pi.document_identity(document))))
        # Through the MODEL as well: every case is a valid document, and the build's path agrees.
        identity = pi.identity_for(ProbeDefinitionV1.model_validate(document), 1)
        got_model = (identity.detector_digest, identity.decision_digest)
        for key in EXPECTED_KEYS:
            if got[key] != case[key]:
                wrong.append((case["name"], key))
        if got_model != (case["detector_digest"], case["decision_digest"]):
            wrong.append((case["name"], "model path"))
    assert wrong == []


def test_the_file_is_what_the_generator_writes():
    """The committed file is the generator's output byte for byte, so the module and the file
    cannot be edited apart (and a hand edit of an expected digest is caught here as well)."""
    if os.environ.get("PROBE_IDENTITY_CASES"):
        pytest.skip("pointed at a copy on purpose")
    assert COMMITTED.read_text(encoding="utf-8") == render_cases_file()


def test_the_file_carries_no_behaviour_data():
    """D28: public test values only — no behaviour key but the placeholder, no catalogue name."""
    cases = load_cases()
    for case in cases["cases"]:
        behaviour = case["document"].get("behaviour")
        assert behaviour is None or behaviour["key"].startswith("placeholder"), case["name"]


def test_the_checker_imports_only_the_standard_library():
    import ast

    tree = ast.parse(Path(__file__).read_text())
    top_level = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    names = {alias.name.split(".")[0] for node in top_level if isinstance(node, ast.Import)
             for alias in node.names}
    names |= {node.module.split(".")[0] for node in top_level
              if isinstance(node, ast.ImportFrom) and node.module}
    assert "src" not in names and "tests" not in names
    assert names <= set(sys.stdlib_module_names) | {"__future__", "pytest"}


class TestTheCasesAreVendoredIdentically:
    """⚠ PIN THE TWO REPOS TO EACH OTHER (the `TestTheContractIsVendoredIdentically` pattern).

    ⚠ X2 (FTID §16): this FAILS whenever miLLM's copy exists and differs — not only under
    `MISTUDIO_REQUIRE_CROSS_REPO_CHECKS=1`. Between miStudio's Slice 1 push and miLLM's re-vendor
    (HANDOFF §6 steps 2 → 5) it is red on any machine with miLLM checked out beside this repo; the
    public mirror, which has no sibling, skips. That red is the signal the re-vendor is owed.
    """

    def test_byte_identical_to_millms_vendored_copy(self):
        here = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(here.parent))
        try:
            from tests.support.sibling_repos import millm_repo
        finally:
            sys.path.pop(0)
        theirs = millm_repo() / "docs" / "schemas" / "probe-identity-cases.json"
        if not theirs.exists():
            if os.environ.get("MISTUDIO_REQUIRE_CROSS_REPO_CHECKS") == "1":
                pytest.fail(
                    f"{theirs} is absent and MISTUDIO_REQUIRE_CROSS_REPO_CHECKS=1. miLLM must vendor "
                    f"probe-identity-cases.json byte-identically (039 FR-9, HANDOFF §6 step 5)."
                )
            pytest.skip(
                f"miLLM's vendored copy is not present ({theirs}); set MILLM_REPO to check it"
            )
        assert COMMITTED.read_bytes() == theirs.read_bytes(), (
            "the two repos' copies of probe-identity-cases.json differ; copy ours across "
            "byte-identically (never regenerate there)"
        )


# ══ the generator (`--write`); imports the module lazily ═══════════════════════

ABOUT = (
    "mistudio.probe-identity/1: the detector and decision digests of a mistudio.probe-definition/v1 "
    "document. Each digest is the lowercase hex SHA-256 of the UTF-8 bytes of the RFC 8785 (JSON "
    "Canonicalization Scheme) form of the wrapper {\"algorithm\": <algorithm>, \"digest\": "
    "\"detector\" | \"decision\", \"payload\": <projection>}. The detector projection is computed over "
    "the parsed JSON document and the decision projection over its `decision` block, each by "
    "`projection`: only the listed keys, every listed key present, an absent key written as its "
    "`projection.defaults` value (null when not listed), nested objects projected the same way. Paths "
    "use `.` between keys, `[]` for each element of a list and `{}` for each value of a map. So "
    "absent, null (for keys whose default is null) and the declared default hash equal, and -0.0 and "
    "0.0 hash equal (RFC 8785 writes both as 0). Numbers are written as ECMAScript Number::toString "
    "writes them; `numbers` lists IEEE-754 doubles (big-endian `bits`) with their exact RFC 8785 "
    "text. Each case is a complete valid document with its expected canonical texts and digests; "
    "`relation` states whether it shares the detector and decision digests of another case. To "
    "reproduce: parse `document`, project, canonicalise, hash, compare. A digest gives integrity, "
    "not authenticity: it proves a document agrees with itself, not who produced it. "
    "decision_revision is not hashed. No case carries behaviour data."
)

FRAMING = {
    "wrapper": {"algorithm": "<algorithm>", "digest": "detector | decision", "payload": "<projection>"},
    "canonicalization": "RFC 8785 (JSON Canonicalization Scheme)",
    "hash": "SHA-256 over the UTF-8 canonical bytes, lowercase hex",
}


def _number_values() -> List[float]:
    """~100 doubles chosen to exercise every branch of ES Number::toString (D2)."""
    values = [
        0.0, -0.0, 1.0, -1.0, 2.0, 10.0, 100.0, 1000.0, 0.5, 0.25, 0.1, 0.2, 0.3, 0.1 + 0.2,
        1.5, -2.75, 3.141592653589793, 2.718281828459045, 1.0000000000000002,
        0.9999999999999999, 4.35, 0.002, 4.5, 123.456, 1 / 3, 2 / 3,
        # exponent boundaries
        1e20, 1e21, 1e22, 1.5e21, 9.999999999999999e20, 123456789012345680000.0,
        1e16, 1e15, 12345678901234567.0, 1e-6, 1e-7, 1.5e-7, 9.99e-7, 0.000001, 0.0000012345,
        1e-10, -1e-10, 1e30, 1e-27, 1e100, 1e-100, 1e300, 1e-300,
        # extremes and subnormals
        5e-324, 1e-323, 2.2250738585072014e-308, 2.225073858507201e-308, 1.7976931348623157e308,
        -1.7976931348623157e308,
        # integers written as floats, around 2^53
        9007199254740991.0, 9007199254740992.0, 9007199254740994.0, -9007199254740992.0,
        4294967296.0, 65536.0, 1e3, 1e6, 1e9, 1e12,
        # long fractions and RFC 8785 Appendix B forms
        333333333.3333333, 1424953923781206.2, 1.2345678901234567e-5, 0.000123456789,
        295147905179352830000.0, 1e23, 5e-7, 0.000001000001, 100000000000000000000.0,
        0.01, 0.0097, 0.009, 0.95, 0.9574, 2.38, -0.5, 11.914362907409668,
    ]
    # float(float32) values, as the builder's `head.weight.tolist()` produces them.
    import random

    rng = random.Random(39)
    for _ in range(24):
        f = rng.choice([rng.gauss(0, 0.05), rng.gauss(0, 3), rng.uniform(-1e-3, 1e-3)])
        values.append(struct.unpack("f", struct.pack("f", f))[0])
    return values


def _vectors() -> List[Dict[str, Any]]:
    return [
        {"messages": [{"role": "user", "content": f"row {i}"}], "token_ids": [1, 2, 3],
         "token_scores": [0.1, 0.2, 0.3], "score": 0.2, "verdict": i % 2 == 0}
        for i in range(8)
    ]


def _residual() -> Dict[str, Any]:
    return {
        "kind": "mistudio.probe-definition/v1",
        "name": "example-probe",
        "description": "an example residual probe",
        "concept": "positive = placeholder",
        "model": {"hf_id": "example-org/example-model", "revision": "0" * 40, "d_model": 4,
                  "n_layers": 8, "architecture": "ExampleForCausalLM",
                  "chat_template_sha256": "a" * 64, "mistudio_model_id": "m_example",
                  "load_dtype": "float16", "quantization": "FP16"},
        "read": {"layer": 3, "hook_point": "resid_post"},
        "scope": "all",
        "basis": "residual",
        "head": {"weights": [0.125, -0.5, 1.0, 1e-7], "bias": -0.25, "norm_mean": [0.0, 0.5, -0.5, 0.1],
                 "norm_std": [1.0, 2.0, 0.5, 1.5], "attention_query": None},
        "sae": None,
        "aggregation": {"rule": "mean", "params": {}, "streamable": True},
        "decision": {"threshold": 2.38, "target_fpr": 0.01, "realised_fpr": 0.0097,
                     "threshold_source": "calibration_set",
                     "calibration": {"hf_id": "example-org/example-dataset", "config": "default",
                                     "split": "test", "revision": None},
                     "windows": None, "length_bands": None},
        "evidence": {"rung": 2, "rung_language": "example", "acknowledgement": None,
                     "evaluations": [], "judge": None},
        "provenance": {"run_id": "pmr_example", "probe_id": "pm_example",
                       "exported_at": "2026-10-23T00:00:00Z"},
        "test_vectors": {"tolerance": 0.05, "authoritative_input": "token_ids",
                         "messages_reproduce_token_ids": None, "vectors": _vectors()},
        "render": {"generation_prompt": True, "add_special_tokens": False,
                   "bos_handling": {"template_wrote_bos": True, "tokenizer_added_bos": False,
                                    "bos_count": 1}},
        "behaviour": None,
    }


def _sae() -> Dict[str, Any]:
    document = _residual()
    document["basis"] = "sae_features"
    document["head"] = {"weights": [0.75, -0.125], "bias": 0.5, "norm_mean": [0.0, 0.0],
                        "norm_std": [1.0, 1.0], "attention_query": None}
    document["sae"] = {"hf_repo": "example-org/example-saes", "path": "layer_3/width_64",
                       "revision": "1" * 40, "weights_sha256": "b" * 64, "architecture": "jumprelu",
                       "d_model": 4, "n_features": 64, "normalization": {"mode": "none"},
                       "feature_indices": [5, 17], "feature_labels": ["first", "second"]}
    return document


def _attention() -> Dict[str, Any]:
    document = _residual()
    document["aggregation"] = {"rule": "attention", "params": {}, "streamable": True}
    document["head"]["attention_query"] = [0.5, 0.25, -0.125, 2.0]
    return document


def _softmax() -> Dict[str, Any]:
    document = _residual()
    document["aggregation"] = {"rule": "softmax", "params": {"tau": 1.0}, "streamable": True}
    return document


def _bands(*thresholds: float) -> List[Dict[str, Any]]:
    out = []
    for index, threshold in enumerate(thresholds):
        last = index == len(thresholds) - 1
        out.append({"min_tokens": index * 100, "max_tokens": None if last else index * 100 + 99,
                    "threshold": threshold, "threshold_source": "band", "target_fpr": 0.01,
                    "realised_fpr": 0.0095, "n_negatives": 400})
    return out


def _windowed() -> Dict[str, Any]:
    document = _residual()
    document["decision"]["windows"] = {
        "prompt": {"threshold": 1.5, "target_fpr": 0.01, "realised_fpr": 0.01, "n_negatives": 400,
                   "scope": "user", "length_bands": _bands(1.25, 1.75)},
        "response": {"threshold": 2.5, "target_fpr": 0.01, "realised_fpr": 0.009,
                     "n_negatives": 400, "scope": "assistant", "length_bands": None},
    }
    document["decision"]["length_bands"] = _bands(2.0, 2.5, 3.0)
    return document


def _set(document: Dict[str, Any], path: str, value: Any) -> Dict[str, Any]:
    out = copy.deepcopy(document)
    node = out
    parts = path.split(".")
    for part in parts[:-1]:
        node = node[int(part)] if part.isdigit() else node[part]
    last = parts[-1]
    if value is _ABSENT:
        del node[last]
    elif last.isdigit():
        node[int(last)] = value
    else:
        node[last] = value
    return out


_ABSENT = object()


def _case_specs() -> List[Tuple[str, str, Dict[str, Any], Any]]:
    """(name, about, document, comparison case or None)."""
    r = _residual()
    specs: List[Tuple[str, str, Dict[str, Any], Any]] = [
        ("residual", "a residual probe: the base most cases change one field of", r, None),
        ("sae", "an SAE-basis probe", _sae(), "residual"),
        ("attention", "an attention probe carrying head.attention_query", _attention(), "residual"),
        ("softmax", "a softmax probe carrying aggregation.params.tau", _softmax(), "residual"),
        ("windows-and-bands", "per-window bars with their own length bands, and a global length table",
         _windowed(), "residual"),
    ]
    included = [
        ("model.hf_id", "example-org/other-model"), ("model.revision", "f" * 40),
        ("model.n_layers", 16), ("model.architecture", "OtherForCausalLM"),
        ("model.chat_template_sha256", "c" * 64), ("model.load_dtype", "bfloat16"),
        ("model.quantization", "Q4"), ("read.layer", 4), ("scope", "response"),
        ("head.weights.0", 0.25), ("head.bias", 0.75), ("head.norm_mean.1", 0.25),
        ("head.norm_std.2", 0.75), ("aggregation.rule", "max"),
        ("render.generation_prompt", False), ("render.bos_handling", None),
    ]
    for path, value in included:
        specs.append((f"detector-{path}", f"included field {path} changed: the detector digest moves",
                      _set(r, path, value), "residual"))
    # (`model.d_model`, `sae.d_model` and `aggregation.streamable` cannot change ALONE in a valid
    # document — the width and rule validators tie them to other fields. `test_probe_identity.py`
    # moves each alone over a parsed dict.)
    specs.append(("detector-render.add_special_tokens", "included field render.add_special_tokens "
                  "changed (no bos_handling, so the start-of-text rule does not tie it)",
                  _set(_set(r, "render.bos_handling", None), "render.add_special_tokens", True),
                  "detector-render.bos_handling"))
    specs.append(("detector-aggregation.params", "included field aggregation.params changed",
                  _set(_softmax(), "aggregation.params.tau", 2.0), "softmax"))
    specs.append(("detector-head.attention_query", "included field head.attention_query changed",
                  _set(_attention(), "head.attention_query.0", 0.75), "attention"))
    for path, value in [("sae.hf_repo", "example-org/other-saes"), ("sae.path", "layer_4/width_64"),
                        ("sae.revision", "2" * 40), ("sae.weights_sha256", "d" * 64),
                        ("sae.architecture", "topk"), ("sae.n_features", 128),
                        ("sae.normalization", {"mode": "constant_norm_rescale", "scale": 1.5}),
                        ("sae.feature_indices.1", 18)]:
        specs.append((f"detector-{path}", f"included field {path} changed", _set(_sae(), path, value),
                      "sae"))
    excluded = [
        ("name", "renamed-probe"), ("description", "another description"),
        ("concept", "positive = other placeholder"), ("decision.threshold", 3.5),
        ("evidence.rung", 3), ("provenance.exported_at", "2026-10-24T00:00:00Z"),
        ("test_vectors.vectors.0.score", 0.75), ("model.mistudio_model_id", "m_other"),
        ("behaviour", {"key": "placeholder-key", "name": "Placeholder", "version": 1}),
    ]
    for path, value in excluded:
        specs.append((f"excluded-{path}", f"excluded field {path} changed: the detector digest stays",
                      _set(r, path, value), "residual"))
    specs.append(("excluded-sae.feature_labels", "excluded field sae.feature_labels changed",
                  _set(_sae(), "sae.feature_labels", ["third", "fourth"]), "sae"))
    # Omitted defaults and absent against null: the digests equal the comparison's.
    no_bias = _set(_set(_set(r, "head.bias", 0.0), "model.load_dtype", None), "render",
                   {"generation_prompt": True, "add_special_tokens": False, "bos_handling": None})
    specs.append(("defaults-explicit", "every optional detector key written as its declared default: "
                  "bias 0, hook_point resid_post, params {}, attention_query null, load_dtype null, "
                  "add_special_tokens false, bos_handling null", no_bias, "residual"))
    for path in ("head.bias", "render.add_special_tokens", "render.bos_handling", "read.hook_point",
                 "aggregation.params", "head.attention_query", "model.load_dtype"):
        specs.append((f"absent-{path}", f"{path} absent: hashes as its default",
                      _set(no_bias, path, _ABSENT), "defaults-explicit"))
    specs.append(("absent-sae.normalization", "sae.normalization absent: hashes as {}",
                  _set(_set(_sae(), "sae.normalization", {}), "sae.normalization", _ABSENT),
                  "detector-sae.normalization-empty"))
    specs.append(("detector-sae.normalization-empty", "sae.normalization written as {}",
                  _set(_sae(), "sae.normalization", {}), "sae"))
    no_render = _set(r, "render", None)
    specs.append(("render-null", "render written as null", no_render, "residual"))
    specs.append(("render-absent", "render absent: hashes as null", _set(r, "render", _ABSENT),
                  "render-null"))
    specs.append(("sae-absent", "sae absent on a residual probe: hashes as null",
                  _set(r, "sae", _ABSENT), "residual"))
    specs.append(("windows-absent", "decision.windows absent: hashes as null",
                  _set(r, "decision.windows", _ABSENT), "residual"))
    specs.append(("length-bands-absent", "decision.length_bands absent: hashes as null",
                  _set(r, "decision.length_bands", _ABSENT), "residual"))
    specs.append(("calibration-absent", "decision.calibration absent",
                  _set(r, "decision.calibration", _ABSENT), "calibration-null"))
    specs.append(("calibration-null", "decision.calibration null: the decision digest moves",
                  _set(r, "decision.calibration", None), "residual"))
    specs.append(("negative-zero", "head.bias -0.0 hashes as 0",
                  _set(no_bias, "head.bias", -0.0), "defaults-explicit"))
    # The decision.
    specs.append(("threshold-null", "threshold null: fire on nothing, a real operating point",
                  _set(r, "decision.threshold", None), "residual"))
    specs.append(("window-threshold", "one window's threshold changed: decision moves, detector stays",
                  _set(_windowed(), "decision.windows.prompt.threshold", 1.625), "windows-and-bands"))
    specs.append(("window-band-threshold", "one window length band changed",
                  _set(_windowed(), "decision.windows.prompt.length_bands.1.threshold", 1.875),
                  "windows-and-bands"))
    specs.append(("global-band-threshold", "one global length band changed",
                  _set(_windowed(), "decision.length_bands.2.threshold", 3.25), "windows-and-bands"))
    specs.append(("window-band-absent-keys", "a window band written with only its required keys "
                  "differs (its omitted keys are null, not the recorded values)",
                  _set(_windowed(), "decision.windows.prompt.length_bands",
                       [{"min_tokens": 0, "max_tokens": 99, "threshold": 1.25},
                        {"min_tokens": 100, "max_tokens": None, "threshold": 1.75}]),
                  "windows-and-bands"))
    # Strings and key order.
    specs.append(("non-ascii", "a non-ASCII model.architecture",
                  _set(r, "model.architecture", "Modèle-Ünïcode-模型"), "residual"))
    specs.append(("non-bmp-key", "aggregation.params keys outside the Basic Multilingual Plane: "
                  "RFC 8785 orders by UTF-16 code units, so U+1F600 sorts before U+FF61",
                  _set(r, "aggregation.params", {"｡": 1, "\U0001f600": 2, "a": 3, "é": 4}),
                  "residual"))
    specs.append(("escapes", "a model.architecture with a quote, a backslash and control characters",
                  _set(r, "model.architecture", 'A"B\\C\n\t\u0001'), "residual"))
    return specs


def build_cases() -> Dict[str, Any]:
    from src.schemas import probe_identity as pi
    from src.schemas.probe_definition import ProbeDefinitionV1

    specs = _case_specs()
    out_cases = []
    by_name = {name: document for name, _a, document, _c in specs}
    for name, about, document, compare in specs:
        ProbeDefinitionV1.model_validate(document)          # every case is a valid document
        detector, decision = pi.document_identity(document)
        entry: Dict[str, Any] = {
            "name": name,
            "about": about,
            "document": document,
            "detector_canonical": pi.canonical_bytes(
                "detector", pi.detector_payload(document)).decode("utf-8"),
            "decision_canonical": pi.canonical_bytes(
                "decision", pi.decision_payload(document["decision"])).decode("utf-8"),
            "detector_digest": detector,
            "decision_digest": decision,
        }
        if compare is not None:
            other = pi.document_identity(by_name[compare])
            entry["relation"] = {
                "to": compare,
                "detector": "equal" if detector == other[0] else "differs",
                "decision": "equal" if decision == other[1] else "differs",
            }
        out_cases.append(entry)
    # Two identity cases: decision_revision absent and present — the digests do not depend on it.
    for revision in (None, 7):
        document = copy.deepcopy(by_name["residual"])
        identity = pi.identity_for(ProbeDefinitionV1.model_validate(document), revision)
        document["identity"] = identity.model_dump(mode="json")
        if revision is None:
            del document["identity"]["decision_revision"]
        ProbeDefinitionV1.model_validate(document)
        detector, decision = pi.document_identity(document)
        out_cases.append({
            "name": f"identity-revision-{'absent' if revision is None else revision}",
            "about": "a document carrying its own identity block; identity is excluded from both "
                     "digests and decision_revision is not hashed",
            "document": document,
            "detector_canonical": pi.canonical_bytes(
                "detector", pi.detector_payload(document)).decode("utf-8"),
            "decision_canonical": pi.canonical_bytes(
                "decision", pi.decision_payload(document["decision"])).decode("utf-8"),
            "detector_digest": detector,
            "decision_digest": decision,
            "relation": {"to": "residual", "detector": "equal", "decision": "equal"},
        })
    numbers = []
    seen = set()
    for value in _number_values():
        bits = struct.pack(">d", value).hex()
        if bits in seen:
            continue
        seen.add(bits)
        text = pi.canonical_bytes("decision", {"n": value}).decode()
        numbers.append({"bits": bits, "jcs": text.split('"payload":{"n":', 1)[1][:-2]})
    return {
        "version": 1,
        "about": ABOUT,
        "algorithm": pi.IDENTITY_ALGORITHM,
        "framing": FRAMING,
        "projection": {
            "detector": {
                "include": {k: list(v) for k, v in pi.DETECTOR_V1.items()},
                "exclude": {k: list(v) for k, v in pi.EXCLUDED_V1.items()},
                "added_after_v1": {k: list(v) for k, v in pi.DETECTOR_ADDED_AFTER_V1.items()},
            },
            "decision": {
                "include": {k: list(v) for k, v in pi.DECISION_V1.items()},
                "exclude": {k: list(v) for k, v in pi.DECISION_EXCLUDED_V1.items()},
                "added_after_v1": {k: list(v) for k, v in pi.DECISION_ADDED_AFTER_V1.items()},
            },
            "defaults": pi.defaults_table(),
        },
        "numbers": numbers,
        "cases": out_cases,
    }


def render_cases_file() -> str:
    """Indented, except each case's `document`, which is written on one line: 70 indented documents
    with their test vectors made a 590 KB file nobody could review as a diff."""
    built = build_cases()
    documents = []
    for index, case in enumerate(built["cases"]):
        documents.append(case["document"])
        case["document"] = f"@@DOCUMENT-{index}@@"
    text = json.dumps(built, indent=2, ensure_ascii=False)
    for index, document in enumerate(documents):
        text = text.replace(
            json.dumps(f"@@DOCUMENT-{index}@@"),
            json.dumps(document, ensure_ascii=False, separators=(", ", ": ")),
            1,
        )
    return text + "\n"


if __name__ == "__main__":  # pragma: no cover - the generator
    if "--write" not in sys.argv:
        print("usage: python -m tests.unit.test_probe_identity_cases --write", file=sys.stderr)
        raise SystemExit(2)
    COMMITTED.write_text(render_cases_file(), encoding="utf-8")
    print(f"wrote {COMMITTED}")
