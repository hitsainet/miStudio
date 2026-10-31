"""`mistudio.probe-identity/1` — the detector and decision projections (039 FR-1, FR-2, FR-3, A5).

What must hold, because a consumer joins verdicts on these values:
* every INCLUDED detector field moves the detector digest, and every EXCLUDED field leaves it equal;
* the projection is complete against the LIVE `model_fields` — a field added to the contract fails
  here until someone decides which side of the identity it is on;
* absent, `null` (for Optional-None keys) and the schema's declared default hash equal;
* the decision revision is the history's MAXIMUM, never its length;
* a window-only re-cut moves the decision digest and leaves the detector digest equal;
* NaN / ±infinity / an integer beyond 2^53 is refused by path, never hashed as `null`;
* numbers are RFC 8785 (`1`, `0`, `1e-7`), never a pydantic or `json.dumps` form.

Fixtures differ from their baseline in exactly the field under test, and behaviour keys are random.
Controls run against this file are recorded in `0xcc/reviews/039_implementation_controls_2026-10-23.md`.
"""
from __future__ import annotations

import copy
import json
import math
import secrets
import struct
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.schemas import probe_identity as pi
from src.schemas import probe_definition as contract
from src.schemas.probe_definition import ProbeDefinitionV1, ProbeHead
from tests.unit.test_probe_definition import definition

CASES = Path(__file__).resolve().parents[3] / "docs" / "schemas" / "probe-identity-cases.json"
BACKEND = Path(__file__).resolve().parents[2]


def _set(document: dict, path: str, value) -> dict:
    """A copy with `a.b.0.c` set — `.N` indexes a list. `value is _DELETE` removes the key."""
    out = copy.deepcopy(document)
    node = out
    parts = path.split(".")
    for part in parts[:-1]:
        node = node[int(part)] if part.isdigit() else node[part]
    last = parts[-1]
    if value is _DELETE:
        del node[last]
    elif last.isdigit():
        node[int(last)] = value
    else:
        node[last] = value
    return out


_DELETE = object()


def _sae_document() -> dict:
    return definition(
        basis="sae_features",
        head={"weights": [0.1, 0.2], "bias": -0.5, "norm_mean": [0.0, 0.0],
              "norm_std": [1.0, 1.0], "attention_query": None},
        sae={
            "hf_repo": "example-org/example-saes", "path": "layer_11/width_16k",
            "revision": "1" * 40, "weights_sha256": "b" * 64, "architecture": "jumprelu",
            "d_model": 4, "n_features": 16384, "normalization": {"mode": "none"},
            "feature_indices": [3, 9], "feature_labels": ["a", "b"],
        },
    )


def _attention_document() -> dict:
    document = definition(aggregation={"rule": "attention", "params": {}, "streamable": True})
    document["head"]["attention_query"] = [0.5, 0.25, 0.125, 0.0625]
    return document


def _softmax_document() -> dict:
    return definition(aggregation={"rule": "softmax", "params": {"tau": 1.0}, "streamable": True})


def _rendered_document() -> dict:
    return definition(render={
        "generation_prompt": True, "add_special_tokens": False,
        "bos_handling": {"template_wrote_bos": True, "tokenizer_added_bos": False, "bos_count": 1},
    })


def _windowed_document() -> dict:
    document = definition()
    document["decision"]["windows"] = {
        "prompt": {"threshold": 1.5, "target_fpr": 0.01, "realised_fpr": 0.01, "n_negatives": 400,
                   "scope": "user", "length_bands": None},
        "response": {"threshold": 2.5, "target_fpr": 0.01, "realised_fpr": 0.009,
                     "n_negatives": 400, "scope": "assistant", "length_bands": None},
    }
    return document


def _digests(document: dict) -> tuple:
    return pi.document_identity(json.loads(json.dumps(document)))


# ── FR-1: every included field moves the detector digest ─────────────────────

#: (case id, baseline builder, path, new value). Covers the FR-1 acceptance list and every other
#: included key, so a key dropped from `DETECTOR_V1` turns ITS case red.
INCLUDED = [
    ("head.weights[0]", definition, "head.weights.0", 0.11),
    ("head.bias", definition, "head.bias", -0.25),
    ("head.norm_mean[2]", definition, "head.norm_mean.2", 0.5),
    ("head.norm_std[1]", definition, "head.norm_std.1", 2.0),
    ("head.attention_query[0]", _attention_document, "head.attention_query.0", 0.75),
    ("read.layer", definition, "read.layer", 12),
    ("read.hook_point", definition, "read.hook_point", "resid_pre"),
    ("scope", definition, "scope", "prompt"),
    ("basis", definition, "basis", "sae_features"),
    ("aggregation.rule", definition, "aggregation.rule", "max"),
    ("aggregation.params value", _softmax_document, "aggregation.params.tau", 2.0),
    ("aggregation.streamable", definition, "aggregation.streamable", False),
    ("sae.feature_indices[1]", _sae_document, "sae.feature_indices.1", 10),
    ("sae.hf_repo", _sae_document, "sae.hf_repo", "example-org/other-saes"),
    ("sae.path", _sae_document, "sae.path", "layer_12/width_16k"),
    ("sae.revision", _sae_document, "sae.revision", "2" * 40),
    ("sae.weights_sha256", _sae_document, "sae.weights_sha256", "c" * 64),
    ("sae.architecture", _sae_document, "sae.architecture", "topk"),
    ("sae.d_model", _sae_document, "sae.d_model", 8),
    ("sae.n_features", _sae_document, "sae.n_features", 32768),
    ("sae.normalization", _sae_document, "sae.normalization.mode", "constant_norm_rescale"),
    ("render.generation_prompt", _rendered_document, "render.generation_prompt", False),
    ("render.add_special_tokens", _rendered_document, "render.add_special_tokens", True),
    ("render.bos_handling.bos_count", _rendered_document, "render.bos_handling.bos_count", 0),
    ("render.bos_handling.template_wrote_bos", _rendered_document,
     "render.bos_handling.template_wrote_bos", False),
    ("render.bos_handling.tokenizer_added_bos", _rendered_document,
     "render.bos_handling.tokenizer_added_bos", True),
    ("model.hf_id", definition, "model.hf_id", "example-org/other-model"),
    ("model.revision", definition, "model.revision", "f" * 40),
    ("model.d_model", definition, "model.d_model", 8),
    ("model.n_layers", definition, "model.n_layers", 40),
    ("model.architecture", definition, "model.architecture", "mistral"),
    ("model.chat_template_sha256", definition, "model.chat_template_sha256", "d" * 64),
    ("model.load_dtype", definition, "model.load_dtype", "bfloat16"),
    ("model.quantization", definition, "model.quantization", "Q4"),
]


@pytest.mark.parametrize("case,base,path,value", INCLUDED, ids=[c[0] for c in INCLUDED])
def test_each_detector_field_moves_the_digest(case, base, path, value):
    baseline = base()
    changed = _set(baseline, path, value)
    assert _digests(changed)[0] != _digests(baseline)[0], f"{case} did not move the detector digest"
    # …and only the detector: the decision block is untouched.
    assert _digests(changed)[1] == _digests(baseline)[1]


def test_the_parametrised_list_covers_every_included_key():
    """A key added to `DETECTOR_V1` without a case here would be included and never proven to bite."""
    covered = {case.split("[")[0].split(" ")[0] for case, *_ in INCLUDED}
    covered |= {"head.weights", "sae.normalization", "aggregation.params"}
    expected = set()
    for path, keys in pi.DETECTOR_V1.items():
        for key in keys:
            full = key if path == "" else f"{path}.{key}"
            if full in pi.DETECTOR_V1:          # an object with its own entry: its keys are covered
                continue
            expected.add(full)
    assert expected - covered == set()


# ── FR-1: excluded fields never move it ───────────────────────────────────────

def _random_behaviour() -> dict:
    return {"key": "b-" + secrets.token_hex(4), "name": "placeholder", "version": 1}


EXCLUDED = [
    ("decision.threshold", definition, "decision.threshold", 9.75),
    ("decision.target_fpr", definition, "decision.target_fpr", 0.001),
    ("evidence.rung", definition, "evidence.rung", 3),
    ("provenance.exported_at", definition, "provenance.exported_at", "2026-10-23T12:00:00Z"),
    ("name", definition, "name", "renamed"),
    ("description", definition, "description", "another description"),
    ("concept", definition, "concept", "positive = something else"),
    ("behaviour (random key 1)", definition, "behaviour", _random_behaviour()),
    ("behaviour (random key 2)", definition, "behaviour", _random_behaviour()),
    ("a test vector", definition, "test_vectors.vectors.0.score", 0.75),
    ("model.mistudio_model_id", definition, "model.mistudio_model_id", "m_other"),
    ("sae.feature_labels", _sae_document, "sae.feature_labels", ["c", "d"]),
]


@pytest.mark.parametrize("case,base,path,value", EXCLUDED, ids=[c[0] for c in EXCLUDED])
def test_excluded_fields_do_not_move_the_detector_digest(case, base, path, value):
    baseline = base()
    assert _digests(_set(baseline, path, value))[0] == _digests(baseline)[0], (
        f"{case} moved the detector digest; it says something ABOUT the detector, not what it is"
    )


def test_two_random_behaviour_keys_differ_and_still_agree():
    one, two = _random_behaviour(), _random_behaviour()
    assert one["key"] != two["key"]
    assert _digests(definition(behaviour=one)) == _digests(definition(behaviour=two))


# ── FR-1 guard: complete against the LIVE schema ──────────────────────────────

def test_projection_is_complete_against_live_model_fields():
    assert pi.unclassified_fields(
        pi.DETECTOR_V1, pi.EXCLUDED_V1, pi.DETECTOR_ADDED_AFTER_V1, ProbeDefinitionV1
    ) == []
    assert pi.unclassified_fields(
        pi.DECISION_V1, pi.DECISION_EXCLUDED_V1, pi.DECISION_ADDED_AFTER_V1, contract.Decision
    ) == []


def test_the_guard_sees_a_field_added_to_a_nested_model():
    """The guard's teeth: a `ProbeHead` with one more optional field is reported, by path."""
    from typing import Optional

    class HeadWithADummy(ProbeHead):
        dummy: Optional[float] = None

    class DefinitionWithADummyHead(ProbeDefinitionV1):
        head: HeadWithADummy

    problems = pi.unclassified_fields(
        pi.DETECTOR_V1, pi.EXCLUDED_V1, pi.DETECTOR_ADDED_AFTER_V1, DefinitionWithADummyHead
    )
    assert problems == ["head.dummy: in neither the projection nor the exclusions"]


def test_the_guard_enters_bos_handling():
    from typing import Optional

    class Bos(contract.BosHandling):
        extra_flag: Optional[bool] = None

    class Render(contract.RenderForm):
        bos_handling: Optional[Bos] = None

    class Doc(ProbeDefinitionV1):
        render: Optional[Render] = None

    assert pi.unclassified_fields(
        pi.DETECTOR_V1, pi.EXCLUDED_V1, pi.DETECTOR_ADDED_AFTER_V1, Doc
    ) == ["render.bos_handling.extra_flag: in neither the projection nor the exclusions"]


def test_the_guard_sees_an_included_nested_model_with_no_path_entry():
    """Review R1: a nested MODEL listed as an included key but given no projection path of its own
    would be hashed as whatever the dump wrote — its sub-fields unclassified, and a field added to it
    later entering every digest silently. The guard reports it by path."""
    from typing import Optional

    class HeadWithANestedModel(ProbeHead):
        nested_extra: Optional[contract.BosHandling] = None

    class Doc(ProbeDefinitionV1):
        head: HeadWithANestedModel

    projection = dict(pi.DETECTOR_V1)
    projection["head"] = tuple(projection["head"]) + ("nested_extra",)
    problems = pi.unclassified_fields(projection, pi.EXCLUDED_V1, pi.DETECTOR_ADDED_AFTER_V1, Doc)
    assert len(problems) == 1 and problems[0].startswith(
        "head.nested_extra: a nested model (BosHandling) with no projection entry"
    ), problems


# ── A3: absent, null and the declared default hash equal ──────────────────────

ABSENT_EQUALS = [
    # (case, baseline builder, path, the explicit value that must equal "absent")
    ("render", definition, "render", None),
    ("sae", definition, "sae", None),
    ("head.bias", lambda: definition(head={"weights": [0.1] * 4, "bias": 0.0,
                                           "norm_mean": [0.0] * 4, "norm_std": [1.0] * 4}),
     "head.bias", 0.0),
    ("head.attention_query", definition, "head.attention_query", None),
    ("read.hook_point", definition, "read.hook_point", "resid_post"),
    ("aggregation.params", definition, "aggregation.params", {}),
    ("sae.normalization", lambda: _set(_sae_document(), "sae.normalization", {}),
     "sae.normalization", {}),
    ("render.add_special_tokens", lambda: definition(render={"generation_prompt": True}),
     "render.add_special_tokens", False),
    ("render.bos_handling", lambda: definition(render={"generation_prompt": True}),
     "render.bos_handling", None),
    ("model.load_dtype", definition, "model.load_dtype", None),
    ("decision.windows", definition, "decision.windows", None),
    ("decision.length_bands", definition, "decision.length_bands", None),
    ("decision.calibration", definition, "decision.calibration", None),
]


@pytest.mark.parametrize("case,base,path,explicit", ABSENT_EQUALS, ids=[c[0] for c in ABSENT_EQUALS])
def test_absent_and_null_and_default_hash_equal(case, base, path, explicit):
    baseline = base()
    absent = _set(baseline, path, _DELETE) if _has(baseline, path) else baseline
    written = _set(baseline, path, explicit)
    assert _digests(absent) == _digests(written), f"absent {case} hashes differently from {explicit!r}"


def _has(document: dict, path: str) -> bool:
    node = document
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return False
        node = node[part]
    return True


def test_the_defaults_are_the_schemas_own():
    table = pi.defaults_table()
    assert table["detector.head.bias"] == ProbeHead.model_fields["bias"].default == 0.0
    assert table["detector.read.hook_point"] == "resid_post"
    assert table["detector.render.add_special_tokens"] is False
    assert table["detector.aggregation.params"] == {} and table["detector.sae.normalization"] == {}
    assert table["detector.render"] is None and table["detector.sae"] is None
    assert table["decision.windows"] is None and table["decision.length_bands"] is None
    # Required keys have no default: a valid document always carries them.
    assert "detector.head.weights" not in table and "decision.length_bands[].min_tokens" not in table


def test_negative_zero_hashes_as_zero():
    base = definition()
    assert _digests(_set(base, "head.bias", -0.0)) == _digests(_set(base, "head.bias", 0.0))
    assert _digests(_set(base, "decision.threshold", -0.0)) == _digests(
        _set(base, "decision.threshold", 0.0)
    )


# ── FR-2: the decision ────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "history,expected",
    [
        ([{"revision": 1}, {"revision": 2}, {"revision": 4}], 4),
        ([{"revision": 4}, {"revision": 1}, {"revision": 2}], 4),
        ([{"revision": 1}], 1),
        ([], 1),
        (None, 1),
    ],
)
def test_revision_is_the_history_max(history, expected):
    assert pi.current_revision(history) == expected


def test_window_only_change_moves_decision_not_detector():
    base = _windowed_document()
    changed = _set(base, "decision.windows.prompt.threshold", 1.75)
    assert _digests(changed)[0] == _digests(base)[0]
    assert _digests(changed)[1] != _digests(base)[1]


def test_global_threshold_change_moves_decision_not_detector():
    base = definition()
    changed = _set(base, "decision.threshold", 2.5)
    assert _digests(changed)[0] == _digests(base)[0]
    assert _digests(changed)[1] != _digests(base)[1]


def test_fire_on_nothing_bar_has_a_digest():
    base = definition()
    nothing = _set(base, "decision.threshold", None)
    digest = _digests(nothing)[1]
    assert len(digest) == 64 and digest != _digests(base)[1]


def test_a_window_length_band_moves_the_decision():
    base = _windowed_document()
    bands = [{"min_tokens": 0, "max_tokens": 99, "threshold": 1.0},
             {"min_tokens": 100, "max_tokens": None, "threshold": 1.4}]
    changed = _set(base, "decision.windows.prompt.length_bands", bands)
    assert _digests(changed)[1] != _digests(base)[1]
    moved = _set(changed, "decision.windows.prompt.length_bands.1.threshold", 1.5)
    assert _digests(moved)[1] != _digests(changed)[1]


def test_the_model_path_and_the_file_path_agree():
    """`identity_for` (the build's path, through the model) equals `document_identity` of the bytes
    the builder writes — the publisher and the pre-039 recompute read files, the build reads models."""
    from src.services.probe_definition_builder import serialise

    for document in (definition(), _sae_document(), _attention_document(), _rendered_document(),
                     _windowed_document()):
        model = ProbeDefinitionV1.model_validate(document)
        identity = pi.identity_for(model, 3)
        assert (identity.detector_digest, identity.decision_digest) == pi.document_identity(
            json.loads(serialise(model))
        )
        assert identity.decision_digest == pi.decision_digest_of(model.decision)


# ── A5: non-finite numbers are refused by path ────────────────────────────────

def test_non_finite_is_refused_by_path():
    # A free-form dict value: the contract's `allow_inf_nan` does not reach it, and pydantic would
    # dump the NaN as `null`.
    model = ProbeDefinitionV1.model_validate(_softmax_document())
    model.aggregation.params["tau"] = float("nan")
    with pytest.raises(pi.NonFiniteNumber) as raised:
        pi.identity_for(model, 1)
    assert raised.value.path == "aggregation.params.tau" and raised.value.code == "non_finite_number"

    # A float field in a parsed document (a file read back).
    document = json.loads(json.dumps(definition()))
    document["head"]["weights"][2] = float("inf")
    with pytest.raises(pi.NonFiniteNumber) as raised:
        pi.document_identity(document)
    assert raised.value.path == "head.weights[2]"


def test_an_integer_beyond_two_to_the_53_is_refused_by_path():
    model = ProbeDefinitionV1.model_validate(_sae_document())
    model.sae.normalization["count"] = 2**60
    with pytest.raises(pi.IdentityInputRefused) as raised:
        pi.identity_for(model, 1)
    assert raised.value.path == "sae.normalization.count"


def test_the_integer_boundary_is_exactly_two_to_the_53_minus_one():
    """Review R1: ±(2^53 − 1) is RFC 8785's exact range and must hash; one past it is refused."""
    pi.refuse_non_finite({"n": 2**53 - 1, "m": -(2**53 - 1)})
    assert pi.canonical_bytes("detector", {"n": 2**53 - 1}).endswith(b'"payload":{"n":9007199254740991}}')
    for value in (2**53, -(2**53)):
        with pytest.raises(pi.IdentityInputRefused) as raised:
            pi.refuse_non_finite({"n": value})
        assert raised.value.path == "n"


#: Paths inside `definition()` whose model has a float field, one per contract model that has one.
FLOAT_FIELDS = [
    ("ProbeHead.weights", lambda d: _set(d, "head.weights.0", "{v}")),
    ("ProbeHead.bias", lambda d: _set(d, "head.bias", "{v}")),
    ("Decision.threshold", lambda d: _set(d, "decision.threshold", "{v}")),
    ("WindowDecision.threshold",
     lambda d: _set(_windowed_document(), "decision.windows.prompt.threshold", "{v}")),
    ("LengthBand.threshold", lambda d: _set(
        d, "decision.length_bands",
        [{"min_tokens": 0, "max_tokens": None, "threshold": "{v}"}])),
    ("JudgeEvidence.per_set_auroc", lambda d: _set(
        d, "evidence.judge", {"model": "j", "prompt_version": "1", "per_set_auroc": {"x": "{v}"}})),
    ("TestVector.score", lambda d: _set(d, "test_vectors.vectors.0.score", "{v}")),
    ("TestVector.token_scores", lambda d: _set(d, "test_vectors.vectors.0.token_scores.0", "{v}")),
    ("TestVectors.tolerance", lambda d: _set(d, "test_vectors.tolerance", "{v}")),
]


def _with_value(document, value):
    """The document with the one `"{v}"` marker replaced by `value` (NaN cannot go through JSON)."""
    def walk(node):
        if isinstance(node, dict):
            return {key: value if item == "{v}" else walk(item) for key, item in node.items()}
        if isinstance(node, list):
            return [value if item == "{v}" else walk(item) for item in node]
        return node
    return walk(document)


NON_FINITE = [
    pytest.param(case, mutate, value, id=f"{case}-{label}")
    for case, mutate in FLOAT_FIELDS
    for label, value in (("nan", float("nan")), ("inf", float("inf")), ("-inf", float("-inf")))
    # -inf is refused by `gt=0` alone; +inf and NaN are what the setting adds there.
    if not (case == "TestVectors.tolerance" and label == "-inf")
]


@pytest.mark.parametrize("case,mutate,value", NON_FINITE)
def test_contract_refuses_nan(case, mutate, value):
    document = _with_value(mutate(definition()), value)
    with pytest.raises(ValidationError):
        ProbeDefinitionV1.model_validate(document)


def test_every_contract_model_refuses_non_finite_numbers():
    """Derived from the module, so a model added later without the setting fails here."""
    from pydantic import BaseModel

    models = [
        value for value in vars(contract).values()
        if isinstance(value, type) and issubclass(value, BaseModel) and value is not BaseModel
        and value.__module__ == contract.__name__
    ]
    assert len(models) >= 21
    lacking = sorted(m.__name__ for m in models if m.model_config.get("allow_inf_nan") is not False)
    assert lacking == []


# ── FR-3: one canonical form, RFC 8785 numbers ────────────────────────────────

def _number_cases():
    return json.loads(CASES.read_text())["numbers"]


def _float(bits: str) -> float:
    return struct.unpack(">d", bytes.fromhex(bits))[0]


def test_number_cases_match():
    """Through the module's OWN framing, so a canonicaliser swapped inside it is caught here."""
    cases = _number_cases()
    assert len(cases) >= 90
    wrong = []
    for case in cases:
        got = pi.canonical_bytes("decision", {"x": _float(case["bits"])}).decode()
        want = '{"algorithm":"mistudio.probe-identity/1","digest":"decision","payload":{"x":%s}}' % (
            case["jcs"]
        )
        if got != want:
            wrong.append((case["bits"], case["jcs"], got))
    assert wrong == []


def test_dump_is_not_canonical():
    """The two forms FR-3 forbids really do differ from RFC 8785 on the case file's numbers."""
    from pydantic_core import to_json

    cases = [(_float(c["bits"]), c["jcs"]) for c in _number_cases()]
    differs_json = [jcs for value, jcs in cases if json.dumps(value) != jcs]
    differs_pydantic = [jcs for value, jcs in cases if to_json(value).decode() != jcs]
    assert "1" in differs_json and "1e-7" in differs_json and "0" in differs_pydantic
    assert len(differs_json) >= 10 and len(differs_pydantic) >= 10


def test_the_framing_is_the_documented_one():
    payload = {"b": 1.0, "a": [0.5, -0.0]}
    assert pi.canonical_bytes("detector", payload) == (
        b'{"algorithm":"mistudio.probe-identity/1","digest":"detector","payload":{"a":[0.5,0],"b":1}}'
    )
    with pytest.raises(ValueError):
        pi.canonical_bytes("other", payload)


# ── __main__ --bench (FTDD §11.3) ─────────────────────────────────────────────

def test_the_bench_prints_both_digests_timings():
    result = subprocess.run(
        [sys.executable, "-m", "src.schemas.probe_identity", "--bench", "--iterations", "1",
         "--d-model", "256"],
        cwd=BACKEND, capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    lines = [line for line in result.stdout.splitlines() if " digest " in line]
    assert [line.split()[0] for line in lines] == ["detector", "decision"]
    assert all("p50=" in line and "p95=" in line and "d_model=256" in line for line in lines)
    assert not math.isnan(float(lines[0].split("p50=")[1].split()[0]))
