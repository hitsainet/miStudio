"""`mistudio.probe-identity/1` — the detector and decision digests of a probe definition (039, BR-025).

WHAT THIS IS FOR. A consumer that holds a verdict needs to know which detector produced it and
which bar it was judged against, in a form it can join on across systems. Two digests answer
that, and they are separate on purpose (D16): a re-cut of the threshold is a new DECISION over the
SAME detector, and a consumer must be able to see exactly that.

  detector_digest = SHA-256( JCS({"algorithm": "mistudio.probe-identity/1",
                                  "digest": "detector", "payload": <detector projection>}) )
  decision_digest = SHA-256( JCS({"algorithm": "mistudio.probe-identity/1",
                                  "digest": "decision", "payload": <decision projection>}) )

JCS is RFC 8785 (the `rfc8785` package). The label sits INSIDE the hashed bytes, so a digest can
never be mistaken for one computed under another algorithm, and the `digest` kind separates the two
domains so a payload cannot be replayed as the other kind.

⚠ PARSED JSON VALUES, NEVER A PYDANTIC DUMP (FR-3). Pydantic writes `1.0`, `-0.0` and `1e-07`;
RFC 8785 writes `1`, `0` and `1e-7`. A digest over a dump would be correct only for implementations
that happen to share pydantic's number formatting — i.e. no consumer. The case file
`docs/schemas/probe-identity-cases.json` carries ~100 number vectors that make this fail loudly.

⚠ THE PROJECTIONS ARE NAMED KEY LISTS, AND EVERY /1 KEY IS ALWAYS PRESENT. An absent key is written
as the schema's declared default (`head.bias` → 0, `read.hook_point` → "resid_post"), or `null`
where the default is None. So absent, `null` (for Optional-None keys) and the explicit default all
hash equal, and a document that omits a defaulted field means the same detector as one that writes
it. The defaults are DERIVED from the live `model_fields` and published in the case file, so a
consumer that does not run pydantic can reproduce them.

⚠ THE LITERALS BELOW ARE THE ONE DELIBERATE HAND-KEPT LIST. The projection IS a named key list —
that is what makes it stable when the contract grows. Its COMPLETENESS is not hand-kept: a test
walks the live `model_fields` of every model the projection enters and fails on any field that is
neither included nor excluded (`unclassified_fields`).

⚠ GROWTH (A6). A key added to a projected model after /1 goes in `DETECTOR_ADDED_AFTER_V1` and is
projected only when non-null and not its default, so every existing document keeps its digest. A
change of MEANING is `/2`, never an edit here.

⚠ INTEGRITY, NOT AUTHENTICITY. A digest proves a document agrees with itself, not who produced it.

This module is part of the contract layer (IDL-58): it imports the contract and nothing from
`services`.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import statistics
import time
import typing
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Type

import rfc8785
from pydantic import BaseModel, ValidationError
from pydantic_core import PydanticUndefined

from .probe_definition import (
    HEX64,
    IDENTITY_ALGORITHM_PATTERN,
    Decision,
    ProbeDefinitionV1,
    ProbeIdentity,
)

__all__ = [
    "IDENTITY_ALGORITHM",
    "IDENTITY_ALGORITHM_PATTERN",
    "KNOWN_ALGORITHMS",
    "HEX64",
    "DETECTOR_V1",
    "EXCLUDED_V1",
    "DETECTOR_ADDED_AFTER_V1",
    "DECISION_V1",
    "DECISION_ADDED_AFTER_V1",
    "PROJECTED",
    "IdentityInputRefused",
    "NonFiniteNumber",
    "IdentityRefused",
    "defaults_table",
    "detector_payload",
    "decision_payload",
    "canonical_bytes",
    "digest",
    "refuse_non_finite",
    "parsed_projection",
    "identity_for",
    "check_identity",
    "identity_refusal",
    "document_identity",
    "decision_digest_of",
    "current_revision",
    "unclassified_fields",
]

IDENTITY_ALGORITHM = "mistudio.probe-identity/1"
KNOWN_ALGORITHMS = frozenset({IDENTITY_ALGORITHM})

# ── the /1 projections ─────────────────────────────────────────────────────────
# Keyed by the PATH of the object inside the projected value ("" = the root). A path segment ending
# in `[]` is each element of a list, `{}` each value of a map. Order inside a tuple is irrelevant
# (JCS sorts keys); it follows the contract's declaration order for readability.

#: The detector: what the probe IS — the activations it reads and the function it computes.
DETECTOR_V1: Dict[str, Tuple[str, ...]] = {
    "": ("model", "read", "scope", "basis", "head", "sae", "aggregation", "render"),
    "model": (
        "hf_id", "revision", "d_model", "n_layers", "architecture",
        "chat_template_sha256", "load_dtype", "quantization",
    ),
    "read": ("layer", "hook_point"),
    "head": ("weights", "bias", "norm_mean", "norm_std", "attention_query"),
    "sae": (
        "hf_repo", "path", "revision", "weights_sha256", "architecture", "d_model",
        "n_features", "normalization", "feature_indices",
    ),
    "aggregation": ("rule", "params", "streamable"),
    "render": ("generation_prompt", "add_special_tokens", "bos_handling"),
    # `bos_handling` is projected key by key rather than "whole", so a field added to `BosHandling`
    # later is classified here (and seen by the completeness guard) instead of silently entering
    # every digest as `null`.
    "render.bos_handling": ("template_wrote_bos", "tokenizer_added_bos", "bos_count"),
}

#: Excluded BY NAME: what is said ABOUT the detector, not what it is. `decision` has its own digest;
#: `mistudio_model_id` is a local row handle; `feature_labels` are human annotations.
EXCLUDED_V1: Dict[str, Tuple[str, ...]] = {
    "": (
        "kind", "name", "description", "concept", "decision", "evidence", "provenance",
        "test_vectors", "behaviour", "identity",
    ),
    "model": ("mistudio_model_id",),
    "sae": ("feature_labels",),
}

#: Keys classified into the detector AFTER /1 (A6): projected only when non-null and not default.
DETECTOR_ADDED_AFTER_V1: Dict[str, Tuple[str, ...]] = {}

#: The decision: EVERY key of `Decision`, and of each object inside it, always present (FR-2).
DECISION_V1: Dict[str, Tuple[str, ...]] = {
    "": (
        "threshold", "target_fpr", "realised_fpr", "threshold_source", "calibration",
        "windows", "length_bands",
    ),
    "calibration": ("hf_id", "config", "split", "revision"),
    "windows{}": ("threshold", "target_fpr", "realised_fpr", "n_negatives", "scope", "length_bands"),
    "windows{}.length_bands[]": (
        "min_tokens", "max_tokens", "threshold", "threshold_source", "target_fpr",
        "realised_fpr", "n_negatives",
    ),
    "length_bands[]": (
        "min_tokens", "max_tokens", "threshold", "threshold_source", "target_fpr",
        "realised_fpr", "n_negatives",
    ),
}
DECISION_EXCLUDED_V1: Dict[str, Tuple[str, ...]] = {}
DECISION_ADDED_AFTER_V1: Dict[str, Tuple[str, ...]] = {}

#: The top-level fields that go through serialise-and-parse. Test vectors (up to 2 MB) never do.
PROJECTED = frozenset(DETECTOR_V1[""]) | {"decision"}

#: RFC 8785 serialises numbers as IEEE-754 doubles; an integer beyond ±(2^53 − 1) has no exact form.
_SAFE_INTEGER = 2**53 - 1


# ── refusals ───────────────────────────────────────────────────────────────────


class IdentityInputRefused(ValueError):
    """A value no digest may be computed over. Carries the dotted `path` and a machine `code`."""

    code = "identity_input_refused"

    def __init__(self, path: str, reason: str) -> None:
        super().__init__(f"{self.code} at {path}: {reason}")
        self.path = path
        self.reason = reason


class NonFiniteNumber(IdentityInputRefused):
    """NaN or ±infinity in a projected value. JSON has no such number, and pydantic would write
    `null` — which, for a threshold, is the "fire on nothing" operating point."""

    code = "non_finite_number"


class IdentityRefused(ValueError):
    """A carried `identity` that is not the document's own (FR-5). `detail` is the structured shape
    every surface returns: `{"refused": "identity_mismatch", "field", "carried", "recomputed",
    "algorithm"}` or `{"refused": "identity_algorithm_unknown", "algorithm", "known"}`."""

    def __init__(self, detail: Dict[str, Any]) -> None:
        super().__init__(json.dumps(detail, sort_keys=True))
        self.detail = detail


# ── models behind the paths (derived, never a second hand-kept map) ───────────


def _unwrap(annotation: Any) -> Any:
    """`Optional[X]` → X, `List[X]` → X, `Dict[K, X]` → X — the model a container holds."""
    origin = typing.get_origin(annotation)
    if origin is typing.Union or str(origin) == "types.UnionType":
        args = [arg for arg in typing.get_args(annotation) if arg is not type(None)]
        return _unwrap(args[0]) if len(args) == 1 else annotation
    if origin in (list, List, Sequence):
        return _unwrap(typing.get_args(annotation)[0])
    if origin in (dict, Dict, Mapping):
        return _unwrap(typing.get_args(annotation)[1])
    return annotation


def _model_at(root: Type[BaseModel], path: str) -> Type[BaseModel]:
    """The pydantic model a projection path names, read from the live field annotations."""
    model: Any = root
    if path == "":
        return model
    for segment in path.split("."):
        name = segment.rstrip("[]{}")
        model = _unwrap(model.model_fields[name].annotation)
    return model


def _is_model(value: Any) -> bool:
    return isinstance(value, type) and issubclass(value, BaseModel)


def _join(path: str, key: str) -> str:
    return key if path == "" else f"{path}.{key}"


def unclassified_fields(
    projection: Mapping[str, Sequence[str]],
    excluded: Mapping[str, Sequence[str]],
    added_after: Mapping[str, Sequence[str]],
    root: Type[BaseModel],
) -> List[str]:
    """Every live field the projection enters that is classified nowhere, or twice (FR-1 guard).

    Walks `model_fields` of the root and every nested model the projection enters, so a field added
    to the contract fails a test until someone decides which side of the identity it is on. Also
    reports a NESTED MODEL included without a path entry of its own (it would be hashed as whatever
    the dump wrote, unclassified), and a path entry naming no field.
    """
    problems: List[str] = []
    paths = set(projection) | set(excluded) | set(added_after)
    for path in sorted(paths):
        try:
            model = _model_at(root, path)
        except KeyError as exc:
            problems.append(f"{path or '<root>'}: names no field ({exc})")
            continue
        included = set(projection.get(path, ())) | set(added_after.get(path, ()))
        out = set(excluded.get(path, ()))
        for both in sorted(included & out):
            problems.append(f"{_join(path, both)}: both included and excluded")
        live = set(model.model_fields)
        for name in sorted(live - included - out):
            problems.append(f"{_join(path, name)}: in neither the projection nor the exclusions")
        for name in sorted((included | out) - live):
            problems.append(f"{_join(path, name)}: classified but not a field of {model.__name__}")
        for name in sorted(included & live):
            annotation = model.model_fields[name].annotation
            inner = _unwrap(annotation)
            if not _is_model(inner):
                continue
            origin_text = str(annotation)
            child = _join(path, name)
            if not any(p in projection for p in (child, f"{child}[]", f"{child}{{}}")):
                problems.append(
                    f"{child}: a nested model ({inner.__name__}) with no projection entry "
                    f"({origin_text})"
                )
    return problems


# ── defaults (A3), derived from the live schema ───────────────────────────────


def _defaults_for(
    projection: Mapping[str, Sequence[str]],
    added_after: Mapping[str, Sequence[str]],
    root: Type[BaseModel],
    prefix: str,
) -> Dict[str, Any]:
    table: Dict[str, Any] = {}
    for path, keys in projection.items():
        model = _model_at(root, path)
        for key in tuple(keys) + tuple(added_after.get(path, ())):
            field = model.model_fields[key]
            if field.is_required():
                continue                    # always present in a valid document
            default = field.get_default(call_default_factory=True)
            if default is PydanticUndefined:
                continue
            if isinstance(default, BaseModel):
                default = json.loads(default.model_dump_json())
            table[_join(prefix, _join(path, key))] = default
    return table


_DEFAULTS: Optional[Dict[str, Any]] = None


def defaults_table() -> Dict[str, Any]:
    """`{dotted path: default}` for every optional projected key — what an ABSENT key is hashed as.

    Paths are prefixed `detector.` / `decision.` and use the projection's `[]`/`{}` markers, e.g.
    `detector.head.bias` → 0.0, `decision.windows{}.length_bands[].max_tokens` → None.
    Published in the case file's `projection.defaults`.
    """
    global _DEFAULTS
    if _DEFAULTS is None:
        table = _defaults_for(DETECTOR_V1, DETECTOR_ADDED_AFTER_V1, ProbeDefinitionV1, "detector")
        table.update(_defaults_for(DECISION_V1, DECISION_ADDED_AFTER_V1, Decision, "decision"))
        _DEFAULTS = table
    return copy.deepcopy(_DEFAULTS)


# ── payloads over parsed JSON values ──────────────────────────────────────────


def _project(
    value: Mapping[str, Any],
    path: str,
    projection: Mapping[str, Sequence[str]],
    added_after: Mapping[str, Sequence[str]],
    defaults: Mapping[str, Any],
    prefix: str,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    def nested(key: str, item: Any) -> Any:
        child = _join(path, key)
        if item is None:
            return None
        if child in projection:
            return _project(item, child, projection, added_after, defaults, prefix)
        if f"{child}[]" in projection:
            return [
                _project(entry, f"{child}[]", projection, added_after, defaults, prefix)
                for entry in item
            ]
        if f"{child}{{}}" in projection:
            return {
                str(name): _project(entry, f"{child}{{}}", projection, added_after, defaults, prefix)
                for name, entry in item.items()
            }
        return copy.deepcopy(item)

    for key in projection.get(path, ()):
        if key in value:
            item = value[key]
        else:
            item = copy.deepcopy(defaults.get(_join(prefix, _join(path, key))))
        out[key] = nested(key, item)
    for key in added_after.get(path, ()):
        if key not in value or value[key] is None:
            continue
        if value[key] == defaults.get(_join(prefix, _join(path, key))):
            continue
        out[key] = nested(key, value[key])
    return out


def detector_payload(document: Mapping[str, Any]) -> Dict[str, Any]:
    """The detector projection of a PARSED document (FR-1). Every /1 key present."""
    return _project(
        document, "", DETECTOR_V1, DETECTOR_ADDED_AFTER_V1, defaults_table(), "detector"
    )


def decision_payload(decision: Mapping[str, Any]) -> Dict[str, Any]:
    """The decision projection of a PARSED `decision` block (FR-2). Every key present."""
    return _project(
        decision, "", DECISION_V1, DECISION_ADDED_AFTER_V1, defaults_table(), "decision"
    )


def canonical_bytes(kind: str, payload: Mapping[str, Any]) -> bytes:
    """RFC 8785 bytes of the framed payload — the exact bytes that are hashed."""
    if kind not in ("detector", "decision"):
        raise ValueError(f"unknown digest kind {kind!r}")
    refuse_non_finite(payload, kind)
    try:
        return rfc8785.dumps({"algorithm": IDENTITY_ALGORITHM, "digest": kind, "payload": payload})
    except rfc8785.CanonicalizationError as exc:
        # The walk above names a path for every case it knows; this is the library's own net.
        raise IdentityInputRefused(kind, str(exc)) from exc


def digest(kind: str, payload: Mapping[str, Any]) -> str:
    """Lower-case hex SHA-256 of `canonical_bytes(kind, payload)`."""
    return hashlib.sha256(canonical_bytes(kind, payload)).hexdigest()


def refuse_non_finite(value: Any, path: str = "") -> None:
    """Raise `NonFiniteNumber` naming the first NaN/±inf, or `IdentityInputRefused` for an integer
    outside JSON's exact range — before anything is serialised (A5).

    ⚠ `allow_inf_nan=False` DOES NOT REACH `Dict[str, Any]` VALUES, and pydantic dumps NaN as
    `null`. Without this walk a NaN in `aggregation.params` would hash as `null` — a digest of a
    document that does not exist.
    """
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise NonFiniteNumber(path or "<root>", f"{value!r} is not a JSON number")
        return
    if isinstance(value, int):
        if abs(value) > _SAFE_INTEGER:
            raise IdentityInputRefused(
                path or "<root>", f"integer {value} is outside ±(2^53 − 1), RFC 8785's exact range"
            )
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            refuse_non_finite(item, _join(path, str(key)))
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            refuse_non_finite(item, f"{path}[{index}]")
        return


def _refuse_constant(name: str) -> Any:
    raise NonFiniteNumber("<json>", f"{name} is not a JSON number")


def parsed_projection(definition: ProbeDefinitionV1) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """`(detector payload, decision payload)` of a model, through its PARSED JSON values (A4).

    Walks the python dump first (the only place a NaN in a free-form dict is still visible), then
    serialises the projected sub-models and parses them back, so the values hashed are exactly what
    a consumer parsing the file sees.
    """
    refuse_non_finite(definition.model_dump(include=set(PROJECTED), mode="python"))
    parsed = json.loads(
        definition.model_dump_json(include=set(PROJECTED)), parse_constant=_refuse_constant
    )
    return detector_payload(parsed), decision_payload(parsed["decision"])


def document_identity(document: Mapping[str, Any]) -> Tuple[str, str]:
    """`(detector_digest, decision_digest)` straight from a parsed document (a file read back).

    Used for definitions built before 039 (FR-17) and by the publisher; constructs no model.
    """
    projected = {key: document.get(key) for key in PROJECTED if key in document}
    refuse_non_finite(projected)
    return (
        digest("detector", detector_payload(document)),
        digest("decision", decision_payload(document.get("decision") or {})),
    )


def decision_digest_of(decision: Decision) -> str:
    """The decision digest of a contract `Decision` — what a stored snapshot is keyed by."""
    refuse_non_finite(decision.model_dump(mode="python"), "decision")
    parsed = json.loads(decision.model_dump_json(), parse_constant=_refuse_constant)
    return digest("decision", decision_payload(parsed))


def current_revision(history: Any) -> int:
    """The current decision revision: the HIGHEST `revision` in the history, 1 when there is none.

    ⚠ NEVER THE LENGTH. A history `[1, 2, 4]` (an entry pruned or skipped) is at revision 4; reading
    `len()` would name revision 3, which a consumer would then look for and not find.
    """
    revisions = [
        int(entry["revision"])
        for entry in (history or [])
        if isinstance(entry, Mapping) and isinstance(entry.get("revision"), int)
        and not isinstance(entry.get("revision"), bool)
    ]
    return max(revisions) if revisions else 1


def identity_for(definition: ProbeDefinitionV1, decision_revision: Optional[int]) -> ProbeIdentity:
    """The identity block a build writes. Raises `IdentityInputRefused` on a value it cannot hash."""
    detector, decision = parsed_projection(definition)
    return ProbeIdentity(
        algorithm=IDENTITY_ALGORITHM,
        detector_digest=digest("detector", detector),
        decision_digest=digest("decision", decision),
        decision_revision=decision_revision,
    )


def check_identity(definition: ProbeDefinitionV1) -> None:
    """Refuse a carried `identity` that is not the document's own (FR-5). `decision_revision` is
    not checked: it is miStudio's history position, not derivable from the document."""
    carried = definition.identity
    if carried is None:
        return
    if carried.algorithm not in KNOWN_ALGORITHMS:
        raise IdentityRefused(
            {
                "refused": "identity_algorithm_unknown",
                "algorithm": carried.algorithm,
                "known": sorted(KNOWN_ALGORITHMS),
            }
        )
    detector, decision = parsed_projection(definition)
    for field, payload_kind, payload in (
        ("detector_digest", "detector", detector),
        ("decision_digest", "decision", decision),
    ):
        recomputed = digest(payload_kind, payload)
        if getattr(carried, field) != recomputed:
            raise IdentityRefused(
                {
                    "refused": "identity_mismatch",
                    "field": field,
                    "carried": getattr(carried, field),
                    "recomputed": recomputed,
                    "algorithm": carried.algorithm,
                }
            )


def identity_refusal(exc: ValidationError) -> Optional[Dict[str, Any]]:
    """The structured refusal inside a `ValidationError`, or None when it is some other error."""
    for error in exc.errors():
        inner = (error.get("ctx") or {}).get("error")
        if isinstance(inner, IdentityRefused):
            return dict(inner.detail)
    return None


# ── `python -m src.schemas.probe_identity --bench` (FTDD §11.3) ────────────────


def _synthetic_definition(d_model: int) -> ProbeDefinitionV1:
    """A valid residual-probe document of width `d_model`, built in memory (no database)."""
    import random

    rng = random.Random(4096)
    vector = [{"messages": [{"role": "user", "content": "x"}], "token_ids": [1, 2, 3],
               "token_scores": [0.1, 0.2, 0.3], "score": 0.3, "verdict": False}] * 8
    return ProbeDefinitionV1.model_validate(
        {
            "name": "bench",
            "model": {"hf_id": "org/model", "revision": "0" * 40, "d_model": d_model,
                      "n_layers": 32, "architecture": "Bench"},
            "read": {"layer": 16},
            "scope": "all",
            "basis": "residual",
            "head": {
                # float(float32) values, as the builder's `head.weight.tolist()` produces them.
                "weights": [float(_f32(rng.gauss(0, 0.05))) for _ in range(d_model)],
                "bias": float(_f32(rng.gauss(0, 1))),
                "norm_mean": [float(_f32(rng.gauss(0, 1))) for _ in range(d_model)],
                "norm_std": [float(_f32(abs(rng.gauss(1, 0.2)) + 0.01)) for _ in range(d_model)],
            },
            "aggregation": {"rule": "mean", "params": {}, "streamable": True},
            "decision": {"threshold": 1.25, "target_fpr": 0.01, "realised_fpr": 0.0095,
                         "threshold_source": "calibration_set"},
            "evidence": {"rung": 2, "rung_language": "bench"},
            "test_vectors": {"tolerance": 0.05, "vectors": vector},
        }
    )


def _f32(value: float) -> float:
    import struct

    return struct.unpack("f", struct.pack("f", value))[0]


def _bench(d_model: int, iterations: int) -> Dict[str, Dict[str, float]]:
    definition = _synthetic_definition(d_model)
    timings: Dict[str, List[float]] = {"detector": [], "decision": []}
    for _ in range(iterations):
        for kind in ("detector", "decision"):
            started = time.perf_counter()
            detector, decision = parsed_projection(definition)
            digest(kind, detector if kind == "detector" else decision)
            timings[kind].append((time.perf_counter() - started) * 1000.0)
    out: Dict[str, Dict[str, float]] = {}
    for kind, values in timings.items():
        ordered = sorted(values)
        p95_index = min(len(ordered) - 1, max(0, math.ceil(0.95 * len(ordered)) - 1))
        out[kind] = {"p50_ms": statistics.median(ordered), "p95_ms": ordered[p95_index],
                     "n": float(len(ordered))}
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.schemas.probe_identity")
    parser.add_argument("--bench", action="store_true", help="time both digests (FTDD §11.3)")
    parser.add_argument("--d-model", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args(argv)
    if not args.bench:
        parser.print_help()
        return 2
    result = _bench(args.d_model, max(1, args.iterations))
    for kind, stats in result.items():
        print(
            f"{kind} digest d_model={args.d_model} n={int(stats['n'])} "
            f"p50={stats['p50_ms']:.2f} ms p95={stats['p95_ms']:.2f} ms"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by test_probe_identity.py via subprocess
    raise SystemExit(main())
