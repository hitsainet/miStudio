"""The vendored runner contract, as data (Feature 040 S4; FTDD Appendix C.2, C.3). STANDARD LIBRARY ONLY.

The backend imports this module (the hash rule, the floor, the reasons, the caller's text rule); the runner image
does too. Nothing here reads a setting, a database or the network.

* **One hash rule** (the contract's ``hash_rule``): ``code_sha256`` = SHA-256 of the canonical JSON of the code-rule
  document — keys sorted by Unicode CODE POINT (not UTF-16 code units, so NOT RFC 8785 / `rule_signals.canonical`,
  which the rule-set identity uses), ``,``/``:`` separators, UTF-8 with non-ASCII written as itself, NaN and
  infinity refused. Pinned by the vendored ``hash`` vectors (non-ASCII keys, code point vs UTF-16 order).
* ``bundle_digest`` is the same hash over ``guest.bundle`` without its own ``bundle_digest`` key.
* The caller's half of ``text_rules``: lone surrogates become U+FFFD BEFORE a text is encoded for the runner.
"""
from __future__ import annotations

import hashlib
import json
import re
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Mapping, Sequence, Tuple

CONTRACT_NAME = "midataworks-runner-v1.json"
CASES_NAME = "runner-cases-v1.json"
#: The package copies (the backend tree: `src/schemas/vendored/`; the image: `/app/schemas/vendored/`).
VENDORED = Path(__file__).resolve().parents[1] / "schemas" / "vendored"
CONTRACT_PATH = VENDORED / CONTRACT_NAME
CASES_PATH = VENDORED / CASES_NAME
FORMAT = "midataworks.runner/v1"


class Reason(str, Enum):
    """The contract's eight reasons (``reasons``), in declaration order."""

    RULE_TIMEOUT = "rule_timeout"
    RULE_RESOURCE_LIMIT = "rule_resource_limit"
    RULE_WORKER_UNAVAILABLE = "rule_worker_unavailable"
    RULE_ERROR = "rule_error"
    RULE_LOAD_REFUSED = "rule_load_refused"
    RULE_OUTPUT_INVALID = "rule_output_invalid"
    RULE_INPUT_TOO_LARGE = "rule_input_too_large"
    RULE_NOT_APPROVED = "rule_not_approved"


REASONS: Tuple[str, ...] = tuple(r.value for r in Reason)
#: Caused by the host or the deployment, not by the operator's code: NEVER cached as a verdict (040-A6).
HOST_FAULTS: FrozenSet[str] = frozenset({Reason.RULE_WORKER_UNAVAILABLE.value, Reason.RULE_TIMEOUT.value})

#: Hostile classes whose ESCAPE-form vectors the operator writes (ENH-006 D48): a guest reaching files, the
#: environment, a socket or a process past the prelude (1, 2, 3, 4), the clock or random source (9), and the
#: guest-source forms that replay or reorder a phase marker or forge a reply line (17, 18, 19). Mirrors the
#: producer's ``runner_contract.coverage.PENDING_HOSTILE_CLASSES``: nothing in miStudio reports these classes as
#: covered by a vector. The host defences they would exercise are pinned at host level by
#: ``rule_runner/tests/test_wasi_table.py``.
PENDING_HOSTILE_CLASSES: FrozenSet[int] = frozenset({1, 2, 3, 4, 9, 17, 18, 19})
#: Reasons no guest-source vector can produce: a host/deployment fault, and the approval reason (emitted where the
#: database is, never by the runner).
REASONS_WITHOUT_VECTOR: FrozenSet[str] = frozenset({"rule_worker_unavailable", "rule_not_approved"})


class CanonicalRefused(ValueError):
    """NaN, infinity or a value JSON cannot carry."""


def canonical_json(value: Any) -> bytes:
    """The contract's canonical JSON. Python's ``sort_keys`` orders ``str`` keys by code point."""
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                          allow_nan=False).encode("utf-8")
    except ValueError as exc:
        raise CanonicalRefused(str(exc)) from None


def canonical_document_bytes(document: Mapping[str, Any]) -> bytes:
    """The exact bytes hashed (and sent to the runner) for a code-rule document."""
    return canonical_json(dict(document))


def code_sha256(document: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_document_bytes(document)).hexdigest()


def bundle_digest(bundle: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json({k: v for k, v in bundle.items() if k != "bundle_digest"})).hexdigest()


_LONE_SURROGATE = re.compile("[\ud800-\udfff]")


def replace_lone_surrogates(text: str) -> str:
    """The caller's half of ``text_rules``: every lone surrogate → U+FFFD; nothing else changes."""
    return _LONE_SURROGATE.sub("�", text)


@lru_cache(maxsize=4)
def _read(path: str) -> Dict[str, Any]:
    data: Dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    return data


def load_contract(path: Path = CONTRACT_PATH) -> Dict[str, Any]:
    return _read(str(path))


def load_cases(path: Path = CASES_PATH) -> Dict[str, Any]:
    return _read(str(path))


def floor(contract: Mapping[str, Any] | None = None) -> Dict[str, int]:
    return {k: int(v) for k, v in (contract or load_contract())["limits_floor"].items()}


def request_limits(contract: Mapping[str, Any] | None = None) -> Tuple[str, ...]:
    return tuple((contract or load_contract())["request_limits"])


def identity(contract: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    """What a code-rule verdict is computed under, from the contract alone (the image is built from it)."""
    c = contract or load_contract()
    return {"format": c["format"], "contract_version": c["contract_version"],
            "bundle_digest": c["guest"]["bundle"]["bundle_digest"],
            "snapshot_sha256": c["guest"]["bundle"]["snapshot"]["sha256"],
            "runtime": {"family": c["runtime"]["family"], "major": c["runtime"]["major"]}}


def missing_coverage(contract: Mapping[str, Any], cases: Mapping[str, Any]) -> Dict[str, List[Any]]:
    """What the contract requires but the cases file does not cover (empty == full). Fails closed on an empty
    contract list. A class in `PENDING_HOSTILE_CLASSES` without a vector is NOT a gap — and is never reported as
    covered either (`covered_hostile_classes`)."""
    def reasons_seen() -> set:
        seen = set()
        for section in ("conformance", "hostile", "limits"):
            for case in cases.get(section, []):
                if "reason" in case.get("expect", {}):
                    seen.add(case["expect"]["reason"])
        return seen

    reasons: Sequence[str] = contract.get("reasons", [])
    modules: Sequence[str] = contract.get("stdlib_allowed", [])
    floor_names = list(contract.get("limits_floor", {}))
    hostile = [h["number"] for h in contract.get("hostile_classes", [])]
    present_r = reasons_seen()
    present_m = {c["module"] for c in cases.get("imports", []) if "module" in c}
    present_l = {c["limit"] for c in cases.get("limits", []) if "limit" in c}
    present_h = covered_hostile_classes(cases)
    out: Dict[str, List[Any]] = {
        "reasons": sorted(r for r in reasons if r not in present_r and r not in REASONS_WITHOUT_VECTOR),
        "modules": sorted(m for m in modules if m not in present_m),
        "limits": sorted(k for k in floor_names if k not in present_l),
        "hostile": sorted(n for n in hostile if n not in present_h and n not in PENDING_HOSTILE_CLASSES),
    }
    if not reasons:
        out["reasons"] = ["<contract has no reasons>"]
    if not hostile:
        out["hostile"] = ["<contract has no hostile classes>"]
    return {k: v for k, v in out.items() if v}


def covered_hostile_classes(cases: Mapping[str, Any]) -> FrozenSet[int]:
    """Classes a vector in the file covers. A pending class is never in this set unless a vector arrives."""
    return frozenset(c["number"] for c in cases.get("hostile", []) if "number" in c)
