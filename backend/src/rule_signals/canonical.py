"""miDataworks' rule-set identity (Feature 040 FR-27 provenance, FTDD §4.3; 016 spec D38-4).

The identity is the lower-case hex SHA-256 of the RFC 8785 (JSON Canonicalization Scheme) bytes of the
document AS RECEIVED — the parsed JSON values, before any default is applied, never a re-serialisation
of a model. An ABSENT list and an EMPTY list are different documents with different digests. An integer
outside ±(2^53 − 1) is refused (`unsafe_integer`). Pinned by the vendored `rule_set_identity` cases of
`rule-signal-cases-v1.json` (``tests/unit/test_rule_signals_conformance.py``).

⚠ NOT ``json.dumps(sort_keys=True)``: that sorts keys by code point, not UTF-16 code unit, writes ``1.0``
for an integral float and escapes non-ASCII, so it hashes a different byte string for the same document
(the FTASKS 18.3 control).
"""
from __future__ import annotations

import hashlib
from typing import Any

import rfc8785


class CanonicalRefused(ValueError):
    """A value RFC 8785 cannot write (NaN, ±inf, an integer outside ±(2^53 − 1))."""


def canonical_bytes(document: Any) -> bytes:
    try:
        return rfc8785.dumps(document)
    except rfc8785.CanonicalizationError as exc:
        raise CanonicalRefused(str(exc)) from exc


def identity(document: Any) -> str:
    return hashlib.sha256(canonical_bytes(document)).hexdigest()
