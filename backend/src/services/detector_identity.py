"""Canonical bytes and identities for the behaviour-detector feature (Feature 040 FR-18, FR-25; TI-4).

`jcs_bytes` is RFC 8785 (JSON Canonicalization Scheme) through the `rfc8785` package — the library 039
added, one algorithm for the suite (FTID D-5). Two identities are built on it:

* a rule set's identity — the lower-case hex SHA-256 of the JCS bytes of the document AS RECEIVED
  (miDataworks 016, spec D38-4; `rule_set_identity`);
* a success report's identity (S2) — the SHA-256 of the JCS bytes of every input that can move a figure,
  with `REPORT_ALGORITHM` among them (`identity`).

⚠ PARSED JSON VALUES, NEVER A PYDANTIC DUMP: pydantic writes ``1.0`` and ``1e-07`` where RFC 8785 writes
``1`` and ``1e-7`` (039's lesson). Callers pass what they parsed.
"""
from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping

import rfc8785

#: The success report's algorithm version. A change of how any figure is computed bumps it, so every stored
#: report reads `stale` instead of silently meaning something else. 2 (040 S3): the `rules_*` forms and
#: `rule_*` subsets are judged (they read `engine_not_available` under 1).
REPORT_ALGORITHM = 2


class CanonicalInputRefused(ValueError):
    """A value RFC 8785 cannot represent (NaN, ±inf, an integer outside ±(2^53 − 1))."""


def jcs_bytes(value: Any) -> bytes:
    try:
        return rfc8785.dumps(value)
    except rfc8785.CanonicalizationError as exc:
        raise CanonicalInputRefused(str(exc)) from exc


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def rule_set_identity(document: Mapping[str, Any]) -> str:
    """016's rule-set identity: SHA-256 of the JCS bytes of the document as received (D38-4)."""
    return sha256_hex(jcs_bytes(document))


def identity(inputs: Mapping[str, Any]) -> str:
    """SHA-256 over the JCS bytes of every input that can move a figure (040-T17)."""
    return sha256_hex(jcs_bytes(dict(inputs)))


def round6(x: Any) -> Any:
    """Figures are float64, rounded to 6 decimals before JCS (FTID §7.3); None and ints pass through."""
    if x is None or isinstance(x, bool) or isinstance(x, int):
        return x
    value = float(x)
    if not math.isfinite(value):
        raise CanonicalInputRefused(f"non-finite figure {value!r}")
    rounded = round(value, 6)
    return 0.0 if rounded == 0 else rounded
