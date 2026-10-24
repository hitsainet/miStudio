"""The blend report's canonical bytes (Feature 041 FR-15; T14, TI9, PT-12).

`report.json` is PLAIN JCS (RFC 8785, the `rfc8785` library 039 added) of the report object after 040's
rounding rule (float64 rounded to 6 decimals, `detector_identity.round6`; ints and bools untouched) — never
`json.dumps` (its key order and float form are not canonical) and never 039's domain-separated
`canonical_bytes(kind, payload)` wrapper. Two evaluations of one request therefore write the same bytes.

The format markers are internal, NOT contracts: no file in `docs/schemas/` (T14; allowlist test).
"""
from __future__ import annotations

from typing import Any

from ..detector_identity import jcs_bytes, round6

FORMAT = "mistudio.blend-report/1"
SUGGESTED_FORMAT = "mistudio.suggested-blend/1"
ENGINE_VERSION = "1"

#: The fixed sentence every report and suggested-blend record carries (FTDD §6.4).
STATEMENT = (
    "A blend has no rung of its own; each armed probe is judged by its own rung. This report is evidence "
    "for an application's blend; it is not exported to or served by miLLM."
)


def round_floats(value: Any) -> Any:
    """Every float rounded by 040's rule, recursively; ints and bools pass through (`isinstance(True, int)`)."""
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        return round6(value)
    if isinstance(value, dict):
        return {str(k): round_floats(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [round_floats(v) for v in value]
    if hasattr(value, "item"):                       # a NumPy scalar
        return round_floats(value.item())
    raise TypeError(f"{type(value).__name__} cannot be written into a blend report")


def canonical_bytes(obj: Any) -> bytes:
    """Round, then plain JCS."""
    return jcs_bytes(round_floats(obj))
