"""How to DESCRIBE the precision an artifact was read at — recorded, inferred, or unknown.

⚠ NOTHING HERE WRITES. A NULL `model_dtype` stays NULL in the database: it means "not recorded".
This module only labels it, at read time, so an operator looking at an SAE or a probe run can see
what it was fitted on without the database claiming a fact nobody recorded.

THE INFERENCE, and exactly when it holds. Before 2026-10-03 every miStudio load path feeding an
extraction or a probe run cast 16-bit rows to float16 (hardcoded; `ml/native_dtype.py` replaced
it), and FP32 rows were mapped to float16 too. So a NULL is inferred float16 only when BOTH:

  * the artifact was created before `PRECISION_RECORDED_SINCE`. After it, a NULL is a row the new
    code has not finished (a queued training, a failed extraction) or one the OLD code wrote before
    the deploy reached it — either way unknown, never inferred (review round 1, M4). The constant
    must not be later than the deploy; earlier only widens "unknown", which is safe.
  * it is not an ON-THE-FLY training. Those loaded through the shared loader, which mapped an FP32
    row to float32 — so for them "float16" is not a deduction.

⚠ J-LENS IS NEVER INFERRED: its readout loaded with `dtype="auto"`, the checkpoint's own precision.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Optional, Union

#: Artifact kinds whose pre-cutover loads were hardcoded float16.
INFERABLE_KINDS = frozenset({"extraction", "training", "probe_run", "sae"})

#: Not later than the deploy of the native-dtype change. See the module docstring.
#: 12:00 UTC on 2026-10-03 — before the first commit of the change existed (c0349451, 12:32 UTC),
#: so no artifact the new code made can be older. Round 3 caught the first value, 16:00 UTC, which
#: was in the FUTURE when set (the clock read 13:25 UTC): a run the new code created before 16:00
#: with a NULL precision would have been labelled "inferred float16".
PRECISION_RECORDED_SINCE = datetime(2026, 10, 3, 12, 0, tzinfo=timezone.utc)

INFERRED_NOTE = (
    "inferred: read before precision was recorded (2026-10-03), when this kind of load ran float16"
)


def _as_utc(value: Union[datetime, str, None]) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if value.tzinfo is None:
        # Naive timestamps here are server-local; treated as UTC, which only moves the boundary
        # by the server's offset — and the boundary is deliberately before the deploy.
        value = value.replace(tzinfo=timezone.utc)
    return value


def describe_dtype(
    recorded: Optional[str],
    kind: str,
    created_at: Union[datetime, str, None] = None,
    *,
    on_the_fly: bool = False,
) -> Dict[str, Optional[str]]:
    """`{"value", "source", "note"}` — source is "recorded", "inferred" or "unknown"."""
    if recorded:
        return {"value": recorded, "source": "recorded", "note": None}
    created = _as_utc(created_at)
    if (
        kind in INFERABLE_KINDS
        and not on_the_fly
        and created is not None
        and created < PRECISION_RECORDED_SINCE
    ):
        return {"value": "float16", "source": "inferred", "note": INFERRED_NOTE}
    return {"value": None, "source": "unknown", "note": None}
