"""Which evaluation views are too large to score whole (ENH-006 G15).

An evaluation scores every row of a view unless the caller names a seeded sample, and a view can be a
corpus: the everyday-text view `everyday-text-openwebtext` holds 1,000,000 rows. Listed among a run's
evaluation sets on 2026-11-03, it was scored whole — 150,730 batches with an ETA of 347 hours — and the
worker grew to ~95 GB until the kernel killed it. Nothing refused it, because the only size guard
(Feature 041 PT-11) looked at the calibration role alone.

The size is the view's REGISTERED count (`counts.positive + counts.negative`, what an evaluation
scores), read at submit, so the refusal costs no load of the data. A view with no recorded counts is
not refused: the count is unknown, never guessed.
"""
from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

from ..schemas.probe_monitor import MAX_UNSAMPLED_EVAL_ROWS


def scored_rows(view: Any) -> Optional[int]:
    """Rows an evaluation of `view` scores whole, from its registered counts; None when not recorded."""
    counts = getattr(view, "counts", None) or {}
    if "positive" not in counts and "negative" not in counts:
        return None
    return int(counts.get("positive") or 0) + int(counts.get("negative") or 0)


def too_large_to_score_whole(
    views: Sequence[Any], sample_rows: Optional[int], limit: int = MAX_UNSAMPLED_EVAL_ROWS
) -> List[Tuple[Any, int]]:
    """Each view an evaluation would score whole although it holds more than `limit` rows.

    Empty whenever `sample_rows` is set: a sample bounds every view, and a smaller view is scored whole.
    """
    if sample_rows is not None:
        return []
    found = []
    for view in views:
        rows = scored_rows(view)
        if rows is not None and rows > limit:
            found.append((view, rows))
    return found


def refusal_detail(found: Sequence[Tuple[Any, int]], knob: str) -> str:
    """The sentence a person reads: which views, how big, and the argument that fixes it."""
    named = "; ".join(f"{view.id} ({getattr(view, 'name', '')}, role {view.role}) holds {rows:,} rows"
                      for view, rows in found)
    return (
        f"{named}. An evaluation set above {MAX_UNSAMPLED_EVAL_ROWS:,} rows is never scored whole: "
        f"set {knob} (100 to 200,000) to score a seeded sample of each view. The seed is the view and "
        f"the size, so every probe evaluated at that size scores the same rows, and a smaller view is "
        f"still scored whole. Rung 2 needs at least 1,000 everyday-text rows."
    )
