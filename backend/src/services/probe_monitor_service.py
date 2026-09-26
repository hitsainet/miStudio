"""Orchestration and DB access for probe monitors (032).

This module owns the decisions the API and the tasks share, so neither can drift
from the other: how a probe dataset's rows are located on disk, what makes a view
scoreable, and which refusals are 422s.

⚠ THE COUNTS ARE COMPUTED ONCE, AT CREATION, AND STORED. Recomputing them at read
time would let a re-downloaded dataset silently disagree with the numbers a run was
submitted against — and `excluded` / `unparseable` are precisely the numbers a
refusal cites. FR-1 wants the mapping auditable after the fact, which means the
mapping's OUTPUT has to be recorded, not just its input.

⚠ AND THE 20-PER-CLASS FLOOR IS ENFORCED HERE, NOT AT THE ROUTE. It needs the data,
so the route cannot decide it; and it must be decided at CREATION rather than at run
submission, because a view that can never be scored is not a view worth storing.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..core.config import settings
from .probe_monitor_inputs import (
    BuildResult,
    build_examples,
    class_counts,
)

logger = logging.getLogger(__name__)

#: Matches `probe_monitor_metrics.MIN_PER_CLASS`. Imported rather than re-declared
#: would be better still, and it is — see `_floor()`; this constant exists only so a
#: reader of this file sees the number.
MIN_PER_CLASS = 20


def _floor() -> int:
    """The one definition of the floor, read from the metrics module.

    A second constant would drift: a view accepted at 20 while the metric refuses
    below 25 produces a dataset that can be created and never scored, and the
    disagreement would show up as an empty report rather than as an error.
    """
    from .probe_monitor_metrics import MIN_PER_CLASS as metric_floor

    return metric_floor


class ProbeDatasetRefused(ValueError):
    """A view that cannot be scored. Carries the counts, because a refusal that does
    not say how far short it fell is a refusal nobody can act on."""

    def __init__(self, message: str, counts: Dict[str, int]):
        super().__init__(message)
        self.counts = counts


@dataclass
class ResolvedColumns:
    input_column: str
    label_column: str
    pair_column: Optional[str] = None


def resolve_dataset_path(raw_path: Optional[str]) -> Path:
    """Where a downloaded dataset's Arrow tree actually is.

    TWO LAYOUTS EXIST ON THIS ESTATE and both are load-bearing. `raw_path` is what
    `save_to_disk` wrote; if it is absent the data may still be in HuggingFace's own
    cache tree, whose directory replaces the FIRST underscore of the repo name with
    three. The samples endpoint already carries this fallback, and a probe dataset
    that cannot find rows a user can see in the UI is an incoherent failure.
    """
    if not raw_path:
        raise FileNotFoundError(
            "this dataset has no raw_path, so its rows are not on disk; download it "
            "before building a probe dataset from it"
        )
    resolved = settings.resolve_data_path(raw_path)
    if resolved.exists():
        return resolved
    # HuggingFace cache layout: `org_name` → `org___name`.
    alternative = resolved.parent / resolved.name.replace("_", "___", 1)
    if alternative.exists():
        logger.info(
            "probe_monitor: %s is absent, reading the HuggingFace cache tree at %s",
            resolved, alternative,
        )
        return alternative
    raise FileNotFoundError(
        f"neither {resolved} nor {alternative} exists; the dataset's files are gone "
        f"even though its row says it was downloaded"
    )


def load_columns(
    path: Path,
    columns: ResolvedColumns,
    *,
    split: Optional[str] = None,
    limit: Optional[int] = None,
) -> Tuple[List[Any], List[Any], Optional[List[Any]], int]:
    """Read the three columns a probe dataset needs. Returns (inputs, labels, pairs, total).

    Reads COLUMNS, not rows: `dataset[column]` on an Arrow table is a column read, so
    a 200k-row dataset does not become 200k Python dicts. This estate has already
    OOMed once by materialising `tokenized_dataset["input_ids"]` as Python lists.
    """
    from datasets import load_from_disk

    data = load_from_disk(str(path))
    # A DatasetDict needs a split; `len()` of one is its SPLIT COUNT, which was read
    # as a row count here before and reported 1 sample for a million rows.
    if hasattr(data, "keys") and not hasattr(data, "column_names"):
        available = list(data.keys())
        chosen = split if split in available else available[0]
        if split and split not in available:
            raise ValueError(
                f"split {split!r} is not in this dataset; available: {available}"
            )
        data = data[chosen]

    missing = [
        name
        for name in (columns.input_column, columns.label_column, columns.pair_column)
        if name and name not in data.column_names
    ]
    if missing:
        raise ValueError(
            f"columns {missing} are not in this dataset; available: "
            f"{sorted(data.column_names)}"
        )

    total = len(data)
    if limit is not None and limit < total:
        data = data.select(range(limit))

    inputs = list(data[columns.input_column])
    labels = list(data[columns.label_column])
    pairs = list(data[columns.pair_column]) if columns.pair_column else None
    return inputs, labels, pairs, total


def build_view(
    inputs: Sequence[Any],
    labels: Sequence[Any],
    label_mapping: Dict[str, str],
    *,
    keyword_filter: Optional[Dict[str, Any]] = None,
    pair_values: Optional[Sequence[Any]] = None,
    role: str = "eval",
) -> BuildResult:
    """Map, filter and count — then REFUSE a view that cannot be scored.

    A calibration set is held to a different standard on purpose: it supplies
    negatives for the FPR threshold and has no positives by definition (BR-003), so
    demanding both classes of it would force a caller to invent a label.
    """
    result = build_examples(
        inputs,
        labels,
        label_mapping,
        keyword_filter=keyword_filter,
        pair_values=pair_values,
    )
    positives, negatives = class_counts(result.examples)
    floor = _floor()

    if role == "calibration":
        if negatives < floor:
            raise ProbeDatasetRefused(
                f"a calibration set needs at least {floor} negatives to place a "
                f"threshold; this mapping produced {negatives}. Counts: "
                f"{result.counts.as_dict()}",
                result.counts.as_dict(),
            )
        return result

    short = [
        name
        for name, count in (("positive", positives), ("negative", negatives))
        if count < floor
    ]
    if short:
        raise ProbeDatasetRefused(
            f"too few {' and '.join(short)} examples: {positives} positive and "
            f"{negatives} negative, and {floor} of each is the floor below which an "
            f"AUROC is noise that reads like a measurement. Counts: "
            f"{result.counts.as_dict()}",
            result.counts.as_dict(),
        )
    return result


def describe_counts(result: BuildResult) -> Dict[str, Any]:
    """What goes in `probe_monitor_datasets.counts`, plus the input-shape histogram.

    The histogram is stored beside the counts because it answers a question the
    counts cannot: a corpus that is 90% plain text and 10% JSON-string chat is one
    where a template change touches a tenth of the rows, and `roles_guessed` says how
    much of it cannot support a role-scoped probe at all.
    """
    payload = result.counts.as_dict()
    payload["kinds"] = dict(result.kinds)
    return payload
