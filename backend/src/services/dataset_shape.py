"""How many rows is this dataset, really?

WHY THIS MODULE EXISTS. `load_dataset(repo)` with no `split=` returns a
**DatasetDict**, and `len()` of a DatasetDict is the number of SPLITS. So
`teknium/OpenHermes-2.5` downloaded 1.6 GB, landed 1,001,551 rows in its `train`
split, and was recorded in the database as `num_samples = 1`.

Nothing raised. The row is READY, the bytes are on disk, and the count is a
plausible-looking small integer — the same shape of failure as the rest of this
arc: a number that is silently wrong rather than absent.

The second site was worse. `tokenize_dataset_task` computed its progress
denominator as `len(dataset)` and only unwrapped the DatasetDict *fourteen lines
later*, so tokenizing any such dataset reported `total_samples=1` for the whole
run — a progress bar over a million rows divided by one.

Both callers now go through here, and `primary_split` is the single definition of
"which split do we mean" so the two cannot drift apart.
"""

from __future__ import annotations

from typing import Any, Optional

__all__ = ["primary_split", "count_rows", "size_in_bytes"]


def _is_split_mapping(dataset: Any) -> bool:
    """A DatasetDict without importing `datasets` at module import time.

    `dataset_tasks` is explicit that `datasets` must not be imported at module
    level (it patches tqdm and the Hub timeouts first), so this duck-types
    instead: a split mapping has `.keys()` and yields datasets, a Dataset does
    not have `.keys()` at all.
    """
    keys = getattr(dataset, "keys", None)
    if keys is None or not callable(keys):
        return False
    # `Dataset` has no `.keys()`; a plain dict of splits and a DatasetDict both
    # do. Requiring `.items()` too keeps a stray object with a `keys` attribute
    # from being mistaken for one.
    return callable(getattr(dataset, "items", None))


def primary_split(dataset: Any) -> Any:
    """The split a caller means when it says "the data".

    `train` when present, otherwise the first split in iteration order. A plain
    Dataset is returned unchanged, so callers may apply this unconditionally.
    A split mapping holding NO splits yields None — there is no data to name,
    and returning the empty mapping would make `len()` say 0 rows, which is
    the split count wearing a different hat.

    This mirrors what `tokenize_dataset_task` already did inline. It is here so
    that the row count and the tokenizer cannot disagree about which split they
    are describing — they disagreed by construction before, because only one of
    them unwrapped at all.
    """
    if not _is_split_mapping(dataset):
        return dataset
    if "train" in dataset:
        return dataset["train"]
    try:
        first = next(iter(dataset.keys()))
    except StopIteration:
        # No splits at all. Returning the mapping would hand back something
        # whose len() is 0 — counting splits again, which is the whole bug.
        return None
    return dataset[first]


def count_rows(dataset: Any) -> Optional[int]:
    """Rows in the dataset — never the number of splits.

    Returns None when the object cannot be counted at all, which is honest;
    the previous code's `hasattr(dataset, "__len__")` guard made a DatasetDict
    look countable, which is how 1,001,551 became 1.
    """
    target = primary_split(dataset)
    if target is None or not hasattr(target, "__len__"):
        return None
    try:
        return len(target)
    except TypeError:
        return None


def size_in_bytes(dataset: Any) -> Optional[int]:
    """On-disk size, summed across splits when there are several.

    `DatasetDict.size_in_bytes` is not reliably present, which is why the
    OpenHermes row recorded `size_bytes = None` beside 1.6 GB of data.
    """
    if _is_split_mapping(dataset):
        total = 0
        seen = False
        for split in dataset.values():
            value = getattr(split, "size_in_bytes", None)
            if value is not None:
                total += value
                seen = True
        return total if seen else None
    return getattr(dataset, "size_in_bytes", None)
