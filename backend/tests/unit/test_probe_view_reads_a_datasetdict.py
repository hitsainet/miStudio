"""A probe view can be created over a multi-split dataset.

⚠ IT COULD NOT. `load_columns` guarded the DatasetDict branch with

    if hasattr(data, "keys") and not hasattr(data, "column_names"):

and a `DatasetDict` in datasets 2.21 HAS `column_names` — it returns `{split: [columns]}`. So the
guard was False for precisely the type it existed to catch: the split was never applied, and the
column check then compared the requested columns against the dict's KEYS. Creating a view over
any multi-split dataset failed with

    columns ['inputs', 'labels', 'pair_id'] are not in this dataset; available: ['eval', 'train']

naming the splits as though they were columns. Found 2026-09-29 registering a generated corpus
saved as train/eval splits.

The lesson is in the guard's shape, not its logic: duck-typing a library type by the ABSENCE of an
attribute breaks silently the moment the library adds one, and nothing about the failure points at
the guard. The type is importable, so it is named.

⚠ These tests build a REAL `DatasetDict` and round-trip it through `save_to_disk`. A stub with a
`keys()` method and no `column_names` would pass against the broken guard — the fixture would
supply the very property whose absence was wrongly assumed.
"""

from __future__ import annotations

import pytest

from src.services.probe_monitor_service import ResolvedColumns, load_columns

datasets = pytest.importorskip("datasets")

COLUMNS = ResolvedColumns(input_column="inputs", label_column="labels", pair_column="pair_id")


def _rows(tag: str, n: int):
    return [
        {
            "inputs": f"{tag} message {i}",
            "labels": "high-stakes" if i % 2 else "low-stakes",
            "pair_id": f"{tag}-pair-{i // 2}",
        }
        for i in range(n)
    ]


@pytest.fixture
def multi_split(tmp_path):
    """A real DatasetDict on disk — the shape that broke."""
    path = tmp_path / "corpus"
    datasets.DatasetDict(
        {
            "train": datasets.Dataset.from_list(_rows("train", 8)),
            "eval": datasets.Dataset.from_list(_rows("eval", 4)),
        }
    ).save_to_disk(str(path))
    return path


@pytest.fixture
def single_split(tmp_path):
    path = tmp_path / "flat"
    datasets.Dataset.from_list(_rows("flat", 6)).save_to_disk(str(path))
    return path


class TestTheGuardMatchesTheRealType:
    def test_a_real_DatasetDict_still_exposes_column_names(self, multi_split):
        """Pins the library behaviour the old guard assumed away. If a future datasets release
        drops `column_names` from DatasetDict this goes red, and the comment in `load_columns`
        stops being true."""
        loaded = datasets.load_from_disk(str(multi_split))
        assert isinstance(loaded, datasets.DatasetDict)
        assert hasattr(loaded, "column_names"), (
            "DatasetDict no longer exposes column_names; the old hasattr guard would now "
            "work by accident and this test's premise needs revisiting"
        )
        assert isinstance(loaded.column_names, dict)


class TestSplitSelection:
    def test_the_requested_split_is_read(self, multi_split):
        inputs, labels, pairs, total = load_columns(multi_split, COLUMNS, split="train")
        assert total == 8, f"read {total} rows; the train split has 8"
        assert all(t.startswith("train ") for t in inputs)
        assert len(labels) == 8 and len(pairs) == 8

    def test_the_other_split_is_read(self, multi_split):
        """⚠ Specificity. Always returning the FIRST split would satisfy the test above while
        making `view.split` decorative."""
        inputs, _labels, _pairs, total = load_columns(multi_split, COLUMNS, split="eval")
        assert total == 4
        assert all(t.startswith("eval ") for t in inputs)

    def test_a_single_split_dataset_is_unaffected(self, single_split):
        """The published corpora are one directory per split, so this is the common path."""
        inputs, _labels, _pairs, total = load_columns(single_split, COLUMNS)
        assert total == 6
        assert all(t.startswith("flat ") for t in inputs)

    def test_an_unknown_split_is_refused_by_name(self, multi_split):
        """Not silently the first one: a typo'd split would otherwise train on the wrong rows
        and nothing would say so."""
        with pytest.raises(ValueError, match="nope"):
            load_columns(multi_split, COLUMNS, split="nope")

    def test_the_columns_are_checked_against_the_SPLIT_not_the_dict(self, multi_split):
        """The observable symptom of the old bug: the refusal named the splits as though they
        were the available columns."""
        bad = ResolvedColumns(input_column="not_a_column", label_column="labels", pair_column=None)
        with pytest.raises(ValueError) as exc:
            load_columns(multi_split, bad, split="train")
        message = str(exc.value)
        assert "not_a_column" in message
        assert "inputs" in message, (
            f"the refusal does not list the split's real columns, so it is still reading the "
            f"DatasetDict rather than the split: {message}"
        )
        assert "'eval'" not in message, (
            f"the refusal offers split names as available columns — the old bug: {message}"
        )
