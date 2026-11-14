"""A DatasetDict's len() is its SPLIT COUNT, and two callers believed otherwise.

MEASURED 2026-09-11. `teknium/OpenHermes-2.5` was downloaded with no `split=`,
so `load_dataset` returned a DatasetDict. 1.6 GB landed on disk, the `train`
split held **1,001,551 rows**, and the database recorded `num_samples = 1`.
Nothing raised: the row went READY with a plausible small integer.

The second site was worse and had no symptom at all in the DB.
`tokenize_dataset_task` emitted its progress denominator as `len(dataset)` and
then unwrapped the DatasetDict *fourteen lines later*, so a tokenization of any
such dataset ran a progress bar of `n/1` over a million rows.

Both now call `services/dataset_shape`, and the AST assertions below pin the
CALL rather than scraping for a name — a substring search would match the
comments that explain this, which is how five guards in this arc were satisfied
by the wrong occurrence.
"""

import ast
import inspect

import pytest

from src.services.dataset_shape import count_rows, primary_split, size_in_bytes


class _Split:
    """Stands in for a `datasets.Dataset`: has __len__, has no .keys()."""

    def __init__(self, n, size=None):
        self._n = n
        if size is not None:
            self.size_in_bytes = size

    def __len__(self):
        return self._n


class _Dict(dict):
    """Stands in for a `datasets.DatasetDict`: a mapping of name -> split.

    Subclassing dict is the point — its `len()` is the split count, exactly
    like the real thing, so the fixture cannot agree with the fix by
    construction.
    """


class TestCountRowsNeverReturnsTheSplitCount:

    def test_the_openhermes_case(self):
        """The measured production failure, reproduced exactly."""
        d = _Dict(train=_Split(1_001_551))

        assert len(d) == 1, "fixture must reproduce the trap, or it proves nothing"
        assert count_rows(d) == 1_001_551

    def test_a_plain_dataset_is_counted_directly(self):
        assert count_rows(_Split(446_762)) == 446_762

    def test_train_wins_over_other_splits(self):
        d = _Dict(test=_Split(7), train=_Split(1_001_551), validation=_Split(9))
        assert count_rows(d) == 1_001_551

    def test_without_a_train_split_the_first_is_used(self):
        d = _Dict()
        d["train_sft"] = _Split(207_865)
        d["test_sft"] = _Split(23_110)
        assert count_rows(d) == 207_865

    def test_an_uncountable_object_returns_none_not_a_guess(self):
        assert count_rows(object()) is None

    def test_an_empty_split_mapping_does_not_raise(self):
        assert count_rows(_Dict()) is None


class TestPrimarySplit:

    def test_a_plain_dataset_passes_through_unchanged(self):
        """Callers apply this unconditionally, so it must be a no-op here."""
        split = _Split(10)
        assert primary_split(split) is split

    def test_it_selects_train(self):
        train = _Split(3)
        d = _Dict(test=_Split(1), train=train)
        assert primary_split(d) is train


class TestSizeInBytes:

    def test_it_sums_across_splits(self):
        d = _Dict(train=_Split(3, size=100), test=_Split(1, size=25))
        assert size_in_bytes(d) == 125

    def test_a_plain_dataset_reports_its_own(self):
        assert size_in_bytes(_Split(3, size=99)) == 99

    def test_absent_everywhere_is_none_not_zero(self):
        """0 bytes and "unknown" are different claims; OpenHermes recorded None
        beside 1.6 GB and that at least did not assert a falsehood."""
        assert size_in_bytes(_Dict(train=_Split(3))) is None
        assert size_in_bytes(_Split(3)) is None


def _calls_in(func_name, module):
    """Every function called inside one function of `module`, by AST."""
    tree = ast.parse(inspect.getsource(module))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            out = set()
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call):
                    if isinstance(sub.func, ast.Name):
                        out.add(sub.func.id)
                    elif isinstance(sub.func, ast.Attribute):
                        out.add(sub.func.attr)
            return out
    raise AssertionError(f"{func_name} not found in {module.__name__}")


class TestBothCallersActuallyUseIt:
    """Reachability. Deleting either call must fail something."""

    def test_the_download_records_rows_through_count_rows(self):
        from src.workers import dataset_tasks

        assert "count_rows" in _calls_in("download_dataset_task", dataset_tasks)

    def test_the_download_records_size_through_size_in_bytes(self):
        from src.workers import dataset_tasks

        assert "size_in_bytes" in _calls_in("download_dataset_task", dataset_tasks)

    def test_tokenization_counts_and_unwraps_through_the_same_module(self):
        from src.workers import dataset_tasks

        calls = _calls_in("tokenize_dataset_task", dataset_tasks)
        assert "count_rows" in calls, "the progress denominator must not be len()"
        assert "primary_split" in calls, "the unwrap must share one definition"

    def test_neither_caller_still_takes_a_bare_len_of_the_dataset(self):
        """NEGATIVE CONTROL for the two tests above. `len(dataset)` is the
        defect verbatim; reintroducing it beside a retained call to
        `count_rows` would pass every assertion above."""
        from src.workers import dataset_tasks

        tree = ast.parse(inspect.getsource(dataset_tasks))
        offenders = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            if node.name not in ("download_dataset_task", "tokenize_dataset_task"):
                continue
            for sub in ast.walk(node):
                if (
                    isinstance(sub, ast.Call)
                    and isinstance(sub.func, ast.Name)
                    and sub.func.id == "len"
                    and len(sub.args) == 1
                    and isinstance(sub.args[0], ast.Name)
                    and sub.args[0].id == "dataset"
                ):
                    offenders.append((node.name, sub.lineno))

        assert offenders == [], (
            f"len(dataset) is the split count on a DatasetDict; found at {offenders}"
        )
