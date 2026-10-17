"""Two configs of one repo are two datasets (032 FR-2).

MUTATION CONTROLS (each verified to fail the suite; recorded in the review):
  M73  `raw_dataset_dirname` ignores `config`        → distinct-path test fails
  M74  the 409 compares `repo_id` alone              → second-config test fails
  M75  `view_key` stops normalising `""` to None     → blank-field test fails
  M76  the download task builds the path inline again → the AST call test fails
  M77  `_is_only_row_for_repo` returns True on error  → fail-closed test fails
  M78  the cancel path adds the HF cache unconditionally → shared-cache test fails
  M79  `get_dataset_by_repo_id` back to scalar_one_or_none → multi-row test fails

⚠ WHAT THIS FEATURE BROKE ON THE WAY IN, AND WHY IT IS TESTED HERE.

1. `download_dataset_task` deletes HuggingFace's arrow tree when a download is
   cancelled, and the comment justifying that says in as many words: "one Dataset
   row per repo_id is enforced at `datasets.py`'s 409 — which is what makes deleting
   them safe". FR-2 removes that invariant. The tree is keyed on the REPO and shared
   by every config, so cancelling config B would delete config A's completed data.

2. `get_dataset_by_repo_id` used `scalar_one_or_none()`, which RAISES
   `MultipleResultsFound` the moment a repo has two rows — a 500 for data FR-2 makes
   legal.

Neither is visible in the diff of the path change. Both are the actual cost of it.
"""
import ast
import inspect
import textwrap
from pathlib import Path

import pytest

from src.services.dataset_view_identity import (
    display_name_for,
    hf_cache_dirname,
    raw_dataset_dirname,
    view_key,
    view_key_of_metadata,
)


class TestTheRawPathCarriesTheView:
    def test_plain_repo_is_unchanged_from_before_FR2(self):
        """Backward compatibility by construction: every row already on disk keeps
        its path, so nothing needs migrating and nothing silently 404s."""
        assert raw_dataset_dirname("Arrrlex/models-under-pressure") == (
            "Arrrlex_models-under-pressure"
        )
        assert raw_dataset_dirname("a/b") == "a_b" == "a/b".replace("/", "_")

    def test_two_configs_of_one_repo_get_distinct_paths(self):
        training = raw_dataset_dirname("Arrrlex/models-under-pressure", "training")
        balanced = raw_dataset_dirname("Arrrlex/models-under-pressure", "toolace_balanced")
        assert training != balanced
        assert training.endswith("__training")
        assert balanced.endswith("__toolace_balanced")

    def test_two_splits_of_one_config_get_distinct_paths(self):
        a = raw_dataset_dirname("r/n", "c", "train")
        b = raw_dataset_dirname("r/n", "c", "test")
        assert a != b

    def test_the_separator_is_unambiguous(self):
        """`org_name` already contains one underscore, so a single-underscore join
        makes `a_b_train` ambiguous between repo `a/b` split `train` and repo
        `a/b_train`. The double underscore cannot be produced by the repo part."""
        assert raw_dataset_dirname("a/b", None, "train") == "a_b__train"
        assert raw_dataset_dirname("a/b_train") == "a_b_train"
        assert raw_dataset_dirname("a/b", None, "train") != raw_dataset_dirname("a/b_train")

    def test_a_traversing_config_cannot_escape_the_datasets_directory(self):
        name = raw_dataset_dirname("r/n", "../../etc/passwd")
        assert "/" not in name and ".." not in name.split("__")[-1].replace(".", "")
        assert name.startswith("r_n__")

    def test_a_pathological_config_cannot_exceed_a_filesystem_entry(self):
        name = raw_dataset_dirname("r/n", "x" * 5000)
        assert all(len(part) <= 64 for part in name.split("__"))

    def test_the_hf_cache_name_is_shared_across_configs(self):
        """Stated as a test because the cancel path depends on it being shared."""
        assert hf_cache_dirname("a/b") == "a___b"
        assert hf_cache_dirname("a/b") == hf_cache_dirname("a/b")


class TestTheViewKeyIsTheIdentity:
    def test_config_and_split_are_part_of_it(self):
        assert view_key("r", "a") != view_key("r", "b")
        assert view_key("r", "a", "train") != view_key("r", "a", "test")

    def test_a_blank_field_is_not_a_second_view(self):
        """A form that submits an empty string must not create a duplicate row."""
        assert view_key("r", "", "") == view_key("r", None, None)

    def test_it_compares_ORIGINAL_values_not_sanitised_ones(self):
        """Two configs whose paths collide after truncation are still two views, and
        calling them one is how a download silently overwrites another. The 409 is
        what must catch the collision — so the key must NOT pre-truncate."""
        long_a = "x" * 70 + "a"
        long_b = "x" * 70 + "b"
        assert raw_dataset_dirname("r/n", long_a) == raw_dataset_dirname("r/n", long_b)
        assert view_key("r/n", long_a) != view_key("r/n", long_b)

    @pytest.mark.parametrize(
        "metadata",
        [None, {}, {"config": None, "split": None}, {"access_token_provided": True}],
    )
    def test_every_shape_a_PRE_FR2_row_has_reads_as_the_plain_view(self, metadata):
        """Rows on this estate carry all of these shapes, so an old row still
        collides with a new plain download — the behaviour that was there before."""
        assert view_key_of_metadata("r/n", metadata) == ("r/n", None, None)

    def test_a_row_with_a_config_reads_as_that_view(self):
        assert view_key_of_metadata("r/n", {"config": "c", "split": "train"}) == (
            "r/n",
            "c",
            "train",
        )


class TestTheDisplayNameNamesTheConfig:
    def test_without_a_config_it_is_the_repo_basename(self):
        assert display_name_for("Arrrlex/models-under-pressure") == "models-under-pressure"

    def test_with_a_config_the_config_is_visible(self):
        """Seven rows named `models-under-pressure` are unusable in a list."""
        assert display_name_for("Arrrlex/models-under-pressure", "training") == (
            "models-under-pressure (training)"
        )

    def test_the_split_is_NOT_in_the_name(self):
        """The name is not an identifier; the split is shown in its own column."""
        assert "train" not in display_name_for("a/b", "cfg")


class TestTheProductionSitesActuallyCALLThese:
    """AST, not a substring scan. Every docstring above mentions these names, and a
    text search matches the prose describing the code it is checking — this repo has
    shipped that mistake in five separate guards."""

    def _calls_in(self, module, function_name):
        tree = ast.parse(inspect.getsource(module))
        target = None
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
                target = node
                break
        assert target is not None, f"{function_name} not found in {module.__name__}"
        names = set()
        for node in ast.walk(target):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    names.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    names.add(node.func.attr)
        return names

    def test_the_download_task_calls_raw_dataset_dirname(self):
        from src.workers import dataset_tasks

        assert "raw_dataset_dirname" in self._calls_in(dataset_tasks, "download_dataset_task"), (
            "the download task no longer calls raw_dataset_dirname, so the path it "
            "writes and the identity the 409 compares can drift apart"
        )

    def test_the_download_task_guards_the_shared_cache(self):
        from src.workers import dataset_tasks

        calls = self._calls_in(dataset_tasks, "download_dataset_task")
        assert "_is_only_row_for_repo" in calls, (
            "the cancel path adds the HF arrow tree without asking whether another "
            "config shares it — cancelling one config would delete another's data"
        )
        assert "hf_cache_dirname" in calls

    def test_the_endpoint_compares_the_VIEW_not_the_repo(self):
        from src.api.v1.endpoints import datasets as endpoint

        calls = self._calls_in(endpoint, "download_dataset")
        assert "get_dataset_by_view" in calls, (
            "the 409 is back to comparing the repo alone, which makes a multi-config "
            "dataset undownloadable"
        )
        assert "get_dataset_by_repo_id" not in calls

    def test_the_endpoint_names_the_row_with_its_config(self):
        from src.api.v1.endpoints import datasets as endpoint

        assert "display_name_for" in self._calls_in(endpoint, "download_dataset")


class TestTheSharedCacheCheckFailsClosed:
    """It runs inside a cancellation handler, where a second exception is caught by
    nothing. Wasting disk is recoverable; deleting a completed neighbour is not."""

    def test_a_database_error_leaves_the_cache_in_place(self, monkeypatch):
        from src.workers import dataset_tasks

        def boom():
            raise RuntimeError("database is down")

        monkeypatch.setattr("src.core.database.SyncSessionLocal", boom)
        assert dataset_tasks._is_only_row_for_repo("a/b", "some-id") is False

    def test_it_is_module_level_so_a_test_can_drive_it(self):
        """Inline in the handler it would be unreachable, and a mutation disabling
        it would leave the suite green — the lesson `remove_partial_download`
        already carries in this same file."""
        from src.workers import dataset_tasks

        assert callable(dataset_tasks._is_only_row_for_repo)

    def test_the_session_is_closed_even_on_failure(self, monkeypatch):
        from src.workers import dataset_tasks

        closed = []

        class Session:
            def query(self, *a, **k):
                raise RuntimeError("query failed")

            def close(self):
                closed.append(True)

        monkeypatch.setattr("src.core.database.SyncSessionLocal", lambda: Session())
        assert dataset_tasks._is_only_row_for_repo("a/b", "x") is False
        assert closed == [True], "the session leaked on the error path"


class TestTheRepoLookupSurvivesSeveralRows:
    """`scalar_one_or_none()` raises `MultipleResultsFound` on two rows, which FR-2
    makes an ordinary state.

    ⚠ THE FIRST VERSION OF THIS CLASS FAILED FOR THE WRONG REASON, AND THAT IS THE
    POINT OF THE AST. It asserted `"scalar_one_or_none" not in inspect.getsource(...)`
    — and the method's own comment EXPLAINS why `scalar_one_or_none` was removed, so
    the scrape matched the prose describing the fix and reported the fix as absent.
    That is this repo's five-times-recorded failure, committed again while writing a
    test about it. Walking the AST for the CALL cannot see a comment.
    """

    def _method_calls(self, func):
        tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    names.add(node.func.attr)
                elif isinstance(node.func, ast.Name):
                    names.add(node.func.id)
        return names

    def test_it_no_longer_CALLS_scalar_one_or_none(self):
        from src.services import dataset_service

        calls = self._method_calls(dataset_service.DatasetService.get_dataset_by_repo_id)
        assert "scalar_one_or_none" not in calls, (
            "get_dataset_by_repo_id raises MultipleResultsFound once a repo has two "
            "configs, which FR-2 permits"
        )
        assert "first" in calls

    def test_it_orders_deterministically(self):
        from src.services import dataset_service

        calls = self._method_calls(dataset_service.DatasetService.get_dataset_by_repo_id)
        assert "order_by" in calls, "which row is 'the' row must not depend on plan order"

    def test_get_dataset_by_view_uses_the_shared_normaliser(self):
        from src.services import dataset_service

        calls = self._method_calls(dataset_service.DatasetService.get_dataset_by_view)
        assert "view_key_of_metadata" in calls
        assert "view_key" in calls

    def test_the_AST_check_can_actually_fail(self):
        """Prove it bites before trusting its silence: a function that DOES call
        `scalar_one_or_none` must be seen to."""
        from src.services import dataset_service

        calls = self._method_calls(dataset_service.DatasetService.get_dataset)
        assert "scalar_one_or_none" in calls, (
            "the AST walk found no scalar_one_or_none even in a method that calls it, "
            "so it is not looking at anything"
        )
