"""The shared autodiscovery reader, and its own negative control.

⚠ WHY THIS FILE EXISTS. `tests/support/celery_introspection.py` was added to replace two
weaker guards, and a mutation found that the replacement had the same hole one level up:
making `assert_autodiscovered` unconditionally true (`listing = list(listing) + [module]`)
left BOTH suites that depend on it green — 114 passed. A helper that decides whether a
capability is reachable is itself load-bearing, so it gets the same treatment as the
thing it guards.

MUTATION CONTROLS (each verified to fail this file):
  W7   `assert_autodiscovered` made unconditionally true  → the refusal test
  W8   the reader looks for a different call name         → the non-empty test
  W9   the reader accepts any call, not autodiscover      → the exclusion test
"""
import pytest

from tests.support.celery_introspection import (
    assert_autodiscovered,
    autodiscovered_modules,
)


class TestTheReaderFindsTheRealList:
    def test_it_is_not_empty(self):
        assert autodiscovered_modules(), "the reader found no module list at all"

    def test_it_contains_a_module_that_is_definitely_there(self):
        assert "src.workers.training_tasks" in autodiscovered_modules()

    def test_every_entry_looks_like_a_worker_module(self):
        """A reader that accidentally collected `task_routes` keys would pick up globs
        and dotted TASK names, which is how it could pass for the wrong reason."""
        for module in autodiscovered_modules():
            assert module.startswith("src.workers."), module
            assert "*" not in module, f"{module} is a glob, so this is not the import list"

    def test_it_does_not_collect_route_keys(self):
        """`task_routes` contains `"src.workers.probe_monitor_tasks.run_probe_monitor"`
        — a module path with a task name appended. If either appears, the reader is
        reading the wrong dictionary and a missing import entry could be masked by a
        route entry that happens to start with the same path."""
        listing = autodiscovered_modules()
        assert "src.workers.probe_monitor_tasks.run_probe_monitor" not in listing
        assert "src.workers.probe_monitor_tasks" in listing


class TestTheAssertionActuallyRefuses:
    """The control W7 exposed: without this, the helper can be made to accept anything
    and every caller stays green."""

    def test_a_module_that_is_not_listed_raises(self):
        with pytest.raises(AssertionError) as caught:
            assert_autodiscovered("src.workers.there_is_no_such_module")
        message = str(caught.value)
        assert "there_is_no_such_module" in message
        assert "no worker will" in message, "the refusal must say what the consequence is"

    def test_a_module_that_IS_listed_does_not_raise(self):
        assert_autodiscovered("src.workers.training_tasks")

    def test_a_near_miss_is_still_refused(self):
        """Substring matching is the failure this helper replaced: a prefix of a real
        entry must not satisfy it."""
        with pytest.raises(AssertionError):
            assert_autodiscovered("src.workers.training_task")
        with pytest.raises(AssertionError):
            assert_autodiscovered("workers.training_tasks")
