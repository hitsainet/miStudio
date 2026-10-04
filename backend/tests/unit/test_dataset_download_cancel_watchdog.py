"""A cancelled or deleted dataset download must actually stop.

WHAT HAPPENED (2026-09-13). An operator queued `codeparrot/github-code-clean` without a
config — the all-languages build, 313.7 GB — then cancelled it and deleted it. The
download ran on for another 30 minutes, its progress stuck at 0% in the UI, blocking the
three small downloads and a tokenization queued behind it on the solo worker, which
finally had to be SIGKILLed. Two independent defects:

  1. The only cancellation checkpoint was a tqdm swap. `datasets` binds its progress-bar
     class when the library is first imported, and the worker imports it at startup, so
     the swapped class never reached the bars that tick. No check ever ran.
  2. `dataset_download` treated a missing row as "continue", so even a working check
     would have kept going once the dataset was deleted.

`CancelWatchdog` stops the blocking call from outside; a deleted row is now a stop signal.
"""

import ast
import inspect
import threading
import time

import pytest

from src.core import cancellation as C
from src.core.cancellation import CancelWatchdog, OperatorCancelled


class _Check:
    """Stands in for CancelCheck: says stop from the Nth poll on."""

    def __init__(self, stop_from_poll: int, reason: str = "deleted"):
        self.polls = 0
        self.stop_from_poll = stop_from_poll
        self._reason = reason
        self.reason = None

    def poll_now(self) -> bool:
        self.polls += 1
        if self.polls >= self.stop_from_poll:
            self.reason = self._reason
            return True
        return False


def _spin(seconds: float) -> int:
    """Pure-Python work with no callback — the shape of a blocking library call."""
    deadline = time.monotonic() + seconds
    n = 0
    while time.monotonic() < deadline:
        n += 1
    return n


def _watchdog_threads(name: str) -> list:
    return [t for t in threading.enumerate() if t.name == name and t.is_alive()]


def test_the_watchdog_stops_a_block_that_offers_no_callback():
    check = _Check(stop_from_poll=2)
    started = time.monotonic()
    with pytest.raises(OperatorCancelled) as raised:
        with CancelWatchdog("dataset_download", "ds-1", interval_s=0.05, checker=check):
            _spin(20)
    assert time.monotonic() - started < 5, "the block ran on after the stop"
    assert raised.value.scope == "dataset_download"
    assert raised.value.target_id == "ds-1"
    assert raised.value.reason == "deleted"


def test_a_block_that_finishes_first_is_never_interrupted_afterwards():
    check = _Check(stop_from_poll=10**9)
    with CancelWatchdog("dataset_download", "ds-2", interval_s=0.05, checker=check):
        _spin(0.2)
    polls_at_exit = check.polls
    check.stop_from_poll = 0  # the operator cancels only after the work finished
    _spin(0.5)  # an injected exception would surface here, in unrelated code

    deadline = time.monotonic() + 2
    while _watchdog_threads("cancel-watchdog:dataset_download:ds-2") and time.monotonic() < deadline:
        _spin(0.05)
    assert not _watchdog_threads("cancel-watchdog:dataset_download:ds-2"), "watchdog thread leaked"
    assert check.polls <= polls_at_exit + 1, "the watchdog kept polling after the block exited"


def test_a_stop_that_races_the_exit_is_not_delivered_into_later_code():
    """A poll can answer "stop" in the instant the block exits; nothing may be injected then.

    Without the armed flag the exception lands in whatever the thread runs next — for a
    download task, the code that marks the dataset READY.
    """
    exited = threading.Event()

    class _AnswersAfterExit:
        reason = "cancelled"

        def poll_now(self):
            exited.wait(5)  # hold the answer until the block has exited
            return True

    with CancelWatchdog("dataset_download", "ds-4", interval_s=0.01, checker=_AnswersAfterExit()):
        _spin(0.05)
    exited.set()
    _spin(0.5)  # a late injection would raise OperatorCancelled here


def test_polling_keeps_going_until_a_stop_arrives():
    check = _Check(stop_from_poll=5)
    with pytest.raises(OperatorCancelled):
        with CancelWatchdog("dataset_download", "ds-3", interval_s=0.02, checker=check):
            _spin(20)
    assert check.polls == 5


@pytest.mark.parametrize("kind", ["dataset_download", "dataset_tokenization"])
def test_deleting_the_row_is_a_stop_signal(kind):
    class _NoRow:
        def query(self, model):
            return self

        def filter(self, *args, **kwargs):
            return self

        def populate_existing(self):
            return self

        def first(self):
            return None

    check = C.cancel_checker(kind, "deleted-id", db=_NoRow())
    assert check.poll_now() is True, f"{kind}: a deleted row must stop the job"
    assert check.reason == "deleted"


def test_load_dataset_and_save_to_disk_run_inside_the_watchdog():
    """The wiring: both blocking calls in the download task sit inside the watchdog."""
    from src.workers import dataset_tasks

    tree = ast.parse(inspect.getsource(dataset_tasks))
    task = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "download_dataset_task"
    )
    guarded = set()
    for node in ast.walk(task):
        if not isinstance(node, ast.With):
            continue
        for item in node.items:
            call = item.context_expr
            if (
                isinstance(call, ast.Call)
                and getattr(call.func, "id", None) == "CancelWatchdog"
                and call.args
                and isinstance(call.args[0], ast.Constant)
                and call.args[0].value == "dataset_download"
            ):
                for inner in ast.walk(node):
                    if isinstance(inner, ast.Call):
                        guarded.add(getattr(inner.func, "id", None) or getattr(inner.func, "attr", None))
    missing = {"load_dataset", "save_to_disk"} - guarded
    assert not missing, f"not inside CancelWatchdog('dataset_download', ...): {sorted(missing)}"
