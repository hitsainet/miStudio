"""
A forked pool worker must not be able to cancel a tokenization.

`Dataset.map(num_proc=N)` forks N workers, and shutting that pool down signals
every one of them -- on the ordinary path where the map finished, not only when
an operator cancels. The task's handler is inherited by those forks, so it is
reached during routine teardown. Only the installing process may act on it.

Live failure this pins (2026-08-25): a 446,762-sample tokenization reached
`446762/446762`, took a signal one millisecond later, and was thrown away at
100% with its row frozen mid-progress and no output written.
"""

import os
import signal

import pytest

from src.workers.dataset_tasks import (
    TokenizationSignalState,
    apply_tokenization_success,
    make_tokenization_signal_handler,
)


class _CleanupSpy:
    def __init__(self):
        self.calls = 0

    def __call__(self):
        self.calls += 1


def test_owner_cancels_an_unfinished_tokenization():
    """The installing process still gets the cancel behaviour it had before."""
    state = TokenizationSignalState()
    cleanup = _CleanupSpy()
    handler = make_tokenization_signal_handler(state, os.getpid(), cleanup=cleanup)

    with pytest.raises(SystemExit):
        handler(signal.SIGTERM, None)

    assert state.shutdown_requested is True
    assert cleanup.calls == 1


def test_owner_lets_a_finished_tokenization_save():
    state = TokenizationSignalState()
    state.complete = True
    cleanup = _CleanupSpy()
    handler = make_tokenization_signal_handler(state, os.getpid(), cleanup=cleanup)

    handler(signal.SIGTERM, None)  # must not raise

    assert state.shutdown_requested is True
    assert cleanup.calls == 0


def test_a_forked_worker_declines_instead_of_cancelling(monkeypatch):
    """
    The load-bearing case. A process that is not the owner must neither raise
    SystemExit nor touch the shared state -- doing either aborts a job that the
    pool was merely tearing down.
    """
    state = TokenizationSignalState()
    cleanup = _CleanupSpy()

    owner_pid = os.getpid()
    forked_pid = owner_pid + 1
    monkeypatch.setattr(os, "getpid", lambda: forked_pid)

    redelivered = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: redelivered.append((pid, sig)))
    dispositions = []
    monkeypatch.setattr(
        signal, "signal", lambda sig, disp: dispositions.append((sig, disp))
    )

    handler = make_tokenization_signal_handler(state, owner_pid, cleanup=cleanup)
    handler(signal.SIGTERM, None)  # must NOT raise

    assert cleanup.calls == 0, "a forked worker reaped the owner's children"
    assert state.shutdown_requested is False, "a forked worker mutated shared state"
    assert state.complete is False

    # It declines by restoring the default disposition and re-delivering, which
    # is the death the pool asked for.
    assert dispositions == [(signal.SIGTERM, signal.SIG_DFL)]
    assert redelivered == [(forked_pid, signal.SIGTERM)]


def test_the_task_installs_an_owner_bound_handler():
    """
    Reachability: the guard is worthless if the task builds its handler some
    other way. Assert the task body binds the handler to its own pid.
    """
    import inspect

    from src.workers import dataset_tasks

    source = inspect.getsource(dataset_tasks.tokenize_dataset_task)
    tree = __import__("ast").parse(source.lstrip())

    calls = [
        n
        for n in __import__("ast").walk(tree)
        if isinstance(n, __import__("ast").Call)
        and getattr(n.func, "id", None) == "make_tokenization_signal_handler"
    ]
    assert len(calls) == 1, "the task no longer builds its handler via the factory"

    second_arg = calls[0].args[1]
    ast = __import__("ast")
    assert (
        isinstance(second_arg, ast.Call)
        and getattr(second_arg.func, "attr", None) == "getpid"
    ), "the handler is not bound to the installing process"


class TestSuccessClearsAStaleError:
    """A row that failed and was re-run must stop advertising the old failure.

    The list renders `error_message` whatever the status, so leaving it set
    showed a READY tokenization with a failure message under it -- observed on
    Bloomberg_Financial_News, 2026-08-25: status READY, 228,742,144 tokens, and
    "Tokenization failed: Task terminated by signal 15" beneath it.
    """

    STATS = {"vocab_size": 262144, "num_tokens": 42, "avg_seq_length": 512.0}

    def _row(self):
        from types import SimpleNamespace

        return SimpleNamespace(
            status=None,
            progress=80.0,
            completed_at=None,
            tokenized_path=None,
            vocab_size=None,
            num_tokens=None,
            avg_seq_length=None,
            error_message="Tokenization failed: Task terminated by signal 15",
        )

    def test_it_clears_the_error_message(self):
        row = self._row()
        apply_tokenization_success(row, "/data/x", self.STATS)
        assert row.error_message is None, (
            "a READY tokenization still carries the failure it recovered from"
        )

    def test_it_still_records_the_result(self):
        """Guard against 'fixing' the message by writing nothing at all."""
        from src.models.dataset_tokenization import TokenizationStatus

        row = self._row()
        apply_tokenization_success(row, "/data/x", self.STATS)

        assert row.status == TokenizationStatus.READY
        assert row.progress == 100.0
        assert row.tokenized_path == "/data/x"
        assert row.num_tokens == 42
        assert row.vocab_size == 262144
        assert row.completed_at is not None
