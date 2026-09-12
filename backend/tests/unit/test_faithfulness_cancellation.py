"""Phase 3 — the faithfulness cancel path was written, wired to nothing, and dead.

`circuit_faithfulness_service` has had a full cancellation path for as long as
faithfulness has existed: a poll at the top of the per-prompt loop, a
`_FaithfulnessCancelled` raise, and a handler in the task that turns it into a
cancelled status without a failure. Every line of it was unreachable, because
the one caller passed `cancel_check=None`.

It also had NO TEST — the surrounding suite never once constructed
`_FaithfulnessCancelled`. So "dead" understates it: nothing would have noticed
if the path had been deleted outright.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.core import cancellation as C


class _Circuit:
    def __init__(self, status="running"):
        self.id = "crc_1"
        self.faithfulness_status = status


class _Query:
    def __init__(self, row):
        self._row = row

    def filter(self, *a, **k):
        return self

    filter_by = filter

    def populate_existing(self):
        return self

    def first(self):
        return self._row


def _raw(celery_task):
    """The plain function behind a bind=True Celery task.

    `.__wrapped__` is a BOUND method — self is already the real Task instance,
    whose `get_db()` would open a real session. `.__func__` is the unbound
    function, so a fake task can be passed as self.
    """
    return celery_task.__wrapped__.__func__


def _session(row):
    db = MagicMock()
    db.query.side_effect = lambda model: _Query(row)
    return db


class TestShapeATheRealLoopStops:
    """Drives `CircuitFaithfulnessService._behavior`, the production loop."""

    def _behavior_kwargs(self, cancel_check, doc_ids):
        """`suppress={}` is the clean-behaviour pass, which touches neither
        `get_hookable_module` nor `resolve_decoder_weight` before the loop —
        so the poll can be observed in isolation from any GPU work."""
        return dict(
            suppress={},
            model=MagicMock(),
            structure=MagicMock(),
            get_hookable_module=MagicMock(),
            saes={},
            readers={},
            dataset=MagicMock(),
            tokenizer=MagicMock(),
            doc_ids=doc_ids,
            down_layer=0,
            down_features=[0],
            device="cpu",
            resolve_decoder_weight=MagicMock(),
            cancel_check=cancel_check,
            _pad_batch=MagicMock(),
        )

    def test_a_cancelled_run_abandons_before_any_model_work(self):
        from src.services.circuit_faithfulness_service import (
            CircuitFaithfulnessService, _FaithfulnessCancelled,
        )

        # The dataset must be NON-EMPTY, or `if doc_id >= len(dataset): continue`
        # skips every prompt and the "no model work happened" assertions below
        # are true by construction of the fixture rather than by the poll's
        # position. A bare MagicMock has len() == 0.
        kwargs = self._behavior_kwargs(lambda: True, doc_ids=list(range(1000)))
        dataset = MagicMock()
        dataset.__len__ = MagicMock(return_value=1000)
        kwargs["dataset"] = dataset
        model = kwargs["model"]

        with pytest.raises(_FaithfulnessCancelled):
            CircuitFaithfulnessService._behavior(**kwargs)

        # THE POINT: the poll is at the TOP of the per-prompt loop, so a
        # cancelled run costs zero forward passes rather than one more prompt.
        assert dataset.__getitem__.call_count == 0, (
            "the loop read a prompt before checking whether it was cancelled"
        )
        assert model.call_count == 0, "a forward pass ran after cancellation"

    def test_the_poll_happens_once_per_prompt(self):
        """Bounded latency: one prompt, not one whole pass."""
        from src.services.circuit_faithfulness_service import (
            CircuitFaithfulnessService, _FaithfulnessCancelled,
        )

        calls = {"n": 0}

        def check():
            calls["n"] += 1
            return calls["n"] > 3

        kwargs = self._behavior_kwargs(check, doc_ids=list(range(1000)))
        dataset = MagicMock()
        dataset.__len__ = MagicMock(return_value=0)  # every prompt skips the body
        kwargs["dataset"] = dataset

        # EXACTLY four, not ">= 1". The loose version passed against a throttled
        # poll — `if i % 50 == 0 and cancel_check()` still calls it at least
        # once and still raises, while the latency becomes fifty prompts.
        with pytest.raises(_FaithfulnessCancelled):
            CircuitFaithfulnessService._behavior(**kwargs)
        assert calls["n"] == 4, (
            f"the poll ran {calls['n']} times over four prompts; it must fire "
            f"once per prompt, or the latency is however many prompts it skips"
        )

    def test_none_still_means_no_polling(self):
        """The old caller passed None and that must remain legal — the service
        is also called from paths that genuinely have no cancel channel."""
        from src.services.circuit_faithfulness_service import CircuitFaithfulnessService

        # `assert X or True` was here until R1 — a literal tautology. What the
        # case is actually for is that `cancel_check=None` stays legal, because
        # other callers genuinely have no cancel channel.
        kwargs = self._behavior_kwargs(None, doc_ids=[])
        result = CircuitFaithfulnessService._behavior(**kwargs)
        assert isinstance(result, float), (
            f"_behavior returned {result!r}; with no prompts and no checker it "
            f"must still produce a behaviour score"
        )


class TestTheTaskSuppliesARealChecker:
    """Reachability: capture what the task actually passes, and drive it."""

    def _captured_cancel_check(self, circuit_row):
        from src.workers import circuit_validation_tasks as tasks

        captured = {}

        def fake_run(db, circuit_id, config, cancel_check=None, progress_cb=None):
            captured["cancel_check"] = cancel_check
            return {"ok": True}

        task = MagicMock()
        from contextlib import contextmanager

        @contextmanager
        def _db():
            yield _session(circuit_row)

        task.get_db = _db

        with patch(
            "src.services.circuit_faithfulness_service.CircuitFaithfulnessService.run",
            side_effect=fake_run,
        ), patch.object(tasks, "emit_circuit_run_completed"), \
             patch.object(tasks, "emit_circuit_run_progress"):
            _raw(tasks.run_circuit_faithfulness)(task, "crc_1", {})
        return captured.get("cancel_check")

    def test_the_task_no_longer_passes_none(self):
        check = self._captured_cancel_check(_Circuit("running"))
        assert check is not None, (
            "cancel_check=None — every line of the service's cancellation path "
            "is unreachable again"
        )

    def test_the_checker_it_passes_sees_a_cancelled_circuit(self):
        """Shape C for this lifecycle: what a writer sets is what this reads.

        A checker pointed at the wrong column, or at the discovery run's
        `validation_status` instead of the circuit's `faithfulness_status`,
        would satisfy the test above and still never fire.
        """
        check = self._captured_cancel_check(_Circuit("cancelled"))
        assert check() is True, (
            "the task's checker cannot see a cancelled faithfulness run — it is "
            "reading a different column or a different row"
        )

    def test_the_checker_stays_quiet_on_a_running_circuit(self):
        check = self._captured_cancel_check(_Circuit("running"))
        assert check() is False


class TestTheHandlerThatCouldNeverFire:
    def test_a_cancelled_run_returns_rather_than_raising(self):
        """RETURNING is what acks the acks_late message. Raising would redeliver
        it against a 12-hour visibility timeout."""
        from contextlib import contextmanager

        from src.workers import circuit_validation_tasks as tasks
        from src.services.circuit_faithfulness_service import _FaithfulnessCancelled

        row = _Circuit("running")
        task = MagicMock()

        @contextmanager
        def _db():
            yield _session(row)

        task.get_db = _db

        with patch(
            "src.services.circuit_faithfulness_service.CircuitFaithfulnessService.run",
            side_effect=_FaithfulnessCancelled(),
        ), patch.object(tasks, "emit_circuit_run_failed"), \
             patch.object(tasks, "emit_circuit_run_progress"):
            out = _raw(tasks.run_circuit_faithfulness)(task, "crc_1", {})

        assert out == {"status": "cancelled", "circuit_id": "crc_1"}
        assert row.faithfulness_status == "cancelled", (
            "the in-flight marker was left set, which wedges the single-GPU guard"
        )


class TestTheScopeIsRegistered:
    def test_the_faithfulness_scope_reads_the_circuits_own_column(self):
        scope = C.get_scope("circuit_faithfulness")
        assert scope.status_field == "faithfulness_status", (
            "faithfulness runs on a CIRCUIT, not on a discovery run"
        )
        assert "cancelled" in scope.cancelled_values
