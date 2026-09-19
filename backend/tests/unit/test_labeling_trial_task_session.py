"""The trial Celery task must hand run_trial a real Session.

Feature 30 shipped the trial system with unit tests that called run_trial with a
proper Session directly. The Celery task — the only way a trial is ever actually
started — passed `self.get_db()` without entering it, so every trial in
production died before doing any work. The failure was masked: the type guard in
run_trial fired correctly, then the error handler called .query() on the context
manager and `finally: db.close()` raised AttributeError over the top of it.

Reproduced against the live pod on 2026-08-30:
    TrialError: run_trial requires a sync Session
    AttributeError: '_GeneratorContextManager' object has no attribute 'close'
"""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest


class _FakeSession:
    """Stands in for a real sync Session; records that it was closed."""

    def __init__(self):
        self.closed = False
        self.committed = False

    def query(self, *a, **k):
        return MagicMock()

    def commit(self):
        self.committed = True

    def close(self):
        self.closed = True


@contextmanager
def _patched_task(session):
    """Patch get_db on the real task object.

    The task is bind=True, so Celery has already bound `self` and
    `__wrapped__` takes only the job id — the task instance cannot be passed
    in, it has to be patched in place.
    """
    from src.workers import labeling_tasks

    task = labeling_tasks.label_features_trial_task

    @contextmanager
    def _get_db():
        try:
            yield session
        finally:
            session.close()

    with patch.object(task, "get_db", _get_db):
        yield task


class TestTheTaskEntersTheContextManager:
    def test_run_trial_receives_a_session_not_a_context_manager(self):
        seen = {}

        class _Svc:
            def __init__(self, db):
                seen["db"] = db

            def run_trial(self, job_id):
                return {"stats": {"ok": True}}

        session = _FakeSession()
        with _patched_task(session) as task, \
             patch("src.services.labeling_trial_service.LabelingTrialService", _Svc):
            result = task.__wrapped__("job1")

        assert seen["db"] is session, (
            f"run_trial got {type(seen['db']).__name__}, not a Session — "
            "every production trial dies before it starts"
        )
        assert result == {"stats": {"ok": True}}

    def test_the_session_is_closed_on_the_way_out(self):
        class _Svc:
            def __init__(self, db):
                pass

            def run_trial(self, job_id):
                return {"stats": {}}

        session = _FakeSession()
        with _patched_task(session) as task, \
             patch("src.services.labeling_trial_service.LabelingTrialService", _Svc):
            task.__wrapped__("job1")
        assert session.closed, "the context manager never exited; sessions leak"

    def test_a_failure_records_the_real_error_not_a_cleanup_error(self):
        """The original bug hid the true cause behind AttributeError on close()."""
        class _Svc:
            def __init__(self, db):
                pass

            def run_trial(self, job_id):
                raise ValueError("the actual problem")

        session = _FakeSession()
        with _patched_task(session) as task, \
             patch("src.services.labeling_trial_service.LabelingTrialService", _Svc):
            with pytest.raises(ValueError, match="the actual problem"):
                task.__wrapped__("job1")
        assert session.closed


class TestNoTaskUsesTheBareForm:
    """A source guard, because the bug is a one-token difference.

    Deliberately narrow: it reads the worker modules for `= self.get_db()`
    outside a `with`. That is the exact shape of the defect and it is cheap to
    check; it is a backstop to the behavioural tests above, not a substitute.
    """

    def test_no_worker_binds_get_db_without_entering_it(self):
        import pathlib
        import re

        offenders = []
        for f in pathlib.Path("src/workers").glob("*.py"):
            for n, line in enumerate(f.read_text().split("\n"), 1):
                if re.search(r"=\s*self\.get_db\(\)\s*$", line):
                    offenders.append(f"{f.name}:{n}")
        assert not offenders, (
            "get_db() bound without `with`; the service receives a context "
            f"manager instead of a Session: {offenders}"
        )
