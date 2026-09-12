"""A cancelled sweep actually stops sweeping.

Shape-A test for the `"labeling_sweep"` scope. The registry requires one, and
the requirement is not bureaucratic: this scope was registered with a PARALLEL
`stop_requested_at` mechanism beside it, so the scope existed, was documented,
and was wired to nothing. `test_cancel_registry_completeness` caught all four
halves — no route called `request_cancel`, nothing polled it, its declared
`progress_field` was not a column, and no test named it.

A sweep is the one lifecycle here where the gap between "the operator asked" and
"the work stopped" is a whole batch — up to 4.4 hours — so the distinction the
timestamp preserves is load-bearing rather than cosmetic.
"""

import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from src.core.cancellation import SCOPES
from src.models.labeling_resume_sweep import LabelingResumeSweep
from src.services.labeling_sweep_service import LabelingSweepService


def _sweep(**overrides) -> LabelingResumeSweep:
    base = dict(
        id=f"sweep_{uuid.uuid4().hex[:8]}",
        extraction_job_id="ext_1",
        config={},
        status="running",
        batches_done=0,
        features_labeled=0,
        features_failed=0,
        max_batches=27,
        batch_size=2000,
        progress=0.0,
    )
    base.update(overrides)
    return LabelingResumeSweep(**base)


class TestTheScopeIsWiredToTheFramework:

    def test_the_scope_names_its_request_field(self):
        scope = SCOPES["labeling_sweep"]
        assert scope.request_field == "cancel_requested_at"

    def test_every_field_the_scope_declares_is_a_real_column(self):
        """`setattr` on a missing field silently creates it on the instance and
        persists nothing — a cancellation that reports success and does nothing."""
        scope = SCOPES["labeling_sweep"]
        columns = set(LabelingResumeSweep.__table__.c.keys())
        for field in (
            scope.id_field,
            scope.status_field,
            scope.request_field,
            scope.error_field,
            scope.progress_field,
            scope.completed_at_field,
        ):
            if field is not None:
                assert field in columns, f"scope declares {field!r}, not a column"

    def test_a_vanished_sweep_reads_as_cancelled(self):
        """Consistent with `labeling`: there is nothing left to record progress
        against and nobody waiting for it."""
        assert SCOPES["labeling_sweep"].missing_row == "cancelled"


class TestACancelledSweepStopsSweeping:

    def test_a_requested_stop_halts_the_next_batch(self):
        service = LabelingSweepService(MagicMock())
        stopped = _sweep(cancel_requested_at=datetime.now(timezone.utc))
        assert "stop requested" in service.should_stop(stopped)

    def test_a_sweep_with_no_request_continues(self):
        """A live sweep must not be stopped by the guard itself.

        The first version of this test ended in `or True`, which made it assert
        nothing — and it still failed, on a TypeError, because `is_cancelled`
        takes a STATUS and was being passed a session. A vacuous assertion hid
        the shape of its own bug.
        """
        service = LabelingSweepService(MagicMock())
        assert service.should_stop(_sweep()) is None

    def test_a_cancelled_status_stops_it_even_with_no_timestamp(self):
        """Another route may set the status directly; the registry's vocabulary
        is what decides what that word means."""
        service = LabelingSweepService(MagicMock())
        assert service.should_stop(_sweep(status="cancelled")) is not None

    def test_the_request_is_written_through_the_registry(self):
        """Not by assigning the column directly. Going through `request_cancel`
        is what lets the janitors and the startup reconciliation see it, and
        what keeps this lifecycle stopping the same way as every other."""
        import inspect

        source = inspect.getsource(LabelingSweepService.request_stop)
        assert 'request_cancel("labeling_sweep"' in source
        assert "cancel_requested_at = datetime" not in source, (
            "writing the column directly bypasses the registry"
        )

    def test_the_task_checks_before_doing_any_work(self):
        import inspect

        from src.workers.labeling_tasks import resume_sweep_step

        source = inspect.getsource(resume_sweep_step)
        assert source.index("should_stop(") < source.index("next_batch_ids(")
