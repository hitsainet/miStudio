"""Deleting a running extraction must stop the work, not just the row.

2026-09-12, from production. An operator stopped and deleted an extraction of
LFM2.5-1.2B-Instruct during its statistics phase. The API removed the row and
the 390 GB directory and reported success. The worker kept computing
statistics for another twenty minutes over unlinked files — the disk space
could not be reclaimed while it held them open — and every queued job,
including the operator's model download, waited behind it.

Two gaps, both closed here:

  * the extraction's cancellation scope said `missing_row="continue"`, so a
    vanished row read as "nothing to do" rather than "stop". That dated from
    when the task created its own row after starting; since 2a982a5f the row is
    committed before dispatch, so absence can only mean deletion;
  * the delete endpoint never asked the job to stop at all, so a task still
    waiting in the queue would run after its row was gone.
"""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

import _cancel_ast as A
from src.core import cancellation as C
from src.services.activation_service import ActivationService


class _VanishingRow:
    """A session whose row exists for the first `alive_polls` reads, then is gone."""

    def __init__(self, alive_polls):
        self.alive_polls = alive_polls
        self.polls = 0
        self._lock = threading.Lock()

    def query(self, _model):
        return self

    def filter(self, *a, **k):
        return self

    def populate_existing(self):
        return self

    def first(self):
        with self._lock:
            self.polls += 1
            if self.polls <= self.alive_polls:
                return SimpleNamespace(status="extracting", progress=90.0)
            return None


def _checker(alive_polls):
    return C.cancel_checker(
        "activation_extraction", "ext_live", db=_VanishingRow(alive_polls), min_interval_s=0
    )


class TestTheScopeTreatsDeletionAsStop:
    def test_a_vanished_row_cancels_with_reason_deleted(self):
        check = _checker(alive_polls=1)
        assert check() is False, "the row still exists on the first poll"
        assert check() is True
        assert check.reason == "deleted"


class TestTheStatisticsPhaseStops:
    def test_deleting_mid_statistics_abandons_the_remaining_chunks(self, monkeypatch):
        from src.workers import model_tasks as MT

        monkeypatch.setattr(MT, "record_progress", lambda *a, **k: True)
        monkeypatch.setattr(MT, "emit_extraction_progress", lambda **k: None)
        monkeypatch.setattr(ActivationService, "CHUNKED_STATISTICS_THRESHOLD_BYTES", 0)

        reduced = []
        real = ActivationService._chunk_accumulators

        def counting(chunk, near_zero):
            reduced.append(1)
            return real(chunk, near_zero)

        monkeypatch.setattr(ActivationService, "_chunk_accumulators", staticmethod(counting))

        rng = np.random.default_rng(0)
        activations = {
            f"layer_{i}_residual": rng.normal(size=(1000, 4, 4)).astype(np.float16)
            for i in range(3)
        }  # 3 layers x 10 chunks of 100 = 30 chunks
        heartbeat = MT.build_statistics_heartbeat("m_1", "ext_live", _checker(alive_polls=3))

        svc = ActivationService.__new__(ActivationService)
        with pytest.raises(C.OperatorCancelled) as stopped:
            svc._calculate_statistics(activations, on_progress=heartbeat)

        assert stopped.value.reason == "deleted"
        assert len(reduced) < 30, (
            f"all {len(reduced)} chunks ran after the extraction was deleted"
        )


class TestTheGpuPassStops:
    def test_the_per_sample_callback_raises_once_the_row_is_gone(self, monkeypatch):
        from src.workers import model_tasks as MT

        monkeypatch.setattr(MT, "emit_extraction_progress", lambda **k: None)
        monkeypatch.setattr(MT.ExtractionDatabaseService, "update_progress", staticmethod(lambda **k: None))
        task = SimpleNamespace(get_db=lambda: MagicMock())

        callback = MT.build_extraction_progress_callback(task, "m_1", "ext_live", _checker(alive_polls=1))

        callback(10, 10_000)  # row still there
        with pytest.raises(C.OperatorCancelled) as stopped:
            callback(20, 10_000)
        assert stopped.value.reason == "deleted"


class TestTheDeleteEndpointAsksTheJobToStop:
    def _calls(self):
        import ast

        from src.api.v1.endpoints import models as models_endpoint

        tree = A._tree(models_endpoint.delete_extractions)
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
        name = lambda c: getattr(c.func, "id", None) or getattr(c.func, "attr", None)
        # `sync_db.delete(...)` specifically. A bare "delete" also matches the
        # `@router.delete(...)` DECORATOR at line 1 of this function's source —
        # the first version of this test did exactly that and compared the
        # cancel against the decorator instead of the row delete.
        row_deletes = [
            c for c in calls
            if isinstance(c.func, ast.Attribute) and c.func.attr == "delete"
            and getattr(c.func.value, "id", None) == "sync_db"
        ]
        return [c for c in calls if name(c) == "request_cancel"], row_deletes

    def test_request_cancel_is_called_for_the_extraction_scope(self):
        cancels, _ = self._calls()
        assert cancels, "deleting a live extraction never asks the job to stop"
        assert any(A.first_string_arg(c) == "activation_extraction" for c in cancels)

    def test_it_revokes_the_queued_task_too(self):
        cancels, _ = self._calls()
        assert any(A.keyword_of(c, "celery_task_id") is not None for c in cancels), (
            "without the task id a queued extraction still starts after its row is gone"
        )

    def test_the_stop_is_requested_before_the_row_is_deleted(self):
        cancels, deletes = self._calls()
        assert cancels and deletes
        assert min(c.lineno for c in cancels) < min(d.lineno for d in deletes), (
            "the row is deleted first, so request_cancel finds nothing and "
            "revokes nothing"
        )
