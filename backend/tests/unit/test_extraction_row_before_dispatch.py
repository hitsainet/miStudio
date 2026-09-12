"""An extraction's row must be committed before its task can be consumed.

2026-09-12, from production. Starting an extraction on LFM2.5-1.2B-Instruct
returned `202 Accepted` and then did nothing visible. The worker log, one
second later:

    Extraction ext_m_88d55564_20260912_110358 has no database row, so this run
    could never be recorded. Refusing to start. (The row is created by the API
    before dispatch; if it is missing the request did not commit.)

Two fixes, each correct alone, contradicted each other:

  * `a328607e` (2026-08-24) made the task refuse any `extraction_id` with no
    row, assuming the id is only ever passed on a retry, when a row exists.
  * `2badda15` (2026-09-05) made the endpoint ALWAYS pass its id, so the id the
    caller holds is the row's id — and never created the row.

So from 2026-09-05 every extraction started from the UI (and every retry) was
refused, and because the refusal had no row to mark failed, nothing reached
the UI. Underneath that sat a third defect: the task's own first-attempt branch
passed `max_seq_length` / `micro_batch_token_budget` to a `create_extraction`
that accepts neither, and the resulting TypeError was swallowed as a warning —
so even a caller passing no id got no row.
"""

import ast
import inspect
from unittest.mock import MagicMock

import pytest

import _cancel_ast as A

ROW = dict(
    extraction_id="ext_m_x_20260912_110358",
    model_id="m_x",
    dataset_id="d",
    layer_indices=[0],
    hook_types=["residual"],
    max_samples=10,
    batch_size=8,
    micro_batch_size=None,
    gpu_id=0,
)


def _service():
    from src.services.extraction_db_service import ExtractionDatabaseService

    return ExtractionDatabaseService


class TestTheRowIsWrittenBeforeTheTaskIsQueued:
    def test_the_row_is_added_and_committed_when_dispatch_runs(self):
        db = MagicMock()
        seen = {}

        def dispatch():
            seen["added_id"] = db.add.call_args.args[0].id if db.add.called else None
            seen["committed"] = db.commit.called
            return MagicMock(id="celery-1")

        _service().create_row_then_dispatch(db, dispatch=dispatch, **ROW)

        assert seen == {"added_id": ROW["extraction_id"], "committed": True}, (
            "the task was queued before its row existed; an idle worker consumes "
            "the message in milliseconds and refuses the job"
        )

    def test_the_celery_task_id_is_recorded_on_the_row(self):
        db = MagicMock()
        row = MagicMock()
        db.query.return_value.filter_by.return_value.first.return_value = row

        task = _service().create_row_then_dispatch(
            db, dispatch=lambda: MagicMock(id="celery-7"), **ROW
        )

        assert task.id == "celery-7"
        assert row.celery_task_id == "celery-7"

    def test_a_failed_dispatch_marks_the_row_failed_and_raises(self, monkeypatch):
        S = _service()
        marked = []
        monkeypatch.setattr(
            S, "mark_failed", staticmethod(lambda db, eid, msg: marked.append((eid, msg)))
        )

        def dispatch():
            raise ConnectionError("broker unreachable")

        with pytest.raises(ConnectionError):
            S.create_row_then_dispatch(MagicMock(), dispatch=dispatch, **ROW)

        assert len(marked) == 1, "a row whose task was never sent stays QUEUED forever"
        assert marked[0][0] == ROW["extraction_id"]
        assert "broker unreachable" in marked[0][1]

    def test_a_row_that_cannot_be_written_is_never_dispatched(self):
        db = MagicMock()
        db.commit.side_effect = RuntimeError("duplicate key value")
        dispatched = []

        with pytest.raises(RuntimeError):
            _service().create_row_then_dispatch(
                db, dispatch=lambda: dispatched.append(True), **ROW
            )

        assert dispatched == [], "a task was queued for a row that does not exist"


def _callee(node: ast.Call):
    return getattr(node.func, "id", None) or getattr(node.func, "attr", None)


class TestEveryEndpointDispatchGoesThroughIt:
    """Structure, not text: one parsed tree per endpoint, so node identity holds."""

    @pytest.mark.parametrize("endpoint", ["extract_model_activations", "retry_extraction"])
    def test_every_delay_is_the_helpers_dispatch(self, endpoint):
        from src.api.v1.endpoints import models as models_endpoint

        tree = A._tree(getattr(models_endpoint, endpoint))
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
        helpers = [c for c in calls if _callee(c) == "create_row_then_dispatch"]
        delays = [c for c in calls if _callee(c) == "delay"]

        assert len(helpers) == 1, f"{endpoint} does not write the row through the helper"
        assert delays, f"{endpoint} no longer dispatches the task"

        dispatch = A.keyword_of(helpers[0], "dispatch")
        assert dispatch is not None, "the helper is called without a dispatch"
        inside = {id(n) for n in ast.walk(dispatch)}
        stray = [d for d in delays if id(d) not in inside]
        assert not stray, (
            f"{endpoint} queues the task outside the helper, so nothing "
            f"guarantees its row exists first"
        )

    @pytest.mark.parametrize("endpoint", ["extract_model_activations", "retry_extraction"])
    def test_the_row_and_the_task_carry_the_same_id(self, endpoint):
        from src.api.v1.endpoints import models as models_endpoint

        tree = A._tree(getattr(models_endpoint, endpoint))
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
        helper = next(c for c in calls if _callee(c) == "create_row_then_dispatch")
        delay = next(c for c in calls if _callee(c) == "delay")

        row_id = A.keyword_of(helper, "extraction_id")
        task_id = A.keyword_of(delay, "extraction_id")
        assert row_id is not None and task_id is not None
        assert ast.dump(row_id) == ast.dump(task_id), (
            "the row is written under one id and the task is handed another"
        )


class TestTheTasksOwnCreateCallFitsTheSignature:
    def test_every_keyword_is_one_create_extraction_accepts(self):
        from src.workers.model_tasks import extract_activations

        calls = A.calls_named(extract_activations, "create_extraction")
        assert calls, "the task no longer creates a row when it mints its own id"

        sig = inspect.signature(_service().create_extraction)
        for call in calls:
            names = [kw.arg for kw in call.keywords if kw.arg]
            try:
                sig.bind_partial(**{name: None for name in names})
            except TypeError as exc:
                pytest.fail(
                    f"the task calls create_extraction with an argument it does "
                    f"not accept ({exc}); the TypeError is swallowed as a warning "
                    f"and no row is created"
                )
