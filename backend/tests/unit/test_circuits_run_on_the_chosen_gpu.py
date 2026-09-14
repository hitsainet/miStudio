"""Circuit GPU jobs run on the card the caller chose — and say which card that was.

Six runs load a model: capture, attribution, validation, faithfulness,
calibration and the steered-transcript recorder (calibration and the recorder
through the shared `steering_core`). Until 2026-09-13 every one of them loaded
onto a bare ``"cuda"`` — CUDA index 0 — which became the 12 GB RTX 3080 Ti when
it was added beside the 3090.

For each run this file proves, by behaviour and never by reading source:

  (a) the endpoint resolves the request (an index becomes the UUID it names
      NOW) and stores it on the row or passes it with the task — payload AND
      call count;
  (b) an unknown card is a 400 with no row created or marked and nothing
      dispatched;
  (c) the worker places the job with that request, records the card, and hands
      `placement.device` to the model loader / `steering_core`;
  (d) a `GpuPlacementError` fails the run with its message.

No GPU is needed: `list_cards` and `place_job` are faked, and the model loaders
are stopped at their first call so only the arguments they received are read.

MUTATION CONTROLS (each applied, run red, restored; see the commit report):
  * capture endpoint stores `body.gpu` instead of the resolved request
  * create_run drops `gpu_request=` from the row constructor
  * attribution endpoint drops `gpu_request=` from `.delay`
  * attribution service resets the CURRENT device's peak stats
  * validation reproduce ignores the manifest's recorded request
  * faithfulness task places with None instead of its request
  * steering_core loads with device_map="auto"
  * recorder task does not record the card on its row
  * faithfulness endpoint resolves AFTER marking the pass in flight
"""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch
from fastapi.testclient import TestClient

from src.services import gpu_placement
from src.services.gpu_placement import GpuCard, GpuPlacementError, Placement

TI_UUID = "GPU-f47ba814-49a2-603f-3595-275284140251"
RTX_UUID = "GPU-247aa582-0d1b-e161-8156-983ed1fefc57"
MISSING_UUID = "GPU-00000000-0000-0000-0000-000000000000"

CARDS = [
    GpuCard(index=0, uuid=TI_UUID, name="NVIDIA GeForce RTX 3080 Ti", total_mb=12_288, free_mb=11_000),
    GpuCard(index=1, uuid=RTX_UUID, name="NVIDIA GeForce RTX 3090", total_mb=24_576, free_mb=23_000),
]

#: The device a placement on the 3090 hands out. Constructible without CUDA.
RTX_DEVICE = torch.device("cuda", 1)
RTX_RECORD = {"request": RTX_UUID, "uuid": RTX_UUID, "name": "NVIDIA GeForce RTX 3090"}
REFUSAL = ("GPU 0 (NVIDIA GeForce RTX 3080 Ti, 11,000 of 12,288 MB free) cannot take "
           "this job: it needs ~15,000 MB. Choose another GPU or Auto.")


class _Stop(Exception):
    """Raised by a faked loader once it has seen its arguments."""


# ── fakes ────────────────────────────────────────────────────────────────


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

    def all(self):
        return [self._row]


class _Session:
    """Every query returns the one row; commits are counted."""

    def __init__(self, row):
        self.row = row
        self.commits = 0

    def query(self, *_models):
        return _Query(self.row)

    def commit(self):
        self.commits += 1

    def rollback(self):
        pass

    def add(self, _obj):
        pass

    def refresh(self, _obj):
        pass


def _fake_task(session):
    task = MagicMock()

    @contextmanager
    def _db():
        yield session

    task.get_db = _db
    return task


def _raw(celery_task):
    """The plain function behind a bind=True task, so a fake task can be self."""
    return celery_task.__wrapped__.__func__


@pytest.fixture
def cards(monkeypatch):
    monkeypatch.setattr(gpu_placement, "list_cards", lambda: CARDS)


@pytest.fixture
def placed(monkeypatch):
    """`place_job` puts every job on the 3090 and records what it was asked for."""
    asked = []

    # `allow_shard` is accepted and ignored here: which run types may split is
    # pinned in test_circuits_run_split.py.
    def fake_place_job(requested="auto", required_mb=None, cards=None, allow_shard=False):
        asked.append(requested)
        return Placement(card=CARDS[1], device=RTX_DEVICE)

    monkeypatch.setattr("src.workers.circuit_gpu.place_job", fake_place_job)
    return asked


@pytest.fixture
def refused(monkeypatch):
    def fake_place_job(requested="auto", required_mb=None, cards=None, allow_shard=False):
        raise GpuPlacementError(REFUSAL, requested=requested, cards=CARDS)

    monkeypatch.setattr("src.workers.circuit_gpu.place_job", fake_place_job)


@pytest.fixture
def client():
    from src.core.database import get_db
    from src.main import app

    session = AsyncMock()
    session.execute = AsyncMock()
    session.commit = AsyncMock()

    async def _override():
        yield session

    app.dependency_overrides[get_db] = _override
    try:
        yield TestClient(app, raise_server_exceptions=False)
    finally:
        app.dependency_overrides.pop(get_db, None)


def _circuit(**kw):
    from datetime import datetime

    from src.models.circuit import Circuit

    return Circuit(
        id=kw.pop("id", "crc_g1"), name="GPU", granularity="feature",
        saes=[{"mistudio_sae_id": "sae_l13", "layer": 13}],
        members=[{"layer": 13, "member_kind": "feature_ref",
                  "feature": {"feature_idx": 1, "strength": 1.0}}],
        edges=[], budget=None, faithfulness=None, rung=0, promoted=False,
        discovery_run_id="dsc1", model_id="m_1",
        created_at=datetime(2026, 9, 13), updated_at=datetime(2026, 9, 13), **kw)


def _discovery(**kw):
    from datetime import datetime

    from src.models.circuit_runs import CircuitDiscoveryRun

    return CircuitDiscoveryRun(
        id="dsc1", capture_run_id="cap1", status="completed", params={},
        candidates=[{"up": {"layer": 1, "feature_idx": 2},
                     "down": {"layer": 3, "feature_idx": 4}}],
        created_at=datetime(2026, 9, 13), updated_at=datetime(2026, 9, 13), **kw)


def _capture(**kw):
    from datetime import datetime

    from src.models.circuit_runs import CircuitCaptureRun

    return CircuitCaptureRun(
        id="cap_test1", status=kw.pop("status", "estimated"),
        manifest={"corpus": {}, "layers": [], "split": {}}, stale=False,
        created_at=datetime(2026, 9, 13), updated_at=datetime(2026, 9, 13), **kw)


def _sync_bridge(row, *, effect=None, events=None, commit_event=None):
    """A `_run_sync` that RUNS the endpoint's sync function on a one-row session.

    Confirm guards and marks in one sync transaction, so a stub that never calls
    the function would skip the very mark under test. The guard's advisory-lock
    query is replaced by a no-op, because `_Session` answers every query with the
    one row; a test that wants the guard to refuse passes `effect`. Each sync
    commit appends `commit_event()` to `events`.
    """

    class _Recording(_Session):
        def commit(self):
            super().commit()
            if events is not None:
                events.append(commit_event() if commit_event else ("commit",))

    async def _call(_db, fn):
        from src.services.circuit_capture_service import CircuitCaptureService

        if effect is not None:
            raise effect
        # `gpu_request`: multi-GPU Phase 3 guards per card, so every endpoint
        # passes the request it is checking.
        with patch.object(CircuitCaptureService, "assert_no_active_gpu_run",
                          lambda _sync_db, gpu_request=None: None):
            return fn(_Recording(row))

    return AsyncMock(side_effect=_call)


# ═════════════════════════════ CAPTURE ════════════════════════════════════


CAPTURE_BODY = {"dataset_id": "ds1", "layers": [{"layer": 13, "sae_id": "sae1"}]}


class TestCaptureEndpoint:
    def _post(self, client, body):
        from src.services.circuit_capture_service import CircuitCaptureService

        created = []

        def fake_create(sync_db, config, gpu_request=None):
            created.append((config, gpu_request))
            return SimpleNamespace(id="cap_new")

        async def run_sync(db, fn):
            return fn(MagicMock())

        run_sync_spy = AsyncMock(side_effect=run_sync)
        with patch.object(CircuitCaptureService, "assert_no_active_gpu_run"), \
             patch.object(CircuitCaptureService, "create_run", side_effect=fake_create), \
             patch("src.api.v1.endpoints.circuit_discovery._run_sync", new=run_sync_spy), \
             patch("src.workers.circuit_capture_tasks.capture_circuit_activations.delay",
                   return_value=MagicMock(id="task_c")) as delay:
            response = client.post("/api/v1/circuit-capture", json=body)
        return response, created, delay, run_sync_spy

    def test_an_index_is_stored_as_the_uuid_it_names(self, client, cards):
        r, created, delay, _ = self._post(client, {**CAPTURE_BODY, "gpu": "1"})

        assert r.status_code == 202, r.text
        assert [g for _c, g in created] == [RTX_UUID]
        assert "gpu" not in created[0][0], "the request leaked into the capture manifest config"
        delay.assert_called_once_with("cap_new", confirmed=False)

    def test_no_gpu_field_is_auto(self, client, cards):
        r, created, _delay, _ = self._post(client, CAPTURE_BODY)
        assert r.status_code == 202, r.text
        assert [g for _c, g in created] == ["auto"]

    def test_an_unknown_card_is_a_400_with_no_row_and_no_dispatch(self, client, cards):
        r, created, delay, run_sync = self._post(client, {**CAPTURE_BODY, "gpu": MISSING_UUID})

        assert r.status_code == 400
        assert "RTX 3090" in r.json()["detail"]
        assert created == [] and run_sync.await_count == 0
        delay.assert_not_called()


class TestCaptureRowStoresTheRequest:
    def test_create_run_puts_the_request_on_the_row(self):
        from src.models.circuit_runs import CircuitCaptureRun
        from src.models.dataset_tokenization import TokenizationStatus
        from src.services.circuit_capture_service import CircuitCaptureService

        row = MagicMock(local_path="saes/s1", layer=13, n_features=16, model_id="m_1")
        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = row
        db.query.return_value.filter.return_value.all.return_value = [row]
        added = []
        db.add.side_effect = added.append
        tokenization = SimpleNamespace(status=TokenizationStatus.READY, id="tok_1")

        with patch("src.workers.model_tasks.select_tokenization_for_model",
                   return_value=tokenization):
            CircuitCaptureService.create_run(
                db, {**CAPTURE_BODY, "model_id": "m_1"}, gpu_request=RTX_UUID)

        runs = [a for a in added if isinstance(a, CircuitCaptureRun)]
        assert len(runs) == 1 and runs[0].gpu_request == RTX_UUID


class TestCaptureConfirmRunsOnTheChosenCard:
    """Confirm honours the picker as it stands when the user confirms.

    User decision (2026-09-13): changing the GPU between the estimate and the
    confirm must change the card the full capture runs on. With no `gpu` in the
    body (an older client, a bare REST call) the estimate's request is kept.

    MUTATION CONTROLS (2026-09-13, each red, restored byte-identically):
      K1 resolve `run.gpu_request` and ignore `body.gpu`
           -> test_a_card_named_at_confirm_replaces_the_estimates fails
      K2 assign `run.gpu_request` before the one-GPU-run guard
           -> test_a_refused_confirm_leaves_the_stored_request_alone fails
    """

    def _confirm(self, client, run, json=None, run_sync_effect=None):
        with patch("src.api.v1.endpoints.circuit_discovery._capture_or_404",
                   new=AsyncMock(return_value=run)), \
             patch("src.api.v1.endpoints.circuit_discovery._run_sync",
                   new=_sync_bridge(run, effect=run_sync_effect)) as run_sync, \
             patch("src.workers.circuit_capture_tasks.capture_circuit_activations.delay",
                   return_value=MagicMock(id="task_cc")) as delay:
            kwargs = {} if json is None else {"json": json}
            response = client.post("/api/v1/circuit-capture/cap_test1/confirm", **kwargs)
        return response, delay, run_sync

    def test_a_card_named_at_confirm_replaces_the_estimates(self, client, cards):
        run = _capture(gpu_request=TI_UUID)
        r, delay, _ = self._confirm(client, run, json={"gpu": RTX_UUID})

        assert r.status_code == 202, r.text
        assert run.gpu_request == RTX_UUID, "the confirm's card was ignored"
        assert run.status == "pending"
        delay.assert_called_once_with("cap_test1", confirmed=True)

    def test_an_index_named_at_confirm_is_stored_as_its_uuid(self, client, cards):
        run = _capture(gpu_request=TI_UUID)
        r, delay, _ = self._confirm(client, run, json={"gpu": "1"})

        assert r.status_code == 202, r.text
        assert run.gpu_request == RTX_UUID
        delay.assert_called_once_with("cap_test1", confirmed=True)

    @pytest.mark.parametrize("body", [None, {}, {"gpu": None}], ids=["no-body", "empty", "null"])
    def test_a_confirm_without_a_card_keeps_the_estimates_request(self, client, cards, body):
        run = _capture(gpu_request=TI_UUID)
        r, delay, _ = self._confirm(client, run, json=body)

        assert r.status_code == 202, r.text
        assert run.gpu_request == TI_UUID
        delay.assert_called_once_with("cap_test1", confirmed=True)

    def test_an_unknown_card_named_at_confirm_is_a_400_and_changes_nothing(self, client, cards):
        run = _capture(gpu_request=TI_UUID)
        r, delay, run_sync = self._confirm(client, run, json={"gpu": MISSING_UUID})

        assert r.status_code == 400
        assert run.gpu_request == TI_UUID and run.status == "estimated"
        assert run_sync.await_count == 0
        delay.assert_not_called()

    def test_a_refused_confirm_leaves_the_stored_request_alone(self, client, cards):
        from src.services.circuit_capture_service import CaptureConflictError

        run = _capture(gpu_request=TI_UUID)
        r, delay, _ = self._confirm(
            client, run, json={"gpu": RTX_UUID},
            run_sync_effect=CaptureConflictError("another GPU circuit run is active"),
        )

        assert r.status_code == 409
        assert run.gpu_request == TI_UUID, "a refused confirm rewrote the request a retry would use"
        delay.assert_not_called()

    def test_a_stored_card_that_left_the_node_is_a_400(self, client, cards):
        run = _capture(gpu_request=MISSING_UUID)
        r, delay, run_sync = self._confirm(client, run)

        assert r.status_code == 400
        assert run.status == "estimated" and run_sync.await_count == 0
        delay.assert_not_called()

    def test_the_confirms_card_is_committed_before_the_task_is_dispatched(self, cards):
        """The worker places from the ROW, so the row must say the new card first.

        Round 1 of the Phase 1 review found the confirm assigning
        `run.gpu_request` and dispatching, then committing: an idle worker read
        the estimate's request and ran the full capture on the old card. The
        tests above only inspect the ORM object, which is right either way.

        MUTATION CONTROL (round 1, 2026-09-13): delete the commit before
        `.delay()` -> this test fails.
        """
        from src.core.database import get_db
        from src.main import app

        run = _capture(gpu_request=TI_UUID)
        events = []

        async def _commit():
            events.append(("commit", run.gpu_request))

        session = AsyncMock()
        session.commit = AsyncMock(side_effect=_commit)

        async def _override():
            yield session

        def _dispatch(*_args, **_kwargs):
            events.append(("dispatch", run.gpu_request))
            return MagicMock(id="task_cc")

        app.dependency_overrides[get_db] = _override
        try:
            with patch("src.api.v1.endpoints.circuit_discovery._capture_or_404",
                       new=AsyncMock(return_value=run)), \
                 patch("src.api.v1.endpoints.circuit_discovery._run_sync",
                       new=_sync_bridge(run, events=events,
                                        commit_event=lambda: ("commit", run.gpu_request))), \
                 patch("src.workers.circuit_capture_tasks.capture_circuit_activations.delay",
                       side_effect=_dispatch) as delay:
                r = TestClient(app, raise_server_exceptions=False).post(
                    "/api/v1/circuit-capture/cap_test1/confirm", json={"gpu": RTX_UUID})
        finally:
            app.dependency_overrides.pop(get_db, None)

        assert r.status_code == 202, r.text
        assert delay.call_count == 1
        dispatched_at = events.index(("dispatch", RTX_UUID))
        assert ("commit", RTX_UUID) in events[:dispatched_at], (
            f"the new card was not committed before the task could read it: {events}")


class TestConfirmNeverOverwritesTheWorkersStatus:
    """The status goes out BEFORE the dispatch; after it, only the task id.

    Review round 2 (2026-09-13). Round 1 committed the card before `.delay()`
    but still set `status = "pending"` after it. An idle worker starts within
    milliseconds, and one that fails at once commits "failed" — which the
    endpoint's flush then overwrote with "pending". The janitor never reaps a
    pending row whose task is terminal, so every circuit GPU run was refused
    with a 409 from then on.

    MUTATION CONTROLS (round 2, each red, restored):
      S1 `run.status = "pending"` moved back after `.delay()`
           -> test_a_failure_the_worker_commits_during_dispatch_survives fails
      S2 the post-dispatch column UPDATE writes `status="pending"` too
           -> the same test fails
      S3 no status reset when the dispatch raises
           -> test_a_dispatch_that_raises_puts_the_status_back fails
    """

    @staticmethod
    def _post(run, dispatch, events):
        from src.core.database import get_db
        from src.main import app

        session = AsyncMock()

        async def _commit():
            events.append(("commit", run.status))

        async def _execute(statement):
            events.append(("execute", dict(statement.compile().params)))

        session.commit = AsyncMock(side_effect=_commit)
        session.execute = AsyncMock(side_effect=_execute)

        async def _override():
            yield session

        app.dependency_overrides[get_db] = _override
        try:
            with patch("src.api.v1.endpoints.circuit_discovery._capture_or_404",
                       new=AsyncMock(return_value=run)), \
                 patch("src.api.v1.endpoints.circuit_discovery._run_sync",
                       new=_sync_bridge(run, events=events,
                                        commit_event=lambda: ("commit", run.status))), \
                 patch("src.workers.circuit_capture_tasks.capture_circuit_activations.delay",
                       side_effect=dispatch) as delay:
                response = TestClient(app, raise_server_exceptions=False).post(
                    "/api/v1/circuit-capture/cap_test1/confirm", json={"gpu": RTX_UUID})
        finally:
            app.dependency_overrides.pop(get_db, None)
        return response, delay

    def test_a_failure_the_worker_commits_during_dispatch_survives(self, cards):
        run = _capture(gpu_request=TI_UUID)
        events = []

        def _worker_fails_at_once(*_args, **_kwargs):
            events.append(("dispatch",))
            run.status = "failed"  # committed by the worker before we return
            return MagicMock(id="task_cc")

        r, delay = self._post(run, _worker_fails_at_once, events)

        assert r.status_code == 202, r.text
        assert delay.call_count == 1
        at = events.index(("dispatch",))
        assert ("commit", "pending") in events[:at], (
            f"'pending' was not committed before the worker could run: {events}")
        after = [e for e in events[at:] if e[0] == "execute"]
        assert len(after) == 1 and after[0][1].get("celery_task_id") == "task_cc", events
        assert "status" not in after[0][1], events
        assert run.status == "failed", (
            f"the endpoint overwrote the worker's status with {run.status!r}")

    def test_a_dispatch_that_raises_puts_the_status_back(self, cards):
        run = _capture(gpu_request=TI_UUID)
        events = []

        def _broker_down(*_args, **_kwargs):
            raise ConnectionError("broker unreachable")

        r, _delay = self._post(run, _broker_down, events)

        assert r.status_code == 500
        resets = [e[1] for e in events if e[0] == "execute"]
        assert len(resets) == 1 and resets[0].get("status") == "estimated", events
        assert events[-1][0] == "commit", events


class TestADoubleConfirmDispatchesOnce:
    """The guard and the mark share ONE locked transaction; a second confirm is a 409.

    Review round 2 (2026-09-13). Every other circuit GPU endpoint guards and
    marks in one advisory-locked transaction. Confirm committed the guard alone
    and marked the run in a second transaction, so a second confirm — a
    double-click, since the Run capture button is not disabled in flight —
    passed the guard before the first had marked the run, and the full capture
    was dispatched twice. The worker never re-checks the row's status.

    MUTATION CONTROLS (round 2, each red, restored byte-identically):
      D1 the status re-check under the lock deleted
           -> test_a_confirm_that_lost_the_race_is_refused fails
      D2 the mark moved back out of the locked transaction (the pre-fix shape:
         guard alone in `_run_sync`, status and card set on the async row)
           -> both tests fail
    """

    @staticmethod
    def _post(client, stale, run_sync):
        with patch("src.api.v1.endpoints.circuit_discovery._capture_or_404",
                   new=AsyncMock(return_value=stale)), \
             patch("src.api.v1.endpoints.circuit_discovery._run_sync",
                   new=run_sync), \
             patch("src.workers.circuit_capture_tasks.capture_circuit_activations.delay",
                   return_value=MagicMock(id="task_cc")) as delay:
            response = client.post("/api/v1/circuit-capture/cap_test1/confirm",
                                   json={"gpu": TI_UUID})
        return response, delay

    def test_a_confirm_that_lost_the_race_is_refused(self, client, cards):
        # What this request read before the lock: still "estimated".
        stale = _capture(gpu_request=TI_UUID)
        # What the row says under the lock: the first confirm has marked it.
        marked = _capture(gpu_request=RTX_UUID, status="pending")
        run_sync = _sync_bridge(marked)

        r, delay = self._post(client, stale, run_sync)

        assert r.status_code == 409, r.text
        assert run_sync.await_count == 1
        delay.assert_not_called()
        assert (marked.status, marked.gpu_request) == ("pending", RTX_UUID), (
            "the losing confirm rewrote the winner's row")

    def test_the_mark_commits_in_the_guards_transaction(self, client, cards):
        from src.services.circuit_capture_service import CircuitCaptureService

        run = _capture(gpu_request=RTX_UUID)
        events = []

        class _Recording(_Session):
            def commit(self):
                super().commit()
                events.append(("commit", run.status, run.gpu_request))

        async def _call(_db, fn):
            with patch.object(CircuitCaptureService, "assert_no_active_gpu_run",
                              lambda _sync_db, gpu_request=None: events.append(("guard", run.status))):
                return fn(_Recording(run))

        run_sync = AsyncMock(side_effect=_call)
        r, delay = self._post(client, _capture(gpu_request=RTX_UUID), run_sync)

        assert r.status_code == 202, r.text
        assert delay.call_count == 1
        assert run_sync.await_count == 1
        assert events == [("guard", "estimated"), ("commit", "pending", TI_UUID)], (
            f"the run was not marked inside the guard's transaction: {events}")


def _post_with_a_broker_down(path, body, row, patches):
    """POST a circuit GPU endpoint whose task dispatch raises, recording the async session.

    `_run_sync` RUNS the endpoint's guard-and-mark on `row`, so the value the
    restore must put back is the one the mark actually replaced.
    """
    from contextlib import ExitStack

    from src.core.database import get_db
    from src.main import app

    events = []
    session = AsyncMock()

    async def _execute(statement):
        events.append(("execute", dict(statement.compile().params)))

    async def _commit():
        events.append(("commit",))

    session.execute = AsyncMock(side_effect=_execute)
    session.commit = AsyncMock(side_effect=_commit)

    async def _override():
        yield session

    app.dependency_overrides[get_db] = _override
    try:
        with ExitStack() as stack:
            stack.enter_context(patch("src.api.v1.endpoints.circuit_discovery._run_sync",
                                      new=_sync_bridge(row)))
            for target, kwargs in patches:
                stack.enter_context(patch(target, **kwargs))
            response = TestClient(app, raise_server_exceptions=False).post(path, json=body)
    finally:
        app.dependency_overrides.pop(get_db, None)
    return response, events


def _broker_down():
    return {"side_effect": ConnectionError("broker unreachable")}


def _failed_dispatch_cases():
    """(id, path, body, row, patches, restored column, value it must be restored to)."""
    from datetime import datetime

    from src.models.validation_manifest import ValidationManifest

    discovery = _discovery()
    reproduced_run = _discovery()
    edge_manifest = ValidationManifest(id="vman_1", kind="edge_batch", discovery_run_id="dsc1",
                                       payload={"config": {"ordering": "coact"}},
                                       created_at=datetime(2026, 9, 13))
    faithful = _circuit()
    calibrated = _circuit()
    recalibrated = _circuit(calibration_status="completed")
    calibration_manifest = ValidationManifest(id="vman_k", kind="calibration",
                                              circuit_id="crc_g1",
                                              payload={"gpu": {"request": TI_UUID}},
                                              created_at=datetime(2026, 9, 13))
    attributed = _discovery(attribution_status="failed")
    created_capture = _capture(status="pending")
    created_discovery = _discovery()
    circuit_get = "src.api.v1.endpoints.circuits.CircuitService.get"
    manifest_get = "src.services.manifest_service.ManifestService.get"
    run_get = "src.api.v1.endpoints.circuit_validation._run_or_404"
    return [
        pytest.param("/api/v1/circuit-discovery/dsc1/validate", {"k": 5}, discovery,
                     [(run_get, {"new": AsyncMock(return_value=discovery)}),
                      ("src.workers.circuit_validation_tasks.validate_circuit_edges.delay",
                       _broker_down())],
                     "validation_status", None, id="validate"),
        pytest.param("/api/v1/validation-manifests/vman_1/reproduce", {}, reproduced_run,
                     [(manifest_get, {"new": AsyncMock(return_value=edge_manifest)}),
                      (run_get, {"new": AsyncMock(return_value=reproduced_run)}),
                      ("src.workers.circuit_validation_tasks.validate_circuit_edges.delay",
                       _broker_down())],
                     "validation_status", None, id="validation-reproduce"),
        pytest.param("/api/v1/circuits/crc_g1/faithfulness", {"mode": "both"}, faithful,
                     [(circuit_get, {"new": AsyncMock(return_value=faithful)}),
                      ("src.workers.circuit_validation_tasks.run_circuit_faithfulness.delay",
                       _broker_down())],
                     "faithfulness_status", None, id="faithfulness"),
        pytest.param("/api/v1/circuits/crc_g1/calibration", {"step_budget": 4}, calibrated,
                     [(circuit_get, {"new": AsyncMock(return_value=calibrated)}),
                      ("src.workers.circuit_calibration_tasks.run_circuit_calibration.delay",
                       _broker_down())],
                     "calibration_status", None, id="calibration"),
        pytest.param("/api/v1/circuits/calibration-manifests/vman_k/reproduce", {}, recalibrated,
                     [(manifest_get, {"new": AsyncMock(return_value=calibration_manifest)}),
                      (circuit_get, {"new": AsyncMock(return_value=recalibrated)}),
                      ("src.workers.circuit_calibration_tasks.reproduce_circuit_calibration.delay",
                       _broker_down())],
                     "calibration_status", "completed", id="calibration-reproduce"),
        # A callable: the cases are built when the class is defined, and
        # RECORD_BODY is defined further down this module.
        pytest.param("/api/v1/circuits/steering-samples", lambda: RECORD_BODY, _circuit(),
                     [("src.workers.circuit_record_tasks.run_circuit_record.delay",
                       _broker_down())],
                     "status", "failed", id="recorder"),
        # Review round 4: the round-3 follow-up missed these two. The attribution
        # row starts "failed" so the restored value cannot agree with a None by
        # construction.
        pytest.param("/api/v1/circuit-discovery/dsc1/attribution", {"prompt_limit": 8},
                     attributed,
                     [("src.api.v1.endpoints.circuit_discovery._discovery_or_404",
                       {"new": AsyncMock(return_value=attributed)}),
                      ("src.workers.circuit_capture_tasks.run_circuit_attribution.delay",
                       _broker_down())],
                     "attribution_status", "failed", id="attribution"),
        pytest.param("/api/v1/circuit-capture", lambda: CAPTURE_BODY, created_capture,
                     [("src.services.circuit_capture_service.CircuitCaptureService.create_run",
                       {"return_value": created_capture}),
                      ("src.workers.circuit_capture_tasks.capture_circuit_activations.delay",
                       _broker_down())],
                     "status", "failed", id="capture"),
        # Not a GPU job, but a pending discovery refuses every later discovery
        # on the same capture store, so it gets the same restore.
        pytest.param("/api/v1/circuit-discovery", {"capture_run_id": "cap1"}, created_discovery,
                     [("src.services.circuit_discovery_service.CircuitDiscoveryService.create_run",
                       {"return_value": created_discovery}),
                      ("src.workers.circuit_capture_tasks.run_circuit_discovery.delay",
                       _broker_down())],
                     "status", "failed", id="discovery"),
    ]


class TestAFailedDispatchLeavesNothingPending:
    """A dispatch that raises must not leave a run marked in flight.

    Review round 3, 2026-09-14. Each endpoint marks its run 'pending' inside the
    guard's locked transaction and then calls `.delay()`. With the broker down
    the dispatch raises, no worker ever moves the mark, and a 'pending' row makes
    `assert_no_active_gpu_run` refuse EVERY circuit GPU job with a 409 until the
    database is edited by hand. Round 3 named reproduce and calibration; the same
    mark-then-dispatch shape was at validate, faithfulness and the recorder, so
    all six share `_dispatch_or_restore`. (Capture confirm has had its own
    restore since round 2; see TestConfirmNeverOverwritesTheWorkersStatus.)

    MUTATION CONTROLS (2026-09-14, each verified red, restored byte-identically):
      F1 `_dispatch_or_restore` re-raises without restoring   -> all six fail
      F2 validate calls `.delay()` directly, not via the helper -> [validate] fails
      F3 the recorder's restore writes status "pending"         -> [recorder] fails

    Review round 4 added [attribution], [capture] and [discovery], which still
    dispatched directly. Controls (2026-09-14): each restore writing "pending"
    -> its own case fails.
    """

    @pytest.mark.parametrize("path, body, row, patches, column, restored", _failed_dispatch_cases())
    def test_the_in_flight_mark_is_put_back(self, cards, path, body, row, patches, column, restored):
        response, events = _post_with_a_broker_down(
            path, body() if callable(body) else body, row, patches)

        assert response.status_code == 500, response.text
        restores = [params for kind, *rest in events if kind == "execute" for params in rest]
        assert len(restores) == 1, f"expected one restoring UPDATE, got {events}"
        assert column in restores[0] and restores[0][column] == restored, restores[0]
        assert events[-1] == ("commit",), f"the restore was never committed: {events}"


class TestCaptureWorker:
    def test_places_with_the_rows_request_and_records_the_card_before_loading(self, placed):
        from src.services.circuit_capture_service import CircuitCaptureService
        from src.workers import circuit_capture_tasks as tasks

        row = SimpleNamespace(id="cap_1", gpu_request=RTX_UUID, gpu_uuid=None,
                              status="pending", error_message=None)
        session = _Session(row)
        seen = []

        def fake_run_capture(db, run_id, *, confirmed, device, placement=None,
                             cancel_check=None, progress_cb=None):
            seen.append((device, row.gpu_uuid, session.commits))
            return {"status": "estimated"}

        with patch.object(CircuitCaptureService, "run_capture", side_effect=fake_run_capture), \
             patch.object(tasks, "emit_circuit_run_completed"):
            _raw(tasks.capture_circuit_activations)(_fake_task(session), "cap_1")

        assert placed == [RTX_UUID]
        assert len(seen) == 1
        device, uuid_at_load, commits_at_load = seen[0]
        assert device == RTX_DEVICE
        assert uuid_at_load == RTX_UUID and commits_at_load >= 1, (
            "the card must be recorded AND committed before the model load")

    def test_a_row_from_before_gpu_choice_is_placed_as_auto(self, placed):
        from src.services.circuit_capture_service import CircuitCaptureService
        from src.workers import circuit_capture_tasks as tasks

        row = SimpleNamespace(id="cap_1", gpu_request=None, gpu_uuid=None, status="pending")
        with patch.object(CircuitCaptureService, "run_capture", return_value={}), \
             patch.object(tasks, "emit_circuit_run_completed"):
            _raw(tasks.capture_circuit_activations)(_fake_task(_Session(row)), "cap_1")
        assert placed == ["auto"]

    def test_a_placement_refusal_fails_the_run_with_its_message(self, refused):
        from src.services.circuit_capture_service import CircuitCaptureService
        from src.workers import circuit_capture_tasks as tasks

        row = SimpleNamespace(id="cap_1", gpu_request=TI_UUID, gpu_uuid=None,
                              status="pending", error_message=None)
        with patch.object(CircuitCaptureService, "run_capture") as run_capture, \
             patch.object(tasks, "emit_circuit_run_failed") as failed:
            with pytest.raises(GpuPlacementError):
                _raw(tasks.capture_circuit_activations)(_fake_task(_Session(row)), "cap_1")

        run_capture.assert_not_called()
        assert row.status == "failed" and REFUSAL in row.error_message
        assert row.gpu_uuid is None
        assert REFUSAL[:100] in failed.call_args.args[2]


class TestCaptureServiceUsesTheDevice:
    def test_the_model_is_loaded_onto_the_placed_device(self):
        from src.services import circuit_capture_service as cap

        manifest = {"model_id": "m_1",
                    "corpus": {"tokenization_id": "tok_1", "sample_cap": 4},
                    "layers": [{"layer": 4, "sae_id": "sae_a", "epsilon": 0.1,
                                "theta_floor": 0.01}]}
        row = MagicMock(manifest=manifest, tokenized_path="datasets/tok_1",
                        file_path=None, quantization="FP16")
        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = row
        seen = {}

        def fake_load(**kwargs):
            seen.update(kwargs)
            raise _Stop

        ds = MagicMock()
        ds.__len__ = lambda self: 4
        ds.select = lambda r: []
        with patch("datasets.load_from_disk", return_value=ds), \
             patch("src.ml.model_loader.load_model_from_hf", side_effect=fake_load):
            with pytest.raises(_Stop):
                cap.CircuitCaptureService.run_capture(
                    db, "cap_1", confirmed=True, device=RTX_DEVICE)
        assert seen["device_map"] == RTX_DEVICE


# ═════════════════════════════ ATTRIBUTION ════════════════════════════════


class TestAttributionEndpoint:
    def _post(self, client, body):
        with patch("src.api.v1.endpoints.circuit_discovery._discovery_or_404",
                   new=AsyncMock(return_value=_discovery())), \
             patch("src.api.v1.endpoints.circuit_discovery._run_sync",
                   new=AsyncMock(return_value=None)) as run_sync, \
             patch("src.workers.circuit_capture_tasks.run_circuit_attribution.delay",
                   return_value=MagicMock(id="task_a")) as delay:
            response = client.post("/api/v1/circuit-discovery/dsc1/attribution", json=body)
        return response, delay, run_sync

    def test_the_resolved_request_travels_with_the_task(self, client, cards):
        r, delay, _ = self._post(client, {"prompt_limit": 8, "gpu": "1"})

        assert r.status_code == 202, r.text
        delay.assert_called_once_with("dsc1", prompt_limit=8, gpu_request=RTX_UUID)

    def test_an_unknown_card_is_a_400_with_nothing_marked_or_dispatched(self, client, cards):
        r, delay, run_sync = self._post(client, {"gpu": MISSING_UUID})

        assert r.status_code == 400
        assert run_sync.await_count == 0
        delay.assert_not_called()


class TestAttributionWorker:
    def test_places_with_the_request_and_hands_over_the_device(self, placed):
        from src.services.circuit_attribution_service import CircuitAttributionService
        from src.workers import circuit_capture_tasks as tasks

        row = SimpleNamespace(id="dsc1", attribution_status="pending")
        with patch.object(CircuitAttributionService, "run", return_value={"status": "completed"}) as run, \
             patch.object(tasks, "emit_circuit_run_completed"):
            _raw(tasks.run_circuit_attribution)(
                _fake_task(_Session(row)), "dsc1", prompt_limit=4, gpu_request=RTX_UUID)

        assert placed == [RTX_UUID]
        assert run.call_count == 1
        assert run.call_args.kwargs["device"] == RTX_DEVICE
        assert run.call_args.kwargs["gpu"] == RTX_RECORD

    def test_a_placement_refusal_fails_the_pass_with_its_message(self, refused):
        from src.services.circuit_attribution_service import CircuitAttributionService
        from src.workers import circuit_capture_tasks as tasks

        row = SimpleNamespace(id="dsc1", attribution_status="pending", attribution_error=None)
        with patch.object(CircuitAttributionService, "run") as run, \
             patch.object(tasks, "emit_circuit_run_failed"):
            with pytest.raises(GpuPlacementError):
                _raw(tasks.run_circuit_attribution)(
                    _fake_task(_Session(row)), "dsc1", gpu_request=TI_UUID)

        run.assert_not_called()
        assert row.attribution_status == "failed" and REFUSAL in row.attribution_error


class TestAttributionServiceUsesTheDevice:
    def test_peak_stats_and_the_load_use_the_placed_device(self, monkeypatch):
        from src.services.circuit_attribution_service import CircuitAttributionService

        resets = []
        monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats",
                            lambda device=None: resets.append(device))
        row = MagicMock(
            candidates=[{"up": {"layer": 1, "feature_idx": 2},
                         "down": {"layer": 3, "feature_idx": 4}}],
            params={}, capture_run_id="cap1", tokenized_path="datasets/t",
            file_path=None, quantization="FP16", repo_id="org/m",
            manifest={"model_id": "m_1", "corpus": {"tokenization_id": "t"}, "layers": []})
        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = row
        seen = {}

        def fake_load(**kwargs):
            seen.update(kwargs)
            raise _Stop

        ds = MagicMock()
        ds.__len__ = lambda self: 4
        ds.select = lambda r: ds
        with patch("datasets.load_from_disk", return_value=ds), \
             patch("src.ml.model_loader.load_model_from_hf", side_effect=fake_load), \
             patch("src.services.extraction_service.cleanup_gpu_memory"):
            with pytest.raises(_Stop):
                CircuitAttributionService.run(db, "dsc1", device=RTX_DEVICE, gpu=RTX_RECORD)

        assert resets == [RTX_DEVICE], "peak stats were reset on some other device"
        assert seen["device_map"] == RTX_DEVICE


# ═════════════════════════════ VALIDATION ═════════════════════════════════


class TestValidationEndpoint:
    def _post(self, client, body):
        with patch("src.api.v1.endpoints.circuit_validation._run_or_404",
                   new=AsyncMock(return_value=_discovery())), \
             patch("src.api.v1.endpoints.circuit_discovery._run_sync",
                   new=AsyncMock(return_value=None)) as run_sync, \
             patch("src.workers.circuit_validation_tasks.validate_circuit_edges.delay",
                   return_value=MagicMock(id="task_v")) as delay:
            response = client.post("/api/v1/circuit-discovery/dsc1/validate", json=body)
        return response, delay, run_sync

    def test_the_resolved_request_travels_with_the_task(self, client, cards):
        r, delay, _ = self._post(client, {"k": 5, "gpu": RTX_UUID.lower()})

        assert r.status_code == 202, r.text
        assert delay.call_count == 1
        (run_id, scope), kwargs = delay.call_args
        assert run_id == "dsc1" and kwargs == {"gpu_request": RTX_UUID}
        assert scope["k"] == 5 and "gpu" not in scope

    def test_an_unknown_card_is_a_400_with_nothing_marked_or_dispatched(self, client, cards):
        r, delay, run_sync = self._post(client, {"gpu": MISSING_UUID})

        assert r.status_code == 400
        assert run_sync.await_count == 0
        delay.assert_not_called()


class TestValidationReproduceReusesTheRecordedRequest:
    """Reproduce reuses the original's request unless the caller names a card.

    The Manifest drawer has a GPU picker and sends `{"gpu": ...}`. Until the
    Phase 1 round-1 review the endpoint took no body, so the picker's choice was
    silently dropped; the frontend test mocked the API and stayed green.

    MUTATION CONTROLS (round 1, 2026-09-13, each red, restored):
      R1 resolve `_manifest_gpu_request(m)` and ignore `body.gpu`
           -> test_a_card_named_at_reproduce_replaces_the_recorded_request fails
      R2 prefer the manifest's request over `body.gpu`
           -> test_a_card_named_at_reproduce_replaces_the_recorded_request fails
    """

    def _reproduce(self, client, payload, json=None):
        from datetime import datetime

        from src.models.validation_manifest import ValidationManifest

        manifest = ValidationManifest(id="vman_1", kind="edge_batch", discovery_run_id="dsc1",
                                      payload=payload, created_at=datetime(2026, 9, 13))
        with patch("src.services.manifest_service.ManifestService.get",
                   new=AsyncMock(return_value=manifest)), \
             patch("src.api.v1.endpoints.circuit_validation._run_or_404",
                   new=AsyncMock(return_value=_discovery())), \
             patch("src.api.v1.endpoints.circuit_discovery._run_sync",
                   new=AsyncMock(return_value=None)) as run_sync, \
             patch("src.workers.circuit_validation_tasks.validate_circuit_edges.delay",
                   return_value=MagicMock(id="task_r")) as delay:
            kwargs = {} if json is None else {"json": json}
            response = client.post("/api/v1/validation-manifests/vman_1/reproduce", **kwargs)
        return response, delay, run_sync

    RECORDED_ON_TI = {
        "config": {"ordering": "coact"},
        "gpu": {"request": TI_UUID, "uuid": TI_UUID, "name": "NVIDIA GeForce RTX 3080 Ti"},
    }

    def test_a_card_named_at_reproduce_replaces_the_recorded_request(self, client, cards):
        r, delay, _ = self._reproduce(client, self.RECORDED_ON_TI, json={"gpu": RTX_UUID})

        assert r.status_code == 202, r.text
        delay.assert_called_once_with(
            "dsc1", {"ordering": "coact", "reproduce_of": "vman_1"}, gpu_request=RTX_UUID)

    def test_an_index_named_at_reproduce_is_sent_as_its_uuid(self, client, cards):
        r, delay, _ = self._reproduce(client, self.RECORDED_ON_TI, json={"gpu": "1"})

        assert r.status_code == 202, r.text
        assert delay.call_args.kwargs == {"gpu_request": RTX_UUID}
        assert delay.call_count == 1

    @pytest.mark.parametrize("body", [{}, {"gpu": None}], ids=["empty", "null"])
    def test_a_reproduce_without_a_card_keeps_the_recorded_request(self, client, cards, body):
        r, delay, _ = self._reproduce(client, self.RECORDED_ON_TI, json=body)

        assert r.status_code == 202, r.text
        assert delay.call_args.kwargs == {"gpu_request": TI_UUID}
        assert delay.call_count == 1

    def test_an_unknown_card_named_at_reproduce_is_a_400_with_nothing_marked(self, client, cards):
        r, delay, run_sync = self._reproduce(client, self.RECORDED_ON_TI, json={"gpu": MISSING_UUID})

        assert r.status_code == 400
        assert run_sync.await_count == 0
        delay.assert_not_called()

    def test_the_original_request_is_reused(self, client, cards):
        r, delay, _ = self._reproduce(client, {
            "config": {"ordering": "coact"},
            "gpu": {"request": TI_UUID, "uuid": TI_UUID, "name": "NVIDIA GeForce RTX 3080 Ti"}})

        assert r.status_code == 202, r.text
        delay.assert_called_once_with(
            "dsc1", {"ordering": "coact", "reproduce_of": "vman_1"}, gpu_request=TI_UUID)

    def test_a_manifest_from_before_gpu_choice_reproduces_on_auto(self, client, cards):
        r, delay, _ = self._reproduce(client, {"config": {"ordering": "coact"}})
        assert r.status_code == 202, r.text
        assert delay.call_args.kwargs == {"gpu_request": "auto"}

    def test_a_recorded_card_that_left_the_node_is_a_400(self, client, cards):
        r, delay, run_sync = self._reproduce(client, {
            "config": {}, "gpu": {"request": MISSING_UUID}})
        assert r.status_code == 400
        assert run_sync.await_count == 0
        delay.assert_not_called()


class TestValidationWorker:
    def test_places_with_the_request_and_hands_over_the_device(self, placed):
        from src.services.circuit_intervention_service import CircuitInterventionService
        from src.workers import circuit_validation_tasks as tasks

        row = SimpleNamespace(id="dsc1", validation_status="pending")
        with patch.object(CircuitInterventionService, "run", return_value={"status": "completed"}) as run, \
             patch.object(tasks, "emit_circuit_run_completed"):
            _raw(tasks.validate_circuit_edges)(
                _fake_task(_Session(row)), "dsc1", {"k": 1}, gpu_request=RTX_UUID)

        assert placed == [RTX_UUID]
        assert run.call_count == 1
        assert run.call_args.kwargs["device"] == RTX_DEVICE
        assert run.call_args.kwargs["gpu"] == RTX_RECORD

    def test_a_placement_refusal_fails_the_pass_with_its_message(self, refused):
        from src.services.circuit_intervention_service import CircuitInterventionService
        from src.workers import circuit_validation_tasks as tasks

        row = SimpleNamespace(id="dsc1", validation_status="pending", validation_error=None)
        with patch.object(CircuitInterventionService, "run") as run, \
             patch.object(tasks, "emit_circuit_run_failed"):
            with pytest.raises(GpuPlacementError):
                _raw(tasks.validate_circuit_edges)(
                    _fake_task(_Session(row)), "dsc1", {}, gpu_request=TI_UUID)

        run.assert_not_called()
        assert row.validation_status == "failed" and REFUSAL in row.validation_error


class TestInterventionServiceUsesTheDevice:
    def test_the_model_is_loaded_onto_the_placed_device(self):
        from src.services.circuit_intervention_service import CircuitInterventionService

        row = MagicMock(
            store_path="circuit_captures/cap1", tokenized_path="datasets/t",
            file_path=None, quantization="FP16", repo_id="org/m",
            candidates=[{"up": {"layer": 1, "feature_idx": 2},
                         "down": {"layer": 3, "feature_idx": 4},
                         "orderings": {"coact_rank": 0}}],
            manifest={"model_id": "m_1", "corpus": {"tokenization_id": "t"}, "layers": []})
        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = row
        seen = {}

        def fake_load(**kwargs):
            seen.update(kwargs)
            raise _Stop

        scope = CircuitInterventionService.create_scope({"k": 1})
        with patch("datasets.load_from_disk", return_value=MagicMock()), \
             patch("src.ml.model_loader.load_model_from_hf", side_effect=fake_load), \
             patch("src.services.extraction_service.cleanup_gpu_memory"):
            with pytest.raises(_Stop):
                CircuitInterventionService.run(db, "dsc1", scope, device=RTX_DEVICE)
        assert seen["device_map"] == RTX_DEVICE


# ═════════════════════════════ FAITHFULNESS ═══════════════════════════════


class TestFaithfulnessEndpoint:
    def _post(self, client, body):
        with patch("src.api.v1.endpoints.circuits.CircuitService.get",
                   new=AsyncMock(return_value=_circuit())), \
             patch("src.api.v1.endpoints.circuit_discovery._run_sync",
                   new=AsyncMock(return_value=None)) as run_sync, \
             patch("src.workers.circuit_validation_tasks.run_circuit_faithfulness.delay",
                   return_value=MagicMock(id="task_f")) as delay:
            response = client.post("/api/v1/circuits/crc_g1/faithfulness", json=body)
        return response, delay, run_sync

    def test_the_resolved_request_travels_with_the_task(self, client, cards):
        r, delay, _ = self._post(client, {"mode": "both", "gpu": "0"})

        assert r.status_code == 202, r.text
        assert delay.call_count == 1
        (circuit_id, config), kwargs = delay.call_args
        assert circuit_id == "crc_g1" and kwargs == {"gpu_request": TI_UUID}
        assert config["mode"] == "both" and "gpu" not in config

    def test_an_unknown_card_is_a_400_with_nothing_marked_or_dispatched(self, client, cards):
        r, delay, run_sync = self._post(client, {"gpu": MISSING_UUID})

        assert r.status_code == 400
        assert run_sync.await_count == 0
        delay.assert_not_called()


class TestFaithfulnessWorker:
    def test_places_with_the_request_and_hands_over_the_device(self, placed):
        from src.services.circuit_faithfulness_service import CircuitFaithfulnessService
        from src.workers import circuit_validation_tasks as tasks

        row = SimpleNamespace(id="crc_g1", faithfulness_status="pending")
        with patch.object(CircuitFaithfulnessService, "run", return_value={"status": "completed"}) as run, \
             patch.object(tasks, "emit_circuit_run_completed"):
            _raw(tasks.run_circuit_faithfulness)(
                _fake_task(_Session(row)), "crc_g1", {}, gpu_request=RTX_UUID)

        assert placed == [RTX_UUID]
        assert run.call_count == 1
        assert run.call_args.kwargs["device"] == RTX_DEVICE
        assert run.call_args.kwargs["gpu"] == RTX_RECORD

    def test_a_placement_refusal_fails_the_pass_with_its_message(self, refused):
        from src.services.circuit_faithfulness_service import CircuitFaithfulnessService
        from src.workers import circuit_validation_tasks as tasks

        row = SimpleNamespace(id="crc_g1", faithfulness_status="pending")
        with patch.object(CircuitFaithfulnessService, "run") as run, \
             patch.object(tasks, "emit_circuit_run_failed") as failed:
            with pytest.raises(GpuPlacementError):
                _raw(tasks.run_circuit_faithfulness)(
                    _fake_task(_Session(row)), "crc_g1", {}, gpu_request=TI_UUID)

        run.assert_not_called()
        assert row.faithfulness_status == "failed"
        assert REFUSAL[:100] in failed.call_args.args[2]


class TestFaithfulnessServiceUsesTheDevice:
    def test_the_model_is_loaded_onto_the_placed_device(self):
        from src.services.circuit_faithfulness_service import CircuitFaithfulnessService

        row = MagicMock(
            members=[{"layer": 3, "member_kind": "feature_ref", "feature": {"feature_idx": 4}}],
            discovery_run_id="dsc1", capture_run_id="cap1",
            store_path="circuit_captures/cap1", tokenized_path="datasets/t",
            file_path=None, quantization="FP16", repo_id="org/m",
            manifest={"model_id": "m_1", "corpus": {"tokenization_id": "t"},
                      "layers": [{"layer": 3, "sae_id": "sae_3"}]})
        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = row
        seen = {}

        def fake_load(**kwargs):
            seen.update(kwargs)
            raise _Stop

        with patch("datasets.load_from_disk", return_value=MagicMock()), \
             patch("src.services.circuit_capture_store.EventReader"), \
             patch.object(CircuitFaithfulnessService, "_select_prompts", return_value=[0]), \
             patch("src.ml.model_loader.load_model_from_hf", side_effect=fake_load), \
             patch("src.services.extraction_service.cleanup_gpu_memory"):
            with pytest.raises(_Stop):
                CircuitFaithfulnessService.run(db, "crc_g1", {}, device=RTX_DEVICE)
        assert seen["device_map"] == RTX_DEVICE


# ═════════════════════════════ CALIBRATION ════════════════════════════════


class TestCalibrationEndpoint:
    def _post(self, client, body):
        with patch("src.api.v1.endpoints.circuits.CircuitService.get",
                   new=AsyncMock(return_value=_circuit())), \
             patch("src.api.v1.endpoints.circuit_discovery._run_sync",
                   new=AsyncMock(return_value=None)) as run_sync, \
             patch("src.workers.circuit_calibration_tasks.run_circuit_calibration.delay",
                   return_value=MagicMock(id="task_k")) as delay:
            response = client.post("/api/v1/circuits/crc_g1/calibration", json=body)
        return response, delay, run_sync

    def test_the_resolved_request_travels_with_the_task(self, client, cards):
        r, delay, _ = self._post(client, {"step_budget": 4, "gpu": "1"})

        assert r.status_code == 202, r.text
        assert delay.call_count == 1
        (circuit_id, config), kwargs = delay.call_args
        assert circuit_id == "crc_g1" and kwargs == {"gpu_request": RTX_UUID}
        assert config["step_budget"] == 4 and "gpu" not in config

    def test_an_unknown_card_is_a_400_with_nothing_marked_or_dispatched(self, client, cards):
        r, delay, run_sync = self._post(client, {"gpu": MISSING_UUID})

        assert r.status_code == 400
        assert run_sync.await_count == 0
        delay.assert_not_called()


class TestCalibrationReproduceReusesTheRecordedRequest:
    """MUTATION CONTROL (review round 3, 2026-09-14), verified red then restored:
      * `chosen` ignores `body.gpu` (the stored request only)
           -> test_a_card_named_at_reproduce_replaces_the_recorded_request fails
    """

    def _reproduce(self, client, payload, json=None):
        from datetime import datetime

        from src.models.validation_manifest import ValidationManifest

        manifest = ValidationManifest(id="vman_k", kind="calibration", circuit_id="crc_g1",
                                      payload=payload, created_at=datetime(2026, 9, 13))
        with patch("src.services.manifest_service.ManifestService.get",
                   new=AsyncMock(return_value=manifest)), \
             patch("src.api.v1.endpoints.circuits.CircuitService.get",
                   new=AsyncMock(return_value=_circuit(calibration_status="completed"))), \
             patch("src.api.v1.endpoints.circuit_discovery._run_sync",
                   new=AsyncMock(return_value=None)) as run_sync, \
             patch("src.workers.circuit_calibration_tasks.reproduce_circuit_calibration.delay",
                   return_value=MagicMock(id="task_kr")) as delay:
            kwargs = {} if json is None else {"json": json}
            response = client.post("/api/v1/circuits/calibration-manifests/vman_k/reproduce",
                                   **kwargs)
        return response, delay, run_sync

    def test_the_original_request_is_reused(self, client, cards):
        r, delay, _ = self._reproduce(client, {"gpu": {"request": TI_UUID}})

        assert r.status_code == 202, r.text
        delay.assert_called_once_with("vman_k", "completed", gpu_request=TI_UUID)

    @pytest.mark.parametrize("body", [{}, {"gpu": None}])
    def test_a_body_without_a_card_keeps_the_recorded_request(self, client, cards, body):
        r, delay, _ = self._reproduce(client, {"gpu": {"request": TI_UUID}}, json=body)

        assert r.status_code == 202, r.text
        delay.assert_called_once_with("vman_k", "completed", gpu_request=TI_UUID)

    def test_a_card_named_at_reproduce_replaces_the_recorded_request(self, client, cards):
        r, delay, _ = self._reproduce(client, {"gpu": {"request": TI_UUID}}, json={"gpu": "1"})

        assert r.status_code == 202, r.text
        delay.assert_called_once_with("vman_k", "completed", gpu_request=RTX_UUID)

    def test_an_unknown_card_named_at_reproduce_is_a_400_with_nothing_marked(self, client, cards):
        r, delay, run_sync = self._reproduce(
            client, {"gpu": {"request": TI_UUID}}, json={"gpu": MISSING_UUID})

        assert r.status_code == 400
        assert run_sync.await_count == 0
        delay.assert_not_called()

    def test_a_recorded_card_that_left_the_node_is_a_400(self, client, cards):
        r, delay, run_sync = self._reproduce(client, {"gpu": {"request": MISSING_UUID}})
        assert r.status_code == 400
        assert run_sync.await_count == 0
        delay.assert_not_called()


class TestCalibrationWorker:
    def test_places_with_the_request_and_hands_over_the_device(self, placed):
        from src.services.circuit_calibration_service import CircuitCalibrationService
        from src.workers import circuit_calibration_tasks as tasks

        row = SimpleNamespace(id="crc_g1", calibration_status="pending")
        with patch.object(CircuitCalibrationService, "run", return_value={"band": None}) as run, \
             patch.object(tasks, "emit_circuit_run_completed"):
            _raw(tasks.run_circuit_calibration)(
                _fake_task(_Session(row)), "crc_g1", {}, gpu_request=RTX_UUID)

        assert placed == [RTX_UUID]
        assert run.call_count == 1
        assert run.call_args.kwargs["device"] == RTX_DEVICE
        assert run.call_args.kwargs["gpu"] == RTX_RECORD

    def test_reproduce_places_with_the_request_it_was_given(self, placed):
        from src.services.circuit_calibration_service import CircuitCalibrationService
        from src.workers import circuit_calibration_tasks as tasks

        row = SimpleNamespace(id="vman_k", circuit_id="crc_g1", calibration_status="pending")
        with patch.object(CircuitCalibrationService, "reproduce", return_value={}) as reproduce, \
             patch.object(tasks, "emit_circuit_run_completed"):
            _raw(tasks.reproduce_circuit_calibration)(
                _fake_task(_Session(row)), "vman_k", "completed", gpu_request=RTX_UUID)

        assert placed == [RTX_UUID]
        assert reproduce.call_count == 1
        assert reproduce.call_args.kwargs["device"] == RTX_DEVICE
        assert row.calibration_status == "completed"

    def test_a_placement_refusal_fails_the_pass_with_its_message(self, refused):
        from src.services.circuit_calibration_service import CircuitCalibrationService
        from src.workers import circuit_calibration_tasks as tasks

        row = SimpleNamespace(id="crc_g1", calibration_status="pending")
        with patch.object(CircuitCalibrationService, "run") as run, \
             patch.object(tasks, "emit_circuit_run_failed") as failed:
            with pytest.raises(GpuPlacementError):
                _raw(tasks.run_circuit_calibration)(
                    _fake_task(_Session(row)), "crc_g1", {}, gpu_request=TI_UUID)

        run.assert_not_called()
        assert row.calibration_status == "failed"
        assert REFUSAL[:100] in failed.call_args.args[2]


class TestCalibrationServiceUsesTheDevice:
    def test_run_hands_the_device_on_and_records_the_card_in_the_manifest(self):
        from src.services.circuit_calibration_service import CircuitCalibrationService

        circuit = SimpleNamespace(id="crc_g1", members=[{"layer": 1}])
        seen = {}

        def fake_fns(circuit_, db, cfg, *, device):
            seen["device"] = device
            return (None, None, None, None, None)

        persisted = {}
        with patch.object(CircuitCalibrationService, "_build_generation_fns", side_effect=fake_fns), \
             patch.object(CircuitCalibrationService, "build_band", return_value={
                 "manifest_payload": {"probes": []}, "judge_reliable": True,
                 "usable_band": False}), \
             patch.object(CircuitCalibrationService, "_persist_manifest",
                          side_effect=lambda db, cid, payload: persisted.update(payload) or "vman_k"), \
             patch.object(CircuitCalibrationService, "_mark_no_usable_band"):
            CircuitCalibrationService.run(_Session(circuit), "crc_g1", {},
                                          device=RTX_DEVICE, gpu=RTX_RECORD)

        assert seen["device"] == RTX_DEVICE
        assert persisted["gpu"] == RTX_RECORD

    def test_the_generation_fns_load_the_model_onto_the_device(self, monkeypatch):
        from src.services import steering_core
        from src.services.circuit_calibration_service import CircuitCalibrationService

        seen = {}

        def fake_load(model_id, db, device):
            seen["device"] = device
            raise _Stop

        monkeypatch.setattr(steering_core, "load_model_and_structure", fake_load)
        with pytest.raises(_Stop):
            CircuitCalibrationService._build_generation_fns(
                SimpleNamespace(model_id="m_1"), None,
                {"seed": 0, "judge_endpoint": "http://judge/v1", "judge_model": "j"},
                device=RTX_DEVICE)
        assert seen["device"] == RTX_DEVICE


class TestSteeringCoreLoadsOntoTheDevice:
    def test_the_loader_is_given_the_device(self):
        from src.services.steering_core import load_model_and_structure

        model_rec = SimpleNamespace(repo_id="org/m", quantization="FP16", file_path=None)
        seen = {}

        def fake_load(**kwargs):
            seen.update(kwargs)
            raise _Stop

        with patch("src.ml.model_loader.load_model_from_hf", side_effect=fake_load):
            with pytest.raises(_Stop):
                load_model_and_structure("m_1", _Session(model_rec), RTX_DEVICE)
        assert seen["device_map"] == RTX_DEVICE


# ═════════════════════════════ RECORDER ═══════════════════════════════════


RECORD_BODY = {"artifact": {"kind": "circuit", "circuit_id": "crc_g1"},
               "dials": [0.5], "prompts": ["hello"]}


class TestRecorderEndpoint:
    def _post(self, client, body):
        from src.models.steering_record_run import SteeringRecordRun
        from src.services.circuit_capture_service import CircuitCaptureService

        added = []
        sync_db = MagicMock()
        sync_db.add.side_effect = added.append

        async def run_sync(db, fn):
            return fn(sync_db)

        run_sync_spy = AsyncMock(side_effect=run_sync)
        with patch.object(CircuitCaptureService, "assert_no_active_gpu_run"), \
             patch("src.api.v1.endpoints.circuit_discovery._run_sync", new=run_sync_spy), \
             patch("src.workers.circuit_record_tasks.run_circuit_record.delay",
                   return_value=MagicMock(id="task_s")) as delay:
            response = client.post("/api/v1/circuits/steering-samples", json=body)
        rows = [a for a in added if isinstance(a, SteeringRecordRun)]
        return response, rows, delay, run_sync_spy

    def test_the_resolved_request_is_stored_on_the_row(self, client, cards):
        r, rows, delay, _ = self._post(client, {**RECORD_BODY, "gpu": "1"})

        assert r.status_code == 202, r.text
        assert len(rows) == 1 and rows[0].gpu_request == RTX_UUID
        assert delay.call_count == 1
        assert "gpu" not in delay.call_args.args[1]

    def test_an_unknown_card_is_a_400_with_no_row_and_no_dispatch(self, client, cards):
        r, rows, delay, run_sync = self._post(client, {**RECORD_BODY, "gpu": MISSING_UUID})

        assert r.status_code == 400
        assert rows == [] and run_sync.await_count == 0
        delay.assert_not_called()


class TestRecorderWorker:
    def _row(self, **kw):
        return SimpleNamespace(id="srr_1", status="pending", gpu_uuid=None, error=None,
                               manifest_ref=None, **kw)

    def test_places_with_the_rows_request_and_records_the_card_before_loading(self, placed):
        from src.services.steering_recorder_service import SteeringRecorderService
        from src.workers import circuit_record_tasks as tasks

        row = self._row(gpu_request=RTX_UUID)
        session = _Session(row)
        seen = []

        def fake_record(db, config, *, progress_cb=None, cancel_check=None, run_id=None,
                        device, gpu=None):
            seen.append((device, gpu, row.gpu_uuid, session.commits))
            return {"manifest_ref": "vman_s"}

        with patch.object(SteeringRecorderService, "record_samples", side_effect=fake_record), \
             patch.object(tasks, "emit_circuit_run_completed"):
            _raw(tasks.run_circuit_record)(_fake_task(session), "srr_1", {})

        assert placed == [RTX_UUID]
        assert len(seen) == 1
        device, gpu, uuid_at_load, commits_at_load = seen[0]
        assert device == RTX_DEVICE and gpu == RTX_RECORD
        assert uuid_at_load == RTX_UUID and commits_at_load >= 1
        assert row.status == "completed"

    def test_a_placement_refusal_fails_the_run_with_its_message(self, refused):
        from src.services.steering_recorder_service import SteeringRecorderService
        from src.workers import circuit_record_tasks as tasks

        row = self._row(gpu_request=TI_UUID)
        with patch.object(SteeringRecorderService, "record_samples") as record, \
             patch.object(tasks, "emit_circuit_run_failed"):
            with pytest.raises(GpuPlacementError):
                _raw(tasks.run_circuit_record)(_fake_task(_Session(row)), "srr_1", {})

        record.assert_not_called()
        assert row.status == "failed" and row.error in REFUSAL
        assert row.error.startswith("GPU 0 (NVIDIA GeForce RTX 3080 Ti")
        assert row.gpu_uuid is None


class TestRecorderServiceUsesTheDevice:
    def test_the_device_reaches_the_loader_and_resolver_and_the_card_the_manifest(self, monkeypatch):
        import src.services.steering_recorder_service as mod
        from src.services.steering_recorder_service import SteeringRecorderService

        seen = {}

        def fake_load(model_id, db, device):
            seen["load"] = device
            return ("M", "T", "S", False, device)

        def fake_resolve(cls, artifact, db, device):
            seen["resolve"] = device
            return ("m_1", [(1, 1, 1.0, "W")])

        monkeypatch.setattr(mod, "load_model_and_structure", fake_load)
        monkeypatch.setattr(SteeringRecorderService, "_resolve", classmethod(fake_resolve))
        monkeypatch.setattr(SteeringRecorderService, "_artifact_model_id",
                            staticmethod(lambda art, db: "m_1"))
        monkeypatch.setattr(SteeringRecorderService, "_model_hf_id",
                            staticmethod(lambda mid, db: "org/m"))
        monkeypatch.setattr(mod, "build_steer_generator",
                            lambda *a, **k: (lambda d, p: "S", lambda p, s: "B"))
        persisted = {}
        monkeypatch.setattr(SteeringRecorderService, "_persist",
                            staticmethod(lambda db, art, payload: persisted.update(payload) or "vman_s"))

        SteeringRecorderService.record_samples(None, dict(RECORD_BODY),
                                               device=RTX_DEVICE, gpu=RTX_RECORD)

        assert seen == {"load": RTX_DEVICE, "resolve": RTX_DEVICE}
        assert persisted["gpu"] == RTX_RECORD


# ═════════════════════════════ MCP ════════════════════════════════════════


class _FakeClient:
    def __init__(self):
        self.posts = []

    async def post(self, path, json_body=None):
        self.posts.append((path, json_body))
        return {"ok": True}

    async def get(self, *a, **k):
        return {}

    async def patch(self, *a, **k):
        return {}

    async def delete(self, *a, **k):
        return {}


MCP_CALLS = {
    "start_circuit_capture": ({"dataset_id": "ds1", "layers": [{"layer": 1, "sae_id": "s"}]},
                              "/circuit-capture"),
    "run_attribution_pass": ({"run_id": "dsc1"}, "/circuit-discovery/dsc1/attribution"),
    "validate_circuit_edges": ({"run_id": "dsc1"}, "/circuit-discovery/dsc1/validate"),
    "run_circuit_faithfulness": ({"circuit_id": "crc_1"}, "/circuits/crc_1/faithfulness"),
    "calibrate_circuit_strength": ({"circuit_id": "crc_1"}, "/circuits/crc_1/calibration"),
    "record_steering_samples": ({"artifact": {"kind": "circuit", "circuit_id": "crc_1"},
                                 "dials": [0.5], "prompts": ["p"]},
                                "/circuits/steering-samples"),
}


def _call_tool(name, args):
    import asyncio

    from mcp.server.fastmcp import FastMCP

    from src.mcp_server.config import MCPSettings
    from src.mcp_server.tools import circuits

    fake = _FakeClient()
    mcp = FastMCP("test")
    circuits.register(mcp, fake, MCPSettings(allow_anonymous=True))
    asyncio.run(mcp.call_tool(name, args))
    return fake.posts


class TestMcpToolsSendTheGpu:
    @pytest.mark.parametrize("tool", sorted(MCP_CALLS))
    def test_a_named_card_is_sent_in_the_body(self, tool):
        args, path = MCP_CALLS[tool]
        posts = _call_tool(tool, {**args, "gpu": RTX_UUID})
        assert len(posts) == 1
        assert posts[0][0] == path and posts[0][1]["gpu"] == RTX_UUID

    @pytest.mark.parametrize("tool", sorted(MCP_CALLS))
    def test_the_default_is_auto(self, tool):
        args, _path = MCP_CALLS[tool]
        posts = _call_tool(tool, args)
        assert len(posts) == 1 and posts[0][1]["gpu"] == "auto"
