"""Multi-GPU Phase 3, review round 2 — round 1's claim and lease fixes, and how they compose.

Round 1's fixes are where this round looked: the ordered release, lease loss, the
redelivered-copy drop, the API-process logit-lens lease and the host RAM reserve.
Every test runs the REAL claim (``services/gpu_job_claim``), the REAL ``@gpu_job``
wrapper where a task is involved, and REAL POSTGRES leases (``gpu_lease_db``) on a
fake inventory of the node's two cards. Concurrency is driven by events with hang
timeouts, never by sleeps.

PROOF (2026-09-14): the fixed sources swapped back to ccd91cc6 (scratchpad
p3-r2/spec_baseline.json, B-R2-claims), this module red on the assertion each test
makes: both renewer interleavings, the allocated reserve, the cause and the context
frames, the lens cancelled while it claims, the download's deleted weights. The lens
lease-loss tests name GpuLeaseLost, which did not exist, and are proven by C-R2-7a–g.
The negative controls (a lapsed lease still taken back while the job runs, a reserve
not yet allocated, a refused claim closed at once, the harness download, an
operator's cancel) were green on both trees.

MUTATION CONTROLS (2026-09-14; scratchpad p3-r2/mutate.py and spec_r2.json; each
alone, restored byte-identically with sha256 checked, `git status` clean after every
run) — all red:
  C-R2-1a no stop check before the take-back          -> SURVIVED first: the check after the
          take-back gave the card back anyway. The test now also asserts a stopped
          renewer never calls acquire (a take-back given up a moment later parks
          another job for 30 s); re-run red
  C-R2-1b a take-back committed after the stop is kept -> the take-back that commits after the stop
  C-R2-1c a take-back records no reserve               -> an allocated reserve after a take-back
  C-R2-1d host_reserve_allocated leaves the claim's reserve -> the same
  C-R2-1e the renewer is given a reserve of 0          -> a reserve not yet allocated after a take-back
  C-R2-2a the chain walk skips __cause__/__context__   -> a cause; a context
  C-R2-2b the wrapper clears no frames                 -> a cause; a context; R1's frames test
  C-R2-3a the claim is closed at once, thread running  -> the card is given back once the claim finishes
  C-R2-3b the claim is awaited unshielded              -> the same
  C-R2-4a no lease-lost branch in the download handler -> the weights are kept, a retryable failure
  C-R2-4b the lease-lost branch records nothing        -> the same (SKIPPED first: the anchor's
          indentation matched nothing; re-anchored, red)
  C-R2-4c the failure branch skips the shared writer   -> test_gpu_model_download_placement (3)
  C-R2-7a no lease check in the lens's batch loop      -> a task lens stops at the next batch
  C-R2-7b lease_lost_reason ignores the request's claim -> the API-process lens lease is what the check reads
  C-R2-7c the lens lease does not set the request's claim -> the same
  C-R2-7d the lens lease does not reset it on exit     -> SURVIVED first: one lease leaves a closed
          claim, which reports nothing either way. Nested leases added; re-run red
  C-R2-7e the push's dashboard step swallows it        -> the push does not skip a lens that lost its lease
  C-R2-7f push_sae_to_local swallows it                -> the push fails with the reason and pushes nothing
  C-R2-7g GpuLeaseLost is not a GpuPlacementError      -> the export fails the job

ROUND 1 CONTROLS RE-RUN on the lines this round changed (spec_r1_rerun.json; anchors
re-spelled where this round moved the line) — all red: C-R1-4a (the take-back),
C-R1-4c, C-R1-4d, C-R1-4e, C-R1-4n, C-R1-4o (the renewer's cards), C-R1-6a, C-R1-6c
(the reserve recorded at take), C-R1-1b (the frames), C-R1-5c, C-R1-5d (the lens
lease's close), M3 (close() releases: 17 red), D8, D9 (the failure writer's gpu_uuids).
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from celery import Task
from sqlalchemy import text

from src.core.cancellation import GPU_LEASE_LOST, OperatorCancelled
from src.core.config import settings
from src.models.model import Model, ModelStatus
from src.models.task_queue import TaskQueue
from src.services import gpu_job_claim as C
from src.services import gpu_leases, gpu_placement
from src.services import logit_lens_service as lens_module
from src.services.gpu_placement import GpuCard, Placement
from src.workers import gpu_job as G
from src.workers import model_tasks
from src.workers.gpu_supervisor import WORKER_GPU_ENV
from tests.unit.gpu_lease_db import clear, lease_engine, session_factory

TI_UUID = "GPU-f47ba814-49a2-603f-3595-275284140251"
RTX_UUID = "GPU-247aa582-0d1b-e161-8156-983ed1fefc57"
TI = GpuCard(0, TI_UUID, "NVIDIA GeForce RTX 3080 Ti", 12_288, 11_900)
RTX = GpuCard(1, RTX_UUID, "NVIDIA GeForce RTX 3090", 24_576, 21_500)
CARDS = [TI, RTX]

#: A thread that should finish at once gets this long before the test fails instead of hanging.
HANG_S = 10.0

RENEWER_THREAD = "gpu-lease-renew"


@pytest.fixture(scope="module")
def engine():
    eng = lease_engine("mistudio_test_gpu_phase3_review_r2")
    yield eng
    eng.dispose()


@pytest.fixture
def db(engine):
    clear(engine)
    return session_factory(engine)


def leases(db):
    with db() as s:
        return gpu_leases.live_leases(s)


def ctx(db, *, worker=RTX, handoff=True, cards=CARDS, available=1_000_000.0, task_id="task-1", **kwargs):
    return C.ClaimContext(
        holder=C.make_holder(kwargs.pop("kind", "test"), task_id),
        task_id=task_id,
        worker_uuid=None if worker is None else worker.uuid,
        handoff=handoff,
        session=kwargs.pop("session", db),
        inventory=lambda: list(cards),
        available_mb=lambda: available,
        **kwargs,
    )


def lapse(engine, holder):
    """The holder's lease expired and its heartbeat went stale, as after a long outage."""
    with engine.begin() as conn:
        conn.execute(
            text("UPDATE gpu_leases SET expires_at = now() - interval '1 second', "
                 "heartbeat_at = now() - interval '11 minutes' WHERE holder = :h"),
            {"h": holder},
        )


def in_renewer() -> bool:
    return threading.current_thread().name.startswith(RENEWER_THREAD)


# ── a renewal in flight when the job releases its lease ─────────────────────


class TestARenewalNeverOutlivesTheRelease:
    """``close()`` stops the renewer with a BOUNDED join (5 s), then deletes the holder's rows.

    A renewal that is still inside the database when the join gives up finds its rows
    gone, reads that as a lapse, and takes the card back with ``acquire`` — AFTER the
    job released it. Nothing renews that row again, so a finished job holds the card
    for a whole lease TTL (600 s) and every other job waits or goes elsewhere.
    """

    def test_a_renewal_in_flight_when_the_job_releases_does_not_take_the_card_back(self, db, monkeypatch):
        """And a stopped renewer does not even TRY: a take-back given up a moment later still
        shows every other claim the card as held for that moment, and parks a job for 30 s."""
        holder = "test:t-r2a:aaaa0000"
        with db() as s:
            assert gpu_leases.acquire(s, [RTX_UUID], holder, task_id="t-r2a")
        gate, entered = threading.Event(), threading.Event()
        renewer_acquires = []
        real_acquire = gpu_leases.acquire

        def acquire(session, uuids, who, **kwargs):
            if in_renewer():
                renewer_acquires.append(who)
            return real_acquire(session, uuids, who, **kwargs)

        monkeypatch.setattr(gpu_leases, "acquire", acquire)

        def session():
            if in_renewer():
                entered.set()
                gate.wait(HANG_S)
            return db()

        renewer = C.LeaseRenewer(holder, session, interval_s=0.01, uuids=[RTX_UUID], task_id="t-r2a")
        renewer.start()
        try:
            assert entered.wait(HANG_S), "the renewal never ran"
            renewer.stop(timeout=0.05)            # close(): a bounded join that gives up...
            assert renewer.is_alive(), "the fixture did not keep the renewal in flight"
            with db() as s:                       # ...then the lease release
                gpu_leases.release(s, holder)
        finally:
            gate.set()
            renewer._thread.join(HANG_S)
        assert not renewer.is_alive()
        assert leases(db) == {}, "a renewal that finished after the release took the card back"
        assert renewer_acquires == [], "a stopped renewer tried to take the card back"

    def test_a_take_back_that_commits_after_the_stop_is_given_up_again(self, db, monkeypatch):
        """The stop can land between the renewer's check and its re-acquire's commit."""
        holder = "test:t-r2b:bbbb0000"
        gate, entered = threading.Event(), threading.Event()
        real_acquire = gpu_leases.acquire

        def acquire(session, uuids, who, **kwargs):
            if in_renewer():
                entered.set()
                gate.wait(HANG_S)
            return real_acquire(session, uuids, who, **kwargs)

        monkeypatch.setattr(gpu_leases, "acquire", acquire)
        # No row: the lease already lapsed (a janitor freed its stale heartbeat).
        renewer = C.LeaseRenewer(holder, db, interval_s=0.01, uuids=[RTX_UUID], task_id="t-r2b")
        renewer.start()
        try:
            assert entered.wait(HANG_S), "the renewal never tried to take the card back"
            renewer.stop(timeout=0.05)
            with db() as s:
                gpu_leases.release(s, holder)
        finally:
            gate.set()
            renewer._thread.join(HANG_S)
        assert not renewer.is_alive()
        assert leases(db) == {}, "a take-back committed after the stop kept the card"

    def test_a_lapsed_lease_is_still_taken_back_while_the_job_runs(self, db, engine):
        """The negative side of the fix: a renewer that has NOT been stopped still re-acquires."""
        claim = ctx(db, worker=RTX, renew_interval_s=3600.0)
        claim.claim("auto", required_mb=4_000)
        try:
            lapse(engine, claim.holder)
            assert claim._renewer.tick() == 1
            assert leases(db) == {RTX_UUID: claim.holder}
        finally:
            claim.close()
        assert leases(db) == {}


# ── a take-back keeps the host reserve the job recorded ─────────────────────


class TestATakenBackLeaseKeepsItsRecordedReserve:
    """R1-6 zeroes a training's host reserve once its pinned buffer exists. A lease the
    renewer takes back is a NEW row, written with no reserve (NULL), which the guard reads
    as "a row from before the column" — the training's 24 GiB is subtracted again for the
    rest of its run, the double count R1-6 removed."""

    @staticmethod
    def _allocated(claim):
        with C._ACTIVE_LOCK:
            C._ACTIVE.append(claim)
        try:
            C.host_reserve_allocated()
        finally:
            with C._ACTIVE_LOCK:
                C._ACTIVE.remove(claim)

    def test_an_allocated_reserve_is_not_counted_again_after_a_take_back(self, db, engine):
        training = ctx(db, worker=RTX, kind="training", renew_interval_s=3600.0)
        training.claim("auto", required_mb=4_000)
        try:
            self._allocated(training)
            other = ctx(db, worker=TI, task_id="task-2")
            assert other._other_reserves_mb({training.holder}) == 0.0
            lapse(engine, training.holder)
            assert training._renewer.tick() == 1
            assert other._other_reserves_mb({training.holder}) == 0.0, (
                "the take-back re-reserved host memory the training had already allocated"
            )
        finally:
            training.close()

    def test_a_reserve_not_yet_allocated_is_still_counted_after_a_take_back(self, db, engine):
        training = ctx(db, worker=RTX, kind="training", renew_interval_s=3600.0)
        training.claim("auto", required_mb=4_000)
        try:
            lapse(engine, training.holder)
            assert training._renewer.tick() == 1
            other = ctx(db, worker=TI, task_id="task-2")
            assert other._other_reserves_mb({training.holder}) == 24 * 1024
        finally:
            training.close()


# ── the traceback of a failed job, through its whole chain ──────────────────


class FakeTask(Task):
    request = None
    name = "tests.review_r2_gpu_task"

    def __init__(self, request):
        self.request = request

    def apply_async(self, args=None, kwargs=None, **options):  # pragma: no cover - no hand-off here
        raise AssertionError("handed off")


def request(**overrides):
    base = dict(id="tid-r2", args=["job-1"], kwargs={"gpu_request": "auto"}, timelimit=(None, None),
                gpu_hops=None, headers=None, argsrepr=None, kwargsrepr=None, delivery_info={})
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.fixture
def node(engine, monkeypatch):
    clear(engine)
    db = session_factory(engine)
    monkeypatch.setattr(settings, "gpu_worker_mode", "per_card")
    monkeypatch.setenv(WORKER_GPU_ENV, RTX_UUID)
    monkeypatch.setattr(C, "_sync_session", db)
    monkeypatch.setattr(C, "list_cards", lambda: list(CARDS))
    monkeypatch.setattr(C, "host_available_mb", lambda: 1_000_000.0)
    return SimpleNamespace(db=db)


class _Held:
    """Stands in for a model a job's frame holds."""


class TestAChainedFailureFreesItsFramesBeforeTheLease:
    """R1-1 clears the failing exception's traceback frames so the release before the
    lease release frees the body's models. It cleared ONE traceback. A body that wraps a
    failure — ``raise RuntimeError(...) from exc``, or a raise inside an ``except`` — keeps
    the original exception, and every frame it unwound, on ``__cause__``/``__context__``.
    The model stays alive through the release and the card is offered still full."""

    @staticmethod
    def _run_and_observe(node, monkeypatch, body):
        refs, seen = [], {}
        real_release = gpu_leases.release

        def spying_release(session, holder, uuids=None):
            seen["alive_at_release"] = refs[0]() is not None
            seen["leases_at_release"] = gpu_leases.live_leases(session)
            return real_release(session, holder, uuids)

        monkeypatch.setattr(gpu_leases, "release", spying_release)

        def load_and_fail():
            model = _Held()
            refs.append(weakref.ref(model))
            raise ValueError("CUDA out of memory")

        @G.gpu_job("test")
        def run(self, job_id, gpu_request="auto"):
            C.claim_cards(gpu_request, required_mb=4_000)
            body(load_and_fail)

        with pytest.raises(RuntimeError):
            run(FakeTask(request()), "job-1")
        return seen

    def test_a_cause(self, node, monkeypatch):
        def body(load_and_fail):
            try:
                load_and_fail()
            except ValueError as exc:
                raise RuntimeError("the load failed") from exc

        seen = self._run_and_observe(node, monkeypatch, body)
        assert list(seen["leases_at_release"]) == [RTX_UUID]
        assert seen["alive_at_release"] is False, "the chained exception's frame still held the model"
        assert leases(node.db) == {}

    def test_a_context(self, node, monkeypatch):
        def body(load_and_fail):
            try:
                load_and_fail()
            except ValueError:
                raise RuntimeError("cleanup after the failed load also failed")

        seen = self._run_and_observe(node, monkeypatch, body)
        assert seen["alive_at_release"] is False, "the context exception's frame still held the model"

    def test_the_message_and_the_chain_survive(self, node, monkeypatch):
        """What the task's failure handler and Celery's result read: the message, the chain, the lines."""
        import traceback as tb

        caught = {}

        def body(load_and_fail):
            try:
                load_and_fail()
            except ValueError as exc:
                raise RuntimeError("the load failed") from exc

        @G.gpu_job("test")
        def run(self, job_id, gpu_request="auto"):
            C.claim_cards(gpu_request, required_mb=4_000)

            def load_and_fail():
                raise ValueError("CUDA out of memory")

            body(load_and_fail)

        with pytest.raises(RuntimeError) as failure:
            run(FakeTask(request()), "job-1")
        caught["text"] = "".join(tb.format_exception(failure.value))
        assert "RuntimeError: the load failed" in caught["text"]
        assert "ValueError: CUDA out of memory" in caught["text"]
        assert "load_and_fail" in caught["text"]


# ── a logit lens in the API process cancelled while it claims ────────────────


@pytest.fixture
def api_node(engine, monkeypatch):
    """Per-card mode, the API process (no worker card, no claim), real leases, CUDA faked."""
    clear(engine)
    db = session_factory(engine)
    index_of = {card.uuid: card.index for card in CARDS}
    monkeypatch.setattr(settings, "gpu_worker_mode", "per_card")
    monkeypatch.delenv(WORKER_GPU_ENV, raising=False)
    monkeypatch.setattr(C, "_sync_session", db)
    monkeypatch.setattr(C, "list_cards", lambda: list(CARDS))
    monkeypatch.setattr(C, "host_available_mb", lambda: 1_000_000.0)
    monkeypatch.setattr(gpu_placement, "list_cards", lambda: list(CARDS))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device", lambda device: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(
        gpu_placement, "torch_device", lambda card: torch.device("cuda", index_of[getattr(card, "uuid", card)]),
    )
    return db


class TestAnApiProcessLensCancelledWhileItClaims:
    """``LogitLensLease.__aenter__`` claims on a thread (``asyncio.to_thread``). When the
    awaiting coroutine is cancelled — a client that disconnects from the synchronous
    dashboard route, an API shutdown during an export — its ``except`` closed the claim AT
    ONCE, while the claim thread was still running. That thread then took the lease and
    started a renewer that nothing will ever stop: the card is leased, and renewed every
    60 s, until the API process restarts."""

    def test_the_card_is_given_back_once_the_abandoned_claim_finishes(self, api_node, monkeypatch):
        gate, entered, claimed, closed_after = (threading.Event() for _ in range(4))
        made = []

        def inventory():
            entered.set()
            gate.wait(HANG_S)
            return list(CARDS)

        monkeypatch.setattr(C, "list_cards", inventory)
        real_claim, real_close = C.ClaimContext.claim, C.ClaimContext.close

        def claim(self, *args, **kwargs):
            made.append(self)
            try:
                return real_claim(self, *args, **kwargs)
            finally:
                claimed.set()

        def close(self):
            real_close(self)
            if claimed.is_set():
                closed_after.set()

        monkeypatch.setattr(C.ClaimContext, "claim", claim)
        monkeypatch.setattr(C.ClaimContext, "close", close)

        async def scenario():
            lease = lens_module.logit_lens_lease(RTX_UUID, kind="dashboard_logit_lens_api", wait_timeout_s=0.0)
            entering = asyncio.ensure_future(lease.__aenter__())
            assert await asyncio.to_thread(entered.wait, HANG_S), "the claim never started"
            entering.cancel()
            with pytest.raises(asyncio.CancelledError):
                await entering
            gate.set()
            assert await asyncio.to_thread(claimed.wait, HANG_S), "the claim thread never finished"
            return await asyncio.to_thread(closed_after.wait, HANG_S)

        try:
            closed = asyncio.run(scenario())
            assert closed, "the claim was closed before its thread took the lease, and never again"
            assert leases(api_node) == {}
            assert all(claim_._renewer is None for claim_ in made), "a renewer still runs for the abandoned claim"
        finally:
            gate.set()
            for claim_ in made:
                real_close(claim_)

    def test_a_claim_that_fails_is_still_closed_at_once(self, api_node, monkeypatch):
        """Unchanged: a claim that raises (a busy card, wait 0) is closed before the error reaches the caller."""
        with api_node() as s:
            assert gpu_leases.acquire(s, [RTX_UUID], "training:someone-else:00000000", task_id="x")

        async def scenario():
            async with lens_module.logit_lens_lease(RTX_UUID, kind="dashboard_logit_lens_api", wait_timeout_s=0.0):
                raise AssertionError("the lens ran on a card another job holds")

        with pytest.raises(gpu_placement.GpuPlacementError, match="Waited 0 min"):
            asyncio.run(scenario())
        assert leases(api_node) == {RTX_UUID: "training:someone-else:00000000"}


# ── a model download whose lease is lost ─────────────────────────────────────

LOST_DETAIL = (
    f"its lease on ['{RTX_UUID}'] lapsed and another job now holds the card; it cannot keep running there"
)


class _Query:
    def __init__(self, first=None, rows=()):
        self._first, self._rows = first, list(rows)

    def filter_by(self, **_):
        return self

    def filter(self, *_):
        return self

    def first(self):
        if self._first is not None:
            return self._first
        return self._rows[0] if self._rows else None

    def all(self):
        return list(self._rows)


class TestADownloadThatLosesItsLeaseKeepsTheModel:
    """The download is PLACED BEFORE the transfer, so its lease is held — and can be lost —
    for the whole download. Its cancellation check after the download raises
    ``OperatorCancelled(reason=gpu_lease_lost)``, and the only handler for that exception
    is the operator's cancel: it deleted the downloaded weights (``rmtree`` of the model's
    cache directory), recorded "Cancelled by user" and closed the retry row as cancelled.
    A 30 GB download was thrown away by a lease lapse nobody asked for."""

    @staticmethod
    def _drive(tmp_path, *, reason=GPU_LEASE_LOST, detail=LOST_DETAIL):
        """Run the real task body with the stop raised after the download; None runs it through."""
        model_row = SimpleNamespace(status=None, progress=None, error_message=None)
        added, recorded = [], []
        session = MagicMock()
        session.add.side_effect = added.append
        session.query.side_effect = lambda cls: _Query(first=model_row) if cls is Model else _Query(rows=())
        db_ctx = MagicMock()
        db_ctx.__enter__ = MagicMock(return_value=session)
        db_ctx.__exit__ = MagicMock(return_value=False)
        cache = tmp_path / "raw" / "m_1"

        def load(**kwargs):
            cache.mkdir(parents=True, exist_ok=True)
            (cache / "model.safetensors").write_bytes(b"weights")
            return MagicMock(), MagicMock(), MagicMock(), {
                "architecture": "llama", "params_count": 1_000, "architecture_config": {},
                "memory_required_bytes": 1, "quantization": "FP16",
            }

        checker = MagicMock()

        def raise_if_cancelled(where=""):
            if reason is not None and "after the download" in where:
                raise OperatorCancelled("model_download", "m_1", reason, detail)

        checker.raise_if_cancelled.side_effect = raise_if_cancelled
        place = MagicMock(return_value=Placement(card=RTX, device=torch.device("cpu")))
        fake_settings = SimpleNamespace(models_dir=tmp_path, resolve_deletable_path=lambda stored: Path(stored))
        outcome = SimpleNamespace(model_row=model_row, added=added, recorded=recorded, result=None,
                                  error=None, weights=cache / "model.safetensors")
        with patch("src.workers.base_task.DatabaseTask.get_db", return_value=db_ctx), \
             patch.object(model_tasks, "get_sync_db", MagicMock(return_value=db_ctx)), \
             patch("src.services.gpu_placement.place_job", place), \
             patch.object(model_tasks, "required_mb_for_load", MagicMock(return_value=None)), \
             patch("src.workers.base_task.mark_task_queue_entries_completed", MagicMock()), \
             patch("src.ml.transformers_compat.patch_transformers_compatibility", MagicMock()), \
             patch.object(model_tasks, "load_model_from_hf", load), \
             patch.object(model_tasks, "send_progress_update", MagicMock()), \
             patch.object(model_tasks, "DownloadProgressMonitor", MagicMock()), \
             patch.object(model_tasks, "clear_cancel_request", MagicMock()), \
             patch.object(model_tasks, "cancel_checker", MagicMock(return_value=checker)), \
             patch.object(model_tasks, "record_progress", lambda *a, **k: recorded.append((a, k)) or True), \
             patch.object(model_tasks, "settings", fake_settings):
            try:
                outcome.result = model_tasks.download_and_load_model.run(
                    "m_1", "vendor/model", "FP16", gpu_request=RTX_UUID,
                )
            except BaseException as exc:  # noqa: BLE001 - the test inspects it
                outcome.error = exc
        return outcome

    def test_the_harness_runs_a_download_through(self, tmp_path):
        outcome = self._drive(tmp_path, reason=None)
        assert outcome.error is None, outcome.error
        assert outcome.model_row.status == ModelStatus.READY and outcome.weights.exists()

    def test_the_downloaded_weights_are_kept_and_the_stop_is_a_retryable_failure(self, tmp_path):
        outcome = self._drive(tmp_path)

        assert outcome.error is None, outcome.error
        assert outcome.weights.exists(), "a lost GPU lease deleted the downloaded model"
        assert outcome.result["status"] == "failed" and outcome.result["reason"] == GPU_LEASE_LOST
        assert outcome.model_row.status == ModelStatus.ERROR
        assert "lease" in outcome.model_row.error_message
        assert "Cancelled by user" not in outcome.model_row.error_message
        assert not any("Cancelled by user" in str(call) for call in outcome.recorded), outcome.recorded
        retry_rows = [row for row in outcome.added if isinstance(row, TaskQueue)]
        assert len(retry_rows) == 1 and retry_rows[0].status == "failed", "no retry was offered"
        assert retry_rows[0].retry_params["gpu_request"] == RTX_UUID
        assert "lease" in retry_rows[0].error_message

    def test_an_operators_cancel_still_removes_the_partial_download(self, tmp_path):
        """Unchanged: the operator's own cancel still owns and deletes its partial output."""
        outcome = self._drive(tmp_path, reason="cancelled", detail="stopped after the download")

        assert outcome.error is None, outcome.error
        assert outcome.result["status"] == "cancelled"
        assert not outcome.weights.exists(), "the operator's cancel no longer removes the partial download"
        assert any("Cancelled by user" in str(call) for call in outcome.recorded)


# ── a logit lens whose job loses its lease ───────────────────────────────────


class _LensDb:
    """The two rows ``compute_logit_lens_for_sae`` reads: the SAE and its model."""

    def __init__(self):
        self.sae = SimpleNamespace(
            id="sae-1", status=lens_module.SAEStatus.READY.value, local_path="saes/sae-1",
            model_id="m-1", model_name=None, n_features=6,
        )
        self.model = SimpleNamespace(quantized_path=None, file_path=None, repo_id="vendor/model", name="model")

    async def get(self, cls, key):
        return self.sae if cls is lens_module.ExternalSAE else self.model


class TestALogitLensStopsWhenItsJobLosesTheLease:
    """R1-4 stops a job that lost its lease at its next cooperative cancellation check.
    A logit lens has none: the Neuronpedia dashboard task, the push and the export compute
    it over an SAE's every feature (minutes to hours), and ran on, on a card another job
    may since have been placed on. The API process's lens lease (R1-5) was worse: its
    claim is not the process-wide claim, so its loss was never read by anything."""

    @staticmethod
    def _service(monkeypatch, tmp_path, on_batch):
        batches = []
        service = lens_module.LogitLensService()
        monkeypatch.setattr(lens_module, "settings", SimpleNamespace(resolve_data_path=lambda stored: tmp_path))
        monkeypatch.setattr(lens_module, "_downloaded_weights_dir", lambda record: tmp_path)
        monkeypatch.setattr(lens_module, "load_sae_auto_detect",
                            lambda path, device: ({"W_dec": torch.randn(4, 6)}, None, None))
        monkeypatch.setattr(lens_module.LogitLensService, "_read_unembedding",
                            lambda self, weights, device: (torch.randn(4, 10), MagicMock()))

        async def batch(self, W_dec, W_U, tokenizer, indices, k):
            batches.append(list(indices))
            on_batch(len(batches))
            return {i: f"r{i}" for i in indices}

        monkeypatch.setattr(lens_module.LogitLensService, "_compute_batch_logit_lens", batch)
        return service, batches

    @staticmethod
    def _compute(service):
        return asyncio.run(service.compute_logit_lens_for_sae(
            _LensDb(), "sae-1", feature_indices=list(range(6)), batch_size=2,
            force_recompute=True, device=torch.device("cpu"),
        ))

    def test_a_task_lens_stops_at_the_batch_after_its_lease_is_lost(self, node, monkeypatch, tmp_path):
        claim = ctx(node.db, worker=RTX, renew_interval_s=3600.0)
        claim.claim("auto", required_mb=4_000)

        def lose_after_the_first(n):
            if n == 1:
                claim._renewer.lost = "its lease lapsed and another job now holds the card"

        service, batches = self._service(monkeypatch, tmp_path, lose_after_the_first)
        with C.claiming(claim):
            with pytest.raises(C.GpuLeaseLost, match="another job now holds the card"):
                self._compute(service)
        assert batches == [[0, 1]], "the lens kept computing on a card it no longer held"
        assert leases(node.db) == {}

    def test_a_task_lens_that_holds_its_lease_computes_every_batch(self, node, monkeypatch, tmp_path):
        claim = ctx(node.db, worker=RTX, renew_interval_s=3600.0)
        claim.claim("auto", required_mb=4_000)
        service, batches = self._service(monkeypatch, tmp_path, lambda n: None)
        with C.claiming(claim):
            assert len(self._compute(service)) == 6
        assert batches == [[0, 1], [2, 3], [4, 5]]

    def test_the_api_process_lens_lease_is_what_the_check_reads(self, api_node):
        seen = {}

        async def scenario():
            lease = lens_module.logit_lens_lease(RTX_UUID, kind="export_logit_lens")
            async with lease:
                seen["held"] = C.lease_lost_reason()
                lease._claim._renewer.lost = "its lease lapsed"
                seen["lost"] = C.lease_lost_reason()
                # The threads the request starts see it too.
                seen["in_thread"] = await asyncio.to_thread(C.lease_lost_reason)
                with pytest.raises(C.GpuLeaseLost, match="its lease lapsed"):
                    C.raise_if_lease_lost("computing the logit lens")
            seen["after"] = C.lease_lost_reason()

        asyncio.run(scenario())
        assert seen == {"held": None, "lost": "its lease lapsed", "in_thread": "its lease lapsed", "after": None}
        assert leases(api_node) == {}

    def test_an_outer_lens_lease_is_read_again_once_an_inner_one_ends(self, api_node):
        """Two leases in one request, one inside the other: once the inner one ends, the check
        reads the OUTER lease again. Left pointing at the inner (closed) claim, it reported
        nothing for the rest of the outer lease's work, however that lease was lost."""
        seen = {}

        async def scenario():
            outer = lens_module.logit_lens_lease(RTX_UUID, kind="export_logit_lens")
            async with outer:
                async with lens_module.logit_lens_lease(TI_UUID, kind="export_logit_lens"):
                    pass
                outer._claim._renewer.lost = "the outer lease lapsed"
                seen["outer"] = C.lease_lost_reason()

        asyncio.run(scenario())
        assert seen == {"outer": "the outer lease lapsed"}, "the ended inner lease hid the outer lease's loss"
        assert leases(api_node) == {}

    def test_another_requests_lens_lease_is_not_read(self, api_node):
        """The API process serves many requests: one request's lost lease stops only that request."""
        seen = {}

        async def scenario():
            gate = asyncio.Event()

            async def lens():
                lease = lens_module.logit_lens_lease(RTX_UUID, kind="export_logit_lens")
                async with lease:
                    lease._claim._renewer.lost = "its lease lapsed"
                    await gate.wait()

            async def bystander():
                await asyncio.sleep(0)
                seen["bystander"] = C.lease_lost_reason()
                gate.set()

            await asyncio.gather(lens(), bystander())

        asyncio.run(scenario())
        assert seen == {"bystander": None}

    def test_the_export_fails_the_job_rather_than_skipping_the_lens(self):
        """`NeuronpediaExportService.execute_export` re-raises a `GpuPlacementError` from its
        logit-lens stage (the job fails with that message) and skips the lens on any other
        error. A lost lease must take the first path: exported without its lens, the job
        answers a different request than the one made."""
        from src.services import neuronpedia_export_service as export_module

        lost = C.GpuLeaseLost("Stopped while computing the logit lens: its lease lapsed. Run it again.")
        assert isinstance(lost, export_module.GpuPlacementError)

    def test_the_push_does_not_skip_a_lens_that_lost_its_lease(self, monkeypatch):
        from src.services import neuronpedia_local_service as push_module

        lens = MagicMock()

        async def compute(**kwargs):
            raise C.GpuLeaseLost("Stopped while computing the logit lens: its lease lapsed. Run it again.")

        lens.compute_logit_lens_for_sae = compute
        monkeypatch.setattr(push_module, "get_logit_lens_service", lambda: lens)
        service = push_module.NeuronpediaLocalPushService.__new__(push_module.NeuronpediaLocalPushService)
        config = SimpleNamespace(compute_dashboard_data=True, feature_indices=[0, 1], logit_lens_k=5,
                                 device=torch.device("cpu"))
        with pytest.raises(C.GpuLeaseLost):
            asyncio.run(service._compute_dashboard_data_if_needed(
                db=MagicMock(), sae=SimpleNamespace(id="sae-1", n_features=2), config=config,
            ))

    def test_a_push_whose_lens_lost_its_lease_fails_with_the_reason_and_pushes_nothing(self, monkeypatch):
        from unittest.mock import AsyncMock

        from src.services import neuronpedia_local_service as push_module

        service = push_module.NeuronpediaLocalPushService.__new__(push_module.NeuronpediaLocalPushService)
        client = AsyncMock()
        service._get_client = AsyncMock(return_value=client)
        service._generate_model_id = lambda name: "model"
        service._generate_source_set_name = lambda n: "res-8k"
        service._generate_source_id = lambda layer, name: "0-res-8k"
        service._compute_dashboard_data_if_needed = AsyncMock(
            side_effect=C.GpuLeaseLost("Stopped while computing the logit lens: its lease lapsed. Run it again."))
        service._load_features = AsyncMock(side_effect=AssertionError("features were pushed after the lease was lost"))
        db = MagicMock()
        db.get = AsyncMock(return_value=SimpleNamespace(
            id="sae-1", name="SAE from model (L0-residual)", model_name="model", model_id=None, layer=0, n_features=2))
        config = push_module.LocalPushConfig(compute_dashboard_data=True, device=torch.device("cpu"))

        result = asyncio.run(service.push_sae_to_local(db=db, sae_id="sae-1", config=config))

        assert result.success is False and "its lease lapsed" in result.error_message
        service._load_features.assert_not_awaited()
