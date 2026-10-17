"""Multi-GPU Phase 3, review round 1 — claims, leases and dispatch.

Every test here runs the REAL claim (``services/gpu_job_claim``) and, where a
task is involved, the REAL ``@gpu_job`` wrapper, against REAL POSTGRES leases
(``gpu_lease_db``) on a fake inventory: the node's RTX 3080 Ti (index 0) and RTX
3090 (index 1). Concurrency is driven by events and explicit ticks, never sleeps.

FINDINGS PINNED HERE (each test was red on c78f0f21 before its fix):
  R1-1  a finished job's memory was released only AFTER its lease (Celery's
        after_return runs once the body, and so `claiming()`, has exited), and on
        a failure the traceback kept the body's frames — models, buffers — alive
        through the release: the card was offered while still full.
  R1-2  the size-free precheck decided with allow_shard=True for EVERY job, so an
        "all" message for a job that can never split was parked while a card was busy.
  R1-4  a lease that lapsed was only logged and the job ran on; a renewal callback
        that raised killed the renewal thread silently.
  R1-5  the Neuronpedia export and the synchronous dashboard route computed the logit
        lens in the API process under no lease.
  (R1-3, jobs waiting for a GPU reaped as dead, is pinned in
  test_gpu_waiting_jobs_are_not_reaped.py.)

PROOFS on c78f0f21: R1-1's frame and order tests red with the four sources swapped
back (B-R1-1b); R1-5's four API-process tests red with the export and endpoint swapped
back (B-R1-5). R1-2 and R1-4 changed a signature the old code does not have, so their
proof is the control that restores the old rule in place: C-R1-2a (the precheck decides
with True again) and C-R1-4e (the callback outside the try, as on c78f0f21).

MUTATION CONTROLS (2026-09-14; scratchpad p3-r1-claim/mutate.py; each applied
alone where it matched exactly once, the named tests run, the source restored
byte-identically with sha256 checked and `git diff` clean) — all red:
  R1-1 (fix 3b857811), proof: the two frame/order tests red on c78f0f21's sources
    C-R1-1a close() skips the memory release           -> task's own release while leased; release given the
                                                           held cards; test_gpu_leases_compose_with_buffer_round2 order
    C-R1-1b the wrapper does not clear a failing body's frames -> a failed job's frames are freed before its lease
    C-R1-1c the release waits without its timeout       -> a release that hangs does not keep the card
    C-R1-1d the wrapper does not wire release_memory     -> task's own release while leased; composition order
    C-R1-1e TrainingTask.release_job_memory does nothing -> composition order (the buffer closed only after)
    C-R1-1f release_job_memory skips empty_cache_on_cards -> composition order (re-run after the rename)
  Phase 3 control re-run on the lines R1-1 touched:
    M3  close() does not release the leases              -> 13 red, including every R1-1 test here
  Composed with 3dac5fb8 (idle copies freed before the leases; cherry-pick resolution b7c12116):
  ONE ordered, bounded release — the task's own memory (only when a card was held), then the
  idle copies (on every exit, a hand-off included), then gc, each leased card's cache and
  pinned host memory — inside `close()`, before the lease release, which a failing step never
  skips. Re-run on the composed lines, each alone — all red:
    C-R1-1g the task hook runs with nothing held          -> a job that placed nothing releases nothing
    C-R1-1d-composed the wrapper does not wire release_memory -> task's own release while leased; the
                                                            idle-copy tests; the buffer order test
    (K1-composed, K2-composed, C-R1-1h: recorded in test_gpu_job_wrapper.py)
  Composed with 870f355a (the supervisor's `gpu-worker-down:<uuid>` lease on a card whose worker is down):
    C-R1-6g the host RAM guard counts that lease as a job -> a card marked unavailable by the supervisor
  R1-4 (fix dd4cc2c4)
    C-R1-4a no re-acquire of a lapsed lease             -> taken back when no other job took the card
    C-R1-4b _lose never records the loss                -> told to stop (another holder; TTL; cancel check)
    C-R1-4c no loss after a whole TTL unrenewed          -> no renewal succeeds for a whole TTL
    C-R1-4d the renewal clock is not reset on success   -> an outage shorter than the TTL resets the clock
    C-R1-4e the callback outside the try (c78f0f21)     -> a raising renewal callback does not end the renewals
    C-R1-4f CancelCheck ignores the lease               -> a cancel check stops it with the reason
    C-R1-4g cooperative_cancel records no failure       -> cooperative_cancel records it as the job's failure
    C-R1-4h owns_its_failure records no failure         -> a J-lens task records it as its failure
    C-R1-4i the training loop passes no lease_lost      -> the training loop asks the claim (AST: the CALL)
    C-R1-4j stop_signal_for ignores lease_lost          -> training stops at its status check
    C-R1-4k calibration's handler always "cancelled"    -> calibration[gpu_lease_lost-failed]
    C-R1-4l the recorder's handler always "cancelled"   -> steering_record[gpu_lease_lost-failed]
    C-R1-4m the J-lens fit's lease branch removed       -> j_lens_fit[gpu_lease_lost-failed]
    C-R1-4n lease_lost() ignores a dead renewer thread  -> a renewal thread that died counts as lost
    C-R1-4o the renewer is not given the held cards     -> 4 red (knows its cards, taken back, stop, cancel check)
  R1-2 (fix 2a5efeaf)
    C-R1-2a the precheck decides with True (c78f0f21)   -> refused at placement not parked; answer from the row
    C-R1-2b the wrapper passes no can_split             -> the same two
    C-R1-2c an unreadable row counts as "cannot split"  -> a row that cannot be read keeps waiting as before
    C-R1-2d the J-lens fit declares nothing             -> the live registry
    C-R1-2e training declares no row answer             -> the live registry
    C-R1-2f training_can_split inverted                 -> training splits only when it loads a base model (3)
    C-R1-2g the spec does not record can_split          -> the live registry
  R1-5 (fix 2a5efeaf)
    C-R1-5a the lease never leases in per-card mode     -> 4 red: export holds; failing lens; route leases; route refuses
    C-R1-5b the route does not give the card back       -> the synchronous route leases the card and gives it back
    C-R1-5c the lease's exit does not close             -> export holds; failing lens; route gives it back
    C-R1-5d a failed enter does not close its claim     -> a leased card this process cannot see is given back
    C-R1-5e the export refuses instead of waiting       -> the export asks for a bounded wait (survived until that test)
  R1-6 (fix 845376cc). The first cut of two tests marked the reserve inside
  `claiming()` and asserted AFTER the block — which releases the training's lease —
  so the job fitted because the card was free, not because the reserve stopped
  counting; the column assertion caught it and both now assert inside the claim.
    C-R1-6a host_reserve_allocated writes nothing        -> already-allocated not counted again; recorded and zeroed
    C-R1-6b the guard ignores recorded reserves          -> already-allocated not counted again
    C-R1-6c a new lease records no reserve               -> recorded when the lease is taken
    C-R1-6d acquire does not store the reserve           -> recorded when the lease is taken
    C-R1-6e the training never marks its reserve         -> the training marks its reserve once its pool exists (AST)
    C-R1-6f a pre-column row counts nothing              -> a row from before the column counts its kind's reserve
  Phase 3 control re-run on the line R1-6 touched:
    M8c other trainings' reserves dropped                -> not-allocated-yet still counts; pre-column row;
                                                            test_gpu_job_claim's training on the other card
"""

from __future__ import annotations

import threading
import weakref
from types import SimpleNamespace

import pytest
from celery import Task

from src.core.config import settings
from src.services import gpu_job_claim as C
from src.services import gpu_leases
from src.services.gpu_placement import GpuCard
from src.workers import gpu_job as G
from src.workers.gpu_supervisor import WORKER_GPU_ENV
from tests.unit.gpu_lease_db import clear, lease_engine, session_factory

TI_UUID = "GPU-f47ba814-49a2-603f-3595-275284140251"
RTX_UUID = "GPU-247aa582-0d1b-e161-8156-983ed1fefc57"
TI = GpuCard(0, TI_UUID, "NVIDIA GeForce RTX 3080 Ti", 12_288, 11_900)
RTX = GpuCard(1, RTX_UUID, "NVIDIA GeForce RTX 3090", 24_576, 21_500)
CARDS = [TI, RTX]
OTHER = "training:someone-else:00000000"

#: A thread that should finish at once is given this long before the test fails
#: instead of hanging.
HANG_S = 10.0


@pytest.fixture(scope="module")
def engine():
    eng = lease_engine("mistudio_test_gpu_claim_review_r1")
    yield eng
    eng.dispose()


@pytest.fixture
def db(engine):
    clear(engine)
    return session_factory(engine)


def leases(db):
    with db() as s:
        return gpu_leases.live_leases(s)


def take(db, uuid, holder=OTHER, task_id="other-task"):
    with db() as s:
        assert gpu_leases.acquire(s, [uuid], holder, task_id=task_id)


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


class FakeTask(Task):
    request = None  # shadows Task.request (a property over the request stack)
    name = "tests.review_r1_gpu_task"

    def __init__(self, request):
        self.request = request
        self.published = []

    def apply_async(self, args=None, kwargs=None, **options):
        self.published.append((args, kwargs, options))
        return SimpleNamespace(id=options.get("task_id"))


def request(**overrides):
    base = dict(id="tid-1", args=["job-1"], kwargs={"gpu_request": "auto"}, timelimit=(None, None),
                gpu_hops=None, headers=None, argsrepr=None, kwargsrepr=None)
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.fixture
def node(engine, monkeypatch):
    """Per-card mode on the two-card node, the 3090's worker, real leases, parking recorded."""
    clear(engine)
    db = session_factory(engine)
    parked = []
    monkeypatch.setattr(settings, "gpu_worker_mode", "per_card")
    monkeypatch.setenv(WORKER_GPU_ENV, RTX_UUID)
    monkeypatch.setattr(C, "_sync_session", db)
    monkeypatch.setattr(C, "list_cards", lambda: list(CARDS))
    monkeypatch.setattr(C, "host_available_mb", lambda: 1_000_000.0)
    monkeypatch.setattr(G, "park_job", lambda *a, **k: parked.append((a, k)))
    return SimpleNamespace(db=db, parked=parked)


# ── R1-1: the card is not offered until the job's memory is released ─────────


class _Held:
    """Stands in for a model or buffer a job's frame holds."""


class TestTheCardIsNotOfferedUntilItsMemoryIsReleased:
    def test_a_failed_jobs_frames_are_freed_before_its_lease_is_released(self, node, monkeypatch):
        """An OOM propagates through the claim. The traceback holds the body's frame,
        and with it the model; the release before the lease release must free it."""
        refs, seen = [], {}
        real_release = gpu_leases.release

        def spying_release(db, holder, uuids=None):
            seen["alive_at_release"] = refs[0]() is not None
            seen["leases_at_release"] = gpu_leases.live_leases(db)
            return real_release(db, holder, uuids)

        monkeypatch.setattr(gpu_leases, "release", spying_release)

        @G.gpu_job("test")
        def run(self, job_id, gpu_request="auto"):
            C.claim_cards(gpu_request, required_mb=4_000)
            model = _Held()
            refs.append(weakref.ref(model))
            raise RuntimeError("CUDA out of memory")

        with pytest.raises(RuntimeError, match="out of memory"):
            run(FakeTask(request()), "job-1")
        assert list(seen["leases_at_release"]) == [RTX_UUID], "the lease was gone before the release ran"
        assert seen["alive_at_release"] is False, "the body's frame still held the model when the card was released"
        assert leases(node.db) == {}

    def test_the_tasks_own_release_runs_while_the_card_is_still_leased(self, node):
        seen = []

        class TrainingLike(FakeTask):
            def release_job_memory(self):
                seen.append(leases(node.db))

        @G.gpu_job("test")
        def run(self, job_id, gpu_request="auto"):
            return C.claim_cards(gpu_request, required_mb=4_000)

        assert run(TrainingLike(request()), "job-1") == (RTX,)
        assert len(seen) == 1 and list(seen[0]) == [RTX_UUID]
        assert leases(node.db) == {}

    def test_a_job_that_placed_nothing_releases_nothing(self, node):
        seen = []

        class TrainingLike(FakeTask):
            def release_job_memory(self):
                seen.append("released")

        @G.gpu_job("test")
        def run(self, job_id, gpu_request="auto"):
            raise ValueError("refused before placing")

        with pytest.raises(ValueError):
            run(TrainingLike(request()), "job-1")
        assert seen == []

    def test_a_release_that_hangs_does_not_keep_the_card(self, db):
        gate = threading.Event()
        claim = ctx(db, worker=RTX, release_memory=lambda held: gate.wait(HANG_S * 3), release_timeout_s=0.2)
        try:
            claim.claim("auto", required_mb=4_000)
            closer = threading.Thread(target=claim.close, daemon=True)
            closer.start()
            closer.join(HANG_S)
            assert not closer.is_alive(), "close() waited on a hung memory release"
            assert leases(db) == {}
        finally:
            gate.set()

    def test_a_release_that_raises_does_not_keep_the_card(self, db):
        def release(held):
            raise RuntimeError("the activation buffer's close failed")

        claim = ctx(db, worker=RTX, release_memory=release)
        claim.claim("auto", required_mb=4_000)
        claim.close()
        assert leases(db) == {}

    def test_the_release_is_given_the_cards_the_job_held(self, db):
        got = []
        claim = ctx(db, worker=RTX, release_memory=lambda held: got.append((tuple(held), leases(db))))
        claim.claim("auto", required_mb=30_000, allow_shard=True)
        claim.close()
        ((held, at_release),) = got
        assert sorted(held) == sorted([RTX_UUID, TI_UUID])
        assert at_release == {RTX_UUID: claim.holder, TI_UUID: claim.holder}
        assert leases(db) == {}


# ── R1-4: a lease that lapses is taken back, or the job stops ────────────────

import ast  # noqa: E402
import pathlib  # noqa: E402

from sqlalchemy import text  # noqa: E402

from src.core import cancellation  # noqa: E402
from src.core.cancellation import GPU_LEASE_LOST, OperatorCancelled  # noqa: E402
from src.models.training import TrainingStatus  # noqa: E402
from src.workers import jlens_progress  # noqa: E402
from src.workers.training_tasks import stop_signal_for  # noqa: E402

SRC = pathlib.Path(__file__).resolve().parents[2] / "src"


def lapse(engine, holder):
    """The holder's lease expired and its heartbeat went stale, as after a long outage."""
    with engine.begin() as conn:
        conn.execute(
            text("UPDATE gpu_leases SET expires_at = now() - interval '1 second', "
                 "heartbeat_at = now() - interval '11 minutes' WHERE holder = :h"),
            {"h": holder},
        )


class TestALeaseThatLapses:
    @staticmethod
    def _claim(db, **kwargs):
        # The timer never fires on its own here; the test drives each renewal with tick().
        claim = ctx(db, worker=RTX, renew_interval_s=3600.0, **kwargs)
        claim.claim("auto", required_mb=4_000)
        return claim

    def test_the_renewal_knows_the_cards_it_holds(self, db):
        claim = self._claim(db, task_id="celery-abc")
        try:
            assert (claim._renewer.uuids, claim._renewer.task_id) == ((RTX_UUID,), "celery-abc")
        finally:
            claim.close()

    def test_it_is_taken_back_when_no_other_job_took_the_card(self, db, engine):
        claim = self._claim(db)
        try:
            lapse(engine, claim.holder)
            assert leases(db) == {}
            assert claim._renewer.tick() == 1
            assert leases(db) == {RTX_UUID: claim.holder}
            assert claim.lease_lost() is None
        finally:
            claim.close()

    def test_the_job_is_told_to_stop_when_another_job_now_holds_the_card(self, db, engine):
        claim = self._claim(db)
        try:
            lapse(engine, claim.holder)
            take(db, RTX_UUID)
            assert claim._renewer.tick() == 0
            assert "another job now holds" in claim.lease_lost()
            with C.claiming(claim):
                assert "another job now holds" in C.lease_lost_reason()
            assert leases(db) == {RTX_UUID: OTHER}
        finally:
            claim.close()
        assert leases(db) == {RTX_UUID: OTHER}, "the job that lost its lease freed the other job's card"

    def test_the_job_is_told_to_stop_when_no_renewal_succeeds_for_a_whole_ttl(self, db):
        clock = {"t": 0.0}
        outage = {"on": False}

        def session():
            if outage["on"]:
                raise ConnectionError("the database is unreachable")
            return db()

        with db() as s:
            assert gpu_leases.acquire(s, [RTX_UUID], "test:t:0000", task_id="t")
        renewer = C.LeaseRenewer("test:t:0000", session, uuids=[RTX_UUID], task_id="t",
                                 ttl_s=600.0, clock=lambda: clock["t"])
        outage["on"] = True
        clock["t"] = 599.0
        assert renewer.tick() == -1 and renewer.lost is None
        clock["t"] = 600.0
        assert renewer.tick() == -1
        assert "could not be renewed for 600 s" in renewer.lost

    def test_a_renewal_after_an_outage_shorter_than_the_ttl_resets_the_clock(self, db):
        clock = {"t": 0.0}
        outage = {"on": True}

        def session():
            if outage["on"]:
                raise ConnectionError("the database is unreachable")
            return db()

        with db() as s:
            assert gpu_leases.acquire(s, [RTX_UUID], "test:t:1111", task_id="t")
        renewer = C.LeaseRenewer("test:t:1111", session, uuids=[RTX_UUID], ttl_s=600.0, clock=lambda: clock["t"])
        clock["t"] = 500.0
        assert renewer.tick() == -1
        outage["on"] = False
        assert renewer.tick() == 1
        outage["on"] = True
        clock["t"] = 1_000.0
        assert renewer.tick() == -1 and renewer.lost is None

    def test_a_raising_renewal_callback_does_not_end_the_renewals(self, db):
        calls, second = [], threading.Event()

        def on_renew(renewed):
            calls.append(renewed)
            if len(calls) == 1:
                raise RuntimeError("a bug in a callback")
            second.set()

        claim = ctx(db, worker=RTX, renew_interval_s=0.02, on_renew=on_renew)
        claim.claim("auto", required_mb=4_000)
        try:
            assert second.wait(HANG_S), "the renewal thread died with its callback"
        finally:
            claim.close()

    def test_a_renewal_thread_that_died_counts_as_a_lost_lease(self, db):
        claim = self._claim(db)
        try:
            dead = threading.Thread(target=lambda: None)
            dead.start()
            dead.join()
            claim._renewer._thread = dead
            assert "stopped while the job still runs" in claim.lease_lost()
        finally:
            claim.close()


class TestAJobThatLostItsLeaseStops:
    @staticmethod
    def _lost_claim(db, engine):
        claim = ctx(db, worker=RTX, renew_interval_s=3600.0)
        claim.claim("auto", required_mb=4_000)
        lapse(engine, claim.holder)
        take(db, RTX_UUID)
        claim._renewer.tick()
        assert claim.lease_lost()
        return claim

    def test_a_cancel_check_stops_it_with_the_reason(self, db, engine):
        claim = self._lost_claim(db, engine)
        try:
            with C.claiming(claim):
                check = cancellation.cancel_checker("activation_extraction", "ext-1", db=object())
                assert check.poll_now() is True
                assert check.reason == GPU_LEASE_LOST
                with pytest.raises(OperatorCancelled) as stopped:
                    check.raise_if_cancelled("at batch 3")
            assert stopped.value.reason == GPU_LEASE_LOST
            assert "another job now holds" in stopped.value.detail and "at batch 3" in stopped.value.detail
        finally:
            claim.close()

    def test_outside_a_claim_the_check_reads_only_the_row(self, monkeypatch):
        check = cancellation.cancel_checker("activation_extraction", "ext-1", db=object())
        assert check.poll_now() is False  # the fake session fails the row read; no lease is consulted

    def test_cooperative_cancel_records_it_as_the_jobs_failure(self, monkeypatch):
        recorded = []
        monkeypatch.setattr(cancellation, "record_progress", lambda *a, **k: recorded.append((a, k)) or True)

        @cancellation.cooperative_cancel("activation_extraction")
        def lost():
            raise OperatorCancelled("activation_extraction", "ext-1", GPU_LEASE_LOST, "its lease lapsed")

        @cancellation.cooperative_cancel("activation_extraction")
        def cancelled():
            raise OperatorCancelled("activation_extraction", "ext-2", "cancelled", "at 3%")

        assert lost()["reason"] == GPU_LEASE_LOST
        assert cancelled()["reason"] == "cancelled"
        assert recorded == [
            (("activation_extraction", "ext-1"),
             {"status": "failed", "error_message": "Stopped: its lease lapsed. Run it again."}),
            (("activation_extraction", "ext-2"), {"error_message": "at 3%"}),
        ]

    def test_a_j_lens_task_records_it_as_its_failure(self, monkeypatch):
        rows = []
        monkeypatch.setattr(jlens_progress, "update_row", lambda *a, **k: rows.append((a, k)) or True)
        monkeypatch.setattr(jlens_progress, "fail_row", lambda *a, **k: pytest.fail("recorded as a crash"))

        @jlens_progress.owns_its_failure
        def lost(self):
            raise OperatorCancelled("jlens_task", "tid-j", GPU_LEASE_LOST, "its lease lapsed")

        @jlens_progress.owns_its_failure
        def cancelled(self):
            raise OperatorCancelled("jlens_task", "tid-k", "cancelled", "")

        task = SimpleNamespace(request=SimpleNamespace(id="tid-j"))
        assert lost(task)["status"] == "failed"
        assert cancelled(SimpleNamespace(request=SimpleNamespace(id="tid-k")))["status"] == "cancelled"
        assert rows == [(("tid-j",), {"status": "failed", "error_message": "Stopped: its lease lapsed. Run it again."})]

    def test_training_stops_at_its_status_check_and_the_operator_still_wins(self):
        running = SimpleNamespace(status=TrainingStatus.RUNNING.value)
        cancelled = SimpleNamespace(status=TrainingStatus.CANCELLED.value)
        assert stop_signal_for(running, 50) is None
        assert stop_signal_for(running, 50, lease_lost="its lease lapsed") == {
            "status": "failed", "step": 50, "reason": GPU_LEASE_LOST, "detail": "its lease lapsed",
        }
        assert stop_signal_for(cancelled, 50, lease_lost="its lease lapsed") == {"status": "cancelled", "step": 50}

    def test_the_training_loop_asks_the_claim_and_records_the_failure(self):
        """The CALL, by AST: the loop's status check passes lease_lost=lease_lost_reason(), and
        the loop records a FAILED status for a lease-loss stop."""
        tree = ast.parse((SRC / "workers" / "training_tasks.py").read_text())
        train = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "train_sae_task")
        checks = [n for n in ast.walk(train) if isinstance(n, ast.Call)
                  and getattr(n.func, "id", None) == "stop_signal_for"]
        assert len(checks) == 1, "the walk found no single status check"
        (keyword,) = [k for k in checks[0].keywords if k.arg == "lease_lost"]
        assert isinstance(keyword.value, ast.Call) and getattr(keyword.value.func, "id", None) == "lease_lost_reason"
        failures = [
            n for n in ast.walk(train)
            if isinstance(n, ast.If) and isinstance(n.test, ast.Compare)
            and any(getattr(c, "id", None) == "GPU_LEASE_LOST" for c in n.test.comparators)
        ]
        assert failures, "no branch on a lease-loss stop"
        writes = [c for f in failures for c in ast.walk(f) if isinstance(c, ast.Call)
                  and getattr(c.func, "id", None) == "record_progress"
                  and any(k.arg == "status" and "FAILED" in ast.unparse(k.value) for k in c.keywords)]
        assert len(writes) == 1


class TestTasksWithTheirOwnStopHandlerRecordALostLeaseAsAFailure:
    """Three GPU tasks catch the stop themselves, ahead of the shared recorders. Each
    must write a lease loss as the job's failure and still write an operator's stop
    as a cancellation. Driven through each task's real body, placement patched to stop."""

    @staticmethod
    def _stop(reason):
        def place(*args, **kwargs):
            raise OperatorCancelled("circuit_calibration", "crc-1", reason, "its lease lapsed")

        return place

    @pytest.mark.parametrize("reason, outcome", [(GPU_LEASE_LOST, "failed"), ("cancelled", "cancelled")])
    def test_calibration(self, monkeypatch, reason, outcome):
        from src.workers import circuit_calibration_tasks as T

        statuses, emitted = [], []
        monkeypatch.setattr(T, "_refuse_if_cancelled", lambda db, cid: True)
        monkeypatch.setattr(T, "_set_status", lambda db, cid, status: statuses.append((cid, status)))
        monkeypatch.setattr(T, "circuit_steering_manifest", lambda db, cid: None)
        monkeypatch.setattr(T, "circuit_required_mb", lambda db, manifest: None)
        monkeypatch.setattr(T, "place_circuit_job", self._stop(reason))
        monkeypatch.setattr(T, "release_circuit_job", lambda *args, **kwargs: None)
        monkeypatch.setattr(T, "emit_circuit_run_failed", lambda kind, cid, message: emitted.append(message))

        result = T.run_circuit_calibration.apply(args=["crc-1", {}, "auto"]).get()
        assert result["status"] == outcome
        assert statuses == [("crc-1", "running"), ("crc-1", outcome)]
        assert emitted == (["stopped: its lease lapsed"] if outcome == "failed" else ["cancelled"])

    @pytest.mark.parametrize("reason, outcome", [(GPU_LEASE_LOST, "failed"), ("cancelled", "cancelled")])
    def test_steering_record(self, monkeypatch, reason, outcome):
        from src.workers import circuit_record_tasks as T

        statuses = []
        monkeypatch.setattr(T, "_refuse_if_cancelled", lambda db, rid: True)
        monkeypatch.setattr(T, "_set_status", lambda db, rid, status, error=None: statuses.append((rid, status, error)))
        monkeypatch.setattr(T, "_place_and_record", self._stop(reason))
        monkeypatch.setattr(T, "release_circuit_job", lambda *args, **kwargs: None)
        monkeypatch.setattr(T, "emit_circuit_run_failed", lambda kind, rid, message: None)

        result = T.run_circuit_record.apply(args=["rec-1", {}]).get()
        assert result["status"] == outcome
        assert statuses == [("rec-1", "running", None), ("rec-1", outcome, "its lease lapsed")]

    @pytest.mark.parametrize("reason, outcome", [(GPU_LEASE_LOST, "failed"), ("cancelled", "cancelled")])
    def test_j_lens_fit(self, monkeypatch, reason, outcome):
        from src.ml import jlens_fitter
        from src.workers import jlens_fit_tasks as T

        rows = []

        class StoppedFitter:
            def __init__(self, *args, **kwargs):
                pass

            def fit(self, prompts, layers=None, on_progress=None):
                raise jlens_progress.TaskCancelled("jlens_task", "tid-fit", reason, "its lease lapsed")

        monkeypatch.setattr(jlens_fitter, "JacobianFitter", StoppedFitter)
        monkeypatch.setattr(jlens_progress, "mark_running", lambda *a, **k: True)
        monkeypatch.setattr(jlens_progress, "update_row", lambda *a, **k: rows.append((a, k)) or True)
        task = SimpleNamespace(request=SimpleNamespace(id="tid-fit"), update_state=lambda **kwargs: None)
        loaded = SimpleNamespace(model=None, tokenizer=None, structure=None)

        result = T._fit_and_publish(
            task, loaded, "m1", "org/repo", ["p"], None, False, "corpus", None, False, False, False, "final", None,
        )
        assert result["status"] == outcome
        if outcome == "failed":
            assert rows == [(("tid-fit",), {"status": "failed", "error_message": "Stopped: its lease lapsed. Run it again."})]
        else:
            assert [k.get("status") for _, k in rows] == [None]


# ── R1-2: a stale "all" for a job that cannot split is refused, never parked ─

from celery.exceptions import Ignore  # noqa: E402

from src.core.celery_app import celery_app  # noqa: E402
from src.services.gpu_dispatch import gpu_job_spec  # noqa: E402
from src.services.gpu_placement import GpuPlacementError  # noqa: E402
from tests.unit.test_gpu_job_wiring import GPU_QUEUE_TASKS  # noqa: E402


class TestAnAllJobThatCannotSplit:
    """The precheck decided with `allow_shard=True` for every job, so an "all" message for a
    job that can never split was PARKED whenever a card was busy — for as long as the card
    stayed busy — before its placement finally refused it. Submit refuses such a request
    now, but a message queued before that, or a training whose row no longer splits, still
    arrives."""

    def test_it_goes_to_its_placement_and_is_refused_not_parked(self, node):
        take(node.db, TI_UUID)
        ran = []

        @G.gpu_job("test", can_split=False)
        def run(self, job_id, gpu_request="auto"):
            ran.append(gpu_request)
            return C.claim_cards(gpu_request, required_mb=4_000, allow_shard=False)

        task = FakeTask(request(kwargs={"gpu_request": "all"}))
        with pytest.raises(GpuPlacementError, match="cannot run split"):
            run(task, "job-1", gpu_request="all")
        assert ran == ["all"]
        assert node.parked == [] and task.published == []
        assert leases(node.db) == {TI_UUID: OTHER}

    def test_a_job_that_can_split_still_waits_for_every_card(self, node):
        take(node.db, TI_UUID)

        @G.gpu_job("test", can_split=True)
        def run(self, job_id, gpu_request="auto"):
            pytest.fail("the job ran while a card it needs was busy")

        with pytest.raises(Ignore):
            run(FakeTask(request(kwargs={"gpu_request": "all"})), "job-1", gpu_request="all")
        assert len(node.parked) == 1

    def test_the_answer_can_come_from_the_row(self, node):
        take(node.db, TI_UUID)
        asked = []

        def from_row(arguments):
            asked.append(arguments["job_id"])
            return False

        @G.gpu_job("test", can_split=from_row)
        def run(self, job_id, gpu_request="auto"):
            return C.claim_cards(gpu_request, required_mb=4_000, allow_shard=False)

        with pytest.raises(GpuPlacementError, match="cannot run split"):
            run(FakeTask(request(kwargs={"gpu_request": "all"})), "job-7", gpu_request="all")
        assert asked == ["job-7"] and node.parked == []

    def test_a_row_that_cannot_be_read_keeps_waiting_as_before(self, node):
        take(node.db, TI_UUID)

        def unreadable(arguments):
            raise ConnectionError("the database is unreachable")

        @G.gpu_job("test", can_split=unreadable)
        def run(self, job_id, gpu_request="auto"):
            pytest.fail("the job ran while a card it needs was busy")

        with pytest.raises(Ignore):
            run(FakeTask(request(kwargs={"gpu_request": "all"})), "job-1", gpu_request="all")
        assert len(node.parked) == 1


class TestEveryGpuQueueTaskSaysWhetherItSplits:
    """Read from the LIVE registry. A new GPU-queue task fails here until it is classified;
    each True names a job whose placement passes allow_shard=True (see
    test_every_gpu_request_says_whether_its_job_splits.py for the endpoints' side)."""

    EXPECTED = {name: True for name in GPU_QUEUE_TASKS} | {
        "src.workers.jlens_fit_tasks.fit_jlens_artifact": False,
        "train_sae": "row",
    }

    def test_the_live_registry(self):
        declared = {}
        for name in GPU_QUEUE_TASKS:
            value = gpu_job_spec(celery_app.tasks[name]).can_split
            declared[name] = "row" if callable(value) else value
        assert declared == self.EXPECTED
        assert gpu_job_spec(celery_app.tasks["train_sae"]).can_split is G.training_can_split

    @pytest.mark.parametrize("extraction_id, extraction_ids, splits", [
        (None, None, True),          # on the fly: loads the base model, may split
        ("ext_1", None, False),      # one cached extraction
        (None, ["ext_1", "ext_2"], False),  # a mixture, with no single extraction_id
    ])
    def test_training_splits_only_when_it_loads_a_base_model(self, monkeypatch, extraction_id, extraction_ids, splits):
        import contextlib

        from src.core import database

        row = SimpleNamespace(id="train_1", extraction_id=extraction_id, extraction_ids=extraction_ids)

        class Query:
            def filter(self, *args):
                return self

            def first(self):
                return row

        @contextlib.contextmanager
        def get_sync_db():
            yield SimpleNamespace(query=lambda model: Query())

        monkeypatch.setattr(database, "get_sync_db", get_sync_db)
        assert G.training_can_split({"training_id": "train_1"}) is splits


# ── R1-5: a logit lens computed in the API process leases its card ─────────

import asyncio  # noqa: E402
import contextlib  # noqa: E402
from unittest.mock import AsyncMock, MagicMock, patch  # noqa: E402

import torch  # noqa: E402
from fastapi import HTTPException  # noqa: E402

from src.api.v1.endpoints import neuronpedia as np_ep  # noqa: E402
from src.services import gpu_placement  # noqa: E402
from src.services import logit_lens_service as lens_module  # noqa: E402
from src.services import neuronpedia_export_service as export_module  # noqa: E402


@pytest.fixture
def api_node(engine, monkeypatch):
    """Per-card mode, the API process (no worker card, no claim), real leases, CUDA faked."""
    clear(engine)
    db = session_factory(engine)
    index_of = {card.uuid: card.index for card in CARDS}
    monkeypatch.setattr(settings, "gpu_worker_mode", "per_card")
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


def _lens(db, seen, fail=False):
    service = MagicMock()

    async def compute(*args, **kwargs):
        seen.append((dict(leases(db)), kwargs.get("device")))
        if fail:
            raise RuntimeError("CUDA out of memory")
        return {3: "r3"}

    service.compute_logit_lens_for_sae = compute
    service.save_logit_lens_results = AsyncMock()
    return service


class TestALogitLensInTheApiProcessLeasesItsCard:
    """The Neuronpedia export (a BackgroundTask) and the synchronous dashboard route
    compute the lens in the API process, outside any Celery task: no lease covered the
    card, so a GPU worker could place a job on the card the lens was loading a model onto."""

    @staticmethod
    def _export(config_gpu, service):
        exporter = export_module.NeuronpediaExportService.__new__(export_module.NeuronpediaExportService)
        with patch.object(export_module, "get_logit_lens_service", return_value=service):
            return asyncio.run(exporter._compute_logit_lens(
                db=None, sae=SimpleNamespace(id="sae-1"), features=[SimpleNamespace(neuron_index=3)],
                config=SimpleNamespace(gpu=config_gpu, logit_lens_k=5),
            ))

    def test_the_export_holds_the_card_while_the_lens_runs(self, api_node):
        seen = []
        self._export(RTX_UUID, _lens(api_node, seen))
        ((held, device),) = seen
        assert list(held) == [RTX_UUID] and held[RTX_UUID].startswith("export_logit_lens:no-task:")
        assert device == torch.device("cuda", 1)
        assert leases(api_node) == {}

    def test_a_lens_that_fails_releases_the_card(self, api_node):
        seen = []
        with pytest.raises(RuntimeError, match="out of memory"):
            self._export(RTX_UUID, _lens(api_node, seen, fail=True))
        assert list(seen[0][0]) == [RTX_UUID]
        assert leases(api_node) == {}

    @staticmethod
    def _route(service, gpu=RTX_UUID):
        request = np_ep.ComputeDashboardDataRequest(
            sae_id="sae_1", feature_indices=[0, 1], include_histograms=False, include_top_tokens=False, gpu=gpu,
        )
        with patch("src.services.logit_lens_service.get_logit_lens_service", return_value=service):
            return asyncio.run(np_ep.compute_dashboard_data(request, background_tasks=MagicMock(), db=AsyncMock()))

    def test_the_synchronous_route_leases_the_card_and_gives_it_back(self, api_node):
        seen = []
        response = self._route(_lens(api_node, seen))
        ((held, device),) = seen
        assert list(held) == [RTX_UUID] and held[RTX_UUID].startswith("dashboard_logit_lens_api:no-task:")
        assert device == torch.device("cuda", 1)
        assert response.status == "completed"
        assert leases(api_node) == {}

    def test_the_synchronous_route_refuses_a_card_another_job_holds_at_once(self, api_node):
        take(api_node, RTX_UUID)
        service = MagicMock()
        service.compute_logit_lens_for_sae = AsyncMock(side_effect=AssertionError("the lens ran on a card another job holds"))
        with pytest.raises(HTTPException) as refused:
            self._route(service)
        assert refused.value.status_code == 409 and "Waited 0 min" in refused.value.detail
        service.compute_logit_lens_for_sae.assert_not_called()
        assert leases(api_node) == {RTX_UUID: OTHER}

    def test_single_mode_resolves_the_device_as_before_and_leases_nothing(self, monkeypatch):
        monkeypatch.setattr(settings, "gpu_worker_mode", "single")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(C, "_sync_session", lambda: pytest.fail("single mode opened a lease session"))
        monkeypatch.setattr(lens_module, "logit_lens_device", lambda requested: ("resolved", requested))

        async def use():
            async with lens_module.logit_lens_lease(RTX_UUID, kind="export_logit_lens") as device:
                return device

        assert asyncio.run(use()) == ("resolved", RTX_UUID)

    def test_the_export_asks_for_its_lease_with_a_bounded_wait_not_a_refusal(self, api_node, monkeypatch):
        """The export is a background job: a card another job holds is waited for, not refused.
        The CALL and its payload: once, the stored request, its own holder kind, no zero wait."""
        calls = []

        class RecordingLease:
            async def __aenter__(self):
                return torch.device("cuda", 1)

            async def __aexit__(self, *exc_info):
                return False

        def recording(*args, **kwargs):
            calls.append((args, kwargs))
            return RecordingLease()

        monkeypatch.setattr(lens_module, "logit_lens_lease", recording)
        self._export(RTX_UUID, _lens(api_node, []))
        assert calls == [((RTX_UUID,), {"kind": "export_logit_lens"})]

    def test_a_leased_card_this_process_cannot_see_is_given_back(self, api_node, monkeypatch):
        def invisible(card):
            raise GpuPlacementError(f"GPU {getattr(card, 'uuid', card)} is not visible to this process.")

        monkeypatch.setattr(gpu_placement, "torch_device", invisible)
        with pytest.raises(GpuPlacementError, match="not visible"):
            self._export(RTX_UUID, _lens(api_node, []))
        assert leases(api_node) == {}


# ── R1-6: a training's pinned host buffer is counted once ───────────────────


class TestTheHostRamGuardCountsAReserveOnce:
    """The guard subtracted a 24 GiB reserve for every other live training, because its
    pinned rolling buffer may not be allocated yet. Once it IS allocated, MemAvailable
    already excludes it, and the reserve counted it twice: on the 124 GB node, a job that
    fit beside a running training was made to wait for as long as the training ran."""

    @staticmethod
    def _training(db):
        """A training holding the 3080 Ti (named, so the claim is RunHere on its own worker)."""
        training = C.ClaimContext(
            holder=C.make_holder("training", "t1"), task_id="t1", worker_uuid=TI_UUID, session=db,
            inventory=lambda: list(CARDS), available_mb=lambda: 1_000_000.0, renew_interval_s=3600.0,
        )
        assert training.claim(TI_UUID, required_mb=2_000) == (TI,)
        return training

    def test_a_training_that_already_allocated_its_buffer_is_not_counted_again(self, db):
        # INSIDE the training's claim: `claiming()` releases its leases on exit, and a
        # test that marked the reserve and left the block would pass because the
        # training's card was free, not because its reserve stopped counting.
        with C.claiming(self._training(db)) as training:
            C.host_reserve_allocated()
            # 30 GB available, 25% kept free = 22.5 GB usable: the 8 GB job fits.
            job = ctx(db, worker=RTX, available=30_000, task_id="e1")
            try:
                assert job.claim("auto", required_mb=8_000) == (RTX,)
                assert leases(db) == {TI_UUID: training.holder, RTX_UUID: job.holder}
            finally:
                job.close()

    def test_a_training_that_has_not_allocated_yet_still_counts(self, db):
        training = self._training(db)
        try:
            with pytest.raises(C.JobHandoff, match="host RAM"):
                ctx(db, worker=RTX, available=30_000, task_id="e1").claim("auto", required_mb=8_000)
        finally:
            training.close()

    def test_the_reserve_is_recorded_when_the_lease_is_taken_and_zeroed_once_allocated(self, db, engine):
        def reserves():
            with engine.connect() as conn:
                return dict(conn.execute(text("SELECT gpu_uuid, host_reserve_mb FROM gpu_leases")).all())

        # INSIDE the training's claim, which releases its leases when the block exits.
        with C.claiming(self._training(db)):
            other = ctx(db, worker=RTX, task_id="e1")
            other.claim(RTX_UUID, required_mb=2_000)
            try:
                assert reserves() == {TI_UUID: C.HOST_RESERVE_MB_BY_KIND["training"], RTX_UUID: 0}
                C.host_reserve_allocated()
                assert reserves() == {TI_UUID: 0, RTX_UUID: 0}
            finally:
                other.close()

    def test_a_row_from_before_the_column_counts_its_kinds_reserve(self, db):
        take(db, TI_UUID, holder="training:old:11111111")
        with pytest.raises(C.JobHandoff, match="host RAM"):
            ctx(db, worker=RTX, available=30_000, task_id="e1").claim("auto", required_mb=8_000)

    def test_a_card_marked_unavailable_by_the_supervisor_is_not_a_job_loading(self, db):
        """`gpu-worker-down:<uuid>` (gpu_supervisor.CardAvailability) holds a card whose worker
        is not running. It loads nothing, so a FIRST job is not host-RAM guarded because of it."""
        from src.workers.gpu_supervisor import CardAvailability

        take(db, TI_UUID, holder=CardAvailability.holder_for(TI_UUID), task_id=None)
        assert ctx(db, worker=RTX, available=10.0).claim("auto", required_mb=8_000) == (RTX,)

    def test_the_not_a_job_kinds_are_the_supervisors_own_holder(self):
        from src.workers.gpu_supervisor import CardAvailability

        assert C.NOT_A_JOB_HOLDER_KINDS == {CardAvailability.HOLDER_PREFIX}
        assert C.holder_kind(CardAvailability.holder_for(TI_UUID)) == CardAvailability.HOLDER_PREFIX

    def test_marking_outside_a_claim_or_without_a_database_never_raises(self, db):
        C.host_reserve_allocated()
        claim = ctx(db, worker=RTX, renew_interval_s=3600.0)
        claim.claim("auto", required_mb=2_000)
        try:
            claim.session = lambda: (_ for _ in ()).throw(ConnectionError("the database is unreachable"))
            with C.claiming(claim):
                C.host_reserve_allocated()
        finally:
            claim.session = db
            claim.close()

    def test_the_training_marks_its_reserve_once_its_pool_exists(self):
        """The CALL, by AST: after the rolling buffer and the pinned pool are built, before the step loop."""
        tree = ast.parse((SRC / "workers" / "training_tasks.py").read_text())
        train = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "train_sae_task")

        def call_lines(name):
            return sorted(n.lineno for n in ast.walk(train) if isinstance(n, ast.Call)
                          and (getattr(n.func, "attr", None) or getattr(n.func, "id", None)) == name)

        marks = call_lines("host_reserve_allocated")
        assert len(marks) == 1, marks
        built = call_lines("RollingActivationBuffer") + call_lines("pin_within_budget")
        assert built and marks[0] > max(built), (marks, built)
        step_loops = [n.lineno for n in ast.walk(train) if isinstance(n, ast.For)
                      and isinstance(n.iter, ast.Call) and getattr(n.iter.func, "id", None) == "range"
                      and [getattr(a, "id", None) for a in n.iter.args] == ["start_step", "total_steps"]]
        assert len(step_loops) == 1 and marks[0] < step_loops[0], (marks, step_loops)
