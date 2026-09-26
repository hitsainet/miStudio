"""A redelivered copy of a job that is still running is dropped, not run on the other card (R1-7).

Multi-GPU Phase 3, review round 1. Every task here acks late (``task_acks_late=True``),
and kombu's Redis transport RESTORES an unacked message to its queue once it is older
than ``visibility_timeout`` (12 h here): ``QoS.restore_visible`` runs in every consuming
worker and does not ask whether the original consumer is alive. A J-lens fit (hours to
days, and a solo worker enforces no time limit) or a long circuit run passes 12 h.

With ONE GPU worker that restored copy waited behind the running original. With a worker
PER CARD the other card's worker takes it at once: a new execution of the same task id,
with its own holder nonce, leases the free card and runs the same job concurrently —
two fits writing one staging directory, two runs writing one row.

The rule now: a delivery the broker RESTORED (kombu sets ``delivery_info['redelivered']``)
that finds a LIVE lease for its own task id, held by an execution whose heartbeat is fresh
(``STALE_HEARTBEAT_S``), is a copy. It is acked and dropped — not run, not parked, not
republished — and says so in the log. A lease whose holder stopped renewing belongs to a
dead execution, so a genuine redelivery after a crash still runs.

ONLY A RESTORED DELIVERY. The first cut of this rule looked at every delivery, and would
have dropped a RETRY: Celery publishes the next attempt before the failing one leaves
`claiming()`, whose memory release can hold the lease for up to two minutes, and
extract_activations, SAE extraction and labeling do retry. A retry and a hand-off are
fresh publishes, never marked redelivered.

PROOF: on the unfixed tree `test_it_is_dropped_before_it_does_anything` and
`test_a_task_that_waits_in_place_is_dropped_at_its_claim` were red (the copy ran and
leased the other card).

MUTATION CONTROLS (2026-09-14, fix a353004d; scratchpad p3-r1-claim/mutate.py, each alone,
restored byte-identically with sha256 checked, `git diff` clean) — all 9 red:
  C-R1-7a the precheck does not refuse a copy          -> dropped before it does anything
  C-R1-7b claim does not refuse a copy                 -> a task that waits in place; a J-lens copy
  C-R1-7c the lease query ignores the heartbeat        -> a copy of a job whose execution died still runs; the query
  C-R1-7d the lease query ignores the task id          -> a different job on the other card; the query
  C-R1-7e the wrapper does not catch the copy          -> dropped; waits in place; a J-lens copy
  C-R1-7f owns_its_failure records the copy's drop     -> a J-lens copy does not fail the original's row
  C-R1-7g claim_exact does not refuse a copy           -> a copy that would reuse a resident model
  C-R1-7h any delivery is judged, not only a restored one -> a retry racing the attempt before it; a hand-off
  C-R1-7i the wrapper never reads delivery_info        -> dropped; waits in place; a J-lens copy
(C-R1-7d first matched two lines — `release_stale_for_task` filters by task id too — and
was skipped by the runner; re-run with a unique anchor.)

REAL POSTGRES leases, a fake two-card inventory, the real ``@gpu_job`` wrapper.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from celery import Task
from celery.exceptions import Ignore
from sqlalchemy import text

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
ORIGINAL = "test:tid-long:aaaaaaaa"


class FakeTask(Task):
    request = None
    name = "tests.redelivered_gpu_task"

    def __init__(self, request):
        self.request = request
        self.published = []

    def apply_async(self, args=None, kwargs=None, **options):
        self.published.append((args, kwargs, options))
        return SimpleNamespace(id=options.get("task_id"))


def request(**overrides):
    """A delivery the broker RESTORED (kombu marks it redelivered), unless overridden."""
    base = dict(id="tid-long", args=["job-1"], kwargs={"gpu_request": "auto"}, timelimit=(None, None),
                gpu_hops=None, headers=None, argsrepr=None, kwargsrepr=None,
                delivery_info={"redelivered": True})
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.fixture(scope="module")
def engine():
    eng = lease_engine("mistudio_test_gpu_redelivered_duplicate")
    yield eng
    eng.dispose()


@pytest.fixture
def node(engine, monkeypatch):
    """Per-card mode; this delivery reaches the 3090's worker while the original runs on the 3080 Ti."""
    clear(engine)
    db = session_factory(engine)
    parked = []
    monkeypatch.setattr(settings, "gpu_worker_mode", "per_card")
    monkeypatch.setenv(WORKER_GPU_ENV, RTX_UUID)
    monkeypatch.setattr(C, "_sync_session", db)
    monkeypatch.setattr(C, "list_cards", lambda: list(CARDS))
    monkeypatch.setattr(C, "host_available_mb", lambda: 1_000_000.0)
    monkeypatch.setattr(G, "park_job", lambda *a, **k: parked.append((a, k)))
    monkeypatch.setattr(G, "mark_waiting", lambda task_id: None)
    with db() as s:
        assert gpu_leases.acquire(s, [TI_UUID], ORIGINAL, task_id="tid-long")
    return SimpleNamespace(db=db, engine=engine, parked=parked)


def leases(db):
    with db() as s:
        return gpu_leases.live_leases(s)


def stale(engine, holder):
    with engine.begin() as conn:
        conn.execute(text("UPDATE gpu_leases SET heartbeat_at = now() - interval '10 minutes' WHERE holder = :h"),
                     {"h": holder})


def job(handoff=True):
    ran = []

    @G.gpu_job("test", handoff=handoff)
    def run(self, job_id, gpu_request="auto"):
        ran.append(job_id)
        return C.claim_cards(gpu_request, required_mb=4_000)

    return run, ran


class TestARedeliveredCopyOfARunningJob:
    def test_it_is_dropped_before_it_does_anything(self, node):
        run, ran = job()
        task = FakeTask(request())
        with pytest.raises(Ignore):
            run(task, "job-1")
        assert ran == []
        assert task.published == [] and node.parked == []
        assert leases(node.db) == {TI_UUID: ORIGINAL}, "the duplicate leased the other card"

    def test_a_copy_of_a_job_whose_execution_died_still_runs(self, node):
        stale(node.engine, ORIGINAL)
        run, ran = job()
        assert run(FakeTask(request()), "job-1") == (RTX,)
        assert ran == ["job-1"]

    def test_a_different_job_on_the_other_card_is_not_affected(self, node):
        run, ran = job()
        assert run(FakeTask(request(id="tid-other")), "job-2") == (RTX,)
        assert ran == ["job-2"]

    def test_a_task_that_waits_in_place_is_dropped_at_its_claim(self, node):
        run, ran = job(handoff=False)
        task = FakeTask(request())
        with pytest.raises(Ignore):
            run(task, "job-1")
        assert ran == ["job-1"]  # its pre-placement work ran; its GPU work did not
        assert leases(node.db) == {TI_UUID: ORIGINAL}

    def test_single_mode_is_unchanged(self, node, monkeypatch):
        monkeypatch.setattr(settings, "gpu_worker_mode", "single")
        monkeypatch.setattr(C, "resolve_cards", lambda *a, **k: (RTX,))
        run, ran = job()
        assert run(FakeTask(request()), "job-1") == (RTX,)


class TestAResidentModelClaim:
    def test_a_copy_that_would_reuse_a_resident_model_is_refused(self, node):
        """`claim_exact` (a J-lens readout reusing a loaded copy, steering reusing its model)."""
        copy = C.ClaimContext(
            holder=C.make_holder("jlens_readout", "tid-long"), task_id="tid-long", worker_uuid=RTX_UUID,
            session=node.db, inventory=lambda: list(CARDS), available_mb=lambda: 1_000_000.0,
            redelivered=True,
        )
        with pytest.raises(C.DuplicateExecution):
            copy.claim_exact([RTX])
        assert leases(node.db) == {TI_UUID: ORIGINAL}


class TestAFreshPublishOfTheSameTaskIdIsNotACopy:
    """A retry is published by the failing attempt BEFORE its lease is released (Celery
    queues the retry, then raises), and extract_activations, SAE extraction and labeling
    retry. Dropping it as a copy would lose the job; only a RESTORED delivery can be one."""

    def test_a_retry_racing_the_attempt_before_it_runs_on_the_free_card(self, node):
        run, ran = job()
        retry = FakeTask(request(delivery_info={"redelivered": False}, retries=1))
        assert run(retry, "job-1") == (RTX,)
        assert ran == ["job-1"]

    def test_a_hand_off_republished_by_apply_async_is_not_a_copy(self, node):
        run, ran = job()
        republished = FakeTask(request(delivery_info={}))
        assert run(republished, "job-1") == (RTX,)
        assert ran == ["job-1"]


class TestAJLensCopyDoesNotFailTheOriginalsRow:
    def test_a_copy_dropped_at_its_claim_records_nothing(self, node, monkeypatch):
        from src.workers import jlens_progress

        failed = []
        monkeypatch.setattr(jlens_progress, "fail_row", lambda task_id, exc: failed.append((task_id, exc)))

        @G.gpu_job("jlens", handoff=False)
        @jlens_progress.owns_its_failure
        def run(self, job_id, gpu_request="auto"):
            return C.claim_cards(gpu_request, required_mb=4_000)

        with pytest.raises(Ignore):
            run(FakeTask(request()), "job-1")
        assert failed == [], "the copy marked the original's row failed"
        assert leases(node.db) == {TI_UUID: ORIGINAL}


class TestTheLeaseQuery:
    def test_only_other_live_fresh_executions_of_the_task_count(self, node):
        with node.db() as s:
            assert gpu_leases.live_holders_for_task(s, "tid-long", fresh_after_s=180) == {ORIGINAL}
            assert gpu_leases.live_holders_for_task(s, "tid-none", fresh_after_s=180) == set()
            assert gpu_leases.live_holders_for_task(s, None, fresh_after_s=180) == set()
        stale(node.engine, ORIGINAL)
        with node.db() as s:
            assert gpu_leases.live_holders_for_task(s, "tid-long", fresh_after_s=180) == set()
