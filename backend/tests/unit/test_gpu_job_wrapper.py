"""Every GPU task runs under a lease claim, and a hand-off is a re-dispatch (Phase 3).

``@gpu_job`` (``workers/gpu_job.py``) is exercised around a real Celery ``Task``
subclass whose ``apply_async`` is recorded, with REAL POSTGRES leases (see
``gpu_lease_db``) on a fake inventory: the RTX 3080 Ti (index 0) and the RTX 3090
(index 1).

MUTATION CONTROLS (2026-09-14; each alone, restored byte-identically, sha256 checked,
`git diff` clean) — all red:
  M9  the precheck is skipped                         -> five hand-off tests (named card, hops, general worker…)
  M10 the re-dispatch drops the task id               -> sent there before it does anything; parked not held
  M11 a hand-off returns instead of raising Ignore    -> seven hand-off tests
  M12 a wait uses Celery countdown instead of parking -> parked not held; passed back and forth; J-lens hand-off
  M13 the wrapper is a pass-through in per-card mode  -> 12 red, including every release test
  M15 owns_its_failure records a JobHandoff           -> a J-lens task does not record a hand-off as its failure
  M2/M3 (gpu_job_claim, also run here)                -> red here as well
  J3  the janitor release ignores the heartbeat       -> a lease its holder still renews is kept
"""

from types import SimpleNamespace

import pytest
from celery import Task
from celery.exceptions import Ignore

from src.core.cancellation import OperatorCancelled, cooperative_cancel
from src.core.config import settings
from src.services import gpu_job_claim as C
from src.services import gpu_leases
from src.services.gpu_claim import AUTO_QUEUE, queue_for
from src.services.gpu_placement import GpuCard
from src.workers import gpu_job as G
from src.workers import jlens_progress
from src.workers.gpu_supervisor import WORKER_GPU_ENV
from tests.unit.gpu_lease_db import clear, lease_engine, session_factory

TI_UUID = "GPU-f47ba814-49a2-603f-3595-275284140251"
RTX_UUID = "GPU-247aa582-0d1b-e161-8156-983ed1fefc57"
TI = GpuCard(0, TI_UUID, "NVIDIA GeForce RTX 3080 Ti", 12_288, 11_900)
RTX = GpuCard(1, RTX_UUID, "NVIDIA GeForce RTX 3090", 24_576, 21_500)
CARDS = [TI, RTX]
OTHER = "training:someone-else:00000000"


class FakeTask(Task):
    request = None  # shadows Task.request (a property over the request stack)
    name = "tests.fake_gpu_task"

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


@pytest.fixture(scope="module")
def engine():
    eng = lease_engine("mistudio_test_gpu_job_wrapper")
    yield eng
    eng.dispose()


@pytest.fixture
def node(engine, monkeypatch):
    """Per-card mode on the two-card node, the 3090's worker, real leases, parking recorded."""
    clear(engine)
    db = session_factory(engine)
    sessions = {"n": 0}

    def counted():
        sessions["n"] += 1
        return db()

    parked = []
    monkeypatch.setattr(settings, "gpu_worker_mode", "per_card")
    monkeypatch.setenv(WORKER_GPU_ENV, RTX_UUID)
    monkeypatch.setattr(C, "_sync_session", counted)
    monkeypatch.setattr(C, "list_cards", lambda: list(CARDS))
    monkeypatch.setattr(C, "host_available_mb", lambda: 1_000_000.0)
    monkeypatch.setattr(G, "park_job", lambda *a, **k: parked.append((a, k)))
    return SimpleNamespace(db=db, sessions=sessions, parked=parked)


def leases(db):
    with db() as s:
        return gpu_leases.live_leases(s)


def take(db, uuid, holder=OTHER):
    with db() as s:
        assert gpu_leases.acquire(s, [uuid], holder, task_id="other-task")


def wrapped(body, **gpu_job_kwargs):
    @G.gpu_job("test", **gpu_job_kwargs)
    def run(self, job_id, gpu_request="auto"):
        return body(self, job_id, gpu_request)

    return run


def places(required_mb=4_000, allow_shard=False):
    def body(self, job_id, gpu_request):
        return C.claim_cards(gpu_request, required_mb=required_mb, allow_shard=allow_shard)

    return body


class TestSingleMode:
    def test_the_wrapper_is_a_pass_through(self, node, monkeypatch):
        monkeypatch.setattr(settings, "gpu_worker_mode", "single")
        run = wrapped(lambda self, job_id, req: ("ran", job_id))
        assert run(FakeTask(request()), "job-1") == ("ran", "job-1")
        assert node.sessions["n"] == 0


class TestTheJobRunsUnderItsLease:
    def test_the_lease_is_held_while_it_runs_and_released_after(self, node):
        seen = {}

        def body(self, job_id, gpu_request):
            cards = C.claim_cards(gpu_request, required_mb=4_000)
            seen["leases"] = leases(node.db)
            return cards

        assert wrapped(body)(FakeTask(request()), "job-1") == (RTX,)
        (holder,) = seen["leases"].values()
        assert seen["leases"].keys() == {RTX_UUID} and holder.startswith("test:tid-1:")
        assert leases(node.db) == {}

    def test_an_oom_releases_the_card(self, node):
        def body(self, job_id, gpu_request):
            C.claim_cards(gpu_request, required_mb=4_000)
            raise RuntimeError("CUDA out of memory")

        with pytest.raises(RuntimeError, match="out of memory"):
            wrapped(body)(FakeTask(request()), "job-1")
        assert leases(node.db) == {}

    def test_a_cooperative_cancel_releases_the_card(self, node):
        @G.gpu_job("test")
        @cooperative_cancel("training")
        def run(self, job_id, gpu_request="auto"):
            C.claim_cards(gpu_request, required_mb=4_000)
            raise OperatorCancelled("training", job_id)

        assert run(FakeTask(request()), "job-1")["status"] == "cancelled"
        assert leases(node.db) == {}

    def test_a_job_whose_row_was_deleted_releases_the_card(self, node):
        def body(self, job_id, gpu_request):
            C.claim_cards(gpu_request, required_mb=4_000)
            return {"status": "cancelled", "reason": "deleted"}

        assert wrapped(body)(FakeTask(request()), "job-1")["reason"] == "deleted"
        assert leases(node.db) == {}


class TestHandOffs:
    def test_a_job_naming_another_card_is_sent_there_before_it_does_anything(self, node):
        ran = []
        task = FakeTask(request(kwargs={"gpu_request": TI_UUID}))
        with pytest.raises(Ignore):
            wrapped(lambda *a: ran.append(a))(task, "job-1", gpu_request=TI_UUID)

        assert ran == []
        assert task.published == [(["job-1"], {"gpu_request": TI_UUID},
                                   {"queue": queue_for(TI_UUID), "task_id": "tid-1", "headers": {"gpu_hops": 1}})]
        assert node.parked == [] and leases(node.db) == {}

    def test_a_job_that_must_wait_is_parked_not_held(self, node, monkeypatch):
        """Precheck: the 3080 Ti is idle. Placement: the job needs 15 GB, which only the busy 3090 has."""
        monkeypatch.setenv(WORKER_GPU_ENV, TI_UUID)
        take(node.db, RTX_UUID)
        task = FakeTask(request())
        with pytest.raises(Ignore):
            wrapped(places(required_mb=15_000))(task, "job-1")

        assert task.published == []
        ((name, args, kwargs, options), park_kwargs), = node.parked
        assert (name, args, kwargs) == (FakeTask.name, ["job-1"], {"gpu_request": "auto"})
        assert options == {"queue": AUTO_QUEUE, "task_id": "tid-1", "headers": {"gpu_hops": 0}}
        assert park_kwargs["due_at"] > 0
        assert leases(node.db) == {RTX_UUID: OTHER}

    def test_a_job_passed_back_and_forth_is_parked(self, node):
        task = FakeTask(request(gpu_hops=G.MAX_IMMEDIATE_HOPS, kwargs={"gpu_request": TI_UUID}))
        with pytest.raises(Ignore):
            wrapped(lambda *a: None)(task, "job-1", gpu_request=TI_UUID)
        assert task.published == []
        assert len(node.parked) == 1

    def test_the_general_worker_forwards_a_gpu_job(self, node, monkeypatch):
        monkeypatch.delenv(WORKER_GPU_ENV)
        task = FakeTask(request())
        with pytest.raises(Ignore):
            wrapped(lambda *a: None)(task, "job-1")
        assert [options["queue"] for _, _, options in task.published] == [AUTO_QUEUE]

    def test_time_limits_and_redacted_arguments_survive_the_hand_off(self, node):
        task = FakeTask(request(kwargs={"gpu_request": TI_UUID}, timelimit=(172_800, 144_000),
                                kwargsrepr="{'token': '***'}"))
        with pytest.raises(Ignore):
            wrapped(lambda *a: None)(task, "job-1", gpu_request=TI_UUID)
        (_, _, options), = task.published
        assert (options["time_limit"], options["soft_time_limit"], options["kwargsrepr"]) == (
            172_800, 144_000, "{'token': '***'}")

    def test_a_row_based_request_is_read_for_the_precheck(self, node):
        task = FakeTask(request(args=["job-1"], kwargs={}))
        with pytest.raises(Ignore):
            wrapped(lambda *a: None, request_from=lambda arguments: TI_UUID)(task, "job-1")
        assert task.published[0][2]["queue"] == queue_for(TI_UUID)

    def test_a_j_lens_task_does_not_record_a_hand_off_as_its_failure(self, node, monkeypatch):
        failed = []
        monkeypatch.setattr(jlens_progress, "fail_row", lambda task_id, exc: failed.append(exc))
        monkeypatch.setenv(WORKER_GPU_ENV, TI_UUID)
        take(node.db, RTX_UUID)

        @G.gpu_job("jlens")
        @jlens_progress.owns_its_failure
        def run(self, job_id, gpu_request="auto"):
            C.claim_cards(gpu_request, required_mb=15_000)

        with pytest.raises(Ignore):
            run(FakeTask(request()), "job-1")
        assert failed == []
        assert len(node.parked) == 1


class TestWaitingInPlace:
    def test_an_in_place_task_is_not_prechecked_and_leases_the_card_it_names(self, node, monkeypatch):
        monkeypatch.setenv(WORKER_GPU_ENV, TI_UUID)
        task = FakeTask(request(kwargs={"gpu_request": RTX_UUID}))
        assert wrapped(places(), handoff=False)(task, "job-1", gpu_request=RTX_UUID) == (RTX,)
        assert task.published == [] and node.parked == []
        assert leases(node.db) == {}


class TestJanitorRelease:
    def test_a_reaped_jobs_stale_lease_is_released(self, node, engine):
        from sqlalchemy import text

        take(node.db, RTX_UUID)
        with engine.begin() as conn:
            conn.execute(text("UPDATE gpu_leases SET heartbeat_at = now() - interval '1 hour'"))
        assert G.release_reaped_leases("other-task", session=node.db) == 1
        assert leases(node.db) == {}

    def test_a_lease_its_holder_still_renews_is_kept(self, node):
        take(node.db, RTX_UUID)
        assert G.release_reaped_leases("other-task", session=node.db) == 0
        assert leases(node.db) == {RTX_UUID: OTHER}

    def test_single_mode_touches_nothing(self, node, monkeypatch):
        monkeypatch.setattr(settings, "gpu_worker_mode", "single")

        def no_session():
            raise AssertionError("single mode opened a session")

        assert G.release_reaped_leases("other-task", session=no_session) == 0


class TestNoIdleCopyOutlivesItsClaim:
    """Phase 3 review round 1, item 4. A copy a job leaves loaded (a J-lens readout with
    `unload_after=False`, the logit lens's cache) sat on a card with no lease once the
    claim ended, freed only by the next placement in the SAME worker. Every other worker
    saw that card as idle and its free memory as taken. The claim now frees idle caches
    as it ends — before its leases go, so no job can be placed on the card first.

    MUTATION CONTROLS (2026-09-14; each alone, restored byte-identically, sha256 checked,
    `git diff` clean) — all red:
      K1 the wrapper no longer calls release_idle_copies_at_claim_end
            -> test_the_copy_is_freed_as_the_job_ends_while_its_lease_is_still_held
               (run with -x: the first red test is the one recorded)
      K2 the release runs AFTER the claim (outside `with claiming`)
            -> the copy is freed as the job ends (the lease was already gone)

    COMPOSED WITH REVIEW ROUND 1's R1-1 (cherry-pick resolution b7c12116): the wrapper's
    `finally` is gone; the idle-copy release is step 2 of `gpu_job.release_job_memory`,
    the claim's one ordered, bounded release, which `ClaimContext.close` runs before the
    leases on every exit — with nothing held too, for a hand-off. Re-run on the composed
    lines, each alone, restored by sha256 — all red:
      K1-composed  release_job_memory no longer calls release_idle_copies_at_claim_end
            -> this class's first three tests
      K2-composed  close() runs no release inside the claim
            -> this class's first three; review r1's release-while-leased tests; the buffer order test
      C-R1-1h      the release is skipped again when nothing is held
            -> test_a_hand_off_frees_it_too
    """

    @pytest.fixture
    def idle_cache(self, node, monkeypatch):
        from src.services import gpu_placement

        freed = []
        monkeypatch.setattr(gpu_placement, "_IDLE_RELEASERS", [lambda: freed.append(leases(node.db))])
        return freed

    def test_the_copy_is_freed_as_the_job_ends_while_its_lease_is_still_held(self, node, idle_cache):
        def body(self, job_id, gpu_request):
            C.claim_cards(gpu_request, required_mb=4_000)
            return "kept a copy"

        assert wrapped(body)(FakeTask(request()), "job-1") == "kept a copy"
        (held_at_release,) = idle_cache
        assert list(held_at_release) == [RTX_UUID], "the card was released before its idle copy was freed"
        assert leases(node.db) == {}

    def test_a_failure_frees_it_too(self, node, idle_cache):
        def body(self, job_id, gpu_request):
            C.claim_cards(gpu_request, required_mb=4_000)
            raise RuntimeError("CUDA out of memory")

        with pytest.raises(RuntimeError):
            wrapped(body)(FakeTask(request()), "job-1")
        assert len(idle_cache) == 1

    def test_a_hand_off_frees_it_too(self, node, idle_cache):
        with pytest.raises(Ignore):
            wrapped(lambda *a: None)(FakeTask(request(kwargs={"gpu_request": TI_UUID})), "job-1", gpu_request=TI_UUID)
        assert len(idle_cache) == 1

    def test_single_mode_keeps_the_copy_as_before(self, node, idle_cache, monkeypatch):
        monkeypatch.setattr(settings, "gpu_worker_mode", "single")
        wrapped(lambda *a: None)(FakeTask(request()), "job-1")
        assert idle_cache == []

    def test_the_j_lens_and_logit_lens_caches_are_what_it_frees(self):
        from src.services import gpu_placement, jlens_model_registry, logit_lens_service

        assert jlens_model_registry.release_idle_gpu_copy in gpu_placement._IDLE_RELEASERS
        assert logit_lens_service._release_idle_logit_lens_models in gpu_placement._IDLE_RELEASERS
