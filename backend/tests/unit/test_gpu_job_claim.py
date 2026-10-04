"""A job places only on the cards it has leased (multi-GPU Phase 3).

``place_job`` asks ``gpu_job_claim.claim_cards`` for its cards. Inside a job the
claim reads the live leases, decides with the job's exact size, takes the leases
(all or none) and returns ONLY those cards; it hands the job off, waits, or
refuses otherwise. These tests run the real claim against REAL POSTGRES leases
(see ``gpu_lease_db``) on a fake inventory: the node's RTX 3080 Ti (index 0) and
RTX 3090 (index 1).

MUTATION CONTROLS (2026-09-14; each alone, applied where it occurred exactly once,
restored byte-identically with sha256 checked and `git diff` clean) — all red:
  M1  place_job resolves over every card instead of claiming    -> place_job places only on the leased card
  M2  RunHere returns its cards without taking the lease        -> 14 red: the card is leased, the race, second placement…
  M3  close() does not release                                  -> release on every exit, renewal, host RAM
  M4  the renewal thread is never started                       -> a running job renews its leases on a timer
  M5  the holder drops its per-execution nonce                  -> another execution of the same task cannot release it
  M6  a second placement is not restricted to the held cards    -> it chooses only among the cards the job holds
  M7  the host RAM refusal is ignored                           -> three host RAM tests
  M8  no host reserve for a training                            -> a training on the other card / its own claim
  M8b a training's OWN reserve dropped                           -> a training's own claim reserves its buffer
  M8c other trainings' reserves dropped                          -> a training on the other card counts against host RAM
  M14 an unclaimed placement is allowed in per-card mode        -> per-card mode refuses an unclaimed placement
  M16 reuse_card does not claim the resident copy's cards       -> a J-lens task reusing a resident copy leases its card
  J3  the janitor release ignores the heartbeat                 -> only a lease its holder stopped renewing is released
  P1  (the previous round's rule, gpu_claim) wait for what can never come -> a job no card could ever hold is refused
"""

import threading
from datetime import datetime, timedelta, timezone

import pytest
import torch
from sqlalchemy import text

from src.core.cancellation import OperatorCancelled
from src.core.config import settings
from src.services import gpu_job_claim as C
from src.services import gpu_leases
from src.services import gpu_placement
from src.services.gpu_claim import AUTO_QUEUE, queue_for
from src.services.gpu_placement import GpuCard, GpuPlacementError
from tests.unit.gpu_lease_db import clear, lease_engine, session_factory

TI_UUID = "GPU-f47ba814-49a2-603f-3595-275284140251"
RTX_UUID = "GPU-247aa582-0d1b-e161-8156-983ed1fefc57"
TI = GpuCard(0, TI_UUID, "NVIDIA GeForce RTX 3080 Ti", 12_288, 11_900)
RTX = GpuCard(1, RTX_UUID, "NVIDIA GeForce RTX 3090", 24_576, 21_500)
CARDS = [TI, RTX]
OTHER = "training:someone-else:00000000"


@pytest.fixture(scope="module")
def engine():
    eng = lease_engine("mistudio_test_gpu_job_claim")
    yield eng
    eng.dispose()


@pytest.fixture
def db(engine):
    clear(engine)
    return session_factory(engine)


@pytest.fixture
def per_card(monkeypatch):
    monkeypatch.setattr(settings, "gpu_worker_mode", "per_card")


def ctx(db, *, worker=RTX, handoff=True, cards=CARDS, available=1_000_000.0, task_id="task-1", **kwargs):
    return C.ClaimContext(
        holder=C.make_holder("test", task_id),
        task_id=task_id,
        worker_uuid=None if worker is None else worker.uuid,
        handoff=handoff,
        session=kwargs.pop("session", db),
        inventory=lambda: list(cards),
        available_mb=lambda: available,
        **kwargs,
    )


def take(db, uuid, holder=OTHER, task_id="other-task"):
    with db() as s:
        assert gpu_leases.acquire(s, [uuid], holder, task_id=task_id)


def leases(db):
    with db() as s:
        return gpu_leases.live_leases(s)


class TestRunHereLeasesExactlyWhatItPlaces:
    def test_the_card_is_leased_and_returned(self, db):
        claim = ctx(db, worker=RTX)
        assert claim.claim("auto", required_mb=4_000) == (RTX,)
        assert leases(db) == {RTX_UUID: claim.holder}

    def test_the_lease_records_the_task(self, db, engine):
        ctx(db, worker=RTX, task_id="celery-xyz").claim("auto", required_mb=4_000)
        with engine.connect() as conn:
            assert conn.execute(text("SELECT task_id FROM gpu_leases")).scalar() == "celery-xyz"

    def test_a_card_another_job_leased_is_never_placed_on(self, db):
        """The 3090 has more free memory, but another job holds it."""
        take(db, RTX_UUID)
        claim = ctx(db, worker=TI)
        assert claim.claim("auto", required_mb=4_000) == (TI,)
        assert leases(db) == {RTX_UUID: OTHER, TI_UUID: claim.holder}

    def test_a_split_leases_every_card(self, db):
        claim = ctx(db, worker=RTX)
        cards = claim.claim("auto", required_mb=30_000, allow_shard=True)
        assert cards == (RTX, TI)
        assert leases(db) == {RTX_UUID: claim.holder, TI_UUID: claim.holder}

    def test_a_split_takes_nothing_while_one_of_its_cards_is_held(self, db):
        take(db, TI_UUID)
        with pytest.raises(C.JobHandoff) as handed:
            ctx(db, worker=RTX).claim("auto", required_mb=30_000, allow_shard=True)
        assert handed.value.queue == AUTO_QUEUE
        assert handed.value.delay_s == C.HANDOFF_WAIT_S
        assert leases(db) == {TI_UUID: OTHER}


class TestHandingOff:
    def test_a_job_naming_another_card_is_sent_there_at_once(self, db):
        with pytest.raises(C.JobHandoff) as handed:
            ctx(db, worker=TI).claim(RTX_UUID, required_mb=4_000)
        assert (handed.value.queue, handed.value.delay_s) == (queue_for(RTX_UUID), 0.0)
        assert leases(db) == {}

    def test_a_named_card_another_job_holds_is_waited_for_on_its_own_queue(self, db):
        take(db, RTX_UUID)
        with pytest.raises(C.JobHandoff) as handed:
            ctx(db, worker=RTX).claim(RTX_UUID, required_mb=4_000)
        assert (handed.value.queue, handed.value.delay_s) == (queue_for(RTX_UUID), C.HANDOFF_WAIT_S)

    def test_every_card_busy_waits_on_the_shared_queue(self, db):
        take(db, RTX_UUID)
        take(db, TI_UUID, holder="extraction:x:11111111")
        with pytest.raises(C.JobHandoff) as handed:
            ctx(db, worker=TI).claim("auto", required_mb=4_000)
        assert (handed.value.queue, handed.value.delay_s) == (AUTO_QUEUE, C.HANDOFF_WAIT_S)

    def test_a_general_worker_forwards_a_gpu_job_to_the_gpu_queue(self, db):
        with pytest.raises(C.JobHandoff) as handed:
            ctx(db, worker=None).claim(TI_UUID, required_mb=4_000)
        assert (handed.value.queue, handed.value.delay_s) == (queue_for(TI_UUID), 0.0)
        assert leases(db) == {}

    def test_a_missing_card_is_refused_with_the_placement_message(self, db):
        with pytest.raises(GpuPlacementError, match="No GPU"):
            ctx(db, worker=RTX).claim("GPU-00000000-0000-0000-0000-000000000000", required_mb=4_000)
        assert leases(db) == {}

    def test_a_job_no_card_could_ever_hold_is_refused_not_parked(self, db):
        take(db, TI_UUID)
        with pytest.raises(GpuPlacementError):
            ctx(db, worker=RTX).claim("auto", required_mb=50_000)


class TestARaceForTheSameCard:
    def test_a_card_leased_between_the_read_and_the_lease_is_never_used(self, db):
        """B reads the leases (empty), then A takes the 3090, then B tries to lease it."""
        rival = ctx(db, worker=RTX, task_id="rival")
        calls = {"n": 0}

        def racing_session():
            calls["n"] += 1
            if calls["n"] == 2:  # B's lease attempt: the rival gets there first
                rival.claim("auto", required_mb=4_000)
            return db()

        loser = ctx(db, worker=RTX, task_id="loser", session=racing_session)
        with pytest.raises(C.JobHandoff) as handed:
            loser.claim("auto", required_mb=4_000)
        # The 3090 is the rival's; the loser is sent to the idle 3080 Ti, never onto the 3090.
        assert handed.value.queue == queue_for(TI_UUID)
        assert leases(db) == {RTX_UUID: rival.holder}


class TestASecondPlacementInOneJob:
    def test_it_chooses_only_among_the_cards_the_job_holds(self, db):
        take(db, RTX_UUID)
        claim = ctx(db, worker=TI)
        assert claim.claim("auto", required_mb=4_000) == (TI,)
        with db() as s:
            gpu_leases.release(s, OTHER)  # the 3090 is now idle and has more free memory

        assert claim.claim("auto", required_mb=2_000) == (TI,)
        assert leases(db) == {TI_UUID: claim.holder}


class TestRelease:
    def test_close_releases_only_this_executions_leases(self, db):
        take(db, RTX_UUID)
        claim = ctx(db, worker=TI)
        claim.claim("auto", required_mb=4_000)
        claim.close()
        assert leases(db) == {RTX_UUID: OTHER}

    def test_another_execution_of_the_same_task_cannot_release_it(self, db):
        first = ctx(db, worker=RTX, task_id="same-task")
        first.claim("auto", required_mb=4_000)
        second = ctx(db, worker=RTX, task_id="same-task")
        assert first.holder != second.holder

        second.close()
        assert leases(db) == {RTX_UUID: first.holder}
        with pytest.raises(C.JobHandoff):
            second.claim(RTX_UUID, required_mb=4_000)

    @pytest.mark.parametrize("raised", [RuntimeError("CUDA out of memory"), OperatorCancelled("training", "t1")])
    def test_claiming_releases_on_every_exit(self, db, raised):
        claim = ctx(db, worker=RTX)
        with pytest.raises(type(raised)):
            with C.claiming(claim):
                claim.claim("auto", required_mb=4_000)
                assert leases(db) == {RTX_UUID: claim.holder}
                raise raised
        assert leases(db) == {}
        assert C.current_claim() is None


class TestWaitingInPlace:
    def test_it_waits_until_the_card_is_released_then_leases_it(self, db):
        take(db, RTX_UUID)
        slept = []

        def sleep(seconds):
            slept.append(seconds)
            with db() as s:
                gpu_leases.release(s, OTHER)

        claim = ctx(db, worker=None, handoff=False, sleep=sleep)
        assert claim.claim("auto", required_mb=15_000) == (RTX,)
        assert slept == [C.IN_PLACE_POLL_S]
        assert leases(db) == {RTX_UUID: claim.holder}

    def test_it_gives_up_with_a_message_after_its_bound(self, db):
        take(db, RTX_UUID)
        clock = {"t": 0.0}

        def sleep(seconds):
            clock["t"] += seconds

        claim = ctx(db, worker=None, handoff=False, wait_timeout_s=25.0,
                    clock=lambda: clock["t"], sleep=sleep)
        with pytest.raises(GpuPlacementError, match="Waited"):
            claim.claim("auto", required_mb=15_000)
        assert leases(db) == {RTX_UUID: OTHER}

    def test_it_never_hands_off_it_runs_on_the_card_asked_for(self, db):
        """A steering worker spawned for the 3080 Ti, asked for the 3090, leases the 3090."""
        claim = ctx(db, worker=TI, handoff=False)
        assert claim.claim(RTX_UUID, required_mb=4_000) == (RTX,)
        assert leases(db) == {RTX_UUID: claim.holder}


class TestHostRam:
    def test_the_refusal_rule(self):
        assert C.host_ram_refusal(None, others_hold_leases=True, available_mb=1.0) is None
        assert C.host_ram_refusal(8_000, others_hold_leases=False, available_mb=1.0) is None
        assert C.host_ram_refusal(8_000, others_hold_leases=True, available_mb=20_000) is None
        # 25% of what is available is kept free: 10,000 * 0.75 = 7,500 < 8,000.
        assert "host RAM" in C.host_ram_refusal(8_000, others_hold_leases=True, available_mb=10_000)

    def test_a_running_training_s_buffer_is_reserved_even_before_it_is_allocated(self):
        # 40,000 * 0.75 = 30,000, less the other training's 24,576 reserve = 5,424.
        assert C.host_ram_refusal(4_000, others_hold_leases=True, available_mb=40_000,
                                  other_reserves_mb=24_576) is None
        assert "host RAM" in C.host_ram_refusal(8_000, others_hold_leases=True, available_mb=40_000,
                                                other_reserves_mb=24_576)

    def test_a_training_reserves_its_own_buffer(self):
        # Its GPU estimate is small; its pinned host buffer is not.
        assert "host RAM" in C.host_ram_refusal(2_000, others_hold_leases=True, available_mb=30_000,
                                                own_reserve_mb=24_576)
        assert C.host_ram_refusal(2_000, others_hold_leases=False, available_mb=30_000,
                                  own_reserve_mb=24_576) is None

    def test_a_second_concurrent_load_waits_for_host_ram(self, db):
        take(db, TI_UUID, holder="extraction:e1:11111111")
        with pytest.raises(C.JobHandoff, match="host RAM") as handed:
            ctx(db, worker=RTX, available=10_000).claim("auto", required_mb=8_000)
        assert handed.value.delay_s == C.HANDOFF_WAIT_S
        assert leases(db) == {TI_UUID: "extraction:e1:11111111"}

    def test_a_training_on_the_other_card_counts_against_host_ram(self, db):
        """40 GB available fits an 8 GB load beside an extraction, not beside a training."""
        take(db, TI_UUID, holder="extraction:e1:11111111")
        beside_extraction = ctx(db, worker=RTX, available=40_000, task_id="a")
        assert beside_extraction.claim("auto", required_mb=8_000) == (RTX,)
        beside_extraction.close()
        with db() as s:
            gpu_leases.release(s, "extraction:e1:11111111")

        take(db, TI_UUID, holder="training:t1:22222222")
        with pytest.raises(C.JobHandoff, match="host RAM"):
            ctx(db, worker=RTX, available=40_000, task_id="b").claim("auto", required_mb=8_000)

    def test_a_training_s_own_claim_reserves_its_buffer(self, db):
        take(db, TI_UUID, holder="extraction:e1:11111111")
        training = C.ClaimContext(
            holder=C.make_holder("training", "t2"), task_id="t2", worker_uuid=RTX_UUID, session=db,
            inventory=lambda: list(CARDS), available_mb=lambda: 30_000.0,
        )
        with pytest.raises(C.JobHandoff, match="host RAM"):
            training.claim("auto", required_mb=2_000)

    def test_a_first_load_is_not_guarded(self, db):
        assert ctx(db, worker=RTX, available=10.0).claim("auto", required_mb=8_000) == (RTX,)


class TestRenewal:
    def test_a_running_job_renews_its_leases_on_a_timer(self, db, engine):
        renewed = threading.Event()
        claim = ctx(db, worker=RTX, renew_interval_s=0.05,
                    on_renew=lambda n: renewed.set() if n == 1 else None)
        claim.claim("auto", required_mb=4_000)
        with engine.connect() as conn:
            before = conn.execute(text("SELECT heartbeat_at FROM gpu_leases")).scalar()

        assert renewed.wait(timeout=10), "the lease was never renewed"
        with engine.connect() as conn:
            after = conn.execute(text("SELECT heartbeat_at FROM gpu_leases")).scalar()
        assert after > before

        renewer = claim._renewer
        claim.close()
        assert renewer is not None and not renewer._thread.is_alive()
        assert leases(db) == {}


class TestTheEntryPlaceJobUses:
    def test_single_mode_without_a_claim_is_phase_2(self, monkeypatch):
        monkeypatch.setattr(settings, "gpu_worker_mode", "single")
        assert C.claim_cards("auto", required_mb=4_000, cards=CARDS) == (RTX,)

    def test_per_card_mode_refuses_an_unclaimed_placement(self, per_card):
        with pytest.raises(GpuPlacementError, match="without a GPU lease claim"):
            C.claim_cards("auto", required_mb=4_000, cards=CARDS)

    def test_place_job_places_only_on_the_leased_card(self, db, per_card, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
        monkeypatch.setattr(gpu_placement, "torch_device", lambda card: torch.device("cuda", card.index))
        take(db, RTX_UUID)
        claim = ctx(db, worker=TI)
        with C.claiming(claim):
            placement = gpu_placement.place_job("auto", required_mb=4_000, cards=CARDS)
            assert placement.card == TI
            assert leases(db) == {RTX_UUID: OTHER, TI_UUID: claim.holder}
        assert leases(db) == {RTX_UUID: OTHER}


class TestAResidentModel:
    def test_its_cards_are_leased(self, db):
        claim = ctx(db, worker=TI)
        claim.claim_exact([TI])
        assert leases(db) == {TI_UUID: claim.holder}

    def test_a_resident_card_another_job_holds_is_waited_for(self, db):
        take(db, TI_UUID)
        with pytest.raises(C.JobHandoff):
            ctx(db, worker=TI).claim_exact([TI], requested="auto")

    def test_single_mode_leases_nothing(self, monkeypatch):
        monkeypatch.setattr(settings, "gpu_worker_mode", "single")
        C.claim_exact_cards([TI])

    def test_a_j_lens_task_reusing_a_resident_copy_leases_its_card(self, db, per_card, monkeypatch):
        """The call site: `reuse_card` runs without a placement, so it must claim the copy's card itself."""
        from src.workers import jlens_progress

        monkeypatch.setattr(jlens_progress, "record_gpu", lambda *args, **kwargs: True)
        monkeypatch.setattr(gpu_placement, "make_current", lambda device: None)
        claim = ctx(db, worker=TI)
        with C.claiming(claim):
            jlens_progress.reuse_card("task-r", TI, torch.device("cuda", 0), "auto")
            assert leases(db) == {TI_UUID: claim.holder}
        assert leases(db) == {}


class TestJanitorRelease:
    T0 = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)

    def test_only_a_lease_its_holder_stopped_renewing_is_released(self, db):
        with db() as s:
            assert gpu_leases.acquire(s, [TI_UUID], "dead:1:aaaa", task_id="dead", now=self.T0)
            assert gpu_leases.acquire(s, [RTX_UUID], "live:2:bbbb", task_id="live",
                                      now=self.T0 + timedelta(seconds=500))
        later = self.T0 + timedelta(seconds=600)
        with db() as s:
            assert gpu_leases.release_stale_for_task(s, "live", stale_after_s=180, now=later) == 0
            assert gpu_leases.release_stale_for_task(s, "dead", stale_after_s=180, now=later) == 1
            assert gpu_leases.release_stale_for_task(s, None, stale_after_s=180, now=later) == 0
        with db() as s:
            assert gpu_leases.live_leases(s, now=later) == {RTX_UUID: "live:2:bbbb"}
