"""Which card a free worker runs a job on (multi-GPU Phase 3, decision 5).

Each card has a worker; Auto and "all" jobs wait in one shared queue, and the
worker that takes one decides. These tests pin every branch of that decision on a
fake inventory: the node's RTX 3080 Ti (index 0) beside the RTX 3090 (index 1),
and a third card where it matters.
"""

import pytest

from src.services.gpu_claim import (
    AUTO_QUEUE,
    Refuse,
    RunHere,
    SendTo,
    WaitForCard,
    decide_claim,
    queue_for,
)
from src.services.gpu_placement import SHARD_RESERVE_MB, GpuCard

TI_UUID = "GPU-f47ba814-49a2-603f-3595-275284140251"
RTX_UUID = "GPU-247aa582-0d1b-e161-8156-983ed1fefc57"
TI = GpuCard(index=0, uuid=TI_UUID, name="NVIDIA GeForce RTX 3080 Ti", total_mb=12_288, free_mb=11_900)
RTX = GpuCard(index=1, uuid=RTX_UUID, name="NVIDIA GeForce RTX 3090", total_mb=24_576, free_mb=21_500)
CARDS = [TI, RTX]
ME = "training:t1"


def claim(worker, requested="auto", required_mb=4_000, allow_shard=False, leases=None, cards=CARDS):
    return decide_claim(
        worker_uuid=worker.uuid, requested=requested, required_mb=required_mb,
        allow_shard=allow_shard, cards=cards, leases=leases or {}, holder=ME,
    )


class TestQueues:
    def test_one_spelling_of_a_card_queue(self):
        assert queue_for(RTX_UUID) == queue_for(RTX_UUID.lower()) == queue_for(RTX_UUID.removeprefix("GPU-"))

    def test_the_shared_queue(self):
        assert AUTO_QUEUE == "gpu.auto"


class TestAuto:
    def test_the_most_free_idle_card_runs_it(self):
        assert claim(RTX) == RunHere((RTX,))

    def test_a_less_free_worker_sends_it_to_the_more_free_idle_card(self):
        """Decision 1: the first free worker does not keep a job another card fits better."""
        decision = claim(TI)
        assert decision == SendTo(queue_for(RTX_UUID), decision.reason)

    def test_a_busy_card_is_not_the_best_choice(self):
        assert claim(TI, leases={RTX_UUID: "extraction:e1"}) == RunHere((TI,))

    def test_a_card_this_job_already_holds_counts_as_free(self):
        assert claim(RTX, leases={RTX_UUID: ME}) == RunHere((RTX,))

    def test_every_card_busy_means_wait(self):
        leases = {TI_UUID: "extraction:e1", RTX_UUID: "training:t2"}
        assert isinstance(claim(RTX, leases=leases), WaitForCard)

    def test_too_big_for_the_idle_card_waits_for_the_busy_one(self):
        """16 GB: only the 3090 can hold it, and it is busy."""
        decision = claim(TI, required_mb=16_000, leases={RTX_UUID: "training:t2"})
        assert isinstance(decision, WaitForCard)

    def test_too_big_for_any_card_with_everything_idle_is_refused(self):
        assert isinstance(claim(RTX, required_mb=40_000), Refuse)

    def test_too_big_for_any_card_is_refused_even_while_a_card_is_busy(self):
        """Waiting is only right if a released card could make room. 30 GB fits no
        card of this node at any time, so a busy 3090 must not park it for ever."""
        assert isinstance(claim(TI, required_mb=30_000, leases={RTX_UUID: "training:t2"}), Refuse)

    def test_a_split_job_waits_when_the_cards_together_could_hold_it(self):
        decision = claim(TI, required_mb=30_000, allow_shard=True, leases={RTX_UUID: "training:t2"})
        assert isinstance(decision, WaitForCard)


class TestAutoSplit:
    BIG = 26_000   # more than either card, less than both

    def test_a_split_over_idle_cards_runs_on_a_worker_in_the_split(self):
        decision = claim(TI, required_mb=self.BIG, allow_shard=True)
        assert decision == RunHere((RTX, TI))

    def test_a_job_that_cannot_split_waits_or_is_refused_instead(self):
        assert isinstance(claim(RTX, required_mb=self.BIG, allow_shard=False), Refuse)

    def test_a_split_needing_a_busy_card_waits(self):
        decision = claim(RTX, required_mb=self.BIG, allow_shard=True, leases={TI_UUID: "extraction:e1"})
        assert isinstance(decision, WaitForCard)

    def test_a_worker_outside_the_split_sends_it_to_the_split(self):
        spare = GpuCard(index=2, uuid="GPU-11111111-2222-3333-4444-555555555555", name="RTX 3060", total_mb=12_288, free_mb=2_000)
        cards = CARDS + [spare]
        decision = claim(spare, required_mb=self.BIG, allow_shard=True, cards=cards)
        assert decision == SendTo(queue_for(RTX_UUID), decision.reason)


class TestAll:
    def test_all_runs_on_every_card_when_all_are_idle(self):
        assert claim(TI, requested="all", allow_shard=True) == RunHere((RTX, TI))

    def test_all_waits_while_any_card_is_busy(self):
        assert isinstance(claim(TI, requested="all", allow_shard=True, leases={RTX_UUID: "training:t2"}), WaitForCard)

    def test_all_is_refused_by_a_job_that_cannot_split(self):
        assert isinstance(claim(TI, requested="all", allow_shard=False), Refuse)

    def test_all_too_big_even_together_is_refused(self):
        assert isinstance(claim(TI, requested="all", allow_shard=True, required_mb=40_000), Refuse)


class TestNamed:
    def test_a_job_naming_this_card_runs_here(self):
        assert claim(TI, requested=TI_UUID) == RunHere((TI,))

    def test_a_job_naming_another_card_is_sent_there(self):
        decision = claim(TI, requested=RTX_UUID)
        assert decision == SendTo(queue_for(RTX_UUID), decision.reason)

    def test_a_named_busy_card_waits(self):
        assert isinstance(claim(TI, requested=TI_UUID, leases={TI_UUID: "extraction:e1"}), WaitForCard)

    def test_a_named_card_without_room_is_refused_not_swapped(self):
        assert isinstance(claim(TI, requested=TI_UUID, required_mb=16_000), Refuse)

    def test_a_card_not_on_the_node_is_refused(self):
        assert isinstance(claim(TI, requested="GPU-00000000-0000-0000-0000-000000000000"), Refuse)


def test_a_worker_whose_card_is_gone_refuses():
    assert isinstance(claim(RTX, cards=[TI]), Refuse)


def test_the_reserve_is_the_one_the_resolver_uses():
    """The split rule is resolve_cards's, not a copy that could drift from it."""
    exactly = (RTX.free_mb - SHARD_RESERVE_MB) + (TI.free_mb - SHARD_RESERVE_MB)
    assert claim(TI, required_mb=exactly, allow_shard=True) == RunHere((RTX, TI))
    assert isinstance(claim(TI, required_mb=exactly + 1, allow_shard=True), Refuse)
