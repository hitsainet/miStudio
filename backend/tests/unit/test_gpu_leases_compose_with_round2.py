"""Multi-GPU Phase 3's GPU leases, composed with review round 2's split planner.

Review round 2 (97fc6e29) maps a steering or J-lens split with transformers' own
map inference before a weight is read, and refuses one that would spill. Phase 3
leases the cards a placement uses. Written in parallel and ported together onto
main (2026-09-14), each property below held for one side by construction:

1. Steering: the planner is handed the LEASED placement, budgeted from the memory
   free as the weights load (round 1's re-read, 756bd9cc), and the load passes the
   plan's budget. Never a card outside the lease: on a three-card node another job
   holds the card with the most free memory.
2. Steering and J-lens: a split the planner refuses is refused before
   ``from_pretrained``, nothing is loaded, and the execution's claim gives back
   every lease it took.

Real Postgres leases (``gpu_lease_db``) and the REAL ``place_job``. The J-lens case
runs transformers' real map inference (round 2's harness); the steering cases
record the planner's inputs, since its fake config cannot be built on meta.

MUTATION CONTROLS (2026-09-14, the port; each applied alone, this module run red,
restored byte-identically and checked by sha256):
  Composition (new with this module):
    Y1  steering's planner is handed every visible card, not the placement's
          -> test_the_plan_gets_the_leased_cards_live_budget_and_the_load_gets_the_plan
  Round 1 and round 2 controls re-run against these tests:
    Y2  (round 1 R1-N1) the split's budget is not re-read before the planner
          -> test_the_plan_gets_the_leased_cards_live_budget_and_the_load_gets_the_plan
    Y3  (round 2 P7) steering's planner call removed
          -> both steering cases
    Y4  (round 2 P2) the J-lens refusal is swallowed (planned = None)
          -> TestAJlensSplitThePlannerRefuses
  Phase 3 controls re-run against these tests:
    M1  place_job resolves over every card instead of claiming -> all three cases
    M3  close() does not release                               -> all three cases
"""

import asyncio
import math
from types import SimpleNamespace

import pytest
import torch

from src.services import gpu_job_claim as C
from tests.unit.gpu_lease_db import clear, lease_engine, session_factory
from tests.unit.test_steering_gpu_placement import (  # noqa: F401 - cuda, leased_node, loader are fixtures
    OTHER_HOLDER,
    RTX_UUID,
    TI_UUID,
    X_UUID,
    _service,
    _steering_claim,
    cuda,
    leased_node,
    loader,
)


def _free_when_the_weights_load(device=None):
    """Less than placement read off the inventory (3080 Ti 11,000 MB, 3090 23,000 MB)."""
    index = device.index if isinstance(device, torch.device) else device
    return {0: 10_900, 1: 22_500, 2: 40_000}[index] * 1024**2, [12_288, 24_576, 48_000][index] * 1024**2


@pytest.fixture
def planner(monkeypatch, leased_node):
    """The split planner, recording what it was handed and the leases live at that moment."""
    from src.ml import split_load

    state = SimpleNamespace(seen=[], refuse=False)

    def plan(config, *, max_memory, **kwargs):
        state.seen.append({"max_memory": dict(max_memory), "leases": leased_node.live()})
        if state.refuse:
            raise split_load.SplitDoesNotFit(
                "Steering model big does not fit on the GPUs it was split across", mapped_mb={}, limit_mb={}
            )
        return split_load.SplitLoadPlan(
            max_memory={1: "18000MiB", 0: "9000MiB"}, device_map={}, mapped_mb={}, reclaimed_mb=0
        )

    monkeypatch.setattr(split_load, "plan_split_load", plan)
    return state


class TestTheSteeringPlannerSeesOnlyTheLeasedCards:
    def test_the_plan_gets_the_leased_cards_live_budget_and_the_load_gets_the_plan(
        self, monkeypatch, loader, leased_node, planner
    ):
        monkeypatch.setattr(torch.cuda, "mem_get_info", _free_when_the_weights_load)
        svc = _service(monkeypatch)
        claim = _steering_claim(leased_node, "tid-plan")

        with C.claiming(claim):
            asyncio.run(svc.load_model("big", gpu="auto"))

        leased = {RTX_UUID: claim.holder, TI_UUID: claim.holder, X_UUID: OTHER_HOLDER}
        assert planner.seen == [{"max_memory": {1: "21476MiB", 0: "9876MiB"}, "leases": leased}], (
            "the planner was not handed the leased cards' budget re-read as the weights load"
        )
        assert [call["max_memory"] for call in loader] == [{1: "18000MiB", 0: "9000MiB"}], (
            "the load did not pass the plan's budget"
        )
        assert leased_node.live() == {X_UUID: OTHER_HOLDER}

    def test_a_split_the_planner_refuses_loads_nothing_and_gives_back_every_lease(
        self, monkeypatch, loader, leased_node, planner
    ):
        planner.refuse = True
        svc = _service(monkeypatch)
        claim = _steering_claim(leased_node, "tid-refused")

        with C.claiming(claim):
            with pytest.raises(RuntimeError, match="does not fit on the GPUs"):
                asyncio.run(svc.load_model("big", gpu="auto"))

        assert [seen["leases"] for seen in planner.seen] == [
            {RTX_UUID: claim.holder, TI_UUID: claim.holder, X_UUID: OTHER_HOLDER}
        ], "the split was planned on cards the claim had not leased"
        assert loader == [], "a refused split went on to from_pretrained"
        assert "big" not in svc._loaded_models
        assert leased_node.live() == {X_UUID: OTHER_HOLDER}, "the refused split kept its cards leased"


@pytest.fixture
def lease_db(monkeypatch):
    from src.core.config import settings
    from src.services import gpu_leases

    engine = lease_engine("mistudio_test_gpu_leases_compose_r2")
    clear(engine)
    db = session_factory(engine)
    monkeypatch.setattr(settings, "gpu_worker_mode", "per_card")

    def live():
        with db() as s:
            return gpu_leases.live_leases(s)

    yield SimpleNamespace(db=db, live=live)
    engine.dispose()


class TestAJlensSplitThePlannerRefusesGivesBackItsLeases:
    def test_refused_before_loading_and_every_lease_released(self, tmp_path, lease_db):
        from tests.unit.test_every_split_load_is_mapped_first import (
            _bf16_config,
            _jlens_hub,
            _short_budgets,
        )
        from tests.unit.test_jlens_split_gpu import _record
        from tests.unit.test_split_load_maps_onto_the_gpus import UUID_RTX, UUID_TI, _cards, _place, _sizes_mib

        config = _bf16_config()
        budget0, budget1 = _short_budgets(_sizes_mib(config))
        cards = _cards(budget0, budget1)
        claim = C.ClaimContext(
            holder=C.make_holder("jlens_readout", "tid-j"), task_id="tid-j",
            worker_uuid=UUID_RTX, handoff=False, session=lease_db.db,
            inventory=lambda: list(cards), available_mb=lambda: 1_000_000.0, sleep=lambda s: None,
        )

        with _jlens_hub(tmp_path, config) as (registry, seen):
            with C.claiming(claim):
                placement = _place(cards, required_mb=math.floor(budget0 + budget1 - 2))
                assert placement.is_shard, "the fixture must be a split"
                held = lease_db.live()
                with pytest.raises(registry.ModelNotAvailable, match="does not fit on the GPUs"):
                    registry.load_for_readout(_record(), placement=placement)
            assert seen.primary == [], "the refusal came after from_pretrained had started"
            assert seen.fallback == [], "the refusal was loaded again through the fallback"

        assert held == {UUID_TI: claim.holder, UUID_RTX: claim.holder}, "the split was not leased before planning"
        assert lease_db.live() == {}, "the refused split kept its cards leased"
