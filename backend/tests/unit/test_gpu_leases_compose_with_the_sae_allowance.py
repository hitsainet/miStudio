"""Multi-GPU Phase 3's GPU leases, composed with review round 2's SAE allowance and J-lens headroom.

Review round 2 (steering reviewer) changed three things that meet Phase 3's claim:

1. Steering moves a request's SAEs off the cards and re-reads the split's budget,
   then hands the planner each layer's SAE size. Phase 3 leases the cards a
   placement uses. The move, the cache release and the plan must stay on the
   LEASED cards: on a three-card node another job holds the card with the most
   free memory.
2. A split the planner refuses because its SAEs do not fit is refused before
   ``from_pretrained``, and the claim still gives back every lease it took.
3. A J-lens task is placed for its weights plus the activation headroom. The
   claim is taken by the same ``place_job`` call, so it must lease for that same
   size: the cards a split needs, not the one card its weights alone would take.

Real Postgres leases (``gpu_lease_db``) and the REAL ``place_job``. The steering
cases record the planner's inputs; the planner itself is tested with transformers'
real map inference in test_split_load_keeps_room_beside_each_layer.py.

MUTATION CONTROLS (review round 2, on the branch re-based onto c78f0f21; scratchpad
p2-r2-steer/mutate.py with mutations_rebased.json, each alone, restored and checked by
sha256, `git diff HEAD` empty after). All 16 killed:
  Composition, this module:
    W5  steering's planner call drops the SAEs          -> both TestTheSaeAllowanceStaysOnTheLeasedCards cases
    W6  the SAEs stay on the cards                       -> test_the_saes_leave_and_the_allowance_is_planned_...
    W7x the SAEs' cache released on EVERY visible card   -> both steering cases (the other job's A6000)
    C1  place_on_card sized at the weights alone         -> TestAJlensClaimLeasesForItsHeadroom (+ 2 headroom cases)
  Phase 3's controls re-run here: M1 place_job resolves instead of claiming -> all three cases;
    M3 close() never releases -> all three; SR1 place_model's reuse branch leases nothing
    -> test_steering_gpu_placement section k (both kept-split cases).
  Round 2 (placement + loader) P7, steering's planner call removed -> test_every_split_load_is_mapped_first
    (3) and test_gpu_leases_compose_with_round2's steering refusal case.
  This reviewer's earlier controls re-run on the re-based code: A1, A5 (planner), R2P-L2 (the
    conflicted planner-breakage line), A7 (circuits), A14 (core loader), A16 (recorder), B4, D1.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from src.services import gpu_job_claim as C
from src.services import gpu_placement
from src.services import steering_service as steering_module
from src.services.steering_service import LoadedSAE
from tests.unit.gpu_lease_db import clear, lease_engine, session_factory
from tests.unit.test_steering_gpu_placement import (  # noqa: F401 - cuda, leased_node, loader are fixtures
    CUDA1,
    OTHER_HOLDER,
    RTX,
    RTX_UUID,
    TI,
    TI_UUID,
    X_UUID,
    _service,
    _steering_claim,
    cuda,
    leased_node,
    loader,
)

#: fp16 SAEs at d_model 5,120 (a 13B-class width), worked by hand:
#:   A: 65,536 features -> (2 x 5,120 x 65,536 + 5,120 + 65,536) = 671,159,296 values x 2 B
#:   B: 16,384 features -> (2 x 5,120 x 16,384 + 5,120 + 16,384) = 167,793,664 values x 2 B
SAE_A_MB = 1_342_318_592 / 1024**2
SAE_B_MB = 335_587_328 / 1024**2
LEASED_DEVICES = {"cuda:0", "cuda:1"}


def _sae(name, layer, d_sae, moves):
    model = MagicMock(name=name)
    model.to.side_effect = lambda device: moves.append((name, str(device)))
    return LoadedSAE(model=model, config=None, layer=layer, d_in=5_120, d_sae=d_sae, device=str(CUDA1))


@pytest.fixture
def planner(monkeypatch, leased_node):
    """The split planner, recording its inputs and the leases live when it was called."""
    from src.ml import split_load

    state = SimpleNamespace(seen=[], refuse_with_saes=False)

    def plan(config, *, max_memory, extra_mb_by_layer=None, **kwargs):
        state.seen.append({
            "max_memory": dict(max_memory),
            "extra": dict(extra_mb_by_layer or {}),
            "leases": leased_node.live(),
        })
        if state.refuse_with_saes and extra_mb_by_layer:
            raise split_load.SplitDoesNotFit(
                "Steering model big does not fit on the GPUs it was split across beside its SAEs",
                mapped_mb={}, limit_mb={},
            )
        return split_load.SplitLoadPlan(
            max_memory={1: "18000MiB", 0: "8000MiB"}, device_map={}, mapped_mb={}, reclaimed_mb=0
        )

    monkeypatch.setattr(split_load, "plan_split_load", plan)
    return state


@pytest.fixture
def emptied(monkeypatch, leased_node):
    """Every cache release steering makes: the devices, and the leases live at that moment."""
    seen = []
    monkeypatch.setattr(
        steering_module, "empty_cache_on",
        lambda devices: seen.append(({str(d) for d in devices}, leased_node.live())),
    )
    return seen


class TestTheSaeAllowanceStaysOnTheLeasedCards:
    def test_the_saes_leave_and_the_allowance_is_planned_only_on_the_leased_cards(
        self, monkeypatch, loader, leased_node, planner, emptied
    ):
        svc = _service(monkeypatch)
        moves = []
        saes = {13: [_sae("A", 13, 65_536, moves)], 30: [_sae("B", 30, 16_384, moves)]}
        claim = _steering_claim(leased_node, "tid-sae")

        with C.claiming(claim):
            asyncio.run(svc.load_model("big", gpu="auto", saes_by_layer=saes))

        both = {RTX_UUID: claim.holder, TI_UUID: claim.holder, X_UUID: OTHER_HOLDER}
        assert moves == [("A", "cpu"), ("B", "cpu")], "the SAEs were not taken off the cards"
        assert emptied, "the SAEs' cache was never released before the budget was read"
        assert all(devices <= LEASED_DEVICES for devices, _ in emptied), (
            f"a cache was released on a card the job had not leased: {emptied}"
        )
        assert emptied[0][1] == both, "the SAEs' cache was released outside the claim"
        assert len(planner.seen) == 1
        assert set(planner.seen[0]["max_memory"]) == {0, 1}, "the planner saw a card outside the lease"
        assert planner.seen[0]["leases"] == both
        assert planner.seen[0]["extra"] == {13: pytest.approx(SAE_A_MB, abs=1e-9),
                                            30: pytest.approx(SAE_B_MB, abs=1e-9)}
        assert [call["max_memory"] for call in loader] == [{1: "18000MiB", 0: "8000MiB"}]
        assert leased_node.live() == {X_UUID: OTHER_HOLDER}

    def test_a_split_refused_for_its_saes_loads_nothing_and_gives_back_every_lease(
        self, monkeypatch, loader, leased_node, planner, emptied
    ):
        planner.refuse_with_saes = True
        svc = _service(monkeypatch)
        saes = {13: [_sae("A", 13, 65_536, [])]}
        claim = _steering_claim(leased_node, "tid-sae-refused")

        with C.claiming(claim):
            with pytest.raises(RuntimeError, match="beside its SAEs"):
                asyncio.run(svc.load_model("big", gpu="auto", saes_by_layer=saes))

        both = {RTX_UUID: claim.holder, TI_UUID: claim.holder, X_UUID: OTHER_HOLDER}
        assert [seen["leases"] for seen in planner.seen] == [both]
        assert loader == [], "a split refused for its SAEs went on to from_pretrained"
        assert "big" not in svc._loaded_models
        assert all(devices <= LEASED_DEVICES for devices, _ in emptied), emptied
        assert leased_node.live() == {X_UUID: OTHER_HOLDER}, "the refused split kept its cards leased"


@pytest.fixture
def lease_db(monkeypatch):
    from src.core.config import settings
    from src.services import gpu_leases

    engine = lease_engine("mistudio_test_gpu_leases_compose_sae_allowance")
    clear(engine)
    db = session_factory(engine)
    monkeypatch.setattr(settings, "gpu_worker_mode", "per_card")

    def live():
        with db() as s:
            return gpu_leases.live_leases(s)

    yield SimpleNamespace(db=db, live=live)
    engine.dispose()


def _jlens_claim(db, task_id):
    return C.ClaimContext(
        holder=C.make_holder("jlens_readout", task_id), task_id=task_id,
        worker_uuid=RTX_UUID, handoff=False, session=db,
        inventory=lambda: [TI, RTX], available_mb=lambda: 1_000_000.0, sleep=lambda s: None,
    )


class TestAJlensClaimLeasesForItsHeadroom:
    def test_the_claim_leases_the_cards_its_weights_and_headroom_need(self, monkeypatch, lease_db):
        from src.workers import jlens_progress

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
        monkeypatch.setattr(gpu_placement, "torch_device", lambda card: torch.device("cuda", card.index))
        monkeypatch.setattr(jlens_progress, "record_gpu", lambda *a, **k: True)

        # 22,500 MB of weights: the 3090 (23,000 MB free) holds them alone, not with
        # 2,048 MB of headroom; the two cards' split budget (21,976 + 9,976) holds both.
        weights_only = _jlens_claim(lease_db.db, "tid-weights")
        with C.claiming(weights_only):
            gpu_placement.place_job("auto", required_mb=22_500.0, allow_shard=True)
            held_for_weights = lease_db.live()
        claim = _jlens_claim(lease_db.db, "tid-headroom")
        with C.claiming(claim):
            placement = jlens_progress.place_on_card("tid-headroom", "auto", required_mb=22_500.0, allow_shard=True)
            held = lease_db.live()

        assert held_for_weights == {RTX_UUID: weights_only.holder}, (
            "precondition: a claim sized at the weights alone leases the 3090 by itself"
        )
        assert placement.is_shard
        assert held == {RTX_UUID: claim.holder, TI_UUID: claim.holder}, (
            "the claim leased for a different size than the placement it was taken for"
        )
        assert lease_db.live() == {}
