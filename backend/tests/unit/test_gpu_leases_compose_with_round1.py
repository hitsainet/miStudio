"""Multi-GPU Phase 3's GPU leases, composed with review round 1's fixes.

Phase 3 (a GPU job places only on the cards it has leased) and review round 1 of
Phase 2 were written in parallel and ported together onto main (2026-09-14).
Each side's tests exercise only its own code, so every property below held for
one side by construction and was pinned by neither:

1. ``release_circuit_job`` (round 1, a66f0d6c) empties caches on EXACTLY the cards
   the job leased, and does it while the job still holds them: another job's card
   is never touched, and the leases go only after the cards have been released,
   so no job is placed on a card whose blocks this one still reserves.
2. A split that the loader's pre-load map refuses (round 1, bd38ae13) gives back
   every lease its placement took, and nothing is loaded.
3. "all" asked of a job that cannot split is refused at SUBMIT first (round 1,
   be889163 / 3da80d83), before a row, a dispatch or a lease; a message that
   reaches a worker anyway (queued before the deploy, or by a site that bypassed
   the endpoint) is refused by the claim (Phase 3) with the same words, before it
   leases anything.

Real Postgres leases (``gpu_lease_db``), a fake two-card inventory, the REAL
``place_job``.

MUTATION CONTROLS (2026-09-14, the port; each applied alone, this module run red,
restored byte-identically and checked by sha256):
  Composition (new with this module):
    X1  release_circuit_job cleans every card this process holds (device=None)
          -> both TestACircuitJob cases
    X4  the loader swallows the split refusal and loads unmapped
          -> TestASplitTheLoaderRefuses
    X5  the trainings endpoint drops can_split (round 1's A2, re-run here)
          -> test_per_card_mode_refuses_all_at_submit_before_a_row_a_dispatch_or_a_lease
    X6  decide_claim stops refusing "all" to a job that cannot split
          -> both test_the_claim_refuses_all_... cases
  Phase 3 controls re-run against these tests:
    M1  place_job resolves over every card instead of claiming
          -> both TestACircuitJob cases, TestASplitTheLoaderRefuses
    M3  close() does not release
          -> both TestACircuitJob cases, TestASplitTheLoaderRefuses
  Round 1 control re-run against these tests:
    R1-N8 the calibration task's finally drops the release
          -> both TestACircuitJob cases
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from fastapi import HTTPException

from src.api.v1.gpu_request import SPLIT_REFUSED
from src.core.config import settings
from src.services import gpu_job_claim as C
from src.services import gpu_leases, gpu_placement
from src.services.gpu_placement import GpuCard, GpuPlacementError
from src.workers import gpu_job as G
from src.workers.gpu_supervisor import WORKER_GPU_ENV
from tests.unit.gpu_lease_db import clear, lease_engine, session_factory

TI_UUID = "GPU-f47ba814-49a2-603f-3595-275284140251"
RTX_UUID = "GPU-247aa582-0d1b-e161-8156-983ed1fefc57"
TI = GpuCard(0, TI_UUID, "NVIDIA GeForce RTX 3080 Ti", 12_288, 11_000)
RTX = GpuCard(1, RTX_UUID, "NVIDIA GeForce RTX 3090", 24_576, 23_000)
CARDS = [TI, RTX]
CUDA0, CUDA1 = torch.device("cuda", 0), torch.device("cuda", 1)
OTHER = "training:someone-else:00000000"


@pytest.fixture(scope="module")
def engine():
    eng = lease_engine("mistudio_test_gpu_leases_compose")
    yield eng
    eng.dispose()


@pytest.fixture
def node(engine, monkeypatch):
    """Per-card mode, the 3090's worker, real leases, and the REAL place_job over a fake inventory."""
    clear(engine)
    db = session_factory(engine)
    monkeypatch.setattr(settings, "gpu_worker_mode", "per_card")
    monkeypatch.setenv(WORKER_GPU_ENV, RTX_UUID)
    monkeypatch.setattr(C, "_sync_session", db)
    monkeypatch.setattr(C, "list_cards", lambda: list(CARDS))
    monkeypatch.setattr(C, "host_available_mb", lambda: 1_000_000.0)
    monkeypatch.setattr(G, "park_job", lambda *a, **k: pytest.fail(f"the job was parked: {a}"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(gpu_placement, "torch_device", lambda card: torch.device("cuda", card.index))
    return SimpleNamespace(db=db)


def leases(db):
    with db() as s:
        return gpu_leases.live_leases(s)


def take(db, uuid, holder=OTHER):
    with db() as s:
        assert gpu_leases.acquire(s, [uuid], holder, task_id="other-task")


# ── 1. a circuit job releases its cards: only its own, and before its leases ──


class TestACircuitJobReleasesOnlyItsLeasedCardsWhileItHoldsThem:
    def _calibrate(self, node, monkeypatch, rows):
        from src.services.circuit_calibration_service import CircuitCalibrationService
        from src.workers import circuit_calibration_tasks as tasks
        from tests.unit.test_circuits_run_split import _Db, _fake_task, _raw

        cleanups = []

        def cleanup(models_to_cleanup=None, context="unknown", device=None):
            cleanups.append({
                "device": None if device is None else list(device),
                "leases": leases(node.db),
            })

        monkeypatch.setattr("src.services.extraction_service.cleanup_gpu_memory", cleanup)
        with patch.object(CircuitCalibrationService, "run", return_value={"band": None}) as run, \
             patch.object(tasks, "emit_circuit_run_completed"):
            _raw(tasks.run_circuit_calibration)(_fake_task(_Db(rows)), "crc_1", {}, gpu_request="auto")
        assert run.call_count == 1
        return cleanups

    def test_one_card_is_released_while_leased_and_the_other_jobs_card_is_untouched(self, node, monkeypatch):
        from tests.unit.test_circuits_run_split import _rows

        take(node.db, TI_UUID)

        cleanups = self._calibrate(node, monkeypatch, _rows())

        assert len(cleanups) == 1, f"the job released its cards {len(cleanups)} times"
        (cleanup,) = cleanups
        assert cleanup["device"] == [CUDA1], "the release reached a card this job did not lease"
        holder = cleanup["leases"].get(RTX_UUID)
        assert holder is not None and holder.startswith("circuit_calibration:"), (
            "the card was released after its lease was (or was never leased): another job could "
            "have been placed on it while this job's blocks were still reserved"
        )
        assert cleanup["leases"][TI_UUID] == OTHER
        assert leases(node.db) == {TI_UUID: OTHER}

    def test_a_split_releases_both_its_cards_while_it_still_holds_them(self, node, monkeypatch):
        from tests.unit.test_circuits_run_split import _rows

        # ~27.9 GB with its SAEs: no single card; the split's budgets (21,976 + 9,976 MB) hold it.
        cleanups = self._calibrate(node, monkeypatch, _rows(params=13_000_000_000))

        assert len(cleanups) == 1
        (cleanup,) = cleanups
        assert cleanup["device"] == [CUDA1, CUDA0], "the release did not cover exactly the split's cards"
        holders = set(cleanup["leases"].values())
        assert set(cleanup["leases"]) == {RTX_UUID, TI_UUID} and len(holders) == 1, cleanup["leases"]
        assert next(iter(holders)).startswith("circuit_calibration:")
        assert leases(node.db) == {}


# ── 2. a split the loader refuses gives back its leases ──────────────────────


class TestASplitTheLoaderRefusesReleasesItsLeases:
    def test_nothing_loads_and_every_card_is_released(self, node, monkeypatch):
        from src.ml import model_loader, split_load
        from src.services import resource_config
        from tests.unit.test_gpu_job_wrapper import FakeTask, request

        seen, loads = {}, []

        class _Config:
            model_type = "llama"

        def refuse(config, *, max_memory, **kwargs):
            seen["max_memory"] = dict(max_memory)
            seen["leases"] = leases(node.db)
            raise split_load.SplitDoesNotFit(
                "org/model does not fit on the GPUs it was split across", mapped_mb={}, limit_mb={}
            )

        monkeypatch.setattr(model_loader.AutoConfig, "from_pretrained", lambda *a, **k: _Config())
        monkeypatch.setattr(model_loader, "extract_architecture_config", lambda config: {})
        monkeypatch.setattr(model_loader, "estimate_parameter_count", lambda config: None)
        monkeypatch.setattr(model_loader, "get_quantization_config", lambda fmt: None)
        monkeypatch.setattr(resource_config, "preflight_gpu_capacity", lambda **kwargs: None)
        monkeypatch.setattr(split_load, "plan_split_load", refuse)
        monkeypatch.setattr(model_loader.AutoModelForCausalLM, "from_pretrained",
                            lambda *a, **k: loads.append(k))

        @G.gpu_job("activation_extraction")
        def run(self, job_id, gpu_request="auto"):
            placement = gpu_placement.place_job(gpu_request, required_mb=30_000, allow_shard=True)
            model_loader.load_model_from_hf(
                "org/model", device_map=placement.device_map, max_memory=placement.max_memory
            )

        with pytest.raises(model_loader.ModelLoadError, match="does not fit"):
            run(FakeTask(request()), "job-1")

        assert set(seen["leases"]) == {RTX_UUID, TI_UUID}, "the split was mapped on cards it had not leased"
        (holder,) = set(seen["leases"].values())
        assert holder.startswith("activation_extraction:tid-1:")
        assert set(seen["max_memory"]) == {0, 1}
        assert loads == [], "a refused split went on to load"
        assert leases(node.db) == {}, "the refused split kept its cards leased"


# ── 3. "all" for a job that cannot split: submit first, then the claim ──────


class TestAllIsRefusedAtSubmitFirstAndByTheClaimBeforeAnyLease:
    @pytest.mark.asyncio
    async def test_per_card_mode_refuses_all_at_submit_before_a_row_a_dispatch_or_a_lease(self, node, monkeypatch):
        from src.api.v1.endpoints import trainings
        from tests.unit.test_all_is_refused_where_a_job_cannot_split import _training

        created, dispatched = [], []

        async def create_training(db, training, gpu_request=None):
            created.append(gpu_request)
            raise AssertionError("a row was created for a request that is refused")

        monkeypatch.setattr(trainings.TrainingService, "create_training", create_training)
        monkeypatch.setattr(trainings, "dispatch_gpu_task", lambda *a, **k: dispatched.append((a, k)))
        monkeypatch.setattr(C, "list_cards", lambda: pytest.fail("the inventory was read to refuse all"))

        with pytest.raises(HTTPException) as refused:
            await trainings.create_training(training=_training(extraction_ids=["ext_m_1_x"]), db=None)

        assert (refused.value.status_code, refused.value.detail) == (400, SPLIT_REFUSED)
        assert created == [] and dispatched == []
        assert leases(node.db) == {}

    @pytest.mark.parametrize("handoff", [True, False], ids=["whole-gpu", "in-place"])
    def test_the_claim_refuses_all_to_a_job_that_cannot_split_before_it_leases(self, node, handoff):
        claim = C.ClaimContext(
            holder=C.make_holder("training", "tid-all"), task_id="tid-all",
            worker_uuid=RTX_UUID, handoff=handoff,
        )
        with C.claiming(claim):
            with pytest.raises(GpuPlacementError) as refused:
                gpu_placement.place_job("all", required_mb=4_000, allow_shard=False)
            assert leases(node.db) == {}, "the refused job leased cards"

        assert SPLIT_REFUSED in str(refused.value), "the worker's refusal does not say what submit says"
