""""all" is refused at submit by a job that never runs split.

Multi-GPU Phase 2, review round 1 (2026-09-14). `GpuRequestStr` accepts "all" on
every GPU job's request, and `resolve_request` stores it for every job. A job
placed without `allow_shard` refuses it only in its worker — after its row was
created and it was queued:

* a training on cached activations (`place_job(..., allow_shard=False)`: it
  loads no model), queued and then failed with "This job cannot run split";
* the logit lens — the dashboard-data task, the local push, and the export —
  which resolves its card with `resolve_card`, where "all" was looked up as a
  card UUID and refused as "No GPU 'all' on this node".

`resolve_gpu_request(..., can_split=False)` now refuses "all" with a 400 before a
row exists, and `resolve_card` refuses it with the honest message. A job that
does run split (an on-the-fly training; a dashboard, push or export without the
logit lens) still passes "all" through.

MUTATION CONTROLS (review round 1, 2026-09-14; scratchpad p2-r1-core/mutate.py,
each alone, this file run, source restored and checked by sha256):
  A1  the helper never refuses "all"                  -> 6 tests: the helper's, both trainings refusals,
                                                          the dashboard, push and export refusals
  A2  trainings drops can_split                        -> both cached-activation refusals
  A3  trainings' predicate inverted                    -> both refusals + an_on_the_fly_training_keeps_all
  A4  compute-dashboard drops can_split                -> dashboard_data_with_the_lens_refuses_all
  A5  push drops can_split                             -> a_push_computing_dashboard_data_refuses_all
  A6  export drops can_split                           -> an_export_with_the_lens_refuses_all...
  A7  resolve_card looks "all" up as a card again      -> the_one_card_resolver_refuses_all_as_a_split...
  A8  compute-dashboard refuses "all" even without the lens -> dashboard_data_without_the_lens_keeps_all
"""

from types import SimpleNamespace

import pytest
from fastapi import BackgroundTasks, HTTPException

from src.api.v1 import gpu_request as gpu_request_module
from src.api.v1.gpu_request import SPLIT_REFUSED, resolve_gpu_request
from src.schemas.neuronpedia import (
    ComputeDashboardDataRequest,
    NeuronpediaExportConfigRequest,
    NeuronpediaExportRequest,
)
from src.schemas.training import TrainingCreate, TrainingHyperparameters
from src.services import gpu_placement as gp
from src.services.gpu_placement import GpuCard, GpuPlacementError

CARDS = [
    GpuCard(index=0, uuid="GPU-f47ba814-49a2-603f-3595-275284140251", name="RTX 3080 Ti",
            total_mb=12_288, free_mb=11_000),
    GpuCard(index=1, uuid="GPU-247aa582-0d1b-e161-8156-983ed1fefc57", name="RTX 3090",
            total_mb=24_576, free_mb=23_000),
]


class TestTheHelper:
    def test_all_is_a_400_for_a_job_that_cannot_split(self, monkeypatch):
        monkeypatch.setattr(gp, "list_cards", lambda: pytest.fail("NVML read to refuse all"))
        with pytest.raises(HTTPException) as refused:
            resolve_gpu_request("all", can_split=False)
        assert refused.value.status_code == 400
        assert refused.value.detail == SPLIT_REFUSED

    def test_all_is_stored_for_a_job_that_can_split(self):
        assert resolve_gpu_request(" ALL ", can_split=True) == gp.ALL

    def test_auto_is_unaffected(self):
        assert resolve_gpu_request("auto", can_split=False) == gp.AUTO


def test_the_one_card_resolver_refuses_all_as_a_split_not_as_an_unknown_card():
    with pytest.raises(GpuPlacementError, match="cannot run split") as refused:
        gp.resolve_card("all", cards=CARDS)
    assert "No GPU" not in str(refused.value)


def _training(**overrides):
    fields = dict(
        model_id="m_1",
        dataset_ids=["12345678-1234-5678-1234-567812345678"],
        hyperparameters=TrainingHyperparameters(
            hidden_dim=768, latent_dim=16_384, l1_alpha=0.001, learning_rate=0.0003,
            batch_size=4_096, total_steps=1_000,
        ),
        gpu="all",
    )
    fields.update(overrides)
    return TrainingCreate(**fields)


class TestTrainings:
    @pytest.fixture
    def created(self, monkeypatch):
        from src.api.v1.endpoints import trainings

        calls = []

        async def create_training(db, training, gpu_request=None):
            calls.append(gpu_request)
            raise ValueError("stopped after the request was resolved")

        monkeypatch.setattr(trainings.TrainingService, "create_training", create_training)
        return trainings, calls

    @pytest.mark.asyncio
    @pytest.mark.parametrize("cached", [{"extraction_ids": ["ext_m_1_x"]}, {"extraction_id": "ext_m_1_x"}])
    async def test_a_training_on_cached_activations_refuses_all_before_a_row_exists(self, created, cached):
        trainings, calls = created
        with pytest.raises(HTTPException) as refused:
            await trainings.create_training(training=_training(**cached), db=None)
        assert (refused.value.status_code, refused.value.detail) == (400, SPLIT_REFUSED)
        assert calls == [], "the training was created before all was refused"

    @pytest.mark.asyncio
    async def test_an_on_the_fly_training_keeps_all(self, created):
        trainings, calls = created
        with pytest.raises(HTTPException, match="stopped after"):
            await trainings.create_training(training=_training(), db=None)
        assert calls == [gp.ALL]


class TestTheLogitLens:
    @pytest.fixture
    def neuronpedia(self, monkeypatch):
        from src.api.v1.endpoints import neuronpedia

        state = SimpleNamespace(tasks=[], exports=[])
        monkeypatch.setattr(
            neuronpedia, "compute_dashboard_data_task",
            SimpleNamespace(delay=lambda **kwargs: state.tasks.append(kwargs) or SimpleNamespace(id="task_1")),
        )

        class _ExportService:
            async def start_export(self, db, sae_id, config):
                state.exports.append(config)
                return "job_1"

        monkeypatch.setattr(neuronpedia, "get_neuronpedia_export_service", lambda: _ExportService())
        monkeypatch.setattr(neuronpedia.settings, "neuronpedia_local_db_url", "postgresql://neuronpedia")
        return neuronpedia, state

    @pytest.mark.asyncio
    async def test_dashboard_data_with_the_lens_refuses_all(self, neuronpedia):
        module, state = neuronpedia
        request = ComputeDashboardDataRequest(sae_id="sae_1", include_logit_lens=True, gpu="all")
        with pytest.raises(HTTPException) as refused:
            await module.compute_dashboard_data(request=request, background_tasks=BackgroundTasks(), db=None)
        assert (refused.value.status_code, refused.value.detail) == (400, SPLIT_REFUSED)
        assert state.tasks == []

    @pytest.mark.asyncio
    async def test_dashboard_data_without_the_lens_keeps_all(self, neuronpedia):
        module, state = neuronpedia
        request = ComputeDashboardDataRequest(sae_id="sae_1", include_logit_lens=False, gpu="all")
        await module.compute_dashboard_data(request=request, background_tasks=BackgroundTasks(), db=None)
        assert [task["gpu_request"] for task in state.tasks] == [gp.ALL]

    @pytest.mark.asyncio
    async def test_an_export_with_the_lens_refuses_all_before_the_job_exists(self, neuronpedia):
        module, state = neuronpedia
        request = NeuronpediaExportRequest(
            sae_id="sae_1", config=NeuronpediaExportConfigRequest(include_logit_lens=True), gpu="all"
        )
        with pytest.raises(HTTPException) as refused:
            await module.start_export(request=request, background_tasks=BackgroundTasks(), db=None)
        assert (refused.value.status_code, refused.value.detail) == (400, SPLIT_REFUSED)
        assert state.exports == []

    @pytest.mark.asyncio
    async def test_an_export_without_the_lens_keeps_all(self, neuronpedia):
        module, state = neuronpedia
        request = NeuronpediaExportRequest(
            sae_id="sae_1", config=NeuronpediaExportConfigRequest(include_logit_lens=False), gpu="all"
        )
        await module.start_export(request=request, background_tasks=BackgroundTasks(), db=None)
        assert [config.gpu for config in state.exports] == [gp.ALL]

    def _push(self, module, db, compute_dashboard_data):
        return module.push_to_local_neuronpedia(
            sae_id="sae_1", include_activations=True, include_explanations=True,
            max_activations_per_feature=20, feature_indices=None, visibility="PUBLIC",
            compute_dashboard_data=compute_dashboard_data, logit_lens_k=20, gpu="all", db=db,
        )

    @pytest.mark.asyncio
    async def test_a_push_computing_dashboard_data_refuses_all(self, neuronpedia):
        module, _ = neuronpedia

        class _NoDb:
            async def get(self, *args):
                pytest.fail("the push looked up its SAE before all was refused")

        with pytest.raises(HTTPException) as refused:
            await self._push(module, _NoDb(), compute_dashboard_data=True)
        assert (refused.value.status_code, refused.value.detail) == (400, SPLIT_REFUSED)

    @pytest.mark.asyncio
    async def test_a_push_without_dashboard_data_keeps_all(self, neuronpedia):
        module, _ = neuronpedia
        looked_up = []

        class _Db:
            async def get(self, model, sae_id):
                looked_up.append(sae_id)
                return None

        with pytest.raises(HTTPException) as missing:
            await self._push(module, _Db(), compute_dashboard_data=False)
        assert missing.value.status_code == 404 and looked_up == ["sae_1"]


def test_the_helper_module_is_what_the_endpoints_import():
    """The endpoints call this module's helper, not a copy: the refusal is one definition."""
    from src.api.v1.endpoints import neuronpedia, trainings

    assert trainings.resolve_gpu_request is gpu_request_module.resolve_gpu_request
    assert neuronpedia.resolve_gpu_request is gpu_request_module.resolve_gpu_request
