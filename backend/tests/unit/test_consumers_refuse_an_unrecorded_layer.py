"""Consumers that read an SAE AT ITS LAYER refuse one recording no layer (review R3-B, R3B-11).

Seven of them turned a NULL ``external_saes.layer`` into ``layer or 0`` and said nothing:
feature extraction hooked layer 0, the Neuronpedia export published layer 0 and the local
push KEYED its source by it (upserting neurons ON CONFLICT (model, layer, index), on top of
whatever really was layer 0), the feature browser reported every feature at layer 0 -- which
is where the steering panel then routed them -- and the steering loader steered layer 0. A
steer at the wrong layer reads as a weak or odd steer, never as an error, which is why this
had to be refused rather than logged.

A RECORDED layer 0 is a real layer and is accepted throughout; only a NULL is refused. The
refusals went in AFTER ``scripts/backfill_sae_hooks.py`` existed, because refusing first
would have failed every historical row at once with nothing to repair it.

Two related defects are pinned here too:
* R3B-13 -- the export wrote ``blocks.L.hook_resid_post`` for an SAE that recorded no hook,
  and re-importing that directory made the resolver RECORD it: a NULL laundered into a claim
  by a round trip through our own artifact. The exported files now say whether the hook was
  recorded, and the resolver honours it.
* B11 -- the Gemma Scope and SAELens loaders invent ``blocks.0.hook_resid_post`` for a file
  that records no layer, and an invented 0 is indistinguishable from a recorded one. Its
  single PRODUCTION consumer is ``steering_service.load_sae``: grepping the tree,
  ``config.hook_point_layer`` is read in exactly two places outside the loaders, and the
  other is ``SAEConverterService.load_auto``, which NOTHING calls. So steering steered layer
  0 believing the checkpoint said so. The loaders now mark the assumption and the loader
  refuses it.
  Two things this is NOT, checked rather than assumed: the download task is unaffected
  (``get_sae_info`` sets ``layer`` only when the path names one, so its fallback was already
  NULL, never 0), and the ``load_auto`` marking below is defensive, not a wired fix.

MUTATION CONTROLS (one line broken at a time, the listed tests run, bytes restored from a
byte copy, sha256 and ``git diff`` verified clean):
  U1  extraction_service.start_extraction_for_sae: the layer refusal removed
        -> the extraction endpoint and service tests
  U2  extraction_service batch path: the layer skip removed
        -> test_a_batch_skips_it_with_its_reason_and_runs_on
  U3  extraction_service worker: ``layer_index = external_sae.layer or 0`` restored
        -> test_the_worker_guards_the_layer_it_hooks
  U4  neuronpedia_export_service.start_export: the layer refusal removed
        -> the export endpoint and service tests
  U5  neuronpedia_export_service.exported_hook: ``recorded`` hard-coded True
        -> test_the_export_says_whether_the_hook_was_recorded
  U6  sae_manager_service.hook_name_from_sae_config: the hook_point_recorded check removed
        -> test_a_re_import_does_not_read_an_assumed_hook_as_a_record
  U7  neuronpedia.py push-local: the layer refusal removed
        -> test_the_push_endpoint_refuses_it
  U8  neuronpedia_local_service.push_sae_to_local: ``layer = sae.layer or 0`` restored
        -> test_the_push_service_refuses_it_too
  U9  saes.py browse_sae_features: ``sae.layer if ... else 0`` restored
        -> test_the_feature_browser_refuses_it
  U10 steering.py resolve_referenced_saes: the layer refusal removed
        -> test_the_steering_resolver_refuses_it
  U11 steering_service.load_sae: ``sae_layer = layer or 0`` restored
        -> test_the_loader_refuses_rather_than_steering_layer_zero
  U12 community_format: ``hook_point_assumed`` hard-coded False
        -> the two loader tests and test_the_converter_does_not_pass_an_invented_layer_on
  U13 sae_converter.load_sae_any: passes config.hook_point_layer regardless of the mark
        -> test_the_converter_does_not_pass_an_invented_layer_on
  U14 steering_service.load_sae: an assumed config layer taken as a record
        -> test_an_assumed_checkpoint_layer_is_not_taken_as_a_record
"""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.models.external_sae import ExternalSAE
from src.services.sae_hook_support import (
    UnrecordedSaeLayer,
    UnsupportedSae,
    UnsupportedSaeHook,
    refuse_unrecorded_layer,
    unrecorded_layer_reason,
)

PHRASE = "records no layer"
HINT = "backfill_sae_hooks"
SRC = Path(__file__).resolve().parents[2] / "src"
#: A well-formed id that exists in no table: a request that gets PAST the refusal reaches
#: the next check and answers 400 (review R3B-17: a malformed id 500'd and "not 422" passed).
MISSING_DATASET = "00000000-0000-4000-8000-00000000abcd"


def _detail(response) -> str:
    body = response.json()
    detail = body.get("detail", body) if isinstance(body, dict) else body
    return detail if isinstance(detail, str) else str(detail)


async def _seed(async_session):
    """Three residual SAEs: no layer, a RECORDED layer 0, and layer 3."""
    from src.models.model import Model, ModelStatus, QuantizationFormat

    async_session.add(Model(
        id="m_layer", name="tiny", architecture="llama", params_count=1_000,
        quantization=QuantizationFormat.FP16, status=ModelStatus.READY,
    ))
    await async_session.flush()
    for sae_id, layer in (("sae_nolayer", None), ("sae_layer0", 0), ("sae_layer3", 3)):
        async_session.add(ExternalSAE(
            id=sae_id, name=sae_id, source="huggingface", status="ready", model_id="m_layer",
            layer=layer, hook_type="residual", n_features=16, d_model=8,
            local_path="/nonexistent/" + sae_id, sae_metadata={}, progress=100.0,
        ))
    await async_session.commit()


def _gemma_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    np.savez(
        root / "params.npz",
        W_enc=np.zeros((8, 16), dtype=np.float32), W_dec=np.zeros((16, 8), dtype=np.float32),
        b_enc=np.zeros(16, dtype=np.float32), b_dec=np.zeros(8, dtype=np.float32),
        threshold=np.zeros(16, dtype=np.float32),
    )
    return root


# ── the rule ────────────────────────────────────────────────────────────────


class TestTheRule:
    @pytest.mark.parametrize("layer", [0, 3, 47])
    def test_a_recorded_layer_is_accepted_including_zero(self, layer):
        """Layer 0 is a real layer. What is refused is a NULL silently BECOMING 0."""
        assert unrecorded_layer_reason(layer, "X", "sae_x") is None
        refuse_unrecorded_layer(layer, "X", "sae_x")

    def test_a_missing_layer_is_refused_by_name_and_says_what_to_do(self):
        reason = unrecorded_layer_reason(None, "Feature extraction", "sae_x")
        assert reason.startswith("Feature extraction")
        assert "sae_x" in reason and "LAYER 0" in reason
        assert HINT in reason, "a refusal with no repair instruction is a dead end"

    def test_both_refusals_are_one_exception_family_so_endpoints_answer_422(self):
        for refusal in (UnrecordedSaeLayer, UnsupportedSaeHook):
            assert issubclass(refusal, UnsupportedSae) and issubclass(refusal, ValueError)
        with pytest.raises(UnrecordedSaeLayer, match=PHRASE):
            refuse_unrecorded_layer(None, "X", "sae_x")


# ── feature extraction ──────────────────────────────────────────────────────


class TestFeatureExtraction:
    @pytest.mark.asyncio
    async def test_the_endpoint_refuses_it_and_not_a_recorded_layer_zero(self, client, async_session):
        await _seed(async_session)

        refused = await client.post(
            f"/api/v1/saes/sae_nolayer/extract-features?dataset_id={MISSING_DATASET}", json={}
        )
        assert refused.status_code == 422 and PHRASE in _detail(refused), refused.text

        allowed = await client.post(
            f"/api/v1/saes/sae_layer0/extract-features?dataset_id={MISSING_DATASET}", json={}
        )
        assert allowed.status_code == 400, allowed.text
        assert PHRASE not in allowed.text

    @pytest.mark.asyncio
    async def test_the_service_refuses_before_a_job_row_exists(self, async_session):
        from src.services.extraction_service import ExtractionService

        await _seed(async_session)
        with pytest.raises(UnrecordedSaeLayer, match=PHRASE):
            await ExtractionService(async_session).start_extraction_for_sae(
                sae_id="sae_nolayer", config=dict(dataset_id=MISSING_DATASET),
            )

    @pytest.mark.asyncio
    async def test_a_batch_skips_it_with_its_reason_and_runs_on(self, async_session):
        from src.services.extraction_service import ExtractionService

        await _seed(async_session)
        result = await ExtractionService(async_session).start_batch_extraction_for_saes(
            sae_ids=["sae_nolayer"], config=dict(dataset_id=MISSING_DATASET),
        )
        skipped = [s for s in result["skipped_saes"] if s["sae_id"] == "sae_nolayer"]
        assert len(skipped) == 1 and PHRASE in skipped[0]["reason"], result

    def test_the_worker_guards_the_layer_it_hooks(self):
        """The entry points refuse first; a job row created BEFORE they did still reaches the
        worker, and ``external_sae.layer or 0`` there is the line that hooked layer 0.

        The AST, not a substring: this file and that file both NAME the defect in prose.
        """
        source = SRC / "services" / "extraction_service.py"
        tree = ast.parse(source.read_text(), filename=str(source))

        holders = []
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for inner in ast.walk(node):
                if isinstance(inner, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "layer_index" for t in inner.targets
                ):
                    assert not isinstance(inner.value, ast.BoolOp), (
                        "the hook layer is back to `layer or 0`: " + ast.unparse(inner)
                    )
                    holders.append(node)
        assert holders, "no function assigns layer_index — this guard would pass over nothing"

        for holder in holders:
            calls = {
                call.func.id for call in ast.walk(holder)
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
            }
            assert "refuse_unrecorded_layer" in calls, (
                f"{holder.name} computes the hook layer without refusing an unrecorded one"
            )


# ── the Neuronpedia export (R3B-11 + R3B-13) ────────────────────────────────


class TestTheNeuronpediaExport:
    @pytest.mark.asyncio
    async def test_the_endpoint_refuses_it(self, client, async_session):
        await _seed(async_session)
        refused = await client.post("/api/v1/neuronpedia/export", json=dict(sae_id="sae_nolayer"))
        assert refused.status_code == 422 and PHRASE in _detail(refused), refused.text

        allowed = await client.post("/api/v1/neuronpedia/export", json=dict(sae_id="sae_layer0"))
        assert PHRASE not in allowed.text, allowed.text

    @pytest.mark.asyncio
    async def test_the_service_refuses_it(self, async_session):
        from src.services.neuronpedia_export_service import get_neuronpedia_export_service

        await _seed(async_session)
        with pytest.raises(ValueError, match=PHRASE):
            await get_neuronpedia_export_service().start_export(async_session, "sae_nolayer")

    def test_the_export_says_whether_the_hook_was_recorded(self):
        """R3B-13. SAELens and Neuronpedia both require a hook name, so one is written for an
        SAE that recorded none -- but it must not be published as a FACT."""
        from src.services.neuronpedia_export_service import get_neuronpedia_export_service

        service = get_neuronpedia_export_service()

        name, recorded = service.exported_hook(
            SimpleNamespace(id="sae_r", layer=3, hook_type="blocks.3.hook_resid_post")
        )
        assert (name, recorded) == ("blocks.3.hook_resid_post", True)

        name, recorded = service.exported_hook(SimpleNamespace(id="sae_u", layer=3, hook_type=None))
        assert name == "blocks.3.hook_resid_post" and recorded is False

        with pytest.raises(UnrecordedSaeLayer, match=PHRASE):
            service.exported_hook(SimpleNamespace(id="sae_n", layer=None, hook_type="residual"))

    def test_the_metadata_file_carries_the_mark(self):
        from src.services.neuronpedia_export_service import get_neuronpedia_export_service

        service = get_neuronpedia_export_service()
        sae = SimpleNamespace(
            id="sae_u", layer=3, hook_type=None, n_features=16_384, d_model=8,
            architecture="jumprelu", name="u",
        )
        metadata = service._generate_metadata_json(sae, "gemma-2-2b")
        assert metadata["hook_point_recorded"] is False
        assert metadata["hook_point"] == "blocks.3.hook_resid_post"

        sae.hook_type = "blocks.3.hook_resid_post"
        assert service._generate_metadata_json(sae, "gemma-2-2b")["hook_point_recorded"] is True

    def test_a_re_import_does_not_read_an_assumed_hook_as_a_record(self):
        """The other half of R3B-13: without this, exporting and re-importing an SAE whose
        hook was never recorded turned that NULL into a claimed resid_post."""
        from src.services.sae_manager_service import hook_name_from_sae_config

        written = dict(hook_name="blocks.3.hook_resid_post")
        assert hook_name_from_sae_config(written) == "blocks.3.hook_resid_post"
        assert hook_name_from_sae_config({**written, "hook_point_recorded": True}) == \
            "blocks.3.hook_resid_post"
        assert hook_name_from_sae_config({**written, "hook_point_recorded": False}) is None


# ── the local Neuronpedia push ──────────────────────────────────────────────


class TestTheLocalPush:
    @pytest.mark.asyncio
    async def test_the_push_endpoint_refuses_it(self, client, async_session, monkeypatch):
        from src.api.v1.endpoints import neuronpedia as endpoint

        monkeypatch.setattr(
            endpoint.settings, "neuronpedia_local_db_url", "postgresql://unused/neuronpedia"
        )
        await _seed(async_session)
        refused = await client.post("/api/v1/neuronpedia/push-local?sae_id=sae_nolayer")
        assert refused.status_code == 422 and PHRASE in _detail(refused), refused.text

    @pytest.mark.asyncio
    async def test_the_push_service_refuses_it_too(self, async_session, monkeypatch):
        """A task queued before the endpoint refused reaches the service directly."""
        from src.services.neuronpedia_local_service import (
            NeuronpediaLocalPushService,
            get_neuronpedia_local_push_service,
        )

        await _seed(async_session)
        monkeypatch.setattr(
            NeuronpediaLocalPushService, "_get_client", AsyncMock(return_value=MagicMock()),
        )
        result = await get_neuronpedia_local_push_service().push_sae_to_local(
            async_session, "sae_nolayer",
        )
        assert result.success is False and PHRASE in (result.error_message or "")


# ── the feature browser (what the steering panel routes by) ─────────────────


class TestTheFeatureBrowser:
    @pytest.mark.asyncio
    async def test_the_feature_browser_refuses_it(self, client, async_session):
        await _seed(async_session)
        refused = await client.get("/api/v1/saes/sae_nolayer/features")
        assert refused.status_code == 422 and PHRASE in _detail(refused), refused.text

    @pytest.mark.asyncio
    async def test_a_recorded_layer_zero_still_browses_and_reports_layer_zero(
        self, client, async_session
    ):
        """Every summary carries the SAE's layer, and the steering panel routes by it -- so
        this asserts the PAYLOAD, not just a 200."""
        from src.models.extraction_job import ExtractionJob, ExtractionStatus
        from src.models.feature import Feature

        await _seed(async_session)
        async_session.add(ExtractionJob(
            id="ext_l0", external_sae_id="sae_layer0", status=ExtractionStatus.COMPLETED, config={},
        ))
        await async_session.flush()
        async_session.add(Feature(
            id="feat_l0_0", external_sae_id="sae_layer0", extraction_job_id="ext_l0",
            neuron_index=0, name="feature_0", activation_frequency=0.01,
            interpretability_score=0.5, max_activation=1.0, mean_activation=0.1,
        ))
        await async_session.commit()

        allowed = await client.get("/api/v1/saes/sae_layer0/features?limit=3")
        assert allowed.status_code == 200, allowed.text
        features = allowed.json()["features"]
        assert features and all(feature["layer"] == 0 for feature in features)


# ── steering ────────────────────────────────────────────────────────────────


def _sae_record(sae_id, layer, hook="residual"):
    return SimpleNamespace(
        id=sae_id, status="ready", local_path="saes/" + sae_id, layer=layer, hook_type=hook,
        n_features=16, d_model=8, architecture="jumprelu",
    )


async def _resolve(request, records):
    from src.api.v1.endpoints import steering

    async def get_sae(db, sid):
        return records[sid]

    with patch.object(steering.SAEManagerService, "get_sae", new=get_sae), \
            patch.object(steering, "settings") as settings:
        settings.resolve_data_path.return_value = MagicMock(exists=MagicMock(return_value=True))
        return await steering.resolve_referenced_saes(request, records[request.sae_id], db=None)


def _steer_request(sae_id, layer=3):
    feature = SimpleNamespace(feature_idx=1, layer=layer, sae_id=None)
    return SimpleNamespace(sae_id=sae_id, selected_features=[feature])


class TestSteering:
    @pytest.mark.asyncio
    async def test_the_steering_resolver_refuses_it(self):
        """The per-feature layer check below it is no substitute: it SKIPS a NULL layer, so
        it agreed with whatever layer the request named."""
        from fastapi import HTTPException

        records = dict(sae_none=_sae_record("sae_none", None), sae_0=_sae_record("sae_0", 0))
        with pytest.raises(HTTPException) as refused:
            await _resolve(_steer_request("sae_none"), records)
        assert refused.value.status_code == 422 and PHRASE in str(refused.value.detail)

        served = await _resolve(_steer_request("sae_0", layer=0), records)
        assert list(served) == ["sae_0"], "a recorded layer 0 is a real layer"

    @pytest.mark.asyncio
    async def test_the_loader_refuses_rather_than_steering_layer_zero(self, monkeypatch):
        import torch

        from src.services import steering_service as mod

        service = mod.SteeringService.__new__(mod.SteeringService)
        service._loaded_saes = {}
        service._device = torch.device("cpu")
        service._placement = None
        monkeypatch.setattr(mod, "load_sae_auto_detect", lambda path, device: ({}, None, "mistudio"))

        with pytest.raises(UnrecordedSaeLayer, match=PHRASE):
            await service.load_sae(Path("/nonexistent"), "sae_x", layer=None)

    @pytest.mark.asyncio
    async def test_an_assumed_checkpoint_layer_is_not_taken_as_a_record(self, monkeypatch):
        """B11. The loaders INVENT ``blocks.0.hook_resid_post`` for a file recording no layer,
        so trusting the checkpoint's layer steered layer 0 believing the file said so."""
        import torch

        from src.ml.community_format import CommunityStandardConfig
        from src.services import steering_service as mod

        assumed = CommunityStandardConfig(
            model_name="tiny", hook_point="blocks.0.hook_resid_post", hook_point_layer=0,
            d_in=8, d_sae=16, extra_metadata={"hook_point_assumed": True},
        )
        service = mod.SteeringService.__new__(mod.SteeringService)
        service._loaded_saes = {}
        service._device = torch.device("cpu")
        service._placement = None
        monkeypatch.setattr(
            mod, "load_sae_auto_detect", lambda path, device: ({}, assumed, "gemma_scope")
        )

        with pytest.raises(UnrecordedSaeLayer, match=PHRASE):
            await service.load_sae(Path("/nonexistent"), "sae_y", layer=None)


# ── B11: an invented hook and layer are marked as invented ──────────────────


class TestAnInventedHookIsMarked:
    def test_a_params_npz_whose_path_names_no_layer_is_marked_assumed(self, tmp_path):
        from src.ml.community_format import load_sae_auto_detect

        _, config, fmt = load_sae_auto_detect(_gemma_dir(tmp_path / "somewhere"), "cpu")
        assert fmt == "gemma_scope"
        # The invented value is still written (the dataclass requires one) -- but labelled.
        assert config.hook_point_layer == 0
        assert config.extra_metadata["hook_point_assumed"] is True

    def test_a_layer_in_the_path_is_a_record_not_an_assumption(self, tmp_path):
        from src.ml.community_format import load_sae_auto_detect

        _, config, _ = load_sae_auto_detect(_gemma_dir(tmp_path / "layer_20" / "width_16k"), "cpu")
        assert config.hook_point_layer == 20
        assert config.extra_metadata["hook_point_assumed"] is False

    def test_the_converter_marks_it_too_although_nothing_calls_the_converter(self, tmp_path):
        """``load_auto`` reads the same invented value and is corrected with the rest -- but
        it has NO production caller, and a test that quietly asserted dead code would look
        like protection it is not.

        So it is said here, with a guard: if someone wires ``load_auto`` up, this fails and
        they must re-check whether an assumed layer can flow on from there. (The path that
        DOES reach the database, the download task, goes through ``get_sae_info``, which
        never invents a layer at all.)
        """
        from src.services.sae_converter import SAEConverterService

        callers = []
        for path in sorted(SRC.rglob("*.py")):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "load_auto"):
                    callers.append(f"{path.relative_to(SRC)}:{node.lineno}")
        assert callers == [], (
            "load_auto now has a production caller: re-check whether an assumed hook or "
            "layer can flow from it into a recorded value. " + ", ".join(callers)
        )

        _, metadata, _ = SAEConverterService.load_auto(
            str(_gemma_dir(tmp_path / "somewhere")), "cpu"
        )
        assert metadata["layer"] is None and metadata["hook_point"] is None

        _, recorded, _ = SAEConverterService.load_auto(
            str(_gemma_dir(tmp_path / "layer_20")), "cpu"
        )
        assert recorded["layer"] == 20 and recorded["hook_point"] == "blocks.20.hook_resid_post"
