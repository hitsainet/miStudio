"""Consumers that read every SAE as the residual stream refuse an MLP or attention SAE (review R1-A, A5).

A training over several hook types trains an MLP-output and an attention-output SAE beside
each residual one, and each imports as an ExternalSAE row with its hook recorded. Three
consumers assumed resid_post whatever the row said:

* feature extraction hooks each layer's decoder output for every SAE (hook_types is fixed
  to residual), so an MLP SAE's features were computed from the residual stream;
* the Neuronpedia export writes hook_resid_post into the SAELens cfg.json of every SAE;
* the local Neuronpedia push names its source by layer and upserts neurons ON CONFLICT
  (model, layer, index), so an MLP SAE overwrote the residual SAE of the same layer.

Each now refuses with the reason. The services raise UnsupportedSaeHook (a ValueError), or
skip the SAE in a batch; the endpoints answer it with 422. The Neuronpedia export's check
lives in the service only, because the endpoint refuses gpu="all" before it touches the
database (test_all_is_refused_where_a_job_cannot_split calls it with db=None). A residual
SAE is not refused.

MUTATION CONTROLS (one line broken at a time, this file run, bytes restored, sha256 and the
working tree verified clean; the full table is in the A5 record):
  N1 extract-features endpoint check removed        -> the extraction endpoint test (400, not 422)
  N2 start_extraction_for_sae check removed          -> the extraction service test
  N3 the batch skip removed                          -> the batch test
  N4 export endpoint's UnsupportedSaeHook -> 422 removed  -> the export endpoint test (400, not 422)
  N5 start_export check removed                      -> both export tests
  N6 push-local check removed                        -> the push test
  N7 the helper stops recognising mlp                -> the helper test and every mlp refusal
"""

import pytest

from src.models.external_sae import ExternalSAE
from src.models.model import Model
from src.services.sae_hook_support import non_residual_hook_reason

PHRASE = "supports residual-stream SAEs only"


@pytest.mark.parametrize("hook", [None, "", "residual", "hook_resid_post", "blocks.3.hook_resid_post"])
def test_a_residual_or_unrecorded_hook_is_accepted(hook):
    assert non_residual_hook_reason(hook, "X") is None


@pytest.mark.parametrize("hook", ["mlp", "attention", "hook_mlp_out", "blocks.3.hook_attn_out", "MLP"])
def test_an_mlp_or_attention_hook_is_refused_with_its_name(hook):
    reason = non_residual_hook_reason(hook, "Feature extraction")
    assert reason is not None and reason.startswith("Feature extraction " + PHRASE) and repr(hook) in reason


async def _seed(async_session):
    from src.models.model import ModelStatus, QuantizationFormat

    async_session.add(Model(
        id="m_hooks", name="tiny", architecture="llama", params_count=1_000,
        quantization=QuantizationFormat.FP16, status=ModelStatus.READY,
    ))
    await async_session.flush()
    for sae_id, hook in (("sae_res", "residual"), ("sae_mlp", "mlp")):
        async_session.add(ExternalSAE(
            id=sae_id, name="L3 " + hook, source="trained", status="ready", model_id="m_hooks",
            layer=3, hook_type=hook, n_features=16, d_model=8, local_path="/nonexistent/" + sae_id,
            sae_metadata=dict(), progress=100.0,
        ))
    await async_session.commit()


def _detail(response):
    body = response.json()
    return str(body.get("detail", body))


@pytest.mark.asyncio
async def test_feature_extraction_endpoint_refuses_an_mlp_sae_and_not_a_residual_one(client, async_session):
    await _seed(async_session)
    refused = await client.post("/api/v1/saes/sae_mlp/extract-features?dataset_id=ds_none", json=dict())
    assert refused.status_code == 422 and PHRASE in _detail(refused), refused.text

    residual = await client.post("/api/v1/saes/sae_res/extract-features?dataset_id=ds_none", json=dict())
    assert PHRASE not in residual.text, residual.text


@pytest.mark.asyncio
async def test_feature_extraction_service_refuses_an_mlp_sae(async_session):
    from src.services.extraction_service import ExtractionService

    await _seed(async_session)
    with pytest.raises(ValueError, match=PHRASE):
        await ExtractionService(async_session).start_extraction_for_sae("sae_mlp", dict(dataset_id="ds_none"))


@pytest.mark.asyncio
async def test_a_batch_skips_the_mlp_sae_with_its_reason(async_session):
    from src.services.extraction_service import ExtractionService

    await _seed(async_session)
    result = await ExtractionService(async_session).start_batch_extraction_for_saes(
        sae_ids=["sae_mlp"], config=dict(dataset_id="ds_none"),
    )
    skipped = [s for s in result["skipped_saes"] if s["sae_id"] == "sae_mlp"]
    assert len(skipped) == 1 and PHRASE in skipped[0]["reason"], result


@pytest.mark.asyncio
async def test_neuronpedia_export_endpoint_refuses_an_mlp_sae_and_not_a_residual_one(client, async_session):
    await _seed(async_session)
    refused = await client.post("/api/v1/neuronpedia/export", json=dict(sae_id="sae_mlp"))
    assert refused.status_code == 422 and PHRASE in _detail(refused), refused.text

    residual = await client.post("/api/v1/neuronpedia/export", json=dict(sae_id="sae_res"))
    assert PHRASE not in residual.text, residual.text


@pytest.mark.asyncio
async def test_neuronpedia_export_service_refuses_an_mlp_sae(async_session):
    from src.services.neuronpedia_export_service import get_neuronpedia_export_service

    await _seed(async_session)
    with pytest.raises(ValueError, match=PHRASE):
        await get_neuronpedia_export_service().start_export(async_session, "sae_mlp")


@pytest.mark.asyncio
async def test_the_local_neuronpedia_push_refuses_an_mlp_sae(client, async_session, monkeypatch):
    from src.api.v1.endpoints import neuronpedia as endpoint

    monkeypatch.setattr(endpoint.settings, "neuronpedia_local_db_url", "postgresql://unused/neuronpedia")
    await _seed(async_session)
    refused = await client.post("/api/v1/neuronpedia/push-local?sae_id=sae_mlp")
    assert refused.status_code == 422 and PHRASE in _detail(refused), refused.text
