"""Consumers that read or write an SAE at its layer's OUTPUT refuse SAEs recorded at another hook (review R2-B).

A5 made feature extraction, the Neuronpedia export and the local push refuse MLP and attention
SAEs, and recorded steering and circuit capture as debt. Round 2 reassessed every other
consumer (record: review_sae_remediation_R2_B_2026-09-15.md). Every one below reads the
decoder layer's output (resid_post) or writes a decoder direction there, whatever the SAE's
hook, and none looked at ``hook_type``:

* circuit capture ENCODES resid_post through each SAE, so an MLP-output SAE's co-activation
  store, and every attribution, validation and faithfulness number computed from it, is
  silently wrong. Refused at ``create_run`` (a 422) and again in ``run_capture`` before the
  model loads, which a confirm or a retry of an existing row reaches without ``create_run``;
* the steering endpoints' shared resolver (async compare, sweep, combined) and the core the
  calibration and the recorder steer through add each decoder direction at the layer output.
  For an attention-output SAE that skips the layer's MLP, and on architectures with a
  post-MLP norm (gemma) an MLP-output direction skips that norm: a plausible, wrong steer;
* the cluster allocation's hazard prior dots one SAE's decoder into another's encoder in the
  residual basis.

The shared rule also now names what A5's substring match missed or passed: Gemma Scope's
``att``, a bare ``hook_z``, transcoders and ``ln2.hook_normalized`` inputs are refused, and so
are ``resid_pre`` and ``resid_mid``, which these consumers would read one layer (or half a
layer) late. A residual or unrecorded hook is never refused.

MUTATION CONTROLS (R2-B; one line broken at a time, this file run, bytes restored, sha256 and
`git diff` verified; the table is in the R2-B record):
  C1 capture create_run's refusal removed                -> the create_run test
  C2 run_capture's refusal removed                       -> the run_capture test
  C3 run_capture's refusal moved after the model load    -> the run_capture test (the model loads)
  S1 the steering resolver's refusal removed             -> both resolver refusals
  S2 the resolver refuses only the per-feature SAEs      -> the request-level refusal
  S3 steering_core's refusal removed                     -> the core test
  A1 the single-layer allocation refusal removed         -> the single-layer test
  A2 the multi-layer allocation refusal removed          -> the multi-layer test
  H1 the helper stops recognising resid_pre              -> the resid_pre cases
  H2 the helper's token match back to a substring match  -> a residual name refused / att missed
"""

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.services.sae_hook_support import non_residual_hook_reason

PHRASE = "supports residual-stream SAEs only"
LATE_PHRASE = "at its layer's output (resid_post)"


# ── the rule ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("hook", [
    None, "", "residual", "resid_post", "hook_resid_post", "blocks.3.hook_resid_post", "res",
    # names that merely CONTAIN the letters of a non-residual hook, not its name
    "hook_resid_post_mlpish", "battery_resid_post", "latent_resid_post", "prefix_resid_post",
])
def test_a_residual_or_unrecorded_hook_is_accepted(hook):
    assert non_residual_hook_reason(hook, "X") is None


@pytest.mark.parametrize("hook", [
    "mlp", "MLP", "hook_mlp_out", "mlp_out", "mlp_in", "blocks.3.mlp.hook_post", "transcoder",
    "blocks.3.ln2.hook_normalized",
])
def test_an_mlp_side_hook_is_refused(hook):
    reason = non_residual_hook_reason(hook, "Steering")
    assert reason is not None and reason.startswith("Steering " + PHRASE) and "MLP" in reason and repr(hook) in reason


@pytest.mark.parametrize("hook", [
    "attention", "attn_out", "blocks.3.hook_attn_out", "att", "hook_z", "blocks.3.attn.hook_z",
])
def test_an_attention_hook_is_refused(hook):
    reason = non_residual_hook_reason(hook, "Steering")
    assert reason is not None and reason.startswith("Steering " + PHRASE) and "attention" in reason
    assert repr(hook) in reason


@pytest.mark.parametrize("hook", ["resid_pre", "hook_resid_pre", "blocks.3.hook_resid_pre", "resid_mid", "blocks.3.hook_resid_mid"])
def test_a_residual_hook_before_the_layer_output_is_refused(hook):
    reason = non_residual_hook_reason(hook, "Circuit capture")
    assert reason is not None and LATE_PHRASE in reason and repr(hook) in reason


# ── circuit capture ──────────────────────────────────────────────────────────


@pytest.fixture
def sync_db(async_engine):
    """A sync session on the test database conftest's async_engine built (as in
    test_circuit_discovery_service)."""
    url = async_engine.url.set(drivername=async_engine.url.drivername.replace("+asyncpg", "").replace("+psycopg", ""))
    engine = create_engine(url)
    session = sessionmaker(bind=engine)()
    yield session
    session.rollback()
    session.close()
    engine.dispose()


def _seed_sync(db):
    from src.models.dataset import Dataset, DatasetStatus
    from src.models.external_sae import ExternalSAE
    from src.models.model import Model, ModelStatus, QuantizationFormat

    db.add(Model(id="m_hooks", name="tiny", architecture="llama", params_count=1_000,
                 quantization=QuantizationFormat.FP16, status=ModelStatus.READY))
    dataset = Dataset(id=uuid.uuid4(), name="corpus", source="HuggingFace", status=DatasetStatus.READY)
    db.add(dataset)
    db.flush()
    for sae_id, hook in (("sae_res", "residual"), ("sae_mlp", "mlp")):
        db.add(ExternalSAE(
            id=sae_id, name="L3 " + hook, source="trained", status="ready", model_id="m_hooks",
            layer=3, hook_type=hook, n_features=16, d_model=8, local_path="/nonexistent/" + sae_id,
            sae_metadata=dict(), progress=100.0,
        ))
    db.commit()
    return str(dataset.id)


def test_capture_create_run_refuses_an_mlp_sae_and_not_a_residual_one(sync_db):
    from src.services.circuit_capture_service import CaptureConfigError, CircuitCaptureService

    dataset_id = _seed_sync(sync_db)
    with pytest.raises(CaptureConfigError, match=PHRASE):
        CircuitCaptureService.create_run(sync_db, dict(dataset_id=dataset_id, layers=[dict(layer=3, sae_id="sae_mlp")]))
    sync_db.rollback()
    try:
        CircuitCaptureService.create_run(sync_db, dict(dataset_id=dataset_id, layers=[dict(layer=3, sae_id="sae_res")]))
    except CaptureConfigError as exc:  # later checks (tokenization, disk) may refuse; not for the hook
        assert PHRASE not in str(exc), exc


def test_run_capture_refuses_an_mlp_sae_before_it_loads_the_model(sync_db, monkeypatch):
    """A confirm or a retry runs an existing row without create_run."""
    from src.ml import model_loader
    from src.models.circuit_runs import CircuitCaptureRun
    from src.services.circuit_capture_service import CaptureConfigError, CircuitCaptureService

    _seed_sync(sync_db)
    run = CircuitCaptureRun(status="pending", store_path="/nonexistent/store", manifest=dict(
        model_id="m_hooks", layers=[dict(layer=3, sae_id="sae_mlp")],
        corpus=dict(dataset_id="ds_x", tokenization_id="tok_x", sample_cap=4),
    ))
    sync_db.add(run)
    sync_db.commit()

    def loaded(*args, **kwargs):
        raise AssertionError("the model was loaded for a capture that must be refused")

    monkeypatch.setattr(model_loader, "load_model_from_hf", loaded)
    with pytest.raises(CaptureConfigError, match=PHRASE):
        CircuitCaptureService.run_capture(sync_db, run.id, confirmed=True, device="cpu")


# ── steering ─────────────────────────────────────────────────────────────────


def _sae_record(sae_id, hook, layer=3):
    return SimpleNamespace(id=sae_id, status="ready", local_path="saes/" + sae_id, layer=layer, hook_type=hook,
                           n_features=16, d_model=8, architecture="jumprelu")


def _steer_request(request_sae, feature_sae):
    feature = SimpleNamespace(feature_idx=1, layer=3, sae_id=feature_sae)
    return SimpleNamespace(sae_id=request_sae, selected_features=[feature])


async def _resolve(request, records):
    from src.api.v1.endpoints import steering

    async def get_sae(db, sid):
        return records[sid]

    with patch.object(steering.SAEManagerService, "get_sae", new=get_sae), \
            patch.object(steering, "settings") as settings:
        settings.resolve_data_path.return_value = MagicMock(exists=MagicMock(return_value=True))
        return await steering.resolve_referenced_saes(request, records[request.sae_id], db=None)


@pytest.mark.asyncio
async def test_the_steering_resolver_refuses_a_per_feature_mlp_sae():
    records = dict(sae_res=_sae_record("sae_res", "residual"), sae_mlp=_sae_record("sae_mlp", "mlp"))
    with pytest.raises(HTTPException) as refused:
        await _resolve(_steer_request("sae_res", "sae_mlp"), records)
    assert refused.value.status_code == 422 and PHRASE in str(refused.value.detail), refused.value.detail


@pytest.mark.asyncio
async def test_the_steering_resolver_refuses_a_request_level_attention_sae_and_not_a_residual_one():
    records = dict(sae_att=_sae_record("sae_att", "attention"), sae_res=_sae_record("sae_res", None))
    with pytest.raises(HTTPException) as refused:
        await _resolve(_steer_request("sae_att", None), records)
    assert refused.value.status_code == 422 and PHRASE in str(refused.value.detail), refused.value.detail

    served = await _resolve(_steer_request("sae_res", None), records)
    assert list(served) == ["sae_res"]


def test_the_steering_core_refuses_an_mlp_sae_before_loading_it(monkeypatch):
    """The core the circuit calibration and the steering recorder steer through."""
    import torch

    from src.services import circuit_capture_service, steering_core, steering_service
    from src.services.steering_core import SteeringCoreError

    rows = dict(sae_mlp=SimpleNamespace(id="sae_mlp", hook_type="hook_mlp_out"),
                sae_res=SimpleNamespace(id="sae_res", hook_type="residual"))
    loads = []

    class _Query:
        def filter(self, condition):
            return SimpleNamespace(first=lambda: rows.get(condition.right.value))

    db = SimpleNamespace(query=lambda model: _Query())
    monkeypatch.setattr(circuit_capture_service, "_load_sae_sync", lambda rec, device: loads.append(rec.id) or rec)
    monkeypatch.setattr(steering_service, "resolve_decoder_weight", lambda sae: torch.zeros(8, 16))

    with pytest.raises(SteeringCoreError, match=PHRASE):
        steering_core._load_wdec_by_layer({3: "sae_mlp"}, db, "cpu")
    assert loads == [], "the SAE was loaded before it was refused"

    assert set(steering_core._load_wdec_by_layer({3: "sae_res"}, db, "cpu")) == {3}
    assert loads == ["sae_res"]


# ── cluster allocation ───────────────────────────────────────────────────────


def _alloc_sae(layer, sae_id, hook):
    sae = MagicMock()
    sae.status, sae.local_path, sae.n_features, sae.layer = "ready", "saes/" + sae_id, 100, layer
    sae.d_model, sae.architecture, sae.hook_type = 16, "jumprelu", hook
    return sae


@pytest.mark.asyncio
async def test_the_single_layer_allocation_refuses_an_mlp_sae():
    from src.api.v1.endpoints.steering import compute_cluster_strength_allocation
    from src.schemas.steering import ClusterAllocationMember, ClusterAllocationRequest

    req = ClusterAllocationRequest(sae_id="sae_x", members=[
        ClusterAllocationMember(feature_idx=i, layer=12, similarity=0.8, activation_frequency=0.2) for i in (0, 1)
    ])
    with patch("src.api.v1.endpoints.steering.SAEManagerService.get_sae",
               new=AsyncMock(return_value=_alloc_sae(12, "sae_x", "mlp"))):
        with pytest.raises(HTTPException) as refused:
            await compute_cluster_strength_allocation(req, db=MagicMock())
    assert refused.value.status_code == 422 and PHRASE in str(refused.value.detail), refused.value.detail


@pytest.mark.asyncio
async def test_the_multi_layer_allocation_refuses_an_attention_sae():
    from src.api.v1.endpoints.steering import compute_cluster_strength_allocation
    from src.schemas.steering import ClusterAllocationMember, ClusterAllocationRequest

    req = ClusterAllocationRequest(sae_id="sae_A", members=[
        ClusterAllocationMember(feature_idx=1, layer=13, similarity=0.8, activation_frequency=0.2, sae_id="sae_A"),
        ClusterAllocationMember(feature_idx=2, layer=14, similarity=0.8, activation_frequency=0.2, sae_id="sae_B"),
    ])
    saes = dict(sae_A=_alloc_sae(13, "sae_A", "residual"), sae_B=_alloc_sae(14, "sae_B", "attention"))

    async def get_sae(db, sid):
        return saes[sid]

    with patch("src.api.v1.endpoints.steering.SAEManagerService.get_sae", new=get_sae), \
            patch("src.api.v1.endpoints.steering.get_steering_service"), \
            patch("src.api.v1.endpoints.steering.settings") as st:
        st.resolve_data_path.return_value = MagicMock(exists=MagicMock(return_value=False))
        st.steering_cluster_constants_json = "{}"
        st.steering_hazard_prior_threshold = 0.5
        with pytest.raises(HTTPException) as refused:
            await compute_cluster_strength_allocation(req, db=MagicMock())
    assert refused.value.status_code == 422 and PHRASE in str(refused.value.detail), refused.value.detail
