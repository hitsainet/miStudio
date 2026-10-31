"""Feature 015: SUBMIT-time validation for combined multi-SAE steering.

The per-feature SAE routing (feature.sae_id ?? request.sae_id) must be validated
BEFORE the GPU worker runs — a layer/SAE mismatch is a 422 listing offenders, an
unknown or not-ready SAE is a 4xx, and a valid single/multi-SAE request threads a
JSON-serializable SaeMeta map to the Celery task.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from src.api.v1.endpoints.steering import submit_async_combined_steering
from src.schemas.steering import CombinedSteeringRequest, SelectedFeature


def _sae(layer, sae_id, n_features=100):
    s = MagicMock()
    s.status = "ready"
    s.local_path = f"saes/{sae_id}"
    s.n_features = n_features
    s.layer = layer
    s.d_model = 16
    s.architecture = "jumprelu"
    s.model_id = "m1"
    s.model_name = "m1"
    return s


def _req(features, sae_id="sae_A"):
    return CombinedSteeringRequest(
        sae_id=sae_id, model_id="m1", prompt="hi", selected_features=features,
    )


def _http_request():
    r = MagicMock()
    r.client = MagicMock(host="127.0.0.1")
    r.headers = {}
    return r


@pytest.fixture(autouse=True)
def _bypass_gates():
    """Rate limiter + worker start are orthogonal to routing validation."""
    with patch("src.api.v1.endpoints.steering._rate_limiter") as rl, patch(
        "src.api.v1.endpoints.steering._ensure_steering_worker_running",
        new=AsyncMock(return_value=(True, 123)),
    ):
        rl.is_allowed.return_value = True
        yield


@pytest.mark.asyncio
async def test_422_layer_mismatch_lists_offenders():
    req = _req([
        SelectedFeature(feature_idx=1, layer=13, sae_id="sae_A", strength=50),
        SelectedFeature(feature_idx=2, layer=99, sae_id="sae_A", strength=50),  # wrong layer
    ])
    with patch(
        "src.api.v1.endpoints.steering.SAEManagerService.get_sae",
        new=AsyncMock(return_value=_sae(13, "sae_A")),
    ), patch("src.api.v1.endpoints.steering.settings") as st:
        st.resolve_data_path.return_value = MagicMock(exists=MagicMock(return_value=True))
        with pytest.raises(HTTPException) as e:
            await submit_async_combined_steering(req, _http_request(), db=MagicMock())
    assert e.value.status_code == 422
    assert e.value.detail["code"] == "sae_layer_mismatch"
    assert any(o["feature_idx"] == 2 and o["layer"] == 99 for o in e.value.detail["offenders"])


@pytest.mark.asyncio
async def test_404_when_referenced_sae_missing():
    req = _req([
        SelectedFeature(feature_idx=1, layer=13, sae_id="sae_A", strength=50),
        SelectedFeature(feature_idx=2, layer=14, sae_id="sae_MISSING", strength=50),
    ])
    saes = {"sae_A": _sae(13, "sae_A")}

    async def get_sae(db, sid):
        return saes.get(sid)

    with patch(
        "src.api.v1.endpoints.steering.SAEManagerService.get_sae", new=get_sae
    ), patch("src.api.v1.endpoints.steering.settings") as st:
        st.resolve_data_path.return_value = MagicMock(exists=MagicMock(return_value=True))
        with pytest.raises(HTTPException) as e:
            await submit_async_combined_steering(req, _http_request(), db=MagicMock())
    assert e.value.status_code == 404
    assert "sae_MISSING" in str(e.value.detail)


@pytest.mark.asyncio
async def test_multi_sae_threads_meta_map_to_task():
    """A valid two-SAE request submits with a two-entry sae_meta_map."""
    req = _req([
        SelectedFeature(feature_idx=1, layer=13, sae_id="sae_A", strength=50),
        SelectedFeature(feature_idx=2, layer=14, sae_id="sae_B", strength=50),
    ])
    saes = {"sae_A": _sae(13, "sae_A"), "sae_B": _sae(14, "sae_B")}

    async def get_sae(db, sid):
        return saes[sid]

    fake_task = MagicMock(id="task-xyz")
    with patch(
        "src.api.v1.endpoints.steering.SAEManagerService.get_sae", new=get_sae
    ), patch("src.api.v1.endpoints.steering.settings") as st, patch(
        "src.api.v1.endpoints.steering.ModelService.get_model", new=AsyncMock(return_value=None)
    ), patch(
        "src.workers.steering_tasks.steering_combined_task"
    ) as task:
        st.resolve_data_path.return_value = MagicMock(exists=MagicMock(return_value=True))
        task.apply_async.return_value = fake_task
        resp = await submit_async_combined_steering(req, _http_request(), db=MagicMock())

    assert resp.task_id == "task-xyz"
    kwargs = task.apply_async.call_args.kwargs["kwargs"]
    meta = kwargs["sae_meta_map"]
    assert set(meta) == {"sae_A", "sae_B"}
    assert meta["sae_A"]["layer"] == 13 and meta["sae_B"]["layer"] == 14


@pytest.mark.asyncio
async def test_single_sae_still_threads_one_entry_meta_map():
    """Single-SAE (no per-feature sae_id) → one-entry map, byte-identical path."""
    req = _req([
        SelectedFeature(feature_idx=1, layer=13, strength=50),
        SelectedFeature(feature_idx=2, layer=13, strength=50),
    ])
    fake_task = MagicMock(id="task-solo")
    with patch(
        "src.api.v1.endpoints.steering.SAEManagerService.get_sae",
        new=AsyncMock(return_value=_sae(13, "sae_A")),
    ), patch("src.api.v1.endpoints.steering.settings") as st, patch(
        "src.api.v1.endpoints.steering.ModelService.get_model", new=AsyncMock(return_value=None)
    ), patch(
        "src.workers.steering_tasks.steering_combined_task"
    ) as task:
        st.resolve_data_path.return_value = MagicMock(exists=MagicMock(return_value=True))
        task.apply_async.return_value = fake_task
        resp = await submit_async_combined_steering(req, _http_request(), db=MagicMock())

    kwargs = task.apply_async.call_args.kwargs["kwargs"]
    assert set(kwargs["sae_meta_map"]) == {"sae_A"}
    assert resp.task_id == "task-solo"
