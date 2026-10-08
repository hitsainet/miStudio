"""The model ROW's quantization travels from the steering endpoints to the load.

⚠ THE LINK THAT WAS MISSING. Each steering endpoint read the Model row, swapped its id for the repo
id and handed the worker only `model_id` and `model_path`, so the worker could not know a row was
Q4 or Q8 and loaded every model 16-bit. `test_steering_gpu_placement.py` pins what the LOAD does
with a quantization and `test_steering_split_keeps_room_for_its_saes.py` pins that the Celery-side
wrapper hands it to the load; this file pins the two hops before that.
"""

from __future__ import annotations

import ast
import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from src.api.v1.endpoints import steering as endpoints
from src.api.v1.endpoints.steering import steering_model_quantization, submit_async_combined_steering
from src.models.model import QuantizationFormat
from src.schemas.steering import CombinedSteeringRequest, SelectedFeature
from src.workers import steering_tasks


class TestTheRowsQuantizationIsRead:
    @pytest.mark.parametrize("value", ["FP32", "FP16", "Q8", "Q4"])
    def test_each_served_format_is_passed_on(self, value):
        row = SimpleNamespace(id="m_1", quantization=QuantizationFormat(value))
        assert steering_model_quantization(row) == value

    def test_no_row_is_none_not_a_guess(self):
        assert steering_model_quantization(None) is None

    def test_q2_is_refused_with_the_reason(self):
        with pytest.raises(HTTPException) as exc:
            steering_model_quantization(SimpleNamespace(id="m_q2", quantization=QuantizationFormat.Q2))
        assert exc.value.status_code == 422 and "Q2" in exc.value.detail


@pytest.fixture
def _bypass_gates():
    with patch("src.api.v1.endpoints.steering._rate_limiter") as rl, patch(
        "src.api.v1.endpoints.steering._ensure_steering_worker_running",
        new=AsyncMock(return_value=(True, 123)),
    ):
        rl.is_allowed.return_value = True
        yield


def _sae():
    s = MagicMock()
    s.status = "ready"
    s.local_path = "saes/sae_A"
    s.n_features = 100
    s.layer = 13
    s.d_model = 16
    s.architecture = "jumprelu"
    s.model_id = "m1"
    s.model_name = "m1"
    return s


def _http_request():
    r = MagicMock()
    r.client = MagicMock(host="127.0.0.1")
    r.headers = {}
    return r


async def _submit(row):
    req = CombinedSteeringRequest(
        sae_id="sae_A", model_id="m1", prompt="hi",
        selected_features=[SelectedFeature(feature_idx=1, layer=13, sae_id="sae_A", strength=50)],
    )

    async def get_sae(db, sid):
        return _sae()

    with patch("src.api.v1.endpoints.steering.SAEManagerService.get_sae", new=get_sae), patch(
        "src.api.v1.endpoints.steering.settings"
    ) as st, patch(
        "src.api.v1.endpoints.steering.ModelService.get_model", new=AsyncMock(return_value=row)
    ), patch("src.workers.steering_tasks.steering_combined_task") as task:
        st.resolve_data_path.return_value = MagicMock(exists=MagicMock(return_value=True))
        task.apply_async.return_value = MagicMock(id="task-1")
        await submit_async_combined_steering(req, _http_request(), db=MagicMock())
    return task


@pytest.mark.asyncio
async def test_the_endpoint_sends_the_rows_quantization(_bypass_gates):
    row = SimpleNamespace(
        id="m1", quantization=QuantizationFormat.Q4, file_path="models/m1", repo_id="org/m1", name="m1",
    )
    task = await _submit(row)
    assert task.apply_async.call_count == 1
    kwargs = task.apply_async.call_args.kwargs["kwargs"]
    assert kwargs["model_quantization"] == "Q4"
    assert kwargs["model_id"] == "org/m1"


@pytest.mark.asyncio
async def test_a_q2_row_is_refused_before_anything_is_queued(_bypass_gates):
    row = SimpleNamespace(
        id="m1", quantization=QuantizationFormat.Q2, file_path="models/m1", repo_id="org/m1", name="m1",
    )
    with pytest.raises(HTTPException) as exc:
        await _submit(row)
    assert exc.value.status_code == 422


def _calls(function, name):
    tree = ast.parse(inspect.getsource(function).lstrip())
    return [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (getattr(node.func, "id", None) == name or getattr(node.func, "attr", None) == name)
    ]


@pytest.mark.parametrize("submit", [
    endpoints.submit_async_steering_comparison,
    endpoints.submit_async_strength_sweep,
    endpoints.submit_async_combined_steering,
])
def test_every_steering_endpoint_dispatches_the_quantization(submit):
    """Compare and sweep share the combined endpoint's shape; one endpoint missing the key is the
    whole defect again, for that kind of steering."""
    assert _calls(submit, "steering_model_quantization"), "the row's quantization is never read"
    [dispatch] = _calls(submit, "apply_async")
    payload = next(k.value for k in dispatch.keywords if k.arg == "kwargs")
    keys = {getattr(key, "value", None): value for key, value in zip(payload.keys, payload.values)}
    assert "model_quantization" in keys
    assert isinstance(keys["model_quantization"], ast.Name)
    assert keys["model_quantization"].id == "model_quantization"


@pytest.mark.parametrize("task, wrapper", [
    (steering_tasks.steering_compare_task, "generate_comparison_sync"),
    (steering_tasks.steering_sweep_task, "generate_strength_sweep_sync"),
    (steering_tasks.steering_combined_task, "generate_combined_sync"),
])
def test_every_task_hands_it_to_the_service(task, wrapper):
    function = getattr(task, "run", task)
    function = getattr(function, "__wrapped__", function)
    assert "model_quantization" in inspect.signature(function).parameters
    [call] = _calls(function, wrapper)
    passed = {k.arg: k.value for k in call.keywords}
    assert isinstance(passed.get("model_quantization"), ast.Name)
    assert passed["model_quantization"].id == "model_quantization"
