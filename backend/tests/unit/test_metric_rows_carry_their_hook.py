"""The metrics endpoint serves each row's hook type, and can serve the aggregate alone (review R1-A, A5).

A training over several hook types writes, per logged step, one aggregate row, one row per
SAE and one held-out row per SAE, for EVERY hook. Two SAEs on one layer were two rows with
the same layer_idx and nothing to tell them apart, and a window of N raw rows held
N / (1 + 2 x layers x hooks) steps: the training card asked for 400 rows and so drew 12
steps of a 5-layer, 3-hook run. aggregate_only asks Postgres for the aggregate rows alone.

MUTATION CONTROLS (one line broken at a time, this file run, bytes restored, sha256 and the
working tree verified clean; the full table is in the A5 record):
  R1 TrainingMetricResponse without hook_type          -> both tests
  R2 get_metrics ignores aggregate_only                 -> the aggregate test
  R3 the endpoint does not pass aggregate_only on       -> the aggregate test
"""

from datetime import datetime, timezone

import pytest

from src.models.model import Model
from src.models.training import Training
from src.models.training_metric import TrainingMetric


def test_the_metrics_response_serves_hook_type_in_the_live_app():
    from src.main import app

    schema = app.openapi()["components"]["schemas"]["TrainingMetricResponse"]
    assert "hook_type" in schema["properties"] and "layer_idx" in schema["properties"]
    route = app.openapi()["paths"]["/api/v1/trainings/{training_id}/metrics"]["get"]
    assert "aggregate_only" in [p["name"] for p in route["parameters"]]


ROWS = [
    # (step, layer_idx, hook_type, fvu_centred)
    (100, None, None, 0.31),
    (100, 11, "residual", 0.90),
    (100, 11, "mlp", 0.70),
    (100, -12, "residual", 0.80),
    (100, -12, "mlp", 0.60),
    (200, None, None, 0.29),
    (200, 11, "residual", 0.85),
    (200, 11, "mlp", 0.65),
    (200, -12, "residual", 0.75),
    (200, -12, "mlp", 0.55),
]


async def _seed(async_session):
    from src.models.model import ModelStatus, QuantizationFormat

    async_session.add(Model(
        id="m_hooks", name="tiny", architecture="llama", params_count=1_000,
        quantization=QuantizationFormat.FP16, status=ModelStatus.READY,
    ))
    async_session.add(Training(
        id="train_hooks", model_id="m_hooks", dataset_id="ds_x", dataset_ids=["ds_x"],
        status="running", progress=50.0, current_step=200, total_steps=400,
        hyperparameters=dict(hidden_dim=8, latent_dim=16, training_layers=[11], hook_types=["residual", "mlp"]),
    ))
    await async_session.flush()
    for step, layer_idx, hook_type, fvu_centred in ROWS:
        async_session.add(TrainingMetric(
            training_id="train_hooks", step=step, loss=0.1, layer_idx=layer_idx, hook_type=hook_type,
            fvu_centred=fvu_centred, timestamp=datetime.now(timezone.utc),
        ))
    await async_session.commit()


def _served(response):
    assert response.status_code == 200, response.text
    return sorted(
        ((r["step"], r["layer_idx"], r["hook_type"], r["fvu_centred"]) for r in response.json()["data"]),
        key=repr,
    )


@pytest.mark.asyncio
async def test_each_row_is_served_with_its_hook_type(client, async_session):
    await _seed(async_session)
    served = _served(await client.get("/api/v1/trainings/train_hooks/metrics?limit=50"))
    assert served == sorted(ROWS, key=repr)


@pytest.mark.asyncio
async def test_aggregate_only_serves_one_row_per_step_however_many_saes_the_run_has(client, async_session):
    await _seed(async_session)

    raw = _served(await client.get("/api/v1/trainings/train_hooks/metrics?limit=2"))
    assert any(layer is not None for _, layer, _, _ in raw), "precondition: a raw window mixes in per-SAE rows"

    served = _served(await client.get("/api/v1/trainings/train_hooks/metrics?limit=2&aggregate_only=true"))
    assert served == [(100, None, None, 0.31), (200, None, None, 0.29)]
