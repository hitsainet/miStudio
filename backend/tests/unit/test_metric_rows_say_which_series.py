"""A metric row says which series it belongs to (review R1-C, 2026-09-15).

One logged step writes an aggregate row (``layer_idx`` NULL), one row per SAE, and
held-out rows (``-1 - layer``); runs before 2026-09-15 also carry spliced-CE rows
(``-1000 - layer``) whose ``loss`` is a cross-entropy. ``GET /trainings/{id}/metrics``
served none of that: ``TrainingMetricResponse`` had no ``layer_idx``. The training
card asked for the last 20 rows and deduplicated by step, so its FVU chart could
show one layer's value, or a held-out value, as the run's — and no client could
have done better, because the rows were indistinguishable.

NEGATIVE CONTROL (applied alone, this file run, restored, sha256 verified):
  L11 the field renamed away from `layer_idx`   RED  every test here
"""

from datetime import datetime, timezone

import pytest
from sqlalchemy import select  # noqa: F401 - imported for parity with the endpoint tests

from src.models.model import Model
from src.models.training import Training
from src.models.training_metric import TrainingMetric


def test_the_metrics_response_serves_layer_idx_in_the_live_app():
    from src.main import app

    schema = app.openapi()["components"]["schemas"]["TrainingMetricResponse"]
    assert "layer_idx" in schema["properties"]


@pytest.mark.asyncio
async def test_the_metrics_route_serves_each_rows_series(client, async_session):
    from src.models.model import ModelStatus, QuantizationFormat

    async_session.add(Model(
        id="m_series", name="tiny", architecture="llama", params_count=1_000,
        quantization=QuantizationFormat.FP16, status=ModelStatus.READY,
    ))
    async_session.add(Training(
        id="train_series", model_id="m_series", dataset_id="ds_x", dataset_ids=["ds_x"],
        status="running", progress=50.0, current_step=100, total_steps=200,
        hyperparameters={"hidden_dim": 8, "latent_dim": 16},
    ))
    await async_session.flush()
    for layer_idx, fvu_centred in ((None, 0.31), (11, 0.9), (-12, 0.8), (-1011, None)):
        async_session.add(TrainingMetric(
            training_id="train_series", step=100, loss=0.1, layer_idx=layer_idx,
            fvu_centred=fvu_centred, timestamp=datetime.now(timezone.utc),
        ))
    await async_session.commit()

    response = await client.get("/api/v1/trainings/train_series/metrics?limit=10")

    assert response.status_code == 200, response.text
    served = {row["layer_idx"]: row["fvu_centred"] for row in response.json()["data"]}
    assert served == {None: 0.31, 11: 0.9, -12: 0.8, -1011: None}
