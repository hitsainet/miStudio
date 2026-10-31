"""An imported SAE carries ITS OWN layer's final metrics, not the run's average.

THE DEFECT. Importing SAEs from a multi-layer training copied
`training.current_loss` and `training.current_l0_sparsity` onto every one.
Those columns are AVERAGES across all the SAEs the run trained, so L11, L12 and
L13 of `train_b14d263e` carried bit-identical `final_l0_sparsity`
(0.00300555…) — a physically impossible result that hid the very per-layer
difference a reader would consult it for. The truth was in `training_metrics`
all along, keyed by `layer_idx`:

    layer   L0 mean   FVU centred
     11      51.36      0.2363
     12      49.46      0.2924
     13      46.91      0.2959

Held-out rows are stored at `layer_idx = -1 - layer`, so selecting the in-sample
series means matching `layer_idx == layer` exactly — a held-out row must never
leak into the in-sample figure.

MUTATION CONTROLS (each alone; suite must go red):
  D1  drop the `layer_idx == layer` filter
        -> test_each_layer_gets_its_own_metrics
  D2  drop the `order_by(step.desc())`
        -> test_the_latest_step_wins
  D3  make the import path use run_metrics_payload unconditionally
        -> test_the_import_path_reads_the_layer_row
  D4  match the held-out series too (e.g. `abs(layer_idx)` style)
        -> test_held_out_rows_never_leak_in
"""

import ast
import inspect
import textwrap
from datetime import datetime, timezone

import pytest

from src.models.model import Model
from src.models.training import Training
from src.models.training_metric import TrainingMetric
from src.services import sae_manager_service
from src.services.sae_manager_service import (
    final_layer_metrics_row,
    layer_metrics_payload,
    run_metrics_payload,
)

TID = "train_perlayer"


async def _seed(session):
    from src.models.model import ModelStatus, QuantizationFormat

    session.add(Model(
        id="m_perlayer", name="tiny", architecture="llama", params_count=1_000,
        quantization=QuantizationFormat.FP16, status=ModelStatus.READY,
    ))
    session.add(Training(
        id=TID, model_id="m_perlayer", dataset_id="ds_x", dataset_ids=["ds_x"],
        status="completed", progress=100.0, current_step=200, total_steps=200,
        # The run-level averages the old code copied onto every layer.
        current_loss=0.35, current_l0_sparsity=0.0030,
        hyperparameters={"hidden_dim": 8, "latent_dim": 16},
    ))
    await session.flush()
    now = datetime.now(timezone.utc)
    rows = [
        # (layer_idx, step, l0_sparsity, fvu_centred)
        (None, 200, 0.0030, 0.270),   # aggregate across every SAE
        (11, 100, 0.0040, 0.400),     # layer 11, an EARLIER step
        (11, 200, 0.0031, 0.236),     # layer 11, final
        (13, 200, 0.0029, 0.296),     # layer 13, final
        # Layer 11 HELD-OUT (-1 - 11), at a LATER step than the in-sample
        # final. At the SAME step, a query that wrongly matched both series
        # would tie on the ordering and could return the right row by luck —
        # which is exactly how control D4 first survived.
        (-12, 210, 0.0031, 0.990),
    ]
    for layer_idx, step, l0, fvu_c in rows:
        session.add(TrainingMetric(
            training_id=TID, step=step, loss=0.3, l0_sparsity=l0,
            fvu_centred=fvu_c, layer_idx=layer_idx,
            hook_type=None if layer_idx is None else "residual",
            timestamp=now,
        ))
    await session.commit()


@pytest.mark.asyncio
async def test_each_layer_gets_its_own_metrics(async_session):
    await _seed(async_session)

    l11 = await final_layer_metrics_row(async_session, TID, 11, "residual")
    l13 = await final_layer_metrics_row(async_session, TID, 13, "residual")

    assert (l11.fvu_centred, l13.fvu_centred) == (0.236, 0.296)
    # The regression itself: two layers of one run must not report one number.
    assert layer_metrics_payload(l11) != layer_metrics_payload(l13)


@pytest.mark.asyncio
async def test_the_latest_step_wins(async_session):
    await _seed(async_session)

    l11 = await final_layer_metrics_row(async_session, TID, 11, "residual")

    assert l11.step == 200 and l11.fvu_centred == 0.236


@pytest.mark.asyncio
async def test_held_out_rows_never_leak_in(async_session):
    await _seed(async_session)

    l11 = await final_layer_metrics_row(async_session, TID, 11, "residual")

    assert l11.layer_idx == 11
    assert l11.fvu_centred != 0.990, "a held-out row was reported as in-sample"


@pytest.mark.asyncio
async def test_a_training_without_layer_rows_falls_back_honestly(async_session):
    await _seed(async_session)

    missing = await final_layer_metrics_row(async_session, TID, 12, "residual")
    training = await async_session.get(Training, TID)

    assert missing is None
    payload = run_metrics_payload(training)
    # A run-level average may only appear if it SAYS it is one.
    assert payload["final_metrics_scope"] == "run"
    assert payload["final_l0_sparsity"] == 0.0030


def test_the_layer_payload_is_labelled_per_layer():
    class Row:
        loss, l0_sparsity, l0_mean, fvu, fvu_centred = 0.3, 0.0031, 51.4, 0.21, 0.236
        dead_neurons, step = 0, 200

    payload = layer_metrics_payload(Row())
    assert payload["final_metrics_scope"] == "layer"
    assert payload["final_l0_mean"] == 51.4 and payload["final_fvu_centred"] == 0.236


def test_the_import_path_reads_the_layer_row():
    """REACHABILITY: the import must CALL final_layer_metrics_row.

    Walks the AST for a Call node rather than scraping text, which would match
    this very module's docstrings and pass for the wrong reason.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(sae_manager_service.SAEManagerService)))
    called = {
        n.func.id for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert "final_layer_metrics_row" in called
    assert "layer_metrics_payload" in called
