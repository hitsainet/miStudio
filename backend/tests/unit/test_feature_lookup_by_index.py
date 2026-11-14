"""Looking a feature up by (SAE or training, neuron index) survives re-extraction.

WHY THIS EXISTS

The steering panel's "View Feature Details" opens a feature that was added by
index, so it first asks for that index's feature id:

  GET /api/v1/trainings/{training_id}/features/by-index/{idx}   (trained SAE)
  GET /api/v1/saes/{sae_id}/features/by-index/{idx}             (downloaded SAE)

Feature rows are unique per (training, EXTRACTION, neuron) — one SAE extracted
twice has two rows for every index. Both lookups ended in
`scalar_one_or_none()`, which RAISES on two rows, so the moment an SAE was
re-extracted the lookup returned a 500 and the panel told the user the feature
"has not been extracted yet". The downloaded-SAE route had never been reached
from the steering selector (the menu item was disabled for downloaded SAEs), so
nothing exercised it at all.

Both now answer with the row from the NEWEST extraction, which is the one the
feature browser and the extraction list put first.

NEGATIVE CONTROLS (2026-10-18, each restored by sha256):
  N1  restore the unordered `scalar_one_or_none()` for the SAE lookup     -> 1 red
  N2  restore it in FeatureService.get_feature_by_index                   -> 1 red
  N3  order the newest-extraction helper ascending                        -> 2 red
  N4  replace the SAE route's @router.get with a no-op decorator          -> 1 red
"""

import datetime as dt
import uuid

import pytest

from src.api.v1.endpoints.features import get_feature_id_by_sae_and_index
from src.models.dataset import Dataset
from src.models.external_sae import ExternalSAE
from src.models.extraction_job import ExtractionJob
from src.models.feature import Feature
from src.models.model import Model
from src.models.training import Training
from src.services.feature_service import FeatureService

T0 = dt.datetime(2026, 10, 1, tzinfo=dt.timezone.utc)


async def _training(session, tid):
    mid, did = f"m_{tid}", str(uuid.uuid4())
    session.add(Model(id=mid, name=f"model {tid}", architecture="test", params_count=1))
    session.add(Dataset(id=did, name=f"dataset {tid}", source="Local"))
    await session.commit()
    session.add(Training(id=tid, status="completed", model_id=mid,
                         dataset_id=did, total_steps=10))
    await session.commit()
    return tid


async def _sae(session, sid, training_id=None):
    session.add(ExternalSAE(id=sid, name=sid, source="trained" if training_id else "huggingface",
                            training_id=training_id))
    await session.commit()
    return sid


async def _extraction(session, eid, sae_id, created_at, training_id=None):
    session.add(ExtractionJob(id=eid, external_sae_id=sae_id, training_id=training_id,
                              config={}, created_at=created_at))
    await session.commit()
    return eid


async def _feature(session, fid, eid, sae_id, neuron_index, training_id=None):
    session.add(Feature(
        id=fid, name=fid, neuron_index=neuron_index,
        extraction_job_id=eid, external_sae_id=sae_id, training_id=training_id,
        activation_frequency=0.1, mean_activation=1.0, max_activation=2.0,
        interpretability_score=0.5,
    ))
    await session.commit()


class TestDownloadedSAELookup:
    async def test_single_extraction_returns_its_feature(self, async_session):
        sid = await _sae(async_session, "sae_dl_one")
        eid = await _extraction(async_session, "extr_dl_one", sid, T0)
        await _feature(async_session, "feat_dl_one_42", eid, sid, 42)

        got = await get_feature_id_by_sae_and_index(sid, 42, db=async_session)
        assert got == {"feature_id": "feat_dl_one_42"}

    async def test_unextracted_index_is_null_not_an_error(self, async_session):
        sid = await _sae(async_session, "sae_dl_none")
        got = await get_feature_id_by_sae_and_index(sid, 7, db=async_session)
        assert got == {"feature_id": None}

    async def test_reextracted_sae_returns_the_newest_extractions_feature(self, async_session):
        """Two rows for one index used to raise MultipleResultsFound -> 500."""
        sid = await _sae(async_session, "sae_dl_two")
        old = await _extraction(async_session, "extr_dl_old", sid, T0)
        new = await _extraction(async_session, "extr_dl_new", sid, T0 + dt.timedelta(days=1))
        # Insert the newer one FIRST so insertion order cannot pass for "newest".
        await _feature(async_session, "feat_dl_new_42", new, sid, 42)
        await _feature(async_session, "feat_dl_old_42", old, sid, 42)

        got = await get_feature_id_by_sae_and_index(sid, 42, db=async_session)
        assert got == {"feature_id": "feat_dl_new_42"}

    async def test_another_saes_feature_is_not_returned(self, async_session):
        a = await _sae(async_session, "sae_dl_a")
        b = await _sae(async_session, "sae_dl_b")
        ea = await _extraction(async_session, "extr_dl_a", a, T0)
        await _feature(async_session, "feat_a_42", ea, a, 42)
        got = await get_feature_id_by_sae_and_index(b, 42, db=async_session)
        assert got == {"feature_id": None}


class TestTrainedSAELookup:
    async def test_reextracted_training_returns_the_newest_extractions_feature(self, async_session):
        tid = await _training(async_session, "train_lookup_two")
        sid = await _sae(async_session, "sae_tr_two", training_id=tid)
        old = await _extraction(async_session, "extr_tr_old", sid, T0, training_id=tid)
        new = await _extraction(async_session, "extr_tr_new", sid, T0 + dt.timedelta(days=1),
                                training_id=tid)
        await _feature(async_session, "feat_tr_new_42", new, sid, 42, training_id=tid)
        await _feature(async_session, "feat_tr_old_42", old, sid, 42, training_id=tid)

        got = await FeatureService(async_session).get_feature_by_index(tid, 42)
        assert got == "feat_tr_new_42"


class TestRoutesAreServed:
    """The routes are what the steering panel calls; read the app's own OpenAPI
    paths (app.routes is not a route list under FastAPI's included routers)."""

    @pytest.mark.parametrize("path", [
        "/api/v1/saes/{sae_id}/features/by-index/{feature_idx}",
        "/api/v1/trainings/{training_id}/features/by-index/{feature_idx}",
    ])
    def test_route_is_in_the_live_app(self, path):
        from src.main import app
        assert "get" in app.openapi()["paths"].get(path, {}), path
