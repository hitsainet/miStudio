"""Deleting a parent must remove exactly the right children.

WHY THIS EXISTS

Mutation M5 flipped ALL THREE `ondelete="CASCADE"` declarations on the `Feature`
model to `RESTRICT` and 211 tests stayed green (MIS-E2E-053). Nothing anywhere
exercised a delete rule.

That blindness is why MIS-E2E-033 went unnoticed: three foreign keys were
declared on models and absent from the database, so `ondelete=` did nothing in
production. A suite that never deletes a parent cannot notice.

Deletion is the one operation in this product that cannot be undone, and it had
no coverage. It is also the mechanism behind MIS-E2E-046: one training delete
cascades to features and takes hand-edited labels, notes and the "protected"
aqua stars with it.

ON THE FIXTURE: these run against the `async_session` schema, which
`create_all()` builds from the ORM. That is only trustworthy because
`test_orm_matches_migrated_schema.py` now fails when the ORM and the migrated
database disagree. The two tests are a pair — this one asserts the behaviour,
that one asserts the schema under test is the real one. Do not delete either
without the other.

NEGATIVE CONTROL: flip any `ondelete="CASCADE"` on Feature to `"RESTRICT"` and
`test_deleting_a_training_removes_its_features` must fail. Verified 2026-08-23.
"""

import uuid

import pytest
import sqlalchemy as sa

from src.models.external_sae import ExternalSAE
from src.models.extraction_job import ExtractionJob
from src.models.feature import Feature
from src.models.feature_analysis_cache import AnalysisType, FeatureAnalysisCache
from src.models.dataset import Dataset
from src.models.model import Model
from src.models.training import Training


async def _training(session, tid="train_cascade"):
    """A training needs a model and a dataset — both NOT NULL FKs.

    `datasets.id` is a UUID column, so the id must be a real UUID; `models.id`
    is a plain string.
    """
    mid, did = f"m_{tid}", str(uuid.uuid4())
    session.add(Model(id=mid, name=f"model for {tid}",
                      architecture="test", params_count=1))
    session.add(Dataset(id=did, name=f"dataset for {tid}", source="Local"))
    await session.commit()
    session.add(Training(id=tid, status="completed", model_id=mid,
                         dataset_id=did, total_steps=100))
    await session.commit()
    return tid


async def _sae(session, sid="sae_cascade", training_id=None):
    session.add(ExternalSAE(id=sid, name="cascade SAE", source="trained",
                            training_id=training_id))
    await session.commit()
    return sid


async def _extraction(session, eid, sae_id, training_id=None):
    session.add(ExtractionJob(id=eid, external_sae_id=sae_id,
                              training_id=training_id, config={}))
    await session.commit()
    return eid


async def _feature(session, fid, extraction_id, sae_id, training_id=None):
    session.add(Feature(
        id=fid, name=fid, neuron_index=0,
        extraction_job_id=extraction_id, external_sae_id=sae_id,
        training_id=training_id,
        activation_frequency=0.5, mean_activation=1.0,
        max_activation=2.0, interpretability_score=0.4,
    ))
    await session.commit()
    return fid


async def _count(session, model, **filters):
    stmt = sa.select(sa.func.count()).select_from(model)
    for k, v in filters.items():
        stmt = stmt.where(getattr(model, k) == v)
    return (await session.execute(stmt)).scalar()


class TestCascadeFromTraining:
    async def test_deleting_a_training_removes_its_features(self, async_session):
        """The rule M5 broke with 211 tests green."""
        tid = await _training(async_session, "train_c1")
        sid = await _sae(async_session, "sae_c1", training_id=tid)
        eid = await _extraction(async_session, "extr_c1", sid)
        await _feature(async_session, "feat_c1", eid, sid, training_id=tid)

        assert await _count(async_session, Feature, id="feat_c1") == 1

        await async_session.execute(
            sa.delete(Training).where(Training.id == tid))
        await async_session.commit()

        assert await _count(async_session, Feature, id="feat_c1") == 0, (
            "features.training_id declares ON DELETE CASCADE; deleting the "
            "training must remove the feature"
        )

    async def test_a_feature_from_another_training_survives(self, async_session):
        """The negative half. Without it, a cascade that deletes EVERYTHING
        passes the test above."""
        t1 = await _training(async_session, "train_c2a")
        t2 = await _training(async_session, "train_c2b")
        sid = await _sae(async_session, "sae_c2")
        eid = await _extraction(async_session, "extr_c2", sid)
        await _feature(async_session, "feat_c2a", eid, sid, training_id=t1)
        await _feature(async_session, "feat_c2b", eid, sid, training_id=t2)

        await async_session.execute(
            sa.delete(Training).where(Training.id == t1))
        await async_session.commit()

        assert await _count(async_session, Feature, id="feat_c2a") == 0
        assert await _count(async_session, Feature, id="feat_c2b") == 1, (
            "deleting one training must not touch another's features"
        )


class TestCascadeFromExtraction:
    async def test_deleting_an_extraction_removes_its_features(self, async_session):
        sid = await _sae(async_session, "sae_c3")
        eid = await _extraction(async_session, "extr_c3", sid)
        await _feature(async_session, "feat_c3", eid, sid)

        await async_session.execute(
            sa.delete(ExtractionJob).where(ExtractionJob.id == eid))
        await async_session.commit()

        assert await _count(async_session, Feature, id="feat_c3") == 0


class TestCascadeIntoAnalysisCache:
    async def test_deleting_a_feature_removes_its_cached_analyses(self, async_session):
        """feature_analysis_cache.feature_id is ON DELETE CASCADE.

        Without this, a deleted feature leaves cache rows that the unique
        constraint then blocks a new feature from re-creating — the neighbouring
        failure mode to MIS-E2E-030.
        """
        sid = await _sae(async_session, "sae_c4")
        eid = await _extraction(async_session, "extr_c4", sid)
        fid = await _feature(async_session, "feat_c4", eid, sid)

        async_session.add(FeatureAnalysisCache(
            feature_id=fid, analysis_type=AnalysisType.CORRELATIONS,
            result={"x": 1},
            computed_at=sa.func.now(), expires_at=sa.func.now(),
        ))
        await async_session.commit()
        assert await _count(async_session, FeatureAnalysisCache, feature_id=fid) == 1

        await async_session.execute(sa.delete(Feature).where(Feature.id == fid))
        await async_session.commit()

        assert await _count(async_session, FeatureAnalysisCache, feature_id=fid) == 0


class TestSetNullRatherThanCascade:
    async def test_deleting_a_labeling_job_nulls_the_reference(self, async_session):
        """features.labeling_job_id is SET NULL, not CASCADE — deleting a
        labeling job must not delete the features it labelled.

        This FK did not exist in the database at all until migration
        d7f3a91c2e08 (MIS-E2E-033), so `ondelete="SET NULL"` was inert and the
        id was left dangling.
        """
        from src.models.labeling_job import LabelingJob

        sid = await _sae(async_session, "sae_c5")
        eid = await _extraction(async_session, "extr_c5", sid)
        fid = await _feature(async_session, "feat_c5", eid, sid)

        async_session.add(LabelingJob(id="lbl_c5", extraction_job_id=eid,
                                      labeling_method="llm", status="completed"))
        await async_session.commit()
        await async_session.execute(
            sa.update(Feature).where(Feature.id == fid)
            .values(labeling_job_id="lbl_c5"))
        await async_session.commit()

        await async_session.execute(
            sa.delete(LabelingJob).where(LabelingJob.id == "lbl_c5"))
        await async_session.commit()

        assert await _count(async_session, Feature, id=fid) == 1, (
            "SET NULL must not delete the feature"
        )
        row = (await async_session.execute(
            sa.select(Feature.labeling_job_id).where(Feature.id == fid))).scalar()
        assert row is None, "the dangling reference must be NULLed, not left"
