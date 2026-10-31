"""Deleting a parent row must leave its children to the database.

2026-09-12, from production. Deleting SAEs from the UI drove the API process to
113 GB of anonymous memory, with 26 transactions idle-in-transaction, until the
GPU node stopped answering SSH. `SAEManagerService.delete_sae` calls
`await db.delete(sae)`; `ExternalSAE.features`/`extraction_jobs`,
`ExtractionJob.features`/`labeling_jobs` and `Feature.activations`/
`analysis_cache` were `cascade="all, delete-orphan"` WITHOUT `passive_deletes`.
So SQLAlchemy SELECTed every extraction job, every feature and every activation
example (each carrying token arrays) in order to delete them one at a time —
although every one of those foreign keys is ON DELETE CASCADE and Postgres
removes the lot from the single parent DELETE.

The SAE-delete tests that existed mocked `delete_sae` away, so nothing ever ran
it. These run the real delete against the unit-test database and record every
SQL statement sent. A delete that reads or writes a child table is the defect,
whatever it returns — at production row counts that statement is the outage.
"""

import re
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import event, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

import src.models  # noqa: F401  registers every table, so create_all can build the FKs
from src.models.external_sae import ExternalSAE, SAEStatus
from src.models.extraction_job import ExtractionJob, ExtractionStatus
from src.models.feature import Feature
from src.models.feature_activation import FeatureActivation
from src.models.feature_analysis_cache import FeatureAnalysisCache
from src.models.labeling_job import LabelingJob
from src.services.sae_manager_service import SAEManagerService

SAE_ID = "sae_delete_test"
JOB_ID = "ext_delete_test"
LABELING_ID = "label_delete_test"
N_FEATURES = 3
EXAMPLES_PER_FEATURE = 4

CHILD_TABLES = {
    ExternalSAE: ("extraction_jobs", "labeling_jobs", "features", "feature_activations", "feature_analysis_cache"),
    ExtractionJob: ("labeling_jobs", "features", "feature_activations", "feature_analysis_cache"),
    Feature: ("feature_activations", "feature_analysis_cache"),
}


class StatementLog:
    """Every SQL statement the engine sends while the block runs."""

    def __init__(self, engine):
        self._engine = engine.sync_engine
        self.statements: list[str] = []

    def _record(self, conn, cursor, statement, parameters, context, executemany):
        self.statements.append(statement)

    def __enter__(self):
        event.listen(self._engine, "before_cursor_execute", self._record)
        return self

    def __exit__(self, *exc):
        event.remove(self._engine, "before_cursor_execute", self._record)

    def touching(self, tables) -> list[str]:
        pattern = re.compile(r"\b(?:FROM|UPDATE|INTO)\s+(?:" + "|".join(tables) + r")\b", re.IGNORECASE)
        return [s for s in self.statements if pattern.search(s)]


def _sessions(engine):
    return async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


def _feature_id(i: int) -> str:
    return f"feat_sae_{SAE_ID}_{i}"


async def _seed(engine) -> None:
    """One SAE → one extraction job → a labeling job, features, examples and cache rows."""
    async with _sessions(engine)() as s:
        s.add(ExternalSAE(id=SAE_ID, name="delete test", source="local", status=SAEStatus.READY.value))
        s.add(ExtractionJob(id=JOB_ID, external_sae_id=SAE_ID, config={}, status=ExtractionStatus.COMPLETED))
        s.add(LabelingJob(id=LABELING_ID, extraction_job_id=JOB_ID, labeling_method="manual"))
        example_id = 1
        for i in range(N_FEATURES):
            s.add(Feature(
                id=_feature_id(i), external_sae_id=SAE_ID, extraction_job_id=JOB_ID, neuron_index=i,
                name=f"feature_{i}", activation_frequency=0.1, interpretability_score=0.5, max_activation=1.0,
            ))
            for j in range(EXAMPLES_PER_FEATURE):
                s.add(FeatureActivation(
                    id=example_id, feature_id=_feature_id(i), sample_index=j,
                    max_activation=1.0, tokens=["a", "b"], activations=[0.1, 0.9],
                ))
                example_id += 1
            s.add(FeatureAnalysisCache(
                feature_id=_feature_id(i), analysis_type="logit_lens", result={},
                expires_at=datetime.now(timezone.utc) + timedelta(days=7),
            ))
        await s.commit()


async def _count(engine, model, *where) -> int:
    async with _sessions(engine)() as s:
        return (await s.execute(select(func.count()).select_from(model).where(*where))).scalar_one()


async def _all_counts(engine) -> dict:
    models = (ExternalSAE, ExtractionJob, LabelingJob, Feature, FeatureActivation, FeatureAnalysisCache)
    return {m.__tablename__: await _count(engine, m) for m in models}


@pytest.mark.asyncio
async def test_the_recorder_sees_a_statement_against_a_child_table(async_engine):
    """NEGATIVE CONTROL: a guard that cannot see a child read would pass forever."""
    await _seed(async_engine)
    with StatementLog(async_engine) as log:
        async with _sessions(async_engine)() as s:
            feature = await s.get(Feature, _feature_id(0))
            await s.run_sync(lambda _: len(feature.activations))
    assert log.touching(("feature_activations",)), log.statements


@pytest.mark.asyncio
async def test_deleting_an_sae_through_the_service_sends_nothing_to_its_children(async_engine):
    await _seed(async_engine)
    assert await _count(async_engine, FeatureActivation) == N_FEATURES * EXAMPLES_PER_FEATURE

    with StatementLog(async_engine) as log:
        async with _sessions(async_engine)() as s:
            assert await SAEManagerService.delete_sae(s, SAE_ID, delete_files=True) is True

    assert log.touching(CHILD_TABLES[ExternalSAE]) == [], (
        "deleting an SAE read or wrote its child tables from Python; at production row "
        "counts that is every activation example loaded into the API process"
    )
    assert await _all_counts(async_engine) == {
        "external_saes": 0, "extraction_jobs": 0, "labeling_jobs": 0,
        "features": 0, "feature_activations": 0, "feature_analysis_cache": 0,
    }, "the database cascade must still remove every derived row"


@pytest.mark.asyncio
async def test_a_soft_delete_keeps_the_row_and_its_features(async_engine):
    await _seed(async_engine)
    async with _sessions(async_engine)() as s:
        assert await SAEManagerService.delete_sae(s, SAE_ID, delete_files=False) is True
    async with _sessions(async_engine)() as s:
        assert (await s.get(ExternalSAE, SAE_ID)).status == SAEStatus.DELETED.value
    assert await _count(async_engine, Feature) == N_FEATURES


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model, row_id",
    [(ExternalSAE, SAE_ID), (ExtractionJob, JOB_ID), (Feature, _feature_id(0))],
    ids=["external_sae", "extraction_job", "feature"],
)
async def test_an_orm_delete_leaves_the_children_to_the_database(async_engine, model, row_id):
    """Every `session.delete(parent)` in the codebase, not only the SAE service."""
    await _seed(async_engine)

    with StatementLog(async_engine) as log:
        async with _sessions(async_engine)() as s:
            await s.delete(await s.get(model, row_id))
            await s.commit()

    assert log.touching(CHILD_TABLES[model]) == [], log.touching(CHILD_TABLES[model])

    if model is Feature:
        assert await _count(async_engine, FeatureActivation, FeatureActivation.feature_id == row_id) == 0
        assert await _count(async_engine, FeatureAnalysisCache, FeatureAnalysisCache.feature_id == row_id) == 0
        assert await _count(async_engine, Feature) == N_FEATURES - 1, "only the deleted feature goes"
        assert await _count(async_engine, FeatureActivation) == (N_FEATURES - 1) * EXAMPLES_PER_FEATURE
    else:
        assert await _count(async_engine, Feature) == 0
        assert await _count(async_engine, FeatureActivation) == 0
        assert await _count(async_engine, LabelingJob) == 0
