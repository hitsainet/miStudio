"""The analysis cache must survive a recompute.

WHY THIS EXISTS

`(feature_id, analysis_type)` is UNIQUE in the database. `_get_cached_analysis`
filters on ``computed_at >= now - CACHE_EXPIRY_DAYS``, so once a row passes the
expiry the READ stops seeing it while the ROW IS STILL THERE — and nothing
prunes expired rows. The old `_cache_analysis` did a plain INSERT, so the
recompute that the expiry was designed to trigger raised a unique violation.

The failure was PERMANENT and user-visible: Logit Lens, Correlations and
Ablation all returned 500 for that feature from the expiry onward, and every
retry re-ran the same path. Observed in production on
``feat_sae_20260726_174056_d1a4_00000`` — the tabs 500'd while Examples and
Token Analysis, which do not touch this cache, worked.

It was invisible to the suite because the unique constraint lived only in
migration ``76918d8aa763`` and not on the ORM model, so `create_all()` built a
test schema WITHOUT it. That is fixed too (`FeatureAnalysisCache.__table_args__`),
which is what lets this test see the constraint at all.

NEGATIVE CONTROL: replace the `pg_insert(...).on_conflict_do_update(...)` in
`_cache_analysis` with the original `self.db.add(FeatureAnalysisCache(...))` and
`test_recompute_replaces_an_existing_entry` must fail with an IntegrityError.
Verified 2026-08-23.
"""

import pytest
from sqlalchemy import select, func

from src.models.external_sae import ExternalSAE
from src.models.extraction_job import ExtractionJob
from src.models.feature import Feature
from src.models.feature_analysis_cache import AnalysisType, FeatureAnalysisCache
from src.services.analysis_service import AnalysisService


async def _make_feature(session, feature_id: str = "feat_cache_test") -> Feature:
    """Create the FK parents this feature needs, then the feature.

    Modelled on the real shape: the SAE row is `source="trained"` and carries
    the `training_id`, and the feature points at the SAE, not the training.
    """
    sae_id = f"sae_{feature_id}"
    extraction_id = f"extr_{feature_id}"
    session.add(ExternalSAE(id=sae_id, name=f"SAE for {feature_id}", source="trained"))
    session.add(ExtractionJob(id=extraction_id, external_sae_id=sae_id, config={}))
    await session.commit()

    feature = Feature(
        id=feature_id,
        # An EXTERNAL-SAE feature, which is what every feature in this product
        # actually is: training runs export to community_format/, that SAE is
        # imported into the registry, and extraction runs against the registry
        # SAE. `training_id` is therefore NULL and the provenance lives on
        # `ExternalSAE.training_id`, one hop away.
        extraction_job_id=extraction_id,
        external_sae_id=sae_id,
        neuron_index=0,
        name="cache_test_feature",
        activation_frequency=0.5,
        mean_activation=1.0,
        max_activation=2.0,
        interpretability_score=0.4,
    )
    session.add(feature)
    await session.commit()
    return feature


class TestTheCacheSurvivesARecompute:
    async def test_the_unique_constraint_is_present_in_the_schema_the_tests_build(
        self, async_session
    ):
        """Without this the rest of the class proves nothing.

        The constraint used to exist only in the migration, so `create_all()`
        gave the suite a table that accepted duplicates and no test could ever
        observe the production behaviour.
        """
        constraints = {
            c.name
            for c in FeatureAnalysisCache.__table__.constraints
            if type(c).__name__ == "UniqueConstraint"
        }
        assert "uq_feature_analysis_cache_feature_type" in constraints

    async def test_recompute_replaces_an_existing_entry(self, async_session):
        """The second write for the same (feature, type) must not raise."""
        await _make_feature(async_session, "feat_cache_recompute")
        service = AnalysisService(async_session)

        await service._cache_analysis(
            "feat_cache_recompute", AnalysisType.CORRELATIONS, {"generation": 1}
        )
        # This is the call that used to raise UniqueViolationError, and with it
        # every subsequent request for this feature's analysis.
        await service._cache_analysis(
            "feat_cache_recompute", AnalysisType.CORRELATIONS, {"generation": 2}
        )

        count = (
            await async_session.execute(
                select(func.count())
                .select_from(FeatureAnalysisCache)
                .where(FeatureAnalysisCache.feature_id == "feat_cache_recompute")
            )
        ).scalar()
        assert count == 1, "the recompute must replace the row, not add one"

        row = (
            await async_session.execute(
                select(FeatureAnalysisCache).where(
                    FeatureAnalysisCache.feature_id == "feat_cache_recompute"
                )
            )
        ).scalar_one()
        assert row.result == {"generation": 2}, (
            "a recompute must REFRESH the cached value; keeping the stale one "
            "would make the expiry meaningless"
        )

    async def test_the_refresh_moves_the_expiry_window(self, async_session):
        """A refreshed entry must be servable again, not instantly stale."""
        await _make_feature(async_session, "feat_cache_expiry")
        service = AnalysisService(async_session)

        await service._cache_analysis(
            "feat_cache_expiry", AnalysisType.LOGIT_LENS, {"generation": 1}
        )
        first = (
            await async_session.execute(
                select(FeatureAnalysisCache).where(
                    FeatureAnalysisCache.feature_id == "feat_cache_expiry"
                )
            )
        ).scalar_one()
        first_expires = first.expires_at

        await service._cache_analysis(
            "feat_cache_expiry", AnalysisType.LOGIT_LENS, {"generation": 2}
        )
        await async_session.refresh(first)

        assert first.expires_at >= first_expires
        cached = await service._get_cached_analysis(
            "feat_cache_expiry", AnalysisType.LOGIT_LENS
        )
        assert cached is not None, "the refreshed entry must be readable again"
        assert cached.result == {"generation": 2}

    async def test_each_analysis_type_keeps_its_own_entry(self, async_session):
        """The upsert must not collapse different analyses onto one row."""
        await _make_feature(async_session, "feat_cache_types")
        service = AnalysisService(async_session)

        for analysis_type in (
            AnalysisType.LOGIT_LENS,
            AnalysisType.CORRELATIONS,
            AnalysisType.ABLATION,
        ):
            await service._cache_analysis(
                "feat_cache_types", analysis_type, {"which": analysis_type.value}
            )

        count = (
            await async_session.execute(
                select(func.count())
                .select_from(FeatureAnalysisCache)
                .where(FeatureAnalysisCache.feature_id == "feat_cache_types")
            )
        ).scalar()
        assert count == 3
