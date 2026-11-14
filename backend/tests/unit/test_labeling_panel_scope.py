"""Apply-mode labeling scoped to an explicit panel.

Until now `POST /api/v1/labeling` could only label a WHOLE extraction:
`label_features_for_extraction` filtered on `extraction_job_id` alone. On the
L46 extraction that is 30,712 features — at the ~16 s/feature measured on
gemma-4-12B-it, roughly five days, and past Celery's 12 h task_time_limit.

The trial path could take a panel but is architecturally forbidden to write a
label. So nothing could label a subset AND persist it. This closes that.

Mutation controls:
  C65 drop the Feature.id.in_() filter from the selection
       -> test_only_the_panel_is_selected
  C66 leave the COUNT unscoped
       -> test_total_features_is_the_panel_not_the_extraction
  C67 accept a partially-resolving panel instead of refusing
       -> test_a_panel_that_does_not_fully_resolve_is_refused
  C68 allow extra keys on LabelingPanelRequest
       -> test_a_typo_cannot_silently_become_a_full_extraction_run
"""

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from src.models.extraction_job import ExtractionJob, ExtractionStatus
from src.models.feature import Feature
from src.models.labeling_job import LabelingJob, LabelingStatus
from src.services.labeling_service import LabelingService


def _async_db(feature_count, extraction_status=ExtractionStatus.COMPLETED.value):
    """The count mock returns whatever it is told, so a test that only reads the
    returned number cannot see an unscoped query. `executed` captures the actual
    statements so the COUNT can be asserted on its SQL."""
    db = AsyncMock()
    db.executed = []
    extraction = Mock(spec=ExtractionJob)
    extraction.id = "extr_x"
    extraction.status = extraction_status
    _results = [
        Mock(scalar_one_or_none=Mock(return_value=extraction)),  # extraction lookup
        Mock(scalar_one_or_none=Mock(return_value=None)),        # no active job
        Mock(scalar_one=Mock(return_value=feature_count)),       # scoped count
    ]

    async def _execute(stmt, *a, **k):
        db.executed.append(str(stmt))
        return _results[len(db.executed) - 1]

    db.execute = AsyncMock(side_effect=_execute)
    db.add = Mock(); db.commit = AsyncMock(); db.refresh = AsyncMock()
    return db


class TestPanelScopingAtJobCreation:

    @pytest.mark.asyncio
    async def test_total_features_is_the_panel_not_the_extraction(self):
        """C66. An unscoped count leaves total_features at the extraction size,
        so progress crawls to 1% then jumps to 1.0 and every ETA is an order of
        magnitude wrong."""
        db = _async_db(feature_count=3)
        job = await LabelingService(db).start_labeling(
            "extr_x", {"labeling_method": "openai", "feature_ids": ["a", "b", "c"]})
        assert job.total_features == 3, (
            f"total_features is {job.total_features}; a panel run must not "
            f"advertise the whole extraction"
        )
        assert job.feature_ids == ["a", "b", "c"], "the panel was not persisted"

        # The count STATEMENT must carry the panel. Asserting only on the
        # returned number cannot detect an unscoped count, because the mock
        # returns 3 either way.
        count_sql = db.executed[2]
        assert "features.id IN" in count_sql, (
            f"the count query was not scoped to the panel, so total_features "
            f"would be the whole extraction:\n{count_sql}"
        )

    @pytest.mark.asyncio
    async def test_a_panel_that_does_not_fully_resolve_is_refused(self):
        """C67. A shrunken panel is not the panel that was requested: two runs
        over 'the same' panel would not be comparable and any rate computed
        from it would be wrong."""
        db = _async_db(feature_count=2)  # only 2 of 3 exist
        with pytest.raises(ValueError, match="panel resolved to 2 of 3"):
            await LabelingService(db).start_labeling(
                "extr_x", {"labeling_method": "openai", "feature_ids": ["a", "b", "zzz"]})

    @pytest.mark.asyncio
    async def test_duplicate_ids_are_collapsed_before_the_count_is_compared(self):
        """['a','a','b'] is a 2-feature panel; comparing against 3 would refuse
        a perfectly valid request."""
        db = _async_db(feature_count=2)
        job = await LabelingService(db).start_labeling(
            "extr_x", {"labeling_method": "openai", "feature_ids": ["a", "a", "b"]})
        assert job.feature_ids == ["a", "b"]
        assert job.total_features == 2

    @pytest.mark.asyncio
    async def test_a_run_with_no_panel_is_unchanged(self):
        """The whole-extraction path must behave exactly as before."""
        db = _async_db(feature_count=30712)
        job = await LabelingService(db).start_labeling(
            "extr_x", {"labeling_method": "openai"})
        assert job.total_features == 30712
        assert job.feature_ids is None


class TestPanelScopingAtSelection:

    def test_only_the_panel_is_selected(self):
        """C65. The query must carry BOTH predicates: the panel, and the
        extraction — so a foreign id cannot pull in another extraction's
        feature."""
        from sqlalchemy.orm import Session
        session = Mock(spec=Session)
        session.commit = Mock()

        job = Mock(spec=LabelingJob)
        job.id = "label_x"; job.extraction_job_id = "extr_x"
        job.status = LabelingStatus.QUEUED.value; job.updated_at = None
        job.statistics = {}; job.prompt_template_id = None
        job.max_tokens = 300; job.feature_ids = ["feat_1", "feat_2"]

        extraction = Mock(spec=ExtractionJob)
        extraction.id = "extr_x"; extraction.status = ExtractionStatus.COMPLETED.value

        filters = []
        def query_side_effect(model):
            q = Mock()
            def _filter(*a, **k):
                filters.append(a)
                return q
            q.filter.side_effect = _filter
            q.order_by.return_value = q
            if model is LabelingJob:
                q.first.return_value = job
            elif model is ExtractionJob:
                q.first.return_value = extraction
            else:
                q.first.return_value = None
                q.all.return_value = []      # empty -> raises, which is fine
            return q
        session.query = Mock(side_effect=query_side_effect)

        with pytest.raises(ValueError):
            LabelingService(session).label_features_for_extraction("label_x")

        preds = [str(c) for group in filters for c in group]

        # Assert the EXACT predicates. An earlier version scanned the joined
        # string for "IN", which was true unconditionally because the word
        # "labelINg" appears in `labeling_jobs.id = :id_1` — the test passed
        # with the panel filter removed.
        assert any(pr.startswith("features.extraction_job_id =") for pr in preds), (
            f"the extraction predicate was dropped: {preds}"
        )
        assert any(pr.startswith("features.id IN") for pr in preds), (
            f"no panel filter was applied to Feature.id; the run would label the "
            f"whole extraction. predicates: {preds}"
        )


class TestTheRequestSchemaCannotSilentlyDropThePanel:

    def test_a_typo_cannot_silently_become_a_full_extraction_run(self):
        """C68. The parent schema permits unknown keys. Without extra='forbid'
        a typo'd `featureIds` is dropped and the request labels all 30,712."""
        from pydantic import ValidationError

        from src.api.v1.endpoints.labeling import LabelingPanelRequest

        with pytest.raises(ValidationError):
            LabelingPanelRequest(
                extraction_job_id="extr_x", labeling_method="openai",
                featureIds=["a"],          # the typo
                feature_ids=["a"],
            )

    def test_the_panel_is_required_and_bounded(self):
        from pydantic import ValidationError

        from src.api.v1.endpoints.labeling import LabelingPanelRequest

        with pytest.raises(ValidationError):
            LabelingPanelRequest(extraction_job_id="e", labeling_method="openai",
                                 feature_ids=[])
        # 2000 is the Celery soft-limit bound at ~16 s/feature (~8.9 h).
        with pytest.raises(ValidationError):
            LabelingPanelRequest(extraction_job_id="e", labeling_method="openai",
                                 feature_ids=[f"f{i}" for i in range(2001)])
        ok = LabelingPanelRequest(extraction_job_id="e", labeling_method="openai",
                                  feature_ids=[f"f{i}" for i in range(2000)])
        assert len(ok.feature_ids) == 2000

    def test_a_malformed_feature_ids_column_is_treated_as_no_panel(self):
        """A non-list value must not reach in_().

        Caught by two existing strict-mock tests, whose job doubles expose a
        MagicMock for feature_ids: it is truthy, so it was handed straight to
        Feature.id.in_() and raised an opaque SQLAlchemy ArgumentError instead
        of running as an ordinary whole-extraction job.
        """
        from sqlalchemy.orm import Session
        session = Mock(spec=Session); session.commit = Mock()
        job = Mock(spec=LabelingJob)
        job.id = "label_x"; job.extraction_job_id = "extr_x"
        job.status = LabelingStatus.QUEUED.value; job.updated_at = None
        job.statistics = {}; job.prompt_template_id = None; job.max_tokens = 300
        job.feature_ids = Mock()          # not a list

        extraction = Mock(spec=ExtractionJob)
        extraction.id = "extr_x"; extraction.status = ExtractionStatus.COMPLETED.value

        def qse(model):
            q = Mock(); q.filter.return_value = q; q.order_by.return_value = q
            q.first.return_value = job if model is LabelingJob else (
                extraction if model is ExtractionJob else None)
            q.all.return_value = []
            return q
        session.query = Mock(side_effect=qse)

        # Must reach the ordinary "no features" refusal, not an ArgumentError.
        with pytest.raises(ValueError, match="No features found"):
            LabelingService(session).label_features_for_extraction("label_x")
