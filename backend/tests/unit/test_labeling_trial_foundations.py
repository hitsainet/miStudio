"""Foundations for labeling trial mode: the two latent defects it depends on.

Both defects here were reachable before trial mode existed, and both get MORE
dangerous as the working set shrinks — which is exactly what a 30-feature
prompt-template panel is.

Mutation controls (each must turn a test in this file red):

  C1  drop the `uuid.uuid4().hex[:6]` suffix from the job id
      -> test_two_starts_in_the_same_second_get_distinct_ids
  C2  delete the `if total_features == 0: raise` junk-filter guard
      -> test_a_fully_filtered_run_raises_instead_of_completing_at_zero
  C3  make `mode`/`feature_ids` non-nullable, or drop them from the model
      -> test_trial_columns_are_additive_and_default_to_apply
"""

from datetime import datetime, timezone
from unittest.mock import Mock, patch

import pytest

from src.models.extraction_job import ExtractionJob, ExtractionStatus
from src.models.feature import Feature
from src.models.labeling_job import LabelingJob, LabelingMode, LabelingStatus
from src.models.labeling_trial_run import LabelingTrialRun, _ltr_id
from src.services.labeling_service import LabelingService


class TestJobIdUniqueness:
    """C1 — the job id was second-resolution and collided on the primary key."""

    @pytest.mark.asyncio
    async def test_two_starts_in_the_same_second_get_distinct_ids(self):
        """Two jobs started inside one second must not share a primary key.

        This drives the REAL start_labeling. An earlier version of this test
        rebuilt the id inline, which only proved the test's own arithmetic — it
        stayed green with the fix reverted, and the source-scrape below was
        carrying the whole guard. Source scrapes fail open, so the behaviour has
        to be exercised.
        """
        from unittest.mock import AsyncMock

        frozen = datetime(2026, 8, 29, 12, 0, 0, tzinfo=timezone.utc)

        def _session():
            db = AsyncMock()
            extraction = Mock(spec=ExtractionJob)
            extraction.id = "extr_x"
            extraction.status = ExtractionStatus.COMPLETED.value

            results = [
                Mock(scalar_one_or_none=Mock(return_value=extraction)),  # extraction
                Mock(scalar_one_or_none=Mock(return_value=None)),        # no active job
                Mock(scalar_one=Mock(return_value=42)),                  # feature count
            ]
            db.execute = AsyncMock(side_effect=results)
            db.add = Mock()
            db.commit = AsyncMock()
            db.refresh = AsyncMock()
            return db

        ids = []
        for _ in range(2):
            db = _session()
            service = LabelingService(db)
            with patch("src.services.labeling_service.datetime") as dt:
                dt.now.return_value = frozen
                job = await service.start_labeling(
                    "extr_x", {"labeling_method": "openai"}
                )
            ids.append(job.id)

        assert ids[0].startswith("label_extr_x_20260829_120000"), ids[0]
        assert ids[0] != ids[1], (
            f"two labeling jobs created in the same second share a primary key: "
            f"{ids[0]} == {ids[1]}"
        )

    def test_the_id_construction_carries_a_random_suffix(self):
        """The suffix must be in the source, not merely in this test's arithmetic."""
        import inspect
        src = inspect.getsource(LabelingService.start_labeling)
        assert "uuid.uuid4().hex[:6]" in src, (
            "start_labeling builds a second-resolution id with no random suffix; "
            "two starts in the same second will collide on the PK"
        )


class TestJunkFilterCannotSilentlySucceed:
    """C2 — a filter that removed everything reported COMPLETED with 0 labels."""

    def _session_with_features(self, labeling_job, extraction_job, features):
        from sqlalchemy.orm import Session
        session = Mock(spec=Session)
        session.commit = Mock()

        def query_side_effect(model):
            q = Mock()
            q.filter.return_value = q
            q.order_by.return_value = q
            if model is LabelingJob:
                q.first.return_value = labeling_job
            elif model is ExtractionJob:
                q.first.return_value = extraction_job
            elif model is Feature:
                q.first.return_value = features[0] if features else None
                q.all.return_value = features
            else:
                q.first.return_value = None
                q.all.return_value = []
            return q

        session.query = Mock(side_effect=query_side_effect)
        return session

    def test_a_fully_filtered_run_raises_instead_of_completing_at_zero(self):
        """When the junk filter drops every feature, the job must FAIL, not pass.

        With total_features == 0 every `range(0, total_features, ...)` label loop
        is a no-op, `labels` stays empty, and the terminal write records
        COMPLETED / progress=1.0 / features_labeled=0. That is a finished-looking
        run that labeled nothing.
        """
        labeling_job = Mock(spec=LabelingJob)
        labeling_job.id = "label_test"
        labeling_job.extraction_job_id = "extr_test"
        labeling_job.status = LabelingStatus.QUEUED.value
        labeling_job.updated_at = None
        labeling_job.statistics = {}
        labeling_job.prompt_template_id = None
        labeling_job.max_tokens = 300

        extraction_job = Mock(spec=ExtractionJob)
        extraction_job.id = "extr_test"
        extraction_job.status = ExtractionStatus.COMPLETED.value

        feature = Mock(spec=Feature)
        feature.id = "feat_1"
        feature.neuron_index = 1

        session = self._session_with_features(labeling_job, extraction_job, [feature])
        service = LabelingService(session)

        # Retrieval yields examples; the junk filter then removes everything.
        with patch.object(
            service, "_retrieve_top_examples_batch_sync",
            return_value={"feat_1": [{"prime_token": ".", "max_activation": 1.0}]},
        ), patch("src.utils.token_filter.get_feature_filter") as get_filter:
            get_filter.return_value.filter_features_from_examples.return_value = (
                [], [], [],
                {"total_features": 1, "features_to_label": 0,
                 "features_skipped": 1, "skip_percentage": 100.0},
            )
            with pytest.raises(ValueError, match="nothing to label"):
                service.label_features_for_extraction("label_test")

        # Assert the ACTUAL terminal state. `!= COMPLETED` is satisfied by
        # LABELING and QUEUED as well, so deleting the FAILED transition in the
        # generic handler left the job in limbo forever and this test stayed
        # green — the same silent-limbo failure the guard exists to prevent.
        assert labeling_job.status == LabelingStatus.FAILED.value, (
            f"a fully-filtered run ended in {labeling_job.status!r}; it must be "
            f"FAILED, carrying the reason, not left in limbo"
        )
        assert labeling_job.error_message, "no reason was recorded on the failure"

    def test_the_guard_is_present_in_the_source(self):
        import inspect
        src = inspect.getsource(LabelingService.label_features_for_extraction)
        assert "nothing to label" in src, (
            "the all-filtered guard is gone; a run that labels nothing will "
            "report COMPLETED at progress 1.0"
        )


class TestTrialModelIsAdditive:
    """C3 — trial mode must not change anything for existing apply jobs."""

    def test_trial_columns_are_additive_and_default_to_apply(self):
        """NULL mode is the pre-existing behaviour and must stay legal."""
        cols = LabelingJob.__table__.columns
        for name in ("mode", "feature_ids", "trial_run_id"):
            assert name in cols, f"LabelingJob.{name} is missing"
            assert cols[name].nullable, (
                f"LabelingJob.{name} is NOT NULL — every pre-existing row would "
                f"violate it and the migration would fail on a populated database"
            )

    def test_apply_is_the_reading_of_a_null_mode(self):
        assert LabelingMode.APPLY.value == "apply"
        assert LabelingMode.TRIAL.value == "trial"

    def test_results_survive_deletion_of_the_job_that_made_them(self):
        """labeling_job_id must be SET NULL, never CASCADE.

        `delete_labeling_job` is a user-reachable endpoint. Deleting the job that
        produced a measurement must not delete the measurement.
        """
        fk = list(LabelingTrialRun.__table__.c.labeling_job_id.foreign_keys)[0]
        assert fk.ondelete == "SET NULL", (
            f"labeling_job_id uses ondelete={fk.ondelete!r}; deleting a labeling "
            f"job would destroy the trial results it produced"
        )

    def test_the_migration_ddl_agrees_with_the_model(self):
        """The ORM says ON DELETE SET NULL; the migration says it again in raw
        SQL. Two independent declarations can drift, and the deployed database
        follows the MIGRATION — so a model-only assertion would stay green while
        production destroyed trial results on job deletion.
        """
        from pathlib import Path
        mig = Path(__file__).resolve().parents[2] / "alembic" / "versions" / \
            "d7e3a91c04b8_labeling_trials.py"
        assert mig.exists(), "the trial migration moved; this guard is vacuous"
        sql = mig.read_text()

        assert "REFERENCES labeling_jobs(id) ON DELETE SET NULL" in sql, (
            "the migration does not declare SET NULL on labeling_job_id; the "
            "deployed schema would delete a measurement with the job that made it"
        )
        assert "REFERENCES extraction_jobs(id) ON DELETE CASCADE" in sql
        # Anchored to a COLUMN DEFINITION, not a bare substring.
        #
        # `col in sql` was fail-open for 8 of 15 names: `mode`, `feature_ids`,
        # `trial_run_id`, `id`, `panel_id`, `extraction_job_id`,
        # `labeling_job_id` and `created_at` all appear again in index
        # definitions, FK references, COMMENT ON statements or the docstring.
        # Deleting `ADD COLUMN IF NOT EXISTS mode VARCHAR(16),` left this green
        # while upgrade() died on the COMMENT ON COLUMN referring to it.
        import re as _re
        TYPES = r"(VARCHAR|JSONB|TIMESTAMP|TEXT|INTEGER|BOOLEAN|FLOAT|DOUBLE)"
        for col in LabelingTrialRun.__table__.columns.keys():
            assert _re.search(rf"\b{col}\s+{TYPES}", sql), (
                f"model column {col!r} has no column DEFINITION in the migration "
                f"(its name may appear in an index or comment, which is not the "
                f"same thing)"
            )
        for col in ("mode", "feature_ids", "trial_run_id"):
            assert _re.search(rf"ADD COLUMN IF NOT EXISTS {col}\s+{TYPES}", sql), (
                f"labeling_jobs.{col} is not ADDED by the migration"
            )

    def test_trial_run_ids_are_unique(self):
        assert len({_ltr_id() for _ in range(200)}) == 200
