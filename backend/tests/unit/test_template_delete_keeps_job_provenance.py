"""OSD-12 — deleting a labeling template must not erase which template a job used.

`labeling_jobs.prompt_template_id` is declared `ondelete="RESTRICT"`, and
`LabelingPromptTemplateService.delete_template` documents that a template in use
cannot be deleted. Neither was true: without `passive_deletes=True` SQLAlchemy
NULLs the child column before issuing the delete, so RESTRICT never fires. Seen
live on 2026-09-22 — deleting `lpt_807f319bf2f843c8` returned 200 and its trial
job's `prompt_template_id` became NULL.
"""
from src.models.labeling_job import LabelingJob
from src.models.labeling_prompt_template import LabelingPromptTemplate


def _relationship():
    return LabelingPromptTemplate.__mapper__.relationships["labeling_jobs"]


class TestTheDatabaseGetsToRefuse:

    def test_the_relationship_leaves_the_delete_to_the_database(self):
        """Read the MAPPER, not the source text: the mapper is the authority."""
        assert _relationship().passive_deletes is True, (
            "without passive_deletes the ORM NULLs labeling_jobs.prompt_template_id "
            "before the DELETE, so RESTRICT never fires and the job forgets which "
            "template produced its labels"
        )

    def test_the_foreign_key_still_restricts(self):
        """The guard is only meaningful while the FK refuses."""
        fk = next(iter(LabelingJob.__table__.c.prompt_template_id.foreign_keys))
        assert fk.ondelete == "RESTRICT", (
            "the FK no longer restricts, so passive_deletes now means the delete "
            "succeeds and the column keeps a dangling id — re-read OSD-12"
        )

    def test_the_relationship_does_not_cascade_deletes_to_jobs(self):
        """A template delete must never take labeling jobs with it."""
        relationship = _relationship()
        assert "delete" not in (relationship.cascade or ""), (
            "cascading would destroy labeling history instead of refusing"
        )


class TestTheServiceStillPromisesThis:

    def test_the_documented_refusal_is_still_documented(self):
        from src.services.labeling_prompt_template_service import (
            LabelingPromptTemplateService,
        )
        doc = LabelingPromptTemplateService.delete_template.__doc__ or ""
        assert "in use cannot be deleted" in doc, (
            "if the promise is dropped, drop passive_deletes too — do not leave "
            "the code and the docstring disagreeing in the other direction"
        )
