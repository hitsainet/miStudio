"""Template updates and imports refuse bad input instead of corrupting a row or 500ing.

Debt lane WS-C, 2026-09-16. Three tracked findings, all in the template services:

R2F-10  A PATCH carrying an explicit null for a NOT NULL column reached the database
        as `SET <column> = NULL`. `exclude_unset` deliberately keeps an explicitly-sent
        null — that is how "clear this description" is expressed — so the IntegrityError
        escaped the endpoint as a 500 on a request that is merely invalid. All FOUR
        template services did it.
R2F-7   Import stored `template_data["hyperparameters"]` exactly as the file gave it:
        any JSON object at all, including one with no `latent_dim`. Nothing validated
        it, so the file's contents became a template that looks like every other
        template and fails only when a training built from it runs.
R2F-8   `TrainingHyperparameters` took pydantic's default `extra="ignore"`, so a stored
        hyperparameter this build has no field for was DROPPED by the first PATCH that
        round-tripped the template — the client sends the template's own stored keys
        back, the schema discarded the unrecognised ones, and the update REPLACES the
        dict. A key written by a newer build was deleted by an unrelated edit, silently.

MUTATION CONTROLS (each applied alone to the source, this file plus
test_training_template_update_r2f.py, test_training_template_service.py,
test_extraction_template.py and test_labeling_prompt_template_service.py run, the edit
confirmed landed, bytes restored and sha256 verified). All RED:

  C16 `reject_null_updates` returns update_data unchanged (the guard is a no-op)
      -> every test in TestANullForARequiredColumnIsRefused (all four services)
  C17 `required_columns` includes only primary keys (an empty required set)
      -> the same four service tests
  C19 the endpoint's `except NullNotAllowed` deleted -> test_the_endpoint_answers_422_not_500
  C20 the labeling endpoint's NullNotAllowed branch removed, so its `except ValueError`
      answers 403 -> test_a_null_is_422_not_the_system_template_403
  C21 `_validated_hyperparameters` returns the raw dict
      -> test_an_import_with_no_latent_dim_is_refused,
         test_an_overwriting_import_is_validated_too
  C22 the create branch stores `template_data["hyperparameters"]` again
      -> test_a_valid_import_stores_what_create_stores
  C23 the overwrite branch stores the raw dict again
      -> test_an_overwriting_import_stores_the_validated_dump_too
  C24 `extra="allow"` removed from TrainingHyperparameters
      -> test_a_key_this_build_does_not_know_survives_a_round_trip

⚠ C22 AND C23 SURVIVED THEIR FIRST RUN, and that is the finding worth keeping from
this lane. `_validated_hyperparameters` is called BEFORE either branch stores anything,
so a mutant that went back to storing the file's own dict still refused every invalid
import — every refusal test stayed green while the line that stores the validated dump
was unprotected. Asserting that bad input is refused says nothing about what good input
is STORED. The two tests named above were added to assert the stored dict equals
`TrainingHyperparameters(**sent).model_dump()` and carries the full field set, exactly
as `create_template` stores it; re-run: both red.

The other controls were killed on their first run. The driver (scratchpad
`debt-c/mutate.py`) restores every file from a byte copy and asserts its sha256;
`grep -rn "MUTANT C" src` is empty afterwards.
"""

import pytest

from src.models.extraction_template import ExtractionTemplate
from src.models.labeling_prompt_template import LabelingPromptTemplate
from src.models.prompt_template import PromptTemplate
from src.models.training_template import TrainingTemplate
from src.schemas.extraction_template import ExtractionTemplateUpdate
from src.schemas.labeling_prompt_template import LabelingPromptTemplateUpdate
from src.schemas.prompt_template import PromptTemplateUpdate
from src.schemas.training import TrainingHyperparameters
from src.schemas.training_template import TrainingTemplateUpdate
from src.services.extraction_template_service import ExtractionTemplateService
from src.services.labeling_prompt_template_service import LabelingPromptTemplateService
from src.services.prompt_template_service import PromptTemplateService
from src.services.template_updates import (
    NullNotAllowed,
    reject_null_updates,
    required_columns,
)
from src.services.training_template_service import TrainingTemplateService

VALID_HP = {
    "hidden_dim": 512,
    "latent_dim": 16384,
    "learning_rate": 0.0001,
    "batch_size": 64,
    "total_steps": 10000,
}


# ── R2F-10: a null for a column that cannot hold one ─────────────────────────


class TestTheRequiredColumnsAreDerived:
    def test_the_required_columns_come_from_the_mapper_not_a_list(self):
        """Derived from the model, so a column added later is covered without anyone
        remembering to extend a list — the failure mode this repo keeps rediscovering."""
        assert "name" in required_columns(TrainingTemplate)
        assert "dataset_ids" in required_columns(TrainingTemplate)
        assert "encoder_type" in required_columns(TrainingTemplate)
        # Nullable columns are not required, and the primary key is not a field a
        # PATCH can null.
        assert "description" not in required_columns(TrainingTemplate)
        assert "id" not in required_columns(TrainingTemplate)
        assert "layer_indices" in required_columns(ExtractionTemplate)
        assert "prompts" in required_columns(PromptTemplate)
        assert "system_message" in required_columns(LabelingPromptTemplate)

    def test_a_python_side_default_does_not_rescue_an_explicit_null(self):
        """`dataset_ids` has `default=list`, but a default applies only when the
        attribute was never set: setting it to None emits SET dataset_ids = NULL."""
        assert "dataset_ids" in required_columns(TrainingTemplate)
        with pytest.raises(NullNotAllowed):
            reject_null_updates({"dataset_ids": None}, TrainingTemplate)

    def test_a_clearable_field_still_clears(self):
        assert reject_null_updates({"description": None}, TrainingTemplate) == {"description": None}

    def test_the_message_names_every_offending_field(self):
        with pytest.raises(NullNotAllowed) as caught:
            reject_null_updates({"name": None, "encoder_type": None}, TrainingTemplate)
        assert "name" in str(caught.value) and "encoder_type" in str(caught.value)


@pytest.mark.asyncio
class TestANullForARequiredColumnIsRefused:
    """THE FINDING, in all four services: a 4xx, not an IntegrityError from the driver."""

    async def test_training_template(self, async_session):
        row = TrainingTemplate(
            name="t", encoder_type="jumprelu", hyperparameters=dict(VALID_HP), dataset_ids=[]
        )
        async_session.add(row)
        await async_session.commit()
        await async_session.refresh(row)

        with pytest.raises(NullNotAllowed):
            await TrainingTemplateService.update_template(
                async_session, row.id, TrainingTemplateUpdate(name=None)
            )

    async def test_extraction_template(self, async_session):
        row = ExtractionTemplate(
            name="e", layer_indices=[0], hook_types=["residual"], max_samples=1000,
            batch_size=32, top_k_examples=10,  # NOT NULL with no default
        )
        async_session.add(row)
        await async_session.commit()
        await async_session.refresh(row)

        with pytest.raises(NullNotAllowed):
            await ExtractionTemplateService.update_template(
                async_session, row.id, ExtractionTemplateUpdate(layer_indices=None)
            )

    async def test_prompt_template(self, async_session):
        row = PromptTemplate(name="p", prompts=["hello"])
        async_session.add(row)
        await async_session.commit()
        await async_session.refresh(row)

        with pytest.raises(NullNotAllowed):
            await PromptTemplateService.update_template(
                async_session, row.id, PromptTemplateUpdate(prompts=None)
            )

    async def test_labeling_prompt_template(self, async_session):
        row = LabelingPromptTemplate(
            id="lpt_guard", name="l", system_message="sys", user_prompt_template="{examples_block}",
            is_system=False,
        )
        async_session.add(row)
        await async_session.commit()
        await async_session.refresh(row)

        with pytest.raises(NullNotAllowed):
            await LabelingPromptTemplateService.update_template(
                async_session, row.id, LabelingPromptTemplateUpdate(system_message=None)
            )


@pytest.mark.asyncio
class TestTheEndpointAnswersABadRequest:
    async def test_the_endpoint_answers_422_not_500(self, client, async_session):
        """Through the live route: the status code a caller actually sees."""
        row = TrainingTemplate(
            name="through-the-route", encoder_type="jumprelu",
            hyperparameters=dict(VALID_HP), dataset_ids=[],
        )
        async_session.add(row)
        await async_session.commit()
        await async_session.refresh(row)

        response = await client.patch(f"/api/v1/training-templates/{row.id}", json={"name": None})

        assert response.status_code == 422, response.text
        assert "name" in response.text

    async def test_a_valid_patch_is_untouched_by_the_guard(self, client, async_session):
        row = TrainingTemplate(
            name="still-works", encoder_type="jumprelu",
            hyperparameters=dict(VALID_HP), dataset_ids=[], description="before",
        )
        async_session.add(row)
        await async_session.commit()
        await async_session.refresh(row)

        response = await client.patch(
            f"/api/v1/training-templates/{row.id}", json={"name": "renamed", "description": None}
        )

        assert response.status_code == 200, response.text
        assert response.json()["name"] == "renamed"
        assert response.json()["description"] is None, "a nullable field must still clear"


@pytest.mark.asyncio
class TestTheLabelingEndpointDistinguishesItsRefusals:
    async def test_a_null_is_422_not_the_system_template_403(self, client, async_session):
        """`NullNotAllowed` derives from ValueError, and this endpoint's existing
        `except ValueError` answers 403 "cannot modify a system template" — a wrong
        explanation for a null field. The NullNotAllowed branch must come FIRST.

        THROUGH THE LIVE ROUTE, because the ordering of two except clauses is only
        observable there. The first version of this test called the service directly,
        which raises NullNotAllowed under either ordering — it asserted nothing about
        the thing it was named for, and control C20 could not have bitten it.
        """
        row = LabelingPromptTemplate(
            id="lpt_order", name="l", system_message="sys",
            user_prompt_template="{examples_block}", is_system=False,
        )
        async_session.add(row)
        await async_session.commit()

        response = await client.patch(
            "/api/v1/labeling-prompt-templates/lpt_order", json={"system_message": None}
        )

        assert response.status_code == 422, response.text
        assert "system_message" in response.text


# ── R2F-7: an import is validated on the way in ──────────────────────────────


@pytest.mark.asyncio
class TestAnImportIsValidated:
    def _payload(self, hyperparameters, name="Imported"):
        return {
            "version": "1.0",
            "templates": [
                {"name": name, "encoder_type": "jumprelu", "hyperparameters": hyperparameters}
            ],
        }

    async def test_a_valid_import_stores_what_create_stores(self, async_session):
        """A file's five required keys become the COMPLETE validated dict, exactly as
        `create_template` stores it — not the five keys as the file wrote them.

        ASSERTING THE REFUSAL WAS NOT ENOUGH. `_validated_hyperparameters` is called
        before either branch stores anything, so a mutant that went back to storing
        `template_data["hyperparameters"]` still refused every bad import and left this
        class green (controls C22 and C23, both SURVIVED on their first run). What the
        import STORES needs its own assertion.
        """
        result = await TrainingTemplateService.import_templates(
            async_session, self._payload(dict(VALID_HP))
        )
        assert result["created"] == 1

        templates, _ = await TrainingTemplateService.list_templates(async_session)
        stored = templates[0].hyperparameters
        assert stored == TrainingHyperparameters(**VALID_HP).model_dump(), (
            "the import stored the file's own dict, not the validated dump"
        )
        assert set(stored) == set(TrainingHyperparameters.model_fields), (
            "an imported template is missing fields that a created one carries"
        )

    async def test_an_overwriting_import_stores_the_validated_dump_too(self, async_session):
        """The overwrite branch has its own store line, and its own mutant (C23)."""
        await TrainingTemplateService.import_templates(
            async_session, self._payload(dict(VALID_HP), name="Existing")
        )
        replacement = {**VALID_HP, "total_steps": 55555}

        result = await TrainingTemplateService.import_templates(
            async_session, self._payload(replacement, name="Existing"), overwrite_duplicates=True
        )

        assert result["updated"] == 1
        templates, _ = await TrainingTemplateService.list_templates(async_session)
        stored = templates[0].hyperparameters
        assert stored == TrainingHyperparameters(**replacement).model_dump()
        assert set(stored) == set(TrainingHyperparameters.model_fields)
        assert stored["total_steps"] == 55555

    async def test_an_import_with_no_latent_dim_is_refused(self, async_session):
        """THE FINDING. Identical to the valid payload above except for the one
        missing required hyperparameter, so nothing else can explain the refusal."""
        broken = {k: v for k, v in VALID_HP.items() if k != "latent_dim"}

        with pytest.raises(ValueError, match="Imported"):
            await TrainingTemplateService.import_templates(async_session, self._payload(broken))

        templates, total = await TrainingTemplateService.list_templates(async_session)
        assert total == 0, "a template with no latent_dim was stored"

    async def test_an_import_whose_hyperparameters_are_not_an_object_is_refused(self, async_session):
        with pytest.raises(ValueError, match="must be an object"):
            await TrainingTemplateService.import_templates(
                async_session, self._payload("not a dict")
            )

    async def test_an_overwriting_import_is_validated_too(self, async_session):
        """The overwrite branch wrote the raw dict over a GOOD template — so a bad
        import could destroy a working one."""
        await TrainingTemplateService.import_templates(
            async_session, self._payload(dict(VALID_HP), name="Existing")
        )
        broken = {k: v for k, v in VALID_HP.items() if k != "latent_dim"}

        with pytest.raises(ValueError, match="Existing"):
            await TrainingTemplateService.import_templates(
                async_session, self._payload(broken, name="Existing"), overwrite_duplicates=True
            )

        templates, _ = await TrainingTemplateService.list_templates(async_session)
        assert templates[0].hyperparameters["latent_dim"] == 16384, "the good template was overwritten"


# ── R2F-8: a key this build does not know is kept ────────────────────────────


@pytest.mark.asyncio
class TestUnknownStoredKeysSurvive:
    async def test_a_key_this_build_does_not_know_survives_a_round_trip(
        self, client, async_session
    ):
        """THE FINDING. The Templates form sends the template's own stored keys back
        with its edits applied, and the update REPLACES the dict. With `extra="ignore"`
        a key written by a newer build was dropped by an edit to an unrelated field."""
        stored = {**VALID_HP, "a_field_from_a_newer_build": 7}
        row = TrainingTemplate(
            name="round-trip", encoder_type="jumprelu",
            hyperparameters=dict(stored), dataset_ids=[],
        )
        async_session.add(row)
        await async_session.commit()
        await async_session.refresh(row)

        response = await client.patch(
            f"/api/v1/training-templates/{row.id}",
            json={"hyperparameters": {**stored, "total_steps": 20000}},
        )

        assert response.status_code == 200, response.text
        result = response.json()["hyperparameters"]
        assert result["a_field_from_a_newer_build"] == 7, "an unknown stored key was dropped"
        assert result["total_steps"] == 20000

    def test_the_schema_keeps_extras_rather_than_ignoring_them(self):
        parsed = TrainingHyperparameters(**{**VALID_HP, "something_unknown": "kept"})
        assert parsed.model_dump()["something_unknown"] == "kept"
