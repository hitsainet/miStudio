"""
A Templates-panel save stores the whole template (R2D-4, review round 2 lane R2-F).

R2-D found that saving a training template in the Templates panel deleted every
hyperparameter the form does not expose. Two halves made that happen. The form built
`hyperparameters` from its own twenty fields. And `update_template` stored
`updates.model_dump(exclude_unset=True)`, which recurses into the nested model, so it
kept only the keys it was sent.

The fix has two halves as well:

* The form overlays its edits on the template's stored hyperparameters
  (`frontend/.../trainingTemplates/templateFormPayload.ts`).
* The update still REPLACES the dict and never merges, so a cleared value cannot
  come back. It now stores the complete validated dump, the same as create. A key
  the form cleared is then stored as the schema default, not left missing. The
  training panel loads a missing key as the framework default, which for JumpReLU
  `target_l0` is 0.05.

These tests go through the live PATCH route with the payload that
`TrainingTemplateForm.r2f.test.tsx` asserts the form sends. The seeded row is template
6a460fbe's 35 stored keys (production export, 2026-09-14), which predates
`lr_decay_steps`, `evaluation_token_budget` and the `holdout_eval_*` fields.

Mutation controls (each red, restored by bytes with sha256 verified; the table is in
.claude/context/sessions/review_sae_remediation_R2_F_2026-09-15.md):
  B1 the complete-dump line deleted, back to exclude_unset recursion
     -> test_form_save_of_the_16k_template, test_update_stores_what_create_stores,
        test_a_cleared_field_is_stored_as_the_default
  B2 a merge with the stored dict (`{**stored, **sent}`)
     -> test_a_cleared_field_is_stored_as_the_default
  B3 `model_dump(exclude_none=True)` -> test_form_save_of_the_16k_template
  B4 top-level `model_dump()` without exclude_unset
     -> test_patch_without_hyperparameters_leaves_them_untouched
  B5 the dataset_id sync deleted -> test_dataset_id_follows_dataset_ids
"""

import copy

import pytest

from src.models.training_template import TrainingTemplate
from src.schemas.training import TrainingHyperparameters

STORED_6A460FBE = {
    "seed": None, "aux_k": None, "top_k": None, "l1_alpha": None, "bandwidth": 0.01,
    "target_l0": None, "batch_size": 2048, "hidden_dim": 2048, "hook_types": ["residual"],
    "latent_dim": 16384, "total_steps": 50000, "adam_epsilon": None, "log_interval": 100,
    "warmup_steps": 2000, "weight_decay": 0.0, "learning_rate": 0.00007,
    "ste_bandwidth": None, "aux_loss_alpha": None, "grad_clip_norm": 1.0,
    "sparsity_coeff": 0.001, "top_k_sparsity": None, "dataset_weights": None,
    "training_layers": [11, 12, 13], "holdout_fraction": 0.0, "architecture_type": "jumprelu",
    "evaluate_ce_delta": True, "initial_threshold": 0.5, "normalize_decoder": True,
    "resample_interval": 5000, "checkpoint_interval": 2000, "dead_neuron_threshold": 10000,
    "normalize_activations": "constant_norm_rescale", "resample_dead_neurons": True,
    "sparsity_warmup_steps": 10000,
}

# What TrainingTemplateForm.r2f.test.tsx asserts the form sends for this template with
# Total Steps edited to 150000. `l1_alpha` and `target_l0` are shown and empty, so they
# are not in it.
FORM_PAYLOAD = {
    "name": "LFM2.5-1.2B · L11-13 residual · JumpReLU 8x",
    "description": None,
    "model_id": None,  # the form sends m_f0271325; no models row exists in this database
    "dataset_ids": [],
    "encoder_type": "jumprelu",
    "is_favorite": True,
    "extra_metadata": {},
    "hyperparameters": {
        "total_steps": 150000,
        "seed": None, "aux_k": None, "top_k": None, "hook_types": ["residual"],
        "adam_epsilon": None, "ste_bandwidth": None, "aux_loss_alpha": None,
        "top_k_sparsity": None, "dataset_weights": None, "training_layers": [11, 12, 13],
        "holdout_fraction": 0, "evaluate_ce_delta": True, "normalize_decoder": True,
        "normalize_activations": "constant_norm_rescale", "sparsity_warmup_steps": 10000,
        "hidden_dim": 2048, "latent_dim": 16384, "architecture_type": "jumprelu",
        "learning_rate": 0.00007, "batch_size": 2048, "warmup_steps": 2000,
        "weight_decay": 0, "grad_clip_norm": 1, "checkpoint_interval": 2000,
        "log_interval": 100, "resample_dead_neurons": True, "resample_interval": 5000,
        "dead_neuron_threshold": 10000, "sparsity_coeff": 0.001, "initial_threshold": 0.5,
        "bandwidth": 0.01, "lr_decay_steps": 0,
    },
}

MINIMAL = {
    "hidden_dim": 768, "latent_dim": 16384, "learning_rate": 3e-4,
    "batch_size": 4096, "total_steps": 10000,
}


async def _seed(session, hyperparameters, **fields):
    row = TrainingTemplate(
        name=fields.pop("name", "LFM2.5-1.2B · L11-13 residual · JumpReLU 8x"),
        encoder_type=fields.pop("encoder_type", "jumprelu"),
        hyperparameters=copy.deepcopy(hyperparameters),
        dataset_ids=fields.pop("dataset_ids", []),
        is_favorite=fields.pop("is_favorite", True),
        extra_metadata=fields.pop("extra_metadata", {}),
        **fields,
    )
    session.add(row)
    await session.commit()
    await session.refresh(row)
    return row


def _url(template_id) -> str:
    return f"/api/v1/training-templates/{template_id}"


@pytest.mark.asyncio
async def test_form_save_of_the_16k_template(client, async_session):
    """Every stored key survives, the edit lands, and nothing is altered."""
    row = await _seed(async_session, STORED_6A460FBE)

    response = await client.patch(_url(row.id), json=FORM_PAYLOAD)
    assert response.status_code == 200, response.text
    stored = (await client.get(_url(row.id))).json()["hyperparameters"]

    for key, value in STORED_6A460FBE.items():
        expected = 150000 if key == "total_steps" else value
        assert key in stored, key
        assert stored[key] == expected, (key, stored[key], expected)

    # The only keys the stored template did not have: the loop field the form always
    # sends, and three fields that postdate the template, stored as schema defaults.
    added = set(stored) - set(STORED_6A460FBE)
    assert added == {
        "lr_decay_steps", "evaluation_token_budget",
        "holdout_eval_tokens", "holdout_eval_chunk_tokens",
    }
    assert stored["lr_decay_steps"] == 0
    fields = TrainingHyperparameters.model_fields
    for key in ("evaluation_token_budget", "holdout_eval_tokens", "holdout_eval_chunk_tokens"):
        assert stored[key] == fields[key].default, key
    # Exposed and empty: null, as stored before the save. A MISSING key would load as
    # the JumpReLU framework default, target_l0 0.05.
    assert stored["target_l0"] is None and stored["l1_alpha"] is None


@pytest.mark.asyncio
async def test_update_stores_what_create_stores(client, async_session):
    """The same hyperparameters store the same dict through create and update."""
    created = await client.post(
        "/api/v1/training-templates",
        json={"name": "created", "encoder_type": "jumprelu", "hyperparameters": MINIMAL},
    )
    assert created.status_code == 201, created.text

    row = await _seed(async_session, STORED_6A460FBE, name="updated")
    updated = await client.patch(_url(row.id), json={"hyperparameters": MINIMAL})
    assert updated.status_code == 200, updated.text

    assert updated.json()["hyperparameters"] == created.json()["hyperparameters"]
    assert set(updated.json()["hyperparameters"]) == set(TrainingHyperparameters.model_fields)


@pytest.mark.asyncio
async def test_a_cleared_field_is_stored_as_the_default(client, async_session):
    """A replace, not a merge: a value the client cleared does not come back."""
    stored = {**STORED_6A460FBE, "grad_clip_norm": 1.0, "target_l0": 0.05}
    row = await _seed(async_session, stored)
    sent = {k: v for k, v in FORM_PAYLOAD["hyperparameters"].items() if k != "grad_clip_norm"}

    response = await client.patch(_url(row.id), json={"hyperparameters": sent})
    assert response.status_code == 200, response.text
    result = response.json()["hyperparameters"]

    assert "grad_clip_norm" in result and result["grad_clip_norm"] is None
    assert "target_l0" in result and result["target_l0"] is None
    assert result["training_layers"] == [11, 12, 13]


@pytest.mark.asyncio
async def test_patch_without_hyperparameters_leaves_them_untouched(client, async_session):
    """The top level stays a PATCH: renaming does not rewrite or fill the dict."""
    row = await _seed(async_session, STORED_6A460FBE, description="kept")

    response = await client.patch(_url(row.id), json={"name": "renamed"})
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["name"] == "renamed"
    assert body["description"] == "kept"
    assert body["hyperparameters"] == STORED_6A460FBE


@pytest.mark.asyncio
async def test_dataset_id_follows_dataset_ids(client, async_session):
    """The template selector matches on `dataset_id`; it must track the list."""
    row = await _seed(async_session, STORED_6A460FBE, dataset_ids=["ds_old"], dataset_id="ds_old")

    moved = await client.patch(_url(row.id), json={"dataset_ids": ["ds_new", "ds_b"]})
    assert moved.status_code == 200, moved.text
    assert moved.json()["dataset_id"] == "ds_new"

    cleared = await client.patch(_url(row.id), json={"dataset_ids": []})
    assert cleared.status_code == 200, cleared.text
    assert cleared.json()["dataset_id"] is None
    assert cleared.json()["dataset_ids"] == []

    renamed = await client.patch(_url(row.id), json={"name": "renamed"})
    assert renamed.json()["dataset_id"] is None
