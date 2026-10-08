"""A cluster definition binds to an SAE recorded at a compatible hook (review R3-B, R2-B's B8).

Member indices name features of ONE dictionary. A multi-hook training imports a residual, an MLP
and an attention SAE of the same layer and the same width, and the import binding matched on
model, layer and n_features only: an MLP definition bound to whichever SAE of that layer and width
the database listed first. Steering then refused it only if the bound row happened to record a
non-residual hook; a residual row bound to an MLP definition steered unrelated features.

The binding now compares hooks by the kind the refusals' rule decides (sae_hook_support), so
``blocks.12.hook_mlp_out`` and ``mlp`` are one kind, and an unrecorded hook reads as residual,
as it does for every refusal.

Every fixture lists the WRONG SAE first: an auto-binding that ignores the hook picks it.

MUTATION CONTROLS (table in review_sae_remediation_R3_B_2026-09-15.md):
  B1 check() no longer blocks a hook conflict -> the same-id and explicit-choice tests
  B2 auto-binding no longer filters by hook -> the auto-binding tests
  B3 ``_local_sae_summaries`` drops hook_type -> the live import route test
  B4 hook_conflict compares raw strings, not kinds -> test_a_raw_hook_name_and_the_vocabulary_word_are_one_kind
  B5 an unrecorded hook no longer reads as residual -> test_an_unrecorded_hook_reads_as_residual_on_both_sides
"""

import pytest
from sqlalchemy import select

from src.models.cluster_profile import ClusterProfile
from src.models.external_sae import ExternalSAE
from src.schemas.cluster_profile import (
    ClusterDefinitionV1,
    DefinitionModelRef,
    DefinitionSAERef,
    ProfileMember,
)
from src.services.cluster_profile_service import decide_compatibility


def _definition(hook_type=None, sae_id=None, layer=12, n_features=16384) -> ClusterDefinitionV1:
    return ClusterDefinitionV1(
        name="cluster",
        model=DefinitionModelRef(hf_id="google/gemma-2-2b"),
        sae=DefinitionSAERef(mistudio_sae_id=sae_id, layer=layer, n_features=n_features, hook_type=hook_type),
        members=[ProfileMember(feature_idx=7, strength=1.0)],
    )


def _local(*hooks):
    """One SAE per hook at the same layer and width, in the order given."""
    return [
        {"id": "sae_" + (hook or "none").split(".")[-1], "n_features": 16384, "layer": 12,
         "model_name": "google/gemma-2-2b", "hook_type": hook}
        for hook in hooks
    ]


def test_auto_binding_picks_the_sae_of_the_definitions_hook_not_the_first_listed():
    local = _local("mlp", "attention", "residual")
    assert decide_compatibility(_definition("residual"), local).sae_id == "sae_residual"
    assert decide_compatibility(_definition("mlp"), _local("residual", "mlp")).sae_id == "sae_mlp"
    assert decide_compatibility(_definition("attention"), local).sae_id == "sae_attention"


def test_a_raw_hook_name_and_the_vocabulary_word_are_one_kind():
    local = _local("residual", "mlp")
    decision = decide_compatibility(_definition("blocks.12.hook_mlp_out"), local)
    assert (decision.action, decision.sae_id) == ("warn_bind", "sae_mlp")
    local = _local("blocks.12.hook_mlp_out", "blocks.12.hook_resid_post")
    assert decide_compatibility(_definition("residual"), local).sae_id == "sae_hook_resid_post"


def test_an_unrecorded_hook_reads_as_residual_on_both_sides():
    assert decide_compatibility(_definition(None), _local("mlp", "residual")).sae_id == "sae_residual"
    assert decide_compatibility(_definition("blocks.12.hook_resid_post"), _local("mlp", None)).sae_id == "sae_none"
    blocked = decide_compatibility(_definition("mlp"), _local(None))
    assert blocked.action == "block" and "compatible" in blocked.warnings[0]


def test_no_sae_at_a_compatible_hook_blocks_with_the_reason():
    decision = decide_compatibility(_definition("attention"), _local("mlp", "residual"))
    assert decision.action == "block" and decision.sae_id is None
    assert "'attention'" in decision.warnings[0]


def test_the_same_id_on_a_conflicting_hook_is_blocked_not_bound_silently():
    decision = decide_compatibility(_definition("mlp", sae_id="sae_residual"), _local("mlp", "residual"))
    assert decision.action == "block" and decision.sae_id is None
    assert "hook mismatch" in decision.warnings[0] and "sae_residual" in decision.warnings[0]


def test_an_explicit_choice_on_a_conflicting_hook_is_blocked():
    decision = decide_compatibility(_definition("residual"), _local("residual", "mlp"), explicit_sae_id="sae_mlp")
    assert decision.action == "block" and "hook mismatch" in decision.warnings[0]
    ok = decide_compatibility(_definition("residual"), _local("residual", "mlp"), explicit_sae_id="sae_residual")
    assert (ok.action, ok.sae_id) == ("bind", "sae_residual")


# ── the live import route reads the hook from the database ──────────────────


@pytest.mark.asyncio
async def test_the_import_route_binds_to_the_sae_recorded_at_the_definitions_hook(client, async_session):
    for sae_id, hook in (("sae_route_mlp", "mlp"), ("sae_route_attn", "blocks.12.hook_attn_out"), ("sae_route_res", "residual")):
        async_session.add(ExternalSAE(
            id=sae_id, name=sae_id, source="trained", status="ready", layer=12, n_features=16384,
            hook_type=hook, model_name="google/gemma-2-2b", sae_metadata={}, progress=100.0,
        ))
    await async_session.commit()

    payloads = {
        "blocks.12.hook_resid_post": "sae_route_res",
        "mlp": "sae_route_mlp",
        "blocks.12.attn.hook_z": "sae_route_attn",
    }
    for hook, expected in payloads.items():
        definition = _definition(hook).model_dump(mode="json")
        response = await client.post("/api/v1/cluster-profiles/import", json={"payload": definition})
        assert response.status_code == 200, response.text
        body = response.json()
        assert (body["imported"], body["blocked"], body["errors"]) == (1, 0, 0), body
        profile_id = body["results"][0]["profile_id"]
        profile = (await async_session.execute(select(ClusterProfile).where(ClusterProfile.id == profile_id))).scalar_one()
        assert profile.sae_id == expected, (hook, profile.sae_id)
