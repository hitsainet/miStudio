"""Family export — the stale-identity blocker (Feature 041 FR-23; controls record § Review R1).

B-08: the blocker for a member whose BUILT definition no longer states the decision in force (039's
`identity_current` false) survived the S3 suite — only "never built" was pinned. Exporting that file would
ship a threshold the probe no longer serves.
"""
from __future__ import annotations

from tests.unit.test_family_export import API, data_dir, family  # noqa: F401 (fixture)


async def test_a_member_whose_built_definition_is_behind_the_decision_blocks(client, async_session, data_dir,
                                                                            request):
    key, _model, probes = await family(client, async_session, data_dir, request)
    stale = probes[1]
    stale.definition_build = {"resolved": {"detector_digest": "a" * 64, "decision_digest": "b" * 64,
                                           "decision_revision": 1}}
    stale.decision_digest = "c" * 64                                     # re-cut since the build
    await async_session.commit()
    response = await client.get(f"{API}/behaviours/{key}/family/export-index")
    assert response.status_code == 409, response.text
    detail = response.json()["detail"]
    assert [(b["probe_id"], b["reason"]) for b in detail["blockers"]] == [(stale.id, "no_current_definition")]
    assert detail["next"][0]["arguments"]["probe_id"] == stale.id
