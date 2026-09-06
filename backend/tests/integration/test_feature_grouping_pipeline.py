"""Integration test: full feature-grouping pipeline (Feature 010).

Seeds a synthetic extraction (SAE → extraction job → features → activations)
in the shared dev database, runs FeatureGroupingService.compute directly
(the exact code path the Celery task uses), and exercises the REST endpoints
against the app. Cleans up after itself.
"""

import uuid

import pytest
from httpx import ASGITransport, AsyncClient

from src.core.database import SyncSessionLocal
from src.main import app
from src.models.extraction_job import ExtractionJob
from src.models.external_sae import ExternalSAE
from src.models.feature import Feature
from src.models.feature_activation import FeatureActivation
from src.models.feature_grouping import FeatureGroupingRun
from src.services.feature_grouping_service import FeatureGroupingService

pytestmark = pytest.mark.integration

SUFFIX = uuid.uuid4().hex[:8]
SAE_ID = f"sae_test_grouping_{SUFFIX}"
EXT_ID = f"ext_sae_test_grouping_{SUFFIX}"

# (neuron, prime token, context) — two "love" context clusters + one "heart"
FIXTURES = [
    (0, "▁love", ["i", "really", "you", "so", "much"]),
    (1, "▁Love", ["i", "truly", "you", "so", "much"]),
    (2, "▁love", ["would", "to", "see", "the", "menu"]),
    (3, "▁heart", ["my", "beats", "fast", "with", "joy"]),
    (4, "▁heart", ["my", "beats", "quick", "with", "joy"]),
]


@pytest.fixture(scope="module")
def seeded_extraction():
    db = SyncSessionLocal()
    try:
        db.add(
            ExternalSAE(
                id=SAE_ID, name="grouping-test-sae", source="local", status="ready",
                n_features=8, d_model=16, layer=1,
            )
        )
        db.add(
            ExtractionJob(
                id=EXT_ID, external_sae_id=SAE_ID, config={}, status="completed",
            )
        )
        db.flush()
        for neuron, prime, ctx in FIXTURES:
            fid = f"feat_sae_{SAE_ID}_{neuron}"
            db.add(
                Feature(
                    id=fid, external_sae_id=SAE_ID, extraction_job_id=EXT_ID,
                    neuron_index=neuron, name=f"feature_{neuron}",
                    label_source="auto", activation_frequency=0.1,
                    interpretability_score=0.5, max_activation=5.0, mean_activation=1.0,
                )
            )
            for i in range(3):
                db.add(
                    FeatureActivation(
                        id=neuron * 10 + i, feature_id=fid, sample_index=i,
                        max_activation=5.0 - i,
                        tokens=ctx[:2] + [prime] + ctx[2:],
                        activations=[0.1] * 6,
                        prefix_tokens=ctx[:2], prime_token=prime,
                        suffix_tokens=ctx[2:], prime_activation_index=2,
                    )
                )
        db.commit()
        yield EXT_ID
    finally:
        # Cleanup (order matters: runs cascade to index/groups/members)
        db.query(FeatureGroupingRun).filter_by(extraction_id=EXT_ID).delete()
        db.query(FeatureActivation).filter(
            FeatureActivation.feature_id.like(f"feat_sae_{SAE_ID}_%")
        ).delete(synchronize_session=False)
        db.query(Feature).filter_by(extraction_job_id=EXT_ID).delete()
        db.query(ExtractionJob).filter_by(id=EXT_ID).delete()
        db.query(ExternalSAE).filter_by(id=SAE_ID).delete()
        db.commit()
        db.close()


class TestGroupingPipeline:
    def test_compute_builds_expected_groups(self, seeded_extraction):
        db = SyncSessionLocal()
        try:
            service = FeatureGroupingService()
            run = service.compute(db, seeded_extraction, params={"min_group_size": 2})
            assert run.status == "completed"
            assert run.feature_count == 5
            # love splits into 2 context clusters; only the romantic pair (n0,n1)
            # reaches min_group_size=2. heart pair forms 1 group.
            assert run.group_count == 2
        finally:
            db.close()

    @pytest.mark.asyncio
    async def test_rest_endpoints_serve_groups(self, seeded_extraction):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test/api/v1") as client:
            status = await client.get(f"/extractions/{seeded_extraction}/feature-groups/status")
            assert status.status_code == 200
            assert status.json()["status"] == "completed"

            groups = await client.get(f"/extractions/{seeded_extraction}/feature-groups")
            body = groups.json()
            assert groups.status_code == 200
            assert body["total"] == 2
            tokens = {g["normalized_token"] for g in body["groups"]}
            assert tokens == {"love", "heart"}

            love = next(g for g in body["groups"] if g["normalized_token"] == "love")
            detail = await client.get(
                f"/extractions/{seeded_extraction}/feature-groups/{love['group_id']}"
            )
            members = detail.json()["members"]
            assert detail.status_code == 200
            assert {m["neuron_index"] for m in members} == {0, 1}
            assert all("*" in (m["context_snippet"] or "") for m in members)

            by_token = await client.get(
                f"/extractions/{seeded_extraction}/features/by-token",
                params={"token": "LOVE", "match": "normalized"},
            )
            assert by_token.status_code == 200
            assert by_token.json()["total"] == 3  # all three love features indexed

            seed_id = f"feat_sae_{SAE_ID}_0"
            related = await client.get(f"/features/{seed_id}/related")
            assert related.status_code == 200
            related_ids = {r["feature_id"] for r in related.json()["related"]}
            assert f"feat_sae_{SAE_ID}_1" in related_ids  # shared token + context

    @pytest.mark.asyncio
    async def test_idempotent_compute_short_circuits(self, seeded_extraction):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test/api/v1") as client:
            response = await client.post(
                f"/extractions/{seeded_extraction}/feature-groups/compute",
                json={"params": {"min_group_size": 2}, "force": False},
            )
            # Identical params + completed run → short-circuit, no new job
            assert response.status_code == 202
            assert response.json()["status"] == "completed"
