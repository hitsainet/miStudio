"""004 Future items, built 2026-10-16 (suite tracker 4.23).

* EDGE-TIER summarisation: the weakest stored activations are summarised in pass 1 as
  their OWN tier and reach the synthesis prompt in their own section — where the concept
  ends, never mixed into what it is most about.
* BATCH enhanced labeling over a subset of one extraction (explicit ids, or the
  hand-starred features), bounded, never queuing a feature twice.
"""

import pytest
from unittest.mock import MagicMock, patch

from src.services.enhanced_labeling_service import EnhancedLabelingService


def _row(prime, act):
    return {"prime_token": prime, "prefix_tokens": ["the"], "suffix_tokens": ["x"],
            "max_activation": act}


class TestTheEdgeTier:
    def _service(self, prompts):
        svc = EnhancedLabelingService(endpoint="http://t/v1", model="m", workers=2)

        def fake_call(prompt, max_tokens, retries=3):
            prompts.append(prompt)
            if "Identify the single unifying concept" in prompt:
                return ('{"name":"france","category":"geo","description":"d",'
                        '"confidence":"high","reasoning":"r"}')
            return "note for " + prompt.split("[", 1)[1].split("]", 1)[0]

        svc._call_llm = fake_call
        return svc

    def test_edge_rows_are_summarised_and_reach_synthesis_in_their_own_section(self):
        prompts = []
        svc = self._service(prompts)
        progress = []
        result = svc.run(
            activation_rows=[_row("Paris", 9.0), _row("Lyon", 8.0)],
            max_examples=2,
            progress_cb=lambda n, total: progress.append((n, total)),
            edge_rows=[_row("capital", 0.4)],
        )
        synthesis = prompts[-1]
        assert "WEAKEST STORED ACTIVATIONS" in synthesis
        edge_section = synthesis.split("WEAKEST STORED ACTIVATIONS", 1)[1]
        assert "capital" in edge_section
        assert "capital" not in synthesis.split("PER-EXAMPLE OBSERVATIONS", 1)[1].split(
            "WEAKEST STORED ACTIVATIONS", 1)[0], "an edge example must not join the core"
        assert [s["tier"] for s in result["pass1_summaries"]] == ["top", "top", "edge"]
        assert progress[-1] == (3, 3), "progress counts both tiers"
        assert "Edge tier" in result["notes"]

    def test_no_edge_rows_no_edge_section(self):
        prompts = []
        svc = self._service(prompts)
        result = svc.run(activation_rows=[_row("Paris", 9.0)], max_examples=1)
        assert "WEAKEST STORED ACTIVATIONS" not in prompts[-1]
        assert "Edge tier" not in result["notes"]


@pytest.mark.asyncio
class TestTheBatchRoute:
    async def _seed(self, async_session):
        from src.models.extraction_job import ExtractionJob
        from src.models.external_sae import ExternalSAE
        from src.models.feature import Feature

        async_session.add(ExternalSAE(id="sae_b", name="s", source="trained"))
        async_session.add(ExtractionJob(id="extr_b", external_sae_id="sae_b", config={}))
        await async_session.commit()
        colours = {0: ("yellow", True), 1: ("yellow", True), 2: ("aqua", True), 3: (None, False)}
        for i, (colour, fav) in colours.items():
            async_session.add(Feature(
                id=f"feat_b_{i}", extraction_job_id="extr_b", external_sae_id="sae_b",
                neuron_index=i, name=f"n{i}", activation_frequency=0.1, mean_activation=0.5,
                max_activation=1.0, interpretability_score=0.5, is_favorite=fav,
                star_color=colour,
            ))
        await async_session.commit()

    @pytest.fixture
    def configured(self, monkeypatch):
        from src.api.v1.endpoints import enhanced_labeling as ep

        async def cfg(db):
            return {"method": "openai_compatible", "endpoint": "http://t/v1", "model": "m",
                    "workers": 2}

        monkeypatch.setattr(ep, "_resolve_labeling_config", cfg)
        delay = MagicMock(return_value=MagicMock(id="celery-1"))
        monkeypatch.setattr(
            "src.workers.enhanced_labeling_tasks.enhanced_label_feature_task.delay", delay)
        return delay

    async def test_starred_only_queues_exactly_the_hand_starred(self, client, async_session, configured):
        await self._seed(async_session)
        resp = await client.post("/api/v1/extractions/extr_b/label/enhanced/batch",
                                 json={"starred_only": True})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["considered"] == 2 and len(body["queued"]) == 2
        assert configured.call_count == 2
        # And again: both are now in flight, so nothing is queued twice.
        again = (await client.post("/api/v1/extractions/extr_b/label/enhanced/batch",
                                   json={"starred_only": True})).json()
        assert again["queued"] == [] and again["considered"] == 0, (
            "queued features turn purple, so a second starred batch finds none"
        )

    async def test_explicit_ids_report_strangers_and_in_flight(self, client, async_session, configured):
        await self._seed(async_session)
        first = (await client.post("/api/v1/extractions/extr_b/label/enhanced/batch",
                                   json={"feature_ids": ["feat_b_3"]})).json()
        assert len(first["queued"]) == 1
        body = (await client.post("/api/v1/extractions/extr_b/label/enhanced/batch",
                                  json={"feature_ids": ["feat_b_3", "feat_elsewhere"]})).json()
        assert body["already_active"] == ["feat_b_3"]
        assert body["skipped"] == [{"feature_id": "feat_elsewhere", "reason": "not in this extraction"}]
        assert configured.call_count == 1

    async def test_exactly_one_selector(self, client, async_session, configured):
        resp = await client.post("/api/v1/extractions/extr_b/label/enhanced/batch", json={})
        assert resp.status_code == 422


@pytest.mark.asyncio
class TestTheTaskFeedsTheEdgeTier:
    """The worker reads the WEAKEST stored rows, disjoint from the top set, and hands
    them to the service — without this the service's edge tier is unreachable."""

    @pytest.mark.parametrize("stored, expected_edge", [
        (30, [0.0, 1.0, 2.0, 3.0, 4.0]),
        # ⚠ A SURVIVING MUTATION ADDED THIS CASE: with 30 rows the weakest five never
        # overlap the top twenty, so dropping the exclusion left the test green. With
        # 22 stored the top set reaches down to 2.0 — only 0.0 and 1.0 are left.
        (22, [0.0, 1.0]),
    ])
    async def test_edge_rows_are_the_weakest_and_never_the_top(self, async_session, monkeypatch, stored, expected_edge):
        from src.core.database import get_sync_db
        from src.models.enhanced_labeling_job import EnhancedLabelingJob
        from src.models.extraction_job import ExtractionJob
        from src.models.external_sae import ExternalSAE
        from src.models.feature import Feature
        from src.models.feature_activation import FeatureActivation
        from src.workers import enhanced_labeling_tasks as tasks

        async_session.add(ExternalSAE(id="sae_t", name="s", source="trained"))
        async_session.add(ExtractionJob(id="extr_t", external_sae_id="sae_t", config={}))
        await async_session.commit()
        async_session.add(Feature(
            id="feat_t", extraction_job_id="extr_t", external_sae_id="sae_t", neuron_index=0,
            name="n", activation_frequency=0.1, mean_activation=0.5, max_activation=1.0,
            interpretability_score=0.5,
        ))
        await async_session.commit()
        for i in range(stored):
            async_session.add(FeatureActivation(
                feature_id="feat_t", sample_index=i, max_activation=float(i), tokens=["a"],
                activations=[float(i)], prime_token=f"t{i}", prefix_tokens=[], suffix_tokens=[],
            ))
        async_session.add(EnhancedLabelingJob(
            id="elj_t", feature_id="feat_t", status="queued", method="openai_compatible",
            endpoint="http://t/v1", model="m", workers=1, examples_total=20,
        ))
        await async_session.commit()

        seen = {}

        class FakeService:
            def __init__(self, **kw):
                pass

            def run(self, activation_rows, max_examples, progress_cb=None, edge_rows=None):
                seen["top"] = [r["max_activation"] for r in activation_rows[:max_examples]]
                seen["edge"] = [r["max_activation"] for r in (edge_rows or [])]
                return {"name": "n", "category": "c", "description": "d", "notes": "x",
                        "raw_synthesis": "{}", "pass1_summaries": []}

            def close(self):
                pass

        monkeypatch.setattr("src.services.enhanced_labeling_service.EnhancedLabelingService", FakeService)
        monkeypatch.setattr(tasks, "_ensure_model_loaded", lambda *a: None)
        for name in ("emit_enhanced_labeling_progress", "emit_enhanced_labeling_completed",
                     "emit_enhanced_labeling_failed"):
            monkeypatch.setattr(f"src.workers.websocket_emitter.{name}", lambda *a, **k: None)

        task = tasks.enhanced_label_feature_task
        task.push_request(id="celery-t")
        try:
            task.run("elj_t")
        finally:
            task.pop_request()

        assert seen["top"] == [float(i) for i in range(stored - 1, stored - 21, -1)]
        assert seen["edge"] == expected_edge
        with get_sync_db() as db:
            job = db.query(EnhancedLabelingJob).filter_by(id="elj_t").one()
        assert job.examples_total == 20 + len(expected_edge), "the bar counts both tiers"
