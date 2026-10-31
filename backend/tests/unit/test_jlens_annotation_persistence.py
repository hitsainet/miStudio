"""024 tasks 4.1, 4.3, 4.4, 4.5 — two surfaces, persistence, the queue, the sweep.

Covers the defects found while wiring them, each with the shape that would have hidden it:
* the route decoded readout tokens with `convert_ids_to_tokens`, so a label's words were
  compared against tokenizer PIECES (`Ġdog`) and every labelled feature disagreed;
* the encoder surface did not exist, so BR-015's "encoder and decoder as separate
  readouts" had no implementation;
* the queue was a flag in one HTTP response and nowhere else.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from src.services import jlens_annotation_store as store
from src.services.jlens_readout_service import IdentityTransport


class PieceTokenizer:
    """`convert_ids_to_tokens` returns pieces; `decode` returns text. Like a real BPE."""

    def __init__(self, words):
        self.words = words

    def convert_ids_to_tokens(self, ids):
        return [f"Ġ{self.words[i]}" for i in ids]

    def decode(self, ids):
        return "".join(f" {self.words[i]}" for i in ids)


class TinySAE(nn.Module):
    def __init__(self, d_model=6, d_sae=4):
        super().__init__()
        self.tied_weights = False
        self.encoder = nn.Linear(d_model, d_sae, bias=False)
        self.decoder = nn.Linear(d_sae, d_model, bias=False)


WORDS = ["dog", "cat", "paris", "the", "of", "run"]


class TestTheReadoutIsDecodedText:
    def test_decode_tokens_returns_text_not_pieces(self):
        assert store.decode_tokens(PieceTokenizer(WORDS), [0, 2]) == [" dog", " paris"]

    def test_a_label_that_matches_its_readout_does_not_disagree(self):
        """The defect: pieces never equal words, so this scored ~1.0 for every feature."""
        W_U = torch.eye(6)
        direction = torch.zeros(6)
        direction[0] = 5.0  # pushes toward "dog"
        rows = store.annotate_rows(
            directions=direction.unsqueeze(0), indices=[0], labels={0: "dog"},
            unembedding=W_U, tokenizer=PieceTokenizer(WORDS), transport=IdentityTransport(),
            layer=3, sae_id="sae_1", surface="decoder", top_k=1,
        )
        assert rows[0]["top_tokens"] == [" dog"]
        assert rows[0]["disagreement_score"] == pytest.approx(0.0)
        assert rows[0]["has_disagreement"] is False

    def test_no_label_is_not_disagreement(self):
        rows = store.annotate_rows(
            directions=torch.eye(6)[:1], indices=[0], labels={},
            unembedding=torch.eye(6), tokenizer=PieceTokenizer(WORDS),
            transport=IdentityTransport(), layer=3, sae_id="s", surface="decoder",
        )
        assert rows[0]["disagreement_score"] is None
        assert rows[0]["has_disagreement"] is False


class TestTwoSurfaces:
    def test_encoder_and_decoder_rows_are_the_right_vectors(self):
        sae = TinySAE()
        surfaces = store.sae_surfaces(sae)
        # Row i of each surface is feature i's direction in d_model space.
        assert torch.allclose(surfaces["decoder"][2], sae.decoder.weight[:, 2].detach())
        assert torch.allclose(surfaces["encoder"][2], sae.encoder.weight[2, :].detach())
        assert not torch.allclose(surfaces["decoder"], surfaces["encoder"])

    def test_the_route_reads_the_requested_surface(self, monkeypatch):
        from src.api.v1.endpoints import jlens as endpoint

        sae = TinySAE()
        record = SimpleNamespace(local_path="x", d_model=6)
        db = SimpleNamespace()
        monkeypatch.setattr(
            "src.services.circuit_capture_service._load_sae_sync", lambda r, d: sae
        )

        class _Q:
            def filter(self, *_):
                return self

            def first(self):
                return record

        class _DB:
            def query(self, *_):
                return _Q()

        from contextlib import contextmanager

        @contextmanager
        def fake_db():
            yield _DB()

        monkeypatch.setattr("src.core.database.get_sync_db", fake_db)
        enc = endpoint._feature_direction("sae_1", "1", "encoder")
        dec = endpoint._feature_direction("sae_1", "1", "decoder")
        assert torch.allclose(enc, sae.encoder.weight[1, :].detach())
        assert torch.allclose(dec, sae.decoder.weight[:, 1].detach())
        del db


@pytest.mark.asyncio
class TestPersistenceAndTheQueue:
    async def test_upsert_then_queue_sorted_worst_first(self, async_session):
        from src.core.database import get_sync_db
        from src.main import app
        from fastapi.testclient import TestClient

        rows = [
            store.annotation_row(
                sae_id="sae_q", feature_index=i, surface="decoder", lens_type="LOGIT_LENS",
                annotation=SimpleNamespace(layer=4, lens_kurtosis=1.0,
                                           workspace_class=SimpleNamespace(value="unknown"),
                                           top_tokens=["a"]),
                compared_label=f"label {i}", score=score,
            )
            for i, score in [(1, 0.2), (2, 0.95), (3, None), (4, 0.85)]
        ]
        with get_sync_db() as db:
            for row in rows:
                store.upsert_annotation(db, row)
            db.commit()
            # Upsert REPLACES: re-annotating feature 1 must not create a second row.
            store.upsert_annotation(db, {**rows[0], "disagreement_score": 0.99,
                                         "has_disagreement": True})
            db.commit()

        client = TestClient(app)
        body = client.get("/api/v1/jlens/annotations",
                          params={"sae_id": "sae_q", "has_disagreement": True}).json()
        assert body["total"] == 3
        assert [a["feature_index"] for a in body["annotations"]] == [1, 2, 4]
        everything = client.get("/api/v1/jlens/annotations", params={"sae_id": "sae_q"}).json()
        assert everything["total"] == 4, "an upsert must replace, not duplicate"
        assert everything["annotations"][-1]["feature_index"] == 3, "unscored rows sort last"

    async def test_the_feature_list_filters_to_the_queue(self, async_session):
        """BR-013: the queue is a filter over the EXISTING feature list — through the
        real list method, so the count and the page must agree."""
        from src.core.database import get_sync_db
        from src.models.external_sae import ExternalSAE
        from src.models.extraction_job import ExtractionJob
        from src.models.feature import Feature
        from src.schemas.feature import FeatureSearchRequest
        from src.services.feature_service import FeatureService

        async_session.add(ExternalSAE(id="sae_f", name="sae_f", source="trained"))
        async_session.add(ExtractionJob(id="extr_f", external_sae_id="sae_f", config={}))
        await async_session.commit()
        for i in (0, 1, 2):
            async_session.add(Feature(
                id=f"feat_sae_f_{i}", extraction_job_id="extr_f", external_sae_id="sae_f",
                neuron_index=i, name=f"n{i}", activation_frequency=0.1, mean_activation=0.5,
                max_activation=1.0, interpretability_score=0.5,
            ))
        await async_session.commit()

        with get_sync_db() as db:
            for i, score in ((0, 0.9), (1, 0.1), (2, 0.95)):
                store.upsert_annotation(db, store.annotation_row(
                    sae_id="sae_f", feature_index=i, surface="decoder", lens_type="LOGIT_LENS",
                    annotation=SimpleNamespace(layer=4, lens_kurtosis=None,
                                               workspace_class=SimpleNamespace(value="unknown"),
                                               top_tokens=["x"]),
                    compared_label="n", score=score,
                ))
            # An ENCODER flag must not put a feature in the decoder queue.
            store.upsert_annotation(db, store.annotation_row(
                sae_id="sae_f", feature_index=1, surface="encoder", lens_type="LOGIT_LENS",
                annotation=SimpleNamespace(layer=4, lens_kurtosis=None,
                                           workspace_class=SimpleNamespace(value="unknown"),
                                           top_tokens=["x"]),
                compared_label="n", score=0.99,
            ))
            db.commit()

        service = FeatureService(async_session)
        queue = await service.list_features_by_extraction(
            "extr_f", FeatureSearchRequest(jlens_disagreement=True,
                                           sort_by="jlens_disagreement", sort_order="desc"),
        )
        assert [f.neuron_index for f in queue.features] == [2, 0], "worst first"
        assert queue.total == 2, "the count must apply the same filter as the page"
        rest = await service.list_features_by_extraction(
            "extr_f", FeatureSearchRequest(jlens_disagreement=False),
        )
        assert sorted(f.neuron_index for f in rest.features) == [1]
        everything = await service.list_features_by_extraction(
            "extr_f", FeatureSearchRequest(sort_by="jlens_disagreement", sort_order="asc"),
        )
        assert [f.neuron_index for f in everything.features] == [1, 0, 2]


class TestTheSweepRouteQueuesABoundedCpuJob:
    def test_it_is_served_routed_to_cpu_and_bounded(self):
        from src.core.celery_app import celery_app
        from src.main import app

        paths = app.openapi()["paths"]
        assert "post" in paths["/api/v1/jlens/annotation-sweeps"]
        assert "get" in paths["/api/v1/jlens/annotations"]
        name = "src.workers.jlens_annotation_tasks.sweep_annotations"
        assert name in celery_app.tasks, "the sweep task is not registered with the worker"
        route = celery_app.amqp.router.route({}, name)
        assert route["queue"].name == "processing", "a CPU sweep must never take a GPU queue"

    def test_the_route_sends_what_was_asked(self, monkeypatch):
        from fastapi.testclient import TestClient

        from src.main import app

        sent = {}

        class _Delay:
            def __call__(self, **kw):
                sent.update(kw)
                return SimpleNamespace(id="t-sweep")

        from contextlib import contextmanager

        @contextmanager
        def fake_db():
            class _Q:
                def filter(self, *_):
                    return self

                def first(self):
                    return SimpleNamespace(id="sae_1")

            yield SimpleNamespace(query=lambda *_: _Q())

        monkeypatch.setattr("src.core.database.get_sync_db", fake_db)
        with patch("src.workers.jlens_annotation_tasks.sweep_annotations.delay", _Delay()), \
                patch("src.workers.jlens_progress.open_row") as open_row:
            resp = TestClient(app).post("/api/v1/jlens/annotation-sweeps", json={
                "sae_id": "sae_1", "surface": "encoder", "offset": 4096, "max_features": 4096,
            })
        assert resp.status_code == 202, resp.text
        assert sent == {"sae_id": "sae_1", "surface": "encoder", "offset": 4096,
                        "max_features": 4096, "top_k": 8}
        assert open_row.call_args.args[0] == "jlens_annotation_sweep"
        too_big = TestClient(app).post("/api/v1/jlens/annotation-sweeps",
                                       json={"sae_id": "sae_1", "max_features": 4097})
        assert too_big.status_code == 422


@pytest.mark.asyncio
class TestTheSweepWorker:
    async def test_it_annotates_the_range_persists_and_reports_the_shape(self, async_session, monkeypatch):
        from src.core.database import get_sync_db
        from src.models.external_sae import ExternalSAE
        from src.models.extraction_job import ExtractionJob
        from src.models.feature import Feature
        from src.models.jlens_feature_annotation import JLensFeatureAnnotation
        from src.models.model import Model
        from src.workers import jlens_annotation_tasks as tasks

        async_session.add(Model(id="m_sw", name="m", architecture="test", params_count=1))
        async_session.add(ExternalSAE(id="sae_sw", name="s", source="trained", model_id="m_sw", layer=4))
        async_session.add(ExtractionJob(id="extr_sw", external_sae_id="sae_sw", config={}))
        await async_session.commit()
        async_session.add(Feature(
            id="feat_sae_sw_1", extraction_job_id="extr_sw", external_sae_id="sae_sw",
            neuron_index=1, name="dog", activation_frequency=0.1, mean_activation=0.5,
            max_activation=1.0, interpretability_score=0.5,
        ))
        await async_session.commit()

        sae = TinySAE(d_model=6, d_sae=4)
        with torch.no_grad():
            sae.decoder.weight.zero_()
            sae.decoder.weight[0, 1] = 5.0  # feature 1 writes toward "dog"
            sae.decoder.weight[2, 2] = 5.0  # feature 2 writes toward "paris"
        monkeypatch.setattr("src.services.circuit_capture_service._load_sae_sync", lambda r, d: sae)
        monkeypatch.setattr(
            "src.services.jlens_annotation_store.unembedding_and_tokenizer",
            lambda record: (torch.eye(6), PieceTokenizer(WORDS)),
        )
        monkeypatch.setattr("src.workers.jlens_progress.update_row", lambda *a, **k: True)

        task = tasks.sweep_annotations
        monkeypatch.setattr(task, "update_state", lambda *a, **k: None)
        task.push_request(id="t-sweep-1")
        try:
            result = task(sae_id="sae_sw", surface="decoder", offset=1, max_features=2, top_k=1)
        finally:
            task.pop_request()

        assert result["annotated"] == 2
        assert result["next_offset"] == 3, "a bounded sweep says where the next one starts"
        assert result["distribution_plausible"] is None, "no band report: cannot say"
        with get_sync_db() as db:
            rows = {r.feature_index: r for r in db.query(JLensFeatureAnnotation)
                    .filter(JLensFeatureAnnotation.sae_id == "sae_sw").all()}
        assert sorted(rows) == [1, 2]
        assert rows[1].top_tokens == [" dog"] and rows[1].disagreement_score == pytest.approx(0.0)
        assert rows[2].compared_label is None and rows[2].disagreement_score is None
        assert {r.sweep_task_id for r in rows.values()} == {"t-sweep-1"}


@pytest.mark.asyncio
class TestTheAnnotateRoutePersistsTheRequestedSurface:
    """The route end to end: the SURFACE travels to the weights, and the row lands."""

    async def test_encoder_request_reads_and_persists_the_encoder(self, client, async_session, monkeypatch):
        from src.core.database import get_sync_db
        from src.models.external_sae import ExternalSAE
        from src.models.jlens_feature_annotation import JLensFeatureAnnotation
        from src.models.model import Model

        async_session.add(Model(id="m_rt", name="m", architecture="test", params_count=1))
        async_session.add(ExternalSAE(id="sae_rt", name="s", source="trained", model_id="m_rt", layer=2))
        await async_session.commit()

        sae = TinySAE(d_model=6, d_sae=4)
        with torch.no_grad():
            sae.encoder.weight.zero_()
            sae.decoder.weight.zero_()
            sae.encoder.weight[3, 2] = 5.0   # feature 3 READS "paris"
            sae.decoder.weight[0, 3] = 5.0   # feature 3 WRITES "dog"
        monkeypatch.setattr("src.services.circuit_capture_service._load_sae_sync", lambda r, d: sae)
        monkeypatch.setattr(
            "src.services.jlens_annotation_store.unembedding_and_tokenizer",
            lambda record: (torch.eye(6), PieceTokenizer(WORDS)),
        )

        resp = await client.post("/api/v1/jlens/annotate", json={
            "model_id": "m_rt", "sae_id": "sae_rt", "layer": 2, "feature_id": "3",
            "surface": "encoder", "top_k": 1,
        })
        assert resp.status_code == 200, resp.text
        assert resp.json()["surface"] == "encoder"
        assert resp.json()["top_tokens"] == [" paris"], "the ENCODER direction must be read"

        resp = await client.post("/api/v1/jlens/annotate", json={
            "model_id": "m_rt", "sae_id": "sae_rt", "layer": 2, "feature_id": "3", "top_k": 1,
        })
        assert resp.json()["top_tokens"] == [" dog"], "decoder stays the default"

        with get_sync_db() as db:
            rows = {r.surface: r for r in db.query(JLensFeatureAnnotation)
                    .filter(JLensFeatureAnnotation.sae_id == "sae_rt").all()}
        assert set(rows) == {"encoder", "decoder"}, "each surface persists its own row"
        assert rows["encoder"].top_tokens == [" paris"]
        assert rows["decoder"].feature_index == 3
