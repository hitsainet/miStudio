"""Export and import for the J-space kinds (027 task 4.1).

`schemas/jspace_contracts.py` had no production consumer. These tests drive the routes
that now produce and read the documents, and pin the two refusals that matter: an
unknown kind/version is never read as a known one, and a missing recipe fact is never
invented.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.schemas.jspace_contracts import (
    KIND_JLENS_ARTIFACT,
    KIND_READOUT_RECORD,
    KIND_WATCHLIST,
    KIND_WORKSPACE_ANNOTATION,
)
from src.services import jspace_documents as docs

CONFIG = """model: lfm2.5-1.2b
d_model: 2048
n_layers: 16
n_vocab: 65536
dtype: fp16
target_layer: penultimate
aggregation: mean
seq_len: 117.4
attention_gradients_requested: full
attention_gradients_applied_to_layers: [2, 5]
fitted_layers: [12, 13]
corpus: wikitext-103
n_prompts: 600
converged: true
"""


def _artifact(root: Path, slug="lfm-lens", config=CONFIG, report=None):
    d = root / slug
    d.mkdir(parents=True)
    (d / "lfm_jacobian_lens.pt").write_bytes(b"x" * 64)
    if config is not None:
        (d / "config.yaml").write_text(config)
    return d


@pytest.fixture
def registry(tmp_path, monkeypatch):
    from src.api.v1.endpoints import jlens as endpoint
    from src.services.jlens_artifact_service import JLensArtifactService

    service = JLensArtifactService(tmp_path)
    monkeypatch.setattr(endpoint, "_service", lambda: service)
    monkeypatch.setattr(service, "fitted_layers", lambda ref: [12, 13])
    return tmp_path, service


class TestImportRefusesWhatItDoesNotImplement:
    def test_every_kind_round_trips(self):
        recipe = {"model_id": "m", "corpus": "c", "n_prompts": 1, "seq_len": 8}
        samples = [
            {"kind": KIND_JLENS_ARTIFACT, "slug": "s", "d_model": 8, "n_layers": 2,
             "n_vocab": 10, "layers": [1], "size_bytes": 9, "recipe": recipe},
            {"kind": KIND_WORKSPACE_ANNOTATION, "sae_id": "sae_1", "recipe": recipe,
             "features": [{"feature": 3, "lens_kurtosis": 2.5}]},
            {"kind": KIND_READOUT_RECORD, "model": "m", "prompt": "p", "meta": {"a": 1},
             "tokens": [{"t": "x"}]},
            {"kind": KIND_WATCHLIST, "name": "w", "recipe": recipe,
             "scoring_definition": "eval minus control", "concepts": [{"token": "x", "threshold": 1.0}]},
        ]
        client = TestClient(app)
        for sample in samples:
            parsed = docs.dump_document(docs.parse_document(sample))
            again = docs.dump_document(docs.parse_document(parsed))
            assert parsed == again
            resp = client.post("/api/v1/jlens/documents/validate", json=sample)
            assert resp.status_code == 200, resp.text
            assert resp.json()["kind"] == sample["kind"]
            assert resp.json()["document"] == parsed

    @pytest.mark.parametrize("kind", [
        "mistudio.jlens-artifact/v2", "mistudio.jspace-watchlist/v0", "nonsense", None,
    ])
    def test_an_unknown_kind_or_version_is_refused_not_downgraded(self, kind):
        resp = TestClient(app).post(
            "/api/v1/jlens/documents/validate",
            json={"kind": kind, "slug": "s", "d_model": 8, "n_layers": 2, "n_vocab": 10,
                  "layers": [], "size_bytes": 1,
                  "recipe": {"model_id": "m", "corpus": "c", "n_prompts": 1, "seq_len": 8}},
        )
        assert resp.status_code == 422
        assert "unknown kind" in json.dumps(resp.json())


class TestArtifactExport:
    def test_a_committed_artifact_exports_its_recorded_recipe(self, registry):
        root, _service = registry
        _artifact(root)
        resp = TestClient(app).get("/api/v1/jlens/artifacts/lfm-lens/document")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["kind"] == KIND_JLENS_ARTIFACT
        assert (body["d_model"], body["n_layers"], body["n_vocab"]) == (2048, 16, 65536)
        assert body["layers"] == [12, 13]
        assert body["recipe"]["corpus"] == "wikitext-103"
        assert body["recipe"]["n_prompts"] == 600
        assert body["recipe"]["seq_len"] == 117
        assert [e["layer"] for e in body["recipe"]["per_layer_applicability"]] == [2, 5]
        assert "J" not in body and "weights" not in body, "Track A carries no weights"
        # No stored report: nothing ran, so nothing is claimed.
        assert body["validation"] == {}

    def test_the_stored_verdicts_travel_as_recorded_including_deferred(self, registry, monkeypatch):
        root, service = registry
        _artifact(root)
        monkeypatch.setattr(service, "stored_report", lambda ref: {"results": [
            {"check": "structural", "status": "pass"},
            {"check": "round_trip", "status": "deferred"},
        ]})
        body = TestClient(app).get("/api/v1/jlens/artifacts/lfm-lens/document").json()
        assert body["validation"] == {"structural": "pass", "round_trip": "deferred"}

    @pytest.mark.parametrize("config, missing", [
        (CONFIG.replace("corpus: wikitext-103\n", ""), "corpus"),
        (CONFIG.replace("corpus: wikitext-103", "corpus: unspecified"), "unspecified"),
        (CONFIG.replace("d_model: 2048\n", ""), "dimensions"),
        (None, "model"),
    ])
    def test_a_missing_recipe_fact_is_refused_not_invented(self, registry, config, missing):
        root, _service = registry
        _artifact(root, config=config)
        resp = TestClient(app).get("/api/v1/jlens/artifacts/lfm-lens/document")
        assert resp.status_code == 409
        assert missing in resp.json()["detail"]

    def test_unknown_artifact_is_404(self, registry):
        assert TestClient(app).get("/api/v1/jlens/artifacts/nope/document").status_code == 404


class TestWatchlistExport:
    BODY = {
        "name": "eval-awareness",
        "artifact_ref": "lfm-lens",
        "scoring_definition": "mean logprob of evaluation tokens minus control mean",
        "concepts": [{"token": " evaluation", "threshold": 0.5}],
    }

    def test_the_watchlist_route_returns_a_portable_document_with_the_recipe(self, registry):
        root, _service = registry
        _artifact(root)
        resp = TestClient(app).post("/api/v1/jlens/watchlists", json=self.BODY)
        assert resp.status_code == 200, resp.text
        document = resp.json()["document"]
        assert document["kind"] == KIND_WATCHLIST
        assert document["recipe"]["corpus"] == "wikitext-103"
        assert document["scoring_definition"] == self.BODY["scoring_definition"]
        assert document["concepts"] == [{"token": " evaluation", "threshold": 0.5}]
        # And it imports.
        back = TestClient(app).post("/api/v1/jlens/documents/validate", json=document)
        assert back.status_code == 200

    def test_no_artifact_no_document_and_the_reason(self, registry):
        resp = TestClient(app).post(
            "/api/v1/jlens/watchlists", json={**self.BODY, "artifact_ref": "missing"}
        )
        assert resp.status_code == 200
        assert resp.json()["document"] is None
        assert "missing" in resp.json()["document_unavailable_reason"]
