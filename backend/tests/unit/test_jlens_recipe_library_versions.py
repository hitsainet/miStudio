"""A J-lens fit records the library versions it ran on, and the recipe carries them.

GPU block 2 (2026-10-17) published the Llama-3.1-8B lens with ``library_versions: {}``:
the recipe schema declared the field and the fitter never wrote it, so the
028 watchlist built on that lens could not say which torch / transformers / fitter
computed its Jacobians. These tests drive the real config writer and read the result
back through the artifact-document and watchlist routes.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.services import jspace_documents as docs
from tests.unit.test_jlens_fitter import _written_config
from tests.unit.test_jspace_documents import CONFIG, _artifact, registry  # noqa: F401


def _versions_line(text: str) -> dict:
    line = next(l for l in text.splitlines() if l.startswith("library_versions:"))
    return json.loads(line.partition(":")[2])


class TestTheFitRecordsItsVersions:
    def test_the_written_config_names_the_real_libraries(self):
        import torch
        import transformers

        versions = _versions_line(_written_config())
        assert versions["torch"] == torch.__version__
        assert versions["transformers"] == transformers.__version__
        assert versions["python"]
        assert versions["mistudio"], "the fitter's own version is part of the recipe"

    def test_the_fitter_version_is_the_build(self, monkeypatch):
        monkeypatch.setenv("MISTUDIO_VERSION", "0.5.0+abc1234")
        assert _versions_line(_written_config())["mistudio"] == "0.5.0+abc1234"

    def test_it_is_a_top_level_line_the_flat_reader_sees(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text(_written_config())
        config = docs.read_config(path)
        assert json.loads(config["library_versions"])["torch"]


class TestTheRecipeCarriesThem:
    VERSIONS = {"torch": "2.9.1", "transformers": "5.1.0", "mistudio": "0.5.0+abc1234"}

    def _with_versions(self):
        return CONFIG + f"library_versions: {json.dumps(self.VERSIONS)}\n"

    def test_the_artifact_document_states_them(self, registry):  # noqa: F811
        root, _service = registry
        _artifact(root, config=self._with_versions())
        body = TestClient(app).get("/api/v1/jlens/artifacts/lfm-lens/document").json()
        assert body["recipe"]["library_versions"] == self.VERSIONS

    def test_the_watchlist_inherits_them(self, registry):  # noqa: F811
        root, _service = registry
        _artifact(root, config=self._with_versions())
        resp = TestClient(app).post("/api/v1/jlens/watchlists", json={
            "name": "w", "artifact_ref": "lfm-lens",
            "scoring_definition": "mean logprob", "concepts": [{"token": " x", "threshold": 0.5}]})
        assert resp.status_code == 200, resp.text
        assert resp.json()["document"]["recipe"]["library_versions"] == self.VERSIONS

    def test_a_round_trip_through_the_real_writer(self, registry):  # noqa: F811
        import torch

        root, _service = registry
        _artifact(root, config=_written_config())
        body = TestClient(app).get("/api/v1/jlens/artifacts/lfm-lens/document").json()
        assert body["recipe"]["library_versions"]["torch"] == torch.__version__

    @pytest.mark.parametrize("line", ["", "library_versions: not-json\n",
                                      "library_versions: [1, 2]\n"])
    def test_an_older_or_unreadable_record_is_not_recorded(self, line):
        recipe = docs.recipe_from_config(
            {**{k.strip(): v.strip() for k, _, v in
                (l.partition(":") for l in (CONFIG + line).splitlines() if ":" in l)}})
        assert recipe.library_versions == {}
