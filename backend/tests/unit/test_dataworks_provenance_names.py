"""The names miDataworks probes in miStudio's SERVED OpenAPI (035 FR-17; BR-018, D21).

miDataworks decides at runtime whether each 035 capability exists by reading
`/api/openapi.json` — see `miDataworks/backend/src/services/detector_sets/capabilities.py`
(`from_openapi`, read at miDataworks origin/main on 2026-10-18):

    download_revision  = "revision" in components.schemas.DatasetDownloadRequest.properties
    per_row_scores     = any GET path starting "/api/v1/probe-monitors/probes/" and ending "/scores"
    runs_train_filter  = (miDataworks 011, D39) `train_dataset_id` query parameter on
                         GET /api/v1/probe-monitors/runs

⚠ A RENAME HERE DOES NOT BREAK miDataworks LOUDLY. It silently reads the capability as "not
served" and falls back (unpinned downloads, miLLM scores, the 200-run scan). This file is the only
thing that turns such a rename red. The predicates below are copied from that file, not re-derived,
and they read `app.openapi()` — never `app.routes`, which since FastAPI 0.141 holds no included
route paths.

MUTATION CONTROL: C20 rename `revision` on `DatasetDownloadRequest` (or the route, or the query
parameter) → red.
"""
from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def document():
    from src.main import app

    return app.openapi()


def _props(document, name):
    return set((document["components"]["schemas"].get(name) or {}).get("properties") or {})


def test_download_revision(document):
    assert "revision" in _props(document, "DatasetDownloadRequest")


def test_the_response_carries_the_revision_too(document):
    assert "revision" in _props(document, "DatasetResponse")


def test_per_row_scores(document):
    paths = document["paths"]
    served = [
        p for p in paths
        if p.startswith("/api/v1/probe-monitors/probes/")
        and p.rstrip("/").endswith("/scores")
        and "get" in (paths[p] or {})
    ]
    assert served == ["/api/v1/probe-monitors/probes/{probe_id}/evaluations/{evaluation_id}/scores"]


def test_the_offline_scorer_is_NOT_mistaken_for_it(document):
    """`…/probes/{probe_id}/score` (singular) is the one-input scorer and must never count."""
    assert not "/api/v1/probe-monitors/probes/{probe_id}/score".endswith("/scores")


def test_runs_train_filter(document):
    params = document["paths"]["/api/v1/probe-monitors/runs"]["get"].get("parameters") or []
    query = {p["name"] for p in params if p.get("in") == "query"}
    assert {"train_dataset_id", "status", "limit"} <= query


def test_the_scores_row_fields(document):
    """D28: the row shape miDataworks reads."""
    assert _props(document, "ProbeScoreRow") == {"position", "score", "label", "n_scored"}
    assert {"evaluation_id", "revision", "reported_auroc", "total", "next_offset", "rows",
            "source_dataset_id"} <= _props(document, "ProbeEvaluationScores")


def test_dataworks_behaviour_names(document):
    """Feature 037 FR-15 (D7, D37): miDataworks sends `behaviour` and `positive_meaning` at registration
    only when it finds both names on `ProbeDatasetCreate` in this document. A rename silently turns the
    send off. MUTATION CONTROL: rename either field → red."""
    assert {"behaviour", "positive_meaning"} <= _props(document, "ProbeDatasetCreate")
    assert {"behaviour", "positive_meaning", "positive_labels_confirmed"} <= _props(
        document, "ProbeDatasetResponse"
    )


def test_dataworks_detector_registration_names(document):
    """Feature 040 S2 FR-12 (FTASKS 6.8): miDataworks 017 sends `group_columns`, `sub_question` and the cited
    `behaviour_version` at registration, and the `blend_fit` role, only when it finds them in this document.
    A rename silently turns the send off. MUTATION CONTROL: rename any of them → red."""
    create = document["components"]["schemas"]["ProbeDatasetCreate"]["properties"]
    assert {"group_columns", "sub_question", "behaviour_version"} <= set(create)
    assert {"group_columns", "sub_question", "behaviour_version"} <= _props(document, "ProbeDatasetResponse")
    assert "blend_fit" in create["role"]["enum"]
