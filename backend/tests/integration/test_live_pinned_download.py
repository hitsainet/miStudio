"""LIVE acceptance for Feature 035 — run by the coordinator after deploy, never in CI (marker `live`).

These talk to a DEPLOYED miStudio over HTTP and (test 1) make it download from the Hugging Face Hub.
They skip unless `MISTUDIO_LIVE_URL` is set. Run exactly:

    cd backend
    MISTUDIO_LIVE_URL=http://k8s-mistudio.hitsai.local \\
      venv/bin/python -m pytest tests/integration/test_live_pinned_download.py -m live -v --no-cov -s

    # optional, for test 3 (pick a probe evaluated AFTER the 035 deploy — older ones answer 409):
    MISTUDIO_LIVE_PROBE=pm_… MISTUDIO_LIVE_EVALUATION=pme_…

1. **T8 / D48** — re-download `Arrrlex/models-under-pressure` config `training` PINNED (no revision
   given → the head commit is resolved and recorded) and compare its row count with the legacy
   (revision-NULL) row of the same config and split. With a commit `datasets` reads that commit's own
   files instead of the Hub's parquet export; any difference is RECORDED (the failure message carries
   both counts and both ids), not hidden. Costs one dataset download on the node; creates one dataset
   row the operator may delete afterwards. It uses no GPU; avoid running it during an operator's own
   large download (the Celery dataset queue is shared).
2. Capabilities: the served OpenAPI states all three names miDataworks probes (read-only, cheap).
3. One evaluation's served scores reproduce its reported AUROC to 5e-4 (read-only), and a 10,000-row
   page is timed (FPRD §8: under 1 s on the node).
"""
from __future__ import annotations

import os
import re
import time

import pytest

pytestmark = [pytest.mark.live, pytest.mark.integration]

URL = os.environ.get("MISTUDIO_LIVE_URL", "").rstrip("/")
REPO = "Arrrlex/models-under-pressure"
CONFIG = "training"

needs_live = pytest.mark.skipif(not URL, reason="MISTUDIO_LIVE_URL is not set (live acceptance only)")


def _client():
    import httpx

    return httpx.Client(base_url=URL, timeout=60)


@needs_live
def test_capabilities_are_served():
    with _client() as c:
        doc = c.get("/api/openapi.json").json()
    props = doc["components"]["schemas"]["DatasetDownloadRequest"]["properties"]
    assert "revision" in props
    assert any(p.startswith("/api/v1/probe-monitors/probes/") and p.endswith("/scores") and "get" in doc["paths"][p]
               for p in doc["paths"])
    names = {p["name"] for p in doc["paths"]["/api/v1/probe-monitors/runs"]["get"]["parameters"]}
    assert "train_dataset_id" in names


@needs_live
def test_T8_pinned_redownload_matches_the_legacy_row_count():
    with _client() as c:
        rows = c.get("/api/v1/datasets", params={"limit": 500}).json()["data"]
        legacy = [
            d for d in rows
            if d.get("hf_repo_id") == REPO and (d.get("metadata") or {}).get("config") == CONFIG
            and d.get("revision") is None and d.get("status") == "ready"
        ]
        assert legacy, f"no legacy (revision-NULL, ready) row for {REPO} config {CONFIG}; nothing to compare"
        base = legacy[0]
        split = (base.get("metadata") or {}).get("split")
        body = {"repo_id": REPO, "config": CONFIG}
        if split:
            body["split"] = split
        r = c.post("/api/v1/datasets/download", json=body)
        if r.status_code == 409:
            m = re.search(r"\(dataset ([0-9a-f-]{36})\)", r.json()["detail"])
            assert m, r.text
            new_id = m.group(1)
        else:
            assert r.status_code == 202, r.text
            new_id = r.json()["id"]
        deadline = time.time() + 45 * 60
        while True:
            pinned = c.get(f"/api/v1/datasets/{new_id}").json()
            if pinned["status"] in ("ready", "error") or time.time() > deadline:
                break
            time.sleep(15)
    assert pinned["status"] == "ready", pinned.get("error_message")
    assert re.fullmatch(r"[0-9a-f]{40}", pinned["revision"] or ""), pinned["revision"]
    assert pinned["raw_path"].endswith(f"__rev-{pinned['revision']}")
    assert pinned["num_samples"] == base["num_samples"], (
        f"D48: the pinned (commit-files) download of {REPO}/{CONFIG} split={split!r} has "
        f"{pinned['num_samples']} rows at {pinned['revision']} (dataset {new_id}); the legacy "
        f"parquet-export row {base['id']} has {base['num_samples']}. Record this difference."
    )


@needs_live
def test_served_scores_reproduce_the_reported_auroc():
    probe = os.environ.get("MISTUDIO_LIVE_PROBE")
    evaluation = os.environ.get("MISTUDIO_LIVE_EVALUATION")
    if not (probe and evaluation):
        pytest.skip("set MISTUDIO_LIVE_PROBE and MISTUDIO_LIVE_EVALUATION (an evaluation made after 035)")
    from src.services.probe_monitor_metrics import auroc

    rows, offset, first_page_s = [], 0, None
    with _client() as c:
        while offset is not None:
            t0 = time.perf_counter()
            r = c.get(f"/api/v1/probe-monitors/probes/{probe}/evaluations/{evaluation}/scores",
                      params={"offset": offset, "limit": 10_000})
            elapsed = time.perf_counter() - t0
            first_page_s = elapsed if first_page_s is None else first_page_s
            assert r.status_code == 200, r.text
            body = r.json()
            rows += body["rows"]
            offset = body["next_offset"]
    assert len(rows) == body["total"]
    assert len({row["position"] for row in rows}) == len(rows), "positions are not unique ids"
    if body["reported_auroc"] is not None:
        got = auroc([row["score"] for row in rows], [row["label"] for row in rows])
        assert abs(got - body["reported_auroc"]) < 5e-4, (got, body["reported_auroc"])
    print(f"first 10,000-row page: {first_page_s:.3f} s (FPRD §8 target < 1 s)")
