"""The cost of Feature 039 on the path it changed: `GET /probe-monitors/probes` (FTASKS 8.1, FTDD §11.3).

500 synthetic probes with random behaviour keys in `mistudio_test`: half carry a cached definition
file built before 039 (no identity), and half of THOSE are already recomputed. The first read — which
recomputes the remaining 125 files once — is timed on its own; then the list is timed 200 times and
p50/p95 reported. The same file runs unchanged on the parent commit (where the route has no
identity read and no backfill), which is the comparison FPRD §8 asks for.

Run with `ENVIRONMENT=test` (SQL echo off). `slow` and OPT-IN (`MISTUDIO_RUN_BENCHMARKS=1`): `slow` is not excluded by this repo's default
`addopts`, and two hundred list reads over five hundred rows do not belong in every unit run. CI
asserts NO wall-clock threshold (suite memory `check-ci-not-just-local-green`): this prints and, with
`BENCH_OUT=<path>`, writes the numbers for the controls record.
"""
from __future__ import annotations

import json
import os
import secrets
import shutil
import statistics
import time
from pathlib import Path

import pytest

from tests.unit.test_probe_decision import make_rows, sync_db  # noqa: F401

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(os.environ.get("MISTUDIO_RUN_BENCHMARKS") != "1",
                       reason="benchmark: set MISTUDIO_RUN_BENCHMARKS=1 to run (FTASKS 8.1)"),
]

N_PROBES = 500
N_READS = 200
PRE_039 = Path(__file__).resolve().parents[1] / "fixtures" / "probe_definition_pre_identity.json"


def _percentile(values, q):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]


async def test_list_route_cost(client, sync_db, tmp_path, monkeypatch):  # noqa: F811
    from src.core import config as config_module
    from src.models.probe_monitor import ProbeMonitor
    from src.services.probe_decision import form_snapshot

    data = tmp_path / "data"
    monkeypatch.setattr(config_module.settings, "data_dir", data, raising=False)
    rows = make_rows(sync_db, tmp_path)
    folder = data / "probe_definitions"
    folder.mkdir(parents=True, exist_ok=True)
    columns = {c.name for c in ProbeMonitor.__table__.columns}
    recomputed = {"vectors": 16, "detector_digest": "d" * 64, "decision_digest": "e" * 64,
                  "decision_revision": None, "identity_source": "recomputed"}
    for index in range(N_PROBES):
        probe = ProbeMonitor(run_id=rows.run.id, layer=1, rule="mean", rule_params={},
                             variant="dense", val_metrics={}, selected=False, streamable=True,
                             rung=2, rung_reasons=[], threshold=1.0 + index, target_fpr=0.01,
                             realised_fpr=0.01, threshold_source="validation_negatives",
                             behaviour="b-" + secrets.token_hex(4))
        # Every probe carries its decision snapshot, as every probe does once Slice 1 is deployed
        # (the run's stage writes it; a read forms it for older ones) — so both commits read the
        # same columns and the comparison isolates the identity read.
        snapshot = form_snapshot(probe, None)
        probe.decision_snapshot, probe.decision_digest = snapshot, snapshot["digest"]
        if index % 2 == 0:
            target = folder / f"bench_{index}.probe.json"
            shutil.copy(PRE_039, target)
            probe.definition_path = str(target)
            probe.definition_sha256 = secrets.token_hex(32)
            if index % 4 == 0 and "built_detector_digest" in columns:
                probe.definition_build = {"resolved": dict(recomputed)}
                probe.built_detector_digest = "d" * 64
            else:
                probe.definition_build = {"resolved": {"vectors": 16}}
        sync_db.add(probe)
    sync_db.commit()

    params = {"run_id": rows.run.id}
    started = time.perf_counter()
    first = await client.get("/api/v1/probe-monitors/probes", params=params)
    first_read = time.perf_counter() - started
    assert first.status_code == 200 and len(first.json()) == N_PROBES + 1

    timings = []
    for _ in range(N_READS):
        started = time.perf_counter()
        response = await client.get("/api/v1/probe-monitors/probes", params=params)
        timings.append(time.perf_counter() - started)
        assert response.status_code == 200
    result = {
        "probes": N_PROBES + 1, "reads": N_READS,
        "first_read_s": round(first_read, 4),
        "p50_s": round(statistics.median(timings), 4),
        "p95_s": round(_percentile(timings, 0.95), 4),
        "identity_fields_present": "identity_source" in first.json()[0],
    }
    print("BENCH", json.dumps(result))
    if os.environ.get("BENCH_OUT"):
        Path(os.environ["BENCH_OUT"]).write_text(json.dumps(result))
