"""FR-18: from a held digest back to the probe (Feature 039 FTASKS 5.4).

A consumer holding an event's decision digest — or a file's detector digest — lists the probes that
state it, and `matched` says how: the decision in force (`current`), a recorded revision (`history`),
or the last built definition (`last_built`, D-T8), with EVERY revision at which it occurs.
"""
from __future__ import annotations

import secrets

from src.services.millm_read import NotConfigured
from tests.unit.test_probe_decision import make_rows, sync_db  # noqa: F401

LIST = "/api/v1/probe-monitors/probes"


def _recut(db, probe_id, target):
    from src.services.probe_recalibration import recalibrate_probe

    return recalibrate_probe(db, probe_id, target_fpr=target, commit=True)


async def test_current_historical_and_last_built_digests_match(client, sync_db, tmp_path, monkeypatch):
    from src.services import millm_read

    monkeypatch.setattr(millm_read, "imported_copies", lambda pid: NotConfigured("x"))
    rows = make_rows(sync_db, tmp_path)
    other = make_rows(sync_db, tmp_path)          # never matches: a control row
    first = _recut(sync_db, rows.probe.id, 0.02)
    a = first["previous"]["decision_digest"]      # revision 1
    b = first["current"]["decision_digest"]       # revision 2
    rows.probe.definition_path = str(tmp_path / "x.probe.json")
    rows.probe.definition_build = {"resolved": {"detector_digest": "d" * 64, "decision_digest": b,
                                                "decision_revision": 2, "identity_source": "built"}}
    rows.probe.built_detector_digest = "d" * 64
    sync_db.commit()
    c = _recut(sync_db, rows.probe.id, 0.005)["current"]["decision_digest"]   # revision 3

    by_current = (await client.get(LIST, params={"decision_digest": c})).json()
    assert [p["id"] for p in by_current] == [rows.probe.id]
    assert by_current[0]["matched"] == {"by": "decision", "source": "current", "revisions": [3]}

    by_history = (await client.get(LIST, params={"decision_digest": a})).json()
    assert [p["id"] for p in by_history] == [rows.probe.id]
    assert by_history[0]["matched"] == {"by": "decision", "source": "history", "revisions": [1]}

    # `b` is in the history AND was the last build's: history wins the `source`.
    by_b = (await client.get(LIST, params={"decision_digest": b})).json()
    assert by_b[0]["matched"]["revisions"] == [2]

    by_detector = (await client.get(LIST, params={"detector_digest": "d" * 64})).json()
    assert [p["id"] for p in by_detector] == [rows.probe.id]
    assert by_detector[0]["matched"]["by"] == "detector"
    assert other.probe.id not in {p["id"] for p in by_detector}


async def test_a_last_built_digest_alone_matches(client, sync_db, tmp_path):
    """D-T8: a digest only an invalidated build stated (a pre-039 history has no digests)."""
    rows = make_rows(sync_db, tmp_path)
    digest = secrets.token_hex(32)
    rows.probe.definition_build = {"invalidated": {"cause": "decision", "was_identity": {
        "detector_digest": "d" * 64, "decision_digest": digest, "decision_revision": 4,
        "identity_source": "built"}}}
    sync_db.commit()
    out = (await client.get(LIST, params={"decision_digest": digest})).json()
    assert [p["id"] for p in out] == [rows.probe.id]
    assert out[0]["matched"] == {"by": "decision", "source": "last_built", "revisions": [4]}


async def test_random_digest_is_empty(client, sync_db, tmp_path):
    make_rows(sync_db, tmp_path)
    out = await client.get(LIST, params={"decision_digest": secrets.token_hex(32)})
    assert out.status_code == 200 and out.json() == []
    out = await client.get(LIST, params={"detector_digest": secrets.token_hex(32)})
    assert out.json() == []


async def test_bad_digest_is_422(client):
    for bad in ("e" * 63, "E" * 64, "g" * 64, "e" * 64 + "'; drop table x; --"):
        assert (await client.get(LIST, params={"decision_digest": bad})).status_code == 422
        assert (await client.get(LIST, params={"detector_digest": bad})).status_code == 422


async def test_a_recurring_digest_lists_every_revision(client, sync_db, tmp_path, monkeypatch):
    """A → B → A: digest A is current at revision 3 AND recorded at revision 1."""
    from src.services import millm_read

    monkeypatch.setattr(millm_read, "imported_copies", lambda pid: NotConfigured("x"))
    rows = make_rows(sync_db, tmp_path)
    a = _recut(sync_db, rows.probe.id, 0.02)["previous"]["decision_digest"]
    back = _recut(sync_db, rows.probe.id, 0.01)
    assert back["current"]["decision_digest"] == a
    out = (await client.get(LIST, params={"decision_digest": a})).json()
    assert out[0]["matched"] == {"by": "decision", "source": "current", "revisions": [1, 3]}
