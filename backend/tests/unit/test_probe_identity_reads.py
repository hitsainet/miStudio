"""The ONE identity read and where it surfaces (Feature 039 FR-16, FR-17; E3, E5; B4; FTDD §7.2–§7.3).

What must hold:
* `identity_fields` follows FTDD §7.2 rules 1 – 6 over the row alone, and NEVER guesses a revision;
* the REST list and report carry the six fields for a built probe (through `probe_summary`, the one
  summary), and the decision route returns the decision in force;
* a definition built before 039 has its identity recomputed ONCE, on the first read, by a
  compare-and-set — a second read does no file I/O, and a concurrent rebuild is never overwritten;
* a READ forms a pre-039 probe's decision snapshot lazily (the slice-1 3.7 follow-up).

Fixtures differ from their baseline in exactly the field under test; behaviour keys are random.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

from src.schemas.probe_identity import document_identity
from src.services import probe_decision as P
from tests.unit.test_probe_decision import make_rows, second_session, sync_db  # noqa: F401

D = "d" * 64
E = "e" * 64
F = "f" * 64
PRE_039 = Path(__file__).resolve().parents[1] / "fixtures" / "probe_definition_pre_identity.json"


def _row(**over):
    """A row as `identity_fields` reads it: a current, built, post-039 probe."""
    fields = dict(
        definition_path="/data/probe_definitions/pm_x.probe.json",
        definition_build={"resolved": {"detector_digest": D, "decision_digest": E,
                                       "decision_revision": 3, "identity_source": "built"}},
        decision_digest=E,
        decision_snapshot={"decision": {"threshold": 1.0}, "digest": E},
        calibration_history=[{"revision": 1}, {"revision": 2}, {"revision": 3}],
    )
    fields.update(over)
    return SimpleNamespace(**fields)


# ── FTDD §7.2 rules ───────────────────────────────────────────────────────────


class TestIdentityFields:
    def test_never_built_is_all_null(self):
        out = P.identity_fields(_row(definition_path=None, definition_build=None))
        assert out == dict.fromkeys(P.IDENTITY_KEYS)

    def test_built_and_current(self):
        assert P.identity_fields(_row()) == {
            "detector_digest": D, "decision_digest": E, "decision_revision": 3,
            "identity_current": True, "identity_stale_reason": None, "identity_source": "built",
        }

    def test_a_built_file_whose_decision_is_not_the_one_in_force(self):
        out = P.identity_fields(_row(decision_digest=F))
        assert out["identity_current"] is False
        assert out["identity_stale_reason"] == "decision_differs_from_current_bar"
        assert out["decision_revision"] == 3, "a BUILT file states its own revision"

    def test_invalidated_by_a_recut(self):
        row = _row(definition_path=None, decision_digest=F, definition_build={
            "invalidated": {"cause": "decision", "reason": "threshold recalibrated",
                            "was_identity": {"detector_digest": D, "decision_digest": E,
                                             "decision_revision": 3,
                                             "identity_source": "built"}}})
        out = P.identity_fields(row)
        assert out["identity_current"] is False
        assert out["identity_stale_reason"] == "decision_recalibrated"
        assert (out["detector_digest"], out["decision_digest"]) == (D, E)

    def test_invalidated_by_a_rung_change_is_still_the_current_decision(self):
        """E3 (a): a rung change moves no digest; the stale FILE is reported by `invalidated` and the
        checklist, not by calling the decision stale."""
        row = _row(definition_path=None, definition_build={
            "invalidated": {"cause": "rung", "reason": "rung changed 2 -> 3",
                            "was_identity": {"detector_digest": D, "decision_digest": E,
                                             "decision_revision": 3, "identity_source": "built"}}})
        out = P.identity_fields(row)
        assert out["identity_current"] is True and out["identity_stale_reason"] is None

    def test_pre_039_recomputed_states_the_history_revision_only_when_current(self):
        resolved = {"detector_digest": D, "decision_digest": E, "decision_revision": None,
                    "identity_source": "recomputed"}
        out = P.identity_fields(_row(definition_build={"resolved": resolved}))
        assert out["identity_source"] == "recomputed" and out["decision_revision"] == 3

    def test_pre_039_recomputed_but_stale_states_NO_revision(self):
        """FR-17 / rule 6: the history's maximum belongs to the decision in force; a file stating
        another decision must not be given that number."""
        resolved = {"detector_digest": D, "decision_digest": E, "decision_revision": None,
                    "identity_source": "recomputed"}
        out = P.identity_fields(_row(definition_build={"resolved": resolved}, decision_digest=F))
        assert out["identity_current"] is False and out["decision_revision"] is None
        assert out["identity_stale_reason"] == "decision_differs_from_current_bar"

    def test_pre_039_invalidated_is_not_recorded_and_the_old_file_is_not_read(self):
        """E5 (a)."""
        row = _row(definition_path=None, definition_build={
            "invalidated": {"cause": "decision", "reason": "x", "was": "/data/gone.json"}})
        out = P.identity_fields(row)
        assert out == {**dict.fromkeys(P.IDENTITY_KEYS), "identity_source": "not_recorded"}

    def test_a_missing_pre_039_file(self):
        out = P.identity_fields(_row(definition_build={"resolved": {"vectors": 16}}),
                                file_missing=True)
        assert out["identity_source"] == "not_recorded"
        assert out["identity_stale_reason"] == "file_missing"
        assert out["decision_revision"] is None

    def test_a_decision_that_cannot_be_formed_is_said_so(self):
        out = P.identity_fields(_row(decision_digest=None,
                                     decision_snapshot={"decision": None, "digest": None}))
        assert out["identity_current"] is False
        assert out["identity_stale_reason"] == "decision_not_formable"


# ── the routes ────────────────────────────────────────────────────────────────


def _built(db, tmp_path, rows, *, identity=True):
    """Give the probe a cached definition file: the pre-039 fixture, with or without an identity in
    its build record."""
    data = tmp_path / "data"
    target = data / "probe_definitions" / f"{rows.probe.id}.probe.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(PRE_039, target)
    detector, decision = document_identity(json.loads(PRE_039.read_text()))
    rows.probe.definition_path = str(target)
    rows.probe.definition_sha256 = "a" * 64
    rows.probe.definition_build = (
        {"resolved": {"vectors": 16, "detector_digest": detector, "decision_digest": decision,
                      "decision_revision": 1, "identity_source": "built"}}
        if identity else {"resolved": {"vectors": 16}}
    )
    db.commit()
    return target, detector, decision


def _match_the_file(db, probe, decision_digest, history=None):
    """Make the row's decision in force the one the fixture file states."""
    doc = json.loads(PRE_039.read_text())
    probe.decision_snapshot = {"decision": doc["decision"], "digest": decision_digest,
                               "formed_at": "2026-10-23T00:00:00+00:00",
                               "threshold_source": doc["decision"]["threshold_source"]}
    probe.decision_digest = decision_digest
    probe.calibration_history = history
    db.commit()


async def test_rest_list_and_report_carry_the_identity(client, sync_db, tmp_path, monkeypatch):
    from src.core import config as config_module

    monkeypatch.setattr(config_module.settings, "data_dir", tmp_path / "data", raising=False)
    rows = make_rows(sync_db, tmp_path)
    _target, detector, decision = _built(sync_db, tmp_path, rows)
    _match_the_file(sync_db, rows.probe, decision)
    listed = (await client.get("/api/v1/probe-monitors/probes", params={"run_id": rows.run.id})).json()
    report = (await client.get(f"/api/v1/probe-monitors/probes/{rows.probe.id}")).json()["probe"]
    for summary in (listed[0], report):
        assert summary["detector_digest"] == detector and summary["decision_digest"] == decision
        assert summary["decision_revision"] == 1 and summary["identity_current"] is True
        assert summary["identity_source"] == "built" and summary["identity_stale_reason"] is None
        assert summary["matched"] is None


async def test_list_recomputes_pre_039(client, sync_db, tmp_path, monkeypatch):
    """B4: the first read recomputes a pre-039 file's identity (`recomputed`) and records it; the
    second read does no file I/O."""
    from src.core import config as config_module

    monkeypatch.setattr(config_module.settings, "data_dir", tmp_path / "data", raising=False)
    rows = make_rows(sync_db, tmp_path)
    _target, detector, decision = _built(sync_db, tmp_path, rows, identity=False)
    _match_the_file(sync_db, rows.probe, decision, history=[{"revision": 1}, {"revision": 2}])
    reads = []
    real = Path.read_text

    def counting(self, *a, **k):
        if self.name.endswith(".probe.json"):
            reads.append(self)
        return real(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", counting)
    first = (await client.get("/api/v1/probe-monitors/probes", params={"run_id": rows.run.id})).json()
    assert len(reads) == 1
    assert first[0]["identity_source"] == "recomputed"
    assert (first[0]["detector_digest"], first[0]["decision_digest"]) == (detector, decision)
    assert first[0]["decision_revision"] == 2 and first[0]["identity_current"] is True
    second = (await client.get("/api/v1/probe-monitors/probes", params={"run_id": rows.run.id})).json()
    assert len(reads) == 1, "the steady-state list read the file again"
    assert second[0]["identity_source"] == "recomputed"
    sync_db.expire_all()
    assert rows.probe.built_detector_digest == detector
    assert rows.probe.definition_sha256 == "a" * 64, "the recompute must not touch the file's sha"


async def test_report_recomputes_and_reports_a_missing_file(client, sync_db, tmp_path, monkeypatch):
    from src.core import config as config_module

    monkeypatch.setattr(config_module.settings, "data_dir", tmp_path / "data", raising=False)
    rows = make_rows(sync_db, tmp_path)
    target, _d, decision = _built(sync_db, tmp_path, rows, identity=False)
    target.unlink()
    report = (await client.get(f"/api/v1/probe-monitors/probes/{rows.probe.id}")).json()["probe"]
    assert report["identity_source"] == "not_recorded"
    assert report["identity_stale_reason"] == "file_missing"
    sync_db.expire_all()
    assert rows.probe.built_detector_digest is None, "nothing is written for a missing file"


def test_backfill_is_cas(sync_db, second_session, tmp_path, monkeypatch):  # noqa: F811
    """A rebuild that changes the sha between the backfill's read and its write wins: the
    recompute's values are NOT written over the new build's record."""
    from src.core import config as config_module
    from src.models.probe_monitor import ProbeMonitor

    monkeypatch.setattr(config_module.settings, "data_dir", tmp_path / "data", raising=False)
    rows = make_rows(sync_db, tmp_path)
    _built(sync_db, tmp_path, rows, identity=False)
    real = P.ensure_decision_snapshot

    def rebuild_meanwhile(db, probe):
        row = second_session.query(ProbeMonitor).filter(ProbeMonitor.id == probe.id).one()
        row.definition_sha256 = "b" * 64
        row.definition_build = {"resolved": {"vectors": 16, "detector_digest": F,
                                             "decision_digest": F, "identity_source": "built"}}
        row.built_detector_digest = F
        second_session.commit()
        return real(db, probe)

    monkeypatch.setattr(P, "ensure_decision_snapshot", rebuild_meanwhile)
    P.backfill_identity([rows.probe.id])
    sync_db.expire_all()
    assert rows.probe.definition_sha256 == "b" * 64
    assert rows.probe.definition_build["resolved"]["identity_source"] == "built"
    assert rows.probe.built_detector_digest == F


async def test_a_read_forms_the_lazy_decision_snapshot(client, sync_db, tmp_path):
    """Slice 1's 3.7 follow-up: a probe cut before 039 (no snapshot) gains its decision snapshot on
    a READ — `GET …/decision` — which is what the coordinator checks live after the deploy."""
    rows = make_rows(sync_db, tmp_path)
    assert rows.probe.decision_snapshot is None
    out = (await client.get(f"/api/v1/probe-monitors/probes/{rows.probe.id}/decision")).json()
    sync_db.expire_all()
    assert rows.probe.decision_snapshot is not None
    assert out["decision_digest"] == rows.probe.decision_digest is not None
    assert out["decision"] == rows.probe.decision_snapshot["decision"]
    assert out["decision"]["calibration"]["hf_id"] == "org/cal-set"
    assert out["decision_revision"] == 1 and out["detector_digest"] is None
    assert out["stale_reason"] is None


async def test_the_decision_route_says_why_a_bar_is_not_formable(client, sync_db, tmp_path):
    rows = make_rows(sync_db, tmp_path, hf_repo_id=None)
    out = (await client.get(f"/api/v1/probe-monitors/probes/{rows.probe.id}/decision")).json()
    assert out["decision"] is None and out["decision_digest"] is None
    assert out["stale_reason"] == "decision_not_formable"
    assert "HuggingFace identifier" in out["detail"]


async def test_the_decision_route_404s_an_unknown_probe(client):
    response = await client.get("/api/v1/probe-monitors/probes/pm_nope/decision")
    assert response.status_code == 404


async def test_the_definition_409_names_the_invalidation_cause(client, sync_db, tmp_path):
    """FTASKS 5.6: an agent can tell a re-cut from a rung change without parsing prose."""
    rows = make_rows(sync_db, tmp_path)
    rows.probe.definition_build = {"invalidated": {"reason": "threshold recalibrated",
                                                   "cause": "decision"}}
    sync_db.commit()
    response = await client.get(f"/api/v1/probe-monitors/probes/{rows.probe.id}/definition")
    assert response.status_code == 409
    assert "cause: decision" in response.json()["detail"]


def test_the_list_route_calls_the_backfill_and_the_summary_calls_identity_fields(monkeypatch):
    """Belt and braces for the wiring: the payload tests above go red when either line goes; this
    names the two call sites by behaviour (a spy), not by a source scrape."""
    from src.api.v1.endpoints import probe_monitors as ep

    seen = []
    monkeypatch.setattr(ep, "identity_fields",
                        lambda probe, file_missing=False: seen.append(file_missing) or {})
    stub = SimpleNamespace(
        id="pm_x", run_id="r", layer=1, rule="mean", rule_params={}, variant="dense", sae_id=None,
        sae_feature_indices=None, val_metrics={}, selected=True, threshold=1.0, target_fpr=0.01,
        realised_fpr=0.01, threshold_source=None, streamable=True, rung=2, rung_reasons=[],
        created_at="2026-10-23T00:00:00Z", window_decisions=None, length_bands=None,
    )
    ep.summary_without_curve(stub, None, file_missing=True)
    assert seen == [True]


def test_the_backfill_records_no_revision_for_a_stale_file(sync_db, tmp_path, monkeypatch):
    """Rule 6 in the STORED record (control C5.1d): a pre-039 file whose decision is not the one in
    force is recorded with `decision_revision: null` — the digest lookup's `last_built` match reads
    this field, so a guessed revision would point an event at a bar that never judged it."""
    from src.core import config as config_module

    monkeypatch.setattr(config_module.settings, "data_dir", tmp_path / "data", raising=False)
    rows = make_rows(sync_db, tmp_path)
    _built(sync_db, tmp_path, rows, identity=False)
    _match_the_file(sync_db, rows.probe, F, history=[{"revision": 1}, {"revision": 2}])
    P.backfill_identity([rows.probe.id])
    sync_db.expire_all()
    resolved = rows.probe.definition_build["resolved"]
    assert resolved["identity_source"] == "recomputed" and resolved["decision_revision"] is None


def test_the_backfill_never_overwrites_a_concurrent_invalidation(
    sync_db, second_session, tmp_path, monkeypatch  # noqa: F811
):
    """Control C5.3b: a re-cut that INVALIDATES the pre-039 definition between the backfill's read
    and its write clears the sha (and leaves `built_detector_digest` null, as a pre-039 row has it).
    Only the sha predicate stops the backfill writing its stale `definition_build` over the
    `invalidated` record — losing the record of what was dropped and why."""
    from src.core import config as config_module
    from src.models.probe_monitor import ProbeMonitor
    from src.services.probe_definition_builder import mark_invalidated

    monkeypatch.setattr(config_module.settings, "data_dir", tmp_path / "data", raising=False)
    rows = make_rows(sync_db, tmp_path)
    _built(sync_db, tmp_path, rows, identity=False)
    real = P.ensure_decision_snapshot

    def invalidate_meanwhile(db, probe):
        row = second_session.query(ProbeMonitor).filter(ProbeMonitor.id == probe.id).one()
        mark_invalidated(row, reason="threshold recalibrated", cause="decision")
        second_session.commit()
        return real(db, probe)

    monkeypatch.setattr(P, "ensure_decision_snapshot", invalidate_meanwhile)
    P.backfill_identity([rows.probe.id])
    sync_db.expire_all()
    assert rows.probe.definition_path is None
    assert rows.probe.definition_build["invalidated"]["cause"] == "decision", (
        "the backfill overwrote the invalidation record with its stale read"
    )
    assert rows.probe.built_detector_digest is None


async def test_the_list_reports_a_missing_file_too(client, sync_db, tmp_path, monkeypatch):
    """Control C5.3d: the LIST passes the backfill's per-request `file_missing` flag to the summary,
    as the report does."""
    from src.core import config as config_module

    monkeypatch.setattr(config_module.settings, "data_dir", tmp_path / "data", raising=False)
    rows = make_rows(sync_db, tmp_path)
    target, _d, _decision = _built(sync_db, tmp_path, rows, identity=False)
    target.unlink()
    listed = (await client.get("/api/v1/probe-monitors/probes", params={"run_id": rows.run.id})).json()
    assert listed[0]["identity_source"] == "not_recorded"
    assert listed[0]["identity_stale_reason"] == "file_missing"


async def test_a_digest_lookup_finds_a_pre_039_file_never_read_before(client, sync_db, tmp_path,
                                                                      monkeypatch):
    """Review R1 (Slice 2): the digest filter ran in SQL BEFORE the one-time recompute, and the
    recompute only ran over the rows the query returned — so a pre-039 definition no unfiltered read
    had touched stated no digest in the database, and a lookup by its own detector or decision digest
    (one miLLM recomputes for its imported copy) answered "no probe". The pending recompute now runs
    before a filtered query."""
    from src.core import config as config_module

    monkeypatch.setattr(config_module.settings, "data_dir", tmp_path / "data", raising=False)
    rows = make_rows(sync_db, tmp_path)
    _target, detector, decision = _built(sync_db, tmp_path, rows, identity=False)
    _match_the_file(sync_db, rows.probe, decision, history=[{"revision": 1}])
    sync_db.expire_all()
    assert rows.probe.built_detector_digest is None, "the fixture must be un-recomputed"
    by_detector = (await client.get("/api/v1/probe-monitors/probes",
                                    params={"detector_digest": detector})).json()
    assert [p["id"] for p in by_detector] == [rows.probe.id]
    assert by_detector[0]["identity_source"] == "recomputed"


async def test_the_decision_route_states_the_revision_in_force(client, sync_db, tmp_path,
                                                             monkeypatch):
    """Review R1 (Slice 2): `millm_recalibrate_probe` sends the route's `decision_revision` to miLLM
    and checks the agent's against it. The only test read a never-re-cut probe, where revision 1 is
    also the constant default — so `decision_revision: 1` survived. After two re-cuts it is 3."""
    from src.services import millm_read
    from src.services.probe_recalibration import recalibrate_probe

    monkeypatch.setattr(millm_read, "imported_copies", lambda pid: millm_read.NotConfigured("x"))
    rows = make_rows(sync_db, tmp_path)
    recalibrate_probe(sync_db, rows.probe.id, target_fpr=0.02, commit=True)
    last = recalibrate_probe(sync_db, rows.probe.id, target_fpr=0.005, commit=True)
    out = (await client.get(f"/api/v1/probe-monitors/probes/{rows.probe.id}/decision")).json()
    assert out["decision_revision"] == 3 == last["current"]["decision_revision"]
    assert out["decision_digest"] == last["current"]["decision_digest"]
