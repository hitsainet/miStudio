"""A build that commits over a re-cut or a rung change (039 C1 — a pre-existing race, closed).

`build_probe_definition` read the probe row before minutes of GPU work and wrote the file's path,
sha and record afterwards without reading it again — so a re-cut committed in between was overwritten
by a file stating the OLD bar, recorded as current. The worker now re-reads the row under a lock after
writing the file; when the decision digest, the decision revision or the rung moved, it keeps the
file, records it invalidated (`cause: "build_race"`), and returns `built_stale` with a `next` call.

These run the REAL task body against real `mistudio_test` rows. `build` is replaced by a stand-in
that captures the bar as the real one does and then lets a SECOND session commit the change, which is
exactly the interleaving. Controls: remove the commit re-check → `completed` with an old bar → red;
compare the digest only → the A → B → A case red (0xcc/reviews/039_implementation_controls_2026-10-23.md).
"""
from __future__ import annotations

import contextlib
import json

import pytest

from src.schemas.probe_definition import ProbeDefinitionV1
from src.schemas.probe_identity import current_revision
from tests.unit.test_probe_decision import make_rows, second_session, sync_db  # noqa: F401
from tests.unit.test_probe_definition import definition as contract_document


@pytest.fixture
def harness(sync_db, second_session, tmp_path, monkeypatch):  # noqa: F811
    import src.services.probe_definition_builder as builder
    import src.workers.probe_monitor_tasks as tasks
    from src.services.probe_decision import ensure_decision_snapshot
    from src.workers.base_task import DatabaseTask

    rows = make_rows(sync_db, tmp_path)
    # ENH-006 D82: a re-cut now recomputes the rung (the everyday-text count is taken at the bar). Give the
    # rung-2 probe the evidence its rung states — an OOD pass and an everyday-text pass — so a re-cut race
    # here is a DECISION race, and the rung stays 2.
    from src.models.probe_monitor import ProbeMonitorDataset, ProbeMonitorEvaluation
    from tests.support.everyday import add_rung2_checks_pass

    ood = ProbeMonitorDataset(name="ood", dataset_id=rows.dataset.id, input_column="i", label_column="l",
                              label_mapping={}, role="eval", distribution="out_of_distribution", counts={})
    sync_db.add(ood)
    sync_db.flush()
    sync_db.add(ProbeMonitorEvaluation(probe_id=rows.probe.id, dataset_id=ood.id, status="completed",
                                       metrics={"scored": True, "ci": {"low": 0.8, "high": 0.9}}))
    add_rung2_checks_pass(sync_db, rows.probe, rows.dataset.id, directory=tmp_path)
    sync_db.commit()
    ensure_decision_snapshot(sync_db, rows.probe)
    state = {"during": None, "events": []}

    def fake_build(db, probe_id, **_kwargs):
        from src.models.probe_monitor import ProbeMonitor

        probe = db.query(ProbeMonitor).filter(ProbeMonitor.id == probe_id).one()
        record = {"resolved": {
            "vectors": 8, "bytes": 0, "sha256": "",
            "detector_digest": "d" * 64,
            "decision_digest": probe.decision_digest,
            "decision_revision": current_revision(probe.calibration_history),
            "identity_source": "built",
            "rung": int(probe.rung or 0),
        }}
        if state["during"] is not None:
            state["during"](probe_id)              # the other writer commits mid-build
        return ProbeDefinitionV1.model_validate(contract_document()), record

    monkeypatch.setattr(builder, "build", fake_build)
    monkeypatch.setattr("src.core.cancellation.record_progress", lambda *a, **k: None)
    monkeypatch.setattr("src.core.config.settings.data_dir", str(tmp_path), raising=False)
    monkeypatch.setattr(tasks, "emit_probe_definition_built",
                        lambda run_id, probe_id, summary: state["events"].append(summary))

    @contextlib.contextmanager
    def _get_db():
        yield sync_db

    # Patched on `DatabaseTask`, not the task object (a PromiseProxy; memory:
    # monkeypatches-that-silently-do-nothing).
    monkeypatch.setattr(DatabaseTask, "get_db", lambda _self: _get_db())

    def recut(target_fpr):
        def _do(probe_id):
            from src.services.probe_recalibration import recalibrate_probe

            recalibrate_probe(second_session, probe_id, target_fpr=target_fpr, commit=True)
        return _do

    def run():
        result = tasks.build_probe_definition.run(probe_id=rows.probe.id)
        sync_db.expire_all()
        return result

    return {"rows": rows, "state": state, "run": run, "recut": recut, "db": sync_db,
            "other": second_session}


def test_an_undisturbed_build_completes(harness):
    result = harness["run"]()
    probe = harness["rows"].probe
    assert result["status"] == "completed" and "next" not in result
    assert probe.definition_path == result["path"]
    assert probe.built_detector_digest == "d" * 64
    assert result["identity"]["decision_digest"] == probe.decision_digest
    assert harness["state"]["events"][-1]["stale"] is False


def test_the_commit_compares_under_a_row_lock(harness):
    """Review R1: the re-check reads the row `FOR UPDATE` before it writes the build's columns.
    Without the lock a re-cut committing between the comparison and the build's UPDATE is still
    overwritten by a file stating the old bar. Read from the SQL the session actually executed, so
    it cannot pass on a comment or an unused helper."""
    from sqlalchemy import event

    engine = harness["db"].get_bind()
    statements = []

    def capture(_conn, _cursor, statement, *_args):
        statements.append(" ".join(statement.split()).upper())

    event.listen(engine, "before_cursor_execute", capture)
    try:
        assert harness["run"]()["status"] == "completed"
    finally:
        event.remove(engine, "before_cursor_execute", capture)
    locks = [i for i, s in enumerate(statements)
             if s.startswith("SELECT") and "FROM PROBE_MONITORS" in s and s.endswith("FOR UPDATE")]
    writes = [i for i, s in enumerate(statements)
              if s.startswith("UPDATE PROBE_MONITORS") and "DEFINITION_PATH" in s]
    assert locks, "the build commit took no row lock on probe_monitors"
    assert writes and locks[0] < writes[-1], (locks, writes)


def test_recut_during_build_marks_stale(harness):
    rows = harness["rows"]
    built_digest = rows.probe.decision_digest
    harness["state"]["during"] = harness["recut"](0.02)
    result = harness["run"]()
    probe = rows.probe
    assert result["status"] == "built_stale", result
    assert result["next"] == [{
        "tool": "build_probe_definition", "arguments": {"probe_id": probe.id},
        "why": result["next"][0]["why"],
    }]
    assert probe.decision_digest != built_digest
    assert probe.definition_path is None, "a file stating the old bar was recorded as current"
    invalidated = probe.definition_build["invalidated"]
    assert invalidated["cause"] == "build_race"
    assert invalidated["reason"] == "decision changed during build"
    assert invalidated["was_identity"]["decision_digest"] == built_digest
    assert invalidated["was"] == result["path"] and invalidated["was_sha256"] == result["sha256"]
    # The file is kept, and consumers holding it hold this detector digest.
    assert json.loads(open(result["path"]).read())["kind"] == "mistudio.probe-definition/v1"
    assert probe.built_detector_digest == "d" * 64
    assert harness["state"]["events"][-1]["stale"] is True


def test_rung_change_during_build_marks_stale(harness):
    from src.models.probe_monitor import ProbeMonitor

    def bump_rung(probe_id):
        other = harness["other"]
        row = other.query(ProbeMonitor).filter(ProbeMonitor.id == probe_id).one()
        row.rung = 3
        other.commit()

    harness["state"]["during"] = bump_rung
    result = harness["run"]()
    assert result["status"] == "built_stale" and result["moved"] == ["rung"]
    invalidated = harness["rows"].probe.definition_build["invalidated"]
    assert invalidated["cause"] == "build_race"
    assert invalidated["reason"] == "rung changed during build"


def test_a_recut_A_to_B_to_A_during_build_is_still_stale(harness):
    """R-4: the digest recurs, the revision does not. Comparing digests alone would call it current."""
    rows = harness["rows"]
    built_digest = rows.probe.decision_digest

    def there_and_back(probe_id):
        harness["recut"](0.02)(probe_id)
        harness["recut"](0.01)(probe_id)

    harness["state"]["during"] = there_and_back
    result = harness["run"]()
    probe = rows.probe
    assert probe.decision_digest == built_digest, "the fixture did not return to bar A"
    assert result["status"] == "built_stale" and result["moved"] == ["decision revision"]


def test_a_race_below_rung_2_names_the_acknowledgement_its_rebuild_needs(harness):
    """Review R1 (4.7): when the rung FELL below 2 during the build, the `next` rebuild "as it
    stands" would be refused without the agent's acknowledgement — so it says so under `needs`,
    from the LOCKED row's rung (the one in force now, not the one the build read)."""
    from src.models.probe_monitor import ProbeMonitor
    from tests.support.next_calls import assert_next_calls_are_live

    def drop_rung(probe_id):
        other = harness["other"]
        row = other.query(ProbeMonitor).filter(ProbeMonitor.id == probe_id).one()
        row.rung = 1
        other.commit()

    harness["state"]["during"] = drop_rung
    result = harness["run"]()
    assert result["status"] == "built_stale"
    assert_next_calls_are_live(result["next"])
    assert result["next"][0]["needs"] == ["acknowledge_below_rung2_reason"]


def test_built_stale_is_logged_with_ids_and_digests(harness, caplog):
    """8.4: one structured `built_stale` line — probe id, what moved, both digests; no weights, no text."""
    import logging

    rows = harness["rows"]
    harness["state"]["during"] = harness["recut"](0.02)
    with caplog.at_level(logging.INFO, logger="src.workers.probe_monitor_tasks"):
        result = harness["run"]()
    assert result["status"] == "built_stale"
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("built_stale")]
    assert len(lines) == 1
    assert f"probe={rows.probe.id}" in lines[0] and "moved=decision" in lines[0]
    assert f"current_decision_digest={rows.probe.decision_digest}" in lines[0]
