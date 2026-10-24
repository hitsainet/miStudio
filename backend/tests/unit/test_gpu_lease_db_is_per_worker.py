"""The lease test database is per xdist worker (CI race fix, 2026-10-19).

``leased_node`` (test_steering_gpu_placement) is shared by
test_gpu_leases_compose_with_the_sae_allowance and test_gpu_leases_compose_with_round2.
Under ``--dist loadfile`` those three files land on different workers, and each
``lease_engine`` call drops and recreates ``gpu_leases`` — so one shared database
let a worker drop the table under another (UndefinedTable / missing lease rows).
These tests assert the WIRING: the engine ``lease_engine`` returns is bound to the
worker's own database, not merely that a naming helper exists.
"""

from __future__ import annotations

from tests.unit.gpu_lease_db import lease_db_name, lease_engine


def test_two_workers_get_different_lease_databases():
    a = lease_db_name("mistudio_test_x", {"PYTEST_XDIST_WORKER": "gw0"})
    b = lease_db_name("mistudio_test_x", {"PYTEST_XDIST_WORKER": "gw1"})
    assert a != b
    assert (a, b) == ("mistudio_test_x_gw0", "mistudio_test_x_gw1")


def test_a_serial_run_keeps_the_module_name():
    assert lease_db_name("mistudio_test_x", {}) == "mistudio_test_x"


def test_the_override_is_also_per_worker():
    env = {"MISTUDIO_GPU_LEASE_TEST_DB": "mine", "PYTEST_XDIST_WORKER": "gw2"}
    assert lease_db_name("mistudio_test_x", env) == "mine_gw2"


def test_lease_engine_binds_to_the_workers_database(monkeypatch):
    monkeypatch.delenv("MISTUDIO_GPU_LEASE_TEST_DB", raising=False)
    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw97")
    engine = lease_engine("mistudio_test_lease_db_isolation")
    try:
        assert engine.url.database == "mistudio_test_lease_db_isolation_gw97"
    finally:
        engine.dispose()
