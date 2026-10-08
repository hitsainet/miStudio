"""Parallel runs isolate per worker — and the gate is 3.3x faster because of it.

Measured 2026-09-25 over `tests/unit` (7,510 tests): serial 16m22s; `-n 8` with
xdist's default distribution **36m03s — 2.2x SLOWER than serial**; `-n 4
--dist loadfile` with BLAS pinned **4m56s**. So the win is the distribution mode
and the thread pinning, not the worker count, and this file pins the parts that
are code.

Why any of it is needed: `async_engine` is function-scoped and drops every table
on teardown, so two workers against one database delete each other's schema.
"""
import ast
import inspect
from pathlib import Path

import pytest

from tests.support import xdist_isolation as iso


class TestTheWorkerGetsItsOwnDatabase:

    def test_serial_changes_nothing(self):
        """No worker id means a plain run, and a plain run must be untouched."""
        env = {"DATABASE_URL": "postgresql+asyncpg://h/mistudio_test"}
        assert iso.plan_worker_environment(env) == {}

    def test_both_database_urls_are_suffixed(self):
        plan = iso.plan_worker_environment({
            "PYTEST_XDIST_WORKER": "gw3",
            "DATABASE_URL": "postgresql+asyncpg://h/mistudio_test",
            "DATABASE_URL_SYNC": "postgresql://h/mistudio_test",
        })
        assert plan["DATABASE_URL"] == "postgresql+asyncpg://h/mistudio_test_gw3"
        assert plan["DATABASE_URL_SYNC"] == "postgresql://h/mistudio_test_gw3"

    def test_the_schema_check_database_is_NOT_suffixed(self):
        """It is one shared database kept at alembic head, and read-only here.

        Suffixing it would point every worker at a database no migration ever
        touched, and the schema guards would refuse — which reads as a schema
        failure rather than as a setup mistake.
        """
        plan = iso.plan_worker_environment({
            "PYTEST_XDIST_WORKER": "gw1",
            "DATABASE_URL": "postgresql+asyncpg://h/mistudio_test",
            "SCHEMA_CHECK_DATABASE_URL": "postgresql://h/mistudio_schema_check",
        })
        assert "SCHEMA_CHECK_DATABASE_URL" not in plan

    def test_applying_it_twice_does_not_stack_suffixes(self):
        """conftest can be imported more than once in a session."""
        once = iso.worker_database_url("postgresql://h/mistudio_test", "gw2")
        assert iso.worker_database_url(once, "gw2") == once

    def test_a_missing_url_is_left_alone(self):
        assert iso.worker_database_url(None, "gw0") is None


class TestThreadsArePinnedUnderXdist:
    """Four torch stacks each sizing their pools to the whole machine is what
    made the 8-worker run slower than serial."""

    @pytest.mark.parametrize("var", iso.THREAD_VARS)
    def test_each_thread_variable_is_pinned_to_one(self, var):
        plan = iso.plan_worker_environment({"PYTEST_XDIST_WORKER": "gw0"})
        assert plan[var] == "1"

    def test_serial_does_not_pin_anything(self):
        assert iso.plan_worker_environment({}) == {}

    def test_an_operator_set_value_is_never_overridden(self):
        plan = iso.plan_worker_environment({
            "PYTEST_XDIST_WORKER": "gw0", "OMP_NUM_THREADS": "4",
        })
        assert "OMP_NUM_THREADS" not in plan


class TestItRunsBeforeSrcIsImported:
    """The ordering IS the fix, so assert the order, not the presence.

    `src/core/database.py` builds its engine at import time, so an isolation
    call placed after the `src` imports would rewrite the fixture's url and
    leave `AsyncSessionLocal` pointing at the shared database — the split that
    once pointed the two halves of the suite at two different databases and
    produced eleven failures blamed on three innocent causes.
    """

    def _conftest_tree(self):
        source = (Path(__file__).resolve().parents[1] / "conftest.py").read_text()
        return ast.parse(source)

    def test_the_isolation_call_precedes_every_src_import(self):
        tree = self._conftest_tree()
        call_lines = [
            node.lineno for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and getattr(node.func, "id", "") == "_isolate_xdist_worker"
        ]
        assert call_lines, "conftest no longer calls the isolation at all"

        src_import_lines = [
            node.lineno for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("src")
        ]
        assert src_import_lines, "the scan found no `src` imports — it broke"
        assert min(call_lines) < min(src_import_lines), (
            "the isolation must run before anything from `src` is imported, or "
            "AsyncSessionLocal keeps the shared database"
        )

    def test_the_call_is_at_module_level_not_inside_a_fixture(self):
        """A fixture runs after collection, which is already too late."""
        tree = self._conftest_tree()
        top_level = {
            node.value.func.id
            for node in tree.body
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
        }
        assert "_isolate_xdist_worker" in top_level


class TestTheDsnWorksWhereCIRunsIt:
    """CI's url carries a driver suffix and this workstation's does not.

    `DATABASE_URL_SYNC` is `postgresql+psycopg2://…` in
    `.github/workflows/backend-tests.yml` and plain `postgresql://…` locally, so
    handing it straight to `psycopg2.connect` passes here and fails only in CI.
    """

    def test_a_sqlalchemy_driver_suffix_is_stripped(self):
        assert iso.libpq_dsn("postgresql+psycopg2://u:p@h:5432/db_gw0") == (
            "postgresql://u:p@h:5432/db_gw0"
        )

    def test_a_plain_url_is_unchanged(self):
        assert iso.libpq_dsn("postgresql://u:p@h/db") == "postgresql://u:p@h/db"

    def test_an_asyncpg_url_normalises_too(self):
        assert iso.libpq_dsn("postgresql+asyncpg://h/db") == "postgresql://h/db"

    def test_provision_uses_it(self):
        """Assert the CALL, not that the helper exists."""
        import ast, inspect
        tree = ast.parse(inspect.getsource(iso.provision))
        assert any(
            isinstance(node, ast.Call) and getattr(node.func, "id", "") == "libpq_dsn"
            for node in ast.walk(tree)
        ), "provision() must normalise the dsn, or a parallel CI run cannot connect"


class TestProvisioningIsWired:
    """A worker database that does not exist fails every test in that worker."""

    def test_isolate_provisions_the_sync_database_it_chose(self, monkeypatch):
        provisioned = []
        monkeypatch.setattr(iso, "provision", provisioned.append)
        env = {
            "PYTEST_XDIST_WORKER": "gw5",
            "DATABASE_URL": "postgresql+asyncpg://h/mistudio_test",
            "DATABASE_URL_SYNC": "postgresql://h/mistudio_test",
        }
        applied = iso.isolate(env)
        assert provisioned == ["postgresql://h/mistudio_test_gw5"], (
            "the chosen database must be the one provisioned"
        )
        assert env["DATABASE_URL"] == applied["DATABASE_URL"]

    def test_serial_provisions_nothing(self, monkeypatch):
        called = []
        monkeypatch.setattr(iso, "provision", called.append)
        assert iso.isolate({"DATABASE_URL": "postgresql://h/mistudio_test"}) == {}
        assert called == []

    def test_provision_is_reachable_from_isolate(self):
        """Not a substring scan: walk for the CALL."""
        tree = ast.parse(inspect.getsource(iso.isolate))
        assert any(
            isinstance(node, ast.Call) and getattr(node.func, "id", "") == "provision"
            for node in ast.walk(tree)
        ), "isolate() no longer calls provision(), so a worker database is never created"
