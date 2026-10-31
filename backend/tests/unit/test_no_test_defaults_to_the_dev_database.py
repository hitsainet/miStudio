"""No test module may default to the local DEVELOPMENT database `mistudio` (OSD-57, 2026-10-23).

Five modules did: each fell back to `postgresql://…/mistudio` when `DATABASE_URL_SYNC` was unset, and
two of them drop and recreate tables. The rule now lives in `tests/support/test_databases.py`.

Two halves, so neither fails open:
* BEHAVIOUR — the helpers never return the dev database, whatever the environment says;
* EVERY STRING CONSTANT in every Python file under `tests/` (parsed with `ast`, so comments and
  docstrings describing the defect are not confused with code — a docstring is a constant too, so
  only `postgresql…` URLs are inspected) that is a PostgreSQL URL must not name `mistudio`. The scan
  proves it can see by finding a planted literal first, and asserts it read the files it should.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.support import test_databases as T

TESTS = Path(__file__).resolve().parents[1]


def _folded(node: ast.AST) -> str | None:
    """The text of a string expression with every interpolation as `{}` — so an f-string
    (`f"postgresql://{host}/mistudio"`) or a concatenation (`"postgresql://h/" + "mistudio"`) is
    read as the URL it builds (review R1: the first form of the scan read only plain constants)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(part.value if isinstance(part, ast.Constant) else "{}" for part in node.values)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _folded(node.left), _folded(node.right)
        if left is not None or right is not None:
            return (left if left is not None else "{}") + (right if right is not None else "{}")
    return None


def _env_defaults(tree: ast.AST) -> list[str]:
    """`os.environ.get("…", "mistudio")` / `os.getenv(…, "mistudio")`: a bare database-name default."""
    found = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("get", "getenv", "setdefault") and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant) and node.args[1].value == T.DEV_DATABASE):
            found.append(f"default {T.DEV_DATABASE!r} at line {node.lineno}")
    return found


def _dev_urls(source: str) -> list[str]:
    found = []
    tree = ast.parse(source)
    for node in ast.walk(tree):
        text = _folded(node)
        if text is not None:
            for token in text.split():
                if (token.startswith(("postgresql://", "postgresql+")) and "://" in token
                        and "/" in token.split("://", 1)[1]):
                    if T.database_name(token.strip("'\",)")) == T.DEV_DATABASE:
                        found.append(token)
    return sorted(set(found)) + _env_defaults(tree)


def test_the_scan_can_see_a_planted_default():
    # Built at run time, so this file's own planted URLs are not string constants the scan reads.
    dev = T.DEV_DATABASE
    planted = f'DB = os.environ.get("DATABASE_URL_SYNC", "postgresql://u:p@localhost:5432/{dev}")\n'
    assert _dev_urls(planted) == [f"postgresql://u:p@localhost:5432/{dev}"]
    assert _dev_urls(f'X = "postgresql://u:p@localhost:5432/{dev}_test"\n') == []
    assert _dev_urls(f'X = "postgresql+asyncpg://u@h/{dev}?sslmode=disable"\n') == [
        f"postgresql+asyncpg://u@h/{dev}?sslmode=disable"]


def test_the_scan_sees_a_default_written_another_way():
    """Review R1: an f-string, a concatenation and a bare name default each name the dev database."""
    dev = T.DEV_DATABASE
    assert _dev_urls(f'U = f"postgresql://{{user}}:{{pw}}@localhost:5432/{dev}"\n') == [
        f"postgresql://{{}}:{{}}@localhost:5432/{dev}"]
    assert _dev_urls(f'U = "postgresql://u:p@localhost:5432/" + "{dev}"\n') == [
        f"postgresql://u:p@localhost:5432/{dev}"]
    assert _dev_urls(f'N = os.environ.get("POSTGRES_DB", "{dev}")\n') != []
    assert _dev_urls(f'U = f"postgresql://{{host}}/{dev}_test"\n') == []
    assert _dev_urls('N = os.environ.get("POSTGRES_DB", "mistudio_test")\n') == []


def test_no_test_module_defaults_to_the_dev_database():
    files = sorted(p for p in TESTS.rglob("*.py") if "__pycache__" not in p.parts)
    assert len(files) > 500, f"only {len(files)} test files found under {TESTS}: the scan drifted"
    offenders = {str(p.relative_to(TESTS)): urls for p in files if (urls := _dev_urls(p.read_text()))}
    assert not offenders, (
        f"test modules naming the DEVELOPMENT database `mistudio`: {offenders}. Use "
        "tests/support/test_databases.sync_test_url() or migrated_url()"
    )


@pytest.mark.parametrize("value, expected", [
    (None, T.LOCAL_TEST_SYNC_URL),
    (f"postgresql://postgres:pw@db:5432/{T.DEV_DATABASE}", "postgresql://postgres:pw@db:5432/mistudio_test"),
    ("postgresql+psycopg2://m:t@localhost:5432/mistudio_test",
     "postgresql+psycopg2://m:t@localhost:5432/mistudio_test"),
])
def test_sync_test_url_never_returns_the_dev_database(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("DATABASE_URL_SYNC", raising=False)
    else:
        monkeypatch.setenv("DATABASE_URL_SYNC", value)
    assert T.sync_test_url() == expected
    assert T.database_name(T.sync_test_url()) != T.DEV_DATABASE


def test_migrated_url_skips_loudly_or_fails_on_ci(monkeypatch):
    monkeypatch.delenv("SCHEMA_CHECK_DATABASE_URL", raising=False)
    monkeypatch.delenv("CI", raising=False)
    with pytest.raises(pytest.skip.Exception, match="SCHEMA_CHECK_DATABASE_URL"):
        T.migrated_url()
    monkeypatch.setenv("CI", "true")
    # ⚠ BaseException, then the TYPE: a `skip` escaping `pytest.raises(fail.Exception)` would SKIP
    # this test rather than fail it (control CDB.3 survived the first form).
    with pytest.raises(BaseException) as raised:
        T.migrated_url()
    assert raised.type is pytest.fail.Exception, raised.type
    monkeypatch.delenv("CI")
    monkeypatch.setenv("SCHEMA_CHECK_DATABASE_URL", f"postgresql://u:p@h:5432/{T.DEV_DATABASE}")
    with pytest.raises(pytest.skip.Exception, match="DEVELOPMENT"):
        T.migrated_url()
    monkeypatch.setenv("SCHEMA_CHECK_DATABASE_URL", "postgresql://u:p@h:5432/schema_guard")
    assert T.migrated_url() == "postgresql://u:p@h:5432/schema_guard"


# ── the app's own engines (review R1) ────────────────────────────────────────────────────────


def test_dev_database_redirect_rewrites_only_the_dev_database():
    dev = T.DEV_DATABASE
    env = {"DATABASE_URL": f"postgresql+asyncpg://u:p@h:5432/{dev}"}
    dotenv = {"DATABASE_URL_SYNC": f"postgresql://u:p@h:5432/{dev}?sslmode=disable",
              "DATABASE_URL": "postgresql+asyncpg://u:p@h:5432/ignored_because_env_wins"}
    assert T.dev_database_redirect(env, dotenv) == {
        "DATABASE_URL": "postgresql+asyncpg://u:p@h:5432/mistudio_test",
        "DATABASE_URL_SYNC": "postgresql://u:p@h:5432/mistudio_test?sslmode=disable",
    }
    assert T.dev_database_redirect({"DATABASE_URL": "postgresql://u@h/other"}, {}) == {}


def test_the_apps_own_engines_never_open_the_dev_database(tmp_path):
    """`SyncSessionLocal` and `AsyncSessionLocal` are built at import time from the settings, which a
    plain shell reads from `backend/.env`. Run a fresh interpreter whose environment AND `.env` name
    the dev database, import the test configuration the way pytest does, and read the URLs the app's
    engines were built with. Nothing connects: engines are lazy."""
    import os
    import subprocess
    import sys

    backend = TESTS.parent
    dev = T.DEV_DATABASE
    env = {k: v for k, v in os.environ.items() if k not in T.URL_VARS
           and k != "PYTEST_XDIST_WORKER"}
    env["DATABASE_URL"] = f"postgresql+asyncpg://u:p@127.0.0.1:9/{dev}"
    env["PYTHONPATH"] = str(backend)
    probe = ("import tests.conftest\n"
             "from src.core import database as d\n"
             "print(d.sync_engine.url.database, d.engine.url.database)\n")
    out = subprocess.run([sys.executable, "-c", probe], cwd=backend, env=env, capture_output=True,
                         text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    names = out.stdout.split()[-2:]
    assert names == [T.TEST_DATABASE, T.TEST_DATABASE], names
