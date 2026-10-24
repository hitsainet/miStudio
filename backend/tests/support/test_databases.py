"""Which database a test may touch — never the local DEVELOPMENT database `mistudio`.

⚠ FIVE TEST MODULES DEFAULTED TO THE DEV DATABASE (found by the 039 6.4 review, fixed 2026-10-23,
OSD-57). Each read `DATABASE_URL_SYNC` and, when it was unset, fell back to
`postgresql://…/mistudio` — the operator's working database. Two of them DROP and recreate tables.
Run without the variable (a fresh shell, an editor's test runner), they pointed at real data. The
miStudio memory `test-database-and-suite-command` records what that cost once: 11 phantom failures,
`test_model_*` rows in the dev database, and a near-miss destructive reset.

The rule, in one place:
* `sync_test_url()` — `DATABASE_URL_SYNC` when it names a non-dev database, else the SAME server's
  `mistudio_test` (the redirect `conftest.async_engine` applies to `DATABASE_URL`), else the local
  default `mistudio_test`. Never `mistudio`.
* `migrated_url()` — `SCHEMA_CHECK_DATABASE_URL` (a database migrated to head). Unset → the test
  SKIPS LOUDLY naming the variable, and FAILS on CI (`CI=true`), where it is always set. It no longer
  falls back to `DATABASE_URL_SYNC` (which, unset, was the dev database).

* `apply_dev_database_redirect()` — called by `conftest.py` BEFORE `src` is imported. The APP's own
  engines (`src.core.database`: `AsyncSessionLocal`, `SyncSessionLocal`) are built at import time from
  `settings.database_url`, which a plain shell reads from `backend/.env` — the dev database. The
  `async_engine` fixture redirected only ITS engine, so every production path a test reached through
  `SyncSessionLocal` (039's identity backfill and `GET …/decision`, every Celery task body) ran
  against `mistudio` (review R1, 2026-10-23). The redirect rewrites `DATABASE_URL`/`DATABASE_URL_SYNC`
  in the process environment, which pydantic-settings prefers over `.env`.

`test_no_test_defaults_to_the_dev_database.py` holds every test module to this.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, MutableMapping, Optional

import pytest

#: The local development database. No test default may name it.
DEV_DATABASE = "mistudio"
#: The local test database (`conftest.async_engine` manages it).
TEST_DATABASE = "mistudio_test"
LOCAL_TEST_SYNC_URL = f"postgresql://postgres:devpassword@localhost:5432/{TEST_DATABASE}"


def database_name(url: str) -> str:
    return url.rsplit("/", 1)[-1].split("?")[0]


def sync_test_url() -> str:
    """A sync URL for the TEST database — never the dev one."""
    url = os.environ.get("DATABASE_URL_SYNC") or LOCAL_TEST_SYNC_URL
    if database_name(url) == DEV_DATABASE:
        url = url.rsplit("/", 1)[0] + "/" + TEST_DATABASE
    return url


def migrated_url() -> str:
    """`SCHEMA_CHECK_DATABASE_URL`, a database migrated to head — or a loud skip (a failure on CI)."""
    url = os.environ.get("SCHEMA_CHECK_DATABASE_URL")
    message = None
    if not url:
        message = ("SCHEMA GUARD SKIPPED LOUDLY: set SCHEMA_CHECK_DATABASE_URL to a database migrated to "
                   "head (never the dev database `mistudio`)")
    elif database_name(url) == DEV_DATABASE:
        message = ("SCHEMA_CHECK_DATABASE_URL names the DEVELOPMENT database `mistudio`; point it at a "
                   "scratch database migrated to head")
    if message is not None:
        if os.environ.get("CI") == "true":
            pytest.fail(message)
        pytest.skip(message)
    return url


#: The variables that name the database the app and the tests connect to.
URL_VARS = ("DATABASE_URL", "DATABASE_URL_SYNC")


def _renamed(url: str, name: str) -> str:
    base, rest = url.rsplit("/", 1)
    _old, sep, query = rest.partition("?")
    return f"{base}/{name}{sep}{query}"


def dev_database_redirect(env: Mapping[str, str],
                          dotenv: Mapping[str, Optional[str]]) -> dict[str, str]:
    """The overrides that point a test process naming the dev database at `mistudio_test` instead.

    Pure. `env` wins over `dotenv` (as it does for pydantic-settings); a variable naming any other
    database is left alone.
    """
    overrides: dict[str, str] = {}
    for var in URL_VARS:
        url = env.get(var) or dotenv.get(var)
        if url and database_name(url) == DEV_DATABASE:
            overrides[var] = _renamed(url, TEST_DATABASE)
    return overrides


def apply_dev_database_redirect(env: Optional[MutableMapping[str, str]] = None,
                                dotenv_path: Optional[Path] = None) -> dict[str, str]:
    """Apply `dev_database_redirect` to the process environment. MUST run before `src` is imported."""
    environment = os.environ if env is None else env
    path = Path(".env") if dotenv_path is None else dotenv_path
    values: Mapping[str, Optional[str]] = {}
    if path.is_file():
        from dotenv import dotenv_values

        values = dotenv_values(path)
    overrides = dev_database_redirect(environment, values)
    environment.update(overrides)
    return overrides
