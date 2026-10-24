"""A `postgresql://` URL with no driver still means psycopg2 (SQLAlchemy 2.1, 2026-10-22).

SQLAlchemy 2.1 made psycopg (v3) the default driver for a PostgreSQL URL that names none. This
backend installs only psycopg2 (`psycopg2-binary`), so under 2.1 every such URL fails at
`create_engine` with `No module named 'psycopg'`. Found on the first run after the bump: the
Celery workers' sync engine (built at import of `src.core.database` from DATABASE_URL with
`+asyncpg` stripped) took down `alembic upgrade head`, and with it every worker.

Two fixes, each pinned here:
* `src/__init__.py` registers psycopg2 as the dialect for a bare `postgresql://`, for every URL
  an operator or a test writes the old way (SCHEMA_CHECK_DATABASE_URL, `schema_parity --url`);
* `src.core.database` names `+psycopg2` when it derives the sync URL, so the workers' engine
  does not depend on the registry at all.
"""

from __future__ import annotations

import sqlalchemy as sa

import src  # noqa: F401 — the registration under test runs at package import


def test_a_bare_postgresql_url_resolves_to_psycopg2():
    engine = sa.create_engine("postgresql://user:pw@localhost:1/db")
    assert engine.dialect.driver == "psycopg2"


def test_an_explicit_driver_is_left_alone():
    assert sa.engine.make_url("postgresql+asyncpg://u@h/d").get_dialect().driver == "asyncpg"


def test_the_workers_sync_engine_names_psycopg2_itself():
    from src.core import database

    assert database.sync_engine.url.drivername == "postgresql+psycopg2"
    assert database.sync_engine.dialect.driver == "psycopg2"
