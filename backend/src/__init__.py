"""miStudio backend package.

A bare ``postgresql://`` URL means psycopg2 here, as it did under SQLAlchemy 2.0.

SQLAlchemy 2.1 made psycopg (v3) the default driver for a URL that names none, and this
backend installs only psycopg2 (``psycopg2-binary`` in requirements.txt). Without this line
every bare URL — the sync engine Celery workers build from DATABASE_URL, the schema guards'
SCHEMA_CHECK_DATABASE_URL, ``python -m src.db.schema_parity --url postgresql://...``, an
operator's secret written the old way — fails with ``No module named 'psycopg'``, and the
Celery workers die at import. Registering the dialect here, in the package every entry point
imports first, keeps the old meaning for all of them; an explicit ``+psycopg2`` /
``+asyncpg`` / ``+psycopg`` driver in a URL is unaffected. Guarded by
tests/unit/test_bare_postgres_url_is_psycopg2.py.
"""

from sqlalchemy.dialects import registry as _registry

_registry.register("postgresql", "sqlalchemy.dialects.postgresql.psycopg2", "PGDialect_psycopg2")
