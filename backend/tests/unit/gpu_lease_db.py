"""Real-Postgres helpers for the multi-GPU Phase 3 lease tests.

A lease is refused by a primary key and guarded by row locks; a fake session
reproduces neither. These helpers create a database (named per test module, or
``MISTUDIO_GPU_LEASE_TEST_DB`` so parallel agents keep apart) holding ONLY the
lease table, and FAIL — never skip — when Postgres is unreachable.
"""

from __future__ import annotations

import contextlib
import os

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from src.models.gpu_lease import GpuLease


def _server_url() -> str:
    from tests.support.test_databases import sync_test_url

    return sync_test_url().rsplit("/", 1)[0]       # never the dev database (OSD-57)


def lease_db_name(default_db: str, env=None) -> str:
    """The lease database for THIS pytest process: the module's (or overridden) name,
    suffixed with the xdist worker id.

    ``lease_engine`` DROPS and recreates ``gpu_leases``. ``test_steering_gpu_placement``'s
    ``leased_node`` fixture is imported by two other files, so under ``--dist loadfile``
    three files on three workers shared one database and dropped the table under each
    other (CI race, 2026-10-19). Suffixing per worker keeps every worker on its own
    database, and does the same for a ``MISTUDIO_GPU_LEASE_TEST_DB`` override.
    """
    env = os.environ if env is None else env
    name = env.get("MISTUDIO_GPU_LEASE_TEST_DB", default_db)
    worker = env.get("PYTEST_XDIST_WORKER", "")
    return f"{name}_{worker}" if worker else name


def lease_engine(default_db: str):
    name = lease_db_name(default_db)
    admin = create_engine(_server_url() + "/postgres", isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        if not conn.execute(text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": name}).scalar():
            conn.execute(text(f'CREATE DATABASE "{name}"'))
    admin.dispose()
    engine = create_engine(_server_url() + "/" + name)
    GpuLease.__table__.drop(engine, checkfirst=True)
    GpuLease.__table__.create(engine)
    return engine


def session_factory(engine):
    """A ``get_sync_db``-shaped factory: each call opens an independent session."""
    make = sessionmaker(bind=engine, expire_on_commit=False)

    @contextlib.contextmanager
    def session():
        db = make()
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    return session


def clear(engine) -> None:
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM gpu_leases"))


def lease_rows(engine) -> list:
    with engine.connect() as conn:
        return [dict(row._mapping) for row in conn.execute(text("SELECT * FROM gpu_leases ORDER BY gpu_uuid"))]
