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
    url = os.environ.get("DATABASE_URL_SYNC") or "postgresql://postgres:devpassword@localhost:5432/mistudio"
    return url.rsplit("/", 1)[0]


def lease_engine(default_db: str):
    name = os.environ.get("MISTUDIO_GPU_LEASE_TEST_DB", default_db)
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
