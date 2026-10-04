"""Building a large index without being able to take the API down.

WHY THIS MODULE EXISTS
----------------------
`docker-entrypoint.sh` runs `alembic upgrade heads` BEFORE uvicorn starts, and
the k8s startupProbe allows 300 seconds. Any migration that can take longer than
that does not merely run slowly — it produces a permanent CrashLoopBackOff, and
under `strategy: Recreate` the old pod is already gone, so the API is 503 for
the duration.

`CREATE INDEX CONCURRENTLY` on a 69 GB table is exactly such a migration. It
does two heap passes plus a sort at the stock 64 MB `maintenance_work_mem`, and
it additionally waits out every transaction that can see the table — including
the three Celery containers that share the pod and that a backend restart does
not stop.

Three failure modes have to be handled together, and handling only some of them
makes things worse rather than better:

1. A build killed mid-flight leaves `indisvalid = false`. `IF NOT EXISTS` then
   SKIPS it on every subsequent run, Alembic stamps the revision, and the
   planner never uses the index again. A silent, permanent no-op.
2. Detecting that and rebuilding turns the silent no-op into an infinite loop:
   drop, rebuild from zero, get killed, repeat.
3. Raising on failure turns the loop into a hard outage, and blocks every other
   migration behind it — including ones that have nothing to do with indexing.

THE RESOLUTION: a performance index is not a correctness requirement. Below a
size threshold, build it inline. Above it, report exactly what to run and stamp
the revision. Nothing is lied about, nothing loops, and the API comes up.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _relation_exists(conn, table: str) -> bool:
    """Whether the target table is present in `public`.

    A missing table is not an error: `tests/conftest.py` builds the schema with
    `Base.metadata.create_all`, so partitions genuinely do not exist there.
    """
    return conn.exec_driver_sql(
        f"SELECT to_regclass('public.{table}') IS NOT NULL"
    ).scalar()


def _index_state(conn, index: str) -> str:
    """'absent' | 'valid' | 'invalid', schema-qualified.

    Qualified deliberately: an unqualified `relname` match would find an index
    of the same name in another schema and then act on the wrong object.
    """
    row = conn.exec_driver_sql(
        "SELECT i.indisvalid FROM pg_class c "
        "JOIN pg_namespace n ON n.oid = c.relnamespace "
        "JOIN pg_index i ON i.indexrelid = c.oid "
        f"WHERE c.relname = '{index}' AND n.nspname = 'public'"
    ).first()
    if row is None:
        return "absent"
    return "valid" if row[0] else "invalid"


def _table_bytes(conn, table: str) -> int:
    """On-disk size, which needs no ANALYZE and cannot be stale.

    NOT `reltuples`. That is a planner estimate maintained by ANALYZE, and it is
    **-1 on PostgreSQL 14 for a table that has never been analysed** — which is
    exactly the state of a freshly `pg_restore`d database, the case where an
    accidental inline build hurts most. `-1 > 5_000_000` is False, so a
    row-count guard would have cheerfully built a 69 GB index inside the startup
    probe on precisely the deployment it exists to protect.

    Bytes are exact, are maintained by the storage layer rather than by
    statistics collection, and are a better proxy for build time anyway: build
    cost tracks pages read and sorted, not tuples.
    """
    value = conn.exec_driver_sql(
        f"SELECT pg_total_relation_size('public.{table}')"
    ).scalar()
    return int(value or 0)


def ensure_index_concurrently(
    op,
    *,
    index: str,
    table: str,
    columns: str,
    revision: str,
    inline_build_max_bytes: int,
) -> None:
    """Create `index` if that can be done without risking the startup budget."""
    conn = op.get_bind()

    if not _relation_exists(conn, table):
        logger.info(
            "[%s] %s absent (unpartitioned or minimal schema); skipping %s",
            revision, table, index,
        )
        return

    state = _index_state(conn, index)
    if state == "valid":
        logger.info("[%s] %s already present and valid", revision, index)
        return

    with op.get_context().autocommit_block():
        if state == "invalid":
            # Clear the corpse of an interrupted build. Do NOT immediately
            # rebuild a large one — that is the infinite loop.
            logger.warning(
                "[%s] %s exists but is INVALID (a previous CONCURRENTLY build "
                "was interrupted). Dropping it.", revision, index,
            )
            op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS public.{index}")

        # MEASURED AFTER THE DROP, NOT BEFORE.
        #
        # `pg_total_relation_size` includes indexes, and an `indisvalid=false`
        # index counts. So a build killed just under the limit leaves a partial
        # index whose bytes push the NEXT measurement over it — and the guard
        # then refuses forever, quoting a size the heap does not have. That is
        # the silent permanent no-op this module exists to prevent,
        # reintroduced by ordering alone.
        size_bytes = _table_bytes(conn, table)

        if size_bytes > inline_build_max_bytes:
            # STAMP AND REPORT, do not build and do not raise.
            logger.warning(
                "[%s] NOT building %s inline: %s is %.1f GB, above the %.1f GB "
                "inline limit. A CONCURRENTLY build at this size can exceed the "
                "container startup probe, and a killed build loops forever. The "
                "migration is stamped; queries still work, just more slowly. "
                "Build it out of band with the workers quiesced:\n"
                "    CREATE INDEX CONCURRENTLY %s ON %s %s;",
                revision, index, table,
                size_bytes / 1e9, inline_build_max_bytes / 1e9,
                index, table, columns,
            )
            return

        op.execute(
            f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {index} "
            f"ON {table} {columns}"
        )

    if _index_state(conn, index) == "invalid":
        # Small table, so this is a real failure rather than a timeout — and
        # leaving an invalid index behind would make the NEXT run skip it.
        with op.get_context().autocommit_block():
            op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS public.{index}")
        logger.error(
            "[%s] %s built INVALID on a small table and was dropped again; "
            "build it manually and investigate", revision, index,
        )


def drop_index_concurrently(op, *, index: str) -> None:
    """Symmetric downgrade. Absent is success."""
    conn = op.get_bind()
    if _index_state(conn, index) == "absent":
        return
    with op.get_context().autocommit_block():
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS public.{index}")
