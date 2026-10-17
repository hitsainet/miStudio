"""A performance index must never be able to take the API down.

WHY THIS EXISTS
---------------
`docker-entrypoint.sh` runs `alembic upgrade heads` BEFORE uvicorn, and the k8s
startupProbe allows 30 x 10s = 300 seconds. Any migration that can exceed that
does not run slowly — it produces a permanent CrashLoopBackOff, and under
`strategy: Recreate` the old pod is already gone, so the API is 503 throughout.

`CREATE INDEX CONCURRENTLY` on the 69 GB / 35.6M-row `feature_activations`
partition is such a migration: two heap passes plus a sort at the stock 64 MB
`maintenance_work_mem`, and it waits out every transaction that can see the
table — including the three Celery containers sharing the pod, which a backend
restart does not stop.

Three failure modes compose, and fixing only some makes it worse:

  1. A killed build leaves `indisvalid = false`; `IF NOT EXISTS` then SKIPS it
     forever while Alembic stamps success. A silent permanent no-op.
  2. Detecting and rebuilding turns that into an infinite loop — drop, rebuild
     from zero, killed, repeat.
  3. Raising turns the loop into a hard outage and blocks every unrelated
     migration behind it.

It came up on 2026-09-10 only because every labeling job had been cancelled
minutes before, so nothing held a conflicting transaction. That is luck.

MUTATION CONTROLS:
  C97 remove the size check so a huge table builds inline
       -> test_a_large_table_is_not_built_inline
  C98 rebuild immediately after dropping an invalid index
       -> test_an_invalid_index_on_a_large_table_is_not_rebuilt
  C99 skip the invalid-index detection (plain IF NOT EXISTS)
       -> test_an_invalid_index_is_dropped_not_skipped
  C100 raise instead of stamping when the table is large
       -> test_a_large_table_does_not_raise
"""

from contextlib import contextmanager

import pytest

from src.db.index_build import (
    drop_index_concurrently,
    ensure_index_concurrently,
)

INDEX = "idx_probe"
TABLE = "probe_table"
COLUMNS = "(a, b DESC, id)"


class _FakeResult:
    def __init__(self, value): self._value = value
    def scalar(self): return self._value
    def first(self): return self._value


class _FakeConn:
    """Answers the three catalogue questions and records DDL.

    Deliberately dispatches on the SQL text the way a real connection would
    have to be given it — a stand-in that accepted anything would let a
    malformed query pass here and fail in production.
    """

    def __init__(self, *, exists=True, index_state="absent", size_bytes=1_000_000):
        self.exists = exists
        self.index_state = index_state
        self.size_bytes = size_bytes
        self.ddl = []

    def exec_driver_sql(self, sql):
        if "to_regclass" in sql:
            return _FakeResult(self.exists)
        if "indisvalid" in sql:
            if self.index_state == "absent":
                return _FakeResult(None)
            return _FakeResult((self.index_state == "valid",))
        if "pg_total_relation_size" in sql:
            return _FakeResult(self.size_bytes)
        raise AssertionError(f"unexpected catalogue query: {sql}")


class _FakeOp:
    """Records DDL, its ORDER, and whether it ran inside an autocommit block.

    The first version yielded from `autocommit_block` without recording it and
    appended DDL to an unordered check. Three classes of defect walked through:

      * replacing `with op.get_context().autocommit_block():` with `if True:`
        passed — in production that is
        "CREATE INDEX CONCURRENTLY cannot run inside a transaction block",
        i.e. every deploy's migration fails and the API exits 1;
      * swapping the index and table identifiers in the CREATE passed, because
        only `startswith`/`in` were asserted;
      * moving the DROP after the CREATE passed, because membership was checked
        with `any(...)` — net effect: nothing built, revision stamped.
    """

    def __init__(self, conn):
        self._conn = conn
        self.autocommit_depth = 0
        self.entered_autocommit = 0

    def get_bind(self):
        return self._conn

    def get_context(self):
        outer = self

        class _Ctx:
            @contextmanager
            def autocommit_block(self_inner):
                outer.autocommit_depth += 1
                outer.entered_autocommit += 1
                try:
                    yield
                finally:
                    outer.autocommit_depth -= 1

        return _Ctx()

    def execute(self, sql):
        if self.autocommit_depth == 0:
            raise AssertionError(
                f"DDL ran OUTSIDE an autocommit block: {sql!r}. PostgreSQL "
                f"rejects CREATE/DROP INDEX CONCURRENTLY inside a transaction, "
                f"so this fails every deploy."
            )
        self._conn.ddl.append(sql)
        # Model the catalogue moving, so the post-build check is exercised
        # rather than reading a frozen "invalid".
        if sql.startswith("CREATE INDEX"):
            self._conn.index_state = "valid"
        elif sql.startswith("DROP INDEX"):
            self._conn.index_state = "absent"


def _run(**kwargs):
    conn = _FakeConn(**kwargs)
    op = _FakeOp(conn)
    ensure_index_concurrently(
        op, index=INDEX, table=TABLE, columns=COLUMNS,
        revision="testrev", inline_build_max_bytes=2_000_000_000,
    )
    return conn.ddl


class TestSmallTablesBuildInline:
    def test_a_small_table_builds_the_index(self):
        """Negative control for every test below.

        A policy that never builds anything satisfies all the safety tests and
        ships no index at all.
        """
        ddl = _run(size_bytes=1_000_000)
        creates = [s for s in ddl if s.startswith("CREATE INDEX")]
        assert len(creates) == 1
        # THE WHOLE STATEMENT. Asserting `startswith` and two substrings let a
        # swapped index/table pair through — the same shape of defect this arc
        # found elsewhere and then reproduced in its own guard.
        assert creates[0] == (
            f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {INDEX} "
            f"ON {TABLE} {COLUMNS}"
        ), creates[0]

    def test_the_build_runs_inside_an_autocommit_block(self):
        """CONCURRENTLY cannot run in a transaction; the block is not optional.

        `_FakeOp.execute` raises if the depth is zero, so this asserts the
        block was entered — replacing it with `if True:` fails every test that
        emits DDL rather than passing silently.
        """
        conn = _FakeConn(size_bytes=1_000_000)
        op = _FakeOp(conn)
        ensure_index_concurrently(
            op, index=INDEX, table=TABLE, columns=COLUMNS,
            revision="testrev", inline_build_max_bytes=2_000_000_000,
        )
        assert op.entered_autocommit >= 1
        assert op.autocommit_depth == 0, "an autocommit block was left open"

    def test_a_healthy_build_is_not_dropped_afterwards(self):
        """The post-build check must not fire on success.

        The stand-in used to freeze `index_state`, so every small-table test
        ran the corrective drop by accident and the check itself was untested.
        """
        ddl = _run(size_bytes=1_000_000)
        assert [s.split()[0:2] for s in ddl] == [["CREATE", "INDEX"]], ddl

    def test_a_valid_index_is_left_alone(self):
        """Re-running a migration must not rebuild what is already there."""
        assert _run(index_state="valid", size_bytes=1_000_000) == []

    def test_a_missing_table_is_not_an_error(self):
        """`conftest` builds the schema with create_all, so partitions are
        genuinely absent in tests. Skipping is correct, raising is not."""
        assert _run(exists=False) == []


class TestLargeTablesAreLeftToAnOperator:
    def test_a_large_table_is_not_built_inline(self):
        """C97. The whole point: above the limit, report and stamp."""
        ddl = _run(size_bytes=69_000_000_000)
        creates = [s for s in ddl if s.startswith("CREATE INDEX")]
        assert not creates, (
            f"a 69 GB table was indexed inside the startup probe; "
            f"a build that outlives the probe loops forever and takes the API "
            f"down with it"
        )

    def test_a_large_table_does_not_raise(self):
        """C100. Correctness does not depend on this index; speed does.

        Raising would block every other migration behind a PERFORMANCE
        optimisation, including ones with nothing to do with indexing.
        """
        _run(size_bytes=69_000_000_000)  # must simply return

    def test_an_invalid_index_on_a_large_table_is_not_rebuilt(self):
        """C98. Drop the corpse, do NOT start again from zero.

        Rebuilding is what converts a silent no-op into an infinite
        CrashLoopBackOff: drop, rebuild, get killed at 300s, repeat.
        """
        ddl = _run(index_state="invalid", size_bytes=69_000_000_000)

        drops = [s for s in ddl if s.startswith("DROP INDEX")]
        creates = [s for s in ddl if s.startswith("CREATE INDEX")]
        assert len(drops) == 1, "the dead index was not cleared"
        assert not creates, (
            "a large index was rebuilt immediately after being dropped — this "
            "is the restart loop, not a recovery"
        )

    def test_a_never_analysed_table_is_still_sized_correctly(self):
        """The guard must not be fooled by missing statistics.

        An earlier version read `pg_class.reltuples`, which is a PLANNER
        ESTIMATE maintained by ANALYZE — and is **-1 on PostgreSQL 14 for a
        table that has never been analysed**. That is exactly the state of a
        freshly `pg_restore`d database, and `-1 > 5_000_000` is False, so the
        guard would have built a 69 GB index inline on precisely the deployment
        it exists to protect.

        On-disk size needs no statistics and cannot be stale.
        """
        ddl = _run(size_bytes=69_000_000_000)
        assert not [s for s in ddl if s.startswith("CREATE INDEX")]

    def test_an_invalid_index_is_dropped_not_skipped(self):
        """C99. `IF NOT EXISTS` alone treats a dead index as done.

        The planner never uses an invalid index, so skipping it stamps success
        over a permanent no-op — a silent fallback, which this codebase has
        recorded as worse than a crash.
        """
        ddl = _run(index_state="invalid", size_bytes=1_000_000)

        # ORDERED, not membership. `any(DROP)` and `any(CREATE)` both hold if
        # the drop runs AFTER the create — net effect nothing built, revision
        # stamped.
        kinds = [s.split()[0] for s in ddl]
        assert kinds == ["DROP", "CREATE"], (
            f"expected a drop then a rebuild, got {kinds}: an INVALID index "
            f"left in place is skipped by every future run, and a drop after "
            f"the create leaves nothing behind"
        )


class TestTheDropIsSchemaQualified:
    def test_ddl_names_the_public_schema(self):
        """An unqualified relname match can find another schema's index and
        then drop the wrong object."""
        ddl = _run(index_state="invalid", size_bytes=1_000_000)
        drops = [s for s in ddl if s.startswith("DROP INDEX")]
        assert all("public." in s for s in drops), drops

    def test_downgrade_is_a_no_op_when_absent(self):
        conn = _FakeConn(index_state="absent")
        drop_index_concurrently(_FakeOp(conn), index=INDEX)
        assert conn.ddl == []

    def test_downgrade_drops_a_present_index(self):
        conn = _FakeConn(index_state="valid")
        drop_index_concurrently(_FakeOp(conn), index=INDEX)
        assert len(conn.ddl) == 1 and conn.ddl[0].startswith("DROP INDEX")
