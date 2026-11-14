"""New Neuronpedia rows name their creator miWorkbench (042 TPD-26; FTASKS 12.20).

The local push writes `owner` on "Model", `creatorName` on "SourceRelease" and "SourceSet"; the export's metadata
carries `source.tool`. Rows written before the rename keep `miStudio` (nothing is migrated, R1); new rows say
`miWorkbench`. The frozen Neuronpedia id segment `-mistudio-res-` (DR-07) is unchanged.

Through the real client methods on a recording connection (no Neuronpedia database): each INSERT's SQL and bound
values are read, and the creator literal must be the new name.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace


class _Conn:
    def __init__(self):
        self.executed: list[tuple[str, tuple]] = []

    async def fetchrow(self, *_a, **_k):
        return None

    async def fetchval(self, *_a, **_k):
        return None

    async def execute(self, sql, *args):
        self.executed.append((sql, args))


class _Pool:
    def __init__(self, conn):
        self.conn = conn

    @asynccontextmanager
    async def acquire(self):
        yield self.conn


def _client():
    from src.services.neuronpedia_local_service import NeuronpediaLocalClient

    client = NeuronpediaLocalClient("postgresql://unused/x")
    conn = _Conn()
    client._pool = _Pool(conn)  # noqa: SLF001
    return client, conn


def _inserts(conn, table):
    return [sql for sql, _ in conn.executed if f'INSERT INTO "{table}"' in sql]


def test_new_rows_carry_miworkbench():
    client, conn = _client()

    async def run():
        await client.create_model("m", "M", layers=2, creator_id="u1", neurons_per_layer=8)
        await client.create_source_release("m", "set", "u1", "d")
        await client.create_source_set("m", "set", "d", "u1")

    asyncio.run(run())
    for table in ("Model", "SourceRelease", "SourceSet"):
        sqls = _inserts(conn, table)
        assert len(sqls) == 1, (table, [s for s, _ in conn.executed])
        assert "'miWorkbench'" in sqls[0] and "'miStudio'" not in sqls[0], (table, sqls[0])

    from src.services.neuronpedia_export_service import NeuronpediaExportService

    sae = SimpleNamespace(id="sae_1", layer=12, n_features=16384, d_model=2048, architecture="standard",
                          hook_name=None, hook_point=None, hook_type="resid_post", name="s")
    service = NeuronpediaExportService.__new__(NeuronpediaExportService)
    meta = service._generate_metadata_json(sae, "org/model")  # noqa: SLF001
    assert meta["source"]["tool"] == "miWorkbench"
    assert "-mistudio-res-" in meta["neuronpedia_id"]  # DR-07, frozen
