"""OSD-30 — the circuit list filters and pages in SQL, not in Python.

`edge_type` was applied by the endpoint as a list comprehension over EVERY
circuit, and the page was then sliced out of that list: a 50-row page cost a
full-table read that grew with the table.

Run against a real Postgres, because the predicate is JSONB containment
(`edges @> '[{"type": …}]'`) and whether that matches is a database question. A
mock would only prove the query compiles.
"""
import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.models.circuit import Circuit
from src.services.circuit_service import CircuitService


def _circuit(cid: str, *, edge_types=(), promoted=False, rung=0,
             granularity="feature") -> Circuit:
    return Circuit(
        id=cid, name=cid, granularity=granularity, promoted=promoted, rung=rung,
        members=[{"feature_id": "f1", "layer": 13}],
        edges=[{"type": t, "source": "f1", "target": "f2"} for t in edge_types],
    )


@pytest.fixture
async def session(async_engine):
    maker = async_sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with maker() as s:
        s.add_all([
            _circuit("crc_a", edge_types=("computed",), promoted=True, rung=2),
            _circuit("crc_b", edge_types=("persistence",), promoted=True, rung=3),
            _circuit("crc_c", edge_types=("computed", "persistence")),
            _circuit("crc_d", edge_types=()),
            _circuit("crc_e", edge_types=("attention_mediated",), granularity="cluster"),
        ])
        await s.commit()
        yield s


class TestTheEdgeTypeFilterIsTheDatabases:

    async def test_it_matches_a_circuit_that_has_that_edge(self, session):
        found = await CircuitService.list(session, edge_type="persistence")
        assert {c.id for c in found} == {"crc_b", "crc_c"}

    async def test_it_excludes_a_circuit_with_no_edges_at_all(self, session):
        found = await CircuitService.list(session, edge_type="computed")
        assert "crc_d" not in {c.id for c in found}

    async def test_a_circuit_with_several_edges_matches_each_of_them(self, session):
        for edge_type in ("computed", "persistence"):
            found = await CircuitService.list(session, edge_type=edge_type)
            assert "crc_c" in {c.id for c in found}

    async def test_an_unmatched_type_returns_nothing_rather_than_everything(self, session):
        """A predicate that silently fails open is worse than one that errors."""
        assert await CircuitService.list(session, edge_type="nonexistent") == []

    async def test_it_composes_with_the_other_filters(self, session):
        found = await CircuitService.list(
            session, edge_type="persistence", promoted=True, min_rung=3
        )
        assert {c.id for c in found} == {"crc_b"}


class TestTheCountMatchesTheFilter:

    async def test_the_count_applies_the_edge_filter_too(self, session):
        assert await CircuitService.count(session, edge_type="persistence") == 2

    async def test_the_count_ignores_pagination(self, session):
        page = await CircuitService.list(session, edge_type="computed", limit=1)
        assert len(page) == 1
        assert await CircuitService.count(session, edge_type="computed") == 2

    async def test_an_unfiltered_count_sees_every_row(self, session):
        assert await CircuitService.count(session) == 5


class TestPaginationHappensInTheQuery:

    async def test_a_limit_returns_at_most_that_many(self, session):
        assert len(await CircuitService.list(session, limit=2)) == 2

    async def test_an_offset_walks_the_same_ordering_without_repeats(self, session):
        first = await CircuitService.list(session, limit=2, offset=0)
        second = await CircuitService.list(session, limit=2, offset=2)
        assert not ({c.id for c in first} & {c.id for c in second})

    async def test_walking_every_page_yields_every_row_exactly_once(self, session):
        seen = []
        for offset in (0, 2, 4):
            seen += [c.id for c in await CircuitService.list(session, limit=2, offset=offset)]
        assert sorted(seen) == ["crc_a", "crc_b", "crc_c", "crc_d", "crc_e"]

    async def test_no_limit_still_returns_everything(self, session):
        assert len(await CircuitService.list(session)) == 5
