"""OSD-14 — hard-negative donors were the 20 lexicographically-first ids.

`ORDER BY fti.feature_id LIMIT :donor_limit` is deterministic, which the
surrounding comment correctly insists on, but it drew the same narrow slice of
the dictionary for every feature. Ordering by md5 of the id keeps determinism and
drops the bias.
"""
import re

import pytest
from sqlalchemy import text

from src.services.labeling_detection_scorer import _HARD_NEGATIVES_SQL


def _donors_cte() -> str:
    sql = str(_HARD_NEGATIVES_SQL)
    return sql.split("donors AS", 1)[1].split("ranked AS", 1)[0]


class TestTheDonorDrawIsHashed:

    def test_the_donor_order_is_md5_of_the_id(self):
        assert re.search(r"ORDER BY\s+md5\(fti\.feature_id", _donors_cte()), (
            "the donor CTE must order by a hash of the id, not by the id"
        )

    def test_it_no_longer_orders_by_the_raw_id(self):
        assert not re.search(r"ORDER BY\s+fti\.feature_id\s*\n", _donors_cte())

    def test_the_draw_is_still_bounded_and_deterministic(self):
        cte = _donors_cte()
        assert "LIMIT :donor_limit" in cte, "the draw must stay bounded"
        assert ":salt" in cte, (
            "the hash must be salted by the binding this query already carries, "
            "so two trials over one panel draw the same negatives and stay paired"
        )


class TestHashOrderIsNotAlphabetical:
    """Proven in Postgres over synthetic ids, so the claim is not just textual."""

    @pytest.mark.asyncio
    async def test_md5_order_differs_from_id_order(self, async_engine):
        ids = [f"feat_x_{i:05d}" for i in range(200)]
        async with async_engine.begin() as connection:
            hashed = (await connection.execute(
                text("SELECT fid FROM unnest(cast(:ids AS text[])) AS fid "
                     "ORDER BY md5(fid || 'salt') LIMIT 20"),
                {"ids": ids},
            )).scalars().all()
            plain = (await connection.execute(
                text("SELECT fid FROM unnest(cast(:ids AS text[])) AS fid "
                     "ORDER BY fid LIMIT 20"),
                {"ids": ids},
            )).scalars().all()

        assert plain == ids[:20], "sanity: plain order really is the first 20"
        assert hashed != plain, "the hash must not reproduce alphabetical order"
        assert len(set(hashed)) == 20
        # The bias this fixes: alphabetical never reaches the tail of the space.
        assert max(hashed) > ids[100], (
            "a hashed draw should reach past the first half of the id space"
        )

    @pytest.mark.asyncio
    async def test_the_same_salt_draws_the_same_donors(self, async_engine):
        ids = [f"feat_x_{i:05d}" for i in range(200)]
        async with async_engine.begin() as connection:
            first = (await connection.execute(
                text("SELECT fid FROM unnest(cast(:ids AS text[])) AS fid "
                     "ORDER BY md5(fid || 's1') LIMIT 20"),
                {"ids": ids},
            )).scalars().all()
            again = (await connection.execute(
                text("SELECT fid FROM unnest(cast(:ids AS text[])) AS fid "
                     "ORDER BY md5(fid || 's1') LIMIT 20"),
                {"ids": ids},
            )).scalars().all()
        assert first == again, "pairing across trials depends on this being stable"
