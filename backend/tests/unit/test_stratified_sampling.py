"""Which stored examples reach the judge.

WHY THIS EXISTS
---------------
Every retrieval in this codebase was `ORDER BY max_activation DESC LIMIT
max_examples`, so a feature's label was an inference from the extreme upper tail
of its activation distribution: the top 10 of a stored top-100, or of a stored
top-20 on the eb48 extraction. A feature that reads as "legal language" in its
top decile may be "formal register" across its real range, and nothing in the
system could show that.

`example_sampling='stratified'` spreads the same `max_examples` budget across
whatever is stored. It needs no schema change and no re-extraction, because rows
already carry `max_activation`.

THE SQL IS TESTED AGAINST A REAL POSTGRES. The band arithmetic is integer
division inside a window function; reasoning about it is not evidence, and a
driver that binds `:max_examples` as numeric would make every band fractional
and silently collapse the LEAST() clamp.

MUTATION CONTROLS:
  C67 fixed-width bands instead of proportional
       -> test_stratified_spans_the_whole_stored_set
  C68 ORDER BY b.rank DESC in the band pick
       -> test_rank_one_is_always_selected
  C69 drop the GREATEST(stored_n, 1) guard
       -> SURVIVES, and correctly so. COUNT(*) OVER (PARTITION BY ...) is at
          least 1 in any partition that exists, so the divisor is never zero
          and the branch is unreachable. Recorded in the SQL's own comment
          rather than asserted here — a test that appeared to cover it would
          be claiming coverage it does not have.
  C69b under-supply: a feature with fewer stored rows than K
       -> test_a_feature_with_fewer_rows_than_k_returns_all_of_them
  C70 make 'top_k' dispatch to the stratified SQL
       -> test_top_k_returns_the_head
  C71 drop rank/stored_n from the top_k output
       -> test_both_strategies_return_the_same_row_shape
"""

import pytest
from sqlalchemy import text
from sqlalchemy.pool import NullPool

from src.core.config import settings

from src.services.labeling_service import (
    STRATIFIED_EXAMPLES_SQL,
    TOP_K_EXAMPLES_SQL,
    examples_sql,
)


@pytest.fixture
def db_session(async_engine):
    """A SYNC session against the same test database.

    The retrieval under test is the sync one — it is what the Celery worker
    calls — and the SQL is the point of these tests, so exercising it through
    the async twin would test a different code path than production runs.

    Depends on `async_engine` purely for ordering: that fixture is what creates
    the tables and the PG enums.
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    url = str(settings.database_url_sync)
    if "postgresql" in url and "test" not in url:
        url = url.rsplit("/", 1)[0] + "/mistudio_test"

    engine = create_engine(url, poolclass=NullPool)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        # NO CLEANUP DELETES.
        #
        # `async_engine` is function-scoped and runs `Base.metadata.drop_all`
        # plus an enum drop on teardown, so every row here is removed anyway.
        # Deleting first was worse than redundant: this sync session holds locks
        # on the same tables, and racing them against that drop produced three
        # different failures depending on ordering — a duplicate pg_type key, a
        # missing feature_activations relation, and an outright DeadlockDetected.
        #
        # Two files defining this fixture passed individually and failed
        # together, so "the suite is green" was order-dependent. Closing and
        # disposing promptly is the whole job.
        session.rollback()
        session.close()
        engine.dispose()


@pytest.fixture
def stored(db_session):
    """Insert N activation rows for one feature, strongest first.

    Built through the ORM, not raw INSERTs: `features` carries a dozen NOT NULL
    columns with Python-side defaults, and a hand-written INSERT has to restate
    every one of them — which is a fixture that breaks whenever the schema grows
    rather than when the behaviour changes.

    Activations descend so `rank` is a known function of the row: rank r has
    activation `1.0 - r/1000`, which makes an assertion on ranks readable.
    """
    from src.models.extraction_job import ExtractionJob
    from src.models.feature import Feature
    from src.models.feature_activation import FeatureActivation

    # `features.extraction_job_id` is a real FK, so the parent must exist.
    if not db_session.get(ExtractionJob, "extr_strat_test"):
        db_session.add(ExtractionJob(
            id="extr_strat_test", config={}, features_extracted=0,
        ))
        db_session.commit()

    def _make(feature_id: str, n: int):
        db_session.add(Feature(
            id=feature_id,
            extraction_job_id="extr_strat_test",
            neuron_index=0,
            name=feature_id,
            activation_frequency=0.1,
            max_activation=1.0,
            mean_activation=0.5,
            interpretability_score=0.5,
        ))
        db_session.flush()
        for r in range(1, n + 1):
            db_session.add(FeatureActivation(
                feature_id=feature_id,
                sample_index=r,
                max_activation=1.0 - r / 1000.0,
                tokens=["a", "p", "b"],
                activations=[0.1, 0.5, 0.1],
                prefix_tokens=["a"],
                prime_token="p",
                suffix_tokens=["b"],
                prime_activation_index=1,
            ))
        db_session.commit()
        return feature_id
    return _make


def _ranks(db_session, sql, feature_id, k):
    rows = db_session.execute(
        text(sql), {"feature_ids": [feature_id], "max_examples": k}
    ).fetchall()
    return [r.rank for r in rows]


class TestStratifiedSpansTheStoredSet:
    def test_stratified_spans_the_whole_stored_set(self, db_session, stored):
        """C67. 100 stored rows, K=10 -> one from each tenth, not the head.

        This is the entire point: the judge must see the feature across its
        stored range, not ten samples from the top 10%.
        """
        fid = stored("feat_strat_100", 100)
        ranks = _ranks(db_session, STRATIFIED_EXAMPLES_SQL, fid, 10)

        assert ranks == [1, 11, 21, 31, 41, 51, 61, 71, 81, 91], (
            f"stratified returned {ranks}; proportional bands over 100 rows at "
            f"K=10 must step by 10"
        )

    def test_a_narrower_extraction_still_stratifies(self, db_session, stored):
        """The eb48 extraction stores 20 rows per feature, not 100.

        The band is computed from the feature's OWN stored count, so the same
        SQL adapts with no version branch and no policy lookup.
        """
        fid = stored("feat_strat_20", 20)
        ranks = _ranks(db_session, STRATIFIED_EXAMPLES_SQL, fid, 10)

        assert ranks == [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]

    def test_rank_one_is_always_selected(self, db_session, stored):
        """C68. The peak example may never be dropped.

        `max(returned max_activation)` must still equal `features.max_activation`,
        because the percent-of-peak display normalises against it. A band pick
        that took the WEAKEST of each band would quietly break that.
        """
        for n in (4, 17, 20, 63, 100):
            fid = stored(f"feat_peak_{n}", n)
            rows = db_session.execute(
                text(STRATIFIED_EXAMPLES_SQL),
                {"feature_ids": [fid], "max_examples": 10},
            ).fetchall()
            assert rows[0].rank == 1, (
                f"with {n} stored rows the peak example was not returned"
            )
            assert max(float(r.max_activation) for r in rows) == pytest.approx(
                1.0 - 1 / 1000.0
            )

    def test_a_feature_with_fewer_rows_than_k_returns_all_of_them(
        self, db_session, stored
    ):
        """C69. Under-supply is normal, not an error.

        A feature that fired only four times has four rows. It must come back
        with all four and no duplicates — an empty band must not repeat its
        neighbour, and a division by a zero count must not raise.
        """
        fid = stored("feat_strat_4", 4)
        rows = db_session.execute(
            text(STRATIFIED_EXAMPLES_SQL),
            {"feature_ids": [fid], "max_examples": 10},
        ).fetchall()

        ranks = [r.rank for r in rows]
        assert ranks == [1, 2, 3, 4]
        assert len(set(r.sample_index for r in rows)) == 4, "duplicate rows"

    def test_a_single_stored_row_is_returned(self, db_session, stored):
        """The degenerate case. GREATEST(stored_n, 1) exists for this."""
        fid = stored("feat_strat_1", 1)
        ranks = _ranks(db_session, STRATIFIED_EXAMPLES_SQL, fid, 10)
        assert ranks == [1]


class TestTopKIsUnchanged:
    def test_top_k_returns_the_head(self, db_session, stored):
        """C70. The historical behaviour must be exactly preserved.

        Existing templates default to `top_k`, and a change to what they show
        the judge would invalidate every comparison against past runs.
        """
        fid = stored("feat_topk_100", 100)
        ranks = _ranks(db_session, TOP_K_EXAMPLES_SQL, fid, 10)
        assert ranks == list(range(1, 11))

    def test_both_strategies_return_the_same_row_shape(self, db_session, stored):
        """C71. A consumer must not be able to tell which strategy ran.

        If the row dict's keys depended on the strategy, a downstream branch
        could behave differently per arm — reintroducing exactly the hidden
        variable this change exists to isolate.
        """
        fid = stored("feat_shape", 30)
        params = {"feature_ids": [fid], "max_examples": 5}
        a = db_session.execute(text(TOP_K_EXAMPLES_SQL), params).fetchall()
        b = db_session.execute(text(STRATIFIED_EXAMPLES_SQL), params).fetchall()

        assert set(a[0]._mapping) == set(b[0]._mapping)
        for key in ("rank", "stored_n"):
            assert key in a[0]._mapping, f"top_k does not report {key}"
            assert key in b[0]._mapping, f"stratified does not report {key}"
        assert a[0].stored_n == 30 and b[0].stored_n == 30


class TestTheStrategySelector:
    def test_an_unknown_strategy_fails_closed_to_top_k(self):
        """A template from a newer build must still be able to label.

        Failing closed to the historical behaviour is the conservative reading;
        raising would take a whole extraction offline over one unknown string.
        """
        assert examples_sql("no_such_strategy") == TOP_K_EXAMPLES_SQL
        assert examples_sql(None) == TOP_K_EXAMPLES_SQL
        assert examples_sql("") == TOP_K_EXAMPLES_SQL

    def test_the_known_strategies_select_their_own_sql(self):
        assert examples_sql("top_k") == TOP_K_EXAMPLES_SQL
        assert examples_sql("stratified") == STRATIFIED_EXAMPLES_SQL
        assert TOP_K_EXAMPLES_SQL != STRATIFIED_EXAMPLES_SQL
