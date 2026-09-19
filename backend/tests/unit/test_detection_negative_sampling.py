"""The negative-sampling SQL, exercised against a REAL PostgreSQL.

These queries were the untested surface of the scorer, and every defect Round 1
found lived here — all three were invisible to a mock, because a mock does not
implement `<> ALL`, md5 collisions, or window functions.

Mutation controls:
  C13 remove the None filter from `exclude`
       -> test_a_null_in_the_exclusion_list_does_not_silently_empty_the_draw
  C14 drop the '\\x1f' separators from the easy-negative ORDER BY
       -> test_the_easy_negative_sort_key_cannot_collide
  C15 order hard negatives by max_activation alone (no per-donor rank)
       -> test_hard_negatives_are_spread_across_donors
  C16 remove the sample_index exclusion from either query
       -> test_a_negative_is_never_one_of_the_targets_own_passages
"""

import os
import re
import uuid

import pytest
from sqlalchemy import create_engine, text

from src.services.labeling_detection_scorer import (
    _EASY_NEGATIVES_SQL,
    _HARD_NEGATIVES_SQL,
    sample_negatives,
)

DB = os.environ.get(
    "DATABASE_URL_SYNC", "postgresql://postgres:devpassword@localhost:5432/mistudio"
)


# Enum types several models declare with create_type=False, so
# Base.metadata.create_all will NOT make them. Mirrors tests/conftest.py; values
# must match the model definitions exactly.
_ENUMS = [
    ("export_status", ["pending", "computing", "packaging", "completed", "failed", "cancelled"]),
    ("label_source_enum", ["auto", "user", "llm", "local_llm", "openai", "enhanced_llm", "mcp_agent"]),
    ("analysis_type_enum", ["logit_lens", "correlations", "ablation", "nlp_analysis"]),
    ("extraction_status_enum", ["queued", "loading", "extracting", "saving", "completed", "failed", "cancelled"]),
]


@pytest.fixture(scope="module")
def engine():
    """A sync engine with the schema guaranteed present.

    Three earlier versions of this fixture each failed differently, and every
    failure was invisible locally:

    1. Connecting straight to DATABASE_URL_SYNC assumed the tables already
       existed — true of a dev database, ERRORs in CI.
    2. Calling create_all on a hand-picked subset missed the transitive FK
       closure (extraction_jobs -> trainings).
    3. create_all over the full metadata still failed on
       `type "export_status" does not exist`, because those enums are declared
       create_type=False.

    Enums first, then tables. Both are idempotent, so this is a no-op against a
    populated database.

    It SKIPS only when PostgreSQL itself is unreachable. A reachable database
    with a missing schema is a hard failure: an earlier version turned that into
    a skip, and five tests reported green while asserting nothing.
    """
    try:
        eng = create_engine(DB)
        with eng.connect() as c:
            c.execute(text("SELECT 1"))
    except Exception as exc:  # pragma: no cover - env dependent
        pytest.skip(f"PostgreSQL is unreachable: {exc}")

    from src.core.database import Base
    from src import models  # noqa: F401 - registers every table

    with eng.begin() as c:
        for name, values in _ENUMS:
            vals = ", ".join(f"'{v}'" for v in values)
            c.execute(text(
                f"DO $$ BEGIN CREATE TYPE {name} AS ENUM ({vals}); "
                f"EXCEPTION WHEN duplicate_object THEN NULL; END $$;"))
    Base.metadata.create_all(eng)
    return eng


@pytest.fixture
def conn(engine):
    """A transaction rolled back at the end — this test writes nothing."""
    with engine.connect() as c:
        trans = c.begin()
        try:
            yield c
        finally:
            trans.rollback()


class TestPostgresSemanticsWeDependOn:
    """These are assumptions about Postgres, not about our code.

    They are pinned because the whole sampler is built on them, and because two
    of them were WRONG in the first implementation.
    """

    def test_not_equal_all_keeps_everything_for_an_empty_array(self, conn):
        n = conn.execute(text(
            "SELECT count(*) FROM (VALUES (1),(2),(3)) t(x) "
            "WHERE x <> ALL(ARRAY[]::int[])"
        )).scalar()
        assert n == 3, "an empty exclusion list must not filter anything out"

    def test_a_null_inside_the_array_rejects_every_row(self, conn):
        """The hazard the None filter exists to prevent.

        This is Postgres behaviour, not ours: `x <> ALL(ARRAY[1,NULL])` is never
        true. If a NULL reached the bound array, the draw would return nothing
        and the feature would be silently unscored.
        """
        n = conn.execute(text(
            "SELECT count(*) FROM (VALUES (1),(2),(3)) t(x) "
            "WHERE x <> ALL(ARRAY[1,NULL]::int[])"
        )).scalar()
        assert n == 0, (
            "Postgres changed <> ALL semantics for NULL-containing arrays; the "
            "None filter in sample_negatives may no longer be needed"
        )

    def test_the_easy_negative_sort_key_cannot_collide(self, conn):
        """C14. Undelimited concatenation collides across pair boundaries.

        This asserts against the MODULE'S OWN query text. The first version built
        its own `SELECT md5(...)` literals and never referenced
        `_EASY_NEGATIVES_SQL`, so deleting the separators from the module left it
        green — it proved a property of Postgres's md5, which was never in doubt.
        """
        # Establish the hazard is real.
        collides = conn.execute(text(
            "SELECT md5('a1' || '' || '23') = md5('a12' || '' || '3')"
        )).scalar()
        assert collides is True, "the collision this separator guards against is gone"

        # Now extract the ORDER BY expression the module actually ships and
        # evaluate IT against the colliding pair.
        sql = str(_EASY_NEGATIVES_SQL)
        m = re.search(r"ORDER BY (md5\(.*?\))\n", sql)
        assert m, f"could not locate the ORDER BY expression in:\n{sql}"
        expr = m.group(1)

        # The salt must be EMPTY here. A non-empty salt sits between the two
        # variable-length fields and acts as a de-facto separator all by itself,
        # so with `salt='s'` the expression cannot collide whether the explicit
        # separators are present or not — and the mutation survives. The
        # separators are what make the guard hold UNCONDITIONALLY, including for
        # a caller that passes an empty salt.
        a = expr.replace("fa.feature_id", "'a1'").replace(
            "fa.sample_index::text", "'23'").replace(":salt", "''")
        b = expr.replace("fa.feature_id", "'a12'").replace(
            "fa.sample_index::text", "'3'").replace(":salt", "''")
        same = conn.execute(text(f"SELECT ({a}) = ({b})")).scalar()
        assert same is False, (
            "the module's own sort expression gives the same key to two distinct "
            "(feature, sample) pairs; the easy-negative draw is not the uniform "
            "sample it claims to be"
        )


@pytest.fixture
def panel(conn):
    """A self-contained extraction built inside the rolled-back transaction.

    The first version of these tests keyed off a hardcoded production extraction
    id. That extraction is not in the dev database, so all three real-data tests
    SKIPPED and asserted nothing while the suite reported green — the same
    fail-open shape as a source scrape, but quieter, because -q hides skips.

    Shape: TARGET plus 3 donors sharing its rank-1 token ("running") and 3 that
    do not, each with distinct sample_index values, plus one donor deliberately
    holding a sample the TARGET also holds, so the disjointness filter has
    something real to exclude.
    """
    # Unique per run. Fixed ids collided between concurrent pytest processes
    # pointed at the same DATABASE_URL_SYNC, and survived an abnormal exit to
    # break every subsequent run from inside a fixture rather than a test.
    tag = uuid.uuid4().hex[:10]
    ext = f"extr_negtest_{tag}"
    sae = f"sae_negtest_{tag}"
    fgr = f"fgr_negtest_{tag}"
    conn.execute(text(
        # EVERY NOT-NULL column without a SERVER default is supplied here.
        # The models declare Python-side `default=` values, which the ORM
        # applies and raw SQL does not — so relying on them worked against a
        # dev database carrying migration-added defaults and violated NOT NULL
        # on a freshly created schema, one column at a time.
        "INSERT INTO external_saes "
        "(id, name, source, status, format, progress, sae_metadata) "
        "VALUES (:sae, 't', 'local', 'ready', 'community_standard', 1.0, "
        "'{}'::jsonb)"),
        {"sae": sae})
    conn.execute(text(
        "INSERT INTO extraction_jobs (id, config, external_sae_id, status) "
        "VALUES (:e, '{}'::jsonb, :sae, 'completed')"), {"e": ext, "sae": sae})
    conn.execute(text(
        "INSERT INTO feature_grouping_runs (id, extraction_id, params, params_hash) "
        "VALUES (:g, :e, '{}'::jsonb, 'h')"), {"g": fgr, "e": ext})

    specs = [("target", "running")] + \
            [(f"hard{i}", "running") for i in range(3)] + \
            [(f"easy{i}", f"other{i}") for i in range(3)]

    sample = 1000
    target_samples = []
    for idx, (name, token) in enumerate(specs):
        fid = f"feat_{name}_{tag}"
        conn.execute(text(
            # check_feature_single_source: exactly one of training_id /
            # external_sae_id must be set.
            "INSERT INTO features (id, extraction_job_id, external_sae_id, "
            "neuron_index, name, label_source, activation_frequency, "
            "interpretability_score, max_activation, is_favorite) "
            "VALUES (:f, :e, :sae, :n, :nm, 'auto', 0.01, 0.5, 5.0, false)"),
            {"f": fid, "e": ext, "sae": sae, "n": idx, "nm": name})
        conn.execute(text(
            "INSERT INTO feature_token_index (run_id, extraction_id, feature_id, "
            "neuron_index, raw_token, normalized_token, token_rank, weight) "
            "VALUES (:g, :e, :f, :n, :t, :t, 1, 0.9)"),
            {"g": fgr, "e": ext, "f": fid, "n": idx, "t": token})
        # Give each donor a DIFFERENT activation scale, so a global
        # "ORDER BY max_activation DESC" would collapse onto one donor.
        for k in range(6):
            conn.execute(text(
                "INSERT INTO feature_activations (feature_id, sample_index, "
                "max_activation, tokens, activations) "
                "VALUES (:f, :s, :a, '[]'::jsonb, '[]'::jsonb)"),
                {"f": fid, "s": sample, "a": (10.0 - idx) - k * 0.1})
            if name == "target":
                target_samples.append(sample)
            sample += 1

    # One hard donor also holds a passage the TARGET holds — the case the
    # disjointness filter exists for.
    conn.execute(text(
        "INSERT INTO feature_activations (feature_id, sample_index, max_activation, "
        "tokens, activations) VALUES (:f, :s, 99.0, '[]'::jsonb, '[]'::jsonb)"),
        {"f": f"feat_hard0_{tag}", "s": target_samples[0]})

    return {"ext": ext, "fid": f"feat_target_{tag}", "own": target_samples,
            "tag": tag}


class TestTheSamplerAgainstRealData:
    """Runs the actual query text against real PostgreSQL and real rows."""

    def test_a_negative_is_never_one_of_the_targets_own_passages(self, conn, panel):
        """C16. A donor holding one of the target's own top passages must not
        supply it as a negative — it is a strong POSITIVE for the target, and
        labelling it 0 punishes a correct label."""
        for sql in (_HARD_NEGATIVES_SQL, _EASY_NEGATIVES_SQL):
            params = {"feature_id": panel["fid"], "extraction_id": panel["ext"],
                      "exclude_samples": panel["own"], "limit": 6}
            params["salt"] = "t"
            if sql is _HARD_NEGATIVES_SQL:
                params["donor_limit"] = 20
            drawn = [dict(r._mapping) for r in conn.execute(sql, params)]
            assert drawn, "no negatives drawn at all — the fixture is not exercising the query"
            overlap = {d["sample_index"] for d in drawn} & set(panel["own"])
            assert not overlap, (
                f"negatives include the target's own passages: {sorted(overlap)}"
            )

    def test_hard_negatives_come_from_the_token_sharing_donors(self, conn, panel):
        drawn = [dict(r._mapping) for r in conn.execute(_HARD_NEGATIVES_SQL, {
            "feature_id": panel["fid"], "extraction_id": panel["ext"],
            "exclude_samples": panel["own"], "donor_limit": 20, "limit": 6,
            "salt": "t"})]
        assert drawn
        assert all(d["feature_id"].startswith("feat_hard") for d in drawn), (
            f"hard negatives came from non-token-sharing donors: "
            f"{sorted({d['feature_id'] for d in drawn})}"
        )

    def test_hard_negatives_are_spread_across_donors(self, conn, panel):
        """C15. Each donor has a different activation scale, so ordering by
        max_activation alone takes all 3 from the single highest-scale donor."""
        drawn = [dict(r._mapping) for r in conn.execute(_HARD_NEGATIVES_SQL, {
            "feature_id": panel["fid"], "extraction_id": panel["ext"],
            "exclude_samples": panel["own"], "donor_limit": 20, "limit": 3,
            "salt": "t"})]
        donors = {d["feature_id"] for d in drawn}
        assert len(drawn) == 3
        assert len(donors) == 3, (
            f"3 hard negatives came from {len(donors)} donor(s) ({sorted(donors)}); "
            f"the draw is monopolised by the highest-activation donor and tests a "
            f"far narrower slice than the item count implies"
        )

    def test_a_null_in_the_exclusion_list_does_not_silently_empty_the_draw(self, conn, panel):
        """C13. `x <> ALL(ARRAY[1,NULL])` is never true, so one NULL would empty
        the entire draw and leave the feature silently unscored."""
        class _Conn:
            def __init__(self, c): self._c = c
            def execute(self, *a, **k): return self._c.execute(*a, **k)

        hard, easy = sample_negatives(
            _Conn(conn), feature_id=panel["fid"], extraction_id=panel["ext"],
            exclude_samples=[None, *panel["own"]], n_hard=3, n_easy=3)
        assert hard, "a NULL in the exclusion list emptied the hard-negative draw"
        assert easy, "a NULL in the exclusion list emptied the easy-negative draw"

    def test_easy_negatives_avoid_the_token_sharing_donors_only_by_chance(self, conn, panel):
        """Easy negatives are drawn from the whole extraction, so they MAY include
        a token-sharing donor. Pinned so the distinction from hard stays honest."""
        drawn = [dict(r._mapping) for r in conn.execute(_EASY_NEGATIVES_SQL, {
            "feature_id": panel["fid"], "extraction_id": panel["ext"],
            "exclude_samples": panel["own"], "salt": "t", "limit": 6})]
        assert drawn
        assert all(d["feature_id"] != panel["fid"] for d in drawn), (
            "an easy negative came from the target feature itself"
        )
