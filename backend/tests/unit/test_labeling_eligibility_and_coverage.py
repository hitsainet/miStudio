"""Which features a resume takes, and which it must never touch again.

THE FIXTURE WHOSE ABSENCE ALLOWED THE CONFUSION: a feature in EVERY
`label_status`, asserted against the real eligibility query. Nothing anywhere
held all five states side by side, so no test could see that `succeeded` and
`failed` were being written to the same three fields in the same shape.

The expensive mistake this prevents is making `uninterpretable` eligible.
`_enforce_refusal` produces it deliberately below a 0.5 fit ratio: it is the
judge's honest verdict that a feature has no coherent pattern, and redoing it
costs ~8 s/feature to reach the same answer. The mutation control for it is
recorded in the class below.
"""

import uuid

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import sqlalchemy as sa

from src.models.dataset import Dataset
from src.models.external_sae import ExternalSAE
from src.models.extraction_job import ExtractionJob
from src.models.feature import Feature
from src.models.model import Model
from src.services import labeling_eligibility as elig

FINGERPRINT = "a" * 64
OTHER_FINGERPRINT = "b" * 64
JUDGE = "gemma-4-31b-GGUF:IQ4_XS"
OTHER_JUDGE = "gpt-4o-mini"


async def _extraction(session, eid):
    mid, did = f"m_{eid}", str(uuid.uuid4())
    session.add(Model(id=mid, name=f"model {eid}", architecture="test", params_count=1))
    session.add(Dataset(id=did, name=f"dataset {eid}", source="Local"))
    await session.commit()
    session.add(ExternalSAE(id=f"sae_{eid}", name=f"sae {eid}", source="trained"))
    await session.commit()
    session.add(ExtractionJob(id=eid, external_sae_id=f"sae_{eid}", config={}))
    await session.commit()
    return eid


async def _feature(session, eid, fid, *, neuron_index=0, **cols):
    session.add(Feature(
        id=fid, name=fid, neuron_index=neuron_index,
        extraction_job_id=eid, external_sae_id=f"sae_{eid}",
        activation_frequency=0.5, mean_activation=1.0,
        max_activation=2.0, interpretability_score=0.4,
        **cols,
    ))
    await session.commit()
    return fid


async def _eligible_ids(session, eid, **kwargs):
    rows = await session.execute(
        sa.select(Feature.id)
        .where(Feature.extraction_job_id == eid)
        .where(elig.eligibility_filter(**kwargs))
    )
    return sorted(r[0] for r in rows.all())


@pytest.fixture
async def every_status(async_session):
    """One feature in each of the five statuses, all in one extraction."""
    eid = await _extraction(async_session, f"ext_{uuid.uuid4().hex[:8]}")
    await _feature(async_session, eid, "f_pending", neuron_index=0,
                   label_status="pending")
    await _feature(async_session, eid, "f_failed", neuron_index=1,
                   label_status="failed", label_error="APIError: reset",
                   label_attempts=1)
    # A real verdict.
    await _feature(async_session, eid, "f_succeeded", neuron_index=2,
                   label_status="succeeded", category="semantic",
                   label_prompt_fingerprint=FINGERPRINT, label_model=JUDGE,
                   label_attempts=1)
    # THE SHARP EDGE: an honest refusal. `succeeded`, because it is a verdict.
    await _feature(async_session, eid, "f_uninterpretable", neuron_index=3,
                   label_status="succeeded", category="uninterpretable",
                   label_prompt_fingerprint=FINGERPRINT, label_model=JUDGE,
                   label_attempts=1)
    await _feature(async_session, eid, "f_skipped", neuron_index=4,
                   label_status="skipped", star_color="aqua")
    await _feature(async_session, eid, "f_in_progress", neuron_index=5,
                   label_status="in_progress", label_attempts=1)
    return eid


class TestRoutineResume:

    async def test_takes_exactly_pending_and_failed(self, async_session, every_status):
        assert await _eligible_ids(async_session, every_status) == [
            "f_failed", "f_pending"
        ]

    async def test_never_takes_an_adjudicated_refusal(self, async_session, every_status):
        """MUTATION CONTROL: add "succeeded" to RETRYABLE_STATUSES and this must
        fail. It is the single most expensive mistake this design prevents —
        `uninterpretable` is a RESULT, and treating it as a gap turns resume
        into a relabel-everything button."""
        eligible = await _eligible_ids(async_session, every_status)
        assert "f_uninterpretable" not in eligible
        assert "f_succeeded" not in eligible

    async def test_never_takes_a_deliberate_skip(self, async_session, every_status):
        """An aqua feature was left alone on purpose; it is not outstanding work."""
        assert "f_skipped" not in await _eligible_ids(async_session, every_status)

    async def test_never_takes_work_another_job_holds(self, async_session, every_status):
        """Two concurrent resumes must not claim the same feature."""
        assert "f_in_progress" not in await _eligible_ids(async_session, every_status)

    async def test_an_unknown_status_is_ineligible_by_default(self, async_session):
        """A status this module has not been taught about must not be silently
        relabelled. `RETRYABLE_STATUSES` lists what IS retryable rather than
        excluding what is not, so a new outcome fails closed."""
        eid = await _extraction(async_session, f"ext_{uuid.uuid4().hex[:8]}")
        await _feature(async_session, eid, "f_weird", label_status="quarantined")
        assert await _eligible_ids(async_session, eid) == []


class TestAttemptCap:

    async def test_stops_offering_a_feature_that_keeps_failing(self, async_session):
        """Without a cap, a feature that fails for a reason no retry can fix is
        re-queued by every resume forever — and because failures sort first, it
        crowds out work that could succeed."""
        eid = await _extraction(async_session, f"ext_{uuid.uuid4().hex[:8]}")
        await _feature(async_session, eid, "f_two", neuron_index=0,
                       label_status="failed", label_attempts=2)
        await _feature(async_session, eid, "f_three", neuron_index=1,
                       label_status="failed", label_attempts=3)
        await _feature(async_session, eid, "f_four", neuron_index=2,
                       label_status="failed", label_attempts=4)
        assert await _eligible_ids(async_session, eid, max_attempts=3) == ["f_two"]

    async def test_the_cap_is_configurable(self, async_session):
        eid = await _extraction(async_session, f"ext_{uuid.uuid4().hex[:8]}")
        await _feature(async_session, eid, "f_four", label_status="failed",
                       label_attempts=4)
        assert await _eligible_ids(async_session, eid, max_attempts=3) == []
        assert await _eligible_ids(async_session, eid, max_attempts=5) == ["f_four"]


class TestStalenessByJudge:
    """Improving a template must be self-servicing: exactly the features the new
    template would answer differently come back into scope, with no SQL."""

    async def test_an_unchanged_judge_leaves_verdicts_alone(self, async_session, every_status):
        eligible = await _eligible_ids(
            async_session, every_status,
            prompt_fingerprint=FINGERPRINT, judge_model=JUDGE,
        )
        assert "f_succeeded" not in eligible
        assert "f_uninterpretable" not in eligible

    async def test_a_changed_prompt_makes_verdicts_eligible(self, async_session, every_status):
        eligible = await _eligible_ids(
            async_session, every_status,
            prompt_fingerprint=OTHER_FINGERPRINT, judge_model=JUDGE,
        )
        assert "f_succeeded" in eligible
        assert "f_uninterpretable" in eligible

    async def test_a_changed_model_makes_verdicts_eligible(self, async_session, every_status):
        """A verdict is the product of a prompt AND a judge. Same template on a
        different model is a different measurement."""
        eligible = await _eligible_ids(
            async_session, every_status,
            prompt_fingerprint=FINGERPRINT, judge_model=OTHER_JUDGE,
        )
        assert "f_succeeded" in eligible

    async def test_an_unknown_judge_is_stale(self, async_session):
        """NULL means "we do not know which judge produced this" — what the
        backfill writes over historical rows. Honest, re-adjudicable on request,
        and never touched by a routine resume."""
        eid = await _extraction(async_session, f"ext_{uuid.uuid4().hex[:8]}")
        await _feature(async_session, eid, "f_historical", label_status="succeeded",
                       category="semantic", label_prompt_fingerprint=None,
                       label_model=None)
        assert await _eligible_ids(async_session, eid) == []
        assert await _eligible_ids(
            async_session, eid, prompt_fingerprint=FINGERPRINT, judge_model=JUDGE,
        ) == ["f_historical"]


class TestResumeBatchOrdering:

    async def test_never_attempted_features_go_first(self, async_session):
        """A batch filled entirely with twice-failed features while untouched
        ones wait behind them is a resume that appears to make no progress."""
        eid = await _extraction(async_session, f"ext_{uuid.uuid4().hex[:8]}")
        await _feature(async_session, eid, "f_tried_twice", neuron_index=0,
                       label_status="failed", label_attempts=2)
        await _feature(async_session, eid, "f_fresh", neuron_index=9,
                       label_status="pending", label_attempts=0)
        rows = await async_session.execute(elig.resume_batch_query(eid, limit=10))
        assert [r[0] for r in rows.all()] == ["f_fresh", "f_tried_twice"]

    async def test_respects_the_limit(self, async_session):
        eid = await _extraction(async_session, f"ext_{uuid.uuid4().hex[:8]}")
        for i in range(5):
            await _feature(async_session, eid, f"f_{i}", neuron_index=i,
                           label_status="pending")
        rows = await async_session.execute(elig.resume_batch_query(eid, limit=2))
        assert len(rows.all()) == 2


class TestSummarise:

    def test_counts_what_a_resume_would_take(self):
        s = elig.summarise({"pending": 10, "failed": 3, "succeeded": 5,
                            "skipped": 2, "in_progress": 1})
        assert s["total"] == 21
        assert s["remaining"] == 13
        assert s["adjudicated"] == 7
        assert s["in_progress"] == 1
        assert s["unclassified"] == 0

    def test_an_unknown_status_is_reported_not_absorbed(self):
        """`remaining` used to be derivable as `total - adjudicated`, which
        silently absorbs any status nobody has been taught about and reports a
        feature stuck in an unknown state as finished."""
        s = elig.summarise({"succeeded": 5, "quarantined": 4})
        assert s["remaining"] == 0
        assert s["unclassified"] == 4


# ── the endpoint ─────────────────────────────────────────────────────────────

class TestCoverageEndpointIsReachable:
    """A capability is not shipped until a test fails when its wiring is removed.

    Asserted against the LIVE registry. Note `app.openapi()["paths"]` and never
    `app.routes`: on this FastAPI version `include_router` appends a
    `_IncludedRouter` with no `.path`, so scanning `app.routes` reports an app
    serving only framework defaults while every route works fine.
    """

    PATH = "/api/v1/labeling/{extraction_job_id}/coverage"

    def test_the_route_is_registered_on_the_built_app(self):
        from src.main import app

        paths = app.openapi()["paths"]
        assert self.PATH in paths, (
            f"{self.PATH} is not in the live OpenAPI. Registered labeling "
            f"paths: {sorted(p for p in paths if 'labeling' in p)}"
        )
        assert "get" in paths[self.PATH]

    def test_it_is_not_shadowed_by_the_job_status_route(self):
        """`/labeling/{labeling_job_id}` is registered first and takes an id in
        the same position. A second path segment is what keeps them apart."""
        from src.main import app

        paths = app.openapi()["paths"]
        assert "/api/v1/labeling/{labeling_job_id}" in paths
        assert self.PATH in paths


class TestCoverageEndpoint:

    async def test_reports_each_status_and_the_resume_batch(self, client, every_status):
        r = await client.get(f"/api/v1/labeling/{every_status}/coverage")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["total"] == 6
        assert body["remaining"] == 2
        assert body["adjudicated"] == 3
        assert body["in_progress"] == 1
        assert sorted(body["resume_feature_ids"]) == ["f_failed", "f_pending"]

    async def test_offers_no_adjudicated_feature_for_resume(self, client, every_status):
        r = await client.get(f"/api/v1/labeling/{every_status}/coverage")
        offered = set(r.json()["resume_feature_ids"])
        assert not offered & {"f_succeeded", "f_uninterpretable", "f_skipped"}

    async def test_stale_is_absent_until_a_judge_is_named(self, client, every_status):
        """Counting staleness needs a judge to compare against. Reporting 0
        without one would read as "nothing is stale", which is not known."""
        r = await client.get(f"/api/v1/labeling/{every_status}/coverage")
        assert r.json()["stale"] is None

    async def test_a_different_judge_makes_verdicts_stale_and_offered(
        self, client, every_status
    ):
        r = await client.get(
            f"/api/v1/labeling/{every_status}/coverage",
            params={"prompt_fingerprint": OTHER_FINGERPRINT, "judge_model": JUDGE},
        )
        body = r.json()
        # The two SUCCEEDED verdicts, and only those. `f_skipped` is adjudicated
        # but not staleable — see the aqua test below.
        assert body["stale"] == 2
        assert "f_uninterpretable" in body["resume_feature_ids"]

    async def test_the_same_judge_makes_nothing_stale(self, client, every_status):
        r = await client.get(
            f"/api/v1/labeling/{every_status}/coverage",
            params={"prompt_fingerprint": FINGERPRINT, "judge_model": JUDGE},
        )
        assert r.json()["stale"] == 0

    async def test_a_skipped_feature_is_never_stale(self, client, every_status):
        """An aqua feature has no fingerprint because no judge was ever asked
        about it. A naive "adjudicated and the fingerprint does not match" test
        marks every one of them stale the moment a template is edited, offering
        hand-verified labels up for relabelling — the exact promise the aqua
        star makes to the user. Nothing a template change does can invalidate a
        decision not to run the judge at all.

        Caught by `test_the_same_judge_makes_nothing_stale`, which reported 1.
        """
        r = await client.get(
            f"/api/v1/labeling/{every_status}/coverage",
            params={"prompt_fingerprint": OTHER_FINGERPRINT,
                    "judge_model": OTHER_JUDGE},
        )
        assert "f_skipped" not in r.json()["resume_feature_ids"]

    async def test_says_so_when_failures_carry_no_reason(self, async_session, client):
        """A failure with no recorded reason is not diagnosable, and the
        response must not present it as though it were."""
        eid = await _extraction(async_session, f"ext_{uuid.uuid4().hex[:8]}")
        await _feature(async_session, eid, "f_mute", label_status="failed",
                       label_error=None)
        body = (await client.get(f"/api/v1/labeling/{eid}/coverage")).json()
        assert body["failures_without_a_recorded_reason"] == 1
        assert body["caveat"] and "no recorded reason" in body["caveat"]

    async def test_the_backfilled_sentinel_counts_as_NO_reason(self, async_session, client):
        """CAUGHT ON LIVE DATA, not by this suite.

        The backfill writes a literal placeholder into `label_error` for every
        historical failure. The detection query tested `IS NULL OR = ''`, so the
        placeholder read as a GENUINE reason: the endpoint reported
        `failures_without_a_recorded_reason: 0` for an extraction whose 14,560
        failures every one predate error capture.

        Every test here passed because they used synthetic error strings and
        never the string the migration actually writes. This one uses it.
        """
        eid = await _extraction(async_session, f"ext_{uuid.uuid4().hex[:8]}")
        await _feature(
            async_session, eid, "f_backfilled", label_status="failed",
            # The literal `b8e4a2d0c517` wrote to production.
            label_error=(
                "(reason not recorded: this failure predates per-feature "
                "error capture)"
            ),
        )
        body = (await client.get(f"/api/v1/labeling/{eid}/coverage")).json()
        assert body["failures_without_a_recorded_reason"] == 1, (
            "a placeholder is not a reason; reporting 0 here tells an operator "
            "these failures are diagnosable when nothing about them is known"
        )
        assert body["caveat"]

    async def test_no_caveat_when_every_failure_explains_itself(
        self, client, every_status
    ):
        body = (await client.get(f"/api/v1/labeling/{every_status}/coverage")).json()
        assert body["failures_without_a_recorded_reason"] == 0
        assert body["caveat"] is None

    async def test_resume_limit_is_capped_at_the_panel_route_limit(
        self, client, every_status
    ):
        """The ids go straight to POST /labeling/panel, which accepts at most
        2000. Offering more would produce a batch that cannot be submitted."""
        r = await client.get(
            f"/api/v1/labeling/{every_status}/coverage", params={"resume_limit": 2001}
        )
        assert r.status_code == 422

    async def test_unknown_extraction_is_404_not_an_empty_report(self, client):
        """An empty coverage report for a typo'd id reads as "all done"."""
        r = await client.get("/api/v1/labeling/ext_does_not_exist/coverage")
        assert r.status_code == 404


class TestBackfillMigrationTreatsARefusalAsAVerdict:
    """The backfill derives `label_status` from the old encoding. Its list of
    failure sentinels is the one place a single wrong word costs the most.

    `uninterpretable` in that list would mark every honest refusal as
    outstanding work — 196 features on the L46 extraction alone — and a resume
    would re-derive, at ~8 s each, verdicts the judge already reached. Nothing
    else in the system would report it as wrong; it would simply be slow forever.
    """

    MIGRATION = "alembic/versions/b8e4a2d0c517_backfill_feature_adjudication_state.py"

    def _failure_categories(self):
        import ast
        import pathlib

        tree = ast.parse(pathlib.Path(self.MIGRATION).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "_FAILURE_CATEGORIES"
                for t in node.targets
            ):
                return {e.value for e in node.value.elts}
        raise AssertionError(
            f"_FAILURE_CATEGORIES not found in {self.MIGRATION}. If the backfill "
            "was renamed, retarget this test — do not delete it."
        )

    def test_uninterpretable_is_not_a_failure(self):
        assert "uninterpretable" not in self._failure_categories()

    def test_it_lists_the_sentinels_the_old_code_actually_wrote(self):
        assert self._failure_categories() == {
            "error_feature", "rate_limited", "empty_features", "uncategorized",
        }

    def test_it_agrees_with_the_running_code(self):
        """The backfill and the live coverage endpoint must classify the same
        rows the same way, or coverage will report work the resume declines."""
        from src.services.labeling_eligibility import ADJUDICATED_STATUSES

        assert "succeeded" in ADJUDICATED_STATUSES
        assert not self._failure_categories() & {"semantic", "structural",
                                                 "uninterpretable", "noise"}


class TestTheSentinelIsDefinedOnce:
    """The migration writes the placeholder; the query recognises it. They carry
    the same literal because a migration must not import application code that
    will have moved on by the time someone replays it — so the two can drift,
    and if they do the endpoint goes back to reporting 0."""

    def test_the_sentinel_literal_matches_the_migration(self):
        import pathlib

        from src.services.labeling_eligibility import UNRECORDED_REASON_PREFIX

        backfill = pathlib.Path(
            "alembic/versions/b8e4a2d0c517_backfill_feature_adjudication_state.py"
        ).read_text()
        nuller = pathlib.Path(
            "alembic/versions/c1f5b3e9a204_null_the_unrecorded_failure_sentinel.py"
        ).read_text()

        assert UNRECORDED_REASON_PREFIX in backfill, (
            "the backfill no longer writes a string the query recognises"
        )
        assert UNRECORDED_REASON_PREFIX in nuller, (
            "the nulling migration no longer matches what the backfill writes"
        )

    def test_a_real_reason_is_never_mistaken_for_the_placeholder(self):
        """The prefix must not swallow a genuine failure reason."""
        from src.services.labeling_eligibility import UNRECORDED_REASON_PREFIX

        for real in (
            "APIError: connection reset",
            "rate limited by the judge endpoint: 429",
            "no activating examples were retrieved for this feature",
            "the judge's response could not be parsed as a label",
        ):
            assert not real.startswith(UNRECORDED_REASON_PREFIX)


# ── WS1: why the failures failed ─────────────────────────────────────────────

#: The twelve real failure paths, in the shape the judge services emit. A tail
#: that varies per feature is the whole point — several of these carry a host,
#: a port or an id, which is what breaks a naive GROUP BY.
REAL_FAILURE_REASONS = [
    "APIError: connection reset to 10.0.0.5:41288",
    "APIError: connection reset to 10.0.0.5:41291",
    "APIError: connection reset to 10.0.0.7:41305",
    "ReadTimeout: timed out after 120.0s waiting on feat_abc123",
    "ReadTimeout: timed out after 120.0s waiting on feat_def456",
    "rate limited by the judge endpoint: 429 (retry after 31s)",
    "rate limited by the judge endpoint: 429 (retry after 12s)",
    "no activating examples were retrieved for this feature",
    "no activating examples were retrieved for this feature",
    "the judge's response could not be parsed as a label",
    "the batch judge returned no result for this feature",
    "JSONDecodeError: Expecting value: line 1 column 1 (char 0)",
]


class TestFailureReasonsAreGrouped:
    """THE test for WS1. A raw `GROUP BY label_error` looks correct on a tidy
    fixture and is useless on real data, because most reasons carry a host, a
    port or a feature id in the tail — so every failure becomes its own group.
    """

    async def test_twelve_failure_paths_do_not_become_twelve_groups(
        self, async_session, client
    ):
        eid = await _extraction(async_session, f"ext_{uuid.uuid4().hex[:8]}")
        for i, reason in enumerate(REAL_FAILURE_REASONS):
            await _feature(async_session, eid, f"f{i}", neuron_index=i,
                           label_status="failed", label_error=reason)

        body = (await client.get(f"/api/v1/labeling/{eid}/coverage")).json()
        reasons = {r["reason"]: r["count"] for r in body["failure_reasons"]}

        # 12 distinct strings collapse to 7 causes. On the raw column it is 10.
        assert len(reasons) < len(REAL_FAILURE_REASONS), (
            f"no grouping happened: {len(reasons)} groups from "
            f"{len(REAL_FAILURE_REASONS)} failures — {sorted(reasons)}"
        )
        assert reasons.get("APIError") == 3, reasons
        assert reasons.get("ReadTimeout") == 2, reasons
        assert reasons.get("rate limited by the judge endpoint") == 2, reasons
        assert reasons.get("no activating examples were retrieved for this feature") == 2

    async def test_a_varying_tail_collapses_to_one_cause(self, async_session, client):
        """The specific shape that defeats a raw GROUP BY: one cause, a
        different address every time."""
        eid = await _extraction(async_session, f"ext_{uuid.uuid4().hex[:8]}")
        for i in range(25):
            await _feature(async_session, eid, f"f{i}", neuron_index=i,
                           label_status="failed",
                           label_error=f"APIError: connection reset to 10.0.0.{i}:4128{i}")

        body = (await client.get(f"/api/v1/labeling/{eid}/coverage")).json()
        assert body["failure_reasons"] == [
            {"reason": "APIError", "count": 25,
             "sample_feature_ids": ["f0", "f1", "f2"]}
        ]

    async def test_reasons_come_back_commonest_first(self, async_session, client):
        eid = await _extraction(async_session, f"ext_{uuid.uuid4().hex[:8]}")
        for i in range(5):
            await _feature(async_session, eid, f"many{i}", neuron_index=i,
                           label_status="failed", label_error=f"APIError: {i}")
        await _feature(async_session, eid, "one", neuron_index=99,
                       label_status="failed", label_error="ValueError: odd")

        body = (await client.get(f"/api/v1/labeling/{eid}/coverage")).json()
        assert [r["reason"] for r in body["failure_reasons"]] == ["APIError", "ValueError"]

    async def test_samples_are_real_features_an_operator_can_open(
        self, async_session, client
    ):
        """A count with no example is a number to be believed. With one, it can
        be checked."""
        eid = await _extraction(async_session, f"ext_{uuid.uuid4().hex[:8]}")
        for i in range(6):
            await _feature(async_session, eid, f"f{i}", neuron_index=10 - i,
                           label_status="failed", label_error="APIError: x")

        body = (await client.get(f"/api/v1/labeling/{eid}/coverage")).json()
        samples = body["failure_reasons"][0]["sample_feature_ids"]
        assert len(samples) == 3
        # Deterministic, lowest neuron_index first, so two operators comparing
        # notes see the same examples.
        assert samples == ["f5", "f4", "f3"]

    async def test_the_backfilled_placeholder_is_named_not_dropped(
        self, async_session, client
    ):
        """16,970 undiagnosable failures are the most important row on this
        report, not an omission."""
        eid = await _extraction(async_session, f"ext_{uuid.uuid4().hex[:8]}")
        await _feature(async_session, eid, "f_mute", label_status="failed",
                       label_error=None)

        body = (await client.get(f"/api/v1/labeling/{eid}/coverage")).json()
        assert body["failure_reasons"][0]["reason"] == "(no reason recorded)"
        assert body["failure_reasons"][0]["count"] == 1

    async def test_a_capped_tail_is_reported_as_other(self, async_session, client):
        """A breakdown that does not add up to the failure count is worse than
        no breakdown."""
        eid = await _extraction(async_session, f"ext_{uuid.uuid4().hex[:8]}")
        for i in range(14):
            await _feature(async_session, eid, f"f{i}", neuron_index=i,
                           label_status="failed", label_error=f"Error{i:02d}: x")

        body = (await client.get(f"/api/v1/labeling/{eid}/coverage")).json()
        total = sum(r["count"] for r in body["failure_reasons"])
        assert total == 14, "the breakdown must account for every failure"
        assert any(r["reason"].startswith("other (") for r in body["failure_reasons"])

    async def test_no_breakdown_when_nothing_failed(self, async_session, client, every_status):
        eid = await _extraction(async_session, f"ext_{uuid.uuid4().hex[:8]}")
        await _feature(async_session, eid, "f_ok", label_status="succeeded",
                       category="semantic")
        body = (await client.get(f"/api/v1/labeling/{eid}/coverage")).json()
        assert body["failure_reasons"] == []


# ── WS2: sample before committing the estate ─────────────────────────────────

class TestSamplingOneOutcome:
    """14,556 retries is 32 GPU-hours. A sample of 20 decides whether to spend
    them — but only if it asks ONE question."""

    async def test_failed_only_draws_no_pending_features(self, client, every_status):
        r = await client.get(f"/api/v1/labeling/{every_status}/coverage",
                             params={"only": "failed", "resume_limit": 20})
        assert r.status_code == 200, r.text
        assert r.json()["resume_feature_ids"] == ["f_failed"]

    async def test_pending_only_draws_no_failed_features(self, client, every_status):
        r = await client.get(f"/api/v1/labeling/{every_status}/coverage",
                             params={"only": "pending", "resume_limit": 20})
        assert r.json()["resume_feature_ids"] == ["f_pending"]

    async def test_the_default_is_unchanged_and_takes_both(self, client, every_status):
        """`only` is opt-in. A normal resume must behave exactly as before."""
        r = await client.get(f"/api/v1/labeling/{every_status}/coverage")
        assert sorted(r.json()["resume_feature_ids"]) == ["f_failed", "f_pending"]

    async def test_an_unrecognised_scope_is_rejected_not_widened(self, client, every_status):
        """Falling through to the full eligible set would silently answer a
        different question than the one asked, and the caller could not tell."""
        r = await client.get(f"/api/v1/labeling/{every_status}/coverage",
                             params={"only": "everything"})
        assert r.status_code == 422

    async def test_an_adjudicated_status_cannot_be_sampled(self, client, every_status):
        """`succeeded` is not eligible work. Offering to resample it would make
        a sample into a quiet relabel."""
        r = await client.get(f"/api/v1/labeling/{every_status}/coverage",
                             params={"only": "succeeded"})
        assert r.status_code == 422

    async def test_the_query_refuses_an_unknown_scope_directly(self, async_session, every_status):
        """The endpoint's enum stops it at the edge; the query refuses too, so a
        second caller cannot bypass the guard by not being HTTP."""
        with pytest.raises(ValueError, match="cannot sample"):
            await async_session.execute(
                elig.resume_batch_query(every_status, limit=5, only="succeeded")
            )

    async def test_sampling_does_not_change_the_counts(self, client, every_status):
        """A sample narrows the BATCH, never the report. An operator taking a
        sample must still see the whole picture."""
        r = await client.get(f"/api/v1/labeling/{every_status}/coverage",
                             params={"only": "failed"})
        body = r.json()
        assert body["total"] == 6
        assert body["remaining"] == 2


# ── WS6: fill the gaps, but only when asked ──────────────────────────────────

class TestSkipAdjudicatedIsOptIn:
    """The whole-extraction Label button relabels everything, and an operator
    pressing it expects exactly that.

    Making it skip adjudicated work by default was the obvious change and is the
    wrong one: it repurposes an existing control silently, and they would quietly
    stop getting the relabel they asked for with nothing on screen to say so.
    """

    def test_the_column_defaults_to_todays_behaviour(self):
        from src.models.labeling_job import LabelingJob

        col = LabelingJob.__table__.c["skip_adjudicated"]
        assert col.nullable is False
        assert col.default.arg is False, "a full relabel is the historical default"
        assert col.server_default.arg == "false"

    def test_the_default_query_is_UNCHANGED_not_merely_equivalent(self):
        """Compares the compiled SQL, not a row count.

        A fixture with no adjudicated features passes either way, which is
        exactly the fixtures-agree-by-construction trap: the filter could be
        applied unconditionally and every count-based test would still be green.
        """
        from sqlalchemy.orm import Query

        from src.services.labeling_eligibility import eligibility_filter

        base = Query(Feature).filter(Feature.extraction_job_id == "ext_1")
        plain = str(base.statement.compile(compile_kwargs={"literal_binds": True}))
        filtered = str(
            base.filter(eligibility_filter()).statement.compile(
                compile_kwargs={"literal_binds": True}
            )
        )
        assert plain != filtered, (
            "the eligibility filter compiles to the same SQL as no filter — "
            "this test cannot detect the thing it exists to detect"
        )
        # The PREDICATE, not the whole statement: `label_status` is a selected
        # column and appears in both. A first version of this assertion checked
        # the full SQL and failed on the column list, proving nothing about the
        # filter either way.
        # SQLAlchemy renders the clause after a newline, not a space.
        plain_where = plain.split("WHERE ", 1)[1]
        filtered_where = filtered.split("WHERE ", 1)[1]
        assert "label_status" not in plain_where, (
            "the default selection must carry NO predicate on label state"
        )
        assert "label_status" in filtered_where

    def test_the_service_applies_it_only_when_the_job_asks(self):
        import inspect

        from src.services.labeling_service import LabelingService

        source = inspect.getsource(LabelingService.label_features_for_extraction)
        assert 'getattr(labeling_job, "skip_adjudicated", False)' in source, (
            "the filter must be gated on the job's own flag"
        )
        # Gated, not unconditional: the filter call must sit INSIDE the guard.
        guard = source.index('getattr(labeling_job, "skip_adjudicated", False)')
        applied = source.index("_q.filter(eligibility_filter())")
        assert guard < applied

    def test_it_reuses_the_shared_eligibility_definition(self):
        """A second definition of "already adjudicated" would let the count the
        UI shows and the work the job takes disagree — silently, and only on the
        rows that matter."""
        import inspect

        from src.services import labeling_service

        assert "from src.services.labeling_eligibility import eligibility_filter" in (
            inspect.getsource(labeling_service)
        )

    async def test_true_skips_adjudicated_features(self, async_session, every_status):
        rows = await async_session.execute(
            sa.select(Feature.id)
            .where(Feature.extraction_job_id == every_status)
            .where(elig.eligibility_filter())
        )
        taken = sorted(r[0] for r in rows.all())
        assert taken == ["f_failed", "f_pending"]
        assert "f_succeeded" not in taken
        assert "f_uninterpretable" not in taken


class TestSamplingCombinedWithStaleness:
    """`only=` and a judge fingerprint are independent narrowings, and nothing
    covered them together.

    R1: the staleness branch widens eligibility to `succeeded` rows adjudicated
    by a different judge, and `only=` then narrows by status. Read carelessly
    that looks like it could re-admit adjudicated work into a sample of
    "failures". It must not.
    """

    async def test_a_failed_sample_stays_failures_even_when_verdicts_are_stale(
        self, client, every_status
    ):
        r = await client.get(
            f"/api/v1/labeling/{every_status}/coverage",
            params={"only": "failed", "prompt_fingerprint": OTHER_FINGERPRINT,
                    "judge_model": OTHER_JUDGE, "resume_limit": 50},
        )
        assert r.status_code == 200, r.text
        offered = r.json()["resume_feature_ids"]
        assert offered == ["f_failed"], (
            "a sample of FAILURES must not pick up stale verdicts just because "
            "a different judge was named"
        )

    async def test_the_stale_count_is_still_reported_alongside(self, client, every_status):
        """Narrowing the batch must not narrow the REPORT — an operator taking a
        sample still needs to see that verdicts went stale."""
        r = await client.get(
            f"/api/v1/labeling/{every_status}/coverage",
            params={"only": "failed", "prompt_fingerprint": OTHER_FINGERPRINT,
                    "judge_model": OTHER_JUDGE},
        )
        assert r.json()["stale"] == 2

    async def test_a_pending_sample_excludes_failures_and_verdicts_alike(
        self, client, every_status
    ):
        r = await client.get(
            f"/api/v1/labeling/{every_status}/coverage",
            params={"only": "pending", "prompt_fingerprint": OTHER_FINGERPRINT,
                    "judge_model": OTHER_JUDGE, "resume_limit": 50},
        )
        assert r.json()["resume_feature_ids"] == ["f_pending"]


class TestTheRequestSchemaDefaultsToTodaysBehaviour:
    """R3 FINDING. `skip_adjudicated` was guarded at the column, in the compiled
    SQL, at the service and on the checkbox — and NOT in the request schema.

    Mutation R3c (flipping the Pydantic default to True) left all 313 tests
    green. An API caller omitting the field would then silently get the
    repurposed behaviour this whole workstream exists to prevent, and the UI
    tests could not see it because the UI always sends the field explicitly.

    Defending four layers and leaving the fifth is how a fix ends up worse than
    the bug: it looks thoroughly guarded.
    """

    def test_omitting_it_means_a_full_relabel(self):
        from src.schemas.labeling import LabelingConfigRequest

        # `labeling_method` is required; everything else must default.
        request = LabelingConfigRequest(
            extraction_job_id="ext_1", labeling_method="openai"
        )
        assert request.skip_adjudicated is False, (
            "an API caller who does not mention skip_adjudicated must get the "
            "behaviour this endpoint has always had — a full relabel"
        )

    def test_the_field_default_is_false_in_the_served_schema(self):
        """What a client generating from OpenAPI actually reads."""
        from src.main import app

        schema = app.openapi()["components"]["schemas"]["LabelingConfigRequest"]
        assert schema["properties"]["skip_adjudicated"]["default"] is False

    def test_it_is_not_required(self):
        """Making it required would break every existing caller, which is a
        different way of changing behaviour without saying so."""
        from src.main import app

        schema = app.openapi()["components"]["schemas"]["LabelingConfigRequest"]
        assert "skip_adjudicated" not in schema.get("required", [])


class TestExhaustedRetriesAreNotOfferedAsRemaining:
    """REPORTED FROM THE UI: a card read "Resume 0 of 39", and clicking it
    answered "Nothing left to label in this extraction."

    `extr_…5ed1_001` has 39 failed features, each with `label_attempts = 3`.
    The batch query excludes them (correctly — the cap is what stops a
    permanently-broken feature being retried forever), while `summarise`
    counted them as `remaining` from status alone.

    `summarise`'s own docstring claimed a button "cannot offer N while a job
    labels a different N". It shared the STATUS constants with the query but
    not the ATTEMPT CAP. The number on a button has to be measured with the
    predicate that produces the work, not with a proxy for it.
    """

    async def _extraction_with_exhausted_failures(self, session, *, attempts):
        eid = await _extraction(session, f"ext_{uuid.uuid4().hex[:8]}")
        for i in range(39):
            await _feature(session, eid, f"spent{i}", neuron_index=i,
                           label_status="failed", label_attempts=attempts,
                           label_error="APIError: reset")
        return eid

    async def test_remaining_matches_what_a_resume_takes(self, async_session, client):
        eid = await self._extraction_with_exhausted_failures(async_session, attempts=3)
        body = (await client.get(f"/api/v1/labeling/{eid}/coverage")).json()

        assert body["remaining"] == len(body["resume_feature_ids"]) == 0, (
            "the button's number must equal the work a resume would take"
        )

    async def test_the_features_are_still_reported_as_outstanding(
        self, async_session, client
    ):
        """They genuinely still need a label. Hiding them would be its own lie."""
        eid = await self._extraction_with_exhausted_failures(async_session, attempts=3)
        body = (await client.get(f"/api/v1/labeling/{eid}/coverage")).json()

        assert body["outstanding"] == 39
        assert body["by_status"]["failed"] == 39
        assert body["adjudicated"] == 0

    async def test_exhausted_explains_the_gap(self, async_session, client):
        """39 outstanding and 0 remaining is arithmetic an operator cannot
        reconcile without being told why."""
        eid = await self._extraction_with_exhausted_failures(async_session, attempts=3)
        body = (await client.get(f"/api/v1/labeling/{eid}/coverage")).json()

        assert body["exhausted"] == 39
        assert body["outstanding"] - body["exhausted"] == body["remaining"]

    async def test_one_more_attempt_available_means_it_is_offered(
        self, async_session, client
    ):
        """The boundary, from the other side — the sibling card that works."""
        eid = await self._extraction_with_exhausted_failures(async_session, attempts=2)
        body = (await client.get(f"/api/v1/labeling/{eid}/coverage")).json()

        assert body["remaining"] == 39
        assert body["exhausted"] == 0
        assert len(body["resume_feature_ids"]) == 39

    async def test_raising_the_cap_makes_them_eligible_again(
        self, async_session, client
    ):
        eid = await self._extraction_with_exhausted_failures(async_session, attempts=3)
        body = (await client.get(
            f"/api/v1/labeling/{eid}/coverage", params={"max_attempts": 5}
        )).json()

        assert body["remaining"] == 39
        assert body["exhausted"] == 0


class TestAResumeCanNameADifferentJudge:
    """Being able to label missed or errored features with a DIFFERENT model is
    the point of resume, not an edge case.

    Resume reuses the job's own judge, which keeps both halves of one run
    comparable. That is useless when the judge is gone — which is exactly when
    someone needs to resume. A job's March-era judge was removed from the server
    months later; resume reused it faithfully, every feature 404'd, and there was
    no route through the UI to pick a model that exists.

    `features.label_model` and `label_prompt_fingerprint` are per-feature, so an
    extraction labelled by two judges stays honest about which produced what.
    """

    # Patched by NAME. Patching `httpx.AsyncClient.get` also patches the test
    # client — both are the same class — so the endpoint never ran and the test
    # asserted against its own mock.
    _EP = "src.api.v1.endpoints.labeling"

    async def _job(self, session, eid, *, model, endpoint="http://millm.local/v1"):
        from src.models.labeling_job import LabelingJob, LabelingStatus

        session.add(LabelingJob(
            id="lbl_judge", extraction_job_id=eid, labeling_method="openai_compatible",
            openai_compatible_endpoint=endpoint, openai_compatible_model=model,
            status=LabelingStatus.COMPLETED.value, progress=1.0, features_labeled=0,
        ))
        await session.commit()
        return "lbl_judge"

    async def test_it_reports_what_the_endpoint_serves(self, async_session, client):
        eid = await _extraction(async_session, f"ext_{uuid.uuid4().hex[:8]}")
        jid = await self._job(async_session, eid, model="granite-3.3-8b-instruct")

        with patch(f"{self._EP}._fetch_served_models", new_callable=AsyncMock) as fetch:
            fetch.return_value = ["gemma-4-31b-GGUF:IQ4_XS", "granite-4.2-8b"]
            body = (await client.get(f"/api/v1/labeling/{jid}/available-judges")).json()

        assert body["models"] == ["gemma-4-31b-GGUF:IQ4_XS", "granite-4.2-8b"]
        assert body["original_model"] == "granite-3.3-8b-instruct"
        assert body["original_available"] is False, (
            "a judge the endpoint does not serve must not be reported as usable"
        )

    async def test_it_confirms_a_judge_that_is_still_served(self, async_session, client):
        eid = await _extraction(async_session, f"ext_{uuid.uuid4().hex[:8]}")
        jid = await self._job(async_session, eid, model="granite-4.2-8b")

        with patch(f"{self._EP}._fetch_served_models", new_callable=AsyncMock) as fetch:
            fetch.return_value = ["granite-4.2-8b"]
            body = (await client.get(f"/api/v1/labeling/{jid}/available-judges")).json()

        assert body["original_available"] is True

    async def test_an_unreachable_endpoint_reports_rather_than_500s(
        self, async_session, client
    ):
        """A picker that errors leaves an operator with no route at all."""
        eid = await _extraction(async_session, f"ext_{uuid.uuid4().hex[:8]}")
        jid = await self._job(async_session, eid, model="anything")

        with patch(f"{self._EP}._fetch_served_models", new_callable=AsyncMock) as fetch:
            fetch.side_effect = OSError("no route to host")
            r = await client.get(f"/api/v1/labeling/{jid}/available-judges")

        assert r.status_code == 200
        assert r.json()["reachable"] is False
        assert r.json()["detail"]

    async def test_it_takes_the_endpoint_from_the_job_not_the_caller(self):
        """Keyed on the job id, so this cannot be pointed at an arbitrary host."""
        from src.main import app

        params = app.openapi()["paths"][
            "/api/v1/labeling/{labeling_job_id}/available-judges"
        ]["get"].get("parameters", [])
        names = {p["name"] for p in params}
        assert names == {"labeling_job_id"}, (
            f"the endpoint accepts {sorted(names - {'labeling_job_id'})} from the "
            "caller; the URL must come from the stored row"
        )

    async def test_unknown_job_is_404(self, client):
        assert (await client.get("/api/v1/labeling/nope/available-judges")).status_code == 404
