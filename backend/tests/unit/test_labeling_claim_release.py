"""A claimed batch must never outlive the run that claimed it.

LIVE DEFECT, 2026-10-20 (tracker 3.4 part E). A labeling job pointed at an
unreachable judge (`http://127.0.0.1:9/v1`) failed in ~2 minutes with
`APIConnectionError`, as designed: an unreachable judge is a JOB failure and
spends no feature's retry. But the batch had been claimed (`in_progress`)
BEFORE the judge was asked, and nothing released it:

* the `JudgeUnavailable` handler promised to "leave every feature exactly as it
  was" and never touched the claim;
* `cleanup_stuck_labeling` scans only QUEUED/LABELING jobs (this one was already
  FAILED) and releases by `labeling_job_id`, which the claim never set — its
  tests seeded that column by hand, so the release matched no production row.

Result: 10 features stranded `in_progress` per unreachable-host run, excluded
from every resume, forever.

These tests drive the REAL `label_features_for_extraction` and the REAL
`_claim_features` against a real database; only the judge, the example
retrieval and the WebSocket emitters are stubbed.
"""
import asyncio
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import httpx
import openai
import pytest
import sqlalchemy as sa

from src.models.dataset import Dataset
from src.models.external_sae import ExternalSAE
from src.models.extraction_job import ExtractionJob
from src.models.feature import Feature
from src.models.labeling_job import LabelingJob, LabelingStatus
from src.models.model import Model
from src.services import labeling_eligibility
from src.services import labeling_service as labeling_service_module
from src.services.labeling_judge_health import JudgeUnavailable
from src.services.labeling_service import LabelingService
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import NullPool

UNREACHABLE = "http://127.0.0.1:9/v1"
N_FEATURES = 20
BATCH = 10


async def _extraction(session):
    eid = f"ext_{uuid.uuid4().hex[:8]}"
    session.add(Model(id=f"m_{eid}", name=f"model {eid}", architecture="test", params_count=1))
    session.add(Dataset(id=str(uuid.uuid4()), name=f"dataset {eid}", source="Local"))
    await session.commit()
    session.add(ExternalSAE(id=f"sae_{eid}", name=f"sae {eid}", source="trained"))
    await session.commit()
    session.add(ExtractionJob(id=eid, external_sae_id=f"sae_{eid}", config={}))
    await session.commit()
    return eid


async def _features(session, eid, n, **fields):
    ids = []
    for i in range(n):
        fid = f"feat_{eid}_{i:05d}"
        base = dict(
            id=fid, name=f"feature_{i}", neuron_index=i,
            extraction_job_id=eid, external_sae_id=f"sae_{eid}",
            activation_frequency=0.5, mean_activation=1.0, max_activation=2.0,
            interpretability_score=0.4, label_status="pending", label_attempts=0,
        )
        base.update(fields)
        session.add(Feature(**base))
        ids.append(fid)
    await session.commit()
    return ids


async def _job(session, eid, job_id, status=LabelingStatus.QUEUED.value, **fields):
    session.add(LabelingJob(
        id=job_id, extraction_job_id=eid, labeling_method="openai_compatible",
        status=status, progress=0.0, features_labeled=0,
        openai_compatible_endpoint=UNREACHABLE,
        openai_compatible_model="Qwen2.5-7B-Instruct",
        statistics={"batch_size": BATCH},
        **fields,
    ))
    await session.commit()
    return job_id


class _UnreachableJudge:
    """Stands in for `OpenAILabelingService` against a host nobody listens on.

    Each request fails exactly as the SDK fails it live:
    `APIConnectionError: Connection error.`. No batched method, so `_label_batch`
    takes the per-feature path with `return_exceptions=True`.
    """

    calls = 0

    def __init__(self, **_kwargs):
        class _Client:
            async def aclose(self):
                return None

        self._http_client = _Client()

    async def generate_label_from_examples(self, **_kwargs):
        type(self).calls += 1
        raise openai.APIConnectionError(request=httpx.Request("POST", UNREACHABLE))


class _JudgeThatGoesAwayAfterOne(_UnreachableJudge):
    """Answers the first request, then the host disappears — a judge lost
    MID-BATCH, so one feature of the claimed batch reaches a verdict before
    the run aborts."""

    async def generate_label_from_examples(self, **_kwargs):
        type(self).calls += 1
        if type(self).calls == 1:
            return {"category": "semantic", "specific": "cat_tokens", "description": "d"}
        raise openai.APIConnectionError(request=httpx.Request("POST", UNREACHABLE))


class _JudgeWithAMalformedAnswer(_UnreachableJudge):
    """A reply the writer cannot store (no `category`): `_persist_label_outcome`
    raises KeyError, which is no JudgeUnavailable — the generic failure path."""

    async def generate_label_from_examples(self, **_kwargs):
        type(self).calls += 1
        return {"specific": "x"}


class _PassThroughFilter:
    def filter_features_from_examples(self, features, examples, all_examples, verdict_examples=None):
        n = len(features)
        return features, examples, all_examples, {
            "features_to_label": n, "total_features": n,
            "features_skipped": 0, "skip_percentage": 0.0,
        }


EXAMPLE = {
    "prefix_tokens": ["the"], "prime_token": "cat", "suffix_tokens": ["sat"],
    "max_activation": 2.0, "token_positions": [], "sample_index": 0,
}


def _run_unreachable_job(sync_db, job_id, judge=_UnreachableJudge):
    """The real worker body, the judge unreachable. Returns what it raised."""
    service = LabelingService(sync_db)
    judge.calls = 0

    def _examples(session=None, feature_ids=(), **_kw):
        return {fid: [dict(EXAMPLE)] for fid in feature_ids}

    with patch.object(service, "_retrieve_top_examples_batch_sync", side_effect=_examples), \
         patch.object(service, "_retrieve_midrange_examples_batch_sync", side_effect=_examples), \
         patch.object(service, "_retrieve_bottom_examples_batch_sync", return_value={}), \
         patch("src.utils.token_filter.get_feature_filter", return_value=_PassThroughFilter()), \
         patch.object(labeling_service_module, "OpenAILabelingService", judge), \
         patch.object(labeling_service_module, "ensure_model_loaded", lambda *a, **k: None), \
         patch.object(labeling_service_module, "emit_labeling_progress", lambda *a, **k: None), \
         patch.object(labeling_service_module, "emit_labeling_result", lambda *a, **k: None), \
         patch.object(labeling_service_module.settings, "labeling_batch_size", 1):
        try:
            service.label_features_for_extraction(job_id)
        except Exception as exc:  # noqa: BLE001 — the test inspects it
            return exc
    return None


async def _run_worker(async_engine, job_id, judge=_UnreachableJudge):
    """Run the worker body as the worker does: a sync session, in a thread with
    no event loop running (the body creates and runs its own loop, which it
    cannot do inside `run_sync`'s greenlet)."""
    url = async_engine.url.set(drivername="postgresql+psycopg2")
    engine = create_engine(url, poolclass=NullPool)

    def _go():
        with Session(engine) as sync_db:
            return _run_unreachable_job(sync_db, job_id, judge)

    try:
        return await asyncio.to_thread(_go)
    finally:
        engine.dispose()


def _statuses(sync_db, eid):
    rows = sync_db.execute(
        sa.select(Feature.label_status, sa.func.count())
        .where(Feature.extraction_job_id == eid)
        .group_by(Feature.label_status)
    ).all()
    return {status: n for status, n in rows}


def _eligible(sync_db, eid):
    return {
        r[0] for r in sync_db.execute(
            sa.select(Feature.id).where(Feature.extraction_job_id == eid)
            .where(labeling_eligibility.eligibility_filter())
        ).all()
    }


@pytest.mark.asyncio
class TestAnUnreachableJudgeStrandsNothing:

    async def test_the_job_fails_and_no_feature_is_left_in_progress(self, async_session, async_engine):
        eid = await _extraction(async_session)
        await _features(async_session, eid, N_FEATURES)
        await _job(async_session, eid, "lbl_unreachable")

        raised = await _run_worker(async_engine, "lbl_unreachable")

        # The precondition the defect needs: the judge was asked, and the run
        # aborted as a JOB-level failure (otherwise this tests something else).
        assert isinstance(raised, JudgeUnavailable), raised
        assert _UnreachableJudge.calls >= 1, "the judge was never asked"

        statuses = await async_session.run_sync(_statuses, eid)
        assert statuses.get("in_progress", 0) == 0, (
            f"{statuses.get('in_progress')} features stranded in_progress by a "
            f"failed job — invisible to every resume"
        )
        assert statuses == {"pending": N_FEATURES}, statuses

        job = await async_session.get(LabelingJob, "lbl_unreachable", populate_existing=True)
        assert job.status == LabelingStatus.FAILED.value
        assert "APIConnectionError" in (job.error_message or "")

    async def test_the_released_batch_is_exactly_what_it_was(self, async_session, async_engine):
        """No attempt spent, no reason, no job id left behind: nothing was
        learned about any of them."""
        eid = await _extraction(async_session)
        ids = await _features(async_session, eid, N_FEATURES, label_attempts=1)
        await _job(async_session, eid, "lbl_unreachable2")

        await _run_worker(async_engine, "lbl_unreachable2")

        rows = (await async_session.execute(
            sa.select(Feature).where(Feature.id.in_(ids)).execution_options(populate_existing=True)
        )).scalars().all()
        for f in rows:
            assert f.label_status == "pending"
            assert f.label_attempts == 1, "a judge-level failure spent a retry"
            assert f.label_error is None
            assert f.labeling_job_id is None, "the claim's job id outlived the claim"

    async def test_a_resume_covers_every_feature_afterwards(self, async_session, async_engine):
        eid = await _extraction(async_session)
        ids = await _features(async_session, eid, N_FEATURES)
        await _job(async_session, eid, "lbl_unreachable3")

        await _run_worker(async_engine, "lbl_unreachable3")

        assert await async_session.run_sync(_eligible, eid) == set(ids), (
            "features the failed run claimed are not offered to a resume"
        )


@pytest.mark.asyncio
class TestAJudgeLostMidBatch:

    async def test_a_verdict_reached_before_the_loss_is_kept(self, async_session, async_engine):
        """The release puts back only what is STILL claimed. A feature the
        judge answered before it vanished keeps its verdict; the rest of the
        batch goes back to `pending`."""
        eid = await _extraction(async_session)
        ids = await _features(async_session, eid, BATCH)
        await _job(async_session, eid, "lbl_midbatch")

        raised = await _run_worker(async_engine, "lbl_midbatch", _JudgeThatGoesAwayAfterOne)
        assert isinstance(raised, JudgeUnavailable), raised

        first = await async_session.get(Feature, ids[0], populate_existing=True)
        assert first.label_status == "succeeded", "a real verdict was released"
        assert first.name == "cat_tokens"
        assert first.labeling_job_id == "lbl_midbatch"
        assert await async_session.run_sync(_statuses, eid) == {
            "succeeded": 1, "pending": BATCH - 1,
        }


@pytest.mark.asyncio
class TestAnyOtherFailureReleasesToo:

    async def test_an_unexpected_error_mid_batch_strands_nothing(self, async_session, async_engine):
        eid = await _extraction(async_session)
        await _features(async_session, eid, BATCH)
        await _job(async_session, eid, "lbl_crash")

        raised = await _run_worker(async_engine, "lbl_crash", _JudgeWithAMalformedAnswer)
        assert isinstance(raised, KeyError), raised

        assert await async_session.run_sync(_statuses, eid) == {"pending": BATCH}


def _run_labeling_janitor(sync_db):
    """The real `cleanup_stuck_labeling` body on a real session (ambient
    liveness probes pinned — see test_labeling_resume_sweep)."""
    from src.workers import cleanup_stuck_labeling as mod

    @contextmanager
    def _session(_self=None):
        yield sync_db

    with patch.object(
        mod.cleanup_stuck_labeling_task, "get_db", lambda: _session(None), create=True
    ), patch.object(mod, "task_looks_alive", lambda *a, **k: False), \
         patch.object(mod, "progress_stalled_seconds", lambda *a, **k: None):
        return mod.cleanup_stuck_labeling_task.run()


def _claim_with_the_real_writer(sync_db, ids, job_id, at):
    service = LabelingService(sync_db)
    features = sync_db.execute(sa.select(Feature).where(Feature.id.in_(ids))).scalars().all()
    service._claim_features(features, at, job_id)


@pytest.mark.asyncio
class TestTheJanitorFindsRealClaims:
    """The existing janitor tests seed `labeling_job_id` on a claimed row by
    hand, which production never did — a fixture that agreed with the janitor
    by construction. These claim through the real writer."""

    async def test_a_dead_workers_claim_is_released_by_its_job(self, async_session):
        eid = await _extraction(async_session)
        ids = await _features(async_session, eid, 3)
        await _job(async_session, eid, "lbl_dead", status=LabelingStatus.LABELING.value)
        # The job is reaped only when its row is stale; leave its claims fresh,
        # so only the per-job release (not the orphan sweep) can take them.
        await async_session.execute(
            sa.update(LabelingJob).where(LabelingJob.id == "lbl_dead")
            .values(updated_at=datetime.now(timezone.utc) - timedelta(hours=3))
        )
        await async_session.commit()
        await async_session.run_sync(
            _claim_with_the_real_writer, ids, "lbl_dead", datetime.now(timezone.utc)
        )

        await async_session.run_sync(_run_labeling_janitor)

        statuses = await async_session.run_sync(_statuses, eid)
        assert statuses == {"failed": 3}, (
            f"{statuses}: the janitor could not find claims made by the real writer"
        )

    async def test_an_orphaned_claim_of_a_finished_job_is_released(self, async_session):
        """The live shape: job already FAILED, claim carries no job id (every
        claim written before the fix)."""
        eid = await _extraction(async_session)
        old = datetime.now(timezone.utc) - timedelta(hours=2)
        ids = await _features(async_session, eid, 10, label_status="in_progress")
        await _job(async_session, eid, "lbl_failed", status=LabelingStatus.FAILED.value)
        await async_session.execute(
            sa.update(Feature).where(Feature.id.in_(ids)).values(updated_at=old)
        )
        await async_session.commit()

        result = await async_session.run_sync(_run_labeling_janitor)

        assert result["orphaned_claims"] == 10
        assert await async_session.run_sync(_statuses, eid) == {"failed": 10}
        assert await async_session.run_sync(_eligible, eid) == set(ids)

    async def test_a_claim_on_an_extraction_with_a_live_job_is_spared(self, async_session):
        """A claim's `updated_at` is the job's START time, so a healthy batch
        can look hours old. Both kinds of extraction are present, so a sweep
        that ignored the live-job guard would release the wrong one."""
        live_eid = await _extraction(async_session)
        dead_eid = await _extraction(async_session)
        old = datetime.now(timezone.utc) - timedelta(hours=2)
        live_ids = await _features(async_session, live_eid, 2, label_status="in_progress")
        dead_ids = await _features(async_session, dead_eid, 2, label_status="in_progress")
        await _job(async_session, live_eid, "lbl_running", status=LabelingStatus.LABELING.value)
        await async_session.execute(
            sa.update(Feature).where(Feature.id.in_(live_ids + dead_ids)).values(updated_at=old)
        )
        await async_session.commit()

        await async_session.run_sync(_run_labeling_janitor)

        assert await async_session.run_sync(_statuses, live_eid) == {"in_progress": 2}
        assert await async_session.run_sync(_statuses, dead_eid) == {"failed": 2}

    async def test_a_fresh_orphan_is_left_for_the_next_cycle(self, async_session):
        eid = await _extraction(async_session)
        await _features(async_session, eid, 2, label_status="in_progress")

        await async_session.run_sync(_run_labeling_janitor)

        assert await async_session.run_sync(_statuses, eid) == {"in_progress": 2}
