"""A sweep runs many batches without ever being a long task.

WHY THIS SHAPE. Finishing the L46 extraction is ~27 batches of 2000 at the
measured ~8 s/feature — about 59 GPU-hours. Celery's soft limit here is 10 h and
the hard limit 12 h, so a task that LOOPED over batches would be killed part-way
through and, on an `acks_late` queue, strand its message for the full 12 h
visibility timeout. That shape has already cost this project an outage.

So the sweep is a row plus a task that does ONE batch and re-enqueues itself.
Every test here exists to hold some part of that.
"""

import inspect
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.models.labeling_job import LabelingJob, LabelingStatus
from src.models.labeling_resume_sweep import LabelingResumeSweep
from src.services.labeling_sweep_service import (
    MAX_ALLOWED_BATCHES,
    LabelingSweepService,
    SweepError,
)


def _sweep(**overrides) -> LabelingResumeSweep:
    base = dict(
        id=f"sweep_{uuid.uuid4().hex[:8]}",
        extraction_job_id="ext_1",
        config={"labeling_method": "openai_compatible"},
        status="running",
        batches_done=0,
        features_labeled=0,
        features_failed=0,
        max_batches=27,
        batch_size=2000,
        progress=0.0,
    )
    base.update(overrides)
    return LabelingResumeSweep(**base)


@pytest.fixture
def service():
    return LabelingSweepService(MagicMock())


# ── the shape that keeps it off the soft limit ───────────────────────────────

class TestOneBatchPerTask:

    def test_the_step_re_enqueues_and_never_loops(self):
        """THE load-bearing property. A loop over batches exceeds the 10 h soft
        limit and strands an acks_late message for 12 h."""
        from src.workers.labeling_tasks import resume_sweep_step

        source = inspect.getsource(resume_sweep_step)
        assert "resume_sweep_step.delay(sweep_id)" in source, (
            "the step must re-enqueue itself; without it a sweep stops after one batch"
        )
        # No loop over batches inside the task.
        assert "for _ in range" not in source
        assert "while True" not in source
        assert source.count("label_features_for_extraction(") == 1, (
            "one batch per task — a second call here is a loop wearing a disguise"
        )

    def test_it_is_registered_with_celery_under_a_qualified_name(self):
        """`task_routes` globs match the TASK NAME, and a short name silently
        lands on the default queue."""
        from src.core.celery_app import celery_app
        from src.workers.labeling_tasks import resume_sweep_step

        assert resume_sweep_step.name == "labeling.resume_sweep_step"
        assert resume_sweep_step.name in celery_app.tasks, (
            "the task is not in the live Celery registry — nothing can run it"
        )

    def test_the_cancel_scope_is_registered(self):
        """A sweep outlives any one of its tasks, so it needs its own scope:
        cancelling the batch in flight is a different intention from stopping
        the remaining twenty-six."""
        from src.core.cancellation import SCOPES

        assert "labeling_sweep" in SCOPES
        assert "labeling" in SCOPES
        assert SCOPES["labeling_sweep"] is not SCOPES["labeling"]


# ── the ceiling ──────────────────────────────────────────────────────────────

class TestTheCeilingIsMandatory:

    def test_a_sweep_cannot_be_created_without_one(self):
        """`max_batches` is keyword-only with no default, so omitting it is a
        TypeError rather than an unbounded sweep."""
        sig = inspect.signature(LabelingSweepService.create)
        param = sig.parameters["max_batches"]
        assert param.default is inspect.Parameter.empty, (
            "max_batches must have NO default: an open-ended sweep is a request "
            "to spend an unknown number of GPU-hours"
        )
        assert param.kind is inspect.Parameter.KEYWORD_ONLY

    def test_the_column_refuses_a_null_ceiling(self):
        col = LabelingResumeSweep.__table__.c["max_batches"]
        assert col.nullable is False
        assert col.default is None and col.server_default is None, (
            "a default here would let an unbounded sweep be inserted directly"
        )

    @pytest.mark.parametrize("bad", [0, -1, MAX_ALLOWED_BATCHES + 1, 10_000])
    def test_an_out_of_range_ceiling_is_refused(self, service, bad):
        with pytest.raises(SweepError, match="max_batches"):
            service.create("ext_1", max_batches=bad, config={})

    def test_a_reached_ceiling_stops_the_sweep(self, service):
        assert service.should_stop(_sweep(batches_done=5, max_batches=5)) is not None
        assert service.should_stop(_sweep(batches_done=4, max_batches=5)) is None


# ── stopping ─────────────────────────────────────────────────────────────────

class TestStoppingIsCooperative:

    def test_a_stop_request_halts_the_next_step(self, service):
        stopped = _sweep(cancel_requested_at=datetime.now(timezone.utc))
        assert "stop requested" in service.should_stop(stopped)

    def test_the_guard_runs_before_any_work(self):
        """A stop honoured only between whole sweeps is not a stop."""
        from src.workers.labeling_tasks import resume_sweep_step

        source = inspect.getsource(resume_sweep_step)
        assert source.index("should_stop(") < source.index("next_batch_ids("), (
            "the sweep must check for a stop BEFORE computing or running a batch"
        )

    def test_a_terminal_sweep_is_not_re_stamped(self, service):
        """Stamping a finished sweep would rewrite history and make a completed
        run look abandoned."""
        done = _sweep(status="completed")
        service.db.query.return_value.filter.return_value.first.return_value = done
        service.request_stop(done.id)
        assert done.cancel_requested_at is None

    def test_a_non_running_sweep_stops_immediately(self, service):
        for status in ("completed", "failed", "cancelled"):
            assert service.should_stop(_sweep(status=status)) is not None

    def test_a_stale_cancel_request_is_cleared_on_the_first_step(self):
        """A leftover request must not outlive the sweep that earned it — the
        same trap that made a cancelled model download undownloadable."""
        from src.workers.labeling_tasks import resume_sweep_step

        source = inspect.getsource(resume_sweep_step)
        assert 'clear_cancel_request("labeling_sweep", sweep_id)' in source
        assert "sweep.batches_done == 0" in source, (
            "clearing on EVERY step would erase a stop the operator just requested"
        )


# ── honest progress ──────────────────────────────────────────────────────────

class TestProgressReflectsWhatWasWritten:

    def test_counters_come_from_the_batch_job_not_the_batch_size(self, service):
        """A sweep whose every batch failed would report "27 of 27 complete" if
        it counted batches."""
        sweep = _sweep()
        job = LabelingJob(
            id="lbl_1", extraction_job_id="ext_1",
            statistics={"successfully_labeled": 1750, "failed_labels": 250},
        )
        service.record_batch(sweep, job)
        assert sweep.batches_done == 1
        assert sweep.features_labeled == 1750
        assert sweep.features_failed == 250

    def test_an_all_failing_batch_advances_no_labelled_count(self, service):
        sweep = _sweep()
        job = LabelingJob(
            id="lbl_1", extraction_job_id="ext_1",
            statistics={"successfully_labeled": 0, "failed_labels": 2000},
        )
        service.record_batch(sweep, job)
        assert sweep.features_labeled == 0
        assert sweep.features_failed == 2000

    def test_a_batch_with_no_statistics_does_not_invent_progress(self, service):
        sweep = _sweep()
        service.record_batch(sweep, LabelingJob(id="lbl_1", extraction_job_id="ext_1"))
        assert sweep.features_labeled == 0
        assert sweep.batches_done == 1


# ── completion ───────────────────────────────────────────────────────────────

class TestNothingLeftIsSuccess:

    def test_an_empty_batch_completes_rather_than_errors(self):
        """The sweep did its job. Reporting failure would send an operator
        looking for a problem that does not exist."""
        from src.workers.labeling_tasks import resume_sweep_step

        source = inspect.getsource(resume_sweep_step)
        assert 'service.finish(sweep, "completed", "no features left to label")' in source

    def test_finish_stamps_a_terminal_row(self, service):
        sweep = _sweep()
        service.finish(sweep, "completed", "nothing left")
        assert sweep.status == "completed"
        assert sweep.completed_at is not None

    def test_a_failure_reason_is_recorded_and_truncated(self, service):
        sweep = _sweep()
        service.finish(sweep, "failed", "x" * 5000)
        assert sweep.status == "failed"
        assert len(sweep.error_message) == 1000


# ── two sweeps must not race ─────────────────────────────────────────────────

class TestOneRunningSweepPerExtraction:
    """Two sweeps on one extraction would compute overlapping batches from the
    same `pending` rows and pay twice for one result. Per-batch claiming narrows
    that window to one batch; refusing the second sweep closes it."""

    def test_a_second_sweep_is_refused_while_one_runs(self, service):
        running = _sweep(status="running")
        service.db.query.return_value.filter.return_value.first.side_effect = [
            object(),   # the extraction exists
            running,    # …and a sweep is already running on it
        ]
        with pytest.raises(SweepError, match="already running"):
            service.create("ext_1", max_batches=5, config={})

    def test_a_new_sweep_is_allowed_once_the_previous_finished(self, service):
        service.db.query.return_value.filter.return_value.first.side_effect = [
            object(),  # extraction exists
            None,      # no RUNNING sweep
        ]
        sweep = service.create("ext_1", max_batches=5, config={})
        assert sweep.status == "running"
        assert sweep.max_batches == 5

    def test_an_unknown_extraction_is_refused(self, service):
        service.db.query.return_value.filter.return_value.first.return_value = None
        with pytest.raises(SweepError, match="not found"):
            service.create("ext_nope", max_batches=1, config={})


class TestTheJudgeConfigIsFrozen:
    """A sweep whose batches drift between models or templates produces labels
    that cannot be compared, and nothing afterwards can say which half came from
    which."""

    def test_the_config_is_copied_not_referenced(self, service):
        service.db.query.return_value.filter.return_value.first.side_effect = [object(), None]
        original = {"labeling_method": "openai", "prompt_template_id": "lpt_a"}
        sweep = service.create("ext_1", max_batches=2, config=original)

        original["prompt_template_id"] = "lpt_b"
        assert sweep.config["prompt_template_id"] == "lpt_a", (
            "the sweep must hold its own copy; a caller mutating the dict "
            "afterwards would change the judge mid-sweep"
        )

    def test_every_batch_is_built_from_the_sweep_config(self):
        from src.workers.labeling_tasks import resume_sweep_step

        source = inspect.getsource(resume_sweep_step)
        assert "config=sweep.config or {}" in source, (
            "a batch built from anything but the frozen config can drift"
        )

    def test_batches_share_one_row_builder_with_the_hand_started_path(self):
        """`start_labeling` is async and the sweep task is sync, so neither can
        call the other. A second construction site would drift — which is how
        `retryLabeling` came to drop the endpoint, the template and every filter
        flag while looking like it worked."""
        from src.services.labeling_service import LabelingService
        from src.workers.labeling_tasks import resume_sweep_step

        assert "LabelingService.build_labeling_job_row(" in inspect.getsource(
            resume_sweep_step
        )
        assert "self.build_labeling_job_row(" in inspect.getsource(
            LabelingService.start_labeling
        )


class TestTheSweepEndpointsAreReachable:
    """A capability is not shipped until a test FAILS when its wiring is removed.
    Asserted against `app.openapi()["paths"]`, never `app.routes` — on this
    FastAPI version the latter reports an app serving only framework defaults."""

    PATHS = {
        "/api/v1/labeling/{extraction_job_id}/resume-sweep": "post",
        "/api/v1/labeling/resume-sweeps/{sweep_id}": "get",
        "/api/v1/labeling/resume-sweeps/{sweep_id}/cancel": "post",
    }

    def test_all_three_are_registered_on_the_built_app(self):
        from src.main import app

        paths = app.openapi()["paths"]
        for path, method in self.PATHS.items():
            assert path in paths, (
                f"{path} is not in the live OpenAPI. Registered: "
                f"{sorted(p for p in paths if 'sweep' in p)}"
            )
            assert method in paths[path]

    def test_max_batches_is_required_in_the_request_schema(self):
        """An agent or a curl must not be able to start an unbounded sweep by
        omitting a field."""
        from src.main import app

        schema = app.openapi()["components"]["schemas"]["ResumeSweepRequest"]
        assert "max_batches" in schema.get("required", []), (
            "max_batches must be REQUIRED, not defaulted"
        )
        assert schema["properties"]["max_batches"]["maximum"] == 100



# ── DB helpers for the behavioural R2 tests ──────────────────────────────────
#
# Real rows, real filters. The Mock-based versions of these tests let mutation
# controls C47, C48 and C51 survive: a Mock returns whatever it was told to
# regardless of the WHERE clause, so the predicate under test never ran.

from datetime import timedelta  # noqa: E402

import sqlalchemy as sa  # noqa: E402

from src.models.dataset import Dataset  # noqa: E402
from src.models.external_sae import ExternalSAE  # noqa: E402
from src.models.extraction_job import ExtractionJob  # noqa: E402
from src.models.feature import Feature  # noqa: E402
from src.models.model import Model  # noqa: E402
from src.services import labeling_eligibility  # noqa: E402


async def _db_extraction(session, eid):
    mid, did = f"m_{eid}", str(uuid.uuid4())
    session.add(Model(id=mid, name=f"model {eid}", architecture="test", params_count=1))
    session.add(Dataset(id=did, name=f"dataset {eid}", source="Local"))
    await session.commit()
    session.add(ExternalSAE(id=f"sae_{eid}", name=f"sae {eid}", source="trained"))
    await session.commit()
    session.add(ExtractionJob(id=eid, external_sae_id=f"sae_{eid}", config={}))
    await session.commit()
    return eid


async def _db_job(session, eid, job_id, status):
    session.add(LabelingJob(
        id=job_id, extraction_job_id=eid, labeling_method="openai", status=status,
        progress=0.0, features_labeled=0,
    ))
    await session.commit()
    return job_id


async def _db_sweep(session, eid, sweep_id, *, updated_at, status="running"):
    session.add(LabelingResumeSweep(
        id=sweep_id, extraction_job_id=eid, config={}, status=status,
        batches_done=1, features_labeled=10, features_failed=0,
        max_batches=27, batch_size=2000, progress=0.0,
    ))
    await session.commit()
    # `updated_at` has an onupdate, so it must be forced after the insert.
    await session.execute(
        sa.update(LabelingResumeSweep)
        .where(LabelingResumeSweep.id == sweep_id)
        .values(updated_at=updated_at)
    )
    await session.commit()
    return sweep_id


def _run_sweep_janitor(sync_db):
    """Run the janitor's body against a real sync session.

    The Celery task opens its own session through `DatabaseTask`, which a unit
    test has no business doing. Patching `get_db` to hand it this session runs
    the REAL code — the query, the filter and the sparing logic — rather than a
    reimplementation of it that could agree with a bug.
    """
    from contextlib import contextmanager
    from unittest.mock import patch

    from src.workers import cleanup_stuck_labeling as mod
    from src.workers.base_task import DatabaseTask

    @contextmanager
    def _session(_self):
        yield sync_db

    # Patch the BASE class: Celery wraps the task in a PromiseProxy, so
    # `type(task)` is not the class that declares `get_db`.
    with patch.object(DatabaseTask, "get_db", _session):
        result = mod.cleanup_stuck_labeling_sweeps_task.run()
    return result["cleaned"]



def _run_sweep_step(sync_db, sweep_id):
    """Run the real task body against a real sync session.

    `resume_sweep_step.delay` is stubbed so the test does not need a broker, and
    so a stood-down duplicate can be shown to enqueue NOTHING — which is the
    property under test. The batch work itself is stubbed too: this is about
    whether the step decides to start one, not about labeling.
    """
    from contextlib import contextmanager
    from unittest.mock import MagicMock, patch

    from src.workers import labeling_tasks as mod
    from src.workers.base_task import DatabaseTask

    @contextmanager
    def _session(_self):
        yield sync_db

    with patch.object(DatabaseTask, "get_db", _session), \
         patch.object(mod.resume_sweep_step, "delay") as delay, \
         patch.object(
             mod.LabelingService, "label_features_for_extraction", MagicMock()
         ):
        result = mod.resume_sweep_step.run(sweep_id)
    return result, delay



async def _db_feature_claimed(session, eid, fid, job_id):
    session.add(Feature(
        id=fid, name=fid, neuron_index=abs(hash(fid)) % 10000,
        extraction_job_id=eid, external_sae_id=f"sae_{eid}",
        activation_frequency=0.5, mean_activation=1.0, max_activation=2.0,
        interpretability_score=0.4,
        label_status="in_progress", label_attempts=1, labeling_job_id=job_id,
    ))
    await session.commit()
    return fid


def _run_labeling_janitor(sync_db):
    """The real `cleanup_stuck_labeling` body against a real sync session.

    `task_looks_alive` and `progress_stalled_seconds` are pinned, as the sibling
    `test_cleanup_stuck_labeling.py` pins them. Both consult AMBIENT STATE —
    Celery's inspect and Redis progress markers — so leaving them live made these
    tests pass alone and fail in the full suite, where other tests had written
    markers that made the job look alive and spared it.

    A test whose result depends on what ran before it is not testing the janitor.
    """
    from contextlib import contextmanager
    from unittest.mock import patch

    from src.workers import cleanup_stuck_labeling as mod
    from src.workers.base_task import DatabaseTask

    @contextmanager
    def _session(_self=None):
        yield sync_db

    # Patch the TASK INSTANCE, not `DatabaseTask`.
    #
    # `test_cleanup_stuck_labeling.py` sets `get_db` on the instance, and an
    # instance attribute SHADOWS the class one — so patching the class made
    # these tests pass alone and fail in the full suite, where the janitor
    # silently ran against the sibling file's Mock session and never saw these
    # rows. Same target, so ordering cannot decide the outcome.
    with patch.object(
        mod.cleanup_stuck_labeling_task, "get_db", lambda: _session(None), create=True
    ), patch.object(mod, "task_looks_alive", lambda *a, **k: False), \
         patch.object(mod, "progress_stalled_seconds", lambda *a, **k: None):
        return mod.cleanup_stuck_labeling_task.run()


# ── R2: the failure modes a static read does not surface ─────────────────────

class TestADuplicateStepStandsDown:
    """R2. `start_labeling` refuses a second job on an extraction that already
    has one QUEUED or LABELING. The sweep task builds its row through
    `build_labeling_job_row` and BYPASSED that guard, so two deliveries of one
    step — an `acks_late` redelivery, a hand re-trigger, a worker that lost its
    connection mid-ack — would each start a batch on the same extraction.

    THESE ARE BEHAVIOURAL, NOT SOURCE SCRAPES. The first version asserted that
    `active_labeling_job(` appeared in the task source, and mutation control C47
    (`if active is not None:` -> `if False:`) SURVIVED it — the call is still
    there, it just does nothing. A guard that reads source text fails open, which
    this repo has shipped twice.
    """

    async def test_the_query_finds_an_in_flight_batch(self, async_session):
        """Against a real session, so the status filter actually runs. A Mock
        returns the row whatever the filter says (control C51)."""
        eid = await _db_extraction(async_session, f"ext_{uuid.uuid4().hex[:8]}")
        await _db_job(async_session, eid, "lbl_running", LabelingStatus.LABELING.value)

        found = await async_session.run_sync(
            lambda db: LabelingSweepService(db).active_labeling_job(eid)
        )
        assert found is not None and found.id == "lbl_running"

    async def test_a_finished_batch_does_not_block(self, async_session):
        eid = await _db_extraction(async_session, f"ext_{uuid.uuid4().hex[:8]}")
        await _db_job(async_session, eid, "lbl_done", LabelingStatus.COMPLETED.value)

        found = await async_session.run_sync(
            lambda db: LabelingSweepService(db).active_labeling_job(eid)
        )
        assert found is None, "a completed batch must not stall the sweep forever"

    async def test_a_queued_batch_blocks_too(self, async_session):
        """QUEUED is in flight: the worker has not picked it up yet, and starting
        a second batch would race it."""
        eid = await _db_extraction(async_session, f"ext_{uuid.uuid4().hex[:8]}")
        await _db_job(async_session, eid, "lbl_q", LabelingStatus.QUEUED.value)

        found = await async_session.run_sync(
            lambda db: LabelingSweepService(db).active_labeling_job(eid)
        )
        assert found is not None

    async def test_the_step_starts_no_batch_while_one_is_in_flight(self, async_session):
        """BEHAVIOURAL, and the one that matters.

        Control C47 (`if active is not None:` -> `if False:`) survived a
        source-scrape version of this test twice: the call was still present, it
        simply did nothing. Only running the step and counting the jobs it
        creates can see that.
        """
        eid = await _db_extraction(async_session, f"ext_{uuid.uuid4().hex[:8]}")
        await _db_sweep(async_session, eid, "sweep_dup",
                        updated_at=datetime.now(timezone.utc))
        await _db_job(async_session, eid, "lbl_inflight", LabelingStatus.LABELING.value)

        before = await async_session.scalar(
            sa.select(sa.func.count()).select_from(LabelingJob)
            .where(LabelingJob.extraction_job_id == eid)
        )
        result, delay = await async_session.run_sync(
            lambda db: _run_sweep_step(db, "sweep_dup")
        )
        after = await async_session.scalar(
            sa.select(sa.func.count()).select_from(LabelingJob)
            .where(LabelingJob.extraction_job_id == eid)
        )

        assert after == before, (
            "the step started a second batch while one was already running — "
            "two chains racing through the same features"
        )
        assert "already in flight" in result.get("reason", "")
        delay.assert_not_called()

    async def test_a_duplicate_enqueues_nothing(self, async_session):
        """Re-enqueueing from a stood-down duplicate forks the sweep into two
        chains, which is worse than the duplicate batch it just avoided."""
        eid = await _db_extraction(async_session, f"ext_{uuid.uuid4().hex[:8]}")
        await _db_sweep(async_session, eid, "sweep_dup2",
                        updated_at=datetime.now(timezone.utc))
        await _db_job(async_session, eid, "lbl_busy", LabelingStatus.QUEUED.value)

        _, delay = await async_session.run_sync(
            lambda db: _run_sweep_step(db, "sweep_dup2")
        )
        delay.assert_not_called()

    async def test_another_extractions_batch_does_not_block(self, async_session):
        mine = await _db_extraction(async_session, f"ext_{uuid.uuid4().hex[:8]}")
        theirs = await _db_extraction(async_session, f"ext_{uuid.uuid4().hex[:8]}")
        await _db_job(async_session, theirs, "lbl_other", LabelingStatus.LABELING.value)

        found = await async_session.run_sync(
            lambda db: LabelingSweepService(db).active_labeling_job(mine)
        )
        assert found is None


class TestAWedgedSweepIsReaped:
    """R2. A step records its batch and THEN re-enqueues. A worker killed in
    that window leaves `status='running'` with no task queued — and because
    `create` refuses a second running sweep, the wedge LOCKS the extraction out
    of sweeping until someone edits the row by hand."""

    def test_the_janitor_exists_and_is_registered(self):
        from src.core.celery_app import celery_app
        from src.workers.cleanup_stuck_labeling import cleanup_stuck_labeling_sweeps_task

        assert cleanup_stuck_labeling_sweeps_task.name == "cleanup_stuck_labeling_sweeps"
        assert cleanup_stuck_labeling_sweeps_task.name in celery_app.tasks

    def test_it_is_actually_scheduled(self):
        """A janitor nothing runs is a janitor that does not exist."""
        from src.core.celery_app import celery_app

        assert "cleanup-stuck-labeling-sweeps" in celery_app.conf.beat_schedule
        assert (
            celery_app.conf.beat_schedule["cleanup-stuck-labeling-sweeps"]["task"]
            == "cleanup_stuck_labeling_sweeps"
        )

    def test_it_is_routed_to_a_real_queue(self):
        from src.core.celery_app import celery_app

        assert "cleanup_stuck_labeling_sweeps" in celery_app.conf.task_routes

    def test_its_threshold_clears_a_full_batch(self):
        """A batch runs up to ~4.4 h and the row is only touched when one
        finishes, so a healthy sweep legitimately looks untouched that long."""
        from src.workers.cleanup_stuck_labeling import _SWEEP_STUCK_THRESHOLD_MINUTES

        assert _SWEEP_STUCK_THRESHOLD_MINUTES > (2000 * 8) / 60

    async def test_it_reaps_a_sweep_with_no_batch_in_flight(self, async_session):
        """BEHAVIOURAL. Control C48 survived a source-scrape version of this."""
        eid = await _db_extraction(async_session, f"ext_{uuid.uuid4().hex[:8]}")
        old = datetime.now(timezone.utc) - timedelta(days=1)
        await _db_sweep(async_session, eid, "sweep_wedged", updated_at=old)

        reaped = await async_session.run_sync(_run_sweep_janitor)
        assert reaped == 1

        sweep = await async_session.get(LabelingResumeSweep, "sweep_wedged")
        assert sweep.status == "failed"
        assert "worker was restarted" in sweep.error_message

    async def test_it_spares_a_sweep_whose_batch_is_still_running(self, async_session):
        """Absence of evidence is not evidence of death. A batch runs for hours
        and does not touch the sweep row while it does."""
        eid = await _db_extraction(async_session, f"ext_{uuid.uuid4().hex[:8]}")
        old = datetime.now(timezone.utc) - timedelta(days=1)
        await _db_sweep(async_session, eid, "sweep_working", updated_at=old)
        await _db_job(async_session, eid, "lbl_inflight", LabelingStatus.LABELING.value)

        reaped = await async_session.run_sync(_run_sweep_janitor)
        assert reaped == 0

        sweep = await async_session.get(LabelingResumeSweep, "sweep_working")
        assert sweep.status == "running", "the janitor killed a working sweep"

    async def test_it_leaves_a_recently_active_sweep_alone(self, async_session):
        eid = await _db_extraction(async_session, f"ext_{uuid.uuid4().hex[:8]}")
        await _db_sweep(async_session, eid, "sweep_fresh",
                        updated_at=datetime.now(timezone.utc))

        assert await async_session.run_sync(_run_sweep_janitor) == 0


class TestAClaimedFeatureIsReleasedWhenItsJobIsReaped:
    """R4 HARDWARE FINDING, and it falsified a docstring.

    `_claim_features` claimed that "stale in_progress rows are reclaimed by the
    existing cleanup_stuck_labeling sweeper". They were not. The janitor marked
    the JOB failed and never touched `features.label_status`, so a feature
    claimed by a worker that died stayed `in_progress` FOREVER — excluded from
    eligibility by design, and therefore invisible to every future resume.

    Observed by restarting the backend pod mid-panel: 10 of 15 succeeded and the
    other 5 were stranded, offered to nobody.

    An unverified sentence in a docstring is how a capability gets believed into
    existence. This is the test that makes it true.
    """

    async def test_the_janitor_releases_what_the_job_held(self, async_session):
        eid = await _db_extraction(async_session, f"ext_{uuid.uuid4().hex[:8]}")
        old = datetime.now(timezone.utc) - timedelta(hours=3)
        await _db_job(async_session, eid, "lbl_dead", LabelingStatus.LABELING.value)
        await async_session.execute(
            sa.update(LabelingJob).where(LabelingJob.id == "lbl_dead")
            .values(updated_at=old)
        )
        for i in range(3):
            await _db_feature_claimed(async_session, eid, f"held{i}", "lbl_dead")
        await async_session.commit()

        await async_session.run_sync(_run_labeling_janitor)

        rows = await async_session.execute(
            sa.select(Feature.label_status, Feature.label_error)
            .where(Feature.labeling_job_id == "lbl_dead")
        )
        for status, error in rows.all():
            assert status == "failed", "a claimed feature was left in_progress"
            assert "worker was restarted" in (error or "")

    async def test_released_features_become_eligible_again(self, async_session):
        """The whole point: a resume must be able to pick them up."""
        eid = await _db_extraction(async_session, f"ext_{uuid.uuid4().hex[:8]}")
        old = datetime.now(timezone.utc) - timedelta(hours=3)
        await _db_job(async_session, eid, "lbl_dead2", LabelingStatus.LABELING.value)
        await async_session.execute(
            sa.update(LabelingJob).where(LabelingJob.id == "lbl_dead2")
            .values(updated_at=old)
        )
        await _db_feature_claimed(async_session, eid, "stranded", "lbl_dead2")
        await async_session.commit()

        before = await async_session.execute(
            elig.resume_batch_query(eid, limit=10) if False else
            sa.select(Feature.id).where(Feature.extraction_job_id == eid)
            .where(labeling_eligibility.eligibility_filter())
        )
        assert [r[0] for r in before.all()] == [], "claimed work must not be offered"

        await async_session.run_sync(_run_labeling_janitor)

        after = await async_session.execute(
            sa.select(Feature.id).where(Feature.extraction_job_id == eid)
            .where(labeling_eligibility.eligibility_filter())
        )
        assert [r[0] for r in after.all()] == ["stranded"]

    async def test_it_does_not_touch_features_of_a_healthy_job(self, async_session):
        """A STUCK job and a HEALTHY one, together.

        The first version of this test created only the healthy job — so the
        janitor found nothing to reap, its loop never ran, and the test passed
        whatever the release query said. Mutation control C59 (dropping the
        `labeling_job_id == job.id` scope, which releases every in_progress
        feature in the database) SURVIVED it.

        The release must be scoped to the job being reaped, and only a fixture
        with both kinds of job present can show that.
        """
        eid = await _db_extraction(async_session, f"ext_{uuid.uuid4().hex[:8]}")
        old_ts = datetime.now(timezone.utc) - timedelta(hours=3)

        await _db_job(async_session, eid, "lbl_reaped", LabelingStatus.LABELING.value)
        await async_session.execute(
            sa.update(LabelingJob).where(LabelingJob.id == "lbl_reaped")
            .values(updated_at=old_ts)
        )
        await _db_feature_claimed(async_session, eid, "stranded2", "lbl_reaped")

        await _db_job(async_session, eid, "lbl_live", LabelingStatus.LABELING.value)
        await _db_feature_claimed(async_session, eid, "busy", "lbl_live")
        await async_session.commit()

        await async_session.run_sync(_run_labeling_janitor)

        reaped = await async_session.get(Feature, "stranded2")
        live = await async_session.get(Feature, "busy")
        await async_session.refresh(reaped)
        await async_session.refresh(live)

        assert reaped.label_status == "failed", "the stuck job's claim was not released"
        assert live.label_status == "in_progress", (
            "the janitor released a feature from a job that is STILL RUNNING — "
            "a second worker would then label it concurrently"
        )
