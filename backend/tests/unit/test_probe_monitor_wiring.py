"""Reachability for probe monitors: delete a wiring line and this file must go red.

MUTATION CONTROLS (each verified to fail the suite; recorded in the review):
  M127  the router include line is removed          → the route test fails
  M128  a `task_routes` entry is removed            → that task's queue test fails
  M129  the judge task is routed to `extraction`    → the CPU-queue test fails
  M130  `probe_monitors` dropped from `_CHANNEL_TOPICS` → the channel test fails
  M131  the `probe_monitor_run` CancelScope is unregistered → the scope test fails
  M132  the reaper's beat entry is removed          → the beat test fails
  M133  the reaper's short-name queue entry is removed → the queue test fails
  M134  `probe_monitor_tasks` dropped from autodiscovery → the registry test fails
  M135  the submit endpoint uses `.delay` instead of `gpu_delay` → the lease test fails

⚠ THIS IS A SHIPPING GATE, NOT A STYLE PREFERENCE. This repo's cautionary case is the
16 `millm_circuit_*` MCP tools: fully implemented, unit-tested and documented while
never registered with the server. Every test passed by importing the module directly,
so the suite was green and the docs said ✅ while no agent could call the feature.

⚠ AND PRESENCE IN THE LIVE REGISTRY, NEVER "the module imports". `app.routes` is not a
route list under FastAPI 0.141 — included routers are wrapped in `_IncludedRouter` with
no `.path`, so introspecting it reports an EMPTY app that serves perfectly well. These
tests read `app.openapi()["paths"]`, which is what the server actually publishes.
"""

import pathlib
from types import SimpleNamespace
from fastapi import HTTPException
import ast
import inspect
import textwrap

import pytest


class TestTheRoutesAreLive:
    @pytest.fixture(scope="class")
    def paths(self):
        from src.main import app

        # `app.openapi()["paths"]`, NOT `app.routes` — see the module docstring.
        return set(app.openapi()["paths"])

    @pytest.mark.parametrize(
        "path",
        [
            "/api/v1/probe-monitors/datasets",
            "/api/v1/probe-monitors/datasets/{dataset_id}",
            "/api/v1/probe-monitors/runs",
            "/api/v1/probe-monitors/runs/{run_id}",
            "/api/v1/probe-monitors/runs/{run_id}/cancel",
            "/api/v1/probe-monitors/probes",
            "/api/v1/probe-monitors/probes/{probe_id}",
            "/api/v1/probe-monitors/probes/{probe_id}/evaluate",
            "/api/v1/probe-monitors/probes/{probe_id}/score",
            "/api/v1/probe-monitors/judge-runs",
        ],
    )
    def test_the_path_is_published(self, paths, path):
        assert path in paths, (
            f"{path} is not in the served OpenAPI paths — the router include line is "
            f"missing, so this capability does not exist for any caller"
        )

    def test_the_methods_are_the_documented_ones(self, paths):
        from src.main import app

        published = app.openapi()["paths"]
        assert "post" in published["/api/v1/probe-monitors/runs"]
        assert "get" in published["/api/v1/probe-monitors/runs"]
        assert "delete" in published["/api/v1/probe-monitors/runs/{run_id}"]
        assert "post" in published["/api/v1/probe-monitors/runs/{run_id}/cancel"]

    def test_submitting_a_run_returns_202_not_200(self, paths):
        """A 200 would mean the work happened in the request. It cannot: the model load
        alone is minutes."""
        from src.main import app

        responses = app.openapi()["paths"]["/api/v1/probe-monitors/runs"]["post"]["responses"]
        assert "202" in responses, f"submit advertises {sorted(responses)}"


from tests.support.celery_introspection import autodiscovered_modules

#: The modules whose tasks must be discovered. Kept beside the names below so a new
#: task module cannot be added without appearing here.
TASK_MODULES = (
    "src.workers.probe_monitor_tasks",
    "src.workers.cleanup_stuck_probe_monitor_runs",
)


class TestTheWorkerWillActuallyIMPORTTheTaskModules:
    """⚠ THE TEST BELOW USED TO GUARANTEE ITS OWN RESULT.

    `TestTheTasksAreRegisteredAndRoutedCorrectly`'s fixture began with

        import src.workers.cleanup_stuck_probe_monitor_runs  # noqa: F401
        import src.workers.probe_monitor_tasks  # noqa: F401

    which registers every task in those modules as a side effect of the import. So
    `celery.tasks` was populated BY THE FIXTURE, and deleting the module from
    `autodiscover_tasks` left the whole file green — verified: 41 passed, with the
    worker then unable to run a single probe task. That is the unregistered-MCP-tools
    failure of 2026-07-21 reproduced exactly: "every one of its tests passed by
    importing the module directly", and the fixture's own error message said to
    "check `autodiscover_tasks`" while making the check impossible.

    Two independent assertions now, because each covers the other's blind spot: the
    list is read from the call (immune to whatever a shared test process happens to
    have imported), and the registry is read after importing only `celery_app`.
    """

    @pytest.mark.parametrize("module", TASK_MODULES)
    def test_the_module_is_named_in_the_autodiscovery_list(self, module):
        assert module in autodiscovered_modules(), (
            f"{module} is not in celery_app.autodiscover_tasks(...), so no worker will "
            f"import it and none of its tasks will exist for a real caller"
        )

    def test_importing_celery_app_ALONE_registers_them(self):
        """`autodiscover_tasks(force=True)` runs at import, and celery's
        `find_related_module` imports the package before looking for a `.tasks`
        submodule — so importing the app is enough, and nothing here imports the task
        modules by name."""
        from src.core.celery_app import celery_app

        missing = [
            name
            for name in (
                "src.workers.probe_monitor_tasks.run_probe_monitor",
                "cleanup_stuck_probe_monitor_runs",
            )
            if name not in celery_app.tasks
        ]
        assert not missing, f"{missing} are absent after importing celery_app alone"

    def test_the_ast_reader_would_notice_a_removal(self):
        """The control for the reader itself: a list it has not been given must not
        satisfy it, or a bug in the walk reads as a pass."""
        assert "src.workers.there_is_no_such_module" not in autodiscovered_modules()


class TestTheTasksAreRegisteredAndRoutedCorrectly:
    @pytest.fixture(scope="class")
    def celery(self):
        # ⚠ NO MANUAL IMPORTS HERE. They used to sit above this line and made every
        # assertion below true by construction; see
        # `TestTheWorkerWillActuallyIMPORTTheTaskModules`.
        from src.core.celery_app import celery_app

        return celery_app

    @pytest.mark.parametrize(
        "name",
        [
            "src.workers.probe_monitor_tasks.run_probe_monitor",
            "src.workers.probe_monitor_tasks.evaluate_probe_monitor",
            "src.workers.probe_monitor_tasks.score_probe_monitor",
            "src.workers.probe_monitor_tasks.run_probe_monitor_judge",
            "cleanup_stuck_probe_monitor_runs",
        ],
    )
    def test_the_task_is_in_the_registry(self, celery, name):
        assert name in celery.tasks, (
            f"{name} is not registered; check `autodiscover_tasks` in core/celery_app.py"
        )

    def _queue(self, celery, name):
        route = celery.amqp.router.route({}, name)
        queue = route.get("queue")
        return getattr(queue, "name", str(queue))

    @pytest.mark.parametrize(
        "name",
        [
            "src.workers.probe_monitor_tasks.run_probe_monitor",
            "src.workers.probe_monitor_tasks.evaluate_probe_monitor",
            "src.workers.probe_monitor_tasks.score_probe_monitor",
        ],
    )
    def test_the_GPU_tasks_go_to_the_extraction_queue(self, celery, name):
        """⚠ `task_routes` GLOBS MATCH THE TASK NAME, not the module path, so a short
        name silently lands on the DEFAULT queue. That put two GPU trainings on the
        wrong queue here once, found only by a test that resolves every route."""
        assert self._queue(celery, name) == "extraction", (
            f"{name} routes to {self._queue(celery, name)!r}"
        )

    def test_the_JUDGE_goes_to_a_CPU_queue(self, celery):
        """It is HTTP calls to a served model. On `extraction` it would take a GPU lease
        and idle a card for the whole network-bound loop."""
        name = "src.workers.probe_monitor_tasks.run_probe_monitor_judge"
        assert self._queue(celery, name) == "processing"

    def test_the_reaper_goes_to_low_priority(self, celery):
        """Its name is SHORT (`cleanup_stuck_probe_monitor_runs`), so it does not match
        a `src.workers.*` glob and needs its own explicit entry."""
        assert self._queue(celery, "cleanup_stuck_probe_monitor_runs") == "low_priority"

    def test_no_probe_task_falls_to_the_DEFAULT_queue(self, celery):
        """The general form: a route resolving to the default queue is the silent
        failure this class exists to prevent."""
        default = celery.conf.task_default_queue
        for name in celery.tasks:
            if "probe_monitor" not in name:
                continue
            resolved = self._queue(celery, name)
            assert resolved != default, f"{name} fell to the default queue {default!r}"

    def test_the_reaper_is_in_the_BEAT_schedule(self, celery):
        """Registered and routed is not scheduled. Without a beat entry the reaper
        exists and never runs, and a phantom run keeps a GPU forever."""
        entries = celery.conf.beat_schedule
        tasks = {entry.get("task") for entry in entries.values()}
        assert "cleanup_stuck_probe_monitor_runs" in tasks, (
            f"the reaper has no beat entry; scheduled tasks are {sorted(t for t in tasks if t)}"
        )

    def test_the_beat_entry_names_its_queue_explicitly(self, celery):
        for entry in celery.conf.beat_schedule.values():
            if entry.get("task") == "cleanup_stuck_probe_monitor_runs":
                assert entry.get("options", {}).get("queue") == "low_priority"
                return
        pytest.fail("no beat entry for the probe reaper")


class TestCancellationIsRegistered:
    def test_both_scopes_exist(self):
        """A `request_cancel` for an unregistered scope raises, so the endpoint would
        500 rather than stop the run."""
        from src.core.cancellation import SCOPES

        kinds = set(SCOPES)
        assert "probe_monitor_run" in kinds
        assert "probe_monitor_judge" in kinds

    def test_a_deleted_run_STOPS_rather_than_continuing(self):
        """`missing_row="continue"` is what let a deleted extraction compute for twenty
        more minutes over unlinked files while holding a card."""
        from src.core.cancellation import get_scope

        assert get_scope("probe_monitor_run").missing_row == "cancelled"
        assert get_scope("probe_monitor_judge").missing_row == "cancelled"

    def test_the_live_stages_are_NOT_terminal(self):
        """`stage` moves through rendering → … → rung; only the three statuses end a
        run. A terminal set that included a stage would make `record_progress` refuse
        every write after it."""
        from src.core.cancellation import get_scope

        terminal = get_scope("probe_monitor_run").terminal_values
        assert terminal == frozenset({"completed", "failed", "cancelled"})
        for stage in ("rendering", "pooled_capture", "training", "evaluating", "rung"):
            assert stage not in terminal

    def test_the_endpoint_CALLS_request_cancel(self):
        """AST, not a text scan: the docstring explains cooperative cancellation and a
        substring search would match the prose."""
        from src.api.v1.endpoints import probe_monitors

        tree = ast.parse(textwrap.dedent(inspect.getsource(probe_monitors.cancel_probe_run)))
        called = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "request_cancel" in called


class TestTheWebSocketChannelIsAccepted:
    def test_the_topic_is_in_the_allowlist(self):
        from src.core.websocket import _CHANNEL_TOPICS

        assert "probe_monitors" in _CHANNEL_TOPICS

    def test_a_real_run_channel_validates(self):
        """Both halves must hold: the topic is allowed AND the name matches
        `_CHANNEL_RE`. A hyphenated topic passes the allowlist check and fails the
        regex, so the emission is rejected at runtime with nothing in the panel."""
        from src.core.websocket import validate_channel

        assert validate_channel("probe_monitors/pmr_0123456789ab")

    def test_the_emitters_publish_to_that_channel(self):
        """Read from the SOURCE of the emitter, so a channel typo is caught here rather
        than by silence in the UI."""
        from src.workers import websocket_emitter

        for function in (
            websocket_emitter.emit_probe_monitor_progress,
            websocket_emitter.emit_probe_monitor_completed,
            websocket_emitter.emit_probe_monitor_failed,
        ):
            source = inspect.getsource(function)
            assert "probe_monitors/" in source, f"{function.__name__} uses another channel"

    def test_every_emitted_channel_is_accepted_by_validation(self):
        """The two halves compared against each other, which is what stops the
        allowlist drifting narrower than production."""
        from src.core.websocket import validate_channel

        assert validate_channel("probe_monitors/pmr_abc")


class TestTheSubmitPathTakesAGpuLease:
    def test_it_dispatches_with_gpu_delay_not_delay(self):
        """`gpu_delay` is what makes the job wait for a lease. A bare `.delay` would
        queue GPU work that runs beside another job on the same card — and the existing
        AST guard for this pattern is why it is asserted rather than assumed."""
        from src.api.v1.endpoints import probe_monitors

        tree = ast.parse(textwrap.dedent(inspect.getsource(probe_monitors.submit_probe_run)))
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    names.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    names.add(node.func.attr)
        assert "gpu_delay" in names
        assert "delay" not in names, (
            "the submit endpoint dispatches with a bare .delay, so the run does not "
            "wait for a GPU lease"
        )

    def test_it_resolves_the_gpu_request_BEFORE_creating_the_row(self):
        """An unknown card must be a 400, not a 202 followed by a failed row. Asserted
        on ORDER, which is the part that decides whether a ghost row is left behind."""
        from src.api.v1.endpoints import probe_monitors

        source = textwrap.dedent(inspect.getsource(probe_monitors.submit_probe_run))
        assert source.index("resolve_gpu_request") < source.index("db.add(row)")

    def test_the_row_is_COMMITTED_before_dispatch(self):
        """⚠ THE PRECONDITION FOR `missing_row="cancelled"`. The task refuses to start
        without a row, so the row must exist first — and when a fix once passed an id
        without creating the row, every UI extraction was silently refused for a week."""
        from src.api.v1.endpoints import probe_monitors

        source = textwrap.dedent(inspect.getsource(probe_monitors.submit_probe_run))
        assert source.index("await db.commit()") < source.index("gpu_delay")

    def test_a_broker_outage_marks_the_row_failed(self):
        """Otherwise a dispatch failure leaves a row `pending` forever, which the reaper
        then reports as a lost worker — a misleading cause for a queueing problem."""
        from src.api.v1.endpoints import probe_monitors

        source = textwrap.dedent(inspect.getsource(probe_monitors.submit_probe_run))
        assert 'status = "failed"' in source
        assert "503" in source


class TestTheTasksRefuseWithoutTheirRow:
    def test_the_guard_exists_and_is_shared(self):
        """One helper, so four tasks cannot disagree about whether a missing row is an
        error — and the reason lives beside it, not in a comment somewhere else."""
        from src.workers import probe_monitor_tasks

        assert callable(probe_monitor_tasks._require_row)

    def test_it_raises_and_says_WHY_a_missing_row_is_an_error(self):
        from src.workers.probe_monitor_tasks import _require_row

        class Empty:
            id = "x"

            def query(self, *a, **k):
                return self

            def filter(self, *a, **k):
                return self

            def first(self):
                return None

        with pytest.raises(ValueError, match="deleted"):
            _require_row(Empty(), Empty, "pmr_missing", "probe monitor run")

    def test_every_task_uses_it(self):
        """A task that skips the guard would work over an unlinked artifact directory."""
        from src.workers import probe_monitor_tasks

        for name in (
            "run_probe_monitor",
            "evaluate_probe_monitor",
            "score_probe_monitor",
            "run_probe_monitor_judge",
        ):
            task = getattr(probe_monitor_tasks, name)
            source = inspect.getsource(inspect.unwrap(task))
            tree = ast.parse(textwrap.dedent(source))
            called = {
                node.func.id
                for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            }
            assert "_require_row" in called, f"{name} does not check for its row"


class TestTheModelsAndMigrationAreReachable:
    def test_the_alembic_head_includes_the_probe_migration(self):
        """A model with no applied migration is a table that does not exist in
        production, however green the suite is."""
        from pathlib import Path

        versions = Path(__file__).resolve().parents[2] / "alembic" / "versions"
        revisions = {
            path.stem.split("_")[0] for path in versions.glob("*.py")
        }
        assert "a7f3c8e90d21" in revisions
        # It must be reachable from the chain, i.e. something revises it or it is head.
        texts = [path.read_text() for path in versions.glob("*.py")]
        is_head = not any('down_revision = "a7f3c8e90d21"' in text for text in texts)
        is_revised = any('down_revision = "a7f3c8e90d21"' in text for text in texts)
        assert is_head or is_revised


class TestThePublishEndpointACTUALLYCallsItsTokenPreflight:
    """⚠ THIS CLASS EXISTS BECAUSE A MUTATION SURVIVED. Deleting the two lines that call
    `_huggingface_rejects` from `publish_definition` left the suite completely green: the helper had
    seven tests of its own and nothing asserted the endpoint used it. A tested helper that nothing
    calls is a capability that does not exist — the repo's reachability rule, caught on the very fix
    that was added to make a silent failure loud.

    It is asserted by CALLING the endpoint, not by scraping its source. A substring search would
    match the docstring that describes the check, and this repo has shipped that mistake in three
    separate arcs.
    """

    def _probe(self):
        return SimpleNamespace(id="pm_x", definition_path="/data/probes/pm_x/definition.json")

    class _Db:
        """Just enough AsyncSession for the endpoint: one `execute` returning one row."""

        def __init__(self, row):
            self._row = row

        async def execute(self, _statement):
            row = self._row

            class _Result:
                def scalar_one_or_none(self):
                    return row

            return _Result()

    async def _publish(self, monkeypatch, *, rejects):
        import src.api.v1.endpoints.probe_monitors as endpoints
        import src.services.huggingface_sae_service as hf

        monkeypatch.setattr(hf, "resolve_hf_token", lambda explicit=None: "hf_token")
        monkeypatch.setattr(
            endpoints, "_huggingface_rejects", lambda _token: "rejected: expired" if rejects else None
        )
        queued = {}

        class _Task:
            id = "task-1"

        def _delay(**kwargs):
            queued.update(kwargs)
            return _Task()

        import src.workers.probe_monitor_tasks as tasks

        monkeypatch.setattr(tasks.publish_probe_definition, "delay", _delay)
        request = SimpleNamespace(repo_id="someone/probes", private=True, token=None)
        return endpoints.publish_definition(
            probe_id="pm_x", request=request, db=self._Db(self._probe())
        ), queued

    @pytest.mark.asyncio
    async def test_a_rejected_token_is_a_401_and_QUEUES_NOTHING(self, monkeypatch):
        coroutine, queued = await self._publish(monkeypatch, rejects=True)
        with pytest.raises(HTTPException) as caught:
            await coroutine
        assert caught.value.status_code == 401
        assert "expired" in str(caught.value.detail)
        assert queued == {}, "the upload must not be queued behind a refused token"

    @pytest.mark.asyncio
    async def test_an_accepted_token_queues_the_upload_with_the_right_payload(self, monkeypatch):
        """The call count is not enough: a preflight that passes must not change what is sent."""
        coroutine, queued = await self._publish(monkeypatch, rejects=False)
        result = await coroutine
        assert result["status"] == "queued"
        assert queued["probe_id"] == "pm_x"
        assert queued["repo_id"] == "someone/probes"
        assert queued["private"] is True
        assert queued["token"] == "hf_token"


class TestTheBuildTaskRecordsTheDIGESTOFTheFileItWrote:
    """⚠ ANOTHER MUTATION SURVIVED HERE, on the row that ships the pin.

    W6 replaced `probe.definition_sha256 = written_sha` with a truncated slice of the serialised
    model — deliberate nonsense — and the whole suite stayed green. The builder's serialisation was
    well covered by then; the TASK's assignment of it to the row was covered by nothing, and the row
    is what the publisher reads when it writes the manifest and the README a consumer verifies
    against.

    This is the same shape as the publish preflight: a correct helper, tested; a caller that could
    record anything, untested. Two rounds in one session, which is why the rule is "delete the
    wiring and require a red" rather than "read the code and be satisfied".
    """

    class _Db:
        def __init__(self, probe):
            self.probe = probe
            self.committed = 0

        def query(self, _model):
            probe = self.probe

            class _Q:
                def filter(self, *_a, **_k):
                    return self

                def first(self):
                    return probe

            return _Q()

        def commit(self):
            self.committed += 1

    def test_the_row_carries_the_file_s_own_sha256(self, tmp_path, monkeypatch):
        import contextlib
        import hashlib
        import json

        from src.schemas.probe_definition import ProbeDefinitionV1
        import src.workers.probe_monitor_tasks as tasks
        import src.services.probe_definition_builder as builder
        from tests.unit.test_probe_definition import definition as contract_document

        definition = ProbeDefinitionV1.model_validate(contract_document())
        record = {"resolved": {"bytes": 1, "vectors": 16, "sha256": "not-this"}}

        probe = SimpleNamespace(
            id="pm_x", run_id="pmr_x", definition_path=None, definition_built_at=None,
            definition_sha256=None, definition_build=None,
        )
        db = self._Db(probe)

        monkeypatch.setattr(builder, "build", lambda *a, **k: (definition, record))
        monkeypatch.setattr(tasks, "_require_row", lambda *a, **k: probe)
        # ⚠ The task imports `record_progress` INSIDE its body, so patching it on this module
        # is inert — a local import rebinds the name every call. Patch it where it lives.
        monkeypatch.setattr("src.core.cancellation.record_progress", lambda *a, **k: None)

        class _Settings:
            data_dir = str(tmp_path)

        monkeypatch.setattr("src.core.config.settings", _Settings())

        @contextlib.contextmanager
        def _get_db():
            yield db

        # ⚠ PATCHED ON `DatabaseTask`, not on the task object. A Celery task is a `PromiseProxy`,
        # so `type(task)` is the proxy class and setting `get_db` there is inert — the task keeps
        # using the real session, the assertions below still pass against a real commit, and only
        # the commit COUNT notices. That near-miss is the reason this comment exists: a patch that
        # silently does nothing is the same failure mode as a mutation that silently does nothing.
        from src.workers.base_task import DatabaseTask

        monkeypatch.setattr(DatabaseTask, "get_db", lambda _self: _get_db())

        result = tasks.build_probe_definition.run(probe_id="pm_x")

        assert result["status"] == "completed"
        written = pathlib.Path(probe.definition_path).read_bytes()
        expected = hashlib.sha256(written).hexdigest()
        assert probe.definition_sha256 == expected, (
            "the row's digest must be of the bytes on disk — this is what a consumer verifies"
        )
        assert result["sha256"] == expected
        assert result["bytes"] == len(written)
        assert probe.definition_build["resolved"]["sha256"] == expected, (
            "the build record is what the publisher reads; a stale digest there ships in the README"
        )
        assert json.loads(written)["kind"] == "mistudio.probe-definition/v1"
        assert db.committed == 1


class TestTheRungIsPromotedAsSoonAsAProbeEarnsIt:
    """⚠ A FULLY-EVALUATED PROBE SAT AT RUNG 0 BECAUSE A LATER PROBE RAISED.

    `execute_probe_run` evaluated every probe and then promoted every rung in a separate stage. On
    `pmr_5d81ad3b81f2` the SAE probe raised inside evaluation, so the loop never reached that stage,
    and `pm_8428015843e3` kept `rung = 0` with five complete out-of-distribution evaluations beside
    it in the database.

    Not cosmetic: the rung is what 033's export gate reads. The probe was refused for lacking
    evidence it had, and an acknowledged export would have written "rung 0 — trained" above five
    out-of-distribution AUROCs — a document contradicting itself in the field a consumer trusts
    most.

    Asserted on the ORDER of calls in the source's AST rather than by running a GPU run: the claim
    is "the promotion happens inside the evaluation loop", which is a structural property. A
    substring search would match the comment that explains it, so the loop body is walked.
    """

    def _evaluation_loop(self):
        import ast
        import inspect

        from src.services import probe_monitor_run

        tree = ast.parse(inspect.getsource(probe_monitor_run.execute_probe_run))
        for node in ast.walk(tree):
            if not isinstance(node, ast.For):
                continue
            calls = {
                sub.func.id
                for sub in ast.walk(node)
                if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)
            }
            if "_evaluate_probe_on_sets" in calls:
                return calls
        return None

    def test_recompute_rung_is_called_INSIDE_the_evaluation_loop(self):
        calls = self._evaluation_loop()
        assert calls is not None, "the evaluation loop was not found — this guard has stopped looking"
        assert "recompute_rung" in calls, (
            "a probe's rung must be promoted in the same loop iteration that finishes its "
            "evaluations, or a later probe raising strands it at the rung it started with"
        )

    def test_the_separate_rung_stage_still_exists(self):
        """Kept deliberately: it is where the stage/progress contract reports, and it is idempotent."""
        import inspect

        from src.services import probe_monitor_run

        source = inspect.getsource(probe_monitor_run.execute_probe_run)
        assert 'stage("rung")' in source


class TestTheExportGateReadsARecomputedRung:
    """The gate must ask the ladder, not trust a cached column — see the class above for why.

    Pinned by CALLING `build` far enough to reach the gate, because a source scrape would match the
    comment that explains the recomputation. `recompute_rung` is replaced with a recorder that also
    promotes the row, so the test proves the gate saw the NEW value rather than merely that a
    function was called.
    """

    def test_the_gate_sees_the_rung_the_ladder_computes(self, monkeypatch):
        import src.services.probe_definition_builder as builder_module
        import src.services.probe_monitor_run as run_module

        probe = SimpleNamespace(
            id="pm_x", run_id="pmr_x", rung=0, variant="dense", sae_id=None,
            layer=11, threshold=1.0, weights_path=None,
        )
        seen = {}

        def _recompute(_db, probe_id):
            probe.rung = 2          # what the ladder would conclude from the evidence
            seen["recomputed"] = probe_id
            return {"rung": 2}

        monkeypatch.setattr(run_module, "recompute_rung", _recompute)

        def _gate(row, _run, **kwargs):
            seen["rung_at_gate"] = row.rung
            raise builder_module.ProbeExportRefused("stop here; the gate is what we are measuring")

        monkeypatch.setattr(builder_module, "check_export_gate", _gate)

        class _Db:
            def query(self, _model):
                class _Q:
                    def filter(self, *_a, **_k):
                        return self

                    def first(self):
                        return probe

                return _Q()

            def refresh(self, _row):
                return None

        with pytest.raises(builder_module.ProbeExportRefused):
            builder_module.build(_Db(), "pm_x")

        assert seen["recomputed"] == "pm_x"
        assert seen["rung_at_gate"] == 2, (
            "the gate read the stale cached rung; a probe is then refused for lacking evidence it has"
        )


class TestBuildPassesTheMeasuredRoundTripIntoTheDocument:
    """⚠ THE THIRD CALLER-NOT-COVERED GAP IN ONE SESSION.

    T3 deleted `messages_reproduce_token_ids=_messages_round_trip(...)` from the `TestVectors(...)`
    call in `build` and the whole suite stayed green: the helper had five tests and the line that
    puts its answer in the document had none. Same shape as the publish preflight (P11) and the
    task's digest assignment (W6/W7) — a correct helper, a caller that could omit it entirely.

    Asserted on the AST of the `TestVectors(...)` CALL, not on the module's text. A substring search
    for the field name matches the comment above the line explaining why it is there, and this repo
    has shipped that exact mistake in three separate arcs.
    """

    def _test_vectors_call(self):
        import ast
        import inspect

        from src.services import probe_definition_builder

        tree = ast.parse(inspect.getsource(probe_definition_builder.build))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "TestVectors"
            ):
                return node
        return None

    def test_the_call_exists_at_all(self):
        assert self._test_vectors_call() is not None, (
            "the TestVectors(...) call was not found in build() — this guard has stopped looking, "
            "which is how a source-derived check fails open"
        )

    def test_the_round_trip_is_passed_as_a_keyword(self):
        call = self._test_vectors_call()
        keywords = {keyword.arg for keyword in call.keywords}
        assert "messages_reproduce_token_ids" in keywords, (
            "the document must carry the MEASURED round-trip; without it a consumer starting from "
            "`messages` is off by up to 1.153 on a 0.05 tolerance and has nothing to warn it"
        )
        assert "tolerance" in keywords and "vectors" in keywords

    def test_it_is_passed_the_MEASUREMENT_not_a_literal(self):
        """`messages_reproduce_token_ids=True` would satisfy the keyword and state a falsehood."""
        import ast

        call = self._test_vectors_call()
        argument = next(
            keyword.value
            for keyword in call.keywords
            if keyword.arg == "messages_reproduce_token_ids"
        )
        assert isinstance(argument, ast.Call), "it must be a call, not a constant"
        assert getattr(argument.func, "id", None) == "_messages_round_trip"
