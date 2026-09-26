"""The WHOLE probe run, end to end, on CPU: render → capture → select → train → evaluate → rung.

MUTATION CONTROLS (each verified to fail the suite; recorded in the review):
  M136  a stage is skipped in `execute_probe_run`     → the stage-order test fails
  M137  the rung is never recomputed                  → the rung test fails
  M138  `selected` is never set                       → the selected-probe test fails
  M139  weights are not written to disk               → the artifact test fails
  M140  the evaluation row is omitted on a refusal    → the refusal-row test fails
  M141  `environment` loses its per-stage timings      → the FR-15 test fails
  M142  the cancel guard is removed from `stage()`     → the cancellation test fails

⚠ WHY THIS EXISTS AND WHY IT IS IN `tests/unit`. Feature 030 shipped a detection path
with THREE independent breaks behind green unit tests: every test called the formatter
directly, and the fixture that exercised the real dispatch was defined and never used.
Unit tests of each stage cannot catch a pipeline that does not connect. This test drives
the real `execute_probe_run` over real rows, a real Arrow dataset on disk and a real
(tiny, random) causal LM.

`tests/integration` would have been the conventional home — and CI passes
`--ignore=tests/integration`, so the one test that proves the pipeline connects would
never run in CI. It lives here instead, with its OWN database so it cannot race the
shared fixtures' drop/create cycle.
"""
import os
import uuid

import numpy as np
import pytest
import torch

pytest.importorskip("datasets")

D_MODEL = 32
N_ROWS = 80


# ── an isolated database, because this test needs a SYNC session ──────────────


@pytest.fixture(scope="module")
def pipeline_db():
    """A dedicated database with every table, and a sync session over it.

    Its own database rather than the shared test one: `execute_probe_run` takes a SYNC
    session while the shared fixtures manage the schema through an async engine, and
    creating tables under both is a race that shows up as flakes in other files.
    """
    import subprocess
    import sys
    from pathlib import Path

    import psycopg2
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    import src.models  # noqa: F401 - registers every table

    # ⚠ DERIVED FROM THE ENVIRONMENT, NEVER HARDCODED. The first version of this fixture
    # wrote `user="postgres", password="devpassword"` — the workstation's credentials. It
    # passed locally and ERRORED 18 TIMES IN CI, where the database is `mistudio` /
    # `testpassword`. A test that only runs on one machine is a test nobody else is
    # protected by, and a credential in a test file is its own problem.
    from urllib.parse import urlparse

    source = os.environ.get("DATABASE_URL_SYNC") or os.environ.get("DATABASE_URL") or ""
    parsed = urlparse(source.replace("+psycopg2", "").replace("+asyncpg", ""))
    if not parsed.hostname or not parsed.username:
        raise AssertionError(
            "DATABASE_URL_SYNC is unset or unparseable, so this test cannot build its "
            f"scratch database (got {source!r}). It is not skipped: skipping would hide "
            "the one test that proves the probe pipeline connects end to end."
        )
    host = parsed.hostname
    port = parsed.port or 5432
    user = parsed.username
    password = parsed.password or ""

    worker = os.getenv("PYTEST_XDIST_WORKER", "solo")
    name = f"mistudio_probe_pipeline_{worker}"
    admin = psycopg2.connect(
        host=host, port=port, user=user, password=password, dbname="postgres"
    )
    admin.autocommit = True
    with admin.cursor() as cursor:
        cursor.execute(f'DROP DATABASE IF EXISTS "{name}"')
        cursor.execute(f'CREATE DATABASE "{name}"')
    admin.close()

    url = f"postgresql://{user}:{password}@{host}:{port}/{name}"
    # ⚠ THE MIGRATIONS, NOT `Base.metadata.create_all`. Several tables declare native
    # Postgres enums with `create_type=False` because their migration creates the type,
    # so `create_all` on a fresh database fails with `type "export_status" does not
    # exist`. Running alembic also means this test exercises the SCHEMA THAT SHIPS,
    # including the probe migration itself.
    backend = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=backend,
        env={**os.environ, "DATABASE_URL_SYNC": url,
             "DATABASE_URL": url.replace("postgresql://", "postgresql+asyncpg://")},
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise AssertionError(
            "alembic could not build the pipeline test's database:\n"
            + completed.stdout[-2000:] + completed.stderr[-2000:]
        )
    engine = create_engine(url)
    # ⚠ THE SESSION MUST BE CONFIGURED LIKE PRODUCTION'S, AND IT WAS NOT.
    #
    # `SyncSessionLocal` sets `autoflush=False` ("Manual flush control"), and that is not
    # a detail: with autoflush ON, any query flushes pending changes first, so an
    # assignment survives a `.populate_existing()` read on the same session. With it OFF
    # — production — the read refreshes every attribute from the database and REVERTS the
    # assignment.
    #
    # That difference hid a real defect and then nearly hid its diagnosis. The Stage 1
    # acceptance run completed with `artifact_dir` and `gpu_uuid` both NULL while its
    # token capture held 7.7 GB on disk; the control written to prove the mechanism
    # PASSED against the defect here, because this fixture autoflushed and production
    # does not. The flags are read from `SyncSessionLocal` itself so the two cannot drift
    # again.
    from src.core.database import SyncSessionLocal as _production_sessionmaker

    _kw = _production_sessionmaker.kw
    Session = sessionmaker(
        bind=engine,
        autoflush=_kw.get("autoflush", True),
        expire_on_commit=_kw.get("expire_on_commit", True),
    )
    session = Session()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()
        admin = psycopg2.connect(
            host=host, port=port, user=user, password=password, dbname="postgres"
        )
        admin.autocommit = True
        with admin.cursor() as cursor:
            cursor.execute(f'DROP DATABASE IF EXISTS "{name}"')
        admin.close()


@pytest.fixture(scope="module")
def tiny_lm():
    from transformers import AutoModelForCausalLM, LlamaConfig

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=D_MODEL,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=256,
    )
    model = AutoModelForCausalLM.from_config(config)
    model.eval()
    return model


@pytest.fixture(scope="module")
def tiny_tokenizer():
    """A real fast tokenizer with a real chat template, over a tiny vocabulary."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    words = ["<unk>", "<turn>", "</turn>", "user", "assistant"]
    # The two class-marker words plus filler. The LABEL IS ENCODED IN TOKEN IDENTITY,
    # which is what makes a 4-layer random model able to separate the classes at all:
    # a random transformer has no semantics, but it does map different tokens to
    # different residuals, so a linear probe can find them.
    words += [f"risk{i}" for i in range(8)]
    words += [f"calm{i}" for i in range(8)]
    words += [f"filler{i}" for i in range(16)]
    vocab = {word: index for index, word in enumerate(words)}
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>")
    tokenizer.chat_template = (
        "{% for m in messages %}<turn> {{ m['role'] }} {{ m['content'] }} </turn> "
        "{% endfor %}"
    )
    tokenizer.pad_token = "<unk>"
    return tokenizer


@pytest.fixture(scope="module")
def arrow_dataset(tmp_path_factory):
    """A real Arrow dataset on disk: `inputs`, `stakes`, `pair`.

    Written with `save_to_disk` so `resolve_dataset_path` and `load_columns` run their
    real code — reading a list in memory would skip the two functions most likely to
    disagree with what a download actually produces.
    """
    from datasets import Dataset as HFDataset

    rng = np.random.default_rng(7)
    inputs, labels, pairs = [], [], []
    for i in range(N_ROWS):
        positive = i % 2 == 0
        marker = "risk" if positive else "calm"
        words = [f"{marker}{rng.integers(0, 8)}" for _ in range(3)]
        words += [f"filler{rng.integers(0, 16)}" for _ in range(2)]
        inputs.append(" ".join(words))
        labels.append("high" if positive else "low")
        pairs.append(f"pair{i // 2}")
    path = tmp_path_factory.mktemp("probe_corpus") / "arrow"
    HFDataset.from_dict({"inputs": inputs, "stakes": labels, "pair": pairs}).save_to_disk(
        str(path)
    )
    return path


@pytest.fixture(scope="module")
def prepared(pipeline_db, arrow_dataset, monkeypatch_module, tiny_lm, tiny_tokenizer):
    """Rows for a model, a dataset, a train view and an out-of-distribution eval view."""
    from src.models.dataset import Dataset, DatasetStatus
    from src.models.model import Model, ModelStatus
    from src.models.probe_monitor import ProbeMonitorDataset, ProbeMonitorRun

    model_row = Model(
        id=f"m_{uuid.uuid4().hex[:12]}",
        name="tiny-random",
        status=ModelStatus.READY,
        file_path=str(arrow_dataset),            # never loaded: the loader is injected
        architecture="LlamaForCausalLM",
        params_count=1_000,                      # NOT NULL on the real schema
    )
    dataset_row = Dataset(
        id=uuid.uuid4(),
        name="probe-corpus",
        source="local",
        status=DatasetStatus.READY,
        raw_path=str(arrow_dataset),
        extra_metadata={},
    )
    pipeline_db.add_all([model_row, dataset_row])
    pipeline_db.commit()

    mapping = {"high": "positive", "low": "negative"}
    train_view = ProbeMonitorDataset(
        name="train", dataset_id=dataset_row.id, input_column="inputs",
        label_column="stakes", label_mapping=mapping, pair_column="pair",
        role="train", counts={},
    )
    eval_view = ProbeMonitorDataset(
        name="unseen", dataset_id=dataset_row.id, input_column="inputs",
        label_column="stakes", label_mapping=mapping,
        role="eval", distribution="out_of_distribution", counts={},
    )
    pipeline_db.add_all([train_view, eval_view])
    pipeline_db.commit()

    run = ProbeMonitorRun(
        model_id=model_row.id,
        train_dataset_id=train_view.id,
        eval_dataset_ids=[eval_view.id],
        config={
            "layers": [1, 2],
            "rules": ["mean", "max"],
            "scope": "all",
            "max_length": 64,
            "val_fraction": 0.25,
            "seed": 1337,
            "top_n_layers": 1,
            "target_fpr": 0.1,
        },
        status="pending",
        environment={},
    )
    pipeline_db.add(run)
    pipeline_db.commit()
    return {"run": run, "train_view": train_view, "eval_view": eval_view, "model": model_row}


@pytest.fixture(scope="module")
def monkeypatch_module():
    from _pytest.monkeypatch import MonkeyPatch

    patcher = MonkeyPatch()
    yield patcher
    patcher.undo()


#: What `place_job(...)` would write. A recognisable value so the test asserts the
#: loader's own placement reached the database, not merely that the column is non-null.
FAKE_GPU_UUID = "GPU-00000000-1111-2222-3333-444444444444"


def _loader_that_records_its_placement(model, tokenizer):
    def loader(row):
        row.gpu_request = "auto"
        row.gpu_uuid = FAKE_GPU_UUID
        return model, tokenizer, "LlamaForCausalLM"

    return loader


@pytest.fixture(scope="module")
def completed_run(pipeline_db, prepared, tiny_lm, tiny_tokenizer, tmp_path_factory,
                  monkeypatch_module):
    """Run the REAL `execute_probe_run` once, and let every test read its result."""
    from src.core import config as config_module
    from src.services import probe_monitor_run as run_module

    # Artifacts under a tmp path: `artifact_dir_for` derives from `settings.data_dir`,
    # and pointing that at the real data volume from a test would write into /data.
    artifacts = tmp_path_factory.mktemp("probe_artifacts")
    monkeypatch_module.setattr(config_module.settings, "data_dir", artifacts, raising=False)

    # ⚠ FORCE MORE THAN ONE BATCH PER CAPTURE, OR THE PROGRESS ASSERTIONS ARE VACUOUS.
    # With the default 16,384-token budget these tiny fixtures fit in ONE batch, so every
    # heartbeat fires exactly once — at `done == total`, reporting the top of its band. The
    # recorded sequence was `evaluating 97.0, 98.5, 100.0`: one value per probe, trivially
    # sorted, and a stage that restarted its band for every probe would read
    # `97.0, 100.0, 100.0` — also sorted. A one-batch fixture cannot see a rewind, which is
    # the whole thing `test_progress_is_monotone` exists to catch, and the live run showed
    # 99.2% → 100.0% → 97.7% as a second probe began.
    #
    # ⚠ PATCHING `DEFAULT_TOKEN_BUDGET` DOES NOT WORK, and that was the first attempt here.
    # It is a DEFAULT ARGUMENT — `token_budget: int = DEFAULT_TOKEN_BUDGET` — so it is bound
    # when the function is DEFINED, at import, and rebinding the module attribute afterwards
    # changes nothing. Two mutation controls survived against that inert patch. The call is
    # patched instead, which is resolved from the module global every time.
    from src.services import probe_monitor_capture as capture_module

    _real_plan_batches = capture_module.plan_batches
    monkeypatch_module.setattr(
        capture_module,
        "plan_batches",
        lambda examples, *, token_budget=None: _real_plan_batches(examples, token_budget=8),
    )

    stages_seen = []
    #: What the row held, as COMMITTED, at each stage boundary. Read through a separate
    #: session so it reflects what another process would see — the run's own session
    #: would show uncommitted state and make a write-once-at-the-end bug invisible.
    committed = {}

    def on_stage(stage, progress, message=None):
        stages_seen.append((stage, progress))
        if stage in committed:
            return
        from sqlalchemy.orm import sessionmaker

        from src.models.probe_monitor import ProbeMonitorRun

        # Bound to the SAME engine as `pipeline_db` — that engine owns the scratch
        # database this fixture built with alembic — but a DIFFERENT session, hence a
        # different connection. `SyncSessionLocal` would point at the shared test
        # database, where these tables do not exist.
        other = sessionmaker(bind=pipeline_db.get_bind())()
        try:
            row = (
                other.query(ProbeMonitorRun)
                .filter(ProbeMonitorRun.id == prepared["run"].id)
                .first()
            )
            committed[stage] = dict((row.environment or {}) if row else {})
        finally:
            other.close()

    result = run_module.execute_probe_run(
        pipeline_db,
        prepared["run"].id,
        on_stage=on_stage,
        # INJECTED so the pipeline runs on CPU. The production loader requires CUDA on
        # purpose — a probe trained on CPU activations of a quantized model is not the
        # probe that will be served — so the test supplies the model rather than
        # relaxing that rule.
        # The injected loader also sets the placement columns, because the production
        # loader does (`place_job(...).gpu_columns()`) and they are lost by the same
        # mechanism as `artifact_dir`. A loader that set nothing would leave `gpu_uuid`
        # untested and half the defect unguarded.
        model_loader=_loader_that_records_its_placement(tiny_lm, tiny_tokenizer),
    )
    return {
        "result": result,
        "stages": stages_seen,
        "artifacts": artifacts,
        "committed": committed,
    }


class TestThePipelineConnects:
    def test_it_completes(self, completed_run, pipeline_db, prepared):
        from src.models.probe_monitor import ProbeMonitorRun

        row = (
            pipeline_db.query(ProbeMonitorRun)
            .filter(ProbeMonitorRun.id == prepared["run"].id)
            .first()
        )
        assert row.status == "completed"
        assert row.progress == 100.0
        assert row.completed_at is not None

    def test_every_stage_ran_IN_ORDER(self, completed_run):
        """A pipeline that skips a stage still 'completes'. The order is the contract.

        ⚠ CONSECUTIVE REPEATS ARE DELIBERATE AND ARE NOT TRANSITIONS. A long stage now
        emits within-stage heartbeats through the same callback — `pooled_capture` took
        over forty minutes on the first Stage 1 acceptance run, and a stage that reports
        only at its boundaries is indistinguishable from a dead worker. So the
        TRANSITIONS are compared here, and `test_the_long_stages_emitted_heartbeats`
        asserts the repeats exist rather than tolerating them silently.

        The de-duplication is of CONSECUTIVE repeats only, so a stage genuinely running
        twice, or out of order, is still caught: `[..., "training", "calibrating",
        "training"]` keeps both `training` entries and fails.
        """
        from src.services.probe_monitor_run import STAGES

        seen = [stage for stage, _ in completed_run["stages"]]
        transitions = [
            stage for index, stage in enumerate(seen)
            if index == 0 or stage != seen[index - 1]
        ]
        assert transitions == [name for name, _ in STAGES], f"stages ran as {seen}"

    def test_the_long_stages_emitted_heartbeats(self, completed_run):
        """The other half: the de-duplication above must not be able to hide a stage
        that stopped reporting. A stage wired with `progress=heartbeat(...)` emits at
        least one extra tick, because the final batch is never throttled."""
        seen = [stage for stage, _ in completed_run["stages"]]
        counts = {name: seen.count(name) for name in set(seen)}
        for stage in ("pooled_capture", "token_capture", "evaluating"):
            assert counts.get(stage, 0) >= 2, (
                f"{stage} was reported {counts.get(stage, 0)} time(s); it is wired with "
                f"a heartbeat, so a single report means the callback is not reaching the "
                f"capture loop"
            )

    def test_the_heartbeat_progress_stays_inside_its_stage_band(self, completed_run):
        """A heartbeat that overshot its band would report a later stage's progress and
        make the bar jump backwards at the next transition."""
        from src.services.probe_monitor_run import STAGE_PROGRESS

        from src.services.probe_monitor_run import heartbeat_band

        for stage, progress in completed_run["stages"]:
            low, high = heartbeat_band(stage)
            assert low <= progress <= high, (
                f"{stage} reported {progress}, outside its band {(low, high)}"
            )
            assert low == STAGE_PROGRESS[stage], (
                f"{stage} heartbeats from {low} while it is ENTERED at "
                f"{STAGE_PROGRESS[stage]}, so the bar moves backwards on the first tick"
            )

    def test_the_environment_is_persisted_AS_THE_RUN_GOES(self, completed_run):
        """⚠ IT USED TO BE WRITTEN ONCE, ON THE SUCCESS PATH.

        A run that failed at `token_capture` therefore stored nothing about how it had
        been configured — no swept layers, no template hash, no seed, no stage timings.
        FR-15 requires a run to be reproducible from what it stored, and a failed run is
        the one a reader most needs to reconstruct: the first two Stage 1 acceptance
        failures had to be diagnosed from the worker log, because the row was empty and
        the API reported `environment: {}` throughout a 20-minute run.

        Read from a SEPARATE session at the `training` boundary, so it asserts what is
        committed rather than what the run's own session is holding.
        """
        environment = completed_run["committed"].get("training")
        assert environment, "no snapshot was taken at the training boundary"
        for key in ("layers_swept", "template_hash", "seed", "scope", "architecture"):
            assert key in environment, (
                f"{key} was not committed by the time training began, so a run failing "
                f"after this point would store nothing about its own configuration; "
                f"committed keys were {sorted(environment)}"
            )
        assert environment.get("stage_seconds", {}).get("pooled_capture") is not None, (
            "the timing of a stage that had already finished was not committed"
        )

    def test_nothing_is_committed_before_it_is_known(self, completed_run):
        """The other direction: the render summary must NOT appear at the `rendering`
        boundary, because rendering has not happened yet. A snapshot that showed it would
        mean the fixture is reading the final state, and every assertion above would pass
        against a write-once-at-the-end implementation."""
        at_rendering = completed_run["committed"].get("rendering")
        assert at_rendering is not None, "no snapshot was taken at the rendering boundary"
        assert "render" not in at_rendering, (
            "the render summary is already committed when rendering BEGINS, so the "
            "snapshots are not reading per-boundary state"
        )

    def test_the_FIRST_stage_commits_before_it_does_its_work(self, completed_run):
        """The half that `finish()` cannot cover, and that a mutation found.

        `finish()` commits after each stage, so removing the commit from `stage()` is
        invisible from the `training` boundary onward — everything has already been
        written by an earlier `finish`. The exception is the FIRST stage: nothing has
        finished yet, so if `stage()` does not commit, a failure during rendering stores
        nothing at all. Rendering is where a bad `input_column` or an empty chat template
        fails, so it is exactly the stage whose configuration a reader will want.
        """
        at_rendering = completed_run["committed"]["rendering"]
        for key in ("layers_swept", "n_layers", "template_hash", "seed", "scope"):
            assert key in at_rendering, (
                f"{key} is not committed when the FIRST stage begins, so a failure "
                f"during rendering would store nothing; committed keys were "
                f"{sorted(at_rendering)}"
            )

    def test_progress_is_monotone(self, completed_run):
        """The bar never goes backwards — across stages AND across probes within a stage.

        The fixture forces several batches per capture (see `completed_run`) so the
        heartbeats emit intermediate values; with one batch each they all reported the top
        of their band and any rewind was invisible.
        """
        values = [progress for _, progress in completed_run["stages"]]
        assert values == sorted(values), (
            "progress went backwards: "
            + ", ".join(
                f"{stage}={progress:.2f}" for stage, progress in completed_run["stages"]
            )
        )
        assert values[-1] == 100.0

    def test_the_heartbeats_really_emitted_INTERMEDIATE_values(self, completed_run):
        """The control for the test above. If every report landed on a band boundary, the
        monotonicity check is comparing a handful of identical numbers and would pass
        against a stage that restarts its band for every probe."""
        from src.services.probe_monitor_run import STAGE_PROGRESS

        boundaries = set(STAGE_PROGRESS.values())
        intermediate = [
            (stage, progress)
            for stage, progress in completed_run["stages"]
            if progress not in boundaries
        ]
        assert intermediate, (
            "no heartbeat reported a value strictly inside its band, so the monotonicity "
            "assertion cannot detect a rewind"
        )

    def test_each_probes_evaluation_gets_its_own_slice(self, completed_run):
        """Several probes evaluate in one stage. Their reports must partition the band
        rather than each starting at its low end."""
        from src.services.probe_monitor_run import heartbeat_band

        low, high = heartbeat_band("evaluating")
        reports = [p for stage, p in completed_run["stages"] if stage == "evaluating"]
        assert len(reports) >= 2, f"only {len(reports)} evaluating report(s)"
        assert reports == sorted(reports), f"the evaluating reports rewind: {reports}"
        assert low <= min(reports) and max(reports) <= high

    def test_the_layer_selection_grid_was_stored(self, pipeline_db, prepared, completed_run):
        from src.models.probe_monitor import ProbeMonitorRun

        row = (
            pipeline_db.query(ProbeMonitorRun)
            .filter(ProbeMonitorRun.id == prepared["run"].id)
            .first()
        )
        grid = (row.layer_selection or {}).get("grid")
        assert grid, "no selection grid was recorded"
        # 2 layers x 2 poolings
        assert len(grid) == 4
        assert row.layer_selection["chosen"]

    def test_a_probe_exists_per_rule(self, pipeline_db, prepared, completed_run):
        from src.models.probe_monitor import ProbeMonitor

        probes = (
            pipeline_db.query(ProbeMonitor)
            .filter(ProbeMonitor.run_id == prepared["run"].id)
            .all()
        )
        assert {p.rule for p in probes} == {"mean", "max"}
        assert all(p.layer in row_layers(pipeline_db, prepared) for p in probes)

    def test_exactly_one_probe_is_SELECTED(self, pipeline_db, prepared, completed_run):
        """The report needs one probe to lead with, and two would make the panel's
        headline number ambiguous."""
        from src.models.probe_monitor import ProbeMonitor

        selected = (
            pipeline_db.query(ProbeMonitor)
            .filter(ProbeMonitor.run_id == prepared["run"].id, ProbeMonitor.selected.is_(True))
            .all()
        )
        assert len(selected) == 1

    def test_the_weights_are_ON_DISK_and_loadable(self, pipeline_db, prepared, completed_run):
        """Tensors never go in the DB, so the row's path must resolve to a real file —
        and 033 reads exactly this file to build an exported definition."""
        from pathlib import Path

        from src.models.probe_monitor import ProbeMonitor
        from src.services.probe_monitor_run import load_probe

        probe = (
            pipeline_db.query(ProbeMonitor)
            .filter(ProbeMonitor.run_id == prepared["run"].id)
            .first()
        )
        assert probe.weights_path
        assert Path(probe.weights_path).exists()
        head, trained = load_probe(pipeline_db, probe.id)
        assert head.weight.shape == (D_MODEL,)
        assert head.layer == probe.layer

    def test_the_loaded_head_carries_the_BIAS(self, pipeline_db, prepared, completed_run):
        """It travels in the safetensors metadata; losing it silently shifts every
        score by a constant, which moves the threshold and not the AUROC — so nothing
        in the metrics would look wrong."""
        from src.models.probe_monitor import ProbeMonitor
        from src.services.probe_monitor_run import load_probe

        probe = (
            pipeline_db.query(ProbeMonitor)
            .filter(ProbeMonitor.run_id == prepared["run"].id)
            .first()
        )
        head, _ = load_probe(pipeline_db, probe.id)
        assert isinstance(head.bias, float)


class TestEvaluationAndRung:
    def test_an_evaluation_ROW_exists_for_the_eval_set(self, pipeline_db, prepared, completed_run):
        from src.models.probe_monitor import ProbeMonitor, ProbeMonitorEvaluation

        probes = (
            pipeline_db.query(ProbeMonitor)
            .filter(ProbeMonitor.run_id == prepared["run"].id)
            .all()
        )
        for probe in probes:
            evaluations = (
                pipeline_db.query(ProbeMonitorEvaluation)
                .filter(ProbeMonitorEvaluation.probe_id == probe.id)
                .all()
            )
            assert evaluations, f"probe {probe.id} has no evaluation row"

    def test_a_REFUSAL_is_still_a_row_with_its_reason(self, pipeline_db, prepared, completed_run):
        """80 rows is under the 20-per-class floor once split, so this set refuses — and
        the refusal must be a ROW. A missing row reads as 'not run yet' and a 0.5 reads
        as 'measured, and chance'."""
        from src.models.probe_monitor import ProbeMonitorEvaluation

        rows = pipeline_db.query(ProbeMonitorEvaluation).all()
        assert rows
        for row in rows:
            assert row.status in ("completed", "refused")
            if row.status == "refused":
                assert row.metrics.get("reason"), "a refusal with no reason"
                assert row.metrics.get("scored") is False
                assert row.n_positive is not None and row.n_negative is not None

    def test_the_evaluation_carries_the_SET_IDENTITY(self, pipeline_db, prepared, completed_run):
        """`evaluate()` echoes `name` and `out_of_distribution`, which is what makes the
        rung computable at all — without them every set grades as in-distribution."""
        from src.models.probe_monitor import ProbeMonitorEvaluation

        row = pipeline_db.query(ProbeMonitorEvaluation).first()
        assert row.metrics.get("name") == "unseen"
        assert row.metrics.get("out_of_distribution") is True

    def test_the_rung_was_COMPUTED_not_left_at_zero(self, pipeline_db, prepared, completed_run):
        from src.models.probe_monitor import ProbeMonitor

        probes = (
            pipeline_db.query(ProbeMonitor)
            .filter(ProbeMonitor.run_id == prepared["run"].id)
            .all()
        )
        for probe in probes:
            assert probe.rung_reasons, f"probe {probe.id} has no rung reasons"
            assert isinstance(probe.rung, int)

    def test_the_rung_reasons_never_use_a_forbidden_word(self, pipeline_db, prepared, completed_run):
        """A probe is a detector; its wording is what miLLM mirrors verbatim."""
        from src.models.probe_monitor import ProbeMonitor
        from src.schemas.evidence_ladder import PROBE_FORBIDDEN_WORDS

        for probe in (
            pipeline_db.query(ProbeMonitor)
            .filter(ProbeMonitor.run_id == prepared["run"].id)
            .all()
        ):
            text = " ".join(probe.rung_reasons).lower()
            for word in PROBE_FORBIDDEN_WORDS:
                assert word not in text, f"rung reason contains {word!r}: {text}"


class TestReproducibilityIsRecorded:
    def test_the_environment_carries_what_FR15_requires(self, pipeline_db, prepared, completed_run):
        from src.models.probe_monitor import ProbeMonitorRun

        row = (
            pipeline_db.query(ProbeMonitorRun)
            .filter(ProbeMonitorRun.id == prepared["run"].id)
            .first()
        )
        environment = row.environment or {}
        for key in ("n_layers", "layers_swept", "template_hash", "seed", "scope", "max_length"):
            assert key in environment, f"{key} is missing from environment"

    def test_the_per_stage_WALL_TIMES_are_recorded(self, pipeline_db, prepared, completed_run):
        """SC timing is reported from these; without them "Stage 1 in under 3 hours" is
        an impression rather than a measurement."""
        from src.models.probe_monitor import ProbeMonitorRun

        row = (
            pipeline_db.query(ProbeMonitorRun)
            .filter(ProbeMonitorRun.id == prepared["run"].id)
            .first()
        )
        timings = (row.environment or {}).get("stage_seconds") or {}
        assert timings, "no per-stage timings were recorded"
        assert set(timings) >= {"rendering", "pooled_capture", "training"}

    def test_the_render_summary_counts_every_row(self, pipeline_db, prepared, completed_run):
        from src.models.probe_monitor import ProbeMonitorRun

        row = (
            pipeline_db.query(ProbeMonitorRun)
            .filter(ProbeMonitorRun.id == prepared["run"].id)
            .first()
        )
        render = (row.environment or {}).get("render") or {}
        assert render.get("rendered", 0) > 0
        assert "role_mask_unreliable" in render
        assert render.get("counts", {}).get("positive", 0) > 0

    def test_the_artifact_directory_holds_the_run_output(self, completed_run, prepared):
        from src.services.probe_monitor_run import artifact_dir_for

        directory = artifact_dir_for(prepared["run"].id)
        assert directory.exists()
        names = {path.name for path in directory.iterdir()}
        assert any(name.endswith(".safetensors") for name in names), names
        assert any(name.startswith("tokens_layer") for name in names), names


# ⚠ THE CANCELLATION TESTS LIVE IN `test_probe_monitor_cancellation.py`, and the
# filename is load-bearing. `test_cancel_registry_completeness` discovers Shape-A
# cancellation tests by globbing `test_*cancel*.py` — so a cancellation test in a file
# named anything else satisfies nothing, and the completeness guard reports that nobody
# has demonstrated the scope stops work. That guard's own docstring records finding a
# previous version of itself satisfied by the file doing the asserting.


def row_layers(db, prepared):
    from src.models.probe_monitor import ProbeMonitorRun

    row = (
        db.query(ProbeMonitorRun)
        .filter(ProbeMonitorRun.id == prepared["run"].id)
        .first()
    )
    return set((row.layer_selection or {}).get("chosen") or [])


class TestTheRunsOwnBookkeepingSurvives:
    """⚠ `artifact_dir` AND THE GPU PLACEMENT WERE BOTH NULL ON A RUN THAT COMPLETED.

    Measured on the Stage 1 acceptance run `pmr_f70190de92e6`: it finished, reached rung 2,
    and stored `artifact_dir = NULL` while
    `/data/probe_monitors/pmr_f70190de92e6/tokens_layer11.f16` held **7.7 GB**. Since
    `DELETE /runs/{id}` reclaims the directory by reading `artifact_dir`, the token capture
    of every successful run was unreclaimable. `gpu_uuid` was NULL too, which FR-15 needs
    and multi-GPU accounting reads.

    THE MECHANISM IS NOT LOCAL TO THIS FILE. `CancelCheck._fetch` and `record_progress`
    both read the row with `.populate_existing()`, deliberately, so a long-lived task
    session can observe a cancel written by the API process (MIS-E2E-057). On the task's
    own session that refreshes every attribute from the database and REVERTS anything
    assigned but not yet committed — and `stage()` runs the cancel check BEFORE its own
    commit, so the first stage boundary threw both values away.

    MUTATION CONTROLS:
      P1  the commit after `row.artifact_dir = …` removed   → the artifact_dir test
      P2  the commit after the loader removed               → the placement test
    """

    def test_the_gpu_placement_is_persisted(self, pipeline_db, prepared, completed_run):
        """The other half, lost by the same mechanism: FR-15 needs the card a run used,
        and multi-GPU accounting reads it."""
        from src.models.probe_monitor import ProbeMonitorRun

        row = (
            pipeline_db.query(ProbeMonitorRun)
            .filter(ProbeMonitorRun.id == prepared["run"].id)
            .first()
        )
        assert row.gpu_uuid == FAKE_GPU_UUID, (
            f"gpu_uuid is {row.gpu_uuid!r}, not the placement the loader took; the card a "
            f"run used is unrecorded"
        )

    def test_the_artifact_dir_is_persisted(self, pipeline_db, prepared, completed_run):
        from src.models.probe_monitor import ProbeMonitorRun

        row = (
            pipeline_db.query(ProbeMonitorRun)
            .filter(ProbeMonitorRun.id == prepared["run"].id)
            .first()
        )
        assert row.artifact_dir, (
            "artifact_dir is unset on a completed run, so its token capture — gigabytes "
            "on a real run — can never be reclaimed by DELETE"
        )

    def test_it_is_committed_and_not_merely_in_the_session(
        self, pipeline_db, prepared, completed_run
    ):
        """The distinction that matters: the value must be in the DATABASE. Read through a
        separate session, because the run's own session would show it either way."""
        from sqlalchemy.orm import sessionmaker

        from src.models.probe_monitor import ProbeMonitorRun

        other = sessionmaker(bind=pipeline_db.get_bind())()
        try:
            row = (
                other.query(ProbeMonitorRun)
                .filter(ProbeMonitorRun.id == prepared["run"].id)
                .first()
            )
            assert row.artifact_dir, "artifact_dir never reached the database"
        finally:
            other.close()

    def test_it_survives_a_populate_existing_read_on_the_same_session(
        self, pipeline_db, prepared, completed_run
    ):
        """The mechanism itself, pinned so the reason is recorded rather than inferred: a
        cancel check on the run's own session must not be able to revert it."""
        from src.core.cancellation import cancel_checker
        from src.models.probe_monitor import ProbeMonitorRun

        row = (
            pipeline_db.query(ProbeMonitorRun)
            .filter(ProbeMonitorRun.id == prepared["run"].id)
            .first()
        )
        before = row.artifact_dir
        assert before, "nothing to protect — see the test above"
        checker = cancel_checker("probe_monitor_run", prepared["run"].id, db=pipeline_db)
        checker()          # a populate_existing() read on this very session
        pipeline_db.refresh(row)
        assert row.artifact_dir == before, (
            "a cancel check on the run's own session reverted artifact_dir, which is "
            "exactly how it was lost in production"
        )

    def test_an_uncommitted_assignment_really_is_reverted(self, pipeline_db, prepared):
        """The control for the claim above. If `populate_existing()` did NOT discard
        pending changes, every assertion here would pass against the original defect and
        this whole class would be documenting a fiction."""
        from src.core.cancellation import cancel_checker
        from src.models.probe_monitor import ProbeMonitorRun

        row = (
            pipeline_db.query(ProbeMonitorRun)
            .filter(ProbeMonitorRun.id == prepared["run"].id)
            .first()
        )
        row.artifact_dir = "/tmp/not-committed-on-purpose"
        checker = cancel_checker("probe_monitor_run", prepared["run"].id, db=pipeline_db)
        checker()
        assert row.artifact_dir != "/tmp/not-committed-on-purpose", (
            "populate_existing() did NOT discard the pending assignment, so the recorded "
            "mechanism is wrong and the real cause is still unexplained"
        )
        pipeline_db.rollback()


class TestTheTrainingCurveIsKept:
    """⚠ NOT STORING IT COST A WRONG DIAGNOSIS.

    `best_epoch` and `epochs_run` say where training stopped and nothing about whether it
    had converged. The Stage 1 acceptance run's attention probe stopped at epoch 88 with
    its best at 48, and from those two numbers the obvious reading was "early stopping cut
    it off"; re-running the same rule for 600 epochs with early stopping disabled gained
    0.006 of AUROC — it had converged, and the real finding was that the rule overfits. The
    loss curve distinguishes those two at a glance, and it had to be regenerated from the
    saved capture in a 26-minute experiment because the run discarded it.

    MUTATION CONTROLS:
      C1  `history` dropped from val_metrics        → the curve test
      C2  only the best epoch's entry kept          → the every-epoch test
      C3  the loss omitted from each entry          → the fields test
    """

    def test_the_curve_is_persisted(self, pipeline_db, prepared, completed_run):
        from src.models.probe_monitor import ProbeMonitor

        probes = (
            pipeline_db.query(ProbeMonitor)
            .filter(ProbeMonitor.run_id == prepared["run"].id)
            .all()
        )
        assert probes
        for probe in probes:
            history = (probe.val_metrics or {}).get("history")
            assert history, f"{probe.rule} stored no training history"

    def test_it_covers_every_epoch_that_ran(self, pipeline_db, prepared, completed_run):
        """A curve that keeps only its best point answers nothing — the question is what
        the loss was doing when training stopped."""
        from src.models.probe_monitor import ProbeMonitor

        for probe in (
            pipeline_db.query(ProbeMonitor)
            .filter(ProbeMonitor.run_id == prepared["run"].id)
            .all()
        ):
            metrics = probe.val_metrics or {}
            history = metrics["history"]
            assert len(history) == metrics["epochs_run"], (
                f"{probe.rule}: {len(history)} entries for {metrics['epochs_run']} epochs"
            )
            assert [e["epoch"] for e in history] == list(range(1, len(history) + 1))

    def test_every_entry_carries_the_loss_and_the_val_auroc(
        self, pipeline_db, prepared, completed_run
    ):
        from src.models.probe_monitor import ProbeMonitor

        probe = (
            pipeline_db.query(ProbeMonitor)
            .filter(ProbeMonitor.run_id == prepared["run"].id)
            .first()
        )
        for entry in (probe.val_metrics or {})["history"]:
            assert "loss" in entry, "without the loss the curve cannot show convergence"
            assert "val_auroc" in entry
            assert isinstance(entry["loss"], float)

    def test_the_best_epoch_really_is_the_best_in_the_curve(
        self, pipeline_db, prepared, completed_run
    ):
        """The two must agree, or the stored curve describes a different run than the
        weights do — the shape of defect that reports one epoch's number over another
        epoch's probe."""
        from src.models.probe_monitor import ProbeMonitor

        for probe in (
            pipeline_db.query(ProbeMonitor)
            .filter(ProbeMonitor.run_id == prepared["run"].id)
            .all()
        ):
            metrics = probe.val_metrics or {}
            scored = [e for e in metrics["history"] if e["val_auroc"] is not None]
            if not scored:
                continue
            best = max(scored, key=lambda e: e["val_auroc"])
            assert best["epoch"] == metrics["best_epoch"], (
                f"{probe.rule}: best_epoch is {metrics['best_epoch']} but the curve peaks "
                f"at {best['epoch']}"
            )
            assert best["val_auroc"] == pytest.approx(metrics["val_auroc"]), (
                f"{probe.rule}: the reported val_auroc is not the curve's peak"
            )
