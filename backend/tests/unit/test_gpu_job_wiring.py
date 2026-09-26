"""Every GPU task is wrapped by its claim and dispatched through the routing helper (Phase 3).

Two guards, both read from the AUTHORITY rather than a copy:

* the LIVE Celery registry says which tasks carry ``@gpu_job`` and how — a task
  whose decorator is removed drops out of the expected set and goes red;
* the AST of ``src/`` says how each GPU task is queued — a ``.delay`` or
  ``.apply_async`` on a GPU-queue task outside ``services/gpu_dispatch.py`` is a
  site that bypasses the routing decision, and a GPU-queue task that no
  ``gpu_delay``/``dispatch_gpu_task`` call ever names is unreachable in per-card
  mode.

The AST walk resolves import aliases and matches CALLS, never text: a text
scrape matches the comments describing a call and passes for the wrong reason.

MUTATION CONTROLS (2026-09-14; each alone, restored byte-identically, sha256 checked,
`git diff` clean) — all red:
  D3 the readout endpoint queues with `compute_readout.delay(` again -> queued around the helper; never dispatched by it
  D4 `@gpu_job` removed from compute_probe                           -> registry sets; handoff; outermost
  D5 `@gpu_job` placed inside `owns_its_failure` on the fit          -> the claim is the outermost wrapper
  D6 the token-redacting queue calls apply_async directly            -> the token-redacting queue goes through the helper
  J1 cleanup_stuck_trainings no longer releases                      -> its janitor payload
  J2 a circuit janitor releases by the wrong task-id column          -> its janitor payload

REVIEW ROUND 1 (2026-09-14): does the guard fail for a GPU task queued around the helper?
Six experiments, each appended alone to src/workers/jlens_probe_tasks.py (scratchpad
p3-r1-claim/spec_wiring.json), this module run, the source restored by sha256:
  W1 a NEW @gpu_job task queued with .delay                -> red before and after (registry sets; around; never dispatched)
  W2 an existing GPU task queued through an import alias   -> red before and after (around the helper)
  W3 celery_app.tasks["…compute_readout"].delay()          -> GREEN before; red after (around the helper)
  W4 send_task(NAME) with the name in a module constant    -> GREEN before; red after (around the helper)
  W5 celery_app.signature("…compute_readout").delay()      -> GREEN before; red after (around the helper)
  W6 a task WITHOUT @gpu_job that calls place_job, .delay  -> GREEN before; red after (a task that places carries its claim)
The four survivors were guard gaps; the extended checks are what turned each red. Limit:
a placement reached only through a service the task calls is not seen statically — the
per-card run-time refusal (UNCLAIMED_MESSAGE) is the backstop there.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from src.core.celery_app import celery_app
from src.services.gpu_dispatch import GPU_JOB_MARKER, gpu_job_spec

SRC = Path(__file__).resolve().parents[2] / "src"

#: Tasks that run on the GPU queues, claim at placement and may hand off.
GPU_QUEUE_TASKS = {
    # 032 probe monitors: the three GPU tasks. The judge is NOT here — it is HTTP
    # calls to a served model and takes no lease, because a lease for a
    # network-bound loop idles a card for its whole duration.
    "src.workers.probe_monitor_tasks.run_probe_monitor",
    "src.workers.probe_monitor_tasks.evaluate_probe_monitor",
    "src.workers.probe_monitor_tasks.score_probe_monitor",
    # 033: building a definition runs real forward passes for its test vectors, so it takes a
    # card. `publish_probe_definition` is NOT here — it is an HTTP upload, and a lease for a
    # network-bound transfer idles a card for its whole duration.
    "src.workers.probe_monitor_tasks.build_probe_definition",
    "workers.model_tasks.extract_activations",
    "train_sae",
    "src.workers.training_evaluation_tasks.evaluate_training",
    "src.workers.extraction_tasks.extract_features_from_sae",
    "src.workers.circuit_capture_tasks.capture_circuit_activations",
    "src.workers.circuit_capture_tasks.run_circuit_attribution",
    "src.workers.circuit_validation_tasks.validate_circuit_edges",
    "src.workers.circuit_validation_tasks.run_circuit_faithfulness",
    "src.workers.circuit_calibration_tasks.run_circuit_calibration",
    "src.workers.circuit_calibration_tasks.reproduce_circuit_calibration",
    "src.workers.circuit_record_tasks.run_circuit_record",
    "src.workers.jlens_fit_tasks.fit_jlens_artifact",
    "src.workers.jlens_fit_tasks.revalidate_staged_artifact",
    "src.workers.jlens_readout_tasks.compute_readout",
    "src.workers.jlens_probe_tasks.compute_probe",
    "src.workers.jlens_band_tasks.compute_band_report",
    "src.workers.jlens_acquire_tasks.acquire_jlens_artifact",
    "src.workers.jlens_intervention_tasks.run_intervention",
}

#: Tasks that keep their own queue and claim a GPU in place, part-way through.
IN_PLACE_TASKS = {
    "workers.model_tasks.download_and_load_model",
    "label_features",
    "labeling.resume_sweep_step",
    "neuronpedia.compute_dashboard_data",
    "push_to_neuronpedia_local",
    "steering.compare",
    "steering.sweep",
    "steering.combined",
}

DISPATCH_HELPERS = {"gpu_delay", "dispatch_gpu_task"}
QUEUEING_METHODS = {"delay", "apply_async", "s", "si", "signature", "apply", "map", "starmap", "chunks"}


def _registry():
    return {name: task for name, task in celery_app.tasks.items() if gpu_job_spec(task) is not None}


class TestTheLiveRegistry:
    def test_every_gpu_queue_task_carries_its_claim(self):
        marked = {name for name, task in _registry().items() if gpu_job_spec(task).gpu_queue}
        assert marked == GPU_QUEUE_TASKS

    def test_every_in_place_task_carries_its_claim(self):
        marked = {name for name, task in _registry().items() if not gpu_job_spec(task).gpu_queue}
        assert marked == IN_PLACE_TASKS

    def test_in_place_tasks_never_hand_off_and_gpu_queue_tasks_do_unless_stated(self):
        registry = _registry()
        assert not any(gpu_job_spec(registry[name]).handoff for name in IN_PLACE_TASKS)
        # The one GPU-queue task that waits in place: acquire fetches an artifact first.
        waiting = {name for name in GPU_QUEUE_TASKS if not gpu_job_spec(registry[name]).handoff}
        assert waiting == {"src.workers.jlens_acquire_tasks.acquire_jlens_artifact"}

    def test_the_claim_is_the_outermost_wrapper(self):
        """Inside `owns_its_failure` or `cooperative_cancel`, a hand-off would be recorded as a failure.

        Walks the `__wrapped__` chain by CODE OBJECT: `functools.wraps` copies
        names and `__dict__` outward (markers included), and Celery adds its own
        wrapper for `autoretry_for`, but a function's `co_qualname` is its own.
        """
        checked = 0
        for name, task in _registry().items():
            chain, fn = [], task.run
            while fn is not None:
                fn = getattr(fn, "__func__", fn)
                chain.append(getattr(getattr(fn, "__code__", None), "co_qualname", ""))
                fn = getattr(fn, "__wrapped__", None)
            claim = [i for i, qualname in enumerate(chain) if qualname.startswith("gpu_job.")]
            assert len(claim) == 1, f"{name}: {chain}"
            inner_decorators = [i for i, qualname in enumerate(chain)
                                if qualname.startswith(("owns_its_failure.", "cooperative_cancel."))]
            assert all(i > claim[0] for i in inner_decorators), f"{name}: @gpu_job is not outermost: {chain}"
            checked += len(inner_decorators)
        assert checked >= 9, "the chain walk found too few inner decorators to be checking anything"


def _gpu_function_names() -> set:
    names = set()
    for name, task in _registry().items():
        if gpu_job_spec(task).gpu_queue:
            names.add(getattr(task.run, "__name__", None))
    names.discard(None)
    return names


def _modules():
    for path in sorted(SRC.rglob("*.py")):
        yield path, ast.parse(path.read_text(), filename=str(path))


def _aliases(tree) -> dict:
    """local name -> imported name, for every `from X import a as b`."""
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                out[alias.asname or alias.name] = alias.name
    return out


def _named(node, aliases) -> str | None:
    if isinstance(node, ast.Name):
        return aliases.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


#: The calls that choose or take a GPU for a job. A task body that calls one must carry `@gpu_job`.
PLACEMENT_ENTRY_POINTS = {
    "place_job", "place_on_card", "place_circuit_job", "reuse_card", "claim_cards", "claim_exact_cards",
}


def _string_constants(tree) -> dict:
    """Module-level `NAME = "a string"` assignments, so a task name held in a constant resolves."""
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    out[target.id] = node.value.value
    return out


def _task_name(node, constants) -> str | None:
    """The task name a call's argument spells: a string literal, or a module constant holding one."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    return None


class TestEveryDispatchGoesThroughTheHelper:
    def test_the_registry_scan_finds_the_task_functions(self):
        names = _gpu_function_names()
        assert {"extract_activations", "train_sae_task", "compute_readout"} <= names, names

    def test_no_gpu_queue_task_is_queued_around_the_helper(self):
        """Every spelling of "queue this task" that names a GPU-queue task, outside the helper.

        Review round 1 (2026-09-14) found three spellings this walked past, each queuing
        `compute_readout` from a real module with the suite green: the registry subscript
        `celery_app.tasks["…"].delay()` (W3), `send_task(NAME)` with the name in a module
        constant (W4), and `celery_app.signature("…").delay()` (W5).
        """
        gpu_functions = _gpu_function_names()
        offenders = []
        for path, tree in _modules():
            if path.name == "gpu_dispatch.py":
                continue
            aliases = _aliases(tree)
            constants = _string_constants(tree)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                where = f"{path.relative_to(SRC)}:{node.lineno}"
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr in QUEUEING_METHODS:
                    if _named(func.value, aliases) in gpu_functions:
                        offenders.append(f"{where} {_named(func.value, aliases)}.{func.attr}")
                    if isinstance(func.value, ast.Subscript) and _task_name(func.value.slice, constants) in GPU_QUEUE_TASKS:
                        offenders.append(f"{where} tasks[{_task_name(func.value.slice, constants)}].{func.attr}")
                called = _named(func, {})
                named = [_task_name(a, constants) for a in node.args[:1]] + \
                        [_task_name(k.value, constants) for k in node.keywords if k.arg == "name"]
                if called == "send_task" and any(n in GPU_QUEUE_TASKS for n in named):
                    offenders.append(f"{where} send_task({named})")
                if called in {"signature", "Signature", "subtask"} and any(n in GPU_QUEUE_TASKS for n in named):
                    offenders.append(f"{where} {called}({named})")
        assert not offenders, "GPU tasks queued around services/gpu_dispatch: " + ", ".join(offenders)

    def test_a_task_that_places_on_a_gpu_carries_its_claim(self):
        """A Celery task whose OWN body calls a placement entry point must carry `@gpu_job`.

        In per-card mode an unwrapped placement is refused at run time
        (`gpu_job_claim.UNCLAIMED_MESSAGE`), but only then: review round 1's W6 added such a
        task, dispatched with `.delay`, and every guard here stayed green. Read from the LIVE
        registry and the AST of each task's own function, located by its code object.
        LIMIT, recorded: a placement reached only through a helper the task calls (a service)
        is not seen here; the run-time refusal remains the backstop for those.
        """
        offenders = []
        for name, task in celery_app.tasks.items():
            if gpu_job_spec(task) is not None:
                continue
            fn = getattr(task, "run", None)
            while getattr(fn, "__wrapped__", None) is not None:
                fn = fn.__wrapped__
            code = getattr(getattr(fn, "__func__", fn), "__code__", None)
            if code is None or not code.co_filename.startswith(str(SRC)):
                continue
            tree = ast.parse(Path(code.co_filename).read_text())
            defs = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and n.name == code.co_name and code.co_firstlineno in
                    {n.lineno, *(d.lineno for d in n.decorator_list)}]
            for function in defs:
                for node in ast.walk(function):
                    if isinstance(node, ast.Call) and _named(node.func, {}) in PLACEMENT_ENTRY_POINTS:
                        offenders.append(f"{name} calls {_named(node.func, {})} at line {node.lineno}")
        assert not offenders, "tasks that place on a GPU without @gpu_job: " + ", ".join(offenders)

    def test_every_gpu_queue_task_is_dispatched_by_the_helper_somewhere(self):
        gpu_functions = _gpu_function_names()
        dispatched = set()
        for path, tree in _modules():
            aliases = _aliases(tree)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and _named(node.func, aliases) in DISPATCH_HELPERS | {"_queue_without_leaking_token"} \
                        and node.args:
                    dispatched.add(_named(node.args[0], aliases))
        assert gpu_functions <= dispatched, f"never dispatched through the helper: {sorted(gpu_functions - dispatched)}"

    def test_the_token_redacting_queue_goes_through_the_helper(self):
        """`acquire` is queued by `_queue_without_leaking_token`, which must route through the helper."""
        tree = ast.parse((SRC / "api" / "v1" / "endpoints" / "jlens.py").read_text())
        (function,) = [n for n in ast.walk(tree)
                       if isinstance(n, ast.FunctionDef) and n.name == "_queue_without_leaking_token"]
        calls = [n for n in ast.walk(function) if isinstance(n, ast.Call)]
        assert any(_named(c.func, {}) == "dispatch_gpu_task" for c in calls)
        assert not any(isinstance(c.func, ast.Attribute) and c.func.attr in {"apply_async", "delay"} for c in calls)


#: Janitor module -> the task-id columns its reaps release leases by. The
#: payload, not just the call: a release keyed by the wrong column frees nothing.
JANITOR_RELEASES = {
    "workers/cleanup_stuck_trainings.py": ["celery_task_id"],
    "workers/cleanup_stuck_activations.py": ["celery_task_id"],
    "workers/cleanup_stuck_extractions.py": ["celery_task_id"],
    "workers/cleanup_stuck_circuit_runs.py": [
        "attribution_task_id", "calibration_task_id", "celery_task_id",
        "faithfulness_task_id", "task_id", "validation_task_id",
    ],
    "workers/cleanup_orphaned_tasks.py": ["task_id"],
    "workers/cleanup_stuck_labeling.py": ["celery_task_id"],
}


class TestJanitorsReleaseWhatTheyReap:
    """A janitor that fails a dead GPU job also frees the card(s) its execution leased.

    `release_reaped_leases` itself is exercised against real Postgres in
    test_gpu_job_wrapper.py; this pins each janitor's CALL and its argument.
    """

    @pytest.mark.parametrize("module", sorted(JANITOR_RELEASES))
    def test_each_reap_releases_by_the_reaped_rows_task_id(self, module):
        tree = ast.parse((SRC / module).read_text())
        released = sorted(
            node.args[0].attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and _named(node.func, {}) == "release_reaped_leases"
            and node.args and isinstance(node.args[0], ast.Attribute)
        )
        assert released == JANITOR_RELEASES[module]
