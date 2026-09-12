"""Task 11 — correctness bugs whose failure mode is a stranded row.

MIS-E2E-029  a 2 GiB capture overflowed a 32-bit column, poisoned the session,
             and left the run `running` forever with its store orphaned.
MIS-E2E-057  cancelling a labeling job could never be observed.
MIS-E2E-098  retry wiped the failure evidence, committed, then 400'd.
MIS-E2E-092  four of five janitors treated PENDING as alive, so they could
             never fire for any row carrying a task id.
"""

import ast
import inspect

import pytest


# ── MIS-E2E-029 · the counters ─────────────────────────────────────────────

@pytest.mark.parametrize("column", ["bytes_total", "events_total"])
def test_capture_counters_are_64_bit(column):
    """`bytes_total` counts BYTES; 32 bits caps at ~2 GiB.

    The capture completes, the final commit overflows, the session is poisoned,
    and the error handler that needs the same session to mark the run failed
    fails too — so the row sits `running` forever and the multi-gigabyte store
    is leaked with nothing pointing at it.
    """
    from src.models.circuit_runs import CircuitCaptureRun

    kind = str(CircuitCaptureRun.__table__.c[column].type).upper()
    assert "BIGINT" in kind, f"{column} is {kind}; a real capture exceeds 2 GiB"


def test_a_realistic_capture_size_exceeds_the_32_bit_ceiling():
    """Negative control for the premise. If 2 GiB were unreachable the fix
    would be unnecessary, so the arithmetic is computed rather than asserted.

    My first version of this test used 50k tokens / 3 layers / 16k features at
    1% sparsity and came to 98 MB — comfortably UNDER the ceiling, which would
    have made the fix unnecessary. The premise survives at realistic scale, but
    not at every scale, and that is worth writing down: the overflow is reached
    by wide multi-layer captures over a large corpus, not by every capture.
    """
    thirty_two_bit_max = 2_147_483_647

    # A wide multi-layer capture: 200k tokens, 6 layers, a 32k-feature
    # dictionary, fp32, at 2% density.
    wide = int(200_000 * 6 * 32_768 * 4 * 0.02)
    assert wide > thirty_two_bit_max, f"{wide / 1024**3:.1f} GiB"

    # And a modest one does NOT — the ceiling is reachable, not universal.
    modest = int(50_000 * 3 * 16_384 * 4 * 0.01)
    assert modest < thirty_two_bit_max


# ── MIS-E2E-057 · the cancel ───────────────────────────────────────────────

def test_the_cancel_check_bypasses_the_identity_map():
    """`expire_on_commit=False` means a plain re-query returns the object
    already in the session — never the cancel another connection wrote.

    DRIVEN, NOT SCRAPED. This asserted that `.populate_existing()` appeared in
    the text of `_raise_if_cancelled` until 2026-09-05, when that method became
    a shim over `core.cancellation` and the call moved one level down. A source
    scrape cannot distinguish "moved" from "deleted" — it failed against code
    whose behaviour was unchanged, and would equally have passed against a
    version that kept the string in a comment. The fake below models the
    identity map, so the assertion is now about what the check can SEE.
    """
    from types import SimpleNamespace

    from src.models.labeling_job import LabelingStatus
    from src.services.labeling_service import LabelingService

    class _Query:
        def __init__(self, session):
            self._session = session
            self._populate = False

        def populate_existing(self):
            self._populate = True
            return self

        def filter(self, *a, **k):
            return self

        def first(self):
            # Without populate_existing the session hands back the row as it
            # was first loaded — which is the whole defect.
            if self._populate:
                self._session.cached = self._session.db_row
            return self._session.cached

    class _Session:
        def __init__(self, row):
            self.cached = row
            self.db_row = row

        def query(self, _model):
            return _Query(self)

    live = SimpleNamespace(id="job_1", status=LabelingStatus.LABELING.value)
    session = _Session(live)

    svc = LabelingService.__new__(LabelingService)
    svc.db = session

    # Another connection cancels the job. The identity map still holds `live`.
    session.db_row = SimpleNamespace(
        id="job_1", status=LabelingStatus.CANCELLED.value
    )

    with pytest.raises(LabelingService._LabelingCancelled):
        svc._raise_if_cancelled("job_1")


def test_the_sibling_cancel_checks_open_a_fresh_session():
    """Why this fix is one site, not four.

    `training_tasks` and `neuronpedia_tasks` open a NEW session per check, so
    they have no stale identity map. Verified rather than assumed — the finding
    did not say how far the trap spread.
    """
    from src.workers import neuronpedia_tasks, training_tasks

    for mod, opener in (
        (training_tasks, "self.get_db()"),
        (neuronpedia_tasks, "sync_session_maker()"),
    ):
        assert opener in inspect.getsource(mod), (
            f"{mod.__name__} no longer opens a fresh session for its status "
            f"check; it now needs populate_existing too"
        )


# ── MIS-E2E-098 · retry must not destroy evidence ──────────────────────────

def _retry_source() -> str:
    from src.api.v1.endpoints import task_queue

    tree = ast.parse(inspect.getsource(task_queue))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "retry_task":
            return ast.unparse(node)
    raise AssertionError("retry_task not found")


def test_retry_refuses_before_it_wipes_the_row():
    """`increment_retry_count` sets error_message=None and COMMITS.

    Running it before the dispatch chain meant an unsupported pair left the row
    queued with its failure evidence destroyed, nothing dispatched, and a 400
    returned — permanently stranded, and invisible under /failed because its
    status was no longer failed.
    """
    body = _retry_source()
    guard = body.index("RETRYABLE_TASK_TYPES")
    wipe = body.index("increment_retry_count")
    assert guard < wipe, (
        "the retry-count reset runs before the supported-type check, so an "
        "unsupported task loses its error message and is stranded"
    )


def test_every_dispatch_branch_is_listed_as_retryable():
    """The allow-list is only worth having if it matches the code it guards.

    Read out of the AST rather than maintained by eye — a hand-list that can
    drift from its own if/elif chain is how the guard becomes wrong later.
    """
    from src.api.v1.endpoints.task_queue import RETRYABLE_TASK_TYPES

    body = _retry_source()
    pairs = set()
    for node in ast.walk(ast.parse(body)):
        if not isinstance(node, ast.BoolOp) or not isinstance(node.op, ast.And):
            continue
        found = {}
        for cmp_node in node.values:
            if not isinstance(cmp_node, ast.Compare):
                continue
            left = cmp_node.left
            right = cmp_node.comparators[0]
            if isinstance(left, ast.Attribute) and isinstance(right, ast.Constant):
                found[left.attr] = right.value
        if "task_type" in found and "entity_type" in found:
            pairs.add((found["task_type"], found["entity_type"]))

    assert pairs, "no dispatch branches found — the scan broke"
    assert pairs == set(RETRYABLE_TASK_TYPES), (
        f"branches {sorted(pairs)} disagree with RETRYABLE_TASK_TYPES "
        f"{sorted(RETRYABLE_TASK_TYPES)}"
    )


# ── MIS-E2E-092 · the janitors ─────────────────────────────────────────────

#: Every stuck-row janitor. Parametrized off this list rather than testing one
#: representative — "fixed one of five" IS the finding.
_JANITORS = [
    "src.workers.cleanup_stuck_trainings",
    "src.workers.cleanup_stuck_extractions",
    "src.workers.cleanup_stuck_activations",
    "src.workers.cleanup_stuck_enhanced_labeling",
    "src.workers.cleanup_stuck_circuit_runs",
    "src.workers.cleanup_stuck_tokenizations",
    # Feature 30. `labeling_jobs` was the only long-running lifecycle here with
    # no janitor, which is what turned its 409 lock into a trap: a job orphaned
    # by a worker restart 409s every future labeling run on that extraction
    # until someone deletes it by hand.
    "src.workers.cleanup_stuck_labeling",
]

#: Janitors that legitimately do NOT consult a Celery task id, with the reason.
#:
#: Found by `test_the_janitor_list_matches_what_is_on_disk`, which discovered a
#: sixth janitor the finding never named. It turned out to be correct by design
#: rather than a missed sibling — which is exactly why the discovery test exists
#: rather than a hand-list: it makes the question get asked.
_JANITORS_WITHOUT_TASK_IDS = {
    "src.workers.cleanup_stuck_nlp": (
        "ExtractionJob.celery_task_id belongs to the EXTRACTION task, not the "
        "NLP pass, and there is no nlp_celery_task_id column — so consulting it "
        "would ask about the wrong task. Documented in the module itself."
    ),
}


@pytest.mark.parametrize("modname", _JANITORS)
def test_no_janitor_treats_pending_as_alive(modname):
    """Celery reports PENDING for any id it holds no result for.

    `state in (PENDING, STARTED, RETRY)` can therefore never be false for a row
    carrying a task id, so these janitors could never fire for the failure they
    were each written to clear.
    """
    import importlib

    src = inspect.getsource(importlib.import_module(modname))
    for shape in ("'PENDING', 'STARTED', 'RETRY'", '"PENDING", "STARTED", "RETRY"'):
        assert shape not in src, (
            f"{modname} still treats PENDING as alive; it can never reclaim a "
            f"row whose worker died"
        )


@pytest.mark.parametrize("modname", _JANITORS)
def test_every_janitor_uses_the_shared_liveness_rule(modname):
    import importlib

    src = inspect.getsource(importlib.import_module(modname))
    assert "task_looks_alive" in src or "looks_abandoned" in src, (
        f"{modname} decides liveness on its own; the rule exists once in "
        f"task_heartbeat and this is how it came to be applied to one of five"
    )


def test_the_janitor_list_matches_what_is_on_disk():
    """Negative control: a new janitor must join this list, not slip past it."""
    from pathlib import Path

    import src.workers as workers_pkg

    on_disk = {
        f"src.workers.{p.stem}"
        for p in Path(workers_pkg.__file__).parent.glob("cleanup_stuck_*.py")
    }
    missing = on_disk - set(_JANITORS) - set(_JANITORS_WITHOUT_TASK_IDS)
    assert not missing, (
        f"janitors with no coverage here: {sorted(missing)} — add them to "
        f"_JANITORS, or to _JANITORS_WITHOUT_TASK_IDS with the reason"
    )


def test_the_exempt_janitor_really_has_no_task_id_to_consult():
    """Pin the exemption's REASON, not just the exemption.

    An exemption nobody re-checks is how a real gap gets waved through.
    """
    from src.models.feature import Feature  # noqa: F401 - registers metadata
    from src.models.extraction_job import ExtractionJob

    assert not hasattr(ExtractionJob, "nlp_celery_task_id"), (
        "ExtractionJob now has an NLP task id, so cleanup_stuck_nlp CAN check "
        "liveness and should join _JANITORS"
    )


def test_the_shared_rule_treats_a_dead_pending_task_as_abandoned():
    """The behaviour itself, not just its presence."""
    from src.workers.task_heartbeat import looks_abandoned

    # PENDING with no info and a stale row clock = the worker died.
    assert looks_abandoned("PENDING", None, 3600.0) is True
    # PENDING on a QUEUED row (no age passed) stays conservative.
    assert looks_abandoned("PENDING", None, None) is False
    # A terminal state is never "abandoned".
    assert looks_abandoned("SUCCESS", None, 99999.0) is False
