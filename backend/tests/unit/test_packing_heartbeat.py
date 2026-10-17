"""A long packing pass must look alive to the stuck-tokenization janitor.

MEASURED 2026-09-12. github-code-clean (7,177,394 files, ~7.3M blocks) was marked
ERROR by cleanup_stuck_tokenizations at 05:47:49 -- "no progress for more than
65 minutes. The worker was lost before it could record an outcome; no output was
written" -- while the worker logged "[PACKING] 3,320,000 blocks assembled" thirty
seconds later. The janitor spares a row whose updated_at is fresh OR whose
progress counter moved. Packing reported only over the WebSocket, so the row kept
progress=80.0 and a stale updated_at and failed both. It also released the
dataset, so the UI offered a second tokenization into the same output directory.
"""

import ast
import inspect

from src.services.tokenization_service import TokenizationService

heartbeat = TokenizationService.packing_heartbeat_progress


class TestTheHeartbeatValue:

    def test_strictly_increasing(self):
        values = [heartbeat(b) for b in (0, 5_000, 100_000, 1_000_000, 3_320_000, 7_300_000, 50_000_000)]
        assert all(a < b for a, b in zip(values, values[1:]))

    def test_stays_inside_the_packing_band(self):
        for blocks in (0, 1, 10**6, 10**9):
            assert 80.0 <= heartbeat(blocks) < 80.9

    def test_moves_between_janitor_sweeps_at_the_measured_pace(self):
        """One 10-minute sweep at ~50,000 blocks/min, late in a large corpus."""
        assert heartbeat(7_500_000) - heartbeat(7_000_000) > 1e-3

    def test_a_negative_count_is_clamped(self):
        assert heartbeat(-5) == heartbeat(0)


class TestTheHeartbeatReachesTheRow:
    """A heartbeat against a scope with no progress column writes nothing, and
    with nothing changed `onupdate` never refreshes updated_at either."""

    def test_the_scope_names_the_progress_column(self):
        from src.core.cancellation import get_scope

        assert get_scope("dataset_tokenization").progress_field == "progress"

    def test_a_live_row_accepts_it_and_a_failed_row_refuses_it(self):
        from src.core.cancellation import guard_allows

        assert guard_allows("dataset_tokenization", "processing", None, writes_progress=True)
        assert not guard_allows("dataset_tokenization", "error", None, writes_progress=True)


def _blocks_generator():
    from src.workers import dataset_tasks

    tree = ast.parse(inspect.getsource(dataset_tasks))
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef) and fn.name == "tokenize_dataset_task":
            for sub in ast.walk(fn):
                if isinstance(sub, ast.FunctionDef) and sub.name == "_blocks":
                    return sub
    raise AssertionError("_blocks generator not found inside tokenize_dataset_task")


def _heartbeat_calls(node):
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) and sub.func.id == "record_progress":
            kw = {k.arg: k.value for k in sub.keywords}
            value = kw.get("progress")
            if (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Attribute)
                and value.func.attr == "packing_heartbeat_progress"
            ):
                yield sub


class TestThePackingLoopWiring:

    def test_packing_writes_a_moving_progress_through_record_progress(self):
        """Assert the VALUE, not just the call: record_progress(progress=80.0)
        refreshes updated_at once and then never moves the counter."""
        assert list(_heartbeat_calls(_blocks_generator())), (
            "_blocks must call record_progress(progress=packing_heartbeat_progress(...))"
        )

    def test_the_write_is_throttled_on_time(self):
        gen = _blocks_generator()
        for node in ast.walk(gen):
            if not isinstance(node, ast.If):
                continue
            names = {n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)}
            if "PACKING_HEARTBEAT_SECONDS" in names and list(_heartbeat_calls(node)):
                return
        raise AssertionError("the heartbeat must sit under an `if ... PACKING_HEARTBEAT_SECONDS` time check")

    def test_the_interval_is_far_inside_the_janitor_threshold(self):
        from src.workers.cleanup_stuck_tokenizations import STUCK_THRESHOLD_MINUTES
        from src.workers.dataset_tasks import PACKING_HEARTBEAT_SECONDS

        assert 0 < PACKING_HEARTBEAT_SECONDS <= STUCK_THRESHOLD_MINUTES * 60 / 10
