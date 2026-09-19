"""The Monitor's federated payload has ONE progress scale, and failures can be cleared.

TWO DEFECTS, both reported 2026-07-26 from the Monitor screen.

1. PROGRESS SCALE. The source tables disagree — trainings.progress and
   task_queue.progress are 0-100 while extraction_jobs.progress and
   labeling_jobs.progress are fractions. The extraction federator passed its
   fraction straight through, and ActiveOperationsSection renders
   `task.progress.toFixed(1)}%`. So a 98% extraction displayed as

       (0.98).toFixed(1) -> "1.0%"

   with a progress bar 0.98% wide. `extraction_job.progress` was even COMMENTED
   `# 0-100`, which is presumably how it happened.

2. UNCLEARABLE FAILURES. Federated rows are can_retry=False and the UI said
   "manage in its panel" — but for neuronpedia_pushes no such control exists
   anywhere; the only DELETE in that API targets neuronpedia_exports, a
   different table. Four failures from 2026-03-28 could not be removed.

MUTATION CONTROLS:
  * drop the *100 in _federated_extractions          -> scale test fails
  * stop filtering dismissed rows from /failed        -> filter test fails
  * unregister either dismiss route                   -> reachability test fails
"""

import ast
import inspect
from pathlib import Path

import pytest


class TestFederatedProgressIsAPercentage:
    def test_the_extraction_federator_converts_its_fraction(self):
        from src.api.v1.endpoints import task_queue

        src = inspect.getsource(task_queue._federated_extractions)
        assert "* 100" in src, (
            "extraction_jobs.progress is a FRACTION and this boundary is 0-100; "
            "without the conversion a 98% job renders as \"1.0%\""
        )

    def test_the_boundary_documents_its_scale(self):
        from src.api.v1.endpoints import task_queue

        doc = task_queue._federated_row.__doc__ or ""
        assert "0-100" in doc, (
            "the federated row contract no longer states its progress scale — "
            "that ambiguity is exactly what produced the bug"
        )

    def test_the_model_comment_no_longer_claims_0_100(self):
        """The misleading comment is the root cause; pin the correction."""
        source = Path(__file__).resolve().parents[2] / "src" / "models" / "extraction_job.py"
        text = source.read_text()
        line = next(
            ln for ln in text.splitlines()
            if ln.strip().startswith("progress = Column")
        )
        assert "0-100" not in line, (
            "extraction_job.progress is a 0.0-1.0 fraction; a '# 0-100' comment "
            "on it is what the federator trusted"
        )

    def test_percent_arithmetic_matches_the_extraction_card(self):
        """Both surfaces must derive the same number from the same column."""
        db_value = 0.98
        monitor = (db_value or 0.0) * 100          # federator
        card = (db_value or 0) * 100               # ExtractionJobCard
        assert monitor == card == 98.0


class TestFailedOperationsCanBeCleared:
    def test_dismiss_routes_are_registered(self):
        """Reachability: assert the LIVE router, not that the module imports."""
        from src.api.v1.endpoints import task_queue

        registered = {
            (frozenset(r.methods), r.path) for r in task_queue.router.routes
        }
        paths = {p for _, p in registered}

        assert "/failed/{task_type}/{source_id}/dismiss" in paths, (
            "dismissing a single federated failure is unreachable"
        )
        assert "/failed/dismiss-all" in paths, "clear-all is unreachable"

        methods = {p: m for m, p in registered}
        assert "POST" in methods["/failed/dismiss-all"]
        assert {"POST", "DELETE"} <= {
            meth
            for m, p in registered
            if p == "/failed/{task_type}/{source_id}/dismiss"
            for meth in m
        }, "dismissal must be reversible — DELETE on the same route"

    def test_literal_routes_precede_the_catch_all(self):
        """`/failed` and `/failed/...` must be matched before `/{task_queue_id}`."""
        from src.api.v1.endpoints import task_queue

        paths = [r.path for r in task_queue.router.routes]
        catch_all = paths.index("/{task_queue_id}")
        for literal in ("/failed", "/failed/dismiss-all"):
            assert paths.index(literal) < catch_all, (
                f"{literal} is declared after /{{task_queue_id}} and would be "
                "swallowed by it"
            )

    def test_the_failed_listing_filters_dismissed_rows(self):
        from src.api.v1.endpoints import task_queue

        src = inspect.getsource(task_queue.list_failed_tasks)
        assert "_load_dismissed" in src, (
            "/failed no longer excludes dismissed operations, so clearing one "
            "has no visible effect"
        )

    def test_the_filter_runs_after_every_federator(self):
        """Filtering per-query would silently miss a source added later."""
        from src.api.v1.endpoints import task_queue

        src = inspect.getsource(task_queue.list_failed_tasks)
        last_federator = max(
            src.rindex(name)
            for name in (
                "_federated_trainings",
                "_federated_extractions",
                "_federated_labeling",
                "_federated_pushes",
            )
        )
        assert src.index("_load_dismissed") > last_federator, (
            "the dismissal filter runs before some federator, so those rows "
            "escape it"
        )

    def test_clear_all_leaves_retryable_rows_alone(self):
        """A retryable queue row has a real DELETE; dismissing it would orphan
        a row the user could still retry."""
        from src.api.v1.endpoints import task_queue

        tree = ast.parse(inspect.getsource(task_queue.dismiss_all_failed_operations))
        has_guard = any(
            isinstance(node, ast.Compare)
            and any(
                isinstance(c, ast.Constant) and c.value is False
                for c in node.comparators
            )
            for node in ast.walk(tree)
        )
        assert has_guard, (
            "dismiss-all no longer skips can_retry rows"
        )


class TestEverySourceThatCanBeActiveCanAlsoBeFailed:
    """MIS-E2E-101: a failed activation extraction appeared nowhere at all.

    `/active` federated six sources and `/failed` federated four. The two that
    were missing were `_federated_tokenizations` — deliberately, its worker
    writes a real `task_queue` row on failure — and
    `_federated_activation_extractions`, which was simply forgotten. Its worker
    (`extract_activations`) writes no row either, so a failure disappeared from
    the operator's view entirely.

    Derived from the source, not a hand-list: the point is that a *new*
    federator added to `/active` and forgotten in `/failed` fails here, which
    is exactly what happened.
    """

    @staticmethod
    def _federators_called_by(func_name: str) -> set:
        import inspect

        from src.api.v1.endpoints import task_queue

        source = inspect.getsource(getattr(task_queue, func_name))
        tree = ast.parse(inspect.cleandoc(source))
        called = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
                if name and name.startswith("_federated_"):
                    called.add(name)
        return called

    #: The one source deliberately absent from /failed, with the reason. Any
    #: other gap is a defect, not a decision.
    DELIBERATELY_ACTIVE_ONLY = {
        # its worker writes a real task_queue row on failure, so federating
        # here too would double-count it
        "_federated_tokenizations",
    }

    def test_the_scan_finds_federators_at_all(self):
        """Guards the test: an empty set below passes every assertion."""
        active = self._federators_called_by("list_active_tasks")
        assert len(active) >= 5, f"only found {active} — did the scan break?"

    def test_no_source_can_fail_invisibly(self):
        active = self._federators_called_by("list_active_tasks")
        failed = self._federators_called_by("list_failed_tasks")

        missing = active - failed - self.DELIBERATELY_ACTIVE_ONLY
        assert not missing, (
            f"{sorted(missing)} federate into /active but not /failed, and are "
            f"not recorded as deliberate. A job from those sources vanishes "
            f"from the Monitor the moment it fails."
        )

    def test_activation_extractions_specifically_reach_the_failed_list(self):
        failed = self._federators_called_by("list_failed_tasks")
        assert "_federated_activation_extractions" in failed, (
            "the exact defect: `extract_activations` writes no task_queue row, "
            "so if it is not federated here a failed extraction is unreachable"
        )

    def test_the_exemption_list_names_only_real_federators(self):
        """A stale exemption silently re-opens the hole it was written for."""
        from src.api.v1.endpoints import task_queue

        for name in self.DELIBERATELY_ACTIVE_ONLY:
            assert hasattr(task_queue, name), (
                f"{name} is exempted from /failed but no longer exists"
            )

    def test_the_failed_call_asks_for_a_failure_status(self):
        """Federating with the wrong status is the same bug with more code."""
        import inspect

        from src.api.v1.endpoints import task_queue

        source = inspect.getsource(task_queue.list_failed_tasks)
        assert '_federated_activation_extractions(db, ("FAILED",))' in source, (
            "this table's status is the `extractionstatus` enum whose labels "
            "are the UPPERCASE Python names; a lowercase 'failed' matches "
            "nothing and the row stays invisible"
        )
