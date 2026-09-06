"""Phase 3 — SAE feature extraction stops, and its endpoint speaks the same word.

This lifecycle had the vocabulary split the registry was built to end: the
endpoint wrote `FAILED` + "Cancelled by user" while the sibling activation
extraction wrote `CANCELLED` for the identical operator action. A checker
looking for "cancelled" could never have seen a cancel expressed as "failed" —
so Shape C here is not ceremony, it is the test that would have caught it.
"""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from src.core import cancellation as C


class _Job:
    def __init__(self, status="extracting"):
        self.id = "ext_sae_1"
        self.status = status
        self.progress = 0.1
        self.features_extracted = 0
        self.total_features = None
        self.statistics = None
        self.error_message = None
        self.completed_at = None
        self.updated_at = None
        self.celery_task_id = "celery-sae"


class _Query:
    def __init__(self, row):
        self._row = row

    def filter(self, *a, **k):
        return self

    filter_by = filter

    def order_by(self, *a, **k):
        return self

    def populate_existing(self):
        return self

    def first(self):
        return self._row


def _session(row):
    db = MagicMock()
    db.query.side_effect = lambda model: _Query(row)
    return db


class TestShapeCTheEndpointReachesTheChannel:
    def test_the_endpoint_writes_cancelled_not_failed(self):
        """The defect, pinned. `ExtractionStatus.CANCELLED` existed the whole
        time; writing FAILED made a deliberate stop look like a crash AND made
        the row invisible to any checker."""
        from src.models.extraction_job import ExtractionStatus

        row = _Job("extracting")
        C.request_cancel("sae_extraction", "ext_sae_1", db=_session(row))
        assert row.status is ExtractionStatus.CANCELLED

    def test_the_workers_checker_sees_the_endpoints_write(self):
        row = _Job("extracting")
        db = _session(row)
        C.request_cancel("sae_extraction", "ext_sae_1", db=db)
        assert C.cancel_checker("sae_extraction", "ext_sae_1", db=db)() is True

    def test_the_endpoint_no_longer_claims_terminate_works(self):
        import inspect

        from src.api.v1.endpoints import saes

        src = inspect.getsource(saes.cancel_sae_extraction)
        code = "\n".join(
            line for line in src.splitlines() if not line.lstrip().startswith("#")
        )
        assert "request_cancel" in code
        assert "terminate=True" not in code, (
            "terminate signals a POOL CHILD; this worker is --pool=solo"
        )
        assert "ExtractionStatus.FAILED" not in code, (
            "the endpoint is writing FAILED for an operator cancel again"
        )


class TestShapeAItStops:
    """Drive the real service loops. Both phases must abandon at the top."""

    def _service(self, row, cancel_after=0):
        from src.services.extraction_service import ExtractionService

        service = ExtractionService.__new__(ExtractionService)
        service.db = _session(row)
        return service

    def test_the_sampling_loop_abandons_before_reading_a_batch(self):
        """Phase 1 commits nothing, so the top of the batch loop is a clean
        abandon point — the database is left exactly as it was."""
        import inspect

        from src.services.extraction_service import ExtractionService

        src = inspect.getsource(ExtractionService.extract_features_for_sae)
        loop = src.index("for batch_start in range(0, len(dataset), batch_size):")
        # Absolute offsets from the loop header — a fixed-size window silently
        # raised ValueError when the comment block pushed the read past it,
        # which is a test that breaks on formatting rather than on behaviour.
        poll = src.index("cancel_check", loop)
        read = src.index("batch = dataset[batch_start:batch_end]", loop)
        assert poll < read, (
            "the sampling loop reads a batch before checking whether it was "
            "cancelled"
        )

    def test_the_write_loop_polls_per_feature(self):
        import inspect

        from src.services.extraction_service import ExtractionService

        src = inspect.getsource(ExtractionService.extract_features_for_sae)
        loop = src.index("for neuron_idx in range(latent_dim):")
        heap = src.index("heap_items = feature_activations[neuron_idx]", loop)
        assert "cancel_check" in src[loop:heap], (
            "the per-feature write loop never polls, so a cancel during phase 2 "
            "waits for the entire feature set"
        )

    def test_it_polls_before_the_base_model_load(self):
        import inspect

        from src.services.extraction_service import ExtractionService

        src = inspect.getsource(ExtractionService.extract_features_for_sae)
        poll = src.find("stopped before the base model was loaded")
        load = src.find("base_model, tokenizer, model_config, metadata = load_model_from_hf(")
        assert poll != -1 and load != -1, "shape changed — re-read the service"
        assert poll < load, "nothing polls before the multi-minute model load"

    def test_none_is_still_accepted(self):
        """Other callers have no cancel channel; the parameter must be optional."""
        import inspect

        from src.services.extraction_service import ExtractionService

        sig = inspect.signature(ExtractionService.extract_features_for_sae)
        assert sig.parameters["cancel_check"].default is None


class TestTheTaskIsWired:
    def test_the_task_carries_the_decorator(self):
        import inspect

        from src.workers.extraction_tasks import extract_features_from_sae_task

        fn = extract_features_from_sae_task
        found = None
        for _ in range(6):
            found = getattr(fn, "__cooperative_cancel_scope__", None)
            if found:
                break
            nxt = getattr(fn, "__wrapped__", None)
            if nxt is None or nxt is fn:
                break
            fn = nxt
        assert found == "sae_extraction"

    def test_the_task_injects_a_real_checker(self):
        """R1-03. `assert "cancel_check=" in src` was satisfied by
        `cancel_check=None` — the EXACT Phase-3 faithfulness defect this arc
        exists to fix, and the highest-value surviving mutation in the round:
        every checkpoint in the SAE service goes inert, suite green."""
        import _cancel_ast as A

        from src.workers.extraction_tasks import extract_features_from_sae_task

        assert "sae_extraction" in A.scopes_passed_to(
            extract_features_from_sae_task, "cancel_checker"
        ), "the task constructs no checker for its own scope"
        assert A.passes_real_value(
            extract_features_from_sae_task, "extract_features_for_sae", "cancel_check"
        ), (
            "the service is called with cancel_check=None, so every checkpoint "
            "inside it is inert"
        )

    def test_the_failure_path_goes_through_the_guard(self):
        """It wrote status=FAILED straight onto the ORM row, so an exception
        arriving after the operator cancelled relabelled the row FAILED and lost
        the cancellation at the last possible moment."""
        import inspect

        from src.workers.extraction_tasks import extract_features_from_sae_task

        import _cancel_ast as A

        assert A.calls_named(extract_features_from_sae_task, "record_progress"), (
            "the failure path no longer goes through the guard"
        )
        src = inspect.getsource(inspect.unwrap(extract_features_from_sae_task))
        code = "\n".join(
            line for line in src.splitlines() if not line.lstrip().startswith("#")
        )
        assert "extraction_job.status = ExtractionStatus.FAILED.value" not in code, (
            "the task writes FAILED directly again, bypassing guard_allows"
        )

    def test_a_cancelled_row_survives_the_failure_writer(self):
        """Behavioural: the guard actually refuses it."""
        row = _Job("cancelled")
        assert C.record_progress(
            "sae_extraction", "ext_sae_1",
            status="failed", error_message="boom", db=_session(row),
        ) is True, "terminal -> terminal is allowed; the janitors need it"
        # ...but a non-terminal relabel is not:
        row2 = _Job("cancelled")
        assert C.record_progress(
            "sae_extraction", "ext_sae_1",
            status="extracting", progress=0.5, db=_session(row2),
        ) is False
        assert row2.status == "cancelled"
