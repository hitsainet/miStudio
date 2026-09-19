"""An extraction with no database row must refuse to start.

2026-08-24, from production. A task ran **3 hours 24 minutes** at 100% GPU
against `ext_m_4a703773_20260824_212308`, whose row was never created. What
followed, all from that one fact:

  * ~300 × `Extraction ... not found for progress update`, each ignored
  * the UI stuck on "Starting extraction..." the entire time, and the Monitor
    showing "No active operations" while the card was pinned
  * the eventual failure could not be recorded either — `not found to mark failed`
  * Celery retried, and each retry re-resolved the model and started a
    **spurious 15 GB download** of a model already on disk
  * 9 GB of orphaned `.npy` files nothing could ever read

Every symptom is downstream of something knowable in the first millisecond:
there is nowhere to write the result. This is the guard for that, plus the
rule that a permanent failure must not be retried.
"""

import ast
import inspect
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src"


def _task_source() -> str:
    """`inspect.unwrap`, NOT a single `__wrapped__` hop.

    A one-hop unwrap reaches celery's bound method, which was the task body
    until `@cooperative_cancel` was added beneath the celery decorator. After
    that, one hop lands on the CANCELLATION WRAPPER and every assertion below
    would have been read against `core/cancellation.py` instead — passing or
    failing for reasons having nothing to do with this task. A guard that reads
    the wrong function is a guard that fails open. `unwrap` follows the whole
    chain to the original.
    """
    from src.workers.model_tasks import extract_activations

    fn = inspect.unwrap(extract_activations)
    return inspect.getsource(fn)


class TestTheGuardExistsAndRunsFirst:
    def test_a_missing_row_raises_a_permanent_error(self):
        src = _task_source()
        assert "PermanentExtractionError" in src, (
            "nothing refuses to start when the extraction row is absent"
        )
        assert "get_extraction" in src, "the row is never looked up"

    def test_the_check_precedes_any_gpu_work(self):
        """Aborting after the model loads would still burn VRAM and minutes."""
        src = _task_source()
        guard = src.find("PermanentExtractionError")
        cuda = src.find("torch.cuda")
        assert guard != -1 and cuda != -1, "shape changed — re-read the task"
        assert guard < cuda, (
            "the row check happens after CUDA work begins; the point is to "
            "refuse before consuming the GPU at all"
        )

    def test_the_error_type_is_distinct_from_transient_failures(self):
        from src.workers.model_tasks import PermanentExtractionError

        from src.services.activation_service import ActivationExtractionError

        assert not issubclass(PermanentExtractionError, ActivationExtractionError), (
            "a permanent error that subclasses the transient one will be "
            "swept back into the retry path"
        )


class TestPermanentFailuresAreNotRetried:
    def test_the_retry_branch_excludes_permanent_errors(self):
        src = _task_source()
        assert "isinstance(exc, PermanentExtractionError)" in src, (
            "the retry branch does not exclude permanent errors; a missing "
            "model row will retry three times, re-resolving and re-downloading "
            "the model each attempt"
        )

    def test_a_missing_model_is_treated_as_permanent(self):
        """The exact string production produced, three times over."""
        src = _task_source()
        assert '"not found in database" in str(exc)' in src

    def test_the_refusal_comes_before_the_generic_retry(self):
        """Ordering against the GENERIC retry, not the first one in the file.

        There are two `self.retry(` sites. The earlier one is an OOM back-off
        that halves the batch size — genuinely transient and correct to keep.
        The one that must not be reached with a permanent error is the generic
        `if self.request.retries < self.max_retries` branch in the outer
        handler.
        """
        src = _task_source()
        refusal = src.find("Not retrying extraction")
        generic = src.find("if self.request.retries < self.max_retries")
        assert refusal != -1 and generic != -1, "shape changed — re-read the task"
        assert refusal < generic, (
            "the permanent-error check must short-circuit before the generic "
            "retry branch"
        )

    def test_the_oom_backoff_is_left_alone(self):
        """Halving the batch on OOM is a real recovery and must survive."""
        src = _task_source()
        assert "Retrying with batch_size=" in src, (
            "the OOM back-off was removed; that one SHOULD retry"
        )


class TestTheProgressWriterStillReportsHonestly:
    """The warn-and-continue path stays — it is right for a row deleted
    mid-run — but it must not be the ONLY thing standing between a phantom
    job and three hours of GPU time."""

    def test_the_warning_still_fires(self, caplog):
        """DRIVEN, NOT SCRAPED. This asserted the phrase was present in
        `extraction_db_service.py` until 2026-09-05, when the writer was routed
        through `core.cancellation.record_progress` and the phrase moved. A
        source scrape cannot tell "moved" from "deleted", and this repo's record
        is that such a guard fails OPEN — so it now calls the writer against a
        missing row and requires the warning to actually reach the log."""
        import logging
        from unittest.mock import MagicMock

        from src.models.activation_extraction import ExtractionStatus
        from src.services.extraction_db_service import ExtractionDatabaseService

        db = MagicMock()
        db.query.return_value.filter.return_value.populate_existing.return_value.first.return_value = None

        with caplog.at_level(logging.WARNING):
            result = ExtractionDatabaseService.update_progress(
                db=db,
                extraction_id="ext_gone",
                progress=42.0,
                status=ExtractionStatus.EXTRACTING,
                samples_processed=10,
            )

        assert result is None
        assert "not found for progress update" in caplog.text
        assert "ext_gone" in caplog.text

    def test_but_it_is_no_longer_the_only_defence(self):
        assert "PermanentExtractionError" in _task_source(), (
            "the task-start guard is gone, leaving only a warning that was "
            "ignored 300 times in production"
        )
