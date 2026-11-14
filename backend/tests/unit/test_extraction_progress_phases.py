"""Extraction progress must not read 100% while minutes of work remain.

REPORTED 2026-07-26: "please check on the current extraction job and why it's
at 100% but doesn't transition to complete."

The job was healthy. Extraction has TWO long phases:

  1. activation sampling  — `progress = batch_end / len(dataset)`, so it hit
     exactly 1.0 on the last batch
  2. writing feature records — updated `features_extracted` only, never
     `progress`, so the field kept the 1.0 from phase 1

Measured on that job: phase 2 ran 19:15:47 -> ~19:25 for 32,768 features
(2,000 per ~32s). Nine minutes of a full bar with no status change.

MUTATION CONTROLS:
  * let sampling reach 1.0 again              -> the ceiling test fails
  * drop `progress=` from the write loop      -> the write-phase test fails
"""

import ast
import inspect
from pathlib import Path

from src.services import extraction_service
from src.services.extraction_service import SAMPLING_PROGRESS_SHARE


class TestSamplingDoesNotClaimTheWholeBar:
    def test_share_leaves_room_for_the_write_phase(self):
        assert 0.0 < SAMPLING_PROGRESS_SHARE < 1.0, (
            "sampling would own the entire bar again"
        )

    def test_sampling_progress_is_scaled_by_the_share(self):
        src = inspect.getsource(extraction_service)
        assert "SAMPLING_PROGRESS_SHARE * batch_end / len(dataset)" in src, (
            "the sampling loop no longer scales its progress, so it reaches "
            "1.0 while the write phase has not started"
        )

    def test_sampling_never_reports_a_full_bar(self):
        """Simulate the loop's arithmetic at the final batch."""
        total = 2000
        for batch_end in (1, 500, 1999, total):
            progress = SAMPLING_PROGRESS_SHARE * batch_end / total
            assert progress < 1.0, (
                f"sampling reported {progress} at batch_end={batch_end}"
            )
        assert SAMPLING_PROGRESS_SHARE * total / total == SAMPLING_PROGRESS_SHARE


class TestWritePhaseAdvancesTheBar:
    def test_the_commit_loop_passes_a_progress_value(self):
        """The defect was an update that omitted `progress` entirely."""
        tree = ast.parse(inspect.getsource(extraction_service))

        found = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if getattr(node.func, "attr", None) != "update_extraction_status_sync":
                continue
            kwargs = {kw.arg for kw in node.keywords}
            if "features_extracted" in kwargs and "statistics" not in kwargs:
                found.append(kwargs)

        assert found, "no incremental status update found in the commit loop"
        for kwargs in found:
            assert "progress" in kwargs, (
                "the write-phase update omits `progress`, so the bar keeps "
                f"whatever sampling last wrote (kwargs seen: {sorted(kwargs)})"
            )

    def test_the_bar_is_monotonic_across_the_phase_boundary(self):
        latent_dim = 32768
        end_of_sampling = SAMPLING_PROGRESS_SHARE * 1.0

        previous = end_of_sampling
        for processed in (2000, 16000, 32000, latent_dim):
            fraction = processed / latent_dim
            progress = (
                SAMPLING_PROGRESS_SHARE
                + (1.0 - SAMPLING_PROGRESS_SHARE) * fraction
            )
            assert progress >= previous, "the bar went backwards"
            assert progress <= 1.0, "the bar exceeded 100%"
            previous = progress

        assert previous == 1.0, "the bar never reaches 100% on completion"

    def test_the_write_phase_names_itself_for_the_ui(self):
        """`message` reaches the card as status_message; without it the two
        phases are indistinguishable to anyone watching."""
        src = inspect.getsource(extraction_service)
        assert "Writing features to database:" in src, (
            "the write phase no longer identifies itself in the UI"
        )
        assert 'event_data["message"] = message' in src, (
            "message is accepted but never emitted, so it cannot reach the UI"
        )
