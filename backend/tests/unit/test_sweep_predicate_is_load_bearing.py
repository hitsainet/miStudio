"""The frozen predicate must reach the query, and the request must reach the row.

WHY THIS FILE EXISTS. Round 4 mutated both halves of round 3's headline sweep
fix and the full 4545-test suite stayed green on each:

  M15  labeling_sweep_service.py   prompt_fingerprint=config.get(...)  -> None
  M16  api/v1/endpoints/labeling.py  the merged config={**..., **fp, **judge}
                                     -> config=body.config.model_dump(...)

Either revert restores the defect `next_batch_ids`' own docstring describes: an
operator is quoted ~27 batches and ~59 GPU-hours against the stale-inclusive
backlog, and the sweep then selects only never-attempted features — a set up to
twenty times smaller than the one it was sized for, while the stale features it
was booked to cover are never touched.

A "was called" assertion is not enough here: the defect sends the RIGHT function
the WRONG arguments. Every test below asserts the payload.
"""

import uuid
from unittest.mock import MagicMock, patch

import pytest

from src.models.labeling_resume_sweep import LabelingResumeSweep
from src.services.labeling_sweep_service import LabelingSweepService

FP = "a" * 64
JUDGE = "gemma-4-31b-it-3MPER0RR-abliterated-GGUF:IQ4_XS"


def _sweep(config) -> LabelingResumeSweep:
    from datetime import datetime, timezone
    return LabelingResumeSweep(
        created_at=datetime.now(timezone.utc),
        id=f"sweep_{uuid.uuid4().hex[:8]}",
        extraction_job_id="extr_x",
        config=config,
        status="running",
        batches_done=0,
        features_labeled=0,
        features_failed=0,
        max_batches=27,
        batch_size=2000,
        progress=0.0,
    )


class TestTheFrozenPredicateReachesTheQuery:
    """M15. The sweep row carries the predicate; the query must receive it."""

    def _call(self, config):
        db = MagicMock()
        db.execute.return_value.all.return_value = [("feat_1",)]
        service = LabelingSweepService(db)
        with patch(
            "src.services.labeling_sweep_service.labeling_eligibility"
            ".resume_batch_query"
        ) as query:
            service.next_batch_ids(_sweep(config))
        assert query.call_count == 1, (
            "the eligibility query must be built exactly once per batch"
        )
        return query.call_args

    def test_the_predicate_is_forwarded_verbatim(self):
        args, kwargs = self._call(
            {"prompt_fingerprint": FP, "judge_model": JUDGE}
        )
        assert kwargs["prompt_fingerprint"] == FP, (
            "the sweep selected without the fingerprint it was sized against"
        )
        assert kwargs["judge_model"] == JUDGE, (
            "the sweep selected without the judge it was sized against"
        )

    def test_the_extraction_and_batch_size_travel_too(self):
        """The predicate must not arrive at the cost of the scope."""
        args, kwargs = self._call(
            {"prompt_fingerprint": FP, "judge_model": JUDGE}
        )
        assert args[0] == "extr_x"
        assert kwargs["limit"] == 2000

    def test_a_sweep_without_a_predicate_still_passes_none(self):
        """The NEGATIVE CONTROL for the test above.

        If `next_batch_ids` hardcoded the fingerprint, the first test would pass
        for the wrong reason. A sweep started without a predicate must send
        None — the pre-round-3 behaviour, which is correct for that sweep.
        """
        args, kwargs = self._call({"labeling_method": "openai_compatible"})
        assert kwargs["prompt_fingerprint"] is None
        assert kwargs["judge_model"] is None


class TestTheRequestReachesTheRow:
    """M16. The REST body's predicate must be frozen onto the sweep config.

    Asserted through the real route, not by re-implementing the merge: a test
    of an extracted helper stays green when the endpoint stops calling it.
    """

    @pytest.mark.asyncio
    async def test_the_endpoint_freezes_the_predicate_into_config(self):
        from src.api.v1.endpoints.labeling import (
            ResumeSweepRequest,
            start_resume_sweep,
        )

        captured = {}

        class _Service:
            def __init__(self, db):
                pass

            def create(self, extraction_job_id, **kwargs):
                captured["extraction_job_id"] = extraction_job_id
                captured.update(kwargs)
                return _sweep(kwargs.get("config") or {})

        body = ResumeSweepRequest(
            max_batches=3,
            batch_size=2000,
            prompt_fingerprint=FP,
            judge_model=JUDGE,
            config={"extraction_job_id": "extr_x",
                    "labeling_method": "openai_compatible"},
        )

        db = MagicMock()

        async def _run_sync(fn, *a, **kw):
            return fn(MagicMock())

        db.run_sync = _run_sync

        with patch(
            "src.services.labeling_sweep_service.LabelingSweepService", _Service
        ), patch("src.api.v1.endpoints.labeling.resume_sweep_step"):
            await start_resume_sweep("extr_x", body, db)

        config = captured["config"]
        assert config["prompt_fingerprint"] == FP, (
            "the sweep row was frozen WITHOUT the fingerprint the operator "
            "sized it against; it will select a different set than it quoted"
        )
        assert config["judge_model"] == JUDGE, (
            "the sweep row was frozen WITHOUT the judge it was sized against"
        )
        assert config["labeling_method"] == "openai_compatible", (
            "the predicate must be added to the judge config, not replace it"
        )

    @pytest.mark.asyncio
    async def test_absent_predicate_is_not_invented(self):
        """NEGATIVE CONTROL. A request with no predicate must freeze none.

        `eligibility_filter` degenerates to a tautology on a ONE-SIDED
        predicate, so inventing half of one would silently widen the sweep to
        the entire staleable estate.
        """
        from src.api.v1.endpoints.labeling import (
            ResumeSweepRequest,
            start_resume_sweep,
        )

        captured = {}

        class _Service:
            def __init__(self, db):
                pass

            def create(self, extraction_job_id, **kwargs):
                captured.update(kwargs)
                return _sweep(kwargs.get("config") or {})

        body = ResumeSweepRequest(
            max_batches=3, batch_size=2000,
            config={"extraction_job_id": "extr_x",
                    "labeling_method": "openai_compatible"},
        )
        db = MagicMock()

        async def _run_sync(fn, *a, **kw):
            return fn(MagicMock())

        db.run_sync = _run_sync

        with patch(
            "src.services.labeling_sweep_service.LabelingSweepService", _Service
        ), patch("src.api.v1.endpoints.labeling.resume_sweep_step"):
            await start_resume_sweep("extr_x", body, db)

        assert "prompt_fingerprint" not in captured["config"]
        assert "judge_model" not in captured["config"]
